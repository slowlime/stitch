use wasm_encoder::{
    ElementSection, EntityType, ExportSection, FunctionSection, GlobalSection,
    ImportSection, MemorySection, StartSection, TableSection, TypeSection,
};

use crate::ir::ty::{ElemType, GlobalType, MemoryType, TableType, Type, ValType};
use crate::ir::{
    self, ExportDef, Expr, Func, FuncId, GlobalDef, GlobalId, ImportDesc, ImportId, MemoryId,
    Module, TableDef, TableId, TypeId,
};
use crate::util::iter::segments;
use crate::util::slot::SeqSlot;

pub struct Encoder<'a> {
    module: &'a mut Module,
    encoder: wasm_encoder::Module,
    types: SeqSlot<TypeId>,
    imports: SeqSlot<ImportId>,
    funcs: SeqSlot<FuncId>,
    tables: SeqSlot<TableId>,
    mems: SeqSlot<MemoryId>,
    globals: SeqSlot<GlobalId>,
}

impl<'a> Encoder<'a> {
    fn new(module: &'a mut Module) -> Self {
        Self {
            module,
            encoder: Default::default(),
            types: Default::default(),
            imports: Default::default(),
            funcs: Default::default(),
            tables: Default::default(),
            mems: Default::default(),
            globals: Default::default(),
        }
    }

    fn encode(mut self) {
        self.module.insert_func_types();
        self.encode_types();
        self.encode_imports();
        self.encode_funcs();
        self.encode_tables();
        self.encode_memories();
        self.encode_globals();
        self.encode_exports();
        self.encode_start();
        self.encode_elements();
        self.encode_code();
        self.encode_data();
    }

    fn encode_types(&mut self) {
        let mut sec = TypeSection::new();

        for (ty_id, ty) in &self.module.types {
            self.types.insert(ty_id).unwrap();

            match ty {
                ir::ty::Type::Func(ty) => {
                    let params = ty.params.iter().map(|val_ty| self.convert_val_type(val_ty));
                    let ret = ty.ret.iter().map(|val_ty| self.convert_val_type(val_ty));

                    sec.function(params, ret);
                }
            }
        }

        self.encoder.section(&sec);
    }

    fn encode_imports(&mut self) {
        let mut sec = ImportSection::new();

        for (import_id, import) in &self.module.imports {
            self.imports.insert(import_id).unwrap();

            let entity = match &import.desc {
                ImportDesc::Func(ty_id) => EntityType::Function(self.types[*ty_id] as u32),
                ImportDesc::Table(table_ty) => EntityType::Table(self.convert_table_type(table_ty)),
                ImportDesc::Memory(mem_ty) => EntityType::Memory(self.convert_mem_type(mem_ty)),
                ImportDesc::Global(global_ty) => {
                    EntityType::Global(self.convert_global_type(global_ty))
                }
            };

            sec.import(&import.module, &import.name, entity);
        }

        self.encoder.section(&sec);
    }

    fn encode_funcs(&mut self) {
        let mut sec = FunctionSection::new();

        // imports go first
        for (func_id, func) in &self.module.funcs {
            if func.is_import() {
                self.funcs.insert(func_id);
            }
        }

        for (func_id, func) in &self.module.funcs {
            let Func::Body(body) = func else { continue };

            let ty_id = self
                .module
                .types
                .get_key(&Type::Func(body.ty.clone()))
                .expect("module should have all func types");

            let ty_idx = self.types.get(ty_id).unwrap();
            sec.function(ty_idx as u32);
            self.funcs.insert(func_id);
        }

        self.encoder.section(&sec);
    }

    fn encode_tables(&mut self) {
        let mut sec = TableSection::new();

        // process imports first
        for (table_id, table) in &self.module.tables {
            if table.def.is_import() {
                self.tables.insert(table_id);
            }
        }

        for (table_id, table) in &self.module.tables {
            if table.def.is_import() {
                continue;
            };

            sec.table(self.convert_table_type(&table.ty));
            self.tables.insert(table_id);
        }

        self.encoder.section(&sec);
    }

    fn encode_memories(&mut self) {
        let mut sec = MemorySection::new();

        // process imports first
        for (mem_id, mem) in &self.module.mems {
            if mem.def.is_import() {
                self.mems.insert(mem_id);
            }
        }

        for (mem_id, mem) in &self.module.mems {
            if mem.def.is_import() {
                continue;
            }

            sec.memory(self.convert_mem_type(&mem.ty));
            self.mems.insert(mem_id);
        }

        self.encoder.section(&sec);
    }

    fn encode_globals(&mut self) {
        let mut sec = GlobalSection::new();

        // process imports first
        for (global_id, global) in &self.module.globals {
            if global.def.is_import() {
                self.globals.insert(global_id);
            }
        }

        for (global_id, global) in &self.module.globals {
            let expr = match global.def {
                GlobalDef::Import(_) => continue,
                GlobalDef::Value(ref expr) => expr,
            };

            sec.global(
                self.convert_global_type(&global.ty),
                &self.convert_const_expr(expr).unwrap(),
            );
            self.globals.insert(global_id);
        }

        self.encoder.section(&sec);
    }

    fn encode_exports(&mut self) {
        use wasm_encoder::ExportKind;

        let mut sec = ExportSection::new();

        for export in self.module.exports.values() {
            let (kind, idx) = match export.def {
                ExportDef::Func(func_id) => (ExportKind::Func, self.funcs[func_id]),
                ExportDef::Table(table_id) => (ExportKind::Table, self.tables[table_id]),
                ExportDef::Memory(mem_id) => (ExportKind::Memory, self.mems[mem_id]),
                ExportDef::Global(global_id) => (ExportKind::Global, self.globals[global_id]),
            };

            sec.export(&export.name, kind, idx as u32);
        }

        self.encoder.section(&sec);
    }

    fn encode_start(&mut self) {
        if let Some(func_id) = self.module.start {
            self.encoder.section(&StartSection {
                function_index: self.funcs[func_id] as u32,
            });
        }
    }

    fn encode_elements(&mut self) {
        use wasm_encoder::{ConstExpr, Elements};

        let mut sec = ElementSection::new();

        for (idx, table) in self.module.tables.values().enumerate() {
            assert_eq!(idx, 0, "multiple memories are not supported");

            match &table.def {
                TableDef::Import(_) => panic!("table imports are not supported"),

                TableDef::Elems(elems) => segments(
                    elems,
                    0,
                    |elem| elem.is_some(),
                    |segment| {
                        let offset = segment.offset();
                        let funcs: Vec<_> = segment
                            .map(|func_id| self.funcs[func_id.unwrap()] as u32)
                            .collect();

                        sec.active(
                            None,
                            &ConstExpr::i32_const(offset as i32),
                            Elements::Functions(&funcs),
                        );
                    },
                ),
            }
        }

        self.encoder.section(&sec);
    }

    fn encode_code(&mut self) {
        todo!()
    }

    fn encode_data(&mut self) {
        todo!()
    }
}

impl Encoder<'_> {
    fn convert_val_type(&self, val_ty: &ValType) -> wasm_encoder::ValType {
        match val_ty {
            ValType::I32 => wasm_encoder::ValType::I32,
            ValType::I64 => wasm_encoder::ValType::I64,
            ValType::F32 => wasm_encoder::ValType::F32,
            ValType::F64 => wasm_encoder::ValType::F64,
        }
    }

    fn convert_elem_type(&self, elem_ty: &ElemType) -> wasm_encoder::RefType {
        match elem_ty {
            ElemType::FuncType => wasm_encoder::RefType::FUNCREF,
        }
    }

    fn convert_table_type(&self, table_ty: &TableType) -> wasm_encoder::TableType {
        wasm_encoder::TableType {
            element_type: self.convert_elem_type(&table_ty.elem_ty),
            minimum: table_ty.limits.min,
            maximum: table_ty.limits.max,
        }
    }

    fn convert_mem_type(&self, mem_type: &MemoryType) -> wasm_encoder::MemoryType {
        wasm_encoder::MemoryType {
            minimum: mem_type.limits.min as u64,
            maximum: mem_type.limits.max.map(|max| max as u64),
            memory64: false,
            shared: false,
        }
    }

    fn convert_global_type(&self, global_ty: &GlobalType) -> wasm_encoder::GlobalType {
        wasm_encoder::GlobalType {
            val_type: self.convert_val_type(&global_ty.val_type),
            mutable: global_ty.mutable,
        }
    }

    fn convert_const_expr(&self, expr: &Expr) -> Option<wasm_encoder::ConstExpr> {
        use wasm_encoder::ConstExpr;

        Some(match *expr {
            Expr::I32(value) => ConstExpr::i32_const(value),
            Expr::I64(value) => ConstExpr::i64_const(value),
            Expr::F32(value) => ConstExpr::f32_const(value),
            Expr::F64(value) => ConstExpr::f64_const(value),
            Expr::GlobalGet(global_id) => ConstExpr::global_get(self.globals[global_id] as u32),
            _ => return None,
        })
    }
}
