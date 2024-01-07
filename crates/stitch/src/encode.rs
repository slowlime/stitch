use std::collections::HashSet;

use wasm_encoder::{
    CodeSection, ElementSection, EntityType, ExportSection, FunctionSection, GlobalSection,
    ImportSection, MemorySection, StartSection, TableSection, TypeSection, DataSection,
};

use crate::ir::expr::MemArg;
use crate::ir::ty::{ElemType, GlobalType, MemoryType, TableType, Type, ValType};
use crate::ir::{
    self, ExportDef, Expr, Func, FuncBody, FuncId, GlobalDef, GlobalId, ImportDesc, ImportId,
    LocalId, MemoryId, Module, TableDef, TableId, TypeId, MemoryDef,
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
    pub fn new(module: &'a mut Module) -> Self {
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

    pub fn encode(mut self) -> Vec<u8> {
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

        self.encoder.finish()
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
            assert_eq!(idx, 0, "multiple tables are not supported");

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
        let mut sec = CodeSection::new();

        for func in self.module.funcs.values() {
            let Func::Body(body) = func else { continue };

            let params = body.params.iter().copied().collect::<HashSet<_>>();
            let mut locals = SeqSlot::new();
            let mut grouped_locals = Vec::<(u32, wasm_encoder::ValType)>::new();

            for &local_id in &body.params {
                locals.insert(local_id);
            }

            for (local_id, val_ty) in &body.locals {
                if !params.contains(&local_id) {
                    locals.insert(local_id);
                    grouped_locals.push((1, self.convert_val_type(val_ty)));
                }
            }

            let mut wr_idx = 0;

            for rd_idx in 1..grouped_locals.len() {
                let (count, val_ty) = grouped_locals[rd_idx];

                if grouped_locals[wr_idx].1 == val_ty {
                    grouped_locals[wr_idx].0 += count;
                } else {
                    wr_idx += 1;
                    grouped_locals[wr_idx] = (count, val_ty);
                }
            }

            grouped_locals.truncate(wr_idx + 1);

            let encoder = BodyEncoder {
                encoder: self,
                func_encoder: wasm_encoder::Function::new(grouped_locals),
                locals: &locals,
                body,
            };
            sec.function(&encoder.encode());
        }

        self.encoder.section(&sec);
    }

    fn encode_data(&mut self) {
        use wasm_encoder::ConstExpr;

        let mut sec = DataSection::new();

        for (idx, mem) in self.module.mems.values().enumerate() {
            assert_eq!(idx, 0, "multiple memories are not supported");

            match &mem.def {
                MemoryDef::Import(_) => panic!("memory imports are not supported"),

                MemoryDef::Bytes(bytes) => segments(
                    bytes.iter().copied(),
                    16,
                    |&byte| byte != 0,
                    |segment| {
                        let offset = segment.offset();
                        let bytes = segment.collect::<Vec<_>>();

                        sec.active(0, &ConstExpr::i32_const(offset as i32), bytes);
                    },
                )
            }
        }

        self.encoder.section(&sec);
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

struct BodyEncoder<'a, 'b> {
    encoder: &'a Encoder<'b>,
    func_encoder: wasm_encoder::Function,
    locals: &'a SeqSlot<LocalId>,
    body: &'a FuncBody,
}

impl<'a> BodyEncoder<'a, '_> {
    fn encode(mut self) -> wasm_encoder::Function {
        self.block(&self.body.body);

        self.func_encoder
    }

    fn block(&mut self, exprs: &[Expr]) {
        for expr in exprs {
            self.expr(expr);
        }
    }

    fn expr(&mut self, expr: &Expr) {
        use wasm_encoder::Instruction;

        match expr {
            Expr::I32(value) => self.nullary(Instruction::I32Const(*value)),
            Expr::I64(value) => self.nullary(Instruction::I64Const(*value)),
            Expr::F32(value) => self.nullary(Instruction::F32Const(*value)),
            Expr::F64(value) => self.nullary(Instruction::F64Const(*value)),

            Expr::I32Clz(inner) => self.unary(inner, Instruction::I32Clz),
            Expr::I32Ctz(inner) => self.unary(inner, Instruction::I32Ctz),
            Expr::I32Popcnt(inner) => self.unary(inner, Instruction::I32Popcnt),

            Expr::I64Clz(inner) => self.unary(inner, Instruction::I64Clz),
            Expr::I64Ctz(inner) => self.unary(inner, Instruction::I64Ctz),
            Expr::I64Popcnt(inner) => self.unary(inner, Instruction::I64Popcnt),

            Expr::F32Abs(inner) => self.unary(inner, Instruction::F32Abs),
            Expr::F32Neg(inner) => self.unary(inner, Instruction::F32Neg),
            Expr::F32Sqrt(inner) => self.unary(inner, Instruction::F32Sqrt),
            Expr::F32Ceil(inner) => self.unary(inner, Instruction::F32Ceil),
            Expr::F32Floor(inner) => self.unary(inner, Instruction::F32Floor),
            Expr::F32Trunc(inner) => self.unary(inner, Instruction::F32Trunc),
            Expr::F32Nearest(inner) => self.unary(inner, Instruction::F32Nearest),

            Expr::F64Abs(inner) => self.unary(inner, Instruction::F64Abs),
            Expr::F64Neg(inner) => self.unary(inner, Instruction::F64Neg),
            Expr::F64Sqrt(inner) => self.unary(inner, Instruction::F64Sqrt),
            Expr::F64Ceil(inner) => self.unary(inner, Instruction::F64Ceil),
            Expr::F64Floor(inner) => self.unary(inner, Instruction::F64Floor),
            Expr::F64Trunc(inner) => self.unary(inner, Instruction::F64Trunc),
            Expr::F64Nearest(inner) => self.unary(inner, Instruction::F64Nearest),

            Expr::I32Add(lhs, rhs) => self.binary(lhs, rhs, Instruction::I32Add),
            Expr::I32Sub(lhs, rhs) => self.binary(lhs, rhs, Instruction::I32Sub),
            Expr::I32Mul(lhs, rhs) => self.binary(lhs, rhs, Instruction::I32Mul),
            Expr::I32DivS(lhs, rhs) => self.binary(lhs, rhs, Instruction::I32DivS),
            Expr::I32DivU(lhs, rhs) => self.binary(lhs, rhs, Instruction::I32DivU),
            Expr::I32RemS(lhs, rhs) => self.binary(lhs, rhs, Instruction::I32RemS),
            Expr::I32RemU(lhs, rhs) => self.binary(lhs, rhs, Instruction::I32RemU),
            Expr::I32And(lhs, rhs) => self.binary(lhs, rhs, Instruction::I32And),
            Expr::I32Or(lhs, rhs) => self.binary(lhs, rhs, Instruction::I32Or),
            Expr::I32Xor(lhs, rhs) => self.binary(lhs, rhs, Instruction::I32Xor),
            Expr::I32Shl(lhs, rhs) => self.binary(lhs, rhs, Instruction::I32Shl),
            Expr::I32ShrS(lhs, rhs) => self.binary(lhs, rhs, Instruction::I32ShrS),
            Expr::I32ShrU(lhs, rhs) => self.binary(lhs, rhs, Instruction::I32ShrU),
            Expr::I32Rotl(lhs, rhs) => self.binary(lhs, rhs, Instruction::I32Rotl),
            Expr::I32Rotr(lhs, rhs) => self.binary(lhs, rhs, Instruction::I32Rotr),

            Expr::I64Add(lhs, rhs) => self.binary(lhs, rhs, Instruction::I64Add),
            Expr::I64Sub(lhs, rhs) => self.binary(lhs, rhs, Instruction::I64Sub),
            Expr::I64Mul(lhs, rhs) => self.binary(lhs, rhs, Instruction::I64Mul),
            Expr::I64DivS(lhs, rhs) => self.binary(lhs, rhs, Instruction::I64DivS),
            Expr::I64DivU(lhs, rhs) => self.binary(lhs, rhs, Instruction::I64DivU),
            Expr::I64RemS(lhs, rhs) => self.binary(lhs, rhs, Instruction::I64RemS),
            Expr::I64RemU(lhs, rhs) => self.binary(lhs, rhs, Instruction::I64RemU),
            Expr::I64And(lhs, rhs) => self.binary(lhs, rhs, Instruction::I64And),
            Expr::I64Or(lhs, rhs) => self.binary(lhs, rhs, Instruction::I64Or),
            Expr::I64Xor(lhs, rhs) => self.binary(lhs, rhs, Instruction::I64Xor),
            Expr::I64Shl(lhs, rhs) => self.binary(lhs, rhs, Instruction::I64Shl),
            Expr::I64ShrS(lhs, rhs) => self.binary(lhs, rhs, Instruction::I64ShrS),
            Expr::I64ShrU(lhs, rhs) => self.binary(lhs, rhs, Instruction::I64ShrU),
            Expr::I64Rotl(lhs, rhs) => self.binary(lhs, rhs, Instruction::I64Rotl),
            Expr::I64Rotr(lhs, rhs) => self.binary(lhs, rhs, Instruction::I64Rotr),

            Expr::F32Add(lhs, rhs) => self.binary(lhs, rhs, Instruction::F32Add),
            Expr::F32Sub(lhs, rhs) => self.binary(lhs, rhs, Instruction::F32Sub),
            Expr::F32Mul(lhs, rhs) => self.binary(lhs, rhs, Instruction::F32Mul),
            Expr::F32Div(lhs, rhs) => self.binary(lhs, rhs, Instruction::F32Div),
            Expr::F32Min(lhs, rhs) => self.binary(lhs, rhs, Instruction::F32Min),
            Expr::F32Max(lhs, rhs) => self.binary(lhs, rhs, Instruction::F32Max),
            Expr::F32Copysign(lhs, rhs) => self.binary(lhs, rhs, Instruction::F32Copysign),

            Expr::F64Add(lhs, rhs) => self.binary(lhs, rhs, Instruction::F64Add),
            Expr::F64Sub(lhs, rhs) => self.binary(lhs, rhs, Instruction::F64Sub),
            Expr::F64Mul(lhs, rhs) => self.binary(lhs, rhs, Instruction::F64Mul),
            Expr::F64Div(lhs, rhs) => self.binary(lhs, rhs, Instruction::F64Div),
            Expr::F64Min(lhs, rhs) => self.binary(lhs, rhs, Instruction::F64Min),
            Expr::F64Max(lhs, rhs) => self.binary(lhs, rhs, Instruction::F64Max),
            Expr::F64Copysign(lhs, rhs) => self.binary(lhs, rhs, Instruction::F64Copysign),

            Expr::I32Eqz(inner) => self.unary(inner, Instruction::I32Eqz),
            Expr::I64Eqz(inner) => self.unary(inner, Instruction::I64Eqz),

            Expr::I32Eq(lhs, rhs) => self.binary(lhs, rhs, Instruction::I32Eq),
            Expr::I32Ne(lhs, rhs) => self.binary(lhs, rhs, Instruction::I32Ne),
            Expr::I32LtS(lhs, rhs) => self.binary(lhs, rhs, Instruction::I32LtS),
            Expr::I32LtU(lhs, rhs) => self.binary(lhs, rhs, Instruction::I32LtU),
            Expr::I32GtS(lhs, rhs) => self.binary(lhs, rhs, Instruction::I32GtS),
            Expr::I32GtU(lhs, rhs) => self.binary(lhs, rhs, Instruction::I32GtU),
            Expr::I32LeS(lhs, rhs) => self.binary(lhs, rhs, Instruction::I32LeS),
            Expr::I32LeU(lhs, rhs) => self.binary(lhs, rhs, Instruction::I32LeU),
            Expr::I32GeS(lhs, rhs) => self.binary(lhs, rhs, Instruction::I32GeS),
            Expr::I32GeU(lhs, rhs) => self.binary(lhs, rhs, Instruction::I32GeU),

            Expr::I64Eq(lhs, rhs) => self.binary(lhs, rhs, Instruction::I64Eq),
            Expr::I64Ne(lhs, rhs) => self.binary(lhs, rhs, Instruction::I64Ne),
            Expr::I64LtS(lhs, rhs) => self.binary(lhs, rhs, Instruction::I64LtS),
            Expr::I64LtU(lhs, rhs) => self.binary(lhs, rhs, Instruction::I64LtU),
            Expr::I64GtS(lhs, rhs) => self.binary(lhs, rhs, Instruction::I64GtS),
            Expr::I64GtU(lhs, rhs) => self.binary(lhs, rhs, Instruction::I64GtU),
            Expr::I64LeS(lhs, rhs) => self.binary(lhs, rhs, Instruction::I64LeS),
            Expr::I64LeU(lhs, rhs) => self.binary(lhs, rhs, Instruction::I64LeU),
            Expr::I64GeS(lhs, rhs) => self.binary(lhs, rhs, Instruction::I64GeS),
            Expr::I64GeU(lhs, rhs) => self.binary(lhs, rhs, Instruction::I64GeU),

            Expr::F32Eq(lhs, rhs) => self.binary(lhs, rhs, Instruction::F32Eq),
            Expr::F32Ne(lhs, rhs) => self.binary(lhs, rhs, Instruction::F32Ne),
            Expr::F32Lt(lhs, rhs) => self.binary(lhs, rhs, Instruction::F32Lt),
            Expr::F32Gt(lhs, rhs) => self.binary(lhs, rhs, Instruction::F32Gt),
            Expr::F32Le(lhs, rhs) => self.binary(lhs, rhs, Instruction::F32Le),
            Expr::F32Ge(lhs, rhs) => self.binary(lhs, rhs, Instruction::F32Ge),

            Expr::F64Eq(lhs, rhs) => self.binary(lhs, rhs, Instruction::F64Eq),
            Expr::F64Ne(lhs, rhs) => self.binary(lhs, rhs, Instruction::F64Ne),
            Expr::F64Lt(lhs, rhs) => self.binary(lhs, rhs, Instruction::F64Lt),
            Expr::F64Gt(lhs, rhs) => self.binary(lhs, rhs, Instruction::F64Gt),
            Expr::F64Le(lhs, rhs) => self.binary(lhs, rhs, Instruction::F64Le),
            Expr::F64Ge(lhs, rhs) => self.binary(lhs, rhs, Instruction::F64Ge),

            Expr::I32WrapI64(inner) => self.unary(inner, Instruction::I32WrapI64),

            Expr::I64ExtendI32S(inner) => self.unary(inner, Instruction::I64ExtendI32S),
            Expr::I64ExtendI32U(inner) => self.unary(inner, Instruction::I64ExtendI32U),

            Expr::I32TruncF32S(inner) => self.unary(inner, Instruction::I32TruncF32S),
            Expr::I32TruncF32U(inner) => self.unary(inner, Instruction::I32TruncF32U),
            Expr::I32TruncF64S(inner) => self.unary(inner, Instruction::I32TruncF64S),
            Expr::I32TruncF64U(inner) => self.unary(inner, Instruction::I32TruncF64U),

            Expr::I64TruncF32S(inner) => self.unary(inner, Instruction::I64TruncF32S),
            Expr::I64TruncF32U(inner) => self.unary(inner, Instruction::I64TruncF32U),
            Expr::I64TruncF64S(inner) => self.unary(inner, Instruction::I64TruncF64S),
            Expr::I64TruncF64U(inner) => self.unary(inner, Instruction::I64TruncF64U),

            Expr::F32DemoteF64(inner) => self.unary(inner, Instruction::F32DemoteF64),
            Expr::F64PromoteF32(inner) => self.unary(inner, Instruction::F64PromoteF32),

            Expr::F32ConvertI32S(inner) => self.unary(inner, Instruction::F32ConvertI32S),
            Expr::F32ConvertI32U(inner) => self.unary(inner, Instruction::F32ConvertI32U),
            Expr::F32ConvertI64S(inner) => self.unary(inner, Instruction::F32ConvertI64S),
            Expr::F32ConvertI64U(inner) => self.unary(inner, Instruction::F32ConvertI64U),
            Expr::F64ConvertI32S(inner) => self.unary(inner, Instruction::F64ConvertI32S),
            Expr::F64ConvertI32U(inner) => self.unary(inner, Instruction::F64ConvertI32U),
            Expr::F64ConvertI64S(inner) => self.unary(inner, Instruction::F64ConvertI64S),
            Expr::F64ConvertI64U(inner) => self.unary(inner, Instruction::F64ConvertI64U),

            Expr::F32ReinterpretI32(inner) => self.unary(inner, Instruction::F32ReinterpretI32),
            Expr::F64ReinterpretI64(inner) => self.unary(inner, Instruction::F64ReinterpretI64),
            Expr::I32ReinterpretF32(inner) => self.unary(inner, Instruction::I32ReinterpretF32),
            Expr::I64ReinterpretF64(inner) => self.unary(inner, Instruction::I64ReinterpretF64),

            Expr::Drop(inner) => self.unary(inner, Instruction::Drop),

            Expr::Select(first, second, condition) => {
                self.ternary(first, second, condition, Instruction::Select)
            }

            Expr::LocalGet(local_id) => {
                self.nullary(Instruction::LocalGet(self.locals[*local_id] as u32))
            }

            Expr::LocalSet(local_id, inner) => {
                self.unary(inner, Instruction::LocalSet(self.locals[*local_id] as u32))
            }

            Expr::LocalTee(local_id, inner) => {
                self.unary(inner, Instruction::LocalTee(self.locals[*local_id] as u32))
            }

            Expr::GlobalGet(global_id) => self.nullary(Instruction::GlobalGet(
                self.encoder.globals[*global_id] as u32,
            )),

            Expr::GlobalSet(global_id, inner) => self.unary(
                inner,
                Instruction::GlobalSet(self.encoder.globals[*global_id] as u32),
            ),

            Expr::I32Load(mem_arg, addr) => {
                self.unary(addr, Instruction::I32Load(self.convert_mem_arg(mem_arg)))
            }

            Expr::I64Load(mem_arg, addr) => {
                self.unary(addr, Instruction::I64Load(self.convert_mem_arg(mem_arg)))
            }

            Expr::F32Load(mem_arg, addr) => {
                self.unary(addr, Instruction::F32Load(self.convert_mem_arg(mem_arg)))
            }

            Expr::F64Load(mem_arg, addr) => {
                self.unary(addr, Instruction::F64Load(self.convert_mem_arg(mem_arg)))
            }

            Expr::I32Store(mem_arg, addr, inner) => self.binary(
                addr,
                inner,
                Instruction::I32Store(self.convert_mem_arg(mem_arg)),
            ),

            Expr::I64Store(mem_arg, addr, inner) => self.binary(
                addr,
                inner,
                Instruction::I64Store(self.convert_mem_arg(mem_arg)),
            ),

            Expr::F32Store(mem_arg, addr, inner) => self.binary(
                addr,
                inner,
                Instruction::F32Store(self.convert_mem_arg(mem_arg)),
            ),

            Expr::F64Store(mem_arg, addr, inner) => self.binary(
                addr,
                inner,
                Instruction::I64Store(self.convert_mem_arg(mem_arg)),
            ),

            Expr::I32Load8S(mem_arg, addr) => {
                self.unary(addr, Instruction::I32Load8S(self.convert_mem_arg(mem_arg)))
            }

            Expr::I32Load8U(mem_arg, addr) => {
                self.unary(addr, Instruction::I32Load8U(self.convert_mem_arg(mem_arg)))
            }

            Expr::I32Load16S(mem_arg, addr) => {
                self.unary(addr, Instruction::I32Load16S(self.convert_mem_arg(mem_arg)))
            }

            Expr::I32Load16U(mem_arg, addr) => {
                self.unary(addr, Instruction::I32Load16U(self.convert_mem_arg(mem_arg)))
            }

            Expr::I64Load8S(mem_arg, addr) => {
                self.unary(addr, Instruction::I64Load8S(self.convert_mem_arg(mem_arg)))
            }

            Expr::I64Load8U(mem_arg, addr) => {
                self.unary(addr, Instruction::I64Load8U(self.convert_mem_arg(mem_arg)))
            }

            Expr::I64Load16S(mem_arg, addr) => {
                self.unary(addr, Instruction::I64Load16S(self.convert_mem_arg(mem_arg)))
            }

            Expr::I64Load16U(mem_arg, addr) => {
                self.unary(addr, Instruction::I64Load16U(self.convert_mem_arg(mem_arg)))
            }

            Expr::I64Load32S(mem_arg, addr) => {
                self.unary(addr, Instruction::I64Load32S(self.convert_mem_arg(mem_arg)))
            }

            Expr::I64Load32U(mem_arg, addr) => {
                self.unary(addr, Instruction::I64Load32U(self.convert_mem_arg(mem_arg)))
            }

            Expr::I32Store8(mem_arg, addr, inner) => self.binary(
                addr,
                inner,
                Instruction::I32Store8(self.convert_mem_arg(mem_arg)),
            ),

            Expr::I32Store16(mem_arg, addr, inner) => self.binary(
                addr,
                inner,
                Instruction::I32Store16(self.convert_mem_arg(mem_arg)),
            ),

            Expr::I64Store8(mem_arg, addr, inner) => self.binary(
                addr,
                inner,
                Instruction::I64Store8(self.convert_mem_arg(mem_arg)),
            ),

            Expr::I64Store16(mem_arg, addr, inner) => self.binary(
                addr,
                inner,
                Instruction::I64Store16(self.convert_mem_arg(mem_arg)),
            ),

            Expr::I64Store32(mem_arg, addr, inner) => self.binary(
                addr,
                inner,
                Instruction::I64Store32(self.convert_mem_arg(mem_arg)),
            ),

            Expr::MemorySize => self.nullary(Instruction::MemorySize(0)),
            Expr::MemoryGrow(size) => self.unary(size, Instruction::MemoryGrow(0)),

            Expr::Nop => self.nullary(Instruction::Nop),
            Expr::Unreachable => self.nullary(Instruction::Unreachable),

            Expr::Block(block_ty, block) => {
                self.nullary(Instruction::Block(self.convert_block_type(block_ty)));
                self.block(block);
                self.nullary(Instruction::End);
            }

            Expr::Loop(block_ty, block) => {
                self.nullary(Instruction::Loop(self.convert_block_type(block_ty)));
                self.block(block);
                self.nullary(Instruction::End);
            }

            Expr::If(block_ty, condition, then_block, else_block) => {
                self.unary(
                    condition,
                    Instruction::If(self.convert_block_type(block_ty)),
                );
                self.block(then_block);

                if !else_block.is_empty() {
                    self.nullary(Instruction::Else);
                    self.block(else_block);
                }

                self.nullary(Instruction::End);
            }

            Expr::Br(label, Some(inner)) => self.unary(inner, Instruction::Br(*label)),
            Expr::Br(label, None) => self.nullary(Instruction::Br(*label)),

            Expr::BrIf(label, condition, Some(inner)) => {
                self.binary(condition, inner, Instruction::BrIf(*label))
            }

            Expr::BrIf(label, condition, None) => self.unary(condition, Instruction::BrIf(*label)),

            Expr::BrTable(labels, default_label, condition, Some(inner)) => self.binary(
                condition,
                inner,
                Instruction::BrTable(labels.into(), *default_label),
            ),

            Expr::BrTable(labels, default_label, condition, None) => {
                self.unary(condition, Instruction::BrTable(labels.into(), *default_label))
            }

            Expr::Return(Some(inner)) => self.unary(inner, Instruction::Return),
            Expr::Return(None) => self.nullary(Instruction::Return),

            Expr::Call(func_id, args) => {
                for expr in args {
                    self.expr(expr);
                }

                self.nullary(Instruction::Call(self.encoder.funcs[*func_id] as u32));
            }

            Expr::CallIndirect(ty_id, idx_expr, args) => {
                for expr in args {
                    self.expr(expr);
                }

                self.unary(idx_expr, Instruction::CallIndirect {
                    ty: self.encoder.types[*ty_id] as u32,
                    table: 0,
                });
            }
        }
    }

    fn nullary(&mut self, instr: wasm_encoder::Instruction) {
        self.func_encoder.instruction(&instr);
    }

    fn unary(&mut self, inner: &Expr, instr: wasm_encoder::Instruction) {
        self.expr(inner);
        self.func_encoder.instruction(&instr);
    }

    fn binary(&mut self, lhs: &Expr, rhs: &Expr, instr: wasm_encoder::Instruction) {
        self.expr(lhs);
        self.expr(rhs);
        self.func_encoder.instruction(&instr);
    }

    fn ternary(
        &mut self,
        expr1: &Expr,
        expr2: &Expr,
        expr3: &Expr,
        instr: wasm_encoder::Instruction,
    ) {
        self.expr(expr1);
        self.expr(expr2);
        self.expr(expr3);
        self.func_encoder.instruction(&instr);
    }

    fn convert_mem_arg(&self, mem_arg: &MemArg) -> wasm_encoder::MemArg {
        wasm_encoder::MemArg {
            offset: mem_arg.offset as u64,
            align: mem_arg.align,
            memory_index: 0,
        }
    }

    fn convert_block_type(&self, block_ty: &Option<ValType>) -> wasm_encoder::BlockType {
        match block_ty {
            Some(val_ty) => wasm_encoder::BlockType::Result(self.encoder.convert_val_type(val_ty)),
            None => wasm_encoder::BlockType::Empty,
        }
    }
}
