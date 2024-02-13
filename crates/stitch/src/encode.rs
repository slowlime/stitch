use std::collections::HashSet;

use log::trace;
use slotmap::SecondaryMap;
use wasm_encoder::{
    CodeSection, DataSection, ElementSection, EntityType, ExportSection, FunctionSection,
    GlobalSection, ImportSection, MemorySection, StartSection, TableSection, TypeSection,
};

use crate::ast::expr::{BinOp, Block, MemArg, NulOp, TernOp, UnOp, Value};
use crate::ast::ty::{BlockType, ElemType, GlobalType, MemoryType, TableType, Type, ValType};
use crate::ast::{
    self, BlockId, ConstExpr, ExportDef, Expr, Func, FuncBody, FuncId, GlobalDef, GlobalId,
    ImportDesc, ImportId, LocalId, MemoryDef, MemoryId, Module, TableDef, TableId, TypeId,
};
use crate::util::iter::segments;
use crate::util::slot::SeqSlot;

struct Encoder<'a> {
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
                ast::ty::Type::Func(ty) => {
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

        for (func_id, func) in &self.module.funcs {
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
            trace!("encoding func {func_id:?}");

            let encoder = BodyEncoder {
                encoder: self,
                func_encoder: wasm_encoder::Function::new(grouped_locals),
                locals: &locals,
                body,
                block_stack: Default::default(),
                block_depths: Default::default(),
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
                ),
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
            ElemType::Funcref => wasm_encoder::RefType::FUNCREF,
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

    fn convert_const_expr(&self, expr: &ConstExpr) -> Option<wasm_encoder::ConstExpr> {
        Some(match *expr {
            ConstExpr::Value(Value::I32(value), _) => wasm_encoder::ConstExpr::i32_const(value),
            ConstExpr::Value(Value::I64(value), _) => wasm_encoder::ConstExpr::i64_const(value),
            ConstExpr::Value(Value::F32(value), _) => {
                wasm_encoder::ConstExpr::f32_const(value.to_f32())
            }
            ConstExpr::Value(Value::F64(value), _) => {
                wasm_encoder::ConstExpr::f64_const(value.to_f64())
            }
            ConstExpr::GlobalGet(global_id) => {
                wasm_encoder::ConstExpr::global_get(self.globals[global_id] as u32)
            }
        })
    }
}

struct BodyEncoder<'a, 'b> {
    encoder: &'a Encoder<'b>,
    func_encoder: wasm_encoder::Function,
    locals: &'a SeqSlot<LocalId>,
    body: &'a FuncBody,
    block_stack: Vec<BlockId>,
    block_depths: SecondaryMap<BlockId, u32>,
}

impl<'a> BodyEncoder<'a, '_> {
    fn encode(mut self) -> wasm_encoder::Function {
        self.block(&self.body.main_block);
        self.nullary(wasm_encoder::Instruction::End);

        self.func_encoder
    }

    fn get_relative_depth(&self, block_id: BlockId) -> u32 {
        self.block_stack.len() as u32 - self.block_depths[block_id] - 1
    }

    fn block(&mut self, block: &Block) {
        trace!("encoding block {block}");
        self.block_depths
            .insert(block.id, self.block_stack.len() as u32);
        self.block_stack.push(block.id);

        for expr in &block.body {
            self.expr(expr);
        }

        assert_eq!(self.block_stack.pop(), Some(block.id));
        self.block_depths.remove(block.id);
    }

    fn expr(&mut self, expr: &Expr) {
        use wasm_encoder::Instruction;

        trace!("encoding expr {expr}");

        match expr {
            Expr::Value(value, _) => self.nullary(match value {
                Value::I32(value) => Instruction::I32Const(*value),
                Value::I64(value) => Instruction::I64Const(*value),
                Value::F32(value) => Instruction::F32Const(value.to_f32()),
                Value::F64(value) => Instruction::F64Const(value.to_f64()),
            }),

            Expr::Nullary(op) => self.nullary(match *op {
                NulOp::Nop => Instruction::Nop,
                NulOp::Unreachable => Instruction::Unreachable,
                NulOp::LocalGet(local_id) => Instruction::LocalGet(self.locals[local_id] as u32),
                NulOp::GlobalGet(global_id) => {
                    Instruction::GlobalGet(self.encoder.globals[global_id] as u32)
                }
                NulOp::MemorySize(mem_id) => {
                    Instruction::MemorySize(self.encoder.mems[mem_id] as u32)
                }
            }),

            Expr::Unary(op, inner) => self.unary(
                inner,
                match *op {
                    UnOp::I32Clz => Instruction::I32Clz,
                    UnOp::I32Ctz => Instruction::I32Ctz,
                    UnOp::I32Popcnt => Instruction::I32Popcnt,

                    UnOp::I64Clz => Instruction::I64Clz,
                    UnOp::I64Ctz => Instruction::I64Ctz,
                    UnOp::I64Popcnt => Instruction::I64Popcnt,

                    UnOp::F32Abs => Instruction::F32Abs,
                    UnOp::F32Neg => Instruction::F32Neg,
                    UnOp::F32Sqrt => Instruction::F32Sqrt,
                    UnOp::F32Ceil => Instruction::F32Ceil,
                    UnOp::F32Floor => Instruction::F32Floor,
                    UnOp::F32Trunc => Instruction::F32Trunc,
                    UnOp::F32Nearest => Instruction::F32Nearest,

                    UnOp::F64Abs => Instruction::F64Abs,
                    UnOp::F64Neg => Instruction::F64Neg,
                    UnOp::F64Sqrt => Instruction::F64Sqrt,
                    UnOp::F64Ceil => Instruction::F64Ceil,
                    UnOp::F64Floor => Instruction::F64Floor,
                    UnOp::F64Trunc => Instruction::F64Trunc,
                    UnOp::F64Nearest => Instruction::F64Nearest,

                    UnOp::I32Eqz => Instruction::I32Eqz,
                    UnOp::I64Eqz => Instruction::I64Eqz,

                    UnOp::I32WrapI64 => Instruction::I32WrapI64,

                    UnOp::I64ExtendI32S => Instruction::I64ExtendI32S,
                    UnOp::I64ExtendI32U => Instruction::I64ExtendI32U,

                    UnOp::I32TruncF32S => Instruction::I32TruncF32S,
                    UnOp::I32TruncF32U => Instruction::I32TruncF32U,
                    UnOp::I32TruncF64S => Instruction::I32TruncF64S,
                    UnOp::I32TruncF64U => Instruction::I32TruncF64U,

                    UnOp::I64TruncF32S => Instruction::I64TruncF32S,
                    UnOp::I64TruncF32U => Instruction::I64TruncF32U,
                    UnOp::I64TruncF64S => Instruction::I64TruncF64S,
                    UnOp::I64TruncF64U => Instruction::I64TruncF64U,

                    UnOp::F32DemoteF64 => Instruction::F32DemoteF64,
                    UnOp::F64PromoteF32 => Instruction::F64PromoteF32,

                    UnOp::F32ConvertI32S => Instruction::F32ConvertI32S,
                    UnOp::F32ConvertI32U => Instruction::F32ConvertI32U,
                    UnOp::F32ConvertI64S => Instruction::F32ConvertI64S,
                    UnOp::F32ConvertI64U => Instruction::F32ConvertI64U,

                    UnOp::F64ConvertI32S => Instruction::F64ConvertI32S,
                    UnOp::F64ConvertI32U => Instruction::F64ConvertI32U,
                    UnOp::F64ConvertI64S => Instruction::F64ConvertI64S,
                    UnOp::F64ConvertI64U => Instruction::F64ConvertI64U,

                    UnOp::F32ReinterpretI32 => Instruction::F32ReinterpretI32,
                    UnOp::F64ReinterpretI64 => Instruction::F64ReinterpretI64,
                    UnOp::I32ReinterpretF32 => Instruction::I32ReinterpretF32,
                    UnOp::I64ReinterpretF64 => Instruction::I64ReinterpretF64,

                    UnOp::I32Extend8S => Instruction::I32Extend8S,
                    UnOp::I32Extend16S => Instruction::I32Extend16S,

                    UnOp::I64Extend8S => Instruction::I64Extend8S,
                    UnOp::I64Extend16S => Instruction::I64Extend16S,
                    UnOp::I64Extend32S => Instruction::I64Extend32S,

                    UnOp::LocalSet(local_id) => Instruction::LocalSet(self.locals[local_id] as u32),
                    UnOp::LocalTee(local_id) => Instruction::LocalTee(self.locals[local_id] as u32),

                    UnOp::GlobalSet(global_id) => {
                        Instruction::GlobalSet(self.encoder.globals[global_id] as u32)
                    }

                    UnOp::I32Load(mem_arg) => Instruction::I32Load(self.convert_mem_arg(&mem_arg)),
                    UnOp::I64Load(mem_arg) => Instruction::I64Load(self.convert_mem_arg(&mem_arg)),
                    UnOp::F32Load(mem_arg) => Instruction::F32Load(self.convert_mem_arg(&mem_arg)),
                    UnOp::F64Load(mem_arg) => Instruction::F64Load(self.convert_mem_arg(&mem_arg)),

                    UnOp::I32Load8S(mem_arg) => {
                        Instruction::I32Load8S(self.convert_mem_arg(&mem_arg))
                    }
                    UnOp::I32Load8U(mem_arg) => {
                        Instruction::I32Load8U(self.convert_mem_arg(&mem_arg))
                    }
                    UnOp::I32Load16S(mem_arg) => {
                        Instruction::I32Load16S(self.convert_mem_arg(&mem_arg))
                    }
                    UnOp::I32Load16U(mem_arg) => {
                        Instruction::I32Load16U(self.convert_mem_arg(&mem_arg))
                    }

                    UnOp::I64Load8S(mem_arg) => {
                        Instruction::I64Load8S(self.convert_mem_arg(&mem_arg))
                    }
                    UnOp::I64Load8U(mem_arg) => {
                        Instruction::I64Load8U(self.convert_mem_arg(&mem_arg))
                    }
                    UnOp::I64Load16S(mem_arg) => {
                        Instruction::I64Load16S(self.convert_mem_arg(&mem_arg))
                    }
                    UnOp::I64Load16U(mem_arg) => {
                        Instruction::I64Load16U(self.convert_mem_arg(&mem_arg))
                    }
                    UnOp::I64Load32S(mem_arg) => {
                        Instruction::I64Load32S(self.convert_mem_arg(&mem_arg))
                    }
                    UnOp::I64Load32U(mem_arg) => {
                        Instruction::I64Load32U(self.convert_mem_arg(&mem_arg))
                    }

                    UnOp::MemoryGrow(mem_id) => {
                        Instruction::MemoryGrow(self.encoder.mems[mem_id] as u32)
                    }

                    UnOp::Drop => Instruction::Drop,
                },
            ),

            Expr::Binary(op, [lhs, rhs]) => self.binary(
                lhs,
                rhs,
                match *op {
                    BinOp::I32Add => Instruction::I32Add,
                    BinOp::I32Sub => Instruction::I32Sub,
                    BinOp::I32Mul => Instruction::I32Mul,
                    BinOp::I32DivS => Instruction::I32DivS,
                    BinOp::I32DivU => Instruction::I32DivU,
                    BinOp::I32RemS => Instruction::I32RemS,
                    BinOp::I32RemU => Instruction::I32RemU,
                    BinOp::I32And => Instruction::I32And,
                    BinOp::I32Or => Instruction::I32Or,
                    BinOp::I32Xor => Instruction::I32Xor,
                    BinOp::I32Shl => Instruction::I32Shl,
                    BinOp::I32ShrS => Instruction::I32ShrS,
                    BinOp::I32ShrU => Instruction::I32ShrU,
                    BinOp::I32Rotl => Instruction::I32Rotl,
                    BinOp::I32Rotr => Instruction::I32Rotr,

                    BinOp::I64Add => Instruction::I64Add,
                    BinOp::I64Sub => Instruction::I64Sub,
                    BinOp::I64Mul => Instruction::I64Mul,
                    BinOp::I64DivS => Instruction::I64DivS,
                    BinOp::I64DivU => Instruction::I64DivU,
                    BinOp::I64RemS => Instruction::I64RemS,
                    BinOp::I64RemU => Instruction::I64RemU,
                    BinOp::I64And => Instruction::I64And,
                    BinOp::I64Or => Instruction::I64Or,
                    BinOp::I64Xor => Instruction::I64Xor,
                    BinOp::I64Shl => Instruction::I64Shl,
                    BinOp::I64ShrS => Instruction::I64ShrS,
                    BinOp::I64ShrU => Instruction::I64ShrU,
                    BinOp::I64Rotl => Instruction::I64Rotl,
                    BinOp::I64Rotr => Instruction::I64Rotr,

                    BinOp::F32Add => Instruction::F32Add,
                    BinOp::F32Sub => Instruction::F32Sub,
                    BinOp::F32Mul => Instruction::F32Mul,
                    BinOp::F32Div => Instruction::F32Div,
                    BinOp::F32Min => Instruction::F32Min,
                    BinOp::F32Max => Instruction::F32Max,
                    BinOp::F32Copysign => Instruction::F32Copysign,

                    BinOp::F64Add => Instruction::F64Add,
                    BinOp::F64Sub => Instruction::F64Sub,
                    BinOp::F64Mul => Instruction::F64Mul,
                    BinOp::F64Div => Instruction::F64Div,
                    BinOp::F64Min => Instruction::F64Min,
                    BinOp::F64Max => Instruction::F64Max,
                    BinOp::F64Copysign => Instruction::F64Copysign,

                    BinOp::I32Eq => Instruction::I32Eq,
                    BinOp::I32Ne => Instruction::I32Ne,
                    BinOp::I32LtS => Instruction::I32LtS,
                    BinOp::I32LtU => Instruction::I32LtU,
                    BinOp::I32GtS => Instruction::I32GtS,
                    BinOp::I32GtU => Instruction::I32GtU,
                    BinOp::I32LeS => Instruction::I32LeS,
                    BinOp::I32LeU => Instruction::I32LeU,
                    BinOp::I32GeS => Instruction::I32GeS,
                    BinOp::I32GeU => Instruction::I32GeU,

                    BinOp::I64Eq => Instruction::I64Eq,
                    BinOp::I64Ne => Instruction::I64Ne,
                    BinOp::I64LtS => Instruction::I64LtS,
                    BinOp::I64LtU => Instruction::I64LtU,
                    BinOp::I64GtS => Instruction::I64GtS,
                    BinOp::I64GtU => Instruction::I64GtU,
                    BinOp::I64LeS => Instruction::I64LeS,
                    BinOp::I64LeU => Instruction::I64LeU,
                    BinOp::I64GeS => Instruction::I64GeS,
                    BinOp::I64GeU => Instruction::I64GeU,

                    BinOp::F32Eq => Instruction::F32Eq,
                    BinOp::F32Ne => Instruction::F32Ne,
                    BinOp::F32Lt => Instruction::F32Lt,
                    BinOp::F32Gt => Instruction::F32Gt,
                    BinOp::F32Le => Instruction::F32Le,
                    BinOp::F32Ge => Instruction::F32Ge,

                    BinOp::F64Eq => Instruction::F64Eq,
                    BinOp::F64Ne => Instruction::F64Ne,
                    BinOp::F64Lt => Instruction::F64Lt,
                    BinOp::F64Gt => Instruction::F64Gt,
                    BinOp::F64Le => Instruction::F64Le,
                    BinOp::F64Ge => Instruction::F64Ge,

                    BinOp::I32Store(mem_arg) => {
                        Instruction::I32Store(self.convert_mem_arg(&mem_arg))
                    }
                    BinOp::I64Store(mem_arg) => {
                        Instruction::I64Store(self.convert_mem_arg(&mem_arg))
                    }
                    BinOp::F32Store(mem_arg) => {
                        Instruction::F32Store(self.convert_mem_arg(&mem_arg))
                    }
                    BinOp::F64Store(mem_arg) => {
                        Instruction::F64Store(self.convert_mem_arg(&mem_arg))
                    }

                    BinOp::I32Store8(mem_arg) => {
                        Instruction::I32Store8(self.convert_mem_arg(&mem_arg))
                    }
                    BinOp::I32Store16(mem_arg) => {
                        Instruction::I32Store16(self.convert_mem_arg(&mem_arg))
                    }

                    BinOp::I64Store8(mem_arg) => {
                        Instruction::I64Store8(self.convert_mem_arg(&mem_arg))
                    }
                    BinOp::I64Store16(mem_arg) => {
                        Instruction::I64Store16(self.convert_mem_arg(&mem_arg))
                    }
                    BinOp::I64Store32(mem_arg) => {
                        Instruction::I64Store32(self.convert_mem_arg(&mem_arg))
                    }
                },
            ),

            Expr::Ternary(TernOp::Select, [first, second, condition]) => {
                self.ternary(first, second, condition, Instruction::Select)
            }

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

                if !else_block.body.is_empty() {
                    self.nullary(Instruction::Else);
                    self.block(else_block);
                }

                self.nullary(Instruction::End);
            }

            Expr::Br(block_id, Some(inner)) => {
                self.unary(inner, Instruction::Br(self.get_relative_depth(*block_id)))
            }
            Expr::Br(block_id, None) => {
                self.nullary(Instruction::Br(self.get_relative_depth(*block_id)))
            }

            Expr::BrIf(block_id, Some(inner), condition) => self.binary(
                inner,
                condition,
                Instruction::BrIf(self.get_relative_depth(*block_id)),
            ),

            Expr::BrIf(block_id, None, condition) => self.unary(
                condition,
                Instruction::BrIf(self.get_relative_depth(*block_id)),
            ),

            Expr::BrTable(block_ids, default_block_id, Some(inner), condition) => self.binary(
                inner,
                condition,
                Instruction::BrTable(
                    block_ids
                        .iter()
                        .map(|&block_id| self.get_relative_depth(block_id))
                        .collect(),
                    self.get_relative_depth(*default_block_id),
                ),
            ),

            Expr::BrTable(block_ids, default_block_id, None, condition) => self.unary(
                condition,
                Instruction::BrTable(
                    block_ids
                        .iter()
                        .map(|&block_id| self.get_relative_depth(block_id))
                        .collect(),
                    self.get_relative_depth(*default_block_id),
                ),
            ),

            Expr::Return(Some(inner)) => self.unary(inner, Instruction::Return),
            Expr::Return(None) => self.nullary(Instruction::Return),

            Expr::Call(func_id, args) => {
                for expr in args {
                    self.expr(expr);
                }

                self.nullary(Instruction::Call(self.encoder.funcs[*func_id] as u32));
            }

            Expr::CallIndirect(ty_id, table_id, args, idx_expr) => {
                for expr in args {
                    self.expr(expr);
                }

                self.unary(
                    idx_expr,
                    Instruction::CallIndirect {
                        ty: self.encoder.types[*ty_id] as u32,
                        table: self.encoder.tables[*table_id] as u32,
                    },
                );
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

    fn convert_block_type(&self, block_ty: &BlockType) -> wasm_encoder::BlockType {
        match block_ty {
            BlockType::Empty => wasm_encoder::BlockType::Empty,
            BlockType::Result(val_ty) => {
                wasm_encoder::BlockType::Result(self.encoder.convert_val_type(val_ty))
            }
        }
    }
}

pub fn encode(module: &mut Module) -> Vec<u8> {
    Encoder::new(module).encode()
}
