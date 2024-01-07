use std::mem;
use std::ops::Range;

use thiserror::Error;
use wasmparser::{
    BinaryReaderError, CompositeType, ExternalKind, Operator, Payload, SubType, WasmFeatures,
};

use crate::ir::expr::{ExprTy, ReturnValueCount};
use crate::ir::{self, ExportId, FuncId, GlobalId, ImportId, LocalId, MemoryId, TableId, TypeId};

const FEATURES: WasmFeatures = WasmFeatures {
    mutable_global: true,
    saturating_float_to_int: false,
    sign_extension: false,
    reference_types: false,
    multi_value: false,
    bulk_memory: false,
    simd: false,
    relaxed_simd: false,
    threads: false,
    tail_call: false,
    floats: true,
    multi_memory: false,
    exceptions: false,
    memory64: false,
    extended_const: false,
    component_model: false,
    function_references: false,
    memory_control: false,
    gc: false,
    component_model_values: false,
};

#[derive(Error, Debug, Clone)]
pub enum ParseError {
    #[error("malformed module: {0}")]
    Parser(#[from] BinaryReaderError),

    #[error("{kind} import ({module}/{name}) is not supported")]
    UnsupportedImport {
        kind: ir::ImportKind,
        module: String,
        name: String,
    },

    #[error("element #{element_idx} (segment #{segment_idx}) stored at #{idx} is out of range for table #{table_idx}")]
    TableElemOutOfRange {
        element_idx: usize,
        segment_idx: usize,
        idx: usize,
        table_idx: usize,
    },

    #[error(
        "data segment #{segment_idx} spans memory bytes 0x{:x}..0x{:x}, \
        which is out of range for memory #{mem_idx} of size 0x{mem_size:x}",
        .range.start, .range.end
    )]
    DataOutOfRange {
        segment_idx: usize,
        range: Range<usize>,
        mem_idx: usize,
        mem_size: usize,
    },
}

fn make_val_type(val_ty: wasmparser::ValType) -> ir::ty::ValType {
    match val_ty {
        wasmparser::ValType::I32 => ir::ty::ValType::I32,
        wasmparser::ValType::I64 => ir::ty::ValType::I64,
        wasmparser::ValType::F32 => ir::ty::ValType::F32,
        wasmparser::ValType::F64 => ir::ty::ValType::F64,
        wasmparser::ValType::V128 | wasmparser::ValType::Ref(_) => {
            unreachable!("unsupported types")
        }
    }
}

fn make_elem_type(ref_ty: wasmparser::RefType) -> ir::ty::ElemType {
    assert!(
        ref_ty.is_func_ref(),
        "unsupported table element type {ref_ty}"
    );

    ir::ty::ElemType::FuncType
}

fn make_table_type(table_ty: wasmparser::TableType) -> ir::ty::TableType {
    let elem_ty = make_elem_type(table_ty.element_type);

    ir::ty::TableType {
        elem_ty,
        limits: ir::ty::Limits {
            min: table_ty.initial,
            max: table_ty.maximum,
        },
    }
}

fn make_mem_type(mem_ty: wasmparser::MemoryType) -> ir::ty::MemoryType {
    assert!(!mem_ty.memory64, "unsupported memory type {mem_ty:?}");

    ir::ty::MemoryType {
        limits: ir::ty::Limits {
            min: mem_ty.initial as _,
            max: mem_ty.maximum.map(|max| max as _),
        },
    }
}

fn make_global_type(global_ty: wasmparser::GlobalType) -> ir::ty::GlobalType {
    ir::ty::GlobalType {
        val_type: make_val_type(global_ty.content_type),
        mutable: global_ty.mutable,
    }
}

fn make_block_type(block_ty: wasmparser::BlockType) -> Option<ir::ty::ValType> {
    match block_ty {
        wasmparser::BlockType::Empty => None,
        wasmparser::BlockType::Type(ty) => Some(make_val_type(ty)),
        _ => unreachable!("func block types are not supported"),
    }
}

fn make_mem_arg(mem_arg: wasmparser::MemArg) -> ir::expr::MemArg {
    ir::expr::MemArg {
        offset: mem_arg.offset as u32,
        align: mem_arg.align as u32,
    }
}

#[derive(Debug, Default)]
struct ImportCount {
    funcs: usize,
    tables: usize,
    mems: usize,
    globals: usize,
}

#[derive(Debug, Default)]
struct Parser {
    module: ir::Module,
    types: Vec<TypeId>,
    funcs: Vec<FuncId>,
    tables: Vec<TableId>,
    mems: Vec<MemoryId>,
    globals: Vec<GlobalId>,
    imports: Vec<ImportId>,

    import_count: ImportCount,
}

pub type Result<T, E = ParseError> = ::std::result::Result<T, E>;

impl Parser {
    fn add_ty(&mut self, ty: ir::ty::Type) -> TypeId {
        let id = self.module.types.insert(ty);
        self.types.push(id);

        id
    }

    fn add_func(&mut self, func: ir::Func) -> FuncId {
        let id = self.module.funcs.insert(func);
        self.funcs.push(id);

        id
    }

    fn add_table(&mut self, table: ir::Table) -> TableId {
        let id = self.module.tables.insert(table);
        self.tables.push(id);

        id
    }

    fn add_mem(&mut self, mem: ir::Memory) -> MemoryId {
        let id = self.module.mems.insert(mem);
        self.mems.push(id);

        id
    }

    fn add_global(&mut self, global: ir::Global) -> GlobalId {
        let id = self.module.globals.insert(global);
        self.globals.push(id);

        id
    }

    fn add_import(&mut self, import: ir::Import) -> Result<ImportId> {
        let id = self.module.imports.insert(import);
        self.imports.push(id);

        let import = &self.module.imports[id];

        match import.desc.kind() {
            ir::ImportKind::Func => self.import_count.funcs += 1,
            ir::ImportKind::Table => self.import_count.tables += 1,
            ir::ImportKind::Memory => self.import_count.mems += 1,
            ir::ImportKind::Global => self.import_count.globals += 1,
        }

        match &import.desc {
            ir::ImportDesc::Func(ty_idx) => {
                self.add_func(ir::Func::Import(ir::func::FuncImport {
                    ty: self.module.types[*ty_idx].as_func().clone(),
                    import: id,
                }));
            }

            desc => {
                return Err(ParseError::UnsupportedImport {
                    kind: desc.kind(),
                    module: import.module.clone(),
                    name: import.name.clone(),
                })
            }
        }

        Ok(id)
    }

    fn add_export(&mut self, export: ir::Export) -> ExportId {
        self.module.exports.insert(export)
    }

    fn parse(mut self, bytes: &[u8]) -> Result<ir::Module> {
        let parser = wasmparser::Parser::new(0);
        let mut validator = wasmparser::Validator::new_with_features(FEATURES);
        let mut code_sections: usize = 0;

        for payload in parser.parse_all(bytes) {
            match payload? {
                Payload::Version {
                    num,
                    encoding,
                    range,
                } => {
                    validator.version(num, encoding, &range)?;
                }

                Payload::TypeSection(reader) => {
                    validator.type_section(&reader)?;
                    self.parse_types(reader)?;
                }

                Payload::ImportSection(reader) => {
                    validator.import_section(&reader)?;
                    self.parse_imports(reader)?;
                }

                Payload::FunctionSection(reader) => {
                    validator.function_section(&reader)?;
                    self.parse_funcs(reader)?;
                }

                Payload::TableSection(reader) => {
                    validator.table_section(&reader)?;
                    self.parse_tables(reader)?;
                }

                Payload::MemorySection(reader) => {
                    validator.memory_section(&reader)?;
                    self.parse_memories(reader)?;
                }

                Payload::GlobalSection(reader) => {
                    validator.global_section(&reader)?;
                    self.parse_globals(reader)?;
                }

                Payload::ExportSection(reader) => {
                    validator.export_section(&reader)?;
                    self.parse_exports(reader)?;
                }

                Payload::StartSection { func, range } => {
                    validator.start_section(func, &range)?;
                    self.parse_start(func)?;
                }

                Payload::ElementSection(reader) => {
                    validator.element_section(&reader)?;
                    self.parse_elements(reader)?;
                }

                Payload::DataSection(reader) => {
                    validator.data_section(&reader)?;
                    self.parse_data(reader)?;
                }

                Payload::CodeSectionStart { count, range, .. } => {
                    validator.code_section_start(count, &range)?;
                }

                Payload::CodeSectionEntry(func) => {
                    validator
                        .code_section_entry(&func)?
                        .into_validator(Default::default())
                        .validate(&func)?;
                    let idx = self.import_count.funcs + code_sections;
                    code_sections += 1;
                    self.parse_code(self.funcs[idx], func)?;
                }

                Payload::CustomSection(_) => {}

                Payload::End(offset) => {
                    validator.end(offset)?;
                }

                payload => {
                    validator.payload(&payload)?;
                }
            }
        }

        Ok(self.module)
    }

    fn parse_types(&mut self, reader: wasmparser::TypeSectionReader<'_>) -> Result<()> {
        for rec_group in reader {
            for SubType { composite_type, .. } in rec_group?.into_types() {
                let CompositeType::Func(func_ty) = composite_type else {
                    unreachable!()
                };

                let params = func_ty
                    .params()
                    .iter()
                    .copied()
                    .map(make_val_type)
                    .collect();
                let ret = func_ty.results().get(0).copied().map(make_val_type);
                let ty = ir::ty::Type::Func(ir::ty::FuncType { params, ret });
                self.add_ty(ty);
            }
        }

        Ok(())
    }

    fn parse_imports(&mut self, reader: wasmparser::ImportSectionReader<'_>) -> Result<()> {
        for import in reader {
            let wasmparser::Import { module, name, ty } = import?;

            self.add_import(ir::Import {
                module: module.to_owned(),
                name: name.to_owned(),
                desc: match ty {
                    wasmparser::TypeRef::Func(idx) => {
                        ir::ImportDesc::Func(self.types[idx as usize])
                    }
                    wasmparser::TypeRef::Table(table_ty) => {
                        ir::ImportDesc::Table(make_table_type(table_ty))
                    }
                    wasmparser::TypeRef::Memory(mem_ty) => {
                        ir::ImportDesc::Memory(make_mem_type(mem_ty))
                    }
                    wasmparser::TypeRef::Global(global_ty) => {
                        ir::ImportDesc::Global(make_global_type(global_ty))
                    }
                    wasmparser::TypeRef::Tag(_) => unreachable!("unsupported type {ty:?}"),
                },
            })?;
        }

        Ok(())
    }

    fn parse_funcs(&mut self, reader: wasmparser::FunctionSectionReader<'_>) -> Result<()> {
        for func_ty_idx in reader {
            let ty = self.module.types[self.types[func_ty_idx? as usize]]
                .as_func()
                .clone();
            let mut body = ir::FuncBody::new(ty);

            for param_ty in &body.ty.params {
                let local_id = body.locals.insert(param_ty.clone());
                body.params.push(local_id);
            }

            self.add_func(ir::Func::Body(body));
        }

        Ok(())
    }

    fn parse_tables(&mut self, reader: wasmparser::TableSectionReader<'_>) -> Result<()> {
        for table in reader {
            let table = table?;
            let table_ty = make_table_type(table.ty);
            assert!(matches!(table.init, wasmparser::TableInit::RefNull));

            self.add_table(ir::Table::new(table_ty));
        }

        Ok(())
    }

    fn parse_memories(&mut self, reader: wasmparser::MemorySectionReader<'_>) -> Result<()> {
        for memory_type in reader {
            let ty = make_mem_type(memory_type?);
            let size = ty.limits.min as usize;

            self.add_mem(ir::Memory {
                ty,
                def: ir::MemoryDef::Bytes(vec![0; size]),
            });
        }

        Ok(())
    }

    fn parse_globals(&mut self, reader: wasmparser::GlobalSectionReader<'_>) -> Result<()> {
        for global in reader {
            let global = global?;
            let ty = make_global_type(global.ty);
            let [expr] = self
                .parse_expr(None, true, global.init_expr.get_operators_reader())?
                .try_into()
                .unwrap();
            let def = ir::GlobalDef::Value(expr);

            self.add_global(ir::Global { ty, def });
        }

        Ok(())
    }

    fn parse_exports(&mut self, reader: wasmparser::ExportSectionReader<'_>) -> Result<()> {
        for export in reader {
            let export = export?;
            let idx = export.index as usize;

            let def = match export.kind {
                ExternalKind::Func => ir::ExportDef::Func(self.funcs[idx]),
                ExternalKind::Table => ir::ExportDef::Table(self.tables[idx]),
                ExternalKind::Memory => ir::ExportDef::Memory(self.mems[idx]),
                ExternalKind::Global => ir::ExportDef::Global(self.globals[idx]),
                _ => unreachable!(),
            };

            self.add_export(ir::Export {
                name: export.name.to_owned(),
                def,
            });
        }

        Ok(())
    }

    fn parse_start(&mut self, func_idx: u32) -> Result<()> {
        self.module.start = Some(self.funcs[func_idx as usize]);

        Ok(())
    }

    fn parse_elements(&mut self, reader: wasmparser::ElementSectionReader<'_>) -> Result<()> {
        for (segment_idx, elem) in reader.into_iter().enumerate() {
            let elem = elem?;

            let wasmparser::ElementKind::Active {
                table_index,
                offset_expr,
            } = elem.kind
            else {
                unreachable!();
            };
            let table_idx = table_index.unwrap_or(0);
            let [offset] = self
                .parse_expr(None, true, offset_expr.get_operators_reader())?
                .try_into()
                .unwrap();
            let offset = offset
                .as_u32()
                .expect("table offset expr must have type i32");

            let table = &mut self.module.tables[self.tables[table_idx as usize]];
            let elems = match table.def {
                ir::TableDef::Elems(ref mut elems) => elems,
                _ => unreachable!("table imports are not supported"),
            };

            let wasmparser::ElementItems::Functions(func_reader) = elem.items else {
                unreachable!("unsupported table content type");
            };

            for (i, func_idx) in func_reader.into_iter().enumerate() {
                let func = self.funcs[func_idx? as usize];
                let idx = offset as usize + i;

                if idx >= elems.len() {
                    return Err(ParseError::TableElemOutOfRange {
                        element_idx: i,
                        segment_idx,
                        idx,
                        table_idx: table_idx as usize,
                    });
                }

                elems[idx] = Some(func);
            }
        }

        Ok(())
    }

    fn parse_data(&mut self, reader: wasmparser::DataSectionReader<'_>) -> Result<()> {
        for (segment_idx, data) in reader.into_iter().enumerate() {
            let data = data?;
            let wasmparser::DataKind::Active {
                memory_index: mem_idx,
                offset_expr,
            } = data.kind
            else {
                unreachable!();
            };

            let mem_idx = mem_idx as usize;
            let [offset] = self
                .parse_expr(None, true, offset_expr.get_operators_reader())?
                .try_into()
                .unwrap();
            let offset = offset.as_u32().expect("data segment offset must be an i32") as usize;
            let range = offset..(data.data.len() + offset as usize);

            let ir::MemoryDef::Bytes(ref mut bytes) = self.module.mems[self.mems[mem_idx]].def
            else {
                unreachable!("memory imports are not supported");
            };

            if range.end > bytes.len() {
                return Err(ParseError::DataOutOfRange {
                    segment_idx,
                    range,
                    mem_idx,
                    mem_size: bytes.len(),
                });
            }

            bytes[range].copy_from_slice(data.data);
        }

        Ok(())
    }

    fn parse_code(&mut self, func_id: FuncId, body: wasmparser::FunctionBody<'_>) -> Result<()> {
        let locals_reader = body.get_locals_reader()?;
        let ops_reader = body.get_operators_reader()?;
        let body = self.module.funcs[func_id].body_mut().unwrap();
        let mut locals = body.params.clone();

        for local in locals_reader {
            let (count, val_type) = local?;
            let ty = make_val_type(val_type);

            for _ in 0..count {
                locals.push(body.locals.insert(ty.clone()));
            }
        }

        let has_return = body.ty.ret.is_some();
        let body = self.parse_expr(Some((&locals, func_id)), has_return, ops_reader)?;
        self.module.funcs[func_id].body_mut().unwrap().body = body;

        Ok(())
    }

    fn parse_expr(
        &mut self,
        func: Option<(&[LocalId], FuncId)>,
        has_return: bool,
        reader: wasmparser::OperatorsReader<'_>,
    ) -> Result<Vec<ir::Expr>> {
        use ir::Expr;

        #[derive(Debug)]
        struct Block {
            branch: u8,
            has_return: bool,
            exprs: Vec<Expr>,
        }

        impl Block {
            fn main(has_return: bool) -> Self {
                Self {
                    branch: 0,
                    has_return,
                    exprs: vec![],
                }
            }

            fn branch(branch: u8, has_return: bool) -> Self {
                Self {
                    branch,
                    has_return,
                    exprs: vec![],
                }
            }
        }

        struct Context<'a> {
            parser: &'a mut Parser,
            blocks: Vec<Block>,
            func: Option<(&'a [LocalId], FuncId)>,
        }

        fn exprs(blocks: &mut Vec<Block>) -> &mut Vec<ir::Expr> {
            &mut blocks.last_mut().unwrap().exprs
        }

        impl Context<'_> {
            fn local(&self, idx: u32) -> LocalId {
                self.func.unwrap().0[idx as usize]
            }

            fn push_expr(&mut self, expr: ir::Expr) {
                exprs(&mut self.blocks).push(expr);
            }

            fn drop_all(&mut self) {
                let mut dropped = vec![];

                while let Some(expr) = self.maybe_pop_expr() {
                    dropped.push(Expr::Drop(Box::new(expr)));
                }

                for expr in dropped {
                    self.push_expr(expr);
                }
            }

            fn capture_br_expr(&mut self, relative_depth: u32) -> Option<Box<ir::Expr>> {
                let expr = self.blocks[self.blocks.len() - relative_depth as usize - 1]
                    .has_return
                    .then(|| Box::new(self.pop_expr()));
                self.drop_all();

                expr
            }

            fn maybe_pop_expr(&mut self) -> Option<ir::Expr> {
                let start_idx = exprs(&mut self.blocks)
                    .iter()
                    .enumerate()
                    .rfind(|(_, expr)| match expr.ret_value_count() {
                        ReturnValueCount::Zero => false,
                        ReturnValueCount::One => true,

                        ReturnValueCount::Call(func) => {
                            self.parser.module.funcs[func].ty().ret.is_some()
                        }

                        ReturnValueCount::CallIndirect(ty) => {
                            self.parser.module.types[ty].as_func().ret.is_some()
                        }

                        ReturnValueCount::Unreachable => {
                            panic!("cannot pop_expr after an unreachable instruction")
                        }
                    })
                    .map(|(idx, _)| idx)?;

                if start_idx + 1 == exprs(&mut self.blocks).len() {
                    Some(exprs(&mut self.blocks).pop().unwrap())
                } else {
                    let ty = match exprs(&mut self.blocks)[start_idx].ty() {
                        ExprTy::Concrete(ty) => ty,
                        ExprTy::Local(local) => self.parser.module.funcs[self.func.unwrap().1]
                            .body()
                            .unwrap()
                            .locals[local]
                            .clone(),

                        ExprTy::Global(global) => {
                            self.parser.module.globals[global].ty.val_type.clone()
                        }

                        ExprTy::Call(func) => {
                            self.parser.module.funcs[func].ty().ret.clone().unwrap()
                        }

                        ExprTy::CallIndirect(ty) => {
                            self.parser.module.types[ty].as_func().ret.clone().unwrap()
                        }

                        ExprTy::Unreachable | ExprTy::Empty => unreachable!(),
                    };

                    Some(Expr::Block(
                        Some(ty),
                        exprs(&mut self.blocks).split_off(start_idx),
                    ))
                }
            }

            fn pop_expr(&mut self) -> ir::Expr {
                self.maybe_pop_expr()
                    .expect("no expr in the block produces a value")
            }

            fn pop_expr2<F, R>(&mut self, f: F) -> R
            where
                F: FnOnce(Box<ir::Expr>, Box<ir::Expr>) -> R,
            {
                let v1 = Box::new(self.pop_expr());
                let v0 = Box::new(self.pop_expr());

                f(v0, v1)
            }

            fn push_block(&mut self, expr: Expr, block: Block) {
                exprs(&mut self.blocks).push(expr);
                self.blocks.push(block);
            }

            fn pop_block(&mut self) -> Block {
                let mut block = self.blocks.pop().unwrap();

                let exprs = match exprs(&mut self.blocks).last_mut().unwrap() {
                    Expr::Block(_, exprs) => exprs,
                    Expr::Loop(_, exprs) => exprs,
                    Expr::If(_, _, exprs, _) if block.branch == 0 => exprs,
                    Expr::If(_, _, _, exprs) => exprs,
                    _ => unreachable!(),
                };
                *exprs = mem::take(&mut block.exprs);

                block
            }
        }

        let mut ctx = Context {
            parser: self,
            blocks: vec![Block::main(has_return)],
            func,
        };

        let mut unreachable_level = 0u32;

        for op in reader {
            let expr = match op? {
                op @ (Operator::Else | Operator::End) => {
                    unreachable_level = unreachable_level.saturating_sub(1);
                    let block = ctx.pop_block();

                    if matches!(op, Operator::Else) {
                        ctx.blocks.push(Block::branch(1, block.has_return));
                    }

                    continue;
                }

                Operator::Block { .. } | Operator::Loop { .. } | Operator::If { .. }
                    if unreachable_level > 0 =>
                {
                    unreachable_level += 1;

                    continue;
                }

                _ if unreachable_level > 0 => continue,

                Operator::Block { blockty } => {
                    let ty = make_block_type(blockty);
                    let has_return = ty.is_some();
                    ctx.push_block(Expr::Block(ty, vec![]), Block::main(has_return));

                    continue;
                }

                Operator::Loop { blockty } => {
                    let ty = make_block_type(blockty);
                    let has_return = ty.is_some();
                    ctx.push_block(Expr::Loop(ty, vec![]), Block::main(has_return));

                    continue;
                }

                Operator::If { blockty } => {
                    let condition = Box::new(ctx.pop_expr());
                    let ty = make_block_type(blockty);
                    let has_return = ty.is_some();
                    ctx.push_block(Expr::If(ty, condition, vec![], vec![]), Block::main(has_return));

                    continue;
                }

                Operator::Unreachable => {
                    ctx.drop_all();

                    Expr::Unreachable
                }

                Operator::Nop => Expr::Nop,

                Operator::Br { relative_depth } => {
                    Expr::Br(relative_depth, ctx.capture_br_expr(relative_depth))
                }

                Operator::BrIf { relative_depth } => {
                    let condition = Box::new(ctx.pop_expr());
                    let ret_expr = ctx.capture_br_expr(relative_depth);

                    Expr::BrIf(relative_depth, condition, ret_expr)
                }

                Operator::BrTable { targets } => {
                    let condition = Box::new(ctx.pop_expr());
                    let ret_expr = ctx.capture_br_expr(targets.default());

                    Expr::BrTable(
                        targets.targets().collect::<Result<Vec<_>, _>>()?,
                        targets.default(),
                        condition,
                        ret_expr,
                    )
                }

                Operator::Return => Expr::Return(ctx.capture_br_expr(0)),

                Operator::Call { function_index } => {
                    let func_id = ctx.parser.funcs[function_index as usize];
                    let func_ty = ctx.parser.module.funcs[func_id].ty();
                    let args = (0..func_ty.params.len()).map(|_| ctx.pop_expr()).collect();

                    Expr::Call(func_id, args)
                }

                Operator::CallIndirect { type_index, .. } => {
                    let type_id = ctx.parser.types[type_index as usize];
                    let param_count = ctx.parser.module.types[type_id].as_func().params.len();
                    let idx_expr = ctx.pop_expr();

                    let mut args = (0..param_count).map(|_| ctx.pop_expr()).collect::<Vec<_>>();
                    args.reverse();

                    Expr::CallIndirect(type_id, Box::new(idx_expr), args)
                }

                Operator::Drop => Expr::Drop(Box::new(ctx.pop_expr())),

                Operator::Select => {
                    let condition = Box::new(ctx.pop_expr());
                    let v1 = Box::new(ctx.pop_expr());
                    let v0 = Box::new(ctx.pop_expr());

                    Expr::Select(v0, v1, condition)
                }

                Operator::LocalGet { local_index } => Expr::LocalGet(ctx.local(local_index)),

                Operator::LocalSet { local_index } => {
                    Expr::LocalSet(ctx.local(local_index), Box::new(ctx.pop_expr()))
                }

                Operator::LocalTee { local_index } => {
                    Expr::LocalTee(ctx.local(local_index), Box::new(ctx.pop_expr()))
                }

                Operator::GlobalGet { global_index } => {
                    Expr::GlobalGet(ctx.parser.globals[global_index as usize])
                }

                Operator::GlobalSet { global_index } => Expr::GlobalSet(
                    ctx.parser.globals[global_index as usize],
                    Box::new(ctx.pop_expr()),
                ),

                Operator::I32Load { memarg } => {
                    Expr::I32Load(make_mem_arg(memarg), Box::new(ctx.pop_expr()))
                }

                Operator::I64Load { memarg } => {
                    Expr::I64Load(make_mem_arg(memarg), Box::new(ctx.pop_expr()))
                }

                Operator::F32Load { memarg } => {
                    Expr::F32Load(make_mem_arg(memarg), Box::new(ctx.pop_expr()))
                }

                Operator::F64Load { memarg } => {
                    Expr::F64Load(make_mem_arg(memarg), Box::new(ctx.pop_expr()))
                }

                Operator::I32Load8S { memarg } => {
                    Expr::I32Load8S(make_mem_arg(memarg), Box::new(ctx.pop_expr()))
                }

                Operator::I32Load8U { memarg } => {
                    Expr::I32Load8U(make_mem_arg(memarg), Box::new(ctx.pop_expr()))
                }

                Operator::I32Load16S { memarg } => {
                    Expr::I32Load16S(make_mem_arg(memarg), Box::new(ctx.pop_expr()))
                }

                Operator::I32Load16U { memarg } => {
                    Expr::I32Load16U(make_mem_arg(memarg), Box::new(ctx.pop_expr()))
                }

                Operator::I64Load8S { memarg } => {
                    Expr::I64Load8S(make_mem_arg(memarg), Box::new(ctx.pop_expr()))
                }

                Operator::I64Load8U { memarg } => {
                    Expr::I64Load8U(make_mem_arg(memarg), Box::new(ctx.pop_expr()))
                }

                Operator::I64Load16S { memarg } => {
                    Expr::I64Load16S(make_mem_arg(memarg), Box::new(ctx.pop_expr()))
                }

                Operator::I64Load16U { memarg } => {
                    Expr::I64Load16U(make_mem_arg(memarg), Box::new(ctx.pop_expr()))
                }

                Operator::I64Load32S { memarg } => {
                    Expr::I64Load32S(make_mem_arg(memarg), Box::new(ctx.pop_expr()))
                }

                Operator::I64Load32U { memarg } => {
                    Expr::I64Load32U(make_mem_arg(memarg), Box::new(ctx.pop_expr()))
                }

                Operator::I32Store { memarg } => {
                    let value = Box::new(ctx.pop_expr());
                    let offset = Box::new(ctx.pop_expr());

                    Expr::I32Store(make_mem_arg(memarg), offset, value)
                }

                Operator::I64Store { memarg } => {
                    let value = Box::new(ctx.pop_expr());
                    let offset = Box::new(ctx.pop_expr());

                    Expr::I64Store(make_mem_arg(memarg), offset, value)
                }

                Operator::F32Store { memarg } => {
                    let value = Box::new(ctx.pop_expr());
                    let offset = Box::new(ctx.pop_expr());

                    Expr::F32Store(make_mem_arg(memarg), offset, value)
                }

                Operator::F64Store { memarg } => {
                    let value = Box::new(ctx.pop_expr());
                    let offset = Box::new(ctx.pop_expr());

                    Expr::F64Store(make_mem_arg(memarg), offset, value)
                }

                Operator::I32Store8 { memarg } => {
                    let value = Box::new(ctx.pop_expr());
                    let offset = Box::new(ctx.pop_expr());

                    Expr::I32Store8(make_mem_arg(memarg), offset, value)
                }

                Operator::I32Store16 { memarg } => {
                    let value = Box::new(ctx.pop_expr());
                    let offset = Box::new(ctx.pop_expr());

                    Expr::I32Store16(make_mem_arg(memarg), offset, value)
                }

                Operator::I64Store8 { memarg } => {
                    let value = Box::new(ctx.pop_expr());
                    let offset = Box::new(ctx.pop_expr());

                    Expr::I64Store8(make_mem_arg(memarg), offset, value)
                }

                Operator::I64Store16 { memarg } => {
                    let value = Box::new(ctx.pop_expr());
                    let offset = Box::new(ctx.pop_expr());

                    Expr::I64Store16(make_mem_arg(memarg), offset, value)
                }

                Operator::I64Store32 { memarg } => {
                    let value = Box::new(ctx.pop_expr());
                    let offset = Box::new(ctx.pop_expr());

                    Expr::I64Store32(make_mem_arg(memarg), offset, value)
                }

                Operator::MemorySize { .. } => Expr::MemorySize,
                Operator::MemoryGrow { .. } => Expr::MemoryGrow(Box::new(ctx.pop_expr())),

                Operator::I32Const { value } => Expr::I32(value),
                Operator::I64Const { value } => Expr::I64(value),
                Operator::F32Const { value } => Expr::F32(f32::from_bits(value.bits())),
                Operator::F64Const { value } => Expr::F64(f64::from_bits(value.bits())),

                Operator::I32Eqz => Expr::I32Eqz(Box::new(ctx.pop_expr())),
                Operator::I32Eq => ctx.pop_expr2(Expr::I32Eq),
                Operator::I32Ne => ctx.pop_expr2(Expr::I32Ne),
                Operator::I32LtS => ctx.pop_expr2(Expr::I32LtS),
                Operator::I32LtU => ctx.pop_expr2(Expr::I32LtU),
                Operator::I32GtS => ctx.pop_expr2(Expr::I32GtS),
                Operator::I32GtU => ctx.pop_expr2(Expr::I32GtU),
                Operator::I32LeS => ctx.pop_expr2(Expr::I32LeS),
                Operator::I32LeU => ctx.pop_expr2(Expr::I32LeU),
                Operator::I32GeS => ctx.pop_expr2(Expr::I32GeS),
                Operator::I32GeU => ctx.pop_expr2(Expr::I32GeU),

                Operator::I64Eqz => Expr::I64Eqz(Box::new(ctx.pop_expr())),
                Operator::I64Eq => ctx.pop_expr2(Expr::I64Eq),
                Operator::I64Ne => ctx.pop_expr2(Expr::I64Ne),
                Operator::I64LtS => ctx.pop_expr2(Expr::I64LtS),
                Operator::I64LtU => ctx.pop_expr2(Expr::I64LtU),
                Operator::I64GtS => ctx.pop_expr2(Expr::I64GtS),
                Operator::I64GtU => ctx.pop_expr2(Expr::I64GtU),
                Operator::I64LeS => ctx.pop_expr2(Expr::I64LeS),
                Operator::I64LeU => ctx.pop_expr2(Expr::I64LeU),
                Operator::I64GeS => ctx.pop_expr2(Expr::I64GeS),
                Operator::I64GeU => ctx.pop_expr2(Expr::I64GeU),

                Operator::F32Eq => ctx.pop_expr2(Expr::F32Eq),
                Operator::F32Ne => ctx.pop_expr2(Expr::F32Ne),
                Operator::F32Lt => ctx.pop_expr2(Expr::F32Lt),
                Operator::F32Gt => ctx.pop_expr2(Expr::F32Gt),
                Operator::F32Le => ctx.pop_expr2(Expr::F32Le),
                Operator::F32Ge => ctx.pop_expr2(Expr::F32Ge),

                Operator::F64Eq => ctx.pop_expr2(Expr::F64Eq),
                Operator::F64Ne => ctx.pop_expr2(Expr::F64Ne),
                Operator::F64Lt => ctx.pop_expr2(Expr::F64Lt),
                Operator::F64Gt => ctx.pop_expr2(Expr::F64Gt),
                Operator::F64Le => ctx.pop_expr2(Expr::F64Le),
                Operator::F64Ge => ctx.pop_expr2(Expr::F64Ge),

                Operator::I32Clz => Expr::I32Clz(Box::new(ctx.pop_expr())),
                Operator::I32Ctz => Expr::I32Ctz(Box::new(ctx.pop_expr())),
                Operator::I32Popcnt => Expr::I32Popcnt(Box::new(ctx.pop_expr())),
                Operator::I32Add => ctx.pop_expr2(Expr::I32Add),
                Operator::I32Sub => ctx.pop_expr2(Expr::I32Sub),
                Operator::I32Mul => ctx.pop_expr2(Expr::I32Mul),
                Operator::I32DivS => ctx.pop_expr2(Expr::I32DivS),
                Operator::I32DivU => ctx.pop_expr2(Expr::I32DivU),
                Operator::I32RemS => ctx.pop_expr2(Expr::I32RemS),
                Operator::I32RemU => ctx.pop_expr2(Expr::I32RemU),
                Operator::I32And => ctx.pop_expr2(Expr::I32And),
                Operator::I32Or => ctx.pop_expr2(Expr::I32Or),
                Operator::I32Xor => ctx.pop_expr2(Expr::I32Xor),
                Operator::I32Shl => ctx.pop_expr2(Expr::I32Shl),
                Operator::I32ShrS => ctx.pop_expr2(Expr::I32ShrS),
                Operator::I32ShrU => ctx.pop_expr2(Expr::I32ShrU),
                Operator::I32Rotl => ctx.pop_expr2(Expr::I32Rotl),
                Operator::I32Rotr => ctx.pop_expr2(Expr::I32Rotr),

                Operator::I64Clz => Expr::I64Clz(Box::new(ctx.pop_expr())),
                Operator::I64Ctz => Expr::I64Ctz(Box::new(ctx.pop_expr())),
                Operator::I64Popcnt => Expr::I64Popcnt(Box::new(ctx.pop_expr())),
                Operator::I64Add => ctx.pop_expr2(Expr::I64Add),
                Operator::I64Sub => ctx.pop_expr2(Expr::I64Sub),
                Operator::I64Mul => ctx.pop_expr2(Expr::I64Mul),
                Operator::I64DivS => ctx.pop_expr2(Expr::I64DivS),
                Operator::I64DivU => ctx.pop_expr2(Expr::I64DivU),
                Operator::I64RemS => ctx.pop_expr2(Expr::I64RemS),
                Operator::I64RemU => ctx.pop_expr2(Expr::I64RemU),
                Operator::I64And => ctx.pop_expr2(Expr::I64And),
                Operator::I64Or => ctx.pop_expr2(Expr::I64Or),
                Operator::I64Xor => ctx.pop_expr2(Expr::I64Xor),
                Operator::I64Shl => ctx.pop_expr2(Expr::I64Shl),
                Operator::I64ShrS => ctx.pop_expr2(Expr::I64ShrS),
                Operator::I64ShrU => ctx.pop_expr2(Expr::I64ShrU),
                Operator::I64Rotl => ctx.pop_expr2(Expr::I64Rotl),
                Operator::I64Rotr => ctx.pop_expr2(Expr::I64Rotr),

                Operator::F32Abs => Expr::F32Abs(Box::new(ctx.pop_expr())),
                Operator::F32Neg => Expr::F32Neg(Box::new(ctx.pop_expr())),
                Operator::F32Ceil => Expr::F32Ceil(Box::new(ctx.pop_expr())),
                Operator::F32Floor => Expr::F32Floor(Box::new(ctx.pop_expr())),
                Operator::F32Trunc => Expr::F32Trunc(Box::new(ctx.pop_expr())),
                Operator::F32Nearest => Expr::F32Nearest(Box::new(ctx.pop_expr())),
                Operator::F32Sqrt => Expr::F32Sqrt(Box::new(ctx.pop_expr())),
                Operator::F32Add => ctx.pop_expr2(Expr::F32Add),
                Operator::F32Sub => ctx.pop_expr2(Expr::F32Sub),
                Operator::F32Mul => ctx.pop_expr2(Expr::F32Mul),
                Operator::F32Div => ctx.pop_expr2(Expr::F32Div),
                Operator::F32Min => ctx.pop_expr2(Expr::F32Min),
                Operator::F32Max => ctx.pop_expr2(Expr::F32Max),
                Operator::F32Copysign => ctx.pop_expr2(Expr::F32Copysign),

                Operator::F64Abs => Expr::F64Abs(Box::new(ctx.pop_expr())),
                Operator::F64Neg => Expr::F64Neg(Box::new(ctx.pop_expr())),
                Operator::F64Ceil => Expr::F64Ceil(Box::new(ctx.pop_expr())),
                Operator::F64Floor => Expr::F64Floor(Box::new(ctx.pop_expr())),
                Operator::F64Trunc => Expr::F64Trunc(Box::new(ctx.pop_expr())),
                Operator::F64Nearest => Expr::F64Nearest(Box::new(ctx.pop_expr())),
                Operator::F64Sqrt => Expr::F64Sqrt(Box::new(ctx.pop_expr())),
                Operator::F64Add => ctx.pop_expr2(Expr::F64Add),
                Operator::F64Sub => ctx.pop_expr2(Expr::F64Sub),
                Operator::F64Mul => ctx.pop_expr2(Expr::F64Mul),
                Operator::F64Div => ctx.pop_expr2(Expr::F64Div),
                Operator::F64Min => ctx.pop_expr2(Expr::F64Min),
                Operator::F64Max => ctx.pop_expr2(Expr::F64Max),
                Operator::F64Copysign => ctx.pop_expr2(Expr::F64Copysign),

                Operator::I32WrapI64 => Expr::I32WrapI64(Box::new(ctx.pop_expr())),
                Operator::I32TruncF32S => Expr::I32TruncF32S(Box::new(ctx.pop_expr())),
                Operator::I32TruncF32U => Expr::I32TruncF32U(Box::new(ctx.pop_expr())),
                Operator::I32TruncF64S => Expr::I32TruncF64S(Box::new(ctx.pop_expr())),
                Operator::I32TruncF64U => Expr::I32TruncF64U(Box::new(ctx.pop_expr())),

                Operator::I64ExtendI32S => Expr::I64ExtendI32S(Box::new(ctx.pop_expr())),
                Operator::I64ExtendI32U => Expr::I64ExtendI32U(Box::new(ctx.pop_expr())),
                Operator::I64TruncF32S => Expr::I64TruncF32S(Box::new(ctx.pop_expr())),
                Operator::I64TruncF32U => Expr::I64TruncF32U(Box::new(ctx.pop_expr())),
                Operator::I64TruncF64S => Expr::I32TruncF64S(Box::new(ctx.pop_expr())),
                Operator::I64TruncF64U => Expr::I64TruncF64U(Box::new(ctx.pop_expr())),

                Operator::F32ConvertI32S => Expr::F32ConvertI32S(Box::new(ctx.pop_expr())),
                Operator::F32ConvertI32U => Expr::F32ConvertI32U(Box::new(ctx.pop_expr())),
                Operator::F32ConvertI64S => Expr::F32ConvertI64S(Box::new(ctx.pop_expr())),
                Operator::F32ConvertI64U => Expr::F32ConvertI64U(Box::new(ctx.pop_expr())),
                Operator::F32DemoteF64 => Expr::F32DemoteF64(Box::new(ctx.pop_expr())),

                Operator::F64ConvertI32S => Expr::F64ConvertI32S(Box::new(ctx.pop_expr())),
                Operator::F64ConvertI32U => Expr::F64ConvertI32U(Box::new(ctx.pop_expr())),
                Operator::F64ConvertI64S => Expr::F64ConvertI64S(Box::new(ctx.pop_expr())),
                Operator::F64ConvertI64U => Expr::F64ConvertI64U(Box::new(ctx.pop_expr())),
                Operator::F64PromoteF32 => Expr::F64PromoteF32(Box::new(ctx.pop_expr())),

                Operator::I32ReinterpretF32 => Expr::I32ReinterpretF32(Box::new(ctx.pop_expr())),
                Operator::I64ReinterpretF64 => Expr::I64ReinterpretF64(Box::new(ctx.pop_expr())),
                Operator::F32ReinterpretI32 => Expr::F32ReinterpretI32(Box::new(ctx.pop_expr())),
                Operator::F64ReinterpretI64 => Expr::F64ReinterpretI64(Box::new(ctx.pop_expr())),

                op => unreachable!("unsupported operation {op:?}"),
            };

            unreachable_level = (expr.ret_value_count() == ReturnValueCount::Unreachable) as u32;
            ctx.push_expr(expr);
        }

        assert_eq!(ctx.blocks.len(), 1);

        Ok(ctx.blocks.pop().unwrap().exprs)
    }
}

pub fn parse(bytes: &[u8]) -> Result<ir::Module> {
    Parser::default().parse(bytes)
}
