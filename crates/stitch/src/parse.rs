use std::mem;
use std::ops::Range;

use log::{log_enabled, trace, warn};
use slotmap::{Key, SecondaryMap, SlotMap};
use thiserror::Error;
use wasmparser::{
    BinaryReaderError, CompositeType, ExternalKind, IndirectNaming, Name, NameSectionReader,
    Naming, Operator, Payload, SubType, WasmFeatures,
};

use crate::ast::expr::{BinOp, ExprTy, NulOp, ReturnValueCount, TernOp, UnOp, Value};
use crate::ast::{
    self, BlockId, ExportId, FuncId, GlobalId, ImportDesc, ImportId, ImportKind, IntrinsicDecl,
    LocalId, MemoryId, TableId, TypeId, PAGE_SIZE, STITCH_MODULE_NAME,
};
use crate::util::float::{F32, F64};

const FEATURES: WasmFeatures = WasmFeatures {
    mutable_global: true,
    saturating_float_to_int: false,
    sign_extension: true,
    reference_types: false,
    multi_value: false,
    bulk_memory: true,
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
        kind: ast::ImportKind,
        module: String,
        name: String,
    },

    #[error(
        "element #{element_idx} (segment #{segment_idx}) stored at #{idx} is out of range \
        for table #{table_idx} of size {table_size}"
    )]
    TableElemOutOfRange {
        element_idx: usize,
        segment_idx: usize,
        idx: usize,
        table_idx: usize,
        table_size: usize,
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

    #[error("unknown intrinsic {kind} stitch/{name}")]
    UnknownIntrinsic { kind: ImportKind, name: String },

    #[error("invalid type for intrinsic stitch/{intrinsic}: expected {expected}, got {actual}")]
    InvalidIntrinsicTy {
        intrinsic: IntrinsicDecl,
        expected: String,
        actual: ast::ty::FuncType,
    },

    #[error("element segment #{segment_idx} has an unsupported mode {mode}")]
    UnsupportedElementSegmentMode {
        segment_idx: usize,
        mode: &'static str,
    },

    #[error("data segment #{segment_idx} has an unsupported mode {mode}")]
    UnsupportedDataSegmentMode {
        segment_idx: usize,
        mode: &'static str,
    },

    #[error("unsupported instruction `{instr}` at byte 0x{offset:x}")]
    UnsupportedInstruction { instr: String, offset: usize },
}

fn make_val_type(val_ty: wasmparser::ValType) -> ast::ty::ValType {
    match val_ty {
        wasmparser::ValType::I32 => ast::ty::ValType::I32,
        wasmparser::ValType::I64 => ast::ty::ValType::I64,
        wasmparser::ValType::F32 => ast::ty::ValType::F32,
        wasmparser::ValType::F64 => ast::ty::ValType::F64,
        wasmparser::ValType::V128 | wasmparser::ValType::Ref(_) => {
            unreachable!("unsupported types")
        }
    }
}

fn make_elem_type(ref_ty: wasmparser::RefType) -> ast::ty::ElemType {
    assert!(
        ref_ty.is_func_ref(),
        "unsupported table element type {ref_ty}"
    );

    ast::ty::ElemType::Funcref
}

fn make_table_type(table_ty: wasmparser::TableType) -> ast::ty::TableType {
    let elem_ty = make_elem_type(table_ty.element_type);

    ast::ty::TableType {
        elem_ty,
        limits: ast::ty::Limits {
            min: table_ty.initial,
            max: table_ty.maximum,
        },
    }
}

fn make_mem_type(mem_ty: wasmparser::MemoryType) -> ast::ty::MemoryType {
    assert!(!mem_ty.memory64, "unsupported memory type {mem_ty:?}");

    ast::ty::MemoryType {
        limits: ast::ty::Limits {
            min: mem_ty.initial as _,
            max: mem_ty.maximum.map(|max| max as _),
        },
    }
}

fn make_global_type(global_ty: wasmparser::GlobalType) -> ast::ty::GlobalType {
    ast::ty::GlobalType {
        val_type: make_val_type(global_ty.content_type),
        mutable: global_ty.mutable,
    }
}

fn make_block_type(block_ty: wasmparser::BlockType) -> ast::ty::BlockType {
    match block_ty {
        wasmparser::BlockType::Empty => ast::ty::BlockType::Empty,
        wasmparser::BlockType::Type(ty) => ast::ty::BlockType::Result(make_val_type(ty)),
        _ => unreachable!("func block types are not supported"),
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
    module: ast::Module,
    types: Vec<TypeId>,
    funcs: Vec<FuncId>,
    locals: SecondaryMap<FuncId, Vec<LocalId>>,
    tables: Vec<TableId>,
    mems: Vec<MemoryId>,
    globals: Vec<GlobalId>,
    imports: Vec<ImportId>,

    import_count: ImportCount,
}

pub type Result<T, E = ParseError> = ::std::result::Result<T, E>;

impl Parser {
    fn add_ty(&mut self, ty: ast::ty::Type) -> TypeId {
        trace!("ty {}: {ty:?}", self.types.len());
        let id = self.module.types.insert(ty);
        self.types.push(id);

        id
    }

    fn add_func(&mut self, func: ast::Func) -> FuncId {
        let id = self.module.funcs.insert(func);
        trace!("added func {id:?}");
        self.funcs.push(id);

        id
    }

    fn add_table(&mut self, table: ast::Table) -> TableId {
        let id = self.module.tables.insert(table);
        trace!("added table {id:?} of type {:?}", self.module.tables[id].ty);
        self.tables.push(id);

        id
    }

    fn add_mem(&mut self, mem: ast::Memory) -> MemoryId {
        let id = self.module.mems.insert(mem);
        trace!("added mem {id:?} of type {:?}", self.module.mems[id].ty);
        self.mems.push(id);

        id
    }

    fn add_global(&mut self, global: ast::Global) -> GlobalId {
        let id = self.module.globals.insert(global);
        trace!(
            "added global {id:?} of type {:?}",
            self.module.globals[id].ty
        );
        self.globals.push(id);

        id
    }

    fn add_import(&mut self, import: ast::Import) -> Result<ImportId> {
        let id = self.module.imports.insert(import);
        self.imports.push(id);

        let import = &self.module.imports[id];

        match import.desc.kind() {
            ast::ImportKind::Func => self.import_count.funcs += 1,
            ast::ImportKind::Table => self.import_count.tables += 1,
            ast::ImportKind::Memory => self.import_count.mems += 1,
            ast::ImportKind::Global => self.import_count.globals += 1,
        }

        match &import.desc {
            ast::ImportDesc::Func(ty_idx) => {
                self.add_func(ast::Func::Import(ast::func::FuncImport {
                    name: None,
                    param_names: Default::default(),
                    ty: self.module.types[*ty_idx].as_func().clone(),
                    import_id: id,
                }));
            }

            ast::ImportDesc::Global(global_ty) => {
                self.add_global(ast::Global {
                    name: None,
                    ty: global_ty.clone(),
                    def: ast::GlobalDef::Import(id),
                });
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

    fn add_export(&mut self, export: ast::Export) -> ExportId {
        self.module.exports.insert(export)
    }

    fn check_intrinsic_types(&self) -> Result<()> {
        for (import_id, import) in &self.module.imports {
            if import.module != STITCH_MODULE_NAME {
                continue;
            }

            let Some(intrinsic) = self.module.get_intrinsic(import_id) else {
                return Err(ParseError::UnknownIntrinsic {
                    kind: import.desc.kind(),
                    name: import.name.clone(),
                });
            };

            let ImportDesc::Func(ty_id) = import.desc else {
                return Err(ParseError::UnknownIntrinsic {
                    kind: import.desc.kind(),
                    name: import.name.clone(),
                });
            };

            let func_ty = self.module.types[ty_id].as_func();

            match intrinsic.check_ty(func_ty) {
                Ok(()) => {}

                Err(expected) => {
                    return Err(ParseError::InvalidIntrinsicTy {
                        intrinsic,
                        expected,
                        actual: func_ty.clone(),
                    });
                }
            }
        }

        Ok(())
    }

    fn parse(mut self, bytes: &[u8]) -> Result<ast::Module> {
        let parser = wasmparser::Parser::new(0);
        let mut validator = wasmparser::Validator::new_with_features(FEATURES);
        let mut code_section_entries: usize = 0;

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
                    let idx = self.import_count.funcs + code_section_entries;
                    code_section_entries += 1;

                    trace!("parsing func {idx}: {:x?}", func.range());
                    self.parse_code(self.funcs[idx], func.clone())?;

                    validator
                        .code_section_entry(&func)?
                        .into_validator(Default::default())
                        .validate(&func)?;
                }

                Payload::CustomSection(reader) if reader.name() == "name" => {
                    self.parse_names(NameSectionReader::new(reader.data(), reader.data_offset()))?;
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

        self.module.default_table = self.tables.get(0).copied().unwrap_or_default();
        self.module.default_mem = self.mems.get(0).copied().unwrap_or_default();

        self.check_intrinsic_types()?;

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
                let ty = ast::ty::Type::Func(ast::ty::FuncType { params, ret });
                self.add_ty(ty);
            }
        }

        Ok(())
    }

    fn parse_imports(&mut self, reader: wasmparser::ImportSectionReader<'_>) -> Result<()> {
        for import in reader {
            let wasmparser::Import { module, name, ty } = import?;

            self.add_import(ast::Import {
                module: module.to_owned(),
                name: name.to_owned(),
                desc: match ty {
                    wasmparser::TypeRef::Func(idx) => {
                        ast::ImportDesc::Func(self.types[idx as usize])
                    }
                    wasmparser::TypeRef::Table(table_ty) => {
                        ast::ImportDesc::Table(make_table_type(table_ty))
                    }
                    wasmparser::TypeRef::Memory(mem_ty) => {
                        ast::ImportDesc::Memory(make_mem_type(mem_ty))
                    }
                    wasmparser::TypeRef::Global(global_ty) => {
                        ast::ImportDesc::Global(make_global_type(global_ty))
                    }
                    wasmparser::TypeRef::Tag(_) => unreachable!("unsupported type {ty:?}"),
                },
            })?;
        }

        Ok(())
    }

    fn parse_funcs(&mut self, reader: wasmparser::FunctionSectionReader<'_>) -> Result<()> {
        let mut idx = self.funcs.len();
        for result in reader {
            let func_ty_idx = result?;
            let ty = self.module.types[self.types[func_ty_idx as usize]]
                .as_func()
                .clone();
            trace!("func {idx}: ({func_ty_idx}) {ty:?}");
            idx += 1;
            let mut body = ast::FuncBody::new(ty);

            for param_ty in &body.ty.params {
                let local_id = body.locals.insert(param_ty.clone());
                body.params.push(local_id);
            }

            self.add_func(ast::Func::Body(body));
        }

        Ok(())
    }

    fn parse_tables(&mut self, reader: wasmparser::TableSectionReader<'_>) -> Result<()> {
        for table in reader {
            let table = table?;
            let table_ty = make_table_type(table.ty);
            assert!(matches!(table.init, wasmparser::TableInit::RefNull));

            self.add_table(ast::Table::new(table_ty));
        }

        Ok(())
    }

    fn parse_memories(&mut self, reader: wasmparser::MemorySectionReader<'_>) -> Result<()> {
        for memory_type in reader {
            let ty = make_mem_type(memory_type?);
            let pages = ty.limits.min as usize;

            self.add_mem(ast::Memory {
                name: None,
                ty,
                def: ast::MemoryDef::Bytes(vec![0; pages * PAGE_SIZE]),
            });
        }

        Ok(())
    }

    fn parse_globals(&mut self, reader: wasmparser::GlobalSectionReader<'_>) -> Result<()> {
        for global in reader {
            let global = global?;
            let ty = make_global_type(global.ty);
            let [expr] = self
                .parse_expr(
                    None,
                    &mut Default::default(),
                    true,
                    global.init_expr.get_operators_reader(),
                )?
                .body
                .try_into()
                .unwrap();
            let def = ast::GlobalDef::Value(expr.try_into().unwrap());

            self.add_global(ast::Global {
                name: None,
                ty,
                def,
            });
        }

        Ok(())
    }

    fn parse_exports(&mut self, reader: wasmparser::ExportSectionReader<'_>) -> Result<()> {
        for export in reader {
            let export = export?;
            let idx = export.index as usize;

            let def = match export.kind {
                ExternalKind::Func => ast::ExportDef::Func(self.funcs[idx]),
                ExternalKind::Table => ast::ExportDef::Table(self.tables[idx]),
                ExternalKind::Memory => ast::ExportDef::Memory(self.mems[idx]),
                ExternalKind::Global => ast::ExportDef::Global(self.globals[idx]),
                _ => unreachable!(),
            };

            self.add_export(ast::Export {
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
                return Err(ParseError::UnsupportedElementSegmentMode {
                    segment_idx,
                    mode: match elem.kind {
                        wasmparser::ElementKind::Active { .. } => unreachable!(),
                        wasmparser::ElementKind::Passive => "passive",
                        wasmparser::ElementKind::Declared => "declarative",
                    },
                });
            };
            let table_idx = table_index.unwrap_or(0);
            let [offset] = self
                .parse_expr(
                    None,
                    &mut Default::default(),
                    true,
                    offset_expr.get_operators_reader(),
                )?
                .body
                .try_into()
                .unwrap();
            let offset = offset
                .to_u32()
                .expect("table offset expr must have type i32");

            let table = &mut self.module.tables[self.tables[table_idx as usize]];
            let elems = match table.def {
                ast::TableDef::Elems(ref mut elems) => elems,
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
                        table_size: elems.len(),
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
                return Err(ParseError::UnsupportedDataSegmentMode {
                    segment_idx,
                    mode: match data.kind {
                        wasmparser::DataKind::Active { .. } => unreachable!(),
                        wasmparser::DataKind::Passive => "passive",
                    },
                });
            };

            let mem_idx = mem_idx as usize;
            let [offset] = self
                .parse_expr(
                    None,
                    &mut Default::default(),
                    true,
                    offset_expr.get_operators_reader(),
                )?
                .body
                .try_into()
                .unwrap();
            let offset = offset.to_u32().expect("data segment offset must be an i32") as usize;
            let range = offset..(data.data.len() + offset as usize);

            let ast::MemoryDef::Bytes(ref mut bytes) = self.module.mems[self.mems[mem_idx]].def
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

        trace!("locals:");

        for (idx, local) in locals.iter().enumerate() {
            trace!("  {idx:>2}: {:?}", body.locals[*local]);
        }

        let has_return = body.ty.ret.is_some();
        let mut blocks = Default::default();
        let body = self.parse_expr(
            Some((&locals, func_id)),
            &mut blocks,
            has_return,
            ops_reader,
        )?;
        let func_body = self.module.funcs[func_id].body_mut().unwrap();
        func_body.main_block = body;
        func_body.blocks = blocks;
        self.locals.insert(func_id, locals);

        Ok(())
    }

    fn parse_names(&mut self, reader: wasmparser::NameSectionReader<'_>) -> Result<()> {
        fn naming<K: Key>(
            reader: wasmparser::NameMap<'_>,
            ids: &[K],
            what: &str,
            mut f: impl FnMut(K, &str),
        ) -> Result<()> {
            for name in reader {
                let Naming { index, name } = name?;

                if let Some(&id) = ids.get(index as usize) {
                    f(id, name);
                } else {
                    warn!("the name section contains a name for a non-existent {what} #{index}");
                }
            }

            Ok(())
        }

        for name in reader {
            match name? {
                Name::Module { name, .. } => {
                    self.module.name = Some(name.into());
                }

                Name::Function(names) => {
                    naming(names, &self.funcs, "function", |func_id, name| {
                        self.module.funcs[func_id].set_name(Some(name.into()));
                    })?;
                }

                Name::Local(names) => {
                    for names in names {
                        let IndirectNaming {
                            index: func_index,
                            names,
                        } = names?;
                        let Some(&func_id) = self.funcs.get(func_index as usize) else {
                            warn!(
                                "the name section contains a name for \
                                a non-existent function #{func_index}",
                            );
                            continue;
                        };

                        for name in names {
                            let Naming { index, name } = name?;

                            match &mut self.module.funcs[func_id] {
                                ast::Func::Import(import) => {
                                    if let Some(param_name) =
                                        import.param_names.get_mut(index as usize)
                                    {
                                        param_name.replace(name.into());
                                        continue;
                                    }
                                }

                                ast::Func::Body(body) => {
                                    if let Some(&local_id) =
                                        self.locals[func_id].get(index as usize)
                                    {
                                        body.local_names.insert(local_id, name.into());
                                        continue;
                                    }
                                }
                            }

                            warn!(
                                "the name section contains a name for a non-existent \
                                local #{index} of function #{func_index} ({})",
                                self.module.func_name(func_id),
                            );
                        }
                    }
                }

                Name::Table(names) => {
                    naming(names, &self.tables, "table", |table_id, name| {
                        self.module.tables[table_id].name = Some(name.into());
                    })?;
                }

                Name::Memory(names) => {
                    naming(names, &self.mems, "memory", |mem_id, name| {
                        self.module.mems[mem_id].name = Some(name.into());
                    })?;
                }

                Name::Global(names) => {
                    naming(names, &self.globals, "global", |global_id, name| {
                        self.module.globals[global_id].name = Some(name.into());
                    })?;
                }

                _ => {}
            }
        }

        Ok(())
    }

    fn parse_expr(
        &mut self,
        func: Option<(&[LocalId], FuncId)>,
        blocks: &mut SlotMap<BlockId, ()>,
        has_return: bool,
        reader: wasmparser::OperatorsReader<'_>,
    ) -> Result<ast::expr::Block> {
        use ast::Expr;

        #[derive(Debug)]
        struct Block {
            branch: u8,
            br_captures_expr: bool,
            block_id: BlockId,
            exprs: Vec<Expr>,
        }

        impl Block {
            fn main(block_id: BlockId, br_captures_expr: bool) -> Block {
                Self {
                    branch: 0,
                    br_captures_expr,
                    block_id,
                    exprs: vec![],
                }
            }

            fn branch(block_id: BlockId, branch: u8, br_captures_expr: bool) -> Block {
                Self {
                    branch,
                    br_captures_expr,
                    block_id,
                    exprs: vec![],
                }
            }
        }

        struct Context<'a> {
            parser: &'a mut Parser,
            blocks: &'a mut SlotMap<BlockId, ()>,
            block_stack: Vec<Block>,
            func: Option<(&'a [LocalId], FuncId)>,
            body: Option<Vec<Expr>>,
        }

        fn exprs(blocks: &mut Vec<Block>) -> &mut Vec<ast::Expr> {
            &mut blocks.last_mut().unwrap().exprs
        }

        impl Context<'_> {
            fn local(&self, idx: u32) -> LocalId {
                self.func.unwrap().0[idx as usize]
            }

            fn push_expr(&mut self, expr: ast::Expr) {
                exprs(&mut self.block_stack).push(expr);
            }

            fn drop_all(&mut self) {
                let mut dropped = vec![];

                while let Some(expr) = self.maybe_pop_expr() {
                    dropped.push(Expr::Unary(UnOp::Drop, Box::new(expr)));
                }

                for expr in dropped {
                    self.push_expr(expr);
                }
            }

            fn get_block_id(&self, relative_depth: u32) -> BlockId {
                self.block_stack[self.block_stack.len() - relative_depth as usize - 1].block_id
            }

            fn pop_br_expr(&mut self, relative_depth: u32) -> Option<Box<ast::Expr>> {
                self.block_stack[self.block_stack.len() - relative_depth as usize - 1]
                    .br_captures_expr
                    .then(|| Box::new(self.pop_expr()))
            }

            fn capture_br_expr(&mut self, relative_depth: u32) -> Option<Box<ast::Expr>> {
                let expr = self.pop_br_expr(relative_depth);
                self.drop_all();

                expr
            }

            fn maybe_pop_expr(&mut self) -> Option<ast::Expr> {
                let start_idx = exprs(&mut self.block_stack)
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

                if start_idx + 1 == exprs(&mut self.block_stack).len() {
                    Some(exprs(&mut self.block_stack).pop().unwrap())
                } else {
                    let ty = match exprs(&mut self.block_stack)[start_idx].ty() {
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

                    let block_id = self.blocks.insert(());

                    Some(Expr::Block(
                        ast::ty::BlockType::Result(ty),
                        ast::expr::Block {
                            body: exprs(&mut self.block_stack).split_off(start_idx),
                            id: block_id,
                        },
                    ))
                }
            }

            fn pop_expr(&mut self) -> Expr {
                self.maybe_pop_expr()
                    .expect("no expr in the block produces a value")
            }

            fn un_expr(&mut self, op: UnOp) -> Expr {
                Expr::Unary(op, Box::new(self.pop_expr()))
            }

            fn bin_expr(&mut self, op: BinOp) -> Expr {
                let rhs = Box::new(self.pop_expr());
                let lhs = Box::new(self.pop_expr());

                Expr::Binary(op, [lhs, rhs])
            }

            fn tern_expr(&mut self, op: TernOp) -> Expr {
                let e2 = Box::new(self.pop_expr());
                let e1 = Box::new(self.pop_expr());
                let e0 = Box::new(self.pop_expr());

                Expr::Ternary(op, [e0, e1, e2])
            }

            fn push_block(&mut self, expr: Expr, block: Block) {
                exprs(&mut self.block_stack).push(expr);
                self.block_stack.push(block);
            }

            fn pop_block(&mut self) -> Block {
                let mut block = self.block_stack.pop().unwrap();

                if self.block_stack.is_empty() {
                    self.body = Some(mem::take(&mut block.exprs));

                    return block;
                }

                let parent_block = match exprs(&mut self.block_stack).last_mut().unwrap() {
                    Expr::Block(_, parent_block) => parent_block,
                    Expr::Loop(_, parent_block) => parent_block,
                    Expr::If(_, _, parent_block, _) if block.branch == 0 => parent_block,
                    Expr::If(_, _, _, parent_block) => parent_block,
                    _ => unreachable!(),
                };
                parent_block.body = mem::take(&mut block.exprs);

                block
            }
        }

        let root_block_id = blocks.insert(());

        let mut ctx = Context {
            parser: self,
            blocks,
            block_stack: vec![Block::main(root_block_id, has_return)],
            func,
            body: None,
        };

        let mut unreachable_level = 0u32;

        for op_offset in reader.into_iter_with_offsets() {
            let (op, offset) = op_offset?;

            if log_enabled!(log::Level::Trace) {
                let indent = ":   ".repeat(ctx.block_stack.len());

                trace!(
                    "{indent}{:<2} {op:?} [{unreachable_level}] @ {offset}",
                    ctx.block_stack.len()
                );

                if let Operator::Call { function_index } = op {
                    trace!(
                        "{indent}   func ty: {:?}",
                        ctx.parser.module.funcs[ctx.parser.funcs[function_index as usize]].ty()
                    );
                }
            }

            let expr = match op {
                op @ (Operator::Else | Operator::End) => {
                    unreachable_level = unreachable_level.saturating_sub(1);
                    let block = ctx.pop_block();

                    if matches!(op, Operator::Else) {
                        let else_block_id = match exprs(&mut ctx.block_stack).last().unwrap() {
                            Expr::If(_, _, _, else_block) => else_block.id,
                            _ => unreachable!(),
                        };
                        ctx.block_stack.push(Block::branch(
                            else_block_id,
                            1,
                            block.br_captures_expr,
                        ));
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
                    let br_captures_expr = !ty.is_empty();
                    let block_id = ctx.blocks.insert(());
                    ctx.push_block(
                        Expr::Block(
                            ty,
                            ast::expr::Block {
                                body: vec![],
                                id: block_id,
                            },
                        ),
                        Block::main(block_id, br_captures_expr),
                    );

                    continue;
                }

                Operator::Loop { blockty } => {
                    let ty = make_block_type(blockty);
                    let block_id = ctx.blocks.insert(());

                    // a branch to a loop restarts the loop without popping any values off the stack
                    let br_captures_expr = false;

                    ctx.push_block(
                        Expr::Loop(
                            ty,
                            ast::expr::Block {
                                body: vec![],
                                id: block_id,
                            },
                        ),
                        Block::main(block_id, br_captures_expr),
                    );

                    continue;
                }

                Operator::If { blockty } => {
                    let condition = Box::new(ctx.pop_expr());
                    let ty = make_block_type(blockty);
                    let br_captures_expr = !ty.is_empty();
                    let then_block_id = ctx.blocks.insert(());
                    let else_block_id = ctx.blocks.insert(());
                    ctx.push_block(
                        Expr::If(
                            ty,
                            condition,
                            ast::expr::Block {
                                body: vec![],
                                id: then_block_id,
                            },
                            ast::expr::Block {
                                body: vec![],
                                id: else_block_id,
                            },
                        ),
                        Block::main(then_block_id, br_captures_expr),
                    );

                    continue;
                }

                Operator::Unreachable => {
                    ctx.drop_all();

                    NulOp::Unreachable.into()
                }

                Operator::Nop => NulOp::Nop.into(),

                Operator::Br { relative_depth } => Expr::Br(
                    ctx.get_block_id(relative_depth),
                    ctx.capture_br_expr(relative_depth),
                ),

                Operator::BrIf { relative_depth } => {
                    let condition = Box::new(ctx.pop_expr());
                    let ret_expr = ctx.pop_br_expr(relative_depth);

                    Expr::BrIf(ctx.get_block_id(relative_depth), ret_expr, condition)
                }

                Operator::BrTable { targets } => {
                    let condition = Box::new(ctx.pop_expr());
                    let ret_expr = ctx.capture_br_expr(targets.default());

                    Expr::BrTable(
                        targets
                            .targets()
                            .map(|relative_depth| {
                                relative_depth
                                    .map(|relative_depth| ctx.get_block_id(relative_depth))
                            })
                            .collect::<Result<Vec<_>, _>>()?,
                        ctx.get_block_id(targets.default()),
                        ret_expr,
                        condition,
                    )
                }

                Operator::Return => {
                    Expr::Return(ctx.capture_br_expr(ctx.block_stack.len() as u32 - 1))
                }

                Operator::Call { function_index } => {
                    let func_id = ctx.parser.funcs[function_index as usize];
                    let func_ty = ctx.parser.module.funcs[func_id].ty();

                    let mut args = (0..func_ty.params.len())
                        .map(|_| ctx.pop_expr())
                        .collect::<Vec<_>>();
                    args.reverse();

                    Expr::Call(func_id, args)
                }

                Operator::CallIndirect {
                    type_index,
                    table_index,
                    ..
                } => {
                    let type_id = ctx.parser.types[type_index as usize];
                    let table_id = ctx.parser.tables[table_index as usize];
                    let param_count = ctx.parser.module.types[type_id].as_func().params.len();
                    let idx_expr = ctx.pop_expr();

                    let mut args = (0..param_count).map(|_| ctx.pop_expr()).collect::<Vec<_>>();
                    args.reverse();

                    Expr::CallIndirect(type_id, table_id, args, Box::new(idx_expr))
                }

                Operator::Drop => ctx.un_expr(UnOp::Drop),
                Operator::Select => ctx.tern_expr(TernOp::Select),

                Operator::LocalGet { local_index } => {
                    NulOp::LocalGet(ctx.local(local_index)).into()
                }
                Operator::LocalSet { local_index } => {
                    ctx.un_expr(UnOp::LocalSet(ctx.local(local_index)))
                }
                Operator::LocalTee { local_index } => {
                    ctx.un_expr(UnOp::LocalTee(ctx.local(local_index)))
                }

                Operator::GlobalGet { global_index } => {
                    NulOp::GlobalGet(ctx.parser.globals[global_index as usize]).into()
                }

                Operator::GlobalSet { global_index } => {
                    ctx.un_expr(UnOp::GlobalSet(ctx.parser.globals[global_index as usize]))
                }

                Operator::I32Load { memarg } => {
                    ctx.un_expr(UnOp::I32Load(ctx.parser.make_mem_arg(memarg)))
                }
                Operator::I64Load { memarg } => {
                    ctx.un_expr(UnOp::I64Load(ctx.parser.make_mem_arg(memarg)))
                }
                Operator::F32Load { memarg } => {
                    ctx.un_expr(UnOp::F32Load(ctx.parser.make_mem_arg(memarg)))
                }
                Operator::F64Load { memarg } => {
                    ctx.un_expr(UnOp::F64Load(ctx.parser.make_mem_arg(memarg)))
                }

                Operator::I32Load8S { memarg } => {
                    ctx.un_expr(UnOp::I32Load8S(ctx.parser.make_mem_arg(memarg)))
                }
                Operator::I32Load8U { memarg } => {
                    ctx.un_expr(UnOp::I32Load8U(ctx.parser.make_mem_arg(memarg)))
                }
                Operator::I32Load16S { memarg } => {
                    ctx.un_expr(UnOp::I32Load16S(ctx.parser.make_mem_arg(memarg)))
                }
                Operator::I32Load16U { memarg } => {
                    ctx.un_expr(UnOp::I32Load16U(ctx.parser.make_mem_arg(memarg)))
                }

                Operator::I64Load8S { memarg } => {
                    ctx.un_expr(UnOp::I64Load8S(ctx.parser.make_mem_arg(memarg)))
                }
                Operator::I64Load8U { memarg } => {
                    ctx.un_expr(UnOp::I64Load8U(ctx.parser.make_mem_arg(memarg)))
                }
                Operator::I64Load16S { memarg } => {
                    ctx.un_expr(UnOp::I64Load16S(ctx.parser.make_mem_arg(memarg)))
                }
                Operator::I64Load16U { memarg } => {
                    ctx.un_expr(UnOp::I64Load16U(ctx.parser.make_mem_arg(memarg)))
                }
                Operator::I64Load32S { memarg } => {
                    ctx.un_expr(UnOp::I64Load32S(ctx.parser.make_mem_arg(memarg)))
                }
                Operator::I64Load32U { memarg } => {
                    ctx.un_expr(UnOp::I64Load32U(ctx.parser.make_mem_arg(memarg)))
                }

                Operator::I32Store { memarg } => {
                    ctx.bin_expr(BinOp::I32Store(ctx.parser.make_mem_arg(memarg)))
                }
                Operator::I64Store { memarg } => {
                    ctx.bin_expr(BinOp::I64Store(ctx.parser.make_mem_arg(memarg)))
                }
                Operator::F32Store { memarg } => {
                    ctx.bin_expr(BinOp::F32Store(ctx.parser.make_mem_arg(memarg)))
                }
                Operator::F64Store { memarg } => {
                    ctx.bin_expr(BinOp::F64Store(ctx.parser.make_mem_arg(memarg)))
                }

                Operator::I32Store8 { memarg } => {
                    ctx.bin_expr(BinOp::I32Store8(ctx.parser.make_mem_arg(memarg)))
                }
                Operator::I32Store16 { memarg } => {
                    ctx.bin_expr(BinOp::I32Store16(ctx.parser.make_mem_arg(memarg)))
                }

                Operator::I64Store8 { memarg } => {
                    ctx.bin_expr(BinOp::I64Store8(ctx.parser.make_mem_arg(memarg)))
                }
                Operator::I64Store16 { memarg } => {
                    ctx.bin_expr(BinOp::I64Store16(ctx.parser.make_mem_arg(memarg)))
                }
                Operator::I64Store32 { memarg } => {
                    ctx.bin_expr(BinOp::I64Store32(ctx.parser.make_mem_arg(memarg)))
                }

                Operator::MemorySize { mem, .. } => {
                    NulOp::MemorySize(ctx.parser.mems[mem as usize]).into()
                }
                Operator::MemoryGrow { mem, .. } => {
                    ctx.un_expr(UnOp::MemoryGrow(ctx.parser.mems[mem as usize]))
                }

                Operator::I32Const { value } => Value::I32(value).into(),
                Operator::I64Const { value } => Value::I64(value).into(),
                Operator::F32Const { value } => Value::F32(F32::from_bits(value.bits())).into(),
                Operator::F64Const { value } => Value::F64(F64::from_bits(value.bits())).into(),

                Operator::I32Eqz => ctx.un_expr(UnOp::I32Eqz),
                Operator::I32Eq => ctx.bin_expr(BinOp::I32Eq),
                Operator::I32Ne => ctx.bin_expr(BinOp::I32Ne),
                Operator::I32LtS => ctx.bin_expr(BinOp::I32LtS),
                Operator::I32LtU => ctx.bin_expr(BinOp::I32LtU),
                Operator::I32GtS => ctx.bin_expr(BinOp::I32GtS),
                Operator::I32GtU => ctx.bin_expr(BinOp::I32GtU),
                Operator::I32LeS => ctx.bin_expr(BinOp::I32LeS),
                Operator::I32LeU => ctx.bin_expr(BinOp::I32LeU),
                Operator::I32GeS => ctx.bin_expr(BinOp::I32GeS),
                Operator::I32GeU => ctx.bin_expr(BinOp::I32GeU),

                Operator::I64Eqz => ctx.un_expr(UnOp::I64Eqz),
                Operator::I64Eq => ctx.bin_expr(BinOp::I64Eq),
                Operator::I64Ne => ctx.bin_expr(BinOp::I64Ne),
                Operator::I64LtS => ctx.bin_expr(BinOp::I64LtS),
                Operator::I64LtU => ctx.bin_expr(BinOp::I64LtU),
                Operator::I64GtS => ctx.bin_expr(BinOp::I64GtS),
                Operator::I64GtU => ctx.bin_expr(BinOp::I64GtU),
                Operator::I64LeS => ctx.bin_expr(BinOp::I64LeS),
                Operator::I64LeU => ctx.bin_expr(BinOp::I64LeU),
                Operator::I64GeS => ctx.bin_expr(BinOp::I64GeS),
                Operator::I64GeU => ctx.bin_expr(BinOp::I64GeU),

                Operator::F32Eq => ctx.bin_expr(BinOp::F32Eq),
                Operator::F32Ne => ctx.bin_expr(BinOp::F32Ne),
                Operator::F32Lt => ctx.bin_expr(BinOp::F32Lt),
                Operator::F32Gt => ctx.bin_expr(BinOp::F32Gt),
                Operator::F32Le => ctx.bin_expr(BinOp::F32Le),
                Operator::F32Ge => ctx.bin_expr(BinOp::F32Ge),

                Operator::F64Eq => ctx.bin_expr(BinOp::F64Eq),
                Operator::F64Ne => ctx.bin_expr(BinOp::F64Ne),
                Operator::F64Lt => ctx.bin_expr(BinOp::F64Lt),
                Operator::F64Gt => ctx.bin_expr(BinOp::F64Gt),
                Operator::F64Le => ctx.bin_expr(BinOp::F64Le),
                Operator::F64Ge => ctx.bin_expr(BinOp::F64Ge),

                Operator::I32Clz => ctx.un_expr(UnOp::I32Clz),
                Operator::I32Ctz => ctx.un_expr(UnOp::I32Ctz),
                Operator::I32Popcnt => ctx.un_expr(UnOp::I32Popcnt),
                Operator::I32Add => ctx.bin_expr(BinOp::I32Add),
                Operator::I32Sub => ctx.bin_expr(BinOp::I32Sub),
                Operator::I32Mul => ctx.bin_expr(BinOp::I32Mul),
                Operator::I32DivS => ctx.bin_expr(BinOp::I32DivS),
                Operator::I32DivU => ctx.bin_expr(BinOp::I32DivU),
                Operator::I32RemS => ctx.bin_expr(BinOp::I32RemS),
                Operator::I32RemU => ctx.bin_expr(BinOp::I32RemU),
                Operator::I32And => ctx.bin_expr(BinOp::I32And),
                Operator::I32Or => ctx.bin_expr(BinOp::I32Or),
                Operator::I32Xor => ctx.bin_expr(BinOp::I32Xor),
                Operator::I32Shl => ctx.bin_expr(BinOp::I32Shl),
                Operator::I32ShrS => ctx.bin_expr(BinOp::I32ShrS),
                Operator::I32ShrU => ctx.bin_expr(BinOp::I32ShrU),
                Operator::I32Rotl => ctx.bin_expr(BinOp::I32Rotl),
                Operator::I32Rotr => ctx.bin_expr(BinOp::I32Rotr),

                Operator::I64Clz => ctx.un_expr(UnOp::I64Clz),
                Operator::I64Ctz => ctx.un_expr(UnOp::I64Ctz),
                Operator::I64Popcnt => ctx.un_expr(UnOp::I64Popcnt),
                Operator::I64Add => ctx.bin_expr(BinOp::I64Add),
                Operator::I64Sub => ctx.bin_expr(BinOp::I64Sub),
                Operator::I64Mul => ctx.bin_expr(BinOp::I64Mul),
                Operator::I64DivS => ctx.bin_expr(BinOp::I64DivS),
                Operator::I64DivU => ctx.bin_expr(BinOp::I64DivU),
                Operator::I64RemS => ctx.bin_expr(BinOp::I64RemS),
                Operator::I64RemU => ctx.bin_expr(BinOp::I64RemU),
                Operator::I64And => ctx.bin_expr(BinOp::I64And),
                Operator::I64Or => ctx.bin_expr(BinOp::I64Or),
                Operator::I64Xor => ctx.bin_expr(BinOp::I64Xor),
                Operator::I64Shl => ctx.bin_expr(BinOp::I64Shl),
                Operator::I64ShrS => ctx.bin_expr(BinOp::I64ShrS),
                Operator::I64ShrU => ctx.bin_expr(BinOp::I64ShrU),
                Operator::I64Rotl => ctx.bin_expr(BinOp::I64Rotl),
                Operator::I64Rotr => ctx.bin_expr(BinOp::I64Rotr),

                Operator::F32Abs => ctx.un_expr(UnOp::F32Abs),
                Operator::F32Neg => ctx.un_expr(UnOp::F32Neg),
                Operator::F32Ceil => ctx.un_expr(UnOp::F32Ceil),
                Operator::F32Floor => ctx.un_expr(UnOp::F32Floor),
                Operator::F32Trunc => ctx.un_expr(UnOp::F32Trunc),
                Operator::F32Nearest => ctx.un_expr(UnOp::F32Nearest),
                Operator::F32Sqrt => ctx.un_expr(UnOp::F32Sqrt),
                Operator::F32Add => ctx.bin_expr(BinOp::F32Add),
                Operator::F32Sub => ctx.bin_expr(BinOp::F32Sub),
                Operator::F32Mul => ctx.bin_expr(BinOp::F32Mul),
                Operator::F32Div => ctx.bin_expr(BinOp::F32Div),
                Operator::F32Min => ctx.bin_expr(BinOp::F32Min),
                Operator::F32Max => ctx.bin_expr(BinOp::F32Max),
                Operator::F32Copysign => ctx.bin_expr(BinOp::F32Copysign),

                Operator::F64Abs => ctx.un_expr(UnOp::F64Abs),
                Operator::F64Neg => ctx.un_expr(UnOp::F64Neg),
                Operator::F64Ceil => ctx.un_expr(UnOp::F64Ceil),
                Operator::F64Floor => ctx.un_expr(UnOp::F64Floor),
                Operator::F64Trunc => ctx.un_expr(UnOp::F64Trunc),
                Operator::F64Nearest => ctx.un_expr(UnOp::F64Nearest),
                Operator::F64Sqrt => ctx.un_expr(UnOp::F64Sqrt),
                Operator::F64Add => ctx.bin_expr(BinOp::F64Add),
                Operator::F64Sub => ctx.bin_expr(BinOp::F64Sub),
                Operator::F64Mul => ctx.bin_expr(BinOp::F64Mul),
                Operator::F64Div => ctx.bin_expr(BinOp::F64Div),
                Operator::F64Min => ctx.bin_expr(BinOp::F64Min),
                Operator::F64Max => ctx.bin_expr(BinOp::F64Max),
                Operator::F64Copysign => ctx.bin_expr(BinOp::F64Copysign),

                Operator::I32WrapI64 => ctx.un_expr(UnOp::I32WrapI64),
                Operator::I32TruncF32S => ctx.un_expr(UnOp::I32TruncF32S),
                Operator::I32TruncF32U => ctx.un_expr(UnOp::I32TruncF32U),
                Operator::I32TruncF64S => ctx.un_expr(UnOp::I32TruncF64S),
                Operator::I32TruncF64U => ctx.un_expr(UnOp::I32TruncF64U),

                Operator::I64ExtendI32S => ctx.un_expr(UnOp::I64ExtendI32S),
                Operator::I64ExtendI32U => ctx.un_expr(UnOp::I64ExtendI32U),
                Operator::I64TruncF32S => ctx.un_expr(UnOp::I64TruncF32S),
                Operator::I64TruncF32U => ctx.un_expr(UnOp::I64TruncF32U),
                Operator::I64TruncF64S => ctx.un_expr(UnOp::I64TruncF64S),
                Operator::I64TruncF64U => ctx.un_expr(UnOp::I64TruncF64U),

                Operator::F32ConvertI32S => ctx.un_expr(UnOp::F32ConvertI32S),
                Operator::F32ConvertI32U => ctx.un_expr(UnOp::F32ConvertI32U),
                Operator::F32ConvertI64S => ctx.un_expr(UnOp::F32ConvertI64S),
                Operator::F32ConvertI64U => ctx.un_expr(UnOp::F32ConvertI64U),
                Operator::F32DemoteF64 => ctx.un_expr(UnOp::F32DemoteF64),

                Operator::F64ConvertI32S => ctx.un_expr(UnOp::F64ConvertI32S),
                Operator::F64ConvertI32U => ctx.un_expr(UnOp::F64ConvertI32U),
                Operator::F64ConvertI64S => ctx.un_expr(UnOp::F64ConvertI64S),
                Operator::F64ConvertI64U => ctx.un_expr(UnOp::F64ConvertI64U),
                Operator::F64PromoteF32 => ctx.un_expr(UnOp::F64PromoteF32),

                Operator::I32ReinterpretF32 => ctx.un_expr(UnOp::I32ReinterpretF32),
                Operator::I64ReinterpretF64 => ctx.un_expr(UnOp::I64ReinterpretF64),
                Operator::F32ReinterpretI32 => ctx.un_expr(UnOp::F32ReinterpretI32),
                Operator::F64ReinterpretI64 => ctx.un_expr(UnOp::F64ReinterpretI64),

                Operator::I32Extend8S => ctx.un_expr(UnOp::I32Extend8S),
                Operator::I32Extend16S => ctx.un_expr(UnOp::I32Extend16S),

                Operator::I64Extend8S => ctx.un_expr(UnOp::I64Extend8S),
                Operator::I64Extend16S => ctx.un_expr(UnOp::I64Extend16S),
                Operator::I64Extend32S => ctx.un_expr(UnOp::I64Extend32S),

                Operator::MemoryInit { .. } => {
                    return Err(ParseError::UnsupportedInstruction {
                        instr: "memory.init".into(),
                        offset,
                    })
                }
                Operator::DataDrop { .. } => {
                    return Err(ParseError::UnsupportedInstruction {
                        instr: "data.drop".into(),
                        offset,
                    })
                }
                Operator::MemoryCopy { dst_mem, src_mem } => ctx.tern_expr(TernOp::MemoryCopy {
                    dst_mem_id: ctx.parser.mems[dst_mem as usize],
                    src_mem_id: ctx.parser.mems[src_mem as usize],
                }),
                Operator::MemoryFill { mem } => ctx.tern_expr(TernOp::MemoryFill {
                    mem_id: ctx.parser.mems[mem as usize],
                }),
                Operator::TableInit { .. } => {
                    return Err(ParseError::UnsupportedInstruction {
                        instr: "table.init".into(),
                        offset,
                    })
                }
                Operator::ElemDrop { .. } => {
                    return Err(ParseError::UnsupportedInstruction {
                        instr: "elem.drop".into(),
                        offset,
                    })
                }
                Operator::TableCopy { .. } => {
                    return Err(ParseError::UnsupportedInstruction {
                        instr: "table.copy".into(),
                        offset,
                    })
                }

                op => unreachable!("unsupported operation {op:?}"),
            };

            unreachable_level = (expr.ret_value_count() == ReturnValueCount::Unreachable) as u32;
            ctx.push_expr(expr);
        }

        assert!(ctx.block_stack.is_empty());

        Ok(ast::expr::Block {
            body: ctx.body.unwrap(),
            id: root_block_id,
        })
    }

    fn make_mem_arg(&self, mem_arg: wasmparser::MemArg) -> ast::expr::MemArg {
        ast::expr::MemArg {
            mem_id: self.mems[mem_arg.memory as usize],
            offset: mem_arg.offset as u32,
            align: mem_arg.align as u32,
        }
    }
}

pub fn parse(bytes: &[u8]) -> Result<ast::Module> {
    Parser::default().parse(bytes)
}
