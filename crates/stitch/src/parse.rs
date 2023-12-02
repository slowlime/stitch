use std::ops::Range;

use thiserror::Error;
use wasmparser::{BinaryReaderError, CompositeType, ExternalKind, Payload, SubType, WasmFeatures};

use crate::ir::{self, ExportId, FuncId, GlobalId, ImportId, MemoryId, TableId, TypeId};

const FEATURES: WasmFeatures = WasmFeatures {
    mutable_global: true,
    saturating_float_to_int: true,
    sign_extension: true,
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

fn make_elem_type(ref_type: wasmparser::RefType) -> ir::ty::ElemType {
    assert!(
        ref_type.is_func_ref(),
        "unsupported table element type {ref_type}"
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

#[derive(Debug, Default)]
struct Parser {
    module: ir::Module,
    types: Vec<TypeId>,
    funcs: Vec<FuncId>,
    tables: Vec<TableId>,
    mems: Vec<MemoryId>,
    globals: Vec<GlobalId>,
    imports: Vec<ImportId>,
}

pub type Result<T, E = ParseError> = ::std::result::Result<T, E>;

impl Parser {
    fn add_ty(&mut self, ty: ir::ty::Type) -> TypeId {
        let id = self.module.types.insert(ty);
        self.types.push(id);

        id
    }

    fn add_func(&mut self, func: ir::func::Func) -> FuncId {
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

        match &import.desc {
            ir::ImportDesc::Func(ty_idx) => {
                self.add_func(ir::func::Func::Import(ir::func::FuncImport {
                    ty: match &self.module.types[*ty_idx] {
                        ir::ty::Type::Func(ty) => ty.clone(),
                    },
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
                    todo!("parse the function body")
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
            let ty = match &self.module.types[self.types[func_ty_idx? as usize]] {
                ir::ty::Type::Func(func_ty) => func_ty.clone(),
            };

            self.add_func(ir::func::Func::Body(ir::func::FuncBody::new(ty)));
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
            let expr = self.parse_expr(global.init_expr.get_operators_reader())?;
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
            let offset = self
                .parse_expr(offset_expr.get_operators_reader())?
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
            let offset = self
                .parse_expr(offset_expr.get_operators_reader())?
                .as_u32()
                .expect("data segment offset must be an i32") as usize;
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

    fn parse_expr(&mut self, reader: wasmparser::OperatorsReader<'_>) -> Result<ir::expr::Expr> {
        todo!()
    }
}

pub fn parse(bytes: &[u8]) -> Result<ir::Module> {
    Parser::default().parse(bytes)
}
