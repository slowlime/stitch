use thiserror::Error;
use wasmparser::{BinaryReaderError, CompositeType, Payload, SubType, WasmFeatures};

use crate::ir::{self, TypeId, FuncId, TableId, MemoryId, GlobalId, ImportId};

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
    assert!(ref_type.is_func_ref(), "unsupported table element type {ref_type}");

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

pub type Result<T> = ::std::result::Result<T, ParseError>;

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

            desc => return Err(ParseError::UnsupportedImport {
                kind: desc.kind(),
                module: import.module.clone(),
                name: import.name.clone(),
            }),
        }

        Ok(id)
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
                    todo!("parse functions")
                }

                Payload::TableSection(reader) => {
                    validator.table_section(&reader)?;
                    todo!("parse tables")
                }

                Payload::MemorySection(reader) => {
                    validator.memory_section(&reader)?;
                    todo!("parse memory contents")
                }

                Payload::GlobalSection(reader) => {
                    validator.global_section(&reader)?;
                    todo!("parse globals")
                }

                Payload::ExportSection(reader) => {
                    validator.export_section(&reader)?;
                    todo!("parse exports")
                }

                Payload::StartSection { func, range } => {
                    validator.start_section(func, &range)?;
                    todo!("parse the start section")
                }

                Payload::ElementSection(reader) => {
                    validator.element_section(&reader)?;
                    todo!("parse elements")
                }

                Payload::DataCountSection { count, range } => {
                    validator.data_count_section(count, &range)?;
                    todo!("???")
                }

                Payload::DataSection(reader) => {
                    validator.data_section(&reader)?;
                    todo!("parse data")
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

    fn parse_types(&mut self, reader: wasmparser::TypeSectionReader) -> Result<()> {
        for rec_group in reader {
            for SubType { composite_type, .. } in rec_group?.into_types() {
                let CompositeType::Func(func_ty) = composite_type else {
                    unreachable!()
                };

                let params = func_ty.params().iter().copied().map(make_val_type).collect();
                let ret = func_ty.results().get(0).copied().map(make_val_type);
                let ty = ir::ty::Type::Func(ir::ty::FuncType { params, ret });
                self.add_ty(ty);
            }
        }

        Ok(())
    }

    fn parse_imports(&mut self, reader: wasmparser::ImportSectionReader) -> Result<()> {
        for import in reader {
            let wasmparser::Import { module, name, ty } = import?;

            self.add_import(ir::Import {
                module: module.to_owned(),
                name: name.to_owned(),
                desc: match ty {
                    wasmparser::TypeRef::Func(idx) => ir::ImportDesc::Func(self.types[idx as usize]),
                    wasmparser::TypeRef::Table(table_ty) => ir::ImportDesc::Table(make_table_type(table_ty)),
                    wasmparser::TypeRef::Memory(mem_ty) => ir::ImportDesc::Memory(make_mem_type(mem_ty)),
                    wasmparser::TypeRef::Global(global_ty) => ir::ImportDesc::Global(make_global_type(global_ty)),
                    wasmparser::TypeRef::Tag(_) => unreachable!("unsupported type {ty:?}"),
                },
            })?;
        }

        Ok(())
    }
}

pub fn parse(bytes: &[u8]) -> Result<ir::Module> {
    Parser::default().parse(bytes)
}
