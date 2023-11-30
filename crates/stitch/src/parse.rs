use wasmparser::{WasmFeatures, BinaryReaderError, Payload};
use thiserror::Error;

use crate::ir;

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
}

pub fn parse(bytes: &[u8]) -> Result<ir::Module, ParseError> {
    let parser = wasmparser::Parser::new(0);
    let mut validator = wasmparser::Validator::new_with_features(FEATURES);

    for payload in parser.parse_all(bytes) {
        match payload? {
            Payload::Version { num, encoding, range } => {
                validator.version(num, encoding, &range)?;
            }

            Payload::TypeSection(reader) => {
                validator.type_section(&reader)?;
                todo!("parse types")
            }

            Payload::ImportSection(reader) => {
                validator.import_section(&reader)?;
                todo!("parse imports")
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
                validator.code_section_entry(&func)?.into_validator(Default::default()).validate(&func)?;
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

    todo!()
}
