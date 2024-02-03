//! A simple AST-based IR for a WebAssembly module.

pub mod expr;
pub mod func;
pub mod ty;

use std::fmt::{self, Display};
use std::ops::Range;

use slotmap::{new_key_type, SlotMap};
use thiserror::Error;

use crate::util::slot::BiSlotMap;

use self::ty::{GlobalType, MemoryType, TableType, Type};

pub use self::expr::Expr;
pub use self::func::{Func, FuncBody};

const STITCH_MODULE_NAME: &str = "stitch";

new_key_type! {
    pub struct TypeId;
    pub struct FuncId;
    pub struct TableId;
    pub struct MemoryId;
    pub struct GlobalId;
    pub struct ImportId;
    pub struct ExportId;
    pub struct LocalId;
    pub struct BlockId;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntrinsicDecl {
    Specialize,
    Unknown,
}

impl Display for IntrinsicDecl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{STITCH_MODULE_NAME}/{}",
            match self {
                Self::Specialize => "specialize",
                Self::Unknown => "unknown",
            }
        )
    }
}

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum MemError {
    #[error("cannot access an imported memory")]
    Import,

    #[error("range 0x{:x}..0x{:x} is out of bounds for a memory of size {size}", .range.start, .range.end)]
    OutOfBounds { range: Range<usize>, size: usize },
}

#[derive(Debug, Default, Clone)]
pub struct Module {
    pub types: BiSlotMap<TypeId, Type>,
    pub funcs: SlotMap<FuncId, Func>,
    pub tables: SlotMap<TableId, Table>,
    pub mems: SlotMap<MemoryId, Memory>,
    pub globals: SlotMap<GlobalId, Global>,
    pub start: Option<FuncId>,
    pub imports: SlotMap<ImportId, Import>,
    pub exports: SlotMap<ExportId, Export>,
}

impl Module {
    pub fn get_intrinsic(&self, import_id: ImportId) -> Option<IntrinsicDecl> {
        let (name, desc) = match &self.imports[import_id] {
            Import { module, name, desc } if module == STITCH_MODULE_NAME => (name, desc),
            _ => return None,
        };

        Some(match (name.as_str(), desc) {
            ("specialize", ImportDesc::Func(_)) => IntrinsicDecl::Specialize,
            ("unknown", ImportDesc::Global(_)) => IntrinsicDecl::Unknown,
            _ => return None,
        })
    }

    pub fn read_mem(&self, mem_id: MemoryId, range: Range<usize>) -> Result<&[u8], MemError> {
        match &self.mems[mem_id].def {
            MemoryDef::Import(_) => Err(MemError::Import),

            MemoryDef::Bytes(bytes) => bytes.get(range.clone()).ok_or(MemError::OutOfBounds {
                range,
                size: bytes.len(),
            }),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Table {
    pub ty: TableType,
    pub def: TableDef,
}

impl Table {
    pub fn new(ty: TableType) -> Self {
        let elems = vec![None; ty.limits.min as usize];

        Self {
            ty,
            def: TableDef::Elems(elems),
        }
    }
}

#[derive(Debug, Clone)]
pub enum TableDef {
    Import(ImportId),
    Elems(Vec<Option<FuncId>>),
}

impl TableDef {
    pub fn is_import(&self) -> bool {
        matches!(self, Self::Import(_))
    }
}

#[derive(Debug, Clone)]
pub struct Memory {
    pub ty: MemoryType,
    pub def: MemoryDef,
}

#[derive(Debug, Clone)]
pub enum MemoryDef {
    Import(ImportId),
    Bytes(Vec<u8>),
}

impl MemoryDef {
    pub fn is_import(&self) -> bool {
        matches!(self, Self::Import(_))
    }
}

#[derive(Debug, Clone)]
pub struct Global {
    pub ty: GlobalType,
    pub def: GlobalDef,
}

#[derive(Debug, Clone)]
pub enum GlobalDef {
    Import(ImportId),
    Value(Expr),
}

impl GlobalDef {
    pub fn is_import(&self) -> bool {
        matches!(self, Self::Import(_))
    }

    pub fn get_intrinsic(&self, module: &Module) -> Option<IntrinsicDecl> {
        match *self {
            Self::Import(import_id) => module.get_intrinsic(import_id),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Import {
    pub module: String,
    pub name: String,
    pub desc: ImportDesc,
}

#[derive(Debug, Clone)]
pub enum ImportDesc {
    Func(TypeId),
    Table(TableType),
    Memory(MemoryType),
    Global(GlobalType),
}

impl ImportDesc {
    pub fn kind(&self) -> ImportKind {
        match self {
            Self::Func(_) => ImportKind::Func,
            Self::Table(_) => ImportKind::Table,
            Self::Memory(_) => ImportKind::Memory,
            Self::Global(_) => ImportKind::Global,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImportKind {
    Func,
    Table,
    Memory,
    Global,
}

impl Display for ImportKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Func => write!(f, "function"),
            Self::Table => write!(f, "table"),
            Self::Memory => write!(f, "memory"),
            Self::Global => write!(f, "global"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Export {
    pub name: String,
    pub def: ExportDef,
}

#[derive(Debug, Clone)]
pub enum ExportDef {
    Func(FuncId),
    Table(TableId),
    Memory(MemoryId),
    Global(GlobalId),
}
