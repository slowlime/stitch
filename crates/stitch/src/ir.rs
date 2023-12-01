//! A simple AST-based IR for a WebAssembly module.

pub mod func;
pub mod ty;
pub mod expr;

use std::fmt::{self, Display};

use slotmap::{SlotMap, new_key_type};

use self::func::Func;
use self::ty::{Type, TableType, GlobalType, MemoryType};
use self::expr::Expr;

new_key_type! {
    pub struct TypeId;
    pub struct FuncId;
    pub struct TableId;
    pub struct MemoryId;
    pub struct GlobalId;
    pub struct ImportId;
    pub struct LocalId;
}

#[derive(Debug, Default, Clone)]
pub struct Module {
    pub types: SlotMap<TypeId, Type>,
    pub funcs: SlotMap<FuncId, Func>,
    pub tables: SlotMap<TableId, Table>,
    pub mems: SlotMap<MemoryId, Memory>,
    pub globals: SlotMap<GlobalId, Global>,
    pub start: Option<FuncId>,
    pub imports: SlotMap<ImportId, Import>,
}

#[derive(Debug, Clone)]
pub struct Table {
    pub ty: TableType,
    pub def: TableDef,
}

#[derive(Debug, Clone)]
pub enum TableDef {
    Import(ImportId),
    Elems(Vec<Expr>),
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
