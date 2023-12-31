//! A simple AST-based IR for a WebAssembly module.

pub mod expr;
pub mod func;
pub mod ty;

use std::fmt::{self, Display};

use slotmap::{new_key_type, SlotMap};

use crate::util::slot::BiSlotMap;

use self::ty::{GlobalType, MemoryType, TableType, Type};

pub use self::expr::Expr;
pub use self::func::{Func, FuncBody};

new_key_type! {
    pub struct TypeId;
    pub struct FuncId;
    pub struct TableId;
    pub struct MemoryId;
    pub struct GlobalId;
    pub struct ImportId;
    pub struct ExportId;
    pub struct LocalId;
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
    /// Inserts the types of the signatures of the module's functions.
    pub fn insert_func_types(&mut self) {
        for func in self.funcs.values() {
            let Func::Body(body) = func else { continue };

            self.types.insert(Type::Func(body.ty.clone()));
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
