//! A simple AST-based IR for a WebAssembly module.

pub mod func;
pub mod ty;
pub mod value;

use slotmap::{SlotMap, new_key_type};

use self::func::Func;
use self::ty::{Type, TableType, GlobalType, MemoryType};
use self::value::Value;

new_key_type! {
    pub struct TypeId;
    pub struct FuncId;
    pub struct TableId;
    pub struct MemoryId;
    pub struct GlobalId;
    pub struct ImportId;
}

#[derive(Debug, Clone)]
pub struct Module {
    pub version: u16,
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
    Elems(Vec<Value>),
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
    Value(Value),
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
