#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ValType {
    I32,
    I64,
    F32,
    F64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    Func(FuncType),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FuncType {
    pub params: Vec<ValType>,
    pub ret: Option<ValType>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ElemType {
    FuncType,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Limits {
    pub min: u32,
    pub max: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TableType {
    pub elem_ty: ElemType,
    pub limits: Limits,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MemoryType {
    pub limits: Limits,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GlobalType {
    pub val_type: ValType,
    pub mutable: bool,
}
