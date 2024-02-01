use std::fmt::{self, Display};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ValType {
    I32,
    I64,
    F32,
    F64,
}

impl Display for ValType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", match self {
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::F32 => "f32",
            Self::F64 => "f64",
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    Func(FuncType),
}

impl Type {
    pub fn as_func(&self) -> &FuncType {
        match self {
            Self::Func(ty) => ty,
        }
    }

    pub fn as_func_mut(&self) -> &FuncType {
        match self {
            Self::Func(ty) => ty,
        }
    }
}

impl Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Func(func_ty) => write!(f, "{}", func_ty),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FuncType {
    pub params: Vec<ValType>,
    pub ret: Option<ValType>,
}

impl Display for FuncType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;

        for (idx, param) in self.params.iter().enumerate() {
            if idx > 0 {
                write!(f, ", ")?;
            }

            write!(f, "{param}")?;
        }

        write!(f, "] -> [")?;

        if let Some(val_ty) = &self.ret {
            write!(f, "{val_ty}")?;
        }

        write!(f, "]")
    }
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
