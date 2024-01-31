use slotmap::SlotMap;

use crate::util::try_match;

use super::expr::Expr;
use super::ty::{FuncType, ValType};
use super::{ImportId, IntrinsicDecl, LocalId, Module};

#[derive(Debug, Clone)]
pub enum Func {
    Import(FuncImport),
    Body(FuncBody),
}

impl Func {
    pub fn get_intrinsic(&self, module: &Module) -> Option<IntrinsicDecl> {
        match *self {
            Self::Import(FuncImport { import_id, .. }) => module.get_intrinsic(import_id),
            _ => None,
        }
    }

    pub fn is_import(&self) -> bool {
        matches!(self, Self::Import(_))
    }

    pub fn ty(&self) -> &FuncType {
        match self {
            Self::Import(import) => &import.ty,
            Self::Body(body) => &body.ty,
        }
    }

    pub fn body(&self) -> Option<&FuncBody> {
        try_match!(self, Self::Body(body) => body)
    }

    pub fn body_mut(&mut self) -> Option<&mut FuncBody> {
        try_match!(self, Self::Body(body) => body)
    }
}

#[derive(Debug, Clone)]
pub struct FuncImport {
    pub ty: FuncType,
    pub import_id: ImportId,
}

#[derive(Debug, Clone)]
pub struct FuncBody {
    pub ty: FuncType,
    pub locals: SlotMap<LocalId, ValType>,
    pub params: Vec<LocalId>,
    pub body: Vec<Expr>,
}

impl FuncBody {
    pub fn new(ty: FuncType) -> Self {
        Self {
            ty,
            locals: Default::default(),
            params: Default::default(),
            body: Default::default(),
        }
    }
}
