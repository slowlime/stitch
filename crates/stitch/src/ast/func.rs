use slotmap::SlotMap;

use crate::util::try_match;

use super::expr::Block;
use super::ty::{FuncType, ValType};
use super::{BlockId, ImportId, IntrinsicDecl, LocalId, Module};

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

    pub fn name(&self) -> Option<&str> {
        match self {
            Self::Import(import) => import.name(),
            Self::Body(body) => body.name(),
        }
    }

    pub fn set_name(&mut self, name: Option<String>) {
        match self {
            Self::Import(import) => import.set_name(name),
            Self::Body(body) => body.set_name(name),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FuncImport {
    pub name: Option<String>,
    pub ty: FuncType,
    pub import_id: ImportId,
}

impl FuncImport {
    pub fn name(&self) -> Option<&str> {
        self.name.as_ref().map(String::as_str)
    }

    pub fn set_name(&mut self, name: Option<String>) {
        self.name = name;
    }
}

#[derive(Debug, Clone)]
pub struct FuncBody {
    pub name: Option<String>,
    pub ty: FuncType,
    pub locals: SlotMap<LocalId, ValType>,
    pub params: Vec<LocalId>,
    pub blocks: SlotMap<BlockId, ()>,
    pub main_block: Block,
}

impl FuncBody {
    pub fn new(ty: FuncType) -> Self {
        let mut blocks = SlotMap::with_key();
        let body = Block {
            body: vec![],
            id: blocks.insert(()),
        };

        Self {
            name: None,
            ty,
            locals: Default::default(),
            params: Default::default(),
            blocks,
            main_block: body,
        }
    }

    pub fn name(&self) -> Option<&str> {
        self.name.as_ref().map(String::as_str)
    }

    pub fn set_name(&mut self, name: Option<String>) {
        self.name = name;
    }
}
