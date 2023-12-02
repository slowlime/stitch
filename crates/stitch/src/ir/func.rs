use slotmap::SlotMap;

use super::{LocalId, ImportId};
use super::ty::{FuncType, ValType};

#[derive(Debug, Clone)]
pub enum Func {
    Import(FuncImport),
    Body(FuncBody),
}

#[derive(Debug, Clone)]
pub struct FuncImport {
    pub ty: FuncType,
    pub import: ImportId,
}

#[derive(Debug, Clone)]
pub struct FuncBody {
    pub ty: FuncType,
    pub locals: SlotMap<LocalId, ValType>
}

impl FuncBody {
    pub fn new(ty: FuncType) -> Self {
        Self {
            ty,
            locals: Default::default(),
        }
    }
}
