use super::Import;
use super::ty::FuncType;

#[derive(Debug, Clone)]
pub enum Func {
    Import(FuncImport),
    Body(FuncBody),
}

#[derive(Debug, Clone)]
pub struct FuncImport {
    pub ty: FuncType,
    pub import_idx: Import,
}

#[derive(Debug, Clone)]
pub struct FuncBody {
    pub ty: FuncType,
}
