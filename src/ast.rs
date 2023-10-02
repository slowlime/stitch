use std::num::NonZeroUsize;

use crate::location::{Location, Spanned};

pub type Name = Spanned<String>;

#[derive(Debug, Clone, PartialEq)]
pub struct Class {
    pub location: Location,
    pub name: Name,
    pub superclass: Option<Name>,
    pub object_fields: Vec<Name>,
    pub object_methods: Vec<Method>,
    pub class_fields: Vec<Name>,
    pub class_methods: Vec<Method>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Method {
    pub location: Location,
    pub selector: Spanned<Selector>,
    pub def: Spanned<MethodDef>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Selector {
    Unary(Name),
    Binary(Name),
    Keyword(Vec<Name>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum MethodDef {
    Primitive,
    Block(Block),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub params: Vec<Name>,
    pub locals: Vec<Name>,
    pub body: Vec<Stmt>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Return(Spanned<Expr>),
    Expr(Spanned<Expr>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Assign(Assign),
    Block(Spanned<Block>),
    Array(ArrayLit),
    Symbol(SymbolLit),
    String(StringLit),
    Int(IntLit),
    Float(FloatLit),
    Dispatch(Dispatch),

    UnresolvedName(UnresolvedName),
    Local(Local),
    Upvalue(Upvalue),
    Field(Field),
    Global(Global),
}

impl Expr {
    pub fn location(&self) -> Location {
        match self {
            Self::Assign(expr) => expr.location,
            Self::Block(expr) => expr.location,
            Self::Array(expr) => expr.0.location,
            Self::Symbol(expr) => expr.location(),
            Self::String(expr) => expr.0.location,
            Self::Int(expr) => expr.0.location,
            Self::Float(expr) => expr.0.location,
            Self::Dispatch(expr) => expr.location,

            Self::UnresolvedName(expr) => expr.0.location,
            Self::Local(expr) => expr.0.location,
            Self::Upvalue(expr) => expr.name.location,
            Self::Field(expr) => expr.0.location,
            Self::Global(expr) => expr.0.location,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum AssignVar {
    UnresolvedName(UnresolvedName),
    Local(Local),
    Upvalue(Upvalue),
    Field(Field),
    Global(Global),
}

impl AssignVar {
    pub fn location(&self) -> Location {
        match self {
            AssignVar::UnresolvedName(name) => name.0.location,
            AssignVar::Local(name) => name.0.location,
            AssignVar::Upvalue(name) => name.name.location,
            AssignVar::Field(name) => name.0.location,
            AssignVar::Global(name) => name.0.location,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Assign {
    pub location: Location,
    pub var: AssignVar,
    pub value: Box<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ArrayLit(pub Spanned<Vec<Expr>>);

#[derive(Debug, Clone, PartialEq)]
pub enum SymbolLit {
    String(Name),
    Selector(Spanned<Selector>),
}

impl SymbolLit {
    pub fn location(&self) -> Location {
        match self {
            Self::String(name) => name.location,
            Self::Selector(sel) => sel.location,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StringLit(pub Name);

#[derive(Debug, Clone, PartialEq)]
pub struct IntLit(pub Spanned<i64>);

#[derive(Debug, Clone, PartialEq)]
pub struct FloatLit(pub Spanned<f64>);

#[derive(Debug, Clone, PartialEq)]
pub struct Dispatch {
    pub location: Location,
    pub recv: Box<Expr>,
    pub supercall: bool,
    pub selector: Selector,
    pub args: Vec<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Local(pub Name);

#[derive(Debug, Clone, PartialEq)]
pub struct Upvalue {
    pub name: Name,
    pub up_frames: NonZeroUsize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Field(pub Name);

#[derive(Debug, Clone, PartialEq)]
pub struct Global(pub Name);

/// A name lookup pending resolution: either a global or a field defined in a superclass.
#[derive(Debug, Clone, PartialEq)]
pub struct UnresolvedName(pub Name);
