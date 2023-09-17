use std::borrow::Cow;

use crate::location::{Location, Spanned};

pub type Name<'buf> = Spanned<Cow<'buf, str>>;

#[derive(Debug, Clone, PartialEq)]
pub struct Class<'buf> {
    pub location: Location,
    pub name: Name<'buf>,
    pub superclass: Option<Name<'buf>>,
    pub object_fields: Vec<Name<'buf>>,
    pub object_methods: Vec<Method<'buf>>,
    pub class_fields: Vec<Name<'buf>>,
    pub class_methods: Vec<Method<'buf>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Method<'buf> {
    pub location: Location,
    pub selector: Spanned<Selector<'buf>>,
    pub def: Spanned<MethodDef<'buf>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Selector<'buf> {
    Unary(Name<'buf>),
    Binary(Name<'buf>),
    Keyword(Vec<Name<'buf>>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum MethodDef<'buf> {
    Primitive,
    Block(Block<'buf>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block<'buf> {
    pub params: Vec<Name<'buf>>,
    pub locals: Vec<Name<'buf>>,
    pub body: Vec<Stmt<'buf>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt<'buf> {
    Return(Spanned<Expr<'buf>>),
    Expr(Spanned<Expr<'buf>>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr<'buf> {
    Assign(Assign<'buf>),
    Var(Var<'buf>),
    Block(Spanned<Block<'buf>>),
    Array(ArrayLit<'buf>),
    Symbol(SymbolLit<'buf>),
    String(StringLit<'buf>),
    Int(IntLit),
    Float(FloatLit),
    Dispatch(Dispatch<'buf>),
}

impl Expr<'_> {
    pub fn location(&self) -> Location {
        match self {
            Self::Assign(expr) => expr.location,
            Self::Var(expr) => expr.0.location,
            Self::Block(expr) => expr.location,
            Self::Array(expr) => expr.0.location,
            Self::Symbol(expr) => expr.location(),
            Self::String(expr) => expr.0.location,
            Self::Int(expr) => expr.0.location,
            Self::Float(expr) => expr.0.location,
            Self::Dispatch(expr) => expr.location,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Assign<'buf> {
    pub location: Location,
    pub var: Name<'buf>,
    pub value: Box<Expr<'buf>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Var<'buf>(pub Name<'buf>);

#[derive(Debug, Clone, PartialEq)]
pub struct ArrayLit<'buf>(pub Spanned<Vec<Expr<'buf>>>);

#[derive(Debug, Clone, PartialEq)]
pub enum SymbolLit<'buf> {
    String(Name<'buf>),
    Selector(Spanned<Selector<'buf>>),
}

impl SymbolLit<'_> {
    pub fn location(&self) -> Location {
        match self {
            Self::String(name) => name.location,
            Self::Selector(sel) => sel.location,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StringLit<'buf>(pub Name<'buf>);

#[derive(Debug, Clone, PartialEq)]
pub struct IntLit(pub Spanned<i64>);

#[derive(Debug, Clone, PartialEq)]
pub struct FloatLit(pub Spanned<f64>);

#[derive(Debug, Clone, PartialEq)]
pub struct Dispatch<'buf> {
    pub location: Location,
    pub recv: Box<Expr<'buf>>,
    pub selector: Selector<'buf>,
    pub args: Vec<Expr<'buf>>,
}
