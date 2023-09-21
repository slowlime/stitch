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
    Var(Var),
    Block(Spanned<Block>),
    Array(ArrayLit),
    Symbol(SymbolLit),
    String(StringLit),
    Int(IntLit),
    Float(FloatLit),
    Dispatch(Dispatch),
}

impl Expr {
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
pub struct Assign {
    pub location: Location,
    pub var: Name,
    pub value: Box<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Var(pub Name);

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
    pub selector: Selector,
    pub args: Vec<Expr>,
}
