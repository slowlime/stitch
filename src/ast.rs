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
    Expr(Expr<'buf>),
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
    Selector(Selector<'buf>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct StringLit<'buf>(Name<'buf>);

#[derive(Debug, Clone, PartialEq)]
pub struct IntLit(Spanned<i64>);

#[derive(Debug, Clone, PartialEq)]
pub struct FloatLit(Spanned<f64>);

#[derive(Debug, Clone, PartialEq)]
pub struct Dispatch<'buf> {
    pub location: Location,
    pub selector: Selector<'buf>,
    pub params: Vec<Expr<'buf>>,
}
