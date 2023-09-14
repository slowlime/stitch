use std::borrow::Cow;

use crate::location::{Location, Spanned};

#[derive(Debug, Clone, PartialEq)]
pub struct Class<'buf> {
    pub location: Location,
    pub name: Spanned<Cow<'buf, str>>,
    pub superclass: Option<Cow<'buf, str>>,
    pub object_fields: Vec<Cow<'buf, str>>,
    pub object_methods: Vec<Method<'buf>>,
    pub class_fields: Vec<Cow<'buf, str>>,
    pub class_methods: Vec<Method<'buf>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Method<'buf> {
    pub location: Location,
    pub selector: Selector<'buf>,
    pub definition: Spanned<MethodDef<'buf>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Selector<'buf> {
    Unary(Spanned<Cow<'buf, str>>),
    Binary(Spanned<Cow<'buf, str>>),
    Keyword(Vec<Spanned<Cow<'buf, str>>>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum MethodDef<'buf> {
    Primitive,
    Block(Block<'buf>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block<'buf> {
    pub params: Vec<Spanned<Cow<'buf, str>>>,
    pub locals: Vec<Spanned<Cow<'buf, str>>>,
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
    Block(Block<'buf>),
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
    pub var: Spanned<Cow<'buf, str>>,
    pub value: Box<Expr<'buf>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Var<'buf>(pub Spanned<Cow<'buf, str>>);

#[derive(Debug, Clone, PartialEq)]
pub struct ArrayLit<'buf>(pub Spanned<Vec<Expr<'buf>>>);

#[derive(Debug, Clone, PartialEq)]
pub enum SymbolLit<'buf> {
    String(Spanned<Cow<'buf, str>>),
    Selector(Selector<'buf>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct StringLit<'buf>(Spanned<Cow<'buf, str>>);

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
