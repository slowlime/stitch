use std::collections::HashMap;
use std::fmt::{self, Display};

use crate::ast;
use crate::location::Location;

use super::gc::GcRefCell;
use super::value::{tag, TypedValue, Value};

pub struct CalleeName<'s, 'gc>(&'s Callee<'gc>);

impl Display for CalleeName<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            Callee::Method(value) => write!(f, "method {}", value.get().selector.value.name()),
            Callee::Block(_) => write!(f, "<block>"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Callee<'gc> {
    Method(TypedValue<'gc, tag::Method>),
    Block(TypedValue<'gc, tag::Block>),
}

impl<'gc> Callee<'gc> {
    pub fn location(&self) -> Location {
        match self {
            Self::Method(value) => value.get().location,
            Self::Block(value) => value.get().location,
        }
    }

    pub fn name(&self) -> CalleeName<'_, 'gc> {
        CalleeName(self)
    }
}

#[derive(Debug, Clone)]
pub struct Frame<'gc> {
    pub callee: Callee<'gc>,
    pub local_map: HashMap<String, usize>,
    pub locals: Vec<Local<'gc>>,
}

#[derive(Debug, Clone)]
pub struct Local<'gc> {
    pub name: ast::Name,
    pub value: GcRefCell<Value<'gc>>,
}
