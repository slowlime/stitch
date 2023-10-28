use std::cell::Cell;
use std::collections::HashMap;
use std::fmt::{self, Display};
use std::pin::Pin;

use crate::location::Location;
use crate::{ast, impl_collect};

use super::gc::{Collect, Finalize, Gc, GcOnceCell, GcRefCell};
use super::value::{tag, TypedValue, Value};

pub struct CalleeName<'s, 'gc>(&'s Callee<'gc>);

impl Display for CalleeName<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            Callee::Method { method } => {
                write!(
                    f,
                    "method `{}` of class `{}`",
                    method.get().selector.value.name(),
                    method.get().holder.get().unwrap().get().name.value,
                )
            }

            Callee::Block { .. } => write!(f, "<block>"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Callee<'gc> {
    Method {
        method: TypedValue<'gc, tag::Method>,
    },

    Block {
        block: TypedValue<'gc, tag::Block>,
    },
}

impl<'gc> Callee<'gc> {
    pub fn location(&self) -> Location {
        match self {
            Self::Method { method } => method.get().location,
            Self::Block { block } => block.get().location,
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
    pub locals: Pin<Box<[Local<'gc>]>>,
}

impl<'gc> Frame<'gc> {
    pub fn get_local_by_name(&self, name: &str) -> Option<Pin<&Local<'gc>>> {
        self.local_map.get(name).map(|&idx| Pin::new(&self.locals[idx]))
    }

    pub fn get_recv(&self) -> Option<&GcRefCell<Value<'gc>>> {
        match &self.callee {
            Callee::Method { .. } => Some(&self.get_local_by_name("self").unwrap().get_ref().value),
            Callee::Block { block } => block
                .get()
                .get_upvalue_by_name("self")
                .map(|upvalue| &upvalue.get_local().value),
        }
    }

    pub fn get_defining_method(&self) -> &TypedValue<'gc, tag::Method> {
        match &self.callee {
            Callee::Method { method } => method,
            Callee::Block { block } => &block.get().defining_method,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Local<'gc> {
    pub name: ast::Name,
    pub value: GcRefCell<Value<'gc>>,
}

impl Finalize for Local<'_> {}

unsafe impl Collect for Local<'_> {
    impl_collect! {
        fn visit(&self) {
            visit(&self.value);
        }
    }
}

#[derive(Debug, Clone)]
pub struct Upvalue<'gc> {
    pub next: GcRefCell<Option<Gc<'gc, Upvalue<'gc>>>>,
    local: Cell<*const Local<'gc>>,
    closed_var: GcOnceCell<Local<'gc>>,
}

impl<'gc> Upvalue<'gc> {
    pub fn new(local: Pin<&Local<'gc>>) -> Self {
        Self {
            next: GcRefCell::new(None),
            local: Cell::new(local.get_ref()),
            closed_var: GcOnceCell::new(),
        }
    }

    pub fn get_local(&self) -> &Local<'gc> {
        unsafe { &*self.local.get() }
    }

    pub fn is_closed(&self) -> bool {
        self.closed_var.get().is_some()
    }

    pub fn close(&self) {
        self.closed_var
            .set(self.get_local().clone())
            .expect("upvalue is already closed");
        self.local.set(self.closed_var.get().unwrap());
    }
}

impl Finalize for Upvalue<'_> {}

unsafe impl Collect for Upvalue<'_> {
    impl_collect! {
        fn visit(&self) {
            // FIXME: recursion for traversing a singly-linked list? really?
            visit(&self.next);
            // skip self.local
            visit(&self.closed_var);
        }
    }
}
