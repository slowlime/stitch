use std::cell::{Cell, OnceCell};
use std::collections::HashMap;
use std::fmt::{self, Display};
use std::rc::Rc;

use crate::location::Location;
use crate::{ast, impl_collect};

use super::gc::{Collect, Finalize, Gc, GcRefCell};
use super::value::{tag, TypedValue, Value};

pub struct CalleeName<'s, 'gc>(&'s Callee<'gc>);

impl Display for CalleeName<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            Callee::Method { method: value, .. } => {
                write!(f, "method {}", value.get().selector.value.name())
            }
            Callee::Block { .. } => write!(f, "<block>"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Callee<'gc> {
    Method {
        method: TypedValue<'gc, tag::Method>,
        nlret_valid_flag: Rc<()>,
    },

    Block {
        block: TypedValue<'gc, tag::Block>,
    },
}

impl<'gc> Callee<'gc> {
    pub fn location(&self) -> Location {
        match self {
            Self::Method { method: value, .. } => value.get().location,
            Self::Block { block: value } => value.get().location,
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

impl<'gc> Frame<'gc> {
    pub fn get_local_by_name(&self, name: &str) -> Option<&Local<'gc>> {
        self.local_map.get(name).map(|&idx| &self.locals[idx])
    }

    pub fn get_recv(&self) -> Option<&GcRefCell<Value<'gc>>> {
        match &self.callee {
            Callee::Method { .. } => Some(&self.get_local_by_name("self").unwrap().value),
            Callee::Block { block } => block
                .get()
                .get_upvalue_by_name("self")
                .map(|upvalue| &upvalue.get().value),
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
    closed_var: OnceCell<Local<'gc>>,
}

impl<'gc> Upvalue<'gc> {
    pub unsafe fn new(local: &Local<'gc>) -> Self {
        Self {
            next: GcRefCell::new(None),
            local: Cell::new(local),
            closed_var: OnceCell::new(),
        }
    }

    pub fn get(&self) -> &Local<'gc> {
        unsafe { &*self.local.get() }
    }

    pub fn close(&self) {
        // FIXME: make sure *local.get() is no longer treated as a gc root
        self.closed_var
            .set(self.get().clone())
            .expect("upvalue is already closed");
        self.local.set(self.closed_var.get().unwrap());
    }
}

impl Finalize for Upvalue<'_> {}

unsafe impl Collect for Upvalue<'_> {
    impl_collect! {
        fn visit(&self) {
            visit(&self.next);
            // skip self.local

            if let Some(local) = self.closed_var.get() {
                visit(local);
            }
        }
    }
}
