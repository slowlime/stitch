pub mod visit;

use std::fmt::{self, Display};
use std::num::NonZeroUsize;

use crate::location::{Location, Spanned};

use self::visit::AstRecurse;

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

impl AstRecurse for Class {
    fn recurse<V: visit::Visitor>(&self, visitor: &mut V) {
        for method in &self.object_methods {
            visitor.visit_method(method);
        }

        for method in &self.class_methods {
            visitor.visit_method(method);
        }
    }

    fn recurse_mut<V: visit::VisitorMut>(&mut self, visitor: &mut V) {
        for method in &mut self.object_methods {
            visitor.visit_method(method);
        }

        for method in &mut self.class_methods {
            visitor.visit_method(method);
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Method {
    pub location: Location,
    pub selector: Spanned<Selector>,
    pub def: Spanned<MethodDef>,
}

impl AstRecurse for Method {
    fn recurse<V: visit::Visitor>(&self, visitor: &mut V) {
        self.def.value.recurse(visitor);
    }

    fn recurse_mut<V: visit::VisitorMut>(&mut self, visitor: &mut V) {
        self.def.value.recurse_mut(visitor);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Selector {
    Unary(Name),
    Binary(Name),
    Keyword(Vec<Name>),
}

impl Display for Selector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unary(name) | Self::Binary(name) => name.value.fmt(f),

            Self::Keyword(names) => {
                for name in names {
                    write!(f, "{}:", name.value)?;
                }

                Ok(())
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum MethodDef {
    Primitive,
    Block(Block),
}

impl AstRecurse for MethodDef {
    fn recurse<V: visit::Visitor>(&self, visitor: &mut V) {
        match self {
            MethodDef::Block(ref block) => visitor.visit_block(block),
            MethodDef::Primitive => {}
        }
    }

    fn recurse_mut<V: visit::VisitorMut>(&mut self, visitor: &mut V) {
        match self {
            MethodDef::Block(ref mut block) => visitor.visit_block(block),
            MethodDef::Primitive => {}
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub params: Vec<Name>,
    pub locals: Vec<Name>,
    pub body: Vec<Stmt>,
}

impl AstRecurse for Block {
    fn recurse<V: visit::Visitor>(&self, visitor: &mut V) {
        for stmt in &self.body {
            visitor.visit_stmt(stmt);
        }
    }

    fn recurse_mut<V: visit::VisitorMut>(&mut self, visitor: &mut V) {
        for stmt in &mut self.body {
            visitor.visit_stmt(stmt);
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub enum Stmt {
    Return(Spanned<Expr>),
    NonLocalReturn(Spanned<Expr>),
    Expr(Spanned<Expr>),

    #[default]
    Dummy,
}

impl AstRecurse for Stmt {
    fn recurse<V: visit::Visitor>(&self, visitor: &mut V) {
        match self {
            Self::Return(expr) | Self::NonLocalReturn(expr) | Self::Expr(expr) => {
                visitor.visit_expr(&expr.value)
            }

            Self::Dummy => panic!("Stmt::Dummy in AST"),
        }
    }

    fn recurse_mut<V: visit::VisitorMut>(&mut self, visitor: &mut V) {
        match self {
            Self::Return(expr) | Self::NonLocalReturn(expr) | Self::Expr(expr) => {
                visitor.visit_expr(&mut expr.value)
            }

            Self::Dummy => panic!("Stmt::Dummy in AST"),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
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

    #[default]
    Dummy,
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

            Self::Dummy => panic!("Expr::Dummy in AST"),
        }
    }
}

impl AstRecurse for Expr {
    fn recurse<V: visit::Visitor>(&self, visitor: &mut V) {
        match self {
            Self::Assign(expr) => visitor.visit_assign(expr),
            Self::Block(expr) => visitor.visit_block(&expr.value),
            Self::Array(expr) => visitor.visit_array(expr),
            Self::Symbol(expr) => visitor.visit_symbol(expr),
            Self::String(expr) => visitor.visit_string(expr),
            Self::Int(expr) => visitor.visit_int(expr),
            Self::Float(expr) => visitor.visit_float(expr),
            Self::Dispatch(expr) => visitor.visit_dispatch(expr),
            Self::UnresolvedName(expr) => visitor.visit_unresolved_name(expr),
            Self::Local(expr) => visitor.visit_local(expr),
            Self::Upvalue(expr) => visitor.visit_upvalue(expr),
            Self::Field(expr) => visitor.visit_field(expr),
            Self::Global(expr) => visitor.visit_global(expr),

            Self::Dummy => panic!("Expr::Dummy in AST"),
        }
    }

    fn recurse_mut<V: visit::VisitorMut>(&mut self, visitor: &mut V) {
        match self {
            Self::Assign(expr) => visitor.visit_assign(expr),
            Self::Block(expr) => visitor.visit_block(&mut expr.value),
            Self::Array(expr) => visitor.visit_array(expr),
            Self::Symbol(expr) => visitor.visit_symbol(expr),
            Self::String(expr) => visitor.visit_string(expr),
            Self::Int(expr) => visitor.visit_int(expr),
            Self::Float(expr) => visitor.visit_float(expr),
            Self::Dispatch(expr) => visitor.visit_dispatch(expr),
            Self::UnresolvedName(expr) => visitor.visit_unresolved_name(expr),
            Self::Local(expr) => visitor.visit_local(expr),
            Self::Upvalue(expr) => visitor.visit_upvalue(expr),
            Self::Field(expr) => visitor.visit_field(expr),
            Self::Global(expr) => visitor.visit_global(expr),

            Self::Dummy => panic!("Expr::Dummy in AST"),
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

impl AstRecurse for AssignVar {
    fn recurse<V: visit::Visitor>(&self, visitor: &mut V) {
        match self {
            Self::UnresolvedName(name) => visitor.visit_unresolved_name(name),
            Self::Local(local) => visitor.visit_local(local),
            Self::Upvalue(upvalue) => visitor.visit_upvalue(upvalue),
            Self::Field(field) => visitor.visit_field(field),
            Self::Global(global) => visitor.visit_global(global),
        }
    }

    fn recurse_mut<V: visit::VisitorMut>(&mut self, visitor: &mut V) {
        match self {
            Self::UnresolvedName(name) => visitor.visit_unresolved_name(name),
            Self::Local(local) => visitor.visit_local(local),
            Self::Upvalue(upvalue) => visitor.visit_upvalue(upvalue),
            Self::Field(field) => visitor.visit_field(field),
            Self::Global(global) => visitor.visit_global(global),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Assign {
    pub location: Location,
    pub var: AssignVar,
    pub value: Box<Expr>,
}

impl AstRecurse for Assign {
    fn recurse<V: visit::Visitor>(&self, visitor: &mut V) {
        self.var.recurse(visitor);
        visitor.visit_expr(&self.value);
    }

    fn recurse_mut<V: visit::VisitorMut>(&mut self, visitor: &mut V) {
        self.var.recurse_mut(visitor);
        visitor.visit_expr(&mut self.value);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ArrayLit(pub Spanned<Vec<Expr>>);

impl AstRecurse for ArrayLit {
    fn recurse<V: visit::Visitor>(&self, visitor: &mut V) {
        for elem in &self.0.value {
            visitor.visit_expr(elem);
        }
    }

    fn recurse_mut<V: visit::VisitorMut>(&mut self, visitor: &mut V) {
        for elem in &mut self.0.value {
            visitor.visit_expr(elem);
        }
    }
}

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

impl AstRecurse for Dispatch {
    fn recurse<V: visit::Visitor>(&self, visitor: &mut V) {
        visitor.visit_expr(&self.recv);

        for arg in &self.args {
            visitor.visit_expr(arg);
        }
    }

    fn recurse_mut<V: visit::VisitorMut>(&mut self, visitor: &mut V) {
        visitor.visit_expr(&mut self.recv);

        for arg in &mut self.args {
            visitor.visit_expr(arg);
        }
    }
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
