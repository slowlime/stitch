pub mod visit;

use std::fmt::{self, Debug, Display, Write};
use std::num::NonZeroUsize;

use crate::location::{Location, Span, Spanned};
use crate::parse::is_ident;
use crate::parse::token::is_bin_op_char;

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
    fn recurse<'a, V: visit::Visitor<'a>>(&'a self, visitor: &mut V) {
        for method in &self.object_methods {
            visitor.visit_method(method);
        }

        for method in &self.class_methods {
            visitor.visit_method(method);
        }
    }

    fn recurse_mut<'a, V: visit::VisitorMut<'a>>(&'a mut self, visitor: &mut V) {
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
    pub selector: SpannedSelector,
    pub def: Spanned<MethodDef>,
}

impl AstRecurse for Method {
    fn recurse<'a, V: visit::Visitor<'a>>(&'a self, visitor: &mut V) {
        self.def.value.recurse(visitor);
    }

    fn recurse_mut<'a, V: visit::VisitorMut<'a>>(&'a mut self, visitor: &mut V) {
        self.def.value.recurse_mut(visitor);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum SelectorKind {
    Unary,
    Binary,
    Keyword(Box<[usize]>),
}

impl SelectorKind {
    pub fn param_count(&self) -> usize {
        match self {
            Self::Unary => 1,
            Self::Binary => 2,
            Self::Keyword(kws) => kws.len(),
        }
    }
}

#[derive(Clone, PartialEq)]
pub struct Selector {
    name: String,
    kind: SelectorKind,
}

impl Selector {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn from_string(name: String) -> Result<Self, String> {
        let kind = if name.chars().all(is_bin_op_char) {
            SelectorKind::Binary
        } else {
            let colons = name
                .match_indices(':')
                .map(|(idx, _)| idx)
                .collect::<Vec<_>>();

            if colons.is_empty() {
                SelectorKind::Unary
            } else {
                // the last colon must be at the very end of the name
                if colons.last().unwrap() + 1 != name.len() {
                    return Err(name);
                }

                SelectorKind::Keyword(colons.into())
            }
        };

        Ok(Self { name, kind })
    }

    pub fn param_count(&self) -> usize {
        self.kind.param_count()
    }

    pub fn keywords(&self) -> KeywordIter<'_> {
        KeywordIter::new(self)
    }
}

impl Debug for Selector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Selector").field(&self.name).finish()
    }
}

impl Display for Selector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

#[derive(Debug, Clone)]
pub struct KeywordIter<'a> {
    selector: &'a Selector,
    next_idx: usize,
}

impl<'a> KeywordIter<'a> {
    fn new(selector: &'a Selector) -> Self {
        Self {
            selector,
            next_idx: 0,
        }
    }
}

impl<'a> Iterator for KeywordIter<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        let result = match &self.selector.kind {
            SelectorKind::Unary | SelectorKind::Binary => {
                (self.next_idx == 0).then_some(self.selector.name.as_str())
            }

            SelectorKind::Keyword(kws) if self.next_idx >= kws.len() => None,

            SelectorKind::Keyword(kws) => {
                let start_pos = self
                    .next_idx
                    .checked_sub(1)
                    .map(|idx| kws[idx] + 1) // kws[idx] is at `:`, and kws[idx]` is at the next char
                    .unwrap_or_default();
                let end_pos = kws[self.next_idx];

                Some(&self.selector.name[start_pos..end_pos])
            }
        };

        if result.is_some() {
            self.next_idx += 1;
        }

        result
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpannedSelector {
    pub location: Location,
    pub value: Selector,
    pub kws: Option<Box<[Location]>>,
}

impl SpannedSelector {
    pub fn from_selector(selector: Selector, location: Location) -> Self {
        match location {
            Location::Builtin => {
                let kws = match selector.kind {
                    SelectorKind::Unary | SelectorKind::Binary => None,
                    SelectorKind::Keyword(ref kws) => {
                        Some(vec![Location::Builtin; kws.len()].into())
                    }
                };

                SpannedSelector {
                    location,
                    value: selector,
                    kws,
                }
            }

            Location::UserCode(span) => {
                let kws = match selector.kind {
                    SelectorKind::Unary | SelectorKind::Binary => None,

                    SelectorKind::Keyword(ref kws) => {
                        let mut result = Vec::with_capacity(kws.len());
                        let offset = span.start().offset();

                        for (idx, colon_pos) in kws.iter().enumerate() {
                            let start_pos =
                                idx.checked_sub(1).map(|idx| kws[idx]).unwrap_or_default();
                            let end_pos = colon_pos + 1;
                            let len = end_pos - start_pos;

                            result.push(Location::UserCode(Span::new_with_extent(
                                start_pos + offset,
                                len,
                            )));
                        }

                        Some(result.into_boxed_slice())
                    }
                };

                SpannedSelector {
                    location,
                    value: selector,
                    kws,
                }
            }
        }
    }

    pub fn new_unary(name: Name) -> Self {
        let Spanned {
            location,
            value: name,
        } = name;
        assert!(!name.contains(':'), "unary selector contains a colon");

        SpannedSelector {
            location,
            value: Selector {
                name,
                kind: SelectorKind::Unary,
            },
            kws: None,
        }
    }

    pub fn new_binary(name: Name) -> Self {
        let Spanned {
            location,
            value: name,
        } = name;

        SpannedSelector {
            location,
            value: Selector {
                name,
                kind: SelectorKind::Binary,
            },
            kws: None,
        }
    }

    pub fn new_keyword(names: Vec<Name>) -> Self {
        assert!(!names.is_empty());

        let location = names
            .iter()
            .map(|name| name.location)
            .reduce(|lhs, rhs| lhs.convex_hull(rhs))
            .unwrap();

        let mut kw_offsets = Vec::with_capacity(names.len() - 1);
        let mut serialized = String::new();

        for name in names.iter() {
            kw_offsets.push(name.value.len());
            write!(serialized, "{}:", name.value).unwrap();
        }

        let kws = Some(names.into_iter().map(|name| name.location).collect());

        SpannedSelector {
            location,
            value: Selector {
                name: serialized,
                kind: SelectorKind::Keyword(kw_offsets.into_boxed_slice()),
            },
            kws,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum MethodDef {
    Primitive { params: Vec<Name> },

    Block(Block),
}

impl AstRecurse for MethodDef {
    fn recurse<'a, V: visit::Visitor<'a>>(&'a self, visitor: &mut V) {
        match self {
            MethodDef::Block(ref block) => visitor.visit_block(block),
            MethodDef::Primitive { .. } => {}
        }
    }

    fn recurse_mut<'a, V: visit::VisitorMut<'a>>(&'a mut self, visitor: &mut V) {
        match self {
            MethodDef::Block(ref mut block) => visitor.visit_block(block),
            MethodDef::Primitive { .. } => {}
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub params: Vec<Name>,
    pub locals: Vec<Name>,
    pub body: Vec<Stmt>,
    pub upvalues: Vec<String>,
}

impl AstRecurse for Block {
    fn recurse<'a, V: visit::Visitor<'a>>(&'a self, visitor: &mut V) {
        for stmt in &self.body {
            visitor.visit_stmt(stmt);
        }
    }

    fn recurse_mut<'a, V: visit::VisitorMut<'a>>(&'a mut self, visitor: &mut V) {
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
    fn recurse<'a, V: visit::Visitor<'a>>(&'a self, visitor: &mut V) {
        match self {
            Self::Return(expr) | Self::NonLocalReturn(expr) | Self::Expr(expr) => {
                visitor.visit_expr(&expr.value)
            }

            Self::Dummy => panic!("Stmt::Dummy in AST"),
        }
    }

    fn recurse_mut<'a, V: visit::VisitorMut<'a>>(&'a mut self, visitor: &mut V) {
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
    fn recurse<'a, V: visit::Visitor<'a>>(&'a self, visitor: &mut V) {
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

    fn recurse_mut<'a, V: visit::VisitorMut<'a>>(&'a mut self, visitor: &mut V) {
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
            AssignVar::Upvalue(upvalue) => upvalue.name.location,
            AssignVar::Field(name) => name.0.location,
            AssignVar::Global(name) => name.0.location,
        }
    }
}

impl AstRecurse for AssignVar {
    fn recurse<'a, V: visit::Visitor<'a>>(&'a self, visitor: &mut V) {
        match self {
            Self::UnresolvedName(name) => visitor.visit_unresolved_name(name),
            Self::Local(local) => visitor.visit_local(local),
            Self::Upvalue(upvalue) => visitor.visit_upvalue(upvalue),
            Self::Field(field) => visitor.visit_field(field),
            Self::Global(global) => visitor.visit_global(global),
        }
    }

    fn recurse_mut<'a, V: visit::VisitorMut<'a>>(&'a mut self, visitor: &mut V) {
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
    fn recurse<'a, V: visit::Visitor<'a>>(&'a self, visitor: &mut V) {
        self.var.recurse(visitor);
        visitor.visit_expr(&self.value);
    }

    fn recurse_mut<'a, V: visit::VisitorMut<'a>>(&'a mut self, visitor: &mut V) {
        self.var.recurse_mut(visitor);
        visitor.visit_expr(&mut self.value);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ArrayLit(pub Spanned<Vec<Expr>>);

impl AstRecurse for ArrayLit {
    fn recurse<'a, V: visit::Visitor<'a>>(&'a self, visitor: &mut V) {
        for elem in &self.0.value {
            visitor.visit_expr(elem);
        }
    }

    fn recurse_mut<'a, V: visit::VisitorMut<'a>>(&'a mut self, visitor: &mut V) {
        for elem in &mut self.0.value {
            visitor.visit_expr(elem);
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SymbolLit {
    String(Name),
    Selector(SpannedSelector),
}

impl SymbolLit {
    pub fn from_string(s: String, location: Location) -> Self {
        let selector = Selector::from_string(s).and_then(|selector| match selector.kind {
            SelectorKind::Binary => Ok(selector),
            _ if selector.keywords().all(is_ident) => Ok(selector),
            _ => Err(selector.name),
        });

        match selector {
            Ok(selector) => Self::Selector(SpannedSelector::from_selector(selector, location)),
            Err(name) => Self::String(Spanned::new(name, location)),
        }
    }

    pub fn location(&self) -> Location {
        match self {
            Self::String(name) => name.location,
            Self::Selector(sel) => sel.location,
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::String(name) => &name.value,
            Self::Selector(selector) => selector.value.name(),
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
    pub selector: SpannedSelector,
    pub args: Vec<Expr>,
}

impl AstRecurse for Dispatch {
    fn recurse<'a, V: visit::Visitor<'a>>(&'a self, visitor: &mut V) {
        visitor.visit_expr(&self.recv);

        for arg in &self.args {
            visitor.visit_expr(arg);
        }
    }

    fn recurse_mut<'a, V: visit::VisitorMut<'a>>(&'a mut self, visitor: &mut V) {
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
