use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::mem;
use std::num::NonZeroUsize;
use std::ops::{Deref, DerefMut};

use miette::{Diagnostic, SourceOffset};
use thiserror::Error;

use super::token::{self, BinOp, Special, Token, TokenType, TokenValue};
use super::{BigNumberBehavior, Lexer, LexerError, ParserOptions};

use crate::ast;
use crate::location::{Location, Offset, Span, Spanned};
use crate::util::{define_yes_no_options, format_list, macro_cond, CloneStatic};

const RECURSION_LIMIT: usize = 8192;

#[derive(Error, Diagnostic, Debug, Clone, PartialEq)]
pub enum ParserError {
    #[error("encountered an unexpected token: {:#} (expected {})", .actual.ty(), format_list!("{}", .expected, "or"))]
    #[diagnostic(code(parser::unexpected_token))]
    UnexpectedToken {
        expected: Vec<Cow<'static, str>>,

        #[label]
        actual: Token<'static>,
    },

    #[error("multiple class member separators present in the definition")]
    #[diagnostic(code(parser::multiple_separators))]
    MultipleSeparators {
        #[label]
        bad_separator: Token<'static>,

        #[label = "first separator specified here"]
        first_separator: Token<'static>,
    },

    #[error(
        "reached a recursion limit ({}) while parsing (too much nesting)",
        RECURSION_LIMIT
    )]
    #[diagnostic(code(parser::recursion_limit))]
    RecursionLimit(#[label] Span),

    #[error("the number literal is too large")]
    NumberTooLarge(#[label] Span),

    #[error("cannot define a variable `{name}`")]
    RecvRedefined {
        name: String,

        #[label]
        span: Span,
    },

    #[error("cannot assign to `{name}`")]
    AssignmentToRecv {
        name: String,

        #[label]
        span: Span,
    },

    #[error("name `{name}` defined twice")]
    NameCollision {
        name: String,

        #[label]
        span: Span,

        #[label = "previous definition here"]
        prev_span: Option<Span>,
    },

    #[error(transparent)]
    #[diagnostic(transparent)]
    LexerError(#[from] LexerError),
}

define_yes_no_options! {
    enum PrimitiveAllowed;
    enum BlockParamsAllowed;
    enum StatementFinal;
    enum SuperRecv;
}

trait Matcher {
    fn matches(&self, token: &Token<'_>) -> bool;

    fn expected(&self) -> Vec<Cow<'static, str>>;
}

impl<const N: usize> Matcher for [TokenType<'_>; N] {
    fn matches(&self, token: &Token<'_>) -> bool {
        self.contains(&token.ty())
    }

    fn expected(&self) -> Vec<Cow<'static, str>> {
        self.iter().map(|t| format!("{:#}", t).into()).collect()
    }
}

impl<const N: usize> Matcher for [Special; N] {
    fn matches(&self, token: &Token<'_>) -> bool {
        match token.ty() {
            TokenType::Special(sym) => self.contains(&sym),
            _ => false,
        }
    }

    fn expected(&self) -> Vec<Cow<'static, str>> {
        self.iter()
            .map(|s| format!("{:#}", s.as_str()).into())
            .collect()
    }
}

impl Matcher for TokenType<'_> {
    fn matches(&self, token: &Token<'_>) -> bool {
        self == &token.ty()
    }

    fn expected(&self) -> Vec<Cow<'static, str>> {
        vec![format!("{:#}", self).into()]
    }
}

impl Matcher for Special {
    fn matches(&self, token: &Token<'_>) -> bool {
        TokenType::Special(*self) == token.ty()
    }

    fn expected(&self) -> Vec<Cow<'static, str>> {
        vec![format!("{:#}", TokenType::Special(*self)).into()]
    }
}

impl Matcher for BinOp<'_> {
    fn matches(&self, token: &Token<'_>) -> bool {
        token
            .value
            .as_bin_op()
            .is_some_and(|bin_op| &*bin_op == self)
    }

    fn expected(&self) -> Vec<Cow<'static, str>> {
        vec![format!("{:#}", TokenType::BinOp(self.clone())).into()]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SeparatorMatcher;

impl Matcher for SeparatorMatcher {
    fn matches(&self, token: &Token<'_>) -> bool {
        match token.value {
            TokenValue::BinOp(ref op) => op.is_separator(),
            _ => false,
        }
    }

    fn expected(&self) -> Vec<Cow<'static, str>> {
        vec!["separator".into()]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BinOpMatcher;

impl Matcher for BinOpMatcher {
    fn matches(&self, token: &Token<'_>) -> bool {
        token.value.as_bin_op().is_some()
    }

    fn expected(&self) -> Vec<Cow<'static, str>> {
        vec!["binary operator".into()]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct VarNameMatcher;

impl Matcher for VarNameMatcher {
    fn matches(&self, token: &Token<'_>) -> bool {
        matches!(
            token.value,
            TokenValue::Special(Special::Primitive) | TokenValue::Ident(_)
        )
    }

    fn expected(&self) -> Vec<Cow<'static, str>> {
        vec!["identifier".into()]
    }
}

macro_rules! lookahead {
    ($self:ident : $( ( $expected:expr ) )? { $( $matcher:expr => $arm:expr, )+ _ => return #error $(,)? }) => ({
        lookahead!($self: $( ( $expected ) )? {
            $( $matcher => $arm, )+

            _ => {
                return match $self.lexer.peek().unwrap().clone() {
                    Ok(actual) => Err($self.make_unexpected_token_err(actual.clone_static())),

                    Err(e) => Err(ParserError::from(e)),
                };
            },
        })
    });

    ($self:ident : $( ( $expected:expr ) )? { $( $matcher:expr => $arm:expr, )+ _ => $default:expr $(,)? }) => ({
        match $self.lexer.peek() {
            $( Some(Ok(token)) if $matcher.matches(token) => $arm, )+
            _ => {
                macro_cond! {
                    if non_empty!($( $expected )?) {
                        $self.attempted_tokens.insert($( $expected.into() )?);
                    } else {
                        $( $self.attempted_tokens.extend($matcher.expected()); )+
                    }
                };

                $default
            }
        }
    });
}

struct BoundedParser<'a, 'parser> {
    parser: &'parser mut Parser<'a>,
    prev_layer_start: SourceOffset,
}

impl Drop for BoundedParser<'_, '_> {
    fn drop(&mut self) {
        self.parser.recursion_limit += 1;
        self.parser.layer_start = self.prev_layer_start;
    }
}

impl<'a> Deref for BoundedParser<'a, '_> {
    type Target = Parser<'a>;

    fn deref(&self) -> &Parser<'a> {
        self.parser
    }
}

impl<'a> DerefMut for BoundedParser<'a, '_> {
    fn deref_mut(&mut self) -> &mut Parser<'a> {
        self.parser
    }
}

macro_rules! expect_impl {
    {
        match $self:ident . $action:ident ! ($matcher:expr) {
            Ok($token_ok:pat) => $on_match:expr,
            Err($token_err:pat) => $on_fail:expr,
        }
    } => {
        match $self.lexer.peek() {
            Some(Ok(token)) if $matcher.matches(token) => {
                $self.attempted_tokens.clear();
                expect_impl!(@ $self.$action($token_ok, token));

                $on_match
            }

            Some(Ok($token_err)) => {
                $self.attempted_tokens.extend($matcher.expected());

                $on_fail
            }

            Some(Err(_)) => Err($self.lexer.next().unwrap().err().unwrap().into()),

            None => panic!("peeking after retrieving the Eof token"),
        }
    };

    (@ $self:ident.peek($token:pat, $ref:expr)) => {
        let $token = $ref;
    };

    (@ $self:ident.try_consume($token:pat, $ref:expr)) => {
        let $token = $self.lexer.next().unwrap().unwrap();
    };
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Var {
    /// A local variable defined in the current block.
    Local(Location),

    /// An implicit `self` or `super` parameter.
    Recv,

    /// A field of the receiver.
    Field(Location),
}

impl Var {
    fn location(&self) -> Location {
        match *self {
            Self::Local(l) | Self::Field(l) => l,
            Self::Recv => Location::Builtin,
        }
    }
}

enum ResolvedVar {
    Local,
    Upvalue {
        up_frames: NonZeroUsize,
    },
    Field,

    /// Either a global or a superclass field.
    Unresolved,
}

#[derive(Default)]
struct Scope {
    vars: HashMap<String, Var>,
}

pub struct Parser<'a> {
    lexer: std::iter::Peekable<Lexer<'a>>,
    recursion_limit: usize,
    layer_start: SourceOffset,
    attempted_tokens: HashSet<Cow<'static, str>>,
    scopes: Vec<Scope>,
    options: ParserOptions,
}

impl<'a> Parser<'a> {
    pub fn new(lexer: Lexer<'a>, options: ParserOptions) -> Self {
        let layer_start = lexer.pos();

        Self {
            lexer: lexer.peekable(),
            recursion_limit: RECURSION_LIMIT,
            layer_start,
            attempted_tokens: Default::default(),
            scopes: vec![],
            options,
        }
    }

    fn bounded(&mut self) -> Result<BoundedParser<'a, '_>, ParserError> {
        self.recursion_limit =
            self.recursion_limit
                .checked_sub(1)
                .ok_or(ParserError::RecursionLimit(
                    self.make_span_from(self.layer_start),
                ))?;

        let layer_start = self.next_pos();
        let prev_layer_start = mem::replace(&mut self.layer_start, layer_start);

        Ok(BoundedParser {
            parser: self,
            prev_layer_start,
        })
    }

    fn make_span_from(&mut self, start: impl Offset) -> Span {
        Span::new_spanning(start.offset()..self.next_pos().offset())
    }

    fn next_pos(&mut self) -> SourceOffset {
        match self.lexer.peek().unwrap().as_ref() {
            Ok(token) => token.span.start(),
            Err(e) => e.span.start(),
        }
    }

    fn make_unexpected_token_err(&mut self, actual: Token<'static>) -> ParserError {
        let mut expected: Vec<_> = mem::take(&mut self.attempted_tokens).into_iter().collect();
        expected.sort();

        ParserError::UnexpectedToken {
            expected,
            actual: actual.clone_static(),
        }
    }

    fn expect(&mut self, matcher: impl Matcher) -> Result<Token<'a>, ParserError> {
        expect_impl! {
            match self.try_consume!(matcher) {
                Ok(token) => Ok(token),

                Err(token) => {
                    let token = token.clone_static();

                    Err(self.make_unexpected_token_err(token))
                },
            }
        }
    }

    fn try_consume(&mut self, matcher: impl Matcher) -> Result<Option<Token<'a>>, ParserError> {
        expect_impl! {
            match self.try_consume!(matcher) {
                Ok(token) => Ok(Some(token)),
                Err(_) => Ok(None),
            }
        }
    }

    fn peek(&mut self, matcher: impl Matcher) -> Result<bool, ParserError> {
        expect_impl! {
            match self.peek!(matcher) {
                Ok(_) => Ok(true),
                Err(_) => Ok(false),
            }
        }
    }

    fn define_var(&mut self, name: String, kind: Var) -> Result<(), ParserError> {
        use std::collections::hash_map::Entry;

        let scope = self.scopes.last_mut().unwrap();

        if kind != Var::Recv && (name == "self" || name == "super") {
            return Err(ParserError::RecvRedefined {
                name,
                span: kind.location().span().unwrap(),
            });
        }

        match scope.vars.entry(name) {
            Entry::Vacant(e) => {
                e.insert(kind);

                Ok(())
            }

            Entry::Occupied(e) => Err(ParserError::NameCollision {
                name: e.key().clone(),
                prev_span: e.get().location().span(),
                span: kind.location().span().unwrap(),
            }),
        }
    }

    fn resolve_var(&mut self, name: &str) -> ResolvedVar {
        // since `super` is only defined once, in the same scope as `self`, this is okay.
        let name = if name == "super" { "self" } else { name };

        for (i, scope) in self.scopes.iter().rev().enumerate() {
            if let Some(var) = scope.vars.get(name) {
                return match var {
                    Var::Local(_) | Var::Recv if i == 0 => ResolvedVar::Local,
                    Var::Local(_) | Var::Recv => ResolvedVar::Upvalue {
                        up_frames: i.try_into().unwrap(),
                    },
                    Var::Field(_) => ResolvedVar::Field,
                };
            }
        }

        ResolvedVar::Unresolved
    }

    pub fn parse(mut self) -> Result<ast::Class, ParserError> {
        let class = self.parse_class()?;
        self.expect(TokenType::Eof)?;

        Ok(class)
    }

    fn parse_class(&mut self) -> Result<ast::Class, ParserError> {
        let name = self.parse_ident(PrimitiveAllowed::No)?;
        self.expect(Special::Equals)?;

        let superclass = if self.peek(Special::ParenLeft)? {
            None
        } else {
            Some(self.parse_ident(PrimitiveAllowed::No)?)
        };

        self.expect(Special::ParenLeft)?;

        let object_fields = self.parse_opt_var_list()?.unwrap_or_default();
        self.scopes.push(Default::default());

        for field in &object_fields {
            self.define_var(Into::into(&field.value), Var::Field(field.location))?;
        }

        let mut object_methods = vec![];

        loop {
            object_methods.push(lookahead!(self: {
                Special::ParenRight => break,
                SeparatorMatcher => break,
                _ => self.parse_method()?,
            }));
        }

        self.scopes.pop();
        assert!(self.scopes.is_empty());

        let (class_fields, class_methods) =
            if let Some(separator) = self.try_consume(SeparatorMatcher)? {
                let class_fields = self.parse_opt_var_list()?.unwrap_or_default();

                self.scopes.push(Default::default());

                for field in &class_fields {
                    self.define_var(Into::into(&field.value), Var::Field(field.location))?;
                }

                let mut class_methods = vec![];

                loop {
                    class_methods.push(lookahead!(self: {
                    Special::ParenRight => break,

                    SeparatorMatcher => {
                        let bad_separator = self.expect(SeparatorMatcher).unwrap().clone_static();

                        return Err(ParserError::MultipleSeparators {
                            bad_separator,
                            first_separator: separator.clone_static(),
                        });
                    },

                    _ => self.parse_method()?,
                }));
                }

                self.scopes.pop();
                assert!(self.scopes.is_empty());

                (class_fields, class_methods)
            } else {
                Default::default()
            };

        let right_paren = self.expect(Special::ParenRight)?;

        Ok(ast::Class {
            location: Location::UserCode(name.span().unwrap().convex_hull(&right_paren.span)),
            name,
            superclass,
            object_fields,
            object_methods,
            class_fields,
            class_methods,
        })
    }

    fn parse_opt_var_list(&mut self) -> Result<Option<Vec<ast::Name>>, ParserError> {
        Ok(
            lookahead!(self: (format!("{:#}", TokenType::Special(Special::Bar))) {
                BinOp::new("||") => {
                    self.expect(BinOp::new("||")).unwrap();

                    Some(vec![])
                },

                Special::Bar => Some(self.parse_var_list()?),

                _ => None,
            }),
        )
    }

    fn parse_var_list(&mut self) -> Result<Vec<ast::Name>, ParserError> {
        self.expect(Special::Bar)?;

        let mut vars = vec![];

        while self.try_consume(Special::Bar)?.is_none() {
            vars.push(self.parse_ident(PrimitiveAllowed::Yes)?);
        }

        Ok(vars)
    }

    fn parse_method(&mut self) -> Result<ast::Method, ParserError> {
        let (selector, params) = self.parse_pattern()?;
        self.expect(Special::Equals)?;

        self.scopes.push(Default::default());

        for param in &params {
            self.define_var(param.value.clone(), Var::Local(param.location))?;
        }

        let def = lookahead!(self: {
            Special::Primitive => {
                let token = self.expect(Special::Primitive).unwrap();

                Spanned::new_spanning(ast::MethodDef::Primitive, token.span)
            },

            Special::ParenLeft => {
                self.define_var("self".into(), Var::Recv).unwrap();

                let mut block = self.parse_block_body(BlockParamsAllowed::No, Special::ParenLeft, Special::ParenRight)?;
                block.value.params = params;
                let span = block.span().unwrap();

                Spanned::new_spanning(ast::MethodDef::Block(block.value), span)
            },

            _ => return #error,
        });

        self.scopes.pop();

        Ok(ast::Method {
            location: Location::UserCode(
                selector.span().unwrap().convex_hull(&def.span().unwrap()),
            ),
            selector,
            def,
        })
    }

    fn parse_pattern(&mut self) -> Result<(Spanned<ast::Selector>, Vec<ast::Name>), ParserError> {
        lookahead!(self: {
            BinOpMatcher => self.parse_bin_pattern(),
            TokenType::Keyword => self.parse_kw_pattern(),
            _ => self.parse_un_pattern(),
        })
    }

    fn parse_bin_pattern(
        &mut self,
    ) -> Result<(Spanned<ast::Selector>, Vec<ast::Name>), ParserError> {
        let Token { span, value } = self.expect(BinOpMatcher).unwrap();
        let name = Spanned::new_spanning(value.as_bin_op().unwrap().into_owned().into_str(), span);
        let param = self.parse_ident(PrimitiveAllowed::Yes)?;

        Ok((
            Spanned::new_spanning(ast::Selector::Binary(name.into_owned()), span),
            vec![param],
        ))
    }

    fn parse_kw_pattern(
        &mut self,
    ) -> Result<(Spanned<ast::Selector>, Vec<ast::Name>), ParserError> {
        let mut kws = vec![];
        let mut params = vec![];

        while let Some(kw) = self.try_consume(TokenType::Keyword)? {
            let Token {
                span,
                value: TokenValue::Keyword(kw),
            } = kw
            else {
                unreachable!()
            };

            kws.push(Spanned::new_spanning(kw.into_owned(), span));
            params.push(self.parse_ident(PrimitiveAllowed::Yes)?);
        }

        debug_assert!(!kws.is_empty());
        let span = kws[0]
            .span()
            .unwrap()
            .convex_hull(&params.last().unwrap().span().unwrap());

        Ok((
            Spanned::new_spanning(ast::Selector::Keyword(kws), span),
            params,
        ))
    }

    fn parse_un_pattern(
        &mut self,
    ) -> Result<(Spanned<ast::Selector>, Vec<ast::Name>), ParserError> {
        let selector = self.parse_ident(PrimitiveAllowed::Yes)?;
        let span = selector.span().unwrap();

        Ok((
            Spanned::new_spanning(ast::Selector::Unary(selector), span),
            vec![],
        ))
    }

    fn parse_block_body(
        &mut self,
        params_allowed: BlockParamsAllowed,
        left_matcher: impl Matcher,
        right_matcher: impl Matcher + Copy,
    ) -> Result<Spanned<ast::Block>, ParserError> {
        let left = self.expect(left_matcher)?;
        let mut params = vec![];

        if params_allowed.is_yes() {
            while let Some(Token { span, value }) = self.try_consume(TokenType::BlockParam)? {
                let TokenValue::BlockParam(param) = value else {
                    unreachable!()
                };

                self.define_var(
                    param.clone().into_owned(),
                    Var::Local(Location::UserCode(span)),
                )?;
                params.push(Spanned::new_spanning(param.into_owned(), span));
            }

            if !params.is_empty() {
                self.expect(Special::Bar)?;
            }
        }

        if let Some(right) = self.try_consume(right_matcher)? {
            return Ok(Spanned::new_spanning(
                ast::Block {
                    params: vec![],
                    locals: vec![],
                    body: vec![],
                },
                left.span.convex_hull(&right.span),
            ));
        }

        let locals = self.parse_opt_var_list()?.unwrap_or_default();

        for local in &locals {
            self.define_var(Into::into(&local.value), Var::Local(local.location))?;
        }

        let mut body = vec![];

        loop {
            let (stmt, terminal) = self.parse_stmt()?;
            body.push(stmt);

            if terminal.is_yes() || self.peek(right_matcher)? {
                break;
            }
        }

        let right = self.expect(right_matcher)?;

        Ok(Spanned::new_spanning(
            ast::Block {
                params,
                locals,
                body,
            },
            left.span.convex_hull(&right.span),
        ))
    }

    fn parse_stmt(&mut self) -> Result<(ast::Stmt, StatementFinal), ParserError> {
        lookahead!(self: {
            Special::Circumflex => self.parse_return_stmt(),
            _ => self.parse_expr_stmt(),
        })
    }

    fn parse_return_stmt(&mut self) -> Result<(ast::Stmt, StatementFinal), ParserError> {
        let circumflex = self.expect(Special::Circumflex).unwrap();
        let (ret_value, _) = self.parse_expr()?;

        let last_span = if let Some(dot) = self.try_consume(Special::Dot)? {
            dot.span
        } else {
            ret_value.location().span().unwrap()
        };

        Ok((
            ast::Stmt::Return(Spanned::new_spanning(
                ret_value,
                circumflex.span.convex_hull(&last_span),
            )),
            StatementFinal::Yes,
        ))
    }

    fn parse_expr_stmt(&mut self) -> Result<(ast::Stmt, StatementFinal), ParserError> {
        let (expr, _) = self.parse_expr()?;
        let expr_span = expr.location().span().unwrap();

        let (span, terminal) = if let Some(dot) = self.try_consume(Special::Dot)? {
            (expr_span.convex_hull(&dot.span), StatementFinal::No)
        } else {
            (expr_span, StatementFinal::Yes)
        };

        Ok((ast::Stmt::Expr(Spanned::new_spanning(expr, span)), terminal))
    }

    fn parse_expr(&mut self) -> Result<(ast::Expr, SuperRecv), ParserError> {
        self.bounded()?.parse_assign_expr()
    }

    fn parse_assign_expr(&mut self) -> Result<(ast::Expr, SuperRecv), ParserError> {
        if self.peek(VarNameMatcher)? {
            let var = self.parse_ident(PrimitiveAllowed::Yes)?;

            if self.try_consume(Special::Assign)?.is_some() {
                if var.value == "self" || var.value == "super" {
                    let span = var.span().unwrap();

                    return Err(ParserError::AssignmentToRecv {
                        name: var.value,
                        span,
                    });
                }

                let var = match self.resolve_var(&var.value) {
                    ResolvedVar::Local => ast::AssignVar::Local(ast::Local(var)),
                    ResolvedVar::Upvalue { up_frames } => ast::AssignVar::Upvalue(ast::Upvalue {
                        name: var,
                        up_frames,
                    }),
                    ResolvedVar::Field => ast::AssignVar::Field(ast::Field(var)),
                    ResolvedVar::Unresolved => {
                        ast::AssignVar::UnresolvedName(ast::UnresolvedName(var))
                    }
                };

                let (value, _) = self.bounded()?.parse_expr()?;
                let span = var
                    .location()
                    .span()
                    .unwrap()
                    .convex_hull(&value.location().span().unwrap());

                Ok((
                    ast::Expr::Assign(ast::Assign {
                        location: Location::UserCode(span),
                        var,
                        value: Box::new(value),
                    }),
                    SuperRecv::No,
                ))
            } else {
                self.bounded()?.parse_kw_dispatch_expr(Some(var))
            }
        } else {
            self.bounded()?.parse_kw_dispatch_expr(None)
        }
    }

    fn parse_kw_dispatch_expr(
        &mut self,
        parsed_recv: Option<ast::Name>,
    ) -> Result<(ast::Expr, SuperRecv), ParserError> {
        let (recv, super_recv) = self.bounded()?.parse_bin_dispatch_expr(parsed_recv)?;

        let mut kws = vec![];
        let mut args = vec![];

        while let Some(kw) = self.try_consume(TokenType::Keyword)? {
            let Token {
                span,
                value: TokenValue::Keyword(kw),
            } = kw
            else {
                unreachable!()
            };
            let (arg, _) = self.bounded()?.parse_bin_dispatch_expr(None)?;

            kws.push(Spanned::new_spanning(kw.into_owned(), span));
            args.push(arg);
        }

        if let Some(kw) = kws.first() {
            let span = kw
                .span()
                .unwrap()
                .convex_hull(&args.last().unwrap().location().span().unwrap());
            let selector = ast::Selector::Keyword(kws);

            Ok((
                ast::Expr::Dispatch(ast::Dispatch {
                    location: Location::UserCode(span),
                    recv: Box::new(recv),
                    supercall: super_recv.is_yes(),
                    selector,
                    args,
                }),
                SuperRecv::No,
            ))
        } else {
            Ok((recv, super_recv))
        }
    }

    fn parse_bin_dispatch_expr(
        &mut self,
        parsed_recv: Option<ast::Name>,
    ) -> Result<(ast::Expr, SuperRecv), ParserError> {
        let (mut recv, mut super_recv) = self.bounded()?.parse_un_dispatch_expr(parsed_recv)?;

        while let Some(Token {
            span: op_span,
            value: op_value,
        }) = self.try_consume(BinOpMatcher)?
        {
            let op = op_value.as_bin_op().unwrap().into_owned().into_str();
            let (arg, _) = self.bounded()?.parse_un_dispatch_expr(None)?;
            let span = recv
                .location()
                .span()
                .unwrap()
                .convex_hull(&arg.location().span().unwrap());
            let selector = ast::Selector::Binary(Spanned::new_spanning(op.into_owned(), op_span));

            recv = ast::Expr::Dispatch(ast::Dispatch {
                location: Location::UserCode(span),
                recv: Box::new(recv),
                supercall: super_recv.is_yes(),
                selector,
                args: vec![arg],
            });
            super_recv = SuperRecv::No;
        }

        Ok((recv, super_recv))
    }

    fn parse_un_dispatch_expr(
        &mut self,
        parsed_recv: Option<ast::Name>,
    ) -> Result<(ast::Expr, SuperRecv), ParserError> {
        let (mut recv, mut super_recv) = self.bounded()?.parse_primary_expr(parsed_recv)?;

        while self.peek(VarNameMatcher)? {
            let name = self.parse_ident(PrimitiveAllowed::Yes)?;
            let name_span = name.location.span().unwrap();
            let selector = ast::Selector::Unary(name);
            let span = recv.location().span().unwrap().convex_hull(&name_span);

            recv = ast::Expr::Dispatch(ast::Dispatch {
                location: Location::UserCode(span),
                recv: Box::new(recv),
                supercall: super_recv.is_yes(),
                selector,
                args: vec![],
            });
            super_recv = SuperRecv::No;
        }

        Ok((recv, super_recv))
    }

    fn parse_primary_expr(
        &mut self,
        parsed_name: Option<ast::Name>,
    ) -> Result<(ast::Expr, SuperRecv), ParserError> {
        if parsed_name.is_some() {
            self.bounded()?.parse_var_expr(parsed_name)
        } else {
            lookahead!(self: {
                VarNameMatcher => self.bounded()?.parse_var_expr(None),
                Special::ParenLeft => self.bounded()?.parse_paren_expr(),
                Special::BracketLeft => self.bounded()?.parse_block_expr().map(|expr| (expr, SuperRecv::No)),
                _ => self.bounded()?.parse_lit_expr().map(|expr| (expr, SuperRecv::No)),
            })
        }
    }

    fn parse_var_expr(
        &mut self,
        parsed_name: Option<ast::Name>,
    ) -> Result<(ast::Expr, SuperRecv), ParserError> {
        let name = match parsed_name {
            Some(name) => name,
            None => self.parse_ident(PrimitiveAllowed::Yes)?,
        };

        let super_recv = (name.value == "super").into();

        let expr = match self.resolve_var(&name.value) {
            ResolvedVar::Local => ast::Expr::Local(ast::Local(name)),
            ResolvedVar::Field => ast::Expr::Field(ast::Field(name)),
            ResolvedVar::Upvalue { up_frames } => {
                ast::Expr::Upvalue(ast::Upvalue { name, up_frames })
            }
            ResolvedVar::Unresolved => ast::Expr::UnresolvedName(ast::UnresolvedName(name)),
        };

        Ok((expr, super_recv))
    }

    fn parse_paren_expr(&mut self) -> Result<(ast::Expr, SuperRecv), ParserError> {
        self.expect(Special::ParenLeft).unwrap();
        let expr = self.bounded()?.parse_expr()?;
        self.expect(Special::ParenRight)?;

        Ok(expr)
    }

    fn parse_block_expr(&mut self) -> Result<ast::Expr, ParserError> {
        self.scopes.push(Default::default());

        let block = self.bounded()?.parse_block_body(
            BlockParamsAllowed::Yes,
            Special::BracketLeft,
            Special::BracketRight,
        )?;

        self.scopes.pop();

        Ok(ast::Expr::Block(block))
    }

    fn parse_lit_expr(&mut self) -> Result<ast::Expr, ParserError> {
        lookahead!(self: ("literal") {
            Special::ArrayLeft => self.bounded()?.parse_array_lit_expr(),
            TokenType::Symbol => self.bounded()?.parse_symbol_lit_expr(),
            TokenType::String => self.bounded()?.parse_string_lit_expr(),
            Special::Minus => self.bounded()?.parse_num_lit_expr(),
            TokenType::Int => self.bounded()?.parse_num_lit_expr(),
            TokenType::Float => self.bounded()?.parse_num_lit_expr(),
            _ => return #error,
        })
    }

    fn parse_array_lit_expr(&mut self) -> Result<ast::Expr, ParserError> {
        let left = self.expect(Special::ArrayLeft).unwrap();
        let mut values = vec![];

        let right = loop {
            if let Some(right) = self.try_consume(Special::ParenRight)? {
                break right;
            }

            values.push(self.parse_lit_expr()?);
        };

        Ok(ast::Expr::Array(ast::ArrayLit(Spanned::new_spanning(
            values,
            left.span.convex_hull(&right.span),
        ))))
    }

    fn parse_symbol_lit_expr(&mut self) -> Result<ast::Expr, ParserError> {
        let Token {
            span,
            value: TokenValue::Symbol(sym),
        } = self.expect(TokenType::Symbol).unwrap()
        else {
            unreachable!()
        };

        let sym = match sym {
            token::Symbol::String(s) => {
                ast::SymbolLit::String(Spanned::new_spanning(s.into_owned(), span))
            }
            token::Symbol::UnarySelector(s) => ast::SymbolLit::Selector(Spanned::new_spanning(
                ast::Selector::Unary(Spanned::new_spanning(s.into_owned(), span)),
                span,
            )),
            token::Symbol::BinarySelector(op) => ast::SymbolLit::Selector(Spanned::new_spanning(
                ast::Selector::Binary(Spanned::new_spanning(op.into_str().into_owned(), span)),
                span,
            )),
            token::Symbol::KeywordSelector(kws) => {
                let kws = kws
                    .into_iter()
                    .map(|token::Keyword { span, kw }| Spanned::new_spanning(kw.into_owned(), span))
                    .collect();

                ast::SymbolLit::Selector(Spanned::new_spanning(ast::Selector::Keyword(kws), span))
            }
        };

        Ok(ast::Expr::Symbol(sym))
    }

    fn parse_string_lit_expr(&mut self) -> Result<ast::Expr, ParserError> {
        let Token {
            span,
            value: TokenValue::String(s),
        } = self.expect(TokenType::String).unwrap()
        else {
            unreachable!()
        };

        Ok(ast::Expr::String(ast::StringLit(Spanned::new_spanning(
            s.into_owned(),
            span,
        ))))
    }

    fn parse_num_lit_expr(&mut self) -> Result<ast::Expr, ParserError> {
        let minus = self.try_consume(Special::Minus)?;
        let Token { span, value } = self.expect([TokenType::Int, TokenType::Float])?;

        let span = match minus {
            Some(ref minus) => minus.span.convex_hull(&span),
            None => span,
        };

        Ok(match value {
            TokenValue::Int(n) => {
                let n = match (n.cmp(&(1 << 63)), minus.is_some()) {
                    (Ordering::Less, false) => n as i64,
                    (Ordering::Less, true) => -(n as i64),
                    (Ordering::Equal, true) => i64::MIN,

                    (_, _) if self.options.big_numbers == BigNumberBehavior::Error => {
                        return Err(ParserError::NumberTooLarge(span))
                    }

                    (Ordering::Equal | Ordering::Greater, false) => i64::MAX,
                    (Ordering::Greater, true) => i64::MIN,
                };

                ast::Expr::Int(ast::IntLit(Spanned::new_spanning(n, span)))
            }

            TokenValue::Float(n) => {
                let n = if minus.is_some() { -n } else { n };

                ast::Expr::Float(ast::FloatLit(Spanned::new_spanning(n, span)))
            }

            _ => unreachable!(),
        })
    }

    fn parse_ident(
        &mut self,
        primitive_allowed: PrimitiveAllowed,
    ) -> Result<ast::Name, ParserError> {
        let Token { span, value } = match primitive_allowed {
            PrimitiveAllowed::No => self.expect(TokenType::Ident)?,
            PrimitiveAllowed::Yes => self.expect(VarNameMatcher)?,
        };

        let name = match value {
            TokenValue::Special(s @ Special::Primitive) => s.as_str().into(),
            TokenValue::Ident(id) => id,
            _ => unreachable!(),
        };

        Ok(Spanned::new_spanning(name.into_owned(), span))
    }
}
