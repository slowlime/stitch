use std::borrow::Cow;
use std::mem;
use std::ops::{Deref, DerefMut};

use miette::{Diagnostic, SourceOffset};
use thiserror::Error;

use super::token::{self, Special, Token, TokenType, TokenValue};
use super::{Lexer, LexerError};

use crate::ast;
use crate::location::{Location, Offset, Span, Spanned};
use crate::util::{define_yes_no_options, format_list, CloneStatic};

const RECURSION_LIMIT: usize = 8192;

#[derive(Error, Diagnostic, Debug, Clone, PartialEq)]
pub enum ParserError {
    #[error("encountered an unexpected token: `{}` (expected {})", .actual.ty(), format_list!("`{}`", &.expected, "or"))]
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

    #[error(transparent)]
    #[diagnostic(transparent)]
    LexerError(#[from] LexerError),
}

define_yes_no_options! {
    enum PrimitiveAllowed;
    enum BlockParamsAllowed;
    enum StatementFinal;
}

trait Matcher<'a> {
    fn matches(&self, token: &Token<'_>) -> bool;

    fn expected_tokens(&self) -> Vec<Cow<'static, str>>;
}

impl<'a, const N: usize> Matcher<'a> for [TokenType<'a>; N] {
    fn matches(&self, token: &Token<'_>) -> bool {
        self.contains(&token.ty())
    }

    fn expected_tokens(&self) -> Vec<Cow<'static, str>> {
        self.iter().map(|t| t.to_string().into()).collect()
    }
}

impl<const N: usize> Matcher<'static> for [Special; N] {
    fn matches(&self, token: &Token<'_>) -> bool {
        match token.ty() {
            TokenType::Special(sym) => self.contains(&sym),
            _ => false,
        }
    }

    fn expected_tokens(&self) -> Vec<Cow<'static, str>> {
        self.iter().map(|s| s.as_str().into()).collect()
    }
}

impl<'a> Matcher<'a> for TokenType<'a> {
    fn matches(&self, token: &Token<'_>) -> bool {
        self == &token.ty()
    }

    fn expected_tokens(&self) -> Vec<Cow<'static, str>> {
        vec![self.to_string().into()]
    }
}

impl Matcher<'static> for Special {
    fn matches(&self, token: &Token<'_>) -> bool {
        TokenType::Special(*self) == token.ty()
    }

    fn expected_tokens(&self) -> Vec<Cow<'static, str>> {
        vec![self.as_str().into()]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SeparatorMatcher;

impl Matcher<'static> for SeparatorMatcher {
    fn matches(&self, token: &Token<'_>) -> bool {
        match token.value {
            TokenValue::BinOp(ref op) => op.is_separator(),
            _ => false,
        }
    }

    fn expected_tokens(&self) -> Vec<Cow<'static, str>> {
        vec!["separator".into()]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BinOpMatcher;

impl Matcher<'static> for BinOpMatcher {
    fn matches(&self, token: &Token<'_>) -> bool {
        token.value.as_bin_op().is_some()
    }

    fn expected_tokens(&self) -> Vec<Cow<'static, str>> {
        vec!["binary operator".into()]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct VarNameMatcher;

impl Matcher<'static> for VarNameMatcher {
    fn matches(&self, token: &Token<'_>) -> bool {
        match token.value {
            TokenValue::Special(Special::Primitive) => true,
            TokenValue::Ident(_) => true,
            _ => false,
        }
    }

    fn expected_tokens(&self) -> Vec<Cow<'static, str>> {
        vec!["identifier".into()]
    }
}

macro_rules! lookahead {
    ($self:ident : { $( $matcher:expr => $arm:expr, )+ _ => return #error $(,)? }) => ({
        lookahead!($self: {
            $( $matcher => $arm, )+

            _ => {
                let mut expected = ::std::collections::HashSet::new();
                $( expected.extend($matcher.expected_tokens()); )+

                return match $self.lexer.peek().unwrap().clone() {
                    Ok(actual) => Err(ParserError::UnexpectedToken {
                        expected: expected.into_iter().collect(),
                        actual: actual.clone_static(),
                    }),

                    Err(e) => Err(ParserError::from(e)),
                };
            },
        })
    });

    ($self:ident : { $( $matcher:expr => $arm:expr, )+ _ => $default:expr $(,)? }) => ({
        match $self.lexer.peek() {
            $( Some(Ok(token)) if $matcher.matches(token) => $arm, )+
            _ => $default,
        }
    });
}

struct BoundedParser<'buf, 'a> {
    parser: &'a mut Parser<'buf>,
    prev_layer_start: SourceOffset,
}

impl Drop for BoundedParser<'_, '_> {
    fn drop(&mut self) {
        self.parser.recursion_limit += 1;
        self.parser.layer_start = self.prev_layer_start;
    }
}

impl<'buf> Deref for BoundedParser<'buf, '_> {
    type Target = Parser<'buf>;

    fn deref(&self) -> &Parser<'buf> {
        self.parser
    }
}

impl<'buf> DerefMut for BoundedParser<'buf, '_> {
    fn deref_mut(&mut self) -> &mut Parser<'buf> {
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
                $self.attempted_tokens.extend_from_slice(&$matcher.expected_tokens());

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

pub struct Parser<'buf> {
    lexer: std::iter::Peekable<Lexer<'buf>>,
    recursion_limit: usize,
    layer_start: SourceOffset,
    attempted_tokens: Vec<Cow<'static, str>>,
}

impl<'buf> Parser<'buf> {
    pub fn new(lexer: Lexer<'buf>) -> Self {
        let layer_start = lexer.pos();

        Self {
            lexer: lexer.peekable(),
            recursion_limit: RECURSION_LIMIT,
            layer_start,
            attempted_tokens: vec![],
        }
    }

    fn bounded(&mut self) -> Result<BoundedParser<'buf, '_>, ParserError> {
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

    fn expect(&mut self, matcher: impl Matcher<'static>) -> Result<Token<'buf>, ParserError> {
        expect_impl! {
            match self.try_consume!(matcher) {
                Ok(token) => Ok(token),

                Err(token) => Err(ParserError::UnexpectedToken {
                    expected: mem::take(&mut self.attempted_tokens),
                    actual: token.clone_static(),
                }),
            }
        }
    }

    fn try_consume(
        &mut self,
        matcher: impl Matcher<'static>,
    ) -> Result<Option<Token<'buf>>, ParserError> {
        expect_impl! {
            match self.try_consume!(matcher) {
                Ok(token) => Ok(Some(token)),
                Err(_) => Ok(None),
            }
        }
    }

    fn peek(&mut self, matcher: impl Matcher<'static>) -> Result<bool, ParserError> {
        expect_impl! {
            match self.peek!(matcher) {
                Ok(_) => Ok(true),
                Err(_) => Ok(false),
            }
        }
    }

    pub fn parse(mut self) -> Result<ast::Class<'buf>, ParserError> {
        let class = self.parse_class()?;
        self.expect(TokenType::Eof)?;

        Ok(class)
    }

    fn parse_class(&mut self) -> Result<ast::Class<'buf>, ParserError> {
        let name = self.parse_ident(PrimitiveAllowed::No)?;
        self.expect(Special::Equals)?;

        let superclass = if self.peek(Special::ParenLeft)? {
            None
        } else {
            Some(self.parse_ident(PrimitiveAllowed::No)?)
        };

        self.expect(Special::ParenLeft)?;

        let object_fields = if self.peek(Special::Bar)? {
            self.parse_var_list()?
        } else {
            vec![]
        };

        let mut object_methods = vec![];

        loop {
            object_methods.push(lookahead!(self: {
                Special::ParenRight => break,
                SeparatorMatcher => break,
                _ => self.parse_method()?,
            }));
        }

        let (class_fields, class_methods) =
            if let Some(separator) = self.try_consume(SeparatorMatcher)? {
                let class_fields = if self.peek(Special::Bar)? {
                    self.parse_var_list()?
                } else {
                    vec![]
                };

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

    fn parse_var_list(&mut self) -> Result<Vec<ast::Name<'buf>>, ParserError> {
        self.expect(Special::Bar)?;

        let mut vars = vec![];

        while self.try_consume(Special::Bar)?.is_none() {
            vars.push(self.parse_ident(PrimitiveAllowed::Yes)?);
        }

        Ok(vars)
    }

    fn parse_method(&mut self) -> Result<ast::Method<'buf>, ParserError> {
        let (selector, params) = self.parse_pattern()?;
        self.expect(Special::Equals)?;

        let def = lookahead!(self: {
            Special::Primitive => {
                let token = self.expect(Special::Primitive).unwrap();

                Spanned::new_spanning(ast::MethodDef::Primitive, token.span)
            },

            Special::ParenLeft => {
                let mut block = self.parse_block_body(BlockParamsAllowed::No, Special::ParenLeft, Special::ParenRight)?;
                block.value.params = params;
                let span = block.span().unwrap();

                Spanned::new_spanning(ast::MethodDef::Block(block.value), span)
            },

            _ => return #error,
        });

        Ok(ast::Method {
            location: Location::UserCode(
                selector.span().unwrap().convex_hull(&def.span().unwrap()),
            ),
            selector,
            def,
        })
    }

    fn parse_pattern(
        &mut self,
    ) -> Result<(Spanned<ast::Selector<'buf>>, Vec<ast::Name<'buf>>), ParserError> {
        lookahead!(self: {
            BinOpMatcher => self.parse_bin_pattern(),
            TokenType::Keyword => self.parse_kw_pattern(),
            _ => self.parse_un_pattern(),
        })
    }

    fn parse_bin_pattern(
        &mut self,
    ) -> Result<(Spanned<ast::Selector<'buf>>, Vec<ast::Name<'buf>>), ParserError> {
        let Token { span, value } = self.expect(BinOpMatcher).unwrap();
        let name = Spanned::new_spanning(value.as_bin_op().unwrap().into_owned().into_str(), span);
        let param = self.parse_ident(PrimitiveAllowed::Yes)?;

        Ok((
            Spanned::new_spanning(ast::Selector::Binary(name), span),
            vec![param],
        ))
    }

    fn parse_kw_pattern(
        &mut self,
    ) -> Result<(Spanned<ast::Selector<'buf>>, Vec<ast::Name<'buf>>), ParserError> {
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

            kws.push(Spanned::new_spanning(kw.into(), span));
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
    ) -> Result<(Spanned<ast::Selector<'buf>>, Vec<ast::Name<'buf>>), ParserError> {
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
        left_matcher: impl Matcher<'static>,
        right_matcher: impl Matcher<'static> + Copy,
    ) -> Result<Spanned<ast::Block<'buf>>, ParserError> {
        let left = self.expect(left_matcher)?;
        let mut params = vec![];

        if params_allowed.is_yes() {
            while let Some(Token { span, value }) = self.try_consume(TokenType::BlockParam)? {
                let TokenValue::BlockParam(param) = value else {
                    unreachable!()
                };

                params.push(Spanned::new_spanning(param.into(), span));
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

        let locals = if self.peek(Special::Bar)? {
            self.parse_var_list()?
        } else {
            vec![]
        };

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

    fn parse_stmt(&mut self) -> Result<(ast::Stmt<'buf>, StatementFinal), ParserError> {
        lookahead!(self: {
            Special::Circumflex => self.parse_return_stmt(),
            _ => self.parse_expr_stmt(),
        })
    }

    fn parse_return_stmt(
        &mut self,
    ) -> Result<(ast::Stmt<'buf>, StatementFinal), ParserError> {
        let circumflex = self.expect(Special::Circumflex).unwrap();
        let ret_value = self.parse_expr()?;

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

    fn parse_expr_stmt(&mut self) -> Result<(ast::Stmt<'buf>, StatementFinal), ParserError> {
        let expr = self.parse_expr()?;
        let expr_span = expr.location().span().unwrap();

        let (span, terminal) = if let Some(dot) = self.try_consume(Special::Dot)? {
            (expr_span.convex_hull(&dot.span), StatementFinal::No)
        } else {
            (expr_span, StatementFinal::Yes)
        };

        Ok((ast::Stmt::Expr(Spanned::new_spanning(expr, span)), terminal))
    }

    fn parse_expr(&mut self) -> Result<ast::Expr<'buf>, ParserError> {
        self.bounded()?.parse_assign_expr()
    }

    fn parse_assign_expr(&mut self) -> Result<ast::Expr<'buf>, ParserError> {
        if self.peek(VarNameMatcher)? {
            let var = self.parse_ident(PrimitiveAllowed::Yes)?;

            if self.try_consume(Special::Assign)?.is_some() {
                let value = self.bounded()?.parse_expr()?;
                let span = var
                    .span()
                    .unwrap()
                    .convex_hull(&value.location().span().unwrap());

                Ok(ast::Expr::Assign(ast::Assign {
                    location: Location::UserCode(span),
                    var,
                    value: Box::new(value),
                }))
            } else {
                self.bounded()?.parse_kw_dispatch_expr(Some(var))
            }
        } else {
            self.bounded()?.parse_kw_dispatch_expr(None)
        }
    }

    fn parse_kw_dispatch_expr(
        &mut self,
        parsed_recv: Option<ast::Name<'buf>>,
    ) -> Result<ast::Expr<'buf>, ParserError> {
        let recv = self.bounded()?.parse_bin_dispatch_expr(parsed_recv)?;

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
            let arg = self.bounded()?.parse_bin_dispatch_expr(None)?;

            kws.push(Spanned::new_spanning(kw.into(), span));
            args.push(arg);
        }

        if let Some(kw) = kws.first() {
            let span = kw
                .span()
                .unwrap()
                .convex_hull(&args.last().unwrap().location().span().unwrap());
            let selector = ast::Selector::Keyword(kws);

            Ok(ast::Expr::Dispatch(ast::Dispatch {
                location: Location::UserCode(span),
                recv: Box::new(recv),
                selector,
                args,
            }))
        } else {
            Ok(recv)
        }
    }

    fn parse_bin_dispatch_expr(
        &mut self,
        parsed_recv: Option<ast::Name<'buf>>,
    ) -> Result<ast::Expr<'buf>, ParserError> {
        let mut recv = self.bounded()?.parse_un_dispatch_expr(parsed_recv)?;

        while let Some(Token {
            span: op_span,
            value: op_value,
        }) = self.try_consume(BinOpMatcher)?
        {
            let op = op_value.as_bin_op().unwrap().into_owned().into_str();
            let arg = self.bounded()?.parse_un_dispatch_expr(None)?;
            let span = recv
                .location()
                .span()
                .unwrap()
                .convex_hull(&arg.location().span().unwrap());
            let selector = ast::Selector::Binary(Spanned::new_spanning(op, op_span));

            recv = ast::Expr::Dispatch(ast::Dispatch {
                location: Location::UserCode(span),
                recv: Box::new(recv),
                selector,
                args: vec![arg],
            });
        }

        Ok(recv)
    }

    fn parse_un_dispatch_expr(
        &mut self,
        parsed_recv: Option<ast::Name<'buf>>,
    ) -> Result<ast::Expr<'buf>, ParserError> {
        let mut recv = self.bounded()?.parse_primary_expr(parsed_recv)?;

        while self.peek(VarNameMatcher)? {
            let name = self.parse_ident(PrimitiveAllowed::Yes)?;
            let name_span = name.location.span().unwrap();
            let selector = ast::Selector::Unary(name);
            let span = recv.location().span().unwrap().convex_hull(&name_span);

            recv = ast::Expr::Dispatch(ast::Dispatch {
                location: Location::UserCode(span),
                recv: Box::new(recv),
                selector,
                args: vec![],
            });
        }

        Ok(recv)
    }

    fn parse_primary_expr(
        &mut self,
        parsed_name: Option<ast::Name<'buf>>,
    ) -> Result<ast::Expr<'buf>, ParserError> {
        if parsed_name.is_some() {
            self.bounded()?.parse_var_expr(parsed_name)
        } else {
            lookahead!(self: {
                VarNameMatcher => self.bounded()?.parse_var_expr(None),
                Special::ParenLeft => self.bounded()?.parse_paren_expr(),
                Special::BracketLeft => self.bounded()?.parse_block_expr(),
                _ => self.bounded()?.parse_lit_expr(),
            })
        }
    }

    fn parse_var_expr(
        &mut self,
        parsed_name: Option<ast::Name<'buf>>,
    ) -> Result<ast::Expr<'buf>, ParserError> {
        let name = match parsed_name {
            Some(name) => name,
            None => self.parse_ident(PrimitiveAllowed::Yes)?,
        };

        Ok(ast::Expr::Var(ast::Var(name)))
    }

    fn parse_paren_expr(&mut self) -> Result<ast::Expr<'buf>, ParserError> {
        self.expect(Special::ParenLeft).unwrap();
        let expr = self.bounded()?.parse_expr()?;
        self.expect(Special::ParenRight)?;

        Ok(expr)
    }

    fn parse_block_expr(&mut self) -> Result<ast::Expr<'buf>, ParserError> {
        let block = self.bounded()?.parse_block_body(
            BlockParamsAllowed::Yes,
            Special::BracketLeft,
            Special::BracketRight,
        )?;

        Ok(ast::Expr::Block(block))
    }

    fn parse_lit_expr(&mut self) -> Result<ast::Expr<'buf>, ParserError> {
        lookahead!(self: {
            Special::ArrayLeft => self.bounded()?.parse_array_lit_expr(),
            TokenType::Symbol => self.bounded()?.parse_symbol_lit_expr(),
            TokenType::String => self.bounded()?.parse_string_lit_expr(),
            TokenType::Int => self.bounded()?.parse_num_lit_expr(),
            TokenType::Float => self.bounded()?.parse_num_lit_expr(),
            _ => return #error,
        })
    }

    fn parse_array_lit_expr(&mut self) -> Result<ast::Expr<'buf>, ParserError> {
        let left = self.expect(Special::ArrayLeft).unwrap();
        let mut values = vec![];

        let right = loop {
            if let Some(right) = self.try_consume(Special::ParenRight)? {
                break right;
            }

            values.push(self.parse_lit_expr()?);
        };

        Ok(ast::Expr::Array(ast::ArrayLit(Spanned::new_spanning(values, left.span.convex_hull(&right.span)))))
    }

    fn parse_symbol_lit_expr(&mut self) -> Result<ast::Expr<'buf>, ParserError> {
        let Token { span, value: TokenValue::Symbol(sym) } = self.expect(TokenType::Symbol).unwrap() else {
            unreachable!()
        };

        let sym = match sym {
            token::Symbol::String(s) => ast::SymbolLit::String(Spanned::new_spanning(s, span)),
            token::Symbol::UnarySelector(s) => ast::SymbolLit::Selector(Spanned::new_spanning(ast::Selector::Unary(Spanned::new_spanning(s.into(), span)), span)),
            token::Symbol::BinarySelector(op) => ast::SymbolLit::Selector(Spanned::new_spanning(ast::Selector::Binary(Spanned::new_spanning(op.into_str(), span)), span)),
            token::Symbol::KeywordSelector(kws) => {
                let kws = kws.into_iter().map(|token::Keyword { span, kw }| Spanned::new_spanning(kw.into(), span)).collect();

                ast::SymbolLit::Selector(Spanned::new_spanning(ast::Selector::Keyword(kws), span))
            },
        };

        Ok(ast::Expr::Symbol(sym))
    }

    fn parse_string_lit_expr(&mut self) -> Result<ast::Expr<'buf>, ParserError> {
        let Token { span, value: TokenValue::String(s) } = self.expect(TokenType::String).unwrap() else { unreachable!() };

        Ok(ast::Expr::String(ast::StringLit(Spanned::new_spanning(s, span))))
    }

    fn parse_num_lit_expr(&mut self) -> Result<ast::Expr<'buf>, ParserError> {
        let Token { span, value } = self.expect([TokenType::Int, TokenType::Float]).unwrap();

        Ok(match value {
            TokenValue::Int(n) => ast::Expr::Int(ast::IntLit(Spanned::new_spanning(n, span))),
            TokenValue::Float(n) => ast::Expr::Float(ast::FloatLit(Spanned::new_spanning(n, span))),
            _ => unreachable!(),
        })
    }

    fn parse_ident(
        &mut self,
        primitive_allowed: PrimitiveAllowed,
    ) -> Result<ast::Name<'buf>, ParserError> {
        let Token { span, value } = match primitive_allowed {
            PrimitiveAllowed::No => self.expect(TokenType::Ident)?,
            PrimitiveAllowed::Yes => self.expect(VarNameMatcher)?,
        };

        let name = match value {
            TokenValue::Special(s @ Special::Primitive) => s.as_str().into(),
            TokenValue::Ident(id) => id,
            _ => unreachable!(),
        };

        Ok(Spanned::new_spanning(name.into(), span))
    }
}
