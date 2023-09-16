use std::mem;
use std::ops::{Deref, DerefMut};

use miette::{Diagnostic, SourceOffset};
use thiserror::Error;

use super::token::{Special, Token, TokenType, TokenValue};
use super::{Lexer, LexerError};

use crate::ast;
use crate::location::{Location, Offset, Span, Spanned};
use crate::parse::token::BinOp;
use crate::util::format_list;

const RECURSION_LIMIT: usize = 1000;

#[derive(Error, Diagnostic, Debug, Clone, PartialEq)]
pub enum ParserError<'buf> {
    #[error("encountered an unexpected token: {} (expected {})", .actual.ty(), format_list(&.expected, "or"))]
    #[diagnostic(code(parser::unexpected_token))]
    UnexpectedToken {
        expected: Vec<TokenType<'buf>>,

        #[label]
        actual: Token<'buf>,
    },

    #[error("multiple class member separators present in the definition")]
    #[diagnostic(code(parser::multiple_separators))]
    MultipleSeparators {
        #[label]
        bad_separator: Token<'buf>,

        #[label = "first separator specified here"]
        first_separator: Token<'buf>,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum PrimitiveAllowed {
    Yes,
    No,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum BlockParamsAllowed {
    Yes,
    No,
}

trait Matcher<'a> {
    fn matches(&self, token: &Token<'_>) -> bool;

    fn expected_tokens(&self) -> Vec<TokenType<'a>>;
}

impl<'a, const N: usize> Matcher<'a> for [TokenType<'a>; N] {
    fn matches(&self, token: &Token<'_>) -> bool {
        self.contains(&token.ty())
    }

    fn expected_tokens(&self) -> Vec<TokenType<'a>> {
        self.to_vec()
    }
}

impl<const N: usize> Matcher<'static> for [Special; N] {
    fn matches(&self, token: &Token<'_>) -> bool {
        match token.ty() {
            TokenType::Special(sym) => self.contains(&sym),
            _ => false,
        }
    }

    fn expected_tokens(&self) -> Vec<TokenType<'static>> {
        self.iter().copied().map(TokenType::Special).collect()
    }
}

impl<'a> Matcher<'a> for TokenType<'a> {
    fn matches(&self, token: &Token<'_>) -> bool {
        self == &token.ty()
    }

    fn expected_tokens(&self) -> Vec<TokenType<'a>> {
        vec![self.clone()]
    }
}

impl Matcher<'static> for Special {
    fn matches(&self, token: &Token<'_>) -> bool {
        TokenType::Special(*self) == token.ty()
    }

    fn expected_tokens(&self) -> Vec<TokenType<'static>> {
        vec![TokenType::Special(*self)]
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

    fn expected_tokens(&self) -> Vec<TokenType<'static>> {
        vec![TokenType::BinOp(BinOp::new("----"))]
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
                        actual,
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
    attempted_tokens: Vec<TokenType<'buf>>,
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

    fn bounded(&mut self) -> Result<BoundedParser<'buf, '_>, ParserError<'buf>> {
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

    fn expect(&mut self, matcher: impl Matcher<'static>) -> Result<Token<'buf>, ParserError<'buf>> {
        expect_impl! {
            match self.try_consume!(matcher) {
                Ok(token) => Ok(token),

                Err(token) => Err(ParserError::UnexpectedToken {
                    expected: mem::take(&mut self.attempted_tokens),
                    actual: token.clone(),
                }),
            }
        }
    }

    fn try_consume(
        &mut self,
        matcher: impl Matcher<'static>,
    ) -> Result<Option<Token<'buf>>, ParserError<'buf>> {
        expect_impl! {
            match self.try_consume!(matcher) {
                Ok(token) => Ok(Some(token)),
                Err(_) => Ok(None),
            }
        }
    }

    fn peek(&mut self, matcher: impl Matcher<'static>) -> Result<bool, ParserError<'buf>> {
        expect_impl! {
            match self.peek!(matcher) {
                Ok(_) => Ok(true),
                Err(_) => Ok(false),
            }
        }
    }

    pub fn parse(mut self) -> Result<ast::Class<'buf>, ParserError<'buf>> {
        let class = self.parse_class()?;
        self.expect(TokenType::Eof)?;

        Ok(class)
    }

    fn parse_class(&mut self) -> Result<ast::Class<'buf>, ParserError<'buf>> {
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
                            let bad_separator = self.expect(SeparatorMatcher).unwrap();

                            return Err(ParserError::MultipleSeparators {
                                bad_separator,
                                first_separator: separator,
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

    fn parse_var_list(&mut self) -> Result<Vec<ast::Name<'buf>>, ParserError<'buf>> {
        self.expect(Special::Bar)?;

        let mut vars = vec![];

        while self.try_consume(Special::Bar)?.is_none() {
            vars.push(self.parse_ident(PrimitiveAllowed::Yes)?);
        }

        Ok(vars)
    }

    fn parse_method(&mut self) -> Result<ast::Method<'buf>, ParserError<'buf>> {
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
            location: Location::UserCode(selector.span().unwrap().convex_hull(&def.span().unwrap())),
            selector,
            def,
        })
    }

    fn parse_pattern(
        &mut self,
    ) -> Result<(Spanned<ast::Selector<'buf>>, Vec<ast::Name<'buf>>), ParserError<'buf>> {
        todo!()
    }

    fn parse_block_body(
        &mut self,
        params_allowed: BlockParamsAllowed,
        left_matcher: impl Matcher<'static>,
        right_matcher: impl Matcher<'static>,
    ) -> Result<Spanned<ast::Block<'buf>>, ParserError<'buf>> {
        todo!()
    }

    fn parse_ident(
        &mut self,
        primitive_allowed: PrimitiveAllowed,
    ) -> Result<ast::Name<'buf>, ParserError<'buf>> {
        todo!()
    }
}
