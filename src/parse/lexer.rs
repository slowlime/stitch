use std::borrow::Cow;
use std::fmt::{self, Display};
use std::iter::FusedIterator;
use std::num::IntErrorKind;

use miette::{Diagnostic, SourceOffset};
use thiserror::Error;

use crate::location::Span;
use crate::parse::cursor::Cursor;
use crate::parse::token::{is_bin_op_char, BinOp};

use super::token::{Keyword, Special, Symbol, Token, TokenValue};

type ScanResult<'a> = Result<TokenValue<'a>, PosLexerError>;

fn is_whitespace(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\r' | '\n')
}

fn is_ident_start(c: char) -> bool {
    // this is stricter than needed (the grammar says \p{Alpha})
    c.is_ascii_alphabetic()
}

fn is_ident_continuation(c: char) -> bool {
    // this is stricter than needed (the grammar says \p{Alpha})
    c.is_ascii_alphanumeric() || c == '_'
}

fn make_ident_matcher() -> impl FnMut(char) -> bool {
    let mut first = true;

    move |c| {
        if first {
            first = false;

            is_ident_start(c)
        } else {
            is_ident_continuation(c)
        }
    }
}

pub fn is_ident(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }

    s.chars().all(make_ident_matcher())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum MatchMode {
    Exact,
    Prefix,
}

fn scan_special(s: &str, mode: MatchMode) -> Option<TokenValue<'_>> {
    let result = match mode {
        MatchMode::Exact => Special::parse_exact(s),
        MatchMode::Prefix => Special::parse_prefix(s),
    };

    result.map(TokenValue::Special)
}

fn format_char(c: char) -> impl Display {
    struct CharFormatter(char);

    impl Display for CharFormatter {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            if self.0.is_ascii_graphic() {
                write!(f, "{}", self.0)
            } else {
                write!(f, "U+{:04x}", self.0 as u32)
            }
        }
    }

    CharFormatter(c)
}

#[derive(Error, Diagnostic, Debug, Clone, Copy, Eq, PartialEq)]
pub enum LexerErrorKind {
    #[error("the comment is not terminated")]
    #[diagnostic(code(lexer::unterminated_comment))]
    UnterminatedComment,

    #[error("the string is not terminated")]
    #[diagnostic(code(lexer::unterminated_string))]
    UnterminatedString,

    #[error("the escape sequence '\\{c}' is invalid")]
    #[diagnostic(code(lexer::invalid_escape))]
    InvalidEscape {
        c: char,

        #[label = "The escape sequence is here"]
        span: Span,
    },

    #[error("`#` must be followed by a selector, string, or a left parenthesis")]
    #[diagnostic(code(lexer::illegal_octothorpe))]
    IllegalOctothorpe,

    #[error("keyword selector must only contain keywords")]
    #[diagnostic(code(lexer::malformed_kw_selector))]
    MalformedKwSelector,

    #[error("encountered an unrecognized character {}", format_char(*.0))]
    #[diagnostic(code(lexer::unrecognized_character))]
    UnrecognizedCharacter(char),
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct PosLexerError {
    end: SourceOffset,
    kind: LexerErrorKind,
}

impl PosLexerError {
    fn with_start(self, start: SourceOffset) -> LexerError {
        LexerError {
            kind: self.kind,
            span: (start.offset()..self.end.offset()).into(),
        }
    }
}

#[derive(Error, Diagnostic, Debug, Clone, Eq, PartialEq)]
pub struct LexerError {
    pub kind: LexerErrorKind,

    #[label]
    pub span: Span,
}

impl Display for LexerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "lexical analysis failed: {}", self.kind)
    }
}

#[derive(Debug, Clone)]
pub struct Lexer<'a> {
    cursor: Cursor<'a>,
    eof: bool,
}

impl<'a> Lexer<'a> {
    pub fn new(cursor: Cursor<'a>) -> Self {
        Self { cursor, eof: false }
    }

    pub fn pos(&self) -> SourceOffset {
        self.cursor.pos()
    }

    fn make_error_at_pos(&self, kind: LexerErrorKind) -> PosLexerError {
        PosLexerError {
            end: self.cursor.pos(),
            kind,
        }
    }

    fn skip_whitespace(&mut self) {
        self.cursor.consume_while(is_whitespace);
    }

    fn skip_comment(&mut self) -> Result<(), PosLexerError> {
        self.cursor.consume_expecting("\"").unwrap();
        self.cursor.consume_while(|c| c != '"');

        self.cursor
            .consume_expecting("\"")
            .map(|_| ())
            .ok_or_else(|| self.make_error_at_pos(LexerErrorKind::UnterminatedComment))
    }

    fn scan_string(&mut self) -> ScanResult<'a> {
        self.cursor.consume_expecting("'").unwrap();

        let buf = self.cursor.remaining();
        let mut token_value = Cow::Borrowed("");
        let mut escape_pos: Option<SourceOffset> = None;
        let mut invalid_escape: Option<(Span, char)> = None;

        loop {
            let c = match self.cursor.next() {
                Some(c) => c,
                None => return Err(self.make_error_at_pos(LexerErrorKind::UnterminatedString)),
            };

            match escape_pos {
                Some(backslash_pos) => {
                    'push: {
                        token_value.to_mut().push(match c {
                            't' => '\t',
                            'b' => '\x08', // backspace
                            'n' => '\n',
                            'r' => '\r',
                            'f' => '\x0c', // form feed
                            '0' => '\0',
                            '\\' | '\'' | '|' => c,

                            _ => {
                                if invalid_escape.is_none() {
                                    invalid_escape = Some((
                                        (backslash_pos.offset()..self.cursor.pos().offset()).into(),
                                        c,
                                    ));
                                }

                                break 'push;
                            }
                        });
                    }

                    escape_pos = None;
                }

                None => match c {
                    '\\' => {
                        // start of an escape sequence: save the position of this backslash for error messages
                        escape_pos = Some((self.pos().offset() - 1).into());
                    }

                    // a terminating quote
                    '\'' => break,

                    // something else — append to the scanned value
                    _ => match token_value {
                        Cow::Owned(ref mut s) => s.push(c),

                        Cow::Borrowed(ref mut s) => {
                            // simply increase the extent of the borrowed slice
                            *s = &buf[0..s.len() + c.len_utf8()];
                        }
                    },
                },
            }
        }

        if let Some((span, c)) = invalid_escape {
            Err(self.make_error_at_pos(LexerErrorKind::InvalidEscape { span, c }))
        } else {
            Ok(TokenValue::String(token_value))
        }
    }

    fn scan_symbol_or_array(&mut self) -> ScanResult<'a> {
        self.cursor.consume_expecting("#").unwrap();

        match self.cursor.peek() {
            Some('\'') => self.scan_string_symbol(),
            Some('(') => self.scan_array(),
            Some(c) if is_bin_op_char(c) => self.scan_bin_selector(),
            Some(c) if is_ident_start(c) => self.scan_un_or_kw_selector(),
            Some(_) | None => Err(self.make_error_at_pos(LexerErrorKind::IllegalOctothorpe)),
        }
    }

    fn scan_array(&mut self) -> ScanResult<'a> {
        self.cursor.consume_expecting("(").unwrap();

        Ok(TokenValue::Special(Special::ArrayLeft))
    }

    fn scan_string_symbol(&mut self) -> ScanResult<'a> {
        let TokenValue::String(value) = self.scan_string()? else {
            unreachable!()
        };

        Ok(TokenValue::Symbol(Symbol::String(value)))
    }

    fn scan_bin_selector(&mut self) -> ScanResult<'a> {
        let TokenValue::BinOp(bin_op) = self.scan_bin_op()? else {
            unreachable!()
        };

        Ok(TokenValue::Symbol(Symbol::BinarySelector(bin_op)))
    }

    fn scan_un_or_kw_selector(&mut self) -> ScanResult<'a> {
        let start = self.pos();

        match self.scan_ident()? {
            TokenValue::Special(s @ Special::Primitive) => {
                Ok(TokenValue::Symbol(Symbol::UnarySelector(s.as_str().into())))
            }
            TokenValue::Ident(id) => Ok(TokenValue::Symbol(Symbol::UnarySelector(id))),
            TokenValue::Keyword(kw) => self.scan_kw_selector(start, kw),
            _ => unreachable!(),
        }
    }

    fn scan_kw_selector(
        &mut self,
        first_kw_start: SourceOffset,
        first_kw: Cow<'a, str>,
    ) -> ScanResult<'a> {
        let mut kws = vec![Keyword {
            span: (first_kw_start..self.pos()).into(),
            kw: first_kw,
        }];

        loop {
            match self.cursor.peek() {
                Some(c) if is_ident_start(c) => {
                    let start = self.pos();

                    match self.scan_ident()? {
                        TokenValue::Keyword(kw) => kws.push(Keyword {
                            span: (start..self.pos()).into(),
                            kw,
                        }),

                        _ => {
                            return Err(self.make_error_at_pos(LexerErrorKind::MalformedKwSelector))
                        }
                    }
                }

                Some(c) if is_bin_op_char(c) => {
                    return Err(self.make_error_at_pos(LexerErrorKind::MalformedKwSelector))
                }

                _ => break,
            }
        }

        Ok(TokenValue::Symbol(Symbol::KeywordSelector(kws)))
    }

    fn scan_bin_op(&mut self) -> ScanResult<'a> {
        let op = self.cursor.consume_while(is_bin_op_char);
        assert!(!op.is_empty());

        Ok(TokenValue::BinOp(BinOp::new(op)))
    }

    fn scan_block_param(&mut self) -> ScanResult<'a> {
        self.cursor.consume_expecting(":").unwrap();
        let id = self.cursor.consume_while(make_ident_matcher());

        // guaranteed by the main scan loop
        debug_assert!(!id.is_empty());

        Ok(TokenValue::BlockParam(id.into()))
    }

    fn scan_ident(&mut self) -> ScanResult<'a> {
        let ident = self.cursor.consume_while(make_ident_matcher());
        debug_assert!(!ident.is_empty());

        Ok(if let Some(s) = scan_special(ident, MatchMode::Exact) {
            s
        } else if !self.cursor.starts_with(":=") && self.cursor.consume_expecting(":").is_some() {
            // the condition ensures x:=y is not tokenized as Keyword("x") Equals Ident("y")
            TokenValue::Keyword(ident.into())
        } else {
            TokenValue::Ident(ident.into())
        })
    }

    fn scan_number(&mut self) -> ScanResult<'a> {
        let buf = self.cursor.remaining();
        let start_pos = self.cursor.pos().offset();
        let integer_part = self.cursor.consume_while(|c| c.is_ascii_digit());
        debug_assert!(!integer_part.is_empty());

        // only eat the dot if it's immediately followed by a digit
        if self.cursor.starts_with(".")
            && self.cursor.peek_nth(1).is_some_and(|c| c.is_ascii_digit())
        {
            self.cursor.consume_expecting(".").unwrap();
            let fractional_part = self.cursor.consume_while(|c| c.is_ascii_digit());
            debug_assert!(!fractional_part.is_empty());

            let end_pos = self.cursor.pos().offset();
            let literal = &buf[0..(end_pos - start_pos)];
            let value = literal.parse::<f64>().unwrap();

            Ok(TokenValue::Float(value))
        } else {
            let end_pos = self.cursor.pos().offset();

            let value = match buf[0..(end_pos - start_pos)].parse::<u64>() {
                Ok(value) => value,

                Err(e) => match e.kind() {
                    IntErrorKind::PosOverflow => u64::MAX,
                    _ => unreachable!(),
                },
            };

            Ok(TokenValue::Int(value))
        }
    }
}

impl<'a> Iterator for Lexer<'a> {
    type Item = Result<Token<'a>, LexerError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.eof {
            return None;
        }

        let mut start;

        let scan_result = loop {
            start = self.cursor.pos();

            break match self.cursor.peek() {
                None if self.eof => return None,

                None => {
                    self.eof = true;

                    return Some(Ok(Token {
                        value: TokenValue::Eof,
                        span: Span::new_with_extent(start, 0),
                    }));
                }

                Some('"') => match self.skip_comment() {
                    Ok(()) => continue,
                    Err(e) => Err(e),
                },

                Some(c) if is_whitespace(c) => {
                    self.skip_whitespace();

                    continue;
                }

                Some('\'') => self.scan_string(),

                Some('#') => self.scan_symbol_or_array(),

                Some(':') if self.cursor.peek_nth(1).is_some_and(is_ident_start) => {
                    self.scan_block_param()
                }

                Some(c)
                    if is_bin_op_char(c) && self.cursor.peek_nth(1).is_some_and(is_bin_op_char) =>
                {
                    self.scan_bin_op()
                }

                Some(c) if c.is_ascii_digit() => self.scan_number(),

                Some(c) if is_ident_start(c) => self.scan_ident(),

                Some(c) => match scan_special(self.cursor.remaining(), MatchMode::Prefix) {
                    Some(value) => {
                        let TokenValue::Special(s) = value else {
                            unreachable!()
                        };
                        self.cursor.consume_n(s.as_str().len());

                        Ok(value)
                    }

                    None => Err(self.make_error_at_pos(LexerErrorKind::UnrecognizedCharacter(c))),
                },
            };
        };

        Some(match scan_result {
            Ok(value) => Ok(Token {
                span: (start..self.pos()).into(),
                value,
            }),

            Err(err) => {
                self.eof = true;

                Err(err.with_start(start))
            }
        })
    }
}

impl<'a> FusedIterator for Lexer<'a> {}
