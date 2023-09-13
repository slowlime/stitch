use std::borrow::Cow;
use std::error::Error;
use std::fmt::Display;
use std::iter::FusedIterator;
use std::num::{IntErrorKind, ParseIntError};

use miette::{SourceOffset, SourceSpan};

use crate::parse::cursor::Cursor;

use super::token::{Special, Token, TokenValue};

type ScanResult<'buf> = Result<TokenValue<'buf>, PosLexerError>;

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

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum LexerErrorKind {
    NumberTooLarge,
    UnterminatedComment,
    UnterminatedString,
    UnrecognizedCharacter(char),
}

impl From<ParseIntError> for LexerErrorKind {
    fn from(err: ParseIntError) -> Self {
        match err.kind() {
            IntErrorKind::PosOverflow | IntErrorKind::NegOverflow => Self::NumberTooLarge,
            _ => unimplemented!(),
        }
    }
}

impl Display for LexerErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let &Self::UnrecognizedCharacter(c) = self {
            write!(f, "encountered an unrecognized character '")?;

            if c.is_ascii_graphic() {
                write!(f, "{}", c)?;
            } else {
                write!(f, "U+{:04x}", c as u32)?;
            }

            return write!(f, "'");
        }

        write!(
            f,
            "{}",
            match self {
                Self::NumberTooLarge => "the number literal is too large",
                Self::UnterminatedComment => "the comment is not terminated",
                Self::UnterminatedString => "the string is not terminated",

                Self::UnrecognizedCharacter(_) => unreachable!(),
            }
        )
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct PosLexerError {
    end: SourceOffset,
    kind: LexerErrorKind,
}

impl PosLexerError {
    fn with_start(self, start: SourceOffset) -> LexerError {
        LexerError {
            span: (start.offset()..self.end.offset()).into(),
            kind: self.kind,
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct LexerError {
    span: SourceSpan,
    kind: LexerErrorKind,
}

impl Display for LexerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "lexical analysis failed: {}", self.kind)
    }
}

impl Error for LexerError {}

#[derive(Debug, Clone)]
pub struct Lexer<'buf> {
    cursor: Cursor<'buf>,
    eof: bool,
}

impl<'buf> Lexer<'buf> {
    pub fn new(cursor: Cursor<'buf>) -> Self {
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
        self.cursor.consume_while(|c| is_whitespace(c));
    }

    fn skip_comment(&mut self) -> Result<(), PosLexerError> {
        self.cursor.consume_expecting("'").unwrap();
        self.cursor.consume_while(|c| c != '\'');

        self.cursor
            .consume_expecting("'")
            .map(|_| ())
            .ok_or_else(|| self.make_error_at_pos(LexerErrorKind::UnterminatedComment))
    }

    fn scan_string(&mut self) -> ScanResult<'buf> {
        self.cursor.consume_expecting("'").unwrap();

        let buf = self.cursor.remaining();
        let mut token_value = Cow::Borrowed("");
        let mut escape_pos: Option<SourceOffset> = None;

        loop {
            let c = match self.cursor.next() {
                Some(c) => c,
                None => return Err(self.make_error_at_pos(LexerErrorKind::UnterminatedString)),
            };

            match escape_pos {
                Some(pos) => {
                    token_value.to_mut().push(match c {
                        't' => '\t',
                        'b' => '\x08', // backspace
                        'n' => '\n',
                        'r' => '\r',
                        'f' => '\x0c', // form feed
                        '0' => '\0',
                        '\'' | '|' => c,
                        _ => todo!("return an error somehow... can't really `return` here cause that's gonna set the start to the beginning of the string and I'd rather it started at the backslash"),
                    });

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

        Ok(TokenValue::String(token_value))
    }

    fn scan_separator(&mut self) -> ScanResult<'buf> {
        self.cursor.consume_expecting("----").unwrap();
        self.cursor.consume_while(|c| c == '-');

        Ok(TokenValue::Separator)
    }

    fn scan_ident(&mut self) -> ScanResult<'buf> {
        let ident = self.cursor.consume_while(is_ident_continuation);

        Ok(scan_special(ident, MatchMode::Exact).unwrap_or(TokenValue::Ident(ident)))
    }

    fn scan_number(&mut self) -> ScanResult<'buf> {
        let buf = self.cursor.remaining();
        let integer_part = self.cursor.consume_while(|c| c.is_ascii_digit());
        debug_assert!(!integer_part.is_empty());

        // only eat the dot if it's immediately followed by a digit
        if self.cursor.starts_with(".")
            && self.cursor.peek_nth(1).is_some_and(|c| c.is_ascii_digit())
        {
            self.cursor.consume_expecting(".").unwrap();
            let fractional_part = self.cursor.consume_while(|c| c.is_ascii_digit());
            debug_assert!(!fractional_part.is_empty());

            let literal = &buf[0..integer_part.len() + 1 + fractional_part.len()];

            Ok(TokenValue::Float(literal.parse().unwrap()))
        } else {
            integer_part
                .parse::<i64>()
                .map_err(|_| self.make_error_at_pos(LexerErrorKind::NumberTooLarge))
                .map(TokenValue::Int)
        }
    }
}

impl<'buf> Iterator for Lexer<'buf> {
    type Item = Result<Token<'buf>, LexerError>;

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
                        span: SourceSpan::new(start, 0.into()),
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

                Some('-') if self.cursor.starts_with("----") => self.scan_separator(),

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
                span: (start.offset()..self.pos().offset()).into(),
                value,
            }),

            Err(err) => {
                self.eof = true;

                Err(err.with_start(start))
            }
        })
    }
}

impl<'buf> FusedIterator for Lexer<'buf> {}
