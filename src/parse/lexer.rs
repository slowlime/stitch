use std::borrow::Cow;
use std::fmt::{self, Display};
use std::iter::FusedIterator;
use std::num::{IntErrorKind, ParseIntError};

use miette::{Diagnostic, SourceOffset, SourceSpan};
use thiserror::Error;

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
    #[error("the number literal is too large")]
    #[diagnostic(code(lexer::number_too_large))]
    NumberTooLarge,

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
        span: SourceSpan,
    },

    #[error("encountered an unrecognized character {}", format_char(*.0))]
    #[diagnostic(code(lexer::unrecognized_character))]
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
    pub span: SourceSpan,
}

impl Display for LexerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "lexical analysis failed: {}", self.kind)
    }
}

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
        let mut invalid_escape: Option<(SourceSpan, char)> = None;

        loop {
            let c = match self.cursor.next() {
                Some(c) => c,
                None => return Err(self.make_error_at_pos(LexerErrorKind::UnterminatedString)),
            };

            match escape_pos {
                Some(backslash_pos) => {
                    token_value.to_mut().push(match c {
                        't' => '\t',
                        'b' => '\x08', // backspace
                        'n' => '\n',
                        'r' => '\r',
                        'f' => '\x0c', // form feed
                        '0' => '\0',
                        '\'' | '|' => c,

                        _ => {
                            if invalid_escape.is_none() {
                                invalid_escape = Some((
                                    (backslash_pos.offset()..self.cursor.pos().offset()).into(),
                                    c,
                                ));
                            }

                            continue;
                        }
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

        if let Some((span, c)) = invalid_escape {
            Err(self.make_error_at_pos(LexerErrorKind::InvalidEscape { span, c }))
        } else {
            Ok(TokenValue::String(token_value))
        }
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
                .map_err(|e| self.make_error_at_pos(e.into()))
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
