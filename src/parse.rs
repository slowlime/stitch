mod cursor;
mod lexer;
mod parser;
pub mod token;

pub use cursor::Cursor;
pub use lexer::{Lexer, LexerError};
pub use parser::{Parser, ParserError};

use crate::ast;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BigNumberBehavior {
    Error,
    Saturate,
}

impl Default for BigNumberBehavior {
    fn default() -> Self {
        Self::Error
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct ParserOptions {
    pub big_numbers: BigNumberBehavior,
}

pub fn parse<'buf>(
    buf: &'buf str,
    options: ParserOptions,
) -> Result<ast::Class<'buf>, ParserError> {
    let cursor = Cursor::new(buf);
    let lexer = Lexer::new(cursor);
    let parser = Parser::new(lexer, options);

    parser.parse()
}
