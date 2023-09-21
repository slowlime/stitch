mod cursor;
mod lexer;
mod parser;
pub mod token;

pub use cursor::Cursor;
pub use lexer::{Lexer, LexerError};
pub use parser::{Parser, ParserError};

use crate::ast;
use crate::sourcemap::SourceFile;

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

pub fn parse<'a>(
    file: &'a SourceFile,
    options: ParserOptions,
) -> Result<ast::Class<'a>, ParserError> {
    let cursor = Cursor::new(file);
    let lexer = Lexer::new(cursor);
    let parser = Parser::new(lexer, options);

    parser.parse()
}
