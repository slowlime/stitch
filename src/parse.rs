mod cursor;
mod lexer;
mod parser;
pub mod token;

pub use cursor::Cursor;
pub use lexer::{Lexer, LexerError};
pub use parser::{Parser, ParserError};

use crate::ast;
use crate::sourcemap::SourceFile;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum BigNumberBehavior {
    #[default]
    Error,

    Saturate,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct ParserOptions {
    pub big_numbers: BigNumberBehavior,
}

pub fn parse(file: &SourceFile, options: ParserOptions) -> Result<ast::Class<'_>, ParserError> {
    let cursor = Cursor::new(file);
    let lexer = Lexer::new(cursor);
    let parser = Parser::new(lexer, options);

    parser.parse()
}
