mod cursor;
mod lexer;
mod parser;
pub mod token;

pub use cursor::Cursor;
pub use lexer::{Lexer, LexerError};
pub use parser::{Parser, ParserError};

use crate::ast;

pub fn parse<'buf>(buf: &'buf str) -> Result<ast::Class<'buf>, ParserError<'buf>> {
    let cursor = Cursor::new(buf);
    let lexer = Lexer::new(cursor);
    let parser = Parser::new(lexer);

    parser.parse()
}
