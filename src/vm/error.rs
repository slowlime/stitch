use std::borrow::Cow;

use miette::Diagnostic;
use thiserror::Error;

use crate::location::Span;
use crate::util::format_list;

use super::value::Ty;

#[derive(Diagnostic, Error, Debug, Clone)]
pub enum VmError {
    #[error("name not defined: `{name}`")]
    UndefinedName {
        #[label]
        span: Option<Span>,

        name: String,
    },

    #[error("illegal type: expected {}, got {actual}", format_list!("{}", .expected, "or"))]
    IllegalTy {
        #[label]
        span: Option<Span>,

        expected: Cow<'static, [Ty]>,
        actual: Ty,
    },
}
