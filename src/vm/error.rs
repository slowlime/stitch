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

    #[error("{}method `{name}` defined twice", if *.class_method { "class "} else { "" })]
    MethodCollision {
        #[label]
        span: Option<Span>,

        #[label = "first definition here"]
        prev_span: Option<Span>,

        name: String,
        class_method: bool,
    },

    #[error("unknown primitive {name} for class {class_name}")]
    UnknownPrimitive {
        #[label]
        span: Option<Span>,

        name: String,
        class_name: String,
    },
}
