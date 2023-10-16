use std::borrow::Cow;

use miette::Diagnostic;
use thiserror::Error;

use crate::location::{Span, Spanned};
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

    #[error("no run method on class {class_name}")]
    NoRunMethod {
        #[label]
        class_span: Option<Span>,

        class_name: String,
    },

    #[error("not enough arguments to {callee_name}: {expected_count} expected, {provided_count} provided")]
    NotEnoughArguments {
        #[label]
        dispatch_span: Option<Span>,

        #[label = "callee defined here"]
        callee_span: Option<Span>,

        callee_name: String,
        expected_count: usize,
        provided_count: usize,
        missing_params: Vec<Spanned<String>>,
    },

    #[error(
        "too many arguments to {callee_name}: {expected_count} expected, {provided_count} provided"
    )]
    TooManyArguments {
        #[label]
        dispatch_span: Option<Span>,

        #[label]
        callee_span: Option<Span>,

        callee_name: String,
        expected_count: usize,
        provided_count: usize,
    },

    #[error("non-local return from an escaped block")]
    NlRetFromEscapedBlock {
        #[label]
        ret_span: Option<Span>,
        // TODO:
        // method_span: Option<Span>,
    },

    #[error("object of class `{class_name}` has no method `{method_name}`")]
    NoSuchMethod {
        #[label]
        span: Option<Span>,

        #[label = "class `{class_name}` defined here"]
        class_span: Option<Span>,

        class_name: String,
        method_name: String,
    },

    #[error("index `{idx}` is out of bounds for an array of size {size}")]
    IndexOutOfBounds {
        #[label]
        span: Option<Span>,

        idx: i64,
        size: usize,
    },

    #[error("array size {size} is too large")]
    ArrayTooLarge {
        #[label]
        span: Option<Span>,

        size: i64,
    },

    #[error("array size {size} is negative")]
    ArraySizeNegative {
        #[label]
        span: Option<Span>,

        size: i64,
    },

    #[error("floating-point value ({value}) is too large for an integer")]
    FloatTooLargeForInt {
        #[label]
        span: Option<Span>,

        value: f64,
    },

    #[error("floating-point value is a NaN and cannot be converted to an integer")]
    NanFloatToInt {
        #[label]
        span: Option<Span>,
    },

    #[error("string does not contain a valid floating-point value")]
    DoubleFromInvalidString {
        #[label]
        span: Option<Span>,

        value: String,
    },

    #[error("string does not contain a valid integer")]
    IntegerFromInvalidString {
        #[label]
        span: Option<Span>,

        value: String,
    },

    #[error("object of class `{class_name}` has no field `{field_name}`")]
    NoSuchField {
        #[label]
        span: Option<Span>,

        #[label = "class `{class_name}` defined here"]
        class_span: Option<Span>,

        class_name: String,
        field_name: String,
    },

    #[error("program exited with code {code}")]
    Exited {
        #[label]
        span: Option<Span>,

        code: i64,
    },

    #[error("start position ({start}) is greater than the end ({end})")]
    StartGtEnd {
        #[label]
        span: Option<Span>,

        start: usize,
        end: usize,
    },
}
