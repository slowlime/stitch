use miette::SourceSpan;

use crate::util::try_match;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Location {
    UserCode {
        file: String,
        span: SourceSpan,
    },

    Builtin,
}

impl From<Location> for Option<SourceSpan> {
    fn from(location: Location) -> Self {
        try_match!(location, Location::UserCode { span, .. } => span)
    }
}

impl Default for Location {
    fn default() -> Self {
        Location::Builtin
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Spanned<T> {
    pub location: Location,
    pub value: T,
}

impl<T> Spanned<T> {
    pub fn new(value: T, location: Location) -> Self {
        Self { location, value }
    }

    pub fn new_builtin(value: T) -> Self {
        Self {
            location: Location::Builtin,
            value,
        }
    }
}
