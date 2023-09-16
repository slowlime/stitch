use std::ops::{Range, RangeBounds};

use miette::{SourceOffset, SourceSpan};

use crate::util::try_match;

pub trait Offset: Copy {
    fn offset(self) -> usize;
}

impl Offset for usize {
    fn offset(self) -> usize {
        self
    }
}

impl Offset for SourceOffset {
    fn offset(self) -> usize {
        self.offset()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Span {
    pub start: usize,
    pub len: usize,
}

impl Span {
    pub fn new_with_extent(start: impl Offset, len: usize) -> Self {
        Self {
            start: start.offset(),
            len,
        }
    }

    pub fn new_spanning<T: Offset>(range: impl RangeBounds<T>) -> Self {
        use std::ops::Bound;

        let start = match range.start_bound() {
            Bound::Included(&start) => start.offset(),
            Bound::Excluded(&start) => start.offset() + 1,
            Bound::Unbounded => panic!("span is unbounded"),
        };

        let end = match range.end_bound() {
            Bound::Included(&end) => end.offset() + 1,
            Bound::Excluded(&end) => end.offset(),
            Bound::Unbounded => panic!("span is unbounded"),
        };

        assert!(start <= end, "end offset precedes start offset");

        Self::new_with_extent(start, end - start)
    }

    pub fn start(&self) -> SourceOffset {
        self.start.into()
    }

    pub fn end(&self) -> SourceOffset {
        (self.start + self.len).into()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn convex_hull(&self, other: &Self) -> Self {
        let start = self.start.min(other.start);
        let end = self.end().offset().max(other.end().offset());

        Self::new_spanning(start..end)
    }
}

impl From<SourceSpan> for Span {
    fn from(span: SourceSpan) -> Self {
        Self {
            start: span.offset(),
            len: span.len(),
        }
    }
}

impl From<Span> for SourceSpan {
    fn from(span: Span) -> Self {
        Self::new(span.start.into(), span.len.into())
    }
}

impl From<Range<usize>> for Span {
    fn from(range: Range<usize>) -> Self {
        Self::new_spanning(range)
    }
}

impl From<Range<SourceOffset>> for Span {
    fn from(range: Range<SourceOffset>) -> Self {
        Self::new_spanning(range.start.offset()..range.end.offset())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Location {
    UserCode(Span),
    Builtin,
}

impl Location {
    pub fn span(&self) -> Option<Span> {
        try_match!(*self, Location::UserCode(span) => span)
    }
}

impl From<Location> for Option<SourceSpan> {
    fn from(location: Location) -> Self {
        try_match!(location, Location::UserCode(span) => span.into())
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

    pub fn new_spanning(value: T, span: Span) -> Self {
        Self {
            location: Location::UserCode(span),
            value,
        }
    }

    pub fn span(&self) -> Option<Span> {
        self.location.span()
    }
}
