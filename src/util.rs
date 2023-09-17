use std::fmt;
use std::fmt::Display;

macro_rules! try_match {
    ($scrutinee:expr, $pattern:pat => $map:expr) => {
        match $scrutinee {
            $pattern => Some($map),
            _ => None,
        }
    };
}

pub(crate) use try_match;

pub fn format_list<'a, T: Display>(items: &'a [T], conjunction: &'a str) -> impl Display + 'a {
    struct ListFormatter<'a, T> {
        items: &'a [T],
        conjunction: &'a str,
    }

    impl<T: Display> Display for ListFormatter<'_, T> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self.items {
                [] => Ok(()),
                [item] => item.fmt(f),
                [first, second] => write!(f, "{} {} {}", first, self.conjunction, second),

                [first, middle @ .., last] => {
                    write!(f, "{}", first)?;

                    for item in middle {
                        write!(f, ", {}", item)?;
                    }

                    write!(f, ", {} {}", self.conjunction, last)
                }
            }
        }
    }

    ListFormatter { items, conjunction }
}

macro_rules! define_yes_no_options {
    { $($vis:vis enum $name:ident ;)* } => {
        $(
            #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
            $vis enum $name {
                Yes,
                No,
            }

            impl $name {
                pub fn is_yes(self) -> bool {
                    self == Self::Yes
                }

                pub fn is_no(self) -> bool {
                    self == Self::No
                }
            }
        )*
    };
}

pub(crate) use define_yes_no_options;
