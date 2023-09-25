use std::borrow::Cow;

macro_rules! try_match {
    ($scrutinee:expr, $pattern:pat => $map:expr) => {
        match $scrutinee {
            $pattern => Some($map),
            _ => None,
        }
    };
}

pub(crate) use try_match;

macro_rules! macro_cond {
    {
        if non_empty!( $( $cond:tt )+ ) {
            $( $true:tt )*
        } else {
            $( $false:tt )*
        }
    } => { $( $true )* };

    {
        if non_empty!() {
            $( $true:tt )*
        } else {
            $( $false:tt )*
        }
    } => { $( $false )* };
}

pub(crate) use macro_cond;

macro_rules! format_list {
    ($item_fmt:literal, $items:expr, $conjunction:expr) => {{
        struct ListFormatter<'a, T> {
            items: &'a [T],
            conjunction: &'a str,
        }

        impl<T: ::std::fmt::Display> ::std::fmt::Display for ListFormatter<'_, T> {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                match self.items {
                    [] => Ok(()),

                    [item] => format_args!($item_fmt, item).fmt(f),

                    [first, second] => write!(
                        f,
                        "{} {} {}",
                        format_args!($item_fmt, first),
                        self.conjunction,
                        format_args!($item_fmt, second)
                    ),

                    [first, middle @ .., last] => {
                        format_args!($item_fmt, first).fmt(f)?;

                        for item in middle {
                            write!(f, ", {}", format_args!($item_fmt, item))?;
                        }

                        write!(
                            f,
                            ", {} {}",
                            self.conjunction,
                            format_args!($item_fmt, last)
                        )
                    }
                }
            }
        }

        ListFormatter {
            items: $items,
            conjunction: $conjunction,
        }
    }};
}

pub(crate) use format_list;

macro_rules! define_yes_no_options {
    { $($( #[ $attr:meta ] )* $vis:vis enum $name:ident ;)* } => {
        $(
            $( #[ $attr ] )*
            #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
            $vis enum $name {
                Yes,
                No,
            }

            impl $name {
                #[allow(dead_code)]
                pub fn is_yes(self) -> bool {
                    self == Self::Yes
                }

                #[allow(dead_code)]
                pub fn is_no(self) -> bool {
                    self == Self::No
                }
            }

            impl From<bool> for $name {
                fn from(value: bool) -> Self {
                    if value { Self::Yes } else { Self::No }
                }
            }
        )*
    };
}

pub(crate) use define_yes_no_options;

pub trait CloneStatic<T>
where
    T: 'static,
{
    fn clone_static(&self) -> T;
}

impl<T: ?Sized + ToOwned> CloneStatic<Cow<'static, T>> for Cow<'_, T>
where
    <T as ToOwned>::Owned: Clone,
    <T as ToOwned>::Owned: 'static,
{
    fn clone_static(&self) -> Cow<'static, T> {
        match self {
            Cow::Borrowed(borrowed) => Cow::Owned((*borrowed).to_owned()),
            Cow::Owned(owned) => Cow::Owned(owned.clone()),
        }
    }
}

impl<T, U> CloneStatic<Option<U>> for Option<T>
where
    T: CloneStatic<U>,
    U: 'static,
{
    fn clone_static(&self) -> Option<U> {
        self.as_ref().map(T::clone_static)
    }
}

impl<T, U> CloneStatic<Vec<U>> for Vec<T>
where
    T: CloneStatic<U>,
    U: 'static,
{
    fn clone_static(&self) -> Vec<U> {
        self.iter().map(T::clone_static).collect()
    }
}
