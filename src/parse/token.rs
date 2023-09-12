use std::collections::HashSet;
use std::borrow::Cow;
use std::fmt::Display;
use std::sync::OnceLock;

use phf::phf_map;
use miette::SourceSpan;

#[derive(Debug, Clone, PartialEq)]
pub struct Token<'buf> {
    pub span: SourceSpan,
    pub value: TokenValue<'buf>,
}

impl Token<'_> {
    pub fn ty(&self) -> TokenType {
        self.value.ty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenType {
    Int,
    Float,
    String,
    Ident,
    Separator,
    Special(Special),
    Eof,
}

impl Display for TokenType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Int => "integer",
                Self::Float => "float",
                Self::String => "string",
                Self::Ident => "identifier",
                Self::Separator => "separator",
                Self::Special(s) => s.as_str(),
                Self::Eof => "end of file",
            }
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TokenValue<'buf> {
    Int(i64),
    Float(f64),
    String(Cow<'buf, str>),
    Ident(&'buf str),
    Separator,
    Special(Special),
    Eof,
}

impl TokenValue<'_> {
    pub fn ty(&self) -> TokenType {
        match *self {
            Self::Int(_) => TokenType::Int,
            Self::Float(_) => TokenType::Float,
            Self::String(_) => TokenType::String,
            Self::Ident(_) => TokenType::Ident,
            Self::Separator => TokenType::Separator,
            Self::Special(s) => TokenType::Special(s),
            Self::Eof => TokenType::Eof,
        }
    }
}

macro_rules! specials {
    { $( $lit:literal => $variant:ident ),+ $(,)? } => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum Special {
            $( $variant, )+
        }

        const fn max_arr<const N: usize>(values: [usize; N]) -> usize {
            const fn max_arr_impl<const N: usize>(acc: usize, idx: usize, values: [usize; N]) -> usize {
                if idx >= N {
                    acc
                } else {
                    let x = values[idx];

                    max_arr_impl(if x > acc { x } else { acc }, idx + 1, values)
                }
            }

            max_arr_impl(0, 0, values)
        }

        impl Special {
            const SPECIALS: phf::Map<&'static str, Special> = phf_map! {
                $( $lit => Self::$variant, )+
            };

            const MAX_LENGTH: usize = max_arr([$( $lit.len(), )+]);

            fn get_prefix_lengths() -> &'static [usize] {
                static PREFIX_LENGTHS: OnceLock<Vec<usize>> = OnceLock::new();

                PREFIX_LENGTHS.get_or_init(|| {
                    let mut lengths = HashSet::new();
                    $( lengths.insert($lit.len()); )+

                    let mut lengths = lengths.into_iter().collect::<Vec<_>>();
                    lengths.sort_unstable();
                    lengths
                })
            }

            /// Tries to parse the beginning of `input` as a special.
            pub fn parse_prefix(input: &str) -> Option<Special> {
                Self::get_prefix_lengths()
                    .iter()
                    .filter_map(|&len| input.get(0..len))
                    .find_map(|prefix| Self::SPECIALS.get(prefix))
                    .copied()
            }

            pub fn parse_exact(input: &str) -> Option<Special> {
                Self::SPECIALS.get(input).copied()
            }

            pub fn as_str(&self) -> &'static str {
                match self {
                    $( Self::$variant => $lit, )+
                }
            }
        }
    };
}

specials! {
    "~" => Tilde,
    "&" => Ampersand,
    "|" => Bar,
    "*" => Asterisk,
    "/" => Slash,
    "\\" => Backslash,
    "+" => Plus,
    "-" => Minus,
    "=" => Equals,
    ">" => Greater,
    "<" => Less,
    "," => Comma,
    "@" => At,
    "%" => Percent,

    "(" => ParenLeft,
    ")" => ParenRight,
    "[" => BracketLeft,
    "]" => BracketRight,
    "." => Dot,
    "#" => Octothorpe,
    "^" => Circumflex,

    "primitive" => Primitive,
}
