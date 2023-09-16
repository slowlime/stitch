use std::collections::HashSet;
use std::borrow::Cow;
use std::fmt::Display;
use std::sync::OnceLock;

use phf::phf_map;
use miette::SourceSpan;

use crate::location::Span;

#[derive(Debug, Clone, PartialEq)]
pub struct Token<'buf> {
    pub span: Span,
    pub value: TokenValue<'buf>,
}

impl<'buf> Token<'buf> {
    pub fn ty(&self) -> TokenType<'buf> {
        self.value.ty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BinOp<'a>(Cow<'a, str>);

impl<'a> BinOp<'a> {
    pub fn new(s: impl Into<Cow<'a, str>>) -> Self {
        Self(s.into())
    }

    pub fn is_separator(&self) -> bool {
        self.0.len() >= 4 && self.0.chars().all(|c| c == '-')
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TokenType<'a> {
    Int,
    Float,
    String,
    Ident,
    BinOp(BinOp<'a>),
    Special(Special),
    Eof,
}

impl Display for TokenType<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Int => "integer",
                Self::Float => "float",
                Self::String => "string",
                Self::Ident => "identifier",
                Self::BinOp(op) => op.as_str(),
                Self::Special(s) => s.as_str(),
                Self::Eof => "end of file",
            }
        )
    }
}

impl<'buf> From<Token<'buf>> for SourceSpan {
    fn from(token: Token<'buf>) -> SourceSpan {
        token.span.into()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TokenValue<'buf> {
    Int(i64),
    Float(f64),
    String(Cow<'buf, str>),
    Ident(&'buf str),
    BinOp(BinOp<'buf>),
    Special(Special),
    Eof,
}

impl<'buf> TokenValue<'buf> {
    pub fn ty(&self) -> TokenType<'buf> {
        match *self {
            Self::Int(_) => TokenType::Int,
            Self::Float(_) => TokenType::Float,
            Self::String(_) => TokenType::String,
            Self::Ident(_) => TokenType::Ident,
            Self::BinOp(ref op) => TokenType::BinOp(op.clone()),
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

        impl Special {
            const SPECIALS: phf::Map<&'static str, Special> = phf_map! {
                $( $lit => Self::$variant, )+
            };

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
