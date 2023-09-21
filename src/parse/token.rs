use std::collections::HashSet;
use std::borrow::Cow;
use std::fmt::Display;
use std::sync::OnceLock;

use phf::phf_map;
use miette::SourceSpan;

use crate::location::Span;
use crate::util::CloneStatic;

pub fn is_bin_op_char(c: char) -> bool {
    matches!(
        c,
        '~' | '&' | '|' | '*' | '/' | '\\' | '+' | '=' | '>' | '<' | ',' | '@' | '%' | '-'
    )
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token<'a> {
    pub span: Span,
    pub value: TokenValue<'a>,
}

impl<'a> Token<'a> {
    pub fn ty(&self) -> TokenType<'a> {
        self.value.ty()
    }
}

impl CloneStatic<Token<'static>> for Token<'_> {
    fn clone_static(&self) -> Token<'static> {
        Token {
            span: self.span,
            value: self.value.clone_static(),
        }
    }
}

impl<'a> From<Token<'a>> for SourceSpan {
    fn from(token: Token<'a>) -> SourceSpan {
        token.span.into()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BinOp<'a>(Cow<'a, str>);

impl<'a> BinOp<'a> {
    pub fn new(s: impl Into<Cow<'a, str>>) -> Self {
        let s: Cow<_> = s.into();
        debug_assert!(s.chars().all(is_bin_op_char));

        Self(s)
    }

    pub fn is_separator(&self) -> bool {
        self.0.len() >= 4 && self.0.chars().all(|c| c == '-')
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn into_str(self) -> Cow<'a, str> {
        self.0
    }
}

impl CloneStatic<BinOp<'static>> for BinOp<'_> {
    fn clone_static(&self) -> BinOp<'static> {
        BinOp(self.0.clone_static())
    }
}

impl TryFrom<Special> for BinOp<'static> {
    type Error = ();

    fn try_from(s: Special) -> Result<Self, Self::Error> {
        let s = s.as_str();

        if s.chars().all(is_bin_op_char) {
            Ok(BinOp::new(s))
        } else {
            Err(())
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TokenType<'a> {
    Int,
    Float,
    String,
    Ident,
    Keyword,
    Symbol,
    BlockParam,
    BinOp(BinOp<'a>),
    Special(Special),
    Eof,
}

impl Display for TokenType<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Int => write!(f, "integer"),
            Self::Float => write!(f, "float"),
            Self::String => write!(f, "string"),
            Self::Ident => write!(f, "identifier"),
            Self::Keyword => write!(f, "keyword"),
            Self::Symbol => write!(f, "symbol"),
            Self::BlockParam => write!(f, "block parameter"),
            Self::BinOp(op) if f.alternate() => write!(f, "`{}`", op.as_str()),
            Self::BinOp(op) => write!(f, "{}", op.as_str()),
            Self::Special(s) if f.alternate() => write!(f, "`{}`", s.as_str()),
            Self::Special(s) => write!(f, "{}", s.as_str()),
            Self::Eof => write!(f, "end of file"),
        }
    }
}

impl CloneStatic<TokenType<'static>> for TokenType<'_> {
    fn clone_static(&self) -> TokenType<'static> {
        match self {
            Self::Int => TokenType::Int,
            Self::Float => TokenType::Float,
            Self::String => TokenType::String,
            Self::Ident => TokenType::Ident,
            Self::Keyword => TokenType::Keyword,
            Self::Symbol => TokenType::Symbol,
            Self::BlockParam => TokenType::BlockParam,
            Self::BinOp(op) => TokenType::BinOp(op.clone_static()),
            Self::Special(s) => TokenType::Special(*s),
            Self::Eof => TokenType::Eof,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Symbol<'a> {
    String(Cow<'a, str>),
    UnarySelector(Cow<'a, str>),
    BinarySelector(BinOp<'a>),
    KeywordSelector(Vec<Keyword<'a>>),
}

impl CloneStatic<Symbol<'static>> for Symbol<'_> {
    fn clone_static(&self) -> Symbol<'static> {
        match self {
            Self::String(s) => Symbol::String(s.clone_static()),
            Self::UnarySelector(id) => Symbol::UnarySelector(id.clone_static()),
            Self::BinarySelector(op) => Symbol::BinarySelector(op.clone_static()),
            Self::KeywordSelector(kws) => Symbol::KeywordSelector(kws.clone_static()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Keyword<'a> {
    pub span: Span,
    pub kw: Cow<'a, str>,
}

impl CloneStatic<Keyword<'static>> for Keyword<'_> {
    fn clone_static(&self) -> Keyword<'static> {
        Keyword {
            span: self.span,
            kw: self.kw.clone_static(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TokenValue<'a> {
    // the lexer treats integer literals as unsigned, but it'll get reduced to i64 during parsing
    Int(u64),
    Float(f64),
    String(Cow<'a, str>),
    Ident(Cow<'a, str>),
    Keyword(Cow<'a, str>),
    Symbol(Symbol<'a>),
    BlockParam(Cow<'a, str>),
    BinOp(BinOp<'a>),
    Special(Special),
    Eof,
}

impl<'a> TokenValue<'a> {
    pub fn ty(&self) -> TokenType<'a> {
        match *self {
            Self::Int(_) => TokenType::Int,
            Self::Float(_) => TokenType::Float,
            Self::String(_) => TokenType::String,
            Self::Ident(_) => TokenType::Ident,
            Self::Keyword(_) => TokenType::Keyword,
            Self::Symbol(_) => TokenType::Symbol,
            Self::BlockParam(_) => TokenType::BlockParam,
            Self::BinOp(ref op) => TokenType::BinOp(op.clone()),
            Self::Special(s) => TokenType::Special(s),
            Self::Eof => TokenType::Eof,
        }
    }

    pub fn as_bin_op(&self) -> Option<Cow<'_, BinOp<'a>>> {
        Some(match *self {
            Self::BinOp(ref op) => Cow::Borrowed(op),
            Self::Special(s) => Cow::Owned(BinOp::try_from(s).ok()?),
            _ => return None,
        })
    }
}

impl CloneStatic<TokenValue<'static>> for TokenValue<'_> {
    fn clone_static(&self) -> TokenValue<'static> {
        match self {
            Self::Int(i) => TokenValue::Int(*i),
            Self::Float(f) => TokenValue::Float(*f),
            Self::String(s) => TokenValue::String(s.clone_static()),
            Self::Ident(id) => TokenValue::Ident(id.clone_static()),
            Self::Keyword(kw) => TokenValue::Keyword(kw.clone_static()),
            Self::Symbol(s) => TokenValue::Symbol(s.clone_static()),
            Self::BlockParam(p) => TokenValue::BlockParam(p.clone_static()),
            Self::BinOp(op) => TokenValue::BinOp(op.clone_static()),
            Self::Special(s) => TokenValue::Special(*s),
            Self::Eof => TokenValue::Eof,
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
    "^" => Circumflex,
    ":=" => Assign,

    "primitive" => Primitive,
    "#(" => ArrayLeft,
}
