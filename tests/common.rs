use regex::Regex;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer};

use stitch::parse::Cursor;

pub fn parse_comment_header<T: DeserializeOwned + Default>(source: &str) -> T {
    let mut cursor = Cursor::new(source);

    if cursor.consume_expecting("\"test:").is_none() {
        return Default::default();
    }

    cursor
        .consume_newline()
        .expect("first line must only contain \"test:");

    let header = cursor.consume_while(|c| c != '"');
    cursor
        .consume_expecting("\"")
        .expect("header must be terminated by a quote");
    assert!(
        header.ends_with("\r\n") || header.ends_with('\n'),
        "terminating quote must be placed on its own line"
    );

    match serde_yaml::from_str(&header) {
        Ok(result) => result,
        Err(e) => panic!("Header is invalid: {e}"),
    }
}

fn deserialize_regex<'de, D>(de: D) -> Result<Regex, D::Error>
where
    D: Deserializer<'de>,
{
    Regex::new(Deserialize::deserialize(de)?).map_err(|e| <D::Error as serde::de::Error>::custom(e))
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub enum Matcher {
    Contains(String),

    #[serde(deserialize_with = "deserialize_regex")]
    Regex(Regex),
}

impl Matcher {
    pub fn check(&self, s: &str) -> Result<(), String> {
        match self {
            Self::Contains(substr) if s.contains(substr) => Ok(()),
            Self::Contains(substr) => Err(format!("expected to contain `{substr}`")),

            Self::Regex(r) if r.is_match(s) => Ok(()),
            Self::Regex(r) => Err(format!("expected to match regexp `{r}`")),
        }
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
pub enum Matchers {
    One(Matcher),
    Many(Vec<Matcher>),
}

impl Matchers {
    pub fn is_empty(&self) -> bool {
        match self {
            Self::One(_) => false,
            Self::Many(v) => v.is_empty(),
        }
    }

    pub fn as_slice(&self) -> &[Matcher] {
        match self {
            Self::One(matcher) => std::slice::from_ref(matcher),
            Self::Many(matchers) => matchers,
        }
    }

    pub fn check(&self, s: &str) -> Result<(), String> {
        self.as_slice()
            .iter()
            .map(|matcher| matcher.check(s))
            .collect()
    }
}

impl Default for Matchers {
    fn default() -> Self {
        Self::Many(vec![])
    }
}
