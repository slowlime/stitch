use serde::de::DeserializeOwned;

use stitch::parse::Cursor;

pub fn parse_comment_header<T: DeserializeOwned + Default>(source: &str) -> T {
    let mut cursor = Cursor::new(source);

    if cursor.consume_expecting("\"").is_none() {
        return Default::default();
    }

    cursor.consume_newline().expect("first line must only contain a quote");

    let header = cursor.consume_while(|c| c != '"');
    cursor.consume_expecting("\"").expect("header must be terminated by a quote");
    assert!(header.ends_with("\r\n") || header.ends_with('\n'), "terminating quote must be placed on its own line");

    serde_yaml::from_str(&header).expect("header is invalid")
}
