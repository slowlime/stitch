mod common;

use std::fs;
use std::path::PathBuf;

use serde::Deserialize;
use test_generator::test_resources;

#[derive(Deserialize, Default)]
struct ParserTest {
    #[serde(default)]
    fail: bool,

    #[serde(default)]
    fail_message: Option<String>,
}

#[test_resources("tests/parser/*.som")]
fn test_parser(source_path: PathBuf) {
    let source = fs::read_to_string(&source_path).expect("Failed to read the test source");
    let mut test: ParserTest = common::parse_comment_header(&source);
    test.fail = test.fail || test.fail_message.is_some();

    let result = stitch::parse::parse(&source);

    match result {
        Ok(_) if test.fail => panic!("Expected parsing failure but parsed successfully"),
        Err(e) if !test.fail => panic!("Parsing failed: {e}"),

        Ok(_class) => {},

        Err(e) => {
            if let Some(fail_message) = test.fail_message.as_ref() {
                let msg = e.to_string();

                assert!(
                    msg.contains(fail_message),
                    "Parsing failed with message {msg} (expected the message to contain {fail_message})"
                );
            }
        },
    }
}
