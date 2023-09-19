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

fn test_parser(source_path: PathBuf, cfg: Option<ParserTest>) {
    miette::set_panic_hook();

    let source = fs::read_to_string(&source_path).expect("Failed to read the test source");

    let mut test = cfg.unwrap_or_else(|| common::parse_comment_header(&source));
    test.fail = test.fail || test.fail_message.is_some();

    let result = match stitch::parse::parse(&source) {
        Ok(class) => Ok(class),
        Err(e) => Err(
            miette::Report::new(e).with_source_code(miette::NamedSource::new(
                source_path.to_string_lossy(),
                source.clone(),
            )),
        ),
    };

    match result {
        Ok(_) if test.fail => panic!("Expected parsing failure but parsed successfully"),
        Err(e) if !test.fail => panic!("Parsing failed:\n{e:?}"),

        Ok(_class) => {}

        Err(e) => {
            if let Some(fail_message) = test.fail_message.as_ref() {
                let msg = e.to_string();

                assert!(
                    msg.contains(fail_message),
                    "Parsing failed with message {msg} (expected the message to contain {fail_message}). Details:\n{e:?}"
                );
            }
        }
    }
}

#[test_resources("tests/parser/*.som")]
fn test_parser_stitch(source_path: PathBuf) {
    test_parser(source_path, None);
}

#[test_resources("third-party/SOM/TestSuite/**/*.som")]
fn test_parser_som_st(source_path: PathBuf) {
    test_parser(source_path, Some(Default::default()))
}
