mod common;

use std::fs;
use std::path::PathBuf;

use serde::Deserialize;
use stitch::parse::{BigNumberBehavior, ParserOptions};
use test_generator::test_resources;

use common::Matchers;

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "camelCase")]
struct ParserTest {
    #[serde(default)]
    fail: bool,

    #[serde(default)]
    fail_message: Matchers,

    #[serde(skip)]
    parser_options: ParserOptions,
}

fn test_parser(source_path: PathBuf, cfg: Option<ParserTest>) {
    miette::set_panic_hook();

    let source = fs::read_to_string(&source_path).expect("Failed to read the test source");

    let mut test = cfg.unwrap_or_else(|| common::parse_comment_header(&source));
    test.fail = test.fail || !test.fail_message.is_empty();

    let result = match stitch::parse::parse(&source, test.parser_options) {
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
        Err(e) if !test.fail => panic!("Parsing failed (expected success):\n{e:?}"),

        Ok(_class) => {}

        Err(e) => {
            if !test.fail_message.is_empty() {
                if let Err(matcher_msg) = test.fail_message.check(&e.to_string()) {
                    panic!(
                        "Parsing failed with unexpected message: {matcher_msg}. Details:\n{e:?}"
                    );
                }
            }
        }
    }
}

#[test_resources("tests/parser/**/*.som")]
fn test_parser_stitch(source_path: PathBuf) {
    test_parser(source_path, None);
}

#[test_resources("third-party/SOM/TestSuite/**/*.som")]
fn test_parser_som_st(source_path: PathBuf) {
    let cfg = ParserTest {
        parser_options: ParserOptions {
            big_numbers: BigNumberBehavior::Saturate,
        },
        ..Default::default()
    };

    test_parser(source_path, Some(cfg))
}
