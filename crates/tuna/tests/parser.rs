mod common;

use std::fs;
use std::path::PathBuf;

use rstest::rstest;
use serde::Deserialize;

use stitch::parse::{BigNumberBehavior, ParserOptions};
use stitch::sourcemap::SourceMap;

use self::common::Matchers;

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "kebab-case")]
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

    let mut source_map = SourceMap::new();
    let source = fs::read_to_string(&source_path).expect("Failed to read the test source");
    let source = source_map.add_source(source_path.to_string_lossy().into_owned(), source);

    let mut test = cfg.unwrap_or_else(|| common::parse_comment_header(source));
    test.fail = test.fail || !test.fail_message.is_empty();

    let result = match stitch::parse::parse(&source, test.parser_options) {
        Ok(class) => Ok(class),

        Err(e) => Err(miette::Report::new(e).with_source_code(source_map)),
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

#[rstest]
fn test_parser_stitch(#[files("tests/parser/**/*.som")] source_path: PathBuf) {
    test_parser(source_path, None);
}

#[rstest]
fn test_parser_som_st(
    #[files("third-party/SOM/TestSuite/**/*.som")]
    #[files("third-part/SOM/Smalltalk/*.som")]
    source_path: PathBuf,
) {
    const FAILING_TEST_NAMES: &[&str] = &[
        // assigns to `self` and `super`, neither of which is supported
        "Self",
    ];

    let fail = FAILING_TEST_NAMES.contains(&&*source_path.file_stem().unwrap().to_string_lossy());

    let cfg = ParserTest {
        fail,
        parser_options: ParserOptions {
            big_numbers: BigNumberBehavior::Saturate,
        },
        ..Default::default()
    };

    test_parser(source_path, Some(cfg))
}
