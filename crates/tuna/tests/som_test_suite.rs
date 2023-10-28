use std::io;

use stitch::file::PathFileLoader;
use stitch::parse::{ParserOptions, BigNumberBehavior};
use stitch::util::FormattedWriter;
use stitch::vm::gc::GarbageCollector;
use stitch::vm::{Vm, VmOptions};

mod common;

#[test]
fn run_som_test_suite() {
    miette::set_panic_hook();

    let gc = GarbageCollector::default();
    let file_loader = PathFileLoader::new(vec![
        "third-party/SOM/Smalltalk".into(),
        "third-party/SOM/TestSuite".into(),
    ]);

    let stdout = Box::new(FormattedWriter(io::stdout()));
    let stderr = Box::new(FormattedWriter(io::stderr()));

    let mut vm = Vm::new(&gc, Box::new(file_loader), stdout, stderr, VmOptions {
        parser_options: ParserOptions {
            big_numbers: BigNumberBehavior::Saturate,
        },
        ..Default::default()
    });

    let test_harness = vm.parse_and_load_user_class("TestHarness")
        .map_err(|e| miette::Report::new(*e).with_source_code(vm.file_loader.get_source().clone()))
        .unwrap();

    if let Err(e) = vm.run(test_harness, vec![vm.make_string("TestHarness".to_owned()).into_value()]) {
        let e = miette::Report::new(*e).with_source_code(vm.file_loader.get_source().clone())
            .wrap_err("VM terminated abnormally");
        panic!("{e:?}");
    };
}
