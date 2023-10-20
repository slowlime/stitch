mod common;

use std::cell::RefCell;
use std::collections::HashMap;
use std::error::Error;
use std::fmt::{self, Write};
use std::fs;
use std::path::PathBuf;
use std::rc::Rc;

use common::Matchers;
use serde::Deserialize;

use stitch::file::FileLoader;
use stitch::sourcemap::{SourceFile, SourceMap};
use stitch::vm::gc::GarbageCollector;
use stitch::vm::Vm;
use test_generator::test_resources;

#[derive(Default)]
struct VmOutput(Rc<RefCell<String>>);

impl Write for VmOutput {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.0.borrow_mut().write_str(s)
    }

    fn write_char(&mut self, c: char) -> fmt::Result {
        self.0.borrow_mut().write_char(c)
    }

    fn write_fmt(&mut self, args: fmt::Arguments<'_>) -> fmt::Result {
        self.0.borrow_mut().write_fmt(args)
    }
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "camelCase")]
struct VmTest {
    #[serde(default)]
    fail: bool,

    #[serde(default)]
    output: Matchers,
}

struct TestFileLoader {
    base_path: PathBuf,
    source_map: SourceMap,
    loaded_builtin_classes: HashMap<String, String>,
    loaded_user_classes: HashMap<String, String>,
}

impl TestFileLoader {
    fn new(base_path: PathBuf) -> Self {
        Self {
            base_path,
            source_map: SourceMap::new(),
            loaded_builtin_classes: Default::default(),
            loaded_user_classes: Default::default(),
        }
    }
}

impl FileLoader for TestFileLoader {
    fn get_source(&self) -> &SourceMap {
        &self.source_map
    }

    fn load_builtin_class(
        &mut self,
        class_name: &str,
    ) -> Result<&SourceFile, Box<dyn Error + Send + Sync>> {
        if let Some(source_name) = self.loaded_builtin_classes.get(class_name) {
            return Ok(self.source_map.get_by_name(source_name).unwrap());
        }

        let mut path = PathBuf::from(file!());
        path.pop();
        path.push("../third-party/SOM/Smalltalk/");
        path.push(format!("{class_name}.som"));

        let contents = fs::read_to_string(&path)
            .map_err(|e| format!("read_to_string({}) failed: {e}", path.display()))?;
        let source_name = path.into_os_string().to_string_lossy().into_owned();
        let source = self.source_map.add_source(source_name.clone(), contents);
        self.loaded_builtin_classes
            .insert(class_name.to_owned(), source_name);

        Ok(source)
    }

    fn load_user_class(
        &mut self,
        class_name: &str,
    ) -> Result<&SourceFile, Box<dyn Error + Send + Sync>> {
        if let Some(source_name) = self.loaded_user_classes.get(class_name) {
            return Ok(self.source_map.get_by_name(source_name).unwrap());
        }

        let path = self.base_path.join(format!("{class_name}.som"));
        let contents = fs::read_to_string(&path)
            .map_err(|e| format!("read_to_string({}) failed: {e}", path.display()))?;
        let source_name = path.into_os_string().to_string_lossy().into_owned();
        let source = self.source_map.add_source(source_name.clone(), contents);
        self.loaded_user_classes
            .insert(class_name.to_owned(), source_name);

        Ok(source)
    }
}

fn run_test_class(base_path: PathBuf) {
    miette::set_panic_hook();

    let class_name = base_path.file_stem().unwrap().to_str().unwrap();

    let gc = GarbageCollector::new();
    let mut file_loader = TestFileLoader::new(base_path.parent().unwrap().to_owned());
    let test: VmTest =
        common::parse_comment_header(file_loader.load_user_class(class_name).unwrap());

    let output = Rc::new(RefCell::default());
    let stdout = Box::new(VmOutput(output.clone()));
    let stderr = Box::new(VmOutput(output.clone()));

    let mut vm = Vm::new(&gc, Box::new(file_loader), stdout, stderr);
    let test_class = vm
        .parse_and_load_user_class(class_name)
        .expect("loading test class failed");

    let result = vm
        .run(test_class)
        .map_err(|e| miette::Report::new(e).with_source_code(vm.file_loader.get_source().clone()));
    let mut output = output.take();

    match result {
        Ok(_) if test.fail => panic!("Expected a VM error but completed successfully"),
        Err(e) if !test.fail => panic!("Test failed (expected success):\n{e:?}"),

        ref result => {
            if let Err(e) = result {
                write!(&mut output, "{e}").unwrap();
            }

            if !test.output.is_empty() {
                if let Err(matcher_msg) = test.output.check(&output) {
                    eprintln!("Test failed with unexpected message: {matcher_msg}.");
                    eprintln!("Test output:\n{output}");

                    if let Err(e) = result {
                        eprintln!("\n---\nVM has terminated with an error:\n{e:?}");
                    }

                    panic!("test failed with unexpected message: {matcher_msg}.");
                }
            }
        }
    }
}

#[test_resources("tests/vm/**/Test.som")]
fn test_vm(source_path: PathBuf) {
    run_test_class(source_path);
}
