mod common;

use std::collections::HashMap;
use std::error::Error;
use std::fmt::Write;
use std::fs;
use std::path::{Path, PathBuf};

use common::SharedStringBuf;
use serde::Deserialize;
use test_generator::test_resources;

use stitch::file::FileLoader;
use stitch::sourcemap::{SourceFile, SourceMap};
use stitch::vm::gc::GarbageCollector;
use stitch::vm::Vm;

use self::common::Matchers;

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "kebab-case")]
struct VmTest {
    #[serde(default)]
    fail: bool,

    #[serde(default)]
    output: Matchers,

    #[serde(default = "VmTest::preload_classes_default")]
    preload_classes: bool,

    #[serde(default)]
    args: Vec<String>,
}

impl VmTest {
    fn preload_classes_default() -> bool {
        true
    }
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
        let source_name = path.to_string_lossy().into_owned();
        let source = self.source_map.add_source(source_name.clone(), contents);
        self.loaded_user_classes
            .insert(class_name.to_owned(), source_name);

        Ok(source)
    }

    fn load_file(&mut self, path: &Path) -> Result<String, Box<dyn Error + Send + Sync>> {
        let contents = fs::read_to_string(&path)
            .map_err(|e| format!("read_to_string({}) failed: {e}", path.display()))?;

        Ok(contents)
    }
}

fn run_test_class(test_class_path: PathBuf) {
    miette::set_panic_hook();

    let class_name = test_class_path.file_stem().unwrap().to_str().unwrap();
    let base_path = test_class_path.parent().unwrap().to_owned();

    let gc = GarbageCollector::default();
    let mut file_loader = TestFileLoader::new(base_path.clone());
    let test: VmTest =
        common::parse_comment_header(file_loader.load_user_class(class_name).unwrap());

    let output = SharedStringBuf::default();
    let stdout = Box::new(output.clone());
    let stderr = Box::new(output.clone());

    let mut vm = Vm::new(
        &gc,
        Box::new(file_loader),
        stdout,
        stderr,
        Default::default(),
    );

    if test.preload_classes {
        let mut entries = fs::read_dir(&base_path)
            .expect("could not open the test directory")
            .map(|entry| entry.map(|entry| entry.path()))
            .filter(|path| match path {
                Ok(path) => {
                    path.extension().is_some_and(|ext| ext == "som")
                        && path.file_stem().is_some_and(|name| name != class_name)
                        && fs::metadata(&path)
                            .expect("could not read metadata")
                            .is_file()
                }

                Err(_) => true,
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        entries.sort_unstable_by_key(|path| {
            path.file_stem().map(|class_name| class_name.to_os_string())
        });

        for path in entries {
            vm.parse_and_load_user_class(&path.file_stem().unwrap().to_string_lossy())
                .map_err(|e| {
                    miette::Report::new(*e)
                        .wrap_err(format!("could not load {}", path.display()))
                        .with_source_code(vm.file_loader.get_source().clone())
                })
                .unwrap();
        }
    }

    let test_class = vm
        .parse_and_load_user_class(class_name)
        .map_err(|e| {
            miette::Report::new(*e)
                .wrap_err(format!("loading class `{class_name}` failed"))
                .with_source_code(vm.file_loader.get_source().clone())
        })
        .unwrap();

    let args = test
        .args
        .into_iter()
        .map(|arg| vm.make_string(arg).into_value())
        .collect();

    let result = vm
        .run(test_class, args)
        .map_err(|e| miette::Report::new(*e).with_source_code(vm.file_loader.get_source().clone()));

    let mut output = output.inner().take();

    match result {
        Ok(_) if test.fail => panic!("Expected a VM error but completed successfully"),
        Err(e) if !test.fail => panic!("Test failed (expected success):\n{e:?}"),

        ref result => {
            if let Err(e) = result {
                write!(&mut output, "{e}").unwrap();
            }

            if !test.output.is_empty() {
                if let Err(matcher_msg) = test.output.check(&output) {
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
