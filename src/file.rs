use std::io;

use crate::sourcemap::{SourceMap, SourceFile};

pub trait FileLoader {
    fn get_source(&self) -> &SourceMap;
    fn load_builtin_class(&self, class_name: &str) -> Result<&SourceFile, io::Error>;
    fn load_user_class(&self, class_name: &str) -> Result<&SourceFile, io::Error>;
}
