use std::collections::HashMap;
use std::error::Error;
use std::path::PathBuf;
use std::{fs, io};

use thiserror::Error;

use crate::sourcemap::{SourceFile, SourceMap};

pub trait FileLoader {
    fn get_source(&self) -> &SourceMap;
    fn load_builtin_class(&mut self, class_name: &str) -> Result<&SourceFile, Box<dyn Error + Send + Sync>>;
    fn load_user_class(&mut self, class_name: &str) -> Result<&SourceFile, Box<dyn Error + Send + Sync>>;
}

#[derive(Error, Debug)]
pub enum PathFileLoaderError {
    #[error("illegal class name: {0}")]
    IllegalClassName(String),

    #[error("could not load file {}", path.display())]
    LoadFailed { path: PathBuf, source: io::Error },

    #[error("class not found: {0}")]
    ClassNotFound(String),
}

#[derive(Debug, Clone)]
pub struct PathFileLoader {
    source_map: SourceMap,
    search_paths: Vec<PathBuf>,

    // maps a class name to its source name
    loaded_classes: HashMap<String, String>,
}

impl PathFileLoader {
    pub fn new(search_paths: Vec<PathBuf>) -> Self {
        Self {
            source_map: SourceMap::new(),
            search_paths,
            loaded_classes: Default::default(),
        }
    }

    pub fn load_class(&mut self, class_name: &str) -> Result<&SourceFile, PathFileLoaderError> {
        use PathFileLoaderError::*;

        if let Some(source_name) = self.loaded_classes.get(class_name) {
            return Ok(self.source_map.get_by_name(source_name).unwrap());
        }

        if class_name.contains(|c: char| !c.is_ascii_alphanumeric() && c != '_') {
            return Err(IllegalClassName(class_name.to_owned()));
        }

        for search_path in &self.search_paths {
            let mut path = search_path.clone();
            path.push(format!("{}.som", class_name));

            let contents = match fs::read_to_string(&path) {
                Ok(contents) => contents,
                Err(e) if e.kind() == io::ErrorKind::NotFound => continue,
                Err(e) => return Err(LoadFailed { path, source: e }),
            };

            let source_name = path.into_os_string().to_string_lossy().into_owned();
            let source = self.source_map.add_source(source_name.clone(), contents);
            self.loaded_classes
                .insert(class_name.to_owned(), source_name);

            return Ok(source);
        }

        Err(ClassNotFound(class_name.to_owned()))
    }
}

impl FileLoader for PathFileLoader {
    fn get_source(&self) -> &SourceMap {
        &self.source_map
    }

    fn load_builtin_class(&mut self, class_name: &str) -> Result<&SourceFile, Box<dyn Error + Send + Sync>> {
        self.load_class(class_name).map_err(Into::into)
    }

    fn load_user_class(&mut self, class_name: &str) -> Result<&SourceFile, Box<dyn Error + Send + Sync>> {
        self.load_class(class_name).map_err(Into::into)
    }
}
