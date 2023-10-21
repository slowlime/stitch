use std::collections::HashMap;

use miette::{MietteError, MietteSpanContents, SourceCode, SourceSpan};

#[derive(Debug, Clone)]
pub struct SourceFile {
    name: String,
    contents: String,
    offset: usize,
}

impl SourceFile {
    pub fn base_offset(&self) -> usize {
        self.offset
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn contents(&self) -> &str {
        &self.contents
    }

    fn next_offset(&self) -> usize {
        // the plus one makes sure empty files have disjoint ranges
        self.offset + self.contents.len() + 1
    }
}

#[derive(Debug, Clone, Default)]
pub struct SourceMap {
    files: Vec<SourceFile>,
    file_names: HashMap<String, usize>,
}

impl SourceMap {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn add_source(&mut self, name: String, contents: String) -> &SourceFile {
        let offset = self.files.last().map(SourceFile::next_offset).unwrap_or(0);
        let file = SourceFile {
            name: name.clone(),
            contents,
            offset,
        };

        self.file_names.insert(name, self.files.len());
        self.files.push(file);

        self.files.last().unwrap()
    }

    pub fn get_by_name(&self, name: impl AsRef<str>) -> Option<&SourceFile> {
        self.file_names
            .get(name.as_ref())
            .map(|&idx| &self.files[idx])
    }
}

impl SourceCode for SourceMap {
    fn read_span<'a>(
        &'a self,
        span: &SourceSpan,
        context_lines_before: usize,
        context_lines_after: usize,
    ) -> Result<Box<dyn miette::SpanContents<'a> + 'a>, MietteError> {
        if self.files.is_empty() {
            return Err(MietteError::OutOfBounds);
        }

        let file_idx = match self
            .files
            .binary_search_by_key(&span.offset(), SourceFile::base_offset)
        {
            Ok(idx) => idx,
            Err(idx) => idx.saturating_sub(1),
        };

        let file = &self.files[file_idx];
        debug_assert!(file.offset <= span.offset());
        let offset = span.offset() - file.offset;
        let len = span.len();

        let span = SourceSpan::new(offset.into(), len.into());
        let contents = file
            .contents
            .read_span(&span, context_lines_before, context_lines_after)?;

        Ok(Box::new(MietteSpanContents::new_named(
            file.name.clone(),
            contents.data(),
            SourceSpan::new(
                (contents.span().offset() + file.offset).into(),
                contents.span().len().into(),
            ),
            contents.line(),
            contents.column(),
            contents.line_count(),
        )))
    }
}
