use miette::SourceOffset;

#[derive(Debug, Clone)]
pub struct Cursor<'buf> {
    iter: std::str::Chars<'buf>,
    pos: SourceOffset,
    prev_pos: Option<SourceOffset>,
    eof: bool,
}

impl<'buf> Cursor<'buf> {
    pub fn new(buf: &'buf str) -> Self {
        Self {
            iter: buf.chars(),
            pos: 0.into(),
            prev_pos: None,
            eof: false,
        }
    }

    /// Returns the position of the immediately following character.
    pub fn pos(&self) -> SourceOffset {
        self.pos
    }

    /// Returns the position of the previously returned character.
    pub fn prev_pos(&self) -> SourceOffset {
        self.prev_pos.unwrap_or(0.into())
    }

    pub fn peek(&self) -> Option<char> {
        self.iter.clone().next()
    }

    pub fn remaining(&self) -> &'buf str {
        self.iter.as_str()
    }

    pub fn starts_with(&self, value: &str) -> bool {
        self.remaining().starts_with(value)
    }

    pub fn consume_expecting(&mut self, expected: &str) -> Option<&'buf str> {
        self.starts_with(expected)
            .then(|| self.consume_n(expected.len()))
    }

    pub fn consume_n(&mut self, n: usize) -> &'buf str {
        let remaining = self.remaining();
        let start = self.pos.offset();

        for _ in 0..n {
            self.next();
        }

        let end = self.pos.offset();

        &remaining[0..(end - start)]
    }

    pub fn consume_while(&mut self, mut predicate: impl FnMut(char) -> bool) -> &'buf str {
        self.consume_n(self.iter.clone().take_while(|&c| predicate(c)).count())
    }
}

impl<'buf> Iterator for Cursor<'buf> {
    type Item = char;

    fn next(&mut self) -> Option<Self::Item> {
        if self.eof {
            return None;
        }

        self.prev_pos = Some(self.pos);

        let c = match self.iter.next() {
            Some(c) => c,
            None => {
                self.eof = true;

                return None;
            }
        };

        self.pos = (self.pos.offset() + c.len_utf8()).into();

        Some(c)
    }
}
