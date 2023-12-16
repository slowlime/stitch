use std::io::{self, Write};

use crate::ir::Module;

pub struct Encoder<'a, W> {
    module: &'a Module,
    writer: W,
}

impl<'a, W: Write> Encoder<'a, W> {
    fn new(module: &'a Module, writer: W) -> Self {
        Self { module, writer }
    }

    fn encode(mut self) -> io::Result<()> {
        self.encode_types()?;
        self.encode_imports()?;
        self.encode_funcs()?;
        self.encode_tables()?;
        self.encode_memories()?;
        self.encode_globals()?;
        self.encode_exports()?;
        self.encode_start()?;
        self.encode_elements()?;
        self.encode_code()?;
        self.encode_data()?;

        Ok(())
    }

    fn encode_types(&mut self) -> io::Result<()> {
        todo!()
    }

    fn encode_imports(&mut self) -> io::Result<()> {
        todo!()
    }

    fn encode_funcs(&mut self) -> io::Result<()> {
        todo!()
    }

    fn encode_tables(&mut self) -> io::Result<()> {
        todo!()
    }

    fn encode_memories(&mut self) -> io::Result<()> {
        todo!()
    }

    fn encode_globals(&mut self) -> io::Result<()> {
        todo!()
    }

    fn encode_exports(&mut self) -> io::Result<()> {
        todo!()
    }

    fn encode_start(&mut self) -> io::Result<()> {
        todo!()
    }

    fn encode_elements(&mut self) -> io::Result<()> {
        todo!()
    }

    fn encode_code(&mut self) -> io::Result<()> {
        todo!()
    }

    fn encode_data(&mut self) -> io::Result<()> {
        todo!()
    }
}
