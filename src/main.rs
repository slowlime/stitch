mod args;

use std::fmt;
use std::io;

use clap::Parser;

use stitch::file::PathFileLoader;
use stitch::vm::error::VmError;
use stitch::vm::gc::GarbageCollector;
use stitch::vm::Vm;

use crate::args::Args;

struct FormattedWriter<W>(W);

impl<W: io::Write> fmt::Write for FormattedWriter<W> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        write!(&mut self.0, "{}", s).map_err(|_| fmt::Error)
    }

    fn write_char(&mut self, c: char) -> fmt::Result {
        write!(&mut self.0, "{}", c).map_err(|_| fmt::Error)
    }

    fn write_fmt(&mut self, args: fmt::Arguments<'_>) -> fmt::Result {
        write!(&mut self.0, "{}", args).map_err(|_| fmt::Error)
    }
}

fn main() -> Result<(), VmError> {
    let args = Args::parse();

    let gc = GarbageCollector::new();
    let file_loader = PathFileLoader::new(args.class_search_paths);
    let stdout = Box::new(FormattedWriter(io::stdout()));
    let stderr = Box::new(FormattedWriter(io::stderr()));
    let mut vm = Vm::new(&gc, Box::new(file_loader), stdout, stderr);
    let main_class = vm.parse_and_load_user_class(&args.main_class_name)?;

    vm.run(main_class).map(|_| ())
}
