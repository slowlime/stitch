mod args;

use std::fmt;
use std::io;
use std::process::ExitCode;

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

fn main() -> ExitCode {
    let args = Args::parse();

    let gc = GarbageCollector::new();
    let file_loader = PathFileLoader::new(args.class_search_paths);
    let stdout = Box::new(FormattedWriter(io::stdout()));
    let stderr = Box::new(FormattedWriter(io::stderr()));
    let mut vm = Vm::new(&gc, Box::new(file_loader), stdout, stderr);
    let result = vm
        .parse_and_load_user_class(&args.main_class_name)
        .and_then(|class| vm.run(class));

    match result {
        Ok(_) | Err(VmError::Exited { code: 0, .. }) => ExitCode::SUCCESS,

        Err(e) => {
            let code = if let VmError::Exited { code, .. } = e {
                code
            } else {
                1
            };

            let e = miette::Report::new(e)
                .wrap_err("VM has terminated with an error")
                .with_source_code(vm.file_loader.get_source().clone());
            eprintln!("{:?}", e);

            ExitCode::from(code as u8)
        }
    }
}
