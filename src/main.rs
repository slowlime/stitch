mod args;

use std::io;
use std::iter;
use std::process::ExitCode;

use clap::Parser;

use stitch::file::PathFileLoader;
use stitch::parse::ParserOptions;
use stitch::util::FormattedWriter;
use stitch::vm::error::VmError;
use stitch::vm::gc::GarbageCollector;
use stitch::vm::Vm;
use stitch::vm::VmOptions;

use crate::args::Args;

fn main() -> ExitCode {
    let args = Args::parse();

    let vm_options = VmOptions {
        print_warnings: args.print_warnings,
        debug: args.verbose,
        load_class_options: Default::default(),
        parser_options: ParserOptions {
            big_numbers: args.big_numbers.into(),
        },
    };

    let gc = GarbageCollector::new();
    let file_loader = PathFileLoader::new(args.class_search_paths);
    let stdout = Box::new(FormattedWriter(io::stdout()));
    let stderr = Box::new(FormattedWriter(io::stderr()));
    let mut vm = Vm::new(&gc, Box::new(file_loader), stdout, stderr, vm_options);

    let run_args = iter::once(args.main_class_name.clone())
        .chain(args.args.into_iter())
        .map(|arg| vm.make_string(arg).into_value())
        .collect();
    let result = vm
        .parse_and_load_user_class(&args.main_class_name)
        .and_then(|class| vm.run(class, run_args));

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
