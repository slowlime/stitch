use std::error::Error;
use std::process::exit;
use std::{env, fs};

use stitch::post::PostProc;
use stitch::spec::Specializer;
use stitch::{encode, parse};

fn main() -> Result<(), Box<dyn Error>> {
    pretty_env_logger::init();

    let mut args = env::args_os().skip(1);

    if args.len() != 2 {
        eprintln!("Usage: stitch <input wasm module> <output module>");
        exit(2);
    }

    let input_path = args.next().unwrap();
    let output_path = args.next().unwrap();

    let module_bytes = fs::read(&input_path)?;
    let mut module = parse::parse(&module_bytes)?;
    Specializer::new(&mut module).process();
    PostProc::new(&mut module).process();
    let module = encode::encode(&mut module);

    fs::write(&output_path, &module)?;

    Ok(())
}
