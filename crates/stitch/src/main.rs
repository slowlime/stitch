use std::process::exit;
use std::{env, fs};

use anyhow::Result;
use stitch::post::PostProc;
use stitch::{encode, parse};

fn main() -> Result<()> {
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

    // TODO: Specializer::new(&mut module).process();
    PostProc::new(&mut module).process();

    let module = encode::encode(&mut module);

    if cfg!(debug_assertions) {
        parse::parse(&module).unwrap();
    }

    fs::write(&output_path, &module)?;

    Ok(())
}
