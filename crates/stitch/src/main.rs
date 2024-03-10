use std::process::exit;
use std::{env, fs};

use anyhow::{Context, Result};
use stitch::interp::Interpreter;
use stitch::post::PostProc;
use stitch::{encode, parse};

fn main() -> Result<()> {
    pretty_env_logger::init();

    let mut args = env::args_os().skip(1);

    if args.len() < 2 {
        eprintln!("Usage: stitch <input wasm module> <output module> [<arg>...]");
        exit(2);
    }

    let input_path = args.next().unwrap();
    let output_path = args.next().unwrap();
    let args = args.map(|arg| arg.into_encoded_bytes()).collect();

    let module_bytes = fs::read(&input_path).context("could not read the input module")?;
    let mut module = parse::parse(&module_bytes).context("could not parse the input module")?;

    Interpreter::new(&mut module, args).process()?;
    PostProc::new(&mut module).process();

    let module = encode::encode(&mut module);

    if cfg!(debug_assertions) {
        if let Err(e) = parse::parse(&module) {
            panic!("emitted a malformed module: {e}");
        }
    }

    fs::write(&output_path, &module).context("could not save the result")?;

    Ok(())
}
