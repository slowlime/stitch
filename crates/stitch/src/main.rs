use std::error::Error;
use std::process::exit;
use std::{env, fs};

use stitch::{parse, encode};

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args_os().skip(1);

    if args.len() != 2 {
        eprintln!("Usage: stitch <input wasm module> <output module>");
        exit(2);
    }

    let input_path = args.next().unwrap();
    let output_path = args.next().unwrap();

    let module_bytes = fs::read(&input_path)?;
    let mut module = parse::parse(&module_bytes)?;
    let module = encode::encode(&mut module);

    fs::write(&output_path, &module)?;

    Ok(())
}
