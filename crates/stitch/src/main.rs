use std::error::Error;
use std::process::exit;
use std::{env, fs, mem};

use stitch::ast::FuncBody;
use stitch::post::PostProc;
use stitch::spec::Specializer;
use stitch::{cfg, encode, parse};

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

    let func_ids = module.funcs.iter().filter(|(_, func)| !func.is_import()).map(|x| x.0).collect::<Vec<_>>();

    for (idx, &func_id) in func_ids.iter().enumerate() {
        if idx > 0 {
            eprintln!("\n");
        }

        let body = module.funcs[func_id].body_mut().unwrap();
        let body = mem::replace(body, FuncBody::new(body.ty.clone()));
        eprintln!("FUNC {func_id:?}:\n{}", body.main_block);
        let cfg = cfg::FuncBody::from_ast(&module, &body);
        eprintln!("\ncfg:\n{cfg}");
        let body = cfg.to_ast();
        eprintln!("\nafter:\n{}", body.main_block);
        *module.funcs[func_id].body_mut().unwrap() = body;
    }

    Specializer::new(&mut module).process();
    PostProc::new(&mut module).process();

    eprintln!("\n\n{}\n", "=".repeat(120));
    let func_ids = module.funcs.iter().filter(|(_, func)| !func.is_import()).map(|x| x.0).collect::<Vec<_>>();

    for (idx, &func_id) in func_ids.iter().enumerate() {
        if idx > 0 {
            eprintln!("\n");
        }

        let body = module.funcs[func_id].body_mut().unwrap();
        let body = mem::replace(body, FuncBody::new(body.ty.clone()));
        eprintln!("FUNC {func_id:?}:\n{}", body.main_block);
        let cfg = cfg::FuncBody::from_ast(&module, &body);
        eprintln!("\ncfg:\n{cfg}");
        let body = cfg.to_ast();
        eprintln!("\nafter:\n{}", body.main_block);
        *module.funcs[func_id].body_mut().unwrap() = body;
    }

    let module = encode::encode(&mut module);

    if cfg!(debug_assertions) {
        parse::parse(&module).unwrap();
    }

    fs::write(&output_path, &module)?;

    Ok(())
}
