use std::path::PathBuf;

use clap::Parser;

#[derive(Parser, Debug)]
pub struct Args {
    #[arg(short = 'p', long)]
    pub class_search_paths: Vec<PathBuf>,

    #[arg(default_value = "Main")]
    pub main_class_name: String,
}
