use std::path::PathBuf;

use clap::{Parser, ValueEnum};

#[derive(ValueEnum, Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum BigNumberBehavior {
    #[default]
    Error,

    Saturate,
}

impl From<BigNumberBehavior> for tuna::parse::BigNumberBehavior {
    fn from(value: BigNumberBehavior) -> Self {
        match value {
            BigNumberBehavior::Error => Self::Error,
            BigNumberBehavior::Saturate => Self::Saturate,
        }
    }
}

#[derive(Parser, Debug)]
pub struct Args {
    #[arg(short = 'w', long)]
    pub print_warnings: bool,

    #[arg(short = 'v', long)]
    pub verbose: bool,

    #[arg(long, value_enum, default_value_t)]
    pub big_numbers: BigNumberBehavior,

    #[arg(short = 'p', long)]
    pub class_search_paths: Vec<PathBuf>,

    #[arg(default_value = "Main")]
    pub main_class_name: String,

    pub args: Vec<String>,
}
