pub mod float;
pub mod iter;
pub mod slot;

use std::fmt::{self, Display};

macro_rules! try_match {
    ($scrutinee:expr, $pattern:pat => $arm:expr) => {
        match $scrutinee {
            $pattern => Some($arm),
            _ => None,
        }
    };
}

pub(crate) use try_match;

#[derive(Debug, Clone, Copy)]
pub struct Indent<I>(pub usize, pub I);

impl<I: Display> Display for Indent<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for _ in 0..self.0 {
            write!(f, "{}", self.1)?;
        }

        Ok(())
    }
}
