pub mod iter;
pub mod slot;

macro_rules! try_match {
    ($scrutinee:expr, $pattern:pat => $arm:expr) => {
        match $scrutinee {
            $pattern => Some($arm),
            _ => None,
        }
    };
}

pub(crate) use try_match;
