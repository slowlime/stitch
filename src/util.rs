macro_rules! try_match {
    ($scrutinee:expr, $pattern:pat => $map:expr) => (match $scrutinee {
        $pattern => Some($map),
        _ => None,
    });
}

pub(crate) use try_match;
