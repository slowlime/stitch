use super::value::Value;

#[derive(Debug, Clone)]
pub struct Frame<'gc> {
    locals: Vec<Local<'gc>>,
}

#[derive(Debug, Clone)]
pub struct Local<'gc> {
    name: String,
    value: Value<'gc>,
}
