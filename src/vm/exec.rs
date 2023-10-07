use std::num::NonZeroUsize;

use crate::ast;

use super::error::VmError;
use super::value::Value;
use super::Vm;

pub enum Effect<'gc> {
    None,
    Return(Value<'gc>),
    Unwind(VmError),

    // Return(value) is equivalent to NonLocalReturn { value, up_frames: 0 }
    NonLocalReturn {
        value: Value<'gc>,
        up_frames: NonZeroUsize,
    },
}

impl ast::Block {
    pub(super) fn exec<'gc>(&self, vm: &mut Vm<'gc>) -> Effect<'gc> {
        for stmt in &self.body {
            match stmt.exec(vm) {
                Effect::None => {}
                eff @ (Effect::Return(_) | Effect::Unwind(_)) => return eff,

                Effect::NonLocalReturn { value, up_frames } => {
                    return match NonZeroUsize::new(up_frames.get() - 1) {
                        Some(up_frames) => Effect::NonLocalReturn { value, up_frames },
                        None => Effect::Return(value),
                    }
                }
            }
        }

        Effect::None
    }
}
