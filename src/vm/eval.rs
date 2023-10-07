use std::num::NonZeroUsize;
use std::rc::Rc;

use crate::ast;

use super::error::VmError;
use super::frame::Callee;
use super::value::Value;
use super::Vm;

pub enum Effect<'gc> {
    None(Value<'gc>),
    Return(Value<'gc>),
    Unwind(VmError),

    // Return(value) is equivalent to NonLocalReturn { value, up_frames: 0 }
    NonLocalReturn {
        value: Value<'gc>,
        up_frames: NonZeroUsize,
    },
}

impl<'gc> Effect<'gc> {
    pub fn then_pure(self) -> Self {
        self
    }

    pub fn then_return(self) -> Self {
        self.and_then(Effect::Return)
    }

    pub fn then_unwind(self, err: VmError) -> Self {
        self.and_then(|_| Effect::Unwind(err))
    }

    pub fn then_nlret(self, up_frames: NonZeroUsize) -> Self {
        match self {
            Self::None(value) => Self::NonLocalReturn { value, up_frames },
            Self::Unwind(_) | Self::Return(_) => self,
            Self::NonLocalReturn {
                value,
                up_frames: inner_up_frames,
            } => Self::NonLocalReturn {
                value,
                up_frames: up_frames.max(inner_up_frames),
            },
        }
    }

    pub fn and_then<F>(self, f: F) -> Self
    where
        F: FnOnce(Value<'gc>) -> Self,
    {
        match self {
            Self::None(value) => f(value),
            Self::Return(_) | Self::Unwind(_) | Self::NonLocalReturn { .. } => self,
        }
    }
}

impl ast::Block {
    pub(super) fn eval<'gc>(&self, vm: &mut Vm<'gc>) -> Effect<'gc> {
        for stmt in &self.body {
            match stmt.eval(vm) {
                Effect::None(_) => {}
                eff @ (Effect::Return(_) | Effect::Unwind(_)) => return eff,

                Effect::NonLocalReturn { value, up_frames } => {
                    return match NonZeroUsize::new(up_frames.get() - 1) {
                        Some(up_frames) => Effect::NonLocalReturn { value, up_frames },
                        None => Effect::Return(value),
                    }
                }
            }
        }

        panic!("block has no return statement");
    }
}

impl ast::Stmt {
    pub(super) fn eval<'gc>(&self, vm: &mut Vm<'gc>) -> Effect<'gc> {
        match self {
            Self::Return(expr) => expr.value.eval(vm).then_return(),
            Self::Expr(expr) => expr.value.eval(vm).then_pure(),
            Self::Dummy => panic!("Stmt::Dummy in AST"),

            Self::NonLocalReturn(expr) => {
                expr.value.eval(vm).and_then(|value| {
                    let flag = match vm.frames.last().unwrap().callee {
                        Callee::Method { .. } => panic!("Stmt::NonLocalReturn in method body"),

                        Callee::Block { value: ref block } => {
                            match block.get().nlret_valid_flag.upgrade() {
                                Some(rc) => rc,

                                None => {
                                    // TODO: dispatch to #escapedBlock:
                                    // ↑ does this mean blocks always have to capture `self`?
                                    // if they do, the validity flag won't be necessary as nlret is valid
                                    // iff `self` is not closed.
                                    // TODO: investigate this as well
                                    return Effect::Unwind(VmError::NlRetFromEscapedBlock {
                                        ret_span: expr.span(),
                                    });
                                }
                            }
                        }
                    };

                    // compute how many frames we have to skip
                    let up_frames = vm
                        .frames
                        .iter()
                        .rev()
                        .enumerate()
                        .skip(1) // skip the current frame, which won't match
                        .find(|(_, frame)| match frame.callee {
                            Callee::Method {
                                ref nlret_valid_flag,
                                ..
                            } => Rc::ptr_eq(nlret_valid_flag, &flag),
                            _ => false,
                        })
                        .and_then(|(idx, _)| NonZeroUsize::new(idx))
                        .expect("non-local return flag valid but no target frame was found");

                    Effect::NonLocalReturn { value, up_frames }
                })
            }
        }
    }
}

impl ast::Expr {
    pub(super) fn eval<'gc>(&self, vm: &mut Vm<'gc>) -> Effect<'gc> {
        todo!()
    }
}
