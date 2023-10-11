use std::num::NonZeroUsize;
use std::ptr;

use crate::ast;
use crate::location::Spanned;

use super::error::VmError;
use super::frame::{Callee, Local, Upvalue};
use super::method::{MethodDef, Primitive};
use super::value::{tag, TypedValue, Value};
use super::Vm;

/// The result of a computation — either a plain value or a triggered effect.
pub enum Effect<'gc> {
    /// The computation has completed successfully and terminated with the given value.
    None(Value<'gc>),

    /// The computation directs the control flow to pop the current frame, returning the given value.
    Return(Value<'gc>),

    /// The computation has encountered an error, which lead to unwinding the stack.
    Unwind(VmError),

    /// The computation directs the control flow to pop `up_frames` frames and then return from the resulting frame with
    /// the given value.
    ///
    /// Note that `Return(value)` would be equivalent to `NonLocalReturn { value, up_frames: 0 }`.
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
        // can i haz monads pls
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
                    let self_local = match vm.frames.last().unwrap().callee {
                        Callee::Method { .. } => panic!("Stmt::NonLocalReturn in method body"),

                        Callee::Block { ref block } => {
                            let Some(self_upval) = block.get().get_upvalue_by_name("self") else {
                                panic!("non-local return in a block that does not capture `self`");
                            };

                            if self_upval.is_closed() {
                                // the method (which defines `self`) is no longer active, meaning the block has escaped.
                                let obj = self_upval.get_local().value.borrow();

                                return if let Some(method) =
                                    obj.get_method_by_name(vm, "escapedBlock:").cloned()
                                {
                                    let args = vec![obj.clone(), block.clone().into_value()];
                                    drop(obj);

                                    method.eval(vm, args)
                                } else {
                                    Effect::Unwind(VmError::NlRetFromEscapedBlock {
                                        ret_span: expr.span(),
                                    })
                                };
                            }

                            self_upval.get_local()
                        }
                    };

                    // compute how many frames we have to skip
                    let up_frames = vm
                        .frames
                        .iter()
                        .rev()
                        .enumerate()
                        .skip(1) // skip the current frame, which won't match
                        .find(|(_, frame)| {
                            frame
                                .get_local_by_name("self")
                                .is_some_and(|local| ptr::eq(local, self_local))
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
        match self {
            Self::Assign(expr) => expr.eval(vm),
            Self::Block(expr) => expr.eval(vm),
            Self::Array(expr) => expr.eval(vm),
            Self::Symbol(expr) => expr.eval(vm),
            Self::String(expr) => expr.eval(vm),
            Self::Int(expr) => expr.eval(vm),
            Self::Float(expr) => expr.eval(vm),
            Self::Dispatch(expr) => expr.eval(vm),

            Self::UnresolvedName(expr) => expr.eval(vm),
            Self::Local(expr) => expr.eval(vm),
            Self::Upvalue(expr) => expr.eval(vm),
            Self::Field(expr) => expr.eval(vm),
            Self::Global(expr) => expr.eval(vm),

            Self::Dummy => panic!("Expr::Dummy in AST"),
        }
    }
}

impl ast::Assign {
    pub(super) fn eval<'gc>(&self, vm: &mut Vm<'gc>) -> Effect<'gc> {
        self.value.eval(vm).and_then(|value| match &self.var {
            ast::AssignVar::UnresolvedName(_) => panic!("UnresolvedName in AST"),

            ast::AssignVar::Local(local) => {
                *local.lookup(vm).value.borrow_mut() = value.clone();

                Effect::None(value)
            }

            ast::AssignVar::Upvalue(upvalue) => {
                *upvalue.lookup(vm).get_local().value.borrow_mut() = value.clone();

                Effect::None(value)
            }

            ast::AssignVar::Field(name) => {
                let recv = vm.frames.last().unwrap().get_recv().unwrap().borrow();
                let obj = recv.get_obj().expect("self has no associated object");
                let Some(mut field) = obj.get_field_by_name_mut(&name.0.value) else {
                    panic!("unknown field `{}`", name.0.value);
                };

                *field = value.clone();

                Effect::None(value)
            }

            ast::AssignVar::Global(name) => {
                if vm.get_global(&name.0.value).is_some() {
                    vm.set_global(name.0.value.clone(), value.clone());

                    Effect::None(value)
                } else {
                    Effect::Unwind(VmError::UndefinedName {
                        span: name.0.span(),
                        name: name.0.value.clone(),
                    })
                }
            }
        })
    }
}

impl Spanned<ast::Block> {
    pub(super) fn eval<'gc>(&self, vm: &mut Vm<'gc>) -> Effect<'gc> {
        // TODO: whoa, cloning the whole ast here seems excessive
        Effect::None(vm.make_block(self.clone()).into_value())
    }
}

impl ast::ArrayLit {
    pub(super) fn eval<'gc>(&self, vm: &mut Vm<'gc>) -> Effect<'gc> {
        let mut values = vec![];

        for expr in &self.0.value {
            match expr.eval(vm) {
                Effect::None(value) => values.push(value),
                eff => return eff,
            }
        }

        Effect::None(vm.make_array(values).into_value())
    }
}

impl ast::SymbolLit {
    pub(super) fn eval<'gc>(&self, vm: &mut Vm<'gc>) -> Effect<'gc> {
        Effect::None(vm.make_symbol(self.clone()).into_value())
    }
}

impl ast::StringLit {
    pub(super) fn eval<'gc>(&self, vm: &mut Vm<'gc>) -> Effect<'gc> {
        Effect::None(vm.make_string(self.0.value.clone()).into_value())
    }
}

impl ast::IntLit {
    pub(super) fn eval<'gc>(&self, vm: &mut Vm<'gc>) -> Effect<'gc> {
        Effect::None(vm.make_int(self.0.value).into_value())
    }
}

impl ast::FloatLit {
    pub(super) fn eval<'gc>(&self, vm: &mut Vm<'gc>) -> Effect<'gc> {
        Effect::None(vm.make_float(self.0.value).into_value())
    }
}

impl ast::Dispatch {
    pub(super) fn eval<'gc>(&self, vm: &mut Vm<'gc>) -> Effect<'gc> {
        let recv = match self.recv.eval(vm) {
            Effect::None(recv) => recv,
            eff => return eff,
        };

        let mut args = vec![recv];

        for expr in &self.args {
            match expr.eval(vm) {
                Effect::None(arg) => args.push(arg),
                eff => return eff,
            }
        }

        let recv = &args[0];

        match recv.get_method_by_name(vm, self.selector.value.name()) {
            Some(method) => method.clone().eval(vm, args),

            None => {
                let class = recv.get_class(vm).get();

                Effect::Unwind(VmError::NoSuchMethod {
                    span: self.location.span(),
                    class_span: class.name.span(),
                    class_name: class.name.value.clone(),
                    method_name: self.selector.value.name().to_owned(),
                })
            }
        }
    }
}

impl ast::UnresolvedName {
    pub(super) fn eval<'gc>(&self, _vm: &mut Vm<'gc>) -> Effect<'gc> {
        panic!("UnresolvedName in AST");
    }
}

impl ast::Local {
    pub(super) fn eval<'gc>(&self, vm: &mut Vm<'gc>) -> Effect<'gc> {
        Effect::None(self.lookup(vm).value.borrow().clone())
    }

    pub(super) fn lookup<'a, 'gc>(&self, vm: &'a Vm<'gc>) -> &'a Local<'gc> {
        let Some(local) = vm.frames.last().unwrap().get_local_by_name(&self.0.value) else {
            panic!("unknown local `{}`", self.0.value);
        };

        local
    }
}

impl ast::Upvalue {
    pub(super) fn eval<'gc>(&self, vm: &mut Vm<'gc>) -> Effect<'gc> {
        Effect::None(self.lookup(vm).get_local().value.borrow().clone())
    }

    pub(super) fn lookup<'a, 'gc>(&self, vm: &'a Vm<'gc>) -> &'a Upvalue<'gc> {
        match &vm.frames.last().unwrap().callee {
            Callee::Method { .. } => panic!("upvalue access in a method"),
            Callee::Block { block } => match block.get().get_upvalue_by_name(&self.name.value) {
                Some(upvalue) => upvalue,
                None => panic!("unknown upvalue `{}`", self.name.value),
            },
        }
    }
}

impl ast::Field {
    pub(super) fn eval<'gc>(&self, vm: &mut Vm<'gc>) -> Effect<'gc> {
        let recv = vm.frames.last().unwrap().get_recv().unwrap().borrow();
        let obj = recv.get_obj().expect("self has no associated object");
        let Some(field) = obj.get_field_by_name(&self.0.value) else {
            panic!("unknown field `{}`", self.0.value);
        };

        Effect::None(field.clone())
    }
}

impl ast::Global {
    pub(super) fn eval<'gc>(&self, vm: &mut Vm<'gc>) -> Effect<'gc> {
        match vm.get_global(&self.0.value) {
            Some(value) => Effect::None(value.clone()),

            None => Effect::Unwind(VmError::UndefinedName {
                span: self.0.span(),
                name: self.0.value.clone(),
            }),
        }
    }
}

impl<'gc> TypedValue<'gc, tag::Method> {
    pub(super) fn eval(&self, vm: &mut Vm<'gc>, args: Vec<Value<'gc>>) -> Effect<'gc> {
        let result = match self.get().def.value {
            MethodDef::Code(ref block) => {
                if let Err(e) = vm.push_frame(
                    block,
                    self.get().location.span(),
                    Callee::Method {
                        method: self.clone(),
                    },
                    args,
                ) {
                    return Effect::Unwind(e);
                }

                let result = block.eval(vm);
                vm.pop_frame();

                result
            }

            MethodDef::Primitive(p) => p.eval(vm, args),
        };

        match result {
            Effect::None(_) => panic!("block has no return statement"),
            Effect::Return(value) => Effect::None(value),
            Effect::Unwind(err) => Effect::Unwind(err),
            Effect::NonLocalReturn { value, up_frames } => {
                match NonZeroUsize::new(up_frames.get() - 1) {
                    Some(up_frames) => Effect::NonLocalReturn { value, up_frames },
                    None => Effect::Return(value),
                }
            }
        }
    }
}

impl Primitive {
    pub(super) fn eval<'gc>(&self, vm: &mut Vm<'gc>, args: Vec<Value<'gc>>) -> Effect<'gc> {
        todo!()
    }
}
