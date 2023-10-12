use std::mem;
use std::num::NonZeroUsize;
use std::ptr;

use crate::ast;
use crate::location::{Span, Spanned};
use crate::vm::gc::GcRef;

use super::error::VmError;
use super::frame::{Callee, Local, Upvalue};
use super::method::{MethodDef, Primitive};
use super::value::{tag, TypedValue, Value};
use super::{check_arg_count, Vm};

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

macro_rules! ok_or_unwind {
    ($e:expr) => {
        match $e {
            Ok(value) => value,
            Err(e) => return Effect::Unwind(e),
        }
    };
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

    pub(super) fn eval_and_pop_frame<'gc>(&self, vm: &mut Vm<'gc>) -> Effect<'gc> {
        let result = self.eval(vm);
        vm.pop_frame();

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

                                return if let Some(method) = obj
                                    .get_class(vm)
                                    .get()
                                    .get_method_by_name("escapedBlock:")
                                    .cloned()
                                {
                                    let args = vec![obj.clone(), block.clone().into_value()];
                                    let arg_spans = vec![None];
                                    drop(obj);

                                    method.eval(vm, expr.span(), args, arg_spans)
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
        let mut arg_spans = vec![self.recv.location().span()];

        for expr in &self.args {
            match expr.eval(vm) {
                Effect::None(arg) => {
                    args.push(arg);
                    arg_spans.push(expr.location().span());
                }

                eff => return eff,
            }
        }

        let recv = &args[0];

        let method = if self.supercall {
            recv.get_class(vm)
                .get()
                .get_supermethod_by_name(self.selector.value.name())
        } else {
            recv.get_class(vm)
                .get()
                .get_method_by_name(self.selector.value.name())
        };

        match method {
            Some(method) => method
                .clone()
                .eval(vm, self.location.span(), args, arg_spans),

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
    pub(super) fn eval(
        &self,
        vm: &mut Vm<'gc>,
        dispatch_span: Option<Span>,
        args: Vec<Value<'gc>>,
        arg_spans: Vec<Option<Span>>,
    ) -> Effect<'gc> {
        match self.get().def.value {
            MethodDef::Code(ref block) => {
                if let Err(e) = vm.push_frame(
                    block,
                    dispatch_span,
                    Callee::Method {
                        method: self.clone(),
                    },
                    args,
                ) {
                    Effect::Unwind(e)
                } else {
                    block.eval_and_pop_frame(vm)
                }
            }

            MethodDef::Primitive {
                primitive: p,
                ref params,
            } => {
                ok_or_unwind!(check_arg_count(
                    &args,
                    params,
                    dispatch_span,
                    self.get().location.span(),
                    p.as_selector().to_string(),
                ));

                p.eval(vm, dispatch_span, args, arg_spans)
            }
        }
    }
}

impl<'gc> TypedValue<'gc, tag::Block> {
    pub(super) fn eval(
        &self,
        vm: &mut Vm<'gc>,
        dispatch_span: Option<Span>,
        args: Vec<Value<'gc>>,
        arg_spans: Vec<Option<Span>>,
    ) -> Effect<'gc> {
        ok_or_unwind!(vm.push_frame(
            &self.get().code,
            dispatch_span,
            Callee::Block {
                block: self.clone(),
            },
            args,
        ));

        self.get().code.eval_and_pop_frame(vm)
    }
}

impl Primitive {
    pub(super) fn eval<'gc>(
        &self,
        vm: &mut Vm<'gc>,
        dispatch_span: Option<Span>,
        args: Vec<Value<'gc>>,
        arg_spans: Vec<Option<Span>>,
    ) -> Effect<'gc> {
        debug_assert_eq!(args.len(), self.param_count());
        debug_assert_eq!(args.len(), arg_spans.len());

        #[inline]
        fn check_arr_idx<'a, 'gc>(
            span: Option<Span>,
            contents: &GcRef<'a, Vec<Value<'gc>>>,
            idx: i64,
        ) -> Result<usize, VmError> {
            match usize::try_from(idx) {
                Ok(idx) if idx < contents.len() => Ok(idx),
                _ => Err(VmError::IndexOutOfBounds {
                    span,
                    idx,
                    size: contents.len(),
                }),
            }
        }

        #[inline]
        fn block_value<'gc>(
            vm: &mut Vm<'gc>,
            dispatch_span: Option<Span>,
            mut args: Vec<Value<'gc>>,
            mut arg_spans: Vec<Option<Span>>,
        ) -> Effect<'gc> {
            let recv = mem::take(&mut args[0]);
            let block = ok_or_unwind!(recv.downcast_or_err::<tag::Block>(arg_spans[0]));

            for _ in args.len()..block.get().code.params.len() {
                args.push(vm.builtins().nil_object.clone().into_value());
                arg_spans.push(None);
            }

            block.eval(vm, dispatch_span, args, arg_spans)
        }

        match self {
            Primitive::ArrayAt => {
                let [recv, idx] = args.try_into().unwrap();
                let arr = ok_or_unwind!(recv.downcast_or_err::<tag::Array>(arg_spans[0]));
                let idx = ok_or_unwind!(idx.downcast_or_err::<tag::Int>(arg_spans[1])).get();
                let contents = arr.get().borrow();
                let idx = ok_or_unwind!(check_arr_idx(dispatch_span, &contents, idx));

                Effect::None(contents[idx].clone())
            }

            Primitive::ArrayAtPut => {
                let [recv, idx, value] = args.try_into().unwrap();
                let arr = ok_or_unwind!(recv.downcast_or_err::<tag::Array>(arg_spans[0]));
                let idx = ok_or_unwind!(idx.downcast_or_err::<tag::Int>(arg_spans[1])).get();
                let idx = ok_or_unwind!(check_arr_idx(dispatch_span, &arr.get().borrow(), idx));

                arr.get().borrow_mut()[idx] = value;

                Effect::None(arr.into_value())
            }

            Primitive::ArrayLength => {
                let [recv] = args.try_into().unwrap();
                let arr = ok_or_unwind!(recv.downcast_or_err::<tag::Array>(arg_spans[0]));
                let len = arr.get().borrow().len();

                Effect::None(vm.make_int(len as _).into_value())
            }

            Primitive::ArrayNew => {
                let [recv, len] = args.try_into().unwrap();
                let _ = ok_or_unwind!(recv.downcast_or_err::<tag::Class>(arg_spans[0]));
                let len = ok_or_unwind!(len.downcast_or_err::<tag::Int>(arg_spans[1])).get();
                let len = if len < 0 {
                    return Effect::Unwind(VmError::ArraySizeNegative {
                        span: dispatch_span,
                        size: len,
                    });
                } else {
                    match usize::try_from(len) {
                        Ok(len) if len < isize::MAX as _ => len,

                        _ => {
                            return Effect::Unwind(VmError::ArrayTooLarge {
                                span: dispatch_span,
                                size: len,
                            })
                        }
                    }
                };

                let mut arr = Vec::with_capacity(len);

                for _ in 0..len {
                    arr.push(vm.builtins().nil_object.clone().into_value());
                }

                Effect::None(vm.make_array(arr).into_value())
            }

            Primitive::BlockValue => block_value(vm, dispatch_span, args, arg_spans),
            Primitive::BlockRestart => todo!("what is this even supposed to do?"),
            Primitive::Block1Value => {
                Primitive::BlockValue.eval(vm, dispatch_span, args, arg_spans)
            }

            Primitive::Block2Value => block_value(vm, dispatch_span, args, arg_spans),
            Primitive::Block3ValueWith => block_value(vm, dispatch_span, args, arg_spans),

            Primitive::ClassName => {
                let [recv] = args.try_into().unwrap();
                let cls = ok_or_unwind!(recv.downcast_or_err::<tag::Class>(arg_spans[0]));

                Effect::None(vm.make_string(cls.get().name.value.clone()).into_value())
            }

            Primitive::ClassNew => {
                let [recv] = args.try_into().unwrap();
                let cls = ok_or_unwind!(recv.downcast_or_err::<tag::Class>(arg_spans[0]));

                Effect::None(vm.make_object(cls).into_value())
            }

            Primitive::ClassSuperclass => {
                let [recv] = args.try_into().unwrap();
                let cls = ok_or_unwind!(recv.downcast_or_err::<tag::Class>(arg_spans[0]));

                Effect::None(match cls.get().superclass {
                    Some(ref superclass) => superclass.clone().into_value(),
                    None => vm.builtins().nil_object.clone().into_value(),
                })
            }

            Primitive::ClassFields => {
                let [recv] = args.try_into().unwrap();
                let obj = match recv.get_obj() {
                    Some(obj) => obj,
                    None => return Effect::None(vm.make_array(vec![]).into_value()),
                };

                let fields = obj.fields.borrow().clone();

                Effect::None(vm.make_array(fields).into_value())
            }

            Primitive::ClassMethods => {
                let [recv] = args.try_into().unwrap();
                let cls = recv.get_class(vm);

                Effect::None(
                    vm.make_array(
                        cls.get()
                            .methods
                            .iter()
                            .map(|method| method.clone().into_value())
                            .collect(),
                    )
                    .into_value(),
                )
            }

            Primitive::DoubleAdd => todo!(),
            Primitive::DoubleSub => todo!(),
            Primitive::DoubleMul => todo!(),
            Primitive::DoubleDiv => todo!(),
            Primitive::DoubleMod => todo!(),
            Primitive::DoubleSqrt => todo!(),
            Primitive::DoubleRound => todo!(),
            Primitive::DoubleAsInteger => todo!(),
            Primitive::DoubleCos => todo!(),
            Primitive::DoubleSin => todo!(),
            Primitive::DoubleEq => todo!(),
            Primitive::DoubleLt => todo!(),
            Primitive::DoubleAsString => todo!(),
            Primitive::DoublePositiveInfinity => todo!(),
            Primitive::DoubleFromString => todo!(),

            Primitive::MethodSignature => todo!(),
            Primitive::MethodHolder => todo!(),
            Primitive::MethodInvokeOnWith => todo!(),

            Primitive::PrimitiveSignature => todo!(),
            Primitive::PrimitiveHolder => todo!(),
            Primitive::PrimitiveInvokeOnWith => todo!(),

            Primitive::SymbolAsString => todo!(),

            Primitive::IntegerAdd => todo!(),
            Primitive::IntegerSub => todo!(),
            Primitive::IntegerMul => todo!(),
            Primitive::IntegerDiv => todo!(),
            Primitive::IntegerFDiv => todo!(),
            Primitive::IntegerMod => todo!(),
            Primitive::IntegerBand => todo!(),
            Primitive::IntegerShl => todo!(),
            Primitive::IntegerShr => todo!(),
            Primitive::IntegerBxor => todo!(),
            Primitive::IntegerSqrt => todo!(),
            Primitive::IntegerAtRandom => todo!(),
            Primitive::IntegerEq => todo!(),
            Primitive::IntegerLt => todo!(),
            Primitive::IntegerAsString => todo!(),
            Primitive::IntegerAs32BitSignedValue => todo!(),
            Primitive::IntegerAs32BitUnsignedValue => todo!(),
            Primitive::IntegerAsDouble => todo!(),
            Primitive::IntegerFromString => todo!(),

            Primitive::ObjectClass => todo!(),
            Primitive::ObjectObjectSize => todo!(),
            Primitive::ObjectRefEq => todo!(),
            Primitive::ObjectHashcode => todo!(),
            Primitive::ObectInspect => todo!(),
            Primitive::ObjectHalt => todo!(),
            Primitive::ObjectPerform => todo!(),
            Primitive::ObjectPerformWithArguments => todo!(),
            Primitive::ObjectPerformInSuperclass => todo!(),
            Primitive::ObjectPerformWithArgmentsInSuperclass => todo!(),
            Primitive::ObjectInstVarAt => todo!(),
            Primitive::ObjectInstVarAtPut => todo!(),
            Primitive::ObjectInstVarNamed => todo!(),

            Primitive::StringConcatenate => todo!(),
            Primitive::StringAsSymbol => todo!(),
            Primitive::StringHashcode => todo!(),
            Primitive::StringLength => todo!(),
            Primitive::StringIsWhitespace => todo!(),
            Primitive::StringIsLetters => todo!(),
            Primitive::StringIsDigits => todo!(),
            Primitive::StringEq => todo!(),
            Primitive::StringPrimSubstringFromTo => todo!(),
            Primitive::SystemGlobal => todo!(),
            Primitive::SystemGlobalPut => todo!(),
            Primitive::SystemHasGlobal => todo!(),
            Primitive::SystemLoadFile => todo!(),
            Primitive::SystemLoad => todo!(),
            Primitive::SystemExit => todo!(),
            Primitive::SystemPrintString => todo!(),
            Primitive::SystemPrintNewline => todo!(),
            Primitive::SystemErrorPrintln => todo!(),
            Primitive::SystemErrorPrint => todo!(),
            Primitive::SystemPrintStackTrace => todo!(),
            Primitive::SystemTime => todo!(),
            Primitive::SystemTicks => todo!(),
            Primitive::SystemFullGC => todo!(),
        }
    }
}
