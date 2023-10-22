use std::mem;
use std::num::NonZeroUsize;
use std::pin::Pin;
use std::ptr;

use crate::ast::{self, SymbolLit as Symbol};
use crate::location::{Span, Spanned};

use super::error::VmError;
use super::frame::{Callee, Frame, Local, Upvalue};
use super::method::{MethodDef, Primitive};
use super::value::{tag, SubstrError, Ty, TypedValue, Value};
use super::{check_arg_count, Vm};

/// The result of a computation — either a plain value or a triggered effect.
#[derive(Debug)]
pub enum Effect<'gc> {
    /// The computation has completed successfully and terminated with the given value.
    None(Value<'gc>),

    /// The computation directs the control flow to pop the current frame, returning the given value.
    Return(Value<'gc>),

    /// The current method should be restarted from the beginning.
    Restart,

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
            Self::Unwind(_) | Self::Restart | Self::Return(_) => self,
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
            Self::Return(_) | Self::Restart | Self::Unwind(_) | Self::NonLocalReturn { .. } => self,
        }
    }
}

impl ast::Block {
    pub(super) fn eval<'gc>(&self, vm: &mut Vm<'gc>) -> Effect<'gc> {
        'restart: loop {
            for stmt in &self.body {
                match stmt.eval(vm) {
                    Effect::None(_) => {}
                    Effect::Restart => continue 'restart,
                    eff @ (Effect::Return(_) | Effect::Unwind(_) | Effect::NonLocalReturn { .. }) => {
                        return eff
                    }
                }
            }

            break;
        }

        panic!("block has no return statement");
    }

    pub(super) fn eval_and_pop_frame<'gc>(&self, vm: &mut Vm<'gc>) -> Effect<'gc> {
        let result = self.eval(vm);
        vm.pop_frame();

        match result {
            Effect::None(_) => panic!("block has no return statement"),
            Effect::Return(value) => Effect::None(value),
            Effect::Restart => Effect::Restart,
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
                                    let arg_spans = vec![None; args.len()];
                                    drop(obj);

                                    method.eval(vm, expr.span(), args, arg_spans).then_return()
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
                let frame = vm.frames.last().expect("frame stack is empty");
                let recv = frame
                    .get_recv()
                    .expect("recv not found for field access")
                    .borrow();
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
        let frame = vm.frames.last().expect("stack frame is empty");

        // TODO: whoa, cloning the whole ast here seems excessive
        Effect::None(
            vm.make_block(frame.get_defining_method().clone(), self.clone())
                .into_value(),
        )
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
            let frame = vm.frames.last().expect("frame stack is empty");
            let holder = frame.get_defining_method().get().holder.get().unwrap();
            holder
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
                let recv = args.remove(0);
                let class = recv.get_class(vm).get();

                if let Some(method) = class
                    .get_method_by_name("doesNotUnderstand:arguments:")
                    .cloned()
                {
                    let args = vm.make_array(args);
                    let sym = vm.make_symbol(Symbol::Selector(self.selector.clone()));
                    let args = vec![recv.clone(), sym.into_value(), args.into_value()];
                    let arg_spans = vec![None; args.len()];

                    method.eval(vm, self.location.span(), args, arg_spans)
                } else {
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

            None => {
                // TODO: make a helper method for this...
                let frame = vm.frames.last().expect("frame stack is empty");
                let recv = frame
                    .get_recv()
                    .expect("`self` was not captured when accessing a global")
                    .borrow();
                let sym = vm.make_symbol(Symbol::String(self.0.clone()));

                if let Some(method) = recv
                    .get_class(vm)
                    .get()
                    .get_method_by_name("unknownGlobal:")
                    .cloned()
                {
                    let args = vec![recv.clone(), sym.into_value()];
                    let arg_spans = vec![None; args.len()];
                    drop(recv);

                    method.eval(vm, self.0.span(), args, arg_spans)
                } else {
                    Effect::Unwind(VmError::UndefinedName {
                        span: self.0.span(),
                        name: self.0.value.clone(),
                    })
                }
            }
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

                p.eval(vm, self, dispatch_span, args, arg_spans)
            }
        }
    }
}

impl<'gc> TypedValue<'gc, tag::Block> {
    pub(super) fn eval(
        &self,
        vm: &mut Vm<'gc>,
        dispatch_span: Option<Span>,
        mut args: Vec<Value<'gc>>,
        _arg_spans: Vec<Option<Span>>,
    ) -> Effect<'gc> {
        // remove the implicit receiver argument (pointing to self)
        args.remove(0);

        // TODO: use arg_spans for diagnostics
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
        this_method: &TypedValue<'gc, tag::Method>,
        dispatch_span: Option<Span>,
        args: Vec<Value<'gc>>,
        arg_spans: Vec<Option<Span>>,
    ) -> Effect<'gc> {
        debug_assert_eq!(
            args.len(),
            self.param_count(),
            "argument count does not match parameter count while calling primitive {:?}",
            self
        );
        debug_assert_eq!(args.len(), arg_spans.len());

        #[inline]
        fn check_arr_idx(
            span: Option<Span>,
            contents_len: usize,
            idx: i64,
        ) -> Result<usize, VmError> {
            match usize::try_from(idx).ok().and_then(|idx| idx.checked_sub(1)) {
                Some(idx) if idx < contents_len => Ok(idx),
                _ => Err(VmError::IndexOutOfBounds {
                    span,
                    idx,
                    size: contents_len,
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
                args.push(vm.builtins.nil_object.clone().into_value());
                arg_spans.push(None);
            }

            block.eval(vm, dispatch_span, args, arg_spans)
        }

        #[inline]
        fn shl(lhs: i64, rhs: i64) -> i64 {
            if rhs < i32::MIN as _ || rhs > i32::MAX as _ {
                0
            } else if rhs < 0 {
                shr(lhs, -rhs)
            } else {
                lhs.checked_shl(rhs as u32).unwrap_or(0)
            }
        }

        #[inline]
        fn shr(lhs: i64, rhs: i64) -> i64 {
            if rhs < i32::MIN as _ || rhs > i32::MAX as _ {
                0
            } else if rhs < 0 {
                shl(lhs, -rhs)
            } else {
                lhs.checked_shr(rhs as u32).unwrap_or(0)
            }
        }

        #[inline]
        #[allow(clippy::too_many_arguments)]
        fn object_perform<'gc>(
            dispatch_span: Option<Span>,
            vm: &mut Vm<'gc>,
            recv: Value<'gc>,
            recv_span: Option<Span>,
            sym: TypedValue<'gc, tag::Symbol>,
            sym_span: Option<Span>,
            class: Option<TypedValue<'gc, tag::Class>>,
            mut args: Vec<Value<'gc>>,
        ) -> Effect<'gc> {
            let class = match class {
                Some(class) if recv.get_class(vm).get().is_subclass_of(class.get()) => class,

                Some(class) => {
                    return Effect::Unwind(VmError::IllegalTy {
                        span: recv_span,
                        expected: vec![Ty::NamedClass(Box::new(class.get().name.value.clone()))]
                            .into(),
                        actual: Ty::NamedClass(Box::new(
                            recv.get_class(vm).get().name.value.clone(),
                        )),
                    })
                }

                None => recv.get_class(vm).clone(),
            };

            let method = match class.get().get_method_by_name(sym.get().as_str()) {
                Some(method) => method,

                None => {
                    return Effect::Unwind(VmError::NoSuchMethod {
                        span: sym_span,
                        class_span: class.get().name.span(),
                        class_name: class.get().name.value.clone(),
                        method_name: sym.get().as_str().to_owned(),
                    })
                }
            };

            args.insert(0, recv);
            let mut arg_spans = vec![None; args.len()];
            arg_spans[0] = recv_span;

            method.eval(vm, dispatch_span, args, arg_spans)
        }

        match self {
            Primitive::ArrayAt => {
                let [recv, idx] = args.try_into().unwrap();
                let arr = ok_or_unwind!(recv.downcast_or_err::<tag::Array>(arg_spans[0]));
                let idx = ok_or_unwind!(idx.downcast_or_err::<tag::Int>(arg_spans[1])).get();
                let contents = arr.get().borrow();
                let idx = ok_or_unwind!(check_arr_idx(dispatch_span, contents.len(), idx));

                Effect::None(contents[idx].clone())
            }

            Primitive::ArrayAtPut => {
                let [recv, idx, value] = args.try_into().unwrap();
                let arr = ok_or_unwind!(recv.downcast_or_err::<tag::Array>(arg_spans[0]));
                let idx = ok_or_unwind!(idx.downcast_or_err::<tag::Int>(arg_spans[1])).get();
                let idx =
                    ok_or_unwind!(check_arr_idx(dispatch_span, arr.get().borrow().len(), idx));

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
                    arr.push(vm.builtins.nil_object.clone().into_value());
                }

                Effect::None(vm.make_array(arr).into_value())
            }

            Primitive::BlockValue | Primitive::Block1Value => {
                block_value(vm, dispatch_span, args, arg_spans)
            }

            Primitive::BlockRestart => {
                let [recv] = args.try_into().unwrap();
                let _ = ok_or_unwind!(recv.downcast_or_err::<tag::Block>(arg_spans[0]));

                Effect::Restart
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

                Effect::None(match cls.get().get_superclass() {
                    Some(superclass) => superclass.clone().into_value(),
                    None => vm.builtins.nil_object.clone().into_value(),
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

            Primitive::DoubleAdd => {
                let [lhs, rhs] = args.try_into().unwrap();
                let lhs = ok_or_unwind!(lhs.downcast_or_err::<tag::Float>(arg_spans[0])).get();
                // TODO: int?
                let rhs = ok_or_unwind!(rhs.downcast_or_err::<tag::Float>(arg_spans[1])).get();

                Effect::None(vm.make_float(lhs + rhs).into_value())
            }

            Primitive::DoubleSub => {
                let [lhs, rhs] = args.try_into().unwrap();
                let lhs = ok_or_unwind!(lhs.downcast_or_err::<tag::Float>(arg_spans[0])).get();
                // TODO: int?
                let rhs = ok_or_unwind!(rhs.downcast_or_err::<tag::Float>(arg_spans[1])).get();

                Effect::None(vm.make_float(lhs - rhs).into_value())
            }

            Primitive::DoubleMul => {
                let [lhs, rhs] = args.try_into().unwrap();
                let lhs = ok_or_unwind!(lhs.downcast_or_err::<tag::Float>(arg_spans[0])).get();
                // TODO: int?
                let rhs = ok_or_unwind!(rhs.downcast_or_err::<tag::Float>(arg_spans[1])).get();

                Effect::None(vm.make_float(lhs * rhs).into_value())
            }

            Primitive::DoubleDiv => {
                let [lhs, rhs] = args.try_into().unwrap();
                let lhs = ok_or_unwind!(lhs.downcast_or_err::<tag::Float>(arg_spans[0])).get();
                // TODO: int?
                let rhs = ok_or_unwind!(rhs.downcast_or_err::<tag::Float>(arg_spans[1])).get();

                Effect::None(vm.make_float(lhs / rhs).into_value())
            }

            Primitive::DoubleMod => {
                let [lhs, rhs] = args.try_into().unwrap();
                let lhs = ok_or_unwind!(lhs.downcast_or_err::<tag::Float>(arg_spans[0])).get();
                // TODO: int?
                let rhs = ok_or_unwind!(rhs.downcast_or_err::<tag::Float>(arg_spans[1])).get();

                Effect::None(vm.make_float(lhs % rhs).into_value())
            }

            Primitive::DoubleSqrt => {
                let [recv] = args.try_into().unwrap();
                let f = ok_or_unwind!(recv.downcast_or_err::<tag::Float>(arg_spans[0])).get();

                Effect::None(vm.make_float(f.sqrt()).into_value())
            }

            Primitive::DoubleRound => {
                let [recv] = args.try_into().unwrap();
                let f = ok_or_unwind!(recv.downcast_or_err::<tag::Float>(arg_spans[0])).get();

                Effect::None(vm.make_float(f.round()).into_value())
            }

            Primitive::DoubleAsInteger => {
                let [recv] = args.try_into().unwrap();
                let f = ok_or_unwind!(recv.downcast_or_err::<tag::Float>(arg_spans[0])).get();

                match f as i64 {
                    _ if f.is_nan() => Effect::Unwind(VmError::NanFloatToInt {
                        span: dispatch_span,
                    }),

                    _ if f > i64::MAX as f64 || f < i64::MIN as f64 => {
                        Effect::Unwind(VmError::FloatTooLargeForInt {
                            span: dispatch_span,
                            value: f,
                        })
                    }

                    i => Effect::None(vm.make_int(i).into_value()),
                }
            }

            Primitive::DoubleCos => {
                let [recv] = args.try_into().unwrap();
                let f = ok_or_unwind!(recv.downcast_or_err::<tag::Float>(arg_spans[0])).get();

                Effect::None(vm.make_float(f.cos()).into_value())
            }

            Primitive::DoubleSin => {
                let [recv] = args.try_into().unwrap();
                let f = ok_or_unwind!(recv.downcast_or_err::<tag::Float>(arg_spans[0])).get();

                Effect::None(vm.make_float(f.sin()).into_value())
            }

            Primitive::DoubleEq => {
                let [lhs, rhs] = args.try_into().unwrap();
                let lhs = ok_or_unwind!(lhs.downcast_or_err::<tag::Float>(arg_spans[0])).get();

                Effect::None(
                    vm.make_boolean(
                        // TODO: int?
                        rhs.downcast::<tag::Float>()
                            .is_ok_and(|rhs| lhs == rhs.get()),
                    )
                    .into_value(),
                )
            }

            Primitive::DoubleLt => {
                let [lhs, rhs] = args.try_into().unwrap();
                let lhs = ok_or_unwind!(lhs.downcast_or_err::<tag::Float>(arg_spans[0])).get();
                // TODO: int?
                let rhs = ok_or_unwind!(rhs.downcast_or_err::<tag::Float>(arg_spans[1])).get();

                Effect::None(vm.make_boolean(lhs < rhs).into_value())
            }

            Primitive::DoubleAsString => {
                let [recv] = args.try_into().unwrap();
                let f = ok_or_unwind!(recv.downcast_or_err::<tag::Float>(arg_spans[0])).get();

                Effect::None(vm.make_string(f.to_string()).into_value())
            }

            Primitive::DoublePositiveInfinity => {
                let [recv] = args.try_into().unwrap();
                let _ = ok_or_unwind!(recv.downcast_or_err::<tag::Class>(arg_spans[0])).get();

                Effect::None(vm.make_float(f64::INFINITY).into_value())
            }

            Primitive::DoubleFromString => {
                let [recv, s] = args.try_into().unwrap();
                let _ = ok_or_unwind!(recv.downcast_or_err::<tag::Class>(arg_spans[0])).get();
                let s = ok_or_unwind!(s.downcast_or_err::<tag::String>(arg_spans[1]));

                match s.get().as_str().parse::<f64>() {
                    Ok(f) => Effect::None(vm.make_float(f).into_value()),
                    Err(_) => Effect::Unwind(VmError::DoubleFromInvalidString {
                        span: dispatch_span,
                        value: s.get().to_string(),
                    }),
                }
            }

            Primitive::MethodSignature | Primitive::PrimitiveSignature => {
                let [recv] = args.try_into().unwrap();
                let method = ok_or_unwind!(recv.downcast_or_err::<tag::Method>(arg_spans[0]));

                Effect::None(
                    vm.make_symbol(ast::SymbolLit::Selector(method.get().selector.clone()))
                        .into_value(),
                )
            }

            Primitive::MethodHolder | Primitive::PrimitiveHolder => {
                let [recv] = args.try_into().unwrap();
                let method = ok_or_unwind!(recv.downcast_or_err::<tag::Method>(arg_spans[0]));

                Effect::None(method.get().holder.get().unwrap().clone().into_value())
            }

            Primitive::MethodInvokeOnWith | Primitive::PrimitiveInvokeOnWith => {
                // this is a nebulous method: neither SOM nor ykSOM seem to define its semantics.
                // let's assume the following is the intended behavior.
                let [recv, obj, call_args] = args.try_into().unwrap();
                let method = ok_or_unwind!(recv.downcast_or_err::<tag::Method>(arg_spans[0]));
                // TODO: check that the receiver type is compatible with the type of the method holder
                // (and in other places too)
                let mut call_args =
                    ok_or_unwind!(call_args.downcast_or_err::<tag::Array>(arg_spans[2]))
                        .get()
                        .borrow()
                        .clone();
                call_args.insert(0, obj);

                // TODO: track argument locations somehow?
                let mut call_arg_spans = vec![None; call_args.len()];
                call_arg_spans.insert(0, arg_spans[1]);

                vm.frames.push(Frame {
                    callee: Callee::Method {
                        method: this_method.clone(),
                    },
                    local_map: Default::default(),
                    locals: Pin::new(Default::default()),
                });
                let result = method.eval(vm, dispatch_span, call_args, call_arg_spans);
                vm.frames.pop();

                result
            }

            Primitive::SymbolAsString => {
                let [recv] = args.try_into().unwrap();
                let sym = ok_or_unwind!(recv.downcast_or_err::<tag::Symbol>(arg_spans[0]));

                Effect::None(vm.make_string(sym.get().as_str().to_owned()).into_value())
            }

            Primitive::IntegerAdd => {
                let [lhs, rhs] = args.try_into().unwrap();
                let lhs = ok_or_unwind!(lhs.downcast_or_err::<tag::Int>(arg_spans[0])).get();
                // TODO: float?
                let rhs = ok_or_unwind!(rhs.downcast_or_err::<tag::Int>(arg_spans[1])).get();

                Effect::None(vm.make_int(lhs.wrapping_add(rhs)).into_value())
            }

            Primitive::IntegerSub => {
                let [lhs, rhs] = args.try_into().unwrap();
                let lhs = ok_or_unwind!(lhs.downcast_or_err::<tag::Int>(arg_spans[0])).get();
                // TODO: float?
                let rhs = ok_or_unwind!(rhs.downcast_or_err::<tag::Int>(arg_spans[1])).get();

                Effect::None(vm.make_int(lhs.wrapping_sub(rhs)).into_value())
            }

            Primitive::IntegerMul => {
                let [lhs, rhs] = args.try_into().unwrap();
                let lhs = ok_or_unwind!(lhs.downcast_or_err::<tag::Int>(arg_spans[0])).get();
                // TODO: float?
                let rhs = ok_or_unwind!(rhs.downcast_or_err::<tag::Int>(arg_spans[1])).get();

                Effect::None(vm.make_int(lhs.wrapping_mul(rhs)).into_value())
            }

            Primitive::IntegerDiv => {
                let [lhs, rhs] = args.try_into().unwrap();
                let lhs = ok_or_unwind!(lhs.downcast_or_err::<tag::Int>(arg_spans[0])).get();
                // TODO: float?
                let rhs = ok_or_unwind!(rhs.downcast_or_err::<tag::Int>(arg_spans[1])).get();

                Effect::None(vm.make_int(lhs.wrapping_div(rhs)).into_value())
            }

            Primitive::IntegerFDiv => {
                let [lhs, rhs] = args.try_into().unwrap();
                let lhs = ok_or_unwind!(lhs.downcast_or_err::<tag::Int>(arg_spans[0])).get() as f64;
                // TODO: float?
                let rhs = ok_or_unwind!(rhs.downcast_or_err::<tag::Int>(arg_spans[1])).get() as f64;

                Effect::None(vm.make_float(lhs / rhs).into_value())
            }

            Primitive::IntegerMod => {
                let [lhs, rhs] = args.try_into().unwrap();
                let lhs = ok_or_unwind!(lhs.downcast_or_err::<tag::Int>(arg_spans[0])).get();
                // TODO: float
                let rhs = ok_or_unwind!(rhs.downcast_or_err::<tag::Int>(arg_spans[1])).get();

                let result = match lhs.checked_rem(rhs).unwrap_or(0) {
                    rem if rem >= 0 && rhs >= 0 || rem < 0 && rhs < 0 => rem,
                    rem => rem - rhs,
                };

                Effect::None(vm.make_int(result).into_value())
            }

            Primitive::IntegerRem => {
                let [lhs, rhs] = args.try_into().unwrap();
                let lhs = ok_or_unwind!(lhs.downcast_or_err::<tag::Int>(arg_spans[0])).get();
                // TODO: float?
                let rhs = ok_or_unwind!(rhs.downcast_or_err::<tag::Int>(arg_spans[1])).get();

                Effect::None(vm.make_int(lhs.checked_rem(rhs).unwrap_or(0)).into_value())
            }

            Primitive::IntegerBand => {
                let [lhs, rhs] = args.try_into().unwrap();
                let lhs = ok_or_unwind!(lhs.downcast_or_err::<tag::Int>(arg_spans[0])).get();
                let rhs = ok_or_unwind!(rhs.downcast_or_err::<tag::Int>(arg_spans[1])).get();

                Effect::None(vm.make_int(lhs & rhs).into_value())
            }

            Primitive::IntegerShl => {
                let [lhs, rhs] = args.try_into().unwrap();
                let lhs = ok_or_unwind!(lhs.downcast_or_err::<tag::Int>(arg_spans[0])).get();
                let rhs = ok_or_unwind!(rhs.downcast_or_err::<tag::Int>(arg_spans[1])).get();

                Effect::None(vm.make_int(shl(lhs, rhs)).into_value())
            }

            Primitive::IntegerShr => {
                let [lhs, rhs] = args.try_into().unwrap();
                let lhs = ok_or_unwind!(lhs.downcast_or_err::<tag::Int>(arg_spans[0])).get();
                let rhs = ok_or_unwind!(rhs.downcast_or_err::<tag::Int>(arg_spans[1])).get();

                Effect::None(vm.make_int(shr(lhs, rhs)).into_value())
            }

            Primitive::IntegerBxor => {
                let [lhs, rhs] = args.try_into().unwrap();
                let lhs = ok_or_unwind!(lhs.downcast_or_err::<tag::Int>(arg_spans[0])).get();
                let rhs = ok_or_unwind!(rhs.downcast_or_err::<tag::Int>(arg_spans[1])).get();

                Effect::None(vm.make_int(lhs ^ rhs).into_value())
            }

            Primitive::IntegerSqrt => {
                let [recv] = args.try_into().unwrap();
                let i = ok_or_unwind!(recv.downcast_or_err::<tag::Int>(arg_spans[0])).get();

                let result = (i as f64).sqrt();

                Effect::None(if result.fract() != 0.0 || result > i64::MAX as f64 {
                    vm.make_float(result).into_value()
                } else {
                    vm.make_int(result as i64).into_value()
                })
            }

            Primitive::IntegerAtRandom => {
                let [recv] = args.try_into().unwrap();
                let _ = ok_or_unwind!(recv.downcast_or_err::<tag::Class>(arg_spans[0])).get();

                // chosen by fair dice roll.
                // guaranteed to be random.
                // (TODO)
                Effect::None(vm.make_int(4).into_value())
            }

            Primitive::IntegerEq => {
                let [lhs, rhs] = args.try_into().unwrap();
                let lhs = ok_or_unwind!(lhs.downcast_or_err::<tag::Int>(arg_spans[0])).get();

                Effect::None(
                    vm.make_boolean(
                        // TODO: float?
                        rhs.downcast::<tag::Int>().is_ok_and(|rhs| lhs == rhs.get()),
                    )
                    .into_value(),
                )
            }

            Primitive::IntegerLt => {
                let [lhs, rhs] = args.try_into().unwrap();
                let lhs = ok_or_unwind!(lhs.downcast_or_err::<tag::Int>(arg_spans[0])).get();
                // TODO: float?
                let rhs = ok_or_unwind!(rhs.downcast_or_err::<tag::Int>(arg_spans[1])).get();

                Effect::None(vm.make_boolean(lhs < rhs).into_value())
            }

            Primitive::IntegerAsString => {
                let [recv] = args.try_into().unwrap();
                let i = ok_or_unwind!(recv.downcast_or_err::<tag::Int>(arg_spans[0])).get();

                Effect::None(vm.make_string(i.to_string()).into_value())
            }

            Primitive::IntegerAs32BitSignedValue => {
                let [recv] = args.try_into().unwrap();
                let i = ok_or_unwind!(recv.downcast_or_err::<tag::Int>(arg_spans[0])).get();

                Effect::None(vm.make_int(i as i32 as i64).into_value())
            }

            Primitive::IntegerAs32BitUnsignedValue => {
                let [recv] = args.try_into().unwrap();
                let i = ok_or_unwind!(recv.downcast_or_err::<tag::Int>(arg_spans[0])).get();

                Effect::None(vm.make_int(i as u32 as i64).into_value())
            }

            Primitive::IntegerAsDouble => {
                let [recv] = args.try_into().unwrap();
                let i = ok_or_unwind!(recv.downcast_or_err::<tag::Int>(arg_spans[0])).get();

                Effect::None(vm.make_float(i as f64).into_value())
            }

            Primitive::IntegerFromString => {
                let [recv, s] = args.try_into().unwrap();
                let _ = ok_or_unwind!(recv.downcast_or_err::<tag::Class>(arg_spans[0])).get();
                let s = ok_or_unwind!(s.downcast_or_err::<tag::String>(arg_spans[1]));

                match s.get().as_str().parse::<i64>() {
                    Ok(i) => Effect::None(vm.make_int(i).into_value()),
                    Err(_) => Effect::Unwind(VmError::IntegerFromInvalidString {
                        span: dispatch_span,
                        value: s.get().to_string(),
                    }),
                }
            }

            Primitive::ObjectClass => {
                let [recv] = args.try_into().unwrap();
                let cls = recv.get_class(vm);

                Effect::None(cls.clone().into_value())
            }

            Primitive::ObjectObjectSize => {
                let [recv] = args.try_into().unwrap();
                let size = recv.size() as i64;

                Effect::None(vm.make_int(size).into_value())
            }

            Primitive::ObjectRefEq => {
                let [lhs, rhs] = args.try_into().unwrap();

                // TODO: compare ints/floats by value
                Effect::None(vm.make_boolean(lhs.ptr_eq(&rhs)).into_value())
            }

            Primitive::ObjectHashcode | Primitive::StringHashcode => {
                let [recv] = args.try_into().unwrap();

                Effect::None(vm.make_int(recv.hash_code() as i64).into_value())
            }

            Primitive::ObjectInspect => todo!("??"),
            Primitive::ObjectHalt => todo!("???????"),

            Primitive::ObjectPerform => {
                let [recv, sym] = args.try_into().unwrap();
                let sym = ok_or_unwind!(sym.downcast_or_err::<tag::Symbol>(arg_spans[1]));

                object_perform(
                    dispatch_span,
                    vm,
                    recv,
                    arg_spans[0],
                    sym,
                    arg_spans[1],
                    None,
                    vec![],
                )
            }

            Primitive::ObjectPerformWithArguments => {
                let [recv, sym, call_args] = args.try_into().unwrap();
                let sym = ok_or_unwind!(sym.downcast_or_err::<tag::Symbol>(arg_spans[1]));
                let call_args =
                    ok_or_unwind!(call_args.downcast_or_err::<tag::Array>(arg_spans[2]))
                        .get()
                        .borrow()
                        .clone();

                object_perform(
                    dispatch_span,
                    vm,
                    recv,
                    arg_spans[0],
                    sym,
                    arg_spans[1],
                    None,
                    call_args,
                )
            }

            Primitive::ObjectPerformInSuperclass => {
                let [recv, sym, superclass] = args.try_into().unwrap();
                let sym = ok_or_unwind!(sym.downcast_or_err::<tag::Symbol>(arg_spans[1]));
                let superclass =
                    ok_or_unwind!(superclass.downcast_or_err::<tag::Class>(arg_spans[2]));

                object_perform(
                    dispatch_span,
                    vm,
                    recv,
                    arg_spans[0],
                    sym,
                    arg_spans[1],
                    Some(superclass),
                    vec![],
                )
            }

            Primitive::ObjectPerformWithArgumentsInSuperclass => {
                let [recv, sym, call_args, superclass] = args.try_into().unwrap();
                let sym = ok_or_unwind!(sym.downcast_or_err::<tag::Symbol>(arg_spans[1]));
                let call_args =
                    ok_or_unwind!(call_args.downcast_or_err::<tag::Array>(arg_spans[2]))
                        .get()
                        .borrow()
                        .clone();
                let superclass =
                    ok_or_unwind!(superclass.downcast_or_err::<tag::Class>(arg_spans[3]));

                object_perform(
                    dispatch_span,
                    vm,
                    recv,
                    arg_spans[0],
                    sym,
                    arg_spans[1],
                    Some(superclass),
                    call_args,
                )
            }

            Primitive::ObjectInstVarAt => {
                let [recv, idx] = args.try_into().unwrap();
                let idx = ok_or_unwind!(idx.downcast_or_err::<tag::Int>(arg_spans[1])).get();

                let _fields;
                let _empty_vec;

                let fields = if let Some(obj) = recv.get_obj() {
                    _fields = obj.fields.borrow();
                    &*_fields
                } else {
                    _empty_vec = vec![];
                    &_empty_vec
                };

                let idx = ok_or_unwind!(check_arr_idx(arg_spans[1], fields.len(), idx));

                // TODO: forbid access to $-fields
                Effect::None(fields[idx].clone())
            }

            Primitive::ObjectInstVarAtPut => {
                let [recv, idx, value] = args.try_into().unwrap();
                let idx = ok_or_unwind!(idx.downcast_or_err::<tag::Int>(arg_spans[1])).get();

                {
                    let mut _fields;
                    let mut _empty_vec;

                    let fields = if let Some(obj) = recv.get_obj() {
                        _fields = obj.fields.borrow_mut();
                        &mut *_fields
                    } else {
                        _empty_vec = vec![];
                        &mut _empty_vec
                    };

                    let idx = ok_or_unwind!(check_arr_idx(arg_spans[1], fields.len(), idx));

                    // TODO: forbid access to $-fields
                    fields[idx] = value;
                }

                Effect::None(recv)
            }

            Primitive::ObjectInstVarNamed => {
                let [recv, sym] = args.try_into().unwrap();
                let sym = ok_or_unwind!(sym.downcast_or_err::<tag::Symbol>(arg_spans[1]));

                match recv
                    .get_obj()
                    .and_then(|obj| obj.get_field_by_name(sym.get().as_str()))
                    .map(|field| field.clone())
                {
                    Some(value) => Effect::None(value),

                    None => {
                        let class = recv.get_class(vm).get();

                        return Effect::Unwind(VmError::NoSuchField {
                            span: dispatch_span,
                            class_span: class.name.span(),
                            class_name: class.name.value.clone(),
                            field_name: sym.get().as_str().to_owned(),
                        });
                    }
                }
            }

            Primitive::StringConcatenate => {
                let [lhs, rhs] = args.try_into().unwrap();
                let lhs = ok_or_unwind!(lhs.downcast_or_err::<tag::String>(arg_spans[0]));
                let rhs = ok_or_unwind!(rhs.downcast_or_err::<tag::String>(arg_spans[1]));

                Effect::None(vm.make_string(lhs.get().concat(rhs.get())).into_value())
            }

            Primitive::StringAsSymbol => {
                let [recv] = args.try_into().unwrap();
                let recv = ok_or_unwind!(recv.downcast_or_err::<tag::String>(arg_spans[0]));

                Effect::None(
                    vm.make_symbol(Symbol::from_string(
                        recv.get().to_string(),
                        arg_spans[0].into(),
                    ))
                    .into_value(),
                )
            }

            Primitive::StringLength => {
                let [recv] = args.try_into().unwrap();
                let recv = ok_or_unwind!(recv.downcast_or_err::<tag::String>(arg_spans[0]));

                Effect::None(vm.make_int(recv.get().char_count() as i64).into_value())
            }

            Primitive::StringIsWhitespace => {
                let [recv] = args.try_into().unwrap();
                let recv = ok_or_unwind!(recv.downcast_or_err::<tag::String>(arg_spans[0]));

                Effect::None(
                    vm.make_boolean(recv.get().chars().all(char::is_whitespace))
                        .into_value(),
                )
            }

            Primitive::StringIsLetters => {
                let [recv] = args.try_into().unwrap();
                let recv = ok_or_unwind!(recv.downcast_or_err::<tag::String>(arg_spans[0]));

                Effect::None(
                    vm.make_boolean(recv.get().chars().all(char::is_alphabetic))
                        .into_value(),
                )
            }

            Primitive::StringIsDigits => {
                let [recv] = args.try_into().unwrap();
                let recv = ok_or_unwind!(recv.downcast_or_err::<tag::String>(arg_spans[0]));

                Effect::None(
                    vm.make_boolean(recv.get().chars().all(char::is_numeric))
                        .into_value(),
                )
            }

            Primitive::StringEq => {
                let [lhs, rhs] = args.try_into().unwrap();
                let lhs = ok_or_unwind!(lhs.downcast_or_err::<tag::String>(arg_spans[0]));
                let rhs = ok_or_unwind!(rhs.downcast_or_err::<tag::String>(arg_spans[1]));

                Effect::None(vm.make_boolean(lhs.get() == rhs.get()).into_value())
            }

            Primitive::StringPrimSubstringFromTo => {
                let [recv, from, to] = args.try_into().unwrap();
                let recv = ok_or_unwind!(recv.downcast_or_err::<tag::String>(arg_spans[0]));
                let from = ok_or_unwind!(from.downcast_or_err::<tag::Int>(arg_spans[1])).get();
                let to = ok_or_unwind!(to.downcast_or_err::<tag::Int>(arg_spans[2])).get();

                let from =
                    ok_or_unwind!(check_arr_idx(arg_spans[1], recv.get().char_count(), from));
                let to = ok_or_unwind!(check_arr_idx(arg_spans[2], recv.get().char_count(), to));

                match recv.get().substr(from..=to) {
                    Ok(s) => Effect::None(vm.make_string(s).into_value()),

                    Err(SubstrError::StartGtEnd) => Effect::Unwind(VmError::StartGtEnd {
                        span: dispatch_span,
                        start: from,
                        end: to,
                    }),

                    Err(SubstrError::StartOutOfBounds | SubstrError::EndOutOfBounds) => {
                        unreachable!("check_arr_idx should have already checked this");
                    }
                }
            }

            Primitive::SystemGlobal => {
                let [_, name] = args.try_into().unwrap();
                let name = ok_or_unwind!(name.downcast_or_err::<tag::Symbol>(arg_spans[1]));

                match vm.get_global(name.get().as_str()) {
                    Some(value) => Effect::None(value.clone()),

                    None => Effect::Unwind(VmError::UndefinedName {
                        span: arg_spans[1],
                        name: name.get().as_str().to_owned(),
                    }),
                }
            }

            Primitive::SystemGlobalPut => {
                let [recv, name, value] = args.try_into().unwrap();
                let name = ok_or_unwind!(name.downcast_or_err::<tag::Symbol>(arg_spans[1]));

                vm.set_global(name.get().as_str().to_owned(), value);

                Effect::None(recv)
            }

            Primitive::SystemHasGlobal => {
                let [_, name] = args.try_into().unwrap();
                let name = ok_or_unwind!(name.downcast_or_err::<tag::Symbol>(arg_spans[1]));

                Effect::None(
                    vm.make_boolean(vm.get_global(name.get().as_str()).is_some())
                        .into_value(),
                )
            }

            Primitive::SystemLoadFile => todo!(),
            Primitive::SystemLoad => todo!(),

            Primitive::SystemExit => {
                let [_, code] = args.try_into().unwrap();
                let code = ok_or_unwind!(code.downcast_or_err::<tag::Int>(arg_spans[1])).get();

                Effect::Unwind(VmError::Exited {
                    span: dispatch_span,
                    code,
                })
            }

            Primitive::SystemPrintString => {
                let [recv, msg] = args.try_into().unwrap();

                match msg.downcast::<tag::Symbol>() {
                    Ok(sym) => vm.print(sym.get().as_str()),
                    Err(msg) => vm.print(
                        ok_or_unwind!(msg.downcast_or_err::<tag::String>(arg_spans[1])).get(),
                    ),
                }

                Effect::None(recv)
            }

            Primitive::SystemPrintNewline => {
                let [recv] = args.try_into().unwrap();
                vm.print('\n');

                Effect::None(recv)
            }

            Primitive::SystemErrorPrintln => {
                let [recv, msg] = args.try_into().unwrap();
                let msg = ok_or_unwind!(msg.downcast_or_err::<tag::String>(arg_spans[1]));

                vm.eprint(format_args!("{}\n", msg.get()));

                Effect::None(recv)
            }

            Primitive::SystemErrorPrint => {
                let [recv, msg] = args.try_into().unwrap();
                let msg = ok_or_unwind!(msg.downcast_or_err::<tag::String>(arg_spans[1]));

                vm.eprint(msg.get());

                Effect::None(recv)
            }

            Primitive::SystemPrintStackTrace => todo!(),
            Primitive::SystemTime => todo!(),

            Primitive::SystemTicks => {
                let [_] = args.try_into().unwrap();

                Effect::None(vm.make_int(vm.ticks().as_micros() as i64).into_value())
            }

            Primitive::SystemFullGC => {
                let [recv] = args.try_into().unwrap();

                vm.full_gc();

                Effect::None(recv)
            }
        }
    }
}
