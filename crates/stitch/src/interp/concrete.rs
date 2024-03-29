use std::ffi::{OsStr, OsString};
use std::fs::File;
use std::io::{ErrorKind, Read, Write};
use std::ops::{Bound, Neg};
use std::rc::Rc;
use std::{array, str};

use anyhow::{anyhow, bail, ensure, Context, Result};
use log::{debug, info, trace, warn};
use regex::Regex;
use slotmap::{Key, SecondaryMap};
use strum::VariantArray;

use crate::ast::expr::{format_value, Never, ValueAttrs};
use crate::ast::ty::{ElemType, FuncType};
use crate::ast::{
    ConstExpr, Export, ExportDef, FuncId, GlobalDef, GlobalId, IntrinsicDecl, MemoryDef, TableDef, PAGE_SIZE
};
use crate::cfg::{BinOp, BlockId, Call, Expr, LocalId, NulOp, Stmt, Terminator, TernOp, UnOp};
use crate::interp::{
    format_arg_list, FuncSpecPolicy, GlobalAttrs, InlineDisposition, SpecializedFunc,
};
use crate::util::float::{F32, F64};

use super::{GlobalValue, Interpreter};

const STACK_POINTER_NAME: &str = "__stack_pointer";

type Ptr<E = i32> = crate::ast::expr::Ptr<E>;
type Value<E = i32> = crate::ast::expr::Value<E>;

#[derive(Debug, Clone, Copy)]
enum Task<'a> {
    Expr(&'a Expr),
    Call(&'a Call),
    Exprs(&'a [Expr]),
}

impl<'a> From<&'a Expr> for Task<'a> {
    fn from(expr: &'a Expr) -> Self {
        Self::Expr(expr)
    }
}

impl<'a> From<&'a Call> for Task<'a> {
    fn from(call: &'a Call) -> Self {
        Self::Call(call)
    }
}

impl<'a> From<&'a [Expr]> for Task<'a> {
    fn from(exprs: &'a [Expr]) -> Self {
        Self::Exprs(exprs)
    }
}

impl<'a, const N: usize> From<&'a Box<[Expr; N]>> for Task<'a> {
    fn from(exprs: &'a Box<[Expr; N]>) -> Self {
        exprs.as_slice().into()
    }
}

impl Value<i32> {
    pub fn into_concrete<T>(self) -> Value<T> {
        match self {
            Self::I32(value) => Value::I32(value),
            Self::I64(value) => Value::I64(value),
            Self::F32(value) => Value::F32(value),
            Self::F64(value) => Value::F64(value),
            Self::Ptr(_) => Value::I32(self.unwrap_i32()),
        }
    }
}

trait ValueExt {
    fn unwrap_i32(&self) -> i32;
    fn unwrap_u32(&self) -> u32;
    fn unwrap_i64(&self) -> i64;
    fn unwrap_u64(&self) -> u64;
    fn unwrap_f32(&self) -> F32;
    fn unwrap_f64(&self) -> F64;
}

impl ValueExt for Value {
    fn unwrap_i32(&self) -> i32 {
        match *self {
            Self::I32(value) => value,
            Self::Ptr(Ptr { base, offset, .. }) => base + offset,
            _ => panic!(),
        }
    }

    fn unwrap_u32(&self) -> u32 {
        self.unwrap_i32() as u32
    }

    fn unwrap_i64(&self) -> i64 {
        self.to_i64().unwrap()
    }

    fn unwrap_u64(&self) -> u64 {
        self.to_u64().unwrap()
    }

    fn unwrap_f32(&self) -> F32 {
        self.to_f32().unwrap()
    }

    fn unwrap_f64(&self) -> F64 {
        self.to_f64().unwrap()
    }
}

#[derive(Debug, Clone)]
struct Frame {
    locals: SecondaryMap<LocalId, (Value, ValueAttrs)>,
    stack: Vec<(Value, ValueAttrs)>,
    func_id: FuncId,
    block_id: BlockId,
    next_stmt: usize,
    ret_local_id: Option<LocalId>,
}

impl Frame {
    fn set_local(&mut self, local_id: LocalId, value: (Value, ValueAttrs)) {
        trace!(
            "frame.locals[{local_id:?}] <- {}",
            format_value(&value.0, value.1)
        );
        self.locals[local_id] = value;
    }

    fn jump(&mut self, block_id: BlockId) {
        self.block_id = block_id;
        self.next_stmt = 0;
    }
}

impl Interpreter<'_> {
    pub fn interpret(
        &mut self,
        func_id: FuncId,
        args: Vec<(Value, ValueAttrs)>,
    ) -> Result<Option<(Value, ValueAttrs)>> {
        let ref mut frames = vec![];
        self.push_frame(frames, func_id, args, None)?;

        'frame: while let Some(frame) = frames.last_mut() {
            let func = Rc::clone(&self.cfgs[frame.func_id]);
            let block = &func.blocks[frame.block_id];

            trace!(
                "func {:?}, block {:?}, stmt {}/{}: {}",
                frame.func_id,
                frame.block_id,
                frame.next_stmt,
                block.body.len(),
                match block.body.get(frame.next_stmt) {
                    Some(stmt) => stmt as &dyn std::fmt::Display,
                    None => &block.term as &dyn std::fmt::Display,
                },
            );

            let frame = frames.last_mut().unwrap();
            let next_stmt = frame.next_stmt;
            frame.next_stmt += 1;

            match block.body.get(next_stmt) {
                Some(stmt) => match stmt {
                    Stmt::Nop => {}
                    Stmt::Drop(expr) => self.eval(frames, expr)?,
                    Stmt::LocalSet(_, expr) => self.eval(frames, expr)?,
                    Stmt::GlobalSet(_, expr) => self.eval(frames, expr)?,
                    Stmt::Store(_, _, exprs) => self.eval(frames, exprs)?,
                    Stmt::Call(call) => self.eval(frames, call)?,
                    Stmt::MemoryCopy { args, .. } => self.eval(frames, args)?,
                    Stmt::MemoryFill(_, exprs) => self.eval(frames, exprs)?,
                },

                None => match &block.term {
                    Terminator::Trap | Terminator::Br(_) | Terminator::Return(None) => {}
                    Terminator::If(expr, _) => self.eval(frames, expr)?,
                    Terminator::Switch(expr, _) => self.eval(frames, expr)?,
                    Terminator::Return(Some(expr)) => self.eval(frames, expr)?,
                },
            }

            let frame = frames.last_mut().unwrap();

            if let Some(stmt) = block.body.get(next_stmt) {
                match *stmt {
                    Stmt::Nop => {}

                    Stmt::Drop(_) => {
                        frame.stack.pop().unwrap();
                    }

                    Stmt::LocalSet(local_id, _) => {
                        let value = frame.stack.pop().unwrap();
                        frame.set_local(local_id, value);
                    }

                    Stmt::GlobalSet(global_id, _) => {
                        let value = frame.stack.pop().unwrap();
                        self.eval_global_set(global_id, value);
                    }

                    Stmt::Store(mem_arg, store, _) => {
                        let (value, attrs) = frame.stack.pop().unwrap();
                        let base_addr = frame.stack.pop().unwrap().0.unwrap_u32();
                        let start = (base_addr + mem_arg.offset) as usize;
                        let range = start
                            ..start
                                .checked_add(store.dst_size())
                                .context("not enough memory")?;
                        let bytes = self.module.get_mem_mut(mem_arg.mem_id, range)?;
                        store.store(bytes, value.into_concrete());
                        trace!(
                            "{store} {} to 0x{base_addr:x} + {offset}: [0x{start:x}..0x{end:x}] <- [{bytes}]",
                            format_value(&value, attrs),
                            offset = mem_arg.offset,
                            end = start + store.dst_size(),
                            bytes = bytes
                                .iter()
                                .map(|b| format!("{b:02x} "))
                                .collect::<String>()
                                .trim_end(),
                        );
                    }

                    Stmt::Call(..) => continue 'frame,

                    Stmt::MemoryCopy {
                        dst_mem_id,
                        src_mem_id,
                        args: _,
                    } => {
                        let len = frame.stack.pop().unwrap().0.unwrap_u32() as usize;
                        let src = frame.stack.pop().unwrap().0.unwrap_u32() as usize;
                        let dst = frame.stack.pop().unwrap().0.unwrap_u32() as usize;
                        self.module.get_mem_mut(
                            dst_mem_id,
                            dst..dst
                                .checked_add(len)
                                .context("the destination range is too large")?,
                        )?;
                        self.module.get_mem(
                            src_mem_id,
                            src..src
                                .checked_add(len)
                                .context("the source range is too large")?,
                        )?;

                        let mut copy = |i| {
                            let src = src + i;
                            let dst = dst + i;
                            let byte = self.module.get_mem(src_mem_id, src..src + 1).unwrap()[0];
                            self.module.get_mem_mut(dst_mem_id, dst..dst + 1).unwrap()[0] = byte;
                        };

                        if dst <= src {
                            for i in 0..len {
                                copy(i);
                            }
                        } else {
                            for i in (0..len).rev() {
                                copy(i);
                            }
                        }
                    }

                    Stmt::MemoryFill(mem_id, _) => {
                        let len = frame.stack.pop().unwrap().0.unwrap_u32() as usize;
                        let byte = frame.stack.pop().unwrap().0.unwrap_u32() as u8;
                        let addr = frame.stack.pop().unwrap().0.unwrap_u32() as usize;

                        self.module
                            .get_mem_mut(
                                mem_id,
                                addr..addr
                                    .checked_add(len)
                                    .context("the destination range is too large")?,
                            )?
                            .fill(byte);
                    }
                }
            } else {
                match block.term {
                    Terminator::Trap => bail!("aborting execution due to an explicit trap"),
                    Terminator::Br(block_id) => frame.jump(block_id),

                    Terminator::If(_, [then_block_id, else_block_id]) => {
                        if frame.stack.pop().unwrap().0.unwrap_i32() == 0 {
                            frame.jump(else_block_id);
                        } else {
                            frame.jump(then_block_id);
                        }
                    }

                    Terminator::Switch(_, ref block_ids) => {
                        let (&default_block_id, block_ids) = block_ids.split_last().unwrap();

                        match block_ids.get(frame.stack.pop().unwrap().0.unwrap_u32() as usize) {
                            Some(&block_id) => frame.jump(block_id),
                            None => frame.jump(default_block_id),
                        }
                    }

                    Terminator::Return(_) => {
                        let mut frame = frames.pop().unwrap();
                        let ret_local_id = frame.ret_local_id;
                        let value = ret_local_id.map(|_| frame.stack.pop().unwrap());

                        trace!(
                            "returning from {:?} (name = {:?}) with {value:?}",
                            frame.func_id,
                            self.module.funcs[frame.func_id].name(),
                        );

                        match frames.last_mut() {
                            Some(frame) => {
                                if let Some(ret_local_id) = ret_local_id {
                                    frame.set_local(ret_local_id, value.unwrap());
                                }
                            }

                            None => {
                                self.flush_globals();
                                return Ok(value);
                            }
                        }
                    }
                }
            }
        }

        unreachable!("reached an empty frame stack without a return");
    }

    fn check_args(&self, args: &[(Value, ValueAttrs)], func_ty: &FuncType) -> Result<()> {
        ensure!(
            args.len() == func_ty.params.len(),
            "invalid number of arguments for function: expected {}, got {}",
            args.len(),
            func_ty.params.len()
        );
        ensure!(
            args.iter()
                .zip(&func_ty.params)
                .all(|((value, _), param_ty)| value.val_ty() == *param_ty),
            "invalid arguments for function: expected {}, got {}",
            format_arg_list(func_ty.params.iter().cloned().map(Some)),
            format_arg_list(args.iter().map(|(value, _)| Some(value.val_ty()))),
        );

        Ok(())
    }

    fn push_frame(
        &mut self,
        frames: &mut Vec<Frame>,
        func_id: FuncId,
        args: Vec<(Value, ValueAttrs)>,
        ret_local_id: Option<LocalId>,
    ) -> Result<()> {
        let func = &self.module.funcs[func_id];
        self.check_args(&args, func.ty())?;
        ensure!(!func.is_import(), "cannot interpret imported function");

        if matches!(
            self.spec_funcs
                .get(func_id)
                .and_then(|spec_sig| self.spec_sigs.get(spec_sig)),
            Some(SpecializedFunc::Pending(_))
        ) {
            bail!("cannot interpret a function while it is being specialized");
        }

        let func = self.get_cfg(func_id).unwrap();
        let mut locals: SecondaryMap<LocalId, (Value, ValueAttrs)> = func
            .locals
            .iter()
            .map(|(local_id, val_ty)| {
                (
                    local_id,
                    (Value::default_for(val_ty), ValueAttrs::default()),
                )
            })
            .collect();

        for (&local_id, arg) in func.params.iter().zip(args) {
            locals[local_id] = arg;
        }

        let mut frame = Frame {
            locals,
            stack: vec![],
            func_id,
            block_id: Default::default(),
            next_stmt: Default::default(),
            ret_local_id,
        };

        frame.jump(func.entry);
        trace!(
            "pushing frame #{}: {func_id:?} (name = {:?})",
            frames.len() + 1,
            self.module.funcs[func_id].name(),
        );
        frames.push(frame);

        Ok(())
    }

    fn flush_globals(&mut self) {
        for (global_id, &global_value) in &self.globals {
            let (value, attrs) = match global_value {
                GlobalValue::Import | GlobalValue::GlobalGet => continue,
                GlobalValue::Value(value, attrs) => (value, attrs)
            };

            self.module.globals[global_id].def = GlobalDef::Value(ConstExpr::Value(
                value.into_concrete(),
                attrs,
            ));
        }
    }

    fn eval<'a>(&mut self, frames: &mut Vec<Frame>, task: impl Into<Task<'a>>) -> Result<()> {
        let mut stack = vec![(0, task.into())];

        while let Some(&mut (ref mut subexpr_idx, task)) = stack.last_mut() {
            let subexpr = match task {
                Task::Expr(expr) => expr.nth_subexpr(*subexpr_idx),
                Task::Call(call) => call.nth_subexpr(*subexpr_idx),
                Task::Exprs(exprs) => exprs.get(*subexpr_idx),
            };

            if let Some(subexpr) = subexpr {
                *subexpr_idx += 1;

                stack.push((0, Task::Expr(subexpr)));
                continue;
            }

            match task {
                Task::Expr(expr) => match *expr {
                    Expr::Value(value, attrs) => self.eval_value(frames, value, attrs)?,
                    Expr::Nullary(op) => self.eval_nul_op(frames, op)?,
                    Expr::Unary(op, _) => self.eval_un_op(frames, op)?,
                    Expr::Binary(op, _) => self.eval_bin_op(frames, op)?,
                    Expr::Ternary(op, _) => self.eval_tern_op(frames, op)?,
                },

                Task::Call(call) => self.eval_call(frames, call)?,

                Task::Exprs(_) => {}
            }

            stack.pop();
        }

        Ok(())
    }

    fn eval_value(
        &mut self,
        frames: &mut Vec<Frame>,
        value: Value<Never>,
        attrs: ValueAttrs,
    ) -> Result<()> {
        let frame = frames.last_mut().unwrap();
        frame.stack.push((value.lift_ptr(), attrs));

        Ok(())
    }

    fn eval_nul_op(&mut self, frames: &mut Vec<Frame>, op: NulOp) -> Result<()> {
        let frame = frames.last_mut().unwrap();

        match op {
            NulOp::LocalGet(local_id) => {
                let value = frame.locals[local_id];
                frame.stack.push(value);
                trace!(
                    "frame.locals[{local_id:?}] -> {}",
                    format_value(&value.0, value.1)
                );
            }

            NulOp::GlobalGet(global_id) => frame.stack.push(self.eval_global_get(global_id)?),

            NulOp::MemorySize(mem_id) => match &self.module.mems[mem_id].def {
                MemoryDef::Import(_) => {
                    bail!("cannot get the size of an imported memory")
                }
                MemoryDef::Bytes(bytes) => frame.stack.push((
                    Value::I32(bytes.len().div_ceil(PAGE_SIZE) as i32),
                    Default::default(),
                )),
            },
        }

        Ok(())
    }

    fn eval_un_op(&mut self, frames: &mut Vec<Frame>, op: UnOp) -> Result<()> {
        let frame = frames.last_mut().unwrap();
        let (arg, arg_attrs) = frame.stack.pop().unwrap();

        let result = match op {
            UnOp::I32Clz => (
                Value::I32(arg.unwrap_u32().leading_zeros() as i32),
                Default::default(),
            ),
            UnOp::I32Ctz => (
                Value::I32(arg.unwrap_u32().trailing_zeros() as i32),
                Default::default(),
            ),
            UnOp::I32Popcnt => (
                Value::I32(arg.unwrap_u32().count_ones() as i32),
                Default::default(),
            ),

            UnOp::I64Clz => (
                Value::I64(arg.unwrap_i64().leading_zeros() as i64),
                Default::default(),
            ),
            UnOp::I64Ctz => (
                Value::I64(arg.unwrap_i64().trailing_zeros() as i64),
                Default::default(),
            ),
            UnOp::I64Popcnt => (
                Value::I64(arg.unwrap_i64().count_ones() as i64),
                Default::default(),
            ),

            UnOp::F32Abs => (Value::F32(arg.unwrap_f32().abs()), arg_attrs),
            UnOp::F32Neg => (Value::F32(arg.unwrap_f32().neg()), arg_attrs),
            UnOp::F32Sqrt => (Value::F32(arg.unwrap_f32().sqrt()), arg_attrs),
            UnOp::F32Ceil => (Value::F32(arg.unwrap_f32().ceil()), arg_attrs),
            UnOp::F32Floor => (Value::F32(arg.unwrap_f32().floor()), arg_attrs),
            UnOp::F32Trunc => (Value::F32(arg.unwrap_f32().trunc()), arg_attrs),
            UnOp::F32Nearest => (Value::F32(arg.unwrap_f32().nearest()), arg_attrs),

            UnOp::F64Abs => (Value::F64(arg.unwrap_f64().abs()), arg_attrs),
            UnOp::F64Neg => (Value::F64(arg.unwrap_f64().neg()), arg_attrs),
            UnOp::F64Sqrt => (Value::F64(arg.unwrap_f64().sqrt()), arg_attrs),
            UnOp::F64Ceil => (Value::F64(arg.unwrap_f64().ceil()), arg_attrs),
            UnOp::F64Floor => (Value::F64(arg.unwrap_f64().floor()), arg_attrs),
            UnOp::F64Trunc => (Value::F64(arg.unwrap_f64().trunc()), arg_attrs),
            UnOp::F64Nearest => (Value::F64(arg.unwrap_f64().nearest()), arg_attrs),

            UnOp::I32Eqz => (
                Value::I32((arg.unwrap_i32() == 0) as i32),
                Default::default(),
            ),
            UnOp::I64Eqz => (
                Value::I32((arg.unwrap_i64() == 0) as i32),
                Default::default(),
            ),

            UnOp::I32WrapI64 => (Value::I32(arg.unwrap_i64() as i32), arg_attrs),

            UnOp::I64ExtendI32S => (Value::I64(arg.unwrap_i32() as i64), arg_attrs),
            UnOp::I64ExtendI32U => (Value::I64(arg.unwrap_u32() as i64), arg_attrs),

            UnOp::I32TruncF32S => (Value::I32(arg.unwrap_f32().trunc_i32()), arg_attrs),
            UnOp::I32TruncF32U => (Value::I32(arg.unwrap_f32().trunc_u32() as i32), arg_attrs),
            UnOp::I32TruncF64S => (Value::I32(arg.unwrap_f64().trunc_i32()), arg_attrs),
            UnOp::I32TruncF64U => (Value::I32(arg.unwrap_f64().trunc_u32() as i32), arg_attrs),

            UnOp::I64TruncF32S => (Value::I64(arg.unwrap_f32().trunc_i64()), arg_attrs),
            UnOp::I64TruncF32U => (Value::I64(arg.unwrap_f32().trunc_u64() as i64), arg_attrs),
            UnOp::I64TruncF64S => (Value::I64(arg.unwrap_f64().trunc_i64()), arg_attrs),
            UnOp::I64TruncF64U => (Value::I64(arg.unwrap_f64().trunc_u64() as i64), arg_attrs),

            UnOp::F32DemoteF64 => (Value::F32(arg.unwrap_f64().demote()), arg_attrs),
            UnOp::F64PromoteF32 => (Value::F64(arg.unwrap_f32().promote()), arg_attrs),

            UnOp::F32ConvertI32S => (Value::F32((arg.unwrap_i32() as f32).into()), arg_attrs),
            UnOp::F32ConvertI32U => (Value::F32((arg.unwrap_u32() as f32).into()), arg_attrs),
            UnOp::F32ConvertI64S => (Value::F32((arg.unwrap_i64() as f32).into()), arg_attrs),
            UnOp::F32ConvertI64U => (Value::F32((arg.unwrap_u64() as f32).into()), arg_attrs),

            UnOp::F64ConvertI32S => (Value::F64((arg.unwrap_i32() as f64).into()), arg_attrs),
            UnOp::F64ConvertI32U => (Value::F64((arg.unwrap_u32() as f64).into()), arg_attrs),
            UnOp::F64ConvertI64S => (Value::F64((arg.unwrap_i64() as f64).into()), arg_attrs),
            UnOp::F64ConvertI64U => (Value::F64((arg.unwrap_u64() as f64).into()), arg_attrs),

            UnOp::F32ReinterpretI32 => (Value::F32(F32::from_bits(arg.unwrap_u32())), arg_attrs),
            UnOp::F64ReinterpretI64 => (Value::F64(F64::from_bits(arg.unwrap_u64())), arg_attrs),
            UnOp::I32ReinterpretF32 => (
                Value::I32(arg.to_f32().unwrap().to_bits() as i32),
                arg_attrs,
            ),
            UnOp::I64ReinterpretF64 => (
                Value::I64(arg.to_f64().unwrap().to_bits() as i64),
                arg_attrs,
            ),

            UnOp::I32Extend8S => (Value::I32(arg.unwrap_u32() as i8 as i32), arg_attrs),
            UnOp::I32Extend16S => (Value::I32(arg.unwrap_u32() as i64 as i32), arg_attrs),

            UnOp::I64Extend8S => (Value::I64(arg.unwrap_u64() as i8 as i64), arg_attrs),
            UnOp::I64Extend16S => (Value::I64(arg.unwrap_u64() as i16 as i64), arg_attrs),
            UnOp::I64Extend32S => (Value::I64(arg.unwrap_u64() as i32 as i64), arg_attrs),

            UnOp::Load(mem_arg, load) => {
                let base_addr = arg.unwrap_u32();
                let start = (base_addr + mem_arg.offset) as usize;
                let range = start..start + load.src_size();
                let bytes = self.module.get_mem(mem_arg.mem_id, range)?;
                let value = load.load(bytes);
                let attrs = arg_attrs.deref_attrs();
                trace!(
                    "{load} {} from 0x{base_addr:x} + {offset}: [0x{start:x}..0x{end:x}] -> [{bytes}]",
                    format_value(&value, attrs),
                    offset = mem_arg.offset,
                    end = start + load.src_size(),
                    bytes = bytes
                        .iter()
                        .map(|b| format!("{b:02x} "))
                        .collect::<String>()
                        .trim_end(),
                );

                (value.lift_ptr(), attrs)
            }

            UnOp::MemoryGrow(mem_id) => {
                let mem = &mut self.module.mems[mem_id];

                match mem.def {
                    MemoryDef::Import(_) => {
                        bail!("cannot resize an imported memory")
                    }

                    MemoryDef::Bytes(ref mut bytes) => {
                        let prev_pages = bytes.len().div_ceil(PAGE_SIZE);
                        let increment = arg.unwrap_u32() as usize;
                        let new_pages = prev_pages + increment;

                        match mem.ty.limits.max {
                            Some(max) if new_pages > max as usize => {
                                (Value::I32(-1), Default::default())
                            }

                            _ => {
                                if let Ok(new_pages_u32) = u32::try_from(new_pages) {
                                    mem.ty.limits.min = mem.ty.limits.min.max(new_pages_u32);
                                    bytes.resize(new_pages * PAGE_SIZE, 0);

                                    (Value::I32(prev_pages as i32), Default::default())
                                } else {
                                    (Value::I32(-1), Default::default())
                                }
                            }
                        }
                    }
                }
            }
        };

        frame.stack.push(result);

        Ok(())
    }

    fn eval_bin_op(&mut self, frames: &mut Vec<Frame>, op: BinOp) -> Result<()> {
        let frame = frames.last_mut().unwrap();
        let (rhs, rhs_attrs) = frame.stack.pop().unwrap();
        let (lhs, lhs_attrs) = frame.stack.pop().unwrap();
        let meet_attrs = lhs_attrs.meet(&rhs_attrs);

        frame.stack.push(match op {
            BinOp::I32Add => (
                match (lhs, rhs) {
                    (Value::I32(offset), Value::Ptr(ptr))
                    | (Value::Ptr(ptr), Value::I32(offset)) => Value::Ptr(ptr.add(offset)),

                    _ => Value::I32(lhs.unwrap_i32().wrapping_add(rhs.unwrap_i32())),
                },
                lhs_attrs.addsub_attrs(&rhs_attrs),
            ),

            BinOp::I32Sub => (
                match (lhs, rhs) {
                    (Value::I32(offset), Value::Ptr(ptr))
                    | (Value::Ptr(ptr), Value::I32(offset)) => Value::Ptr(ptr.sub(offset)),

                    _ => Value::I32(lhs.unwrap_i32().wrapping_sub(rhs.unwrap_i32())),
                },
                lhs_attrs.addsub_attrs(&rhs_attrs),
            ),

            BinOp::I32Mul => (
                Value::I32(lhs.unwrap_i32().wrapping_mul(rhs.unwrap_i32())),
                meet_attrs,
            ),
            BinOp::I32DivS => match (lhs.unwrap_i32(), rhs.unwrap_i32()) {
                (_, 0) => bail!("division by zero"),
                (lhs, -1) if lhs == i32::MIN => {
                    bail!("dividing i32::MIN by -1 resulted in an overflow")
                }
                (lhs, rhs) => (Value::I32(lhs / rhs), meet_attrs),
            },
            BinOp::I32DivU => match (lhs.unwrap_u32(), rhs.unwrap_u32()) {
                (_, 0) => bail!("division by zero"),
                (lhs, rhs) => (Value::I32((lhs / rhs) as i32), meet_attrs),
            },
            BinOp::I32RemS => match (lhs.unwrap_i32(), rhs.unwrap_i32()) {
                (_, 0) => bail!("taking the remainder of dividing by zero"),
                (lhs, rhs) => (Value::I32(lhs % rhs), meet_attrs),
            },
            BinOp::I32RemU => match (lhs.unwrap_u32(), rhs.unwrap_u32()) {
                (_, 0) => bail!("taking the remainder of dividing by zero"),
                (lhs, rhs) => (Value::I32((lhs % rhs) as i32), meet_attrs),
            },
            BinOp::I32And => (Value::I32(lhs.unwrap_i32() & rhs.unwrap_i32()), meet_attrs),
            BinOp::I32Or => (Value::I32(lhs.unwrap_i32() | rhs.unwrap_i32()), meet_attrs),
            BinOp::I32Xor => (Value::I32(lhs.unwrap_i32() ^ rhs.unwrap_i32()), meet_attrs),
            BinOp::I32Shl => (
                Value::I32(lhs.unwrap_i32().wrapping_shl(rhs.unwrap_u32())),
                meet_attrs,
            ),
            BinOp::I32ShrS => (
                Value::I32(lhs.unwrap_i32().wrapping_shr(rhs.unwrap_u32())),
                meet_attrs,
            ),
            BinOp::I32ShrU => (
                Value::I32(lhs.unwrap_u32().wrapping_shr(rhs.unwrap_u32()) as i32),
                meet_attrs,
            ),
            BinOp::I32Rotl => (
                Value::I32(lhs.unwrap_i32().rotate_left(rhs.unwrap_u32())),
                meet_attrs,
            ),
            BinOp::I32Rotr => (
                Value::I32(lhs.unwrap_i32().rotate_right(rhs.unwrap_u32())),
                meet_attrs,
            ),

            BinOp::I64Add => (
                Value::I64(lhs.unwrap_i64().wrapping_add(rhs.unwrap_i64())),
                meet_attrs,
            ),
            BinOp::I64Sub => (
                Value::I64(lhs.unwrap_i64().wrapping_sub(rhs.unwrap_i64())),
                meet_attrs,
            ),
            BinOp::I64Mul => (
                Value::I64(lhs.unwrap_i64().wrapping_mul(rhs.unwrap_i64())),
                meet_attrs,
            ),
            BinOp::I64DivS => match (lhs.unwrap_i64(), rhs.unwrap_i64()) {
                (_, 0) => bail!("division by zero"),
                (lhs, -1) if lhs == i64::MIN => {
                    bail!("dividing i64::MIN by -1 resulted in an overflow")
                }
                (lhs, rhs) => (Value::I64(lhs / rhs), meet_attrs),
            },
            BinOp::I64DivU => match (lhs.unwrap_u64(), rhs.unwrap_u64()) {
                (_, 0) => bail!("division by zero"),
                (lhs, rhs) => (Value::I64((lhs / rhs) as i64), meet_attrs),
            },
            BinOp::I64RemS => match (lhs.unwrap_i64(), rhs.unwrap_i64()) {
                (_, 0) => bail!("taking the remainder of dividing by zero"),
                (lhs, rhs) => (Value::I64(lhs % rhs), meet_attrs),
            },
            BinOp::I64RemU => match (lhs.unwrap_u64(), rhs.unwrap_u64()) {
                (_, 0) => bail!("taking the remainder of dividing by zero"),
                (lhs, rhs) => (Value::I64((lhs % rhs) as i64), meet_attrs),
            },
            BinOp::I64And => (Value::I64(lhs.unwrap_i64() & rhs.unwrap_i64()), meet_attrs),
            BinOp::I64Or => (Value::I64(lhs.unwrap_i64() | rhs.unwrap_i64()), meet_attrs),
            BinOp::I64Xor => (Value::I64(lhs.unwrap_i64() ^ rhs.unwrap_i64()), meet_attrs),
            BinOp::I64Shl => (
                Value::I64(lhs.unwrap_i64().wrapping_shl(rhs.unwrap_u64() as u32)),
                meet_attrs,
            ),
            BinOp::I64ShrS => (
                Value::I64(lhs.unwrap_i64().wrapping_shr(rhs.unwrap_u64() as u32)),
                meet_attrs,
            ),
            BinOp::I64ShrU => (
                Value::I64(lhs.unwrap_u64().wrapping_shr(rhs.unwrap_u64() as u32) as i64),
                meet_attrs,
            ),
            BinOp::I64Rotl => (
                Value::I64(lhs.unwrap_i64().rotate_left(rhs.unwrap_u64() as u32)),
                meet_attrs,
            ),
            BinOp::I64Rotr => (
                Value::I64(lhs.unwrap_i64().rotate_right(rhs.unwrap_u64() as u32)),
                meet_attrs,
            ),

            BinOp::F32Add => (Value::F32(lhs.unwrap_f32() + rhs.unwrap_f32()), meet_attrs),
            BinOp::F32Sub => (Value::F32(lhs.unwrap_f32() - rhs.unwrap_f32()), meet_attrs),
            BinOp::F32Mul => (Value::F32(lhs.unwrap_f32() * rhs.unwrap_f32()), meet_attrs),
            BinOp::F32Div => (Value::F32(lhs.unwrap_f32() / rhs.unwrap_f32()), meet_attrs),
            BinOp::F32Min => (
                Value::F32(lhs.unwrap_f32().min(rhs.unwrap_f32())),
                meet_attrs,
            ),
            BinOp::F32Max => (
                Value::F32(lhs.unwrap_f32().max(rhs.unwrap_f32())),
                meet_attrs,
            ),
            BinOp::F32Copysign => (
                Value::F32(lhs.unwrap_f32().copysign(rhs.unwrap_f32())),
                meet_attrs,
            ),

            BinOp::F64Add => (Value::F64(lhs.unwrap_f64() + rhs.unwrap_f64()), meet_attrs),
            BinOp::F64Sub => (Value::F64(lhs.unwrap_f64() - rhs.unwrap_f64()), meet_attrs),
            BinOp::F64Mul => (Value::F64(lhs.unwrap_f64() * rhs.unwrap_f64()), meet_attrs),
            BinOp::F64Div => (Value::F64(lhs.unwrap_f64() / rhs.unwrap_f64()), meet_attrs),
            BinOp::F64Min => (
                Value::F64(lhs.unwrap_f64().min(rhs.unwrap_f64())),
                meet_attrs,
            ),
            BinOp::F64Max => (
                Value::F64(lhs.unwrap_f64().max(rhs.unwrap_f64())),
                meet_attrs,
            ),
            BinOp::F64Copysign => (
                Value::F64(lhs.unwrap_f64().copysign(rhs.unwrap_f64())),
                meet_attrs,
            ),

            BinOp::I32Eq => (
                Value::I32((lhs.unwrap_i32() == rhs.unwrap_i32()) as i32),
                Default::default(),
            ),
            BinOp::I32Ne => (
                Value::I32((lhs.unwrap_i32() != rhs.unwrap_i32()) as i32),
                Default::default(),
            ),
            BinOp::I32LtS => (
                Value::I32((lhs.unwrap_i32() < rhs.unwrap_i32()) as i32),
                Default::default(),
            ),
            BinOp::I32LtU => (
                Value::I32((lhs.unwrap_u32() < rhs.unwrap_u32()) as i32),
                Default::default(),
            ),
            BinOp::I32GtS => (
                Value::I32((lhs.unwrap_i32() > rhs.unwrap_i32()) as i32),
                Default::default(),
            ),
            BinOp::I32GtU => (
                Value::I32((lhs.unwrap_u32() > rhs.unwrap_u32()) as i32),
                Default::default(),
            ),
            BinOp::I32LeS => (
                Value::I32((lhs.unwrap_i32() <= rhs.unwrap_i32()) as i32),
                Default::default(),
            ),
            BinOp::I32LeU => (
                Value::I32((lhs.unwrap_u32() <= rhs.unwrap_u32()) as i32),
                Default::default(),
            ),
            BinOp::I32GeS => (
                Value::I32((lhs.unwrap_i32() >= rhs.unwrap_i32()) as i32),
                Default::default(),
            ),
            BinOp::I32GeU => (
                Value::I32((lhs.unwrap_u32() >= rhs.unwrap_u32()) as i32),
                Default::default(),
            ),

            BinOp::I64Eq => (
                Value::I32((lhs.unwrap_i64() == rhs.unwrap_i64()) as i32),
                Default::default(),
            ),
            BinOp::I64Ne => (
                Value::I32((lhs.unwrap_i64() != rhs.unwrap_i64()) as i32),
                Default::default(),
            ),
            BinOp::I64LtS => (
                Value::I32((lhs.unwrap_i64() < rhs.unwrap_i64()) as i32),
                Default::default(),
            ),
            BinOp::I64LtU => (
                Value::I32((lhs.unwrap_u64() < rhs.unwrap_u64()) as i32),
                Default::default(),
            ),
            BinOp::I64GtS => (
                Value::I32((lhs.unwrap_i64() > rhs.unwrap_i64()) as i32),
                Default::default(),
            ),
            BinOp::I64GtU => (
                Value::I32((lhs.unwrap_u64() > rhs.unwrap_u64()) as i32),
                Default::default(),
            ),
            BinOp::I64LeS => (
                Value::I32((lhs.unwrap_i64() <= rhs.unwrap_i64()) as i32),
                Default::default(),
            ),
            BinOp::I64LeU => (
                Value::I32((lhs.unwrap_u64() <= rhs.unwrap_u64()) as i32),
                Default::default(),
            ),
            BinOp::I64GeS => (
                Value::I32((lhs.unwrap_i64() >= rhs.unwrap_i64()) as i32),
                Default::default(),
            ),
            BinOp::I64GeU => (
                Value::I32((lhs.unwrap_u64() >= rhs.unwrap_u64()) as i32),
                Default::default(),
            ),

            BinOp::F32Eq => (
                Value::I32((lhs.unwrap_f32() == rhs.unwrap_f32()) as i32),
                Default::default(),
            ),
            BinOp::F32Ne => (
                Value::I32((lhs.unwrap_f32() != rhs.unwrap_f32()) as i32),
                Default::default(),
            ),
            BinOp::F32Lt => (
                Value::I32((lhs.unwrap_f32() < rhs.unwrap_f32()) as i32),
                Default::default(),
            ),
            BinOp::F32Gt => (
                Value::I32((lhs.unwrap_f32() > rhs.unwrap_f32()) as i32),
                Default::default(),
            ),
            BinOp::F32Le => (
                Value::I32((lhs.unwrap_f32() <= rhs.unwrap_f32()) as i32),
                Default::default(),
            ),
            BinOp::F32Ge => (
                Value::I32((lhs.unwrap_f32() >= rhs.unwrap_f32()) as i32),
                Default::default(),
            ),

            BinOp::F64Eq => (
                Value::I32((lhs.unwrap_f64() == rhs.unwrap_f64()) as i32),
                Default::default(),
            ),
            BinOp::F64Ne => (
                Value::I32((lhs.unwrap_f64() != rhs.unwrap_f64()) as i32),
                Default::default(),
            ),
            BinOp::F64Lt => (
                Value::I32((lhs.unwrap_f64() < rhs.unwrap_f64()) as i32),
                Default::default(),
            ),
            BinOp::F64Gt => (
                Value::I32((lhs.unwrap_f64() > rhs.unwrap_f64()) as i32),
                Default::default(),
            ),
            BinOp::F64Le => (
                Value::I32((lhs.unwrap_f64() <= rhs.unwrap_f64()) as i32),
                Default::default(),
            ),
            BinOp::F64Ge => (
                Value::I32((lhs.unwrap_f64() >= rhs.unwrap_f64()) as i32),
                Default::default(),
            ),
        });

        Ok(())
    }

    fn eval_tern_op(&mut self, frames: &mut Vec<Frame>, op: TernOp) -> Result<()> {
        let frame = frames.last_mut().unwrap();
        let mut args = array::from_fn(|_| frame.stack.pop().unwrap());
        args.reverse();

        frame.stack.push(match op {
            TernOp::Select => {
                let [lhs, rhs, (condition, _)] = args;

                match condition.unwrap_i32() {
                    0 => rhs,
                    _ => lhs,
                }
            }
        });

        Ok(())
    }

    fn eval_call(&mut self, frames: &mut Vec<Frame>, call: &Call) -> Result<()> {
        let frame = frames.last_mut().unwrap();
        let mut args = frame
            .stack
            .drain(frame.stack.len() - call.subexpr_count()..)
            .collect::<Vec<_>>();

        let func_id = match *call {
            Call::Direct { func_id, .. } => func_id,

            Call::Indirect { table_id, .. } => match &self.module.tables[table_id].def {
                TableDef::Import(_) => {
                    bail!("cannot perform an indirect call into an entry of an imported table")
                }

                TableDef::Elems(elems) => {
                    let idx = args.pop().unwrap().0.unwrap_u32() as usize;

                    elems
                        .get(idx)
                        .with_context(|| {
                            anyhow!(
                            "an indirect call index ({idx}) is out of range for a table of size {}",
                            elems.len()
                        )
                        })?
                        .with_context(|| {
                            anyhow!("an indirect call's table entry (index {idx}) is uninitialized")
                        })?
                }
            },
        };

        if let Call::Indirect { ty_id, .. } = *call {
            let actual_func_ty = self.module.funcs[func_id].ty();
            let claimed_func_ty = self.module.types[ty_id].as_func();

            ensure!(
                actual_func_ty == claimed_func_ty,
                "the claimed type of an indirectly called function ({claimed_func_ty}) \
                does not match the actual type: {actual_func_ty}"
            );
        }

        let ret_local_id = call.ret_local_id();

        if let Some(intrinsic) = self.module.funcs[func_id].get_intrinsic(&self.module) {
            trace!("evaluating an intrinsic {intrinsic}");

            match intrinsic {
                IntrinsicDecl::ArgCount => self.eval_intr_arg_count(frames)?,
                IntrinsicDecl::ArgLen => self.eval_intr_arg_len(frames, args)?,
                IntrinsicDecl::ArgRead => self.eval_intr_arg_read(frames, args)?,
                IntrinsicDecl::Specialize => self.eval_intr_specialize(frames, args)?,
                IntrinsicDecl::Unknown => self.eval_intr_unknown(frames, func_id)?,
                IntrinsicDecl::ConstPtr => self.eval_intr_const_ptr(frames, args)?,
                IntrinsicDecl::SymbolicPtr => self.eval_intr_symbolic_ptr(frames, args)?,
                IntrinsicDecl::ConcretePtr => self.eval_intr_concrete_ptr(frames, args)?,
                IntrinsicDecl::PropagateLoad => self.eval_intr_propagate_load(frames, args)?,
                IntrinsicDecl::PrintValue => self.eval_intr_print_value(args)?,
                IntrinsicDecl::PrintStr => self.eval_intr_print_str(args)?,
                IntrinsicDecl::IsSpecializing => self.eval_intr_is_specializing(frames)?,
                IntrinsicDecl::Inline => self.eval_intr_inline(frames, args)?,
                IntrinsicDecl::NoInline => self.eval_intr_no_inline(frames, args)?,
                IntrinsicDecl::FileOpen => self.eval_intr_file_open(frames, args)?,
                IntrinsicDecl::FileRead => self.eval_intr_file_read(frames, args)?,
                IntrinsicDecl::FileClose => self.eval_intr_file_close(frames, args)?,
                IntrinsicDecl::FuncSpecPolicy => self.eval_intr_func_spec_policy(args)?,
                IntrinsicDecl::SymbolicStackPtr => self.eval_intr_symbolic_stack_ptr()?,
            }

            let frame = frames.last_mut().unwrap();

            if let Some(ret_local_id) = ret_local_id {
                let ret_value = frame.stack.pop().unwrap();
                frame.set_local(ret_local_id, ret_value);
            }
        } else {
            self.push_frame(frames, func_id, args, ret_local_id)?;
        }

        Ok(())
    }

    fn eval_global_set(&mut self, global_id: GlobalId, (value, attrs): (Value, ValueAttrs)) {
        self.globals
            .insert(global_id, GlobalValue::Value(value, attrs));
    }

    fn eval_global_get(&self, global_id: GlobalId) -> Result<(Value, ValueAttrs)> {
        match self.globals[global_id] {
            GlobalValue::Import => bail!("cannot evaluate an imported global {global_id:?}"),
            GlobalValue::GlobalGet => bail!(
                "cannot evaluate a global {global_id:?} initialized with \
                the value of an imported global",
            ),
            GlobalValue::Value(value, attrs) => Ok((value, attrs)),
        }
    }

    fn make_symbolic_ptr(&mut self, base: i32) -> Ptr {
        let id = self.next_symbolic_ptr_id;
        self.next_symbolic_ptr_id = id.checked_add(1).unwrap();

        Ptr {
            base,
            id,
            offset: 0,
        }
    }

    fn eval_intr_arg_count(&mut self, frames: &mut Vec<Frame>) -> Result<()> {
        let frame = frames.last_mut().unwrap();
        frame.stack.push((
            Value::I32(u32::try_from(self.args.len()).unwrap() as i32),
            Default::default(),
        ));

        Ok(())
    }

    fn eval_intr_arg_len(
        &mut self,
        frames: &mut Vec<Frame>,
        args: Vec<(Value, ValueAttrs)>,
    ) -> Result<()> {
        let idx = args[0].0.unwrap_u32() as usize;
        let arg = self.args.get(idx).with_context(|| {
            anyhow!(
                "an interpreter argument index {idx} is out of bounds (provided {})",
                self.args.len(),
            )
        })?;
        let frame = frames.last_mut().unwrap();
        frame.stack.push((
            Value::I32(u32::try_from(arg.len()).unwrap() as i32),
            Default::default(),
        ));

        Ok(())
    }

    fn eval_intr_arg_read(
        &mut self,
        frames: &mut Vec<Frame>,
        args: Vec<(Value, ValueAttrs)>,
    ) -> Result<()> {
        let [idx, buf, size, offset] = array::from_fn(|i| args[i].0.unwrap_u32() as usize);
        let arg = self.args.get(idx).with_context(|| {
            anyhow!(
                "an interpreter argument index {idx} is out of bounds (provided {})",
                self.args.len(),
            )
        })?;
        let arg = arg.get(offset..).with_context(|| {
            anyhow!(
                "an offset {offset} is out of bounds for the argument #{idx} of length {}",
                arg.len(),
            )
        })?;
        ensure!(
            !self.module.default_mem.is_null(),
            "the module does not define a memory"
        );

        let write_count = self
            .module
            .get_mem_mut(
                self.module.default_mem,
                buf..buf.checked_add(size).context("the size is too large")?,
            )
            .context("could not write the argument value")?
            .write(arg)?;

        let frame = frames.last_mut().unwrap();
        frame
            .stack
            .push((Value::I32(write_count as i32), Default::default()));

        Ok(())
    }

    fn eval_intr_specialize(
        &mut self,
        frames: &mut Vec<Frame>,
        mut args: Vec<(Value, ValueAttrs)>,
    ) -> Result<()> {
        let [elem_idx, name_addr, name_len] = array::from_fn(|i| args[i].0.unwrap_u32() as usize);
        args.drain(..3);

        ensure!(
            !self.module.default_table.is_null(),
            "the module does not define a function table"
        );
        let table = &self.module.tables[self.module.default_table];
        ensure!(
            table.ty.elem_ty == ElemType::Funcref,
            "the default table has a wrong type: expected {}, got {}",
            ElemType::Funcref,
            table.ty.elem_ty,
        );

        let func_id = match &table.def {
            TableDef::Import(_) => {
                bail!("cannot specialize a function referenced in an imported table")
            }

            TableDef::Elems(elems) => elems
                .get(elem_idx)
                .with_context(|| {
                    anyhow!(
                        "a table entry index {elem_idx} is out of range for a table of size {}",
                        elems.len()
                    )
                })?
                .with_context(|| anyhow!("a table entry (index {elem_idx}) is uninitialized"))?,
        };

        let name = if name_len > 0 {
            ensure!(
                !self.module.default_mem.is_null(),
                "the module does not define a memory"
            );

            let name_bytes = self
                .module
                .get_mem(
                    self.module.default_mem,
                    name_addr
                        ..name_addr
                            .checked_add(name_len)
                            .context("the name is too large")?,
                )
                .context("could not read the export name")?;
            let name = str::from_utf8(name_bytes)
                .context("the export name is not a valid utf-8 string")?;

            ensure!(
                !self
                    .module
                    .exports
                    .values()
                    .any(|export| &export.name == name),
                "the export name is already in use: {name}"
            );

            Some(name.to_owned())
        } else {
            None
        };

        let args = args
            .into_iter()
            .map(|(value, attrs)| {
                if attrs.contains(ValueAttrs::UNKNOWN) {
                    None
                } else {
                    Some((value.with_dropped_ptr_base(), attrs))
                }
            })
            .collect();

        trace!("function to specialize: {func_id:?}");
        let spec_func_id = self.specialize(func_id, args)?;

        let table = &mut self.module.tables[self.module.default_table];
        let TableDef::Elems(elems) = &mut table.def else {
            unreachable!()
        };

        let idx = match elems
            .iter_mut()
            .enumerate()
            .skip(1)
            .find(|(_, elem)| elem.is_none())
        {
            Some((idx, elem)) => {
                trace!(
                    "replacing an empty function table slot {idx} with the specialized function id"
                );
                *elem = Some(spec_func_id);

                idx
            }

            None => {
                if elems.is_empty() {
                    elems.push(None);
                }

                trace!("growing the function table to insert the specialized function id");
                elems.push(Some(spec_func_id));
                table.ty.limits.min = table.ty.limits.min.max(elems.len().try_into().unwrap());

                if let Some(max) = &mut table.ty.limits.max {
                    if 2.max(elems.len() + 1) > *max as usize {
                        warn!(
                            "growing the function table past its maximum size ({max}): expanding to {}",
                            elems.len(),
                        );
                        *max = elems.len().try_into().unwrap();
                    }
                }

                elems.len() - 1
            }
        };

        if let Some(name) = name {
            self.module.exports.insert(Export {
                name,
                def: ExportDef::Func(spec_func_id),
            });
        }

        let frame = frames.last_mut().unwrap();
        frame.stack.push((
            Value::I32(u32::try_from(idx).unwrap() as i32),
            Default::default(),
        ));

        Ok(())
    }

    fn eval_intr_unknown(&mut self, frames: &mut Vec<Frame>, func_id: FuncId) -> Result<()> {
        let frame = frames.last_mut().unwrap();
        let func_ty = self.module.funcs[func_id].ty();
        let val_ty = func_ty.ret.as_ref().unwrap();
        frame.stack.push((
            Value::default_for(&val_ty),
            ValueAttrs::default() | ValueAttrs::UNKNOWN,
        ));

        Ok(())
    }

    fn eval_intr_const_ptr(
        &mut self,
        frames: &mut Vec<Frame>,
        args: Vec<(Value, ValueAttrs)>,
    ) -> Result<()> {
        let frame = frames.last_mut().unwrap();
        frame
            .stack
            .push((args[0].0, args[0].1 | ValueAttrs::CONST_PTR));

        Ok(())
    }

    fn eval_intr_symbolic_ptr(
        &mut self,
        frames: &mut Vec<Frame>,
        args: Vec<(Value, ValueAttrs)>,
    ) -> Result<()> {
        let frame = frames.last_mut().unwrap();
        frame.stack.push((
            Value::Ptr(self.make_symbolic_ptr(args[0].0.unwrap_i32())),
            args[0].1,
        ));

        Ok(())
    }

    fn eval_intr_concrete_ptr(
        &mut self,
        frames: &mut Vec<Frame>,
        args: Vec<(Value, ValueAttrs)>,
    ) -> Result<()> {
        let frame = frames.last_mut().unwrap();
        frame
            .stack
            .push((Value::I32(args[0].0.unwrap_i32()), args[0].1));

        Ok(())
    }

    fn eval_intr_propagate_load(
        &mut self,
        frames: &mut Vec<Frame>,
        args: Vec<(Value, ValueAttrs)>,
    ) -> Result<()> {
        let frame = frames.last_mut().unwrap();
        frame
            .stack
            .push((args[0].0, args[0].1 | ValueAttrs::PROPAGATE_LOAD));

        Ok(())
    }

    fn eval_intr_print_value(&mut self, args: Vec<(Value, ValueAttrs)>) -> Result<()> {
        let intr = IntrinsicDecl::PrintValue;

        match args[0].0 {
            Value::I32(value) => info!("{intr}: {value}"),
            Value::I64(value) => info!("{intr}: {value}"),
            Value::F32(value) => info!("{intr}: {value}"),
            Value::F64(value) => info!("{intr}: {value}"),
            Value::Ptr(Ptr { base, id, offset }) => {
                info!("{intr}: ptr({base} + {offset}, id={id})")
            }
        }

        Ok(())
    }

    fn eval_intr_print_str(&mut self, args: Vec<(Value, ValueAttrs)>) -> Result<()> {
        let [addr, len] = array::from_fn(|i| args[i].0.unwrap_u32() as usize);
        ensure!(
            !self.module.default_mem.is_null(),
            "the module does not define a memory",
        );
        let bytes = self
            .module
            .get_mem(
                self.module.default_mem,
                addr..addr.checked_add(len).context("the string is too large")?,
            )
            .context("could not read the string")?;
        info!(
            "{}: {}",
            IntrinsicDecl::PrintStr,
            String::from_utf8_lossy(bytes)
        );

        Ok(())
    }

    fn eval_intr_is_specializing(&mut self, frames: &mut Vec<Frame>) -> Result<()> {
        let frame = frames.last_mut().unwrap();
        frame.stack.push((Value::I32(0), Default::default()));

        Ok(())
    }

    fn eval_intr_inline(
        &mut self,
        frames: &mut Vec<Frame>,
        args: Vec<(Value, ValueAttrs)>,
    ) -> Result<()> {
        let frame = frames.last_mut().unwrap();
        let (value, mut attrs) = args[0];
        attrs |= ValueAttrs::INLINE;
        attrs.remove(ValueAttrs::NO_INLINE);
        frame.stack.push((value, attrs));

        Ok(())
    }

    fn eval_intr_no_inline(
        &mut self,
        frames: &mut Vec<Frame>,
        args: Vec<(Value, ValueAttrs)>,
    ) -> Result<()> {
        let frame = frames.last_mut().unwrap();
        let (value, mut attrs) = args[0];
        attrs |= ValueAttrs::NO_INLINE;
        attrs.remove(ValueAttrs::INLINE);
        frame.stack.push((value, attrs));

        Ok(())
    }

    fn eval_intr_file_open(
        &mut self,
        frames: &mut Vec<Frame>,
        args: Vec<(Value, ValueAttrs)>,
    ) -> Result<()> {
        let frame = frames.last_mut().unwrap();
        let [path_ptr, path_len, out_fd_ptr] = array::from_fn(|i| args[i].0.unwrap_u32() as usize);
        ensure!(
            !self.module.default_mem.is_null(),
            "the module does not define a memory",
        );
        let path = self
            .module
            .get_mem(
                self.module.default_mem,
                path_ptr
                    ..path_ptr
                        .checked_add(path_len)
                        .context("the path is too large")?,
            )
            .context("could not read the path")?;

        #[cfg(not(unix))]
        let path = OsString::from(str::from_utf8(path).context("could not decode the path")?);
        #[cfg(unix)]
        let path = OsString::from(<OsStr as std::os::unix::ffi::OsStrExt>::from_bytes(path));

        match File::open(&path) {
            Ok(file) => {
                let idx = if let Some((idx, entry)) = self
                    .files
                    .iter_mut()
                    .enumerate()
                    .find(|(_, entry)| entry.is_none())
                {
                    *entry = Some(file);

                    idx
                } else {
                    self.files.push(Some(file));

                    self.files.len() - 1
                };

                self.module
                    .get_mem_mut(
                        self.module.default_mem,
                        out_fd_ptr
                            ..out_fd_ptr
                                .checked_add(4)
                                .context("not enough memory for writing the file descriptor")?,
                    )
                    .context("could not write the file descriptor")?
                    .copy_from_slice(&u32::try_from(idx).unwrap().to_le_bytes());
                frame.stack.push((Value::I32(0), Default::default()));
            }

            Err(e) => {
                let errno = e
                    .raw_os_error()
                    .context("File::open returned an unknown error code")?;
                frame.stack.push((Value::I32(errno), Default::default()));
            }
        }

        Ok(())
    }

    fn eval_intr_file_read(
        &mut self,
        frames: &mut Vec<Frame>,
        args: Vec<(Value, ValueAttrs)>,
    ) -> Result<()> {
        let frame = frames.last_mut().unwrap();
        let [fd, buf_ptr, size, out_count_ptr] =
            array::from_fn(|i| args[i].0.unwrap_u32() as usize);
        ensure!(
            !self.module.default_mem.is_null(),
            "the module does not define a memory",
        );

        let file = if let Some(Some(file)) = self.files.get_mut(fd) {
            file
        } else {
            frame.stack.push((
                Value::I32(ErrorKind::InvalidInput as i32),
                Default::default(),
            ));

            return Ok(());
        };

        let buf = self
            .module
            .get_mem_mut(
                self.module.default_mem,
                buf_ptr
                    ..buf_ptr
                        .checked_add(size)
                        .context("the buffer is too large")?,
            )
            .context("the buffer is invalid")?;

        match file.read(buf) {
            Ok(size) => {
                self.module
                    .get_mem_mut(
                        self.module.default_mem,
                        out_count_ptr
                            ..out_count_ptr.checked_add(4).context(
                                "not enough memory for writing the number of bytes read",
                            )?,
                    )
                    .context("could not write the number of bytes read")?
                    .copy_from_slice(&u32::try_from(size).unwrap().to_le_bytes());
                frame.stack.push((Value::I32(0), Default::default()));
            }

            Err(e) => {
                let errno = e
                    .raw_os_error()
                    .context("File::open returned an unknown error code")?;
                frame.stack.push((Value::I32(errno), Default::default()));
            }
        }

        Ok(())
    }

    fn eval_intr_file_close(
        &mut self,
        frames: &mut Vec<Frame>,
        args: Vec<(Value, ValueAttrs)>,
    ) -> Result<()> {
        let frame = frames.last_mut().unwrap();
        let fd = args[0].0.unwrap_u32() as usize;

        if let Some(file @ Some(_)) = self.files.get_mut(fd) {
            *file = None;

            if self.files.len() == fd + 1 {
                // remove the tail None elements
                let last_valid_idx = self
                    .files
                    .iter()
                    .enumerate()
                    .rfind(|(_, file)| file.is_some())
                    .map(|(idx, _)| Bound::Excluded(idx))
                    .unwrap_or(Bound::Unbounded);
                self.files.drain((last_valid_idx, Bound::Unbounded));
            }

            frame.stack.push((Value::I32(0), Default::default()));
        } else {
            frame.stack.push((
                Value::I32(ErrorKind::InvalidInput as i32),
                Default::default(),
            ));
        }

        Ok(())
    }

    fn eval_intr_func_spec_policy(&mut self, args: Vec<(Value, ValueAttrs)>) -> Result<()> {
        let [name_regexp_ptr, name_regexp_len, inline_policy] =
            array::from_fn(|i| args[i].0.unwrap_u32());
        let name_regexp_ptr = name_regexp_ptr as usize;
        let name_regexp_len = name_regexp_len as usize;
        ensure!(
            !self.module.default_mem.is_null(),
            "the module does not define a memory",
        );

        let name_regexp_bytes = self
            .module
            .get_mem(
                self.module.default_mem,
                name_regexp_ptr
                    ..name_regexp_ptr
                        .checked_add(name_regexp_len)
                        .context("the name regexp is too long")?,
            )
            .context("could not read the name regexp")?;
        let name_regexp_str =
            str::from_utf8(name_regexp_bytes).context("could not decode the name regexp")?;
        let name_regexp =
            Regex::new(name_regexp_str).context("could not compile the name regexp")?;

        let Some(inline_policy) = u8::try_from(inline_policy)
            .ok()
            .and_then(|inline_policy| InlineDisposition::from_repr(inline_policy))
        else {
            warn!("invalid inline policy {inline_policy}");
            warn!(
                "valid values are: {}",
                InlineDisposition::VARIANTS
                    .iter()
                    .map(|variant| format!("{} ({variant})", *variant as u8))
                    .collect::<Vec<_>>()
                    .join(", "),
            );
            bail!("invalid inline policy {inline_policy}");
        };

        let func_spec_policy = FuncSpecPolicy { inline_policy };
        let mut empty = true;

        for (func_id, func) in &self.module.funcs {
            if func.name().is_some_and(|name| name_regexp.is_match(name)) {
                debug!(
                    "applying a specialization policy to {func_id:?} ({}): {func_spec_policy:?}",
                    func.name().unwrap(),
                );
                self.func_spec_policies
                    .insert(func_id, func_spec_policy.clone());
                empty = false;
            }
        }

        if empty {
            warn!(
                "the name regexp didn't match any function: {}",
                name_regexp.as_str()
            );
        }

        Ok(())
    }

    fn eval_intr_symbolic_stack_ptr(&mut self) -> Result<()> {
        let Some((global_id, _)) = self.module.globals.iter().find(|&(_, global)| {
            global
                .name
                .as_ref()
                .is_some_and(|name| name == STACK_POINTER_NAME)
        }) else {
            warn!("no stack pointer found (searched for ${STACK_POINTER_NAME})");
            return Ok(());
        };

        let (value, attrs) = self.eval_global_get(global_id)?;

        match value {
            Value::I32(base) => {
                debug!("marking {global_id:?} as symbolic");
                let value = (Value::Ptr(self.make_symbolic_ptr(base)), attrs);
                self.eval_global_set(global_id, value);
            }

            Value::Ptr(_) => {
                debug!("marking {global_id:?} as symbolic");
            }

            _ => {
                warn!(
                    "the global ${STACK_POINTER_NAME} has a wrong type: expected i32, got {}",
                    value.val_ty()
                );
                return Ok(());
            }
        }

        *self.global_attrs.entry(global_id).unwrap().or_default() |= GlobalAttrs::CONST;

        Ok(())
    }
}
