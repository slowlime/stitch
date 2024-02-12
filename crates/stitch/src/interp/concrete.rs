use std::array;
use std::ops::Neg;
use std::rc::Rc;

use anyhow::{anyhow, bail, ensure, Context, Result};
use slotmap::SecondaryMap;

use crate::ast::expr::{Id, Value, ValueAttrs, F32, F64};
use crate::ast::ty::FuncType;
use crate::ast::{ConstExpr, FuncId, GlobalDef, IntrinsicDecl, MemoryDef, TableDef, PAGE_SIZE};
use crate::cfg::{
    BinOp, BlockId, Call, Expr, FuncBody, LocalId, NulOp, Stmt, Terminator, TernOp, UnOp,
};
use crate::interp::{format_arg_list, SpecializedFunc};

use super::Interpreter;

#[derive(Clone, Copy)]
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

trait ValueExt {
    fn try_to_i32(&self) -> Result<i32>;
    fn try_to_u32(&self) -> Result<u32>;
    fn unwrap_i64(&self) -> i64;
    fn unwrap_u64(&self) -> u64;
    fn unwrap_f32(&self) -> f32;
    fn unwrap_f64(&self) -> f64;
}

impl ValueExt for Value {
    fn try_to_i32(&self) -> Result<i32> {
        match self {
            Self::Id(id) => bail!("cannot cast (index_of {id:?}) to i32"),
            _ => Ok(self.to_i32().unwrap()),
        }
    }

    fn try_to_u32(&self) -> Result<u32> {
        self.try_to_i32().map(|x| x as u32)
    }

    fn unwrap_i64(&self) -> i64 {
        self.to_i64().unwrap()
    }

    fn unwrap_u64(&self) -> u64 {
        self.to_u64().unwrap()
    }

    fn unwrap_f32(&self) -> f32 {
        self.to_f32().unwrap().to_f32()
    }

    fn unwrap_f64(&self) -> f64 {
        self.to_f64().unwrap().to_f64()
    }
}

#[derive(Debug, Clone)]
struct Frame {
    locals: SecondaryMap<LocalId, (Value, ValueAttrs)>,
    stack: Vec<(Value, ValueAttrs)>,
    func_id: FuncId,
    block_id: BlockId,
    next_stmt: usize,
    subexpr_stack: Vec<usize>,
}

impl Frame {
    fn jump(&mut self, block_id: BlockId) {
        self.block_id = block_id;
        self.next_stmt = 0;
        self.subexpr_stack = vec![0];
    }
}

#[derive(Debug, Clone)]
enum EvalResult {
    Ok,
    Yielded,
}

impl Interpreter<'_> {
    pub fn interpret(
        &mut self,
        func_id: FuncId,
        args: Vec<(Value, ValueAttrs)>,
    ) -> Result<Option<(Value, ValueAttrs)>> {
        let ref mut frames = vec![];
        self.push_frame(frames, func_id, args)?;

        'frame: while let Some(frame) = frames.last_mut() {
            let func = Rc::clone(&self.cfgs[frame.func_id]);
            let block = &func.blocks[frame.block_id];

            let result = match block.body.get(frame.next_stmt) {
                Some(stmt) => match stmt {
                    Stmt::Nop => EvalResult::Ok,
                    Stmt::Drop(expr) => self.eval(frames, expr)?,
                    Stmt::LocalSet(_, expr) => self.eval(frames, expr)?,
                    Stmt::GlobalSet(_, expr) => self.eval(frames, expr)?,
                    Stmt::Store(_, _, exprs) => self.eval(frames, exprs.as_slice())?,
                    Stmt::Call(call) => self.eval(frames, call)?,
                },

                None => match &block.term {
                    Terminator::Trap | Terminator::Br(_) | Terminator::Return(None) => {
                        EvalResult::Ok
                    }
                    Terminator::If(expr, _) => self.eval(frames, expr)?,
                    Terminator::Switch(expr, _) => self.eval(frames, expr)?,
                    Terminator::Return(Some(expr)) => self.eval(frames, expr)?,
                },
            };

            match result {
                EvalResult::Ok => {}
                EvalResult::Yielded => continue 'frame,
            }

            let frame = frames.last_mut().unwrap();

            if let Some(stmt) = block.body.get(frame.next_stmt) {
                match *stmt {
                    Stmt::Nop => {}

                    Stmt::Drop(_) => {
                        frame.stack.pop().unwrap();
                    }

                    Stmt::LocalSet(local_id, _) => {
                        let value = frame.stack.pop().unwrap();
                        frame.locals[local_id] = value;
                    }

                    Stmt::GlobalSet(global_id, _) => {
                        let (value, attrs) = frame.stack.pop().unwrap();

                        match self.module.globals[global_id].def {
                            GlobalDef::Import(_) => bail!("cannot assign to an imported global"),
                            GlobalDef::Value(ref mut expr) => {
                                *expr = ConstExpr::Value(value, attrs)
                            }
                        }
                    }

                    Stmt::Store(mem_arg, store, _) => {
                        let (value, _) = frame.stack.pop().unwrap();
                        let base_addr = frame.stack.pop().unwrap().0.try_to_u32()?;
                        let start = (base_addr + mem_arg.offset) as usize;
                        let range = start..start + store.dst_size();
                        let bytes = self.module.get_mem_mut(mem_arg.mem_id, range)?;
                        store.store(bytes, value);
                    }

                    Stmt::Call(_) => {}
                }

                frame.next_stmt += 1;
            } else {
                match block.term {
                    Terminator::Trap => bail!("aborting execution due to an explicit trap"),
                    Terminator::Br(block_id) => frame.jump(block_id),

                    Terminator::If(_, [then_block_id, else_block_id]) => {
                        if frame.stack.pop().unwrap().0.try_to_i32()? == 0 {
                            frame.jump(else_block_id);
                        } else {
                            frame.jump(then_block_id);
                        }
                    }

                    Terminator::Switch(_, ref block_ids) => {
                        let (&default_block_id, block_ids) = block_ids.split_last().unwrap();

                        match block_ids.get(frame.stack.pop().unwrap().0.try_to_u32()? as usize) {
                            Some(&block_id) => frame.jump(block_id),
                            None => frame.jump(default_block_id),
                        }
                    }

                    Terminator::Return(ref expr) => {
                        let value = expr.is_some().then(|| frame.stack.pop().unwrap());
                        frames.pop().unwrap();

                        match frames.last_mut() {
                            Some(frame) => frame.stack.extend(value),
                            None => return Ok(value),
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
    ) -> Result<()> {
        let func = &self.module.funcs[func_id];
        self.check_args(&args, func.ty())?;

        let Some(body) = func.body() else {
            bail!("cannot interpret imported function");
        };
        if matches!(
            self.spec_funcs
                .get(func_id)
                .and_then(|spec_sig| self.spec_sigs.get(spec_sig)),
            Some(SpecializedFunc::Pending(_))
        ) {
            bail!("cannot interpret a function while it is being specialized");
        }

        let func = self
            .cfgs
            .entry(func_id)
            .unwrap()
            .or_insert_with(|| Rc::new(FuncBody::from_ast(self.module, body)));

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
            subexpr_stack: Default::default(),
        };

        frame.jump(func.entry);
        frames.push(frame);

        Ok(())
    }

    fn eval<'a>(
        &mut self,
        frames: &mut Vec<Frame>,
        task: impl Into<Task<'a>>,
    ) -> Result<EvalResult> {
        let mut stack = vec![task.into()];

        while let Some(&task) = stack.last() {
            let frame = frames.last_mut().unwrap();
            let subexpr_idx = &mut frame.subexpr_stack[stack.len() - 1];

            let subexpr = match task {
                Task::Expr(expr) => expr.nth_subexpr(*subexpr_idx),
                Task::Call(call) => call.nth_subexpr(*subexpr_idx),
                Task::Exprs(exprs) => exprs.get(*subexpr_idx),
            };

            if let Some(subexpr) = subexpr {
                *subexpr_idx += 1;

                if frame.subexpr_stack.len() == stack.len() {
                    frame.subexpr_stack.push(0);
                }

                stack.push(Task::Expr(subexpr));
                continue;
            }

            debug_assert_eq!(stack.len(), frame.subexpr_stack.len());

            match task {
                Task::Expr(expr) => match *expr {
                    Expr::Value(value, attrs) => frame.stack.push((value, attrs)),
                    Expr::Nullary(op) => self.eval_nul_op(frames, op)?,
                    Expr::Unary(op, _) => self.eval_un_op(frames, op)?,
                    Expr::Binary(op, _) => self.eval_bin_op(frames, op)?,
                    Expr::Ternary(op, _) => self.eval_tern_op(frames, op)?,
                    Expr::Call(ref call) => match self.eval_call(frames, call)? {
                        EvalResult::Ok => {}
                        EvalResult::Yielded => return Ok(EvalResult::Yielded),
                    },
                },

                Task::Call(call) => match self.eval_call(frames, call)? {
                    EvalResult::Ok => {}
                    EvalResult::Yielded => return Ok(EvalResult::Yielded),
                },

                Task::Exprs(_) => {}
            }

            stack.pop();
            frames.last_mut().unwrap().subexpr_stack.pop();
        }

        Ok(EvalResult::Ok)
    }

    fn eval_nul_op(&mut self, frames: &mut Vec<Frame>, op: NulOp) -> Result<()> {
        let frame = frames.last_mut().unwrap();

        match op {
            NulOp::LocalGet(local_id) => frame.stack.push(*frame.locals.get(local_id).unwrap()),

            NulOp::GlobalGet(global_id) => {
                match self.module.globals[global_id].def {
                    GlobalDef::Import(_) => bail!("cannot evaluate an imported global"),

                    GlobalDef::Value(ConstExpr::Value(value, attrs)) => {
                        frame.stack.push((value, attrs))
                    }

                    GlobalDef::Value(ConstExpr::GlobalGet(_)) => {
                        bail!("cannot evaluate a global initialized with the value of an imported global")
                    }
                }
            }

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
                Value::I32(arg.try_to_i32()?.leading_zeros() as i32),
                Default::default(),
            ),
            UnOp::I32Ctz => (
                Value::I32(arg.try_to_i32()?.trailing_zeros() as i32),
                Default::default(),
            ),
            UnOp::I32Popcnt => (
                Value::I32(arg.try_to_i32()?.count_ones() as i32),
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

            UnOp::F32Abs => (Value::F32(arg.unwrap_f32().abs().into()), arg_attrs),
            UnOp::F32Neg => (Value::F32(arg.unwrap_f32().neg().into()), arg_attrs),
            UnOp::F32Sqrt => (Value::F32(arg.unwrap_f32().sqrt().into()), arg_attrs),
            UnOp::F32Ceil => (Value::F32(arg.unwrap_f32().ceil().into()), arg_attrs),
            UnOp::F32Floor => (Value::F32(arg.unwrap_f32().floor().into()), arg_attrs),
            UnOp::F32Trunc => (Value::F32(arg.unwrap_f32().trunc().into()), arg_attrs),
            UnOp::F32Nearest => (Value::F32(arg.to_f32().unwrap().nearest()), arg_attrs),

            UnOp::F64Abs => (Value::F64(arg.unwrap_f64().abs().into()), arg_attrs),
            UnOp::F64Neg => (Value::F64(arg.unwrap_f64().neg().into()), arg_attrs),
            UnOp::F64Sqrt => (Value::F64(arg.unwrap_f64().sqrt().into()), arg_attrs),
            UnOp::F64Ceil => (Value::F64(arg.unwrap_f64().ceil().into()), arg_attrs),
            UnOp::F64Floor => (Value::F64(arg.unwrap_f64().floor().into()), arg_attrs),
            UnOp::F64Trunc => (Value::F64(arg.unwrap_f64().trunc().into()), arg_attrs),
            UnOp::F64Nearest => (Value::F64(arg.to_f64().unwrap().nearest()), arg_attrs),

            UnOp::I32Eqz => (
                Value::I32((arg.try_to_i32()? == 0) as i32),
                Default::default(),
            ),
            UnOp::I64Eqz => (
                Value::I32((arg.try_to_i32()? == 0) as i32),
                Default::default(),
            ),

            UnOp::I32WrapI64 => (Value::I32(arg.unwrap_i64() as i32), arg_attrs),

            UnOp::I64ExtendI32S => (Value::I64(arg.try_to_i32()? as i64), arg_attrs),
            UnOp::I64ExtendI32U => (Value::I64(arg.try_to_u32()? as i64), arg_attrs),

            UnOp::I32TruncF32S => (Value::I32(arg.unwrap_f32() as i32), arg_attrs),
            UnOp::I32TruncF32U => (Value::I32(arg.unwrap_f32() as u32 as i32), arg_attrs),
            UnOp::I32TruncF64S => (Value::I32(arg.unwrap_f64() as i32), arg_attrs),
            UnOp::I32TruncF64U => (Value::I32(arg.unwrap_f64() as u32 as i32), arg_attrs),

            UnOp::I64TruncF32S => (Value::I64(arg.unwrap_f32() as i64), arg_attrs),
            UnOp::I64TruncF32U => (Value::I64(arg.unwrap_f32() as u64 as i64), arg_attrs),
            UnOp::I64TruncF64S => (Value::I64(arg.unwrap_f64() as i64), arg_attrs),
            UnOp::I64TruncF64U => (Value::I64(arg.unwrap_f64() as u64 as i64), arg_attrs),

            UnOp::F32DemoteF64 => (Value::F32((arg.unwrap_f64() as f32).into()), arg_attrs),
            UnOp::F64PromoteF32 => (Value::F64((arg.unwrap_f32() as f64).into()), arg_attrs),

            UnOp::F32ConvertI32S => (Value::F32((arg.try_to_i32()? as f32).into()), arg_attrs),
            UnOp::F32ConvertI32U => (Value::F32((arg.try_to_u32()? as f32).into()), arg_attrs),
            UnOp::F32ConvertI64S => (Value::F32((arg.unwrap_i64() as f32).into()), arg_attrs),
            UnOp::F32ConvertI64U => (Value::F32((arg.unwrap_u64() as f32).into()), arg_attrs),

            UnOp::F64ConvertI32S => (Value::F64((arg.try_to_i32()? as f64).into()), arg_attrs),
            UnOp::F64ConvertI32U => (Value::F64((arg.try_to_u32()? as f64).into()), arg_attrs),
            UnOp::F64ConvertI64S => (Value::F64((arg.unwrap_i64() as f64).into()), arg_attrs),
            UnOp::F64ConvertI64U => (Value::F64((arg.unwrap_u64() as f64).into()), arg_attrs),

            UnOp::F32ReinterpretI32 => (Value::F32(F32::from_bits(arg.try_to_u32()?)), arg_attrs),
            UnOp::F64ReinterpretI64 => (Value::F64(F64::from_bits(arg.unwrap_u64())), arg_attrs),
            UnOp::I32ReinterpretF32 => (
                Value::I32(arg.to_f32().unwrap().to_bits() as i32),
                arg_attrs,
            ),
            UnOp::I64ReinterpretF64 => (
                Value::I64(arg.to_f64().unwrap().to_bits() as i64),
                arg_attrs,
            ),

            UnOp::I32Extend8S => (Value::I32(arg.try_to_u32()? as i8 as i32), arg_attrs),
            UnOp::I32Extend16S => (Value::I32(arg.try_to_u32()? as i64 as i32), arg_attrs),

            UnOp::I64Extend8S => (Value::I64(arg.unwrap_u64() as i8 as i64), arg_attrs),
            UnOp::I64Extend16S => (Value::I64(arg.unwrap_u64() as i16 as i64), arg_attrs),
            UnOp::I64Extend32S => (Value::I64(arg.unwrap_u64() as i32 as i64), arg_attrs),

            UnOp::LocalTee(local_id) => {
                frame.locals[local_id] = (arg, arg_attrs);

                (arg, arg_attrs)
            }

            UnOp::Load(mem_arg, load) => {
                let base_addr = arg.try_to_u32()?;
                let start = (base_addr + mem_arg.offset) as usize;
                let range = start..start + load.src_size();
                let bytes = self.module.get_mem(mem_arg.mem_id, range)?;
                let value = load.load(bytes);

                (value, arg_attrs.deref_attrs())
            }

            UnOp::MemoryGrow(mem_id) => {
                let mem = &mut self.module.mems[mem_id];

                match mem.def {
                    MemoryDef::Import(_) => {
                        bail!("cannot resize an imported memory")
                    }

                    MemoryDef::Bytes(ref mut bytes) => {
                        let prev_pages = bytes.len().div_ceil(PAGE_SIZE);
                        let increment = arg.try_to_u32()? as usize;

                        match mem.ty.limits.max {
                            Some(max) if prev_pages + increment > max as usize => {
                                (Value::I32(-1), Default::default())
                            }

                            _ => {
                                bytes.resize((prev_pages + increment) * PAGE_SIZE, 0);

                                (Value::I32(prev_pages as i32), Default::default())
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
                Value::I32(lhs.try_to_i32()?.wrapping_add(rhs.try_to_i32()?)),
                meet_attrs,
            ),
            BinOp::I32Sub => (
                Value::I32(lhs.try_to_i32()?.wrapping_add(rhs.try_to_i32()?)),
                meet_attrs,
            ),
            BinOp::I32Mul => (
                Value::I32(lhs.try_to_i32()?.wrapping_mul(rhs.try_to_i32()?)),
                meet_attrs,
            ),
            BinOp::I32DivS => match (lhs.try_to_i32()?, rhs.try_to_i32()?) {
                (_, 0) => bail!("division by zero"),
                (lhs, -1) if lhs == i32::MIN => {
                    bail!("dividing i32::MIN by -1 resulted in an overflow")
                }
                (lhs, rhs) => (Value::I32(lhs / rhs), meet_attrs),
            },
            BinOp::I32DivU => match (lhs.try_to_u32()?, rhs.try_to_u32()?) {
                (_, 0) => bail!("division by zero"),
                (lhs, rhs) => (Value::I32((lhs / rhs) as i32), meet_attrs),
            },
            BinOp::I32RemS => match (lhs.try_to_i32()?, rhs.try_to_i32()?) {
                (_, 0) => bail!("taking the remainder of dividing by zero"),
                (lhs, rhs) => (Value::I32(lhs % rhs), meet_attrs),
            },
            BinOp::I32RemU => match (lhs.try_to_u32()?, rhs.try_to_u32()?) {
                (_, 0) => bail!("taking the remainder of dividing by zero"),
                (lhs, rhs) => (Value::I32((lhs % rhs) as i32), meet_attrs),
            },
            BinOp::I32And => (
                Value::I32(lhs.try_to_i32()? & rhs.try_to_i32()?),
                meet_attrs,
            ),
            BinOp::I32Or => (
                Value::I32(lhs.try_to_i32()? & rhs.try_to_i32()?),
                meet_attrs,
            ),
            BinOp::I32Xor => (
                Value::I32(lhs.try_to_i32()? ^ rhs.try_to_i32()?),
                meet_attrs,
            ),
            BinOp::I32Shl => (
                Value::I32(lhs.try_to_i32()?.wrapping_shl(rhs.try_to_u32()?)),
                meet_attrs,
            ),
            BinOp::I32ShrS => (
                Value::I32(lhs.try_to_i32()?.wrapping_shr(rhs.try_to_u32()?)),
                meet_attrs,
            ),
            BinOp::I32ShrU => (
                Value::I32(lhs.try_to_u32()?.wrapping_shr(rhs.try_to_u32()?) as i32),
                meet_attrs,
            ),
            BinOp::I32Rotl => (
                Value::I32(lhs.try_to_i32()?.rotate_left(rhs.try_to_u32()?)),
                meet_attrs,
            ),
            BinOp::I32Rotr => (
                Value::I32(lhs.try_to_i32()?.rotate_right(rhs.try_to_u32()?)),
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

            BinOp::F32Add => (
                Value::F32((lhs.unwrap_f32() + rhs.unwrap_f32()).into()),
                meet_attrs,
            ),
            BinOp::F32Sub => (
                Value::F32((lhs.unwrap_f32() - rhs.unwrap_f32()).into()),
                meet_attrs,
            ),
            BinOp::F32Mul => (
                Value::F32((lhs.unwrap_f32() * rhs.unwrap_f32()).into()),
                meet_attrs,
            ),
            BinOp::F32Div => (
                Value::F32((lhs.unwrap_f32() / rhs.unwrap_f32()).into()),
                meet_attrs,
            ),
            BinOp::F32Min => (
                Value::F32(lhs.unwrap_f32().min(rhs.unwrap_f32()).into()),
                meet_attrs,
            ),
            BinOp::F32Max => (
                Value::F32(lhs.unwrap_f32().max(rhs.unwrap_f32()).into()),
                meet_attrs,
            ),
            BinOp::F32Copysign => (
                Value::F32(lhs.unwrap_f32().copysign(rhs.unwrap_f32()).into()),
                meet_attrs,
            ),

            BinOp::F64Add => (
                Value::F64((lhs.unwrap_f64() + rhs.unwrap_f64()).into()),
                meet_attrs,
            ),
            BinOp::F64Sub => (
                Value::F64((lhs.unwrap_f64() - rhs.unwrap_f64()).into()),
                meet_attrs,
            ),
            BinOp::F64Mul => (
                Value::F64((lhs.unwrap_f64() * rhs.unwrap_f64()).into()),
                meet_attrs,
            ),
            BinOp::F64Div => (
                Value::F64((lhs.unwrap_f64() / rhs.unwrap_f64()).into()),
                meet_attrs,
            ),
            BinOp::F64Min => (
                Value::F64(lhs.unwrap_f64().min(rhs.unwrap_f64()).into()),
                meet_attrs,
            ),
            BinOp::F64Max => (
                Value::F64(lhs.unwrap_f64().max(rhs.unwrap_f64()).into()),
                meet_attrs,
            ),
            BinOp::F64Copysign => (
                Value::F64(lhs.unwrap_f64().copysign(rhs.unwrap_f64()).into()),
                meet_attrs,
            ),

            BinOp::I32Eq => (
                Value::I32(match (lhs, rhs) {
                    (Value::Id(Id::Func(lhs)), Value::Id(Id::Func(rhs))) => (lhs == rhs) as i32,
                    _ => (lhs.try_to_i32()? == rhs.try_to_i32()?) as i32,
                }),
                Default::default(),
            ),
            BinOp::I32Ne => (
                Value::I32(match (lhs, rhs) {
                    (Value::Id(Id::Func(lhs)), Value::Id(Id::Func(rhs))) => (lhs != rhs) as i32,
                    _ => (lhs.try_to_i32()? != rhs.try_to_i32()?) as i32,
                }),
                Default::default(),
            ),
            BinOp::I32LtS => (
                Value::I32((lhs.try_to_i32()? < rhs.try_to_i32()?) as i32),
                Default::default(),
            ),
            BinOp::I32LtU => (
                Value::I32((lhs.try_to_u32()? < rhs.try_to_u32()?) as i32),
                Default::default(),
            ),
            BinOp::I32GtS => (
                Value::I32((lhs.try_to_i32()? > rhs.try_to_i32()?) as i32),
                Default::default(),
            ),
            BinOp::I32GtU => (
                Value::I32((lhs.try_to_u32()? > rhs.try_to_u32()?) as i32),
                Default::default(),
            ),
            BinOp::I32LeS => (
                Value::I32((lhs.try_to_i32()? <= rhs.try_to_i32()?) as i32),
                Default::default(),
            ),
            BinOp::I32LeU => (
                Value::I32((lhs.try_to_u32()? <= rhs.try_to_u32()?) as i32),
                Default::default(),
            ),
            BinOp::I32GeS => (
                Value::I32((lhs.try_to_i32()? >= rhs.try_to_i32()?) as i32),
                Default::default(),
            ),
            BinOp::I32GeU => (
                Value::I32((lhs.try_to_u32()? >= rhs.try_to_u32()?) as i32),
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

                match condition.try_to_i32()? {
                    0 => rhs,
                    _ => lhs,
                }
            }
        });

        Ok(())
    }

    fn eval_call(&mut self, frames: &mut Vec<Frame>, call: &Call) -> Result<EvalResult> {
        let frame = frames.last_mut().unwrap();
        let subexpr_idx = frame.subexpr_stack.last_mut().unwrap();

        // if subexpr_idx equals the number of the call's subexprs, we enqueue a call.
        // if it's greater, the subroutine has returned; the results are already on this frame's stack.
        if *subexpr_idx > call.subexpr_count() {
            return Ok(EvalResult::Ok);
        }

        *subexpr_idx += 1;
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
                    let idx = args.pop().unwrap().0.try_to_u32()? as usize;

                    elems.get(idx)
                        .with_context(|| anyhow!(
                            "the indirect call index ({idx}) is out of range for a table of size {}",
                            elems.len()
                        ))?
                        .with_context(|| anyhow!("an indirect call's table entry (index {idx}) is uninitialized"))?
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

        if let Some(intrinsic) = self.module.funcs[func_id].get_intrinsic(&self.module) {
            match intrinsic {
                IntrinsicDecl::Specialize => self.eval_intr_specialize(func_id, args)?,
                IntrinsicDecl::Unknown => self.eval_intr_unknown(func_id)?,
            }

            Ok(EvalResult::Ok)
        } else {
            self.push_frame(frames, func_id, args)?;

            Ok(EvalResult::Yielded)
        }
    }

    fn eval_intr_specialize(&mut self, func_id: FuncId, args: Vec<(Value, ValueAttrs)>) -> Result<()> {
        todo!()
    }

    fn eval_intr_unknown(&mut self, func_id: FuncId) -> Result<()> {
        todo!()
    }
}
