use std::collections::{HashMap, HashSet};
use std::ops::Neg;
use std::rc::Rc;
use std::{array, iter, mem};

use anyhow::{ensure, Result};
use log::warn;
use slotmap::{Key, SecondaryMap};

use crate::ast::expr::{Value, ValueAttrs};
use crate::ast::{ConstExpr, FuncId, GlobalDef, GlobalId, IntrinsicDecl, Module, TableDef};
use crate::cfg::{
    BinOp, BlockId, Call, Expr, FuncBody, LocalId, NulOp, Stmt, Terminator, TernOp, UnOp,
};
use crate::util::float::{F32, F64};

use super::{Interpreter, SpecSignature};

type OrigBlockId = BlockId;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct HashableEnv {
    globals: Vec<(GlobalId, (Value, ValueAttrs))>,
    locals: Vec<(LocalId, (Value, ValueAttrs))>,
}

#[derive(Debug, Default, Clone)]
struct Env {
    globals: SecondaryMap<GlobalId, (Value, ValueAttrs)>,
    locals: SecondaryMap<LocalId, (Value, ValueAttrs)>,
}

impl Env {
    fn entry(module: &Module, cfg: &FuncBody, args: &[Option<(Value, ValueAttrs)>]) -> Self {
        let globals = module
            .globals
            .iter()
            .filter_map(|(global_id, global)| match global.def {
                GlobalDef::Import(_) => None,
                GlobalDef::Value(ConstExpr::Value(value, attrs)) => {
                    Some((global_id, (value, attrs)))
                }
                GlobalDef::Value(ConstExpr::GlobalGet(_)) => None,
            })
            .collect();
        let locals = args
            .iter()
            .zip(&cfg.params)
            .filter_map(|(arg, &local_id)| arg.map(|value| (local_id, value)))
            .collect();

        Self { globals, locals }
    }

    fn merge(&mut self, other: &Self) -> bool {
        fn merge_maps<K: Key>(
            lhs_map: &mut SecondaryMap<K, (Value, ValueAttrs)>,
            rhs_map: &SecondaryMap<K, (Value, ValueAttrs)>,
        ) -> bool {
            let mut changed = false;
            let mut to_remove = vec![];

            for (id, (lhs, lhs_attrs)) in &mut *lhs_map {
                let new = rhs_map.get(id).and_then(|(rhs, rhs_attrs)| {
                    lhs.meet(rhs)
                        .map(|value| (value, lhs_attrs.meet(rhs_attrs)))
                });

                if let Some((new, new_attrs)) = new {
                    changed = *lhs != new || changed;
                    *lhs = new;
                    *lhs_attrs = new_attrs;
                } else {
                    changed = true;
                    to_remove.push(id);
                }
            }

            for id in to_remove {
                lhs_map.remove(id);
            }

            changed
        }

        merge_maps(&mut self.globals, &other.globals) | merge_maps(&mut self.locals, &other.locals)
    }

    fn to_hashable(&self) -> HashableEnv {
        let mut result = HashableEnv {
            globals: self
                .globals
                .iter()
                .map(|(global_id, &value)| (global_id, value))
                .collect(),
            locals: self
                .locals
                .iter()
                .map(|(local_id, &value)| (local_id, value))
                .collect(),
        };

        result
            .globals
            .sort_unstable_by_key(|&(global_id, _)| global_id);
        result
            .locals
            .sort_unstable_by_key(|&(local_id, _)| local_id);

        result
    }
}

#[derive(Debug, Clone)]
struct BlockInfo {
    orig_block_id: OrigBlockId,
    entry_env: Env,
    exit_env: Env,
}

enum Loop {
    Kept(BlockId),
    Unrolled,
}

impl Loop {
    fn is_unrolled(&self) -> bool {
        matches!(self, Self::Unrolled)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Task(BlockId);

pub struct Specializer<'a, 'i> {
    interp: &'i mut Interpreter<'a>,
    cfg: Rc<FuncBody>,
    func: FuncBody,
    block_map: HashMap<(OrigBlockId, HashableEnv), BlockId>,
    blocks: SecondaryMap<BlockId, BlockInfo>,
    args: Vec<Option<(Value, ValueAttrs)>>,
    tasks: HashSet<Task>,
    loop_headers: HashSet<OrigBlockId>,
    loops: HashMap<OrigBlockId, Loop>,
}

impl<'a, 'i> Specializer<'a, 'i> {
    pub fn new(
        interp: &'i mut Interpreter<'a>,
        SpecSignature { args, .. }: SpecSignature,
        cfg: Rc<FuncBody>,
    ) -> Self {
        let func = FuncBody::new(cfg.ty.clone());
        let loop_headers = cfg.loop_headers(&cfg.rpo());

        Self {
            interp,
            cfg,
            func,
            block_map: Default::default(),
            blocks: Default::default(),
            args,
            tasks: Default::default(),
            loop_headers,
            loops: Default::default(),
        }
    }

    pub fn run(mut self) -> Result<FuncBody> {
        self.process_branch(
            self.cfg.entry,
            Env::entry(&self.interp.module, &self.cfg, &self.args),
        );

        while let Some(&Task(block_id)) = self.tasks.iter().next() {
            self.process_block(block_id)?;
        }

        Ok(self.func)
    }

    fn process_block(&mut self, block_id: BlockId) -> Result<()> {
        let info = &mut self.blocks[block_id];
        info.exit_env = info.entry_env.clone();
        let orig_block_id = info.orig_block_id;
        let cfg = Rc::clone(&self.cfg);
        let mut stmt_idx = 0;
        let ref mut stack = vec![];

        while let Some(stmt) = cfg.blocks[orig_block_id].body.get(stmt_idx) {
            stmt_idx += 1;
            stack.clear();

            match stmt {
                Stmt::Nop => {}
                Stmt::Drop(expr) => self.process_expr(block_id, stack, expr)?,
                Stmt::LocalSet(_, expr) => self.process_expr(block_id, stack, expr)?,
                Stmt::GlobalSet(_, expr) => self.process_expr(block_id, stack, expr)?,
                Stmt::Store(_, _, exprs) => {
                    for expr in &**exprs {
                        self.process_expr(block_id, stack, expr)?;
                    }
                }
                Stmt::Call(Call::Direct { args, .. }) => {
                    for expr in args {
                        self.process_expr(block_id, stack, expr)?;
                    }
                }
                Stmt::Call(Call::Indirect { args, index, .. }) => {
                    for expr in args.iter().chain(iter::once(&**index)) {
                        self.process_expr(block_id, stack, expr)?;
                    }
                }
            }

            let mut block = &mut self.func.blocks[block_id];
            let info = &mut self.blocks[block_id];

            match *stmt {
                Stmt::Nop => {}

                Stmt::Drop(_) => {
                    let expr = stack.pop().unwrap();

                    if expr.has_side_effect() {
                        block.body.push(Stmt::Drop(expr));
                    }
                }

                Stmt::LocalSet(local_id, _) => {
                    let expr = stack.pop().unwrap();

                    if let Some(value) = expr.to_value() {
                        info.exit_env.locals[local_id] = value;
                    } else {
                        block.body.push(Stmt::LocalSet(local_id, expr));
                    }
                }

                Stmt::GlobalSet(global_id, _) => {
                    let expr = stack.pop().unwrap();

                    if let Some(value) = expr.to_value() {
                        info.exit_env.globals[global_id] = value;
                    } else {
                        block.body.push(Stmt::GlobalSet(global_id, expr));
                    }
                }

                Stmt::Store(mem_arg, store, _) => {
                    let value = stack.pop().unwrap();
                    let base_addr = stack.pop().unwrap();

                    if let Some((_, addr_attrs)) = base_addr.to_value() {
                        ensure!(
                            !addr_attrs.contains(ValueAttrs::CONST_PTR),
                            "encountered a memory write via a constant pointer"
                        );
                    }

                    block
                        .body
                        .push(Stmt::Store(mem_arg, store, Box::new([base_addr, value])));
                }

                Stmt::Call(Call::Direct {
                    ret_local_id,
                    func_id,
                    ref args,
                }) => {
                    let args = stack.drain(stack.len() - args.len()..).collect();
                    self.process_call(block_id, &mut stmt_idx, ret_local_id, func_id, args)?;
                }

                Stmt::Call(Call::Indirect {
                    ret_local_id,
                    ty_id,
                    table_id,
                    ref args,
                    ..
                }) => {
                    let mut index = Some(stack.pop().unwrap());
                    let mut args = stack.drain(stack.len() - args.len()..).collect();

                    'spec: {
                        let Some((idx, _)) = index.as_ref().unwrap().to_value() else {
                            break 'spec;
                        };

                        match &self.interp.module.tables[table_id].def {
                            TableDef::Import(_) => {}

                            TableDef::Elems(elems) => {
                                let idx = idx.to_u32().unwrap();
                                let Some(&Some(func_id)) = elems.get(idx as usize) else {
                                    break 'spec;
                                };

                                let actual_func_ty = self.interp.module.funcs[func_id].ty();
                                let claimed_func_ty = self.interp.module.types[ty_id].as_func();

                                if actual_func_ty != claimed_func_ty {
                                    break 'spec;
                                }

                                index = None;
                                let args = mem::take(&mut args);
                                self.process_call(
                                    block_id,
                                    &mut stmt_idx,
                                    ret_local_id,
                                    func_id,
                                    args,
                                )?;
                                block = &mut self.func.blocks[block_id];
                            }
                        }
                    }

                    if let Some(index) = index {
                        block.body.push(Stmt::Call(Call::Indirect {
                            ret_local_id,
                            ty_id,
                            table_id,
                            args,
                            index: Box::new(index),
                        }));
                    }
                }
            }
        }

        match &cfg.blocks[orig_block_id].term {
            Terminator::Trap | Terminator::Br(_) | Terminator::Return(None) => {}
            Terminator::If(expr, _) => self.process_expr(block_id, stack, expr)?,
            Terminator::Switch(expr, _) => self.process_expr(block_id, stack, expr)?,
            Terminator::Return(Some(expr)) => self.process_expr(block_id, stack, expr)?,
        }

        let info = &mut self.blocks[block_id];

        match cfg.blocks[orig_block_id].term {
            Terminator::Trap => {
                self.func.blocks[block_id].term = Terminator::Trap;
            }

            Terminator::Br(orig_target_block_id) => {
                let exit_env = info.exit_env.clone();
                let target_block_id = self.process_branch(orig_target_block_id, exit_env);
                self.func.blocks[block_id].term = Terminator::Br(target_block_id);
            }

            Terminator::If(_, [orig_then_block_id, orig_else_block_id]) => {
                let condition = stack.pop().unwrap();
                let exit_env = info.exit_env.clone();

                match condition
                    .to_value()
                    .map(|(value, _)| value.to_i32().unwrap())
                {
                    Some(0) => {
                        let else_block_id = self.process_branch(orig_else_block_id, exit_env);
                        self.func.blocks[block_id].term = Terminator::Br(else_block_id);
                    }

                    Some(_) => {
                        let then_block_id = self.process_branch(orig_then_block_id, exit_env);
                        self.func.blocks[block_id].term = Terminator::Br(then_block_id);
                    }

                    None => {
                        let then_block_id =
                            self.process_branch(orig_then_block_id, exit_env.clone());
                        let else_block_id = self.process_branch(orig_else_block_id, exit_env);
                        self.func.blocks[block_id].term =
                            Terminator::If(condition, [then_block_id, else_block_id]);
                    }
                }
            }

            Terminator::Switch(_, ref orig_block_ids) => {
                let index = stack.pop().unwrap();
                let exit_env = info.exit_env.clone();

                let orig_target_block_id = index.to_value().map(|(value, _)| {
                    let (&default_orig_block_id, orig_block_ids) =
                        orig_block_ids.split_last().unwrap();

                    orig_block_ids
                        .get(value.to_u32().unwrap() as usize)
                        .copied()
                        .unwrap_or(default_orig_block_id)
                });

                if let Some(orig_block_id) = orig_target_block_id {
                    let target_block_id = self.process_branch(orig_block_id, exit_env);
                    self.func.blocks[block_id].term = Terminator::Br(target_block_id);
                } else {
                    let block_ids = orig_block_ids
                        .iter()
                        .map(|&orig_block_id| self.process_branch(orig_block_id, exit_env.clone()))
                        .collect();

                    self.func.blocks[block_id].term = Terminator::Switch(index, block_ids);
                }
            }

            Terminator::Return(ref expr) => {
                let expr = expr.is_some().then(|| stack.pop().unwrap());
                self.func.blocks[block_id].term = Terminator::Return(expr);
            }
        }

        Ok(())
    }

    fn process_expr(
        &mut self,
        block_id: BlockId,
        stack: &mut Vec<Expr>,
        expr: &Expr,
    ) -> Result<()> {
        let mut tasks = vec![(0, expr)];

        while let Some(&mut (ref mut subexpr_idx, expr)) = tasks.last_mut() {
            if let Some(subexpr) = expr.nth_subexpr(*subexpr_idx) {
                *subexpr_idx += 1;
                tasks.push((0, subexpr));
            } else {
                match *expr {
                    Expr::Value(value, attrs) => stack.push(Expr::Value(value, attrs)),
                    Expr::Nullary(op) => self.process_nul_op(block_id, stack, op)?,
                    Expr::Unary(op, _) => self.process_un_op(block_id, stack, op)?,
                    Expr::Binary(op, _) => self.process_bin_op(block_id, stack, op)?,
                    Expr::Ternary(op, _) => self.process_tern_op(block_id, stack, op)?,
                }
            }
        }

        Ok(())
    }

    fn process_nul_op(
        &mut self,
        block_id: BlockId,
        stack: &mut Vec<Expr>,
        op: NulOp,
    ) -> Result<()> {
        let info = &mut self.blocks[block_id];

        let try_process = || -> Option<Expr> {
            Some(match op {
                NulOp::LocalGet(local_id) => info.exit_env.locals.get(local_id).copied()?.into(),
                NulOp::GlobalGet(global_id) => {
                    info.exit_env.globals.get(global_id).copied()?.into()
                }
                NulOp::MemorySize(_mem_id) => return None,
            })
        };

        stack.push(try_process().unwrap_or_else(|| Expr::Nullary(op)));

        Ok(())
    }

    fn process_un_op(&mut self, block_id: BlockId, stack: &mut Vec<Expr>, op: UnOp) -> Result<()> {
        let info = &mut self.blocks[block_id];
        let arg = stack.pop().unwrap();

        let mut try_process = || -> Option<Expr> {
            let &Expr::Value(value, attrs) = &arg else {
                return None;
            };

            Some(match op {
                UnOp::I32Clz => Expr::Value(
                    Value::I32(value.to_i32()?.leading_zeros() as i32),
                    Default::default(),
                ),
                UnOp::I32Ctz => Expr::Value(
                    Value::I32(value.to_i32()?.trailing_zeros() as i32),
                    Default::default(),
                ),
                UnOp::I32Popcnt => Expr::Value(
                    Value::I32(value.to_i32()?.count_ones() as i32),
                    Default::default(),
                ),

                UnOp::I64Clz => Expr::Value(
                    Value::I64(value.to_i64()?.leading_zeros() as i64),
                    Default::default(),
                ),
                UnOp::I64Ctz => Expr::Value(
                    Value::I64(value.to_i64()?.trailing_zeros() as i64),
                    Default::default(),
                ),
                UnOp::I64Popcnt => Expr::Value(
                    Value::I64(value.to_i64()?.count_ones() as i64),
                    Default::default(),
                ),

                UnOp::F32Abs => Expr::Value(Value::F32(value.to_f32()?.abs()), attrs),
                UnOp::F32Neg => Expr::Value(Value::F32(value.to_f32()?.neg()), attrs),
                UnOp::F32Sqrt => Expr::Value(Value::F32(value.to_f32()?.sqrt()), attrs),
                UnOp::F32Ceil => Expr::Value(Value::F32(value.to_f32()?.ceil()), attrs),
                UnOp::F32Floor => Expr::Value(Value::F32(value.to_f32()?.floor()), attrs),
                UnOp::F32Trunc => Expr::Value(Value::F32(value.to_f32()?.trunc()), attrs),
                UnOp::F32Nearest => Expr::Value(Value::F32(value.to_f32()?.nearest()), attrs),

                UnOp::F64Abs => Expr::Value(Value::F64(value.to_f64()?.abs()), attrs),
                UnOp::F64Neg => Expr::Value(Value::F64(value.to_f64()?.neg()), attrs),
                UnOp::F64Sqrt => Expr::Value(Value::F64(value.to_f64()?.sqrt()), attrs),
                UnOp::F64Ceil => Expr::Value(Value::F64(value.to_f64()?.ceil()), attrs),
                UnOp::F64Floor => Expr::Value(Value::F64(value.to_f64()?.floor()), attrs),
                UnOp::F64Trunc => Expr::Value(Value::F64(value.to_f64()?.trunc()), attrs),
                UnOp::F64Nearest => Expr::Value(Value::F64(value.to_f64()?.nearest()), attrs),

                UnOp::I32Eqz => Expr::Value(
                    Value::I32((value.to_i32()? == 0) as i32),
                    Default::default(),
                ),
                UnOp::I64Eqz => Expr::Value(
                    Value::I32((value.to_i64()? == 0) as i32),
                    Default::default(),
                ),

                UnOp::I32WrapI64 => Expr::Value(Value::I32(value.to_i64()? as i32), attrs),

                UnOp::I64ExtendI32S => Expr::Value(Value::I64(value.to_i32()? as i64), attrs),
                UnOp::I64ExtendI32U => Expr::Value(Value::I64(value.to_u32()? as i64), attrs),

                UnOp::I32TruncF32S => Expr::Value(Value::I32(value.to_f32()?.trunc_i32()), attrs),
                UnOp::I32TruncF32U => {
                    Expr::Value(Value::I32(value.to_f32()?.trunc_u32() as i32), attrs)
                }
                UnOp::I32TruncF64S => Expr::Value(Value::I32(value.to_f64()?.trunc_i32()), attrs),
                UnOp::I32TruncF64U => {
                    Expr::Value(Value::I32(value.to_f64()?.trunc_u32() as i32), attrs)
                }

                UnOp::I64TruncF32S => Expr::Value(Value::I64(value.to_f32()?.trunc_i64()), attrs),
                UnOp::I64TruncF32U => {
                    Expr::Value(Value::I64(value.to_f32()?.trunc_u64() as i64), attrs)
                }
                UnOp::I64TruncF64S => Expr::Value(Value::I64(value.to_f64()?.trunc_i64()), attrs),
                UnOp::I64TruncF64U => {
                    Expr::Value(Value::I64(value.to_f64()?.trunc_u64() as i64), attrs)
                }

                UnOp::F32DemoteF64 => Expr::Value(Value::F32(value.to_f64()?.demote()), attrs),
                UnOp::F64PromoteF32 => Expr::Value(Value::F64(value.to_f32()?.promote()), attrs),

                UnOp::F32ConvertI32S => {
                    Expr::Value(Value::F32((value.to_i32()? as f32).into()), attrs)
                }
                UnOp::F32ConvertI32U => {
                    Expr::Value(Value::F32((value.to_u32()? as f32).into()), attrs)
                }
                UnOp::F32ConvertI64S => {
                    Expr::Value(Value::F32((value.to_i64()? as f32).into()), attrs)
                }
                UnOp::F32ConvertI64U => {
                    Expr::Value(Value::F32((value.to_u64()? as f32).into()), attrs)
                }

                UnOp::F64ConvertI32S => {
                    Expr::Value(Value::F64((value.to_i32()? as f64).into()), attrs)
                }
                UnOp::F64ConvertI32U => {
                    Expr::Value(Value::F64((value.to_u32()? as f64).into()), attrs)
                }
                UnOp::F64ConvertI64S => {
                    Expr::Value(Value::F64((value.to_i64()? as f64).into()), attrs)
                }
                UnOp::F64ConvertI64U => {
                    Expr::Value(Value::F64((value.to_u64()? as f64).into()), attrs)
                }

                UnOp::F32ReinterpretI32 => {
                    Expr::Value(Value::F32(F32::from_bits(value.to_u32()?)), attrs)
                }
                UnOp::F64ReinterpretI64 => {
                    Expr::Value(Value::F64(F64::from_bits(value.to_u64()?)), attrs)
                }
                UnOp::I32ReinterpretF32 => {
                    Expr::Value(Value::I32(value.to_f32()?.to_bits() as i32), attrs)
                }
                UnOp::I64ReinterpretF64 => {
                    Expr::Value(Value::I64(value.to_f64()?.to_bits() as i64), attrs)
                }

                UnOp::I32Extend8S => Expr::Value(Value::I32(value.to_i32()? as i8 as i32), attrs),
                UnOp::I32Extend16S => Expr::Value(Value::I32(value.to_i32()? as i16 as i32), attrs),

                UnOp::I64Extend8S => Expr::Value(Value::I64(value.to_i64()? as i8 as i64), attrs),
                UnOp::I64Extend16S => Expr::Value(Value::I64(value.to_i64()? as i16 as i64), attrs),
                UnOp::I64Extend32S => Expr::Value(Value::I64(value.to_i64()? as i32 as i64), attrs),

                UnOp::LocalTee(local_id) => {
                    info.exit_env.locals.insert(local_id, (value, attrs));

                    Expr::Value(value, attrs)
                }

                UnOp::Load(mem_arg, load) => {
                    if !attrs.contains(ValueAttrs::CONST_PTR) {
                        return None;
                    }

                    let base_addr = value.to_u32()?;
                    let start = (base_addr + mem_arg.offset) as usize;
                    let range = start..start + load.src_size();

                    let bytes = match self.interp.module.get_mem(mem_arg.mem_id, range) {
                        Ok(bytes) => bytes,

                        Err(e) => {
                            warn!("encountered an error while specializing a constant memory load: {e}");

                            return None;
                        }
                    };

                    Expr::Value(load.load(bytes), attrs.deref_attrs())
                }

                UnOp::MemoryGrow(_mem_id) => return None,
            })
        };

        stack.push(try_process().unwrap_or_else(|| Expr::Unary(op, Box::new(arg))));

        Ok(())
    }

    fn process_bin_op(
        &mut self,
        _block_id: BlockId,
        stack: &mut Vec<Expr>,
        op: BinOp,
    ) -> Result<()> {
        let mut args: [_; 2] = array::from_fn(|_| stack.pop().unwrap());
        args.reverse();

        let try_process = || -> Option<Expr> {
            let &[Expr::Value(lhs, lhs_attrs), Expr::Value(rhs, rhs_attrs)] = &args else {
                return None;
            };
            let meet_attrs = lhs_attrs.meet(&rhs_attrs);

            Some(match op {
                BinOp::I32Add => Expr::Value(
                    Value::I32(lhs.to_i32()?.wrapping_add(rhs.to_i32()?)),
                    lhs_attrs.addsub_attrs(&rhs_attrs),
                ),
                BinOp::I32Sub => Expr::Value(
                    Value::I32(lhs.to_i32()?.wrapping_sub(rhs.to_i32()?)),
                    lhs_attrs.addsub_attrs(&rhs_attrs),
                ),
                BinOp::I32Mul => Expr::Value(
                    Value::I32(lhs.to_i32()?.wrapping_mul(rhs.to_i32()?)),
                    meet_attrs,
                ),
                BinOp::I32DivS => match (lhs.to_i32()?, rhs.to_i32()?) {
                    (_, 0) => {
                        warn!("encountered division by zero during specialization");
                        // TODO: stop processing the block body and terminate it with a trap
                        return None;
                    }
                    (lhs, -1) if lhs == i32::MIN => {
                        warn!("encountered an overflow while performing signed division");
                        // TODO: see above
                        return None;
                    }
                    (lhs, rhs) => Expr::Value(Value::I32(lhs / rhs), meet_attrs),
                },
                BinOp::I32DivU => match (lhs.to_u32()?, rhs.to_u32()?) {
                    (_, 0) => {
                        warn!("encountered division by zero during specialization");
                        // TODO: see above
                        return None;
                    }
                    (lhs, rhs) => Expr::Value(Value::I32((lhs / rhs) as i32), meet_attrs),
                },
                BinOp::I32RemS => match (lhs.to_i32()?, rhs.to_i32()?) {
                    (_, 0) => {
                        warn!("trying to take the remainder of division by zero during specialization");
                        // TODO: see above
                        return None;
                    }
                    (lhs, rhs) => Expr::Value(Value::I32(lhs % rhs), meet_attrs),
                },
                BinOp::I32RemU => match (lhs.to_u32()?, rhs.to_u32()?) {
                    (_, 0) => {
                        warn!("trying to take the remainder of division by zero during specialization");
                        // TODO: see above
                        return None;
                    }
                    (lhs, rhs) => Expr::Value(Value::I32((lhs % rhs) as i32), meet_attrs),
                },
                BinOp::I32And => Expr::Value(Value::I32(lhs.to_i32()? & rhs.to_i32()?), meet_attrs),
                BinOp::I32Or => Expr::Value(Value::I32(lhs.to_i32()? | rhs.to_i32()?), meet_attrs),
                BinOp::I32Xor => Expr::Value(Value::I32(lhs.to_i32()? ^ rhs.to_i32()?), meet_attrs),
                BinOp::I32Shl => Expr::Value(
                    Value::I32(lhs.to_i32()?.wrapping_shl(rhs.to_u32()?)),
                    meet_attrs,
                ),
                BinOp::I32ShrS => Expr::Value(
                    Value::I32(lhs.to_i32()?.wrapping_shr(rhs.to_u32()?)),
                    meet_attrs,
                ),
                BinOp::I32ShrU => Expr::Value(
                    Value::I32(lhs.to_u32()?.wrapping_shr(rhs.to_u32()?) as i32),
                    meet_attrs,
                ),
                BinOp::I32Rotl => Expr::Value(
                    Value::I32(lhs.to_i32()?.rotate_left(rhs.to_u32()?)),
                    meet_attrs,
                ),
                BinOp::I32Rotr => Expr::Value(
                    Value::I32(lhs.to_i32()?.rotate_right(rhs.to_u32()?)),
                    meet_attrs,
                ),

                BinOp::I64Add => Expr::Value(
                    Value::I64(lhs.to_i64()?.wrapping_add(rhs.to_i64()?)),
                    meet_attrs,
                ),
                BinOp::I64Sub => Expr::Value(
                    Value::I64(lhs.to_i64()?.wrapping_sub(rhs.to_i64()?)),
                    meet_attrs,
                ),
                BinOp::I64Mul => Expr::Value(
                    Value::I64(lhs.to_i64()?.wrapping_mul(rhs.to_i64()?)),
                    meet_attrs,
                ),
                BinOp::I64DivS => match (lhs.to_i64()?, rhs.to_i64()?) {
                    (_, 0) => {
                        warn!("encountered division by zero during specialization");
                        // TODO: see above
                        return None;
                    }
                    (lhs, -1) if lhs == i64::MIN => {
                        warn!("encountered an overflow while performing signed division");
                        // TODO: see above
                        return None;
                    }
                    (lhs, rhs) => Expr::Value(Value::I64(lhs / rhs), meet_attrs),
                },
                BinOp::I64DivU => match (lhs.to_u64()?, rhs.to_u64()?) {
                    (_, 0) => {
                        warn!("encountered division by zero during specialization");
                        // TODO: see above
                        return None;
                    }
                    (lhs, rhs) => Expr::Value(Value::I64((lhs / rhs) as i64), meet_attrs),
                },
                BinOp::I64RemS => match (lhs.to_i64()?, rhs.to_i64()?) {
                    (_, 0) => {
                        warn!("trying to take the remainder of division by zero during specialization");
                        // TODO: see above
                        return None;
                    }
                    (lhs, rhs) => Expr::Value(Value::I64(lhs % rhs), meet_attrs),
                },
                BinOp::I64RemU => match (lhs.to_i64()?, rhs.to_i64()?) {
                    (_, 0) => {
                        warn!("trying to take the remainder of division by zero during specialization");
                        // TODO: see above
                        return None;
                    }
                    (lhs, rhs) => Expr::Value(Value::I64((lhs % rhs) as i64), meet_attrs),
                },
                BinOp::I64And => Expr::Value(Value::I64(lhs.to_i64()? & rhs.to_i64()?), meet_attrs),
                BinOp::I64Or => Expr::Value(Value::I64(lhs.to_i64()? | rhs.to_i64()?), meet_attrs),
                BinOp::I64Xor => Expr::Value(Value::I64(lhs.to_i64()? ^ rhs.to_i64()?), meet_attrs),
                BinOp::I64Shl => Expr::Value(
                    Value::I64(lhs.to_i64()?.wrapping_shl(rhs.to_u64()? as u32)),
                    meet_attrs,
                ),
                BinOp::I64ShrS => Expr::Value(
                    Value::I64(lhs.to_i64()?.wrapping_shr(rhs.to_u64()? as u32)),
                    meet_attrs,
                ),
                BinOp::I64ShrU => Expr::Value(
                    Value::I64(lhs.to_u64()?.wrapping_shr(rhs.to_u64()? as u32) as i64),
                    meet_attrs,
                ),
                BinOp::I64Rotl => Expr::Value(
                    Value::I64(lhs.to_i64()?.rotate_left(rhs.to_u64()? as u32)),
                    meet_attrs,
                ),
                BinOp::I64Rotr => Expr::Value(
                    Value::I64(lhs.to_i64()?.rotate_right(rhs.to_u64()? as u32)),
                    meet_attrs,
                ),

                BinOp::F32Add => Expr::Value(Value::F32(lhs.to_f32()? + rhs.to_f32()?), meet_attrs),
                BinOp::F32Sub => Expr::Value(Value::F32(lhs.to_f32()? - rhs.to_f32()?), meet_attrs),
                BinOp::F32Mul => Expr::Value(Value::F32(lhs.to_f32()? * rhs.to_f32()?), meet_attrs),
                BinOp::F32Div => Expr::Value(Value::F32(lhs.to_f32()? / rhs.to_f32()?), meet_attrs),
                BinOp::F32Min => {
                    Expr::Value(Value::F32(lhs.to_f32()?.min(rhs.to_f32()?)), meet_attrs)
                }
                BinOp::F32Max => {
                    Expr::Value(Value::F32(lhs.to_f32()?.max(rhs.to_f32()?)), meet_attrs)
                }
                BinOp::F32Copysign => Expr::Value(
                    Value::F32(lhs.to_f32()?.copysign(rhs.to_f32()?)),
                    meet_attrs,
                ),

                BinOp::F64Add => Expr::Value(Value::F64(lhs.to_f64()? + rhs.to_f64()?), meet_attrs),
                BinOp::F64Sub => Expr::Value(Value::F64(lhs.to_f64()? - rhs.to_f64()?), meet_attrs),
                BinOp::F64Mul => Expr::Value(Value::F64(lhs.to_f64()? * rhs.to_f64()?), meet_attrs),
                BinOp::F64Div => Expr::Value(Value::F64(lhs.to_f64()? / rhs.to_f64()?), meet_attrs),
                BinOp::F64Min => {
                    Expr::Value(Value::F64(lhs.to_f64()?.min(rhs.to_f64()?)), meet_attrs)
                }
                BinOp::F64Max => {
                    Expr::Value(Value::F64(lhs.to_f64()?.max(rhs.to_f64()?)), meet_attrs)
                }
                BinOp::F64Copysign => Expr::Value(
                    Value::F64(lhs.to_f64()?.copysign(rhs.to_f64()?)),
                    meet_attrs,
                ),

                BinOp::I32Eq => Expr::Value(
                    Value::I32((lhs.to_i32()? == rhs.to_i32()?) as i32),
                    Default::default(),
                ),
                BinOp::I32Ne => Expr::Value(
                    Value::I32((lhs.to_i32()? != rhs.to_i32()?) as i32),
                    Default::default(),
                ),
                BinOp::I32LtS => Expr::Value(
                    Value::I32((lhs.to_i32()? < rhs.to_i32()?) as i32),
                    Default::default(),
                ),
                BinOp::I32LtU => Expr::Value(
                    Value::I32((lhs.to_u32()? < rhs.to_u32()?) as i32),
                    Default::default(),
                ),
                BinOp::I32GtS => Expr::Value(
                    Value::I32((lhs.to_i32()? > rhs.to_i32()?) as i32),
                    Default::default(),
                ),
                BinOp::I32GtU => Expr::Value(
                    Value::I32((lhs.to_u32()? > rhs.to_u32()?) as i32),
                    Default::default(),
                ),
                BinOp::I32LeS => Expr::Value(
                    Value::I32((lhs.to_i32()? <= rhs.to_i32()?) as i32),
                    Default::default(),
                ),
                BinOp::I32LeU => Expr::Value(
                    Value::I32((lhs.to_u32()? <= rhs.to_u32()?) as i32),
                    Default::default(),
                ),
                BinOp::I32GeS => Expr::Value(
                    Value::I32((lhs.to_i32()? >= rhs.to_i32()?) as i32),
                    Default::default(),
                ),
                BinOp::I32GeU => Expr::Value(
                    Value::I32((lhs.to_u32()? >= rhs.to_u32()?) as i32),
                    Default::default(),
                ),

                BinOp::I64Eq => Expr::Value(
                    Value::I32((lhs.to_i64()? == rhs.to_i64()?) as i32),
                    Default::default(),
                ),
                BinOp::I64Ne => Expr::Value(
                    Value::I32((lhs.to_i64()? != rhs.to_i64()?) as i32),
                    Default::default(),
                ),
                BinOp::I64LtS => Expr::Value(
                    Value::I32((lhs.to_i64()? < rhs.to_i64()?) as i32),
                    Default::default(),
                ),
                BinOp::I64LtU => Expr::Value(
                    Value::I32((lhs.to_u64()? < rhs.to_u64()?) as i32),
                    Default::default(),
                ),
                BinOp::I64GtS => Expr::Value(
                    Value::I32((lhs.to_i64()? > rhs.to_i64()?) as i32),
                    Default::default(),
                ),
                BinOp::I64GtU => Expr::Value(
                    Value::I32((lhs.to_u64()? > rhs.to_u64()?) as i32),
                    Default::default(),
                ),
                BinOp::I64LeS => Expr::Value(
                    Value::I32((lhs.to_i64()? <= rhs.to_i64()?) as i32),
                    Default::default(),
                ),
                BinOp::I64LeU => Expr::Value(
                    Value::I32((lhs.to_u64()? <= rhs.to_u64()?) as i32),
                    Default::default(),
                ),
                BinOp::I64GeS => Expr::Value(
                    Value::I32((lhs.to_i64()? >= rhs.to_i64()?) as i32),
                    Default::default(),
                ),
                BinOp::I64GeU => Expr::Value(
                    Value::I32((lhs.to_u64()? >= rhs.to_u64()?) as i32),
                    Default::default(),
                ),

                BinOp::F32Eq => Expr::Value(
                    Value::I32((lhs.to_f32()? == rhs.to_f32()?) as i32),
                    Default::default(),
                ),
                BinOp::F32Ne => Expr::Value(
                    Value::I32((lhs.to_f32()? != rhs.to_f32()?) as i32),
                    Default::default(),
                ),
                BinOp::F32Lt => Expr::Value(
                    Value::I32((lhs.to_f32()? < rhs.to_f32()?) as i32),
                    Default::default(),
                ),
                BinOp::F32Gt => Expr::Value(
                    Value::I32((lhs.to_f32()? > rhs.to_f32()?) as i32),
                    Default::default(),
                ),
                BinOp::F32Le => Expr::Value(
                    Value::I32((lhs.to_f32()? <= rhs.to_f32()?) as i32),
                    Default::default(),
                ),
                BinOp::F32Ge => Expr::Value(
                    Value::I32((lhs.to_f32()? >= rhs.to_f32()?) as i32),
                    Default::default(),
                ),

                BinOp::F64Eq => Expr::Value(
                    Value::I32((lhs.to_f64()? == rhs.to_f64()?) as i32),
                    Default::default(),
                ),
                BinOp::F64Ne => Expr::Value(
                    Value::I32((lhs.to_f64()? != rhs.to_f64()?) as i32),
                    Default::default(),
                ),
                BinOp::F64Lt => Expr::Value(
                    Value::I32((lhs.to_f64()? < rhs.to_f64()?) as i32),
                    Default::default(),
                ),
                BinOp::F64Gt => Expr::Value(
                    Value::I32((lhs.to_f64()? > rhs.to_f64()?) as i32),
                    Default::default(),
                ),
                BinOp::F64Le => Expr::Value(
                    Value::I32((lhs.to_f64()? <= rhs.to_f64()?) as i32),
                    Default::default(),
                ),
                BinOp::F64Ge => Expr::Value(
                    Value::I32((lhs.to_f64()? >= rhs.to_f64()?) as i32),
                    Default::default(),
                ),
            })
        };

        stack.push(try_process().unwrap_or_else(|| Expr::Binary(op, Box::new(args))));

        Ok(())
    }

    fn process_tern_op(
        &mut self,
        _block_id: BlockId,
        stack: &mut Vec<Expr>,
        op: TernOp,
    ) -> Result<()> {
        let [arg2, arg1, arg0] = array::from_fn(|_| stack.pop().unwrap());

        let result = 'result: {
            match op {
                TernOp::Select => match arg2 {
                    Expr::Value(Value::I32(0), _) => break 'result arg1,
                    Expr::Value(Value::I32(_), _) => break 'result arg0,
                    _ => {}
                },
            }

            Expr::Ternary(op, Box::new([arg0, arg1, arg2]))
        };

        stack.push(result);

        Ok(())
    }

    fn process_call(
        &mut self,
        block_id: BlockId,
        _stmt_idx: &mut usize,
        ret_local_id: Option<LocalId>,
        func_id: FuncId,
        args: Vec<Expr>,
    ) -> Result<()> {
        if let Some(intrinsic) =
            self.interp.module.funcs[func_id].get_intrinsic(&self.interp.module)
        {
            match intrinsic {
                IntrinsicDecl::Specialize => {
                    self.process_intr_specialize(block_id, ret_local_id.unwrap())?
                }
                IntrinsicDecl::Unknown => {
                    self.process_intr_unknown(block_id, ret_local_id.unwrap(), func_id)?
                }
            }
        } else {
            // TODO: inlining
            self.flush_globals(block_id);
            self.func.blocks[block_id]
                .body
                .push(Stmt::Call(Call::Direct {
                    ret_local_id,
                    func_id,
                    args,
                }));
            self.assume_clobbered_globals(block_id);
        }

        Ok(())
    }

    fn process_intr_specialize(&mut self, block_id: BlockId, ret_local_id: LocalId) -> Result<()> {
        warn!("encountered stitch/specialize during specialization");
        self.func.blocks[block_id].body.push(Stmt::LocalSet(
            ret_local_id,
            Expr::Value(Value::I32(0), Default::default()),
        ));

        Ok(())
    }

    fn process_intr_unknown(
        &mut self,
        block_id: BlockId,
        ret_local_id: LocalId,
        func_id: FuncId,
    ) -> Result<()> {
        let val_ty = self.interp.module.funcs[func_id].ty().ret.as_ref().unwrap();
        self.func.blocks[block_id].body.push(Stmt::LocalSet(
            ret_local_id,
            Expr::Value(Value::default_for(val_ty), Default::default()),
        ));

        Ok(())
    }

    fn get_branch_target(&mut self, orig_block_id: OrigBlockId, env: Env) -> BlockId {
        let is_loop_header = self.loop_headers.contains(&orig_block_id);

        if is_loop_header {
            if let Some(&Loop::Kept(block_id)) = self.loops.get(&orig_block_id) {
                return block_id;
            }
        }

        let block_map_key = (orig_block_id, env.to_hashable());

        if let Some(&block_id) = self.block_map.get(&block_map_key) {
            return block_id;
        }

        let block_id = self.func.blocks.insert(Default::default());
        self.blocks.insert(
            block_id,
            BlockInfo {
                orig_block_id,
                entry_env: env.clone(),
                exit_env: Default::default(),
            },
        );

        if is_loop_header && !self.should_unroll(orig_block_id) {
            self.loops.insert(orig_block_id, Loop::Kept(block_id));
        } else {
            self.block_map.insert(block_map_key, block_id);
        }

        self.tasks.insert(Task(block_id));

        block_id
    }

    fn process_branch(&mut self, orig_block_id: OrigBlockId, env: Env) -> BlockId {
        let block_id = self.get_branch_target(orig_block_id, env.clone());

        if self.blocks[block_id].entry_env.merge(&env) {
            self.tasks.insert(Task(block_id));
        }

        block_id
    }

    fn should_unroll(&self, orig_block_id: OrigBlockId) -> bool {
        debug_assert!(self.loop_headers.contains(&orig_block_id));

        // TODO
        false
    }

    fn flush_globals(&mut self, block_id: BlockId) {
        let info = &self.blocks[block_id];
        let flushed_globals = info
            .exit_env
            .globals
            .iter()
            .filter(|&(global_id, (exit_value, _))| {
                !info
                    .entry_env
                    .globals
                    .get(global_id)
                    .is_some_and(|(entry_value, _)| exit_value == entry_value)
            })
            .collect::<Vec<_>>();

        for (global_id, &(value, attrs)) in flushed_globals {
            self.func.blocks[block_id]
                .body
                .push(Stmt::GlobalSet(global_id, Expr::Value(value, attrs)));
        }
    }

    fn assume_clobbered_globals(&mut self, block_id: BlockId) {
        let info = &mut self.blocks[block_id];
        let clobbered_global_ids = info
            .exit_env
            .globals
            .keys()
            .filter(|&global_id| self.interp.module.globals[global_id].ty.mutable)
            .collect::<Vec<_>>();

        for global_id in clobbered_global_ids {
            info.exit_env.globals.remove(global_id);
        }
    }
}
