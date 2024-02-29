use std::ops::Neg;
use std::{array, fmt, iter, mem};

use anyhow::{ensure, Result};
use hashbrown::{HashMap, HashSet};
use log::{trace, warn};
use slotmap::{new_key_type, Key, SecondaryMap, SlotMap};

use crate::ast::expr::{Value, ValueAttrs};
use crate::ast::{ConstExpr, FuncId, GlobalDef, GlobalId, IntrinsicDecl, Module, TableDef};
use crate::cfg::{
    BinOp, Block, BlockId, Call, Expr, FuncBody, LocalId, NulOp, Stmt, Terminator, TernOp, UnOp,
};
use crate::util::float::{F32, F64};

use super::{Interpreter, SpecSignature};

new_key_type! {
    struct FuncCtxId;
}

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
    fn entry<F>(
        module: &Module,
        cfg: &FuncBody,
        args: &[Option<(Value, ValueAttrs)>],
        const_global_ids: &HashSet<GlobalId>,
        map_local: F,
    ) -> Self
    where
        F: Fn(LocalId) -> LocalId,
    {
        let globals = module
            .globals
            .iter()
            .filter_map(|(global_id, global)| match global.def {
                GlobalDef::Import(_) => None,
                GlobalDef::Value(ConstExpr::Value(value, attrs)) => {
                    if !global.ty.mutable || const_global_ids.contains(&global_id) {
                        Some((global_id, (value, attrs)))
                    } else {
                        None
                    }
                }
                GlobalDef::Value(ConstExpr::GlobalGet(_)) => None,
            })
            .collect();

        let mut locals: SecondaryMap<_, _> = cfg
            .locals
            .iter()
            .map(|(local_id, val_ty)| {
                (
                    map_local(local_id),
                    (Value::default_for(val_ty), Default::default()),
                )
            })
            .collect();

        for (local_id, arg) in cfg.params.iter().copied().map(map_local).zip(args) {
            if let &Some(value) = arg {
                locals[local_id] = value;
            } else {
                locals.remove(local_id);
            }
        }

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
                    if *lhs != new {
                        trace!("merge_maps: {id:?} changed");
                    }

                    changed = *lhs != new || changed;
                    *lhs = new;
                    *lhs_attrs = new_attrs;
                } else {
                    trace!("merge_maps: {id:?} removed");
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum BlockKind {
    Regular,
    Canonical,
    Return(FuncCtxId),
}

#[derive(Debug, Clone)]
struct BlockInfo {
    func_ctx_id: FuncCtxId,
    orig_block_id: OrigBlockId,
    start_stmt_idx: usize,
    kind: BlockKind,
    entry_env: Env,
    exit_env: Env,
}

impl BlockInfo {
    fn block_map_key(&self) -> BlockMapKey {
        let &Self {
            func_ctx_id,
            orig_block_id,
            ref entry_env,
            start_stmt_idx,
            ..
        } = self;

        match self.kind {
            BlockKind::Regular => BlockMapKey::Regular {
                func_ctx_id,
                orig_block_id,
                entry_env: entry_env.to_hashable(),
                start_stmt_idx,
            },

            BlockKind::Canonical => BlockMapKey::Canonical {
                func_ctx_id,
                orig_block_id,
                start_stmt_idx,
            },

            BlockKind::Return(from_func_ctx_id) => BlockMapKey::Return {
                func_ctx_id,
                orig_block_id,
                start_stmt_idx,
                from_func_ctx_id,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Task(BlockId);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum BlockMapKey {
    Regular {
        func_ctx_id: FuncCtxId,
        orig_block_id: OrigBlockId,
        entry_env: HashableEnv,
        start_stmt_idx: usize,
    },

    Canonical {
        func_ctx_id: FuncCtxId,
        orig_block_id: OrigBlockId,
        start_stmt_idx: usize,
    },

    Return {
        func_ctx_id: FuncCtxId,
        orig_block_id: OrigBlockId,
        start_stmt_idx: usize,
        from_func_ctx_id: FuncCtxId,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum InlineMode {
    /// Emit a call to a function.
    No,

    /// Emit a call to a specialized function.
    Outline,

    /// Inline a function body.
    Inline,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
enum InlineDisposition {
    #[default]
    Allow,

    Force,
    Deny,
}

impl From<ValueAttrs> for InlineDisposition {
    fn from(attrs: ValueAttrs) -> Self {
        if attrs.contains(ValueAttrs::NO_INLINE) {
            Self::Deny
        } else if attrs.contains(ValueAttrs::INLINE) {
            Self::Force
        } else {
            Self::Allow
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ParentCtx {
    func_ctx_id: FuncCtxId,
    call_block_id: BlockId,
    ret_local_id: Option<LocalId>,
    ret_orig_block_id: OrigBlockId,
    ret_start_stmt_idx: usize,
    ret_canonical: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FuncCtx {
    parent: Option<ParentCtx>,
    func_id: FuncId,
    shape: Vec<bool>,
}

pub struct Specializer<'a, 'i> {
    interp: &'i mut Interpreter<'a>,
    orig_func_id: FuncId,
    func: FuncBody,
    block_map: HashMap<BlockMapKey, BlockId>,
    blocks: SecondaryMap<BlockId, BlockInfo>,
    args: Vec<Option<(Value, ValueAttrs)>>,
    tasks: HashSet<Task>,
    local_map: HashMap<(FuncCtxId, LocalId), LocalId>,
    func_ctxs: SlotMap<FuncCtxId, FuncCtx>,
    func_ctx_map: HashMap<FuncCtx, FuncCtxId>,
}

impl<'a, 'i> Specializer<'a, 'i> {
    pub fn new(
        interp: &'i mut Interpreter<'a>,
        SpecSignature { orig_func_id, args }: SpecSignature,
        mut func: FuncBody,
    ) -> Self {
        func.blocks.clear();

        Self {
            interp,
            orig_func_id,
            func,
            block_map: Default::default(),
            blocks: Default::default(),
            args,
            tasks: Default::default(),
            local_map: Default::default(),
            func_ctxs: Default::default(),
            func_ctx_map: Default::default(),
        }
    }

    pub fn run(mut self) -> Result<FuncBody> {
        let func_ctx_id = self.add_func_ctx(FuncCtx {
            parent: None,
            func_id: self.orig_func_id,
            shape: self.args.iter().map(Option::is_some).collect(),
        });

        self.process_params(func_ctx_id);
        let cfg = self.interp.get_cfg(self.orig_func_id).unwrap();
        self.func.entry = self.process_branch(
            None,
            func_ctx_id,
            cfg.entry,
            Env::entry(
                &self.interp.module,
                &cfg,
                &self.args,
                &self.interp.const_global_ids,
                |local_id| self.local_map[&(func_ctx_id, local_id)],
            ),
            0,
            BlockKind::Regular,
        );

        while let Some(&Task(block_id)) = self.tasks.iter().next() {
            self.tasks.remove(&Task(block_id));
            self.process_block(block_id)?;
        }

        self.flush_envs();
        self.func.remove_unreachable_blocks();
        self.func.merge_blocks();
        self.set_block_name(self.func.entry, format_args!("entry"));

        Ok(self.func)
    }

    fn process_params(&mut self, orig_func_ctx_id: FuncCtxId) {
        let cfg = self.interp.get_cfg(self.orig_func_id).unwrap();
        self.process_locals(orig_func_ctx_id);
        self.func.params = cfg
            .params
            .iter()
            .enumerate()
            .filter(|&(idx, _)| self.args[idx].is_none())
            .map(|(_, &local_id)| self.local_map[&(orig_func_ctx_id, local_id)])
            .collect();
    }

    fn process_locals(&mut self, func_ctx_id: FuncCtxId) {
        let func_id = self.func_ctxs[func_ctx_id].func_id;
        let cfg = self.interp.get_cfg(func_id).unwrap();

        for (local_id, val_ty) in &cfg.locals {
            self.local_map
                .entry((func_ctx_id, local_id))
                .or_insert_with(|| self.func.locals.insert(val_ty.clone()));
        }
    }

    fn process_block(&mut self, block_id: BlockId) -> Result<()> {
        self.func.blocks[block_id].body.clear();
        let info = &mut self.blocks[block_id];
        info.exit_env = info.entry_env.clone();
        let info = &self.blocks[block_id];
        let orig_block_id = info.orig_block_id;
        let func_id = self.func_ctxs[info.func_ctx_id].func_id;
        let cfg = self.interp.get_cfg(func_id).unwrap();
        let mut stmt_idx = info.start_stmt_idx;
        let ref mut stack = vec![];

        trace!(
            "specializing a block {func_id:?}/{orig_block_id:?} (function {:?}, to {block_id:?}, kind {:?})",
            self.interp.module.funcs[func_id].name(),
            info.kind,
        );
        self.set_block_name(
            block_id,
            format_args!(
                "spec:{orig_block_id:?}.{}{}",
                format!("{:?}", info.kind).to_lowercase(),
                if stmt_idx > 0 {
                    format!(":{stmt_idx}")
                } else {
                    String::new()
                },
            ),
        );
        let info = &mut self.blocks[block_id];
        trace!(
            "starting from stmt {stmt_idx}/{}",
            cfg.blocks[orig_block_id].body.len()
        );
        assert!(stmt_idx <= cfg.blocks[orig_block_id].body.len());
        trace!("locals:");

        for (local_id, &(value, attrs)) in &info.exit_env.locals {
            trace!("  {local_id:?} = {}", Expr::Value(value, attrs));
        }

        trace!("globals:");

        for (global_id, &(value, attrs)) in &info.exit_env.globals {
            trace!("  {global_id:?} = {}", Expr::Value(value, attrs));
        }

        let mut skip_terminator = false;

        'body: while let Some(stmt) = cfg.blocks[orig_block_id].body.get(stmt_idx) {
            trace!(
                "processing stmt {stmt_idx}/{}: {stmt}",
                cfg.blocks[orig_block_id].body.len()
            );
            stmt_idx += 1;
            debug_assert!(stack.is_empty());

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
                        trace!("emitting Stmt::Drop for {expr}: has a side effect");
                        block.body.push(Stmt::Drop(expr));
                    } else {
                        trace!("dropping {expr} from the output block: side-effect free");
                    }
                }

                Stmt::LocalSet(local_id, _) => {
                    let local_id = self.local_map[&(info.func_ctx_id, local_id)];
                    let expr = stack.pop().unwrap();
                    self.set_local(block_id, local_id, expr);
                }

                Stmt::GlobalSet(global_id, _) => {
                    let expr = stack.pop().unwrap();

                    if let Some(value) = expr.to_value() {
                        trace!("set exit_env.globals[{global_id:?}] <- {expr}");
                        info.exit_env.globals.insert(global_id, value);
                    } else {
                        trace!("forgetting exit_env.globals[{global_id:?}]: {expr} is not a value");
                        info.exit_env.globals.remove(global_id);
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
                    let ret_local_id = ret_local_id
                        .map(|ret_local_id| self.local_map[&(info.func_ctx_id, ret_local_id)]);
                    let args = stack.drain(stack.len() - args.len()..).collect();

                    if self.process_call(
                        block_id,
                        stmt_idx,
                        ret_local_id,
                        func_id,
                        args,
                        Default::default(),
                    )? {
                        skip_terminator = true;
                        break;
                    }
                }

                Stmt::Call(Call::Indirect {
                    ret_local_id,
                    ty_id,
                    table_id,
                    ref args,
                    ..
                }) => {
                    let ret_local_id = ret_local_id
                        .map(|ret_local_id| self.local_map[&(info.func_ctx_id, ret_local_id)]);
                    let mut index = Some(stack.pop().unwrap());
                    let mut args = stack.drain(stack.len() - args.len()..).collect();

                    'spec: {
                        let Some((idx, idx_attrs)) = index.as_ref().unwrap().to_value() else {
                            trace!(
                                "could not resolve an indirect call target: index {} is not a value",
                                index.as_ref().unwrap(),
                            );
                            break 'spec;
                        };

                        match &self.interp.module.tables[table_id].def {
                            TableDef::Import(_) => {}

                            TableDef::Elems(elems) => {
                                let idx = idx.to_u32().unwrap();
                                let Some(&Some(func_id)) = elems.get(idx as usize) else {
                                    trace!(
                                        "could not resolve an indirect call target: \
                                        index {idx} is out of bounds"
                                    );
                                    break 'spec;
                                };

                                let actual_func_ty = self.interp.module.funcs[func_id].ty();
                                let claimed_func_ty = self.interp.module.types[ty_id].as_func();

                                if actual_func_ty != claimed_func_ty {
                                    trace!(
                                        "could not resolve an indirect call target: \
                                        claimed function type ({claimed_func_ty}) differs from \
                                        the actual type {actual_func_ty}"
                                    );
                                    break 'spec;
                                }

                                index = None;
                                let args = mem::take(&mut args);

                                if self.process_call(
                                    block_id,
                                    stmt_idx,
                                    ret_local_id,
                                    func_id,
                                    args,
                                    InlineDisposition::from(idx_attrs),
                                )? {
                                    skip_terminator = true;
                                    break 'body;
                                }

                                block = &mut self.func.blocks[block_id];
                            }
                        }
                    }

                    if let Some(index) = index {
                        trace!("emitting an indirect call");
                        block.body.push(Stmt::Call(Call::Indirect {
                            ret_local_id,
                            ty_id,
                            table_id,
                            args,
                            index: Box::new(index),
                        }));

                        if let Some(ret_local_id) = ret_local_id {
                            trace!(
                                "forgetting exit_env.locals[{ret_local_id:?}]: \
                                holds the return value of an indirect call"
                            );
                            self.blocks[block_id].exit_env.locals.remove(ret_local_id);
                        }
                    }
                }
            }
        }

        if skip_terminator {
            trace!("skipping the block's terminator");
        } else {
            trace!(
                "processing a terminator: {}",
                cfg.blocks[orig_block_id].term
            );

            match &cfg.blocks[orig_block_id].term {
                Terminator::Trap | Terminator::Br(_) | Terminator::Return(None) => {}
                Terminator::If(expr, _) => self.process_expr(block_id, stack, expr)?,
                Terminator::Switch(expr, _) => self.process_expr(block_id, stack, expr)?,
                Terminator::Return(Some(expr)) => self.process_expr(block_id, stack, expr)?,
            }

            let info = &mut self.blocks[block_id];
            let func_ctx_id = info.func_ctx_id;

            match cfg.blocks[orig_block_id].term {
                Terminator::Trap => {
                    self.func.blocks[block_id].term = Terminator::Trap;
                }

                Terminator::Br(orig_target_block_id) => {
                    let exit_env = info.exit_env.clone();
                    let target_block_id = self.process_branch(
                        Some(block_id),
                        func_ctx_id,
                        orig_target_block_id,
                        exit_env,
                        0,
                        BlockKind::Regular,
                    );
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
                            let else_block_id = self.process_branch(
                                Some(block_id),
                                func_ctx_id,
                                orig_else_block_id,
                                exit_env,
                                0,
                                BlockKind::Regular,
                            );
                            self.func.blocks[block_id].term = Terminator::Br(else_block_id);
                        }

                        Some(_) => {
                            let then_block_id = self.process_branch(
                                Some(block_id),
                                func_ctx_id,
                                orig_then_block_id,
                                exit_env,
                                0,
                                BlockKind::Regular,
                            );
                            self.func.blocks[block_id].term = Terminator::Br(then_block_id);
                        }

                        None => {
                            let then_block_id = self.process_branch(
                                Some(block_id),
                                func_ctx_id,
                                orig_then_block_id,
                                exit_env.clone(),
                                0,
                                BlockKind::Canonical,
                            );
                            let else_block_id = self.process_branch(
                                Some(block_id),
                                func_ctx_id,
                                orig_else_block_id,
                                exit_env,
                                0,
                                BlockKind::Canonical,
                            );
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
                        let target_block_id = self.process_branch(
                            Some(block_id),
                            func_ctx_id,
                            orig_block_id,
                            exit_env,
                            0,
                            BlockKind::Regular,
                        );
                        self.func.blocks[block_id].term = Terminator::Br(target_block_id);
                    } else {
                        let block_ids = orig_block_ids
                            .iter()
                            .map(|&orig_block_id| {
                                self.process_branch(
                                    Some(block_id),
                                    func_ctx_id,
                                    orig_block_id,
                                    exit_env.clone(),
                                    0,
                                    BlockKind::Canonical,
                                )
                            })
                            .collect();

                        self.func.blocks[block_id].term = Terminator::Switch(index, block_ids);
                    }
                }

                Terminator::Return(ref expr) => {
                    let func_ctx_id = self.blocks[block_id].func_ctx_id;
                    let func_ctx = &self.func_ctxs[func_ctx_id];

                    if let Some(ParentCtx {
                        func_ctx_id: parent_func_ctx_id,
                        ret_local_id,
                        ret_orig_block_id,
                        ret_start_stmt_idx,
                        ret_canonical,
                        ..
                    }) = func_ctx.parent
                    {
                        if let Some(ret_local_id) = ret_local_id {
                            let expr = stack.pop().unwrap();
                            self.set_local(block_id, ret_local_id, expr);
                        }

                        let ret_block_kind = if ret_canonical {
                            BlockKind::Canonical
                        } else {
                            BlockKind::Return(func_ctx_id)
                        };

                        let target_block_id = self.process_branch(
                            Some(block_id),
                            parent_func_ctx_id,
                            ret_orig_block_id,
                            self.blocks[block_id].exit_env.clone(),
                            ret_start_stmt_idx,
                            ret_block_kind,
                        );
                        self.func.blocks[block_id].term = Terminator::Br(target_block_id);
                    } else {
                        self.flush_globals(block_id);
                        let expr = expr.is_some().then(|| stack.pop().unwrap());
                        self.func.blocks[block_id].term = Terminator::Return(expr);
                    }
                }
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
                trace!("processing a subexpr (#{subexpr_idx}) of {expr}");
                *subexpr_idx += 1;
                tasks.push((0, subexpr));
            } else {
                trace!("processing an expr: {expr}");

                match *expr {
                    Expr::Value(value, attrs) => stack.push(Expr::Value(value, attrs)),
                    Expr::Nullary(op) => self.process_nul_op(block_id, stack, op)?,
                    Expr::Unary(op, _) => self.process_un_op(block_id, stack, op)?,
                    Expr::Binary(op, _) => self.process_bin_op(block_id, stack, op)?,
                    Expr::Ternary(op, _) => self.process_tern_op(block_id, stack, op)?,
                }

                trace!("pushed {}", stack.last().unwrap());

                tasks.pop().unwrap();
            }
        }

        Ok(())
    }

    fn process_nul_op(
        &mut self,
        block_id: BlockId,
        stack: &mut Vec<Expr>,
        mut op: NulOp,
    ) -> Result<()> {
        let info = &mut self.blocks[block_id];

        if let NulOp::LocalGet(local_id) = &mut op {
            *local_id = self.local_map[&(info.func_ctx_id, *local_id)];
        };

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

    fn process_un_op(&mut self, _block_id: BlockId, stack: &mut Vec<Expr>, op: UnOp) -> Result<()> {
        let arg = stack.pop().unwrap();

        let try_process = || -> Option<Expr> {
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
        next_stmt_idx: usize,
        ret_local_id: Option<LocalId>,
        func_id: FuncId,
        args: Vec<Expr>,
        inline_disposition: InlineDisposition,
    ) -> Result<bool> {
        trace!(
            "processing a call: {ret_local_id:?} <- {func_id:?} ({:?}) with {} args",
            self.interp.module.funcs[func_id].name(),
            args.len(),
        );

        if let Some(intrinsic) =
            self.interp.module.funcs[func_id].get_intrinsic(&self.interp.module)
        {
            trace!("processing an intrinsic {intrinsic}");
            let processed = match intrinsic {
                IntrinsicDecl::ArgCount => {
                    self.process_intr_arg_count(block_id, ret_local_id.unwrap())?
                }
                IntrinsicDecl::ArgLen => {
                    self.process_intr_arg_len(block_id, ret_local_id.unwrap(), &args)?
                }
                IntrinsicDecl::ArgRead => {
                    self.process_intr_arg_read(block_id, ret_local_id.unwrap())?
                }
                IntrinsicDecl::Specialize => {
                    self.process_intr_specialize(block_id, ret_local_id.unwrap())?
                }
                IntrinsicDecl::Unknown => {
                    self.process_intr_unknown(block_id, ret_local_id.unwrap(), func_id)?
                }
                IntrinsicDecl::ConstPtr => {
                    self.process_intr_const_ptr(block_id, ret_local_id.unwrap(), &args)?
                }
                IntrinsicDecl::PropagateLoad => {
                    self.process_intr_propagate_load(block_id, ret_local_id.unwrap(), &args)?
                }
                IntrinsicDecl::PrintValue => self.process_intr_print_value()?,
                IntrinsicDecl::PrintStr => self.process_intr_print_str()?,
                IntrinsicDecl::IsSpecializing => {
                    self.process_intr_is_specializing(block_id, ret_local_id.unwrap())?
                }
                IntrinsicDecl::Inline => {
                    self.process_intr_inline(block_id, ret_local_id.unwrap(), &args)?
                }
                IntrinsicDecl::NoInline => {
                    self.process_intr_no_inline(block_id, ret_local_id.unwrap(), &args)?
                }
            };

            if processed {
                return Ok(false);
            }
        }

        let inline_mode = if self.interp.module.funcs[func_id].is_import() {
            trace!("not inlining {func_id:?}: this is an imported function");

            InlineMode::No
        } else if inline_disposition == InlineDisposition::Deny {
            trace!("not inlining {func_id:?}: denied via an attribute");

            InlineMode::No
        } else if !self.is_recursive_call(block_id, func_id, &args) {
            trace!("inlining {func_id:?}: the call is not recursive");

            InlineMode::Inline
        } else if inline_disposition == InlineDisposition::Force {
            trace!("inlining {func_id:?}: forced via an attribute");

            InlineMode::Inline
        } else if self.blocks[block_id].kind == BlockKind::Canonical {
            trace!("not inlining {func_id:?}: the recursive call site is in a canonical block");

            InlineMode::No
        } else {
            trace!("outlining {func_id:?}: the call is recursive");

            InlineMode::Outline
        };

        match inline_mode {
            InlineMode::Inline => {
                trace!("inlining the body of {func_id:?}");
                let info = &self.blocks[block_id];
                let func_ctx_id = self.add_func_ctx(FuncCtx {
                    parent: Some(ParentCtx {
                        func_ctx_id: info.func_ctx_id,
                        call_block_id: block_id,
                        ret_local_id,
                        ret_orig_block_id: info.orig_block_id,
                        ret_start_stmt_idx: next_stmt_idx,
                        ret_canonical: info.kind == BlockKind::Canonical,
                    }),
                    func_id,
                    shape: args.iter().map(|arg| arg.to_value().is_some()).collect(),
                });

                let target_cfg = self.interp.get_cfg(func_id).unwrap();
                self.process_locals(func_ctx_id);
                let exit_env = &mut self.blocks[block_id].exit_env;

                for (local_id, val_ty) in &target_cfg.locals {
                    exit_env.locals.insert(
                        self.local_map[&(func_ctx_id, local_id)],
                        (Value::default_for(val_ty), Default::default()),
                    );
                }

                for (idx, (&local_id, arg)) in target_cfg.params.iter().zip(args).enumerate() {
                    let local_id = self.local_map[&(func_ctx_id, local_id)];
                    trace!("argument #{idx} -> {local_id:?}");
                    self.set_local(block_id, local_id, arg);
                }

                let target_block_id = self.process_branch(
                    Some(block_id),
                    func_ctx_id,
                    target_cfg.entry,
                    self.blocks[block_id].exit_env.clone(),
                    0,
                    BlockKind::Regular,
                );

                self.func.blocks[block_id].term = Terminator::Br(target_block_id);

                Ok(true)
            }

            InlineMode::Outline | InlineMode::No => {
                let target_func_id = if inline_mode == InlineMode::Outline {
                    trace!("replacing a direct call to {func_id:?} with a specialized version");

                    match self
                        .interp
                        .specialize(func_id, args.iter().map(Expr::to_value).collect())
                    {
                        Ok(spec_func_id) => spec_func_id,
                        Err(e) => {
                            warn!("specialization failed: {e}");

                            func_id
                        }
                    }
                } else {
                    func_id
                };

                trace!("emitting a direct call to {target_func_id:?}");
                self.flush_globals(block_id);
                self.func.blocks[block_id]
                    .body
                    .push(Stmt::Call(Call::Direct {
                        ret_local_id,
                        func_id: target_func_id,
                        args,
                    }));
                self.assume_clobbered_globals(block_id);

                if let Some(ret_local_id) = ret_local_id {
                    trace!(
                        "forgetting exit_env.locals[{ret_local_id:?}]: \
                        holds the return value of a call"
                    );
                    self.blocks[block_id].exit_env.locals.remove(ret_local_id);
                }

                Ok(false)
            }
        }
    }

    fn process_intr_arg_count(&mut self, block_id: BlockId, ret_local_id: LocalId) -> Result<bool> {
        self.set_local(
            block_id,
            ret_local_id,
            Expr::Value(
                Value::I32(u32::try_from(self.interp.args.len()).unwrap() as i32),
                Default::default(),
            ),
        );

        Ok(true)
    }

    fn process_intr_arg_len(
        &mut self,
        block_id: BlockId,
        ret_local_id: LocalId,
        args: &Vec<Expr>,
    ) -> Result<bool> {
        let Some(idx) = args[0].to_value().map(|(value, _)| value.to_u32().unwrap()) else {
            return Ok(false);
        };
        let Some(arg) = self.interp.args.get(idx as usize) else {
            warn!(
                "while processing {}: an interpreter argument index {idx} is out of bounds (provided {})",
                IntrinsicDecl::ArgLen,
                self.interp.args.len(),
            );

            return Ok(false);
        };
        self.set_local(
            block_id,
            ret_local_id,
            Expr::Value(
                Value::I32(u32::try_from(arg.len()).unwrap() as i32),
                Default::default(),
            ),
        );

        Ok(true)
    }

    fn process_intr_arg_read(&mut self, block_id: BlockId, ret_local_id: LocalId) -> Result<bool> {
        warn!(
            "encountered {} during specialization",
            IntrinsicDecl::ArgRead,
        );
        self.set_local(
            block_id,
            ret_local_id,
            Expr::Value(Value::I32(0), Default::default()),
        );

        Ok(true)
    }

    fn process_intr_specialize(
        &mut self,
        block_id: BlockId,
        ret_local_id: LocalId,
    ) -> Result<bool> {
        warn!(
            "encountered {} during specialization",
            IntrinsicDecl::Specialize,
        );
        self.set_local(
            block_id,
            ret_local_id,
            Expr::Value(Value::I32(0), Default::default()),
        );

        Ok(true)
    }

    fn process_intr_unknown(
        &mut self,
        block_id: BlockId,
        ret_local_id: LocalId,
        func_id: FuncId,
    ) -> Result<bool> {
        let val_ty = self.interp.module.funcs[func_id].ty().ret.as_ref().unwrap();
        self.set_local(
            block_id,
            ret_local_id,
            Expr::Value(Value::default_for(val_ty), Default::default()),
        );

        Ok(true)
    }

    fn process_intr_const_ptr(
        &mut self,
        block_id: BlockId,
        ret_local_id: LocalId,
        args: &Vec<Expr>,
    ) -> Result<bool> {
        self.set_local(
            block_id,
            ret_local_id,
            match args[0].to_value() {
                Some((value, attrs)) => Expr::Value(value, attrs | ValueAttrs::CONST_PTR),
                None => args[0].clone(),
            },
        );

        Ok(true)
    }

    fn process_intr_propagate_load(
        &mut self,
        block_id: BlockId,
        ret_local_id: LocalId,
        args: &Vec<Expr>,
    ) -> Result<bool> {
        self.set_local(
            block_id,
            ret_local_id,
            match args[0].to_value() {
                Some((value, attrs)) => Expr::Value(value, attrs | ValueAttrs::PROPAGATE_LOAD),
                None => args[0].clone(),
            },
        );

        Ok(true)
    }

    fn process_intr_print_value(&mut self) -> Result<bool> {
        warn!(
            "encountered {} during specialization",
            IntrinsicDecl::PrintValue,
        );

        Ok(false)
    }

    fn process_intr_print_str(&mut self) -> Result<bool> {
        warn!(
            "encountered {} during specialization",
            IntrinsicDecl::PrintStr,
        );

        Ok(false)
    }

    fn process_intr_is_specializing(
        &mut self,
        block_id: BlockId,
        ret_local_id: LocalId,
    ) -> Result<bool> {
        self.set_local(
            block_id,
            ret_local_id,
            Expr::Value(Value::I32(1), Default::default()),
        );

        Ok(true)
    }

    fn process_intr_inline(
        &mut self,
        block_id: BlockId,
        ret_local_id: LocalId,
        args: &Vec<Expr>,
    ) -> Result<bool> {
        self.set_local(
            block_id,
            ret_local_id,
            match args[0].to_value() {
                Some((value, mut attrs)) => {
                    attrs |= ValueAttrs::INLINE;
                    attrs.remove(ValueAttrs::NO_INLINE);

                    Expr::Value(value, attrs)
                }

                None => args[0].clone(),
            },
        );

        Ok(true)
    }

    fn process_intr_no_inline(
        &mut self,
        block_id: BlockId,
        ret_local_id: LocalId,
        args: &Vec<Expr>,
    ) -> Result<bool> {
        self.set_local(
            block_id,
            ret_local_id,
            match args[0].to_value() {
                Some((value, mut attrs)) => {
                    attrs |= ValueAttrs::NO_INLINE;
                    attrs.remove(ValueAttrs::INLINE);

                    Expr::Value(value, attrs)
                }

                None => args[0].clone(),
            },
        );

        Ok(true)
    }

    fn get_branch_target(
        &mut self,
        func_ctx_id: FuncCtxId,
        orig_block_id: OrigBlockId,
        env: &Env,
        target_kind: BlockKind,
        start_stmt_idx: usize,
    ) -> BlockId {
        let block_map_key = match target_kind {
            BlockKind::Regular => BlockMapKey::Regular {
                func_ctx_id,
                orig_block_id,
                entry_env: env.to_hashable(),
                start_stmt_idx,
            },

            BlockKind::Canonical => BlockMapKey::Canonical {
                func_ctx_id,
                orig_block_id,
                start_stmt_idx,
            },

            BlockKind::Return(from_func_ctx_id) => BlockMapKey::Return {
                func_ctx_id,
                orig_block_id,
                start_stmt_idx,
                from_func_ctx_id,
            },
        };
        let func_id = self.func_ctxs[func_ctx_id].func_id;

        if let Some(&block_id) = self.block_map.get(&block_map_key) {
            trace!("the branch target for {func_id:?}/{orig_block_id:?}: {block_id:?} (cached)");

            return block_id;
        }

        let block_id = self.func.blocks.insert(Default::default());
        self.blocks.insert(
            block_id,
            BlockInfo {
                func_ctx_id,
                orig_block_id,
                start_stmt_idx,
                kind: target_kind,
                entry_env: env.clone(),
                exit_env: Default::default(),
            },
        );

        trace!("the branch target for {func_id:?}/{orig_block_id:?}: {block_id:?} (created)");
        self.block_map.insert(block_map_key, block_id);
        self.tasks.insert(Task(block_id));

        block_id
    }

    fn process_branch(
        &mut self,
        from_block_id: Option<BlockId>,
        func_ctx_id: FuncCtxId,
        to_orig_block_id: OrigBlockId,
        env: Env,
        start_stmt_idx: usize,
        target_kind: BlockKind,
    ) -> BlockId {
        trace!(
            "processing a branch from {from_block_id:?} to {:?}/{to_orig_block_id:?} \
            (target_kind={target_kind:?}, starting from a stmt #{start_stmt_idx})",
            self.func_ctxs[func_ctx_id].func_id,
        );

        let target_kind = match target_kind {
            BlockKind::Return(_) | BlockKind::Canonical => target_kind,

            BlockKind::Regular
                if from_block_id
                    .is_some_and(|block_id| self.blocks[block_id].kind == BlockKind::Canonical) =>
            {
                trace!(
                    "overriding the target block kind to BlockKind::Canonical: \
                    the branch originates from a canonical block"
                );

                BlockKind::Canonical
            }

            BlockKind::Regular => BlockKind::Regular,
        };

        let block_id = self.get_branch_target(
            func_ctx_id,
            to_orig_block_id,
            &env,
            target_kind,
            start_stmt_idx,
        );

        let block_info = &mut self.blocks[block_id];
        let old_block_map_key = block_info.block_map_key();

        if block_info.entry_env.merge(&env) {
            trace!("the entry environment has changed");
            let new_block_map_key = block_info.block_map_key();
            self.block_map.remove(&old_block_map_key);
            self.block_map.insert(new_block_map_key, block_id);
            self.tasks.insert(Task(block_id));
        }

        block_id
    }

    fn flush_globals_in(entry_env: &Env, exit_env: &Env, block: &mut Block) {
        let flushed_globals = exit_env
            .globals
            .iter()
            .filter(|&(global_id, (exit_value, _))| {
                !entry_env
                    .globals
                    .get(global_id)
                    .is_some_and(|(entry_value, _)| exit_value == entry_value)
            })
            .collect::<Vec<_>>();

        for (global_id, &(value, attrs)) in flushed_globals {
            block
                .body
                .push(Stmt::GlobalSet(global_id, Expr::Value(value, attrs)));
        }
    }

    fn flush_globals(&mut self, block_id: BlockId) {
        let info = &self.blocks[block_id];
        Self::flush_globals_in(
            &info.entry_env,
            &info.exit_env,
            &mut self.func.blocks[block_id],
        );
    }

    fn flush_envs(&mut self) {
        let block_ids = self.func.blocks.keys().collect::<Vec<_>>();

        for block_id in block_ids {
            let mut successors = self.func.blocks[block_id]
                .successors()
                .iter()
                .map(|&succ_block_id| (succ_block_id, succ_block_id))
                .collect::<HashMap<_, _>>();

            for (&succ_block_id, new_succ_block_id) in &mut successors {
                let exit_env = &self.blocks[block_id].exit_env;
                let entry_env = &self.blocks[succ_block_id].entry_env;
                let ref mut bridge_block_id = None;

                fn flush<K: Key, F>(
                    func: &mut FuncBody,
                    bridge_block_id: &mut Option<BlockId>,
                    succ_block_id: BlockId,
                    exit_values: &SecondaryMap<K, (Value, ValueAttrs)>,
                    entry_values: &SecondaryMap<K, (Value, ValueAttrs)>,
                    make_stmt: F,
                ) where
                    F: Fn(K, Expr) -> Stmt,
                {
                    for (id, &(lhs_value, lhs_attrs)) in exit_values {
                        if entry_values.contains_key(id) {
                            continue;
                        }

                        let bridge_block_id = *bridge_block_id.get_or_insert_with(|| {
                            let bridge_block_id = func.blocks.insert(Block {
                                term: Terminator::Br(succ_block_id),
                                ..Default::default()
                            });
                            trace!("  created {bridge_block_id:?}");

                            bridge_block_id
                        });

                        trace!("  saving {id:?} = {}", Expr::Value(lhs_value, lhs_attrs));
                        func.blocks[bridge_block_id]
                            .body
                            .push(make_stmt(id, Expr::Value(lhs_value, lhs_attrs)));
                    }
                }

                trace!("flushing the environment along the edge {block_id:?} -> {succ_block_id:?}");

                flush(
                    &mut self.func,
                    bridge_block_id,
                    succ_block_id,
                    &exit_env.locals,
                    &entry_env.locals,
                    Stmt::LocalSet,
                );
                flush(
                    &mut self.func,
                    bridge_block_id,
                    succ_block_id,
                    &exit_env.globals,
                    &entry_env.globals,
                    Stmt::GlobalSet,
                );

                if let Some(bridge_block_id) = *bridge_block_id {
                    let func_id = self.func_ctxs[self.blocks[block_id].func_ctx_id].func_id;
                    self.func.blocks[bridge_block_id].name = Some(self.make_block_name(
                        func_id,
                        format_args!("env:{block_id:?}:{succ_block_id:?}"),
                    ));
                    *new_succ_block_id = bridge_block_id;
                }
            }

            for succ_block_id in self.func.blocks[block_id].successors_mut() {
                *succ_block_id = successors[succ_block_id];
            }
        }
    }

    fn assume_clobbered_globals(&mut self, block_id: BlockId) {
        trace!("assuming all globals are clobbered");
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

    fn is_recursive_call(&self, block_id: BlockId, func_id: FuncId, args: &[Expr]) -> bool {
        let mut func_ctx_id = self.blocks[block_id].func_ctx_id;

        loop {
            let func_ctx = &self.func_ctxs[func_ctx_id];

            if func_ctx.func_id == func_id
                && func_ctx.shape.iter().all({
                    let mut iter = args.iter();

                    move |&concrete| concrete == iter.next().unwrap().to_value().is_some()
                })
            {
                return true;
            }

            if let Some(parent) = &func_ctx.parent {
                func_ctx_id = parent.func_ctx_id;
            } else {
                return false;
            }
        }
    }

    fn set_local(&mut self, block_id: BlockId, local_id: LocalId, expr: Expr) {
        let exit_env = &mut self.blocks[block_id].exit_env;

        if let Some(value) = expr.to_value() {
            trace!("set exit_env.locals[{local_id:?}] <- {expr}");
            exit_env.locals.insert(local_id, value);
        } else {
            trace!("forgetting exit_env.locals[{local_id:?}]: {expr} is not a value");
            exit_env.locals.remove(local_id);
            self.func.blocks[block_id]
                .body
                .push(Stmt::LocalSet(local_id, expr));
        }
    }

    fn add_func_ctx(&mut self, func_ctx: FuncCtx) -> FuncCtxId {
        *self
            .func_ctx_map
            .entry(func_ctx)
            .or_insert_with_key(|func_ctx| self.func_ctxs.insert(func_ctx.clone()))
    }

    fn set_block_name(&mut self, block_id: BlockId, suffix: fmt::Arguments<'_>) {
        let func_id = self.func_ctxs[self.blocks[block_id].func_ctx_id].func_id;
        self.func.blocks[block_id].name = Some(self.make_block_name(func_id, suffix));
    }

    fn make_block_name(&self, func_id: FuncId, suffix: fmt::Arguments<'_>) -> String {
        match self.interp.module.funcs[func_id].name() {
            Some(name) => format!("{name}.{suffix}"),
            None => format!("{func_id:?}.{suffix}"),
        }
    }
}
