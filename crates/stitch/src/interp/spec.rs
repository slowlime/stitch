mod expr;
mod preprocess;

use std::fmt::{self, Debug};
use std::hash::{Hash, Hasher};
use std::ops::{Bound, Range, RangeBounds};
use std::rc::Rc;
use std::{iter, mem};

use anyhow::{ensure, Result};
use bitvec::prelude::*;
use hashbrown::{HashMap, HashSet};
use log::{log_enabled, trace, warn};
use slotmap::{new_key_type, Key, SecondaryMap, SlotMap, SparseSecondaryMap};

use crate::ast::expr::{format_value, MemArg, Ptr, Value, ValueAttrs};
use crate::ast::ty::ValType;
use crate::ast::{ConstExpr, FuncId, GlobalDef, GlobalId, IntrinsicDecl, MemoryId, TableDef};
use crate::cfg::{
    BinOp, Block, BlockId, Call, Expr, FuncBody, I32Store, I64Store, Load, LocalId, NulOp, Stmt,
    Store, Terminator, UnOp,
};

use super::{InlineDisposition, Interpreter, SpecSignature};

new_key_type! {
    struct FuncCtxId;
}

type OrigBlockId = BlockId;

#[derive(Default, Clone, Eq)]
struct MemSlice {
    // invariant: bytes.len() == init.len() ≥ zero_idx ∧ zero_idx ≤ |isize::MIN|
    bytes: Vec<u8>,
    init: BitVec,
    zero_idx: usize,
}

impl MemSlice {
    /// Reserves enough space to make `idx` accessible.
    fn reserve(&mut self, idx: isize) {
        if idx < 0 && idx.unsigned_abs() > self.zero_idx {
            let extra = idx.unsigned_abs() - self.zero_idx;
            self.bytes.splice(..0, iter::repeat(0).take(extra));
            self.init.splice(..0, iter::repeat(false).take(extra));
        } else if idx >= 0 && self.zero_idx + idx as usize >= self.bytes.len() {
            let extra = self.zero_idx + idx as usize + 1 - self.bytes.len();
            self.bytes.extend(iter::repeat(0).take(extra));
            self.init.extend(iter::repeat(false).take(extra));
        }
    }

    fn reserve_range<R: RangeBounds<isize>>(&mut self, range: R) {
        match range.start_bound() {
            Bound::Included(&idx) => self.reserve(idx),
            Bound::Excluded(&idx) => self.reserve(idx.checked_add(1).unwrap()),
            Bound::Unbounded => panic!("the start bound is infinite"),
        }

        match range.end_bound() {
            Bound::Included(&idx) => self.reserve(idx),
            Bound::Excluded(&idx) => self.reserve(idx.checked_sub(1).unwrap()),
            Bound::Unbounded => panic!("the end bound is infinite"),
        }
    }

    fn translate_range(&self, Range { start, end }: Range<isize>) -> Option<Range<usize>> {
        Some(Range {
            start: self.zero_idx.checked_add_signed(start)?,
            end: self.zero_idx.checked_add_signed(end)?,
        })
    }

    fn invalidate(&mut self, start: isize, end: Option<isize>) {
        let start = self.zero_idx.checked_add_signed(start).unwrap_or(0);
        let end = end
            .and_then(|end| self.zero_idx.checked_add_signed(end))
            .unwrap_or(self.init.len());

        self.init[start..end].fill(false);
    }

    fn read(&self, range: Range<isize>) -> Option<&[u8]> {
        trace!(
            "read({range:?}) -> {:?}",
            self.translate_range(range.clone())
        );
        let range = self.translate_range(range)?;
        self.init[range.clone()].all().then(|| &self.bytes[range])
    }

    fn read_u8(&self, offset: isize) -> Option<u8> {
        Some(self.read(offset..offset + 1)?[0])
    }

    fn read_u16(&self, offset: isize) -> Option<u16> {
        Some(u16::from_le_bytes(
            self.read(offset..offset + 2)?.try_into().unwrap(),
        ))
    }

    fn read_u32(&self, offset: isize) -> Option<u32> {
        Some(u32::from_le_bytes(
            self.read(offset..offset + 4)?.try_into().unwrap(),
        ))
    }

    fn read_u64(&self, offset: isize) -> Option<u64> {
        Some(u64::from_le_bytes(
            self.read(offset..offset + 8)?.try_into().unwrap(),
        ))
    }

    fn write(&mut self, range: Range<isize>) -> &mut [u8] {
        self.reserve_range(range.clone());
        let range = self.translate_range(range).unwrap();
        self.init[range.clone()].fill(true);

        &mut self.bytes[range]
    }

    fn merge(&mut self, other: &Self) -> bool {
        let mut changed = false;

        for (idx, bit) in self.init.iter_mut().enumerate().filter(|(_, bit)| **bit) {
            if let Some(other_idx) = other
                .zero_idx
                .checked_add_signed((idx - self.zero_idx) as isize)
            {
                if other.init.get(other_idx).is_some_and(|rhs| *rhs) {
                    if self.bytes[idx] == other.bytes[other_idx] {
                        continue;
                    }
                }
            }

            changed = true;
            bit.commit(false);
        }

        changed
    }
}

impl Debug for MemSlice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MemSlice[{}..{}] [",
            self.zero_idx.wrapping_neg() as isize,
            self.init.len() - self.zero_idx,
        )?;

        for (idx, (known, byte)) in self.init.iter().by_vals().zip(&self.bytes).enumerate() {
            if idx > 0 {
                write!(f, " ")?;
            }

            if known {
                write!(f, "{byte:02x}")?;
            } else {
                write!(f, "??")?;
            }
        }

        write!(f, "]")
    }
}

impl PartialEq for MemSlice {
    fn eq(&self, other: &Self) -> bool {
        if self.zero_idx > other.zero_idx {
            return other == self;
        }

        let other_offset = other.zero_idx - self.zero_idx;

        if other.init[..other_offset].any() {
            return false;
        }

        let self_last_one = self.init.last_one();

        if other.init.last_one().map(|idx| idx - other.zero_idx)
            != self_last_one.map(|idx| idx - self.zero_idx)
        {
            return false;
        }

        let init_size = self_last_one.map(|idx| idx + 1).unwrap_or(0);

        if self.init[..init_size] != other.init[other_offset..other_offset + init_size] {
            return false;
        }

        for idx in self.init[..init_size].iter_ones() {
            if self.bytes[idx] != other.bytes[idx + other_offset] {
                return false;
            }
        }

        true
    }
}

impl Hash for MemSlice {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.init
            .first_one()
            .zip(self.init.last_one())
            .map(|(start, end)| &self.init[start..=end])
            .hash(state);

        for idx in self.init.iter_ones() {
            self.bytes[idx].hash(state);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct HashableEnv {
    globals: Vec<(GlobalId, (Value<()>, ValueAttrs))>,
    locals: Vec<(LocalId, (Value<()>, ValueAttrs))>,
    mem: Vec<MemSlice>,
}

#[derive(Debug, Default, Clone)]
struct Env {
    globals: SecondaryMap<GlobalId, (Value<()>, ValueAttrs)>,
    locals: SecondaryMap<LocalId, (Value<()>, ValueAttrs)>,
    mem: Vec<MemSlice>,
}

impl Env {
    fn merge(&mut self, other: &Self) -> bool {
        fn merge_maps<K: Key>(
            changed: &mut bool,
            lhs_map: &mut SecondaryMap<K, (Value<()>, ValueAttrs)>,
            rhs_map: &SecondaryMap<K, (Value<()>, ValueAttrs)>,
        ) {
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

                    *changed = *lhs != new || *changed;
                    *lhs = new;
                    *lhs_attrs = new_attrs;
                } else {
                    trace!("merge_maps: {id:?} removed");
                    *changed = true;
                    to_remove.push(id);
                }
            }

            for id in to_remove {
                lhs_map.remove(id);
            }
        }

        let mut changed = false;
        merge_maps(&mut changed, &mut self.globals, &other.globals);
        merge_maps(&mut changed, &mut self.locals, &other.locals);

        let empty_slice = MemSlice::default();

        for (idx, lhs) in self.mem.iter_mut().enumerate() {
            changed |= lhs.merge(other.mem.get(idx).unwrap_or(&empty_slice));
        }

        changed
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
            mem: self.mem.clone(),
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

impl InlineDisposition {
    pub fn from_attrs(attrs: ValueAttrs) -> Option<InlineDisposition> {
        if attrs.contains(ValueAttrs::NO_INLINE) {
            Some(Self::Deny)
        } else if attrs.contains(ValueAttrs::INLINE) {
            Some(Self::ForceInline)
        } else {
            None
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

trait BlockProvider {
    fn get(&mut self) -> &mut Block;
}

#[derive(Debug, Clone)]
struct SymbolicPtrInfo {
    /// The id of the immutable local that stores the base address of the allocation.
    local_id: LocalId,

    /// Whether all pointers to the allocation are controlled by the specialized function.
    owned: bool,
}

pub struct Specializer<'a, 'i> {
    interp: &'i mut Interpreter<'a>,
    preprocessed_cfgs: SparseSecondaryMap<FuncId, Rc<FuncBody>>,
    orig_func_id: FuncId,
    func: FuncBody,
    block_map: HashMap<BlockMapKey, BlockId>,
    blocks: SecondaryMap<BlockId, BlockInfo>,
    args: Vec<Option<(Value<()>, ValueAttrs)>>,
    tasks: HashSet<BlockId>,
    local_map: HashMap<(FuncCtxId, LocalId), LocalId>,
    func_ctxs: SlotMap<FuncCtxId, FuncCtx>,
    func_ctx_map: HashMap<FuncCtx, FuncCtxId>,
    symbolic_ptrs: Vec<SymbolicPtrInfo>,
    symbolic_ptr_locals: SparseSecondaryMap<LocalId, u32>,
    symbolic_ptr_id_map: HashMap<u32, u32>,
}

impl<'a, 'i> Specializer<'a, 'i> {
    pub fn new(
        interp: &'i mut Interpreter<'a>,
        SpecSignature { orig_func_id, args }: SpecSignature,
        func: FuncBody,
    ) -> Self {
        Self {
            interp,
            preprocessed_cfgs: Default::default(),
            orig_func_id,
            func,
            block_map: Default::default(),
            blocks: Default::default(),
            args,
            tasks: Default::default(),
            local_map: Default::default(),
            func_ctxs: Default::default(),
            func_ctx_map: Default::default(),
            symbolic_ptrs: Default::default(),
            symbolic_ptr_locals: Default::default(),
            symbolic_ptr_id_map: Default::default(),
        }
    }

    pub fn run(mut self) -> Result<FuncBody> {
        self.process_entry();

        while let Some(&block_id) = self.tasks.iter().next() {
            self.tasks.remove(&block_id);
            self.process_block(block_id)?;
        }

        self.flush_envs();
        self.func.remove_unreachable_blocks();
        self.func.merge_blocks();
        self.set_block_name(self.func.entry, format_args!("entry"));

        Ok(self.func)
    }

    fn get_cfg(&mut self, func_id: FuncId) -> Option<Rc<FuncBody>> {
        if let Some(cfg) = self.preprocessed_cfgs.get(func_id) {
            return Some(Rc::clone(cfg));
        }

        let mut cfg = Rc::unwrap_or_clone(self.interp.get_cfg(func_id)?);
        trace!("preprocessing {func_id:?} before specialization");
        self.preprocess_cfg(&mut cfg);
        let cfg = Rc::new(cfg);
        self.preprocessed_cfgs.insert(func_id, Rc::clone(&cfg));

        Some(cfg)
    }

    fn process_arg(&mut self, value: Value<()>, load: impl FnOnce() -> Expr) -> Value<()> {
        match value {
            Value::I32(_) | Value::I64(_) | Value::F32(_) | Value::F64(_) => value,

            Value::Ptr(Ptr { id, offset, .. }) => {
                let id = *self.symbolic_ptr_id_map.entry(id).or_insert_with(|| {
                    let id = self.symbolic_ptrs.len().try_into().unwrap();
                    let base_local_id = self.func.locals.insert(ValType::I32);
                    self.func
                        .local_names
                        .entry(base_local_id)
                        .unwrap()
                        .or_insert_with(|| format!("ptr-base.{id}"));
                    self.symbolic_ptrs.push(SymbolicPtrInfo {
                        local_id: base_local_id,
                        owned: false,
                    });
                    self.symbolic_ptr_locals.insert(base_local_id, id);
                    self.func.blocks[self.func.entry].body.push(Stmt::LocalSet(
                        base_local_id,
                        match offset {
                            0 => load(),
                            _ => Expr::Binary(
                                BinOp::I32Sub,
                                Box::new([
                                    load(),
                                    Expr::Value(Value::I32(offset), Default::default()),
                                ]),
                            ),
                        },
                    ));

                    id
                });

                Value::Ptr(Ptr {
                    base: (),
                    id,
                    offset,
                })
            }
        }
    }

    fn process_entry(&mut self) {
        self.func.blocks[self.func.entry].body.clear();

        let func_ctx_id = self.add_func_ctx(FuncCtx {
            parent: None,
            func_id: self.orig_func_id,
            shape: self.args.iter().map(Option::is_some).collect(),
        });

        self.process_params(func_ctx_id);
        let cfg = self.get_cfg(self.orig_func_id).unwrap();
        let mut exit_env = Env::default();

        let globals = mem::take(&mut self.interp.module.globals);

        for (global_id, global) in &globals {
            match global.def {
                GlobalDef::Import(_) => {}
                GlobalDef::Value(ConstExpr::Value(value, attrs)) => {
                    if !global.ty.mutable || self.interp.const_global_ids.contains(&global_id) {
                        exit_env.globals.insert(
                            global_id,
                            (
                                self.process_arg(value.lift_ptr(), || {
                                    Expr::Nullary(NulOp::GlobalGet(global_id))
                                }),
                                attrs,
                            ),
                        );
                    }
                }
                GlobalDef::Value(ConstExpr::GlobalGet(_)) => {}
            }
        }

        self.interp.module.globals = globals;

        for (local_id, val_ty) in &cfg.locals {
            let local_id = self.local_map[&(func_ctx_id, local_id)];
            exit_env
                .locals
                .insert(local_id, (Value::default_for(val_ty), Default::default()));
        }

        let args = mem::take(&mut self.args);

        for (&local_id, arg) in cfg.params.iter().zip(&args) {
            let local_id = self.local_map[&(func_ctx_id, local_id)];

            if let &Some((value, attrs)) = arg {
                exit_env.locals[local_id] = (
                    self.process_arg(value, || Expr::Nullary(NulOp::LocalGet(local_id))),
                    attrs,
                );
            } else {
                exit_env.locals.remove(local_id);
            }
        }

        self.args = args;

        self.blocks.insert(
            self.func.entry,
            BlockInfo {
                func_ctx_id,
                orig_block_id: Default::default(),
                start_stmt_idx: 0,
                kind: BlockKind::Regular,
                entry_env: Env::default(),
                exit_env: exit_env.clone(),
            },
        );

        let entry_block_id = self.process_branch(
            None,
            func_ctx_id,
            cfg.entry,
            exit_env,
            0,
            BlockKind::Regular,
        );
        self.func.blocks[self.func.entry].term = Terminator::Br(entry_block_id);
    }

    fn process_params(&mut self, orig_func_ctx_id: FuncCtxId) {
        let cfg = self.get_cfg(self.orig_func_id).unwrap();
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
        let cfg = self.get_cfg(func_id).unwrap();

        for (local_id, val_ty) in &cfg.locals {
            self.local_map
                .entry((func_ctx_id, local_id))
                .or_insert_with(|| {
                    let mapped_local_id = self.func.locals.insert(val_ty.clone());

                    if let Some(name) = cfg.local_names.get(local_id) {
                        self.func.local_names.insert(mapped_local_id, name.clone());
                    }

                    mapped_local_id
                });
        }
    }

    fn process_block(&mut self, block_id: BlockId) -> Result<()> {
        self.func.blocks[block_id].body.clear();
        let info = &mut self.blocks[block_id];
        info.exit_env = info.entry_env.clone();
        let info = &self.blocks[block_id];
        let orig_block_id = info.orig_block_id;
        let func_id = self.func_ctxs[info.func_ctx_id].func_id;
        let cfg = self.get_cfg(func_id).unwrap();
        let info = &self.blocks[block_id];
        let mut stmt_idx = info.start_stmt_idx;
        let ref mut stack = vec![];

        trace!(
            "specializing a block {func_id:?}/{orig_block_id:?} (function {:?}, to {block_id:?}, kind {:?})",
            self.interp.module.funcs[func_id].name(),
            info.kind,
        );

        if log_enabled!(log::Level::Trace) {
            trace!("func stack (the current func ctx is at the top):");
            let mut func_ctx_id = info.func_ctx_id;

            loop {
                let func_ctx = &self.func_ctxs[func_ctx_id];
                trace!(
                    "  {func_ctx_id:?}: func {:?} ({}), shape [{}]",
                    func_ctx.func_id,
                    self.interp.module.func_name(func_ctx.func_id),
                    func_ctx
                        .shape
                        .iter()
                        .map(|present| if *present { "#" } else { "_" })
                        .map(str::to_string)
                        .reduce(|lhs, rhs| format!("{lhs} {rhs}"))
                        .unwrap_or_default(),
                );

                if let Some(parent) = &func_ctx.parent {
                    trace!(
                        "    return to {}{:?} (from stmt #{}), local {:?}",
                        if parent.ret_canonical {
                            "canonical "
                        } else {
                            ""
                        },
                        parent.ret_orig_block_id,
                        parent.ret_start_stmt_idx,
                        parent.ret_local_id,
                    );
                    func_ctx_id = parent.func_ctx_id;
                } else {
                    break;
                }
            }
        }

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

        if log_enabled!(log::Level::Trace) {
            trace!("locals:");

            for (local_id, &(value, attrs)) in &info.exit_env.locals {
                trace!("  {local_id:?} = {}", format_value(&value, attrs));
            }

            trace!("globals:");

            for (global_id, &(value, attrs)) in &info.exit_env.globals {
                trace!("  {global_id:?} = {}", format_value(&value, attrs));
            }

            trace!("mem:");

            for (idx, mem_slice) in info.exit_env.mem.iter().enumerate() {
                trace!("  {idx}: {mem_slice:?}");
            }
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
                Stmt::MemoryCopy { .. } | Stmt::MemoryFill(..) => unreachable!(),
            }

            let block = &mut self.func.blocks[block_id];
            let info = &mut self.blocks[block_id];

            match *stmt {
                Stmt::Nop => {}

                Stmt::Drop(_) => {
                    let expr = Self::convert_expr(&self.symbolic_ptrs, stack.pop().unwrap());

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

                    match expr {
                        Expr::Value(value @ Value::Ptr(_), attrs) => {
                            trace!("set exit_env.globals[{global_id:?}] <- {expr}");
                            info.exit_env.globals.insert(global_id, (value, attrs));
                            let stmt = Stmt::GlobalSet(
                                global_id,
                                Self::convert_expr(&self.symbolic_ptrs, expr),
                            );
                            trace!("emitting {stmt} for the symbolic pointer");
                            block.body.push(stmt);
                        }

                        Expr::Value(value, attrs) => {
                            trace!("set exit_env.globals[{global_id:?}] <- {expr}");
                            info.exit_env.globals.insert(global_id, (value, attrs));
                        }

                        _ => {
                            trace!(
                                "forgetting exit_env.globals[{global_id:?}]: {expr} is not a value"
                            );
                            info.exit_env.globals.remove(global_id);
                            block.body.push(Stmt::GlobalSet(
                                global_id,
                                Self::convert_expr(&self.symbolic_ptrs, expr),
                            ));
                        }
                    }
                }

                Stmt::Store(mem_arg, store, _) => {
                    let value = Self::convert_expr(&self.symbolic_ptrs, stack.pop().unwrap());
                    let addr = stack.pop().unwrap();

                    if let Some(stmt) = self.process_store(block_id, mem_arg, store, addr, value)? {
                        trace!("emitting a store");
                        self.func.blocks[block_id].body.push(stmt);
                    }
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
                    let mut index = Some(Self::convert_expr(
                        &self.symbolic_ptrs,
                        stack.pop().unwrap(),
                    ));
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
                                    InlineDisposition::from_attrs(idx_attrs),
                                )? {
                                    skip_terminator = true;
                                    break 'body;
                                }
                            }
                        }
                    }

                    if let Some(index) = index {
                        trace!("emitting an indirect call");
                        self.flush_global_env(block_id);
                        let preserved_ptrs = self.find_preserved_ptrs(&args);
                        self.func.blocks[block_id]
                            .body
                            .push(Stmt::Call(Call::Indirect {
                                ret_local_id,
                                ty_id,
                                table_id,
                                args: args
                                    .into_iter()
                                    .map(|arg| Self::convert_expr(&self.symbolic_ptrs, arg))
                                    .collect(),
                                index: Box::new(index),
                            }));
                        self.assume_clobbered_env(block_id, preserved_ptrs);

                        if let Some(ret_local_id) = ret_local_id {
                            trace!(
                                "forgetting exit_env.locals[{ret_local_id:?}]: \
                                holds the return value of an indirect call"
                            );
                            self.blocks[block_id].exit_env.locals.remove(ret_local_id);
                        }
                    }
                }

                Stmt::MemoryCopy { .. } | Stmt::MemoryFill(..) => unreachable!(),
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
                    let condition = Self::convert_expr(&self.symbolic_ptrs, stack.pop().unwrap());
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
                    let index = Self::convert_expr(&self.symbolic_ptrs, stack.pop().unwrap());
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
                        self.flush_global_env(block_id);
                        let expr = expr
                            .is_some()
                            .then(|| Self::convert_expr(&self.symbolic_ptrs, stack.pop().unwrap()));
                        self.func.blocks[block_id].term = Terminator::Return(expr);
                    }
                }
            }
        }

        Ok(())
    }

    fn process_load(
        &mut self,
        block_id: BlockId,
        mem_arg: MemArg,
        load: Load,
        addr: Expr<()>,
    ) -> Expr<()> {
        let try_process = || -> Option<Expr<()>> {
            match &addr {
                &Expr::Value(Value::I32(base_addr), attrs)
                    if attrs.contains(ValueAttrs::CONST_PTR) =>
                {
                    let base_addr = base_addr as u32;
                    let start = (base_addr + mem_arg.offset) as usize;
                    let range = start..start.checked_add(load.src_size())?;

                    let bytes = match self.interp.module.get_mem(mem_arg.mem_id, range) {
                        Ok(bytes) => bytes,

                        Err(e) => {
                            warn!("encountered an error while specializing a constant memory load: {e}");

                            return None;
                        }
                    };

                    Some(Expr::Value(
                        load.load(bytes).lift_ptr(),
                        attrs.deref_attrs(),
                    ))
                }

                &Expr::Value(Value::Ptr(Ptr { id, offset, .. }), attrs)
                    if mem_arg.mem_id == self.interp.module.default_mem =>
                {
                    trace!("performing symbolic load");
                    let start = offset.wrapping_add_unsigned(mem_arg.offset) as isize;
                    let range = start..start.checked_add_unsigned(load.src_size())?;
                    let bytes = self.blocks[block_id]
                        .exit_env
                        .mem
                        .get(id as usize)?
                        .read(range)?;

                    Some(Expr::Value(
                        load.load(bytes).lift_ptr(),
                        attrs.deref_attrs(),
                    ))
                }

                _ => None,
            }
        };

        try_process().unwrap_or_else(|| {
            Expr::Unary(
                UnOp::Load(mem_arg, load),
                Box::new(Self::convert_expr(&self.symbolic_ptrs, addr)),
            )
        })
    }

    fn process_store(
        &mut self,
        block_id: BlockId,
        mem_arg: MemArg,
        store: Store,
        addr: Expr<()>,
        value: Expr,
    ) -> Result<Option<Stmt>> {
        if let Some((_, addr_attrs)) = addr.to_value() {
            ensure!(
                !addr_attrs.contains(ValueAttrs::CONST_PTR),
                "encountered a memory write via a constant pointer",
            );
        }

        let mut try_process = || -> bool {
            if mem_arg.mem_id != self.interp.module.default_mem {
                return false;
            }

            let &Expr::Value(Value::Ptr(Ptr { id, offset, .. }), _) = &addr else {
                return false;
            };

            let id = id as usize;
            let start = offset.wrapping_add_unsigned(mem_arg.offset) as isize;
            let end = start.checked_add_unsigned(store.dst_size());
            let mem = &mut self.blocks[block_id].exit_env.mem;

            if let (Some((value, _)), Some(end)) = (value.to_value(), end) {
                trace!("performing symbolic store: {start}..{end}");

                if mem.len() <= id {
                    mem.resize_with(id, Default::default);
                }

                store.store(mem[id].write(start..end), value);

                return true;
            }

            if let Some(mem_slice) = mem.get_mut(id) {
                trace!("invalidating {start}..{end:?}: {value} is not a value");
                mem_slice.invalidate(start, end);
            }

            false
        };

        Ok((!try_process()).then(|| {
            Stmt::Store(
                mem_arg,
                store,
                Box::new([Self::convert_expr(&self.symbolic_ptrs, addr), value]),
            )
        }))
    }

    fn process_call(
        &mut self,
        block_id: BlockId,
        next_stmt_idx: usize,
        ret_local_id: Option<LocalId>,
        func_id: FuncId,
        args: Vec<Expr<()>>,
        attr_inline_disposition: Option<InlineDisposition>,
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
                IntrinsicDecl::Unknown => {
                    self.process_intr_unknown(block_id, ret_local_id.unwrap(), func_id)?
                }
                IntrinsicDecl::ConstPtr => {
                    self.process_intr_const_ptr(block_id, ret_local_id.unwrap(), &args)?
                }
                IntrinsicDecl::SymbolicPtr => {
                    self.process_intr_symbolic_ptr(block_id, ret_local_id.unwrap(), &args)?
                }
                IntrinsicDecl::PropagateLoad => {
                    self.process_intr_propagate_load(block_id, ret_local_id.unwrap(), &args)?
                }
                IntrinsicDecl::IsSpecializing => {
                    self.process_intr_is_specializing(block_id, ret_local_id.unwrap())?
                }
                IntrinsicDecl::Inline => {
                    self.process_intr_inline(block_id, ret_local_id.unwrap(), &args)?
                }
                IntrinsicDecl::NoInline => {
                    self.process_intr_no_inline(block_id, ret_local_id.unwrap(), &args)?
                }

                IntrinsicDecl::ArgRead
                | IntrinsicDecl::Specialize
                | IntrinsicDecl::PrintValue
                | IntrinsicDecl::PrintStr
                | IntrinsicDecl::FileOpen
                | IntrinsicDecl::FileRead
                | IntrinsicDecl::FileClose
                | IntrinsicDecl::FuncSpecPolicy => {
                    warn!("encountered {intrinsic} during specialization");

                    false
                }
            };

            if processed {
                return Ok(false);
            }
        }

        match self.inline_mode(func_id, block_id, &args, attr_inline_disposition) {
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

                let target_cfg = self.get_cfg(func_id).unwrap();
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

            inline_mode @ (InlineMode::Outline | InlineMode::No) => {
                let target_func_id = if inline_mode == InlineMode::Outline {
                    trace!("replacing a direct call to {func_id:?} with a specialized version");

                    let args = args
                        .iter()
                        .map(|arg| match arg.to_value() {
                            Some((Value::Ptr(_), _)) => None,
                            Some((value, attrs)) => Some((value.unwrap_concrete(), attrs)),
                            None => None,
                        })
                        .collect();

                    match self.interp.specialize(func_id, args) {
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
                self.flush_global_env(block_id);
                let preserved_ptrs = self.find_preserved_ptrs(&args);
                self.func.blocks[block_id]
                    .body
                    .push(Stmt::Call(Call::Direct {
                        ret_local_id,
                        func_id: target_func_id,
                        args: args
                            .into_iter()
                            .map(|expr| Self::convert_expr(&self.symbolic_ptrs, expr))
                            .collect(),
                    }));
                self.assume_clobbered_env(block_id, preserved_ptrs);

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
        args: &Vec<Expr<()>>,
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
        args: &Vec<Expr<()>>,
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

    fn process_intr_symbolic_ptr(
        &mut self,
        block_id: BlockId,
        ret_local_id: LocalId,
        args: &Vec<Expr<()>>,
    ) -> Result<bool> {
        match args[0].clone() {
            expr @ Expr::Value(Value::Ptr(_), _) => {
                self.set_local(block_id, ret_local_id, expr);
            }

            expr => {
                let id = *self
                    .symbolic_ptr_locals
                    .entry(ret_local_id)
                    .unwrap()
                    .or_insert_with(|| {
                        let id = self.symbolic_ptrs.len().try_into().unwrap();
                        self.symbolic_ptrs.push(SymbolicPtrInfo {
                            local_id: ret_local_id,
                            owned: true,
                        });

                        id
                    });

                let result = Value::Ptr(Ptr {
                    base: (),
                    id,
                    offset: 0,
                });

                trace!(
                    "set exit_env.locals[{ret_local_id:?}] <- {}",
                    format_value(&result, Default::default()),
                );
                self.func.blocks[block_id].body.push(Stmt::LocalSet(
                    ret_local_id,
                    Self::convert_expr(&self.symbolic_ptrs, expr),
                ));
                self.blocks[block_id]
                    .exit_env
                    .locals
                    .insert(ret_local_id, (result, Default::default()));
            }
        }

        Ok(true)
    }

    fn process_intr_propagate_load(
        &mut self,
        block_id: BlockId,
        ret_local_id: LocalId,
        args: &Vec<Expr<()>>,
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
        args: &Vec<Expr<()>>,
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
        args: &Vec<Expr<()>>,
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
        self.tasks.insert(block_id);

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
            self.tasks.insert(block_id);
        }

        block_id
    }

    fn flush_env<B: BlockProvider>(
        symbolic_ptrs: &Vec<SymbolicPtrInfo>,
        mem_id: MemoryId,
        exit_env: &Env,
        entry_env: Option<&Env>,
        global_env_only: bool,
        mut target_block: B,
    ) {
        fn flush_vars<K: Key>(
            block: &mut impl BlockProvider,
            exit_vars: &SecondaryMap<K, (Value<()>, ValueAttrs)>,
            entry_vars: Option<&SecondaryMap<K, (Value<()>, ValueAttrs)>>,
            make_stmt: impl Fn(K, Expr) -> Stmt,
        ) {
            for (id, &(lhs_value, lhs_attrs)) in exit_vars {
                if entry_vars.is_some_and(|entry_vars| entry_vars.contains_key(id)) {
                    continue;
                }

                let lhs_value = match lhs_value {
                    Value::Ptr(_) => continue,
                    _ => lhs_value.unwrap_concrete(),
                };

                trace!("  saving {id:?} = {}", format_value(&lhs_value, lhs_attrs));
                block
                    .get()
                    .body
                    .push(make_stmt(id, Expr::Value(lhs_value, lhs_attrs)));
            }
        }

        trace!("flushing environment");

        if !global_env_only {
            flush_vars(
                &mut target_block,
                &exit_env.locals,
                entry_env.map(|entry_env| &entry_env.locals),
                Stmt::LocalSet,
            );
        }

        flush_vars(
            &mut target_block,
            &exit_env.globals,
            entry_env.map(|entry_env| &entry_env.globals),
            Stmt::GlobalSet,
        );

        for (mem_idx, (mem_slice, symbolic_ptr_info)) in
            exit_env.mem.iter().zip(symbolic_ptrs).enumerate()
        {
            let mut flush_mask = mem_slice.init.clone();
            let rhs_iter = entry_env
                .and_then(|entry_env| entry_env.mem.get(mem_idx))
                .map(|mem_slice| mem_slice.init.iter().by_vals())
                .into_iter()
                .flatten();

            for (lhs, rhs) in flush_mask.iter_mut().zip(rhs_iter) {
                let new_value = *lhs & !rhs;
                lhs.commit(new_value);
            }

            let mut idx_iter = flush_mask.iter_ones();

            while let Some(idx) = idx_iter.next() {
                let store = if flush_mask.get(idx..idx + 8).is_some_and(BitSlice::all) {
                    Store::I64(I64Store::Eight)
                } else if flush_mask.get(idx..idx + 4).is_some_and(BitSlice::all) {
                    Store::I32(I32Store::Four)
                } else if flush_mask.get(idx..idx + 2).is_some_and(BitSlice::all) {
                    Store::I32(I32Store::Two)
                } else {
                    Store::I32(I32Store::One)
                };

                for i in 0..(store.dst_size() - 1) {
                    debug_assert_eq!(idx_iter.next().unwrap(), idx + i + 1);
                }

                let start = (idx - mem_slice.zero_idx) as isize;
                let stmt = Stmt::Store(
                    MemArg {
                        mem_id,
                        offset: idx.try_into().unwrap(),
                        align: 1,
                    },
                    store,
                    Box::new([
                        Expr::Nullary(NulOp::LocalGet(symbolic_ptr_info.local_id)),
                        Expr::Value(
                            match store {
                                Store::I64(_) => {
                                    Value::I64(mem_slice.read_u64(start).unwrap() as i64)
                                }
                                Store::I32(store) => Value::I32(match store {
                                    I32Store::Four => mem_slice.read_u32(start).unwrap() as i32,
                                    I32Store::Two => mem_slice.read_u16(start).unwrap() as i32,
                                    I32Store::One => mem_slice.read_u8(start).unwrap() as i32,
                                }),
                                _ => unreachable!(),
                            },
                            Default::default(),
                        ),
                    ]),
                );
                trace!(
                    "  emitting a store for the symbolically tracked allocation #{mem_idx}: {stmt}"
                );
                target_block.get().body.push(stmt);
            }
        }
    }

    fn flush_global_env(&mut self, block_id: BlockId) {
        struct TargetBlockProvider<'a> {
            func: &'a mut FuncBody,
            block_id: BlockId,
        }

        impl BlockProvider for TargetBlockProvider<'_> {
            fn get(&mut self) -> &mut Block {
                &mut self.func.blocks[self.block_id]
            }
        }

        let info = &self.blocks[block_id];
        Self::flush_env(
            &self.symbolic_ptrs,
            self.interp.module.default_mem,
            &info.exit_env,
            None,
            true,
            TargetBlockProvider {
                func: &mut self.func,
                block_id,
            },
        );
    }

    fn flush_envs(&mut self) {
        struct BridgeBlockProvider<'a> {
            func: &'a mut FuncBody,
            bridge_block_id: &'a mut Option<BlockId>,
            succ_block_id: BlockId,
        }

        impl BlockProvider for BridgeBlockProvider<'_> {
            fn get(&mut self) -> &mut Block {
                let bridge_block_id = *self.bridge_block_id.get_or_insert_with(|| {
                    let bridge_block_id = self.func.blocks.insert(Block {
                        term: Terminator::Br(self.succ_block_id),
                        ..Default::default()
                    });
                    trace!("  created {bridge_block_id:?}");

                    bridge_block_id
                });

                &mut self.func.blocks[bridge_block_id]
            }
        }

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

                trace!("flushing the environment along the edge {block_id:?} -> {succ_block_id:?}");
                Self::flush_env(
                    &self.symbolic_ptrs,
                    self.interp.module.default_mem,
                    exit_env,
                    Some(entry_env),
                    false,
                    BridgeBlockProvider {
                        func: &mut self.func,
                        bridge_block_id,
                        succ_block_id,
                    },
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

    fn find_preserved_ptrs(&self, args: &[Expr<()>]) -> HashSet<u32> {
        let mut result = self
            .symbolic_ptrs
            .iter()
            .enumerate()
            .filter(|(_, info)| info.owned)
            .map(|(idx, _)| idx.try_into().unwrap())
            .collect::<HashSet<_>>();

        for arg in args {
            match arg {
                Expr::Value(Value::Ptr(Ptr { id, .. }), _) => {
                    result.remove(id);
                }

                Expr::Value(..) => continue,

                // TODO: this is very pessimistic
                // (can be improved by tagging expressions derived from symbolic pointers)
                _ => return Default::default(),
            }
        }

        result
    }

    fn assume_clobbered_env(&mut self, block_id: BlockId, preserved_ptrs: HashSet<u32>) {
        trace!("assuming the global environment is clobbered");
        let info = &mut self.blocks[block_id];
        info.exit_env
            .globals
            .retain(|global_id, _| !self.interp.module.globals[global_id].ty.mutable);

        for (idx, mem_slice) in info.exit_env.mem.iter_mut().enumerate() {
            if !preserved_ptrs.contains(&u32::try_from(idx).unwrap()) {
                mem_slice.invalidate(0, None);
            } else {
                trace!("  preserved {idx}");
            }
        }
    }

    fn is_recursive_call(&self, block_id: BlockId, func_id: FuncId, args: &[Expr<()>]) -> bool {
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

    fn set_local(&mut self, block_id: BlockId, local_id: LocalId, expr: Expr<()>) {
        let exit_env = &mut self.blocks[block_id].exit_env;

        match expr {
            Expr::Value(value @ Value::Ptr(_), attrs) => {
                trace!("set exit_env.locals[{local_id:?}] <- {expr}");
                exit_env.locals.insert(local_id, (value, attrs));
                let stmt = Stmt::LocalSet(local_id, Self::convert_expr(&self.symbolic_ptrs, expr));
                trace!("emitting {stmt} for the symbolic pointer");
                self.func.blocks[block_id].body.push(stmt);
            }

            Expr::Value(value, attrs) => {
                trace!("set exit_env.locals[{local_id:?}] <- {expr}");
                exit_env.locals.insert(local_id, (value, attrs));
            }

            _ => {
                trace!("forgetting exit_env.locals[{local_id:?}]: {expr} is not a value");
                exit_env.locals.remove(local_id);
                self.func.blocks[block_id].body.push(Stmt::LocalSet(
                    local_id,
                    Self::convert_expr(&self.symbolic_ptrs, expr),
                ));
            }
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

    fn convert_expr(symbolic_ptrs: &Vec<SymbolicPtrInfo>, expr: Expr<()>) -> Expr {
        match expr {
            Expr::Value(Value::Ptr(Ptr { id, offset: 0, .. }), _) => {
                Expr::Nullary(NulOp::LocalGet(symbolic_ptrs[id as usize].local_id))
            }

            Expr::Value(Value::Ptr(Ptr { id, offset, .. }), _) => Expr::Binary(
                BinOp::I32Add,
                Box::new([
                    Expr::Nullary(NulOp::LocalGet(symbolic_ptrs[id as usize].local_id)),
                    Expr::Value(Value::I32(offset), Default::default()),
                ]),
            ),

            Expr::Value(value, attrs) => Expr::Value(value.unwrap_concrete(), attrs),
            Expr::Nullary(op) => Expr::Nullary(op),
            Expr::Unary(op, expr) => Expr::Unary(op, expr),
            Expr::Binary(op, exprs) => Expr::Binary(op, exprs),
            Expr::Ternary(op, exprs) => Expr::Ternary(op, exprs),
        }
    }

    fn inline_mode(
        &self,
        func_id: FuncId,
        block_id: BlockId,
        args: &[Expr<()>],
        attr_inline_disposition: Option<InlineDisposition>,
    ) -> InlineMode {
        let func_inline_policy = self
            .interp
            .func_spec_policies
            .get(func_id)
            .map(|policy| policy.inline_policy);
        let (inline_disposition, inline_disposition_source) =
            match (func_inline_policy, attr_inline_disposition) {
                (None, None) => (Default::default(), "the default inline policy"),
                (Some(inline_disposition), None) => {
                    (inline_disposition, "a function specialization policy")
                }
                (_, Some(inline_disposition)) => (inline_disposition, "an attribute"),
            };

        if self.interp.module.funcs[func_id].is_import() {
            trace!("not inlining {func_id:?}: this is an imported function");

            InlineMode::No
        } else if inline_disposition == InlineDisposition::Deny {
            trace!("not inlining {func_id:?}: denied via {inline_disposition_source}");

            InlineMode::No
        } else if inline_disposition == InlineDisposition::ForceOutline {
            trace!("outlining {func_id:?}: forced via {inline_disposition_source}");

            InlineMode::Outline
        } else if !self.is_recursive_call(block_id, func_id, &args) {
            trace!("inlining {func_id:?}: the call is not recursive");

            InlineMode::Inline
        } else if self.blocks[block_id].kind == BlockKind::Regular {
            trace!("inlining {func_id:?}: the call site occurs in a regular block");

            InlineMode::Inline
        } else if inline_disposition == InlineDisposition::ForceInline {
            trace!("inlining {func_id:?}: forced via {inline_disposition_source}");

            InlineMode::Inline
        } else if self.blocks[block_id].kind == BlockKind::Canonical {
            trace!("not inlining {func_id:?}: the recursive call site is in a canonical block");

            InlineMode::No
        } else {
            trace!("outlining {func_id:?}: the call is recursive");

            InlineMode::Outline
        }
    }
}
