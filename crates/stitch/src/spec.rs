use std::collections::{HashMap, VecDeque};
use std::mem;

use log::{trace, warn};
use slotmap::{SecondaryMap, SparseSecondaryMap};

use crate::ast::expr::{
    BinOp, Block, Intrinsic, Load, MemArg, NulOp, PtrAttr, TernOp, UnOp, Value, ValueAttrs,
    VisitContext, F32, F64,
};
use crate::ast::ty::Type;
use crate::ast::{
    BlockId, Export, ExportDef, Expr, Func, FuncBody, FuncId, GlobalDef, LocalId, MemError,
    MemoryId, Module, TableDef, TableId,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SpecSignature {
    orig_func_id: FuncId,
    args: Vec<Option<Value>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpecializedFunc {
    Pending(FuncId),
    Finished(FuncId),
}

impl SpecializedFunc {
    pub fn func_id(&self) -> FuncId {
        match *self {
            Self::Pending(func_id) | Self::Finished(func_id) => func_id,
        }
    }
}

pub struct Specializer<'a> {
    module: &'a mut Module,
    spec_sigs: HashMap<SpecSignature, SpecializedFunc>,
    spec_funcs: SparseSecondaryMap<FuncId, SpecSignature>,
}

impl<'a> Specializer<'a> {
    pub fn new(module: &'a mut Module) -> Self {
        Self {
            module,
            spec_sigs: Default::default(),
            spec_funcs: Default::default(),
        }
    }

    pub fn process(mut self) -> Option<FuncId> {
        let export = self
            .module
            .exports
            .values()
            .find(|export| export.name == "stitch-start")?;

        let ExportDef::Func(func_id) = export.def else {
            warn!("stitch-start is not a function");
            return None;
        };

        let func_ty = self.module.funcs[func_id].ty();

        if !func_ty.params.is_empty() || func_ty.ret.is_some() {
            warn!("stitch-start has a wrong type: expected [] -> []");
            return None;
        }

        let spec_func_id = self.specialize(SpecSignature {
            orig_func_id: func_id,
            args: vec![],
        });

        self.module.funcs.remove(func_id);
        self.module.exports.retain(|_, export| match export.def {
            ExportDef::Func(export_func_id) if func_id == export_func_id => false,
            _ => true,
        });

        Some(spec_func_id)
    }

    pub fn specialize(&mut self, sig: SpecSignature) -> FuncId {
        assert!(self.module.funcs.contains_key(sig.orig_func_id));

        if matches!(
            self.spec_funcs
                .get(sig.orig_func_id)
                .map(|sig| self.spec_sigs[sig]),
            Some(SpecializedFunc::Pending(_))
        ) {
            panic!("trying to specialize a function pending specialization");
        }

        if let Some(&spec) = self.spec_sigs.get(&sig) {
            return spec.func_id();
        }

        let Some(orig_func) = self.module.funcs[sig.orig_func_id].body() else {
            panic!("cannot specialize an imported function");
        };

        let mut func_ty = orig_func.ty.clone();

        assert_eq!(
            sig.args.len(),
            func_ty.params.len(),
            "argument mismatch: signature specifies {}, the function has {}",
            sig.args.len(),
            func_ty.params.len()
        );

        func_ty.params.retain({
            let mut iter = sig.args.iter();

            move |_| iter.next().unwrap().is_none()
        });

        let mut body = FuncBody {
            ty: func_ty,
            ..orig_func.clone()
        };

        let func_id = self
            .module
            .funcs
            .insert(Func::Body(FuncBody::new(body.ty.clone())));
        trace!(
            "specializing {:?} as {func_id:?}: {:?}",
            sig.orig_func_id,
            &sig.args
        );
        self.spec_sigs
            .insert(sig.clone(), SpecializedFunc::Pending(func_id));
        self.spec_funcs.insert(func_id, sig.clone());

        FuncSpecializer::specialize(self, sig.clone(), &mut body);

        *self.module.funcs[func_id].body_mut().unwrap() = body;
        *self.spec_sigs.get_mut(&sig).unwrap() = SpecializedFunc::Finished(func_id);

        func_id
    }
}

#[derive(Debug, Default, Clone)]
struct SpecContext {
    abort_specialization: bool,
}

impl SpecContext {
    pub fn abort_specialization(&mut self) {
        self.abort_specialization = true;
    }
}

struct FuncSpecializer<'a, 'b, 'm> {
    spec: &'a mut Specializer<'m>,
    sig: SpecSignature,
    locals: SecondaryMap<LocalId, (Value, ValueAttrs)>,
    func: &'b mut FuncBody,
}

impl<'a, 'b, 'm> FuncSpecializer<'a, 'b, 'm> {
    fn specialize(spec: &'a mut Specializer<'m>, sig: SpecSignature, func: &'b mut FuncBody) {
        let body = mem::take(&mut func.main_block);
        let mut this = FuncSpecializer {
            spec,
            sig,
            locals: Default::default(),
            func,
        };

        for (idx, arg) in this.sig.args.iter().enumerate() {
            if let &Some(value) = arg {
                this.locals
                    .insert(this.func.params[idx], (value, Default::default()));
            }
        }

        let body = this.block(&mut Default::default(), &body);
        let sig = this.sig;
        func.main_block = body;

        func.params.retain({
            let mut iter = sig.args.iter();

            move |_| iter.next().unwrap().is_none()
        });
    }

    fn block(&mut self, ctx: &mut SpecContext, block: &Block) -> Block {
        let mut body = vec![];

        'outer: for expr in &block.body {
            let mut exprs = VecDeque::new();
            exprs.push_back(self.expr(ctx, expr));

            while let Some(expr) = exprs.pop_front() {
                match expr {
                    Expr::Nullary(NulOp::Nop) => continue,

                    Expr::Unary(UnOp::Drop, inner) if !inner.has_side_effect() => {
                        continue;
                    }

                    Expr::Block(_, block)
                        if block.body.iter().all(|expr| !expr.branches_to(block.id)) =>
                    {
                        for expr in block.body.into_iter().rev() {
                            exprs.push_front(expr);
                        }

                        self.func.blocks.remove(block.id);

                        continue;
                    }

                    _ => {}
                }

                let diverges = expr.diverges();
                body.push(expr);

                if diverges {
                    break 'outer;
                }
            }
        }

        Block { body, id: block.id }
    }

    fn expr(&mut self, ctx: &mut SpecContext, expr: &Expr) -> Expr {
        if ctx.abort_specialization {
            return expr.clone();
        }

        let mut expr = match expr {
            Expr::Value(_, _) | Expr::Index(_) | Expr::Intrinsic(_) | Expr::Nullary(_) => {
                expr.clone()
            }
            Expr::Unary(op, inner) => Expr::Unary(*op, Box::new(self.expr(ctx, inner))),
            Expr::Binary(op, [lhs, rhs]) => Expr::Binary(
                *op,
                [Box::new(self.expr(ctx, lhs)),
                Box::new(self.expr(ctx, rhs))],
            ),
            Expr::Ternary(op, [first, second, third]) => Expr::Ternary(
                *op,
                [Box::new(self.expr(ctx, first)),
                Box::new(self.expr(ctx, second)),
                Box::new(self.expr(ctx, third))],
            ),

            Expr::Block(block_ty, exprs) => Expr::Block(block_ty.clone(), self.block(ctx, exprs)),
            Expr::Loop(block_ty, exprs) => Expr::Block(block_ty.clone(), self.block(ctx, exprs)),
            Expr::If(block_ty, cond, then_block, else_block) => Expr::If(
                block_ty.clone(),
                Box::new(self.expr(ctx, cond)),
                then_block.clone(),
                else_block.clone(),
            ),

            Expr::Br(relative_depth, value) => Expr::Br(
                *relative_depth,
                value.as_ref().map(|expr| Box::new(self.expr(ctx, expr))),
            ),
            Expr::BrIf(relative_depth, value, condition) => Expr::BrIf(
                *relative_depth,
                value.as_ref().map(|expr| Box::new(self.expr(ctx, expr))),
                Box::new(self.expr(ctx, condition)),
            ),
            Expr::BrTable(labels, default_label, value, index) => Expr::BrTable(
                labels.clone(),
                *default_label,
                value.as_ref().map(|expr| Box::new(self.expr(ctx, expr))),
                Box::new(self.expr(ctx, index)),
            ),
            Expr::Return(value) => {
                Expr::Return(value.as_ref().map(|expr| Box::new(self.expr(ctx, expr))))
            }

            Expr::Call(func_id, args) => Expr::Call(
                *func_id,
                args.iter().map(|expr| self.expr(ctx, expr)).collect(),
            ),
            Expr::CallIndirect(ty_id, table_id, args, index) => Expr::CallIndirect(
                *ty_id,
                *table_id,
                args.iter().map(|expr| self.expr(ctx, expr)).collect(),
                Box::new(self.expr(ctx, index)),
            ),
        };

        // could've changed while evaluting subexpressions
        if ctx.abort_specialization {
            return expr;
        }

        macro_rules! try_i32 {
            ($expr:expr) => {{
                match $expr.to_i32() {
                    Some(value) => value,
                    None => return expr,
                }
            }};
        }

        macro_rules! try_u32 {
            ($expr:expr) => {
                try_i32!($expr) as u32
            };
        }

        macro_rules! try_i64 {
            ($expr:expr) => {{
                match $expr.to_i64() {
                    Some(value) => value,
                    None => return expr,
                }
            }};
        }

        macro_rules! try_u64 {
            ($expr:expr) => {
                try_i64!($expr) as u64
            };
        }

        macro_rules! try_f32 {
            ($expr:expr) => {{
                match $expr.to_f32() {
                    Some(value) => value,
                    None => return expr,
                }
            }};
        }

        macro_rules! try_f64 {
            ($expr:expr) => {{
                match $expr.to_f64() {
                    Some(value) => value,
                    None => return expr,
                }
            }};
        }

        match expr.to_load() {
            Some((
                MemArg { mem_id, offset, .. },
                &Expr::Value(Value::I32(addr), addr_attrs),
                load,
            )) if addr_attrs.ptr >= PtrAttr::Const => {
                let start = (addr as u32 + offset) as usize;
                let end = start + load.src_size() as usize;

                match self.spec.module.read_mem(mem_id, start..end) {
                    Ok(src) => {
                        debug_assert!(src.len() <= 8);

                        // little-endian
                        let mut value = src.iter().rfold(0u64, |acc, &byte| acc << 8 | byte as u64);

                        if load.sign_extend() {
                            value |= -((value & 1 << 8 * load.src_size() - 1) as i64) as u64;
                        }

                        let value = match load {
                            Load::I32 { .. } => Value::I32(value as i32),
                            Load::I64 { .. } => Value::I64(value as i64),
                            Load::F32 => Value::F32(F32::from_bits(value as u32)),
                            Load::F64 => Value::F64(F64::from_bits(value)),
                        };

                        let attrs = if addr_attrs.propagate {
                            ValueAttrs {
                                propagate: false,
                                ..addr_attrs
                            }
                        } else {
                            addr_attrs
                        };

                        return Expr::Value(value, attrs);
                    }

                    Err(MemError::Import) => {}

                    Err(MemError::OutOfBounds { .. }) => {
                        warn!("Out-of-bounds constant read of {start}..{end}");
                    }
                }
            }

            _ => {}
        }

        /* FIXME: this is absolutely wrong.
        match expr.to_store() {
            Some((
                MemArg { mem_id, offset, .. },
                &Expr::Value(Value::I32(addr), addr_attrs),
                &Expr::Value(value, _),
                store,
            )) if addr_attrs.ptr >= PtrAttr::Owned => {
                match &mut self.spec.module.mems[mem_id].def {
                    MemoryDef::Import(_) => {}

                    MemoryDef::Bytes(bytes) => {
                        let start = (addr as u32 + offset) as usize;
                        let end = start + store.dst_size() as usize;

                        if let Some(dst) = bytes.get_mut(start..end) {
                            debug_assert!(dst.len() <= 8);

                            let src = match value {
                                Value::I32(value) => value as u64,
                                Value::I64(value) => value as u64,
                                Value::F32(value) => value.to_bits() as u64,
                                Value::F64(value) => value.to_bits(),
                            };

                            dst.copy_from_slice(&src.to_le_bytes()[0..store.dst_size() as usize]);

                            return NulOp::Nop.into();
                        } else {
                            warn!("Out-of-bounds owned write of {start}..{end}");
                        }
                    }
                }
            }

            _ => {}
        }
        */

        match expr {
            Expr::Value(_, _) | Expr::Index(_) => expr,

            Expr::Intrinsic(ref mut intr) => {
                let result = match *intr {
                    Intrinsic::Specialize {
                        table_id,
                        elem_idx,
                        mem_id,
                        name_addr,
                        name_len,
                        ref mut args,
                    } => self.intr_specialize(
                        ctx, table_id, elem_idx, mem_id, name_addr, name_len, args,
                    ),
                    Intrinsic::Unknown(_) => todo!(),
                };

                result.unwrap_or(expr)
            }

            Expr::Nullary(NulOp::LocalGet(local_id)) => match self.locals.get(local_id) {
                Some(&(value, attrs)) => Expr::Value(value, attrs),
                None => expr,
            },

            Expr::Unary(UnOp::LocalSet(local_id), ref inner) => match **inner {
                Expr::Value(value, attrs) => {
                    self.locals.insert(local_id, (value, attrs));

                    NulOp::Nop.into()
                }

                _ => {
                    self.locals.remove(local_id);

                    expr
                }
            },

            Expr::Unary(UnOp::LocalTee(local_id), ref inner) => match **inner {
                Expr::Value(value, attrs) => {
                    self.locals.insert(local_id, (value, attrs));

                    expr
                }

                _ => {
                    self.locals.remove(local_id);

                    expr
                }
            },

            Expr::Nullary(NulOp::GlobalGet(global_id)) => {
                let global = &self.spec.module.globals[global_id];

                match &global.def {
                    GlobalDef::Value(expr) if !global.ty.mutable => self.expr(ctx, &expr.clone()),
                    _ => expr,
                }
            }

            Expr::Unary(UnOp::GlobalSet(_), _) => expr, // TODO

            Expr::Nullary(op) => match op {
                // might want to make an instrinsic to assume the size stays constant
                NulOp::MemorySize(_) => expr,

                NulOp::Nop => expr,

                NulOp::Unreachable => {
                    warn!("aborting specialization: encountered `unreachable`");
                    ctx.abort_specialization();

                    expr
                }

                NulOp::LocalGet(_) | NulOp::GlobalGet(_) => unreachable!(),
            },

            Expr::Unary(op, ref inner) => {
                let Expr::Value(value, attr) = **inner else {
                    return expr;
                };

                let result = match op {
                    UnOp::I32Clz => Value::I32(try_i32!(value).leading_zeros() as i32),
                    UnOp::I32Ctz => Value::I32(try_i32!(value).trailing_zeros() as i32),
                    UnOp::I32Popcnt => Value::I32(try_i32!(value).count_ones() as i32),

                    UnOp::I64Clz => Value::I64(try_i64!(value).leading_zeros() as i64),
                    UnOp::I64Ctz => Value::I64(try_i64!(value).trailing_zeros() as i64),
                    UnOp::I64Popcnt => Value::I64(try_i64!(value).count_ones() as i64),

                    UnOp::F32Abs => Value::F32(try_f32!(value).to_f32().abs().into()),
                    UnOp::F32Neg => Value::F32((-try_f32!(value).to_f32()).into()),
                    UnOp::F32Sqrt => Value::F32(try_f32!(value).to_f32().sqrt().into()),
                    UnOp::F32Ceil => Value::F32(try_f32!(value).to_f32().ceil().into()),
                    UnOp::F32Floor => Value::F32(try_f32!(value).to_f32().floor().into()),
                    UnOp::F32Trunc => Value::F32(try_f32!(value).to_f32().trunc().into()),
                    UnOp::F32Nearest => return expr, // TODO

                    UnOp::F64Abs => Value::F64(try_f64!(value).to_f64().abs().into()),
                    UnOp::F64Neg => Value::F64((-try_f64!(value).to_f64()).into()),
                    UnOp::F64Sqrt => Value::F64(try_f64!(value).to_f64().sqrt().into()),
                    UnOp::F64Ceil => Value::F64(try_f64!(value).to_f64().ceil().into()),
                    UnOp::F64Floor => Value::F64(try_f64!(value).to_f64().floor().into()),
                    UnOp::F64Trunc => Value::F64(try_f64!(value).to_f64().trunc().into()),
                    UnOp::F64Nearest => return expr, // TODO

                    UnOp::I32Eqz => Value::I32((try_i32!(value) == 0) as i32),
                    UnOp::I64Eqz => Value::I32((try_i64!(value) == 0) as i32),

                    UnOp::I32WrapI64 => Value::I32(try_i64!(value) as i32),

                    UnOp::I64ExtendI32S => Value::I64(try_i32!(value) as i64),
                    UnOp::I64ExtendI32U => Value::I64(try_u32!(value) as i64),

                    UnOp::I32TruncF32S => Value::I32(try_f32!(value).to_f32() as i32),
                    UnOp::I32TruncF32U => Value::I32(try_f32!(value).to_f32() as u32 as i32),
                    UnOp::I32TruncF64S => Value::I32(try_f64!(value).to_f64() as i32),
                    UnOp::I32TruncF64U => Value::I32(try_f64!(value).to_f64() as u32 as i32),

                    UnOp::I64TruncF32S => Value::I64(try_f32!(value).to_f32() as i64),
                    UnOp::I64TruncF32U => Value::I64(try_f32!(value).to_f32() as u64 as i64),
                    UnOp::I64TruncF64S => Value::I64(try_f64!(value).to_f64() as i64),
                    UnOp::I64TruncF64U => Value::I64(try_f64!(value).to_f64() as u64 as i64),

                    UnOp::F32DemoteF64 => Value::F32((try_f64!(value).to_f64() as f32).into()),
                    UnOp::F64PromoteF32 => Value::F64((try_f32!(value).to_f32() as f64).into()),

                    UnOp::F32ConvertI32S => Value::F32((try_i32!(value) as f32).into()),
                    UnOp::F32ConvertI32U => Value::F32((try_u32!(value) as f32).into()),
                    UnOp::F32ConvertI64S => Value::F32((try_i64!(value) as f32).into()),
                    UnOp::F32ConvertI64U => Value::F32((try_u64!(value) as f32).into()),

                    UnOp::F64ConvertI32S => Value::F64((try_i32!(value) as f64).into()),
                    UnOp::F64ConvertI32U => Value::F64((try_u32!(value) as f64).into()),
                    UnOp::F64ConvertI64S => Value::F64((try_i64!(value) as f64).into()),
                    UnOp::F64ConvertI64U => Value::F64((try_u64!(value) as f64).into()),

                    UnOp::F32ReinterpretI32 => Value::F32(F32::from_bits(try_u32!(value))),
                    UnOp::F64ReinterpretI64 => Value::F64(F64::from_bits(try_u64!(value))),
                    UnOp::I32ReinterpretF32 => Value::I32(try_f32!(value).to_bits() as i32),
                    UnOp::I64ReinterpretF64 => Value::I64(try_f64!(value).to_bits() as i64),

                    UnOp::I32Extend8S => Value::I32(try_u32!(value) as i8 as i32),
                    UnOp::I32Extend16S => Value::I32(try_u32!(value) as i16 as i32),

                    UnOp::I64Extend8S => Value::I64(try_u64!(value) as i8 as i64),
                    UnOp::I64Extend16S => Value::I64(try_u64!(value) as i16 as i64),
                    UnOp::I64Extend32S => Value::I64(try_u64!(value) as i32 as i64),

                    UnOp::Drop => return expr,

                    UnOp::LocalSet(_) | UnOp::LocalTee(_) | UnOp::GlobalSet(_) => {
                        unreachable!()
                    }

                    UnOp::I32Load(_)
                    | UnOp::I64Load(_)
                    | UnOp::F32Load(_)
                    | UnOp::F64Load(_)
                    | UnOp::I32Load8S(_)
                    | UnOp::I32Load8U(_)
                    | UnOp::I32Load16S(_)
                    | UnOp::I32Load16U(_)
                    | UnOp::I64Load8S(_)
                    | UnOp::I64Load8U(_)
                    | UnOp::I64Load16S(_)
                    | UnOp::I64Load16U(_)
                    | UnOp::I64Load32S(_)
                    | UnOp::I64Load32U(_)
                    | UnOp::MemoryGrow(_) => return expr,
                };

                Expr::Value(result, attr)
            }

            Expr::Binary(op, [ref lhs, ref rhs]) => {
                let (&Expr::Value(lhs, lhs_attr), &Expr::Value(rhs, rhs_attr)) = (&**lhs, &**rhs)
                else {
                    return expr;
                };

                let attr = match op {
                    BinOp::I32Add | BinOp::I32Sub => ValueAttrs {
                        ptr: lhs_attr.ptr.max(rhs_attr.ptr),
                        propagate: false,
                    },

                    _ => lhs_attr.meet(&rhs_attr),
                };

                let result = match op {
                    BinOp::I32Add => Value::I32(try_i32!(lhs).wrapping_add(try_i32!(rhs))),
                    BinOp::I32Sub => Value::I32(try_i32!(lhs).wrapping_sub(try_i32!(rhs))),
                    BinOp::I32Mul => Value::I32(try_i32!(lhs).wrapping_mul(try_i32!(rhs))),

                    BinOp::I32DivS => match try_i32!(rhs) {
                        0 => {
                            warn!("aborting specialization due to division by zero");
                            ctx.abort_specialization();

                            return expr;
                        }

                        rhs => Value::I32(try_i32!(lhs).wrapping_div(rhs)),
                    },

                    BinOp::I32DivU => match try_u32!(rhs) {
                        0 => {
                            warn!("aborting specialization due to division by zero");
                            ctx.abort_specialization();

                            return expr;
                        }

                        rhs => Value::I32(try_u32!(lhs).wrapping_div(rhs) as i32),
                    },

                    BinOp::I32RemS => match try_i32!(rhs) {
                        0 => return expr, // undefined
                        rhs => Value::I32(try_i32!(lhs).wrapping_rem(rhs)),
                    },
                    BinOp::I32RemU => match try_u32!(rhs) {
                        0 => return expr, // undefined
                        rhs => Value::I32(try_u32!(lhs).wrapping_rem(rhs) as i32),
                    },
                    BinOp::I32And => Value::I32(try_i32!(lhs) & try_i32!(rhs)),
                    BinOp::I32Or => Value::I32(try_i32!(lhs) | try_i32!(rhs)),
                    BinOp::I32Xor => Value::I32(try_i32!(lhs) ^ try_i32!(rhs)),
                    BinOp::I32Shl => Value::I32(try_i32!(lhs).wrapping_shl(try_u32!(rhs))),
                    BinOp::I32ShrS => Value::I32(try_i32!(lhs).wrapping_shr(try_u32!(rhs))),
                    BinOp::I32ShrU => Value::I32(try_u32!(lhs).wrapping_shr(try_u32!(rhs)) as i32),
                    BinOp::I32Rotl => Value::I32(try_i32!(lhs).rotate_left(try_u32!(rhs))),
                    BinOp::I32Rotr => Value::I32(try_i32!(lhs).rotate_right(try_u32!(rhs))),

                    BinOp::I64Add => Value::I64(try_i64!(lhs).wrapping_add(try_i64!(rhs))),
                    BinOp::I64Sub => Value::I64(try_i64!(lhs).wrapping_sub(try_i64!(rhs))),
                    BinOp::I64Mul => Value::I64(try_i64!(lhs).wrapping_mul(try_i64!(rhs))),

                    BinOp::I64DivS => match try_i64!(rhs) {
                        0 => {
                            warn!("aborting specialization due to division by zero");
                            ctx.abort_specialization();

                            return expr;
                        }

                        rhs => Value::I64(try_i64!(lhs).wrapping_div(rhs)),
                    },

                    BinOp::I64DivU => match try_u64!(rhs) {
                        0 => {
                            warn!("aborting specialization due to division by zero");
                            ctx.abort_specialization();

                            return expr;
                        }

                        rhs => Value::I64(try_u64!(lhs).wrapping_div(rhs) as i64),
                    },

                    BinOp::I64RemS => match try_i64!(rhs) {
                        0 => return expr,
                        rhs => Value::I64(try_i64!(lhs).wrapping_rem(rhs)),
                    },
                    BinOp::I64RemU => match try_u64!(rhs) {
                        0 => return expr,
                        rhs => Value::I64(try_u64!(lhs).wrapping_rem(rhs) as i64),
                    },
                    BinOp::I64And => Value::I64(try_i64!(lhs) & try_i64!(rhs)),
                    BinOp::I64Or => Value::I64(try_i64!(lhs) | try_i64!(rhs)),
                    BinOp::I64Xor => Value::I64(try_i64!(lhs) ^ try_i64!(rhs)),
                    BinOp::I64Shl => Value::I64(try_i64!(lhs).wrapping_shl(try_u64!(rhs) as u32)),
                    BinOp::I64ShrS => Value::I64(try_i64!(lhs).wrapping_shr(try_u64!(rhs) as u32)),
                    BinOp::I64ShrU => {
                        Value::I64(try_u64!(lhs).wrapping_shr(try_u64!(rhs) as u32) as i64)
                    }
                    BinOp::I64Rotl => Value::I64(try_i64!(lhs).rotate_left(try_u64!(rhs) as u32)),
                    BinOp::I64Rotr => Value::I64(try_i64!(lhs).rotate_right(try_u64!(rhs) as u32)),

                    BinOp::F32Add => {
                        Value::F32((try_f32!(lhs).to_f32() + try_f32!(rhs).to_f32()).into())
                    }
                    BinOp::F32Sub => {
                        Value::F32((try_f32!(lhs).to_f32() - try_f32!(rhs).to_f32()).into())
                    }
                    BinOp::F32Mul => {
                        Value::F32((try_f32!(lhs).to_f32() * try_f32!(rhs).to_f32()).into())
                    }
                    BinOp::F32Div => {
                        Value::F32((try_f32!(lhs).to_f32() / try_f32!(rhs).to_f32()).into())
                    }
                    BinOp::F32Min => {
                        Value::F32(try_f32!(lhs).to_f32().min(try_f32!(rhs).to_f32()).into())
                    }
                    BinOp::F32Max => {
                        Value::F32(try_f32!(lhs).to_f32().max(try_f32!(rhs).to_f32()).into())
                    }
                    BinOp::F32Copysign => Value::F32(
                        try_f32!(lhs)
                            .to_f32()
                            .copysign(try_f32!(rhs).to_f32())
                            .into(),
                    ),

                    BinOp::F64Add => {
                        Value::F64((try_f64!(lhs).to_f64() + try_f64!(rhs).to_f64()).into())
                    }
                    BinOp::F64Sub => {
                        Value::F64((try_f64!(lhs).to_f64() - try_f64!(rhs).to_f64()).into())
                    }
                    BinOp::F64Mul => {
                        Value::F64((try_f64!(lhs).to_f64() * try_f64!(rhs).to_f64()).into())
                    }
                    BinOp::F64Div => {
                        Value::F64((try_f64!(lhs).to_f64() / try_f64!(rhs).to_f64()).into())
                    }
                    BinOp::F64Min => {
                        Value::F64(try_f64!(lhs).to_f64().min(try_f64!(rhs).to_f64()).into())
                    }
                    BinOp::F64Max => {
                        Value::F64(try_f64!(lhs).to_f64().max(try_f64!(rhs).to_f64()).into())
                    }
                    BinOp::F64Copysign => Value::F64(
                        try_f64!(lhs)
                            .to_f64()
                            .copysign(try_f64!(rhs).to_f64())
                            .into(),
                    ),

                    BinOp::I32Eq => Value::I32((try_i32!(lhs) == try_i32!(rhs)) as i32),
                    BinOp::I32Ne => Value::I32((try_i32!(lhs) != try_i32!(rhs)) as i32),
                    BinOp::I32LtS => Value::I32((try_i32!(lhs) < try_i32!(rhs)) as i32),
                    BinOp::I32LtU => Value::I32((try_u32!(lhs) < try_u32!(rhs)) as i32),
                    BinOp::I32GtS => Value::I32((try_i32!(lhs) > try_i32!(rhs)) as i32),
                    BinOp::I32GtU => Value::I32((try_u32!(lhs) > try_u32!(rhs)) as i32),
                    BinOp::I32LeS => Value::I32((try_i32!(lhs) <= try_i32!(rhs)) as i32),
                    BinOp::I32LeU => Value::I32((try_u32!(lhs) <= try_u32!(rhs)) as i32),
                    BinOp::I32GeS => Value::I32((try_i32!(lhs) >= try_i32!(rhs)) as i32),
                    BinOp::I32GeU => Value::I32((try_u32!(lhs) >= try_u32!(rhs)) as i32),

                    BinOp::I64Eq => Value::I32((try_i64!(lhs) == try_i64!(rhs)) as i32),
                    BinOp::I64Ne => Value::I32((try_i64!(lhs) != try_i64!(rhs)) as i32),
                    BinOp::I64LtS => Value::I32((try_i64!(lhs) < try_i64!(rhs)) as i32),
                    BinOp::I64LtU => Value::I32((try_u64!(lhs) < try_u64!(rhs)) as i32),
                    BinOp::I64GtS => Value::I32((try_i64!(lhs) > try_i64!(rhs)) as i32),
                    BinOp::I64GtU => Value::I32((try_u64!(lhs) > try_u64!(rhs)) as i32),
                    BinOp::I64LeS => Value::I32((try_i64!(lhs) <= try_i64!(rhs)) as i32),
                    BinOp::I64LeU => Value::I32((try_u64!(lhs) <= try_u64!(rhs)) as i32),
                    BinOp::I64GeS => Value::I32((try_i64!(lhs) >= try_i64!(rhs)) as i32),
                    BinOp::I64GeU => Value::I32((try_u64!(lhs) >= try_u64!(rhs)) as i32),

                    BinOp::F32Eq => Value::I32((try_f32!(lhs) == try_f32!(rhs)) as i32),
                    BinOp::F32Ne => Value::I32((try_f32!(lhs) != try_f32!(rhs)) as i32),
                    BinOp::F32Lt => Value::I32((try_f32!(lhs) < try_f32!(rhs)) as i32),
                    BinOp::F32Gt => Value::I32((try_f32!(lhs) > try_f32!(rhs)) as i32),
                    BinOp::F32Le => Value::I32((try_f32!(lhs) <= try_f32!(rhs)) as i32),
                    BinOp::F32Ge => Value::I32((try_f32!(lhs) >= try_f32!(rhs)) as i32),

                    BinOp::F64Eq => Value::I32((try_f64!(lhs) == try_f64!(rhs)) as i32),
                    BinOp::F64Ne => Value::I32((try_f64!(lhs) != try_f64!(rhs)) as i32),
                    BinOp::F64Lt => Value::I32((try_f64!(lhs) < try_f64!(rhs)) as i32),
                    BinOp::F64Gt => Value::I32((try_f64!(lhs) > try_f64!(rhs)) as i32),
                    BinOp::F64Le => Value::I32((try_f64!(lhs) <= try_f64!(rhs)) as i32),
                    BinOp::F64Ge => Value::I32((try_f64!(lhs) >= try_f64!(rhs)) as i32),

                    BinOp::I32Store(_)
                    | BinOp::I64Store(_)
                    | BinOp::F32Store(_)
                    | BinOp::F64Store(_)
                    | BinOp::I32Store8(_)
                    | BinOp::I32Store16(_)
                    | BinOp::I64Store8(_)
                    | BinOp::I64Store16(_)
                    | BinOp::I64Store32(_) => return expr,
                };

                Expr::Value(result, attr)
            }

            Expr::Ternary(TernOp::Select, [first, second, condition]) => match condition.to_i32() {
                Some(0) => *second,
                Some(_) => *first,
                _ => Expr::Ternary(TernOp::Select, [first, second, condition]),
            },

            Expr::Block(_, block) | Expr::Loop(_, block) if block.body.is_empty() => {
                NulOp::Nop.into()
            }

            Expr::Block(_, ref mut block) | Expr::Loop(_, ref mut block)
                if block.body.len() == 1 =>
            {
                let mut nop = Expr::Nullary(NulOp::Nop);
                let mut inner = &mut block.body[0];

                loop {
                    inner = match inner {
                        Expr::Br(br_block_id, _) if *br_block_id == block.id => {
                            let Expr::Br(_, inner) = inner else {
                                unreachable!()
                            };

                            match inner {
                                None => &mut nop,
                                Some(inner) => &mut **inner,
                            }
                        }

                        _ => {
                            if inner.branches_to(block.id) {
                                break expr;
                            } else {
                                self.func.blocks.remove(block.id);
                                break mem::take(inner);
                            }
                        }
                    };
                }
            }

            Expr::Block(..) | Expr::Loop(..) => expr,

            Expr::If(block_ty, condition, then_block, else_block) => match *condition {
                Expr::Value(Value::I32(0), _) => {
                    self.expr(ctx, &Expr::Block(block_ty, else_block))
                }

                Expr::Value(Value::I32(_), _) => {
                    self.expr(ctx, &Expr::Block(block_ty, then_block))
                }

                _ => Expr::If(
                    block_ty,
                    condition,
                    self.block(ctx, &then_block),
                    self.block(ctx, &else_block),
                ),
            },

            Expr::Br(..) | Expr::BrIf(..) | Expr::BrTable(..) | Expr::Return(_) => expr,

            Expr::Call(func_id, args) => self.call(ctx, func_id, args),

            Expr::CallIndirect(ty_id, table_id, ref mut args, ref idx) => {
                match &self.spec.module.tables[table_id].def {
                    TableDef::Import(_) => expr,

                    TableDef::Elems(elems) => match idx.to_u32() {
                        Some(idx) => {
                            let Some(&elem) = elems.get(idx as usize) else {
                                warn!(
                                    "aborting specialization: an out-of-bounds indirect call (index {idx}, table size {})",
                                    elems.len(),
                                );
                                ctx.abort_specialization();

                                return expr;
                            };

                            let Some(func_id) = elem else {
                                warn!("aborting specialization: an indirect call to an uninitialized element (index {idx})");
                                ctx.abort_specialization();

                                return expr;
                            };

                            let func_ty = self.spec.module.funcs[func_id].ty();

                            match &self.spec.module.types[ty_id] {
                                Type::Func(annotation_ty) if func_ty == annotation_ty => {}

                                annotation_ty => {
                                    warn!(
                                        "aborting specialization: an indirect call is annotated with a wrong type \
                                        (index {idx}, annotated as {annotation_ty}, actual type is {func_ty})",
                                    );
                                    ctx.abort_specialization();

                                    return expr;
                                }
                            }

                            self.call(ctx, func_id, mem::take(args))
                        }

                        None => expr,
                    },
                }
            }
        }
    }

    fn call(&mut self, ctx: &mut SpecContext, func_id: FuncId, mut args: Vec<Expr>) -> Expr {
        if ctx.abort_specialization {
            return Expr::Call(func_id, args);
        }

        if let Some(body) = self.spec.module.funcs[func_id].body() {
            if body.params.len() != args.len() {
                warn!(
                    "aborting specialization: call to a function with invalid number of arguments: expected {}, got {}",
                    body.params.len(),
                    args.len(),
                );
                ctx.abort_specialization();

                return Expr::Call(func_id, args);
            }
        }

        let spec_func_id = self.spec.specialize(SpecSignature {
            orig_func_id: func_id,
            args: args.iter().map(|expr| expr.to_value()).collect(),
        });

        trace!("specialized Expr::Call({func_id:?}, {args:?}) -> {spec_func_id:?}");

        args.retain(|expr| expr.to_value().is_none());

        self.inline(ctx, spec_func_id, args)
    }

    fn inline(&mut self, ctx: &mut SpecContext, func_id: FuncId, args: Vec<Expr>) -> Expr {
        if matches!(
            self.spec
                .spec_funcs
                .get(func_id)
                .and_then(|sig| self.spec.spec_sigs.get(sig)),
            Some(SpecializedFunc::Pending(_))
        ) {
            trace!("aborting inlining {func_id:?}: this is a pending-specialization function");
            return Expr::Call(func_id, args);
        }

        let Some(body) = self.spec.module.funcs[func_id].body() else {
            trace!("aborting inlining {func_id:?}: this is an import");
            return Expr::Call(func_id, args);
        };

        let mut locals = SparseSecondaryMap::<LocalId, LocalId>::new();

        for (local_id, val_ty) in &body.locals {
            locals.insert(local_id, self.func.locals.insert(val_ty.clone()));
        }

        let mut blocks = SecondaryMap::<BlockId, BlockId>::new();

        for block_id in body.blocks.keys() {
            blocks.insert(block_id, self.func.blocks.insert(()));
        }

        self.expr(
            ctx,
            &Expr::Block(
                body.ty.ret.clone().into(),
                Block {
                    body: args
                        .into_iter()
                        .enumerate()
                        .map(|(idx, expr)| {
                            Expr::Unary(UnOp::LocalSet(locals[body.params[idx]]), Box::new(expr))
                        })
                        .chain(body.main_block.body.iter().map(|expr| {
                            expr.map(&mut |mut expr: Expr, _: &mut VisitContext| match expr {
                                Expr::Nullary(NulOp::LocalGet(ref mut local_id))
                                | Expr::Unary(
                                    UnOp::LocalSet(ref mut local_id)
                                    | UnOp::LocalTee(ref mut local_id),
                                    _,
                                ) => {
                                    *local_id = locals[*local_id];
                                    expr
                                }

                                Expr::Block(_, ref mut block) | Expr::Loop(_, ref mut block) => {
                                    block.id = blocks[block.id];
                                    expr
                                }

                                Expr::If(_, _, ref mut then_block, ref mut else_block) => {
                                    then_block.id = blocks[then_block.id];
                                    else_block.id = blocks[else_block.id];
                                    expr
                                }

                                Expr::Br(ref mut block_id, _)
                                | Expr::BrIf(ref mut block_id, _, _) => {
                                    *block_id = blocks[*block_id];
                                    expr
                                }

                                Expr::BrTable(ref mut block_ids, ref mut block_id, _, _) => {
                                    for block_id in block_ids {
                                        *block_id = blocks[*block_id];
                                    }

                                    *block_id = blocks[*block_id];

                                    expr
                                }

                                Expr::Return(inner) => Expr::Br(blocks[body.main_block.id], inner),

                                _ => expr,
                            })
                        }))
                        .collect(),
                    id: blocks[body.main_block.id],
                },
            ),
        )
    }

    fn intr_specialize(
        &mut self,
        ctx: &mut SpecContext,
        table_id: TableId,
        elem_idx: u32,
        mem_id: MemoryId,
        name_addr: u32,
        name_len: u32,
        args: &mut Vec<Option<Value>>,
    ) -> Result<Expr, ()> {
        let table = &self.spec.module.tables[table_id];

        let elems = match &table.def {
            TableDef::Import(_) => {
                warn!("stitch/specialize references an imported table");
                ctx.abort_specialization();
                return Err(());
            }

            TableDef::Elems(elems) => elems,
        };

        let &func_id = elems.get(elem_idx as usize).ok_or_else(|| {
            warn!("stitch/specialize references an out-of-bounds table entry ({elem_idx})");
        })?;
        let func_id = func_id.ok_or_else(|| {
            warn!("stitch/specialize references an uninitialized table entry ({elem_idx})");
        })?;

        let range = (name_addr as usize)..(name_addr.saturating_add(name_len) as usize);

        let name =
            match name_len {
                0 => None,
                _ => Some(
                    String::from_utf8_lossy(self.spec.module.read_mem(mem_id, range).map_err(
                        |e| warn!("stitch/specialize does not provide a valid name: {e}"),
                    )?)
                    .into_owned(),
                ),
            };

        let spec_func_id = self.spec.specialize(SpecSignature {
            orig_func_id: func_id,
            args: mem::take(args),
        });

        if let Some(name) = name {
            self.spec.module.exports.insert(Export {
                name,
                def: ExportDef::Func(spec_func_id),
            });
        }

        Ok(Expr::Index(spec_func_id.into()))
    }
}
