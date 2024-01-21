use std::collections::HashMap;
use std::mem;

use log::warn;
use slotmap::{SecondaryMap, SparseSecondaryMap};

use crate::ir::expr::{
    BinOp, Load, MemArg, NulOp, PtrAttr, TernOp, UnOp, Value, ValueAttrs, F32, F64,
};
use crate::ir::{Expr, Func, FuncBody, FuncId, LocalId, MemoryDef, Module};

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
        self.spec_sigs
            .insert(sig.clone(), SpecializedFunc::Pending(func_id));
        self.spec_funcs.insert(func_id, sig.clone());

        FuncSpecializer::specialize(self, sig, &mut body);

        *self.module.funcs[func_id].body_mut().unwrap() = body;

        func_id
    }
}

struct FuncSpecializer<'a, 'b, 'm> {
    spec: &'a mut Specializer<'m>,
    sig: SpecSignature,
    locals: SecondaryMap<LocalId, (Value, ValueAttrs)>,
    func: &'b FuncBody,
}

impl<'a, 'b, 'm> FuncSpecializer<'a, 'b, 'm> {
    fn specialize(spec: &'a mut Specializer<'m>, sig: SpecSignature, func: &'b mut FuncBody) {
        let body = mem::take(&mut func.body);
        let mut this = FuncSpecializer {
            spec,
            sig,
            locals: Default::default(),
            func,
        };

        for (idx, arg) in this.sig.args.iter().enumerate() {
            if let &Some(value) = arg {
                this.locals
                    .insert(func.params[idx], (value, Default::default()));
            }
        }

        func.body = this.block(&body);
    }

    fn block(&mut self, exprs: &[Expr]) -> Vec<Expr> {
        exprs
            .iter()
            // FIXME: this eagerly evaluates block bodies, which is not a great idea
            .map(|expr| expr.map(&mut |expr| self.expr(expr)))
            .collect()
    }

    fn expr(&mut self, expr: Expr) -> Expr {
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
                match &self.spec.module.mems[mem_id].def {
                    MemoryDef::Import(_) => {}

                    MemoryDef::Bytes(bytes) => {
                        let start = (addr as u32 + offset) as usize;
                        let end = start + load.src_size() as usize;

                        if let Some(src) = bytes.get(start..end) {
                            debug_assert!(src.len() <= 8);

                            // little-endian
                            let mut value =
                                src.iter().rfold(0u64, |acc, &byte| acc << 8 | byte as u64);

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
                        } else {
                            warn!("Out-of-bounds constant read of {start}..{end}");
                        }
                    }
                }
            }

            _ => {}
        }

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

        match expr {
            Expr::Value(_, _) => expr,

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

            Expr::Nullary(NulOp::GlobalGet(_)) => expr, // TODO
            Expr::Unary(UnOp::GlobalSet(_), _) => expr, // TODO

            Expr::Nullary(op) => match op {
                NulOp::MemorySize(mem_id) => todo!(),
                NulOp::Nop => todo!(),
                NulOp::Unreachable => todo!(),

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
                    | UnOp::I64Load32U(_) => return expr,

                    UnOp::MemoryGrow(mem_id) => todo!(),
                };

                Expr::Value(result, attr)
            }

            Expr::Binary(op, ref lhs, ref rhs) => {
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
                    BinOp::I32DivS => Value::I32(try_i32!(lhs).wrapping_div(try_i32!(rhs))),
                    BinOp::I32DivU => Value::I32(try_u32!(lhs).wrapping_div(try_u32!(rhs)) as i32),
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
                    BinOp::I64DivS => Value::I64(try_i64!(lhs).wrapping_div(try_i64!(rhs))),
                    BinOp::I64DivU => Value::I64(try_u64!(lhs).wrapping_div(try_u64!(rhs)) as i64),
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

            Expr::Ternary(TernOp::Select, first, second, condition) => match condition.to_i32() {
                Some(0) => *second,
                Some(_) => *first,
                _ => Expr::Ternary(TernOp::Select, first, second, condition),
            },

            Expr::Block(_, _) => todo!(),
            Expr::Loop(_, _) => todo!(),
            Expr::If(_, _, _, _) => todo!(),
            Expr::Br(_, _) => todo!(),
            Expr::BrIf(_, _, _) => todo!(),
            Expr::BrTable(_, _, _, _) => todo!(),
            Expr::Return(_) => todo!(),
            Expr::Call(_, _) => todo!(),
            Expr::CallIndirect(_, _, _) => todo!(),
        }
    }
}
