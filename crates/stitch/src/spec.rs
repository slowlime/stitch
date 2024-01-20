use std::collections::HashMap;
use std::mem;

use slotmap::{SecondaryMap, SparseSecondaryMap};

use crate::ir::expr::{BinOp, NulOp, TernOp, UnOp, Value, F32, F64};
use crate::ir::{Expr, Func, FuncBody, FuncId, LocalId, Module};

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
    locals: SecondaryMap<LocalId, Value>,
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
                this.locals.insert(func.params[idx], value);
            }
        }

        func.body = this.block(&body);
    }

    fn block(&mut self, exprs: &[Expr]) -> Vec<Expr> {
        exprs
            .iter()
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

        match expr {
            Expr::I32(_) | Expr::I64(_) | Expr::F32(_) | Expr::F64(_) => expr,

            Expr::Nullary(NulOp::LocalGet(local_id)) => match self.locals.get(local_id) {
                Some(value) => value.to_expr(),
                None => expr,
            },

            Expr::Unary(UnOp::LocalSet(local_id), ref inner) => match inner.to_value() {
                Some(value) => {
                    self.locals.insert(local_id, value);

                    NulOp::Nop.into()
                }

                None => {
                    self.locals.remove(local_id);

                    expr
                }
            },

            Expr::Unary(UnOp::LocalTee(local_id), ref inner) => match inner.to_value() {
                Some(value) => {
                    self.locals.insert(local_id, value);

                    value.to_expr()
                }

                None => {
                    self.locals.remove(local_id);

                    expr
                }
            },

            Expr::Nullary(NulOp::GlobalGet(_)) => expr, // TODO
            Expr::Unary(UnOp::GlobalSet(_), _) => expr, // TODO

            Expr::Nullary(op) => match op {
                NulOp::MemorySize => todo!(),
                NulOp::Nop => todo!(),
                NulOp::Unreachable => todo!(),

                NulOp::LocalGet(_) | NulOp::GlobalGet(_) => unreachable!(),
            },

            Expr::Unary(op, ref inner) => match op {
                UnOp::I32Clz => Expr::I32(try_i32!(inner).leading_zeros() as i32),
                UnOp::I32Ctz => Expr::I32(try_i32!(inner).trailing_zeros() as i32),
                UnOp::I32Popcnt => Expr::I32(try_i32!(inner).count_ones() as i32),

                UnOp::I64Clz => Expr::I64(try_i64!(inner).leading_zeros() as i64),
                UnOp::I64Ctz => Expr::I64(try_i64!(inner).trailing_zeros() as i64),
                UnOp::I64Popcnt => Expr::I64(try_i64!(inner).count_ones() as i64),

                UnOp::F32Abs => Expr::F32(try_f32!(inner).to_f32().abs().into()),
                UnOp::F32Neg => Expr::F32((-try_f32!(inner).to_f32()).into()),
                UnOp::F32Sqrt => Expr::F32(try_f32!(inner).to_f32().sqrt().into()),
                UnOp::F32Ceil => Expr::F32(try_f32!(inner).to_f32().ceil().into()),
                UnOp::F32Floor => Expr::F32(try_f32!(inner).to_f32().floor().into()),
                UnOp::F32Trunc => Expr::F32(try_f32!(inner).to_f32().trunc().into()),
                UnOp::F32Nearest => expr, // TODO

                UnOp::F64Abs => Expr::F64(try_f64!(inner).to_f64().abs().into()),
                UnOp::F64Neg => Expr::F64((-try_f64!(inner).to_f64()).into()),
                UnOp::F64Sqrt => Expr::F64(try_f64!(inner).to_f64().sqrt().into()),
                UnOp::F64Ceil => Expr::F64(try_f64!(inner).to_f64().ceil().into()),
                UnOp::F64Floor => Expr::F64(try_f64!(inner).to_f64().floor().into()),
                UnOp::F64Trunc => Expr::F64(try_f64!(inner).to_f64().trunc().into()),
                UnOp::F64Nearest => expr, // TODO

                UnOp::I32Eqz => Expr::I32((try_i32!(inner) == 0) as i32),
                UnOp::I64Eqz => Expr::I32((try_i64!(inner) == 0) as i32),

                UnOp::I32WrapI64 => Expr::I32(try_i64!(inner) as i32),

                UnOp::I64ExtendI32S => Expr::I64(try_i32!(inner) as i64),
                UnOp::I64ExtendI32U => Expr::I64(try_u32!(inner) as i64),

                UnOp::I32TruncF32S => Expr::I32(try_f32!(inner).to_f32() as i32),
                UnOp::I32TruncF32U => Expr::I32(try_f32!(inner).to_f32() as u32 as i32),
                UnOp::I32TruncF64S => Expr::I32(try_f64!(inner).to_f64() as i32),
                UnOp::I32TruncF64U => Expr::I32(try_f64!(inner).to_f64() as u32 as i32),

                UnOp::I64TruncF32S => Expr::I64(try_f32!(inner).to_f32() as i64),
                UnOp::I64TruncF32U => Expr::I64(try_f32!(inner).to_f32() as u64 as i64),
                UnOp::I64TruncF64S => Expr::I64(try_f64!(inner).to_f64() as i64),
                UnOp::I64TruncF64U => Expr::I64(try_f64!(inner).to_f64() as u64 as i64),

                UnOp::F32DemoteF64 => Expr::F32((try_f64!(inner).to_f64() as f32).into()),
                UnOp::F64PromoteF32 => Expr::F64((try_f32!(inner).to_f32() as f64).into()),

                UnOp::F32ConvertI32S => Expr::F32((try_i32!(inner) as f32).into()),
                UnOp::F32ConvertI32U => Expr::F32((try_u32!(inner) as f32).into()),
                UnOp::F32ConvertI64S => Expr::F32((try_i64!(inner) as f32).into()),
                UnOp::F32ConvertI64U => Expr::F32((try_u64!(inner) as f32).into()),

                UnOp::F64ConvertI32S => Expr::F64((try_i32!(inner) as f64).into()),
                UnOp::F64ConvertI32U => Expr::F64((try_u32!(inner) as f64).into()),
                UnOp::F64ConvertI64S => Expr::F64((try_i64!(inner) as f64).into()),
                UnOp::F64ConvertI64U => Expr::F64((try_u64!(inner) as f64).into()),

                UnOp::F32ReinterpretI32 => Expr::F32(F32::from_bits(try_u32!(inner))),
                UnOp::F64ReinterpretI64 => Expr::F64(F64::from_bits(try_u64!(inner))),
                UnOp::I32ReinterpretF32 => Expr::I32(try_f32!(inner).to_bits() as i32),
                UnOp::I64ReinterpretF64 => Expr::I64(try_f64!(inner).to_bits() as i64),

                UnOp::I32Extend8S => Expr::I32(try_u32!(inner) as i8 as i32),
                UnOp::I32Extend16S => Expr::I32(try_u32!(inner) as i16 as i32),

                UnOp::I64Extend8S => Expr::I64(try_u64!(inner) as i8 as i64),
                UnOp::I64Extend16S => Expr::I64(try_u64!(inner) as i16 as i64),
                UnOp::I64Extend32S => Expr::I64(try_u64!(inner) as i32 as i64),

                UnOp::Drop => expr,

                UnOp::LocalSet(_) | UnOp::LocalTee(_) | UnOp::GlobalSet(_) => unreachable!(),

                UnOp::I32Load(_) => todo!(),
                UnOp::I64Load(_) => todo!(),
                UnOp::F32Load(_) => todo!(),
                UnOp::F64Load(_) => todo!(),

                UnOp::I32Load8S(_) => todo!(),
                UnOp::I32Load8U(_) => todo!(),
                UnOp::I32Load16S(_) => todo!(),
                UnOp::I32Load16U(_) => todo!(),

                UnOp::I64Load8S(_) => todo!(),
                UnOp::I64Load8U(_) => todo!(),
                UnOp::I64Load16S(_) => todo!(),
                UnOp::I64Load16U(_) => todo!(),
                UnOp::I64Load32S(_) => todo!(),
                UnOp::I64Load32U(_) => todo!(),

                UnOp::MemoryGrow => todo!(),
            },

            Expr::Binary(op, ref lhs, ref rhs) => match op {
                BinOp::I32Add => Expr::I32(try_i32!(lhs).wrapping_add(try_i32!(rhs))),
                BinOp::I32Sub => Expr::I32(try_i32!(lhs).wrapping_sub(try_i32!(rhs))),
                BinOp::I32Mul => Expr::I32(try_i32!(lhs).wrapping_mul(try_i32!(rhs))),
                BinOp::I32DivS => Expr::I32(try_i32!(lhs).wrapping_div(try_i32!(rhs))),
                BinOp::I32DivU => Expr::I32(try_u32!(lhs).wrapping_div(try_u32!(rhs)) as i32),
                BinOp::I32RemS => match try_i32!(rhs) {
                    0 => expr, // undefined
                    rhs => Expr::I32(try_i32!(lhs).wrapping_rem(rhs)),
                },
                BinOp::I32RemU => match try_u32!(rhs) {
                    0 => expr, // undefined
                    rhs => Expr::I32(try_u32!(lhs).wrapping_rem(rhs) as i32),
                },
                BinOp::I32And => Expr::I32(try_i32!(lhs) & try_i32!(rhs)),
                BinOp::I32Or => Expr::I32(try_i32!(lhs) | try_i32!(rhs)),
                BinOp::I32Xor => Expr::I32(try_i32!(lhs) ^ try_i32!(rhs)),
                BinOp::I32Shl => Expr::I32(try_i32!(lhs).wrapping_shl(try_u32!(rhs))),
                BinOp::I32ShrS => Expr::I32(try_i32!(lhs).wrapping_shr(try_u32!(rhs))),
                BinOp::I32ShrU => Expr::I32(try_u32!(lhs).wrapping_shr(try_u32!(rhs)) as i32),
                BinOp::I32Rotl => Expr::I32(try_i32!(lhs).rotate_left(try_u32!(rhs))),
                BinOp::I32Rotr => Expr::I32(try_i32!(lhs).rotate_right(try_u32!(rhs))),

                BinOp::I64Add => Expr::I64(try_i64!(lhs).wrapping_add(try_i64!(rhs))),
                BinOp::I64Sub => Expr::I64(try_i64!(lhs).wrapping_sub(try_i64!(rhs))),
                BinOp::I64Mul => Expr::I64(try_i64!(lhs).wrapping_mul(try_i64!(rhs))),
                BinOp::I64DivS => Expr::I64(try_i64!(lhs).wrapping_div(try_i64!(rhs))),
                BinOp::I64DivU => Expr::I64(try_u64!(lhs).wrapping_div(try_u64!(rhs)) as i64),
                BinOp::I64RemS => match try_i64!(rhs) {
                    0 => expr,
                    rhs => Expr::I64(try_i64!(lhs).wrapping_rem(rhs)),
                },
                BinOp::I64RemU => match try_u64!(rhs) {
                    0 => expr,
                    rhs => Expr::I64(try_u64!(lhs).wrapping_rem(rhs) as i64),
                },
                BinOp::I64And => Expr::I64(try_i64!(lhs) & try_i64!(rhs)),
                BinOp::I64Or => Expr::I64(try_i64!(lhs) | try_i64!(rhs)),
                BinOp::I64Xor => Expr::I64(try_i64!(lhs) ^ try_i64!(rhs)),
                BinOp::I64Shl => Expr::I64(try_i64!(lhs).wrapping_shl(try_u64!(rhs) as u32)),
                BinOp::I64ShrS => Expr::I64(try_i64!(lhs).wrapping_shr(try_u64!(rhs) as u32)),
                BinOp::I64ShrU => {
                    Expr::I64(try_u64!(lhs).wrapping_shr(try_u64!(rhs) as u32) as i64)
                }
                BinOp::I64Rotl => Expr::I64(try_i64!(lhs).rotate_left(try_u64!(rhs) as u32)),
                BinOp::I64Rotr => Expr::I64(try_i64!(lhs).rotate_right(try_u64!(rhs) as u32)),

                BinOp::F32Add => {
                    Expr::F32((try_f32!(lhs).to_f32() + try_f32!(rhs).to_f32()).into())
                }
                BinOp::F32Sub => {
                    Expr::F32((try_f32!(lhs).to_f32() - try_f32!(rhs).to_f32()).into())
                }
                BinOp::F32Mul => {
                    Expr::F32((try_f32!(lhs).to_f32() * try_f32!(rhs).to_f32()).into())
                }
                BinOp::F32Div => {
                    Expr::F32((try_f32!(lhs).to_f32() / try_f32!(rhs).to_f32()).into())
                }
                BinOp::F32Min => {
                    Expr::F32(try_f32!(lhs).to_f32().min(try_f32!(rhs).to_f32()).into())
                }
                BinOp::F32Max => {
                    Expr::F32(try_f32!(lhs).to_f32().max(try_f32!(rhs).to_f32()).into())
                }
                BinOp::F32Copysign => Expr::F32(
                    try_f32!(lhs)
                        .to_f32()
                        .copysign(try_f32!(rhs).to_f32())
                        .into(),
                ),

                BinOp::F64Add => {
                    Expr::F64((try_f64!(lhs).to_f64() + try_f64!(rhs).to_f64()).into())
                }
                BinOp::F64Sub => {
                    Expr::F64((try_f64!(lhs).to_f64() - try_f64!(rhs).to_f64()).into())
                }
                BinOp::F64Mul => {
                    Expr::F64((try_f64!(lhs).to_f64() * try_f64!(rhs).to_f64()).into())
                }
                BinOp::F64Div => {
                    Expr::F64((try_f64!(lhs).to_f64() / try_f64!(rhs).to_f64()).into())
                }
                BinOp::F64Min => {
                    Expr::F64(try_f64!(lhs).to_f64().min(try_f64!(rhs).to_f64()).into())
                }
                BinOp::F64Max => {
                    Expr::F64(try_f64!(lhs).to_f64().max(try_f64!(rhs).to_f64()).into())
                }
                BinOp::F64Copysign => Expr::F64(
                    try_f64!(lhs)
                        .to_f64()
                        .copysign(try_f64!(rhs).to_f64())
                        .into(),
                ),

                BinOp::I32Eq => Expr::I32((try_i32!(lhs) == try_i32!(rhs)) as i32),
                BinOp::I32Ne => Expr::I32((try_i32!(lhs) != try_i32!(rhs)) as i32),
                BinOp::I32LtS => Expr::I32((try_i32!(lhs) < try_i32!(rhs)) as i32),
                BinOp::I32LtU => Expr::I32((try_u32!(lhs) < try_u32!(rhs)) as i32),
                BinOp::I32GtS => Expr::I32((try_i32!(lhs) > try_i32!(rhs)) as i32),
                BinOp::I32GtU => Expr::I32((try_u32!(lhs) > try_u32!(rhs)) as i32),
                BinOp::I32LeS => Expr::I32((try_i32!(lhs) <= try_i32!(rhs)) as i32),
                BinOp::I32LeU => Expr::I32((try_u32!(lhs) <= try_u32!(rhs)) as i32),
                BinOp::I32GeS => Expr::I32((try_i32!(lhs) >= try_i32!(rhs)) as i32),
                BinOp::I32GeU => Expr::I32((try_u32!(lhs) >= try_u32!(rhs)) as i32),

                BinOp::I64Eq => Expr::I32((try_i64!(lhs) == try_i64!(rhs)) as i32),
                BinOp::I64Ne => Expr::I32((try_i64!(lhs) != try_i64!(rhs)) as i32),
                BinOp::I64LtS => Expr::I32((try_i64!(lhs) < try_i64!(rhs)) as i32),
                BinOp::I64LtU => Expr::I32((try_u64!(lhs) < try_u64!(rhs)) as i32),
                BinOp::I64GtS => Expr::I32((try_i64!(lhs) > try_i64!(rhs)) as i32),
                BinOp::I64GtU => Expr::I32((try_u64!(lhs) > try_u64!(rhs)) as i32),
                BinOp::I64LeS => Expr::I32((try_i64!(lhs) <= try_i64!(rhs)) as i32),
                BinOp::I64LeU => Expr::I32((try_u64!(lhs) <= try_u64!(rhs)) as i32),
                BinOp::I64GeS => Expr::I32((try_i64!(lhs) >= try_i64!(rhs)) as i32),
                BinOp::I64GeU => Expr::I32((try_u64!(lhs) >= try_u64!(rhs)) as i32),

                BinOp::F32Eq => Expr::I32((try_f32!(lhs) == try_f32!(rhs)) as i32),
                BinOp::F32Ne => Expr::I32((try_f32!(lhs) != try_f32!(rhs)) as i32),
                BinOp::F32Lt => Expr::I32((try_f32!(lhs) < try_f32!(rhs)) as i32),
                BinOp::F32Gt => Expr::I32((try_f32!(lhs) > try_f32!(rhs)) as i32),
                BinOp::F32Le => Expr::I32((try_f32!(lhs) <= try_f32!(rhs)) as i32),
                BinOp::F32Ge => Expr::I32((try_f32!(lhs) >= try_f32!(rhs)) as i32),

                BinOp::F64Eq => Expr::I32((try_f64!(lhs) == try_f64!(rhs)) as i32),
                BinOp::F64Ne => Expr::I32((try_f64!(lhs) != try_f64!(rhs)) as i32),
                BinOp::F64Lt => Expr::I32((try_f64!(lhs) < try_f64!(rhs)) as i32),
                BinOp::F64Gt => Expr::I32((try_f64!(lhs) > try_f64!(rhs)) as i32),
                BinOp::F64Le => Expr::I32((try_f64!(lhs) <= try_f64!(rhs)) as i32),
                BinOp::F64Ge => Expr::I32((try_f64!(lhs) >= try_f64!(rhs)) as i32),

                BinOp::I32Store(_) => todo!(),
                BinOp::I64Store(_) => todo!(),
                BinOp::F32Store(_) => todo!(),
                BinOp::F64Store(_) => todo!(),

                BinOp::I32Store8(_) => todo!(),
                BinOp::I32Store16(_) => todo!(),

                BinOp::I64Store8(_) => todo!(),
                BinOp::I64Store16(_) => todo!(),
                BinOp::I64Store32(_) => todo!(),
            },

            Expr::Ternary(TernOp::Select, first, second, condition) => match *condition {
                Expr::I32(0) => *second,
                Expr::I32(_) => *first,
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
