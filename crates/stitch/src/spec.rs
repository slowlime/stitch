use std::collections::HashMap;
use std::mem;

use slotmap::{SecondaryMap, SparseSecondaryMap};

use crate::ir::expr::{Value, F32, F64};
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
        exprs.iter().map(|expr| expr.map(&mut |expr| self.expr(expr))).collect()
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

        match &expr {
            Expr::I32(_) | Expr::I64(_) | Expr::F32(_) | Expr::F64(_) => expr,

            Expr::I32Clz(inner) => Expr::I32(try_i32!(inner).leading_zeros() as i32),
            Expr::I32Ctz(inner) => Expr::I32(try_i32!(inner).trailing_zeros() as i32),
            Expr::I32Popcnt(inner) => Expr::I32(try_i32!(inner).count_ones() as i32),

            Expr::I64Clz(inner) => Expr::I64(try_i64!(inner).leading_zeros() as i64),
            Expr::I64Ctz(inner) => Expr::I64(try_i64!(inner).trailing_zeros() as i64),
            Expr::I64Popcnt(inner) => Expr::I64(try_i64!(inner).count_ones() as i64),

            Expr::F32Abs(inner) => Expr::F32(try_f32!(inner).to_f32().abs().into()),
            Expr::F32Neg(inner) => Expr::F32((-try_f32!(inner).to_f32()).into()),
            Expr::F32Sqrt(inner) => Expr::F32(try_f32!(inner).to_f32().sqrt().into()),
            Expr::F32Ceil(inner) => Expr::F32(try_f32!(inner).to_f32().ceil().into()),
            Expr::F32Floor(inner) => Expr::F32(try_f32!(inner).to_f32().floor().into()),
            Expr::F32Trunc(inner) => Expr::F32(try_f32!(inner).to_f32().trunc().into()),
            Expr::F32Nearest(_) => expr, // TODO

            Expr::F64Abs(inner) => Expr::F64(try_f64!(inner).to_f64().abs().into()),
            Expr::F64Neg(inner) => Expr::F64((-try_f64!(inner).to_f64()).into()),
            Expr::F64Sqrt(inner) => Expr::F64(try_f64!(inner).to_f64().sqrt().into()),
            Expr::F64Ceil(inner) => Expr::F64(try_f64!(inner).to_f64().ceil().into()),
            Expr::F64Floor(inner) => Expr::F64(try_f64!(inner).to_f64().floor().into()),
            Expr::F64Trunc(inner) => Expr::F64(try_f64!(inner).to_f64().trunc().into()),
            Expr::F64Nearest(_) => expr, // TODO

            Expr::I32Add(lhs, rhs) => Expr::I32(try_i32!(lhs).wrapping_add(try_i32!(rhs))),
            Expr::I32Sub(lhs, rhs) => Expr::I32(try_i32!(lhs).wrapping_sub(try_i32!(rhs))),
            Expr::I32Mul(lhs, rhs) => Expr::I32(try_i32!(lhs).wrapping_mul(try_i32!(rhs))),
            Expr::I32DivS(lhs, rhs) => Expr::I32(try_i32!(lhs).wrapping_div(try_i32!(rhs))),
            Expr::I32DivU(lhs, rhs) => Expr::I32(try_u32!(lhs).wrapping_div(try_u32!(rhs)) as i32),

            Expr::I32RemS(lhs, rhs) => match try_i32!(rhs) {
                0 => expr, // undefined
                rhs => Expr::I32(try_i32!(lhs).wrapping_rem(rhs)),
            },

            Expr::I32RemU(lhs, rhs) => match try_u32!(rhs) {
                0 => expr, // undefined
                rhs => Expr::I32(try_u32!(lhs).wrapping_rem(rhs) as i32),
            },

            Expr::I32And(lhs, rhs) => Expr::I32(try_i32!(lhs) & try_i32!(rhs)),
            Expr::I32Or(lhs, rhs) => Expr::I32(try_i32!(lhs) | try_i32!(rhs)),
            Expr::I32Xor(lhs, rhs) => Expr::I32(try_i32!(lhs) ^ try_i32!(rhs)),
            Expr::I32Shl(lhs, rhs) => Expr::I32(try_i32!(lhs).wrapping_shl(try_u32!(rhs))),
            Expr::I32ShrS(lhs, rhs) => Expr::I32(try_i32!(lhs).wrapping_shr(try_u32!(rhs))),
            Expr::I32ShrU(lhs, rhs) => Expr::I32(try_u32!(lhs).wrapping_shr(try_u32!(rhs)) as i32),
            Expr::I32Rotl(lhs, rhs) => Expr::I32(try_i32!(lhs).rotate_left(try_u32!(rhs))),
            Expr::I32Rotr(lhs, rhs) => Expr::I32(try_i32!(lhs).rotate_right(try_u32!(rhs))),

            Expr::I64Add(lhs, rhs) => Expr::I64(try_i64!(lhs).wrapping_add(try_i64!(rhs))),
            Expr::I64Sub(lhs, rhs) => Expr::I64(try_i64!(lhs).wrapping_sub(try_i64!(rhs))),
            Expr::I64Mul(lhs, rhs) => Expr::I64(try_i64!(lhs).wrapping_mul(try_i64!(rhs))),
            Expr::I64DivS(lhs, rhs) => Expr::I64(try_i64!(lhs).wrapping_div(try_i64!(rhs))),
            Expr::I64DivU(lhs, rhs) => Expr::I64(try_u64!(lhs).wrapping_div(try_u64!(rhs)) as i64),

            Expr::I64RemS(lhs, rhs) => match try_i64!(rhs) {
                0 => expr,
                rhs => Expr::I64(try_i64!(lhs).wrapping_rem(rhs)),
            },

            Expr::I64RemU(lhs, rhs) => match try_u64!(rhs) {
                0 => expr,
                rhs => Expr::I64(try_u64!(lhs).wrapping_rem(rhs) as i64),
            },

            Expr::I64And(lhs, rhs) => Expr::I64(try_i64!(lhs) & try_i64!(rhs)),
            Expr::I64Or(lhs, rhs) => Expr::I64(try_i64!(lhs) | try_i64!(rhs)),
            Expr::I64Xor(lhs, rhs) => Expr::I64(try_i64!(lhs) ^ try_i64!(rhs)),
            Expr::I64Shl(lhs, rhs) => Expr::I64(try_i64!(lhs).wrapping_shl(try_u64!(rhs) as u32)),
            Expr::I64ShrS(lhs, rhs) => Expr::I64(try_i64!(lhs).wrapping_shr(try_u64!(rhs) as u32)),

            Expr::I64ShrU(lhs, rhs) => {
                Expr::I64(try_u64!(lhs).wrapping_shr(try_u64!(rhs) as u32) as i64)
            }

            Expr::I64Rotl(lhs, rhs) => Expr::I64(try_i64!(lhs).rotate_left(try_u64!(rhs) as u32)),
            Expr::I64Rotr(lhs, rhs) => Expr::I64(try_i64!(lhs).rotate_right(try_u64!(rhs) as u32)),

            Expr::F32Add(lhs, rhs) => {
                Expr::F32((try_f32!(lhs).to_f32() + try_f32!(rhs).to_f32()).into())
            }

            Expr::F32Sub(lhs, rhs) => {
                Expr::F32((try_f32!(lhs).to_f32() - try_f32!(rhs).to_f32()).into())
            }

            Expr::F32Mul(lhs, rhs) => {
                Expr::F32((try_f32!(lhs).to_f32() * try_f32!(rhs).to_f32()).into())
            }

            Expr::F32Div(lhs, rhs) => {
                Expr::F32((try_f32!(lhs).to_f32() / try_f32!(rhs).to_f32()).into())
            }

            Expr::F32Min(lhs, rhs) => {
                Expr::F32(try_f32!(lhs).to_f32().min(try_f32!(rhs).to_f32()).into())
            }

            Expr::F32Max(lhs, rhs) => {
                Expr::F32(try_f32!(lhs).to_f32().max(try_f32!(rhs).to_f32()).into())
            }

            Expr::F32Copysign(lhs, rhs) => Expr::F32(
                try_f32!(lhs)
                    .to_f32()
                    .copysign(try_f32!(rhs).to_f32())
                    .into(),
            ),

            Expr::F64Add(lhs, rhs) => {
                Expr::F64((try_f64!(lhs).to_f64() + try_f64!(rhs).to_f64()).into())
            }

            Expr::F64Sub(lhs, rhs) => {
                Expr::F64((try_f64!(lhs).to_f64() - try_f64!(rhs).to_f64()).into())
            }

            Expr::F64Mul(lhs, rhs) => {
                Expr::F64((try_f64!(lhs).to_f64() * try_f64!(rhs).to_f64()).into())
            }

            Expr::F64Div(lhs, rhs) => {
                Expr::F64((try_f64!(lhs).to_f64() / try_f64!(rhs).to_f64()).into())
            }

            Expr::F64Min(lhs, rhs) => {
                Expr::F64(try_f64!(lhs).to_f64().min(try_f64!(rhs).to_f64()).into())
            }

            Expr::F64Max(lhs, rhs) => {
                Expr::F64(try_f64!(lhs).to_f64().max(try_f64!(rhs).to_f64()).into())
            }

            Expr::F64Copysign(lhs, rhs) => Expr::F64(
                try_f64!(lhs)
                    .to_f64()
                    .copysign(try_f64!(rhs).to_f64())
                    .into(),
            ),

            Expr::I32Eqz(inner) => Expr::I32((try_i32!(inner) == 0) as i32),
            Expr::I64Eqz(inner) => Expr::I32((try_i64!(inner) == 0) as i32),

            Expr::I32Eq(lhs, rhs) => Expr::I32((try_i32!(lhs) == try_i32!(rhs)) as i32),
            Expr::I32Ne(lhs, rhs) => Expr::I32((try_i32!(lhs) != try_i32!(rhs)) as i32),
            Expr::I32LtS(lhs, rhs) => Expr::I32((try_i32!(lhs) < try_i32!(rhs)) as i32),
            Expr::I32LtU(lhs, rhs) => Expr::I32((try_u32!(lhs) < try_u32!(rhs)) as i32),
            Expr::I32GtS(lhs, rhs) => Expr::I32((try_i32!(lhs) > try_i32!(rhs)) as i32),
            Expr::I32GtU(lhs, rhs) => Expr::I32((try_u32!(lhs) > try_u32!(rhs)) as i32),
            Expr::I32LeS(lhs, rhs) => Expr::I32((try_i32!(lhs) <= try_i32!(rhs)) as i32),
            Expr::I32LeU(lhs, rhs) => Expr::I32((try_u32!(lhs) <= try_u32!(rhs)) as i32),
            Expr::I32GeS(lhs, rhs) => Expr::I32((try_i32!(lhs) >= try_i32!(rhs)) as i32),
            Expr::I32GeU(lhs, rhs) => Expr::I32((try_u32!(lhs) >= try_u32!(rhs)) as i32),

            Expr::I64Eq(lhs, rhs) => Expr::I32((try_i64!(lhs) == try_i64!(rhs)) as i32),
            Expr::I64Ne(lhs, rhs) => Expr::I32((try_i64!(lhs) != try_i64!(rhs)) as i32),
            Expr::I64LtS(lhs, rhs) => Expr::I32((try_i64!(lhs) < try_i64!(rhs)) as i32),
            Expr::I64LtU(lhs, rhs) => Expr::I32((try_u64!(lhs) < try_u64!(rhs)) as i32),
            Expr::I64GtS(lhs, rhs) => Expr::I32((try_i64!(lhs) > try_i64!(rhs)) as i32),
            Expr::I64GtU(lhs, rhs) => Expr::I32((try_u64!(lhs) > try_u64!(rhs)) as i32),
            Expr::I64LeS(lhs, rhs) => Expr::I32((try_i64!(lhs) <= try_i64!(rhs)) as i32),
            Expr::I64LeU(lhs, rhs) => Expr::I32((try_u64!(lhs) <= try_u64!(rhs)) as i32),
            Expr::I64GeS(lhs, rhs) => Expr::I32((try_i64!(lhs) >= try_i64!(rhs)) as i32),
            Expr::I64GeU(lhs, rhs) => Expr::I32((try_u64!(lhs) >= try_u64!(rhs)) as i32),

            Expr::F32Eq(lhs, rhs) => Expr::I32((try_f32!(lhs) == try_f32!(rhs)) as i32),
            Expr::F32Ne(lhs, rhs) => Expr::I32((try_f32!(lhs) != try_f32!(rhs)) as i32),
            Expr::F32Lt(lhs, rhs) => Expr::I32((try_f32!(lhs) < try_f32!(rhs)) as i32),
            Expr::F32Gt(lhs, rhs) => Expr::I32((try_f32!(lhs) > try_f32!(rhs)) as i32),
            Expr::F32Le(lhs, rhs) => Expr::I32((try_f32!(lhs) <= try_f32!(rhs)) as i32),
            Expr::F32Ge(lhs, rhs) => Expr::I32((try_f32!(lhs) >= try_f32!(rhs)) as i32),

            Expr::F64Eq(lhs, rhs) => Expr::I32((try_f64!(lhs) == try_f64!(rhs)) as i32),
            Expr::F64Ne(lhs, rhs) => Expr::I32((try_f64!(lhs) != try_f64!(rhs)) as i32),
            Expr::F64Lt(lhs, rhs) => Expr::I32((try_f64!(lhs) < try_f64!(rhs)) as i32),
            Expr::F64Gt(lhs, rhs) => Expr::I32((try_f64!(lhs) > try_f64!(rhs)) as i32),
            Expr::F64Le(lhs, rhs) => Expr::I32((try_f64!(lhs) <= try_f64!(rhs)) as i32),
            Expr::F64Ge(lhs, rhs) => Expr::I32((try_f64!(lhs) >= try_f64!(rhs)) as i32),

            Expr::I32WrapI64(inner) => Expr::I32(try_i64!(inner) as i32),

            Expr::I64ExtendI32S(inner) => Expr::I64(try_i32!(inner) as i64),
            Expr::I64ExtendI32U(inner) => Expr::I64(try_u32!(inner) as i64),

            Expr::I32TruncF32S(inner) => Expr::I32(try_f32!(inner).to_f32() as i32),
            Expr::I32TruncF32U(inner) => Expr::I32(try_f32!(inner).to_f32() as u32 as i32),
            Expr::I32TruncF64S(inner) => Expr::I32(try_f64!(inner).to_f64() as i32),
            Expr::I32TruncF64U(inner) => Expr::I32(try_f64!(inner).to_f64() as u32 as i32),

            Expr::I64TruncF32S(inner) => Expr::I64(try_f32!(inner).to_f32() as i64),
            Expr::I64TruncF32U(inner) => Expr::I64(try_f32!(inner).to_f32() as u64 as i64),
            Expr::I64TruncF64S(inner) => Expr::I64(try_f64!(inner).to_f64() as i64),
            Expr::I64TruncF64U(inner) => Expr::I64(try_f64!(inner).to_f64() as u64 as i64),

            Expr::F32DemoteF64(inner) => Expr::F32((try_f64!(inner).to_f64() as f32).into()),
            Expr::F64PromoteF32(inner) => Expr::F64((try_f32!(inner).to_f32() as f64).into()),

            Expr::F32ConvertI32S(inner) => Expr::F32((try_i32!(inner) as f32).into()),
            Expr::F32ConvertI32U(inner) => Expr::F32((try_u32!(inner) as f32).into()),
            Expr::F32ConvertI64S(inner) => Expr::F32((try_i64!(inner) as f32).into()),
            Expr::F32ConvertI64U(inner) => Expr::F32((try_u64!(inner) as f32).into()),

            Expr::F64ConvertI32S(inner) => Expr::F64((try_i32!(inner) as f64).into()),
            Expr::F64ConvertI32U(inner) => Expr::F64((try_u32!(inner) as f64).into()),
            Expr::F64ConvertI64S(inner) => Expr::F64((try_i64!(inner) as f64).into()),
            Expr::F64ConvertI64U(inner) => Expr::F64((try_u64!(inner) as f64).into()),

            Expr::F32ReinterpretI32(inner) => Expr::F32(F32::from_bits(try_u32!(inner))),
            Expr::F64ReinterpretI64(inner) => Expr::F64(F64::from_bits(try_u64!(inner))),
            Expr::I32ReinterpretF32(inner) => Expr::I32(try_f32!(inner).to_bits() as i32),
            Expr::I64ReinterpretF64(inner) => Expr::I64(try_f64!(inner).to_bits() as i64),

            Expr::I32Extend8S(inner) => Expr::I32(try_u32!(inner) as i8 as i32),
            Expr::I32Extend16S(inner) => Expr::I32(try_u32!(inner) as i16 as i32),
            Expr::I64Extend8S(inner) => Expr::I64(try_u64!(inner) as i8 as i64),
            Expr::I64Extend16S(inner) => Expr::I64(try_u64!(inner) as i16 as i64),
            Expr::I64Extend32S(inner) => Expr::I64(try_u64!(inner) as i32 as i64),

            Expr::Drop(_) => todo!(),
            Expr::Select(_, _, _) => todo!(),

            Expr::LocalGet(_) => todo!(),
            Expr::LocalSet(_, _) => todo!(),
            Expr::LocalTee(_, _) => todo!(),

            Expr::GlobalGet(_) => todo!(),
            Expr::GlobalSet(_, _) => todo!(),

            Expr::I32Load(_, _) => todo!(),
            Expr::I64Load(_, _) => todo!(),
            Expr::F32Load(_, _) => todo!(),
            Expr::F64Load(_, _) => todo!(),

            Expr::I32Store(_, _, _) => todo!(),
            Expr::I64Store(_, _, _) => todo!(),
            Expr::F32Store(_, _, _) => todo!(),
            Expr::F64Store(_, _, _) => todo!(),

            Expr::I32Load8S(_, _) => todo!(),
            Expr::I32Load8U(_, _) => todo!(),
            Expr::I32Load16S(_, _) => todo!(),
            Expr::I32Load16U(_, _) => todo!(),

            Expr::I64Load8S(_, _) => todo!(),
            Expr::I64Load8U(_, _) => todo!(),
            Expr::I64Load16S(_, _) => todo!(),
            Expr::I64Load16U(_, _) => todo!(),
            Expr::I64Load32S(_, _) => todo!(),
            Expr::I64Load32U(_, _) => todo!(),

            Expr::I32Store8(_, _, _) => todo!(),
            Expr::I32Store16(_, _, _) => todo!(),

            Expr::I64Store8(_, _, _) => todo!(),
            Expr::I64Store16(_, _, _) => todo!(),
            Expr::I64Store32(_, _, _) => todo!(),

            Expr::MemorySize => todo!(),
            Expr::MemoryGrow(_) => todo!(),

            Expr::Nop => todo!(),
            Expr::Unreachable => todo!(),

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
