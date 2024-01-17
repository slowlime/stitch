use std::collections::HashMap;
use std::mem;

use slotmap::{SecondaryMap, SparseSecondaryMap};

use crate::ir::expr::Value;
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
        exprs.iter().map(|expr| self.expr(expr)).collect()
    }

    fn expr(&mut self, expr: &Expr) -> Expr {
        macro_rules! try_expr {
            ($expr:expr, $variant:ident) => {{
                match **$expr {
                    Expr::$variant(value) => value,

                    ref expr => {
                        return expr.clone();
                    }
                }
            }};
        }

        macro_rules! try_i32 {
            ($expr:expr) => {
                try_expr!($expr, I32)
            };
        }

        macro_rules! try_i64 {
            ($expr:expr) => {
                try_expr!($expr, I64)
            };
        }

        match expr {
            Expr::I32(_) | Expr::I64(_) | Expr::F32(_) | Expr::F64(_) => expr.clone(),

            Expr::I32Clz(inner) => Expr::I32(try_i32!(inner).leading_zeros() as i32),
            Expr::I32Ctz(inner) => Expr::I32(try_i32!(inner).trailing_zeros() as i32),
            Expr::I32Popcnt(inner) => Expr::I32(try_i32!(inner).count_ones() as i32),

            Expr::I64Clz(inner) => Expr::I64(try_i64!(inner).leading_zeros() as i64),
            Expr::I64Ctz(inner) => Expr::I64(try_i64!(inner).trailing_zeros() as i64),
            Expr::I64Popcnt(inner) => Expr::I64(try_i64!(inner).count_ones() as i64),

            Expr::F32Abs(_) => todo!(),
            Expr::F32Neg(_) => todo!(),
            Expr::F32Sqrt(_) => todo!(),
            Expr::F32Ceil(_) => todo!(),
            Expr::F32Floor(_) => todo!(),
            Expr::F32Trunc(_) => todo!(),
            Expr::F32Nearest(_) => todo!(),

            Expr::F64Abs(_) => todo!(),
            Expr::F64Neg(_) => todo!(),
            Expr::F64Sqrt(_) => todo!(),
            Expr::F64Ceil(_) => todo!(),
            Expr::F64Floor(_) => todo!(),
            Expr::F64Trunc(_) => todo!(),
            Expr::F64Nearest(_) => todo!(),

            Expr::I32Add(_, _) => todo!(),
            Expr::I32Sub(_, _) => todo!(),
            Expr::I32Mul(_, _) => todo!(),
            Expr::I32DivS(_, _) => todo!(),
            Expr::I32DivU(_, _) => todo!(),
            Expr::I32RemS(_, _) => todo!(),
            Expr::I32RemU(_, _) => todo!(),
            Expr::I32And(_, _) => todo!(),
            Expr::I32Or(_, _) => todo!(),
            Expr::I32Xor(_, _) => todo!(),
            Expr::I32Shl(_, _) => todo!(),
            Expr::I32ShrS(_, _) => todo!(),
            Expr::I32ShrU(_, _) => todo!(),
            Expr::I32Rotl(_, _) => todo!(),
            Expr::I32Rotr(_, _) => todo!(),

            Expr::I64Add(_, _) => todo!(),
            Expr::I64Sub(_, _) => todo!(),
            Expr::I64Mul(_, _) => todo!(),
            Expr::I64DivS(_, _) => todo!(),
            Expr::I64DivU(_, _) => todo!(),
            Expr::I64RemS(_, _) => todo!(),
            Expr::I64RemU(_, _) => todo!(),
            Expr::I64And(_, _) => todo!(),
            Expr::I64Or(_, _) => todo!(),
            Expr::I64Xor(_, _) => todo!(),
            Expr::I64Shl(_, _) => todo!(),
            Expr::I64ShrS(_, _) => todo!(),
            Expr::I64ShrU(_, _) => todo!(),
            Expr::I64Rotl(_, _) => todo!(),
            Expr::I64Rotr(_, _) => todo!(),

            Expr::F32Add(_, _) => todo!(),
            Expr::F32Sub(_, _) => todo!(),
            Expr::F32Mul(_, _) => todo!(),
            Expr::F32Div(_, _) => todo!(),
            Expr::F32Min(_, _) => todo!(),
            Expr::F32Max(_, _) => todo!(),
            Expr::F32Copysign(_, _) => todo!(),

            Expr::F64Add(_, _) => todo!(),
            Expr::F64Sub(_, _) => todo!(),
            Expr::F64Mul(_, _) => todo!(),
            Expr::F64Div(_, _) => todo!(),
            Expr::F64Min(_, _) => todo!(),
            Expr::F64Max(_, _) => todo!(),
            Expr::F64Copysign(_, _) => todo!(),

            Expr::I32Eqz(_) => todo!(),
            Expr::I64Eqz(_) => todo!(),

            Expr::I32Eq(_, _) => todo!(),
            Expr::I32Ne(_, _) => todo!(),
            Expr::I32LtS(_, _) => todo!(),
            Expr::I32LtU(_, _) => todo!(),
            Expr::I32GtS(_, _) => todo!(),
            Expr::I32GtU(_, _) => todo!(),
            Expr::I32LeS(_, _) => todo!(),
            Expr::I32LeU(_, _) => todo!(),
            Expr::I32GeS(_, _) => todo!(),
            Expr::I32GeU(_, _) => todo!(),

            Expr::I64Eq(_, _) => todo!(),
            Expr::I64Ne(_, _) => todo!(),
            Expr::I64LtS(_, _) => todo!(),
            Expr::I64LtU(_, _) => todo!(),
            Expr::I64GtS(_, _) => todo!(),
            Expr::I64GtU(_, _) => todo!(),
            Expr::I64LeS(_, _) => todo!(),
            Expr::I64LeU(_, _) => todo!(),
            Expr::I64GeS(_, _) => todo!(),
            Expr::I64GeU(_, _) => todo!(),

            Expr::F32Eq(_, _) => todo!(),
            Expr::F32Ne(_, _) => todo!(),
            Expr::F32Lt(_, _) => todo!(),
            Expr::F32Gt(_, _) => todo!(),
            Expr::F32Le(_, _) => todo!(),
            Expr::F32Ge(_, _) => todo!(),

            Expr::F64Eq(_, _) => todo!(),
            Expr::F64Ne(_, _) => todo!(),
            Expr::F64Lt(_, _) => todo!(),
            Expr::F64Gt(_, _) => todo!(),
            Expr::F64Le(_, _) => todo!(),
            Expr::F64Ge(_, _) => todo!(),

            Expr::I32WrapI64(_) => todo!(),

            Expr::I64ExtendI32S(_) => todo!(),
            Expr::I64ExtendI32U(_) => todo!(),

            Expr::I32TruncF32S(_) => todo!(),
            Expr::I32TruncF32U(_) => todo!(),
            Expr::I32TruncF64S(_) => todo!(),
            Expr::I32TruncF64U(_) => todo!(),

            Expr::I64TruncF32S(_) => todo!(),
            Expr::I64TruncF32U(_) => todo!(),
            Expr::I64TruncF64S(_) => todo!(),
            Expr::I64TruncF64U(_) => todo!(),

            Expr::F32DemoteF64(_) => todo!(),
            Expr::F64PromoteF32(_) => todo!(),

            Expr::F32ConvertI32S(_) => todo!(),
            Expr::F32ConvertI32U(_) => todo!(),
            Expr::F32ConvertI64S(_) => todo!(),
            Expr::F32ConvertI64U(_) => todo!(),

            Expr::F64ConvertI32S(_) => todo!(),
            Expr::F64ConvertI32U(_) => todo!(),
            Expr::F64ConvertI64S(_) => todo!(),
            Expr::F64ConvertI64U(_) => todo!(),

            Expr::F32ReinterpretI32(_) => todo!(),
            Expr::F64ReinterpretI64(_) => todo!(),
            Expr::I32ReinterpretF32(_) => todo!(),
            Expr::I64ReinterpretF64(_) => todo!(),

            Expr::I32Extend8S(_) => todo!(),
            Expr::I32Extend16S(_) => todo!(),
            Expr::I64Extend8S(_) => todo!(),
            Expr::I64Extend16S(_) => todo!(),
            Expr::I64Extend32S(_) => todo!(),

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
