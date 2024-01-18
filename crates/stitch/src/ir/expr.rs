use std::fmt::{self, Debug, Display};

use super::ty::ValType;
use super::{FuncId, GlobalId, LocalId, TypeId};

type BExpr = Box<Expr>;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct F32(u32);

impl F32 {
    pub fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    pub fn from_f32(value: f32) -> Self {
        Self::from_bits(value.to_bits())
    }

    pub fn to_bits(&self) -> u32 {
        self.0
    }

    pub fn to_f32(&self) -> f32 {
        f32::from_bits(self.0)
    }
}

impl Debug for F32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.to_f32())
    }
}

impl Display for F32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

impl PartialOrd for F32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.to_f32().partial_cmp(&other.to_f32())
    }
}

impl From<f32> for F32 {
    fn from(value: f32) -> Self {
        Self::from_f32(value)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct F64(u64);

impl F64 {
    pub fn from_bits(bits: u64) -> Self {
        Self(bits)
    }

    pub fn from_f64(value: f64) -> Self {
        Self::from_bits(value.to_bits())
    }

    pub fn to_bits(&self) -> u64 {
        self.0
    }

    pub fn to_f64(&self) -> f64 {
        f64::from_bits(self.0)
    }
}

impl Debug for F64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.to_f64())
    }
}

impl Display for F64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_f64())
    }
}

impl PartialOrd for F64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.to_f64().partial_cmp(&other.to_f64())
    }
}

impl From<f64> for F64 {
    fn from(value: f64) -> Self {
        Self::from_f64(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Value {
    I32(i32),
    I64(i64),
    F32(F32),
    F64(F64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemArg {
    pub offset: u32,
    pub align: u32,
}

#[derive(Debug, Default, Clone)]
pub enum Expr {
    // numeric instructions
    I32(i32),
    I64(i64),
    F32(F32),
    F64(F64),

    I32Clz(BExpr),
    I32Ctz(BExpr),
    I32Popcnt(BExpr),

    I64Clz(BExpr),
    I64Ctz(BExpr),
    I64Popcnt(BExpr),

    F32Abs(BExpr),
    F32Neg(BExpr),
    F32Sqrt(BExpr),
    F32Ceil(BExpr),
    F32Floor(BExpr),
    F32Trunc(BExpr),
    F32Nearest(BExpr),

    F64Abs(BExpr),
    F64Neg(BExpr),
    F64Sqrt(BExpr),
    F64Ceil(BExpr),
    F64Floor(BExpr),
    F64Trunc(BExpr),
    F64Nearest(BExpr),

    I32Add(BExpr, BExpr),
    I32Sub(BExpr, BExpr),
    I32Mul(BExpr, BExpr),
    I32DivS(BExpr, BExpr),
    I32DivU(BExpr, BExpr),
    I32RemS(BExpr, BExpr),
    I32RemU(BExpr, BExpr),
    I32And(BExpr, BExpr),
    I32Or(BExpr, BExpr),
    I32Xor(BExpr, BExpr),
    I32Shl(BExpr, BExpr),
    I32ShrS(BExpr, BExpr),
    I32ShrU(BExpr, BExpr),
    I32Rotl(BExpr, BExpr),
    I32Rotr(BExpr, BExpr),

    I64Add(BExpr, BExpr),
    I64Sub(BExpr, BExpr),
    I64Mul(BExpr, BExpr),
    I64DivS(BExpr, BExpr),
    I64DivU(BExpr, BExpr),
    I64RemS(BExpr, BExpr),
    I64RemU(BExpr, BExpr),
    I64And(BExpr, BExpr),
    I64Or(BExpr, BExpr),
    I64Xor(BExpr, BExpr),
    I64Shl(BExpr, BExpr),
    I64ShrS(BExpr, BExpr),
    I64ShrU(BExpr, BExpr),
    I64Rotl(BExpr, BExpr),
    I64Rotr(BExpr, BExpr),

    F32Add(BExpr, BExpr),
    F32Sub(BExpr, BExpr),
    F32Mul(BExpr, BExpr),
    F32Div(BExpr, BExpr),
    F32Min(BExpr, BExpr),
    F32Max(BExpr, BExpr),
    F32Copysign(BExpr, BExpr),

    F64Add(BExpr, BExpr),
    F64Sub(BExpr, BExpr),
    F64Mul(BExpr, BExpr),
    F64Div(BExpr, BExpr),
    F64Min(BExpr, BExpr),
    F64Max(BExpr, BExpr),
    F64Copysign(BExpr, BExpr),

    I32Eqz(BExpr),
    I64Eqz(BExpr),

    I32Eq(BExpr, BExpr),
    I32Ne(BExpr, BExpr),
    I32LtS(BExpr, BExpr),
    I32LtU(BExpr, BExpr),
    I32GtS(BExpr, BExpr),
    I32GtU(BExpr, BExpr),
    I32LeS(BExpr, BExpr),
    I32LeU(BExpr, BExpr),
    I32GeS(BExpr, BExpr),
    I32GeU(BExpr, BExpr),

    I64Eq(BExpr, BExpr),
    I64Ne(BExpr, BExpr),
    I64LtS(BExpr, BExpr),
    I64LtU(BExpr, BExpr),
    I64GtS(BExpr, BExpr),
    I64GtU(BExpr, BExpr),
    I64LeS(BExpr, BExpr),
    I64LeU(BExpr, BExpr),
    I64GeS(BExpr, BExpr),
    I64GeU(BExpr, BExpr),

    F32Eq(BExpr, BExpr),
    F32Ne(BExpr, BExpr),
    F32Lt(BExpr, BExpr),
    F32Gt(BExpr, BExpr),
    F32Le(BExpr, BExpr),
    F32Ge(BExpr, BExpr),

    F64Eq(BExpr, BExpr),
    F64Ne(BExpr, BExpr),
    F64Lt(BExpr, BExpr),
    F64Gt(BExpr, BExpr),
    F64Le(BExpr, BExpr),
    F64Ge(BExpr, BExpr),

    I32WrapI64(BExpr),

    I64ExtendI32S(BExpr),
    I64ExtendI32U(BExpr),

    I32TruncF32S(BExpr),
    I32TruncF32U(BExpr),
    I32TruncF64S(BExpr),
    I32TruncF64U(BExpr),

    I64TruncF32S(BExpr),
    I64TruncF32U(BExpr),
    I64TruncF64S(BExpr),
    I64TruncF64U(BExpr),

    F32DemoteF64(BExpr),
    F64PromoteF32(BExpr),

    F32ConvertI32S(BExpr),
    F32ConvertI32U(BExpr),
    F32ConvertI64S(BExpr),
    F32ConvertI64U(BExpr),

    F64ConvertI32S(BExpr),
    F64ConvertI32U(BExpr),
    F64ConvertI64S(BExpr),
    F64ConvertI64U(BExpr),

    F32ReinterpretI32(BExpr),
    F64ReinterpretI64(BExpr),
    I32ReinterpretF32(BExpr),
    I64ReinterpretF64(BExpr),

    I32Extend8S(BExpr),
    I32Extend16S(BExpr),

    I64Extend8S(BExpr),
    I64Extend16S(BExpr),
    I64Extend32S(BExpr),

    // parametric instructions
    Drop(BExpr),
    Select(BExpr, BExpr, BExpr),

    // variable instructions
    LocalGet(LocalId),
    LocalSet(LocalId, BExpr),
    LocalTee(LocalId, BExpr),
    GlobalGet(GlobalId),
    GlobalSet(GlobalId, BExpr),

    // memory instructions
    I32Load(MemArg, BExpr),
    I64Load(MemArg, BExpr),
    F32Load(MemArg, BExpr),
    F64Load(MemArg, BExpr),

    I32Store(MemArg, BExpr, BExpr),
    I64Store(MemArg, BExpr, BExpr),
    F32Store(MemArg, BExpr, BExpr),
    F64Store(MemArg, BExpr, BExpr),

    I32Load8S(MemArg, BExpr),
    I32Load8U(MemArg, BExpr),
    I32Load16S(MemArg, BExpr),
    I32Load16U(MemArg, BExpr),

    I64Load8S(MemArg, BExpr),
    I64Load8U(MemArg, BExpr),
    I64Load16S(MemArg, BExpr),
    I64Load16U(MemArg, BExpr),
    I64Load32S(MemArg, BExpr),
    I64Load32U(MemArg, BExpr),

    I32Store8(MemArg, BExpr, BExpr),
    I32Store16(MemArg, BExpr, BExpr),

    I64Store8(MemArg, BExpr, BExpr),
    I64Store16(MemArg, BExpr, BExpr),
    I64Store32(MemArg, BExpr, BExpr),

    MemorySize,
    MemoryGrow(BExpr),

    // control instructions
    #[default]
    Nop,
    Unreachable,
    Block(Option<ValType>, Vec<Expr>),
    Loop(Option<ValType>, Vec<Expr>),
    If(Option<ValType>, BExpr, Vec<Expr>, Vec<Expr>),
    Br(u32, Option<BExpr>),
    BrIf(u32, BExpr, Option<BExpr>),
    BrTable(Vec<u32>, u32, BExpr, Option<BExpr>),
    Return(Option<BExpr>),
    Call(FuncId, Vec<Expr>),
    CallIndirect(TypeId, BExpr, Vec<Expr>),
}

impl Expr {
    pub fn to_value(&self) -> Option<Value> {
        match *self {
            Self::I32(value) => Some(Value::I32(value)),
            Self::I64(value) => Some(Value::I64(value)),
            Self::F32(value) => Some(Value::F32(value)),
            Self::F64(value) => Some(Value::F64(value)),
            _ => None,
        }
    }

    pub fn to_i32(&self) -> Option<i32> {
        match *self {
            Self::I32(value) => Some(value),
            _ => None,
        }
    }

    pub fn to_u32(&self) -> Option<u32> {
        match *self {
            Self::I32(value) => Some(value as u32),
            _ => None,
        }
    }

    pub fn to_i64(&self) -> Option<i64> {
        match *self {
            Self::I64(value) => Some(value),
            _ => None,
        }
    }

    pub fn to_u64(&self) -> Option<u64> {
        match *self {
            Self::I64(value) => Some(value as u64),
            _ => None,
        }
    }

    pub fn to_f32(&self) -> Option<F32> {
        match *self {
            Self::F32(value) => Some(value),
            _ => None,
        }
    }

    pub fn to_f64(&self) -> Option<F64> {
        match *self {
            Self::F64(value) => Some(value),
            _ => None,
        }
    }

    /// Returns the number of values the expression evaluates to.
    pub fn ret_value_count(&self) -> ReturnValueCount {
        match self {
            Self::Drop(_)
            | Self::LocalSet(_, _)
            | Self::GlobalSet(_, _)
            | Self::I32Store(_, _, _)
            | Self::I64Store(_, _, _)
            | Self::F32Store(_, _, _)
            | Self::F64Store(_, _, _)
            | Self::I32Store8(_, _, _)
            | Self::I32Store16(_, _, _)
            | Self::I64Store8(_, _, _)
            | Self::I64Store16(_, _, _)
            | Self::I64Store32(_, _, _)
            | Self::Nop => ReturnValueCount::Zero,

            Self::Block(ty, _) | Self::Loop(ty, _) | Self::If(ty, _, _, _) => {
                if ty.is_some() {
                    ReturnValueCount::One
                } else {
                    ReturnValueCount::Zero
                }
            }

            Self::Call(func, _) => ReturnValueCount::Call(*func),
            Self::CallIndirect(ty, _, _) => ReturnValueCount::CallIndirect(*ty),

            Self::BrIf(_, _, Some(_)) => ReturnValueCount::One,
            Self::BrIf(_, _, None) => ReturnValueCount::Zero,

            Self::Br(_, _)
            | Self::BrTable(_, _, _, _)
            | Self::Return(_)
            | Self::Unreachable => ReturnValueCount::Unreachable,

            _ => ReturnValueCount::One,
        }
    }

    pub fn ty(&self) -> ExprTy {
        match self {
            Self::I32(_) => ValType::I32.into(),
            Self::I64(_) => ValType::I64.into(),
            Self::F32(_) => ValType::F32.into(),
            Self::F64(_) => ValType::F64.into(),

            Self::I32Clz(_) | Self::I32Ctz(_) | Self::I32Popcnt(_) => ValType::I32.into(),

            Self::I64Clz(_) | Self::I64Ctz(_) | Self::I64Popcnt(_) => ValType::I64.into(),

            Self::F32Abs(_)
            | Self::F32Neg(_)
            | Self::F32Sqrt(_)
            | Self::F32Ceil(_)
            | Self::F32Floor(_)
            | Self::F32Trunc(_)
            | Self::F32Nearest(_) => ValType::F32.into(),

            Self::F64Abs(_)
            | Self::F64Neg(_)
            | Self::F64Sqrt(_)
            | Self::F64Ceil(_)
            | Self::F64Floor(_)
            | Self::F64Trunc(_)
            | Self::F64Nearest(_) => ValType::F64.into(),

            Self::I32Add(_, _)
            | Self::I32Sub(_, _)
            | Self::I32Mul(_, _)
            | Self::I32DivS(_, _)
            | Self::I32DivU(_, _)
            | Self::I32RemS(_, _)
            | Self::I32RemU(_, _)
            | Self::I32And(_, _)
            | Self::I32Or(_, _)
            | Self::I32Xor(_, _)
            | Self::I32Shl(_, _)
            | Self::I32ShrS(_, _)
            | Self::I32ShrU(_, _)
            | Self::I32Rotl(_, _)
            | Self::I32Rotr(_, _) => ValType::I32.into(),

            Self::I64Add(_, _)
            | Self::I64Sub(_, _)
            | Self::I64Mul(_, _)
            | Self::I64DivS(_, _)
            | Self::I64DivU(_, _)
            | Self::I64RemS(_, _)
            | Self::I64RemU(_, _)
            | Self::I64And(_, _)
            | Self::I64Or(_, _)
            | Self::I64Xor(_, _)
            | Self::I64Shl(_, _)
            | Self::I64ShrS(_, _)
            | Self::I64ShrU(_, _)
            | Self::I64Rotl(_, _)
            | Self::I64Rotr(_, _) => ValType::I64.into(),

            Self::F32Add(_, _)
            | Self::F32Sub(_, _)
            | Self::F32Mul(_, _)
            | Self::F32Div(_, _)
            | Self::F32Min(_, _)
            | Self::F32Max(_, _)
            | Self::F32Copysign(_, _) => ValType::F32.into(),

            Self::F64Add(_, _)
            | Self::F64Sub(_, _)
            | Self::F64Mul(_, _)
            | Self::F64Div(_, _)
            | Self::F64Min(_, _)
            | Self::F64Max(_, _)
            | Self::F64Copysign(_, _) => ValType::F64.into(),

            Self::I32Eqz(_) => ValType::I32.into(),
            Self::I64Eqz(_) => ValType::I64.into(),

            Self::I32Eq(_, _)
            | Self::I32Ne(_, _)
            | Self::I32LtS(_, _)
            | Self::I32LtU(_, _)
            | Self::I32GtS(_, _)
            | Self::I32GtU(_, _)
            | Self::I32LeS(_, _)
            | Self::I32LeU(_, _)
            | Self::I32GeS(_, _)
            | Self::I32GeU(_, _) => ValType::I32.into(),

            Self::I64Eq(_, _)
            | Self::I64Ne(_, _)
            | Self::I64LtS(_, _)
            | Self::I64LtU(_, _)
            | Self::I64GtS(_, _)
            | Self::I64GtU(_, _)
            | Self::I64LeS(_, _)
            | Self::I64LeU(_, _)
            | Self::I64GeS(_, _)
            | Self::I64GeU(_, _) => ValType::I64.into(),

            Self::F32Eq(_, _)
            | Self::F32Ne(_, _)
            | Self::F32Lt(_, _)
            | Self::F32Gt(_, _)
            | Self::F32Le(_, _)
            | Self::F32Ge(_, _) => ValType::F32.into(),

            Self::F64Eq(_, _)
            | Self::F64Ne(_, _)
            | Self::F64Lt(_, _)
            | Self::F64Gt(_, _)
            | Self::F64Le(_, _)
            | Self::F64Ge(_, _) => ValType::F64.into(),

            Self::I32WrapI64(_) => ValType::I32.into(),

            Self::I64ExtendI32S(_) | Self::I64ExtendI32U(_) => ValType::I64.into(),

            Self::I32TruncF32S(_)
            | Self::I32TruncF32U(_)
            | Self::I32TruncF64S(_)
            | Self::I32TruncF64U(_) => ValType::I32.into(),

            Self::I64TruncF32S(_)
            | Self::I64TruncF32U(_)
            | Self::I64TruncF64S(_)
            | Self::I64TruncF64U(_) => ValType::I64.into(),

            Self::F32DemoteF64(_) => ValType::F32.into(),
            Self::F64PromoteF32(_) => ValType::F64.into(),

            Self::F32ConvertI32S(_)
            | Self::F32ConvertI32U(_)
            | Self::F32ConvertI64S(_)
            | Self::F32ConvertI64U(_) => ValType::F32.into(),

            Self::F64ConvertI32S(_)
            | Self::F64ConvertI32U(_)
            | Self::F64ConvertI64S(_)
            | Self::F64ConvertI64U(_) => ValType::F64.into(),

            Self::F32ReinterpretI32(_) => ValType::F32.into(),
            Self::F64ReinterpretI64(_) => ValType::F64.into(),
            Self::I32ReinterpretF32(_) => ValType::I32.into(),
            Self::I64ReinterpretF64(_) => ValType::I64.into(),

            Self::I32Extend8S(_) | Self::I32Extend16S(_) => ValType::I32.into(),
            Self::I64Extend8S(_) | Self::I64Extend16S(_) | Self::I64Extend32S(_) => ValType::I64.into(),

            Self::Drop(_) => ExprTy::Empty,
            Self::Select(lhs, _, _) => lhs.ty(),

            Self::LocalGet(local) | Self::LocalSet(local, _) | Self::LocalTee(local, _) => {
                ExprTy::Local(*local)
            }

            Self::GlobalGet(global) | Self::GlobalSet(global, _) => ExprTy::Global(*global),

            Self::I32Load(_, _) => ValType::I32.into(),
            Self::I64Load(_, _) => ValType::I64.into(),
            Self::F32Load(_, _) => ValType::F32.into(),
            Self::F64Load(_, _) => ValType::F64.into(),

            Self::I32Store(_, _, _)
            | Self::I64Store(_, _, _)
            | Self::F32Store(_, _, _)
            | Self::F64Store(_, _, _) => ExprTy::Empty,

            Self::I32Load8S(_, _)
            | Self::I32Load8U(_, _)
            | Self::I32Load16S(_, _)
            | Self::I32Load16U(_, _) => ValType::I32.into(),

            Self::I64Load8S(_, _)
            | Self::I64Load8U(_, _)
            | Self::I64Load16S(_, _)
            | Self::I64Load16U(_, _)
            | Self::I64Load32S(_, _)
            | Self::I64Load32U(_, _) => ValType::I64.into(),

            Self::I32Store8(_, _, _) | Self::I32Store16(_, _, _) => ValType::I32.into(),

            Self::I64Store8(_, _, _) | Self::I64Store16(_, _, _) | Self::I64Store32(_, _, _) => {
                ValType::I64.into()
            }

            Self::MemorySize | Self::MemoryGrow(_) => ValType::I32.into(),

            Self::Nop => ExprTy::Empty,
            Self::Unreachable => ExprTy::Unreachable,

            Self::Block(ty, _) | Self::Loop(ty, _) | Self::If(ty, _, _, _) => match ty {
                Some(ty) => ExprTy::Concrete(ty.clone()),
                None => ExprTy::Empty,
            },

            Self::BrIf(_, _, Some(expr)) => expr.ty(),
            Self::BrIf(_, _, None) => ExprTy::Empty,

            Self::Br(_, _) | Self::BrTable(_, _, _, _) | Self::Return(_) => {
                ExprTy::Unreachable
            }

            Self::Call(func, _) => ExprTy::Call(*func),
            Self::CallIndirect(ty, _, _) => ExprTy::CallIndirect(*ty),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReturnValueCount {
    Zero,
    One,
    Call(FuncId),
    CallIndirect(TypeId),
    Unreachable,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ExprTy {
    Concrete(ValType),
    Local(LocalId),
    Global(GlobalId),
    Call(FuncId),
    CallIndirect(TypeId),

    Unreachable,

    /// The expression evaluates to no values.
    Empty,
}

impl From<ValType> for ExprTy {
    fn from(val_ty: ValType) -> Self {
        Self::Concrete(val_ty)
    }
}
