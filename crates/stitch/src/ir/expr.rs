use super::ty::ValType;
use super::{FuncId, GlobalId, LocalId, TypeId};

type BExpr = Box<Expr>;

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
    F32(f32),
    F64(f64),

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
    I32Rolr(BExpr, BExpr),

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
    I64Rolr(BExpr, BExpr),

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
    If(Option<ValType>, Vec<Expr>, Vec<Expr>),
    Br(u32, Option<BExpr>),
    BrIf(u32, BExpr, Option<BExpr>),
    BrTable(Vec<u32>, u32, BExpr, Option<BExpr>),
    Return(BExpr),
    Call(FuncId, Vec<Expr>),
    CallIndirect(TypeId, BExpr, Vec<Expr>),
}

impl Expr {
    pub fn as_i32(&self) -> Option<i32> {
        match *self {
            Self::I32(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_u32(&self) -> Option<u32> {
        match *self {
            Self::I32(value) => Some(value as u32),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match *self {
            Self::I64(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Self::I64(value) => Some(value as u64),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            Self::F32(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Self::F64(value) => Some(value),
            _ => None,
        }
    }
}
