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

impl Value {
    pub fn to_expr(&self) -> Expr {
        match *self {
            Self::I32(value) => Expr::I32(value),
            Self::I64(value) => Expr::I64(value),
            Self::F32(value) => Expr::F32(value),
            Self::F64(value) => Expr::F64(value),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemArg {
    pub offset: u32,
    pub align: u32,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum NulOp {
    #[default]
    Nop,
    Unreachable,
    LocalGet(LocalId),
    GlobalGet(GlobalId),
    MemorySize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnOp {
    I32Clz,
    I32Ctz,
    I32Popcnt,

    I64Clz,
    I64Ctz,
    I64Popcnt,

    F32Abs,
    F32Neg,
    F32Sqrt,
    F32Ceil,
    F32Floor,
    F32Trunc,
    F32Nearest,

    F64Abs,
    F64Neg,
    F64Sqrt,
    F64Ceil,
    F64Floor,
    F64Trunc,
    F64Nearest,

    I32Eqz,
    I64Eqz,

    I32WrapI64,

    I64ExtendI32S,
    I64ExtendI32U,

    I32TruncF32S,
    I32TruncF32U,
    I32TruncF64S,
    I32TruncF64U,

    I64TruncF32S,
    I64TruncF32U,
    I64TruncF64S,
    I64TruncF64U,

    F32DemoteF64,
    F64PromoteF32,

    F32ConvertI32S,
    F32ConvertI32U,
    F32ConvertI64S,
    F32ConvertI64U,

    F64ConvertI32S,
    F64ConvertI32U,
    F64ConvertI64S,
    F64ConvertI64U,

    F32ReinterpretI32,
    F64ReinterpretI64,
    I32ReinterpretF32,
    I64ReinterpretF64,

    I32Extend8S,
    I32Extend16S,

    I64Extend8S,
    I64Extend16S,
    I64Extend32S,

    LocalSet(LocalId),
    LocalTee(LocalId),
    GlobalSet(GlobalId),

    I32Load(MemArg),
    I64Load(MemArg),
    F32Load(MemArg),
    F64Load(MemArg),

    I32Load8S(MemArg),
    I32Load8U(MemArg),
    I32Load16S(MemArg),
    I32Load16U(MemArg),

    I64Load8S(MemArg),
    I64Load8U(MemArg),
    I64Load16S(MemArg),
    I64Load16U(MemArg),
    I64Load32S(MemArg),
    I64Load32U(MemArg),

    MemoryGrow,

    Drop,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    I32Add,
    I32Sub,
    I32Mul,
    I32DivS,
    I32DivU,
    I32RemS,
    I32RemU,
    I32And,
    I32Or,
    I32Xor,
    I32Shl,
    I32ShrS,
    I32ShrU,
    I32Rotl,
    I32Rotr,

    I64Add,
    I64Sub,
    I64Mul,
    I64DivS,
    I64DivU,
    I64RemS,
    I64RemU,
    I64And,
    I64Or,
    I64Xor,
    I64Shl,
    I64ShrS,
    I64ShrU,
    I64Rotl,
    I64Rotr,

    F32Add,
    F32Sub,
    F32Mul,
    F32Div,
    F32Min,
    F32Max,
    F32Copysign,

    F64Add,
    F64Sub,
    F64Mul,
    F64Div,
    F64Min,
    F64Max,
    F64Copysign,

    I32Eq,
    I32Ne,
    I32LtS,
    I32LtU,
    I32GtS,
    I32GtU,
    I32LeS,
    I32LeU,
    I32GeS,
    I32GeU,

    I64Eq,
    I64Ne,
    I64LtS,
    I64LtU,
    I64GtS,
    I64GtU,
    I64LeS,
    I64LeU,
    I64GeS,
    I64GeU,

    F32Eq,
    F32Ne,
    F32Lt,
    F32Gt,
    F32Le,
    F32Ge,

    F64Eq,
    F64Ne,
    F64Lt,
    F64Gt,
    F64Le,
    F64Ge,

    I32Store(MemArg),
    I64Store(MemArg),
    F32Store(MemArg),
    F64Store(MemArg),

    I32Store8(MemArg),
    I32Store16(MemArg),

    I64Store8(MemArg),
    I64Store16(MemArg),
    I64Store32(MemArg),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TernOp {
    Select,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Nullary(NulOp),
    Unary(UnOp),
    Binary(BinOp),
    Ternary(TernOp),
}

impl Default for Op {
    fn default() -> Self {
        NulOp::default().into()
    }
}

impl From<NulOp> for Op {
    fn from(op: NulOp) -> Self {
        Self::Nullary(op)
    }
}

impl From<UnOp> for Op {
    fn from(op: UnOp) -> Self {
        Self::Unary(op)
    }
}

impl From<BinOp> for Op {
    fn from(op: BinOp) -> Self {
        Self::Binary(op)
    }
}

impl From<TernOp> for Op {
    fn from(op: TernOp) -> Self {
        Self::Ternary(op)
    }
}

#[derive(Debug, Clone)]
pub enum Expr {
    I32(i32),
    I64(i64),
    F32(F32),
    F64(F64),

    Nullary(NulOp),
    Unary(UnOp, Box<Expr>),
    Binary(BinOp, Box<Expr>, Box<Expr>),
    Ternary(TernOp, Box<Expr>, Box<Expr>, Box<Expr>),

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

impl From<NulOp> for Expr {
    fn from(op: NulOp) -> Self {
        Self::Nullary(op)
    }
}

pub trait Visitor {
    fn visit(&mut self, expr: Expr) -> Expr;
}

impl<F> Visitor for F
where
    F: FnMut(Expr) -> Expr,
{
    fn visit(&mut self, expr: Expr) -> Expr {
        self(expr)
    }
}

impl Expr {
    pub fn map<V: Visitor>(&self, v: &mut V) -> Expr {
        let expr = match self {
            Self::I32(_) | Self::I64(_) | Self::F32(_) | Self::F64(_) | Self::Nullary(_) => {
                return v.visit(self.clone())
            }

            Self::Unary(op, inner) => Self::Unary(*op, Box::new(inner.map(v))),
            Self::Binary(op, lhs, rhs) => {
                Self::Binary(*op, Box::new(lhs.map(v)), Box::new(rhs.map(v)))
            }

            Self::Ternary(op, first, second, third) => Self::Ternary(
                *op,
                Box::new(first.map(v)),
                Box::new(second.map(v)),
                Box::new(third.map(v)),
            ),

            Self::Block(block_ty, exprs) => Self::Block(
                block_ty.clone(),
                exprs.iter().map(|expr| expr.map(v)).collect(),
            ),

            Self::Loop(block_ty, exprs) => Self::Loop(
                block_ty.clone(),
                exprs.iter().map(|expr| expr.map(v)).collect(),
            ),

            Self::If(block_ty, condition, then_block, else_block) => Self::If(
                block_ty.clone(),
                Box::new(condition.map(v)),
                then_block.iter().map(|expr| expr.map(v)).collect(),
                else_block.iter().map(|expr| expr.map(v)).collect(),
            ),

            Self::Br(relative_depth, inner) => Self::Br(
                *relative_depth,
                inner.as_ref().map(|expr| Box::new(expr.map(v))),
            ),

            Self::BrIf(relative_depth, condition, inner) => Self::BrIf(
                *relative_depth,
                Box::new(condition.map(v)),
                inner.as_ref().map(|expr| Box::new(expr.map(v))),
            ),

            Self::BrTable(labels, default_label, index_expr, inner) => Self::BrTable(
                labels.clone(),
                *default_label,
                Box::new(index_expr.map(v)),
                inner.as_ref().map(|expr| Box::new(expr.map(v))),
            ),

            Self::Return(inner) => Self::Return(inner.as_ref().map(|expr| Box::new(expr.map(v)))),

            Self::Call(func_id, args) => {
                Self::Call(*func_id, args.iter().map(|expr| expr.map(v)).collect())
            }

            Self::CallIndirect(ty_id, index_expr, args) => Self::CallIndirect(
                *ty_id,
                Box::new(index_expr.map(v)),
                args.iter().map(|expr| expr.map(v)).collect(),
            ),
        };

        v.visit(expr)
    }

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
            Self::Nullary(NulOp::Nop | NulOp::Unreachable)
            | Self::Unary(UnOp::Drop | UnOp::LocalSet(_) | UnOp::GlobalSet(_), _)
            | Self::Binary(
                BinOp::I32Store(_)
                | BinOp::I64Store(_)
                | BinOp::F32Store(_)
                | BinOp::F64Store(_)
                | BinOp::I32Store8(_)
                | BinOp::I32Store16(_)
                | BinOp::I64Store8(_)
                | BinOp::I64Store16(_)
                | BinOp::I64Store32(_),
                _,
                _,
            ) => ReturnValueCount::Zero,

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

            Self::Br(_, _) | Self::BrTable(_, _, _, _) | Self::Return(_) => {
                ReturnValueCount::Unreachable
            }

            _ => ReturnValueCount::One,
        }
    }

    pub fn ty(&self) -> ExprTy {
        match self {
            Self::I32(_) => ValType::I32.into(),
            Self::I64(_) => ValType::I64.into(),
            Self::F32(_) => ValType::F32.into(),
            Self::F64(_) => ValType::F64.into(),

            Self::Unary(UnOp::I32Clz | UnOp::I32Ctz | UnOp::I32Popcnt, _) => ValType::I32.into(),
            Self::Unary(UnOp::I64Clz | UnOp::I64Ctz | UnOp::I64Popcnt, _) => ValType::I64.into(),

            Self::Unary(
                UnOp::F32Abs
                | UnOp::F32Neg
                | UnOp::F32Sqrt
                | UnOp::F32Ceil
                | UnOp::F32Floor
                | UnOp::F32Trunc
                | UnOp::F32Nearest,
                _,
            ) => ValType::F32.into(),

            Self::Unary(
                UnOp::F64Abs
                | UnOp::F64Neg
                | UnOp::F64Sqrt
                | UnOp::F64Ceil
                | UnOp::F64Floor
                | UnOp::F64Trunc
                | UnOp::F64Nearest,
                _,
            ) => ValType::F64.into(),

            Self::Binary(
                BinOp::I32Add
                | BinOp::I32Sub
                | BinOp::I32Mul
                | BinOp::I32DivS
                | BinOp::I32DivU
                | BinOp::I32RemS
                | BinOp::I32RemU
                | BinOp::I32And
                | BinOp::I32Or
                | BinOp::I32Xor
                | BinOp::I32Shl
                | BinOp::I32ShrS
                | BinOp::I32ShrU
                | BinOp::I32Rotl
                | BinOp::I32Rotr,
                _,
                _,
            ) => ValType::I32.into(),

            Self::Binary(
                BinOp::I64Add
                | BinOp::I64Sub
                | BinOp::I64Mul
                | BinOp::I64DivS
                | BinOp::I64DivU
                | BinOp::I64RemS
                | BinOp::I64RemU
                | BinOp::I64And
                | BinOp::I64Or
                | BinOp::I64Xor
                | BinOp::I64Shl
                | BinOp::I64ShrS
                | BinOp::I64ShrU
                | BinOp::I64Rotl
                | BinOp::I64Rotr,
                _,
                _,
            ) => ValType::I64.into(),

            Self::Binary(
                BinOp::F32Add
                | BinOp::F32Sub
                | BinOp::F32Mul
                | BinOp::F32Div
                | BinOp::F32Min
                | BinOp::F32Max
                | BinOp::F32Copysign,
                _,
                _,
            ) => ValType::F32.into(),

            Self::Binary(
                BinOp::F64Add
                | BinOp::F64Sub
                | BinOp::F64Mul
                | BinOp::F64Div
                | BinOp::F64Min
                | BinOp::F64Max
                | BinOp::F64Copysign,
                _,
                _,
            ) => ValType::F64.into(),

            Self::Unary(UnOp::I32Eqz, _) => ValType::I32.into(),
            Self::Unary(UnOp::I64Eqz, _) => ValType::I64.into(),

            Self::Binary(
                BinOp::I32Eq
                | BinOp::I32Ne
                | BinOp::I32LtS
                | BinOp::I32LtU
                | BinOp::I32GtS
                | BinOp::I32GtU
                | BinOp::I32LeS
                | BinOp::I32LeU
                | BinOp::I32GeS
                | BinOp::I32GeU,
                _,
                _,
            ) => ValType::I32.into(),

            Self::Binary(
                BinOp::I64Eq
                | BinOp::I64Ne
                | BinOp::I64LtS
                | BinOp::I64LtU
                | BinOp::I64GtS
                | BinOp::I64GtU
                | BinOp::I64LeS
                | BinOp::I64LeU
                | BinOp::I64GeS
                | BinOp::I64GeU,
                _,
                _,
            ) => ValType::I64.into(),

            Self::Binary(
                BinOp::F32Eq
                | BinOp::F32Ne
                | BinOp::F32Lt
                | BinOp::F32Gt
                | BinOp::F32Le
                | BinOp::F32Ge,
                _,
                _,
            ) => ValType::F32.into(),

            Self::Binary(
                BinOp::F64Eq
                | BinOp::F64Ne
                | BinOp::F64Lt
                | BinOp::F64Gt
                | BinOp::F64Le
                | BinOp::F64Ge,
                _,
                _,
            ) => ValType::F64.into(),

            Self::Unary(UnOp::I32WrapI64, _) => ValType::I32.into(),
            Self::Unary(UnOp::I64ExtendI32S | UnOp::I64ExtendI32U, _) => ValType::I64.into(),

            Self::Unary(
                UnOp::I32TruncF32S | UnOp::I32TruncF32U | UnOp::I32TruncF64S | UnOp::I32TruncF64U,
                _,
            ) => ValType::I32.into(),

            Self::Unary(
                UnOp::I64TruncF32S | UnOp::I64TruncF32U | UnOp::I64TruncF64S | UnOp::I64TruncF64U,
                _,
            ) => ValType::I64.into(),

            Self::Unary(UnOp::F32DemoteF64, _) => ValType::F32.into(),
            Self::Unary(UnOp::F64PromoteF32, _) => ValType::F64.into(),

            Self::Unary(
                UnOp::F32ConvertI32S
                | UnOp::F32ConvertI32U
                | UnOp::F32ConvertI64S
                | UnOp::F32ConvertI64U,
                _,
            ) => ValType::F32.into(),

            Self::Unary(
                UnOp::F64ConvertI32S
                | UnOp::F64ConvertI32U
                | UnOp::F64ConvertI64S
                | UnOp::F64ConvertI64U,
                _,
            ) => ValType::F64.into(),

            Self::Unary(UnOp::F32ReinterpretI32, _) => ValType::F32.into(),
            Self::Unary(UnOp::F64ReinterpretI64, _) => ValType::F64.into(),
            Self::Unary(UnOp::I32ReinterpretF32, _) => ValType::I32.into(),
            Self::Unary(UnOp::I64ReinterpretF64, _) => ValType::I64.into(),

            Self::Unary(UnOp::I32Extend8S | UnOp::I32Extend16S, _) => ValType::I32.into(),
            Self::Unary(UnOp::I64Extend8S | UnOp::I64Extend16S | UnOp::I64Extend32S, _) => {
                ValType::I64.into()
            }

            Self::Unary(UnOp::Drop, _) => ExprTy::Empty,
            Self::Ternary(TernOp::Select, lhs, _, _) => lhs.ty(),

            Self::Nullary(NulOp::LocalGet(local))
            | Self::Unary(UnOp::LocalSet(local) | UnOp::LocalTee(local), _) => {
                ExprTy::Local(*local)
            }

            Self::Nullary(NulOp::GlobalGet(global)) | Self::Unary(UnOp::GlobalSet(global), _) => {
                ExprTy::Global(*global)
            }

            Self::Unary(UnOp::I32Load(_), _) => ValType::I32.into(),
            Self::Unary(UnOp::I64Load(_), _) => ValType::I64.into(),
            Self::Unary(UnOp::F32Load(_), _) => ValType::F32.into(),
            Self::Unary(UnOp::F64Load(_), _) => ValType::F64.into(),

            Self::Binary(
                BinOp::I32Store(_)
                | BinOp::I64Store(_)
                | BinOp::F32Store(_)
                | BinOp::F64Store(_),
                _,
                _,
            ) => ExprTy::Empty,

            Self::Unary(
                UnOp::I32Load8S(_) | UnOp::I32Load8U(_) | UnOp::I32Load16S(_) | UnOp::I32Load16U(_),
                _,
            ) => ValType::I32.into(),

            Self::Unary(
                UnOp::I64Load8S(_)
                | UnOp::I64Load8U(_)
                | UnOp::I64Load16S(_)
                | UnOp::I64Load16U(_)
                | UnOp::I64Load32S(_)
                | UnOp::I64Load32U(_),
                _,
            ) => ValType::I64.into(),

            Self::Binary(BinOp::I32Store8(_) | BinOp::I32Store16(_), _, _) => ValType::I32.into(),

            Self::Binary(
                BinOp::I64Store8(_) | BinOp::I64Store16(_) | BinOp::I64Store32(_),
                _,
                _,
            ) => ValType::I64.into(),

            Self::Nullary(NulOp::MemorySize) | Self::Unary(UnOp::MemoryGrow, _) => {
                ValType::I32.into()
            }

            Self::Nullary(NulOp::Nop) => ExprTy::Empty,
            Self::Nullary(NulOp::Unreachable) => ExprTy::Unreachable,

            Self::Block(ty, _) | Self::Loop(ty, _) | Self::If(ty, _, _, _) => match ty {
                Some(ty) => ExprTy::Concrete(ty.clone()),
                None => ExprTy::Empty,
            },

            Self::BrIf(_, _, Some(expr)) => expr.ty(),
            Self::BrIf(_, _, None) => ExprTy::Empty,

            Self::Br(_, _) | Self::BrTable(_, _, _, _) | Self::Return(_) => ExprTy::Unreachable,

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
