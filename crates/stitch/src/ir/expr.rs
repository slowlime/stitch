use std::fmt::{self, Debug, Display};

use crate::util::try_match;

use super::ty::ValType;
use super::{FuncId, GlobalId, LocalId, MemoryId, TableId, TypeId};

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
    pub fn to_i32(&self) -> Option<i32> {
        try_match!(*self, Self::I32(value) => value)
    }

    pub fn to_u32(&self) -> Option<u32> {
        try_match!(*self, Self::I32(value) => value as u32)
    }

    pub fn to_i64(&self) -> Option<i64> {
        try_match!(*self, Self::I64(value) => value)
    }

    pub fn to_u64(&self) -> Option<u64> {
        try_match!(*self, Self::I64(value) => value as u64)
    }

    pub fn to_f32(&self) -> Option<F32> {
        try_match!(*self, Self::F32(value) => value)
    }

    pub fn to_f64(&self) -> Option<F64> {
        try_match!(*self, Self::F64(value) => value)
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PtrAttr {
    #[default]
    None,

    /// Allows inlining loads from addresses derived from a value.
    Const,

    /// Allows inlining stores to addresses derived from a value in addition to loads.
    Owned,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueAttrs {
    pub ptr: PtrAttr,

    /// Whether the specializer can propagate these attributes to the value loaded from this address.
    pub propagate: bool,
}

impl ValueAttrs {
    pub fn meet(&self, other: &Self) -> Self {
        Self {
            ptr: self.ptr.min(other.ptr),
            propagate: self.propagate && other.propagate,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemArg {
    pub mem_id: MemoryId,
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
    MemorySize(MemoryId),
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

    MemoryGrow(MemoryId),

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
pub enum Intrinsic {
    Specialize {
        table_id: TableId,
        elem_idx: u32,
        mem_id: MemoryId,
        name_addr: u32,
        name_len: u32,
        args: Vec<Option<Value>>,
    },

    Unknown(ValType),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Id {
    Func(FuncId),
}

impl From<FuncId> for Id {
    fn from(func_id: FuncId) -> Self {
        Self::Func(func_id)
    }
}

#[derive(Debug, Clone)]
pub enum Expr {
    Value(Value, ValueAttrs),

    Intrinsic(Intrinsic),
    Index(Id),

    Nullary(NulOp),
    Unary(UnOp, Box<Expr>),
    Binary(BinOp, Box<Expr>, Box<Expr>),
    Ternary(TernOp, Box<Expr>, Box<Expr>, Box<Expr>),

    Block(Option<ValType>, Vec<Expr>),
    Loop(Option<ValType>, Vec<Expr>),
    If(Option<ValType>, Box<Expr>, Vec<Expr>, Vec<Expr>),
    Br(u32, Option<Box<Expr>>),
    BrIf(u32, Option<Box<Expr>>, Box<Expr>),
    BrTable(Vec<u32>, u32, Option<Box<Expr>>, Box<Expr>),
    Return(Option<Box<Expr>>),
    Call(FuncId, Vec<Expr>),
    CallIndirect(TypeId, TableId, Vec<Expr>, Box<Expr>),
}

impl From<Value> for Expr {
    fn from(value: Value) -> Self {
        Self::Value(value, ValueAttrs::default())
    }
}

impl From<NulOp> for Expr {
    fn from(op: NulOp) -> Self {
        Self::Nullary(op)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Load {
    I32 { src_size: u8, sign_extend: bool },
    I64 { src_size: u8, sign_extend: bool },
    F32,
    F64,
}

impl Load {
    pub fn src_size(&self) -> u8 {
        match *self {
            Self::I32 { src_size, .. } | Self::I64 { src_size, .. } => src_size,
            Self::F32 => 4,
            Self::F64 => 8,
        }
    }

    pub fn sign_extend(&self) -> bool {
        match *self {
            Self::I32 { sign_extend, .. } | Self::I64 { sign_extend, .. } => sign_extend,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Store {
    I32 { dst_size: u8 },
    I64 { dst_size: u8 },
    F32,
    F64,
}

impl Store {
    pub fn dst_size(&self) -> u8 {
        match *self {
            Self::I32 { dst_size } | Self::I64 { dst_size } => dst_size,
            Self::F32 => 4,
            Self::F64 => 8,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct VisitContext {
    relative_depth: usize,
}

impl VisitContext {
    pub fn relative_depth(&self) -> usize {
        self.relative_depth
    }
}

pub trait Visitor {
    fn pre(&mut self, ctx: &mut VisitContext, expr: &Expr) {}
    fn post(&mut self, ctx: &mut VisitContext, expr: Expr) -> Expr;
}

impl<F> Visitor for F
where
    F: FnMut(Expr, &mut VisitContext) -> Expr,
{
    fn post(&mut self, ctx: &mut VisitContext, expr: Expr) -> Expr {
        self(expr, ctx)
    }
}

impl Expr {
    pub fn map(&self, v: &mut impl Visitor) -> Self {
        fn visit_block(v: &mut impl Visitor, ctx: &mut VisitContext, exprs: &[Expr]) -> Vec<Expr> {
            ctx.relative_depth += 1;
            let exprs = exprs.iter().map(|expr| visit(v, ctx, expr)).collect();
            ctx.relative_depth -= 1;

            exprs
        }

        fn visit(v: &mut impl Visitor, ctx: &mut VisitContext, expr: &Expr) -> Expr {
            v.pre(ctx, expr);

            let result = match expr {
                Expr::Value(_, _) | Expr::Intrinsic(_) | Expr::Index(_) | Expr::Nullary(_) => {
                    expr.clone()
                }

                Expr::Unary(op, inner) => Expr::Unary(*op, Box::new(visit(v, ctx, inner))),
                Expr::Binary(op, lhs, rhs) => Expr::Binary(
                    *op,
                    Box::new(visit(v, ctx, lhs)),
                    Box::new(visit(v, ctx, rhs)),
                ),
                Expr::Ternary(op, first, second, third) => Expr::Ternary(
                    *op,
                    Box::new(visit(v, ctx, first)),
                    Box::new(visit(v, ctx, second)),
                    Box::new(visit(v, ctx, third)),
                ),

                Expr::Block(val_ty, block) => {
                    Expr::Block(val_ty.clone(), visit_block(v, ctx, block))
                }

                Expr::Loop(val_ty, block) => Expr::Loop(val_ty.clone(), visit_block(v, ctx, block)),

                Expr::If(val_ty, condition, then_block, else_block) => Expr::If(
                    val_ty.clone(),
                    Box::new(visit(v, ctx, condition)),
                    visit_block(v, ctx, then_block),
                    visit_block(v, ctx, else_block),
                ),

                Expr::Br(relative_depth, inner) => Expr::Br(
                    *relative_depth,
                    inner.as_ref().map(|inner| Box::new(visit(v, ctx, inner))),
                ),

                Expr::BrIf(relative_depth, inner, condition) => Expr::BrIf(
                    *relative_depth,
                    inner.as_ref().map(|inner| Box::new(visit(v, ctx, inner))),
                    Box::new(visit(v, ctx, condition)),
                ),

                Expr::BrTable(labels, default_label, inner, index) => Expr::BrTable(
                    labels.clone(),
                    *default_label,
                    inner.as_ref().map(|inner| Box::new(visit(v, ctx, inner))),
                    Box::new(visit(v, ctx, index)),
                ),

                Expr::Return(inner) => {
                    Expr::Return(inner.as_ref().map(|inner| Box::new(visit(v, ctx, inner))))
                }

                Expr::Call(func_id, args) => Expr::Call(
                    *func_id,
                    args.iter().map(|arg| visit(v, ctx, arg)).collect(),
                ),

                Expr::CallIndirect(ty_id, table_id, args, index) => Expr::CallIndirect(
                    *ty_id,
                    *table_id,
                    args.iter().map(|arg| visit(v, ctx, arg)).collect(),
                    Box::new(index.map(v)),
                ),
            };

            v.post(ctx, result)
        }

        visit(v, &mut Default::default(), self)
    }

    pub fn all(&self, predicate: &mut impl FnMut(&Expr) -> bool) -> bool {
        (match self {
            Expr::Value(_, _) | Expr::Intrinsic(_) | Expr::Index(_) | Expr::Nullary(_) => true,
            Expr::Unary(_, inner) => inner.all(predicate),
            Expr::Binary(_, lhs, rhs) => lhs.all(predicate) && rhs.all(predicate),
            Expr::Ternary(_, first, second, third) => {
                first.all(predicate) && second.all(predicate) && third.all(predicate)
            }
            Expr::Block(_, block) | Expr::Loop(_, block) => block.iter().all(&mut *predicate),
            Expr::If(_, condition, then_block, else_block) => {
                condition.all(predicate)
                    && then_block.iter().all(&mut *predicate)
                    && else_block.iter().all(&mut *predicate)
            }
            Expr::Br(_, inner) | Expr::Return(inner) => {
                inner.as_ref().map(|inner| predicate(inner)).unwrap_or(true)
            }
            Expr::BrIf(_, inner, condition) => {
                inner.as_ref().map(|inner| predicate(inner)).unwrap_or(true)
                    && condition.all(predicate)
            }
            Expr::BrTable(_, _, inner, index) => {
                inner.as_ref().map(|inner| predicate(inner)).unwrap_or(true) && index.all(predicate)
            }
            Expr::Call(_, args) => args.iter().all(&mut *predicate),
            Expr::CallIndirect(_, _, args, index) => {
                args.iter().all(&mut *predicate) && index.all(&mut *predicate)
            }
        }) && predicate(self)
    }

    pub fn to_value(&self) -> Option<Value> {
        try_match!(*self, Self::Value(value, _) => value)
    }

    pub fn to_i32(&self) -> Option<i32> {
        self.to_value()?.to_i32()
    }

    pub fn to_u32(&self) -> Option<u32> {
        self.to_value()?.to_u32()
    }

    pub fn to_i64(&self) -> Option<i64> {
        self.to_value()?.to_i64()
    }

    pub fn to_u64(&self) -> Option<u64> {
        self.to_value()?.to_u64()
    }

    pub fn to_f32(&self) -> Option<F32> {
        self.to_value()?.to_f32()
    }

    pub fn to_f64(&self) -> Option<F64> {
        self.to_value()?.to_f64()
    }

    pub fn to_load(&self) -> Option<(MemArg, &Expr, Load)> {
        Some(match self {
            Self::Unary(UnOp::I32Load(mem_arg), addr) => (
                *mem_arg,
                addr,
                Load::I32 {
                    src_size: 4,
                    sign_extend: false,
                },
            ),

            Self::Unary(UnOp::I64Load(mem_arg), addr) => (
                *mem_arg,
                addr,
                Load::I64 {
                    src_size: 8,
                    sign_extend: false,
                },
            ),

            Self::Unary(UnOp::F32Load(mem_arg), addr) => (*mem_arg, addr, Load::F32),
            Self::Unary(UnOp::F64Load(mem_arg), addr) => (*mem_arg, addr, Load::F64),

            Self::Unary(UnOp::I32Load8S(mem_arg), addr) => (
                *mem_arg,
                addr,
                Load::I32 {
                    src_size: 1,
                    sign_extend: true,
                },
            ),

            Self::Unary(UnOp::I32Load8U(mem_arg), addr) => (
                *mem_arg,
                addr,
                Load::I32 {
                    src_size: 1,
                    sign_extend: false,
                },
            ),

            Self::Unary(UnOp::I32Load16S(mem_arg), addr) => (
                *mem_arg,
                addr,
                Load::I32 {
                    src_size: 2,
                    sign_extend: true,
                },
            ),

            Self::Unary(UnOp::I32Load16U(mem_arg), addr) => (
                *mem_arg,
                addr,
                Load::I32 {
                    src_size: 2,
                    sign_extend: false,
                },
            ),

            Self::Unary(UnOp::I64Load8S(mem_arg), addr) => (
                *mem_arg,
                addr,
                Load::I64 {
                    src_size: 1,
                    sign_extend: true,
                },
            ),

            Self::Unary(UnOp::I64Load8U(mem_arg), addr) => (
                *mem_arg,
                addr,
                Load::I64 {
                    src_size: 1,
                    sign_extend: false,
                },
            ),

            Self::Unary(UnOp::I64Load16S(mem_arg), addr) => (
                *mem_arg,
                addr,
                Load::I64 {
                    src_size: 2,
                    sign_extend: true,
                },
            ),

            Self::Unary(UnOp::I64Load16U(mem_arg), addr) => (
                *mem_arg,
                addr,
                Load::I64 {
                    src_size: 2,
                    sign_extend: false,
                },
            ),

            Self::Unary(UnOp::I64Load32S(mem_arg), addr) => (
                *mem_arg,
                addr,
                Load::I64 {
                    src_size: 4,
                    sign_extend: true,
                },
            ),

            Self::Unary(UnOp::I64Load32U(mem_arg), addr) => (
                *mem_arg,
                addr,
                Load::I64 {
                    src_size: 4,
                    sign_extend: false,
                },
            ),

            _ => return None,
        })
    }

    pub fn to_store(&self) -> Option<(MemArg, &Expr, &Expr, Store)> {
        match self {
            Self::Binary(op, addr, value) => {
                let (mem_arg, store) = match *op {
                    BinOp::I32Store(mem_arg) => (mem_arg, Store::I32 { dst_size: 4 }),
                    BinOp::I64Store(mem_arg) => (mem_arg, Store::I64 { dst_size: 8 }),
                    BinOp::F32Store(mem_arg) => (mem_arg, Store::F32),
                    BinOp::F64Store(mem_arg) => (mem_arg, Store::F64),

                    BinOp::I32Store8(mem_arg) => (mem_arg, Store::I32 { dst_size: 1 }),
                    BinOp::I32Store16(mem_arg) => (mem_arg, Store::I32 { dst_size: 2 }),

                    BinOp::I64Store8(mem_arg) => (mem_arg, Store::I64 { dst_size: 1 }),
                    BinOp::I64Store16(mem_arg) => (mem_arg, Store::I64 { dst_size: 2 }),
                    BinOp::I64Store32(mem_arg) => (mem_arg, Store::I64 { dst_size: 4 }),

                    _ => return None,
                };

                Some((mem_arg, addr, value, store))
            }

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
            Self::CallIndirect(ty, _, _, _) => ReturnValueCount::CallIndirect(*ty),

            Self::BrIf(_, Some(_), _) => ReturnValueCount::One,
            Self::BrIf(_, None, _) => ReturnValueCount::Zero,

            Self::Br(_, _) | Self::BrTable(_, _, _, _) | Self::Return(_) => {
                ReturnValueCount::Unreachable
            }

            _ => ReturnValueCount::One,
        }
    }

    pub fn ty(&self) -> ExprTy {
        match self {
            Self::Value(Value::I32(_), _) => ValType::I32.into(),
            Self::Value(Value::I64(_), _) => ValType::I64.into(),
            Self::Value(Value::F32(_), _) => ValType::F32.into(),
            Self::Value(Value::F64(_), _) => ValType::F64.into(),

            Self::Index(_) => ValType::I32.into(),

            Self::Intrinsic(Intrinsic::Specialize { .. }) => ValType::I32.into(),
            Self::Intrinsic(Intrinsic::Unknown(val_ty)) => val_ty.clone().into(),

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
                BinOp::I32Store(_) | BinOp::I64Store(_) | BinOp::F32Store(_) | BinOp::F64Store(_),
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

            Self::Nullary(NulOp::MemorySize(_)) | Self::Unary(UnOp::MemoryGrow(_), _) => {
                ValType::I32.into()
            }

            Self::Nullary(NulOp::Nop) => ExprTy::Empty,
            Self::Nullary(NulOp::Unreachable) => ExprTy::Unreachable,

            Self::Block(ty, _) | Self::Loop(ty, _) | Self::If(ty, _, _, _) => match ty {
                Some(ty) => ExprTy::Concrete(ty.clone()),
                None => ExprTy::Empty,
            },

            Self::BrIf(_, Some(expr), _) => expr.ty(),
            Self::BrIf(_, None, _) => ExprTy::Empty,

            Self::Br(_, _) | Self::BrTable(_, _, _, _) | Self::Return(_) => ExprTy::Unreachable,

            Self::Call(func, _) => ExprTy::Call(*func),
            Self::CallIndirect(ty, _, _, _) => ExprTy::CallIndirect(*ty),
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
