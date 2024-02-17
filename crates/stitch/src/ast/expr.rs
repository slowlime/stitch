use std::fmt::{self, Debug, Display};
use std::iter;

use bitflags::bitflags;

use crate::util::{try_match, Indent};
use crate::util::float::{F32, F64};

use super::ty::{BlockType, ValType};
use super::{BlockId, FuncId, GlobalId, LocalId, MemoryId, TableId, TypeId};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Value {
    I32(i32),
    I64(i64),
    F32(F32),
    F64(F64),
}

impl Value {
    pub fn default_for(val_ty: &ValType) -> Self {
        match val_ty {
            ValType::I32 => Self::I32(0),
            ValType::I64 => Self::I64(0),
            ValType::F32 => Self::F32(Default::default()),
            ValType::F64 => Self::F64(Default::default()),
        }
    }

    pub fn val_ty(&self) -> ValType {
        match self {
            Self::I32(_) => ValType::I32,
            Self::I64(_) => ValType::I64,
            Self::F32(_) => ValType::F32,
            Self::F64(_) => ValType::F64,
        }
    }

    pub fn meet(&self, other: &Self) -> Option<Self> {
        match (self, other) {
            (Self::I32(lhs), Self::I32(rhs)) if lhs == rhs => Some(self.clone()),
            (Self::I64(lhs), Self::I64(rhs)) if lhs == rhs => Some(self.clone()),
            (Self::F32(lhs), Self::F32(rhs)) if lhs == rhs => Some(self.clone()),
            (Self::F64(lhs), Self::F64(rhs)) if lhs == rhs => Some(self.clone()),
            _ => None,
        }
    }

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

bitflags! {
    #[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ValueAttrs: u8 {
        const CONST_PTR = 1 << 0;
        const PROPAGATE_LOAD = 1 << 1;
        const UNKNOWN = 1 << 2;
    }
}

impl ValueAttrs {
    pub fn meet(&self, other: &Self) -> Self {
        const MEET_OR: ValueAttrs = ValueAttrs::UNKNOWN;
        const MEET_AND: ValueAttrs = MEET_OR.complement();

        *self & *other & MEET_AND | (*self | *other) & MEET_OR
    }

    pub fn deref_attrs(&self) -> ValueAttrs {
        if self.contains(Self::PROPAGATE_LOAD) {
            *self & !(Self::PROPAGATE_LOAD | Self::UNKNOWN)
        } else {
            Default::default()
        }
    }

    pub fn addsub_attrs(&self, other: &Self) -> ValueAttrs {
        self.meet(other) | (*self | *other) & Self::CONST_PTR
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConstExpr {
    Value(Value, ValueAttrs),
    GlobalGet(GlobalId),
}

impl TryFrom<Expr> for ConstExpr {
    type Error = ();

    fn try_from(expr: Expr) -> Result<Self, Self::Error> {
        match expr {
            Expr::Value(value, attrs) => Ok(Self::Value(value, attrs)),
            Expr::Nullary(NulOp::GlobalGet(global_id)) => Ok(Self::GlobalGet(global_id)),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemArg {
    pub mem_id: MemoryId,
    pub offset: u32,
    pub align: u32,
}

impl Display for MemArg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self {
            mem_id,
            offset,
            align,
        } = self;
        write!(f, "mem={mem_id:?} offset={offset} align={align}")
    }
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

impl Display for NulOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Nop => write!(f, "nop"),
            Self::Unreachable => write!(f, "unreachable"),
            Self::LocalGet(local_id) => write!(f, "local.get {local_id:?}"),
            Self::GlobalGet(global_id) => write!(f, "global.get {global_id:?}"),
            Self::MemorySize(mem_id) => write!(f, "memory.size {mem_id:?}"),
        }
    }
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

impl Display for UnOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::I32Clz => write!(f, "i32.clz"),
            Self::I32Ctz => write!(f, "i32.ctz"),
            Self::I32Popcnt => write!(f, "i32.popcnt"),

            Self::I64Clz => write!(f, "i64.clz"),
            Self::I64Ctz => write!(f, "i64.ctz"),
            Self::I64Popcnt => write!(f, "i64.popcnt"),

            Self::F32Abs => write!(f, "f32.abs"),
            Self::F32Neg => write!(f, "f32.neg"),
            Self::F32Sqrt => write!(f, "f32.sqrt"),
            Self::F32Ceil => write!(f, "f32.ceil"),
            Self::F32Floor => write!(f, "f32.floor"),
            Self::F32Trunc => write!(f, "f32.trunc"),
            Self::F32Nearest => write!(f, "f32.nearest"),

            Self::F64Abs => write!(f, "f64.abs"),
            Self::F64Neg => write!(f, "f64.neg"),
            Self::F64Sqrt => write!(f, "f64.sqrt"),
            Self::F64Ceil => write!(f, "f64.ceil"),
            Self::F64Floor => write!(f, "f64.floor"),
            Self::F64Trunc => write!(f, "f64.trunc"),
            Self::F64Nearest => write!(f, "f64.nearest"),

            Self::I32Eqz => write!(f, "i32.eqz"),
            Self::I64Eqz => write!(f, "i64.eqz"),

            Self::I32WrapI64 => write!(f, "i32.wrap_i64"),

            Self::I64ExtendI32S => write!(f, "i64.extend_i32_s"),
            Self::I64ExtendI32U => write!(f, "i64.extend_i32_u"),

            Self::I32TruncF32S => write!(f, "i32.trunc_f32_s"),
            Self::I32TruncF32U => write!(f, "i32.trunc_f32_u"),
            Self::I32TruncF64S => write!(f, "i32.trunc_f64_s"),
            Self::I32TruncF64U => write!(f, "i32.trunc_f64_u"),

            Self::I64TruncF32S => write!(f, "i64.trunc_f32_s"),
            Self::I64TruncF32U => write!(f, "i64.trunc_f32_u"),
            Self::I64TruncF64S => write!(f, "i64.trunc_f64_s"),
            Self::I64TruncF64U => write!(f, "i64.trunc_f64_u"),

            Self::F32DemoteF64 => write!(f, "f32.demote_f64"),
            Self::F64PromoteF32 => write!(f, "f64.promote_f32"),

            Self::F32ConvertI32S => write!(f, "f32.convert_i32_s"),
            Self::F32ConvertI32U => write!(f, "f32.convert_i32_u"),
            Self::F32ConvertI64S => write!(f, "f32.convert_i64_s"),
            Self::F32ConvertI64U => write!(f, "f32.convert_i64_u"),

            Self::F64ConvertI32S => write!(f, "f64.convert_i32_s"),
            Self::F64ConvertI32U => write!(f, "f64.convert_i32_u"),
            Self::F64ConvertI64S => write!(f, "f64.convert_i64_s"),
            Self::F64ConvertI64U => write!(f, "f64.convert_i64_u"),

            Self::F32ReinterpretI32 => write!(f, "f32.reinterpret_i32"),
            Self::F64ReinterpretI64 => write!(f, "f64.reinterpret_i64"),
            Self::I32ReinterpretF32 => write!(f, "i32.reinterpret_f32"),
            Self::I64ReinterpretF64 => write!(f, "i64.reinterpret_f64"),

            Self::I32Extend8S => write!(f, "i32.extend8_s"),
            Self::I32Extend16S => write!(f, "i32.extend16_s"),

            Self::I64Extend8S => write!(f, "i64.extend8_s"),
            Self::I64Extend16S => write!(f, "i64.extend16_s"),
            Self::I64Extend32S => write!(f, "i64.extend32_s"),

            Self::LocalSet(local_id) => write!(f, "local.set {local_id:?}"),
            Self::LocalTee(local_id) => write!(f, "local.tee {local_id:?}"),

            Self::GlobalSet(global_id) => write!(f, "global.set {global_id:?}"),

            Self::I32Load(mem_arg) => write!(f, "i32.load {mem_arg}"),
            Self::I64Load(mem_arg) => write!(f, "i64.load {mem_arg}"),
            Self::F32Load(mem_arg) => write!(f, "f32.load {mem_arg}"),
            Self::F64Load(mem_arg) => write!(f, "f64.load {mem_arg}"),

            Self::I32Load8S(mem_arg) => write!(f, "i32.load8_s {mem_arg}"),
            Self::I32Load8U(mem_arg) => write!(f, "i32.load8_u {mem_arg}"),
            Self::I32Load16S(mem_arg) => write!(f, "i32.load16_s {mem_arg}"),
            Self::I32Load16U(mem_arg) => write!(f, "i32.load16_u {mem_arg}"),

            Self::I64Load8S(mem_arg) => write!(f, "i64.load8_s {mem_arg}"),
            Self::I64Load8U(mem_arg) => write!(f, "i64.load8_u {mem_arg}"),
            Self::I64Load16S(mem_arg) => write!(f, "i64.load16_s {mem_arg}"),
            Self::I64Load16U(mem_arg) => write!(f, "i64.load16_u {mem_arg}"),
            Self::I64Load32S(mem_arg) => write!(f, "i64.load32_s {mem_arg}"),
            Self::I64Load32U(mem_arg) => write!(f, "i64.load32_u {mem_arg}"),

            Self::MemoryGrow(mem_id) => write!(f, "memory.grow {mem_id:?}"),

            Self::Drop => write!(f, "drop"),
        }
    }
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

impl Display for BinOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::I32Add => write!(f, "i32.add"),
            Self::I32Sub => write!(f, "i32.sub"),
            Self::I32Mul => write!(f, "i32.mul"),
            Self::I32DivS => write!(f, "i32.div_s"),
            Self::I32DivU => write!(f, "i32.div_u"),
            Self::I32RemS => write!(f, "i32.rem_s"),
            Self::I32RemU => write!(f, "i32.rem_u"),
            Self::I32And => write!(f, "i32.and"),
            Self::I32Or => write!(f, "i32.or"),
            Self::I32Xor => write!(f, "i32.xor"),
            Self::I32Shl => write!(f, "i32.shl"),
            Self::I32ShrS => write!(f, "i32.shr_s"),
            Self::I32ShrU => write!(f, "i32.shr_u"),
            Self::I32Rotl => write!(f, "i32.rotl"),
            Self::I32Rotr => write!(f, "i32.rotr"),

            Self::I64Add => write!(f, "i64.add"),
            Self::I64Sub => write!(f, "i64.sub"),
            Self::I64Mul => write!(f, "i64.mul"),
            Self::I64DivS => write!(f, "i64.div_s"),
            Self::I64DivU => write!(f, "i64.div_u"),
            Self::I64RemS => write!(f, "i64.rem_s"),
            Self::I64RemU => write!(f, "i64.rem_u"),
            Self::I64And => write!(f, "i64.and"),
            Self::I64Or => write!(f, "i64.or"),
            Self::I64Xor => write!(f, "i64.xor"),
            Self::I64Shl => write!(f, "i64.shl"),
            Self::I64ShrS => write!(f, "i64.shr_s"),
            Self::I64ShrU => write!(f, "i64.shr_u"),
            Self::I64Rotl => write!(f, "i64.rotl"),
            Self::I64Rotr => write!(f, "i64.rotr"),

            Self::F32Add => write!(f, "f32.add"),
            Self::F32Sub => write!(f, "f32.sub"),
            Self::F32Mul => write!(f, "f32.mul"),
            Self::F32Div => write!(f, "f32.div"),
            Self::F32Min => write!(f, "f32.min"),
            Self::F32Max => write!(f, "f32.max"),
            Self::F32Copysign => write!(f, "f32.copysign"),

            Self::F64Add => write!(f, "f64.add"),
            Self::F64Sub => write!(f, "f64.sub"),
            Self::F64Mul => write!(f, "f64.mul"),
            Self::F64Div => write!(f, "f64.div"),
            Self::F64Min => write!(f, "f64.min"),
            Self::F64Max => write!(f, "f64.max"),
            Self::F64Copysign => write!(f, "f64.copysign"),

            Self::I32Eq => write!(f, "i32.eq"),
            Self::I32Ne => write!(f, "i32.ne"),
            Self::I32LtS => write!(f, "i32.lt_s"),
            Self::I32LtU => write!(f, "i32.lt_u"),
            Self::I32GtS => write!(f, "i32.gt_s"),
            Self::I32GtU => write!(f, "i32.gt_u"),
            Self::I32LeS => write!(f, "i32.le_s"),
            Self::I32LeU => write!(f, "i32.le_u"),
            Self::I32GeS => write!(f, "i32.ge_s"),
            Self::I32GeU => write!(f, "i32.ge_u"),

            Self::I64Eq => write!(f, "i64.eq"),
            Self::I64Ne => write!(f, "i64.ne"),
            Self::I64LtS => write!(f, "i64.lt_s"),
            Self::I64LtU => write!(f, "i64.lt_u"),
            Self::I64GtS => write!(f, "i64.gt_s"),
            Self::I64GtU => write!(f, "i64.gt_u"),
            Self::I64LeS => write!(f, "i64.le_s"),
            Self::I64LeU => write!(f, "i64.le_u"),
            Self::I64GeS => write!(f, "i64.ge_s"),
            Self::I64GeU => write!(f, "i64.ge_u"),

            Self::F32Eq => write!(f, "f32.eq"),
            Self::F32Ne => write!(f, "f32.ne"),
            Self::F32Lt => write!(f, "f32.lt"),
            Self::F32Gt => write!(f, "f32.gt"),
            Self::F32Le => write!(f, "f32.le"),
            Self::F32Ge => write!(f, "f32.ge"),

            Self::F64Eq => write!(f, "f64.eq"),
            Self::F64Ne => write!(f, "f64.ne"),
            Self::F64Lt => write!(f, "f64.lt"),
            Self::F64Gt => write!(f, "f64.gt"),
            Self::F64Le => write!(f, "f64.le"),
            Self::F64Ge => write!(f, "f64.ge"),

            Self::I32Store(mem_arg) => write!(f, "i32.store {mem_arg}"),
            Self::I64Store(mem_arg) => write!(f, "i64.store {mem_arg}"),
            Self::F32Store(mem_arg) => write!(f, "f32.store {mem_arg}"),
            Self::F64Store(mem_arg) => write!(f, "f64.store {mem_arg}"),

            Self::I32Store8(mem_arg) => write!(f, "i32.store8 {mem_arg}"),
            Self::I32Store16(mem_arg) => write!(f, "i32.store16 {mem_arg}"),

            Self::I64Store8(mem_arg) => write!(f, "i64.store8 {mem_arg}"),
            Self::I64Store16(mem_arg) => write!(f, "i64.store16 {mem_arg}"),
            Self::I64Store32(mem_arg) => write!(f, "i64.store32 {mem_arg}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TernOp {
    Select,
}

impl Display for TernOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Select => write!(f, "select"),
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct Block {
    pub body: Vec<Expr>,
    pub id: BlockId,
}

impl Display for Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(block id={:?}", self.id)?;

        for expr in &self.body {
            write!(f, "\n{}", expr.printer(1, ": "))?;
        }

        write!(f, ")")
    }
}

#[derive(Debug, Clone)]
pub enum Expr {
    Value(Value, ValueAttrs),

    Nullary(NulOp),
    Unary(UnOp, Box<Expr>),
    Binary(BinOp, [Box<Expr>; 2]),
    Ternary(TernOp, [Box<Expr>; 3]),

    Block(BlockType, Block),
    Loop(BlockType, Block),
    If(BlockType, Box<Expr>, Block, Block),
    Br(BlockId, Option<Box<Expr>>),
    BrIf(BlockId, Option<Box<Expr>>, Box<Expr>),
    BrTable(Vec<BlockId>, BlockId, Option<Box<Expr>>, Box<Expr>),
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

impl Default for Expr {
    fn default() -> Self {
        NulOp::Nop.into()
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
    #[allow(unused_variables)]
    fn pre(&mut self, ctx: &mut VisitContext, expr: &Expr) {}

    #[allow(unused_variables)]
    fn post(&mut self, ctx: &mut VisitContext, expr: Expr) -> Expr {
        expr
    }
}

impl<F> Visitor for F
where
    F: FnMut(Expr, &mut VisitContext) -> Expr,
{
    fn post(&mut self, ctx: &mut VisitContext, expr: Expr) -> Expr {
        self(expr, ctx)
    }
}

pub fn make_visitor(f: impl FnMut(&Expr, &mut VisitContext)) -> impl Visitor {
    struct V<F>(F);

    impl<F: FnMut(&Expr, &mut VisitContext)> Visitor for V<F> {
        fn pre(&mut self, ctx: &mut VisitContext, expr: &Expr) {
            (self.0)(expr, ctx);
        }
    }

    V(f)
}

impl Expr {
    pub fn map(&self, v: &mut impl Visitor) -> Self {
        fn visit_block(v: &mut impl Visitor, ctx: &mut VisitContext, block: &Block) -> Block {
            ctx.relative_depth += 1;
            let body = block.body.iter().map(|expr| visit(v, ctx, expr)).collect();
            ctx.relative_depth -= 1;

            Block { body, id: block.id }
        }

        fn visit(v: &mut impl Visitor, ctx: &mut VisitContext, expr: &Expr) -> Expr {
            v.pre(ctx, expr);

            let result = match expr {
                Expr::Value(_, _) | Expr::Nullary(_) => expr.clone(),

                Expr::Unary(op, inner) => Expr::Unary(*op, Box::new(visit(v, ctx, inner))),
                Expr::Binary(op, [lhs, rhs]) => Expr::Binary(
                    *op,
                    [Box::new(visit(v, ctx, lhs)), Box::new(visit(v, ctx, rhs))],
                ),
                Expr::Ternary(op, [first, second, third]) => Expr::Ternary(
                    *op,
                    [
                        Box::new(visit(v, ctx, first)),
                        Box::new(visit(v, ctx, second)),
                        Box::new(visit(v, ctx, third)),
                    ],
                ),

                Expr::Block(block_ty, block) => {
                    Expr::Block(block_ty.clone(), visit_block(v, ctx, block))
                }
                Expr::Loop(block_ty, block) => {
                    Expr::Loop(block_ty.clone(), visit_block(v, ctx, block))
                }

                Expr::If(block_ty, condition, then_block, else_block) => Expr::If(
                    block_ty.clone(),
                    Box::new(visit(v, ctx, condition)),
                    visit_block(v, ctx, then_block),
                    visit_block(v, ctx, else_block),
                ),

                Expr::Br(block_id, inner) => Expr::Br(
                    *block_id,
                    inner.as_ref().map(|inner| Box::new(visit(v, ctx, inner))),
                ),

                Expr::BrIf(block_id, inner, condition) => Expr::BrIf(
                    *block_id,
                    inner.as_ref().map(|inner| Box::new(visit(v, ctx, inner))),
                    Box::new(visit(v, ctx, condition)),
                ),

                Expr::BrTable(block_ids, default_block_id, inner, index) => Expr::BrTable(
                    block_ids.clone(),
                    *default_block_id,
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
            Expr::Value(_, _) | Expr::Nullary(_) => true,
            Expr::Unary(_, inner) => inner.all(predicate),
            Expr::Binary(_, exprs) => exprs.iter().all(|arg| arg.all(predicate)),
            Expr::Ternary(_, exprs) => exprs.iter().all(|arg| arg.all(predicate)),
            Expr::Block(_, block) | Expr::Loop(_, block) => {
                block.body.iter().all(|expr| expr.all(predicate))
            }
            Expr::If(_, condition, then_block, else_block) => {
                condition.all(predicate)
                    && then_block.body.iter().all(|expr| expr.all(predicate))
                    && else_block.body.iter().all(|expr| expr.all(predicate))
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
            Expr::Call(_, args) => args.iter().all(|expr| expr.all(predicate)),
            Expr::CallIndirect(_, _, args, index) => args
                .iter()
                .chain(iter::once(&**index))
                .all(|expr| expr.all(predicate)),
        }) && predicate(self)
    }

    pub fn any(&self, predicate: &mut impl FnMut(&Expr) -> bool) -> bool {
        !self.all(&mut |expr| !predicate(expr))
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
            Self::Binary(op, [addr, value]) => {
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
            Self::Nullary(NulOp::Nop)
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
            ) => ReturnValueCount::Zero,

            Self::Nullary(NulOp::Unreachable) => ReturnValueCount::Unreachable,

            Self::Block(block_ty, _) | Self::Loop(block_ty, _) | Self::If(block_ty, _, _, _) => {
                if block_ty.is_empty() {
                    ReturnValueCount::Zero
                } else {
                    ReturnValueCount::One
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

    pub fn diverges(&self) -> bool {
        matches!(self.ret_value_count(), ReturnValueCount::Unreachable)
    }

    pub fn ty(&self) -> ExprTy {
        match self {
            Self::Value(value, _) => value.val_ty().into(),

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
            ) => ValType::I64.into(),

            Self::Binary(
                BinOp::F32Eq
                | BinOp::F32Ne
                | BinOp::F32Lt
                | BinOp::F32Gt
                | BinOp::F32Le
                | BinOp::F32Ge,
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
            Self::Ternary(TernOp::Select, [lhs, _, _]) => lhs.ty(),

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

            Self::Binary(BinOp::I32Store8(_) | BinOp::I32Store16(_), _) => ValType::I32.into(),

            Self::Binary(BinOp::I64Store8(_) | BinOp::I64Store16(_) | BinOp::I64Store32(_), _) => {
                ValType::I64.into()
            }

            Self::Nullary(NulOp::MemorySize(_)) | Self::Unary(UnOp::MemoryGrow(_), _) => {
                ValType::I32.into()
            }

            Self::Nullary(NulOp::Nop) => ExprTy::Empty,
            Self::Nullary(NulOp::Unreachable) => ExprTy::Unreachable,

            Self::Block(block_ty, _) | Self::Loop(block_ty, _) | Self::If(block_ty, _, _, _) => {
                match block_ty {
                    BlockType::Empty => ExprTy::Empty,
                    BlockType::Result(ty) => ExprTy::Concrete(ty.clone()),
                }
            }

            Self::BrIf(_, Some(expr), _) => expr.ty(),
            Self::BrIf(_, None, _) => ExprTy::Empty,

            Self::Br(_, _) | Self::BrTable(_, _, _, _) | Self::Return(_) => ExprTy::Unreachable,

            Self::Call(func, _) => ExprTy::Call(*func),
            Self::CallIndirect(ty, _, _, _) => ExprTy::CallIndirect(*ty),
        }
    }

    pub fn has_side_effect(&self) -> bool {
        self.any(&mut |expr| match expr {
            Expr::Value(..) => false,

            Expr::Nullary(op) => match op {
                NulOp::Nop => false,
                NulOp::Unreachable => true,

                NulOp::LocalGet(_) | NulOp::GlobalGet(_) | NulOp::MemorySize(_) => false,
            },

            Expr::Unary(op, _) => match op {
                UnOp::I32Clz
                | UnOp::I32Ctz
                | UnOp::I32Popcnt
                | UnOp::I64Clz
                | UnOp::I64Ctz
                | UnOp::I64Popcnt
                | UnOp::F32Abs
                | UnOp::F32Neg
                | UnOp::F32Sqrt
                | UnOp::F32Ceil
                | UnOp::F32Floor
                | UnOp::F32Trunc
                | UnOp::F32Nearest
                | UnOp::F64Abs
                | UnOp::F64Neg
                | UnOp::F64Sqrt
                | UnOp::F64Ceil
                | UnOp::F64Floor
                | UnOp::F64Trunc
                | UnOp::F64Nearest
                | UnOp::I32Eqz
                | UnOp::I64Eqz
                | UnOp::I32WrapI64
                | UnOp::I64ExtendI32S
                | UnOp::I64ExtendI32U
                | UnOp::I32TruncF32S
                | UnOp::I32TruncF32U
                | UnOp::I32TruncF64S
                | UnOp::I32TruncF64U
                | UnOp::I64TruncF32S
                | UnOp::I64TruncF32U
                | UnOp::I64TruncF64S
                | UnOp::I64TruncF64U
                | UnOp::F32DemoteF64
                | UnOp::F64PromoteF32
                | UnOp::F32ConvertI32S
                | UnOp::F32ConvertI32U
                | UnOp::F32ConvertI64S
                | UnOp::F32ConvertI64U
                | UnOp::F64ConvertI32S
                | UnOp::F64ConvertI32U
                | UnOp::F64ConvertI64S
                | UnOp::F64ConvertI64U
                | UnOp::F32ReinterpretI32
                | UnOp::F64ReinterpretI64
                | UnOp::I32ReinterpretF32
                | UnOp::I64ReinterpretF64
                | UnOp::I32Extend8S
                | UnOp::I32Extend16S
                | UnOp::I64Extend8S
                | UnOp::I64Extend16S
                | UnOp::I64Extend32S => false,

                UnOp::LocalSet(_) | UnOp::LocalTee(_) => true,
                UnOp::GlobalSet(_) => true,

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
                | UnOp::I64Load32U(_) => false,

                UnOp::MemoryGrow(_) => true,
                UnOp::Drop => false,
            },

            Expr::Binary(op, ..) => match op {
                BinOp::I32DivS
                | BinOp::I32DivU
                | BinOp::I32RemS
                | BinOp::I32RemU
                | BinOp::I64DivS
                | BinOp::I64DivU
                | BinOp::I64RemS
                | BinOp::I64RemU => true, // conservative

                BinOp::I32Add
                | BinOp::I32Sub
                | BinOp::I32Mul
                | BinOp::I32And
                | BinOp::I32Or
                | BinOp::I32Xor
                | BinOp::I32Shl
                | BinOp::I32ShrS
                | BinOp::I32ShrU
                | BinOp::I32Rotl
                | BinOp::I32Rotr
                | BinOp::I64Add
                | BinOp::I64Sub
                | BinOp::I64Mul
                | BinOp::I64And
                | BinOp::I64Or
                | BinOp::I64Xor
                | BinOp::I64Shl
                | BinOp::I64ShrS
                | BinOp::I64ShrU
                | BinOp::I64Rotl
                | BinOp::I64Rotr
                | BinOp::F32Add
                | BinOp::F32Sub
                | BinOp::F32Mul
                | BinOp::F32Div
                | BinOp::F32Min
                | BinOp::F32Max
                | BinOp::F32Copysign
                | BinOp::F64Add
                | BinOp::F64Sub
                | BinOp::F64Mul
                | BinOp::F64Div
                | BinOp::F64Min
                | BinOp::F64Max
                | BinOp::F64Copysign
                | BinOp::I32Eq
                | BinOp::I32Ne
                | BinOp::I32LtS
                | BinOp::I32LtU
                | BinOp::I32GtS
                | BinOp::I32GtU
                | BinOp::I32LeS
                | BinOp::I32LeU
                | BinOp::I32GeS
                | BinOp::I32GeU
                | BinOp::I64Eq
                | BinOp::I64Ne
                | BinOp::I64LtS
                | BinOp::I64LtU
                | BinOp::I64GtS
                | BinOp::I64GtU
                | BinOp::I64LeS
                | BinOp::I64LeU
                | BinOp::I64GeS
                | BinOp::I64GeU
                | BinOp::F32Eq
                | BinOp::F32Ne
                | BinOp::F32Lt
                | BinOp::F32Gt
                | BinOp::F32Le
                | BinOp::F32Ge
                | BinOp::F64Eq
                | BinOp::F64Ne
                | BinOp::F64Lt
                | BinOp::F64Gt
                | BinOp::F64Le
                | BinOp::F64Ge => false,

                BinOp::I32Store(_)
                | BinOp::I64Store(_)
                | BinOp::F32Store(_)
                | BinOp::F64Store(_)
                | BinOp::I32Store8(_)
                | BinOp::I32Store16(_)
                | BinOp::I64Store8(_)
                | BinOp::I64Store16(_)
                | BinOp::I64Store32(_) => true,
            },

            Expr::Ternary(op, ..) => match op {
                TernOp::Select => false,
            },

            Expr::Block(..) => false,
            Expr::Loop(..) => true,
            Expr::If(..) => false,
            Expr::Br(..) | Expr::BrIf(..) | Expr::BrTable(..) | Expr::Return(..) => true,
            Expr::Call(..) | Expr::CallIndirect(..) => true, // conservative
        })
    }

    pub fn branches_to(&self, block_id: BlockId) -> bool {
        self.any(&mut |expr| match expr {
            Expr::Br(br_block_id, _) | Expr::BrIf(br_block_id, ..) => block_id == *br_block_id,

            Expr::BrTable(br_block_ids, default_br_block_id, ..) => {
                *default_br_block_id == block_id || br_block_ids.contains(&block_id)
            }

            _ => false,
        })
    }

    pub fn printer(&self, indent_level: usize, indent: &'static str) -> impl Display + '_ {
        struct Printer<'a, 'b> {
            f: &'a mut fmt::Formatter<'b>,
            indent: Indent<&'static str>,
            first_line: bool,
        }

        impl Printer<'_, '_> {
            fn visit_block(&mut self, block: &Block) -> fmt::Result {
                for expr in &block.body {
                    self.visit(expr)?;
                }

                Ok(())
            }

            fn visit(&mut self, expr: &Expr) -> fmt::Result {
                write!(
                    self.f,
                    "{}{}(",
                    if self.first_line { "" } else { "\n" },
                    self.indent,
                )?;
                self.first_line = false;

                match expr {
                    Expr::Value(value, attrs) => {
                        match value {
                            Value::I32(value) => write!(self.f, "i32.const {value}")?,
                            Value::I64(value) => write!(self.f, "i64.const {value}")?,
                            Value::F32(value) => write!(self.f, "f32.const {value}")?,
                            Value::F64(value) => write!(self.f, "f64.const {value}")?,
                        }

                        write!(self.f, " {attrs:?}")?
                    }

                    Expr::Nullary(op) => write!(self.f, "{op}")?,
                    Expr::Unary(op, _) => write!(self.f, "{op}")?,
                    Expr::Binary(op, ..) => write!(self.f, "{op}")?,
                    Expr::Ternary(op, ..) => write!(self.f, "{op}")?,

                    Expr::Block(block_ty, Block { id, .. }) => {
                        write!(self.f, "block {block_ty} id={id:?}")?
                    }

                    Expr::Loop(block_ty, Block { id, .. }) => {
                        write!(self.f, "loop {block_ty} id={id:?}")?
                    }

                    Expr::If(block_ty, condition, then_block, else_block) => {
                        write!(self.f, "if {block_ty}")?;
                        self.indent.0 += 1;
                        self.visit(condition)?;
                        write!(self.f, "\n{}(then", self.indent)?;
                        self.indent.0 += 1;
                        self.visit_block(then_block)?;
                        self.indent.0 -= 1;
                        write!(self.f, ")\n{}(else", self.indent)?;
                        self.indent.0 += 1;
                        self.visit_block(else_block)?;
                        self.indent.0 -= 2;
                        write!(self.f, ")")?;
                    }

                    Expr::Br(block_id, _) => write!(self.f, "br {block_id:?}")?,
                    Expr::BrIf(block_id, ..) => write!(self.f, "br_if {block_id:?}")?,

                    Expr::BrTable(block_ids, block_id, ..) => {
                        write!(self.f, "br_table {block_ids:?} {block_id:?}",)?
                    }

                    Expr::Return(_) => write!(self.f, "return")?,
                    Expr::Call(func_id, _) => write!(self.f, "call {func_id:?}")?,
                    Expr::CallIndirect(ty_id, table_id, ..) => {
                        write!(self.f, "call_indirect {ty_id:?} {table_id:?}",)?
                    }
                }

                self.indent.0 += 1;

                match expr {
                    Expr::Value(..) | Expr::Nullary(_) => {}

                    Expr::Unary(_, inner) => self.visit(inner)?,

                    Expr::Binary(_, exprs) => {
                        for expr in exprs {
                            self.visit(expr)?;
                        }
                    }

                    Expr::Ternary(_, exprs) => {
                        for expr in exprs {
                            self.visit(expr)?;
                        }
                    }

                    Expr::Block(_, block) | Expr::Loop(_, block) => self.visit_block(block)?,
                    Expr::If(..) => {}

                    Expr::Br(_, inner) => {
                        if let Some(inner) = inner {
                            self.visit(inner)?;
                        }
                    }

                    Expr::BrIf(_, inner, condition) => {
                        if let Some(inner) = inner {
                            self.visit(inner)?;
                        }

                        self.visit(condition)?;
                    }

                    Expr::BrTable(.., inner, index) => {
                        if let Some(inner) = inner {
                            self.visit(inner)?;
                        }

                        self.visit(index)?;
                    }

                    Expr::Return(inner) => {
                        if let Some(inner) = inner {
                            self.visit(inner)?;
                        }
                    }

                    Expr::Call(_, args) => {
                        for arg in args {
                            self.visit(arg)?;
                        }
                    }

                    Expr::CallIndirect(.., args, index) => {
                        for arg in args {
                            self.visit(arg)?;
                        }

                        self.visit(index)?;
                    }
                }

                self.indent.0 -= 1;

                write!(self.f, ")")
            }
        }

        struct Wrapper<'a>(Indent<&'static str>, &'a Expr);

        impl Display for Wrapper<'_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let mut printer = Printer {
                    f,
                    indent: self.0,
                    first_line: true,
                };

                printer.visit(self.1)
            }
        }

        Wrapper(Indent(indent_level, indent), self)
    }
}

impl Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.printer(0, ": "))
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
