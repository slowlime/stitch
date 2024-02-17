mod dom_tree;
mod dot;
mod from_ast;
mod loops;
mod merge_blocks;
mod predecessors;
mod printer;
mod remove_unreachable_blocks;
mod rpo;
mod to_ast;

use std::{iter, slice};

use slotmap::{new_key_type, SlotMap};

use crate::ast::expr::{MemArg, Value, ValueAttrs};
use crate::ast::ty::{FuncType, ValType};
use crate::ast::{FuncId, GlobalId, MemoryId, TableId, TypeId};
use crate::util::try_match;

pub use self::dom_tree::DomTree;
pub use self::predecessors::Predecessors;
pub use self::rpo::Rpo;

new_key_type! {
    pub struct LocalId;
    pub struct BlockId;
    pub struct ParamId;
}

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum I32Load {
    Four = 0b0_100,
    TwoS = 0b1_010,
    TwoU = 0b0_010,
    OneS = 0b1_001,
    OneU = 0b0_001,
}

impl I32Load {
    pub fn src_size(&self) -> usize {
        *self as usize & 0b111
    }

    pub fn sign_extend(&self) -> bool {
        *self as u8 & 0b1_000 != 0
    }

    pub fn load(&self, src: &[u8]) -> i32 {
        match self {
            Self::Four => i32::from_le_bytes(src.try_into().unwrap()),
            Self::TwoS => i16::from_le_bytes(src.try_into().unwrap()) as i32,
            Self::TwoU => u16::from_le_bytes(src.try_into().unwrap()) as i32,
            Self::OneS => i8::from_le_bytes(src.try_into().unwrap()) as i32,
            Self::OneU => u8::from_le_bytes(src.try_into().unwrap()) as i32,
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum I64Load {
    Eight = 0b0_1000,
    FourS = 0b1_0100,
    FourU = 0b0_0100,
    TwoS = 0b1_0010,
    TwoU = 0b0_0010,
    OneS = 0b1_0001,
    OneU = 0b0_0001,
}

impl I64Load {
    pub fn src_size(&self) -> usize {
        *self as usize & 0b1111
    }

    pub fn sign_extend(&self) -> bool {
        *self as u8 & 0b1_0000 != 0
    }

    pub fn load(&self, src: &[u8]) -> i64 {
        match self {
            Self::Eight => i64::from_le_bytes(src.try_into().unwrap()),
            Self::FourS => i32::from_le_bytes(src.try_into().unwrap()) as i64,
            Self::FourU => u32::from_le_bytes(src.try_into().unwrap()) as i64,
            Self::TwoS => i16::from_le_bytes(src.try_into().unwrap()) as i64,
            Self::TwoU => u16::from_le_bytes(src.try_into().unwrap()) as i64,
            Self::OneS => i8::from_le_bytes(src.try_into().unwrap()) as i64,
            Self::OneU => u8::from_le_bytes(src.try_into().unwrap()) as i64,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Load {
    I32(I32Load),
    I64(I64Load),
    F32,
    F64,
}

impl Load {
    pub fn src_size(&self) -> usize {
        match self {
            Self::I32(load) => load.src_size(),
            Self::I64(load) => load.src_size(),
            Self::F32 => 4,
            Self::F64 => 8,
        }
    }

    pub fn sign_extend(&self) -> bool {
        match self {
            Self::I32(load) => load.sign_extend(),
            Self::I64(load) => load.sign_extend(),
            Self::F32 | Self::F64 => false,
        }
    }

    pub fn load(&self, src: &[u8]) -> Value {
        match self {
            Self::I32(load) => Value::I32(load.load(src)),
            Self::I64(load) => Value::I64(load.load(src)),
            Self::F32 => Value::F32(f32::from_le_bytes(src.try_into().unwrap()).into()),
            Self::F64 => Value::F64(f64::from_le_bytes(src.try_into().unwrap()).into()),
        }
    }
}

impl From<I32Load> for Load {
    fn from(load: I32Load) -> Self {
        Self::I32(load)
    }
}

impl From<I64Load> for Load {
    fn from(load: I64Load) -> Self {
        Self::I64(load)
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum I32Store {
    Four = 4,
    Two = 2,
    One = 1,
}

impl I32Store {
    pub fn dst_size(&self) -> usize {
        *self as usize
    }

    pub fn store(&self, dst: &mut [u8], value: i32) {
        let src = value.to_le_bytes();
        dst.copy_from_slice(&src[0..*self as usize]);
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum I64Store {
    Eight = 8,
    Four = 4,
    Two = 2,
    One = 1,
}

impl I64Store {
    pub fn dst_size(&self) -> usize {
        *self as usize
    }

    pub fn store(&self, dst: &mut [u8], value: i64) {
        let src = value.to_le_bytes();
        dst.copy_from_slice(&src[0..*self as usize]);
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Store {
    I32(I32Store),
    I64(I64Store),
    F32,
    F64,
}

impl Store {
    pub fn dst_size(&self) -> usize {
        match self {
            Self::I32(store) => store.dst_size(),
            Self::I64(store) => store.dst_size(),
            Self::F32 => 4,
            Self::F64 => 8,
        }
    }

    pub fn store(&self, dst: &mut [u8], value: Value) {
        match *self {
            Self::I32(store) => store.store(dst, value.to_i32().unwrap()),
            Self::I64(store) => store.store(dst, value.to_i64().unwrap()),
            Self::F32 => dst.copy_from_slice(&value.to_f32().unwrap().to_bits().to_le_bytes()),
            Self::F64 => dst.copy_from_slice(&value.to_f64().unwrap().to_bits().to_le_bytes()),
        }
    }
}

impl From<I32Store> for Store {
    fn from(store: I32Store) -> Self {
        Self::I32(store)
    }
}

impl From<I64Store> for Store {
    fn from(store: I64Store) -> Self {
        Self::I64(store)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum NulOp {
    LocalGet(LocalId),
    GlobalGet(GlobalId),
    MemorySize(MemoryId),
}

#[derive(Debug, Clone, Copy)]
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

    Load(MemArg, Load),
    MemoryGrow(MemoryId),
}

#[derive(Debug, Clone, Copy)]
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
}

#[derive(Debug, Clone, Copy)]
pub enum TernOp {
    Select,
}

#[derive(Debug, Clone)]
pub enum Call {
    Direct {
        ret_local_id: Option<LocalId>,
        func_id: FuncId,
        args: Vec<Expr>,
    },

    Indirect {
        ret_local_id: Option<LocalId>,
        ty_id: TypeId,
        table_id: TableId,
        args: Vec<Expr>,
        index: Box<Expr>,
    },
}

impl Call {
    pub fn ret_local_id(&self) -> Option<LocalId> {
        match *self {
            Self::Direct { ret_local_id, .. } | Self::Indirect { ret_local_id, .. } => ret_local_id,
        }
    }

    pub fn nth_subexpr(&self, n: usize) -> Option<&Expr> {
        match self {
            Self::Direct { args, .. } => args.get(n),
            Self::Indirect { args, index, .. } => args.iter().chain(iter::once(&**index)).nth(n),
        }
    }

    pub fn subexpr_count(&self) -> usize {
        match self {
            Self::Direct { args, .. } => args.len(),
            Self::Indirect { args, .. } => args.len() + 1,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Expr {
    Value(Value, ValueAttrs),
    Nullary(NulOp),
    Unary(UnOp, Box<Expr>),
    Binary(BinOp, Box<[Expr; 2]>),
    Ternary(TernOp, Box<[Expr; 3]>),
}

impl Expr {
    pub fn to_value(&self) -> Option<(Value, ValueAttrs)> {
        try_match!(*self, Self::Value(value, attrs) => (value, attrs))
    }

    pub fn ty(&self) -> ExprTy {
        match self {
            Self::Value(value, _) => value.val_ty().into(),

            Self::Nullary(op) => match *op {
                NulOp::LocalGet(local_id) => ExprTy::Local(local_id),
                NulOp::GlobalGet(global_id) => ExprTy::Global(global_id),
                NulOp::MemorySize(_) => ValType::I32.into(),
            },

            Self::Unary(op, _) => match *op {
                UnOp::I32Clz | UnOp::I32Ctz | UnOp::I32Popcnt => ValType::I32.into(),

                UnOp::I64Clz | UnOp::I64Ctz | UnOp::I64Popcnt => ValType::I64.into(),

                UnOp::F32Abs
                | UnOp::F32Neg
                | UnOp::F32Sqrt
                | UnOp::F32Ceil
                | UnOp::F32Floor
                | UnOp::F32Trunc
                | UnOp::F32Nearest => ValType::F32.into(),

                UnOp::F64Abs
                | UnOp::F64Neg
                | UnOp::F64Sqrt
                | UnOp::F64Ceil
                | UnOp::F64Floor
                | UnOp::F64Trunc
                | UnOp::F64Nearest => ValType::F64.into(),

                UnOp::I32Eqz | UnOp::I64Eqz => ValType::I32.into(),

                UnOp::I32WrapI64 => ValType::I32.into(),
                UnOp::I64ExtendI32S | UnOp::I64ExtendI32U => ValType::I64.into(),

                UnOp::I32TruncF32S
                | UnOp::I32TruncF32U
                | UnOp::I32TruncF64S
                | UnOp::I32TruncF64U => ValType::I32.into(),

                UnOp::I64TruncF32S
                | UnOp::I64TruncF32U
                | UnOp::I64TruncF64S
                | UnOp::I64TruncF64U => ValType::I64.into(),

                UnOp::F32DemoteF64 => ValType::F32.into(),
                UnOp::F64PromoteF32 => ValType::F64.into(),

                UnOp::F32ConvertI32S
                | UnOp::F32ConvertI32U
                | UnOp::F32ConvertI64S
                | UnOp::F32ConvertI64U => ValType::F32.into(),

                UnOp::F64ConvertI32S
                | UnOp::F64ConvertI32U
                | UnOp::F64ConvertI64S
                | UnOp::F64ConvertI64U => ValType::F64.into(),

                UnOp::F32ReinterpretI32 => ValType::F32.into(),
                UnOp::F64ReinterpretI64 => ValType::F64.into(),
                UnOp::I32ReinterpretF32 => ValType::I32.into(),
                UnOp::I64ReinterpretF64 => ValType::I64.into(),

                UnOp::I32Extend8S | UnOp::I32Extend16S => ValType::I32.into(),

                UnOp::I64Extend8S | UnOp::I64Extend16S | UnOp::I64Extend32S => ValType::I64.into(),

                UnOp::Load(_, load) => match load {
                    Load::I32(_) => ValType::I32.into(),
                    Load::I64(_) => ValType::I64.into(),
                    Load::F32 => ValType::F32.into(),
                    Load::F64 => ValType::F64.into(),
                },

                UnOp::MemoryGrow(_) => ValType::I32.into(),
            },

            Self::Binary(op, _) => match *op {
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
                | BinOp::I32Rotr => ValType::I32.into(),

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
                | BinOp::I64Rotr => ValType::I64.into(),

                BinOp::F32Add
                | BinOp::F32Sub
                | BinOp::F32Mul
                | BinOp::F32Div
                | BinOp::F32Min
                | BinOp::F32Max
                | BinOp::F32Copysign => ValType::F32.into(),

                BinOp::F64Add
                | BinOp::F64Sub
                | BinOp::F64Mul
                | BinOp::F64Div
                | BinOp::F64Min
                | BinOp::F64Max
                | BinOp::F64Copysign => ValType::F64.into(),

                BinOp::I32Eq
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
                | BinOp::F64Ge => ValType::I32.into(),
            },

            Self::Ternary(op, exprs) => match *op {
                TernOp::Select => exprs[0].ty(),
            },
        }
    }

    pub fn nth_subexpr(&self, n: usize) -> Option<&Self> {
        match self {
            Self::Value(..) | Self::Nullary(_) => None,
            Self::Unary(_, expr) => (n == 0).then_some(&**expr),
            Self::Binary(_, exprs) => exprs.get(n),
            Self::Ternary(_, exprs) => exprs.get(n),
        }
    }

    pub fn subexpr_count(&self) -> usize {
        match self {
            Self::Value(..) | Self::Nullary(_) => 0,
            Self::Unary(..) => 1,
            Self::Binary(..) => 2,
            Self::Ternary(..) => 3,
        }
    }

    pub fn has_side_effect(&self) -> bool {
        match self {
            Self::Value(..) => false,

            Self::Nullary(op) => match *op {
                NulOp::LocalGet(..) => false,
                NulOp::GlobalGet(..) => false,
                NulOp::MemorySize(..) => false,
            },

            Self::Unary(op, _) => match *op {
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
                UnOp::Load(_, _) => false,
                UnOp::MemoryGrow(_) => true,
            },

            Self::Binary(op, _) => match *op {
                BinOp::I32Add | BinOp::I32Sub | BinOp::I32Mul => false,

                BinOp::I32DivS | BinOp::I32DivU | BinOp::I32RemS | BinOp::I32RemU => true,

                BinOp::I32And
                | BinOp::I32Or
                | BinOp::I32Xor
                | BinOp::I32Shl
                | BinOp::I32ShrS
                | BinOp::I32ShrU
                | BinOp::I32Rotl
                | BinOp::I32Rotr
                | BinOp::I64Add
                | BinOp::I64Sub
                | BinOp::I64Mul => false,

                BinOp::I64DivS | BinOp::I64DivU | BinOp::I64RemS | BinOp::I64RemU => true,

                BinOp::I64And
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
            },

            Self::Ternary(op, _) => match *op {
                TernOp::Select => false,
            },
        }
    }
}

impl From<(Value, ValueAttrs)> for Expr {
    fn from((value, attrs): (Value, ValueAttrs)) -> Self {
        Self::Value(value, attrs)
    }
}

#[derive(Debug, Clone)]
pub enum ExprTy {
    Concrete(ValType),
    Local(LocalId),
    Global(GlobalId),
}

impl From<ValType> for ExprTy {
    fn from(val_ty: ValType) -> Self {
        Self::Concrete(val_ty)
    }
}

#[derive(Debug, Default, Clone)]
pub enum Stmt {
    #[default]
    Nop,
    Drop(Expr),
    LocalSet(LocalId, Expr),
    GlobalSet(GlobalId, Expr),
    Store(MemArg, Store, Box<[Expr; 2]>),
    Call(Call),
}

#[derive(Debug, Default, Clone)]
pub enum Terminator {
    #[default]
    Trap,
    Br(BlockId),
    If(Expr, [BlockId; 2]),
    Switch(Expr, Vec<BlockId>),
    Return(Option<Expr>),
}

impl Terminator {
    pub fn successors(&self) -> &[BlockId] {
        match self {
            Self::Trap => &[],
            Self::Br(block_id) => slice::from_ref(block_id),
            Self::If(_, successors) => successors,
            Self::Switch(_, successors) => successors,
            Self::Return(_) => &[],
        }
    }

    pub fn successors_mut(&mut self) -> &mut [BlockId] {
        match self {
            Self::Trap => &mut [],
            Self::Br(block_id) => slice::from_mut(block_id),
            Self::If(_, successors) => successors,
            Self::Switch(_, successors) => successors,
            Self::Return(_) => &mut [],
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct Block {
    pub body: Vec<Stmt>,
    pub term: Terminator,
}

impl Block {
    pub fn successors(&self) -> &[BlockId] {
        self.term.successors()
    }

    pub fn successors_mut(&mut self) -> &mut [BlockId] {
        self.term.successors_mut()
    }
}

#[derive(Debug, Clone)]
pub struct FuncBody {
    pub blocks: SlotMap<BlockId, Block>,
    pub entry: BlockId,
    pub ty: FuncType,
    pub locals: SlotMap<LocalId, ValType>,
    pub params: Vec<LocalId>,
}

impl FuncBody {
    pub fn new(ty: FuncType) -> Self {
        let mut blocks = SlotMap::<BlockId, Block>::with_key();
        let entry = blocks.insert(Default::default());

        Self {
            blocks,
            entry,
            ty,
            locals: Default::default(),
            params: Default::default(),
        }
    }
}
