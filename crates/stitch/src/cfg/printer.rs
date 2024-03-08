use std::fmt::{self, Display};

use crate::ast::expr::format_value;

use super::{
    BinOp, Call, Expr, FuncBody, I32Load, I32Store, I64Load, I64Store, Load, NulOp, Stmt, Store,
    Terminator, TernOp, UnOp,
};

impl Display for Load {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::I32(I32Load::Four) => write!(f, "i32.load"),
            Self::I32(I32Load::TwoS) => write!(f, "i32.load16_s"),
            Self::I32(I32Load::TwoU) => write!(f, "i32.load16_u"),
            Self::I32(I32Load::OneS) => write!(f, "i32.load8_s"),
            Self::I32(I32Load::OneU) => write!(f, "i32.load8_u"),

            Self::I64(I64Load::Eight) => write!(f, "i64.load"),
            Self::I64(I64Load::FourS) => write!(f, "i64.load32_s"),
            Self::I64(I64Load::FourU) => write!(f, "i64.load32_u"),
            Self::I64(I64Load::TwoS) => write!(f, "i64.load16_s"),
            Self::I64(I64Load::TwoU) => write!(f, "i64.load16_u"),
            Self::I64(I64Load::OneS) => write!(f, "i64.load8_s"),
            Self::I64(I64Load::OneU) => write!(f, "i64.load8_u"),

            Self::F32 => write!(f, "f32.load"),
            Self::F64 => write!(f, "f64.load"),
        }
    }
}

impl Display for Store {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::I32(I32Store::Four) => write!(f, "i32.store"),
            Self::I32(I32Store::Two) => write!(f, "i32.store16"),
            Self::I32(I32Store::One) => write!(f, "i32.store8"),

            Self::I64(I64Store::Eight) => write!(f, "i64.store"),
            Self::I64(I64Store::Four) => write!(f, "i64.store32"),
            Self::I64(I64Store::Two) => write!(f, "i64.store16"),
            Self::I64(I64Store::One) => write!(f, "i64.store8"),

            Self::F32 => write!(f, "f32.store"),
            Self::F64 => write!(f, "f64.store"),
        }
    }
}

impl Display for NulOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LocalGet(local_id) => write!(f, "local.get {local_id:?}"),
            Self::GlobalGet(global_id) => write!(f, "global.get {global_id:?}"),
            Self::MemorySize(mem_id) => write!(f, "memory.size {mem_id:?}"),
        }
    }
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

            Self::Load(mem_arg, load) => write!(f, "{load} {mem_arg}"),
            Self::MemoryGrow(mem_id) => write!(f, "memory.grow {mem_id:?}"),
        }
    }
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
        }
    }
}

impl Display for TernOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Select => write!(f, "select"),
        }
    }
}

impl Display for Call {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Direct { func_id, args, .. } => {
                write!(f, "call {func_id:?}")?;

                for arg in args {
                    write!(f, " {arg}")?;
                }
            }

            Self::Indirect {
                ty_id,
                table_id,
                args,
                index,
                ..
            } => {
                write!(f, "call_indirect {ty_id:?} {table_id:?}")?;

                for arg in args {
                    write!(f, " {arg}")?;
                }

                write!(f, " {index}")?;
            }
        }

        Ok(())
    }
}

impl<E> Display for Expr<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Value(value, attrs) => write!(f, "{}", format_value(value, *attrs))?,
            Self::Nullary(op) => write!(f, "({op})")?,
            Self::Unary(op, expr) => write!(f, "({op} {expr})")?,
            Self::Binary(op, exprs) => write!(f, "({op} {} {})", exprs[0], exprs[1])?,
            Self::Ternary(op, exprs) => write!(f, "({op} {} {} {})", exprs[0], exprs[1], exprs[2])?,
        }

        Ok(())
    }
}

impl Display for Stmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Nop => write!(f, "(nop)"),
            Self::Drop(expr) => write!(f, "(drop {expr})"),
            Self::LocalSet(local_id, expr) => write!(f, "(local.set {local_id:?} {expr})"),
            Self::GlobalSet(global_id, expr) => write!(f, "(global.set {global_id:?} {expr})"),
            Self::Store(mem_arg, store, args) => {
                write!(f, "({store} {mem_arg} {} {})", args[0], args[1])
            }
            Self::Call(call) => match call.ret_local_id() {
                Some(ret_local_id) => write!(f, "(local.set {ret_local_id:?} ({call}))"),
                None => write!(f, "({call})"),
            },
        }
    }
}

impl Display for Terminator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Trap => write!(f, "(trap)"),
            Self::Br(block_id) => write!(f, "(br {block_id:?})"),
            Self::If(condition, [then_block_id, else_block_id]) => {
                write!(f, "(if {then_block_id:?} {else_block_id:?} {condition})")
            }
            Self::Switch(index, block_ids) => write!(f, "(switch {index} {block_ids:?})"),
            Self::Return(Some(inner)) => write!(f, "(return {inner})"),
            Self::Return(None) => write!(f, "(return)"),
        }
    }
}

impl Display for FuncBody {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rpo = self.rpo();
        let preds = self.predecessors();

        for (idx, block_id) in rpo.order.into_iter().enumerate() {
            let block = &self.blocks[block_id];

            write!(
                f,
                "{}{block_id:?}: predecessors {:?}, name {:?}\n",
                if idx > 0 { "\n\n" } else { "" },
                &preds[block_id],
                block.name,
            )?;

            for stmt in &block.body {
                write!(f, "  {stmt}\n")?;
            }

            write!(f, "  {}", block.term)?;
        }

        Ok(())
    }
}
