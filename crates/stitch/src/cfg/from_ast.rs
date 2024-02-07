use std::iter;
use std::mem;

use slotmap::{Key, SecondaryMap};

use crate::ast::expr::{
    BinOp as AstBinOp, Block as AstBlock, Expr as AstExpr, NulOp as AstNulOp, TernOp as AstTernOp,
    UnOp as AstUnOp,
};
use crate::ast::func::FuncBody as AstFunc;
use crate::ast::ty::BlockType;
use crate::ast::Module;
use crate::ast::{BlockId as AstBlockId, LocalId as AstLocalId};
use crate::cfg::Call;
use crate::cfg::{Expr, I32Load, I32Store, I64Load, I64Store, Load, Stmt, Store};
use crate::util::try_match;

use super::ExprTy;
use super::{BinOp, Block, BlockId, FuncBody, LocalId, NulOp, Terminator, TernOp, UnOp};

impl super::FuncBody {
    pub fn from_ast(module: &Module, func: &AstFunc) -> Self {
        let translator = Translator {
            module,
            ast: func,
            func: Self::new(func.ty.clone()),
            local_map: Default::default(),
            block_map: Default::default(),
            current_block_id: Default::default(),
            block_results: Default::default(),
        };

        translator.translate()
    }
}

#[derive(Debug, Clone, Copy)]
enum BlockTarget {
    Block(BlockId),
    FuncBody,
}

impl BlockTarget {
    fn block(&self) -> Option<BlockId> {
        try_match!(*self, Self::Block(block_id) => block_id)
    }
}

#[derive(Debug, Clone, Copy)]
enum ExprResult {
    Unreachable,
    DefinedAt(BlockId, usize),
    BlockResult(AstBlockId),
}

struct Translator<'a> {
    module: &'a Module,
    ast: &'a AstFunc,
    func: FuncBody,
    local_map: SecondaryMap<AstLocalId, LocalId>,
    block_map: SecondaryMap<AstBlockId, BlockTarget>,
    current_block_id: BlockId,
    block_results: SecondaryMap<AstBlockId, LocalId>,
}

impl Translator<'_> {
    fn translate(mut self) -> super::FuncBody {
        self.translate_params();
        self.current_block_id = self.func.entry;
        self.block_map
            .insert(self.ast.main_block.id, BlockTarget::FuncBody);

        if self.ast.ty.ret.is_some() {
            self.block_results
                .insert(self.ast.main_block.id, LocalId::default());
        }

        self.translate_block(&self.ast.main_block, None);
        // TODO: self.func.remove_unreachable_blocks();
        // TODO: self.func.merge_blocks();

        self.func
    }

    fn translate_params(&mut self) {
        for &local_id in &self.ast.params {
            self.local_map.insert(
                local_id,
                self.func.locals.insert(self.ast.locals[local_id].clone()),
            );
            self.func.params.push(self.local_map[local_id]);
        }
    }

    fn translate_block(&mut self, block: &AstBlock, exit_block_id: Option<BlockId>) {
        for expr in &block.body {
            if matches!(self.translate_expr(expr), ExprResult::Unreachable) {
                return;
            }
        }

        let expr = self.block_results.contains_key(block.id).then(|| {
            let idx = self
                .current_block()
                .body
                .iter()
                .enumerate()
                .rfind(|(_, stmt)| matches!(stmt, Stmt::Drop(_)))
                .unwrap()
                .0;

            self.use_expr(ExprResult::DefinedAt(self.current_block_id, idx), true)
        });

        self.branch_to(block.id, exit_block_id, expr);
    }

    fn translate_expr(&mut self, expr: &AstExpr) -> ExprResult {
        macro_rules! translate_expr {
            ($expr:expr) => {
                match self.translate_expr($expr) {
                    ExprResult::Unreachable => return ExprResult::Unreachable,
                    result => result,
                }
            };

            (? $expr:expr) => {
                match $expr {
                    Some(expr) => match self.translate_expr(expr) {
                        ExprResult::Unreachable => return ExprResult::Unreachable,
                        result => Some(result),
                    },

                    None => None,
                }
            };
        }

        match expr {
            &AstExpr::Value(value, attrs) => self.push_drop(Expr::Value(value, attrs)),

            AstExpr::Intrinsic(intrinsic) => self.push_drop(Expr::Intrinsic(intrinsic.clone())),

            &AstExpr::Index(id) => self.push_drop(Expr::Index(id)),

            AstExpr::Nullary(op) => match *op {
                AstNulOp::Nop => self.push_stmt(Stmt::Nop),

                AstNulOp::Unreachable => {
                    self.current_block_mut().term = Terminator::Trap;
                    self.switch_to(None);

                    ExprResult::Unreachable
                }

                AstNulOp::LocalGet(local_id) => {
                    let local_id = self.get_local(local_id);

                    self.push_drop(Expr::Nullary(NulOp::LocalGet(local_id)))
                }

                AstNulOp::GlobalGet(global_id) => {
                    self.push_drop(Expr::Nullary(NulOp::GlobalGet(global_id)))
                }

                AstNulOp::MemorySize(mem_id) => {
                    self.push_drop(Expr::Nullary(NulOp::MemorySize(mem_id)))
                }
            },

            AstExpr::Unary(op, inner) => {
                let inner = translate_expr!(inner);
                let inner = self.use_expr(inner, true);

                macro_rules! push {
                    ($( $op:tt )+) => {
                        self.push_drop(Expr::Unary(UnOp::$( $op )+, Box::new(inner)))
                    };
                }

                match *op {
                    AstUnOp::I32Clz => push!(I32Clz),
                    AstUnOp::I32Ctz => push!(I32Ctz),
                    AstUnOp::I32Popcnt => push!(I32Popcnt),
                    AstUnOp::I64Clz => push!(I64Clz),
                    AstUnOp::I64Ctz => push!(I64Ctz),
                    AstUnOp::I64Popcnt => push!(I64Popcnt),
                    AstUnOp::F32Abs => push!(F32Abs),
                    AstUnOp::F32Neg => push!(F32Neg),
                    AstUnOp::F32Sqrt => push!(F32Sqrt),
                    AstUnOp::F32Ceil => push!(F32Ceil),
                    AstUnOp::F32Floor => push!(F32Floor),
                    AstUnOp::F32Trunc => push!(F32Trunc),
                    AstUnOp::F32Nearest => push!(F32Nearest),
                    AstUnOp::F64Abs => push!(F64Abs),
                    AstUnOp::F64Neg => push!(F64Neg),
                    AstUnOp::F64Sqrt => push!(F64Sqrt),
                    AstUnOp::F64Ceil => push!(F64Ceil),
                    AstUnOp::F64Floor => push!(F64Floor),
                    AstUnOp::F64Trunc => push!(F64Trunc),
                    AstUnOp::F64Nearest => push!(F64Nearest),
                    AstUnOp::I32Eqz => push!(I32Eqz),
                    AstUnOp::I64Eqz => push!(I64Eqz),
                    AstUnOp::I32WrapI64 => push!(I32WrapI64),
                    AstUnOp::I64ExtendI32S => push!(I64ExtendI32S),
                    AstUnOp::I64ExtendI32U => push!(I64ExtendI32U),
                    AstUnOp::I32TruncF32S => push!(I32TruncF32S),
                    AstUnOp::I32TruncF32U => push!(I32TruncF32U),
                    AstUnOp::I32TruncF64S => push!(I32TruncF64S),
                    AstUnOp::I32TruncF64U => push!(I32TruncF64U),
                    AstUnOp::I64TruncF32S => push!(I64TruncF32S),
                    AstUnOp::I64TruncF32U => push!(I64TruncF32U),
                    AstUnOp::I64TruncF64S => push!(I64TruncF64S),
                    AstUnOp::I64TruncF64U => push!(I64TruncF64U),
                    AstUnOp::F32DemoteF64 => push!(F32DemoteF64),
                    AstUnOp::F64PromoteF32 => push!(F64PromoteF32),
                    AstUnOp::F32ConvertI32S => push!(F32ConvertI32S),
                    AstUnOp::F32ConvertI32U => push!(F32ConvertI32U),
                    AstUnOp::F32ConvertI64S => push!(F32ConvertI64S),
                    AstUnOp::F32ConvertI64U => push!(F32ConvertI64U),
                    AstUnOp::F64ConvertI32S => push!(F64ConvertI32S),
                    AstUnOp::F64ConvertI32U => push!(F64ConvertI32U),
                    AstUnOp::F64ConvertI64S => push!(F64ConvertI64S),
                    AstUnOp::F64ConvertI64U => push!(F64ConvertI64U),
                    AstUnOp::F32ReinterpretI32 => push!(F32ReinterpretI32),
                    AstUnOp::F64ReinterpretI64 => push!(F64ReinterpretI64),
                    AstUnOp::I32ReinterpretF32 => push!(I32ReinterpretF32),
                    AstUnOp::I64ReinterpretF64 => push!(I64ReinterpretF64),
                    AstUnOp::I32Extend8S => push!(I32Extend8S),
                    AstUnOp::I32Extend16S => push!(I32Extend16S),
                    AstUnOp::I64Extend8S => push!(I64Extend8S),
                    AstUnOp::I64Extend16S => push!(I64Extend16S),
                    AstUnOp::I64Extend32S => push!(I64Extend32S),

                    AstUnOp::LocalSet(local_id) => {
                        let local_id = self.get_local(local_id);

                        self.push_stmt(Stmt::LocalSet(local_id, inner))
                    }

                    AstUnOp::LocalTee(local_id) => {
                        let local_id = self.get_local(local_id);

                        push!(LocalTee(local_id))
                    }

                    AstUnOp::GlobalSet(global_id) => {
                        self.push_stmt(Stmt::GlobalSet(global_id, inner))
                    }

                    AstUnOp::I32Load(mem_arg) => push!(Load(mem_arg, I32Load::Four.into())),
                    AstUnOp::I64Load(mem_arg) => push!(Load(mem_arg, I64Load::Eight.into())),
                    AstUnOp::F32Load(mem_arg) => push!(Load(mem_arg, Load::F32)),
                    AstUnOp::F64Load(mem_arg) => push!(Load(mem_arg, Load::F64)),
                    AstUnOp::I32Load8S(mem_arg) => push!(Load(mem_arg, I32Load::OneS.into())),
                    AstUnOp::I32Load8U(mem_arg) => push!(Load(mem_arg, I32Load::OneU.into())),
                    AstUnOp::I32Load16S(mem_arg) => push!(Load(mem_arg, I32Load::TwoS.into())),
                    AstUnOp::I32Load16U(mem_arg) => push!(Load(mem_arg, I32Load::TwoU.into())),
                    AstUnOp::I64Load8S(mem_arg) => push!(Load(mem_arg, I64Load::OneS.into())),
                    AstUnOp::I64Load8U(mem_arg) => push!(Load(mem_arg, I64Load::OneU.into())),
                    AstUnOp::I64Load16S(mem_arg) => push!(Load(mem_arg, I64Load::TwoS.into())),
                    AstUnOp::I64Load16U(mem_arg) => push!(Load(mem_arg, I64Load::TwoU.into())),
                    AstUnOp::I64Load32S(mem_arg) => push!(Load(mem_arg, I64Load::FourS.into())),
                    AstUnOp::I64Load32U(mem_arg) => push!(Load(mem_arg, I64Load::FourU.into())),

                    AstUnOp::MemoryGrow(mem_id) => push!(MemoryGrow(mem_id)),

                    AstUnOp::Drop => self.push_drop(inner),
                }
            }

            AstExpr::Binary(op, [lhs, rhs]) => {
                let lhs = translate_expr!(lhs);
                let rhs = translate_expr!(rhs);
                let rhs = self.use_expr(rhs, true);
                let lhs = self.use_expr(lhs, true);
                let exprs = Box::new([lhs, rhs]);

                macro_rules! push {
                    ($( $op:tt )+) => {
                        self.push_drop(Expr::Binary(BinOp::$( $op )+, exprs))
                    };
                }

                match *op {
                    AstBinOp::I32Add => push!(I32Add),
                    AstBinOp::I32Sub => push!(I32Sub),
                    AstBinOp::I32Mul => push!(I32Mul),
                    AstBinOp::I32DivS => push!(I32DivS),
                    AstBinOp::I32DivU => push!(I32DivU),
                    AstBinOp::I32RemS => push!(I32RemS),
                    AstBinOp::I32RemU => push!(I32RemU),
                    AstBinOp::I32And => push!(I32And),
                    AstBinOp::I32Or => push!(I32Or),
                    AstBinOp::I32Xor => push!(I32Xor),
                    AstBinOp::I32Shl => push!(I32Shl),
                    AstBinOp::I32ShrS => push!(I32ShrS),
                    AstBinOp::I32ShrU => push!(I32ShrU),
                    AstBinOp::I32Rotl => push!(I32Rotl),
                    AstBinOp::I32Rotr => push!(I32Rotr),
                    AstBinOp::I64Add => push!(I64Add),
                    AstBinOp::I64Sub => push!(I64Sub),
                    AstBinOp::I64Mul => push!(I64Mul),
                    AstBinOp::I64DivS => push!(I64DivS),
                    AstBinOp::I64DivU => push!(I64DivU),
                    AstBinOp::I64RemS => push!(I64RemS),
                    AstBinOp::I64RemU => push!(I64RemU),
                    AstBinOp::I64And => push!(I64And),
                    AstBinOp::I64Or => push!(I64Or),
                    AstBinOp::I64Xor => push!(I64Xor),
                    AstBinOp::I64Shl => push!(I64Shl),
                    AstBinOp::I64ShrS => push!(I64ShrS),
                    AstBinOp::I64ShrU => push!(I64ShrU),
                    AstBinOp::I64Rotl => push!(I64Rotl),
                    AstBinOp::I64Rotr => push!(I64Rotr),
                    AstBinOp::F32Add => push!(F32Add),
                    AstBinOp::F32Sub => push!(F32Sub),
                    AstBinOp::F32Mul => push!(F32Mul),
                    AstBinOp::F32Div => push!(F32Div),
                    AstBinOp::F32Min => push!(F32Min),
                    AstBinOp::F32Max => push!(F32Max),
                    AstBinOp::F32Copysign => push!(F32Copysign),
                    AstBinOp::F64Add => push!(F64Add),
                    AstBinOp::F64Sub => push!(F64Sub),
                    AstBinOp::F64Mul => push!(F64Mul),
                    AstBinOp::F64Div => push!(F64Div),
                    AstBinOp::F64Min => push!(F64Min),
                    AstBinOp::F64Max => push!(F64Max),
                    AstBinOp::F64Copysign => push!(F64Copysign),
                    AstBinOp::I32Eq => push!(I32Eq),
                    AstBinOp::I32Ne => push!(I32Ne),
                    AstBinOp::I32LtS => push!(I32LtS),
                    AstBinOp::I32LtU => push!(I32LtU),
                    AstBinOp::I32GtS => push!(I32GtS),
                    AstBinOp::I32GtU => push!(I32GtU),
                    AstBinOp::I32LeS => push!(I32LeS),
                    AstBinOp::I32LeU => push!(I32LeU),
                    AstBinOp::I32GeS => push!(I32GeS),
                    AstBinOp::I32GeU => push!(I32GeU),
                    AstBinOp::I64Eq => push!(I64Eq),
                    AstBinOp::I64Ne => push!(I64Ne),
                    AstBinOp::I64LtS => push!(I64LtS),
                    AstBinOp::I64LtU => push!(I64LtU),
                    AstBinOp::I64GtS => push!(I64GtS),
                    AstBinOp::I64GtU => push!(I64GtU),
                    AstBinOp::I64LeS => push!(I64LeS),
                    AstBinOp::I64LeU => push!(I64LeU),
                    AstBinOp::I64GeS => push!(I64GeS),
                    AstBinOp::I64GeU => push!(I64GeU),
                    AstBinOp::F32Eq => push!(F32Eq),
                    AstBinOp::F32Ne => push!(F32Ne),
                    AstBinOp::F32Lt => push!(F32Lt),
                    AstBinOp::F32Gt => push!(F32Gt),
                    AstBinOp::F32Le => push!(F32Le),
                    AstBinOp::F32Ge => push!(F32Ge),
                    AstBinOp::F64Eq => push!(F64Eq),
                    AstBinOp::F64Ne => push!(F64Ne),
                    AstBinOp::F64Lt => push!(F64Lt),
                    AstBinOp::F64Gt => push!(F64Gt),
                    AstBinOp::F64Le => push!(F64Le),
                    AstBinOp::F64Ge => push!(F64Ge),

                    AstBinOp::I32Store(mem_arg) => {
                        self.push_stmt(Stmt::Store(mem_arg, I32Store::Four.into()))
                    }
                    AstBinOp::I64Store(mem_arg) => {
                        self.push_stmt(Stmt::Store(mem_arg, I64Store::Eight.into()))
                    }
                    AstBinOp::F32Store(mem_arg) => self.push_stmt(Stmt::Store(mem_arg, Store::F32)),
                    AstBinOp::F64Store(mem_arg) => self.push_stmt(Stmt::Store(mem_arg, Store::F64)),
                    AstBinOp::I32Store8(mem_arg) => {
                        self.push_stmt(Stmt::Store(mem_arg, I32Store::One.into()))
                    }
                    AstBinOp::I32Store16(mem_arg) => {
                        self.push_stmt(Stmt::Store(mem_arg, I32Store::Two.into()))
                    }
                    AstBinOp::I64Store8(mem_arg) => {
                        self.push_stmt(Stmt::Store(mem_arg, I64Store::One.into()))
                    }
                    AstBinOp::I64Store16(mem_arg) => {
                        self.push_stmt(Stmt::Store(mem_arg, I64Store::Two.into()))
                    }
                    AstBinOp::I64Store32(mem_arg) => {
                        self.push_stmt(Stmt::Store(mem_arg, I64Store::Four.into()))
                    }
                }
            }

            AstExpr::Ternary(op, [e0, e1, e2]) => match *op {
                AstTernOp::Select => {
                    let e0 = translate_expr!(e0);
                    let e1 = translate_expr!(e1);
                    let e2 = translate_expr!(e2);

                    let e2 = self.use_expr(e2, true);
                    let e1 = self.use_expr(e1, true);
                    let e0 = self.use_expr(e0, true);

                    let exprs = Box::new([e0, e1, e2]);

                    self.push_drop(Expr::Ternary(TernOp::Select, exprs))
                }
            },

            AstExpr::Block(block_ty, block) => {
                self.reserve_block_result(block_ty, &[block.id]);
                let exit_block_id = self.func.blocks.insert(Default::default());
                self.block_map
                    .insert(block.id, BlockTarget::Block(exit_block_id));
                self.translate_block(block, None);
                self.switch_to(Some(exit_block_id));

                ExprResult::BlockResult(block.id)
            }

            AstExpr::Loop(block_ty, block) => {
                self.reserve_block_result(block_ty, &[block.id]);
                let loop_block_id = self.func.blocks.insert(Default::default());
                let exit_block_id = self.func.blocks.insert(Default::default());
                self.block_map
                    .insert(block.id, BlockTarget::Block(loop_block_id));
                self.current_block_mut().term = Terminator::Br(loop_block_id);
                self.switch_to(Some(loop_block_id));
                self.translate_block(block, Some(exit_block_id));
                self.switch_to(Some(exit_block_id));

                ExprResult::BlockResult(block.id)
            }

            AstExpr::If(block_ty, condition, then_block, else_block) => {
                let condition = translate_expr!(condition);
                let condition = self.use_expr(condition, true);

                self.reserve_block_result(block_ty, &[then_block.id, else_block.id]);
                let then_block_id = self.func.blocks.insert(Default::default());
                let else_block_id = self.func.blocks.insert(Default::default());
                let exit_block_id = self.func.blocks.insert(Default::default());
                self.block_map
                    .insert(then_block.id, BlockTarget::Block(exit_block_id));
                self.block_map
                    .insert(else_block.id, BlockTarget::Block(exit_block_id));
                self.current_block_mut().term =
                    Terminator::If(condition, [then_block_id, else_block_id]);
                self.switch_to(Some(then_block_id));
                self.translate_block(then_block, None);
                self.switch_to(Some(else_block_id));
                self.translate_block(else_block, None);
                self.switch_to(Some(exit_block_id));

                ExprResult::BlockResult(then_block.id)
            }

            AstExpr::Br(block_id, inner) => {
                let expr = translate_expr!(?inner);
                let expr = expr.map(|expr| self.use_expr(expr, true));

                self.branch_to(*block_id, None, expr);
                self.switch_to(None);

                ExprResult::Unreachable
            }

            AstExpr::BrIf(block_id, inner, condition) => {
                let expr_result = translate_expr!(?inner);
                let condition = translate_expr!(condition);
                let condition = self.use_expr(condition, true);
                let expr = expr_result
                    .clone()
                    .map(|expr_result| self.use_expr(expr_result, false));

                let then_block_id = self.func.blocks.insert(Default::default());
                let else_block_id = self.func.blocks.insert(Default::default());
                self.current_block_mut().term =
                    Terminator::If(condition, [then_block_id, else_block_id]);
                self.switch_to(Some(then_block_id));
                self.branch_to(*block_id, None, expr);
                self.switch_to(Some(else_block_id));

                match expr_result {
                    Some(expr_result) => expr_result,
                    None => ExprResult::BlockResult(Default::default()),
                }
            }

            AstExpr::BrTable(block_ids, default_block_id, inner, index) => {
                let expr = translate_expr!(?inner);
                let index = translate_expr!(index);
                let index = self.use_expr(index, true);

                let all_block_ids = || {
                    block_ids
                        .iter()
                        .chain(iter::once(default_block_id))
                        .copied()
                };

                let has_return = all_block_ids()
                    .find(|&block_id| matches!(self.block_map[block_id], BlockTarget::FuncBody))
                    .is_some();

                if expr.is_some() || has_return {
                    let expr = expr.map(|expr| self.use_expr(expr, false));

                    let trampolines: Vec<_> =
                        iter::repeat_with(|| self.func.blocks.insert(Default::default()))
                            .take(block_ids.len() + 1)
                            .collect();
                    let (&default_block_id, block_ids) = trampolines.split_last().unwrap();
                    self.current_block_mut().term =
                        Terminator::Switch(index, block_ids.to_owned(), default_block_id);

                    for (&trampoline_block_id, target_block_id) in
                        trampolines.iter().zip(all_block_ids())
                    {
                        self.switch_to(Some(trampoline_block_id));
                        self.branch_to(target_block_id, None, expr.clone());
                    }
                } else {
                    self.current_block_mut().term = Terminator::Switch(
                        index,
                        block_ids
                            .iter()
                            .map(|&block_id| self.block_map[block_id].block().unwrap())
                            .collect(),
                        self.block_map[*default_block_id].block().unwrap(),
                    );
                }

                self.switch_to(None);

                ExprResult::Unreachable
            }

            AstExpr::Return(inner) => {
                let expr = translate_expr!(?inner);
                let expr = expr.map(|expr| self.use_expr(expr, true));
                self.current_block_mut().term = Terminator::Return(expr);
                self.switch_to(None);

                ExprResult::Unreachable
            }

            AstExpr::Call(func_id, args) => {
                let is_stmt = self.module.funcs[*func_id].ty().ret.is_some();

                let mut arg_results = Vec::with_capacity(args.len());

                for arg in args {
                    arg_results.push(translate_expr!(arg));
                }

                let mut args = arg_results
                    .into_iter()
                    .rev()
                    .map(|arg| self.use_expr(arg, true))
                    .collect::<Vec<_>>();
                args.reverse();

                let call = Call::Direct {
                    func_id: *func_id,
                    args,
                };

                if is_stmt {
                    self.push_stmt(Stmt::Call(call))
                } else {
                    self.push_drop(Expr::Call(call))
                }
            }

            AstExpr::CallIndirect(ty_id, table_id, args, index) => {
                let is_stmt = self.module.types[*ty_id].as_func().ret.is_some();

                let mut arg_results = Vec::with_capacity(args.len());

                for arg in args {
                    arg_results.push(translate_expr!(arg));
                }

                let index = translate_expr!(index);
                let index = self.use_expr(index, true);

                let mut args = arg_results
                    .into_iter()
                    .rev()
                    .map(|arg| self.use_expr(arg, true))
                    .collect::<Vec<_>>();
                args.reverse();

                let call = Call::Indirect {
                    ty_id: *ty_id,
                    table_id: *table_id,
                    args,
                    index: Box::new(index),
                };

                if is_stmt {
                    self.push_stmt(Stmt::Call(call))
                } else {
                    self.push_drop(Expr::Call(call))
                }
            }
        }
    }

    fn current_block(&self) -> &Block {
        &self.func.blocks[self.current_block_id]
    }

    fn current_block_mut(&mut self) -> &mut Block {
        &mut self.func.blocks[self.current_block_id]
    }

    fn get_local(&mut self, local_id: AstLocalId) -> LocalId {
        *self
            .local_map
            .entry(local_id)
            .unwrap()
            .or_insert_with(|| self.func.locals.insert(self.ast.locals[local_id].clone()))
    }

    fn use_expr(&mut self, expr_result: ExprResult, take_ownership: bool) -> Expr {
        match expr_result {
            ExprResult::Unreachable => panic!("using the result of a divergent expression"),

            ExprResult::DefinedAt(block_id, idx) => {
                match mem::take(&mut self.func.blocks[block_id].body[idx]) {
                    Stmt::Drop(expr)
                        if take_ownership
                            && self.current_block_id == block_id
                            && self
                                .current_block()
                                .body
                                .get(idx + 1..)
                                .unwrap()
                                .iter()
                                .all(|stmt| matches!(stmt, Stmt::Nop)) =>
                    {
                        expr
                    }

                    Stmt::Drop(expr) => {
                        let local_id = self.func.locals.insert(match expr.ty() {
                            ExprTy::Concrete(val_ty) => val_ty,
                            ExprTy::Local(local_id) => self.func.locals[local_id].clone(),
                            ExprTy::Global(global_id) => {
                                self.module.globals[global_id].ty.val_type.clone()
                            }
                            ExprTy::Call(func_id) => {
                                self.module.funcs[func_id].ty().ret.clone().unwrap()
                            }
                            ExprTy::CallIndirect(ty_id) => {
                                self.module.types[ty_id].as_func().ret.clone().unwrap()
                            }
                        });

                        self.func.blocks[block_id].body[idx] = Stmt::LocalSet(local_id, expr);

                        Expr::Nullary(NulOp::LocalGet(local_id))
                    }

                    Stmt::LocalSet(local_id, expr) => {
                        self.func.blocks[block_id].body[idx] = Stmt::LocalSet(local_id, expr);

                        Expr::Nullary(NulOp::LocalGet(local_id))
                    }

                    stmt => panic!("cannot use {stmt:?} as an expression"),
                }
            }

            ExprResult::BlockResult(block_id) => {
                Expr::Nullary(NulOp::LocalGet(self.block_results[block_id]))
            }
        }
    }

    fn switch_to(&mut self, block_id: Option<BlockId>) -> BlockId {
        let block_id = block_id.unwrap_or_else(|| self.func.blocks.insert(Default::default()));
        self.current_block_id = block_id;

        block_id
    }

    fn push_stmt(&mut self, stmt: Stmt) -> ExprResult {
        let block = self.current_block_mut();
        let idx = block.body.len();
        block.body.push(stmt);

        ExprResult::DefinedAt(self.current_block_id, idx)
    }

    fn push_drop(&mut self, expr: Expr) -> ExprResult {
        self.push_stmt(Stmt::Drop(expr))
    }

    fn reserve_block_result(&mut self, block_ty: &BlockType, block_ids: &[AstBlockId]) {
        if let Some(val_ty) = block_ty.to_val_ty() {
            let local_id = self.func.locals.insert(val_ty);

            for &block_id in block_ids {
                self.block_results.insert(block_id, local_id);
            }
        }
    }

    fn branch_to(
        &mut self,
        ast_block_id: AstBlockId,
        exit_block_id: Option<BlockId>,
        expr: Option<Expr>,
    ) {
        let target = match exit_block_id {
            Some(exit_block_id) => BlockTarget::Block(exit_block_id),
            None => self.block_map[ast_block_id],
        };

        match target {
            BlockTarget::FuncBody => {
                self.current_block_mut().term = Terminator::Return(expr);
            }

            BlockTarget::Block(block_id) => {
                if let Some(expr) = expr {
                    let local_id = self.block_results[ast_block_id];
                    assert!(local_id.is_null());

                    self.push_stmt(Stmt::LocalSet(local_id, expr));
                }

                self.current_block_mut().term = Terminator::Br(block_id);
            }
        }
    }
}
