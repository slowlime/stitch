use std::array;
use std::iter;
use std::mem;

use slotmap::{Key, SecondaryMap};

use crate::ast::expr::{
    BinOp as AstBinOp, Block as AstBlock, Expr as AstExpr, NulOp as AstNulOp, TernOp as AstTernOp,
    UnOp as AstUnOp,
};
use crate::ast::func::FuncBody as AstFunc;
use crate::ast::ty::BlockType;
use crate::ast::FuncId;
use crate::ast::Module;
use crate::ast::TableId;
use crate::ast::TypeId;
use crate::ast::{BlockId as AstBlockId, LocalId as AstLocalId};
use crate::cfg::Call;
use crate::cfg::{Expr, I32Load, I32Store, I64Load, I64Store, Load, Stmt, Store};
use crate::util::try_match;

use super::ExprTy;
use super::{BinOp, Block, BlockId, FuncBody, LocalId, NulOp, Terminator, TernOp, UnOp};

impl FuncBody {
    pub fn from_ast(module: &Module, func: &AstFunc) -> Self {
        let translator = Translator {
            module,
            ast: func,
            func: Self::new(func.ty.clone()),
            local_map: Default::default(),
            block_map: Default::default(),
            current_block_id: Default::default(),
            block_results: Default::default(),
            tasks: vec![],
            task_results: vec![],
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

impl ExprResult {
    fn is_unreachable(&self) -> bool {
        matches!(self, Self::Unreachable)
    }
}

#[derive(Debug)]
enum Task<'a> {
    TranslateBlock {
        block: &'a AstBlock,
        exit_block_id: Option<BlockId>,
        next_child: usize,
        tail_result: Option<ExprResult>,
    },

    TranslateExpr {
        expr: &'a AstExpr,
    },

    TranslateExprUnary {
        op: AstUnOp,
    },

    TranslateExprBinary {
        op: AstBinOp,
        exprs: &'a [Box<AstExpr>; 2],
        next_subexpr: usize,
    },

    TranslateExprTernary {
        op: AstTernOp,
        exprs: &'a [Box<AstExpr>; 3],
        next_subexpr: usize,
    },

    TranslateExprBlockEnd {
        exit_block_id: BlockId,
        result_block_id: AstBlockId,
    },

    TranslateExprIfThen {
        block_ty: &'a BlockType,
        then_block: &'a AstBlock,
        else_block: &'a AstBlock,
    },

    TranslateExprIfElse {
        else_block: &'a AstBlock,
        else_block_id: BlockId,
        exit_block_id: BlockId,
        result_block_id: AstBlockId,
    },

    TranslateExprBr {
        target_block_id: AstBlockId,
        has_inner: bool,
    },

    TranslateExprBrIf {
        target_block_id: AstBlockId,
        condition: Option<&'a AstExpr>,
        has_inner: bool,
    },

    TranslateExprBrTable {
        block_ids: &'a [AstBlockId],
        default_block_id: AstBlockId,
        index: Option<&'a AstExpr>,
        has_inner: bool,
    },

    TranslateExprReturn,

    TranslateExprCall {
        func_id: FuncId,
        args: &'a [AstExpr],
        next_subexpr: usize,
    },

    TranslateExprCallIndirect {
        ty_id: TypeId,
        table_id: TableId,
        args: &'a [AstExpr],
        index: &'a AstExpr,
        next_subexpr: usize,
    },
}

struct Translator<'a> {
    module: &'a Module,
    ast: &'a AstFunc,
    func: FuncBody,
    local_map: SecondaryMap<AstLocalId, LocalId>,
    block_map: SecondaryMap<AstBlockId, BlockTarget>,
    current_block_id: BlockId,
    block_results: SecondaryMap<AstBlockId, LocalId>,
    tasks: Vec<Task<'a>>,
    task_results: Vec<ExprResult>,
}

impl<'a> Translator<'a> {
    fn translate(mut self) -> super::FuncBody {
        self.translate_params();
        self.current_block_id = self.func.entry;
        self.block_map
            .insert(self.ast.main_block.id, BlockTarget::FuncBody);

        if self.ast.ty.ret.is_some() {
            self.block_results
                .insert(self.ast.main_block.id, LocalId::default());
        }

        self.tasks.push(Task::TranslateBlock {
            block: &self.ast.main_block,
            exit_block_id: None,
            next_child: 0,
            tail_result: None,
        });
        self.process_tasks();
        self.remove_nops();
        self.func.remove_unreachable_blocks();
        self.func.merge_blocks();

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

    fn process_tasks(&mut self) {
        while let Some(task) = self.tasks.pop() {
            match task {
                Task::TranslateBlock {
                    block,
                    exit_block_id,
                    next_child,
                    tail_result,
                } => self.do_translate_block(block, exit_block_id, next_child, tail_result),

                Task::TranslateExpr { expr } => self.do_translate_expr(expr),

                Task::TranslateExprUnary { op } => self.do_translate_expr_unary(op),

                Task::TranslateExprBinary {
                    op,
                    exprs,
                    next_subexpr,
                } => self.do_translate_expr_binary(op, exprs, next_subexpr),

                Task::TranslateExprTernary {
                    op,
                    exprs,
                    next_subexpr,
                } => self.do_translate_expr_ternary(op, exprs, next_subexpr),

                Task::TranslateExprBlockEnd {
                    exit_block_id,
                    result_block_id,
                } => self.do_translate_expr_block_end(exit_block_id, result_block_id),

                Task::TranslateExprIfThen {
                    block_ty,
                    then_block,
                    else_block,
                } => self.do_translate_expr_if_then(block_ty, then_block, else_block),

                Task::TranslateExprIfElse {
                    else_block,
                    else_block_id,
                    exit_block_id,
                    result_block_id,
                } => self.do_translate_expr_if_else(
                    else_block,
                    else_block_id,
                    exit_block_id,
                    result_block_id,
                ),

                Task::TranslateExprBr {
                    target_block_id,
                    has_inner,
                } => self.do_translate_expr_br(target_block_id, has_inner),

                Task::TranslateExprBrIf {
                    target_block_id,
                    condition,
                    has_inner,
                } => self.do_translate_expr_br_if(target_block_id, condition, has_inner),

                Task::TranslateExprBrTable {
                    block_ids,
                    default_block_id,
                    index,
                    has_inner,
                } => self.do_translate_expr_br_table(block_ids, default_block_id, index, has_inner),

                Task::TranslateExprReturn => self.do_translate_expr_return(true),

                Task::TranslateExprCall {
                    func_id,
                    args,
                    next_subexpr,
                } => self.do_translate_expr_call(func_id, args, next_subexpr),

                Task::TranslateExprCallIndirect {
                    ty_id,
                    table_id,
                    args,
                    index,
                    next_subexpr,
                } => {
                    self.do_translate_expr_call_indirect(ty_id, table_id, args, index, next_subexpr)
                }
            }
        }
    }

    fn is_task_result_unreachable(&self) -> bool {
        self.task_results.last().unwrap().is_unreachable()
    }

    fn do_translate_block(
        &mut self,
        block: &'a AstBlock,
        exit_block_id: Option<BlockId>,
        next_child: usize,
        mut tail_result: Option<ExprResult>,
    ) {
        if next_child > 0 {
            let last_result = self.task_results.pop().unwrap();

            let is_tail = match last_result {
                ExprResult::Unreachable => return,
                ExprResult::DefinedAt(block_id, idx) => {
                    matches!(self.func.blocks[block_id].body[idx], Stmt::Drop(_))
                }
                ExprResult::BlockResult(block_id) => self.block_results.contains_key(block_id),
            };

            if is_tail {
                tail_result = Some(last_result);
            }
        }

        if let Some(expr) = block.body.get(next_child) {
            self.tasks.push(Task::TranslateBlock {
                block,
                exit_block_id,
                next_child: next_child + 1,
                tail_result,
            });
            self.tasks.push(Task::TranslateExpr { expr });
        } else {
            let contains_key = self.block_results.contains_key(block.id);

            if contains_key && tail_result.is_none() {
                // no block subexpression supplies a usable value:
                // unless there's a bug, this means the block exit is unreachable, so emit a trap
                self.current_block_mut().term = Terminator::Trap;
            } else {
                let expr = contains_key.then(|| self.use_expr(tail_result.unwrap(), true));

                self.branch_to(block.id, exit_block_id, expr);
            }
        }
    }

    fn do_translate_expr(&mut self, expr: &'a AstExpr) {
        let result = match expr {
            &AstExpr::Value(value, attrs) => self.push_drop(Expr::Value(value, attrs)),

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
                self.tasks.push(Task::TranslateExprUnary { op: *op });
                self.tasks.push(Task::TranslateExpr { expr: inner });

                return;
            }

            AstExpr::Binary(op, exprs) => {
                self.tasks.push(Task::TranslateExprBinary {
                    op: *op,
                    exprs,
                    next_subexpr: 0,
                });

                return;
            }

            AstExpr::Ternary(op, exprs) => {
                self.tasks.push(Task::TranslateExprTernary {
                    op: *op,
                    exprs,
                    next_subexpr: 0,
                });

                return;
            }

            AstExpr::Block(block_ty, block) => {
                self.reserve_block_result(block_ty, &[block.id]);
                let exit_block_id = self.func.blocks.insert(Default::default());
                self.block_map
                    .insert(block.id, BlockTarget::Block(exit_block_id));
                self.tasks.push(Task::TranslateExprBlockEnd {
                    exit_block_id,
                    result_block_id: block.id,
                });
                self.tasks.push(Task::TranslateBlock {
                    block,
                    exit_block_id: None,
                    next_child: 0,
                    tail_result: None,
                });

                return;
            }

            AstExpr::Loop(block_ty, block) => {
                self.reserve_block_result(block_ty, &[block.id]);
                let loop_block_id = self.func.blocks.insert(Default::default());
                let exit_block_id = self.func.blocks.insert(Default::default());
                self.block_map
                    .insert(block.id, BlockTarget::Block(loop_block_id));
                self.current_block_mut().term = Terminator::Br(loop_block_id);
                self.switch_to(Some(loop_block_id));
                self.tasks.push(Task::TranslateExprBlockEnd {
                    exit_block_id,
                    result_block_id: block.id,
                });
                self.tasks.push(Task::TranslateBlock {
                    block,
                    exit_block_id: Some(exit_block_id),
                    next_child: 0,
                    tail_result: None,
                });

                return;
            }

            AstExpr::If(block_ty, condition, then_block, else_block) => {
                self.tasks.push(Task::TranslateExprIfThen {
                    block_ty,
                    then_block,
                    else_block,
                });
                self.tasks.push(Task::TranslateExpr { expr: condition });

                return;
            }

            AstExpr::Br(block_id, inner) => {
                self.tasks.push(Task::TranslateExprBr {
                    target_block_id: *block_id,
                    has_inner: inner.is_some(),
                });

                if let Some(inner) = inner {
                    self.tasks.push(Task::TranslateExpr { expr: inner });
                }

                return;
            }

            AstExpr::BrIf(block_id, inner, condition) => {
                self.tasks.push(Task::TranslateExprBrIf {
                    target_block_id: *block_id,
                    condition: Some(condition),
                    has_inner: inner.is_some(),
                });

                if let Some(inner) = inner {
                    self.tasks.push(Task::TranslateExpr { expr: inner });
                }

                return;
            }

            AstExpr::BrTable(block_ids, default_block_id, inner, index) => {
                self.tasks.push(Task::TranslateExprBrTable {
                    block_ids,
                    default_block_id: *default_block_id,
                    index: Some(index),
                    has_inner: inner.is_some(),
                });

                if let Some(inner) = inner {
                    self.tasks.push(Task::TranslateExpr { expr: inner });
                }

                return;
            }

            AstExpr::Return(inner) => {
                if let Some(inner) = inner {
                    self.tasks.push(Task::TranslateExprReturn);
                    self.tasks.push(Task::TranslateExpr { expr: inner });
                } else {
                    self.do_translate_expr_return(false);
                }

                return;
            }

            AstExpr::Call(func_id, args) => {
                self.tasks.push(Task::TranslateExprCall {
                    func_id: *func_id,
                    args,
                    next_subexpr: 0,
                });

                return;
            }

            AstExpr::CallIndirect(ty_id, table_id, args, index) => {
                self.tasks.push(Task::TranslateExprCallIndirect {
                    ty_id: *ty_id,
                    table_id: *table_id,
                    args,
                    index,
                    next_subexpr: 0,
                });

                return;
            }
        };

        self.task_results.push(result);
    }

    fn do_translate_expr_unary(&mut self, op: AstUnOp) {
        if self.is_task_result_unreachable() {
            return;
        }

        let inner = self.task_results.pop().unwrap();
        let inner = self.use_expr(inner, true);

        macro_rules! push {
            ($( $op:tt )+) => {
                self.push_drop(Expr::Unary(UnOp::$( $op )+, Box::new(inner)))
            };
        }

        let result = match op {
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

            AstUnOp::LocalSet(local_id) | AstUnOp::LocalTee(local_id) => {
                let local_id = self.get_local(local_id);

                self.push_stmt(Stmt::LocalSet(local_id, inner))
            }

            AstUnOp::GlobalSet(global_id) => self.push_stmt(Stmt::GlobalSet(global_id, inner)),

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
        };

        self.task_results.push(result);
    }

    fn do_translate_expr_binary(
        &mut self,
        op: AstBinOp,
        exprs: &'a [Box<AstExpr>; 2],
        next_subexpr: usize,
    ) {
        if (1..=exprs.len()).contains(&next_subexpr) && self.is_task_result_unreachable() {
            self.task_results.splice(
                (self.task_results.len() - next_subexpr)..,
                [ExprResult::Unreachable],
            );
        } else if next_subexpr < exprs.len() {
            self.tasks.push(Task::TranslateExprBinary {
                op,
                exprs,
                next_subexpr: next_subexpr + 1,
            });
            self.tasks.push(Task::TranslateExpr {
                expr: &exprs[next_subexpr],
            });
        } else {
            let mut exprs = Box::new(array::from_fn(|_| {
                let expr = self.task_results.pop().unwrap();
                self.use_expr(expr, true)
            }));
            exprs.reverse();

            macro_rules! push {
                ($( $op:tt )+) => {
                    self.push_drop(Expr::Binary(BinOp::$( $op )+, exprs))
                };
            }

            let result = match op {
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
                    self.push_stmt(Stmt::Store(mem_arg, I32Store::Four.into(), exprs))
                }
                AstBinOp::I64Store(mem_arg) => {
                    self.push_stmt(Stmt::Store(mem_arg, I64Store::Eight.into(), exprs))
                }
                AstBinOp::F32Store(mem_arg) => {
                    self.push_stmt(Stmt::Store(mem_arg, Store::F32, exprs))
                }
                AstBinOp::F64Store(mem_arg) => {
                    self.push_stmt(Stmt::Store(mem_arg, Store::F64, exprs))
                }
                AstBinOp::I32Store8(mem_arg) => {
                    self.push_stmt(Stmt::Store(mem_arg, I32Store::One.into(), exprs))
                }
                AstBinOp::I32Store16(mem_arg) => {
                    self.push_stmt(Stmt::Store(mem_arg, I32Store::Two.into(), exprs))
                }
                AstBinOp::I64Store8(mem_arg) => {
                    self.push_stmt(Stmt::Store(mem_arg, I64Store::One.into(), exprs))
                }
                AstBinOp::I64Store16(mem_arg) => {
                    self.push_stmt(Stmt::Store(mem_arg, I64Store::Two.into(), exprs))
                }
                AstBinOp::I64Store32(mem_arg) => {
                    self.push_stmt(Stmt::Store(mem_arg, I64Store::Four.into(), exprs))
                }
            };

            self.task_results.push(result);
        }
    }

    fn do_translate_expr_ternary(
        &mut self,
        op: AstTernOp,
        exprs: &'a [Box<AstExpr>; 3],
        next_subexpr: usize,
    ) {
        if (1..=exprs.len()).contains(&next_subexpr) && self.is_task_result_unreachable() {
            self.task_results.splice(
                (self.task_results.len() - next_subexpr)..,
                [ExprResult::Unreachable],
            );
        } else if next_subexpr < exprs.len() {
            self.tasks.push(Task::TranslateExprTernary {
                op,
                exprs,
                next_subexpr: next_subexpr + 1,
            });
            self.tasks.push(Task::TranslateExpr {
                expr: &exprs[next_subexpr],
            });
        } else {
            let mut exprs = Box::new(array::from_fn(|_| {
                let expr = self.task_results.pop().unwrap();
                self.use_expr(expr, true)
            }));
            exprs.reverse();

            let result = match op {
                AstTernOp::Select => self.push_drop(Expr::Ternary(TernOp::Select, exprs)),
            };

            self.task_results.push(result);
        }
    }

    fn do_translate_expr_block_end(&mut self, exit_block_id: BlockId, result_block_id: AstBlockId) {
        self.switch_to(Some(exit_block_id));
        self.task_results
            .push(ExprResult::BlockResult(result_block_id));
    }

    fn do_translate_expr_if_then(
        &mut self,
        block_ty: &'a BlockType,
        then_block: &'a AstBlock,
        else_block: &'a AstBlock,
    ) {
        if self.is_task_result_unreachable() {
            return;
        }

        let condition = self.task_results.pop().unwrap();
        let condition = self.use_expr(condition, true);

        self.reserve_block_result(block_ty, &[then_block.id, else_block.id]);
        let then_block_id = self.func.blocks.insert(Default::default());
        let else_block_id = self.func.blocks.insert(Default::default());
        let exit_block_id = self.func.blocks.insert(Default::default());
        self.block_map
            .insert(then_block.id, BlockTarget::Block(exit_block_id));
        self.block_map
            .insert(else_block.id, BlockTarget::Block(exit_block_id));
        self.current_block_mut().term = Terminator::If(condition, [then_block_id, else_block_id]);
        self.switch_to(Some(then_block_id));
        self.tasks.push(Task::TranslateExprIfElse {
            else_block,
            else_block_id,
            exit_block_id,
            result_block_id: then_block.id,
        });
        self.tasks.push(Task::TranslateBlock {
            block: then_block,
            exit_block_id: None,
            next_child: 0,
            tail_result: None,
        });
    }

    fn do_translate_expr_if_else(
        &mut self,
        else_block: &'a AstBlock,
        else_block_id: BlockId,
        exit_block_id: BlockId,
        result_block_id: AstBlockId,
    ) {
        self.switch_to(Some(else_block_id));
        self.tasks.push(Task::TranslateExprBlockEnd {
            exit_block_id,
            result_block_id,
        });
        self.tasks.push(Task::TranslateBlock {
            block: else_block,
            exit_block_id: None,
            next_child: 0,
            tail_result: None,
        });
    }

    fn do_translate_expr_br(&mut self, target_block_id: AstBlockId, has_inner: bool) {
        if has_inner && self.is_task_result_unreachable() {
            return;
        }

        let expr = has_inner
            .then(|| self.task_results.pop().unwrap())
            .map(|expr| self.use_expr(expr, true));
        self.branch_to(target_block_id, None, expr);
        self.switch_to(None);
        self.task_results.push(ExprResult::Unreachable);
    }

    fn do_translate_expr_br_if(
        &mut self,
        target_block_id: AstBlockId,
        condition: Option<&'a AstExpr>,
        has_inner: bool,
    ) {
        if let Some(condition) = condition {
            if has_inner && self.is_task_result_unreachable() {
                return;
            }

            self.tasks.push(Task::TranslateExprBrIf {
                target_block_id,
                condition: None,
                has_inner,
            });
            self.tasks.push(Task::TranslateExpr { expr: condition });
        } else {
            if self.is_task_result_unreachable() {
                if has_inner {
                    self.task_results.remove(self.task_results.len() - 2);
                }

                return;
            }

            let condition = self.task_results.pop().unwrap();
            let condition = self.use_expr(condition, true);
            let expr_result = has_inner.then(|| self.task_results.pop().unwrap());
            let expr = expr_result
                .clone()
                .map(|expr_result| self.use_expr(expr_result, false));

            let then_block_id = self.func.blocks.insert(Default::default());
            let else_block_id = self.func.blocks.insert(Default::default());
            self.current_block_mut().term =
                Terminator::If(condition, [then_block_id, else_block_id]);
            self.switch_to(Some(then_block_id));
            self.branch_to(target_block_id, None, expr);
            self.switch_to(Some(else_block_id));

            self.task_results.push(match expr_result {
                Some(expr_result) => expr_result,
                None => ExprResult::BlockResult(Default::default()),
            });
        }
    }

    fn do_translate_expr_br_table(
        &mut self,
        block_ids: &'a [AstBlockId],
        default_block_id: AstBlockId,
        index: Option<&'a AstExpr>,
        has_inner: bool,
    ) {
        if let Some(index) = index {
            if has_inner && self.is_task_result_unreachable() {
                return;
            }

            self.tasks.push(Task::TranslateExprBrTable {
                block_ids,
                default_block_id,
                index: None,
                has_inner,
            });
            self.tasks.push(Task::TranslateExpr { expr: index });
        } else {
            if self.is_task_result_unreachable() {
                if has_inner {
                    self.task_results.remove(self.task_results.len() - 2);
                }

                return;
            }

            let index = self.task_results.pop().unwrap();
            let index = self.use_expr(index, true);

            let all_block_ids = || {
                block_ids
                    .iter()
                    .copied()
                    .chain(iter::once(default_block_id))
            };

            let has_return = all_block_ids()
                .find(|&block_id| matches!(self.block_map[block_id], BlockTarget::FuncBody))
                .is_some();

            if has_inner || has_return {
                let expr = has_inner
                    .then(|| self.task_results.pop().unwrap())
                    .map(|expr| self.use_expr(expr, false));

                let trampolines: Vec<_> =
                    iter::repeat_with(|| self.func.blocks.insert(Default::default()))
                        .take(block_ids.len() + 1)
                        .collect();
                self.current_block_mut().term = Terminator::Switch(index, trampolines.clone());

                for (&trampoline_block_id, target_block_id) in
                    trampolines.iter().zip(all_block_ids())
                {
                    self.switch_to(Some(trampoline_block_id));
                    self.branch_to(target_block_id, None, expr.clone());
                }
            } else {
                self.current_block_mut().term = Terminator::Switch(
                    index,
                    all_block_ids()
                        .map(|block_id| self.block_map[block_id].block().unwrap())
                        .collect(),
                );
            }

            self.switch_to(None);
            self.task_results.push(ExprResult::Unreachable);
        }
    }

    fn do_translate_expr_return(&mut self, has_inner: bool) {
        let expr = has_inner
            .then(|| self.task_results.pop().unwrap())
            .map(|expr| self.use_expr(expr, true));
        self.current_block_mut().term = Terminator::Return(expr);
        self.switch_to(None);
        self.task_results.push(ExprResult::Unreachable);
    }

    fn do_translate_expr_call(
        &mut self,
        func_id: FuncId,
        args: &'a [AstExpr],
        next_subexpr: usize,
    ) {
        if (1..=args.len()).contains(&next_subexpr) && self.is_task_result_unreachable() {
            self.task_results.splice(
                (self.task_results.len() - next_subexpr)..,
                [ExprResult::Unreachable],
            );
            return;
        } else if next_subexpr < args.len() {
            self.tasks.push(Task::TranslateExprCall {
                func_id,
                args,
                next_subexpr: next_subexpr + 1,
            });
            self.tasks.push(Task::TranslateExpr {
                expr: &args[next_subexpr],
            });
        } else {
            let ret_ty = self.module.funcs[func_id].ty().ret.clone();

            let mut args = (0..args.len())
                .map(|_| {
                    let expr = self.task_results.pop().unwrap();
                    self.use_expr(expr, true)
                })
                .collect::<Vec<_>>();
            args.reverse();

            let ret_local_id = ret_ty.map(|val_ty| self.func.locals.insert(val_ty));
            let call = Call::Direct { ret_local_id, func_id, args };
            let result = self.push_stmt(Stmt::Call(call));
            self.task_results.push(result);
        }
    }

    fn do_translate_expr_call_indirect(
        &mut self,
        ty_id: TypeId,
        table_id: TableId,
        args: &'a [AstExpr],
        index: &'a AstExpr,
        next_subexpr: usize,
    ) {
        if (1..=args.len() + 1).contains(&next_subexpr) && self.is_task_result_unreachable() {
            self.task_results.splice(
                (self.task_results.len() - next_subexpr)..,
                [ExprResult::Unreachable],
            );
            return;
        } else if next_subexpr < args.len() + 1 {
            self.tasks.push(Task::TranslateExprCallIndirect {
                ty_id,
                table_id,
                args,
                index,
                next_subexpr: next_subexpr + 1,
            });
            self.tasks.push(Task::TranslateExpr {
                expr: args.get(next_subexpr).unwrap_or(index),
            });
        } else {
            let ret_ty = self.module.types[ty_id].as_func().ret.clone();

            let mut args = (0..args.len() + 1)
                .map(|_| {
                    let expr = self.task_results.pop().unwrap();
                    self.use_expr(expr, true)
                })
                .collect::<Vec<_>>();
            args.reverse();
            let index = args.pop().unwrap();

            let ret_local_id = ret_ty.map(|val_ty| self.func.locals.insert(val_ty));
            let call = Call::Indirect {
                ret_local_id,
                ty_id,
                table_id,
                args,
                index: Box::new(index),
            };
            let result = self.push_stmt(Stmt::Call(call));
            self.task_results.push(result);
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
                        });

                        self.func.blocks[block_id].body[idx] = Stmt::LocalSet(local_id, expr);

                        Expr::Nullary(NulOp::LocalGet(local_id))
                    }

                    Stmt::LocalSet(local_id, expr) => {
                        self.func.blocks[block_id].body[idx] = Stmt::LocalSet(local_id, expr);

                        Expr::Nullary(NulOp::LocalGet(local_id))
                    }

                    Stmt::Call(call) if call.ret_local_id().is_some() => {
                        let ret_local_id = call.ret_local_id().unwrap();
                        self.func.blocks[block_id].body[idx] = Stmt::Call(call);

                        Expr::Nullary(NulOp::LocalGet(ret_local_id))
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
                    assert!(!local_id.is_null());

                    self.push_stmt(Stmt::LocalSet(local_id, expr));
                }

                self.current_block_mut().term = Terminator::Br(block_id);
            }
        }
    }

    fn remove_nops(&mut self) {
        for block in self.func.blocks.values_mut() {
            block.body.retain(|stmt| !matches!(stmt, Stmt::Nop));
        }
    }
}
