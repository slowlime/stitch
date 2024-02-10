use std::array;

use log::trace;
use slotmap::{SecondaryMap, SparseSecondaryMap};

use super::{
    BinOp, BlockId, Call, DomTree, Expr, FuncBody, I32Load, I64Load, Load, LocalId, NulOp,
    Predecessors, Rpo, Stmt, Store, TernOp, UnOp,
};
use crate::ast::expr::{
    BinOp as AstBinOp, Block as AstBlock, NulOp as AstNulOp, TernOp as AstTernOp, UnOp as AstUnOp,
};
use crate::ast::ty::BlockType;
use crate::ast::{
    BlockId as AstBlockId, Expr as AstExpr, FuncBody as AstFunc, LocalId as AstLocalId,
};
use crate::cfg::{I32Store, I64Store, Terminator};

// See: https://dl.acm.org/doi/pdf/10.1145/3547621

impl FuncBody {
    pub fn to_ast(&self) -> AstFunc {
        let ast = AstFunc::new(self.ty.clone());
        let rpo = self.rpo();
        let preds = self.predecessors();
        let dom_tree = self.dom_tree(&preds, &rpo);

        let translator = Translator {
            func: self,
            ast,
            rpo,
            preds,
            dom_tree,
            local_map: Default::default(),
            ctx: Default::default(),
            loop_headers: Default::default(),
            merge_nodes: Default::default(),
        };

        translator.translate()
    }
}

enum Context {
    BlockFollowedBy(BlockId),
    LoopHeadedBy(BlockId),
}

impl Context {
    fn matches(&self, block_id: BlockId) -> bool {
        match *self {
            Self::BlockFollowedBy(self_block_id) | Self::LoopHeadedBy(self_block_id) => {
                block_id == self_block_id
            }
        }
    }
}

struct Translator<'a> {
    func: &'a FuncBody,
    ast: AstFunc,
    rpo: Rpo,
    preds: Predecessors,
    dom_tree: DomTree,
    local_map: SecondaryMap<LocalId, AstLocalId>,
    ctx: Vec<(Context, AstBlockId)>,
    loop_headers: SparseSecondaryMap<BlockId, ()>,
    merge_nodes: SparseSecondaryMap<BlockId, ()>,
}

impl Translator<'_> {
    fn translate(mut self) -> AstFunc {
        self.analyze_cfg();
        self.translate_params();
        let mut body = vec![];
        self.do_tree(self.func.entry, &mut body);

        if self.func.ty.ret.is_some() && !body.last().is_some_and(AstExpr::diverges) {
            body.push(AstExpr::Nullary(AstNulOp::Unreachable));
        }

        self.ast.main_block.body = body;

        self.ast
    }

    fn analyze_cfg(&mut self) {
        for &block_id in &self.rpo.order {
            let mut in_fwd_edges = 0usize;

            for &pred_block_id in &self.preds[block_id] {
                if self.rpo.idx[block_id] <= self.rpo.idx[pred_block_id] {
                    // backward edge
                    assert!(self.dom_tree.dominates(block_id, pred_block_id));
                    self.loop_headers.insert(block_id, ());
                } else {
                    // forward edge
                    in_fwd_edges += 1;
                }
            }

            if in_fwd_edges > 1 {
                self.merge_nodes.insert(block_id, ());
            }

            if let Terminator::Switch(_, ref block_ids) = self.func.blocks[block_id].term {
                // mark all of switch targets as merge nodes so they get their own block
                self.merge_nodes
                    .extend(block_ids.iter().map(|&block_id| (block_id, ())));
            }
        }
    }

    fn translate_params(&mut self) {
        self.ast.params = self
            .func
            .params
            .iter()
            .map(|&local_id| self.translate_local(local_id))
            .collect();
    }

    fn lookup_block_id(&self, block_id: BlockId) -> AstBlockId {
        self.ctx
            .iter()
            .rev()
            .find(|(ctx, _)| ctx.matches(block_id))
            .unwrap()
            .1
    }

    fn do_tree(&mut self, root: BlockId, exprs: &mut Vec<AstExpr>) {
        let merge_children = self.dom_tree.succ[root]
            .iter()
            .rev() // blocks with smaller rpo numbers should go after those with greater
            .copied()
            .filter(|&succ_block_id| self.is_merge_node(succ_block_id))
            .collect::<Vec<_>>();

        trace!(
            "do_tree({root:?}): {}",
            if self.is_loop_header(root) {
                "loop header"
            } else {
                "plain"
            }
        );

        if self.is_loop_header(root) {
            let ast_block_id = self.ast.blocks.insert(());
            self.ctx.push((Context::LoopHeadedBy(root), ast_block_id));
            let mut body = vec![];
            self.node_within(root, &merge_children, &mut body);

            exprs.push(AstExpr::Loop(
                BlockType::Empty,
                AstBlock {
                    body,
                    id: ast_block_id,
                },
            ));
            self.ctx.pop();
        } else {
            self.node_within(root, &merge_children, exprs)
        }
    }

    fn node_within(
        &mut self,
        block_id: BlockId,
        merge_children: &[BlockId],
        exprs: &mut Vec<AstExpr>,
    ) {
        trace!("node_within({block_id:?}, {merge_children:?})");

        if let Some((&head, tail)) = merge_children.split_first() {
            let mut body = vec![];
            let ast_block_id = self.ast.blocks.insert(());
            self.ctx
                .push((Context::BlockFollowedBy(head), ast_block_id));
            self.node_within(block_id, tail, &mut body);
            self.ctx.pop();

            exprs.push(AstExpr::Block(
                BlockType::Empty,
                AstBlock {
                    body,
                    id: ast_block_id,
                },
            ));
            self.do_tree(head, exprs);
        } else {
            trace!("translating the body of {block_id:?}");
            self.translate_body(block_id, exprs);

            trace!(
                "translating the terminator of {block_id:?}: {}",
                match &self.func.blocks[block_id].term {
                    Terminator::Trap => "trap".to_owned(),
                    Terminator::Br(target_block_id) => format!("br to {target_block_id:?}"),
                    Terminator::If(_, [then_block_id, else_block_id]) =>
                        format!("if then {then_block_id:?} else {else_block_id:?}"),
                    Terminator::Switch(_, block_ids) => format!("switch [{block_ids:?}]"),
                    Terminator::Return(_) => format!("return"),
                }
            );

            match &self.func.blocks[block_id].term {
                Terminator::Trap => exprs.push(AstExpr::Nullary(AstNulOp::Unreachable)),
                &Terminator::Br(target_block_id) => {
                    self.do_branch(block_id, target_block_id, exprs)
                }

                &Terminator::If(ref condition, [then_block_id, else_block_id]) => {
                    let ast_then_block_id = self.ast.blocks.insert(());
                    let ast_else_block_id = self.ast.blocks.insert(());

                    let mut then_body = vec![];
                    self.do_branch(block_id, then_block_id, &mut then_body);

                    let mut else_body = vec![];
                    self.do_branch(block_id, else_block_id, &mut else_body);

                    exprs.push(AstExpr::If(
                        BlockType::Empty,
                        Box::new(self.translate_expr(condition)),
                        AstBlock {
                            body: then_body,
                            id: ast_then_block_id,
                        },
                        AstBlock {
                            body: else_body,
                            id: ast_else_block_id,
                        },
                    ));
                }

                Terminator::Switch(index, block_ids) => {
                    let block_ids = block_ids
                        .into_iter()
                        // all targets are merge nodes, so the switch is nested within their blocks
                        .map(|&block_id| self.lookup_block_id(block_id))
                        .collect::<Vec<_>>();
                    let (&default_block_id, block_ids) = block_ids.split_first().unwrap();

                    exprs.push(AstExpr::BrTable(
                        block_ids.to_owned(),
                        default_block_id,
                        None,
                        Box::new(self.translate_expr(index)),
                    ));
                }

                Terminator::Return(expr) => {
                    exprs.push(AstExpr::Return(
                        expr.as_ref()
                            .map(|expr| Box::new(self.translate_expr(expr))),
                    ));
                }
            }
        }
    }

    fn do_branch(
        &mut self,
        from_block_id: BlockId,
        to_block_id: BlockId,
        exprs: &mut Vec<AstExpr>,
    ) {
        if self.rpo.idx[from_block_id] >= self.rpo.idx[to_block_id] // backward edge
            || self.is_merge_node(to_block_id)
        {
            trace!(
                "do_branch({from_block_id:?}, {to_block_id:?}) -> (br {:?})",
                self.lookup_block_id(to_block_id),
            );
            exprs.push(AstExpr::Br(self.lookup_block_id(to_block_id), None));
        } else {
            trace!("do_branch({from_block_id:?}, {to_block_id:?}) -> do_tree({to_block_id:?})");
            self.do_tree(to_block_id, exprs);
        }
    }

    fn translate_body(&mut self, block_id: BlockId, exprs: &mut Vec<AstExpr>) {
        for stmt in &self.func.blocks[block_id].body {
            self.translate_stmt(stmt, exprs);
        }
    }

    fn translate_stmt(&mut self, stmt: &Stmt, exprs: &mut Vec<AstExpr>) {
        match stmt {
            Stmt::Nop => {}

            Stmt::Drop(expr) => exprs.push(AstExpr::Unary(
                AstUnOp::Drop,
                Box::new(self.translate_expr(expr)),
            )),

            Stmt::LocalSet(local_id, expr) => exprs.push(AstExpr::Unary(
                AstUnOp::LocalSet(self.translate_local(*local_id)),
                Box::new(self.translate_expr(expr)),
            )),

            Stmt::GlobalSet(global_id, expr) => exprs.push(AstExpr::Unary(
                AstUnOp::GlobalSet(*global_id),
                Box::new(self.translate_expr(expr)),
            )),

            Stmt::Store(mem_arg, store, args) => {
                let [addr, expr] = &**args;

                let op = match store {
                    Store::I32(I32Store::Four) => AstBinOp::I32Store,
                    Store::I32(I32Store::Two) => AstBinOp::I32Store16,
                    Store::I32(I32Store::One) => AstBinOp::I32Store8,

                    Store::I64(I64Store::Eight) => AstBinOp::I64Store,
                    Store::I64(I64Store::Four) => AstBinOp::I64Store32,
                    Store::I64(I64Store::Two) => AstBinOp::I64Store16,
                    Store::I64(I64Store::One) => AstBinOp::I64Store8,

                    Store::F32 => AstBinOp::F32Store,
                    Store::F64 => AstBinOp::F64Store,
                };

                exprs.push(AstExpr::Binary(
                    op(*mem_arg),
                    [
                        Box::new(self.translate_expr(addr)),
                        Box::new(self.translate_expr(expr)),
                    ],
                ));
            }

            Stmt::Call(call) => exprs.push(self.translate_call(call)),
        }
    }

    fn translate_expr(&mut self, expr: &Expr) -> AstExpr {
        match expr {
            &Expr::Value(value, attrs) => AstExpr::Value(value, attrs),
            Expr::Intrinsic(intrinsic) => AstExpr::Intrinsic(intrinsic.clone()),
            &Expr::Index(id) => AstExpr::Index(id),

            Expr::Nullary(op) => AstExpr::Nullary(match *op {
                NulOp::LocalGet(local_id) => AstNulOp::LocalGet(self.translate_local(local_id)),
                NulOp::GlobalGet(global_id) => AstNulOp::GlobalGet(global_id),
                NulOp::MemorySize(mem_id) => AstNulOp::MemorySize(mem_id),
            }),

            Expr::Unary(op, inner) => AstExpr::Unary(
                match *op {
                    UnOp::I32Clz => AstUnOp::I32Clz,
                    UnOp::I32Ctz => AstUnOp::I32Ctz,
                    UnOp::I32Popcnt => AstUnOp::I32Popcnt,
                    UnOp::I64Clz => AstUnOp::I64Clz,
                    UnOp::I64Ctz => AstUnOp::I64Ctz,
                    UnOp::I64Popcnt => AstUnOp::I64Popcnt,
                    UnOp::F32Abs => AstUnOp::F32Abs,
                    UnOp::F32Neg => AstUnOp::F32Neg,
                    UnOp::F32Sqrt => AstUnOp::F32Sqrt,
                    UnOp::F32Ceil => AstUnOp::F32Ceil,
                    UnOp::F32Floor => AstUnOp::F32Floor,
                    UnOp::F32Trunc => AstUnOp::F32Trunc,
                    UnOp::F32Nearest => AstUnOp::F32Nearest,
                    UnOp::F64Abs => AstUnOp::F64Abs,
                    UnOp::F64Neg => AstUnOp::F64Neg,
                    UnOp::F64Sqrt => AstUnOp::F64Sqrt,
                    UnOp::F64Ceil => AstUnOp::F64Ceil,
                    UnOp::F64Floor => AstUnOp::F64Floor,
                    UnOp::F64Trunc => AstUnOp::F64Trunc,
                    UnOp::F64Nearest => AstUnOp::F64Nearest,
                    UnOp::I32Eqz => AstUnOp::I32Eqz,
                    UnOp::I64Eqz => AstUnOp::I64Eqz,
                    UnOp::I32WrapI64 => AstUnOp::I32WrapI64,
                    UnOp::I64ExtendI32S => AstUnOp::I64ExtendI32S,
                    UnOp::I64ExtendI32U => AstUnOp::I64ExtendI32U,
                    UnOp::I32TruncF32S => AstUnOp::I32TruncF32S,
                    UnOp::I32TruncF32U => AstUnOp::I32TruncF32U,
                    UnOp::I32TruncF64S => AstUnOp::I32TruncF64S,
                    UnOp::I32TruncF64U => AstUnOp::I32TruncF64U,
                    UnOp::I64TruncF32S => AstUnOp::I64TruncF32S,
                    UnOp::I64TruncF32U => AstUnOp::I64TruncF32U,
                    UnOp::I64TruncF64S => AstUnOp::I64TruncF64S,
                    UnOp::I64TruncF64U => AstUnOp::I64TruncF64U,
                    UnOp::F32DemoteF64 => AstUnOp::F32DemoteF64,
                    UnOp::F64PromoteF32 => AstUnOp::F64PromoteF32,
                    UnOp::F32ConvertI32S => AstUnOp::F32ConvertI32S,
                    UnOp::F32ConvertI32U => AstUnOp::F32ConvertI32U,
                    UnOp::F32ConvertI64S => AstUnOp::F32ConvertI64S,
                    UnOp::F32ConvertI64U => AstUnOp::F32ConvertI64U,
                    UnOp::F64ConvertI32S => AstUnOp::F64ConvertI32S,
                    UnOp::F64ConvertI32U => AstUnOp::F64ConvertI32U,
                    UnOp::F64ConvertI64S => AstUnOp::F64ConvertI64S,
                    UnOp::F64ConvertI64U => AstUnOp::F64ConvertI64U,
                    UnOp::F32ReinterpretI32 => AstUnOp::F32ReinterpretI32,
                    UnOp::F64ReinterpretI64 => AstUnOp::F64ReinterpretI64,
                    UnOp::I32ReinterpretF32 => AstUnOp::I32ReinterpretF32,
                    UnOp::I64ReinterpretF64 => AstUnOp::I64ReinterpretF64,
                    UnOp::I32Extend8S => AstUnOp::I32Extend8S,
                    UnOp::I32Extend16S => AstUnOp::I32Extend16S,
                    UnOp::I64Extend8S => AstUnOp::I64Extend8S,
                    UnOp::I64Extend16S => AstUnOp::I64Extend16S,
                    UnOp::I64Extend32S => AstUnOp::I64Extend32S,
                    UnOp::LocalTee(local_id) => AstUnOp::LocalTee(self.translate_local(local_id)),

                    UnOp::Load(mem_arg, load) => match load {
                        Load::I32(I32Load::Four) => AstUnOp::I32Load(mem_arg),
                        Load::I32(I32Load::TwoS) => AstUnOp::I32Load16S(mem_arg),
                        Load::I32(I32Load::TwoU) => AstUnOp::I32Load16U(mem_arg),
                        Load::I32(I32Load::OneS) => AstUnOp::I32Load8S(mem_arg),
                        Load::I32(I32Load::OneU) => AstUnOp::I32Load8U(mem_arg),

                        Load::I64(I64Load::Eight) => AstUnOp::I64Load(mem_arg),
                        Load::I64(I64Load::FourS) => AstUnOp::I64Load32S(mem_arg),
                        Load::I64(I64Load::FourU) => AstUnOp::I64Load32U(mem_arg),
                        Load::I64(I64Load::TwoS) => AstUnOp::I64Load16S(mem_arg),
                        Load::I64(I64Load::TwoU) => AstUnOp::I64Load16U(mem_arg),
                        Load::I64(I64Load::OneS) => AstUnOp::I64Load8S(mem_arg),
                        Load::I64(I64Load::OneU) => AstUnOp::I64Load8U(mem_arg),

                        Load::F32 => AstUnOp::F32Load(mem_arg),
                        Load::F64 => AstUnOp::F64Load(mem_arg),
                    },

                    UnOp::MemoryGrow(mem_id) => AstUnOp::MemoryGrow(mem_id),
                },
                Box::new(self.translate_expr(inner)),
            ),

            Expr::Binary(op, exprs) => AstExpr::Binary(
                match *op {
                    BinOp::I32Add => AstBinOp::I32Add,
                    BinOp::I32Sub => AstBinOp::I32Sub,
                    BinOp::I32Mul => AstBinOp::I32Mul,
                    BinOp::I32DivS => AstBinOp::I32DivS,
                    BinOp::I32DivU => AstBinOp::I32DivU,
                    BinOp::I32RemS => AstBinOp::I32RemS,
                    BinOp::I32RemU => AstBinOp::I32RemU,
                    BinOp::I32And => AstBinOp::I32And,
                    BinOp::I32Or => AstBinOp::I32Or,
                    BinOp::I32Xor => AstBinOp::I32Xor,
                    BinOp::I32Shl => AstBinOp::I32Shl,
                    BinOp::I32ShrS => AstBinOp::I32ShrS,
                    BinOp::I32ShrU => AstBinOp::I32ShrU,
                    BinOp::I32Rotl => AstBinOp::I32Rotl,
                    BinOp::I32Rotr => AstBinOp::I32Rotr,
                    BinOp::I64Add => AstBinOp::I64Add,
                    BinOp::I64Sub => AstBinOp::I64Sub,
                    BinOp::I64Mul => AstBinOp::I64Mul,
                    BinOp::I64DivS => AstBinOp::I64DivS,
                    BinOp::I64DivU => AstBinOp::I64DivU,
                    BinOp::I64RemS => AstBinOp::I64RemS,
                    BinOp::I64RemU => AstBinOp::I64RemU,
                    BinOp::I64And => AstBinOp::I64And,
                    BinOp::I64Or => AstBinOp::I64Or,
                    BinOp::I64Xor => AstBinOp::I64Xor,
                    BinOp::I64Shl => AstBinOp::I64Shl,
                    BinOp::I64ShrS => AstBinOp::I64ShrS,
                    BinOp::I64ShrU => AstBinOp::I64ShrU,
                    BinOp::I64Rotl => AstBinOp::I64Rotl,
                    BinOp::I64Rotr => AstBinOp::I64Rotr,
                    BinOp::F32Add => AstBinOp::F32Add,
                    BinOp::F32Sub => AstBinOp::F32Sub,
                    BinOp::F32Mul => AstBinOp::F32Mul,
                    BinOp::F32Div => AstBinOp::F32Div,
                    BinOp::F32Min => AstBinOp::F32Min,
                    BinOp::F32Max => AstBinOp::F32Max,
                    BinOp::F32Copysign => AstBinOp::F32Copysign,
                    BinOp::F64Add => AstBinOp::F64Add,
                    BinOp::F64Sub => AstBinOp::F64Sub,
                    BinOp::F64Mul => AstBinOp::F64Mul,
                    BinOp::F64Div => AstBinOp::F64Div,
                    BinOp::F64Min => AstBinOp::F64Min,
                    BinOp::F64Max => AstBinOp::F64Max,
                    BinOp::F64Copysign => AstBinOp::F64Copysign,
                    BinOp::I32Eq => AstBinOp::I32Eq,
                    BinOp::I32Ne => AstBinOp::I32Ne,
                    BinOp::I32LtS => AstBinOp::I32LtS,
                    BinOp::I32LtU => AstBinOp::I32LtU,
                    BinOp::I32GtS => AstBinOp::I32GtS,
                    BinOp::I32GtU => AstBinOp::I32GtU,
                    BinOp::I32LeS => AstBinOp::I32LeS,
                    BinOp::I32LeU => AstBinOp::I32LeU,
                    BinOp::I32GeS => AstBinOp::I32GeS,
                    BinOp::I32GeU => AstBinOp::I32GeU,
                    BinOp::I64Eq => AstBinOp::I64Eq,
                    BinOp::I64Ne => AstBinOp::I64Ne,
                    BinOp::I64LtS => AstBinOp::I64LtS,
                    BinOp::I64LtU => AstBinOp::I64LtU,
                    BinOp::I64GtS => AstBinOp::I64GtS,
                    BinOp::I64GtU => AstBinOp::I64GtU,
                    BinOp::I64LeS => AstBinOp::I64LeS,
                    BinOp::I64LeU => AstBinOp::I64LeU,
                    BinOp::I64GeS => AstBinOp::I64GeS,
                    BinOp::I64GeU => AstBinOp::I64GeU,
                    BinOp::F32Eq => AstBinOp::F32Eq,
                    BinOp::F32Ne => AstBinOp::F32Ne,
                    BinOp::F32Lt => AstBinOp::F32Lt,
                    BinOp::F32Gt => AstBinOp::F32Gt,
                    BinOp::F32Le => AstBinOp::F32Le,
                    BinOp::F32Ge => AstBinOp::F32Ge,
                    BinOp::F64Eq => AstBinOp::F64Eq,
                    BinOp::F64Ne => AstBinOp::F64Ne,
                    BinOp::F64Lt => AstBinOp::F64Lt,
                    BinOp::F64Gt => AstBinOp::F64Gt,
                    BinOp::F64Le => AstBinOp::F64Le,
                    BinOp::F64Ge => AstBinOp::F64Ge,
                },
                array::from_fn(|i| Box::new(self.translate_expr(&exprs[i]))),
            ),

            Expr::Ternary(op, exprs) => AstExpr::Ternary(
                match *op {
                    TernOp::Select => AstTernOp::Select,
                },
                array::from_fn(|i| Box::new(self.translate_expr(&exprs[i]))),
            ),

            Expr::Call(call) => self.translate_call(call),
        }
    }

    fn translate_call(&mut self, call: &Call) -> AstExpr {
        match *call {
            Call::Direct { func_id, ref args } => AstExpr::Call(
                func_id,
                args.iter().map(|expr| self.translate_expr(expr)).collect(),
            ),

            Call::Indirect {
                ty_id,
                table_id,
                ref args,
                ref index,
            } => AstExpr::CallIndirect(
                ty_id,
                table_id,
                args.iter().map(|expr| self.translate_expr(expr)).collect(),
                Box::new(self.translate_expr(index)),
            ),
        }
    }

    fn translate_local(&mut self, local_id: LocalId) -> AstLocalId {
        *self
            .local_map
            .entry(local_id)
            .unwrap()
            .or_insert_with(|| self.ast.locals.insert(self.func.locals[local_id].clone()))
    }

    fn is_merge_node(&self, block_id: BlockId) -> bool {
        self.merge_nodes.contains_key(block_id)
    }

    fn is_loop_header(&self, block_id: BlockId) -> bool {
        self.loop_headers.contains_key(block_id)
    }
}
