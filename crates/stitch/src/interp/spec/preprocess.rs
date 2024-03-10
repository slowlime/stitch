use log::trace;

use crate::ast::expr::{MemArg, Value};
use crate::ast::ty::ValType;
use crate::cfg::{
    BinOp, Block, BlockId, Expr, FuncBody, I32Load, I32Store, NulOp, Stmt, Terminator, UnOp,
};

use super::Specializer;

impl Specializer<'_, '_> {
    pub(super) fn preprocess_cfg(&mut self, cfg: &mut FuncBody) {
        self.remove_memory_stmts(cfg);
    }

    fn remove_memory_stmts(&mut self, cfg: &mut FuncBody) {
        let block_ids = cfg.blocks.keys().collect::<Vec<_>>();

        for block_id in block_ids {
            for stmt_idx in (0..cfg.blocks[block_id].body.len()).rev() {
                match cfg.blocks[block_id].body[stmt_idx] {
                    Stmt::MemoryCopy { .. } => {
                        self.remove_memory_copy_stmt(cfg, block_id, stmt_idx)
                    }
                    Stmt::MemoryFill { .. } => {
                        self.remove_memory_fill_stmt(cfg, block_id, stmt_idx)
                    }
                    _ => continue,
                }
            }
        }
    }

    fn remove_memory_copy_stmt(&mut self, cfg: &mut FuncBody, block_id: BlockId, stmt_idx: usize) {
        trace!("removing memory.copy from {block_id:?} (stmt {stmt_idx})");
        let Stmt::MemoryCopy {
            dst_mem_id,
            src_mem_id,
            args,
        } = cfg.blocks[block_id].body.remove(stmt_idx)
        else {
            unreachable!()
        };
        let exit_block_id = cfg.split_block_before(block_id, stmt_idx);

        let [dst_local_id, src_local_id, count_local_id] = args.map(|arg| {
            let local_id = cfg.locals.insert(ValType::I32);
            cfg.blocks[block_id]
                .body
                .push(Stmt::LocalSet(local_id, arg));

            local_id
        });

        let fwd_block_id = cfg.blocks.insert(Block {
            name: Some("memmove.forward".into()),
            ..Default::default()
        });
        let bwd_pre_block_id = cfg.blocks.insert(Block {
            name: Some("memmove.backward.pre".into()),
            ..Default::default()
        });
        cfg.blocks[block_id].term = Terminator::If(
            Expr::Binary(
                BinOp::I32LeU,
                Box::new([
                    Expr::Nullary(NulOp::LocalGet(dst_local_id)),
                    Expr::Nullary(NulOp::LocalGet(src_local_id)),
                ]),
            ),
            [fwd_block_id, bwd_pre_block_id],
        );

        let count_minus_one = Expr::Binary(
            BinOp::I32Sub,
            Box::new([
                Expr::Nullary(NulOp::LocalGet(count_local_id)),
                Expr::Value(Value::I32(1), Default::default()),
            ]),
        );
        cfg.blocks[bwd_pre_block_id]
            .body
            .extend([dst_local_id, src_local_id].into_iter().map(|local_id| {
                Stmt::LocalSet(
                    local_id,
                    Expr::Binary(
                        BinOp::I32Add,
                        Box::new([
                            Expr::Nullary(NulOp::LocalGet(local_id)),
                            count_minus_one.clone(),
                        ]),
                    ),
                )
            }));

        let bwd_block_id = cfg.blocks.insert(Block {
            name: Some("memmove.backward".into()),
            ..Default::default()
        });
        cfg.blocks[bwd_pre_block_id].term = Terminator::Br(bwd_block_id);

        for (loop_header, forward) in [(fwd_block_id, true), (bwd_block_id, false)] {
            let body_block_id = cfg.blocks.insert(Block {
                name: cfg.blocks[loop_header]
                    .name
                    .as_ref()
                    .map(|name| format!("{name}.body")),
                ..Default::default()
            });
            cfg.blocks[loop_header].term = Terminator::If(
                Expr::Unary(
                    UnOp::I32Eqz,
                    Box::new(Expr::Nullary(NulOp::LocalGet(count_local_id))),
                ),
                [exit_block_id, body_block_id],
            );

            let next_byte_op = if forward {
                BinOp::I32Add
            } else {
                BinOp::I32Sub
            };

            cfg.blocks[body_block_id].body.extend([
                Stmt::Store(
                    MemArg {
                        mem_id: dst_mem_id,
                        offset: 0,
                        align: 0,
                    },
                    I32Store::One.into(),
                    Box::new([
                        Expr::Nullary(NulOp::LocalGet(dst_local_id)),
                        Expr::Unary(
                            UnOp::Load(
                                MemArg {
                                    mem_id: src_mem_id,
                                    offset: 0,
                                    align: 0,
                                },
                                I32Load::OneU.into(),
                            ),
                            Box::new(Expr::Nullary(NulOp::LocalGet(src_local_id))),
                        ),
                    ]),
                ),
                Stmt::LocalSet(
                    dst_local_id,
                    Expr::Binary(
                        next_byte_op,
                        Box::new([
                            Expr::Nullary(NulOp::LocalGet(dst_local_id)),
                            Expr::Value(Value::I32(1), Default::default()),
                        ]),
                    ),
                ),
                Stmt::LocalSet(
                    src_local_id,
                    Expr::Binary(
                        next_byte_op,
                        Box::new([
                            Expr::Nullary(NulOp::LocalGet(src_local_id)),
                            Expr::Value(Value::I32(1), Default::default()),
                        ]),
                    ),
                ),
                Stmt::LocalSet(
                    count_local_id,
                    Expr::Binary(
                        BinOp::I32Sub,
                        Box::new([
                            Expr::Nullary(NulOp::LocalGet(count_local_id)),
                            Expr::Value(Value::I32(1), Default::default()),
                        ]),
                    ),
                ),
            ]);
            cfg.blocks[body_block_id].term = Terminator::Br(loop_header);
        }
    }

    fn remove_memory_fill_stmt(&mut self, cfg: &mut FuncBody, block_id: BlockId, stmt_idx: usize) {
        trace!("removing memory.fill from {block_id:?} (stmt {stmt_idx})");
        let Stmt::MemoryFill(mem_id, args) = cfg.blocks[block_id].body.remove(stmt_idx) else {
            unreachable!()
        };
        let exit_block_id = cfg.split_block_before(block_id, stmt_idx);

        let [dst_local_id, val_local_id, count_local_id] = args.map(|arg| {
            let local_id = cfg.locals.insert(ValType::I32);
            cfg.blocks[block_id]
                .body
                .push(Stmt::LocalSet(local_id, arg));

            local_id
        });

        let body_block_id = cfg.blocks.insert(Block {
            name: Some("memset.body".into()),
            ..Default::default()
        });
        let loop_header = cfg.blocks.insert(Block {
            name: Some("memset".into()),
            term: Terminator::If(
                Expr::Unary(
                    UnOp::I32Eqz,
                    Box::new(Expr::Nullary(NulOp::LocalGet(count_local_id))),
                ),
                [exit_block_id, body_block_id],
            ),
            ..Default::default()
        });

        cfg.blocks[body_block_id].body.extend([
            Stmt::Store(
                MemArg {
                    mem_id,
                    offset: 0,
                    align: 0,
                },
                I32Store::One.into(),
                Box::new([
                    Expr::Nullary(NulOp::LocalGet(dst_local_id)),
                    Expr::Nullary(NulOp::LocalGet(val_local_id)),
                ]),
            ),
            Stmt::LocalSet(
                dst_local_id,
                Expr::Binary(
                    BinOp::I32Add,
                    Box::new([
                        Expr::Nullary(NulOp::LocalGet(dst_local_id)),
                        Expr::Value(Value::I32(1), Default::default()),
                    ]),
                ),
            ),
            Stmt::LocalSet(
                count_local_id,
                Expr::Binary(
                    BinOp::I32Sub,
                    Box::new([
                        Expr::Nullary(NulOp::LocalGet(count_local_id)),
                        Expr::Value(Value::I32(1), Default::default()),
                    ]),
                ),
            ),
        ]);
        cfg.blocks[body_block_id].term = Terminator::Br(loop_header);
    }
}
