use crate::ast::expr::{NulOp, Value};
use crate::ast::func::FuncImport;
use crate::ast::{Expr, Func, FuncBody, FuncId};

use super::PostProc;

impl PostProc<'_> {
    fn replace_intr<F>(&mut self, func_id: FuncId, f: F)
    where
        F: FnOnce(&mut Self, &mut FuncBody),
    {
        let Func::Import(FuncImport { name, ty, .. }) = &self.module.funcs[func_id] else {
            panic!()
        };
        let mut body = FuncBody::with_locals(ty.clone());
        body.name = name.clone();
        f(self, &mut body);
        self.module.funcs[func_id] = Func::Body(body);
    }

    pub(super) fn replace_intr_arg_count(&mut self, func_id: FuncId) {
        self.replace_intr(func_id, |_, body| {
            body.main_block
                .body
                .push(Expr::Value(Value::I32(0), Default::default()))
        });
    }

    pub(super) fn replace_intr_arg_len(&mut self, func_id: FuncId) {
        self.replace_intr(func_id, |_, body| {
            body.main_block.body.push(NulOp::Unreachable.into())
        });
    }

    pub(super) fn replace_intr_arg_read(&mut self, func_id: FuncId) {
        self.replace_intr(func_id, |_, body| {
            body.main_block.body.push(NulOp::Unreachable.into())
        });
    }

    pub(super) fn replace_intr_specialize(&mut self, func_id: FuncId) {
        self.replace_intr(func_id, |_, body| {
            body.main_block
                .body
                .push(Expr::Value(Value::I32(0), Default::default()))
        });
    }

    pub(super) fn replace_intr_unknown(&mut self, func_id: FuncId) {
        self.replace_intr(func_id, |_, body| {
            body.main_block.body.push(NulOp::Unreachable.into())
        });
    }

    pub(super) fn replace_intr_const_ptr(&mut self, func_id: FuncId) {
        self.replace_intr(func_id, |_, body| {
            body.main_block
                .body
                .push(NulOp::LocalGet(body.params[0]).into())
        });
    }

    pub(super) fn replace_intr_symbolic_ptr(&mut self, func_id: FuncId) {
        self.replace_intr(func_id, |_, body| {
            body.main_block
                .body
                .push(NulOp::LocalGet(body.params[0]).into())
        });
    }

    pub(super) fn replace_intr_concrete_ptr(&mut self, func_id: FuncId) {
        self.replace_intr(func_id, |_, body| {
            body.main_block
                .body
                .push(NulOp::LocalGet(body.params[0]).into())
        });
    }

    pub(super) fn replace_intr_propagate_load(&mut self, func_id: FuncId) {
        self.replace_intr(func_id, |_, body| {
            body.main_block
                .body
                .push(NulOp::LocalGet(body.params[0]).into())
        });
    }

    pub(super) fn replace_intr_print_value(&mut self, func_id: FuncId) {
        self.replace_intr(func_id, |_, _| {});
    }

    pub(super) fn replace_intr_print_str(&mut self, func_id: FuncId) {
        self.replace_intr(func_id, |_, _| {});
    }

    pub(super) fn replace_intr_is_specializing(&mut self, func_id: FuncId) {
        self.replace_intr(func_id, |_, body| {
            body.main_block
                .body
                .push(Expr::Value(Value::I32(0), Default::default()))
        });
    }

    pub(super) fn replace_intr_inline(&mut self, func_id: FuncId) {
        self.replace_intr(func_id, |_, body| {
            body.main_block
                .body
                .push(NulOp::LocalGet(body.params[0]).into())
        });
    }

    pub(super) fn replace_intr_no_inline(&mut self, func_id: FuncId) {
        self.replace_intr(func_id, |_, body| {
            body.main_block
                .body
                .push(NulOp::LocalGet(body.params[0]).into())
        });
    }

    pub(super) fn replace_intr_file_open(&mut self, func_id: FuncId) {
        self.replace_intr(func_id, |_, body| {
            body.main_block
                .body
                .push(Expr::Value(Value::I32(-1), Default::default()))
        });
    }

    pub(super) fn replace_intr_file_read(&mut self, func_id: FuncId) {
        self.replace_intr(func_id, |_, body| {
            body.main_block
                .body
                .push(Expr::Value(Value::I32(-1), Default::default()))
        });
    }

    pub(super) fn replace_intr_file_close(&mut self, func_id: FuncId) {
        self.replace_intr(func_id, |_, body| {
            body.main_block
                .body
                .push(Expr::Value(Value::I32(-1), Default::default()))
        });
    }

    pub(super) fn replace_intr_func_spec_policy(&mut self, func_id: FuncId) {
        self.replace_intr(func_id, |_, _| {})
    }

    pub(super) fn replace_intr_symbolic_stack_ptr(&mut self, func_id: FuncId) {
        self.replace_intr(func_id, |_, _| {})
    }
}
