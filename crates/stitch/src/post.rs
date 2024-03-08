mod replace_intrinsics;

use std::collections::HashSet;

use crate::ast::expr::{make_visitor, NulOp, UnOp};
use crate::ast::ty::Type;
use crate::ast::{Expr, IntrinsicDecl, Module};

pub struct PostProc<'a> {
    module: &'a mut Module,
}

impl<'a> PostProc<'a> {
    pub fn new(module: &'a mut Module) -> Self {
        Self { module }
    }

    pub fn process(mut self) {
        self.replace_intrinsics();
        self.insert_func_types();
        self.remove_unused_locals();
    }

    fn replace_intrinsics(&mut self) {
        let intrinsics = self
            .module
            .funcs
            .iter()
            .filter_map(|(func_id, func)| Some(func_id).zip(func.get_intrinsic(&self.module)))
            .collect::<HashSet<_>>();

        for (func_id, intrinsic) in intrinsics {
            match intrinsic {
                IntrinsicDecl::ArgCount => self.replace_intr_arg_count(func_id),
                IntrinsicDecl::ArgLen => self.replace_intr_arg_len(func_id),
                IntrinsicDecl::ArgRead => self.replace_intr_arg_read(func_id),
                IntrinsicDecl::Specialize => self.replace_intr_specialize(func_id),
                IntrinsicDecl::Unknown => self.replace_intr_unknown(func_id),
                IntrinsicDecl::ConstPtr => self.replace_intr_const_ptr(func_id),
                IntrinsicDecl::SymbolicPtr => self.replace_intr_symbolic_ptr(func_id),
                IntrinsicDecl::PropagateLoad => self.replace_intr_propagate_load(func_id),
                IntrinsicDecl::PrintValue => self.replace_intr_print_value(func_id),
                IntrinsicDecl::PrintStr => self.replace_intr_print_str(func_id),
                IntrinsicDecl::IsSpecializing => self.replace_intr_is_specializing(func_id),
                IntrinsicDecl::Inline => self.replace_intr_inline(func_id),
                IntrinsicDecl::NoInline => self.replace_intr_no_inline(func_id),
            }
        }

        let import_ids = self
            .module
            .imports
            .keys()
            .filter(|&import_id| self.module.get_intrinsic(import_id).is_some())
            .collect::<Vec<_>>();
        self.module
            .imports
            .retain(|import_id, _| !import_ids.contains(&import_id));
    }

    fn insert_func_types(&mut self) {
        for func in self.module.funcs.values() {
            let Some(body) = func.body() else { continue };
            self.module.types.insert(Type::Func(body.ty.clone()));
        }
    }

    fn remove_unused_locals(&mut self) {
        for func in self.module.funcs.values_mut() {
            let Some(body) = func.body_mut() else {
                continue;
            };
            let mut unused_locals = body.locals.keys().collect::<HashSet<_>>();

            for local_id in &body.params {
                unused_locals.remove(local_id);
            }

            for expr in &body.main_block.body {
                expr.map(&mut make_visitor(|expr, _| match expr {
                    Expr::Nullary(NulOp::LocalGet(local_id))
                    | Expr::Unary(UnOp::LocalSet(local_id) | UnOp::LocalTee(local_id), _) => {
                        unused_locals.remove(local_id);
                    }

                    _ => {}
                }));
            }

            body.locals
                .retain(|local_id, _| !unused_locals.contains(&local_id));
        }
    }
}
