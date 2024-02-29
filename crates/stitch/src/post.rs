use std::collections::HashSet;

use log::warn;

use crate::ast::expr::{make_visitor, NulOp, UnOp};
use crate::ast::ty::Type;
use crate::ast::{Expr, Func, FuncId, ImportId, Module, TableDef};

pub struct PostProc<'a> {
    module: &'a mut Module,
}

impl<'a> PostProc<'a> {
    pub fn new(module: &'a mut Module) -> Self {
        Self { module }
    }

    pub fn process(mut self) {
        // FIXME: replace intrinsics with dummy functions
        // self.remove_intrinsics();
        self.insert_func_types();
        self.remove_unused_locals();
    }

    fn remove_intrinsics(&mut self) {
        let mut func_ids = self
            .module
            .funcs
            .iter()
            .filter(|(_, func)| func.get_intrinsic(&self.module).is_some())
            .map(|(func_id, _)| func_id)
            .collect::<HashSet<_>>();
        let mut import_ids = self
            .module
            .imports
            .keys()
            .filter(|&import_id| self.module.get_intrinsic(import_id).is_some())
            .collect::<HashSet<_>>();

        fn check_expr<'a>(
            module: &'a Module,
            func_ids: &'a mut HashSet<FuncId>,
            import_ids: &'a mut HashSet<ImportId>,
        ) -> impl FnMut(&Expr) -> bool + 'a {
            move |expr: &Expr| match expr {
                Expr::Call(func_id, _) if func_ids.remove(func_id) => {
                    warn!("Expr::Call references an intrinsic");
                    import_ids.remove(match &module.funcs[*func_id] {
                        Func::Import(import) => &import.import_id,
                        _ => unreachable!(),
                    });

                    true
                }

                _ => true,
            }
        }

        for func in self.module.funcs.values() {
            let Some(body) = func.body() else { continue };

            for expr in &body.main_block.body {
                expr.all(&mut check_expr(
                    &self.module,
                    &mut func_ids,
                    &mut import_ids,
                ));
            }
        }

        for table in self.module.tables.values() {
            let TableDef::Elems(elems) = &table.def else {
                continue;
            };

            for func_id in elems.iter().flatten() {
                if func_ids.remove(func_id) {
                    warn!("a table contains a reference to an intrinsic");
                }
            }
        }

        self.module
            .funcs
            .retain(|func_id, _| !func_ids.contains(&func_id));
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
