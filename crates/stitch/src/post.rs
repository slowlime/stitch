use std::collections::HashSet;

use log::warn;

use crate::ir::expr::{Id, NulOp, UnOp};
use crate::ir::ty::Type;
use crate::ir::{Expr, FuncId, GlobalDef, GlobalId, Module, TableDef};

pub struct PostProc<'a> {
    module: &'a mut Module,
}

impl<'a> PostProc<'a> {
    pub fn new(module: &'a mut Module) -> Self {
        Self { module }
    }

    pub fn process(mut self) {
        self.remove_intrinsics();
        self.insert_func_types();
    }

    fn remove_intrinsics(&mut self) {
        let mut func_ids = self
            .module
            .funcs
            .iter()
            .filter(|(_, func)| func.get_intrinsic(&self.module).is_some())
            .map(|(func_id, _)| func_id)
            .collect::<HashSet<_>>();
        let mut global_ids = self
            .module
            .globals
            .iter()
            .filter(|(_, global)| global.def.get_intrinsic(&self.module).is_some())
            .map(|(global_id, _)| global_id)
            .collect::<HashSet<_>>();
        let import_ids = self
            .module
            .imports
            .keys()
            .filter(|&import_id| self.module.get_intrinsic(import_id).is_some())
            .collect::<HashSet<_>>();

        fn check_expr<'a>(
            func_ids: &'a mut HashSet<FuncId>,
            global_ids: &'a mut HashSet<GlobalId>,
        ) -> impl FnMut(&Expr) -> bool + 'a {
            move |expr: &Expr| match expr {
                Expr::Index(Id::Func(func_id)) if func_ids.remove(func_id) => {
                    warn!("Expr::Index references an intrinsic");
                    true
                }

                Expr::Intrinsic(_) => {
                    warn!("encountered an unprocessed intrinsic expression");
                    true
                }

                Expr::Nullary(NulOp::GlobalGet(global_id))
                | Expr::Unary(UnOp::GlobalSet(global_id), _)
                    if global_ids.remove(global_id) =>
                {
                    warn!("a global variable instruction references an intrinsic");
                    true
                }

                Expr::Call(func_id, _) if func_ids.remove(func_id) => {
                    warn!("Expr::Call references an intrinsic");
                    true
                }

                _ => true,
            }
        }

        for func in self.module.funcs.values() {
            let Some(body) = func.body() else { continue };

            for expr in &body.body {
                expr.all(&mut check_expr(&mut func_ids, &mut global_ids));
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

        for global in self.module.globals.values() {
            let GlobalDef::Value(expr) = &global.def else {
                continue;
            };

            expr.all(&mut check_expr(&mut func_ids, &mut global_ids));
        }

        self.module
            .funcs
            .retain(|func_id, _| !func_ids.contains(&func_id));
        self.module
            .globals
            .retain(|global_id, _| !global_ids.contains(&global_id));
        self.module
            .imports
            .retain(|import_id, _| !import_ids.contains(&import_id));
    }

    pub fn insert_func_types(&mut self) {
        for func in self.module.funcs.values() {
            let Some(body) = func.body() else { continue };
            self.module.types.insert(Type::Func(body.ty.clone()));
        }
    }
}
