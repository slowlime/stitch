mod concrete;
mod spec;

use std::collections::HashMap;
use std::rc::Rc;

use anyhow::{bail, ensure, Result};
use log::trace;
use slotmap::SparseSecondaryMap;

use crate::ast::expr::{Value, ValueAttrs};
use crate::ast::ty::ValType;
use crate::ast::{self, ExportDef, Func, FuncId, Module};
use crate::cfg::FuncBody;
use crate::interp::spec::Specializer;

pub const START_FUNC_NAME: &str = "stitch-start";

fn format_arg_list(args: impl Iterator<Item = Option<ValType>>) -> String {
    format!(
        "[{}]",
        args.map(|arg| match arg {
            Some(val_ty) => val_ty.to_string(),
            None => "_".to_owned(),
        })
        .reduce(|lhs, rhs| format!("{lhs}, {rhs}"))
        .unwrap_or_default()
    )
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SpecSignature {
    orig_func_id: FuncId,
    args: Vec<Option<(Value, ValueAttrs)>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpecializedFunc {
    Pending(FuncId),
    Finished(FuncId),
}

impl SpecializedFunc {
    pub fn func_id(&self) -> FuncId {
        match *self {
            Self::Pending(func_id) | Self::Finished(func_id) => func_id,
        }
    }
}

pub struct Interpreter<'a> {
    module: &'a mut Module,
    spec_sigs: HashMap<SpecSignature, SpecializedFunc>,
    spec_funcs: SparseSecondaryMap<FuncId, SpecSignature>,
    cfgs: SparseSecondaryMap<FuncId, Rc<FuncBody>>,
}

impl<'a> Interpreter<'a> {
    pub fn new(module: &'a mut Module) -> Self {
        Self {
            module,
            spec_sigs: Default::default(),
            spec_funcs: Default::default(),
            cfgs: Default::default(),
        }
    }

    pub fn process(mut self) -> Result<bool> {
        let Some(export) = self
            .module
            .exports
            .values()
            .find(|export| export.name == START_FUNC_NAME)
        else {
            return Ok(false);
        };

        let ExportDef::Func(start_func_id) = export.def else {
            bail!("{} is not a function", START_FUNC_NAME);
        };

        let func_ty = self.module.funcs[start_func_id].ty();

        if !func_ty.params.is_empty() || func_ty.ret.is_some() {
            bail!("{} has a wrong type: expected [] -> []", START_FUNC_NAME);
        }

        self.interpret(start_func_id, vec![])?;
        self.module.remove_func(start_func_id);

        Ok(true)
    }

    pub fn specialize(
        &mut self,
        func_id: FuncId,
        args: Vec<Option<(Value, ValueAttrs)>>,
    ) -> Result<FuncId> {
        let func = &self.module.funcs[func_id];

        ensure!(
            args.len() == func.ty().params.len(),
            "invalid number of arguments for function: expected {}, got {}",
            func.ty().params.len(),
            args.len(),
        );
        ensure!(
            args.iter()
                .zip(&func.ty().params)
                .filter_map(|(arg, param_ty)| arg.map(|arg| (arg, param_ty)))
                .all(|((value, _), param_ty)| value.val_ty() == *param_ty),
            "invalid arguments for function: expected {}, got {}",
            format_arg_list(func.ty().params.iter().cloned().map(Some)),
            format_arg_list(args.iter().map(|arg| arg.map(|(value, _)| value.val_ty()))),
        );

        let mut spec_sig = SpecSignature {
            orig_func_id: func_id,
            args,
        };

        if let Some(&spec) = self.spec_sigs.get(&spec_sig) {
            return Ok(spec.func_id());
        }

        if let Some(pending_sig) = self.spec_funcs.get(func_id) {
            spec_sig.orig_func_id = pending_sig.orig_func_id;

            let mut args = spec_sig.args.into_iter();
            spec_sig.args = pending_sig
                .args
                .iter()
                .map(move |arg| match arg {
                    Some(_) => arg.clone(),
                    None => args.next().unwrap(),
                })
                .collect();

            return self.specialize(spec_sig.orig_func_id, spec_sig.args);
        }

        let Some(func) = func.body() else {
            bail!("cannot specialize an imported function");
        };

        let func = &*self
            .cfgs
            .entry(func_id)
            .unwrap()
            .or_insert_with(|| Rc::new(FuncBody::from_ast(self.module, func)));

        let mut func_ty = func.ty.clone();
        func_ty.params.retain({
            let mut iter = spec_sig.args.iter();
            move |_| iter.next().unwrap().is_none()
        });

        let body = FuncBody::new(func_ty);
        let func_id = self
            .module
            .funcs
            .insert(Func::Body(ast::FuncBody::new(body.ty.clone())));
        trace!(
            "specializing {:?} as {func_id:?}: {}",
            spec_sig.orig_func_id,
            format_arg_list(
                spec_sig
                    .args
                    .iter()
                    .map(|arg| arg.map(|(value, _)| value.val_ty()))
            ),
        );
        self.spec_sigs
            .insert(spec_sig.clone(), SpecializedFunc::Pending(func_id));
        self.spec_funcs.insert(func_id, spec_sig.clone());

        let func = Rc::clone(func);
        let body = Specializer::new(self, spec_sig.clone(), func).run()?;

        *self.module.funcs[func_id].body_mut().unwrap() = body.to_ast();
        *self.spec_sigs.get_mut(&spec_sig).unwrap() = SpecializedFunc::Finished(func_id);

        Ok(func_id)
    }
}
