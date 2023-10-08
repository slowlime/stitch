mod error;
mod eval;
mod frame;
pub mod gc;
mod method;
mod value;

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::mem;
use std::num::NonZeroUsize;

use crate::ast;
use crate::ast::visit::{AstRecurse, DefaultVisitor, DefaultVisitorMut};
use crate::location::{Location, Span, Spanned};
use crate::vm::frame::Local;

use self::error::VmError;
use self::eval::Effect;
use self::frame::{Callee, Frame};
use self::gc::{GarbageCollector, GcRefCell};
use self::method::{MethodDef, Primitive};
use self::value::{tag, Class, IntoValue, Method, Object, TypedValue, Value};

pub const RUN_METHOD_NAME: &str = "run";

fn check_method_name_collisions(
    class_method: bool,
    methods: &[ast::Method],
) -> Result<(), VmError> {
    let mut names = HashMap::new();

    for method in methods {
        if let Some(prev_span) = names.insert(
            method.selector.value.to_string(),
            method.selector.location.span(),
        ) {
            return Err(VmError::MethodCollision {
                span: method.selector.location.span(),
                prev_span,
                name: method.selector.value.to_string(),
                class_method,
            });
        }
    }

    Ok(())
}

fn resolve_names(block: &mut ast::Block, fields: &HashSet<&str>) {
    struct NameResolver<'a> {
        fields: &'a HashSet<&'a str>,
    }

    impl DefaultVisitorMut<'_> for NameResolver<'_> {
        fn visit_expr(&mut self, expr: &mut ast::Expr) {
            if let ast::Expr::UnresolvedName(_) = expr {
                let ast::Expr::UnresolvedName(name) = mem::take(expr) else {
                    unreachable!()
                };

                *expr = if self.fields.contains(name.0.value.as_str()) {
                    ast::Expr::Field(ast::Field(name.0))
                } else {
                    ast::Expr::Global(ast::Global(name.0))
                };
            } else {
                expr.recurse_mut(self);
            }
        }
    }

    NameResolver { fields }.visit_block(block)
}

fn resolve_upvalues(code: &mut ast::Block) {
    type UpvalueVec = Vec<String>;

    #[derive(Default)]
    struct UpvalueResolver<'a> {
        frames: Vec<&'a mut UpvalueVec>,
    }

    impl UpvalueResolver<'_> {
        fn capture_var(&mut self, name: &str, up_frames: NonZeroUsize) {
            // frames[frames.len() - 1 - up_frames] is the scope that defines the local
            // so frames[(frames.len() - up_frames)..] would all be capturing the variable
            for upvalues in self.frames.iter_mut().rev().take(up_frames.get()) {
                if !upvalues.iter().any(|upvalue| upvalue == name) {
                    upvalues.push(name.to_owned());
                } else {
                    // only register each upvalue once
                    break;
                }
            }
        }
    }

    impl<'a> DefaultVisitorMut<'a> for UpvalueResolver<'a> {
        fn visit_block(&mut self, block: &'a mut ast::Block) {
            self.frames.push(&mut block.upvalues);

            for stmt in &mut block.body {
                self.visit_stmt(stmt);
            }
        }

        fn visit_field(&mut self, _field: &'a mut ast::Field) {
            if self.frames.len() == 1 {
                // `self` is a local (not an upvalue) -- nothing to do
                return;
            }

            // otherwise field access implicitly captures `self` in the outer scope
            self.capture_var("self", NonZeroUsize::new(self.frames.len() - 1).unwrap());
        }

        fn visit_upvalue(&mut self, upvalue: &'a mut ast::Upvalue) {
            self.capture_var(&upvalue.name.value, upvalue.up_frames);
        }
    }

    assert!(
        code.upvalues.is_empty(),
        "method must not capture anything!"
    );
    UpvalueResolver::default().visit_block(code);
}

fn add_implicit_returns(code: &mut ast::Block) {
    match code.body.last() {
        Some(ast::Stmt::Return(_)) => {}
        Some(ast::Stmt::NonLocalReturn(_)) => unreachable!(),
        Some(ast::Stmt::Dummy) => panic!("Stmt::Dummy in AST"),

        Some(ast::Stmt::Expr(_)) | None => {
            let local = ast::Expr::Local(ast::Local(Spanned::new_builtin("self".into())));
            code.body
                .push(ast::Stmt::Return(Spanned::new_builtin(local)));
        }
    }

    struct BlockReturns;

    impl DefaultVisitorMut<'_> for BlockReturns {
        fn visit_block(&mut self, block: &mut ast::Block) {
            match block.body.last_mut() {
                Some(stmt) => {
                    *stmt = match mem::take(stmt) {
                        ast::Stmt::Return(expr) => ast::Stmt::NonLocalReturn(expr),
                        ast::Stmt::NonLocalReturn(_) => {
                            unreachable!("NonLocalReturn must not be present before this run")
                        }
                        ast::Stmt::Expr(expr) => ast::Stmt::Return(expr),
                        ast::Stmt::Dummy => panic!("Stmt::Dummy in AST"),
                    };
                }

                None => {
                    let nil = ast::Expr::Global(ast::Global(Spanned::new_builtin("nil".into())));
                    block
                        .body
                        .push(ast::Stmt::Return(Spanned::new_builtin(nil)));
                }
            }

            block.recurse_mut(self);
        }
    }

    // add implicit returns in block bodies
    code.recurse_mut(&mut BlockReturns);
}

fn check_method_code(code: &ast::Block, fields: &HashSet<&str>) {
    struct Checker<'a> {
        frame_idx: usize,
        captured_upvalues: &'a [String],
        locals: &'a [Spanned<String>],
        fields: &'a HashSet<&'a str>,
    }

    impl<'a> DefaultVisitor<'a> for Checker<'a> {
        fn visit_unresolved_name(&mut self, name: &'a ast::UnresolvedName) {
            panic!("name {} left unresolved", name.0.value);
        }

        fn visit_block(&mut self, block: &'a ast::Block) {
            self.frame_idx += 1;
            let captured_upvalues = mem::replace(&mut self.captured_upvalues, &block.upvalues);

            for name in &block.upvalues {
                assert!(
                    block.locals.iter().any(|local| local.value == *name)
                        || captured_upvalues.contains(name),
                    "upvalue {name} must either capture an upvalue or a local in the immediately enclosing frame"
                );
            }

            let locals = mem::replace(&mut self.locals, &block.locals);

            block.recurse(self);

            match block.body.last() {
                Some(ast::Stmt::Return(_) | ast::Stmt::NonLocalReturn(_)) => {}
                _ => panic!("block must be terminated with Stmt::Return / Stmt::NonLocalReturn"),
            }

            self.locals = locals;
            self.captured_upvalues = captured_upvalues;
            self.frame_idx -= 1;
        }

        fn visit_stmt(&mut self, stmt: &'a ast::Stmt) {
            match stmt {
                ast::Stmt::NonLocalReturn(_) if self.frame_idx <= 1 => {
                    panic!("Stmt::NonLocalReturn in method body")
                }
                ast::Stmt::Dummy => panic!("Stmt::Dummy in AST"),
                _ => {}
            }
        }

        fn visit_expr(&mut self, expr: &'a ast::Expr) {
            match expr {
                ast::Expr::Dummy => panic!("Expr::Dummy in AST"),
                _ => {}
            }
        }

        fn visit_field(&mut self, field: &'a ast::Field) {
            if self.frame_idx > 1 {
                assert!(
                    self.captured_upvalues.iter().any(|name| name == "self"),
                    "`self` implicitly captured by field access but missing from the upvalue list"
                );
            }

            assert!(
                self.fields.contains(field.0.value.as_str()),
                "unknown field `{}`",
                field.0.value
            );
        }

        fn visit_upvalue(&mut self, upvalue: &'a ast::Upvalue) {
            assert!(
                self.captured_upvalues.contains(&upvalue.name.value),
                "upvalue {} not captured",
                upvalue.name.value
            );
        }

        fn visit_local(&mut self, local: &'a ast::Local) {
            assert!(
                self.locals.iter().any(|name| name.value == local.0.value),
                "unknown local `{}`",
                local.0.value
            );
        }
    }

    assert!(code.upvalues.is_empty(), "method captures a variable");

    let mut checker = Checker {
        frame_idx: 0,
        captured_upvalues: &[],
        locals: &[],
        fields,
    };

    checker.visit_block(code);
}

pub struct Builtins<'gc> {
    pub array: TypedValue<'gc, tag::Class>,
    pub object: TypedValue<'gc, tag::Class>,
    pub object_class: TypedValue<'gc, tag::Class>,
    pub metaclass: TypedValue<'gc, tag::Class>,
    pub metaclass_class: TypedValue<'gc, tag::Class>,
    pub method: TypedValue<'gc, tag::Class>,
    pub nil_object: TypedValue<'gc, tag::Object>,
}

pub struct Vm<'gc> {
    gc: &'gc GarbageCollector,
    globals: HashMap<String, Value<'gc>>,
    frames: Vec<Frame<'gc>>,
    builtins: Option<Builtins<'gc>>,
}

impl<'gc> Vm<'gc> {
    pub fn new(gc: &'gc GarbageCollector) -> Self {
        Self {
            gc,
            globals: Default::default(),
            frames: vec![],
            builtins: None,
        }
    }

    pub fn builtins(&self) -> &Builtins<'gc> {
        self.builtins.as_ref().expect("Builtins not initialized")
    }

    pub fn load_class(
        &mut self,
        class: ast::Class,
    ) -> Result<TypedValue<'gc, tag::Class>, VmError> {
        let superclass = match class.superclass {
            Some(name) => {
                let value = match self.globals.get(&name.value) {
                    Some(value) => value.clone(),
                    None => {
                        return Err(VmError::UndefinedName {
                            span: name.span(),
                            name: name.value,
                        })
                    }
                };

                value.downcast_or_err::<tag::Class>(name.span())?
            }

            None => self.builtins().object.clone(),
        };

        check_method_name_collisions(false, &class.object_methods)?;
        check_method_name_collisions(true, &class.class_methods)?;

        let class_fields = class.class_fields;
        let mut object_fields = superclass.get().instance_fields.clone();
        object_fields.extend(class.object_fields);

        let class_field_set = class_fields
            .iter()
            .map(|name| name.value.as_str())
            .collect();
        let class_methods = class
            .class_methods
            .into_iter()
            .map(|method| self.load_method(method, &class.name.value, &class_field_set))
            .collect::<Result<Vec<_>, _>>()?;

        let object_field_set = object_fields
            .iter()
            .map(|name| name.value.as_str())
            .collect();
        let object_methods = class
            .object_methods
            .into_iter()
            .map(|method| self.load_method(method, &class.name.value, &object_field_set))
            .collect::<Result<Vec<_>, _>>()?;

        // TODO: set fields
        let metaclass = self.make_class(
            Spanned::new(format!("{} class", class.name.value), class.name.location),
            self.builtins().metaclass.clone(),
            Some(self.builtins().object_class.clone()),
            class_methods,
            class_fields,
        );

        for method in &metaclass.get().methods {
            method.get().holder.set(metaclass.clone()).unwrap();
        }

        let cls = self.make_class(
            class.name,
            metaclass,
            Some(superclass),
            object_methods,
            object_fields,
        );

        for method in &cls.get().methods {
            method.get().holder.set(cls.clone()).unwrap();
        }

        self.set_global(cls.get().name.value.clone(), cls.clone().into_value());

        Ok(cls)
    }

    fn load_method(
        &mut self,
        method: ast::Method,
        class_name: &str,
        fields: &HashSet<&str>,
    ) -> Result<TypedValue<'gc, tag::Method>, VmError> {
        let Spanned {
            location: def_location,
            value: def,
        } = method.def;

        let code = match def {
            ast::MethodDef::Block(blk) => MethodDef::Code(self.process_method_code(blk, fields)),
            ast::MethodDef::Primitive => {
                MethodDef::Primitive(self.resolve_primitive(class_name, &method.selector)?)
            }
        };
        let def = Spanned::new(code, def_location);
        let method = self.make_method(method.selector, method.location, def);

        Ok(method)
    }

    fn process_method_code(&self, mut code: ast::Block, fields: &HashSet<&str>) -> ast::Block {
        resolve_names(&mut code, fields);
        resolve_upvalues(&mut code);
        add_implicit_returns(&mut code);

        if cfg!(debug_assertions) {
            check_method_code(&code, fields);
        }

        code
    }

    fn resolve_primitive(
        &self,
        class_name: &str,
        selector: &ast::SpannedSelector,
    ) -> Result<Primitive, VmError> {
        Primitive::from_selector(class_name, &selector.value).ok_or_else(|| {
            VmError::UnknownPrimitive {
                span: selector.location.span(),
                name: selector.value.to_string(),
                class_name: class_name.to_owned(),
            }
        })
    }

    fn make_method(
        &self,
        selector: ast::SpannedSelector,
        location: Location,
        def: Spanned<MethodDef>,
    ) -> TypedValue<'gc, tag::Method> {
        let method = Method {
            selector,
            location,
            obj: Default::default(),
            holder: Default::default(),
            def,
        };

        let value = method.into_value(self.gc);

        // TODO: make sure field assignments are consistent
        let obj = self.make_object(self.builtins().method.clone());
        obj.get().fields.borrow_mut()[obj.get().field_idx("$method").unwrap()] =
            value.clone().into_value();
        value.get().obj.set(obj).unwrap();

        value
    }

    fn make_class(
        &self,
        name: Spanned<String>,
        metaclass: TypedValue<'gc, tag::Class>,
        superclass: Option<TypedValue<'gc, tag::Class>>,
        methods: Vec<TypedValue<'gc, tag::Method>>,
        instance_fields: Vec<ast::Name>,
    ) -> TypedValue<'gc, tag::Class> {
        let method_map = methods
            .iter()
            .enumerate()
            .map(|(idx, method)| (method.get().selector.value.to_string(), idx))
            .collect();
        let instance_field_map = instance_fields
            .iter()
            .enumerate()
            .map(|(idx, field)| (field.value.clone(), idx))
            .collect();

        let cls = Class {
            name,
            obj: Default::default(),
            superclass,
            method_map,
            methods,
            instance_field_map,
            instance_fields,
        };

        let value = cls.into_value(self.gc);

        // TODO: make sure field assignments are consistent
        let obj = self.make_object(metaclass);
        obj.get().fields.borrow_mut()[obj.get().field_idx("$class").unwrap()] =
            value.clone().into_value();
        value.get().obj.set(obj).unwrap();

        value
    }

    fn make_object(&self, class: TypedValue<'gc, tag::Class>) -> TypedValue<'gc, tag::Object> {
        let field_count = class.get().instance_fields.len();
        let mut fields = Vec::with_capacity(field_count);

        for _ in 0..field_count {
            fields.push(self.builtins().nil_object.clone().into_value());
        }

        let obj = Object {
            class,
            fields: GcRefCell::new(fields),
        };

        obj.into_value(self.gc)
    }

    fn set_global(&mut self, name: String, value: Value<'gc>) {
        self.globals.insert(name, value);
    }

    fn get_global(&self, name: &str) -> Option<&Value<'gc>> {
        self.globals.get(name)
    }

    pub fn run(&mut self, class: TypedValue<'gc, tag::Class>) -> Result<Value<'gc>, VmError> {
        let method = match class.get().get_method_by_name(RUN_METHOD_NAME) {
            Some(method) => method.clone(),

            None => {
                return Err(VmError::NoRunMethod {
                    class_span: class.get().name.span(),
                    class_name: class.get().name.value.clone(),
                })
            }
        };

        self.execute_method(method, vec![])
    }

    pub fn execute_method(
        &mut self,
        method: TypedValue<'gc, tag::Method>,
        params: Vec<Value<'gc>>,
    ) -> Result<Value<'gc>, VmError> {
        match method.get().def.value {
            MethodDef::Code(ref block) => {
                self.push_frame(
                    block,
                    None,
                    Callee::Method {
                        method: method.clone(),
                        nlret_valid_flag: Default::default(),
                    },
                    params,
                )?;
                self.execute(block)
            }

            MethodDef::Primitive(p) => self.execute_primitive(p, params),
        }
    }

    pub fn execute_block(
        &mut self,
        block: TypedValue<'gc, tag::Block>,
        params: Vec<Value<'gc>>,
    ) -> Result<Value<'gc>, VmError> {
        todo!()
    }

    fn execute_primitive(
        &mut self,
        p: Primitive,
        params: Vec<Value<'gc>>,
    ) -> Result<Value<'gc>, VmError> {
        todo!()
    }

    // assumes the frame is already pushed
    fn execute(&mut self, block: &ast::Block) -> Result<Value<'gc>, VmError> {
        assert!(
            self.frames.len() == 1,
            "expected exactly one frame on the stack"
        );

        let result = match block.eval(self) {
            Effect::None(_) => panic!("block has no return statement"),
            Effect::Return(value) => Ok(value),
            Effect::NonLocalReturn { value, up_frames } => {
                panic!("NonLocalReturn through top-level frame")
            }
            Effect::Unwind(e) => Err(e),
        };

        self.pop_frame();
        assert!(self.frames.is_empty(), "frame push/pop mismatch");

        result
    }

    fn push_frame(
        &mut self,
        block: &ast::Block,
        dispatch_span: Option<Span>,
        callee: Callee<'gc>,
        params: Vec<Value<'gc>>,
    ) -> Result<(), VmError> {
        match params.len().cmp(&block.params.len()) {
            Ordering::Less => {
                return Err(VmError::NotEnoughArguments {
                    dispatch_span,
                    callee_span: callee.location().span(),
                    callee_name: callee.name().to_string(),
                    expected_count: block.params.len(),
                    provided_count: params.len(),
                    missing_params: block.params[params.len()..].to_vec(),
                })
            }

            Ordering::Greater => {
                return Err(VmError::TooManyArguments {
                    dispatch_span,
                    callee_span: callee.location().span(),
                    callee_name: callee.name().to_string(),
                    expected_count: block.params.len(),
                    provided_count: params.len(),
                })
            }

            Ordering::Equal => {}
        }

        let mut locals = vec![];
        let mut local_map = HashMap::new();

        for (name, value) in block.params.iter().cloned().zip(params) {
            local_map.insert(name.value.clone(), locals.len());
            locals.push(Local {
                name,
                value: GcRefCell::new(value),
            });
        }

        for local in &block.locals {
            local_map.insert(local.value.clone(), locals.len());
            locals.push(Local {
                name: local.clone(),
                value: GcRefCell::new(self.builtins().nil_object.clone().into_value()),
            });
        }

        self.frames.push(Frame {
            callee,
            local_map,
            locals,
        });

        Ok(())
    }

    fn pop_frame(&mut self) {
        todo!("close upvalues and pop the frame off the stack")
    }
}
