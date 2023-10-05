mod error;
mod frame;
pub mod gc;
mod method;
mod value;

use std::collections::{HashMap, HashSet};
use std::mem;

use crate::ast;
use crate::ast::visit::{AstRecurse, DefaultVisitorMut};
use crate::location::{Location, Spanned};

use self::error::VmError;
use self::frame::Frame;
use self::gc::{GarbageCollector, GcRefCell};
use self::method::{MethodDef, Primitive};
use self::value::{tag, Class, IntoValue, Method, TypedValue, Value, Object};

fn check_method_name_collisions(
    class_method: bool,
    methods: &[ast::Method],
) -> Result<(), VmError> {
    let mut names = HashMap::new();

    for method in methods {
        if let Some(prev_span) =
            names.insert(method.selector.value.to_string(), method.selector.span())
        {
            return Err(VmError::MethodCollision {
                span: method.selector.span(),
                prev_span,
                name: method.selector.value.to_string(),
                class_method,
            });
        }
    }

    Ok(())
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
            *method.get().holder.borrow_mut() = metaclass.clone();
        }

        let cls = self.make_class(
            class.name,
            metaclass,
            Some(superclass),
            object_methods,
            object_fields,
        );

        for method in &cls.get().methods {
            *method.get().holder.borrow_mut() = cls.clone();
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
        struct NameResolver<'a> {
            fields: &'a HashSet<&'a str>,
        }

        impl DefaultVisitorMut for NameResolver<'_> {
            fn visit_expr(&mut self, expr: &mut ast::Expr) {
                if let ast::Expr::UnresolvedName(_) = expr {
                    let ast::Expr::UnresolvedName(name) = mem::take(expr) else { unreachable!() };

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

        // resolve UnresolvedNames now that all information is available
        NameResolver { fields }.visit_block(&mut code);

        // add an implicit return in the method body
        match code.body.last() {
            Some(ast::Stmt::Return(_)) => {},
            Some(ast::Stmt::NonLocalReturn(_)) => unreachable!(),
            Some(ast::Stmt::Dummy) => panic!("Stmt::Dummy in AST"),

            Some(ast::Stmt::Expr(_)) | None => {
                let local = ast::Expr::Local(ast::Local(Spanned::new_builtin("self".into())));
                code.body.push(ast::Stmt::Return(Spanned::new_builtin(local)));
            }
        }

        struct BlockReturns;

        impl DefaultVisitorMut for BlockReturns {
            fn visit_block(&mut self, block: &mut ast::Block) {
                match block.body.last_mut() {
                    Some(stmt) => {
                        *stmt = match mem::take(stmt) {
                            ast::Stmt::Return(expr) => ast::Stmt::NonLocalReturn(expr),
                            ast::Stmt::NonLocalReturn(_) =>
                                unreachable!("NonLocalReturn must not be present before this run"),
                            ast::Stmt::Expr(expr) => ast::Stmt::Return(expr),
                            ast::Stmt::Dummy => panic!("Stmt::Dummy in AST"),
                        };
                    }

                    None => {
                        let nil = ast::Expr::Global(ast::Global(Spanned::new_builtin("nil".into())));
                        block.body.push(ast::Stmt::Return(Spanned::new_builtin(nil)));
                    }
                }

                block.recurse_mut(self);
            }
        }

        // add implicit returns in block bodies
        code.recurse_mut(&mut BlockReturns);

        code
    }

    fn resolve_primitive(
        &self,
        class_name: &str,
        selector: &Spanned<ast::Selector>,
    ) -> Result<Primitive, VmError> {
        Primitive::from_selector(class_name, &selector.value).ok_or_else(|| VmError::UnknownPrimitive {
            span: selector.span(),
            name: selector.value.to_string(),
            class_name: class_name.to_owned(),
        })
    }

    fn make_method(
        &self,
        selector: Spanned<ast::Selector>,
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
        obj.get().fields.borrow_mut()[obj.get().field_idx("$method").unwrap()] = value.clone().into_value();

        *value.get().obj.borrow_mut() = obj;

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
        let cls = Class {
            name,
            // TODO: put an actual object here
            obj: Default::default(),
            superclass,
            methods,
            instance_fields,
        };

        let value = cls.into_value(self.gc);

        // TODO: make sure field assignments are consistent
        let obj = self.make_object(metaclass);
        obj.get().fields.borrow_mut()[obj.get().field_idx("$class").unwrap()] = value.clone().into_value();

        *value.get().obj.borrow_mut() = obj;

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
}
