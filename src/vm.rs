pub mod error;
mod eval;
mod frame;
pub mod gc;
mod method;
mod value;

use std::cell::{Cell, RefCell};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fmt::{Display, Write};
use std::mem;
use std::num::NonZeroUsize;
use std::pin::Pin;
use std::ptr;
use std::time::{Duration, Instant};

use crate::ast;
use crate::ast::visit::{AstRecurse, DefaultVisitor, DefaultVisitorMut};
use crate::file::FileLoader;
use crate::location::{Location, Span, Spanned};
use crate::parse::{parse, ParserOptions};
use crate::sourcemap::SourceFile;
use crate::vm::frame::Local;
use crate::vm::value::Block;

use self::error::VmError;
use self::eval::Effect;
use self::frame::{Callee, Frame, Upvalue};
use self::gc::{GarbageCollector, Gc, GcOnceCell, GcRefCell};
use self::method::{MethodDef, Primitive};
use self::value::{tag, Class, IntoValue, Method, Object, SomString, TypedValue, Value};

pub const RUN_METHOD_NAME: &str = "run";

#[inline(always)]
fn check_arg_count<'gc>(
    args: &[Value<'gc>],
    params: &[Spanned<String>],
    dispatch_span: Option<Span>,
    callee_span: Option<Span>,
    callee_name: String,
) -> Result<(), VmError> {
    match args.len().cmp(&params.len()) {
        Ordering::Less => {
            return Err(VmError::NotEnoughArguments {
                dispatch_span,
                callee_span,
                callee_name,
                expected_count: params.len(),
                provided_count: args.len(),
                missing_params: params[args.len()..].to_vec(),
            })
        }

        Ordering::Greater => {
            return Err(VmError::TooManyArguments {
                dispatch_span,
                callee_span,
                callee_name,
                expected_count: params.len(),
                provided_count: args.len(),
            })
        }

        Ordering::Equal => Ok(()),
    }
}

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

        fn visit_stmt(&mut self, stmt: &'a mut ast::Stmt) {
            match stmt {
                ast::Stmt::NonLocalReturn(_) if self.frames.len() > 1 => {
                    // capture `self` -- needed for non-local returns
                    self.capture_var("self", NonZeroUsize::new(self.frames.len() - 1).unwrap());
                }

                _ => {}
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
                    block.locals.iter().chain(&block.params).any(|local| local.value == *name)
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

                ast::Stmt::NonLocalReturn(_)
                    if !self.captured_upvalues.iter().any(|name| name == "self") =>
                {
                    panic!("Stmt::NonLocalReturn in a block but `self` was not captured");
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

pub struct LoadClassOptions {
    pub allow_nil_superclass: bool,
}

impl Default for LoadClassOptions {
    fn default() -> Self {
        Self {
            allow_nil_superclass: false,
        }
    }
}

#[derive(Default)]
pub struct Builtins<'gc> {
    pub object: TypedValue<'gc, tag::Class>,
    pub object_class: TypedValue<'gc, tag::Class>,
    pub class: TypedValue<'gc, tag::Class>,
    pub metaclass: TypedValue<'gc, tag::Class>,
    pub metaclass_class: TypedValue<'gc, tag::Class>,
    pub method: TypedValue<'gc, tag::Class>,
    pub nil_object: TypedValue<'gc, tag::Object>,
    pub array: TypedValue<'gc, tag::Class>,
    pub block: TypedValue<'gc, tag::Class>,
    pub block1: TypedValue<'gc, tag::Class>,
    pub block2: TypedValue<'gc, tag::Class>,
    pub block3: TypedValue<'gc, tag::Class>,
    pub integer: TypedValue<'gc, tag::Class>,
    pub double: TypedValue<'gc, tag::Class>,
    pub symbol: TypedValue<'gc, tag::Class>,
    pub primitive: TypedValue<'gc, tag::Class>,
    pub string: TypedValue<'gc, tag::Class>,
    pub true_object: TypedValue<'gc, tag::Object>,
    pub false_object: TypedValue<'gc, tag::Object>,
}

pub struct Vm<'gc> {
    gc: &'gc GarbageCollector,
    globals: HashMap<String, Value<'gc>>,
    frames: Vec<Frame<'gc>>,
    builtins: Builtins<'gc>,
    upvalues: RefCell<Option<Gc<'gc, Upvalue<'gc>>>>,
    start_time: Cell<Instant>,
    stdout: Box<dyn Write>,
    stderr: Box<dyn Write>,
    pub file_loader: Box<dyn FileLoader>,
}

impl<'gc> Vm<'gc> {
    pub fn new(
        gc: &'gc GarbageCollector,
        file_loader: Box<dyn FileLoader>,
        stdout: Box<dyn Write>,
        stderr: Box<dyn Write>,
    ) -> Self {
        let mut result = Self {
            gc,
            globals: Default::default(),
            frames: vec![],
            builtins: Default::default(),
            upvalues: RefCell::new(None),
            start_time: Cell::new(Instant::now()),
            stdout,
            stderr,
            file_loader,
        };

        result.initialize();

        result
    }

    fn initialize(&mut self) {
        // Object:
        // - superclass: None
        // - metaclass: Object class
        // Object class:
        // - superclass: <uninit>
        // - metaclass: <uninit>
        self.builtins.object = self
            .parse_and_load_builtin(
                "Object",
                LoadClassOptions {
                    allow_nil_superclass: true,
                    ..Default::default()
                },
            )
            .unwrap();
        self.builtins.object_class = self.builtins.object.get().get_metaclass().clone();

        // Class:
        // - superclass: Object
        // - metaclass: Class class
        // Class class:
        // - superclass: Object class
        // - metaclass: <uninit>
        self.builtins.class = self
            .parse_and_load_builtin("Class", Default::default())
            .unwrap();

        // Metaclass:
        // - superclass: Object
        // - metaclass: Metaclass class
        // Metaclass class:
        // - superclass: Object class
        // - metaclass: <uninit>
        self.builtins.metaclass = self
            .parse_and_load_builtin("Metaclass", Default::default())
            .unwrap();
        self.builtins.metaclass_class = self.builtins.metaclass.get().get_metaclass().clone();

        // set metaclasses
        let uninit_metaclass_classes = [
            &self.builtins.metaclass_class,
            &self.builtins.object_class,
            &self.builtins.class.get().get_obj().get().class.get().unwrap(),
        ];

        for class in uninit_metaclass_classes {
            class.get().get_obj().get().class.set(self.builtins.metaclass.clone()).unwrap();
        }

        // [Object class].superclass = Class
        self.builtins
            .object_class
            .get()
            .superclass
            .set(Some(self.builtins.class.clone()))
            .unwrap();

        self.builtins.method = self
            .parse_and_load_builtin("Method", Default::default())
            .unwrap();

        let uninit_method_classes = [
            &self.builtins.object,
            &self.builtins.metaclass,
            &self.builtins.class,
            &self.builtins.method,
        ];

        for class in uninit_method_classes {
            for method in &class.get().methods {
                let method_obj = method.get().obj.get().unwrap();
                method_obj
                    .get()
                    .class
                    .set(self.builtins.method.clone())
                    .unwrap();
            }

            for method in &class.get().get_metaclass().get().methods {
                let method_obj = method.get().obj.get().unwrap();
                method_obj
                    .get()
                    .class
                    .set(self.builtins.method.clone())
                    .unwrap();
            }
        }

        // at this point all object references in the classes created so far should be initialized.

        let nil = self
            .parse_and_load_builtin("Nil", Default::default())
            .unwrap();
        self.builtins.nil_object = self.make_object(nil);

        self.builtins.array = self
            .parse_and_load_builtin("Array", Default::default())
            .unwrap();
        self.builtins.block = self
            .parse_and_load_builtin("Block", Default::default())
            .unwrap();
        self.builtins.block1 = self
            .parse_and_load_builtin("Block1", Default::default())
            .unwrap();
        self.builtins.block2 = self
            .parse_and_load_builtin("Block2", Default::default())
            .unwrap();
        self.builtins.block3 = self
            .parse_and_load_builtin("Block3", Default::default())
            .unwrap();
        self.builtins.integer = self
            .parse_and_load_builtin("Integer", Default::default())
            .unwrap();
        self.builtins.double = self
            .parse_and_load_builtin("Double", Default::default())
            .unwrap();
        self.builtins.string = self
            .parse_and_load_builtin("String", Default::default())
            .unwrap();
        self.builtins.symbol = self
            .parse_and_load_builtin("Symbol", Default::default())
            .unwrap();
        self.builtins.primitive = self
            .parse_and_load_builtin("Primitive", Default::default())
            .unwrap();

        self.parse_and_load_builtin("Boolean", Default::default()).unwrap();
        let r#true = self
            .parse_and_load_builtin("True", Default::default())
            .unwrap();
        let r#false = self
            .parse_and_load_builtin("False", Default::default())
            .unwrap();
        self.builtins.true_object = self.make_object(r#true);
        self.builtins.false_object = self.make_object(r#false);
        self.set_global("true".into(), self.builtins.true_object.clone().into_value());
        self.set_global("false".into(), self.builtins.false_object.clone().into_value());

        let system = self.parse_and_load_builtin("System", Default::default()).unwrap();
        self.set_global("system".into(), self.make_object(system).into_value());
    }

    fn parse_and_load_builtin(
        &mut self,
        class_name: &str,
        options: LoadClassOptions,
    ) -> Result<TypedValue<'gc, tag::Class>, VmError> {
        let file = self
            .file_loader
            .load_builtin_class(class_name)
            .map_err(VmError::FileLoadError)?;
        let ast = parse(file, Default::default())?;

        self.load_class(ast, options)
    }

    pub fn parse_and_load_user_class(
        &mut self,
        class_name: &str,
    ) -> Result<TypedValue<'gc, tag::Class>, VmError> {
        let file = self
            .file_loader
            .load_user_class(class_name)
            .map_err(VmError::FileLoadError)?;
        let ast = parse(file, Default::default())?;

        self.load_class(ast, Default::default())
    }

    pub fn parse_and_load_class(
        &mut self,
        file: &SourceFile,
        options: ParserOptions,
    ) -> Result<TypedValue<'gc, tag::Class>, VmError> {
        self.load_class(parse(file, options)?, Default::default())
    }

    pub fn load_class(
        &mut self,
        class: ast::Class,
        options: LoadClassOptions,
    ) -> Result<TypedValue<'gc, tag::Class>, VmError> {
        let superclass = match class.superclass {
            Some(name) => 'superclass: {
                let value = match self.globals.get(&name.value) {
                    Some(value) => value.clone(),
                    None if options.allow_nil_superclass => break 'superclass None,

                    None => {
                        return Err(VmError::UndefinedName {
                            span: name.span(),
                            name: name.value,
                        })
                    }
                };

                Some(value.downcast_or_err::<tag::Class>(name.span())?)
            }

            None => Some(self.builtins.object.clone()),
        };

        check_method_name_collisions(false, &class.object_methods)?;
        check_method_name_collisions(true, &class.class_methods)?;

        let class_fields = class.class_fields;
        let mut object_fields = superclass
            .as_ref()
            .and_then(|superclass| superclass.checked_get())
            .map(|superclass| superclass.instance_fields.clone())
            .unwrap_or_default();
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

        let metaclass = self.make_class(
            Spanned::new(format!("{} class", class.name.value), class.name.location),
            self.builtins.metaclass.clone(),
            Some(self.builtins.object_class.clone()),
            class_methods,
            class_fields,
        );

        for method in &metaclass.get().methods {
            method.get().holder.set(metaclass.clone()).unwrap();
        }

        let cls = self.make_class(
            class.name,
            metaclass,
            superclass,
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
            ast::MethodDef::Primitive { params } => MethodDef::Primitive {
                primitive: self.resolve_primitive(class_name, &method.selector)?,
                params,
            },
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

        let obj = self.make_object(self.builtins.method.clone());
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
            superclass: match superclass {
                Some(ref class) if class.is_legal() => GcOnceCell::new_init(superclass),
                Some(_) => GcOnceCell::new(),
                None => GcOnceCell::new_init(None),
            },
            method_map,
            methods,
            instance_field_map,
            instance_fields,
        };

        let value = cls.into_value(self.gc);

        let obj = self.make_object(metaclass);
        value.get().obj.set(obj).unwrap();

        value
    }

    fn make_block(&mut self, code: Spanned<ast::Block>) -> TypedValue<'gc, tag::Block> {
        let upvalues = code
            .value
            .upvalues
            .iter()
            .map(|name| {
                let frame = self.frames.last().unwrap();
                let local = frame
                    .get_local_by_name(name)
                    .unwrap_or_else(|| match &frame.callee {
                        Callee::Method { .. } => panic!("unknown upvalue `{}`", name),
                        Callee::Block { block } => match block.get().get_upvalue_by_name(name) {
                            Some(upvalue) => upvalue.get_local(),
                            None => panic!("unknown upvalue `{}`", name),
                        },
                    });

                self.capture_local(local)
            })
            .collect::<Vec<_>>();

        let upvalue_map = upvalues
            .iter()
            .enumerate()
            .map(|(idx, upvalue)| (upvalue.get_local().name.value.clone(), idx))
            .collect();

        let block = Block {
            location: code.location,
            obj: Default::default(),
            code: code.value,
            upvalue_map,
            upvalues,
        };

        let class = match block.code.params.len() {
            0 => self.builtins.block1.clone(),
            1 => self.builtins.block2.clone(),
            2 => self.builtins.block3.clone(),
            _ => self.builtins.block.clone(),
        };

        let block = block.into_value(self.gc);
        let obj = self.make_object(class);
        block.get().obj.set(obj).unwrap();

        block
    }

    fn make_array(&self, values: Vec<Value<'gc>>) -> TypedValue<'gc, tag::Array> {
        GcRefCell::new(values).into_value(self.gc)
    }

    fn make_symbol(&self, sym: ast::SymbolLit) -> TypedValue<'gc, tag::Symbol> {
        sym.into_value(self.gc)
    }

    fn make_string(&self, s: impl Into<SomString>) -> TypedValue<'gc, tag::String> {
        Into::<SomString>::into(s).into_value(self.gc)
    }

    fn make_int(&self, int: i64) -> TypedValue<'gc, tag::Int> {
        int.into_value(self.gc)
    }

    fn make_float(&self, float: f64) -> TypedValue<'gc, tag::Float> {
        float.into_value(self.gc)
    }

    fn make_boolean(&self, value: bool) -> TypedValue<'gc, tag::Object> {
        if value {
            self.builtins.true_object.clone()
        } else {
            self.builtins.false_object.clone()
        }
    }

    fn make_object(&self, class: TypedValue<'gc, tag::Class>) -> TypedValue<'gc, tag::Object> {
        let field_count = class
            .checked_get()
            .map(|class| class.instance_fields.len())
            .unwrap_or(0);
        let mut fields = Vec::with_capacity(field_count);

        for _ in 0..field_count {
            fields.push(self.builtins.nil_object.clone().into_value());
        }

        let obj = Object {
            class: if class.is_legal() {
                GcOnceCell::new_init(class)
            } else {
                GcOnceCell::new()
            },
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

        let recv = self.make_object(class.clone());
        self.execute_method(method, vec![recv.into_value()])
    }

    pub fn execute_method(
        &mut self,
        method: TypedValue<'gc, tag::Method>,
        args: Vec<Value<'gc>>,
    ) -> Result<Value<'gc>, VmError> {
        assert!(
            self.frames.len() == 0,
            "execute_method called with non-empty frame stack"
        );

        self.start_time.set(Instant::now());

        let result = match method.eval(self, None, args, vec![None]) {
            Effect::None(value) => Ok(value),
            Effect::Unwind(e) => Err(e),

            Effect::Return(_) | Effect::NonLocalReturn { .. } => {
                panic!("non-local return through top-level frame")
            }
        };

        assert!(self.frames.is_empty(), "frame push/pop mismatch");

        result
    }

    fn push_frame(
        &mut self,
        block: &ast::Block,
        dispatch_span: Option<Span>,
        callee: Callee<'gc>,
        args: Vec<Value<'gc>>,
    ) -> Result<(), VmError> {
        check_arg_count(
            &args,
            &block.params,
            dispatch_span,
            callee.location().span(),
            callee.name().to_string(),
        )?;

        let mut locals = Vec::with_capacity(args.len() + block.locals.len());
        let mut local_map = HashMap::new();

        for (name, value) in block.params.iter().cloned().zip(args) {
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
                value: GcRefCell::new(self.builtins.nil_object.clone().into_value()),
            });
        }

        self.frames.push(Frame {
            callee,
            local_map,
            locals: Pin::new(locals.into_boxed_slice()),
        });

        Ok(())
    }

    fn pop_frame(&mut self) {
        let frame = self.frames.pop().expect("trying to pop an empty frame stack");
        let local_addresses: HashSet<_> = frame.locals.iter().map(|local| local as *const _).collect();

        let mut maybe_upvalue = self.upvalues.borrow().clone();

        while let Some(upvalue) = maybe_upvalue {
            if local_addresses.contains(&(upvalue.get_local() as *const _)) {
                upvalue.close();
            }

            maybe_upvalue = upvalue.next.borrow().clone();
        }
    }

    fn capture_local(&self, local: &Local<'gc>) -> Gc<'gc, Upvalue<'gc>> {
        let mut next = self.upvalues.borrow().clone();

        while let Some(upvalue) = next {
            if ptr::eq(upvalue.get_local(), local) {
                return upvalue;
            }

            next = upvalue.next.borrow().clone();
        }

        let upvalue = Gc::new(self.gc, unsafe { Upvalue::new(local) });
        mem::swap(
            &mut *self.upvalues.borrow_mut(),
            &mut *upvalue.next.borrow_mut(),
        );
        *self.upvalues.borrow_mut() = Some(upvalue.clone());

        upvalue
    }

    fn print(&mut self, msg: impl Display) {
        let _ = write!(&mut self.stdout, "{}", msg);
    }

    fn eprint(&mut self, msg: impl Display) {
        let _ = write!(&mut self.stderr, "{}", msg);
    }

    fn full_gc(&self) {
        self.gc.collect()
    }

    fn ticks(&self) -> Duration {
        Instant::now().duration_since(self.start_time.get())
    }
}
