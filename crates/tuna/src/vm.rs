pub mod error;
mod eval;
mod frame;
pub mod gc;
mod method;
mod value;

use std::cell::Cell;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fmt::{Display, Write};
use std::mem;
use std::num::NonZeroUsize;
use std::path::Path;
use std::pin::Pin;
use std::ptr;
use std::rc::Rc;
use std::time::{Duration, Instant};

use miette::{Diagnostic, SourceCode};
use thiserror::Error;

use crate::ast;
use crate::ast::visit::{AstRecurse, DefaultVisitor, DefaultVisitorMut};
use crate::file::FileLoader;
use crate::location::{Location, Span, Spanned};
use crate::parse::{parse, ParserOptions};
use crate::sourcemap::{SourceFile, SourceMap};
use crate::vm::value::SomArray;

use self::error::{VmError, VmErrorKind};
use self::eval::Effect;
use self::frame::{Callee, Frame, Local, Upvalue};
use self::gc::{GarbageCollector, Gc, GcOnceCell, GcRefCell};
use self::method::{MethodDef, Primitive};
use self::value::{tag, Block, Class, IntoValue, Method, Object, SomString, SomSymbol, TypedValue, Value};

pub const RUN_METHOD_NAME: &str = "run";
pub const RUN_ARGS_METHOD_NAME: &str = "run:";

#[inline(always)]
fn check_arg_count<'gc>(
    args: &[Value<'gc>],
    params: &[Spanned<String>],
    dispatch_span: Option<Span>,
    callee_span: Option<Span>,
    callee_name: String,
) -> Result<(), VmError> {
    match args.len().cmp(&params.len()) {
        Ordering::Less => Err(VmErrorKind::NotEnoughArguments {
            dispatch_span,
            callee_span,
            callee_name,
            expected_count: params.len(),
            provided_count: args.len(),
            missing_params: params[args.len()..].to_vec(),
        }.into()),

        Ordering::Greater => Err(VmErrorKind::TooManyArguments {
            dispatch_span,
            callee_span,
            callee_name,
            expected_count: params.len(),
            provided_count: args.len(),
        }.into()),

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
            return Err(VmErrorKind::MethodCollision {
                span: method.selector.location.span(),
                prev_span,
                name: method.selector.value.to_string(),
                class_method,
            }.into());
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
            match expr {
                ast::Expr::UnresolvedName(_) => {
                    let ast::Expr::UnresolvedName(name) = mem::take(expr) else {
                        unreachable!()
                    };

                    *expr = if self.fields.contains(name.0.value.as_str()) {
                        ast::Expr::Field(ast::Field(name.0))
                    } else {
                        ast::Expr::Global(ast::Global(name.0))
                    };
                }

                ast::Expr::Assign(assign)
                    if matches!(assign.var, ast::AssignVar::UnresolvedName(_)) =>
                {
                    let ast::Expr::Assign(mut assign) = mem::take(expr) else {
                        unreachable!()
                    };
                    let ast::AssignVar::UnresolvedName(name) = assign.var else {
                        unreachable!()
                    };

                    assign.var = if self.fields.contains(name.0.value.as_str()) {
                        ast::AssignVar::Field(ast::Field(name.0))
                    } else {
                        ast::AssignVar::Global(ast::Global(name.0))
                    };

                    *expr = ast::Expr::Assign(assign);
                }

                _ => {}
            }

            expr.recurse_mut(self);
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
                }
            }
        }

        fn capture_recv(&mut self) {
            if self.frames.len() <= 1 {
                // `self` is a local and does not need to be captured
                return;
            }

            self.capture_var("self", NonZeroUsize::new(self.frames.len() - 1).unwrap());
        }
    }

    impl<'a> DefaultVisitorMut<'a> for UpvalueResolver<'a> {
        fn visit_block(&mut self, block: &'a mut ast::Block) {
            self.frames.push(&mut block.upvalues);

            for stmt in &mut block.body {
                self.visit_stmt(stmt);
            }

            self.frames.pop().expect("empty frame stack");
        }

        fn visit_expr(&mut self, expr: &'a mut ast::Expr) {
            if matches!(expr, ast::Expr::Global(_)) {
                // for dispatching to unknownGlobal:
                self.capture_recv();
            }

            expr.recurse_mut(self);
        }

        fn visit_stmt(&mut self, stmt: &'a mut ast::Stmt) {
            if matches!(stmt, ast::Stmt::NonLocalReturn(_)) {
                // to track nlret validity (and dispatch to escapedBlock: to)
                self.capture_recv();
            }

            stmt.recurse_mut(self);
        }

        fn visit_field(&mut self, _field: &'a mut ast::Field) {
            // field access implicitly captures `self`
            self.capture_recv();
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

#[cfg(debug_assertions)]
fn check_method_code(
    class_name: &str,
    method_name: &str,
    source: &SourceMap,
    code: &ast::Block,
    fields: &HashSet<&str>,
) {
    use miette::{diagnostic, LabeledSpan, MietteDiagnostic};

    struct Checker<'a> {
        frame_idx: usize,
        captured_upvalues: &'a [String],
        locals: &'a [Spanned<String>],
        params: &'a [Spanned<String>],
        fields: &'a HashSet<&'a str>,
        diagnostics: Vec<MietteDiagnostic>,
    }

    impl Checker<'_> {
        fn push_diagnostic(&mut self, diagnostic: MietteDiagnostic) -> &mut MietteDiagnostic {
            self.diagnostics.push(diagnostic);

            self.diagnostics.last_mut().unwrap()
        }

        fn locals_iter(&self) -> impl Iterator<Item = &'_ str> {
            self.locals
                .iter()
                .chain(self.params)
                .map(|local| local.value.as_str())
        }

        fn locals(&self) -> Vec<String> {
            self.locals_iter().map(|name| name.to_owned()).collect()
        }

        fn is_recv_available(&self) -> bool {
            self.frame_idx <= 1 || self.captured_upvalues.iter().any(|name| name == "self")
        }
    }

    impl<'a> DefaultVisitor<'a> for Checker<'a> {
        fn visit_unresolved_name(&mut self, name: &'a ast::UnresolvedName) {
            let mut diagnostic = diagnostic!("name {} left unresolved", name.0.value);

            if let Some(span) = name.0.span() {
                diagnostic = diagnostic.and_label(LabeledSpan::underline(span));
            }

            self.push_diagnostic(diagnostic);
        }

        fn visit_block(&mut self, block: &'a ast::Block) {
            self.frame_idx += 1;
            let captured_upvalues = mem::replace(&mut self.captured_upvalues, &block.upvalues);

            for name in &block.upvalues {
                if !self.locals_iter().any(|local| local == name)
                    && !captured_upvalues.contains(name)
                {
                    let mut diagnostic = diagnostic!(
                        help = format!(
                            "captured upvalues: {:?}; available locals: {:?}",
                            captured_upvalues,
                            self.locals(),
                        ),
                        "declared upvalue `{name}` must capture either an upvalue or a local in the immediately enclosing frame"
                    );

                    if let Some(span) = block.body_location().and_then(|loc| loc.span()) {
                        diagnostic =
                            diagnostic.and_label(LabeledSpan::at(span, "block defined here"));
                    }

                    self.push_diagnostic(diagnostic);
                }
            }

            let locals = mem::replace(&mut self.locals, &block.locals);
            let params = mem::replace(&mut self.params, &block.params);

            block.recurse(self);

            match block.body.last() {
                Some(ast::Stmt::Return(_) | ast::Stmt::NonLocalReturn(_)) => {}

                _ => {
                    let mut diagnostic = diagnostic!(
                        "block must be terminated with Stmt::Return / Stmt::NonLocalReturn"
                    );

                    if let Some(span) = block.body_location().and_then(|loc| loc.span()) {
                        diagnostic =
                            diagnostic.and_label(LabeledSpan::at(span, "block defined here"));
                    }

                    self.push_diagnostic(diagnostic);
                }
            }

            self.locals = locals;
            self.params = params;
            self.captured_upvalues = captured_upvalues;
            self.frame_idx -= 1;
        }

        fn visit_stmt(&mut self, stmt: &'a ast::Stmt) {
            match stmt {
                ast::Stmt::NonLocalReturn(_) if self.frame_idx <= 1 => {
                    let mut diagnostic = diagnostic!("Stmt::NonLocalReturn in method body");

                    if let Some(span) = stmt.location().span() {
                        diagnostic = diagnostic.and_label(LabeledSpan::underline(span));
                    }

                    self.push_diagnostic(diagnostic);
                }

                ast::Stmt::NonLocalReturn(_) if !self.is_recv_available() => {
                    let mut diagnostic = diagnostic!(
                        help = format!("captured upvalues: {:?}", self.captured_upvalues),
                        "Stmt::NonLocalReturn in a block but `self` was not captured"
                    );

                    if let Some(span) = stmt.location().span() {
                        diagnostic = diagnostic.and_label(LabeledSpan::underline(span));
                    }

                    self.push_diagnostic(diagnostic);
                }

                ast::Stmt::Dummy => panic!("Stmt::Dummy in AST"),

                _ => stmt.recurse(self),
            }
        }

        fn visit_expr(&mut self, expr: &'a ast::Expr) {
            match expr {
                ast::Expr::Dummy => panic!("Expr::Dummy in AST"),

                ast::Expr::Global(expr) if !self.is_recv_available() => {
                    let mut diagnostic = diagnostic!(
                        help = format!("captured upvalues: {:?}", self.captured_upvalues),
                        "`self` implicitly captured by a global but missing from the upvalue list"
                    );

                    if let Some(span) = expr.0.span() {
                        diagnostic = diagnostic
                            .and_label(LabeledSpan::at(span, "captured by this global access"));
                    }

                    self.push_diagnostic(diagnostic);
                }

                _ => {}
            }

            expr.recurse(self);
        }

        fn visit_field(&mut self, field: &'a ast::Field) {
            if !self.is_recv_available() {
                let mut diagnostic = diagnostic!(
                    help = format!("captured upvalues: {:?}", self.captured_upvalues),
                    "`self` implicitly captured by field access but missing from the upvalue list"
                );

                if let Some(span) = field.0.span() {
                    diagnostic = diagnostic
                        .and_label(LabeledSpan::at(span, "captured by this field access"));
                }

                self.push_diagnostic(diagnostic);
            }

            if !self.fields.contains(field.0.value.as_str()) {
                let mut diagnostic = diagnostic!(
                    help = format!("available fields: {:?}", self.fields),
                    "unknown field `{}`",
                    field.0.value
                );

                if let Some(span) = field.0.span() {
                    diagnostic = diagnostic.and_label(LabeledSpan::at(span, "field accessed here"));
                }

                self.push_diagnostic(diagnostic);
            }
        }

        fn visit_upvalue(&mut self, upvalue: &'a ast::Upvalue) {
            if !self.captured_upvalues.contains(&upvalue.name.value) {
                let mut diagnostic = diagnostic!(
                    help = format!("captured upvalues: {:?}", self.captured_upvalues),
                    "upvalue `{}` not captured",
                    upvalue.name.value
                );

                if let Some(span) = upvalue.name.span() {
                    diagnostic =
                        diagnostic.and_label(LabeledSpan::at(span, "upvalue accessed here"));
                }

                self.push_diagnostic(diagnostic);
            }
        }

        fn visit_local(&mut self, local: &'a ast::Local) {
            if !self.locals_iter().any(|name| name == local.0.value) {
                let mut diagnostic = diagnostic!(
                    help = format!("available locals: {:?}", self.locals()),
                    "unknown local `{}`",
                    local.0.value
                );

                if let Some(span) = local.0.span() {
                    diagnostic = diagnostic.and_label(LabeledSpan::at(span, "local accessed here"));
                }

                self.push_diagnostic(diagnostic);
            }
        }
    }

    let mut checker = Checker {
        frame_idx: 0,
        captured_upvalues: &[],
        locals: &[],
        params: &[],
        fields,
        diagnostics: vec![],
    };

    if !code.upvalues.is_empty() {
        checker.push_diagnostic(diagnostic!(
            help = format!("upvalue list: {:?}", code.upvalues),
            "method captures a variable"
        ));
    }

    checker.visit_block(code);

    if !checker.diagnostics.is_empty() {
        #[derive(Diagnostic, Error, Debug)]
        #[error("code validation failed (while checking method `{method_name}` of class `{class_name}`)")]
        struct AggregateDiagnostics {
            class_name: String,
            method_name: String,

            #[related]
            diagnostics: Vec<MietteDiagnostic>,
        }

        let report = miette::Report::new(AggregateDiagnostics {
            class_name: class_name.to_owned(),
            method_name: method_name.to_owned(),
            diagnostics: checker.diagnostics,
        })
        .with_source_code(source.clone());

        panic!("{:?}", report);
    }
}

#[derive(Debug, Clone)]
pub struct LoadClassOptions {
    pub allow_nil_superclass: bool,
    pub resolve_superclass: bool,
}

impl Default for LoadClassOptions {
    fn default() -> Self {
        Self {
            allow_nil_superclass: false,
            resolve_superclass: true,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct VmOptions {
    pub print_warnings: bool,
    pub debug: bool,
    pub load_class_options: LoadClassOptions,
    pub parser_options: ParserOptions,
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

macro_rules! vm_debug {
    ($self:expr) => (vm_debug!($self, ""));

    ($self:expr, $( $arg:tt )*) => {
        if $self.options.debug {
            $self.eprint(format_args!("{}\n", format_args!($( $arg )*)));
        }
    };
}

pub(self) use vm_debug;

pub struct Vm<'gc> {
    gc: &'gc GarbageCollector,
    globals: HashMap<String, Value<'gc>>,
    frames: Vec<Frame<'gc>>,
    builtins: Builtins<'gc>,
    upvalues: GcRefCell<Option<Gc<'gc, Upvalue<'gc>>>>,
    start_time: Cell<Instant>,
    stdout: Cell<Option<Box<dyn Write>>>,
    stderr: Cell<Option<Box<dyn Write>>>,
    options: VmOptions,
    load_in_progress: HashSet<String>,
    pub file_loader: Box<dyn FileLoader>,
}

impl<'gc> Vm<'gc> {
    pub fn new(
        gc: &'gc GarbageCollector,
        file_loader: Box<dyn FileLoader>,
        stdout: Box<dyn Write>,
        stderr: Box<dyn Write>,
        options: VmOptions,
    ) -> Self {
        let mut result = Self {
            gc,
            globals: Default::default(),
            frames: vec![],
            builtins: Default::default(),
            upvalues: GcRefCell::new(None),
            start_time: Cell::new(Instant::now()),
            stdout: Cell::new(Some(stdout)),
            stderr: Cell::new(Some(stderr)),
            options,
            load_in_progress: Default::default(),
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
                    resolve_superclass: false,
                },
            )
            .unwrap();
        self.builtins.object_class = self.builtins.object.get().get_metaclass().clone();

        let load_options = LoadClassOptions {
            allow_nil_superclass: false,
            resolve_superclass: false,
        };

        // Class:
        // - superclass: Object
        // - metaclass: Class class
        // Class class:
        // - superclass: Object class
        // - metaclass: <uninit>
        self.builtins.class = self
            .parse_and_load_builtin("Class", load_options.clone())
            .unwrap();

        // Metaclass:
        // - superclass: Object
        // - metaclass: Metaclass class
        // Metaclass class:
        // - superclass: Object class
        // - metaclass: <uninit>
        self.builtins.metaclass = self
            .parse_and_load_builtin("Metaclass", load_options.clone())
            .unwrap();
        self.builtins.metaclass_class = self.builtins.metaclass.get().get_metaclass().clone();

        // set metaclasses
        let uninit_metaclass_classes = [
            &self.builtins.metaclass_class,
            &self.builtins.object_class,
            &self
                .builtins
                .class
                .get()
                .get_obj()
                .get()
                .class
                .get()
                .unwrap(),
        ];

        for class in uninit_metaclass_classes {
            class
                .get()
                .get_obj()
                .get()
                .class
                .set(self.builtins.metaclass.clone())
                .unwrap();
        }

        // [Object class].superclass = Class
        self.builtins
            .object_class
            .get()
            .superclass
            .set(Some(self.builtins.class.clone()))
            .unwrap();

        self.builtins.method = self
            .parse_and_load_builtin("Method", load_options.clone())
            .unwrap();
        self.builtins.primitive = self
            .parse_and_load_builtin("Primitive", load_options.clone())
            .unwrap();

        let uninit_method_classes = [
            &self.builtins.object,
            &self.builtins.metaclass,
            &self.builtins.class,
            &self.builtins.method,
        ];

        fn fix_method_obj_class<'gc>(vm: &Vm<'gc>, method: &TypedValue<'gc, tag::Method>) {
            let method_obj = method.get().obj.get().unwrap();

            let class = match method.get().def.value {
                MethodDef::Code(_) => vm.builtins.method.clone(),
                MethodDef::Primitive { .. } => vm.builtins.primitive.clone(),
            };

            method_obj
                .get()
                .class
                .set(class)
                .unwrap();
        }

        for class in uninit_method_classes {
            for method in &class.get().methods {
                fix_method_obj_class(self, method);
            }

            for method in &class.get().get_metaclass().get().methods {
                fix_method_obj_class(self, method);
            }
        }

        // at this point all object references in the classes created so far should be initialized.

        let nil = self
            .parse_and_load_builtin("Nil", load_options.clone())
            .unwrap();
        self.builtins.nil_object = self.make_object(nil);
        self.set_global("nil".into(), self.builtins.nil_object.clone().into_value());

        self.builtins.array = self
            .parse_and_load_builtin("Array", load_options.clone())
            .unwrap();
        self.builtins.block = self
            .parse_and_load_builtin("Block", load_options.clone())
            .unwrap();
        self.builtins.block1 = self
            .parse_and_load_builtin("Block1", load_options.clone())
            .unwrap();
        self.builtins.block2 = self
            .parse_and_load_builtin("Block2", load_options.clone())
            .unwrap();
        self.builtins.block3 = self
            .parse_and_load_builtin("Block3", load_options.clone())
            .unwrap();
        self.builtins.integer = self
            .parse_and_load_builtin("Integer", load_options.clone())
            .unwrap();
        self.builtins.double = self
            .parse_and_load_builtin("Double", load_options.clone())
            .unwrap();
        self.builtins.string = self
            .parse_and_load_builtin("String", load_options.clone())
            .unwrap();
        self.builtins.symbol = self
            .parse_and_load_builtin("Symbol", load_options.clone())
            .unwrap();

        self.parse_and_load_builtin("Boolean", load_options.clone())
            .unwrap();
        let r#true = self
            .parse_and_load_builtin("True", load_options.clone())
            .unwrap();
        let r#false = self
            .parse_and_load_builtin("False", load_options.clone())
            .unwrap();
        self.builtins.true_object = self.make_object(r#true);
        self.builtins.false_object = self.make_object(r#false);
        self.set_global(
            "true".into(),
            self.builtins.true_object.clone().into_value(),
        );
        self.set_global(
            "false".into(),
            self.builtins.false_object.clone().into_value(),
        );

        let system = self
            .parse_and_load_builtin("System", load_options.clone())
            .unwrap();
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
            .map_err(VmErrorKind::FileLoadError)?;
        let ast = parse(file, Default::default())?;

        self.load_class(ast, options)
    }

    pub fn options(&self) -> &VmOptions {
        &self.options
    }

    pub fn parse_and_load_user_class(
        &mut self,
        class_name: &str,
    ) -> Result<TypedValue<'gc, tag::Class>, VmError> {
        let file = self
            .file_loader
            .load_user_class(class_name)
            .map_err(VmErrorKind::FileLoadError)?;
        let ast = parse(file, self.options.parser_options.clone())?;

        self.load_class(ast, self.options.load_class_options.clone())
    }

    pub fn load_file_as_string(
        &mut self,
        path: &Path,
    ) -> Result<TypedValue<'gc, tag::String>, VmError> {
        let contents = self
            .file_loader
            .load_file(path)
            .map_err(VmErrorKind::FileLoadError)?;

        Ok(self.make_string(contents))
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
        if self.options.debug {
            self.eprint(format_args!("load_class({:?}, {:?})\n", class.name.value, options));
        }

        if !self.load_in_progress.insert(class.name.value.clone()) {
            return Err(VmErrorKind::ClassLoadCycle {
                span: class.name.span(),
                class_name: class.name.value.clone(),
            }.into());
        }

        let superclass = match class.superclass {
            Some(name) => 'superclass: {
                let value = if let Some(value) = self.globals.get(&name.value) {
                    value.clone().downcast_or_err::<tag::Class>(name.span())?
                } else if options.resolve_superclass {
                    self.parse_and_load_user_class(&name.value)
                        .map_err(|e| VmErrorKind::RecursiveLoadError {
                            span: class.name.span(),
                            class_name: class.name.value.clone(),
                            superclass_name: name.value,
                            source: e,
                        })?
                } else if options.allow_nil_superclass {
                    break 'superclass None
                } else {
                    return Err(VmErrorKind::UndefinedName {
                        span: name.span(),
                        name: name.value,
                    }.into())
                };

                Some(value)
            }

            None => Some(self.builtins.object.clone()),
        };

        check_method_name_collisions(false, &class.object_methods)?;
        check_method_name_collisions(true, &class.class_methods)?;

        let superclass_metaclass = superclass
            .as_ref()
            .and_then(|superclass| superclass.checked_get())
            .and_then(|superclass| superclass.obj.get())
            .and_then(|superclass_obj| superclass_obj.checked_get())
            .and_then(|superclass_obj| superclass_obj.class.get())
            .cloned();

         let mut class_fields = superclass_metaclass
            .as_ref()
            .and_then(|superclass_metaclass| superclass_metaclass.checked_get())
            .map(|superclass_metaclass| superclass_metaclass.instance_fields.clone())
            .unwrap_or_default();
        class_fields.extend(class.class_fields);

        let class_field_set = class_fields
            .iter()
            .map(|name| name.value.as_str())
            .collect();
        let class_methods = class
            .class_methods
            .into_iter()
            .map(|method| self.load_method(method, &class.name.value, &class_field_set))
            .collect::<Result<Vec<_>, _>>()?;

        let mut object_fields = superclass
            .as_ref()
            .and_then(|superclass| superclass.checked_get())
            .map(|superclass| superclass.instance_fields.clone())
            .unwrap_or_default();
        object_fields.extend(class.object_fields);

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
            Some(superclass_metaclass.unwrap_or_default()),
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

        assert!(self.load_in_progress.remove(&cls.get().name.value));

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
            ast::MethodDef::Block(blk) => MethodDef::Code(self.process_method_code(
                blk,
                fields,
                #[cfg(debug_assertions)]
                class_name,
                #[cfg(debug_assertions)]
                method.selector.value.name(),
            )),

            ast::MethodDef::Primitive { params } => MethodDef::Primitive {
                primitive: self.resolve_primitive(class_name, &method.selector)?,
                params,
            },
        };

        let def = Spanned::new(code, def_location);
        let method = self.make_method(method.selector, method.location, def);

        Ok(method)
    }

    fn process_method_code(
        &self,
        mut code: ast::Block,
        fields: &HashSet<&str>,
        #[cfg(debug_assertions)] class_name: &str,
        #[cfg(debug_assertions)] method_name: &str,
    ) -> ast::Block {
        resolve_names(&mut code, fields);
        add_implicit_returns(&mut code);
        resolve_upvalues(&mut code);

        #[cfg(debug_assertions)]
        {
            check_method_code(
                class_name,
                method_name,
                self.file_loader.get_source(),
                &code,
                fields,
            );
        }

        code
    }

    fn resolve_primitive(
        &self,
        class_name: &str,
        selector: &ast::SpannedSelector,
    ) -> Result<Primitive, VmError> {
        Primitive::from_selector(class_name, &selector.value).ok_or_else(|| {
            VmErrorKind::UnknownPrimitive {
                span: selector.location.span(),
                name: selector.value.to_string(),
                class_name: class_name.to_owned(),
            }.into()
        })
    }

    fn make_method(
        &self,
        selector: ast::SpannedSelector,
        location: Location,
        def: Spanned<MethodDef>,
    ) -> TypedValue<'gc, tag::Method> {
        let class = match def.value {
            MethodDef::Code(_) => self.builtins.method.clone(),
            MethodDef::Primitive { .. } => self.builtins.primitive.clone(),
        };

        let method = Method {
            selector,
            location,
            obj: Default::default(),
            holder: Default::default(),
            def,
        };

        let value = method.into_value(self.gc);

        let obj = self.make_object(class);
        value.get().obj.set(obj).unwrap();

        vm_debug!(self, "alloc:  method {:?}\n", value.get().selector.value.name());

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

        vm_debug!(self, "alloc:   class {:?}", value.get().name.value);

        value
    }

    fn make_block(
        &mut self,
        defining_method: TypedValue<'gc, tag::Method>,
        code: Spanned<Rc<ast::Block>>,
    ) -> TypedValue<'gc, tag::Block> {
        let upvalues = code
            .value
            .upvalues
            .iter()
            .map(|name| {
                let frame = self.frames.last().unwrap();

                match frame.get_local_by_name(name) {
                    Some(local) => self.capture_local(local),

                    None => match &frame.callee {
                        Callee::Method { .. } => panic!("unknown upvalue `{}`", name),

                        Callee::Block { block, .. } => {
                            match block.get().get_upvalue_by_name(name) {
                                Some(upvalue) => Gc::clone(upvalue),
                                None => panic!("unknown upvalue `{}`", name),
                            }
                        }
                    }
                }
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
            defining_method,
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

        vm_debug!(self, "alloc:   block in {:?}", block.get().defining_method.get().selector.value.name());

        block
    }

    pub fn make_array(&self, values: Vec<Value<'gc>>) -> TypedValue<'gc, tag::Array> {
        vm_debug!(self, "alloc:   array of {} elements (capacity {}, size {} MB)", values.len(), values.capacity(),
            (values.capacity() * mem::size_of::<Value>()) as f64 * 1e-6);
        self.gc.collect();

        SomArray::new(values).into_value(self.gc)
    }

    pub fn make_symbol(&self, sym: impl Into<SomSymbol>) -> TypedValue<'gc, tag::Symbol> {
        let sym = Into::<SomSymbol>::into(sym);
        vm_debug!(self, "alloc:  symbol {}", sym.value.as_str());
        Into::<SomSymbol>::into(sym).into_value(self.gc)
    }

    pub fn make_string(&self, s: impl Into<SomString>) -> TypedValue<'gc, tag::String> {
        let s = Into::<SomString>::into(s);
        vm_debug!(self, "alloc:  string {}", crate::vm::value::StringOps::as_str(&s));
        Into::<SomString>::into(s).into_value(self.gc)
    }

    pub fn make_int(&self, int: i64) -> TypedValue<'gc, tag::Int> {
        vm_debug!(self, "alloc:     int {}", int);
        int.into_value(self.gc)
    }

    pub fn make_float(&self, float: f64) -> TypedValue<'gc, tag::Float> {
        vm_debug!(self, "alloc:   float {}", float);
        float.into_value(self.gc)
    }

    pub fn make_boolean(&self, value: bool) -> TypedValue<'gc, tag::Object> {
        if value {
            self.builtins.true_object.clone()
        } else {
            self.builtins.false_object.clone()
        }
    }

    pub fn make_object(&self, class: TypedValue<'gc, tag::Class>) -> TypedValue<'gc, tag::Object> {
        if self.options.debug {
            if class.is_legal() {
                vm_debug!(self, "alloc:  object of class {}", class.get().name.value);
            } else {
                vm_debug!(self, "alloc:  object of class <uninit>");
            }
        }

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

    pub fn run(
        &mut self,
        class: TypedValue<'gc, tag::Class>,
        run_args: Vec<Value<'gc>>,
    ) -> Result<Value<'gc>, VmError> {
        let recv = self.make_object(class.clone());
        let mut args = vec![recv.into_value()];

        let method = if let Some(method) = class.get().get_method_by_name(RUN_ARGS_METHOD_NAME) {
            args.push(self.make_array(run_args).into_value());

            method
        } else if let Some(method) = class.get().get_method_by_name(RUN_METHOD_NAME) {
            method
        } else {
            return Err(VmErrorKind::NoRunMethod {
                class_span: class.get().name.span(),
                class_name: class.get().name.value.clone(),
            }.into());
        };

        self.execute_method(method.clone(), args)
    }

    pub fn execute_method(
        &mut self,
        method: TypedValue<'gc, tag::Method>,
        args: Vec<Value<'gc>>,
    ) -> Result<Value<'gc>, VmError> {
        assert!(
            self.frames.is_empty(),
            "execute_method called with non-empty frame stack"
        );

        self.start_time.set(Instant::now());

        let result = match method.eval(self, None, args, vec![None]) {
            Effect::None(value) => Ok(value),
            Effect::Unwind(e) => Err(e),
            Effect::Restart => panic!("method execution resulted in a restart"),

            Effect::Return(_) | Effect::NonLocalReturn { .. } => {
                panic!("non-local return through top-level frame")
            }
        };

        assert!(self.frames.is_empty(), "frame push/pop mismatch");

        result
    }

    fn debug_call(&self, dispatch_span: Option<Span>, callee: &Callee, args: &[Value<'gc>]) {
        if self.options.debug {
            vm_debug!(self, " call: {} with arg count {}: {:?}", callee.name(), args.len(), args);

            if let Some(span) = dispatch_span {
                if let Ok(contents) = self.file_loader.get_source().read_span(&span.into(), 0, 0) {
                    vm_debug!(self, "       span: line {}, column {} in file {}",
                        contents.line() + 1,
                        contents.column() + 1,
                        contents.name().unwrap());
                    let span_start_line = contents.line();

                    if let Ok(contents) = self.file_loader.get_source().read_span(&span.into(), 2, 2) {
                        for (i, line) in std::str::from_utf8(contents.data()).unwrap().lines().enumerate() {
                            vm_debug!(self, "           {:>4} {} {}",
                                contents.line() + i + 1,
                                if contents.line() + i == span_start_line { "|" } else { ":" },
                                line);
                        }
                    }
                }
            }

            if matches!(callee, Callee::Method { .. }) {
                vm_debug!(self, "       recv class: {}", args[0].get_class(self).get().name.value);
            }

            for frame in self.frames.iter().rev() {
                vm_debug!(self, "       at: {}", frame.callee.name());
            }
        }
    }

    fn push_frame(
        &mut self,
        block: &ast::Block,
        dispatch_span: Option<Span>,
        callee: Callee<'gc>,
        args: Vec<Value<'gc>>,
    ) -> Result<(), VmError> {
        self.debug_call(dispatch_span, &callee, &args);

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
        let frame = self
            .frames
            .pop()
            .expect("trying to pop an empty frame stack");
        let local_addresses: HashSet<_> =
            frame.locals.iter().map(|local| local as *const _).collect();

        let mut cell = &self.upvalues;
        let mut upvalue_gc: Gc<'gc, Upvalue<'gc>>;

        loop {
            let inner = cell.borrow();
            let Some(upvalue) = &*inner else { break };

            if local_addresses.contains(&(upvalue.get_local() as *const _)) {
                upvalue.close();
                let next = upvalue.next.borrow_mut().take();
                drop(inner);
                *cell.borrow_mut() = next;
            } else {
                let upvalue_clone = Gc::clone(upvalue);
                drop(inner);
                upvalue_gc = upvalue_clone;
                cell = &upvalue_gc.next;
            }
        }
    }

    fn capture_local(&self, local: Pin<&Local<'gc>>) -> Gc<'gc, Upvalue<'gc>> {
        let mut next = self.upvalues.borrow().clone();

        while let Some(upvalue) = next {
            if ptr::eq(upvalue.get_local(), local.clone().get_ref()) {
                return upvalue;
            }

            next = upvalue.next.borrow().clone();
        }

        let upvalue = Gc::new(self.gc, Upvalue::new(local));
        mem::swap(
            &mut *self.upvalues.borrow_mut(),
            &mut *upvalue.next.borrow_mut(),
        );
        *self.upvalues.borrow_mut() = Some(upvalue.clone());

        upvalue
    }

    fn print(&self, msg: impl Display) {
        if let Some(mut stdout) = self.stdout.take() {
            let _ = write!(&mut stdout, "{}", msg);
            self.stdout.set(Some(stdout));
        }
    }

    fn eprint(&self, msg: impl Display) {
        if let Some(mut stderr) = self.stderr.take() {
            let _ = write!(&mut stderr, "{}", msg);
            self.stderr.set(Some(stderr));
        }
    }

    fn full_gc(&self) {
        self.gc.collect()
    }

    fn ticks(&self) -> Duration {
        Instant::now().duration_since(self.start_time.get())
    }
}
