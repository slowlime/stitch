mod error;
mod frame;
pub mod gc;
mod value;

use std::collections::HashMap;

use crate::ast;
use crate::vm::value::IntoValue;

use self::error::VmError;
use self::frame::Frame;
use self::gc::GarbageCollector;
use self::value::{tag, Class, TypedValue, Value};

pub struct Builtins<'gc> {
    pub array: TypedValue<'gc, tag::Class>,
    pub object: TypedValue<'gc, tag::Class>,
    pub object_class: TypedValue<'gc, tag::Class>,
    pub metaclass: TypedValue<'gc, tag::Class>,
    pub metaclass_class: TypedValue<'gc, tag::Class>,
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

        let class_methods = class
            .class_methods
            .into_iter()
            .map(|method| self.load_method(method))
            .collect::<Result<Vec<_>, _>>()?;
        let object_methods = class
            .object_methods
            .into_iter()
            .map(|method| self.load_method(method))
            .collect::<Result<Vec<_>, _>>()?;

        let class_fields = class.class_fields;
        let object_fields = class.object_fields;

        let metaclass = self.make_class(
            self.builtins().metaclass.clone(),
            Some(self.builtins().object_class.clone()),
            class_methods,
            // TODO: add fields required by Object class
            vec![],
            class_fields.clone(),
        );
        let metaclass = metaclass.into_value(self.gc);

        let cls = self.make_class(
            metaclass,
            Some(superclass),
            object_methods,
            class_fields,
            object_fields,
        );
        let cls = cls.into_value(self.gc);

        todo!()
    }

    fn load_method(
        &mut self,
        method: ast::Method,
    ) -> Result<TypedValue<'gc, tag::Method>, VmError> {
        todo!()
    }

    fn make_class(
        &mut self,
        metaclass: TypedValue<'gc, tag::Class>,
        superclass: Option<TypedValue<'gc, tag::Class>>,
        methods: Vec<TypedValue<'gc, tag::Method>>,
        class_fields: Vec<ast::Name>,
        instance_fields: Vec<ast::Name>,
    ) -> Class<'gc> {
        todo!()
    }
}
