use std::collections::HashMap;
use std::fmt::{self, Display};
use std::ops::{Deref, DerefMut};
use std::marker::PhantomData;
use std::rc::Weak;

use crate::ast::{self, SymbolLit as Symbol};
use crate::impl_collect;
use crate::location::{Location, Span, Spanned};

use super::error::VmError;
use super::frame::Upvalue;
use super::gc::{GcRefCell, GcRef, GcRefMut, GcOnceCell};
use super::gc::{Collect, Gc};
use super::gc::{Finalize, GarbageCollector};
use super::method::MethodDef;

#[derive(Debug, Clone, Default)]
pub struct Value<'gc>(Option<Gc<'gc, ValueKind<'gc>>>);

impl<'gc> Value<'gc> {
    pub fn new_illegal() -> Self {
        Self::default()
    }

    pub fn is_legal(&self) -> bool {
        self.0.is_some()
    }

    pub fn ensure_legal(&self) -> &Self {
        assert!(self.0.is_some(), "Value is illegal");

        self
    }

    pub fn is<T: Tag>(&self) -> bool {
        match self.0 {
            Some(ref gc) => T::is_tag_of(gc),
            None => true,
        }
    }

    pub fn downcast<T: Tag>(self) -> Option<TypedValue<'gc, T>> {
        if self.is::<T>() {
            Some(unsafe { TypedValue::new(self) })
        } else {
            None
        }
    }

    pub fn downcast_or_err<T: Tag>(
        self,
        span: impl Into<Option<Span>>,
    ) -> Result<TypedValue<'gc, T>, VmError> {
        if self.is::<T>() {
            Ok(unsafe { TypedValue::new(self) })
        } else {
            Err(VmError::IllegalTy {
                span: span.into(),
                expected: [T::TY].as_slice().into(),
                actual: self.0.as_ref().unwrap().ty(),
            })
        }
    }

    pub fn get_obj(&self) -> Option<&Object<'gc>> {
        self.0.as_ref()?.get_obj()
    }
}

impl Finalize for Value<'_> {}

unsafe impl Collect for Value<'_> {
    impl_collect! {
        fn visit(&self) {
            visit(&self.0);
        }
    }
}

pub trait IntoValue<'gc> {
    type Tag: Tag;

    fn into_value(self, gc: &'gc GarbageCollector) -> TypedValue<'gc, Self::Tag>;
}

macro_rules! define_value_kind {
    {
        $( #[ $attr:meta ] )*
        pub enum ValueKind<'gc> {
            $( $name:ident ($display:literal, $ty:ty => $value:ty): $pat:pat => $arm:expr, )+
        }
    } => {
        $( #[ $attr ] )*
        pub enum ValueKind<'gc> {
            $( $name($ty), )+
        }

        impl ValueKind<'_> {
            pub fn ty(&self) -> Ty {
                match self {
                    $( Self::$name(_) => Ty::$name, )+
                }
            }
        }

        mod private {
            pub trait Sealed {}
        }

        pub mod tag {
            $(
                #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
                pub struct $name;

                impl super::private::Sealed for $name {}
            )+
        }

        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum Ty {
            $( $name, )+
        }

        impl Display for Ty {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self {
                    $( Self::$name => $display.fmt(f), )+
                }
            }
        }

        pub trait Tag: private::Sealed {
            type Value<'a, 'gc: 'a>;

            const TY: Ty;

            fn is_tag_of(val: &ValueKind<'_>) -> bool;

            fn get<'a, 'gc>(val: &'a ValueKind<'gc>) -> Self::Value<'a, 'gc>;

            unsafe fn get_unchecked<'a, 'gc>(val: &'a ValueKind<'gc>) -> Self::Value<'a, 'gc>;
        }

        $(
            impl Tag for tag::$name {
                type Value<'a, 'gc: 'a> = $value;

                const TY: Ty = Ty::$name;

                fn is_tag_of(val: &ValueKind) -> bool {
                    matches!(val, ValueKind::$name(_))
                }

                fn get<'a, 'gc>(val: &'a ValueKind<'gc>) -> Self::Value<'a, 'gc> {
                    match *val {
                        ValueKind::$name($pat) => $arm,
                        _ => panic!("invalid tag"),
                    }
                }

                unsafe fn get_unchecked<'a, 'gc>(val: &'a ValueKind<'gc>) -> Self::Value<'a, 'gc> {
                    match *val {
                        ValueKind::$name($pat) => $arm,
                        _ => std::hint::unreachable_unchecked(),
                    }
                }
            }

            impl<'gc> IntoValue<'gc> for $ty {
                type Tag = tag::$name;

                fn into_value(self, gc: &'gc GarbageCollector) -> TypedValue<'gc, Self::Tag> {
                    TypedValue(Value(Some(Gc::new(gc, ValueKind::$name(self)))), PhantomData)
                }
            }
        )+
    }
}

define_value_kind! {
    #[derive(Debug)]
    pub enum ValueKind<'gc> {
        Int("integer", i64 => i64): i => i,
        Float("double", f64 => f64): f => f,
        Array("array", GcRefCell<Vec<Value<'gc>>> => &'a GcRefCell<Vec<Value<'gc>>>): ref arr => arr,
        Block("block", Block<'gc> => &'a Block<'gc>): ref blk => blk,
        Class("class", Class<'gc> => &'a Class<'gc>): ref cls => cls,
        Method("method", Method<'gc> => &'a Method<'gc>): ref method => method,
        Symbol("symbol", Symbol => &'a Symbol): ref sym => sym,
        Object("object", Object<'gc> => &'a Object<'gc>): ref obj => obj,
        String("string", String => &'a str): ref s => s,
    }
}

impl<'gc> ValueKind<'gc> {
    pub fn get_obj(&self) -> Option<&Object<'gc>> {
        match self {
            Self::Int(_) => None,
            Self::Float(_) => None,
            Self::Array(_) => None,
            Self::Block(block) => block.obj.get().map(|obj| obj.get()),
            Self::Class(class) => class.obj.get().map(|obj| obj.get()),
            Self::Method(method) => method.obj.get().map(|obj| obj.get()),
            Self::Symbol(_) => None,
            Self::Object(obj) => Some(obj),
            Self::String(_) => None,
        }
    }
}

impl Finalize for ValueKind<'_> {}

unsafe impl Collect for ValueKind<'_> {
    impl_collect! {
        fn visit(&self) {
            match self {
                Self::Array(arr) => visit(arr),
                Self::Block(block) => visit(block),
                Self::Class(class) => visit(class),
                Self::Method(method) => visit(method),
                Self::Object(obj) => visit(obj),

                Self::Int(_) | Self::Float(_) | Self::String(_) | Self::Symbol(_) => {},
            }
        }
    }
}

#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct TypedValue<'gc, T: Tag>(Value<'gc>, PhantomData<T>);

impl<'gc, T: Tag> TypedValue<'gc, T> {
    pub unsafe fn new(val: Value<'gc>) -> Self {
        Self(val, PhantomData)
    }

    pub fn new_illegal() -> Self {
        Default::default()
    }

    pub fn is_legal(&self) -> bool {
        self.0.is_legal()
    }

    pub fn ensure_legal(&self) -> &Self {
        self.0.ensure_legal();
        self
    }

    pub fn get(&self) -> T::Value<'_, 'gc> {
        assert!(
            self.0.is_legal(),
            "attempt to get the value of an illegal Value"
        );

        unsafe { T::get_unchecked(self.0 .0.as_ref().unwrap()) }
    }

    pub fn as_value(&self) -> &Value<'gc> {
        &self.0
    }

    pub fn into_value(self) -> Value<'gc> {
        self.0
    }
}

impl<'gc, T: Tag> Default for TypedValue<'gc, T> {
    fn default() -> Self {
        Self(Value::new_illegal(), PhantomData)
    }
}

impl<T: Tag> Finalize for TypedValue<'_, T> {}

unsafe impl<T: Tag> Collect for TypedValue<'_, T> {
    impl_collect! {
        fn visit(&self) {
            visit(&self.0);
        }
    }
}

#[derive(Debug, Clone)]
pub struct Block<'gc> {
    pub location: Location,
    pub obj: GcOnceCell<TypedValue<'gc, tag::Object>>,
    // the strong reference is held in Frame
    pub nlret_valid_flag: Weak<()>,
    pub code: ast::Block,
    pub upvalue_map: HashMap<String, usize>,
    pub upvalues: Vec<Gc<'gc, Upvalue<'gc>>>,
}

impl<'gc> Block<'gc> {
    pub fn get_upvalue_by_name(&self, name: &str) -> Option<&Upvalue<'gc>> {
        self.upvalue_map.get(name).map(|&idx| &*self.upvalues[idx])
    }
}

impl Finalize for Block<'_> {}

unsafe impl Collect for Block<'_> {
    impl_collect! {
        fn visit(&self) {
            visit(&self.obj);
            visit(&self.upvalues);
        }
    }
}

#[derive(Debug, Clone)]
pub struct Class<'gc> {
    pub name: Spanned<String>,
    pub obj: GcOnceCell<TypedValue<'gc, tag::Object>>,
    pub superclass: Option<TypedValue<'gc, tag::Class>>,
    pub method_map: HashMap<String, usize>,
    pub methods: Vec<TypedValue<'gc, tag::Method>>,
    pub instance_field_map: HashMap<String, usize>,
    pub instance_fields: Vec<ast::Name>,
}

impl<'gc> Class<'gc> {
    pub fn get_method_by_name(&self, name: &str) -> Option<&TypedValue<'gc, tag::Method>> {
        self.method_map.get(name).map(|&idx| &self.methods[idx])
    }
}

impl Finalize for Class<'_> {}

unsafe impl Collect for Class<'_> {
    impl_collect! {
        fn visit(&self) {
            visit(&self.obj);
            visit(&self.superclass);
            visit(&self.methods);
        }
    }
}

#[derive(Debug, Clone)]
pub struct Method<'gc> {
    pub selector: ast::SpannedSelector,
    pub location: Location,
    pub obj: GcOnceCell<TypedValue<'gc, tag::Object>>,
    pub holder: GcOnceCell<TypedValue<'gc, tag::Class>>,
    pub def: Spanned<MethodDef>,
}

impl Finalize for Method<'_> {}

unsafe impl Collect for Method<'_> {
    impl_collect! {
        fn visit(&self) {
            visit(&self.obj);
            visit(&self.holder);
        }
    }
}

#[derive(Debug, Clone)]
pub struct Object<'gc> {
    pub class: TypedValue<'gc, tag::Class>,
    pub fields: GcRefCell<Vec<Value<'gc>>>,
}

pub struct FieldProj<'a, 'gc> {
    inner: GcRef<'a, Vec<Value<'gc>>>,
    idx: usize,
}

impl<'gc> Deref for FieldProj<'_, 'gc> {
    type Target = Value<'gc>;

    fn deref(&self) -> &Self::Target {
        &self.inner[self.idx]
    }
}

pub struct FieldProjMut<'a, 'gc> {
    inner: GcRefMut<'a, Vec<Value<'gc>>>,
    idx: usize,
}

impl<'gc> Deref for FieldProjMut<'_, 'gc> {
    type Target = Value<'gc>;

    fn deref(&self) -> &Self::Target {
        &self.inner[self.idx]
    }
}

impl<'gc> DerefMut for FieldProjMut<'_, 'gc> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner[self.idx]
    }
}

impl<'gc> Object<'gc> {
    pub fn field_idx(&self, name: &str) -> Option<usize> {
        self.class.get().instance_field_map.get(name).copied()
    }

    pub fn get_field_by_name(&self, name: &str) -> Option<FieldProj<'_, 'gc>> {
        self.field_idx(name).map(|idx| FieldProj {
            inner: self.fields.borrow(),
            idx,
        })
    }

    pub fn get_field_by_name_mut(&self, name: &str) -> Option<FieldProjMut<'_, 'gc>> {
        self.field_idx(name).map(|idx| FieldProjMut {
            inner: self.fields.borrow_mut(),
            idx,
        })
    }
}

impl Finalize for Object<'_> {}

unsafe impl Collect for Object<'_> {
    impl_collect! {
        fn visit(&self) {
            visit(&self.class);
            visit(&self.fields);
        }
    }
}
