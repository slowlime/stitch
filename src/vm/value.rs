use std::cell::Cell;
use std::fmt::{self, Display};
use std::marker::PhantomData;
use std::rc::Rc;

use crate::ast::{self, SymbolLit as Symbol};
use crate::impl_collect;
use crate::location::Span;

use super::error::VmError;
use super::gc::GcRefCell;
use super::gc::{Collect, Gc};
use super::gc::{Finalize, GarbageCollector};

#[derive(Debug, Clone)]
pub struct Value<'gc>(Gc<'gc, ValueKind<'gc>>);

impl<'gc> Value<'gc> {
    pub fn is<T: Tag>(&self) -> bool {
        T::is_tag_of(self)
    }

    pub fn downcast<T: Tag>(self) -> Option<TypedValue<'gc, T>> {
        if T::is_tag_of(&self) {
            Some(unsafe { TypedValue::new(self) })
        } else {
            None
        }
    }

    pub fn downcast_or_err<T: Tag>(
        self,
        span: impl Into<Option<Span>>,
    ) -> Result<TypedValue<'gc, T>, VmError> {
        if T::is_tag_of(&self) {
            Ok(unsafe { TypedValue::new(self) })
        } else {
            Err(VmError::IllegalTy {
                span: span.into(),
                expected: [T::TY].as_slice().into(),
                actual: self.0.ty(),
            })
        }
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

            fn is_tag_of(val: &Value<'_>) -> bool;

            fn get<'a, 'gc>(val: &'a Value<'gc>) -> Self::Value<'a, 'gc>;

            unsafe fn get_unchecked<'a, 'gc>(val: &'a Value<'gc>) -> Self::Value<'a, 'gc>;
        }

        $(
            impl Tag for tag::$name {
                type Value<'a, 'gc: 'a> = $value;

                const TY: Ty = Ty::$name;

                fn is_tag_of(val: &Value) -> bool {
                    matches!(*val.0, ValueKind::$name(_))
                }

                fn get<'a, 'gc>(val: &'a Value<'gc>) -> Self::Value<'a, 'gc> {
                    match *val.0 {
                        ValueKind::$name($pat) => $arm,
                        _ => panic!("invalid tag"),
                    }
                }

                unsafe fn get_unchecked<'a, 'gc>(val: &'a Value<'gc>) -> Self::Value<'a, 'gc> {
                    match *val.0 {
                        ValueKind::$name($pat) => $arm,
                        _ => std::hint::unreachable_unchecked(),
                    }
                }
            }

            impl<'gc> IntoValue<'gc> for $ty {
                type Tag = tag::$name;

                fn into_value(self, gc: &'gc GarbageCollector) -> TypedValue<'gc, Self::Tag> {
                    TypedValue(Value(Gc::new(gc, ValueKind::$name(self))), PhantomData)
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
        Primitive("primitive", Primitive => &'a Primitive): ref p => p,
        Symbol("symbol", Symbol => &'a Symbol): ref sym => sym,
        Object("object", Object<'gc> => &'a Object<'gc>): ref obj => obj,
        String("string", String => &'a str): ref s => s,
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

                Self::Int(_) | Self::Float(_) | Self::String(_) | Self::Primitive(_) | Self::Symbol(_) => {},
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

    pub fn get(&self) -> T::Value<'_, 'gc> {
        unsafe { T::get_unchecked(&self.0) }
    }

    pub fn as_value(&self) -> &Value<'gc> {
        &self.0
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
    nlret_valid_flag: Rc<Cell<bool>>,
    code: ast::Block,
    // TODO
    upvalues: Vec<Value<'gc>>,
}

impl Finalize for Block<'_> {}

unsafe impl Collect for Block<'_> {
    impl_collect! {
        fn visit(&self) {
            visit(&self.upvalues);
        }
    }
}

#[derive(Debug, Clone)]
pub struct Class<'gc> {
    obj: TypedValue<'gc, tag::Object>,
    superclass: Option<TypedValue<'gc, tag::Class>>,
    methods: Vec<TypedValue<'gc, tag::Method>>,
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
    obj: TypedValue<'gc, tag::Object>,
    code: ast::Block,
}

impl Finalize for Method<'_> {}

unsafe impl Collect for Method<'_> {
    impl_collect! {
        fn visit(&self) {
            visit(&self.obj);
        }
    }
}

#[derive(Debug, Clone)]
pub enum Primitive {}

impl Finalize for Primitive {}

unsafe impl Collect for Primitive {
    impl_collect!();
}

#[derive(Debug, Clone)]
pub struct Object<'gc> {
    class: TypedValue<'gc, tag::Class>,
    fields: Vec<Value<'gc>>,
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
