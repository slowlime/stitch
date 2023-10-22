use std::collections::HashMap;
use std::fmt::{self, Display};
use std::hash::{self, Hash};
use std::marker::PhantomData;
use std::mem;
use std::ops::{Deref, DerefMut, RangeBounds};
use std::ptr;

use thiserror::Error;

use crate::ast::{self, SymbolLit as Symbol};
use crate::impl_collect;
use crate::location::{Location, Span, Spanned};

use super::error::VmError;
use super::frame::Upvalue;
use super::gc::{Collect, Gc};
use super::gc::{Finalize, GarbageCollector};
use super::gc::{GcOnceCell, GcRef, GcRefCell, GcRefMut};
use super::method::MethodDef;
use super::Vm;

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
                expected: vec![T::TY].into(),
                actual: self.0.as_ref().unwrap().ty(),
            })
        }
    }

    pub fn get_class<'a>(&'a self, vm: &'a Vm<'gc>) -> &'a TypedValue<'gc, tag::Class> {
        self.ensure_legal().0.as_ref().unwrap().get_class(vm)
    }

    pub fn get_obj(&self) -> Option<&Object<'gc>> {
        self.0.as_ref()?.get_obj()
    }

    pub fn size(&self) -> usize {
        self.0.as_ref().map(|value| value.size()).unwrap_or(0)
    }

    pub fn ptr_eq(&self, other: &Self) -> bool {
        match (&self.0, &other.0) {
            (Some(lhs), Some(rhs)) => lhs.ptr_eq(rhs),
            _ => false,
        }
    }

    pub fn as_ptr(&self) -> *const ValueKind<'gc> {
        match &self.0 {
            Some(inner) => &**inner,
            None => ptr::null(),
        }
    }

    pub fn hash_code(&self) -> usize {
        fxhash::hash(self)
    }
}

impl Hash for Value<'_> {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        mem::discriminant(&self.0).hash(state);

        match &self.0 {
            Some(inner) => {
                let kind: &ValueKind = inner;
                mem::discriminant(kind).hash(state);

                match kind {
                    ValueKind::Int(i) => i.hash(state),
                    ValueKind::Float(f) if f.is_nan() => 0x7ff8000000000000u64.hash(state),
                    ValueKind::Float(f) if *f == 0.0 && f.is_sign_negative() => {
                        0f64.to_bits().hash(state)
                    }
                    ValueKind::Float(f) => f.to_bits().hash(state),
                    ValueKind::Array(arr) => arr.borrow().hash(state),
                    ValueKind::Block(block) => block.obj.get().unwrap().hash(state),
                    ValueKind::Class(class) => class.obj.get().unwrap().hash(state),
                    ValueKind::Method(method) => method.obj.get().unwrap().hash(state),
                    ValueKind::Object(_) => self.as_ptr().hash(state),
                    ValueKind::String(s) => s.hash(state),

                    ValueKind::Symbol(sym) => {
                        mem::discriminant(sym).hash(state);

                        match sym {
                            Symbol::String(s) => s.hash(state),
                            Symbol::Selector(selector) => selector.value.name().hash(state),
                        }
                    }
                }
            }

            None => {}
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

        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum Ty {
            NamedClass(Box<String>),

            $( $name, )+
        }

        impl Display for Ty {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self {
                    Self::NamedClass(name) => name.fmt(f),

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
        String("string", SomString => &'a SomString): ref s => s,
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

    pub fn get_class<'a>(&'a self, vm: &'a Vm<'gc>) -> &'a TypedValue<'gc, tag::Class> {
        match self {
            Self::Int(_) => &vm.builtins.integer,
            Self::Float(_) => &vm.builtins.double,
            Self::Array(_) => &vm.builtins.array,
            Self::Block(block) => block.obj.get().unwrap().get().get_class(),
            Self::Class(class) => class.obj.get().unwrap().get().get_class(),
            Self::Method(method) => method.obj.get().unwrap().get().get_class(),
            Self::Symbol(_) => &vm.builtins.symbol,
            Self::Object(obj) => obj.get_class(),
            Self::String(_) => &vm.builtins.string,
        }
    }

    pub fn size(&self) -> usize {
        mem::size_of_val(self)
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

    pub fn checked_get(&self) -> Option<T::Value<'_, 'gc>> {
        self.is_legal().then(|| self.get())
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

impl<'gc, T: Tag> Hash for TypedValue<'gc, T> {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
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
    pub code: ast::Block,
    pub upvalue_map: HashMap<String, usize>,
    pub upvalues: Vec<Gc<'gc, Upvalue<'gc>>>,
    pub defining_method: TypedValue<'gc, tag::Method>,
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
    pub superclass: GcOnceCell<Option<TypedValue<'gc, tag::Class>>>,
    pub method_map: HashMap<String, usize>,
    pub methods: Vec<TypedValue<'gc, tag::Method>>,
    pub instance_field_map: HashMap<String, usize>,
    pub instance_fields: Vec<ast::Name>,
}

impl<'gc> Class<'gc> {
    pub fn get_superclass(&self) -> Option<&TypedValue<'gc, tag::Class>> {
        self.superclass.get().unwrap().as_ref()
    }

    pub fn get_method_by_name(&self, name: &str) -> Option<&TypedValue<'gc, tag::Method>> {
        let mut next_class = Some(self);

        while let Some(class) = next_class {
            next_class = class.get_superclass().map(|class| class.get());

            if let method @ Some(_) = class.get_local_method_by_name(name) {
                return method;
            }
        }

        None
    }

    pub fn get_supermethod_by_name(&self, name: &str) -> Option<&TypedValue<'gc, tag::Method>> {
        let mut next_class = self.get_superclass();

        while let Some(class) = next_class {
            next_class = class.get().get_superclass();

            if let method @ Some(_) = class.get().get_local_method_by_name(name) {
                return method;
            }
        }

        None
    }

    pub fn get_local_method_by_name(&self, name: &str) -> Option<&TypedValue<'gc, tag::Method>> {
        self.method_map.get(name).map(|&idx| &self.methods[idx])
    }

    pub fn is_subclass_of(&self, superclass: &Class<'gc>) -> bool {
        let mut cls = self;

        loop {
            if ptr::eq(cls, superclass) {
                return true;
            }

            if let Some(parent) = cls.get_superclass() {
                cls = parent.get();
            } else {
                return false;
            }
        }
    }

    pub fn get_obj(&self) -> &TypedValue<'gc, tag::Object> {
        self.obj.get().unwrap()
    }

    pub fn get_metaclass(&self) -> &TypedValue<'gc, tag::Class> {
        self.get_obj().get().get_class()
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
    pub class: GcOnceCell<TypedValue<'gc, tag::Class>>,
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
    pub fn get_class(&self) -> &TypedValue<'gc, tag::Class> {
        self.class.get().unwrap()
    }

    pub fn field_idx(&self, name: &str) -> Option<usize> {
        self.get_class().get().instance_field_map.get(name).copied()
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

    pub fn get_method_by_name(&self, name: &str) -> Option<&TypedValue<'gc, tag::Method>> {
        self.get_class().get().get_method_by_name(name)
    }

    pub fn get_supermethod_by_name(&self, name: &str) -> Option<&TypedValue<'gc, tag::Method>> {
        self.get_class()
            .get()
            .get_superclass()?
            .get()
            .get_method_by_name(name)
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

#[derive(Error, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SubstrError {
    #[error("start position comes after end")]
    StartGtEnd,

    #[error("start position is out of bounds")]
    StartOutOfBounds,

    #[error("end position is out of bounds")]
    EndOutOfBounds,
}

#[derive(Debug, Clone)]
pub struct SomString {
    value: Box<str>,
    char_count: usize,
}

impl SomString {
    pub fn new(value: Box<str>) -> Self {
        let char_count = value.chars().count();

        Self { value, char_count }
    }

    pub fn as_str(&self) -> &str {
        &self.value
    }

    pub fn char_count(&self) -> usize {
        self.char_count
    }

    pub fn chars(&self) -> std::str::Chars<'_> {
        self.value.chars()
    }

    pub fn concat(&self, other: &Self) -> Self {
        let mut result = String::with_capacity(self.value.len() + other.value.len());
        result.push_str(&self.value);
        result.push_str(&other.value);

        Self {
            value: result.into_boxed_str(),
            char_count: self.char_count + other.char_count,
        }
    }

    pub fn substr(&self, range: impl RangeBounds<usize>) -> Result<Self, SubstrError> {
        use std::ops::Bound;

        let start = match range.start_bound() {
            Bound::Included(&pos) => pos,
            Bound::Excluded(&pos) => pos + 1,
            Bound::Unbounded => 0,
        };

        let end = match range.end_bound() {
            Bound::Included(&pos) => pos + 1,
            Bound::Excluded(&pos) => pos,
            Bound::Unbounded => self.char_count,
        };

        if start > end {
            Err(SubstrError::StartGtEnd)
        } else if start > self.char_count {
            Err(SubstrError::StartOutOfBounds)
        } else if end > self.char_count {
            Err(SubstrError::EndOutOfBounds)
        } else if start == end {
            Ok(Self {
                value: String::new().into_boxed_str(),
                char_count: 0,
            })
        } else {
            let mut chars = self.value.char_indices().map(|(idx, _)| idx).skip(start);
            let first_char_pos = chars.next().unwrap();
            let end_char_pos = chars.nth(end - start - 1).unwrap_or(self.value.len());
            let value = self.value[first_char_pos..end_char_pos]
                .to_owned()
                .into_boxed_str();

            Ok(Self {
                value,
                char_count: end - start,
            })
        }
    }
}

impl PartialEq for SomString {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl Eq for SomString {}

impl Hash for SomString {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

impl Display for SomString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.value.fmt(f)
    }
}

impl From<String> for SomString {
    fn from(s: String) -> Self {
        Self::new(s.into_boxed_str())
    }
}

impl From<Box<str>> for SomString {
    fn from(s: Box<str>) -> Self {
        Self::new(s)
    }
}

impl From<SomString> for Box<str> {
    fn from(s: SomString) -> Self {
        s.value
    }
}

impl From<SomString> for String {
    fn from(s: SomString) -> Self {
        s.value.into()
    }
}
