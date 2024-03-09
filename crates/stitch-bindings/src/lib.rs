use std::ptr;

pub mod ffi {
    #[link(wasm_import_module = "stitch")]
    extern {
        #[link_name = "arg-count"]
        pub fn arg_count() -> usize;

        #[link_name = "arg-len"]
        pub fn arg_len(idx: usize) -> usize;

        #[link_name = "arg-read"]
        pub fn arg_read(idx: usize, buf: *mut u8, size: usize, offset: usize) -> usize;

        #[link_name = "const-ptr"]
        pub fn const_ptr(ptr: *const ()) -> *const ();

        #[link_name = "symbolic-ptr"]
        pub fn symbolic_ptr(ptr: *const ()) -> *const ();

        #[link_name = "propagate-load"]
        pub fn propagate_load(ptr: *const ()) -> *const ();

        #[link_name = "print-value#i32"]
        pub fn print_value_i32(value: i32);

        #[link_name = "print-value#i64"]
        pub fn print_value_i64(value: i64);

        #[link_name = "print-value#f32"]
        pub fn print_value_f32(value: f32);

        #[link_name = "print-value#f64"]
        pub fn print_value_f64(value: f64);

        #[link_name = "print-str"]
        pub fn print_str(ptr: *const u8, len: usize);

        #[link_name = "is-specializing"]
        pub fn is_specializing() -> bool;

        #[link_name = "inline"]
        pub fn inline(f: extern fn()) -> extern fn();

        #[link_name = "no-inline"]
        pub fn no_inline(f: extern fn()) -> extern fn();

        #[link_name = "file-open"]
        pub fn file_open(path: *const u8, path_size: usize, out_fd: *mut u32) -> u32;

        #[link_name = "file-read"]
        pub fn file_read(fd: u32, buf: *mut u8, size: usize, out_count: *mut usize) -> u32;

        #[link_name = "file-close"]
        pub fn file_close(fd: u32) -> u32;
    }
}

pub trait Ptr {
    fn into_ptr(self) -> *const ();
    unsafe fn from_ptr(ptr: *const ()) -> Self;
}

impl<T> Ptr for *const T {
    fn into_ptr(self) -> *const () {
        self.cast()
    }

    unsafe fn from_ptr(ptr: *const ()) -> Self {
        ptr.cast()
    }
}

impl<T> Ptr for *mut T {
    fn into_ptr(self) -> *const () {
        self as *const ()
    }

    unsafe fn from_ptr(ptr: *const ()) -> Self {
        ptr as *mut T
    }
}

impl<'a, T> Ptr for &'a T {
    fn into_ptr(self) -> *const () {
        ptr::from_ref(self).into_ptr()
    }

    unsafe fn from_ptr(ptr: *const ()) -> Self {
        &*ptr.cast::<T>()
    }
}

impl<'a, T> Ptr for &'a mut T {
    fn into_ptr(self) -> *const () {
        ptr::from_ref(self).into_ptr()
    }

    unsafe fn from_ptr(ptr: *const ()) -> Self {
        &mut *(ptr as *mut T)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Value {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
}

impl From<i32> for Value {
    fn from(value: i32) -> Self {
        Self::I32(value)
    }
}

impl From<i64> for Value {
    fn from(value: i64) -> Self {
        Self::I64(value)
    }
}

impl From<f32> for Value {
    fn from(value: f32) -> Self {
        Self::F32(value)
    }
}

impl From<f64> for Value {
    fn from(value: f64) -> Self {
        Self::F64(value)
    }
}

pub fn arg_count() -> usize {
    unsafe { ffi::arg_count() }
}

pub fn arg_len(idx: usize) -> usize {
    unsafe { ffi::arg_len(idx) }
}

pub fn arg_read(idx: usize, buf: &mut [u8], offset: usize) -> usize {
    unsafe { ffi::arg_read(idx, buf.as_mut_ptr(), buf.len(), offset) }
}

#[macro_export]
macro_rules! specialize {
    ($label:literal: fn( $($param:tt)*, ) $(-> $ret:ty)?) => {
        specialize!(@parse $label {$($param)*} $({$ret})?: orig = {}, spec = {}, params = {})
    };

    (
        @parse $label:literal {$(,)?} $({$ret:ty})?:
        orig = {$($orig:ident: $orig_ty:ty,)*},
        spec = {$($spec_ty:ty,)*},
        params = {$($param:ident: $param_ty:ty,)*},
        args = {$($arg:expr,)*}
    ) => {{
        unsafe fn ensure_unsafe() {}

        ensure_unsafe();

        unsafe fn specialize(
            f: extern fn($($orig_ty),*) $(-> $ret)?,
            name: Option<&str>,
            $($param: $param_ty,)*
        ) -> Option<extern fn($($spec_ty),*) $(-> $ret)?> {
            #[link(wasm_import_module = "stitch")]
            extern {
                #[link_name = "specialize#" $label]
                fn specialize(
                    f: extern fn($($orig_ty),*) $(-> $ret)?,
                    name_ptr: *const u8,
                    name_len: usize,
                    $($param: $param_ty,)*
                ) -> Option<extern fn($($spec_ty),*) $(-> $ret)?>;

                #[link_name = "unknown"]
                fn unknown() -> i32;
            }

            let (name_ptr, name_len) = if let Some(name) = name {
                (name.as_ptr(), name.len())
            } else {
                (::std::ptr::null(), 0)
            };

            specialize(f, name_ptr, name_len, $($arg),*)
        }

        specialize
    }};

    // param_name: ty
    (
        @parse $label:literal {$next_param:ident: $next_param_ty:ty, $($rest:tt)*} $({$ret:ty})?:
        orig = {$($orig:ident: $orig_ty:ty,)*},
        spec = {$($spec_ty:ty,)*},
        params = {$($param:ident: $param_ty:ty,)*},
        args = {$($arg:expr,)*}
    ) => {
        specialize!(
            @parse $label {$($rest)*} $({$ret})?:
            orig = {$($orig: $orig_ty:ty,)* $next_param: $next_param_ty,},
            spec = {$($spec_ty:ty,)*},
            params = {$($param: $param_ty,)* $next_param: $next_param_ty,},
            args = {$($arg,)* $next_param,}
        )
    };

    // param_name?: ty
    (
        @parse $label:literal {$next_param:ident?: $next_param_ty:ty, $($rest:tt)*} $({$ret:ty})?:
        orig = {$($orig:ident: $orig_ty:ty,)*},
        spec = {$($spec_ty:ty,)*},
        params = {$($param:ident: $param_ty:ty,)*},
        args = {$($arg:expr,)*}
    ) => {
        specialize!(
            @parse $label {$($rest)*} $({$ret})?:
            orig = {$($orig: $orig_ty:ty,)*},
            spec = {$($spec_ty:ty,)* $next_param_ty,},
            params = {$($param: $param_ty,)*},
            args = {$($arg,)* ::std::mem::transmute(unknown())}
        )
    };
}

pub unsafe fn const_ptr<P: Ptr>(ptr: P) -> P {
    unsafe { Ptr::from_ptr(ffi::const_ptr(ptr.into_ptr())) }
}

pub unsafe fn symbolic_ptr<P: Ptr>(ptr: P) -> P {
    unsafe { Ptr::from_ptr(ffi::symbolic_ptr(ptr.into_ptr())) }
}

pub unsafe fn propagate_load<P: Ptr>(ptr: P) -> P {
    unsafe { Ptr::from_ptr(ffi::propagate_load(ptr.into_ptr())) }
}

pub fn print_value(value: impl Into<Value>) {
    match value.into() {
        Value::I32(value) => unsafe { ffi::print_value_i32(value) },
        Value::I64(value) => unsafe { ffi::print_value_i64(value) },
        Value::F32(value) => unsafe { ffi::print_value_f32(value) },
        Value::F64(value) => unsafe { ffi::print_value_f64(value) },
    }
}

pub fn print_str(s: &str) {
    unsafe { ffi::print_str(s.as_ptr(), s.len()) }
}

pub fn is_specializing() -> bool {
    unsafe { ffi::is_specializing() }
}
