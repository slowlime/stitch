use std::alloc::{GlobalAlloc, Layout, System as SystemAlloc};
use std::ptr;

pub mod ffi {
    #[link(wasm_import_module = "stitch")]
    extern "C" {
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

        #[link_name = "concrete-ptr"]
        pub fn concrete_ptr(ptr: *const ()) -> *const ();

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
        pub fn inline(f: extern "C" fn()) -> extern "C" fn();

        #[link_name = "no-inline"]
        pub fn no_inline(f: extern "C" fn()) -> extern "C" fn();

        #[link_name = "file-open"]
        pub fn file_open(path: *const u8, path_size: usize, out_fd: *mut u32) -> u32;

        #[link_name = "file-read"]
        pub fn file_read(fd: u32, buf: *mut u8, size: usize, out_count: *mut usize) -> u32;

        #[link_name = "file-close"]
        pub fn file_close(fd: u32) -> u32;

        #[link_name = "func-spec-policy"]
        pub fn func_spec_policy(name_regexp: *const u8, name_regexp_len: usize, inline_policy: InlinePolicy);
    }

    #[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
    #[repr(u8)]
    pub enum InlinePolicy {
        #[default]
        Allow = 0,
        Deny = 1,
        ForceInline = 2,
        ForceOutline = 3,
    }
}

pub trait Ptr {
    fn into_ptr(self) -> *const ();
    unsafe fn from_ptr(ptr: *const ()) -> Self;
}

impl<T> Ptr for *const T {
    #[inline(always)]
    fn into_ptr(self) -> *const () {
        self.cast()
    }

    #[inline(always)]
    unsafe fn from_ptr(ptr: *const ()) -> Self {
        ptr.cast()
    }
}

impl<T> Ptr for *mut T {
    #[inline(always)]
    fn into_ptr(self) -> *const () {
        self as *const ()
    }

    #[inline(always)]
    unsafe fn from_ptr(ptr: *const ()) -> Self {
        ptr as *mut T
    }
}

impl<'a, T> Ptr for &'a T {
    #[inline(always)]
    fn into_ptr(self) -> *const () {
        ptr::from_ref(self).into_ptr()
    }

    #[inline(always)]
    unsafe fn from_ptr(ptr: *const ()) -> Self {
        &*ptr.cast::<T>()
    }
}

impl<'a, T> Ptr for &'a mut T {
    #[inline(always)]
    fn into_ptr(self) -> *const () {
        ptr::from_ref(self).into_ptr()
    }

    #[inline(always)]
    unsafe fn from_ptr(ptr: *const ()) -> Self {
        &mut *(ptr as *mut T)
    }
}

impl<T> Ptr for Box<T> {
    #[inline(always)]
    fn into_ptr(self) -> *const () {
        Box::into_raw(self) as *const ()
    }

    #[inline(always)]
    unsafe fn from_ptr(ptr: *const ()) -> Self {
        Box::from_raw(ptr as *mut T)
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

#[inline(always)]
pub fn arg_count() -> usize {
    unsafe { ffi::arg_count() }
}

#[inline(always)]
pub fn arg_len(idx: usize) -> usize {
    unsafe { ffi::arg_len(idx) }
}

#[inline(always)]
pub fn arg_read(idx: usize, buf: &mut [u8], offset: usize) -> usize {
    unsafe { ffi::arg_read(idx, buf.as_mut_ptr(), buf.len(), offset) }
}

pub unsafe trait WasmReturnSafe {}

unsafe impl WasmReturnSafe for i8 {}
unsafe impl WasmReturnSafe for u8 {}
unsafe impl WasmReturnSafe for i16 {}
unsafe impl WasmReturnSafe for u16 {}
unsafe impl WasmReturnSafe for i32 {}
unsafe impl WasmReturnSafe for u32 {}
unsafe impl WasmReturnSafe for i64 {}
unsafe impl WasmReturnSafe for u64 {}
unsafe impl WasmReturnSafe for isize {}
unsafe impl WasmReturnSafe for usize {}
unsafe impl WasmReturnSafe for () {}
unsafe impl<T> WasmReturnSafe for *const T {}
unsafe impl<T> WasmReturnSafe for *mut T {}
unsafe impl<T: Sized> WasmReturnSafe for Box<T> {}

#[macro_export]
macro_rules! specialize {
    ($label:literal: $($unsafe:tt)? extern fn( $($param:tt)* $(,)? ) $(-> $ret:ty)?) => {{
        $(specialize!(@unsafe $unsafe);)?
        specialize!(@parse $label $($unsafe)? {$($param)*,} $({$ret})?: orig = {}, spec = {}, params = {}, args = {})
    }};

    (@unsafe unsafe) => {{}};

    (
        @parse $label:literal $($unsafe:tt)? {$(,)?} $({$ret:ty})?:
        orig = {$($orig:ident: $orig_ty:ty,)*},
        spec = {$($spec_ty:ty,)*},
        params = {$($param:ident: $param_ty:ty,)*},
        args = {$($arg:expr,)*}
    ) => {{
        unsafe fn ensure_unsafe() {}
        ensure_unsafe();

        $(
            fn assert_return_safe<T: $crate::WasmReturnSafe>() {}
            assert_return_safe::<$ret>();
        )?

        #[inline(always)]
        unsafe fn specialize(
            f: $($unsafe)? extern fn($($orig_ty),*) $(-> $ret)?,
            name: Option<&str>,
            $($param: $param_ty,)*
        ) -> Option<$($unsafe)? extern fn($($spec_ty),*) $(-> $ret)?> {
            #[link(wasm_import_module = "stitch")]
            extern {
                #[link_name = concat!("specialize#", $label)]
                fn specialize(
                    f: $($unsafe)? extern fn($($orig_ty),*) $(-> $ret)?,
                    name_ptr: *const u8,
                    name_len: usize,
                    $($param: $param_ty,)*
                ) -> Option<$($unsafe)? extern fn($($spec_ty),*) $(-> $ret)?>;

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
        @parse $label:literal $($unsafe:tt)? {$next_param:ident: $next_param_ty:ty, $($rest:tt)*} $({$ret:ty})?:
        orig = {$($orig:ident: $orig_ty:ty,)*},
        spec = {$($spec_ty:ty,)*},
        params = {$($param:ident: $param_ty:ty,)*},
        args = {$($arg:expr,)*}
    ) => {
        specialize!(
            @parse $label $($unsafe)? {$($rest)*} $({$ret})?:
            orig = {$($orig: $orig_ty:ty,)* $next_param: $next_param_ty,},
            spec = {$($spec_ty:ty,)*},
            params = {$($param: $param_ty,)* $next_param: $next_param_ty,},
            args = {$($arg,)* $next_param,}
        )
    };

    // param_name?: ty
    (
        @parse $label:literal $($unsafe:tt)? {$next_param:ident?: $next_param_ty:ty, $($rest:tt)*} $({$ret:ty})?:
        orig = {$($orig:ident: $orig_ty:ty,)*},
        spec = {$($spec_ty:ty,)*},
        params = {$($param:ident: $param_ty:ty,)*},
        args = {$($arg:expr,)*}
    ) => {
        specialize!(
            @parse $label $($unsafe)? {$($rest)*} $({$ret})?:
            orig = {$($orig: $orig_ty:ty,)*},
            spec = {$($spec_ty:ty,)* $next_param_ty,},
            params = {$($param: $param_ty,)*},
            args = {$($arg,)* ::std::mem::transmute(unknown())}
        )
    };
}

#[inline(always)]
pub unsafe fn const_ptr<P: Ptr>(ptr: P) -> P {
    unsafe { Ptr::from_ptr(ffi::const_ptr(ptr.into_ptr())) }
}

#[inline(always)]
pub unsafe fn symbolic_ptr<P: Ptr>(ptr: P) -> P {
    unsafe { Ptr::from_ptr(ffi::symbolic_ptr(ptr.into_ptr())) }
}

#[inline(always)]
pub unsafe fn concrete_ptr<P: Ptr>(ptr: P) -> P {
    unsafe { Ptr::from_ptr(ffi::concrete_ptr(ptr.into_ptr())) }
}

#[inline(always)]
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

#[inline(always)]
pub fn print_str(s: &str) {
    unsafe { ffi::print_str(s.as_ptr(), s.len()) }
}

#[inline(always)]
pub fn is_specializing() -> bool {
    unsafe { ffi::is_specializing() }
}

#[macro_export]
macro_rules! inline {
    (fn($($param:ty),* $(,)?) $(-> $ret:ty)?) => {
        inline!({} ($($param),*) $($ret)?)
    };
    (extern $($abi:literal)? fn($($param:ty),* $(,)?) $(-> $ret:ty)?) => {
        inline!({extern $($abi)?} ($($param),*) $($ret)?)
    };
    (unsafe fn($($param:ty),* $(,)?) $(-> $ret:ty)?) => {
        inline!({unsafe} ($($param),*) $($ret)?)
    };
    (unsafe extern $($abi:literal)? fn($($param:ty),* $(,)?) $(-> $ret:ty)?) => {
        inline!({unsafe extern $($abi)?} ($($param),*) $($ret)?)
    };

    ({$($attr:tt)*} ($($param:ty),*) $($ret:ty)?) => {{
        #[inline(always)]
        fn inline(f: $($attr)* fn($($param),*) $(-> $ret)?) -> $($attr)* fn($($param),*) $(-> $ret)? {
            use ::std::mem::transmute;

            unsafe { transmute($crate::ffi::inline(transmute::<_, extern fn()>(f))) }
        }

        inline
    }};
}

#[macro_export]
macro_rules! no_inline {
    (fn($($param:ty),* $(,)?) $(-> $ret:ty)?) => {
        no_inline!({} ($($param),*) $($ret)?)
    };
    (extern $($abi:literal)? fn($($param:ty),* $(,)?) $(-> $ret:ty)?) => {
        no_inline!({extern $($abi)?} ($($param),*) $($ret)?)
    };
    (unsafe fn($($param:ty),* $(,)?) $(-> $ret:ty)?) => {
        no_inline!({unsafe} ($($param),*) $($ret)?)
    };
    (unsafe extern $($abi:literal)? fn($($param:ty),* $(,)?) $(-> $ret:ty)?) => {
        no_inline!({unsafe extern $($abi)?} ($($param),*) $($ret)?)
    };

    ({$($attr:tt)*} ($($param:ty),*) $($ret:ty)?) => {{
        #[inline(always)]
        fn no_inline(f: $($attr)* fn($($param),*) $(-> $ret)?) -> $($attr)* fn($($param),*) $(-> $ret)? {
            use ::std::mem::transmute;

            unsafe { transmute($crate::ffi::no_inline(transmute::<_, extern fn()>(f))) }
        }

        no_inline
    }};
}

#[macro_export]
macro_rules! print_str {
    ($($param:tt)+) => {{
        if let Some(s) = ::std::format_args!($($param)+).as_str() {
            $crate::print_str(s);
        } else {
            $crate::print_str(&::std::format!($($param)+).to_string());
        }
    }};
}

pub struct SymbolicAlloc;

unsafe impl GlobalAlloc for SymbolicAlloc {
    #[inline(always)]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        symbolic_ptr(no_inline!(unsafe fn(&SystemAlloc, Layout) -> *mut u8)(
            SystemAlloc::alloc,
        )(&SystemAlloc, layout))
    }

    #[inline(always)]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        no_inline!(unsafe fn(&SystemAlloc, *mut u8, Layout))(SystemAlloc::dealloc)(
            &SystemAlloc,
            ptr,
            layout,
        )
    }

    #[inline(always)]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        no_inline!(unsafe fn(&SystemAlloc, Layout) -> *mut u8)(SystemAlloc::alloc_zeroed)(
            &SystemAlloc,
            layout,
        )
    }

    #[inline(always)]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        no_inline!(unsafe fn(&SystemAlloc, *mut u8, Layout, usize) -> *mut u8)(SystemAlloc::realloc)(
            &SystemAlloc,
            ptr,
            layout,
            new_size,
        )
    }
}

pub use ffi::InlinePolicy;

#[derive(Debug, Clone)]
pub struct FuncSpecPolicy {
    pub inline_policy: InlinePolicy,
}

pub fn func_spec_policy(name_regexp: &str, policy: &FuncSpecPolicy) {
    unsafe {
        ffi::func_spec_policy(
            name_regexp.as_ptr(),
            name_regexp.len(),
            policy.inline_policy,
        );
    }
}

pub fn configure_rust_func_spec_policies() {
    let ref no_inline_policy = FuncSpecPolicy {
        inline_policy: InlinePolicy::Deny,
    };

    func_spec_policy("^rust_begin_unwind$", &no_inline_policy);
    func_spec_policy("^([cm]|re)alloc$", &no_inline_policy);
    func_spec_policy("^free$", &no_inline_policy);
}
