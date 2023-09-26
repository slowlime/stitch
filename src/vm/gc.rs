//! A primitive garbage collector.
//!
//! Parallels the design of `Rc`/`RefCell`:
//! - `Gc` is an immutable shared reference
//! - `GcCell` allows mutation by enforcing the borrowing rules at runtime

use std::alloc::Layout;
use std::cell::{Cell, UnsafeCell};
use std::collections::HashSet;
use std::marker::PhantomData;
use std::mem::{align_of, size_of_val};
use std::ops::{Deref, DerefMut};
use std::ptr::{addr_of, NonNull};

use crate::util::define_yes_no_options;

define_yes_no_options! {
    enum Marked;
    enum Strong;
    enum Collecting;
}

// determines how often GC runs: we launch a collection cycle if `used/heap_size >= LOAD_FACTOR/u8::MAX`.
const LOAD_FACTOR: u8 = 191; // ~75%

pub trait HasVTable {
    const VTABLE: &'static GcVTable;
}

pub unsafe trait Collect {
    fn mark(&self);

    unsafe fn inc_ref_count(&self);

    unsafe fn dec_ref_count(&self);

    unsafe fn finalize(&self);
}

impl<T: Collect> HasVTable for T {
    const VTABLE: &'static GcVTable = &GcVTable::new_for::<T>();
}

type NextCell = Cell<Option<GcErasedBox>>;

pub struct GarbageCollector {
    // gc values are arranged in an intrusive linked list; this is the head of that list
    head: NextCell,

    // how much memory is used by Gc objects
    used: Cell<usize>,

    // a conceptual max heap size: we double it each time we exceed the `LOAD_FACTOR`
    // this adjusts the garbage collection frequency
    // must be a power of two
    heap_size: Cell<usize>,

    collecting: Cell<Collecting>,
}

impl GarbageCollector {
    pub fn new() -> Self {
        Self {
            head: Cell::new(None),
            used: Cell::new(0),
            heap_size: Cell::new(4096),
            collecting: Cell::new(Collecting::No),
        }
    }

    fn try_collect(&self) {
        if self.collecting.get().is_yes() {
            return;
        }

        let used = self.used.get();
        let heap_size = self.heap_size.get();
        let threshold = LOAD_FACTOR as usize * (heap_size >> 7);

        if used > threshold {
            self.collect();

            if self.used.get() > threshold {
                self.heap_size.set(heap_size * 2);
            }
        }
    }

    pub fn collect(&self) {
        if self.collecting.replace(Collecting::Yes).is_yes() {
            return;
        }

        self.mark();
        let unmarked = self.get_unmarked();

        // run finalizers
        for &(_, obj) in &unmarked {
            unsafe {
                obj.finalize();
            }
        }

        // do another marking because finalizers could have changed anything...
        self.mark();
        let mut condemned = self.get_condemned(unmarked.into_iter().map(|(_, obj)| obj).collect());

        // drop the condemned and fix the linked list
        // (must do it in reverse order, or the cell will point elsewhere)
        while let Some((cell, obj)) = condemned.pop() {
            unsafe {
                let cell = &*cell;
                debug_assert_eq!(
                    cell.get().unwrap(),
                    obj,
                    "linked list was broken while deallocating"
                );

                assert!(
                    obj.header_ref().marked().is_no(),
                    "trying to deallocate a marked object"
                );

                // remove from the linked list
                cell.set(obj.header_ref().next.get());

                // drop the box and update the used size
                let size = obj.drop();
                self.used.set(
                    self.used
                        .get()
                        .checked_sub(size)
                        .expect("inconsistent size"),
                );
            }
        }

        self.collecting.set(Collecting::No);
    }

    fn mark(&self) {
        let mut next = self.head.get();

        while let Some(obj) = next {
            unsafe {
                next = obj.next();

                if obj.ref_count() == 0 {
                    // not directly referenced by a root
                    continue;
                }

                obj.mark_header();
                obj.mark_value();
            }
        }
    }

    fn foreach_unmarked<F>(&self, mut unmarked: F)
    where
        F: FnMut(*const NextCell, GcErasedBox),
    {
        let mut next = &self.head as *const NextCell;

        unsafe {
            while let Some(obj) = (*next).get() {
                let current = next;
                next = addr_of!((*obj.header_ptr()).next);

                if obj.header_ref().set_marked(Marked::No).is_no() {
                    unmarked(current, obj);
                }
            }
        }
    }

    fn get_unmarked(&self) -> Vec<(*const NextCell, GcErasedBox)> {
        let mut unmarked = vec![];

        self.foreach_unmarked(|cell, obj| unmarked.push((cell, obj)));

        unmarked
    }

    fn get_condemned(
        &self,
        finalized: HashSet<GcErasedBox>,
    ) -> Vec<(*const NextCell, GcErasedBox)> {
        let mut condemned = vec![];

        self.foreach_unmarked(|cell, obj| {
            if finalized.contains(&obj) {
                condemned.push((cell, obj));
            }
        });

        condemned
    }
}

impl Drop for GarbageCollector {
    fn drop(&mut self) {
        // since Gc<T>s are bound to GarbageCollector by a lifetime parameter,
        // they should all be dead or forgotten at this point
        self.collect();
    }
}

struct Header {
    vtable: &'static GcVTable,
    ref_mark: Cell<u32>,
    next: NextCell,
}

impl Header {
    const MARK_MASK: u32 = 1 << 31;
    const REF_MASK: u32 = !Self::MARK_MASK;
    const MAX_REFS: u32 = Self::REF_MASK;

    fn new(vtable: &'static GcVTable) -> Self {
        Self {
            vtable,
            ref_mark: Cell::new(1),
            next: Cell::new(None),
        }
    }

    fn ref_count(&self) -> usize {
        (self.ref_mark.get() & Self::REF_MASK) as usize
    }

    fn ref_mark(&self) -> (usize, Marked) {
        let ref_mark = self.ref_mark.get();
        let ref_count = ref_mark & Self::REF_MASK;
        let marked = Marked::from(ref_mark & Self::MARK_MASK != 0);

        (ref_count as usize, marked)
    }

    fn set_ref_mark(&self, ref_count: usize, marked: Marked) {
        assert!(
            ref_count <= Self::MAX_REFS as usize,
            "reference count limit exceeded"
        );
        self.ref_mark
            .set(Self::MARK_MASK * marked.is_yes() as u32 | ref_count as u32);
    }

    fn inc_ref_count(&self) {
        let (ref_count, marked) = self.ref_mark();
        self.set_ref_mark(ref_count + 1, marked)
    }

    fn dec_ref_count(&self) {
        let (ref_count, marked) = self.ref_mark();
        assert!(ref_count > 0, "attempt to decrement 0 ref count");
        self.set_ref_mark(ref_count - 1, marked);
    }

    fn marked(&self) -> Marked {
        self.ref_mark().1
    }

    fn set_marked(&self, marked: Marked) -> Marked {
        let (ref_count, old_marked) = self.ref_mark();
        self.set_ref_mark(ref_count, marked);

        old_marked
    }

    fn mark(&self) -> Marked {
        self.set_marked(Marked::Yes)
    }
}

pub struct GcVTable {
    offset: usize,
    mark: unsafe fn(value: *const ()),
    finalize: unsafe fn(value: *const ()),
    drop: unsafe fn(gc_box: *mut ()) -> usize,
}

impl GcVTable {
    pub const fn new_for<T: Collect>() -> GcVTable {
        unsafe fn mark<T: Collect>(this: *const ()) {
            let this = this as *const T;
            this.as_ref().unwrap().mark();
        }

        unsafe fn finalize<T: Collect>(this: *const ()) {
            let this = this as *const T;
            this.as_ref().unwrap().finalize()
        }

        unsafe fn drop<T: Collect>(gc_box: *mut ()) -> usize {
            let gc_box = gc_box as *mut GcBox<T>;

            let gc_box = Box::from_raw(gc_box);
            let size = size_of_val(&*gc_box);

            // the box goes out of scope and gets deallocated
            size
        }

        let offset = GcBox::<T>::data_offset();

        GcVTable {
            offset,
            mark: mark::<T>,
            finalize: finalize::<T>,
            drop: drop::<T>,
        }
    }
}

#[repr(C)]
struct GcBox<'gc, T: ?Sized> {
    header: Header,
    _marker: PhantomData<&'gc GarbageCollector>,
    value: T,
}

impl<T: ?Sized> GcBox<'_, T> {
    const fn data_offset_for(value_align: usize) -> usize {
        let size = Layout::new::<GcBox<()>>().size();
        let misalignment = size % value_align;

        size + if misalignment == 0 {
            0
        } else {
            value_align - misalignment
        }
    }
}

impl<T> GcBox<'_, T> {
    const fn data_offset() -> usize {
        Self::data_offset_for(align_of::<T>())
    }
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct GcErasedBox(NonNull<()>);

impl GcErasedBox {
    fn new<'gc, T>(gc_box: NonNull<GcBox<'gc, T>>) -> Self {
        GcErasedBox(gc_box.cast())
    }

    fn header_ptr(&self) -> *const Header {
        self.0.as_ptr() as *const Header
    }

    unsafe fn header_ref(&self) -> &Header {
        unsafe { &*self.header_ptr() }
    }

    unsafe fn next(&self) -> Option<GcErasedBox> {
        unsafe { (*self.header_ptr()).next.get() }
    }

    unsafe fn vtable(&self) -> &'static GcVTable {
        unsafe { (*self.header_ptr()).vtable }
    }

    unsafe fn ref_count(&self) -> usize {
        unsafe { (*self.header_ptr()).ref_count() }
    }

    unsafe fn value_ptr(&self) -> *const () {
        unsafe {
            (self.0.as_ptr() as *const u8)
                .add(self.vtable().offset)
                .cast()
        }
    }

    unsafe fn mark_header(&self) -> Marked {
        unsafe { (*self.header_ptr()).mark() }
    }

    unsafe fn mark_value(&self) {
        let vtable = self.vtable();
        (vtable.mark)(self.value_ptr());
    }

    unsafe fn finalize(&self) {
        let vtable = self.vtable();
        (vtable.finalize)(self.value_ptr());
    }

    unsafe fn drop(self) -> usize {
        let vtable = self.vtable();
        (vtable.drop)(self.0.as_ptr())
    }
}

pub struct Gc<'gc, T: ?Sized> {
    inner: NonNull<GcBox<'gc, T>>,

    // basically tracks if we need to dec_ref_count on the header on Drop
    strong: Cell<Strong>,

    // to avoid dropck fiascoes
    _marker: PhantomData<GcBox<'gc, T>>,
}

impl<'gc, T: Collect> Gc<'gc, T> {
    pub fn new(gc: &'gc GarbageCollector, value: T) -> Self {
        // allocate the actual box that's gonna store the value
        let inner = NonNull::new(Box::into_raw(Box::new(GcBox {
            header: Header::new(T::VTABLE),
            _marker: PhantomData,
            value,
        })))
        .unwrap();

        // make sure we aren't putting this guy in a pile of garbage
        gc.used
            .set(gc.used.get() + size_of_val(unsafe { inner.as_ref() }));
        gc.try_collect();

        // stick it into the list
        let obj = GcErasedBox::new(inner);
        let next = gc.head.replace(Some(obj));

        unsafe {
            obj.header_ref().next.set(next);

            // this one's important: roots inside the value should no longer be treated as such
            inner.as_ref().value.dec_ref_count();
        }

        Gc {
            inner,
            strong: Cell::new(Strong::Yes),
            _marker: PhantomData,
        }
    }
}

impl<'gc, T: ?Sized> Gc<'gc, T> {
    // unsafe because it allows direct access to the reference counter among other things...
    unsafe fn inner(&self) -> &GcBox<'gc, T> {
        unsafe { self.inner.as_ref() }
    }
}

impl<T: ?Sized> Drop for Gc<'_, T> {
    fn drop(&mut self) {
        if self.strong.get().is_yes() {
            unsafe {
                self.inner().header.dec_ref_count();
            }
        }
    }
}

impl<T: ?Sized> Clone for Gc<'_, T> {
    fn clone(&self) -> Self {
        let result = Gc {
            inner: self.inner,
            strong: Cell::new(Strong::Yes),
            _marker: PhantomData,
        };

        unsafe {
            // a new strong reference!
            self.inner().header.inc_ref_count();
        }

        result
    }
}

impl<T: ?Sized> Deref for Gc<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &self.inner().value }
    }
}

unsafe impl<T: Collect> Collect for Gc<'_, T> {
    fn mark(&self) {
        unsafe {
            if self.inner().header.mark().is_no() {
                self.inner().value.mark()
            }
        }
    }

    unsafe fn inc_ref_count(&self) {
        assert!(self.strong.get().is_no(), "trying to upgrade a root");

        let inner = self.inner();
        inner.header.inc_ref_count();

        self.strong.set(Strong::Yes);
    }

    unsafe fn dec_ref_count(&self) {
        assert!(self.strong.get().is_yes(), "trying to downgrade a non-root");

        let inner = self.inner();
        inner.header.dec_ref_count();

        self.strong.set(Strong::No);
    }

    unsafe fn finalize(&self) {
        // if T holds the only reference to a Gc, that Gc will be collected anyway (cause it won't get marked),
        // so we don't need to run finalize on it here.
        // otherwise that Gc, if any, would be still alive and thus must not be finalized here.
        // so in either case the only reasonable course of action here is doing absolutely nothing.
    }
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
struct RefCellStatus(u32);

impl RefCellStatus {
    const STRONG_MASK: u32 = 1 << 31;
    const REF_MASK: u32 = !Self::STRONG_MASK;

    fn new() -> Self {
        Self(Self::STRONG_MASK)
    }

    pub fn writing(&self) -> bool {
        // indicated by all ones in the ref count part
        self.0 & Self::REF_MASK == Self::REF_MASK
    }

    pub fn reading(&self) -> bool {
        // maps the ref count part:
        // - all ones (borrowed for writing) to 0
        // - all zeros (no borrows) to 1
        // - everything else (borrowed for reading) to values above 1
        self.0 + 1 & Self::REF_MASK > 1
    }

    fn with_reader_added(self) -> Option<Self> {
        if self.0 & Self::REF_MASK >= Self::REF_MASK - 1 {
            // writing or reader count overflow
            None
        } else {
            Some(Self(self.0 + 1))
        }
    }

    fn with_reader_dropped(self) -> Option<Self> {
        if self.0 + 1 & Self::REF_MASK <= 1 {
            // writing or zero reader count
            None
        } else {
            Some(Self(self.0 - 1))
        }
    }

    fn with_writer_added(self) -> Option<Self> {
        if self.0 & Self::REF_MASK != 0 {
            // writing or reading
            None
        } else {
            Some(Self(self.0 | Self::REF_MASK))
        }
    }

    fn with_writer_dropped(self) -> Option<Self> {
        if !self.writing() {
            None
        } else {
            Some(Self(self.0 & !Self::REF_MASK))
        }
    }

    fn strong(&self) -> Strong {
        Strong::from(self.0 & Self::STRONG_MASK != 0)
    }

    fn with_strong(self, strong: Strong) -> Self {
        let strong = if strong.is_yes() {
            Self::STRONG_MASK
        } else {
            0
        };

        Self(self.0 & Self::REF_MASK | strong)
    }
}

pub struct GcRefCell<T: ?Sized> {
    status: Cell<RefCellStatus>,
    inner: UnsafeCell<T>,
}

impl<T> GcRefCell<T> {
    const STRONG_MASK: u32 = 1 << 31;
    const REF_MASK: u32 = !Self::STRONG_MASK;

    pub fn new(value: T) -> Self {
        Self {
            status: Cell::new(RefCellStatus::new()),
            inner: UnsafeCell::new(value),
        }
    }
}

impl<T: ?Sized> GcRefCell<T> {
    pub fn borrow(&self) -> GcRef<'_, T> {
        GcRef::new(self)
    }
}

impl<T: Collect + ?Sized> GcRefCell<T> {
    pub fn borrow_mut(&self) -> GcRefMut<'_, T> {
        GcRefMut::new(self)
    }
}

unsafe impl<T: Collect> Collect for GcRefCell<T> {
    fn mark(&self) {
        // if writing, the contents are treated as a root, so it'll be fine
        if !self.status.get().writing() {
            unsafe {
                (*self.inner.get()).mark();
            }
        }
    }

    unsafe fn inc_ref_count(&self) {
        assert!(
            self.status.get().strong().is_no(),
            "trying to upgrade a root"
        );

        self.status.set(self.status.get().with_strong(Strong::Yes));

        if !self.status.get().writing() {
            unsafe {
                (*self.inner.get()).inc_ref_count();
            }
        }
    }

    unsafe fn dec_ref_count(&self) {
        assert!(
            self.status.get().strong().is_yes(),
            "trying to downgrade a non-root"
        );

        self.status.set(self.status.get().with_strong(Strong::No));

        if !self.status.get().writing() {
            unsafe {
                (*self.inner.get()).dec_ref_count();
            }
        }
    }

    unsafe fn finalize(&self) {
        if !self.status.get().writing() {
            unsafe {
                (*self.inner.get()).finalize();
            }
        }
    }
}

pub struct GcRef<'a, T: ?Sized> {
    cell: &'a GcRefCell<T>,
    value: NonNull<T>,
}

impl<'a, T: ?Sized> GcRef<'a, T> {
    fn new(cell: &'a GcRefCell<T>) -> Self {
        let status = &cell.status;
        assert!(
            !status.get().writing(),
            "GcRefCell is already mutably borrowed"
        );
        status.set(
            status
                .get()
                .with_reader_added()
                .expect("GcRefCell reader count overflow"),
        );

        Self {
            cell,
            value: NonNull::new(cell.inner.get()).unwrap(),
        }
    }
}

impl<T: ?Sized> Deref for GcRef<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { self.value.as_ref() }
    }
}

impl<T: ?Sized> Drop for GcRef<'_, T> {
    fn drop(&mut self) {
        self.cell.status.set(
            self.cell
                .status
                .get()
                .with_reader_dropped()
                .expect("cell has no readers"),
        )
    }
}

pub struct GcRefMut<'a, T: Collect + ?Sized> {
    cell: &'a GcRefCell<T>,
    value: NonNull<T>,
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T: Collect + ?Sized> GcRefMut<'a, T> {
    fn new(cell: &'a GcRefCell<T>) -> Self {
        let status = &cell.status;
        assert!(
            !status.get().writing(),
            "GcRefCell is already mutably borrowed"
        );
        status.set(
            status
                .get()
                .with_writer_added()
                .expect("cannot borrow GcRefCell as mutable while it's immutably borrowed"),
        );

        unsafe {
            if status.get().strong().is_no() {
                (*cell.inner.get().cast_const()).inc_ref_count();
            }
        }

        Self {
            cell,
            value: NonNull::new(cell.inner.get()).unwrap(),
            _marker: PhantomData,
        }
    }
}

impl<T: Collect + ?Sized> Deref for GcRefMut<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { self.value.as_ref() }
    }
}

impl<T: Collect + ?Sized> DerefMut for GcRefMut<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.value.as_mut() }
    }
}

impl<T: Collect + ?Sized> Drop for GcRefMut<'_, T> {
    fn drop(&mut self) {
        self.cell.status.set(
            self.cell
                .status
                .get()
                .with_writer_dropped()
                .expect("cell has no writers"),
        );

        if self.cell.status.get().strong().is_no() {
            unsafe {
                (*self.cell.inner.get().cast_const()).dec_ref_count();
            }
        }
    }
}

// Collect impls ///////////////////////////////////////////////////////////////////////////////////////////////////////

#[macro_export]
macro_rules! impl_collect {
    {} => {
        impl_collect! {
            fn visit(&self) {
                let _ = visit::<()>;
            }
        }
    };

    {
        fn visit(&$self:ident) $body:block
    } => {
        fn mark(&$self) {
            fn visit<T: Collect>(v: &T) {
                <T as Collect>::mark(v)
            }

            $body
        }

        unsafe fn inc_ref_count(&$self) {
            unsafe fn visit<T: Collect>(v: &T) {
                <T as Collect>::inc_ref_count(v)
            }

            $body
        }

        unsafe fn dec_ref_count(&$self) {
            unsafe fn visit<T: Collect>(v: &T) {
                <T as Collect>::dec_ref_count(v)
            }

            $body
        }

        unsafe fn finalize(&$self) {
            unsafe fn visit<T: Collect>(v: &T) {
                <T as Collect>::finalize(v)
            }

            $body
        }
    };
}

macro_rules! impl_empty_collect {
    (unsafe impl Collect for $( $ty:ty ),+;) => {
        $( unsafe impl Collect for $ty { impl_collect!(); } )+
    };
}

macro_rules! impl_tuple_collect {
    ($( ( $( $param:ident ),+ ) ),+ $(,)?) => {
        $(
            unsafe impl<$( $param: Collect ),+> Collect for ($( $param, )+) {
                impl_collect! {
                    fn visit(&self) {
                        #[allow(non_snake_case)]
                        let &( $( ref $param, )+ ) = self;
                        $( visit::<$param>($param); )+
                    }
                }
            }
        )+
    }
}

impl_empty_collect! {
    unsafe impl Collect for
        (),
        u8, u16, u32, u64, u128, usize,
        i8, i16, i32, i64, i128, isize;
}

impl_tuple_collect! {
    (A),
    (A, B),
    (A, B, C),
    (A, B, C, D),
    (A, B, C, D, E),
    (A, B, C, D, E, F),
}

unsafe impl<T: Collect> Collect for Box<T> {
    impl_collect! {
        fn visit(&self) {
            visit::<T>(self);
        }
    }
}

unsafe impl<T: Collect> Collect for Option<T> {
    impl_collect! {
        fn visit(&self) {
            match *self {
                Some(ref value) => visit::<T>(value),
                None => {},
            }
        }
    }
}

unsafe impl<T: Collect> Collect for Vec<T> {
    impl_collect! {
        fn visit(&self) {
            for item in self {
                visit::<T>(item);
            }
        }
    }
}
