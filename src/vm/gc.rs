//! A primitive garbage collector.
//!
//! Parallels the design of `Rc`/`RefCell`:
//! - `Gc` is an immutable shared reference
//! - `GcCell` allows mutation by enforcing the borrowing rules at runtime

use std::alloc::Layout;
use std::borrow::Borrow;
use std::cell::{Cell, UnsafeCell};
use std::collections::HashSet;
use std::fmt::{self, Debug, Display};
use std::hash::{self, Hash};
use std::marker::PhantomData;
use std::mem::{align_of, size_of_val, MaybeUninit};
use std::ops::{Deref, DerefMut};
use std::ptr::{addr_of, NonNull};

use crate::util::define_yes_no_options;

use thiserror::Error;

define_yes_no_options! {
    enum Marked;
    enum Strong;
    enum Collecting;
    enum Init;
}

// Garbage collector ///////////////////////////////////////////////////////////////////////////////////////////////////

// determines how often GC runs: we launch a collection cycle if `used/heap_size >= LOAD_FACTOR/u8::MAX`.
const LOAD_FACTOR: u8 = 191; // ~75%

pub trait HasVTable {
    const VTABLE: &'static GcVTable;
}

pub trait Finalize {
    unsafe fn finalize(&self) {}
}

pub unsafe trait Collect: Finalize {
    /// Marks all contained objects.
    unsafe fn mark(&self);

    /// Calls `inc_ref_count` on all contained objects.
    ///
    /// The caller ensures this method is not called twice consecutively.
    unsafe fn inc_ref_count(&self);

    /// Calls `dec_ref_count` on all contained objects.
    ///
    /// The caller ensures this method is not called twice consecutively.
    unsafe fn dec_ref_count(&self);

    /// Runs `Finalize::finalize` on self and calls `run_finalizers` on all contained objects.
    unsafe fn run_finalizers(&self);
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
                obj.run_finalizers();
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

// GcBox<T> ////////////////////////////////////////////////////////////////////////////////////////////////////////////

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
    run_finalizers: unsafe fn(value: *const ()),
    drop: unsafe fn(gc_box: *mut ()) -> usize,
}

impl GcVTable {
    pub const fn new_for<T: Collect>() -> GcVTable {
        unsafe fn mark<T: Collect>(this: *const ()) {
            let this = this as *const T;
            this.as_ref().unwrap().mark();
        }

        unsafe fn run_finalizers<T: Collect>(this: *const ()) {
            let this = this as *const T;
            this.as_ref().unwrap().run_finalizers()
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
            run_finalizers: run_finalizers::<T>,
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

    unsafe fn run_finalizers(&self) {
        let vtable = self.vtable();
        (vtable.run_finalizers)(self.value_ptr());
    }

    unsafe fn drop(self) -> usize {
        let vtable = self.vtable();
        (vtable.drop)(self.0.as_ptr())
    }
}

// Gc<T> ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

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

    pub fn ptr_eq(&self, other: &Self) -> bool {
        self.inner.as_ptr() == other.inner.as_ptr()
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

impl<T: ?Sized> AsRef<T> for Gc<'_, T> {
    fn as_ref(&self) -> &T {
        self
    }
}

impl<T: ?Sized> Borrow<T> for Gc<'_, T> {
    fn borrow(&self) -> &T {
        self
    }
}

impl<T: Debug + ?Sized> Debug for Gc<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Gc").field(&&**self).finish()
    }
}

impl<T: Display + ?Sized> Display for Gc<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <T as Display>::fmt(self, f)
    }
}

impl<T: Hash + ?Sized> Hash for Gc<'_, T> {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        <T as Hash>::hash(self, state)
    }
}

impl<T: Ord> Ord for Gc<'_, T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        <T as Ord>::cmp(self, other)
    }
}

impl<'a, 'b, T: PartialOrd + ?Sized> PartialOrd<Gc<'a, T>> for Gc<'b, T> {
    fn partial_cmp(&self, other: &Gc<'a, T>) -> Option<std::cmp::Ordering> {
        <T as PartialOrd>::partial_cmp(self, other)
    }

    fn lt(&self, other: &Gc<'a, T>) -> bool {
        <T as PartialOrd>::lt(self, other)
    }

    fn le(&self, other: &Gc<'a, T>) -> bool {
        <T as PartialOrd>::le(self, other)
    }

    fn gt(&self, other: &Gc<'a, T>) -> bool {
        <T as PartialOrd>::gt(self, other)
    }

    fn ge(&self, other: &Gc<'a, T>) -> bool {
        <T as PartialOrd>::ge(self, other)
    }
}

impl<T: Eq + ?Sized> Eq for Gc<'_, T> {}

impl<'a, 'b, T: PartialEq + ?Sized> PartialEq<Gc<'a, T>> for Gc<'b, T> {
    fn eq(&self, other: &Gc<'a, T>) -> bool {
        <T as PartialEq>::eq(self, other)
    }

    fn ne(&self, other: &Gc<'a, T>) -> bool {
        <T as PartialEq>::ne(self, other)
    }
}

impl<T> Finalize for Gc<'_, T> {}

unsafe impl<T: Collect> Collect for Gc<'_, T> {
    unsafe fn mark(&self) {
        if self.inner().header.mark().is_no() {
            self.inner().value.mark()
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

    unsafe fn run_finalizers(&self) {
        // if T holds the only reference to a Gc, that Gc will be collected anyway (cause it won't get marked),
        // so we don't need to run finalize on it here.
        // otherwise that Gc, if any, would be still alive and thus must not be finalized here.
        // so in either case the only reasonable course of action here is doing absolutely nothing.
    }
}

// GcRefCell<T> ////////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Error, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BorrowError {
    #[error("already mutably borrowed")]
    MutablyBorrowed,
    #[error("reader borrow count overflow")]
    TooManyBorrows,
}

#[derive(Error, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BorrowMutError {
    #[error("already mutably borrowed")]
    MutablyBorrowed,
    #[error("already immutably borrowed")]
    ImmutablyBorrowed,
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
    pub fn try_borrow(&self) -> Result<GcRef<'_, T>, BorrowError> {
        GcRef::new(self)
    }

    pub fn borrow(&self) -> GcRef<'_, T> {
        self.try_borrow().unwrap()
    }
}

impl<T: Collect + ?Sized> GcRefCell<T> {
    pub fn try_borrow_mut(&self) -> Result<GcRefMut<'_, T>, BorrowMutError> {
        GcRefMut::new(self)
    }

    pub fn borrow_mut(&self) -> GcRefMut<'_, T> {
        self.try_borrow_mut().unwrap()
    }
}

impl<T: Clone> Clone for GcRefCell<T> {
    fn clone(&self) -> Self {
        GcRefCell::new(self.borrow().clone())
    }
}

impl<T: Default> Default for GcRefCell<T> {
    fn default() -> Self {
        GcRefCell::new(T::default())
    }
}

impl<T: Debug + ?Sized> Debug for GcRefCell<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut r = f.debug_tuple("GcRefCell");

        let r = match self.try_borrow() {
            Ok(value) => r.field(&&*value),
            Err(BorrowError::MutablyBorrowed) => r.field(&"<mutably borrowed>"),
            Err(BorrowError::TooManyBorrows) => r.field(&"<too many borrows>"),
        };

        r.finish()
    }
}

impl<T: Ord + ?Sized> Ord for GcRefCell<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        <T as Ord>::cmp(&self.borrow(), &other.borrow())
    }
}

impl<T: PartialOrd + ?Sized> PartialOrd for GcRefCell<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        <T as PartialOrd>::partial_cmp(&self.borrow(), &other.borrow())
    }
}

impl<T: Eq + ?Sized> Eq for GcRefCell<T> {}

impl<T: PartialEq + ?Sized> PartialEq for GcRefCell<T> {
    fn eq(&self, other: &Self) -> bool {
        <T as PartialEq>::eq(&self.borrow(), &other.borrow())
    }

    fn ne(&self, other: &Self) -> bool {
        <T as PartialEq>::ne(&self.borrow(), &other.borrow())
    }
}

impl<T> Finalize for GcRefCell<T> {}

unsafe impl<T: Collect> Collect for GcRefCell<T> {
    unsafe fn mark(&self) {
        // if writing, the contents are treated as a root, so it'll be fine
        if !self.status.get().writing() {
            (*self.inner.get()).mark();
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

    unsafe fn run_finalizers(&self) {
        self.finalize();

        if !self.status.get().writing() {
            unsafe {
                (*self.inner.get()).run_finalizers();
            }
        }
    }
}

pub struct GcRef<'a, T: ?Sized> {
    cell: &'a GcRefCell<T>,
    value: NonNull<T>,
}

impl<'a, T: ?Sized> GcRef<'a, T> {
    fn new(cell: &'a GcRefCell<T>) -> Result<Self, BorrowError> {
        let status = &cell.status;

        if status.get().writing() {
            return Err(BorrowError::MutablyBorrowed);
        }

        status.set(
            status
                .get()
                .with_reader_added()
                .ok_or(BorrowError::TooManyBorrows)?,
        );

        Ok(Self {
            cell,
            value: NonNull::new(cell.inner.get()).unwrap(),
        })
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
    fn new(cell: &'a GcRefCell<T>) -> Result<Self, BorrowMutError> {
        let status = &cell.status;

        if status.get().writing() {
            return Err(BorrowMutError::MutablyBorrowed);
        }

        status.set(
            status
                .get()
                .with_writer_added()
                .ok_or(BorrowMutError::ImmutablyBorrowed)?,
        );

        unsafe {
            if status.get().strong().is_no() {
                (*cell.inner.get().cast_const()).inc_ref_count();
            }
        }

        Ok(Self {
            cell,
            value: NonNull::new(cell.inner.get()).unwrap(),
            _marker: PhantomData,
        })
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

// GcOnceCell<T> ///////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
enum OnceCellStatus {
    UninitWeak,

    #[default]
    UninitStrong,

    InitWeak,
    InitStrong,
}

impl OnceCellStatus {
    fn strong(&self) -> Strong {
        matches!(
            self,
            OnceCellStatus::UninitStrong | OnceCellStatus::InitStrong
        )
        .into()
    }

    fn init(&self) -> Init {
        matches!(self, OnceCellStatus::InitWeak | OnceCellStatus::InitStrong).into()
    }

    fn with_strong(self, strong: Strong) -> Self {
        if strong.is_yes() {
            match self {
                Self::UninitWeak => Self::UninitStrong,
                Self::InitWeak => Self::InitStrong,
                _ => self,
            }
        } else {
            match self {
                Self::UninitStrong => Self::UninitWeak,
                Self::InitStrong => Self::InitWeak,
                _ => self,
            }
        }
    }

    fn with_init(self, init: Init) -> Self {
        if init.is_yes() {
            match self {
                Self::UninitWeak => Self::InitWeak,
                Self::UninitStrong => Self::InitStrong,
                _ => self,
            }
        } else {
            match self {
                Self::InitWeak => Self::UninitWeak,
                Self::InitStrong => Self::UninitStrong,
                _ => self,
            }
        }
    }
}

pub struct GcOnceCell<T> {
    status: Cell<OnceCellStatus>,
    inner: UnsafeCell<MaybeUninit<T>>,
}

impl<T> GcOnceCell<T> {
    pub fn new() -> Self {
        Self {
            status: Default::default(),
            inner: UnsafeCell::new(MaybeUninit::uninit()),
        }
    }

    pub fn new_init(value: T) -> Self {
        Self {
            status: Cell::new(OnceCellStatus::InitStrong),
            inner: UnsafeCell::new(MaybeUninit::new(value)),
        }
    }

    pub fn get(&self) -> Option<&T> {
        if self.status.get().init().is_yes() {
            Some(unsafe { (*self.inner.get()).assume_init_ref() })
        } else {
            None
        }
    }
}

impl<T: Collect> GcOnceCell<T> {
    pub fn set(&self, value: T) -> Result<(), T> {
        let status = self.status.get();

        if status.init().is_yes() {
            return Err(value);
        }

        let inner = unsafe { (*self.inner.get()).write(value) };

        if status.strong().is_no() {
            // self is on the garbage-collected heap, so value is no longer a root
            unsafe {
                inner.dec_ref_count();
            }
        }

        self.status.set(status.with_init(Init::Yes));

        Ok(())
    }

    pub fn get_or_init<F>(&self, f: F) -> &T
    where
        F: FnOnce() -> T,
    {
        if let Some(value) = self.get() {
            return value;
        }

        let value = f();
        assert!(self.set(value).is_ok(), "reentrant initialization");

        self.get().unwrap()
    }
}

impl<T> Drop for GcOnceCell<T> {
    fn drop(&mut self) {
        if self.status.get().init().is_yes() {
            unsafe { self.inner.get_mut().assume_init_drop() };
        }
    }
}

impl<T> Default for GcOnceCell<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Debug> Debug for GcOnceCell<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut d = f.debug_tuple("GcOnceCell");

        match self.get() {
            Some(value) => d.field(value),
            None => d.field(&"<uninit>"),
        };

        d.finish()
    }
}

impl<T: Clone> Clone for GcOnceCell<T> {
    fn clone(&self) -> Self {
        match self.get() {
            Some(value) => Self::new_init(value.clone()),
            None => Self::new(),
        }
    }
}

impl<T: PartialEq> PartialEq for GcOnceCell<T> {
    fn eq(&self, other: &Self) -> bool {
        self.get() == other.get()
    }
}

impl<T: Eq> Eq for GcOnceCell<T> {}

impl<T> Finalize for GcOnceCell<T> {}

unsafe impl<T: Collect> Collect for GcOnceCell<T> {
    unsafe fn mark(&self) {
        if let Some(value) = self.get() {
            value.mark();
        }
    }

    unsafe fn inc_ref_count(&self) {
        let status = self.status.get();
        assert!(status.strong().is_no(), "trying to upgrade a root");
        self.status.set(status.with_strong(Strong::Yes));

        if let Some(value) = self.get() {
            value.inc_ref_count();
        }
    }

    unsafe fn dec_ref_count(&self) {
        let status = self.status.get();
        assert!(status.strong().is_yes(), "trying to downgrade a non-root");
        self.status.set(status.with_strong(Strong::No));

        if let Some(value) = self.get() {
            value.dec_ref_count();
        }
    }

    unsafe fn run_finalizers(&self) {
        self.finalize();

        if let Some(value) = self.get() {
            value.run_finalizers();
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
        unsafe fn mark(&$self) {
            unsafe fn visit<T: Collect>(v: &T) {
                <T as $crate::vm::gc::Collect>::mark(v)
            }

            $body
        }

        unsafe fn inc_ref_count(&$self) {
            unsafe fn visit<T: Collect>(v: &T) {
                <T as $crate::vm::gc::Collect>::inc_ref_count(v)
            }

            $body
        }

        unsafe fn dec_ref_count(&$self) {
            unsafe fn visit<T: Collect>(v: &T) {
                <T as $crate::vm::gc::Collect>::dec_ref_count(v)
            }

            $body
        }

        unsafe fn run_finalizers(&$self) {
            unsafe fn visit<T: Collect>(v: &T) {
                <T as $crate::vm::gc::Collect>::run_finalizers(v)
            }

            <Self as $crate::vm::gc::Finalize>::finalize($self);

            $body
        }
    };
}

macro_rules! impl_empty_collect {
    (unsafe impl Collect for $( $ty:ty ),+;) => {
        $(
            impl Finalize for $ty {}
            unsafe impl Collect for $ty { impl_collect!(); }
        )+
    };
}

macro_rules! impl_tuple_collect {
    ($( ( $( $param:ident ),+ ) ),+ $(,)?) => {
        $(
            impl<$( $param ),+> Finalize for ($( $param, )+) {}

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

impl<T> Finalize for Box<T> {}

unsafe impl<T: Collect> Collect for Box<T> {
    impl_collect! {
        fn visit(&self) {
            visit::<T>(self);
        }
    }
}

impl<T> Finalize for Option<T> {}

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

impl<T> Finalize for Vec<T> {}

unsafe impl<T: Collect> Collect for Vec<T> {
    impl_collect! {
        fn visit(&self) {
            for item in self {
                visit::<T>(item);
            }
        }
    }
}
