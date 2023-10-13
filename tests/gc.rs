use std::cell::Cell;
use std::mem;
use std::rc::Rc;

use stitch::impl_collect;
use stitch::vm::gc::{Collect, Finalize, GarbageCollector, Gc, GcOnceCell, GcRefCell};

#[derive(Debug)]
struct DropFlagger(Rc<Cell<usize>>);

impl DropFlagger {
    fn new() -> (Self, Rc<Cell<usize>>) {
        let cell = Rc::new(Cell::new(0));
        let cell_clone = cell.clone();

        (Self(cell), cell_clone)
    }
}

impl Drop for DropFlagger {
    fn drop(&mut self) {
        let cell = &self.0;
        cell.set(cell.get() + 1);
    }
}

impl Finalize for DropFlagger {}

unsafe impl Collect for DropFlagger {
    impl_collect!();
}

#[test]
fn test_gc() {
    let gc = GarbageCollector::new();

    let a = Gc::new(&gc, ());
    let b = Gc::new(&gc, (a.clone(), a.clone()));
    drop(a);
    drop(b);
}

#[test]
fn test_gc_ref_cell() {
    struct Bomb<'gc>(GcRefCell<Option<Gc<'gc, Bomb<'gc>>>>);

    impl Finalize for Bomb<'_> {}

    unsafe impl Collect for Bomb<'_> {
        impl_collect! {
            fn visit(&self) {
                visit(&self.0);
            }
        }
    }

    let gc = GarbageCollector::new();

    let a = Gc::new(&gc, Bomb(GcRefCell::new(None)));
    let b = Gc::new(&gc, Bomb(GcRefCell::new(Some(a.clone()))));
    *a.0.borrow_mut() = Some(b);

    drop(a);
}

#[test]
fn test_gc_once_cell() {
    let cell = GcOnceCell::<Gc<DropFlagger>>::new();
    // should do nothing
    drop(cell);

    let gc = GarbageCollector::new();

    let cell = Gc::new(&gc, GcOnceCell::<Gc<DropFlagger>>::new());
    // should do nothing
    drop(cell);

    let cell = Gc::new(&gc, GcOnceCell::<Gc<DropFlagger>>::new());
    let (flagger, flag) = DropFlagger::new();
    cell.set(Gc::new(&gc, flagger)).expect("init must succeed");
    assert_eq!(flag.get(), 0);
    drop(cell);
    gc.collect();
    assert_eq!(flag.get(), 1);
}

#[test]
fn test_gc_finalization() {
    struct Bomb<'gc> {
        from: GcRefCell<Gc<'gc, DropFlagger>>,
        to: Gc<'gc, GcRefCell<Gc<'gc, DropFlagger>>>,
    }

    unsafe impl Collect for Bomb<'_> {
        impl_collect! {
            fn visit(&self) {
                visit(&self.from);
                visit(&self.to);
            }
        }
    }

    impl Finalize for Bomb<'_> {
        unsafe fn finalize(&self) {
            mem::swap(&mut self.from.borrow_mut(), &mut self.to.borrow_mut());
        }
    }

    let gc = GarbageCollector::new();

    let (flagger1, flag1) = DropFlagger::new();
    let (flagger2, flag2) = DropFlagger::new();
    let (flagger3, flag3) = DropFlagger::new();

    let from = GcRefCell::new(Gc::new(&gc, flagger1));
    let to = Gc::new(&gc, GcRefCell::new(Gc::new(&gc, flagger2)));
    let flagger_gc3 = Gc::new(&gc, flagger3);
    let to_root = to.clone();

    // { from: Some(flagger1), to: Some(flagger2) }
    let bomb = Bomb { from, to };

    // { from: Some(flagger1), to: Some(flagger3) }
    // flagger2 is no longer alive, but it has yet to be collected
    *bomb.to.borrow_mut() = flagger_gc3;
    assert_eq!(flag2.get(), 0);

    // bomb is no longer alive...
    drop(bomb);
    assert_eq!(flag1.get(), 0);
    assert_eq!(flag3.get(), 0);

    // ...and flagger3 would get collected except the bomb's finalizer puts it into a rooted GcRefCell
    gc.collect();

    assert_eq!(flag1.get(), 1);
    assert_eq!(flag2.get(), 1);
    assert_eq!(flag3.get(), 0);

    drop(to_root);
    gc.collect();
    assert_eq!(flag3.get(), 1);
}
