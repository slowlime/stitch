use stitch::impl_collect;
use stitch::vm::gc::{Collect, GarbageCollector, Gc, GcRefCell};

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

    unsafe impl<'gc> Collect for Bomb<'gc> {
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
