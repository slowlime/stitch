use std::ops::Index;

use slotmap::{Key, SecondaryMap};

#[derive(Debug, Default, Clone)]
pub struct SeqSlot<K: Key> {
    map: SecondaryMap<K, usize>,
    next_idx: usize,
}

impl<K: Key> SeqSlot<K> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, key: K) -> Option<usize> {
        use slotmap::secondary::Entry;

        Some(match self.map.entry(key)? {
            Entry::Occupied(entry) => *entry.get(),

            Entry::Vacant(entry) => {
                let idx = self.next_idx;
                self.next_idx += 1;

                *entry.insert(idx)
            }
        })
    }

    pub fn get(&self, key: K) -> Option<usize> {
        self.map.get(key).copied()
    }
}

impl<K: Key> Index<K> for SeqSlot<K> {
    type Output = usize;

    fn index(&self, key: K) -> &Self::Output {
        &self.map[key]
    }
}
