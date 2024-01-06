pub mod seq {
    use std::ops::Index;

    use slotmap::{Key, SecondaryMap};

    /// Maps a SlotMap's entries to consecutive indices.
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
}

pub use seq::SeqSlot;

pub mod bi {
    use std::borrow::Borrow;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::ops::Index;

    use hashbrown::HashTable;
    use slotmap::{Key, SlotMap};
    use thiserror::Error;

    pub type Iter<'a, K, V> = slotmap::basic::Iter<'a, K, V>;
    pub type IntoIter<K, V> = slotmap::basic::IntoIter<K, V>;
    pub type Keys<'a, K, V> = slotmap::basic::Keys<'a, K, V>;
    pub type Values<'a, K, V> = slotmap::basic::Values<'a, K, V>;

    /// A bidirectional SlotMap.
    #[derive(Debug, Clone)]
    pub struct BiSlotMap<K: Key, V> {
        values: SlotMap<K, V>,
        keys: HashTable<K>,
    }

    impl<K: Key, V: Hash + Eq> BiSlotMap<K, V> {
        pub fn new() -> Self {
            Self::default()
        }

        pub fn with_key() -> Self {
            Self {
                values: SlotMap::with_key(),
                keys: Default::default(),
            }
        }

        pub fn len(&self) -> usize {
            self.values.len()
        }

        pub fn is_empty(&self) -> bool {
            self.values.is_empty()
        }

        pub fn contains_key(&self, key: K) -> bool {
            self.values.contains_key(key)
        }

        pub fn contains_value<U>(&self, v: &U) -> bool
        where
            V: Borrow<U>,
            U: Hash + Eq + ?Sized,
        {
            self.get_key(v).is_some()
        }

        pub fn insert(&mut self, value: V) -> K {
            use hashbrown::hash_table::Entry;

            let entry = self.keys.entry(
                Self::hash_value(&value),
                |&k| self.values[k] == value,
                |&k| Self::hash_key(&self.values, k).unwrap(),
            );

            match entry {
                Entry::Occupied(entry) => *entry.get(),

                Entry::Vacant(entry) => {
                    let key = self.values.insert(value);
                    entry.insert(key);

                    key
                }
            }
        }

        pub fn remove(&mut self, key: K) -> Option<V> {
            let value = self.values.remove(key)?;
            self.keys
                .find_entry(Self::hash_value(&value), |&k| k == key)
                .unwrap()
                .remove();

            Some(value)
        }

        pub fn remove_value<U>(&mut self, v: &U) -> Option<V>
        where
            V: Borrow<U>,
            U: Hash + Eq + ?Sized,
        {
            let entry = self
                .keys
                .find_entry(Self::hash_value(v), |&k| self.values[k].borrow() == v)
                .ok()?;
            let value = self.values.remove(*entry.get()).unwrap();
            entry.remove();

            Some(value)
        }

        pub fn clear(&mut self) {
            self.values.clear();
            self.keys.clear();
        }

        pub fn get(&self, key: K) -> Option<&V> {
            self.values.get(key)
        }

        pub fn get_key<U>(&self, v: &U) -> Option<K>
        where
            V: Borrow<U>,
            U: Hash + Eq + ?Sized,
        {
            self.keys
                .find(Self::hash_value(v), |&k| self[k].borrow() == v)
                .copied()
        }

        pub fn iter(&self) -> Iter<'_, K, V> {
            self.values.iter()
        }

        pub fn keys(&self) -> Keys<'_, K, V> {
            self.values.keys()
        }

        pub fn values(&self) -> Values<'_, K, V> {
            self.values.values()
        }

        fn hash_value<U>(v: &U) -> u64
        where
            V: Borrow<U>,
            U: Hash + ?Sized,
        {
            let mut hasher = DefaultHasher::new();
            v.hash(&mut hasher);

            hasher.finish()
        }

        fn hash_key(values: &SlotMap<K, V>, k: K) -> Option<u64> {
            values.get(k).map(Self::hash_value)
        }
    }

    impl<K: Key, V: Hash + Eq> Default for BiSlotMap<K, V> {
        fn default() -> Self {
            Self::with_key()
        }
    }

    impl<K: Key, V> Index<K> for BiSlotMap<K, V> {
        type Output = V;

        fn index(&self, key: K) -> &Self::Output {
            &self.values[key]
        }
    }

    impl<'a, K: Key, V> IntoIterator for &'a BiSlotMap<K, V> {
        type Item = (K, &'a V);
        type IntoIter = Iter<'a, K, V>;

        fn into_iter(self) -> Self::IntoIter {
            self.values.iter()
        }
    }

    impl<K: Key, V> IntoIterator for BiSlotMap<K, V> {
        type Item = (K, V);
        type IntoIter = IntoIter<K, V>;

        fn into_iter(self) -> Self::IntoIter {
            self.values.into_iter()
        }
    }

    #[derive(Error, Debug, Clone, PartialEq, Eq, Hash)]
    #[error("slot map contains a duplicate value")]
    pub struct DuplicateValueError<V> {
        pub value: V,
    }

    impl<K: Key, V: Hash + Eq> BiSlotMap<K, V> {
        pub fn from_slot_map_lossy(values: SlotMap<K, V>) -> Self {
            match Self::from_slot_map(values, false) {
                Ok(result) => result,
                Err(_) => unreachable!(),
            }
        }

        fn from_slot_map(values: SlotMap<K, V>, fail_on_duplicate: bool) -> Result<Self, DuplicateValueError<V>> {
            use hashbrown::hash_table::Entry;

            let mut result = Self {
                values,
                keys: Default::default(),
            };

            for (key, value) in &result.values {
                let entry = result.keys.entry(
                    Self::hash_value(&value),
                    |&k| &result.values[k] == value,
                    |&k| Self::hash_key(&result.values, k).unwrap(),
                );

                match entry {
                    Entry::Occupied(_) if fail_on_duplicate => {
                        let value = result.values.remove(key).unwrap();

                        return Err(DuplicateValueError { value });
                    }

                    Entry::Occupied(_) => {
                        // ignore
                    }

                    Entry::Vacant(entry) => {
                        entry.insert(key);
                    }
                }
            }

            Ok(result)
        }
    }

    impl<K: Key, V: Hash + Eq> TryFrom<SlotMap<K, V>> for BiSlotMap<K, V> {
        type Error = DuplicateValueError<V>;

        fn try_from(values: SlotMap<K, V>) -> Result<Self, Self::Error> {
            Self::from_slot_map(values, true)
        }
    }
}

pub use bi::BiSlotMap;
