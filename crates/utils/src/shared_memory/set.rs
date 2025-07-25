use crate::shared_memory::bitmap::Bitmap;
use std::mem::MaybeUninit;

/// A fixed-capacity indexed set that uses a bitmap to track occupied slots.
/// This is more like a slot map or object pool than a traditional set.
///
/// Note: Consider renaming to `SlotMap` or `IndexedPool` to avoid confusion
/// with standard library collections.
#[repr(C)]
pub struct Set<T, const N: usize> {
    data: [MaybeUninit<T>; N],
    bitmap: Bitmap<N>,
    len: usize,
}

impl<T, const N: usize> Set<T, N> {
    /// Creates a new empty set with capacity N.
    pub fn new() -> Self {
        Self {
            data: [const { MaybeUninit::uninit() }; N],
            bitmap: Bitmap::new(),
            len: 0,
        }
    }

    /// Returns the number of elements in the set.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns the total capacity of the set.
    pub fn capacity(&self) -> usize {
        N
    }

    /// Returns true if the set is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns true if the slot at the given index is occupied.
    pub fn contains(&self, index: usize) -> bool {
        index < N && self.bitmap.test(index as u32)
    }

    /// Inserts a value into the next available slot.
    /// Returns the index of the inserted value, or None if the set is full.
    pub fn insert(&mut self, value: T) -> Option<usize> {
        if let Some(index) = self.bitmap.find_next_zero(0, Some(N as u32)) {
            let index = index as usize;

            // Mark the slot as occupied
            self.bitmap.test_and_set(index as u32, true);

            // Store the value
            self.data[index].write(value);
            self.len += 1;

            Some(index)
        } else {
            None
        }
    }

    /// Removes the value at the given index.
    /// Returns the removed value, or None if the slot was empty.
    pub fn remove(&mut self, index: usize) -> Option<T> {
        if index >= N || !self.bitmap.test(index as u32) {
            return None;
        }

        // Mark the slot as free
        self.bitmap.test_and_set(index as u32, false);
        self.len -= 1;

        // Safety: We know this slot was occupied and contains a valid T
        unsafe { Some(self.data[index].assume_init_read()) }
    }

    /// Gets a reference to the value at the given index.
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= N || !self.bitmap.test(index as u32) {
            return None;
        }

        // Safety: We know this slot is occupied and contains a valid T
        unsafe { Some(self.data[index].assume_init_ref()) }
    }

    /// Gets a mutable reference to the value at the given index.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index >= N || !self.bitmap.test(index as u32) {
            return None;
        }

        // Safety: We know this slot is occupied and contains a valid T
        unsafe { Some(self.data[index].assume_init_mut()) }
    }

    /// Removes all elements from the set.
    pub fn clear(&mut self) {
        for i in 0..N {
            if self.bitmap.test(i as u32) {
                // Safety: We know this slot is occupied
                unsafe {
                    self.data[i].assume_init_drop();
                }
                self.bitmap.test_and_set(i as u32, false);
            }
        }
        self.len = 0;
    }

    /// Returns an iterator over the occupied indices and their values.
    pub fn iter(&self) -> SetIter<'_, T, N> {
        SetIter {
            set: self,
            current: 0,
        }
    }

    /// Returns an iterator over the occupied indices.
    pub fn indices(&self) -> impl Iterator<Item = usize> + '_ {
        (0..N).filter(|&i| self.bitmap.test(i as u32))
    }

    /// Returns an iterator over the values (without indices).
    pub fn values(&self) -> impl Iterator<Item = &T> + '_ {
        self.iter().map(|(_, value)| value)
    }
}

impl<T, const N: usize> Default for Set<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator over occupied slots and their values.
pub struct SetIter<'a, T, const N: usize> {
    set: &'a Set<T, N>,
    current: usize,
}

impl<'a, T, const N: usize> Iterator for SetIter<'a, T, N> {
    type Item = (usize, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        while self.current < N {
            let index = self.current;
            self.current += 1;

            if self.set.bitmap.test(index as u32) {
                // Safety: We know this slot is occupied
                let value = unsafe { self.set.data[index].assume_init_ref() };
                return Some((index, value));
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_set() {
        let set: Set<i32, 10> = Set::new();
        assert_eq!(set.len(), 0);
        assert_eq!(set.capacity(), 10);
        assert!(set.is_empty());
    }

    #[test]
    fn test_insert_and_get() {
        let mut set: Set<String, 5> = Set::new();

        // Insert some values
        let idx1 = set.insert("hello".to_string()).unwrap();
        let idx2 = set.insert("world".to_string()).unwrap();

        assert_eq!(set.len(), 2);
        assert!(!set.is_empty());

        // Check we can get the values
        assert_eq!(set.get(idx1).unwrap(), "hello");
        assert_eq!(set.get(idx2).unwrap(), "world");

        // Check contains
        assert!(set.contains(idx1));
        assert!(set.contains(idx2));
        assert!(!set.contains(999)); // Out of bounds
    }

    #[test]
    fn test_insert_until_full() {
        let mut set: Set<i32, 3> = Set::new();

        // Fill the set
        let idx1 = set.insert(10).unwrap();
        let idx2 = set.insert(20).unwrap();
        let idx3 = set.insert(30).unwrap();

        assert_eq!(set.len(), 3);

        // Try to insert one more - should fail
        assert!(set.insert(40).is_none());
        assert_eq!(set.len(), 3);

        // Verify all values are there
        assert_eq!(*set.get(idx1).unwrap(), 10);
        assert_eq!(*set.get(idx2).unwrap(), 20);
        assert_eq!(*set.get(idx3).unwrap(), 30);
    }

    #[test]
    fn test_remove() {
        let mut set: Set<i32, 10> = Set::new();

        let idx1 = set.insert(100).unwrap();
        let idx2 = set.insert(200).unwrap();
        let idx3 = set.insert(300).unwrap();

        assert_eq!(set.len(), 3);

        // Remove middle element
        let removed = set.remove(idx2).unwrap();
        assert_eq!(removed, 200);
        assert_eq!(set.len(), 2);

        // Check it's no longer accessible
        assert!(!set.contains(idx2));
        assert!(set.get(idx2).is_none());

        // Other elements should still be there
        assert_eq!(*set.get(idx1).unwrap(), 100);
        assert_eq!(*set.get(idx3).unwrap(), 300);

        // Try to remove the same element again
        assert!(set.remove(idx2).is_none());

        // Try to remove non-existent index
        assert!(set.remove(999).is_none());
    }

    #[test]
    fn test_reuse_slots() {
        let mut set: Set<i32, 3> = Set::new();

        // Fill the set
        let _idx1 = set.insert(1).unwrap();
        let idx2 = set.insert(2).unwrap();
        let _idx3 = set.insert(3).unwrap();

        // Remove middle element
        set.remove(idx2);

        // Insert new element - should reuse the freed slot
        let new_idx = set.insert(42).unwrap();
        assert_eq!(new_idx, idx2); // Should reuse the same slot
        assert_eq!(*set.get(new_idx).unwrap(), 42);
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn test_get_mut() {
        let mut set: Set<i32, 5> = Set::new();

        let idx = set.insert(100).unwrap();

        // Modify through mutable reference
        *set.get_mut(idx).unwrap() = 999;

        assert_eq!(*set.get(idx).unwrap(), 999);

        // Try to get mut for non-existent index
        assert!(set.get_mut(999).is_none());
    }

    #[test]
    fn test_clear() {
        let mut set: Set<String, 5> = Set::new();

        set.insert("a".to_string());
        set.insert("b".to_string());
        set.insert("c".to_string());

        assert_eq!(set.len(), 3);

        set.clear();

        assert_eq!(set.len(), 0);
        assert!(set.is_empty());

        // Should be able to insert again
        let idx = set.insert("new".to_string()).unwrap();
        assert_eq!(set.get(idx).unwrap(), "new");
    }

    #[test]
    fn test_iterator() {
        let mut set: Set<i32, 10> = Set::new();

        let idx1 = set.insert(10).unwrap();
        let idx2 = set.insert(20).unwrap();
        let idx3 = set.insert(30).unwrap();

        // Remove middle element
        set.remove(idx2);

        // Collect all items
        let items: Vec<(usize, &i32)> = set.iter().collect();
        assert_eq!(items.len(), 2);

        // Should contain idx1 and idx3, but not idx2
        assert!(items.contains(&(idx1, &10)));
        assert!(items.contains(&(idx3, &30)));
        assert!(!items.iter().any(|(idx, _)| *idx == idx2));
    }

    #[test]
    fn test_indices_iterator() {
        let mut set: Set<char, 5> = Set::new();

        let idx1 = set.insert('a').unwrap();
        let idx2 = set.insert('b').unwrap();
        set.remove(idx1); // Remove first
        let idx3 = set.insert('c').unwrap();

        let indices: Vec<usize> = set.indices().collect();

        // Should contain idx2 and idx3 (and possibly idx1 if reused)
        assert!(indices.contains(&idx2));
        assert!(indices.contains(&idx3));
        assert_eq!(indices.len(), 2);
    }

    #[test]
    fn test_values_iterator() {
        let mut set: Set<i32, 5> = Set::new();

        set.insert(100);
        set.insert(200);
        set.insert(300);

        let values: Vec<&i32> = set.values().collect();
        assert_eq!(values.len(), 3);

        // Values should be present (order might vary)
        assert!(values.contains(&&100));
        assert!(values.contains(&&200));
        assert!(values.contains(&&300));
    }

    #[test]
    fn test_edge_cases() {
        let mut set: Set<i32, 1> = Set::new(); // Capacity of 1

        // Insert one element
        let idx = set.insert(42).unwrap();
        assert_eq!(set.len(), 1);

        // Try to insert another - should fail
        assert!(set.insert(43).is_none());

        // Remove the element
        assert_eq!(set.remove(idx).unwrap(), 42);
        assert_eq!(set.len(), 0);

        // Insert again - should work
        let new_idx = set.insert(44).unwrap();
        assert_eq!(*set.get(new_idx).unwrap(), 44);
    }

    #[test]
    fn test_large_capacity() {
        let mut set: Set<usize, 1000> = Set::new();

        // Insert many elements
        let mut indices = Vec::new();
        for i in 0..100 {
            let idx = set.insert(i * 10).unwrap();
            indices.push(idx);
        }

        assert_eq!(set.len(), 100);

        // Verify all elements
        for (i, &idx) in indices.iter().enumerate() {
            assert_eq!(*set.get(idx).unwrap(), i * 10);
        }

        // Remove every other element
        for &idx in indices.iter().step_by(2) {
            set.remove(idx);
        }

        assert_eq!(set.len(), 50);

        // Verify remaining elements
        for (i, &idx) in indices.iter().enumerate().skip(1).step_by(2) {
            assert_eq!(*set.get(idx).unwrap(), i * 10);
        }
    }
}
