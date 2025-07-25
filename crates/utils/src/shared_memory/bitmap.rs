macro_rules! div_round_up {
    ($n:expr, $d:expr) => {
        $n.div_ceil($d)
    };
}

#[repr(C)]
pub struct Bitmap<const N: usize>([u64; N]);

impl<const N: usize> Default for Bitmap<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> Bitmap<N> {
    pub fn new() -> Self {
        Self([0; N])
    }

    /// Returns the bit of the `offset` position.
    /// true - 1
    /// false - 0
    pub fn test(&self, offset: u32) -> bool {
        let bit_mask = Self::bit_mask(offset);
        let idx = (offset / u64::BITS) as usize;
        let row = self.0[idx];
        (row & bit_mask) == bit_mask
    }

    /// Set the bit at the `offset` position to `val`,
    /// and return the value before it was set.
    /// true - 1
    /// false - 0
    pub fn test_and_set(&mut self, offset: u32, val: bool) -> bool {
        let bit_mask = Self::bit_mask(offset);
        let idx = (offset / u64::BITS) as usize;
        let row = self.0[idx];
        self.0[idx] = if val { row | bit_mask } else { row & !bit_mask };
        (row & bit_mask) == bit_mask
    }

    /// Returns the position of the next 0,
    /// after `offset` (including `offset`) and before `end` (excluding `end`).
    /// None means not existing
    pub fn find_next_zero(&self, offset: u32, end: Option<u32>) -> Option<u32> {
        let mut next_zero = None;
        let col = offset & (u64::BITS - 1);
        if col != 0 {
            // offset in the middle of usize
            let row = offset / u64::BITS;
            let num = self.0[row as usize] | (((1_u64 << col) - 1) << (u64::BITS - col));

            if num != u64::MAX {
                next_zero = Some(row * u64::BITS + num.leading_ones());
            }
        }

        if next_zero.is_none() {
            for i in div_round_up!(offset, u64::BITS)..self.0.len() as u32 {
                let num = unsafe { *self.0.get_unchecked(i as usize) };
                if num == 0 {
                    next_zero = Some(i * u64::BITS);
                    break;
                } else if num == u64::MAX {
                    continue;
                } else {
                    next_zero = Some(i * u64::BITS + num.leading_ones());
                    break;
                }
            }
        }
        next_zero.and_then(|nz| match end {
            Some(end) if nz >= end => None,
            _ => Some(nz),
        })
    }

    #[inline(always)]
    fn bit_mask(offset: u32) -> u64 {
        (1 << (u64::BITS - 1)) >> (offset & (u64::BITS - 1))
    }
}

#[cfg(test)]
mod test {

    use super::Bitmap;

    #[test]
    fn bitmap_test() {
        let mut bitmap = Bitmap::<2>::new();
        assert!(!bitmap.test(1));
        bitmap.test_and_set(1, true);
        assert!(bitmap.test(1));

        assert!(!bitmap.test(127));
        bitmap.test_and_set(127, true);
        assert!(bitmap.test(127));
    }

    #[test]
    fn bitmap_test_and_set() {
        let mut bitmap = Bitmap::<512>::new();
        assert!(!bitmap.test_and_set(0, true));
        assert!(bitmap.test_and_set(0, true));

        assert!(!bitmap.test_and_set(1, true));
        assert!(bitmap.test_and_set(1, true));

        assert!(!bitmap.test_and_set(63, true));
        assert!(bitmap.test_and_set(63, true));
        assert!(!bitmap.test_and_set(64, true));
        assert!(bitmap.test_and_set(64, true));
        assert!(!bitmap.test_and_set(65, true));
        assert!(bitmap.test_and_set(65, true));

        let len = bitmap.0.len() as u32 * u64::BITS;
        assert!(!bitmap.test_and_set(len - 1, true));
        assert!(bitmap.test_and_set(len - 1, true));
    }

    #[test]
    fn bitmap_clear() {
        let mut bitmap = Bitmap::<512>::new();
        bitmap.test_and_set(0, true);
        bitmap.test_and_set(0, false);
        assert!(!bitmap.test_and_set(0, true));

        bitmap.test_and_set(1, true);
        bitmap.test_and_set(2, true);
        bitmap.test_and_set(0, false);
        assert!(!bitmap.test_and_set(0, true));
        assert!(bitmap.test_and_set(1, true));

        bitmap.test_and_set(1, false);
        assert!(!bitmap.test_and_set(1, true));

        bitmap.test_and_set(63, true);
        bitmap.test_and_set(63, false);
        assert!(!bitmap.test_and_set(63, true));

        bitmap.test_and_set(64, true);
        bitmap.test_and_set(64, false);
        assert!(!bitmap.test_and_set(64, true));

        bitmap.test_and_set(100, false);
        assert!(!bitmap.test_and_set(100, true));
    }

    #[test]
    fn bitmap_find_next_zero() {
        let mut bitmap = Bitmap::<512>::new();
        assert_eq!(bitmap.find_next_zero(0, None), Some(0));

        assert!(!bitmap.test_and_set(63, true));
        assert_eq!(bitmap.find_next_zero(0, None), Some(0));
        assert_eq!(bitmap.find_next_zero(63, None), Some(64));

        assert!(!bitmap.test_and_set(0, true));
        assert_eq!(bitmap.find_next_zero(0, None), Some(1));

        assert!(!bitmap.test_and_set(1, true));
        assert_eq!(bitmap.find_next_zero(0, None), Some(2));

        assert!(!bitmap.test_and_set(300, true));
        assert_eq!(bitmap.find_next_zero(300, None), Some(301));
        assert_eq!(bitmap.find_next_zero(400, None), Some(400));

        assert!(!bitmap.test_and_set(64, true));
        assert_eq!(bitmap.find_next_zero(64, None), Some(65));

        assert!(!bitmap.test_and_set(65, true));
        assert_eq!(bitmap.find_next_zero(64, None), Some(66));

        assert!(!bitmap.test_and_set(32767, true));
        assert_eq!(bitmap.find_next_zero(32766, None), Some(32766));
        assert_eq!(bitmap.find_next_zero(32767, None), None);

        let mut bitmap = Bitmap::<512>::new();
        for i in 0..=32766 {
            bitmap.test_and_set(i, true);
        }
        assert_eq!(bitmap.find_next_zero(0, None), Some(32767));
        bitmap.test_and_set(32767, true);
        assert_eq!(bitmap.find_next_zero(0, None), None);
    }

    #[test]
    fn bitmap_find_next_zero_with_end() {
        let mut bitmap = Bitmap::<2>::new();
        assert_eq!(bitmap.find_next_zero(0, Some(10)), Some(0));

        bitmap.test_and_set(0, true);
        bitmap.test_and_set(1, true);
        assert_eq!(bitmap.find_next_zero(0, None), Some(2));
        assert_eq!(bitmap.find_next_zero(0, Some(3)), Some(2));
        assert_eq!(bitmap.find_next_zero(0, Some(2)), None);
    }
}
