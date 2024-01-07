/// Splits an iterator into contiguous segments in the input sequence matching a predicate
/// and calls the provided callback for each segment.
///
/// The segments allow for a gap (comprised of elements failing the predicate)
/// up to the provided size.
pub fn segments<S, I, F, P>(source: S, max_gap: usize, mut predicate: P, mut callback: F)
where
    S: IntoIterator<Item = I>,
    F: FnMut(&mut Segment<S::IntoIter, P, I>),
    P: FnMut(&I) -> bool,
{
    let mut iter = source.into_iter();
    let mut offset = 0;
    let buffer = &mut vec![];

    while let Some(item) = iter.next() {
        if predicate(&item) {
            let mut segment = Segment {
                first_item: Some(item),
                buffer,
                iter: &mut iter,
                offset,
                gap: 0,
                max_gap,
                predicate,
            };
            callback(&mut segment);

            // consume the rest of the segment
            for _ in &mut segment {}
            offset = segment.offset;
            predicate = segment.predicate;
            buffer.clear();
        }

        offset += 1;
    }
}

pub struct Segment<'a, I, P, T> {
    first_item: Option<T>,
    iter: &'a mut I,
    offset: usize,
    max_gap: usize,
    predicate: P,

    // invariants:
    // 1. gap == 0 || gap == max_gap || gap == buffer.len()
    // 2. gap > max_gap: the gap is too wide
    // 3. gap == 0 && !buffer.empty(): the gap has ended; buffer.last() matched the predicate
    // 4. 0 < gap < max_gap: buffer.len() == gap and we're in the middle of a gap
    buffer: &'a mut Vec<T>,
    gap: usize,
}

impl<I, P, T> Segment<'_, I, P, T> {
    /// Returns the index of the last value returned from `next` in the original sequence.
    pub fn offset(&self) -> usize {
        self.offset - self.buffer.len()
    }
}

impl<'a, I, P, T> Iterator for Segment<'a, I, P, T>
where
    I: Iterator<Item = T>,
    P: FnMut(&T) -> bool,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(item) = self.first_item.take() {
            return Some(item);
        } else if self.gap == 0 && !self.buffer.is_empty() {
            return Some(self.buffer.remove(0));
        } else if self.gap > self.max_gap {
            return None;
        }

        loop {
            self.offset += 1;
            let item = self.iter.next()?;

            if (self.predicate)(&item) {
                if self.gap == 0 {
                    // we can return the item right away
                    return Some(item);
                }

                // have to process the gap items first
                let gap_item = self.buffer.remove(0);
                self.buffer.push(item);
                self.gap = 0;

                return Some(gap_item);
            }

            // we're in the middle of a gap
            self.gap += 1;

            if self.gap > self.max_gap {
                // the gap is too wide
                return None;
            }

            if self.buffer.capacity() == 0 {
                self.buffer.reserve_exact(self.max_gap);
            }

            self.buffer.push(item);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::segments;

    #[test]
    fn test_segments_empty() {
        segments(
            iter::empty::<i32>(),
            1,
            |&x| x == 42,
            |_| panic!("must not have any segments"),
        );
    }

    #[test]
    fn test_segments_none() {
        segments(
            [1, 2, 3, 4, 5],
            2,
            |&x| x > 10,
            |_| panic!("must not have any segments"),
        );
    }

    #[test]
    fn test_segments_zero_gap() {
        let expected: Vec<Vec<(usize, i32)>> = vec![
            vec![(0, 1), (1, 3)],
            vec![(3, 5), (4, 7), (5, 9)],
            vec![(8, 13)],
        ];
        let mut idx = 0;

        segments(
            [1, 3, 4, 5, 7, 9, 10, 12, 13, 14, 16, 18],
            0,
            |&x| x % 2 == 1,
            |segment| {
                let mut actual = vec![];

                while let Some(item) = segment.next() {
                    actual.push((segment.offset(), item));
                }

                assert_eq!(expected[idx], actual);
                idx += 1;
            },
        );
    }

    #[test]
    fn test_segments_non_zero_gap() {
        let expected: Vec<Vec<(usize, i32)>> = vec![
            vec![(0, 1), (1, 3), (2, 4), (3, 6), (4, 8), (5, 9)],
            vec![(10, 17), (11, 19)],
        ];
        let mut idx = 0;

        segments(
            [1, 3, 4, 6, 8, 9, 10, 12, 14, 16, 17, 19, 20],
            3,
            |&x| x % 2 == 1,
            |segment| {
                let mut actual = vec![];

                while let Some(item) = segment.next() {
                    actual.push((segment.offset(), item));
                }

                assert_eq!(expected[idx], actual);
                idx += 1;
            },
        );
    }
}
