//! Tag trees (T.800 B.10.2).

#[derive(Debug, Clone, Copy)]
struct Node {
    parent: Option<usize>,
    value: i32,
    low: i32,
}

/// A tag tree over a `w` x `h` grid of leaves.
#[derive(Debug, Clone)]
pub(crate) struct TagTree {
    nodes: Vec<Node>,
    leaves_w: usize,
}

const UNKNOWN: i32 = i32::MAX;

impl TagTree {
    pub(crate) fn new(w: usize, h: usize) -> Self {
        let mut nodes = Vec::new();
        let mut level_dims = vec![(w, h)];
        let (mut lw, mut lh) = (w, h);
        while lw > 1 || lh > 1 {
            lw = lw.div_ceil(2);
            lh = lh.div_ceil(2);
            level_dims.push((lw, lh));
        }
        // Level offsets in `nodes`.
        let mut offsets = Vec::new();
        let mut total = 0;
        for &(a, b) in &level_dims {
            offsets.push(total);
            total += a * b;
        }
        nodes.resize(
            total,
            Node {
                parent: None,
                value: UNKNOWN,
                low: 0,
            },
        );
        for (l, &(a, b)) in level_dims.iter().enumerate() {
            if l + 1 == level_dims.len() {
                break;
            }
            let (pa, _) = level_dims[l + 1];
            for y in 0..b {
                for x in 0..a {
                    nodes[offsets[l] + y * a + x].parent =
                        Some(offsets[l + 1] + (y / 2) * pa + x / 2);
                }
            }
        }
        TagTree { nodes, leaves_w: w }
    }

    /// Decode whether the value of leaf (x, y) is below `threshold`, reading
    /// bits from `bit` as needed.
    pub(crate) fn decode(
        &mut self,
        x: usize,
        y: usize,
        threshold: i32,
        bit: &mut dyn FnMut() -> Option<bool>,
    ) -> Option<bool> {
        let leaf = y * self.leaves_w + x;
        let mut path = Vec::with_capacity(16);
        let mut n = Some(leaf);
        while let Some(i) = n {
            path.push(i);
            n = self.nodes[i].parent;
        }
        let mut low = 0;
        for &i in path.iter().rev() {
            let node = &mut self.nodes[i];
            if low > node.low {
                node.low = low;
            } else {
                low = node.low;
            }
            while low < threshold && low < node.value {
                if bit()? {
                    node.value = low;
                } else {
                    low += 1;
                }
            }
            node.low = low;
        }
        Some(self.nodes[leaf].value < threshold)
    }

    /// Fully decode the value of leaf (x, y) (used for the number of zero bit-planes).
    pub(crate) fn decode_value(
        &mut self,
        x: usize,
        y: usize,
        bit: &mut dyn FnMut() -> Option<bool>,
    ) -> Option<i32> {
        let mut t = 1;
        loop {
            if self.decode(x, y, t, bit)? {
                return Some(t - 1);
            }
            t += 1;
            if t > 1 << 20 {
                return None;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Example from T.800 B.10.2 / Figure B.12 style: encode values by
    /// hand with the standard's procedure and decode them back.
    #[test]
    fn decodes_known_sequence() {
        // 1x1 tree with value 2: bits "0 0 1" for threshold 3.
        let mut t = TagTree::new(1, 1);
        let bits = [false, false, true];
        let mut i = 0;
        let mut next = || {
            let b = bits.get(i).copied();
            i += 1;
            b
        };
        assert_eq!(t.decode_value(0, 0, &mut next), Some(2));
    }
}
