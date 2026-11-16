//! Canonical prefix (Huffman) codes, as used by JPEG (T.81 Annex C / K) and
//! JPEG XL (RFC 7932 code layout).

/// Length-limited Huffman code lengths for `freq` (0 = unused symbol).
///
/// The code is built with Huffman's algorithm and then limited to
/// `max_len` bits with the count adjustment of T.81 K.3; the lengths are
/// assigned to the symbols in order of their Huffman depth (ties by symbol
/// index), as T.81 K.2 does. A single used symbol gets length 1.
pub fn code_lengths(freq: &[u64], max_len: u32) -> Vec<u8> {
    let used: Vec<usize> = (0..freq.len()).filter(|&i| freq[i] > 0).collect();
    let mut lengths = vec![0u8; freq.len()];
    match used.len() {
        0 => return lengths,
        1 => {
            lengths[used[0]] = 1;
            return lengths;
        }
        _ => {}
    }
    assert!(
        used.len() as u64 <= 1u64 << max_len,
        "too many symbols for the code length limit"
    );
    // Huffman tree over the used symbols: leaves 0..n, internal nodes n.. .
    let n = used.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| (freq[used[i]], i));
    let mut weight: Vec<u64> = order.iter().map(|&i| freq[used[i]]).collect();
    weight.resize(2 * n - 1, 0);
    let mut parent = vec![usize::MAX; 2 * n - 1];
    let (mut li, mut ii) = (0usize, n);
    for next in n..2 * n - 1 {
        let mut pick = || {
            if li < n && (ii >= next || weight[li] <= weight[ii]) {
                li += 1;
                li - 1
            } else {
                ii += 1;
                ii - 1
            }
        };
        let (a, b) = (pick(), pick());
        weight[next] = weight[a] + weight[b];
        parent[a] = next;
        parent[b] = next;
    }
    let mut depth = vec![0u32; 2 * n - 1];
    for i in (0..2 * n - 2).rev() {
        depth[i] = depth[parent[i]] + 1;
    }
    // Symbols by (depth, symbol index); leaf k is used[order[k]].
    let mut syms: Vec<(u32, usize)> = (0..n).map(|k| (depth[k], used[order[k]])).collect();
    syms.sort_unstable();
    // Counts per length, limited to max_len (T.81 K.3).
    let longest = syms.last().map_or(0, |s| s.0) as usize;
    let mut bits = vec![0u64; longest.max(max_len as usize) + 1];
    for &(d, _) in &syms {
        bits[d as usize] += 1;
    }
    let max = max_len as usize;
    for i in (max + 1..bits.len()).rev() {
        while bits[i] > 0 {
            let mut j = i - 2;
            while bits[j] == 0 {
                j -= 1;
            }
            bits[i] -= 2;
            bits[i - 1] += 1;
            bits[j + 1] += 2;
            bits[j] -= 1;
        }
    }
    let mut it = syms.iter();
    for (len, &count) in bits.iter().enumerate().take(max + 1) {
        for _ in 0..count {
            let &(_, s) = it.next().expect("counts match the symbols");
            lengths[s] = len as u8;
        }
    }
    lengths
}

/// Canonical codes for the lengths: symbols ordered by (length, symbol)
/// receive consecutive codes; the result holds each code MSB first.
pub fn canonical_codes(lengths: &[u8]) -> Vec<u32> {
    let mut codes = vec![0u32; lengths.len()];
    let max = lengths.iter().copied().max().unwrap_or(0);
    let mut code = 0u32;
    for len in 1..=max {
        for (i, &l) in lengths.iter().enumerate() {
            if l == len {
                codes[i] = code;
                code += 1;
            }
        }
        code <<= 1;
    }
    codes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kraft(lengths: &[u8]) -> f64 {
        lengths
            .iter()
            .filter(|&&l| l > 0)
            .map(|&l| 0.5f64.powi(l as i32))
            .sum()
    }

    #[test]
    fn complete_and_limited() {
        // Fibonacci frequencies produce a very deep unlimited tree.
        let mut freq = vec![1u64, 1];
        for i in 2..30 {
            freq.push(freq[i - 1] + freq[i - 2]);
        }
        for limit in [8, 12, 15, 16] {
            let l = code_lengths(&freq, limit);
            assert!(l.iter().all(|&x| x as u32 <= limit && x > 0));
            assert!((kraft(&l) - 1.0).abs() < 1e-12, "limit {limit}");
        }
    }

    #[test]
    fn canonical_is_prefix_free() {
        let freq = [5u64, 0, 9, 12, 13, 16, 45];
        let l = code_lengths(&freq, 15);
        assert_eq!(l[1], 0);
        let c = canonical_codes(&l);
        for i in 0..freq.len() {
            for j in 0..freq.len() {
                if i == j || l[i] == 0 || l[j] == 0 || l[i] > l[j] {
                    continue;
                }
                assert_ne!(c[j] >> (l[j] - l[i]), c[i], "{i} is a prefix of {j}");
            }
        }
    }

    #[test]
    fn single_symbol() {
        assert_eq!(code_lengths(&[0, 7, 0], 15), vec![0, 1, 0]);
    }
}
