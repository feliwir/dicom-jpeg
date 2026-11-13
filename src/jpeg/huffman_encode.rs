//! Huffman encoding: optimal table generation (T.81 K.2, as in libjpeg's
//! `jpeg_gen_optimal_table`) and an entropy-coded segment bit writer.

/// A Huffman table ready for encoding, with its DHT specification.
#[derive(Debug, Clone)]
pub(crate) struct EncodingTable {
    /// Code length counts for lengths 1..=16.
    pub(crate) bits: [u8; 16],
    /// Symbols in order of increasing code length.
    pub(crate) values: Vec<u8>,
    code: [u16; 256],
    size: [u8; 256],
}

impl EncodingTable {
    /// Build an optimal length-limited table for the given symbol frequencies.
    pub(crate) fn optimal(freq: &[u64; 256]) -> Self {
        let mut freq: Vec<u64> = freq.iter().copied().chain([1]).collect(); // [256] reserves all-ones
        let mut codesize = [0usize; 257];
        let mut others = [-1i32; 257];
        loop {
            // Smallest non-zero frequency (ties: larger symbol), then second smallest.
            let mut c1: i32 = -1;
            let mut v = u64::MAX;
            for (i, &f) in freq.iter().enumerate() {
                if f != 0 && f <= v {
                    v = f;
                    c1 = i as i32;
                }
            }
            let mut c2: i32 = -1;
            let mut v = u64::MAX;
            for (i, &f) in freq.iter().enumerate() {
                if f != 0 && f <= v && i as i32 != c1 {
                    v = f;
                    c2 = i as i32;
                }
            }
            if c2 < 0 {
                break;
            }
            let (mut a, mut b) = (c1 as usize, c2 as usize);
            freq[a] += freq[b];
            freq[b] = 0;
            codesize[a] += 1;
            while others[a] >= 0 {
                a = others[a] as usize;
                codesize[a] += 1;
            }
            others[a] = c2;
            codesize[b] += 1;
            while others[b] >= 0 {
                b = others[b] as usize;
                codesize[b] += 1;
            }
        }
        let mut bits = [0u32; 33];
        for &cs in &codesize {
            if cs > 0 {
                bits[cs.min(32)] += 1;
            }
        }
        // Limit code lengths to 16 bits (K.3).
        for i in (17..=32).rev() {
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
        // Remove the reserved code point from the longest length.
        let mut i = 16;
        while bits[i] == 0 {
            i -= 1;
        }
        bits[i] -= 1;
        let mut values = Vec::new();
        for len in 1..=32 {
            for (sym, &cs) in codesize.iter().enumerate().take(256) {
                if cs == len {
                    values.push(sym as u8);
                }
            }
        }
        let mut counts = [0u8; 16];
        for l in 1..=16 {
            counts[l - 1] = bits[l] as u8;
        }
        Self::from_spec(counts, values)
    }

    /// Build from a DHT specification (C.2).
    pub(crate) fn from_spec(bits: [u8; 16], values: Vec<u8>) -> Self {
        let mut code = [0u16; 256];
        let mut size = [0u8; 256];
        let mut c: u32 = 0;
        let mut k = 0;
        for len in 1..=16u8 {
            for _ in 0..bits[len as usize - 1] {
                let sym = values[k] as usize;
                code[sym] = c as u16;
                size[sym] = len;
                c += 1;
                k += 1;
            }
            c <<= 1;
        }
        EncodingTable {
            bits,
            values,
            code,
            size,
        }
    }

    /// Append the DHT payload for this table (class `tc`, destination `th`).
    pub(crate) fn write_dht(&self, tc: u8, th: u8, out: &mut Vec<u8>) {
        out.push((tc << 4) | th);
        out.extend_from_slice(&self.bits);
        out.extend_from_slice(&self.values);
    }

    #[inline]
    pub(crate) fn emit(&self, w: &mut BitWriter, symbol: u8) {
        let s = symbol as usize;
        debug_assert!(self.size[s] > 0, "symbol {symbol} has no code");
        w.write(self.code[s] as u32, self.size[s] as u32);
    }
}

/// Entropy-coded segment writer with 0xFF byte stuffing.
#[derive(Debug, Default)]
pub(crate) struct BitWriter {
    pub(crate) out: Vec<u8>,
    acc: u64,
    nbits: u32,
}

impl BitWriter {
    pub(crate) fn new(out: Vec<u8>) -> Self {
        BitWriter {
            out,
            acc: 0,
            nbits: 0,
        }
    }

    /// Write the low `n` (<= 24) bits of `value`.
    #[inline]
    pub(crate) fn write(&mut self, value: u32, n: u32) {
        if n == 0 {
            return;
        }
        self.acc = (self.acc << n) | (value as u64 & ((1 << n) - 1));
        self.nbits += n;
        while self.nbits >= 8 {
            self.nbits -= 8;
            let byte = (self.acc >> self.nbits) as u8;
            self.out.push(byte);
            if byte == 0xFF {
                self.out.push(0);
            }
        }
    }

    /// Pad the last byte with 1-bits (F.1.2.3).
    pub(crate) fn flush(&mut self) {
        if self.nbits > 0 {
            let pad = 8 - self.nbits;
            self.write((1 << pad) - 1, pad);
        }
    }

    /// Emit a restart marker after flushing.
    pub(crate) fn restart(&mut self, n: u8) {
        self.flush();
        self.out.extend_from_slice(&[0xFF, 0xD0 + (n & 7)]);
    }
}

/// Magnitude category (number of bits) of a value.
#[inline]
pub(crate) fn category(v: i32) -> u32 {
    32 - v.unsigned_abs().leading_zeros()
}

/// The additional bits for a value of category `s` (F.1.2.1).
#[inline]
pub(crate) fn magnitude_bits(v: i32, s: u32) -> u32 {
    if v < 0 {
        (v - 1) as u32 & ((1 << s) - 1)
    } else {
        v as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimal_table_is_prefix_free_and_limited() {
        let mut freq = [0u64; 256];
        for (i, f) in freq.iter_mut().enumerate().take(200) {
            *f = 1 + (i as u64 * 7919) % 1000 * (1 << (i % 20));
        }
        let t = EncodingTable::optimal(&freq);
        let total: usize = t.bits.iter().map(|&b| b as usize).sum();
        assert_eq!(total, 200);
        // Kraft inequality with the reserved all-ones code.
        let kraft: f64 = (1..=16)
            .map(|l| t.bits[l - 1] as f64 / 2f64.powi(l as i32))
            .sum();
        assert!(kraft < 1.0);
        // Round trip through the decoder's table builder.
        assert!(super::super::huffman::HuffmanTable::new(&t.bits, &t.values).is_ok());
    }

    #[test]
    fn single_symbol_gets_a_code() {
        let mut freq = [0u64; 256];
        freq[5] = 100;
        let t = EncodingTable::optimal(&freq);
        assert_eq!(t.values, vec![5]);
        assert_eq!(t.bits[0], 1);
    }
}
