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
    ///
    /// As in T.81 K.2, a reserved symbol with frequency 1 takes the all-ones
    /// code of the longest length, which is then removed from the table.
    pub(crate) fn optimal(freq: &[u64; 256]) -> Self {
        let mut f: Vec<u64> = freq.to_vec();
        f.push(1); // reserved symbol 256
        let lengths = dicom_jpeg_common::huffman::code_lengths(&f, 16);
        let mut bits = [0u8; 16];
        let mut values = Vec::new();
        for len in 1..=16u8 {
            for (sym, &l) in lengths.iter().enumerate().take(256) {
                if l == len {
                    bits[len as usize - 1] += 1;
                    values.push(sym as u8);
                }
            }
        }
        Self::from_spec(bits, values)
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

/// Entropy-coded segment writer (T.81 F.1.2.3 byte stuffing, 1-bit padding).
pub(crate) type BitWriter = dicom_jpeg_common::bitio::BitWriter<dicom_jpeg_common::bitio::T81>;

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
