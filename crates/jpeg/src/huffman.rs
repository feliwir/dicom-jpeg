//! Huffman tables (T.81 Annex C) with a lookup-table fast path for decoding.

use super::bitreader::BitReader;
use dicom_jpeg_common::{Error, Result};

use super::CODEC;

const LOOKAHEAD: u32 = 9;

/// A decoding table built from a DHT specification.
#[derive(Debug, Clone)]
pub(crate) struct HuffmanTable {
    /// `lookup[peek9] = (length << 8) | value`, length 0 = not in fast path.
    lookup: Box<[u16; 1 << LOOKAHEAD]>,
    /// Largest code of each length (index 1..=16), -1 if none; `maxcode[17]` sentinel.
    maxcode: [i32; 18],
    /// `values` index offset for each length.
    valoffset: [i32; 17],
    values: Vec<u8>,
    /// For AC tables, decoding a code and its magnitude bits at once when
    /// together they fit the lookahead (as stb_image): `fast_ac[peek9] =
    /// value << 16 | run << 8 | total length`, 0 = not available.
    fast_ac: Box<[i32; 1 << LOOKAHEAD]>,
    /// For lossless difference tables, the same with the category as the
    /// symbol: `fast_diff[peek9] = value << 8 | total length`, 0 = not
    /// available.
    fast_diff: Box<[i32; 1 << LOOKAHEAD]>,
}

impl HuffmanTable {
    /// Build from the 16 code-length counts and the symbol values.
    pub(crate) fn new(counts: &[u8; 16], values: &[u8]) -> Result<Self> {
        let total: usize = counts.iter().map(|&c| c as usize).sum();
        if total != values.len() || total > 256 {
            return Err(Error::invalid(CODEC, "inconsistent Huffman table"));
        }
        // C.2: generate code sizes and codes.
        let mut maxcode = [-1i32; 18];
        let mut valoffset = [0i32; 17];
        let mut lookup = Box::new([0u16; 1 << LOOKAHEAD]);
        let mut code: u32 = 0;
        let mut k = 0usize;
        for len in 1..=16u32 {
            let n = counts[len as usize - 1] as usize;
            if n > 0 {
                valoffset[len as usize] = k as i32 - code as i32;
                for _ in 0..n {
                    if code >= (1 << len) {
                        return Err(Error::invalid(CODEC, "Huffman code lengths overflow"));
                    }
                    if len <= LOOKAHEAD {
                        let shift = LOOKAHEAD - len;
                        let base = (code << shift) as usize;
                        for slot in &mut lookup[base..base + (1 << shift)] {
                            *slot = ((len as u16) << 8) | values[k] as u16;
                        }
                    }
                    code += 1;
                    k += 1;
                }
                maxcode[len as usize] = code as i32 - 1;
            }
            code <<= 1;
        }
        maxcode[17] = i32::MAX;
        let mut fast_ac = Box::new([0i32; 1 << LOOKAHEAD]);
        for (i, f) in fast_ac.iter_mut().enumerate() {
            let entry = lookup[i];
            let (len, rs) = ((entry >> 8) as u32, entry as u8);
            let (run, s) = ((rs >> 4) as i32, (rs & 15) as u32);
            if len == 0 || s == 0 || len + s > LOOKAHEAD {
                continue;
            }
            let m = ((i as u32) >> (LOOKAHEAD - len - s)) & ((1 << s) - 1);
            let v = extend(m, s);
            *f = (v << 16) | (run << 8) | (len + s) as i32;
        }
        let mut fast_diff = Box::new([0i32; 1 << LOOKAHEAD]);
        for (i, f) in fast_diff.iter_mut().enumerate() {
            let entry = lookup[i];
            let (len, t) = ((entry >> 8) as u32, (entry & 0xFF) as u32);
            // Category 16 is 32768 with no extra bits (H.1.2.2).
            let (extra, v) = match t {
                0 => (0, 0),
                1..=15 if len + t <= LOOKAHEAD => {
                    let m = ((i as u32) >> (LOOKAHEAD - len - t)) & ((1 << t) - 1);
                    (t, extend(m, t))
                }
                16 => (0, 32768),
                _ => continue,
            };
            if len > 0 {
                *f = (v << 8) | (len + extra) as i32;
            }
        }
        Ok(HuffmanTable {
            lookup,
            maxcode,
            valoffset,
            values: values.to_vec(),
            fast_ac,
            fast_diff,
        })
    }

    /// Decode an AC code with its magnitude bits from the fast table:
    /// `Some((run, value))`, or `None` if the slow path is needed.
    #[inline]
    pub(crate) fn decode_fast_ac(&self, reader: &mut BitReader<'_>) -> Option<(usize, i32)> {
        let f = self.fast_ac[(reader.peek16() >> (16 - LOOKAHEAD)) as usize];
        if f == 0 {
            return None;
        }
        reader.consume((f & 0xFF) as u32);
        Some((((f >> 8) & 0xFF) as usize, f >> 16))
    }

    /// Decode a lossless difference: its category, then its extra bits
    /// (H.1.2.2).
    #[inline(always)]
    pub(crate) fn decode_diff(&self, reader: &mut BitReader<'_>) -> Result<i32> {
        let f = self.fast_diff[(reader.peek16() >> (16 - LOOKAHEAD)) as usize];
        if f != 0 {
            reader.consume((f & 0xFF) as u32);
            return Ok(f >> 8);
        }
        match self.decode(reader)? as u32 {
            t @ 0..=15 => Ok(reader.receive_extend(t)),
            16 => Ok(32768),
            _ => Err(Error::invalid(CODEC, "lossless difference category > 16")),
        }
    }

    /// Decode one symbol.
    #[inline]
    pub(crate) fn decode(&self, reader: &mut BitReader<'_>) -> Result<u8> {
        let peek = reader.peek16();
        let entry = self.lookup[(peek >> (16 - LOOKAHEAD)) as usize];
        if entry != 0 {
            reader.consume((entry >> 8) as u32);
            return Ok(entry as u8);
        }
        // Slow path: F.2.2.3 DECODE.
        let mut len = LOOKAHEAD + 1;
        while len <= 16 {
            let code = (peek >> (16 - len)) as i32;
            if code <= self.maxcode[len as usize] {
                reader.consume(len);
                let idx = (self.valoffset[len as usize] + code) as usize;
                return self
                    .values
                    .get(idx)
                    .copied()
                    .ok_or_else(|| Error::invalid(CODEC, "invalid Huffman code"));
            }
            len += 1;
        }
        Err(Error::invalid(CODEC, "invalid Huffman code"))
    }
}

/// F.2.2.1 EXTEND: the signed value of `s` magnitude bits `v`.
#[inline]
pub(crate) fn extend(v: u32, s: u32) -> i32 {
    let v = v as i32;
    if v < (1 << (s - 1)) {
        v - (1 << s) + 1
    } else {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_standard_dc_table() {
        // Table K.3 (luminance DC)
        let counts = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
        let values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let t = HuffmanTable::new(&counts, &values).unwrap();
        // codes: 0:00 1:010 2:011 ... 11:111111110 (9 bits)
        let data = [0b0001_0011, 0b1111_1111, 0b0000_0000];
        let mut r = BitReader::new(&data, 0);
        assert_eq!(t.decode(&mut r).unwrap(), 0);
        assert_eq!(t.decode(&mut r).unwrap(), 1);
        assert_eq!(t.decode(&mut r).unwrap(), 2);
    }

    #[test]
    fn rejects_overfull_table() {
        let counts = [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        assert!(HuffmanTable::new(&counts, &[0, 1, 2]).is_err());
    }
}
