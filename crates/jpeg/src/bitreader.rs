//! Entropy-coded segment bit reader (T.81 F.2.2.5): removes stuffed zero
//! bytes and stops at markers.

/// Reads bits MSB-first from an entropy-coded segment.
///
/// When a marker (or the end of the data) is reached, zero bits are supplied
/// from then on, matching the behaviour of libjpeg for truncated data.
#[derive(Debug)]
pub(crate) struct BitReader<'a> {
    data: &'a [u8],
    /// Next byte to load into `bits`.
    pos: usize,
    bits: u64,
    /// Number of valid bits in `bits` (left aligned).
    nbits: u32,
    /// A marker (or end of data) was hit at `pos`.
    at_marker: bool,
    /// Number of zero bits supplied after the end of the segment.
    pub(crate) padding_bits: u64,
}

impl<'a> BitReader<'a> {
    pub(crate) fn new(data: &'a [u8], pos: usize) -> Self {
        BitReader {
            data,
            pos,
            bits: 0,
            nbits: 0,
            at_marker: false,
            padding_bits: 0,
        }
    }

    #[inline]
    fn fill(&mut self) {
        // Fast path: as many whole bytes as fit, taken from one 8-byte load,
        // when none of them is 0xFF (no stuffing, no marker).
        if !self.at_marker
            && let Some(chunk) = self.data.get(self.pos..self.pos + 8)
        {
            let w = u64::from_be_bytes(chunk.try_into().expect("8 bytes"));
            let n = (64 - self.nbits) / 8;
            // Flags bytes of `w` that are 0xFF (and possibly bytes before
            // one, which only sends us to the slow path).
            let v = !w;
            let ff = v.wrapping_sub(0x0101_0101_0101_0101) & !v & 0x8080_8080_8080_8080;
            let mask = if n >= 8 {
                u64::MAX
            } else {
                !(u64::MAX >> (8 * n))
            };
            if n > 0 && ff & mask == 0 {
                self.bits |= (w & mask) >> self.nbits;
                self.nbits += 8 * n;
                self.pos += n as usize;
                return;
            }
        }
        while self.nbits <= 56 {
            let byte = if self.at_marker {
                self.padding_bits += 8;
                0
            } else {
                match self.data.get(self.pos) {
                    None => {
                        self.at_marker = true;
                        self.padding_bits += 8;
                        0
                    }
                    Some(&0xFF) => match self.data.get(self.pos + 1) {
                        Some(&0x00) => {
                            self.pos += 2;
                            0xFF
                        }
                        // Fill bytes (FF FF ...) preceding a marker
                        Some(&0xFF) => {
                            self.pos += 1;
                            continue;
                        }
                        _ => {
                            self.at_marker = true;
                            self.padding_bits += 8;
                            0
                        }
                    },
                    Some(&b) => {
                        self.pos += 1;
                        b
                    }
                }
            };
            self.bits |= (byte as u64) << (56 - self.nbits);
            self.nbits += 8;
        }
    }

    /// Look at the next 16 bits without consuming them.
    #[inline]
    pub(crate) fn peek16(&mut self) -> u32 {
        if self.nbits < 16 {
            self.fill();
        }
        (self.bits >> 48) as u32
    }

    #[inline]
    pub(crate) fn consume(&mut self, n: u32) {
        debug_assert!(n <= self.nbits);
        self.bits <<= n;
        self.nbits -= n;
    }

    /// Read `n` (0..=16) bits.
    #[inline]
    pub(crate) fn bits(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        if self.nbits < n {
            self.fill();
        }
        let v = (self.bits >> (64 - n)) as u32;
        self.consume(n);
        v
    }

    /// Read `n` bits and sign-extend them as a DC difference / AC value (F.2.2.1 EXTEND).
    #[inline]
    pub(crate) fn receive_extend(&mut self, n: u32) -> i32 {
        if n == 0 {
            return 0;
        }
        let v = self.bits(n) as i32;
        if v < (1 << (n - 1)) {
            v - (1 << n) + 1
        } else {
            v
        }
    }

    /// Discard buffered bits and position at the next marker, returning it
    /// (without consuming it). `None` if the data ends first.
    fn seek_marker(&mut self) -> Option<u8> {
        self.bits = 0;
        self.nbits = 0;
        self.at_marker = false;
        let mut i = self.pos;
        while i + 1 < self.data.len() {
            if self.data[i] == 0xFF {
                let m = self.data[i + 1];
                if m != 0x00 && m != 0xFF {
                    self.pos = i;
                    return Some(m);
                }
            }
            i += 1;
        }
        self.pos = self.data.len();
        None
    }

    /// Handle a restart marker: byte-align, expect `RST(expected)`.
    /// Returns `false` if no restart marker was found at the expected place
    /// (the reader then supplies zeros until the end of the scan).
    pub(crate) fn restart(&mut self, expected: u8) -> bool {
        match self.seek_marker() {
            Some(m) if (0xD0..=0xD7).contains(&m) => {
                // Accept out-of-sequence numbers; libjpeg resynchronises similarly.
                let _ = expected;
                self.pos += 2;
                true
            }
            _ => {
                self.at_marker = true;
                false
            }
        }
    }

    /// Byte position just after the scan data (the start of the next marker).
    pub(crate) fn end_position(mut self) -> usize {
        self.seek_marker();
        self.pos
    }
}
