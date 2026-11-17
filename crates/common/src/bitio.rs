//! MSB-first bit reading from JPEG-family entropy-coded segments.
//!
//! T.81 (JPEG) and T.87 (JPEG-LS) both write bits MSB-first and protect
//! markers by byte stuffing, but stuff differently; [`Stuffing`] selects
//! the rule. The reader keeps up to 64 bits left-aligned in a register and
//! refills it 8 bytes at a time when they contain no 0xFF.

/// A byte-stuffing rule for [`BitReader`].
pub trait Stuffing {
    /// The most bits [`Stuffing::after_ff`] appends at once.
    const MAX_UNIT: u32;

    /// What a 0xFF byte followed by `next` (if any) means.
    fn after_ff(next: Option<u8>) -> AfterFf;

    /// Whether 0xFF followed by `next` starts a marker, when searching for
    /// one at the end of a segment.
    fn is_marker(next: u8) -> bool;
}

/// The meaning of a 0xFF byte in an entropy-coded segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AfterFf {
    /// The `width` low bits of `value` are data; skip `advance` bytes.
    Data {
        /// Data bits.
        value: u64,
        /// Number of data bits.
        width: u32,
        /// Bytes consumed.
        advance: usize,
    },
    /// A fill byte: skip it.
    Fill,
    /// A marker starts at the 0xFF.
    Marker,
    /// The 0xFF is 8 data bits, and a marker starts right after it.
    DataThenMarker,
}

/// T.81 F.1.2.3: 0xFF is followed by a stuffed 0x00; 0xFF 0xFF is fill.
#[derive(Debug, Clone, Copy)]
pub struct T81;

impl Stuffing for T81 {
    const MAX_UNIT: u32 = 8;

    #[inline]
    fn after_ff(next: Option<u8>) -> AfterFf {
        match next {
            Some(0x00) => AfterFf::Data {
                value: 0xFF,
                width: 8,
                advance: 2,
            },
            // Fill bytes (FF FF ...) preceding a marker.
            Some(0xFF) => AfterFf::Fill,
            _ => AfterFf::Marker,
        }
    }

    #[inline]
    fn is_marker(next: u8) -> bool {
        next != 0x00 && next != 0xFF
    }
}

/// T.87 A.1: the byte after 0xFF carries only 7 bits (its MSB is a stuffed
/// 0); 0xFF followed by a byte with the MSB set is a marker.
#[derive(Debug, Clone, Copy)]
pub struct T87;

impl Stuffing for T87 {
    const MAX_UNIT: u32 = 15;

    #[inline]
    fn after_ff(next: Option<u8>) -> AfterFf {
        match next {
            Some(b) if b & 0x80 == 0 => AfterFf::Data {
                value: 0xFF << 7 | b as u64,
                width: 15,
                advance: 2,
            },
            // The end of the data: the 0xFF is still data.
            None => AfterFf::Data {
                value: 0xFF,
                width: 8,
                advance: 1,
            },
            Some(_) => AfterFf::DataThenMarker,
        }
    }

    #[inline]
    fn is_marker(next: u8) -> bool {
        next & 0x80 != 0 && next != 0xFF
    }
}

/// Reads bits MSB-first from an entropy-coded segment.
///
/// When a marker (or the end of the data) is reached, zero bits are
/// supplied from then on (as libjpeg does for truncated data), and counted
/// in [`BitReader::padding_bits`].
#[derive(Debug)]
pub struct BitReader<'a, S> {
    data: &'a [u8],
    /// Next byte to load.
    pos: usize,
    /// Buffered bits, left-aligned.
    bits: u64,
    /// Number of valid bits in `bits`.
    nbits: u32,
    /// A marker (or the end of the data) is at `pos`.
    at_marker: bool,
    padding_bits: u64,
    stuffing: std::marker::PhantomData<S>,
}

impl<'a, S: Stuffing> BitReader<'a, S> {
    /// A reader of the segment starting at `data[pos]`.
    pub fn new(data: &'a [u8], pos: usize) -> Self {
        BitReader {
            data,
            pos,
            bits: 0,
            nbits: 0,
            at_marker: false,
            padding_bits: 0,
            stuffing: std::marker::PhantomData,
        }
    }

    /// Buffer at least `64 - S::MAX_UNIT + 1` bits.
    #[inline]
    pub fn fill(&mut self) {
        // Fast path: as many whole bytes as fit, from one 8-byte load, when
        // none of them is 0xFF (no stuffing, no marker).
        if !self.at_marker
            && let Some(chunk) = self.data.get(self.pos..self.pos + 8)
        {
            let w = u64::from_be_bytes(chunk.try_into().expect("8 bytes"));
            let n = (64 - self.nbits) / 8;
            // Flags the bytes of `w` that are 0xFF (and possibly bytes
            // before one, which only sends us to the slow path).
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
                if self.nbits > 64 - S::MAX_UNIT {
                    return;
                }
            }
        }
        self.fill_slow();
    }

    #[cold]
    fn fill_slow(&mut self) {
        while self.nbits <= 64 - S::MAX_UNIT {
            let (value, width) = if self.at_marker {
                self.padding_bits += 8;
                (0, 8)
            } else {
                match self.data.get(self.pos) {
                    None => {
                        self.at_marker = true;
                        self.padding_bits += 8;
                        (0, 8)
                    }
                    Some(&0xFF) => match S::after_ff(self.data.get(self.pos + 1).copied()) {
                        AfterFf::Data {
                            value,
                            width,
                            advance,
                        } => {
                            self.pos += advance;
                            (value, width)
                        }
                        AfterFf::Fill => {
                            self.pos += 1;
                            continue;
                        }
                        AfterFf::Marker => {
                            self.at_marker = true;
                            self.padding_bits += 8;
                            (0, 8)
                        }
                        AfterFf::DataThenMarker => {
                            self.at_marker = true;
                            (0xFF, 8)
                        }
                    },
                    Some(&b) => {
                        self.pos += 1;
                        (b as u64, 8)
                    }
                }
            };
            self.bits |= value << (64 - width - self.nbits);
            self.nbits += width;
        }
    }

    /// The buffered bits, left-aligned; [`BitReader::available`] of them
    /// are valid, the rest are zero.
    #[inline]
    pub fn buffer(&self) -> u64 {
        self.bits
    }

    /// Number of buffered bits.
    #[inline]
    pub fn available(&self) -> u32 {
        self.nbits
    }

    /// Look at the next 16 bits without consuming them.
    #[inline]
    pub fn peek16(&mut self) -> u32 {
        if self.nbits < 16 {
            self.fill();
        }
        (self.bits >> 48) as u32
    }

    /// Drop `n` buffered bits (`n <= available()`, `n < 64`).
    #[inline]
    pub fn consume(&mut self, n: u32) {
        debug_assert!(n <= self.nbits && n < 64);
        self.bits <<= n;
        self.nbits -= n;
    }

    /// Read `n` (0..=32) bits.
    #[inline]
    pub fn bits(&mut self, n: u32) -> u32 {
        debug_assert!(n <= 32);
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

    /// Read `n` (0..=16) bits as a T.81 DC difference or AC value
    /// (F.2.2.1 EXTEND).
    #[inline]
    pub fn receive_extend(&mut self, n: u32) -> i32 {
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

    /// Zero bits supplied after the end of the segment so far.
    #[inline]
    pub fn padding_bits(&self) -> u64 {
        self.padding_bits
    }

    /// Discard buffered bits and position at the next marker, returning it
    /// (without consuming it). `None` if the data ends first.
    fn seek_marker(&mut self) -> Option<u8> {
        self.bits = 0;
        self.nbits = 0;
        self.at_marker = false;
        let mut i = self.pos;
        while i + 1 < self.data.len() {
            if self.data[i] == 0xFF && S::is_marker(self.data[i + 1]) {
                self.pos = i;
                return Some(self.data[i + 1]);
            }
            i += 1;
        }
        self.pos = self.data.len();
        None
    }

    /// Byte-align and consume the RSTm marker that should follow. Returns
    /// `false` (and supplies zeros from then on) if the next marker is not
    /// a restart marker.
    pub fn restart(&mut self) -> bool {
        match self.seek_marker() {
            Some(0xD0..=0xD7) => {
                self.pos += 2;
                true
            }
            _ => {
                self.at_marker = true;
                false
            }
        }
    }

    /// Byte position of the marker that follows the segment.
    pub fn end_position(mut self) -> usize {
        self.seek_marker();
        self.pos
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t81_unstuffs_and_pads() {
        let data = [0x12, 0xFF, 0x00, 0x34, 0xFF, 0xFF, 0xFF, 0xD9];
        let mut r = BitReader::<T81>::new(&data, 0);
        assert_eq!(r.bits(8), 0x12);
        assert_eq!(r.bits(16), 0xFF34);
        assert_eq!(r.bits(8), 0);
        assert!(r.padding_bits() > 0);
        assert_eq!(r.end_position(), 6);
    }

    #[test]
    fn t87_drops_stuffed_bit() {
        // FF 7F: 8 + 7 one bits; FF 80 is a marker after 8 data bits.
        let data = [0xFF, 0x7F, 0x00, 0xFF, 0x80];
        let mut r = BitReader::<T87>::new(&data, 0);
        assert_eq!(r.bits(15), 0x7FFF);
        assert_eq!(r.bits(8), 0);
        assert_eq!(r.bits(8), 0xFF);
        assert_eq!(r.bits(8), 0);
        assert_eq!(r.end_position(), 3);
    }

    #[test]
    fn fast_and_slow_paths_agree() {
        // A long stream with 0xFF 0x00 pairs at varying places.
        let mut data = Vec::new();
        let mut expected = Vec::new();
        for i in 0..400u32 {
            let b = (i.wrapping_mul(97) % 251) as u8;
            if i % 13 == 0 {
                data.extend_from_slice(&[0xFF, 0x00]);
                expected.push(0xFF);
            } else {
                data.push(b);
                expected.push(b);
            }
        }
        for step in 1..=32 {
            let mut r = BitReader::<T81>::new(&data, 0);
            let mut bits: Vec<u8> = Vec::new();
            let total = expected.len() * 8;
            let mut read = 0;
            while read + step <= total {
                let v = r.bits(step as u32);
                for k in (0..step).rev() {
                    bits.push((v >> k) as u8 & 1);
                }
                read += step;
            }
            for (i, &b) in bits.iter().enumerate() {
                assert_eq!(b, expected[i / 8] >> (7 - i % 8) & 1, "step {step} bit {i}");
            }
        }
    }
}
