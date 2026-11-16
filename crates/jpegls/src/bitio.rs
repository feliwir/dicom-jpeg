//! JPEG-LS bit stream I/O (T.87 A.1): after a 0xFF byte the next byte carries
//! only 7 bits (its MSB is a stuffed 0); 0xFF followed by a byte with MSB set
//! is a marker.

use super::CODEC;
use dicom_jpeg_common::{Error, Result};

#[derive(Debug)]
pub(crate) struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    acc: u64,
    nbits: u32,
    prev_ff: bool,
    at_marker: bool,
    /// Zero bits supplied past the end of the scan data.
    padding: u64,
}

impl<'a> BitReader<'a> {
    pub(crate) fn new(data: &'a [u8], pos: usize) -> Self {
        BitReader {
            data,
            pos,
            acc: 0,
            nbits: 0,
            prev_ff: false,
            at_marker: false,
            padding: 0,
        }
    }

    #[inline]
    fn load_byte(&mut self) {
        let (value, width) = if self.at_marker {
            (0, 8)
        } else {
            match self.data.get(self.pos) {
                Some(&b) if self.prev_ff && b & 0x80 != 0 => {
                    self.at_marker = true;
                    // The FF already consumed belongs to the marker; step back.
                    self.pos -= 1;
                    (0, 8)
                }
                Some(&b) => {
                    self.pos += 1;
                    let width = if self.prev_ff { 7 } else { 8 };
                    self.prev_ff = b == 0xFF;
                    (b as u64 & ((1 << width) - 1), width)
                }
                None => {
                    self.at_marker = true;
                    (0, 8)
                }
            }
        };
        if self.at_marker {
            self.padding += 8;
        }
        self.acc = (self.acc << width) | value;
        self.nbits += width;
    }

    #[inline]
    fn ensure(&mut self, n: u32) -> Result<()> {
        while self.nbits < n {
            self.load_byte();
        }
        if self.padding > 1 << 16 {
            return Err(Error::Truncated { codec: CODEC });
        }
        Ok(())
    }

    #[inline]
    pub(crate) fn bit(&mut self) -> Result<bool> {
        self.ensure(1)?;
        self.nbits -= 1;
        Ok((self.acc >> self.nbits) & 1 == 1)
    }

    /// Read `n` (<= 32) bits.
    #[inline]
    pub(crate) fn bits(&mut self, n: u32) -> Result<i32> {
        if n == 0 {
            return Ok(0);
        }
        self.ensure(n)?;
        self.nbits -= n;
        Ok(((self.acc >> self.nbits) & ((1u64 << n) - 1)) as i32)
    }

    /// Count zero bits up to the next 1 bit (consumed), at most `max`.
    #[inline]
    pub(crate) fn unary(&mut self, max: u32) -> Result<u32> {
        let mut n = 0;
        while !self.bit()? {
            n += 1;
            if n > max {
                return Err(Error::invalid(CODEC, "unary code too long"));
            }
        }
        Ok(n)
    }

    /// Decode a limited-length Golomb code (A.5.3).
    #[inline]
    pub(crate) fn golomb(&mut self, k: u32, limit: u32, qbpp: u32) -> Result<i32> {
        let high = self.unary(limit)?;
        if high >= limit - qbpp - 1 {
            return Ok(self.bits(qbpp)? + 1);
        }
        if k == 0 {
            return Ok(high as i32);
        }
        Ok(((high as i32) << k) + self.bits(k)?)
    }

    /// Skip to the marker following the current position (byte aligned) and
    /// return it without consuming it.
    fn seek_marker(&mut self) -> Option<u8> {
        self.acc = 0;
        self.nbits = 0;
        self.prev_ff = false;
        self.at_marker = false;
        let mut i = self.pos;
        while i + 1 < self.data.len() {
            if self.data[i] == 0xFF && self.data[i + 1] & 0x80 != 0 && self.data[i + 1] != 0xFF {
                self.pos = i;
                return Some(self.data[i + 1]);
            }
            i += 1;
        }
        self.pos = self.data.len();
        None
    }

    /// Consume an RSTm marker at the next byte boundary.
    pub(crate) fn restart(&mut self) -> Result<()> {
        match self.seek_marker() {
            Some(0xD0..=0xD7) => {
                self.pos += 2;
                Ok(())
            }
            _ => Err(Error::invalid(CODEC, "expected restart marker")),
        }
    }

    /// Byte position of the marker that follows the scan.
    pub(crate) fn end_position(mut self) -> usize {
        self.seek_marker();
        self.pos
    }
}

#[cfg(feature = "encode")]
#[derive(Debug, Default)]
pub(crate) struct BitWriter {
    pub(crate) out: Vec<u8>,
    acc: u64,
    nbits: u32,
    prev_ff: bool,
}

#[cfg(feature = "encode")]
impl BitWriter {
    pub(crate) fn new(out: Vec<u8>) -> Self {
        BitWriter {
            out,
            ..Default::default()
        }
    }

    #[inline]
    fn drain(&mut self) {
        loop {
            let width = if self.prev_ff { 7 } else { 8 };
            if self.nbits < width {
                break;
            }
            self.nbits -= width;
            let byte = ((self.acc >> self.nbits) & ((1 << width) - 1)) as u8;
            self.out.push(byte);
            self.prev_ff = byte == 0xFF;
        }
    }

    /// Write the low `n` (<= 32) bits of `value`.
    #[inline]
    pub(crate) fn write(&mut self, value: u32, n: u32) {
        if n == 0 {
            return;
        }
        self.acc = (self.acc << n) | (value as u64 & ((1u64 << n) - 1));
        self.nbits += n;
        self.drain();
    }

    /// `n` zero bits followed by a one bit.
    #[inline]
    pub(crate) fn unary(&mut self, mut n: u32) {
        while n >= 24 {
            self.write(0, 24);
            n -= 24;
        }
        self.write(1, n + 1);
    }

    /// Limited-length Golomb code (A.5.3).
    #[inline]
    pub(crate) fn golomb(&mut self, value: i32, k: u32, limit: u32, qbpp: u32) {
        let value = value as u32;
        let high = value >> k;
        if high < limit - qbpp - 1 {
            self.unary(high);
            self.write(value & ((1 << k) - 1), k);
        } else {
            self.unary(limit - qbpp - 1);
            self.write((value - 1) & ((1 << qbpp) - 1), qbpp);
        }
    }

    /// Pad to a byte boundary with zero bits.
    pub(crate) fn flush(&mut self) {
        let width = if self.prev_ff { 7 } else { 8 };
        if self.nbits > 0 {
            let pad = width - self.nbits;
            self.write(0, pad);
        }
        if self.prev_ff {
            // A byte with MSB 0 must follow 0xFF so it is not read as a marker.
            self.out.push(0);
            self.prev_ff = false;
        }
    }
}

#[cfg(all(test, feature = "encode"))]
mod tests {
    use super::*;

    #[cfg(feature = "encode")]
    #[test]
    fn stuffing_round_trip() {
        let mut w = BitWriter::new(Vec::new());
        // Lots of 1 bits produce FF bytes and stuffing.
        let values: Vec<(u32, u32)> = (0..500)
            .map(|i| ((i * 7919) as u32 & 0xFFFF, 1 + (i % 16) as u32))
            .collect();
        for &(v, n) in &values {
            w.write(v | 0xFFF0, n);
        }
        w.flush();
        let mut data = w.out;
        assert!(data.windows(2).all(|p| !(p[0] == 0xFF && p[1] & 0x80 != 0)));
        data.extend_from_slice(&[0xFF, 0xD9]);
        let mut r = BitReader::new(&data, 0);
        for &(v, n) in &values {
            let expected = (v | 0xFFF0) & ((1 << n) - 1);
            assert_eq!(r.bits(n).unwrap() as u32, expected);
        }
        assert_eq!(r.end_position(), data.len() - 2);
    }

    #[cfg(feature = "encode")]
    #[test]
    fn golomb_round_trip() {
        let mut w = BitWriter::new(Vec::new());
        let cases: Vec<(i32, u32)> = (0..300)
            .map(|i| ((i * 37) % 4000, (i % 7) as u32))
            .collect();
        for &(v, k) in &cases {
            w.golomb(v, k, 32, 12);
        }
        w.flush();
        let mut r = BitReader::new(&w.out, 0);
        for &(v, k) in &cases {
            assert_eq!(r.golomb(k, 32, 12).unwrap(), v, "value {v} k {k}");
        }
    }
}
