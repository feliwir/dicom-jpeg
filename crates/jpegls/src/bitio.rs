//! JPEG-LS bit stream I/O (T.87 A.1): after a 0xFF byte the next byte carries
//! only 7 bits (its MSB is a stuffed 0); 0xFF followed by a byte with MSB set
//! is a marker.

use super::CODEC;
use dicom_jpeg_common::bitio::{self, T87};
use dicom_jpeg_common::{Error, Result};

/// Reads JPEG-LS scan data; past the end of the data it supplies zero bits
/// until 64 Kibit of them make the stream count as truncated.
#[derive(Debug)]
pub(crate) struct BitReader<'a> {
    r: bitio::BitReader<'a, T87>,
}

impl<'a> BitReader<'a> {
    pub(crate) fn new(data: &'a [u8], pos: usize) -> Self {
        BitReader {
            r: bitio::BitReader::new(data, pos),
        }
    }

    #[inline]
    fn check(&self) -> Result<()> {
        if self.r.padding_bits() > 1 << 16 {
            return Err(Error::Truncated { codec: CODEC });
        }
        Ok(())
    }

    #[inline]
    pub(crate) fn bit(&mut self) -> Result<bool> {
        let b = self.r.bits(1) == 1;
        self.check()?;
        Ok(b)
    }

    /// Read `n` (<= 32) bits.
    #[inline]
    pub(crate) fn bits(&mut self, n: u32) -> Result<i32> {
        let v = self.r.bits(n) as i32;
        self.check()?;
        Ok(v)
    }

    /// Count zero bits up to the next 1 bit (consumed), at most `max`.
    #[inline]
    pub(crate) fn unary(&mut self, max: u32) -> Result<u32> {
        let mut n = 0;
        loop {
            if self.r.available() < 32 {
                self.r.fill();
                self.check()?;
            }
            // Bits past `available()` are zero, so a 1 found is a real one.
            let zeros = self.r.buffer().leading_zeros();
            let avail = self.r.available();
            if zeros < avail {
                n += zeros;
                if n > max {
                    return Err(Error::invalid(CODEC, "unary code too long"));
                }
                self.r.consume(zeros);
                self.r.consume(1);
                return Ok(n);
            }
            n += avail;
            if n > max {
                return Err(Error::invalid(CODEC, "unary code too long"));
            }
            self.r.consume(avail / 2);
            self.r.consume(avail - avail / 2);
        }
    }

    /// Decode a limited-length Golomb code (A.5.3).
    #[inline]
    pub(crate) fn golomb(&mut self, k: u32, limit: u32, qbpp: u32) -> Result<i32> {
        // Fast path: the unary prefix, its terminating 1 and the k low bits
        // are all buffered, and the code is not an escape.
        if self.r.available() < 32 {
            self.r.fill();
            self.check()?;
        }
        let (buf, avail) = (self.r.buffer(), self.r.available());
        let zeros = buf.leading_zeros();
        let len = zeros + 1 + k;
        if zeros < limit - qbpp - 1 && len < avail.min(64) {
            let low = if k == 0 {
                0
            } else {
                ((buf << (zeros + 1)) >> (64 - k)) as i32
            };
            self.r.consume(len);
            return Ok(((zeros as i32) << k) + low);
        }
        let high = self.unary(limit)?;
        if high >= limit - qbpp - 1 {
            return Ok(self.bits(qbpp)? + 1);
        }
        if k == 0 {
            return Ok(high as i32);
        }
        Ok(((high as i32) << k) + self.bits(k)?)
    }

    /// Consume an RSTm marker at the next byte boundary.
    pub(crate) fn restart(&mut self) -> Result<()> {
        if self.r.restart() {
            Ok(())
        } else {
            Err(Error::invalid(CODEC, "expected restart marker"))
        }
    }

    /// Byte position of the marker that follows the scan.
    pub(crate) fn end_position(self) -> usize {
        self.r.end_position()
    }
}

/// Writes JPEG-LS scan data.
#[cfg(feature = "encode")]
#[derive(Debug)]
pub(crate) struct BitWriter {
    w: bitio::BitWriter<T87>,
}

#[cfg(feature = "encode")]
impl BitWriter {
    pub(crate) fn new(out: Vec<u8>) -> Self {
        BitWriter {
            w: bitio::BitWriter::new(out),
        }
    }

    /// The output written so far (complete bytes only).
    pub(crate) fn out(&mut self) -> &mut Vec<u8> {
        &mut self.w.out
    }

    pub(crate) fn into_out(self) -> Vec<u8> {
        self.w.out
    }

    /// Write the low `n` (<= 32) bits of `value`.
    #[inline]
    pub(crate) fn write(&mut self, value: u32, n: u32) {
        self.w.write(value, n);
    }

    /// `n` zero bits followed by a one bit.
    #[inline]
    pub(crate) fn unary(&mut self, mut n: u32) {
        while n >= 32 {
            self.write(0, 32);
            n -= 32;
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
        self.w.flush();
    }

    /// Flush and write the restart marker RSTn (n modulo 8).
    pub(crate) fn restart(&mut self, n: u8) {
        self.w.restart(n);
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
        let mut data = w.into_out();
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
        let data = w.into_out();
        let mut r = BitReader::new(&data, 0);
        for &(v, k) in &cases {
            assert_eq!(r.golomb(k, 32, 12).unwrap(), v, "value {v} k {k}");
        }
    }
}
