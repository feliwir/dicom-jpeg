//! Bit reader for JPEG XL codestreams (ISO/IEC 18181-1 clause 4, B.2):
//! bits are read least significant first, fields use the `U32`, `U64`,
//! `F16` and `Enum` distributions.

use super::CODEC;
use crate::error::{Error, Result};

pub(crate) fn invalid(msg: &'static str) -> Error {
    Error::invalid(CODEC, msg)
}

pub(crate) fn truncated() -> Error {
    Error::Truncated { codec: CODEC }
}

/// One `U32` distribution entry: `offset + u(bits)`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct U32D(pub u32, pub u32);

/// Shorthand for a fixed value.
pub(crate) const fn val(v: u32) -> U32D {
    U32D(v, 0)
}

/// Shorthand for `offset + u(bits)`.
pub(crate) const fn bits(n: u32, offset: u32) -> U32D {
    U32D(offset, n)
}

#[derive(Clone)]
pub(crate) struct BitReader<'a> {
    data: &'a [u8],
    /// Next byte of `data` to load into `buf`.
    pos: usize,
    buf: u64,
    nbits: u32,
    /// Zero bits supplied past the end of the data.
    overrun: u64,
}

impl<'a> BitReader<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        BitReader {
            data,
            pos: 0,
            buf: 0,
            nbits: 0,
            overrun: 0,
        }
    }

    #[inline]
    fn refill(&mut self) {
        while self.nbits <= 56 {
            if self.pos < self.data.len() {
                self.buf |= (self.data[self.pos] as u64) << self.nbits;
                self.pos += 1;
            } else {
                // Past the end: supply zero bits, remembered as overrun.
                self.overrun += 8;
            }
            self.nbits += 8;
        }
    }

    /// The next `n` (<= 32) bits without consuming them.
    #[inline]
    pub(crate) fn peek(&mut self, n: u32) -> u32 {
        debug_assert!(n <= 32);
        if self.nbits < n {
            self.refill();
        }
        (self.buf & ((1u64 << n) - 1)) as u32
    }

    /// Consume `n` bits after a `peek`.
    #[inline]
    pub(crate) fn consume(&mut self, n: u32) -> Result<()> {
        if self.nbits < n {
            self.refill();
        }
        self.buf >>= n;
        self.nbits -= n;
        self.check()
    }

    #[inline]
    pub(crate) fn read(&mut self, n: u32) -> Result<u32> {
        if n == 0 {
            return Ok(0);
        }
        let v = self.peek(n);
        self.consume(n)?;
        Ok(v)
    }

    /// Fails once more bits were consumed than the data holds.
    #[inline]
    pub(crate) fn check(&self) -> Result<()> {
        if self.overrun > 0 && self.bits_read() > self.data.len() as u64 * 8 {
            Err(truncated())
        } else {
            Ok(())
        }
    }

    pub(crate) fn read_bool(&mut self) -> Result<bool> {
        Ok(self.read(1)? == 1)
    }

    /// `U32(d0, d1, d2, d3)`.
    pub(crate) fn u32(&mut self, d: [U32D; 4]) -> Result<u32> {
        let U32D(offset, n) = d[self.read(2)? as usize];
        let v = self.read(n)?;
        offset
            .checked_add(v)
            .ok_or_else(|| invalid("U32 value overflows"))
    }

    /// `U64()`.
    pub(crate) fn u64(&mut self) -> Result<u64> {
        Ok(match self.read(2)? {
            0 => 0,
            1 => 1 + self.read(4)? as u64,
            2 => 17 + self.read(8)? as u64,
            _ => {
                let mut v = self.read(12)? as u64;
                let mut shift = 12;
                while self.read_bool()? {
                    if shift == 60 {
                        v |= (self.read(4)? as u64) << 60;
                        break;
                    }
                    v |= (self.read(8)? as u64) << shift;
                    shift += 8;
                }
                v
            }
        })
    }

    /// `F16()`: a finite IEEE half-precision value.
    pub(crate) fn f16(&mut self) -> Result<f32> {
        let b = self.read(16)?;
        let sign = if b >> 15 == 1 { -1.0 } else { 1.0 };
        let exp = (b >> 10) & 0x1F;
        let mant = (b & 0x3FF) as f32;
        if exp == 31 {
            return Err(invalid("F16 value is not finite"));
        }
        Ok(if exp == 0 {
            sign * mant * 2f32.powi(-24)
        } else {
            sign * (1.0 + mant / 1024.0) * 2f32.powi(exp as i32 - 15)
        })
    }

    /// `Enum()`.
    pub(crate) fn enumeration(&mut self) -> Result<u32> {
        let v = self.u32([val(0), val(1), bits(4, 2), bits(6, 18)])?;
        if v > 63 {
            return Err(invalid("enum value out of range"));
        }
        Ok(v)
    }

    /// Number of bits consumed so far (including zero bits past the end).
    pub(crate) fn bits_read(&self) -> u64 {
        self.pos as u64 * 8 + self.overrun - self.nbits as u64
    }

    /// `ZeroPadToByte()`.
    pub(crate) fn zero_pad_to_byte(&mut self) -> Result<()> {
        let n = (8 - (self.bits_read() % 8) as u32) % 8;
        if self.read(n)? != 0 {
            return Err(invalid("non-zero padding bits"));
        }
        Ok(())
    }

    /// Skip `n` bits.
    pub(crate) fn skip(&mut self, mut n: u64) -> Result<()> {
        while n > 0 {
            let k = n.min(32) as u32;
            self.read(k)?;
            n -= k as u64;
        }
        Ok(())
    }

    /// Byte offset of the reader, which must be byte aligned.
    pub(crate) fn byte_pos(&self) -> usize {
        (self.bits_read() / 8) as usize
    }
}

/// Unpack a zig-zag coded signed value.
#[inline]
pub(crate) fn unpack_signed(u: u32) -> i32 {
    ((u >> 1) as i32) ^ -((u & 1) as i32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_lsb_first() {
        let mut br = BitReader::new(&[0b1010_0110, 0xFF]);
        assert_eq!(br.read(3).unwrap(), 0b110);
        assert_eq!(br.read(5).unwrap(), 0b10100);
        assert_eq!(br.bits_read(), 8);
        assert_eq!(br.read(8).unwrap(), 0xFF);
        assert!(br.read(1).is_err());
    }

    #[test]
    fn half_floats() {
        // 1.0, -2.0, 0.5
        let data = [0x00, 0x3C, 0x00, 0xC0, 0x00, 0x38];
        let mut br = BitReader::new(&data);
        assert_eq!(br.f16().unwrap(), 1.0);
        assert_eq!(br.f16().unwrap(), -2.0);
        assert_eq!(br.f16().unwrap(), 0.5);
    }

    #[test]
    fn zigzag() {
        assert_eq!(unpack_signed(0), 0);
        assert_eq!(unpack_signed(1), -1);
        assert_eq!(unpack_signed(2), 1);
        assert_eq!(unpack_signed(u32::MAX), i32::MIN);
    }
}
