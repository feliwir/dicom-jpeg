//! Image description and sample buffers.

use crate::error::{Error, Result};
use crate::options::Limits;

/// Colour model of the samples in an [`Image`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ColorSpace {
    /// One component, greyscale (or palette indices).
    Gray,
    /// Three components, R G B.
    Rgb,
    /// Three components, Y Cb Cr (full range, ITU-R BT.601 / JFIF).
    YCbCr,
    /// Any other component interpretation.
    Unknown,
}

/// Geometry and sample format of an image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImageInfo {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Number of components (samples per pixel).
    pub components: u16,
    /// Significant bits per sample (1..=16).
    pub bits_per_sample: u8,
    /// Whether samples are signed (two's complement in the low `bits_per_sample` bits).
    pub signed: bool,
    /// Colour model of the samples.
    pub color_space: ColorSpace,
}

impl ImageInfo {
    /// Bytes needed to store one sample (1 for <= 8 bits, otherwise 2).
    pub fn bytes_per_sample(&self) -> usize {
        if self.bits_per_sample <= 8 { 1 } else { 2 }
    }

    /// Total number of samples (`width * height * components`), if it fits in `usize`.
    pub fn sample_count(&self) -> Option<usize> {
        (self.width as usize)
            .checked_mul(self.height as usize)?
            .checked_mul(self.components as usize)
    }

    /// Validate the description against basic sanity rules and `limits`.
    pub fn validate(&self, limits: &Limits) -> Result<()> {
        if self.width == 0 || self.height == 0 {
            return Err(Error::input("image dimensions must be non-zero"));
        }
        if self.components == 0 {
            return Err(Error::input("image must have at least one component"));
        }
        if !(1..=16).contains(&self.bits_per_sample) {
            return Err(Error::input("bits per sample must be within 1..=16"));
        }
        limits.check_image(self)
    }
}

/// Interleaved sample storage (DICOM planar configuration 0).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Samples {
    /// Samples of at most 8 bits.
    U8(Vec<u8>),
    /// Samples of 9 to 16 bits (signed samples are stored as their bit pattern).
    U16(Vec<u16>),
}

impl Samples {
    /// Number of samples.
    pub fn len(&self) -> usize {
        match self {
            Samples::U8(v) => v.len(),
            Samples::U16(v) => v.len(),
        }
    }

    /// Whether there are no samples.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A decoded image, or an image to be encoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image {
    /// Geometry and sample format.
    pub info: ImageInfo,
    /// Interleaved samples, row-major.
    pub samples: Samples,
}

impl Image {
    /// Create an image, checking that the buffer matches the description.
    pub fn new(info: ImageInfo, samples: Samples) -> Result<Self> {
        let expected = info
            .sample_count()
            .ok_or_else(|| Error::input("image too large"))?;
        if samples.len() != expected {
            return Err(Error::input(format!(
                "sample buffer has {} samples, expected {expected}",
                samples.len()
            )));
        }
        match (&samples, info.bytes_per_sample()) {
            (Samples::U8(_), 1) | (Samples::U16(_), 2) => {}
            (Samples::U16(_), 1) if info.bits_per_sample <= 8 => {}
            _ => {
                return Err(Error::input(
                    "sample storage type does not match bits per sample",
                ));
            }
        }
        Ok(Image { info, samples })
    }

    /// Build an image from little-endian bytes with the given storage width
    /// (`bytes_per_sample` of 1 or 2, e.g. DICOM Bits Allocated / 8).
    ///
    /// Bits above `info.bits_per_sample` are masked off.
    pub fn from_le_bytes(info: ImageInfo, bytes: &[u8], bytes_per_sample: usize) -> Result<Self> {
        let count = info
            .sample_count()
            .ok_or_else(|| Error::input("image too large"))?;
        let needed = count
            .checked_mul(bytes_per_sample)
            .ok_or_else(|| Error::input("image too large"))?;
        if bytes.len() < needed {
            return Err(Error::input(format!(
                "pixel buffer has {} bytes, expected {needed}",
                bytes.len()
            )));
        }
        let mask: u16 = if info.bits_per_sample >= 16 {
            u16::MAX
        } else {
            (1u16 << info.bits_per_sample) - 1
        };
        let samples = match (bytes_per_sample, info.bytes_per_sample()) {
            (1, 1) => Samples::U8(bytes[..count].iter().map(|&b| b & mask as u8).collect()),
            (2, 1) => Samples::U8(
                bytes[..needed]
                    .chunks_exact(2)
                    .map(|c| (u16::from_le_bytes([c[0], c[1]]) & mask) as u8)
                    .collect(),
            ),
            (2, 2) => Samples::U16(
                bytes[..needed]
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]) & mask)
                    .collect(),
            ),
            _ => {
                return Err(Error::input(
                    "unsupported storage width for the given bits per sample",
                ));
            }
        };
        Image::new(info, samples)
    }

    /// Append the samples as little-endian bytes using `bytes_per_sample` (1 or 2)
    /// bytes of storage per sample.
    ///
    /// Signed samples narrower than the storage are sign-extended.
    pub fn write_le_bytes(&self, bytes_per_sample: usize, dst: &mut Vec<u8>) -> Result<()> {
        let bits = self.info.bits_per_sample as u32;
        let sign_extend = |v: u16| -> u16 {
            if self.info.signed && bits < 16 && v & (1 << (bits - 1)) != 0 {
                v | !((1u16 << bits) - 1)
            } else {
                v
            }
        };
        match (&self.samples, bytes_per_sample) {
            (Samples::U8(v), 1) => dst.extend_from_slice(v),
            (Samples::U8(v), 2) => {
                dst.reserve(v.len() * 2);
                for &s in v {
                    dst.extend_from_slice(&sign_extend(s as u16).to_le_bytes());
                }
            }
            (Samples::U16(v), 2) => {
                dst.reserve(v.len() * 2);
                for &s in v {
                    dst.extend_from_slice(&sign_extend(s).to_le_bytes());
                }
            }
            (Samples::U16(_), 1) => {
                return Err(Error::input(
                    "image has more than 8 bits per sample, cannot store in 1 byte",
                ));
            }
            _ => return Err(Error::input("storage must be 1 or 2 bytes per sample")),
        }
        Ok(())
    }
}
