//! Image description and sample buffers.

use crate::error::{Error, Result};
use crate::interleave;
use crate::options::Limits;
use crate::simd::Level;

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
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| (u16::from_le_bytes([c[0], c[1]]) & mask) as u8)
                    .collect(),
            ),
            (2, 2) => Samples::U16(
                bytes[..needed]
                    .as_chunks::<2>()
                    .0
                    .iter()
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

    /// Split the interleaved samples into one plane per component. Values
    /// are masked to `bits_per_sample` and sign-extended for signed images.
    pub fn planes(&self) -> Vec<Vec<i32>> {
        let bits = self.info.bits_per_sample as u32;
        let mask = ((1u64 << bits) - 1) as i32;
        let sign = 1i32 << (bits - 1);
        if self.info.signed {
            // Two's complement sign extension of the low `bits` bits.
            self.planes_with(|v| ((v as i32 & mask) ^ sign) - sign)
        } else {
            self.planes_with(|v| v as i32 & mask)
        }
    }

    /// Split the interleaved samples into one plane per component, mapping
    /// each stored sample (as `u16`) through `convert`.
    pub fn planes_with<T: Copy + Default>(&self, convert: impl Fn(u16) -> T) -> Vec<Vec<T>> {
        let nc = self.info.components as usize;
        match &self.samples {
            Samples::U8(v) => deinterleave(v, nc, convert),
            Samples::U16(v) => deinterleave(v, nc, convert),
        }
    }

    /// Interleave component planes into an image. Values are clamped to the
    /// range of `info` (signed values are stored in two's complement).
    pub fn from_planes(info: ImageInfo, planes: &[Vec<i32>]) -> Result<Self> {
        let bits = info.bits_per_sample as u32;
        let (lo, hi) = if info.signed {
            (-(1i32 << (bits - 1)), (1i32 << (bits - 1)) - 1)
        } else {
            (0, ((1u32 << bits) - 1) as i32)
        };
        let mask = ((1u32 << bits) - 1) as i32;
        let planes: Vec<&[i32]> = planes.iter().map(Vec::as_slice).collect();
        Self::from_planes_with(info, &planes, |v| (v.clamp(lo, hi) & mask) as u16)
    }

    /// An image from component planes of `width * height` stored samples
    /// (within the range of `info`), interleaved with the SIMD kernels of
    /// `level`. A single 16-bit plane is moved, not copied.
    pub fn from_u16_planes(
        level: Level,
        info: ImageInfo,
        mut planes: Vec<Vec<u16>>,
    ) -> Result<Self> {
        let nc = info.components as usize;
        let n = info.width as usize * info.height as usize;
        if planes.len() != nc || planes.iter().any(|p| p.len() != n) {
            return Err(Error::input("planes do not match the image description"));
        }
        if !(1..=16).contains(&info.bits_per_sample) {
            return Err(Error::input("bits per sample must be within 1..=16"));
        }
        let narrow = info.bits_per_sample <= 8;
        let samples = match planes.as_mut_slice() {
            [p] if narrow => Samples::U8(p.iter().map(|&v| v as u8).collect()),
            [p] => Samples::U16(std::mem::take(p)),
            [a, b, c] if narrow => {
                let mut out = vec![0u8; 3 * n];
                interleave::interleave3_u8(level, a, b, c, &mut out);
                Samples::U8(out)
            }
            [a, b, c] => {
                let mut out = vec![0u16; 3 * n];
                interleave::interleave3_u16(level, a, b, c, &mut out);
                Samples::U16(out)
            }
            _ => {
                let planes: Vec<&[u16]> = planes.iter().map(Vec::as_slice).collect();
                return Self::from_planes_with(info, &planes, |v| v);
            }
        };
        Ok(Image { info, samples })
    }

    /// Interleave component planes of `width * height` values into an
    /// image, mapping each value to its stored sample through `convert`.
    /// Results are truncated to 8 bits when `info` has at most 8 bits per
    /// sample, so `convert` must keep them in range.
    pub fn from_planes_with<T: Copy>(
        info: ImageInfo,
        planes: &[&[T]],
        convert: impl Fn(T) -> u16,
    ) -> Result<Self> {
        let nc = info.components as usize;
        let n = info.width as usize * info.height as usize;
        if planes.len() != nc || planes.iter().any(|p| p.len() != n) {
            return Err(Error::input("planes do not match the image description"));
        }
        if !(1..=16).contains(&info.bits_per_sample) {
            return Err(Error::input("bits per sample must be within 1..=16"));
        }
        let samples = if info.bits_per_sample <= 8 {
            Samples::U8(interleave(planes, n, convert))
        } else {
            Samples::U16(interleave(planes, n, convert))
        };
        Ok(Image { info, samples })
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

/// A storage type of [`Samples`].
trait Stored: Copy + Default {
    fn from_u16(v: u16) -> Self;
    fn to_u16(self) -> u16;
}

impl Stored for u8 {
    #[inline(always)]
    fn from_u16(v: u16) -> Self {
        v as u8
    }
    #[inline(always)]
    fn to_u16(self) -> u16 {
        self as u16
    }
}

impl Stored for u16 {
    #[inline(always)]
    fn from_u16(v: u16) -> Self {
        v
    }
    #[inline(always)]
    fn to_u16(self) -> u16 {
        self
    }
}

/// Interleave the first `n` values of each plane, in one pass for the
/// common one- and three-component cases.
fn interleave<S: Stored, T: Copy>(planes: &[&[T]], n: usize, f: impl Fn(T) -> u16) -> Vec<S> {
    let mut out = vec![S::default(); n * planes.len()];
    match planes {
        [p] => {
            for (o, &v) in out.iter_mut().zip(&p[..n]) {
                *o = S::from_u16(f(v));
            }
        }
        [a, b, c] => {
            for (o, ((&x, &y), &z)) in out
                .as_chunks_mut::<3>()
                .0
                .iter_mut()
                .zip(a[..n].iter().zip(&b[..n]).zip(&c[..n]))
            {
                *o = [S::from_u16(f(x)), S::from_u16(f(y)), S::from_u16(f(z))];
            }
        }
        _ => {
            let nc = planes.len();
            for (c, p) in planes.iter().enumerate() {
                for (o, &v) in out[c..].iter_mut().step_by(nc).zip(&p[..n]) {
                    *o = S::from_u16(f(v));
                }
            }
        }
    }
    out
}

/// The inverse of [`interleave`].
fn deinterleave<S: Stored, T: Copy + Default>(
    src: &[S],
    nc: usize,
    f: impl Fn(u16) -> T,
) -> Vec<Vec<T>> {
    match nc {
        1 => vec![src.iter().map(|&s| f(s.to_u16())).collect()],
        3 => {
            let px = src.as_chunks::<3>().0;
            let mut planes = [(); 3].map(|_| Vec::with_capacity(px.len()));
            for p in px {
                for (plane, &s) in planes.iter_mut().zip(p) {
                    plane.push(f(s.to_u16()));
                }
            }
            planes.into()
        }
        _ => (0..nc)
            .map(|c| {
                src[c..]
                    .iter()
                    .step_by(nc)
                    .map(|&s| f(s.to_u16()))
                    .collect()
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(components: u16, bits: u8, signed: bool) -> ImageInfo {
        ImageInfo {
            width: 3,
            height: 2,
            components,
            bits_per_sample: bits,
            signed,
            color_space: ColorSpace::Unknown,
        }
    }

    #[test]
    fn planes_round_trip() {
        for nc in [1u16, 2, 3, 4] {
            for (bits, signed) in [(8, false), (12, true), (16, false), (16, true)] {
                let info = info(nc, bits, signed);
                let (lo, hi) = if signed {
                    (-(1i32 << (bits - 1)), (1i32 << (bits - 1)) - 1)
                } else {
                    (0, (1i32 << bits) - 1)
                };
                let planes: Vec<Vec<i32>> = (0..nc as i32)
                    .map(|c| {
                        (0..6)
                            .map(|i| [lo, hi, 0, 1, lo + c, hi - i][i as usize])
                            .collect()
                    })
                    .collect();
                let image = Image::from_planes(info, &planes).unwrap();
                assert_eq!(image.samples.len(), 6 * nc as usize);
                assert_eq!(image.planes(), planes, "{nc} x {bits} bit, signed {signed}");
            }
        }
    }

    #[test]
    fn from_planes_clamps_and_masks() {
        let image =
            Image::from_planes(info(1, 12, true), &[vec![-5000, 5000, -1, 0, 1, 2]]).unwrap();
        assert_eq!(
            image.samples,
            Samples::U16(vec![0x800, 0x7FF, 0xFFF, 0, 1, 2])
        );
    }
}
