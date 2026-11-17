//! Decoded planes and their upsampling to full resolution, one output row at
//! a time, so the decoder can assemble the final interleaved image in a
//! single pass over the decoded planes. The kernels are in `upsample` and
//! `ycbcr`.

use dicom_jpeg_common::interleave;
use dicom_jpeg_common::simd::Level;

use super::upsample;

/// A decoded component plane.
#[derive(Debug, Clone)]
pub(crate) struct Plane {
    pub(crate) data: Vec<u16>,
    pub(crate) stride: usize,
    /// Real (downsampled) width / height of the component.
    pub(crate) width: usize,
    pub(crate) height: usize,
}

impl Plane {
    #[inline]
    fn row(&self, y: usize) -> &[u16] {
        &self.data[y * self.stride..y * self.stride + self.width]
    }
}

#[derive(Debug)]
enum Mode {
    /// Full resolution: rows are used in place.
    Full,
    /// libjpeg h2v1 "fancy" (triangle filter) upsampling.
    H2V1,
    /// libjpeg h2v2 "fancy" upsampling.
    H2V2,
    /// Sample replication for any other factors; `sx` maps output columns.
    Replicate { sx: Vec<usize> },
}

/// Produces the rows of one component at full image resolution.
#[derive(Debug)]
pub(crate) struct Upsampler {
    mode: Mode,
    level: Level,
    precision: u8,
    v: usize,
    vmax: usize,
    row: Vec<u16>,
}

impl Upsampler {
    /// For a component with sampling factors (`h`, `v`) in a frame with
    /// maxima (`hmax`, `vmax`), output width `width` and sample precision
    /// `precision`, using kernels for `level`.
    pub(crate) fn new(
        plane: &Plane,
        (h, v): (usize, usize),
        (hmax, vmax): (usize, usize),
        width: usize,
        level: Level,
        precision: u8,
    ) -> Self {
        let (fx, fy) = (hmax / h.max(1), vmax / v.max(1));
        let integral = hmax.is_multiple_of(h) && vmax.is_multiple_of(v);
        let fancy = integral && fx == 2 && fy <= 2 && plane.width > 2;
        let mode = if fx == 1 && fy == 1 && integral {
            Mode::Full
        } else if fancy && fy == 1 {
            Mode::H2V1
        } else if fancy {
            Mode::H2V2
        } else {
            Mode::Replicate {
                sx: (0..width)
                    .map(|x| ((x * h) / hmax).min(plane.width - 1))
                    .collect(),
            }
        };
        Upsampler {
            mode,
            level,
            precision,
            v,
            vmax,
            row: vec![0; 2 * plane.width.max(width)],
        }
    }

    /// Output row `y` (at least `width` samples).
    pub(crate) fn row<'s>(&'s mut self, plane: &'s Plane, y: usize) -> &'s [u16] {
        match &mut self.mode {
            Mode::Full => plane.row(y),
            Mode::H2V1 => {
                upsample::h2v1(self.level, self.precision, plane.row(y), &mut self.row);
                &self.row
            }
            Mode::H2V2 => {
                let last = plane.height - 1;
                let r = (y / 2).min(last);
                let nr = if y.is_multiple_of(2) {
                    r.saturating_sub(1)
                } else {
                    (r + 1).min(last)
                };
                upsample::h2v2(
                    self.level,
                    self.precision,
                    plane.row(r),
                    plane.row(nr),
                    &mut self.row,
                );
                &self.row
            }
            Mode::Replicate { sx } => {
                let sy = ((y * self.v) / self.vmax).min(plane.height - 1);
                let src = plane.row(sy);
                for (d, &x) in self.row.iter_mut().zip(sx.iter()) {
                    *d = src[x];
                }
                &self.row
            }
        }
    }
}

/// An output sample type.
pub(crate) trait Sample: Copy + Default {
    fn from_u16(v: u16) -> Self;

    /// Interleave three rows into `out` (truncating to the sample type).
    fn interleave3(level: Level, a: &[u16], b: &[u16], c: &[u16], out: &mut [Self]);
}

impl Sample for u8 {
    #[inline(always)]
    fn from_u16(v: u16) -> Self {
        v as u8
    }

    fn interleave3(level: Level, a: &[u16], b: &[u16], c: &[u16], out: &mut [Self]) {
        interleave::interleave3_u8(level, a, b, c, out);
    }
}

impl Sample for u16 {
    #[inline(always)]
    fn from_u16(v: u16) -> Self {
        v
    }

    fn interleave3(level: Level, a: &[u16], b: &[u16], c: &[u16], out: &mut [Self]) {
        interleave::interleave3_u16(level, a, b, c, out);
    }
}
