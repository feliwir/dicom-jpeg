//! The Gaborish restoration filter and the XYB to RGB colour conversion
//! (ISO/IEC 18181-1 Annex J and L); the edge-preserving filter is in `epf`.

use dicom_jpeg_common::simd;

use super::gaborish;
use super::headers::OpsinInverse;
use super::xyb::{self, Xyb};

/// Mirror an index into `0..len` (whole-sample symmetric).
#[inline]
fn mirror(mut i: isize, len: usize) -> usize {
    let len = len as isize;
    loop {
        if i < 0 {
            i = -i - 1;
        } else if i >= len {
            i = 2 * len - i - 1;
        } else {
            return i as usize;
        }
    }
}

/// A copy of the `w` x `h` plane with a mirrored border of `b` samples
/// (row stride `w + 2b`).
pub(crate) fn padded(plane: &[f32], w: usize, h: usize, b: usize) -> Vec<f32> {
    let pw = w + 2 * b;
    let cols: Vec<usize> = (0..pw)
        .map(|x| mirror(x as isize - b as isize, w))
        .collect();
    // Appended row by row, so the buffer is never zeroed first.
    let mut out = Vec::with_capacity(pw * (h + 2 * b));
    for py in 0..h + 2 * b {
        let src = &plane[mirror(py as isize - b as isize, h) * w..][..w];
        out.extend(cols[..b].iter().map(|&x| src[x]));
        out.extend_from_slice(src);
        out.extend(cols[b + w..].iter().map(|&x| src[x]));
    }
    out
}

/// Gaborish: a 3x3 convolution per channel with mirrored edges.
pub(crate) fn gaborish(planes: &mut [Vec<f32>; 3], w: usize, h: usize, weights: [[f32; 2]; 3]) {
    let level = simd::level();
    let pw = w + 2;
    for (c, plane) in planes.iter_mut().enumerate() {
        let [w0, w1] = weights[c];
        let norm = 1.0 / (1.0 + 4.0 * w0 + 4.0 * w1);
        let src = padded(plane, w, h, 1);
        for (y, dst) in plane[..w * h].chunks_exact_mut(w).enumerate() {
            let (up, mid, down) = (
                &src[y * pw..][..pw],
                &src[(y + 1) * pw..][..pw],
                &src[(y + 2) * pw..][..pw],
            );
            gaborish::row(level, up, mid, down, dst, [w0, w1, norm]);
        }
    }
}

/// XYB to linear sRGB (in place: X, Y, B -> R, G, B).
pub(crate) fn xyb_to_linear(
    planes: &mut [Vec<f32>; 3],
    opsin: &OpsinInverse,
    intensity_target: f32,
) {
    let k = Xyb {
        cbrt_bias: opsin.bias.map(|b| b.cbrt()),
        bias: opsin.bias,
        itscale: 255.0 / intensity_target,
        matrix: opsin.matrix,
    };
    let n = planes[0].len();
    let [p0, p1, p2] = planes;
    xyb::to_linear(
        simd::level(),
        [&mut p0[..n], &mut p1[..n], &mut p2[..n]],
        &k,
    );
}

/// Linear light to the encoded transfer function of the output colour space.
pub(crate) enum Transfer {
    Linear,
    Srgb,
    Bt709,
    Dci,
    /// Encoded = linear^gamma.
    Gamma(f32),
}

impl Transfer {
    /// `dst[i] = self.encode(src[i])`, with the choice of function made
    /// once rather than per sample.
    pub(crate) fn encode_into(&self, src: &[f32], dst: &mut [f32]) {
        fn map(src: &[f32], dst: &mut [f32], f: impl Fn(f32) -> f32) {
            for (d, &v) in dst.iter_mut().zip(src) {
                *d = f(v);
            }
        }
        match *self {
            Transfer::Linear => map(src, dst, |v| Transfer::Linear.encode(v)),
            Transfer::Srgb => map(src, dst, |v| Transfer::Srgb.encode(v)),
            Transfer::Bt709 => map(src, dst, |v| Transfer::Bt709.encode(v)),
            Transfer::Dci => map(src, dst, |v| Transfer::Dci.encode(v)),
            Transfer::Gamma(g) => map(src, dst, |v| Transfer::Gamma(g).encode(v)),
        }
    }

    #[inline(always)]
    pub(crate) fn encode(&self, v: f32) -> f32 {
        let v = v.clamp(0.0, 1.0);
        match *self {
            Transfer::Linear => v,
            Transfer::Srgb => {
                if v <= 0.003_130_8 {
                    12.92 * v
                } else {
                    1.055 * v.powf(1.0 / 2.4) - 0.055
                }
            }
            Transfer::Bt709 => {
                if v < 0.018 {
                    4.5 * v
                } else {
                    1.099 * v.powf(0.45) - 0.099
                }
            }
            Transfer::Dci => v.powf(1.0 / 2.6),
            Transfer::Gamma(g) => v.powf(g),
        }
    }
}
