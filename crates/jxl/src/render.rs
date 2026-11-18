//! The Gaborish restoration filter and the XYB to RGB colour conversion
//! (ISO/IEC 18181-1 Annex J and L); the edge-preserving filter is in `epf`.

use super::headers::OpsinInverse;

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
    let mut out = vec![0f32; pw * (h + 2 * b)];
    for (py, dst) in out.chunks_exact_mut(pw).enumerate() {
        let src = &plane[mirror(py as isize - b as isize, h) * w..][..w];
        dst[b..b + w].copy_from_slice(src);
        for x in (0..b).chain(b + w..pw) {
            dst[x] = src[cols[x]];
        }
    }
    out
}

/// Gaborish: a 3x3 convolution per channel with mirrored edges.
pub(crate) fn gaborish(planes: &mut [Vec<f32>; 3], w: usize, h: usize, weights: [[f32; 2]; 3]) {
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
            for (x, d) in dst.iter_mut().enumerate() {
                let side = mid[x] + mid[x + 2] + up[x + 1] + down[x + 1];
                let diag = up[x] + up[x + 2] + down[x] + down[x + 2];
                *d = (mid[x + 1] + side * w0 + diag * w1) * norm;
            }
        }
    }
}

/// XYB to linear sRGB (in place: X, Y, B -> R, G, B).
pub(crate) fn xyb_to_linear(
    planes: &mut [Vec<f32>; 3],
    opsin: &OpsinInverse,
    intensity_target: f32,
) {
    let itscale = 255.0 / intensity_target;
    let cbrt_bias = opsin.bias.map(|b| b.cbrt());
    let m = &opsin.matrix;
    let n = planes[0].len();
    for i in 0..n {
        let (x, y, b) = (planes[0][i], planes[1][i], planes[2][i]);
        let l = y + x - cbrt_bias[0];
        let mm = y - x - cbrt_bias[1];
        let s = b - cbrt_bias[2];
        let mixed = [
            (l * l * l + opsin.bias[0]) * itscale,
            (mm * mm * mm + opsin.bias[1]) * itscale,
            (s * s * s + opsin.bias[2]) * itscale,
        ];
        for c in 0..3 {
            planes[c][i] = m[c * 3] * mixed[0] + m[c * 3 + 1] * mixed[1] + m[c * 3 + 2] * mixed[2];
        }
    }
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
