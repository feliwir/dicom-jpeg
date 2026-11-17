//! Restoration filters (Gaborish, edge-preserving filter) and the XYB to
//! RGB colour conversion (ISO/IEC 18181-1 Annex J and L).

use super::frame::{EpfParams, FrameHeader};
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
fn padded(plane: &[f32], w: usize, h: usize, b: usize) -> Vec<f32> {
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

/// One EPF step: the kernel taps, and the distance taps around each.
trait EpfStep {
    const KERNEL: &'static [(usize, usize)];
    const DIST: &'static [(usize, usize)];
}

// Offsets are (dx + 3, dy + 3), indices into a 7 x 7 window.
struct Step0;
impl EpfStep for Step0 {
    const KERNEL: &'static [(usize, usize)] = &[
        (3, 1),
        (2, 2),
        (3, 2),
        (4, 2),
        (1, 3),
        (2, 3),
        (4, 3),
        (5, 3),
        (2, 4),
        (3, 4),
        (4, 4),
        (3, 5),
    ];
    const DIST: &'static [(usize, usize)] = &[(3, 2), (4, 3), (3, 3), (2, 3), (3, 4)];
}

struct Step1;
impl EpfStep for Step1 {
    const KERNEL: &'static [(usize, usize)] = &[(3, 2), (3, 4), (2, 3), (4, 3)];
    const DIST: &'static [(usize, usize)] = &[(3, 2), (3, 3), (3, 4), (2, 3), (4, 3)];
}

struct Step2;
impl EpfStep for Step2 {
    const KERNEL: &'static [(usize, usize)] = Step1::KERNEL;
    const DIST: &'static [(usize, usize)] = &[(3, 3)];
}

/// Border of the padded planes: kernel reach (2) plus distance reach (1).
const EPF_BORDER: usize = 3;

/// One EPF step on `planes`, in place.
fn epf_step<S: EpfStep>(
    planes: &mut [Vec<f32>; 3],
    w: usize,
    h: usize,
    sigma: &[f32],
    sigma_stride: usize,
    params: &EpfParams,
    step_mul: f32,
) {
    const B: usize = EPF_BORDER;
    let pw = w + 2 * B;
    let src: [Vec<f32>; 3] = std::array::from_fn(|c| padded(&planes[c], w, h, B));
    let scale = params.channel_scale;
    let mul = [step_mul, step_mul * params.border_sad_mul];
    for y in 0..h {
        let y_border = (y + 1) & 0b110 == 0;
        let sigma_row = &sigma[(y / 8) * sigma_stride..];
        // Rows y - 3 ..= y + 3 of each padded channel.
        let rows: [[&[f32]; 7]; 3] =
            std::array::from_fn(|c| std::array::from_fn(|j| &src[c][(y + j) * pw..][..pw]));
        let neg_inv_sigma = |s: f32, border: bool| {
            6.6 * (std::f32::consts::FRAC_1_SQRT_2 - 1.0) / s * mul[border as usize]
        };
        for x0 in (0..w).step_by(8) {
            // sigma is constant over the 8 columns of a block.
            let s = sigma_row[x0 / 8];
            if s < 0.3 {
                // Unfiltered: `planes` already holds the input.
                continue;
            }
            if x0 + 8 <= w {
                let nis: [f32; 8] =
                    std::array::from_fn(|l| neg_inv_sigma(s, y_border || l == 0 || l == 7));
                // Columns x0 - 3 .. x0 + 11 of rows y - 3 ..= y + 3.
                let win: [[&[f32; 14]; 7]; 3] = std::array::from_fn(|c| {
                    std::array::from_fn(|j| rows[c][j][x0..].first_chunk().expect("padded row"))
                });
                let out = epf_lanes::<S>(&win, &nis, &scale);
                for (c, o) in out.iter().enumerate() {
                    planes[c][y * w + x0..][..8].copy_from_slice(o);
                }
                continue;
            }
            for x in x0..w {
                let nis = neg_inv_sigma(s, y_border || matches!(x & 7, 0 | 7));
                let win: [[&[f32; 7]; 7]; 3] = std::array::from_fn(|c| {
                    std::array::from_fn(|j| rows[c][j][x..].first_chunk().expect("padded row"))
                });
                let out = epf_pixel::<S>(&win, nis, &scale);
                for (c, &o) in out.iter().enumerate() {
                    planes[c][y * w + x] = o;
                }
            }
        }
    }
}

/// The EPF of one pixel from its 7 x 7 window per channel.
#[inline(always)]
fn epf_pixel<S: EpfStep>(win: &[[&[f32; 7]; 7]; 3], nis: f32, scale: &[f32; 3]) -> [f32; 3] {
    let at = |c: usize, (dx, dy): (usize, usize)| win[c][dy][dx];
    let mut sum_w = 1.0f32;
    let mut sums: [f32; 3] = std::array::from_fn(|c| at(c, (3, 3)));
    for &(kx, ky) in S::KERNEL {
        let mut d = 0f32;
        for (c, &sc) in scale.iter().enumerate() {
            let mut acc = 0f32;
            for &(ox, oy) in S::DIST {
                acc += (at(c, (kx + ox - 3, ky + oy - 3)) - at(c, (ox, oy))).abs();
            }
            d += sc * acc;
        }
        let weight = (1.0 + d * nis).max(0.0);
        sum_w += weight;
        for (c, sum) in sums.iter_mut().enumerate() {
            *sum += weight * at(c, (kx, ky));
        }
    }
    sums.map(|s| s / sum_w)
}

/// `epf_pixel` for 8 adjacent pixels at once (the same operations per
/// lane), from their 14 x 7 window per channel.
#[inline(always)]
fn epf_lanes<S: EpfStep>(
    win: &[[&[f32; 14]; 7]; 3],
    nis: &[f32; 8],
    scale: &[f32; 3],
) -> [[f32; 8]; 3] {
    let at = |c: usize, (dx, dy): (usize, usize)| -> [f32; 8] {
        win[c][dy][dx..dx + 8].try_into().expect("8 lanes")
    };
    let mut sum_w = [1.0f32; 8];
    let mut sums: [[f32; 8]; 3] = std::array::from_fn(|c| at(c, (3, 3)));
    for &(kx, ky) in S::KERNEL {
        let mut d = [0f32; 8];
        for (c, &sc) in scale.iter().enumerate() {
            let mut acc = [0f32; 8];
            for &(ox, oy) in S::DIST {
                let (a, b) = (at(c, (kx + ox - 3, ky + oy - 3)), at(c, (ox, oy)));
                for l in 0..8 {
                    acc[l] += (a[l] - b[l]).abs();
                }
            }
            for l in 0..8 {
                d[l] += sc * acc[l];
            }
        }
        let mut weight = [0f32; 8];
        for l in 0..8 {
            weight[l] = (1.0 + d[l] * nis[l]).max(0.0);
            sum_w[l] += weight[l];
        }
        for (c, sum) in sums.iter_mut().enumerate() {
            let v = at(c, (kx, ky));
            for l in 0..8 {
                sum[l] += weight[l] * v[l];
            }
        }
    }
    let mut out = [[0f32; 8]; 3];
    for c in 0..3 {
        for l in 0..8 {
            out[c][l] = sums[c][l] / sum_w[l];
        }
    }
    out
}

/// Edge-preserving filter with a per-8x8-block sigma map.
pub(crate) fn epf(
    planes: &mut [Vec<f32>; 3],
    w: usize,
    h: usize,
    sigma: &[f32],
    sigma_stride: usize,
    fh: &FrameHeader,
) {
    let iters = fh.epf_iters;
    let p = &fh.epf;
    if iters == 3 {
        epf_step::<Step0>(planes, w, h, sigma, sigma_stride, p, p.pass0_sigma_scale);
    }
    if iters >= 1 {
        epf_step::<Step1>(planes, w, h, sigma, sigma_stride, p, 1.0);
    }
    if iters >= 2 {
        epf_step::<Step2>(planes, w, h, sigma, sigma_stride, p, p.pass2_sigma_scale);
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
