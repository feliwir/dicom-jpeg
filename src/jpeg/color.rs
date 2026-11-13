//! Chroma upsampling and YCbCr -> RGB conversion, bit-exact with IJG libjpeg 6b
//! (jdsample.c "fancy" upsampling, jdcolor.c fixed-point conversion).

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
    fn at(&self, x: usize, y: usize) -> u16 {
        self.data[y * self.stride + x]
    }
}

/// Upsample `plane` (sampling factors `h`, `v` of `hmax`, `vmax`) to
/// `width` x `height` samples.
pub(crate) fn upsample(
    plane: &Plane,
    h: usize,
    v: usize,
    hmax: usize,
    vmax: usize,
    width: usize,
    height: usize,
) -> Vec<u16> {
    let (fx, fy) = (hmax / h.max(1), vmax / v.max(1));
    let integral = hmax % h == 0 && vmax % v == 0;
    let mut out = vec![0u16; width * height];
    if fx == 1 && fy == 1 && integral {
        for y in 0..height {
            out[y * width..(y + 1) * width]
                .copy_from_slice(&plane.data[y * plane.stride..y * plane.stride + width]);
        }
        return out;
    }
    let fancy = integral && fx == 2 && fy <= 2 && plane.width > 2;
    if fancy && fy == 1 {
        // h2v1_fancy_upsample
        let w = plane.width;
        for y in 0..height {
            let row = &plane.data[y * plane.stride..y * plane.stride + w];
            let dst = &mut out[y * width..(y + 1) * width];
            for (x, d) in dst.iter_mut().enumerate() {
                let i = x / 2;
                let cur = row[i] as u32 * 3;
                let v = if x % 2 == 0 {
                    (cur + row[i.saturating_sub(1)] as u32 + 1) >> 2
                } else {
                    (cur + row[(i + 1).min(w - 1)] as u32 + 2) >> 2
                };
                *d = v as u16;
            }
        }
        return out;
    }
    if fancy && fy == 2 {
        // h2v2_fancy_upsample: rows of column sums with the nearer neighbour row.
        let w = plane.width;
        let last_row = plane.height - 1;
        let mut colsum = vec![0u32; w];
        for y in 0..height {
            let r = (y / 2).min(last_row);
            let nr = if y % 2 == 0 {
                r.saturating_sub(1)
            } else {
                (r + 1).min(last_row)
            };
            for (i, cs) in colsum.iter_mut().enumerate() {
                *cs = plane.at(i, r) as u32 * 3 + plane.at(i, nr) as u32;
            }
            let dst = &mut out[y * width..(y + 1) * width];
            for (x, d) in dst.iter_mut().enumerate() {
                let i = x / 2;
                let this = colsum[i] * 3;
                let v = if x % 2 == 0 {
                    (this + colsum[i.saturating_sub(1)] + 8) >> 4
                } else {
                    (this + colsum[(i + 1).min(w - 1)] + 7) >> 4
                };
                *d = v as u16;
            }
        }
        return out;
    }
    // Replication (also used for non-integral ratios via nearest neighbour).
    for y in 0..height {
        let sy = ((y * v) / vmax).min(plane.height - 1);
        let dst = &mut out[y * width..(y + 1) * width];
        for (x, d) in dst.iter_mut().enumerate() {
            let sx = ((x * h) / hmax).min(plane.width - 1);
            *d = plane.at(sx, sy);
        }
    }
    out
}

const SCALEBITS: u32 = 16;
const ONE_HALF: i64 = 1 << (SCALEBITS - 1);

const fn fix(x: f64) -> i64 {
    (x * (1u64 << SCALEBITS) as f64 + 0.5) as i64
}

/// Convert full-resolution Y, Cb, Cr planes to interleaved RGB in place of `out`.
pub(crate) fn ycbcr_to_rgb(y: &[u16], cb: &[u16], cr: &[u16], precision: u8, out: &mut [u16]) {
    let max = (1i64 << precision) - 1;
    let center = 1i64 << (precision - 1);
    let (fr, fb, fgr, fgb) = (fix(1.40200), fix(1.77200), fix(0.71414), fix(0.34414));
    for (i, px) in out.chunks_exact_mut(3).enumerate() {
        let yy = y[i] as i64;
        let cbv = cb[i] as i64 - center;
        let crv = cr[i] as i64 - center;
        let r = yy + ((fr * crv + ONE_HALF) >> SCALEBITS);
        let g = yy + ((-fgb * cbv + ONE_HALF - fgr * crv) >> SCALEBITS);
        let b = yy + ((fb * cbv + ONE_HALF) >> SCALEBITS);
        px[0] = r.clamp(0, max) as u16;
        px[1] = g.clamp(0, max) as u16;
        px[2] = b.clamp(0, max) as u16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gray_maps_to_gray() {
        let mut out = [0u16; 3];
        ycbcr_to_rgb(&[77], &[128], &[128], 8, &mut out);
        assert_eq!(out, [77, 77, 77]);
    }

    #[test]
    fn h2v1_fancy_matches_libjpeg_formula() {
        let plane = Plane {
            data: vec![10, 20, 40],
            stride: 3,
            width: 3,
            height: 1,
        };
        let out = upsample(&plane, 1, 1, 2, 1, 6, 1);
        // libjpeg: [10, (30+20+2)>>2=13, (60+10+1)>>2=17, (60+40+2)>>2=25, (120+20+1)>>2=35, 40]
        assert_eq!(out, vec![10, 13, 17, 25, 35, 40]);
    }
}
