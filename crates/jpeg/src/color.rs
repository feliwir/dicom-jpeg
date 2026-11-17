//! Chroma upsampling and YCbCr -> RGB conversion, bit-exact with IJG libjpeg 6b
//! (jdsample.c "fancy" upsampling, jdcolor.c fixed-point conversion).
//!
//! Both work one output row at a time, so the decoder can assemble the final
//! interleaved image in a single pass over the decoded planes.

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
    /// libjpeg h2v2 "fancy" upsampling; `colsum` is scratch.
    H2V2 { colsum: Vec<u32> },
    /// Sample replication for any other factors; `sx` maps output columns.
    Replicate { sx: Vec<usize> },
}

/// Produces the rows of one component at full image resolution.
#[derive(Debug)]
pub(crate) struct Upsampler {
    mode: Mode,
    v: usize,
    vmax: usize,
    row: Vec<u16>,
}

impl Upsampler {
    /// For a component with sampling factors (`h`, `v`) in a frame with
    /// maxima (`hmax`, `vmax`) and output width `width`.
    pub(crate) fn new(
        plane: &Plane,
        (h, v): (usize, usize),
        (hmax, vmax): (usize, usize),
        width: usize,
    ) -> Self {
        let (fx, fy) = (hmax / h.max(1), vmax / v.max(1));
        let integral = hmax.is_multiple_of(h) && vmax.is_multiple_of(v);
        let fancy = integral && fx == 2 && fy <= 2 && plane.width > 2;
        let mode = if fx == 1 && fy == 1 && integral {
            Mode::Full
        } else if fancy && fy == 1 {
            Mode::H2V1
        } else if fancy {
            Mode::H2V2 {
                colsum: vec![0; plane.width],
            }
        } else {
            Mode::Replicate {
                sx: (0..width)
                    .map(|x| ((x * h) / hmax).min(plane.width - 1))
                    .collect(),
            }
        };
        Upsampler {
            mode,
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
                h2v1(plane.row(y), &mut self.row);
                &self.row
            }
            Mode::H2V2 { colsum } => {
                let last = plane.height - 1;
                let r = (y / 2).min(last);
                let nr = if y.is_multiple_of(2) {
                    r.saturating_sub(1)
                } else {
                    (r + 1).min(last)
                };
                for ((cs, &a), &b) in colsum.iter_mut().zip(plane.row(r)).zip(plane.row(nr)) {
                    *cs = a as u32 * 3 + b as u32;
                }
                h2v2(colsum, &mut self.row);
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

/// Horizontal triangle filter: output 2i gets 3/4 of input i and 1/4 of
/// i-1 (rounding bias 1), output 2i+1 3/4 of i and 1/4 of i+1 (bias 2).
fn h2v1(src: &[u16], dst: &mut [u16]) {
    let w = src.len();
    let (first, last) = (src[0], src[w - 1]);
    dst[0] = first;
    dst[1] = ((first as u32 * 3 + src[1] as u32 + 2) >> 2) as u16;
    for (d, s) in dst[2..2 * w - 2]
        .as_chunks_mut::<2>()
        .0
        .iter_mut()
        .zip(src.windows(3))
    {
        let c = s[1] as u32 * 3;
        d[0] = ((c + s[0] as u32 + 1) >> 2) as u16;
        d[1] = ((c + s[2] as u32 + 2) >> 2) as u16;
    }
    dst[2 * w - 2] = ((last as u32 * 3 + src[w - 2] as u32 + 1) >> 2) as u16;
    dst[2 * w - 1] = last;
}

/// The horizontal step of h2v2 on column sums (3 * near + far): weights
/// 3:1 again, rounding bias 8 for even and 7 for odd outputs.
fn h2v2(cs: &[u32], dst: &mut [u16]) {
    let w = cs.len();
    dst[0] = ((cs[0] * 4 + 8) >> 4) as u16;
    dst[1] = ((cs[0] * 3 + cs[1] + 7) >> 4) as u16;
    for (d, s) in dst[2..2 * w - 2]
        .as_chunks_mut::<2>()
        .0
        .iter_mut()
        .zip(cs.windows(3))
    {
        let c = s[1] * 3;
        d[0] = ((c + s[0] + 8) >> 4) as u16;
        d[1] = ((c + s[2] + 7) >> 4) as u16;
    }
    dst[2 * w - 2] = ((cs[w - 1] * 3 + cs[w - 2] + 8) >> 4) as u16;
    dst[2 * w - 1] = ((cs[w - 1] * 4 + 7) >> 4) as u16;
}

/// An output sample type.
pub(crate) trait Sample: Copy + Default {
    fn from_u16(v: u16) -> Self;
}

impl Sample for u8 {
    #[inline(always)]
    fn from_u16(v: u16) -> Self {
        v as u8
    }
}

impl Sample for u16 {
    #[inline(always)]
    fn from_u16(v: u16) -> Self {
        v
    }
}

const SCALEBITS: u32 = 16;
const ONE_HALF: i32 = 1 << (SCALEBITS - 1);

const fn fix(x: f64) -> i32 {
    (x * (1u64 << SCALEBITS) as f64 + 0.5) as i32
}

/// Convert one row of Y, Cb, Cr samples to interleaved RGB.
pub(crate) fn ycbcr_to_rgb<S: Sample>(
    y: &[u16],
    cb: &[u16],
    cr: &[u16],
    precision: u8,
    out: &mut [S],
) {
    let max = (1i32 << precision) - 1;
    let center = 1i32 << (precision - 1);
    let (fr, fb, fgr, fgb) = (fix(1.40200), fix(1.77200), fix(0.71414), fix(0.34414));
    for (((px, &yy), &cb), &cr) in out.as_chunks_mut::<3>().0.iter_mut().zip(y).zip(cb).zip(cr) {
        let yy = yy as i32;
        let cbv = cb as i32 - center;
        let crv = cr as i32 - center;
        let r = yy + ((fr * crv + ONE_HALF) >> SCALEBITS);
        let g = yy + ((-fgb * cbv + ONE_HALF - fgr * crv) >> SCALEBITS);
        let b = yy + ((fb * cbv + ONE_HALF) >> SCALEBITS);
        px[0] = S::from_u16(r.clamp(0, max) as u16);
        px[1] = S::from_u16(g.clamp(0, max) as u16);
        px[2] = S::from_u16(b.clamp(0, max) as u16);
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
        let mut up = Upsampler::new(&plane, (1, 1), (2, 1), 6);
        // libjpeg: [10, (30+20+2)>>2=13, (60+10+1)>>2=17, (60+40+2)>>2=25, (120+20+1)>>2=35, 40]
        assert_eq!(up.row(&plane, 0)[..6], [10, 13, 17, 25, 35, 40]);
    }
}
