//! Accurate integer forward DCT ("ISLOW", IJG jfdctint.c) and quantisation.
//!
//! Output coefficients are scaled up by 8 relative to T.81 A.3.3, as in
//! libjpeg; the quantizer divides by `8 * q` and rounds half away from
//! zero. As in libjpeg, 8-bit samples are transformed in 32-bit and 12-bit
//! samples in 64-bit arithmetic, which cannot overflow for samples of the
//! given precision. 8-bit data has SIMD versions; they quantise by `f32`
//! division, which is exact here (see [`QuantTable`]).

use dicom_jpeg_common::dispatch;
use dicom_jpeg_common::simd::Level;

mod scalar;

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "x86_64")]
mod sse41;

const CONST_BITS: u32 = 13;

const FIX_0_298631336: i32 = 2446;
const FIX_0_390180644: i32 = 3196;
const FIX_0_541196100: i32 = 4433;
const FIX_0_765366865: i32 = 6270;
const FIX_0_899976223: i32 = 7373;
const FIX_1_175875602: i32 = 9633;
const FIX_1_501321110: i32 = 12299;
const FIX_1_847759065: i32 = 15137;
const FIX_1_961570560: i32 = 16069;
const FIX_2_053119869: i32 = 16819;
const FIX_2_562915447: i32 = 20995;
const FIX_3_072711026: i32 = 25172;

/// A divisor with its reciprocal for exact division by multiplication
/// (Lemire et al., "Faster remainder by direct computation"): for any
/// 32-bit `n`, `n / d == (m * n) >> 64` with `m = 2^64 / d + 1`.
#[derive(Debug, Clone, Copy)]
struct Divisor {
    half: u32,
    m: u64,
}

impl Divisor {
    fn new(d: u32) -> Self {
        Divisor {
            half: d >> 1,
            m: u64::MAX / d as u64 + 1,
        }
    }

    /// libjpeg quantisation: round half away from zero.
    #[inline(always)]
    fn quantize(self, c: i32) -> i32 {
        let a = c.unsigned_abs() + self.half;
        let v = ((self.m as u128 * a as u128) >> 64) as i32;
        if c < 0 { -v } else { v }
    }
}

/// The divisors `8 * q` of a quantisation table (natural order), and the
/// coefficient limit.
///
/// SIMD versions compute `(|c| + d / 2) / d` as the truncated `f32`
/// quotient, which equals the integer quotient: with `a = |c| + d / 2` and
/// `a + d < 2^24` (8-bit coefficients stay below 2^15 and `d <= 8 * 65535`),
/// the correctly rounded quotient never reaches the next integer, as that
/// is at least `1 / d` away while half a unit in the last place there is at
/// most `(a / d + 1) * 2^-24 < 1 / d`.
#[derive(Debug)]
pub struct QuantTable {
    div: [Divisor; 64],
    divf: [f32; 64],
    half: [i32; 64],
    max: i32,
}

impl QuantTable {
    /// For quantisation values `q` (natural order) and coefficients limited
    /// to `-max..=max`.
    pub fn new(q: &[u16; 64], max: i32) -> Self {
        let d = |k: usize| q[k].max(1) as u32 * 8;
        QuantTable {
            div: std::array::from_fn(|k| Divisor::new(d(k))),
            divf: std::array::from_fn(|k| d(k) as f32),
            half: std::array::from_fn(|k| (d(k) >> 1) as i32),
            max,
        }
    }
}

/// Forward DCT and quantisation (natural order) of the blocks of one block
/// row: block `i` is at `src[8 * i..]` with row stride `stride`.
pub fn fdct_quantize(
    level: Level,
    src: &[u16],
    stride: usize,
    precision: u8,
    t: &QuantTable,
    out: &mut [[i32; 64]],
) {
    if let Some(n) = out.len().checked_sub(1) {
        // The last row of the last block.
        assert!(src.len() >= 7 * stride + 8 * n + 8);
    }
    if precision == 12 {
        return dispatch!(level => fdct_quantize12(src, stride, t, out));
    }
    if precision > 8 {
        for (i, o) in out.iter_mut().enumerate() {
            scalar::fdct_block(&src[8 * i..], stride, precision, o);
            scalar::quantize(t, o);
        }
        return;
    }
    dispatch!(level => fdct_quantize8(src, stride, t, out));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simd_matches_scalar() {
        let mut seed = 3u32;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        let (bw, stride) = (13usize, 13 * 8 + 5);
        let samples = |precision: u8, rnd: &mut dyn FnMut() -> u32| -> Vec<u16> {
            let max = (1u32 << precision) - 1;
            (0..8 * stride)
                .map(|i| match (i / 64) % 4 {
                    0 => (rnd() % (max + 1)) as u16,
                    1 => [0, max as u16][(i + i / stride) % 2],
                    2 => max as u16,
                    _ => (i as u32 % (max + 1)) as u16,
                })
                .collect()
        };
        let tables = [
            QuantTable::new(&[1; 64], 1023),
            QuantTable::new(&std::array::from_fn(|k| 1 + (k as u16 * 37) % 255), 1023),
            QuantTable::new(&std::array::from_fn(|_| 1 + (rnd() as u16)), 16383),
            QuantTable::new(&[2; 64], 30),
        ];
        for precision in [8u8, 12] {
            let samples = samples(precision, &mut rnd);
            for t in &tables {
                let mut want = vec![[0i32; 64]; bw];
                fdct_quantize(Level::Scalar, &samples, stride, precision, t, &mut want);
                for level in Level::supported() {
                    for n in [bw, bw - 1, 1] {
                        let mut got = vec![[0i32; 64]; n];
                        fdct_quantize(level, &samples, stride, precision, t, &mut got);
                        assert_eq!(got, want[..n], "{precision}-bit {level:?} {n} blocks");
                    }
                }
            }
        }
    }

    #[test]
    fn float_quotient_is_exact() {
        // Exhaustive over the 12-bit coefficient range (8-bit is within it)
        // for a few divisors.
        for q in [1u16, 2, 3, 7, 10, 99, 255, 1000, 65535] {
            let d = Divisor::new(q as u32 * 8);
            let df = (q as u32 * 8) as f32;
            for c in -(1i32 << 18)..(1 << 18) {
                let a = c.unsigned_abs() + d.half;
                let v = (a as f32 / df) as i32;
                assert_eq!(if c < 0 { -v } else { v }, d.quantize(c), "q {q} c {c}");
            }
        }
    }
}
