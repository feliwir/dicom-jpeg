//! Accurate integer inverse DCT.
//!
//! This is the "ISLOW" algorithm of the IJG library (jidctint.c, Loeffler,
//! Ligtenberg & Moschytz), reproduced bit-exactly so decoded output matches
//! libjpeg 6b / DCMTK / libjpeg-turbo. Intermediate values use `i64`, so
//! corrupt coefficients cannot overflow.

const CONST_BITS: u32 = 13;

const FIX_0_298631336: i64 = 2446;
const FIX_0_390180644: i64 = 3196;
const FIX_0_541196100: i64 = 4433;
const FIX_0_765366865: i64 = 6270;
const FIX_0_899976223: i64 = 7373;
const FIX_1_175875602: i64 = 9633;
const FIX_1_501321110: i64 = 12299;
const FIX_1_847759065: i64 = 15137;
const FIX_1_961570560: i64 = 16069;
const FIX_2_053119869: i64 = 16819;
const FIX_2_562915447: i64 = 20995;
const FIX_3_072711026: i64 = 25172;

#[inline(always)]
fn descale(x: i64, n: u32) -> i64 {
    (x + (1 << (n - 1))) >> n
}

/// One 8-point 1-D IDCT (shared by both passes). Returns the eight outputs
/// before the final descaling: (tmp10 ± tmp3, tmp11 ± tmp2, tmp12 ± tmp1, tmp13 ± tmp0).
#[inline(always)]
fn idct_1d(i: [i64; 8], dc_shift: u32) -> [i64; 8] {
    // Even part
    let z2 = i[2];
    let z3 = i[6];
    let z1 = (z2 + z3) * FIX_0_541196100;
    let tmp2 = z1 + z3 * -FIX_1_847759065;
    let tmp3 = z1 + z2 * FIX_0_765366865;
    let tmp0 = (i[0] + i[4]) << dc_shift;
    let tmp1 = (i[0] - i[4]) << dc_shift;
    let tmp10 = tmp0 + tmp3;
    let tmp13 = tmp0 - tmp3;
    let tmp11 = tmp1 + tmp2;
    let tmp12 = tmp1 - tmp2;

    // Odd part
    let (mut t0, mut t1, mut t2, mut t3) = (i[7], i[5], i[3], i[1]);
    let z1 = t0 + t3;
    let z2 = t1 + t2;
    let z3 = t0 + t2;
    let z4 = t1 + t3;
    let z5 = (z3 + z4) * FIX_1_175875602;
    t0 *= FIX_0_298631336;
    t1 *= FIX_2_053119869;
    t2 *= FIX_3_072711026;
    t3 *= FIX_1_501321110;
    let z1 = z1 * -FIX_0_899976223;
    let z2 = z2 * -FIX_2_562915447;
    let z3 = z3 * -FIX_1_961570560 + z5;
    let z4 = z4 * -FIX_0_390180644 + z5;
    t0 += z1 + z3;
    t1 += z2 + z4;
    t2 += z2 + z3;
    t3 += z1 + z4;

    [
        tmp10 + t3,
        tmp11 + t2,
        tmp12 + t1,
        tmp13 + t0,
        tmp13 - t0,
        tmp12 - t1,
        tmp11 - t2,
        tmp10 - t3,
    ]
}

/// Inverse DCT of one block of dequantized coefficients (natural order),
/// writing level-shifted, clamped samples of `precision` (8 or 12) bits.
pub(crate) fn idct_block(coef: &[i32; 64], precision: u8, out: &mut [u16], stride: usize) {
    // libjpeg uses PASS1_BITS = 2 for 8-bit and 1 for 12-bit samples.
    let pass1_bits: u32 = if precision > 8 { 1 } else { 2 };
    let max = (1i64 << precision) - 1;
    let center = 1i64 << (precision - 1);
    let mut ws = [0i64; 64];

    // Pass 1: columns.
    for c in 0..8 {
        let col: [i64; 8] = std::array::from_fn(|r| coef[r * 8 + c] as i64);
        if col[1..].iter().all(|&v| v == 0) {
            let dc = col[0] << pass1_bits;
            for r in 0..8 {
                ws[r * 8 + c] = dc;
            }
            continue;
        }
        let o = idct_1d(col, CONST_BITS);
        for r in 0..8 {
            ws[r * 8 + c] = descale(o[r], CONST_BITS - pass1_bits);
        }
    }

    // Pass 2: rows.
    let shift = CONST_BITS + pass1_bits + 3;
    for r in 0..8 {
        let row: [i64; 8] = std::array::from_fn(|c| ws[r * 8 + c]);
        let dst = &mut out[r * stride..r * stride + 8];
        if row[1..].iter().all(|&v| v == 0) {
            let v = (descale(row[0], pass1_bits + 3) + center).clamp(0, max) as u16;
            dst.fill(v);
            continue;
        }
        let o = idct_1d(row, CONST_BITS);
        for c in 0..8 {
            dst[c] = (descale(o[c], shift) + center).clamp(0, max) as u16;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference IDCT in double precision (T.81 A.3.3).
    fn reference(coef: &[i32; 64]) -> [f64; 64] {
        let mut out = [0f64; 64];
        for y in 0..8 {
            for x in 0..8 {
                let mut s = 0.0;
                for v in 0..8 {
                    for u in 0..8 {
                        let cu = if u == 0 {
                            std::f64::consts::FRAC_1_SQRT_2
                        } else {
                            1.0
                        };
                        let cv = if v == 0 {
                            std::f64::consts::FRAC_1_SQRT_2
                        } else {
                            1.0
                        };
                        s += cu
                            * cv
                            * coef[v * 8 + u] as f64
                            * (((2 * x + 1) as f64 * u as f64 * std::f64::consts::PI) / 16.0).cos()
                            * (((2 * y + 1) as f64 * v as f64 * std::f64::consts::PI) / 16.0).cos();
                    }
                }
                out[y * 8 + x] = s / 4.0;
            }
        }
        out
    }

    /// IEEE 1180-style accuracy check: pseudo-random blocks, peak error <= 1.
    #[test]
    fn accuracy_against_float_reference() {
        let mut seed: u32 = 12345;
        let mut rnd = |range: i32| {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            ((seed >> 8) as i32 % (2 * range + 1)) - range
        };
        for precision in [8u8, 12] {
            let range = if precision == 8 { 256 } else { 2048 };
            let max = (1i64 << precision) - 1;
            for _ in 0..2000 {
                let coef: [i32; 64] = std::array::from_fn(|i| if i < 20 { rnd(range) } else { 0 });
                let mut out = [0u16; 64];
                idct_block(&coef, precision, &mut out, 8);
                let r = reference(&coef);
                for i in 0..64 {
                    let expected = (r[i].round() as i64 + (1 << (precision - 1))).clamp(0, max);
                    assert!(
                        (out[i] as i64 - expected).abs() <= 1,
                        "precision {precision}: {} vs {expected}",
                        out[i]
                    );
                }
            }
        }
    }

    #[test]
    fn dc_only_block() {
        let mut coef = [0i32; 64];
        coef[0] = 80; // DC of 80 -> +10 after /8
        let mut out = [0u16; 64];
        idct_block(&coef, 8, &mut out, 8);
        assert!(out.iter().all(|&v| v == 138));
    }
}
