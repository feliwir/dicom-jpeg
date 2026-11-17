//! Scalar ISLOW IDCT (libjpeg jidctint.c), with libjpeg's shortcuts for
//! columns and rows whose AC terms are all zero.

use super::*;

/// Defines `$idct`, the IDCT in `$t` arithmetic: columns, then rows, each
/// with libjpeg's shortcut for all-zero AC terms. libjpeg uses PASS1_BITS = 2
/// for 8-bit and 1 for 12-bit samples.
macro_rules! idct_impl {
    ($idct:ident, $t:ty) => {
        fn $idct(
            coef: &[i32; 64],
            pass1_bits: u32,
            center: i32,
            max: i32,
            out: &mut [u16],
            stride: usize,
        ) {
            #[inline(always)]
            fn descale(x: $t, n: u32) -> $t {
                x.wrapping_add(1 << (n - 1)) >> n
            }

            /// One 8-point 1-D IDCT. Returns the eight outputs before the
            /// final descaling: (tmp10 ± tmp3, tmp11 ± tmp2, tmp12 ± tmp1,
            /// tmp13 ± tmp0).
            #[inline(always)]
            fn idct_1d(i: [$t; 8]) -> [$t; 8] {
                let mul = |a: $t, k: i32| a.wrapping_mul(k as $t);
                // Even part
                let z2 = i[2];
                let z3 = i[6];
                let z1 = mul(z2.wrapping_add(z3), FIX_0_541196100);
                let tmp2 = z1.wrapping_add(mul(z3, -FIX_1_847759065));
                let tmp3 = z1.wrapping_add(mul(z2, FIX_0_765366865));
                let tmp0 = i[0].wrapping_add(i[4]).wrapping_shl(CONST_BITS);
                let tmp1 = i[0].wrapping_sub(i[4]).wrapping_shl(CONST_BITS);
                let tmp10 = tmp0.wrapping_add(tmp3);
                let tmp13 = tmp0.wrapping_sub(tmp3);
                let tmp11 = tmp1.wrapping_add(tmp2);
                let tmp12 = tmp1.wrapping_sub(tmp2);

                // Odd part
                let (t0, t1, t2, t3) = (i[7], i[5], i[3], i[1]);
                let z1 = t0.wrapping_add(t3);
                let z2 = t1.wrapping_add(t2);
                let z3 = t0.wrapping_add(t2);
                let z4 = t1.wrapping_add(t3);
                let z5 = mul(z3.wrapping_add(z4), FIX_1_175875602);
                let t0 = mul(t0, FIX_0_298631336);
                let t1 = mul(t1, FIX_2_053119869);
                let t2 = mul(t2, FIX_3_072711026);
                let t3 = mul(t3, FIX_1_501321110);
                let z1 = mul(z1, -FIX_0_899976223);
                let z2 = mul(z2, -FIX_2_562915447);
                let z3 = mul(z3, -FIX_1_961570560).wrapping_add(z5);
                let z4 = mul(z4, -FIX_0_390180644).wrapping_add(z5);
                let t0 = t0.wrapping_add(z1).wrapping_add(z3);
                let t1 = t1.wrapping_add(z2).wrapping_add(z4);
                let t2 = t2.wrapping_add(z2).wrapping_add(z3);
                let t3 = t3.wrapping_add(z1).wrapping_add(z4);

                [
                    tmp10.wrapping_add(t3),
                    tmp11.wrapping_add(t2),
                    tmp12.wrapping_add(t1),
                    tmp13.wrapping_add(t0),
                    tmp13.wrapping_sub(t0),
                    tmp12.wrapping_sub(t1),
                    tmp11.wrapping_sub(t2),
                    tmp10.wrapping_sub(t3),
                ]
            }

            let mut ws: [$t; 64] = [0; 64];
            for c in 0..8 {
                let col: [$t; 8] = std::array::from_fn(|r| coef[r * 8 + c] as $t);
                if col[1..].iter().all(|&v| v == 0) {
                    let dc = col[0].wrapping_shl(pass1_bits);
                    for r in 0..8 {
                        ws[r * 8 + c] = dc;
                    }
                    continue;
                }
                let o = idct_1d(col);
                for r in 0..8 {
                    ws[r * 8 + c] = descale(o[r], CONST_BITS - pass1_bits);
                }
            }
            let shift = CONST_BITS + pass1_bits + 3;
            let level = |v: $t| (v as i32).wrapping_add(center).clamp(0, max) as u16;
            for r in 0..8 {
                let row: [$t; 8] = std::array::from_fn(|c| ws[r * 8 + c]);
                let dst = &mut out[r * stride..r * stride + 8];
                if row[1..].iter().all(|&v| v == 0) {
                    dst.fill(level(descale(row[0], pass1_bits + 3)));
                    continue;
                }
                let o = idct_1d(row);
                for (d, &v) in dst.iter_mut().zip(&o) {
                    *d = level(descale(v, shift));
                }
            }
        }
    };
}

idct_impl!(idct_i32, i32);
idct_impl!(idct_i64, i64);

/// Inverse DCT of one block of dequantized coefficients (natural order),
/// writing level-shifted, clamped samples of `precision` (8 or 12) bits.
pub(super) fn idct_block(coef: &[i32; 64], precision: u8, out: &mut [u16], stride: usize) {
    let max = (1i32 << precision) - 1;
    let center = 1i32 << (precision - 1);
    if precision > 8 {
        idct_i64(coef, 1, center, max, out, stride);
    } else {
        idct_i32(coef, 2, center, max, out, stride);
    }
}

/// 8-bit IDCT of each block into the matching 8 x 8 output block.
pub(super) fn idct8(coefs: &[[i32; 64]], out: &mut [[u16; 64]]) {
    for (c, o) in coefs.iter().zip(out.iter_mut()) {
        idct_i32(c, 2, 128, 255, o, 8);
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
