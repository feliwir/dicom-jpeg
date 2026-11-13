//! Accurate integer forward DCT ("ISLOW", IJG jfdctint.c).
//!
//! Output coefficients are scaled up by 8 relative to T.81 A.3.3, as in libjpeg;
//! the quantizer divides by `8 * q`.

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

/// 1-D forward DCT; `even_shift` is applied to outputs 0/4 (as a left shift if
/// `even_left`, otherwise as descale), the odd/rotated outputs are descaled by `odd_descale`.
#[inline(always)]
fn fdct_1d(d: [i64; 8], even: impl Fn(i64) -> i64, odd_descale: u32) -> [i64; 8] {
    let tmp0 = d[0] + d[7];
    let tmp7 = d[0] - d[7];
    let tmp1 = d[1] + d[6];
    let tmp6 = d[1] - d[6];
    let tmp2 = d[2] + d[5];
    let tmp5 = d[2] - d[5];
    let tmp3 = d[3] + d[4];
    let tmp4 = d[3] - d[4];

    let tmp10 = tmp0 + tmp3;
    let tmp13 = tmp0 - tmp3;
    let tmp11 = tmp1 + tmp2;
    let tmp12 = tmp1 - tmp2;

    let mut o = [0i64; 8];
    o[0] = even(tmp10 + tmp11);
    o[4] = even(tmp10 - tmp11);
    let z1 = (tmp12 + tmp13) * FIX_0_541196100;
    o[2] = descale(z1 + tmp13 * FIX_0_765366865, odd_descale);
    o[6] = descale(z1 + tmp12 * -FIX_1_847759065, odd_descale);

    let z1 = tmp4 + tmp7;
    let z2 = tmp5 + tmp6;
    let z3 = tmp4 + tmp6;
    let z4 = tmp5 + tmp7;
    let z5 = (z3 + z4) * FIX_1_175875602;
    let t4 = tmp4 * FIX_0_298631336;
    let t5 = tmp5 * FIX_2_053119869;
    let t6 = tmp6 * FIX_3_072711026;
    let t7 = tmp7 * FIX_1_501321110;
    let z1 = z1 * -FIX_0_899976223;
    let z2 = z2 * -FIX_2_562915447;
    let z3 = z3 * -FIX_1_961570560 + z5;
    let z4 = z4 * -FIX_0_390180644 + z5;
    o[7] = descale(t4 + z1 + z3, odd_descale);
    o[5] = descale(t5 + z2 + z4, odd_descale);
    o[3] = descale(t6 + z2 + z3, odd_descale);
    o[1] = descale(t7 + z1 + z4, odd_descale);
    o
}

/// Forward DCT of the 8x8 block at `src` (row stride `stride`) of unsigned
/// samples with `precision` bits; output in natural order, scaled by 8.
pub(crate) fn fdct_block(src: &[u16], stride: usize, precision: u8, out: &mut [i32; 64]) {
    let pass1_bits: u32 = if precision > 8 { 1 } else { 2 };
    let center = 1i64 << (precision - 1);
    let mut ws = [0i64; 64];
    for r in 0..8 {
        let row: [i64; 8] = std::array::from_fn(|c| src[r * stride + c] as i64 - center);
        let o = fdct_1d(row, |v| v << pass1_bits, CONST_BITS - pass1_bits);
        ws[r * 8..r * 8 + 8].copy_from_slice(&o);
    }
    for c in 0..8 {
        let col: [i64; 8] = std::array::from_fn(|r| ws[r * 8 + c]);
        let o = fdct_1d(col, |v| descale(v, pass1_bits), CONST_BITS + pass1_bits);
        for r in 0..8 {
            out[r * 8 + c] = o[r] as i32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_block_has_only_dc() {
        let src = [200u16; 64];
        let mut out = [0i32; 64];
        fdct_block(&src, 8, 8, &mut out);
        // (200 - 128) * 64 = 4608 = 8 * DC(576)
        assert_eq!(out[0], 4608);
        assert!(out[1..].iter().all(|&v| v == 0));
    }

    #[test]
    fn fdct_idct_roundtrip_with_unit_quantization() {
        let mut seed = 7u32;
        for precision in [8u8, 12] {
            let max = (1u32 << precision) - 1;
            for _ in 0..200 {
                let src: [u16; 64] = std::array::from_fn(|_| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    ((seed >> 8) % (max + 1)) as u16
                });
                let mut coef = [0i32; 64];
                fdct_block(&src, 8, precision, &mut coef);
                // Quantize with q = 1 (divide by 8, rounding) and transform back.
                let deq: [i32; 64] = std::array::from_fn(|i| {
                    let c = coef[i];
                    let v = (c.abs() + 4) / 8;
                    if c < 0 { -v } else { v }
                });
                let mut back = [0u16; 64];
                super::super::idct::idct_block(&deq, precision, &mut back, 8);
                for i in 0..64 {
                    assert!(
                        (back[i] as i32 - src[i] as i32).abs() <= 1,
                        "{precision}: {} vs {}",
                        back[i],
                        src[i]
                    );
                }
            }
        }
    }
}
