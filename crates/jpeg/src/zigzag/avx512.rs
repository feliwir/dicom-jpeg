//! AVX-512 zig-zag reorder: narrow the block to 64 i16 in two vectors and
//! permute them with two `vpermt2w`; the mask comes from `vptestmw`.
#![allow(unsafe_code)]

use std::arch::x86_64::*;

use crate::decoder::ZIGZAG;

/// Permute indices of zig-zag positions 0..32 and 32..64 into the natural
/// order held as two vectors (indices 32..64 select the second one).
const IDX: [[i16; 32]; 2] = {
    let mut idx = [[0i16; 32]; 2];
    let mut k = 0;
    while k < 64 {
        idx[k / 32][k % 32] = ZIGZAG[k] as i16;
        k += 1;
    }
    idx
};

#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx2,fma")]
pub(super) fn zigzag(coefs: &[[i32; 64]], zz: &mut [[i16; 64]], nonzero: &mut [u64]) {
    // SAFETY: IDX holds two rows of 32 i16.
    let (i0, i1) = unsafe {
        (
            _mm512_loadu_si512(IDX[0].as_ptr().cast()),
            _mm512_loadu_si512(IDX[1].as_ptr().cast()),
        )
    };
    for ((c, z), m) in coefs.iter().zip(zz.iter_mut()).zip(nonzero.iter_mut()) {
        // SAFETY: `c` holds 64 i32 and `z` 64 i16; unaligned loads and
        // stores are allowed.
        unsafe {
            let p = c.as_ptr();
            // Natural order 0..32 and 32..64, narrowed (values fit in i16).
            let q: [__m256i; 4] = std::array::from_fn(|i| {
                _mm512_cvtepi32_epi16(_mm512_loadu_si512(p.add(16 * i).cast()))
            });
            let lo = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(q[0]), q[1]);
            let hi = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(q[2]), q[3]);
            let z0 = _mm512_permutex2var_epi16(lo, i0, hi);
            let z1 = _mm512_permutex2var_epi16(lo, i1, hi);
            _mm512_storeu_si512(z.as_mut_ptr().cast(), z0);
            _mm512_storeu_si512(z.as_mut_ptr().add(32).cast(), z1);
            *m = _mm512_test_epi16_mask(z0, z0) as u64
                | (_mm512_test_epi16_mask(z1, z1) as u64) << 32;
        }
    }
}
