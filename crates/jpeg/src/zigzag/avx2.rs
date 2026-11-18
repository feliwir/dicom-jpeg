//! AVX2 zig-zag reorder: the table gather, with the mask from vector
//! compares (AVX2 has no word permute across a whole vector).
#![allow(unsafe_code)]

use std::arch::x86_64::*;

use super::gather;

#[target_feature(enable = "avx2,fma")]
pub(super) fn zigzag(coefs: &[[i32; 64]], zz: &mut [[i16; 64]], nonzero: &mut [u64]) {
    let zero = _mm256_setzero_si256();
    for ((c, z), m) in coefs.iter().zip(zz.iter_mut()).zip(nonzero.iter_mut()) {
        gather(c, z);
        // SAFETY: `z` holds 64 i16 (four vectors of 16).
        let v: [__m256i; 4] =
            std::array::from_fn(|i| unsafe { _mm256_loadu_si256(z.as_ptr().add(16 * i).cast()) });
        // Bytes 0xFF where a coefficient is zero; packing interleaves the
        // 128-bit lanes, which the permute undoes.
        let pack = |a, b| {
            let p = _mm256_packs_epi16(_mm256_cmpeq_epi16(a, zero), _mm256_cmpeq_epi16(b, zero));
            _mm256_permute4x64_epi64::<0b11_01_10_00>(p)
        };
        let lo = _mm256_movemask_epi8(pack(v[0], v[1])) as u32 as u64;
        let hi = _mm256_movemask_epi8(pack(v[2], v[3])) as u32 as u64;
        *m = !(lo | hi << 32);
    }
}
