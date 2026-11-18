//! SSE4.1 zig-zag reorder: the table gather, with the mask from vector
//! compares.
#![allow(unsafe_code)]

use std::arch::x86_64::*;

use super::gather;

#[target_feature(enable = "sse4.1")]
pub(super) fn zigzag(coefs: &[[i32; 64]], zz: &mut [[i16; 64]], nonzero: &mut [u64]) {
    let zero = _mm_setzero_si128();
    for ((c, z), m) in coefs.iter().zip(zz.iter_mut()).zip(nonzero.iter_mut()) {
        gather(c, z);
        let mut zeros = 0u64;
        for i in 0..4 {
            // SAFETY: `z` holds 64 i16; 16 are read from 16 * i.
            let (a, b) = unsafe {
                let p = z.as_ptr().add(16 * i);
                (_mm_loadu_si128(p.cast()), _mm_loadu_si128(p.add(8).cast()))
            };
            // Bytes 0xFF where a coefficient is zero.
            let p = _mm_packs_epi16(_mm_cmpeq_epi16(a, zero), _mm_cmpeq_epi16(b, zero));
            zeros |= (_mm_movemask_epi8(p) as u16 as u64) << (16 * i);
        }
        *m = !zeros;
    }
}
