//! NEON zig-zag reorder: the table gather, with the mask from vector
//! compares (each lane's bit weight, summed across the vector).
#![allow(unsafe_code)]

use std::arch::aarch64::*;

use super::gather;

#[target_feature(enable = "neon")]
pub(super) fn zigzag(coefs: &[[i32; 64]], zz: &mut [[i16; 64]], nonzero: &mut [u64]) {
    const WEIGHTS: [u16; 8] = [1, 2, 4, 8, 16, 32, 64, 128];
    // SAFETY: WEIGHTS holds 8 u16.
    let w = unsafe { vld1q_u16(WEIGHTS.as_ptr()) };
    for ((c, z), m) in coefs.iter().zip(zz.iter_mut()).zip(nonzero.iter_mut()) {
        gather(c, z);
        let mut bits = 0u64;
        for i in 0..8 {
            // SAFETY: `z` holds 64 i16; 8 are read from 8 * i.
            let v = unsafe { vld1q_s16(z.as_ptr().add(8 * i)) };
            // All ones where non-zero, then one bit per lane.
            let nz = vtstq_s16(v, v);
            bits |= (vaddvq_u16(vandq_u16(nz, w)) as u64) << (8 * i);
        }
        *m = bits;
    }
}
