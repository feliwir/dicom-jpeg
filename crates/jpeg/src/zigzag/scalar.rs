//! Portable zig-zag reorder.

use super::{gather, mask_of};

pub(super) fn zigzag(coefs: &[[i32; 64]], zz: &mut [[i16; 64]], nonzero: &mut [u64]) {
    for ((c, z), m) in coefs.iter().zip(zz.iter_mut()).zip(nonzero.iter_mut()) {
        gather(c, z);
        *m = mask_of(z);
    }
}
