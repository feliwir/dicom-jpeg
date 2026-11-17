//! Portable interleaving.

pub(super) fn interleave3_u8(a: &[u16], b: &[u16], c: &[u16], out: &mut [u8]) {
    for (((o, &x), &y), &z) in out.as_chunks_mut::<3>().0.iter_mut().zip(a).zip(b).zip(c) {
        *o = [x as u8, y as u8, z as u8];
    }
}

pub(super) fn interleave3_u16(a: &[u16], b: &[u16], c: &[u16], out: &mut [u16]) {
    for (((o, &x), &y), &z) in out.as_chunks_mut::<3>().0.iter_mut().zip(a).zip(b).zip(c) {
        *o = [x, y, z];
    }
}

pub(super) fn deinterleave3_u8(src: &[u8], a: &mut [u16], b: &mut [u16], c: &mut [u16]) {
    for (((p, x), y), z) in src.as_chunks::<3>().0.iter().zip(a).zip(b).zip(c) {
        (*x, *y, *z) = (p[0] as u16, p[1] as u16, p[2] as u16);
    }
}

pub(super) fn deinterleave3_u16(src: &[u16], a: &mut [u16], b: &mut [u16], c: &mut [u16]) {
    for (((p, x), y), z) in src.as_chunks::<3>().0.iter().zip(a).zip(b).zip(c) {
        (*x, *y, *z) = (p[0], p[1], p[2]);
    }
}
