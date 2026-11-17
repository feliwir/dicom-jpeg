//! Portable lifting kernels. Integer steps wrap like the SIMD versions (and
//! like release builds of the previous code), so corrupt data cannot panic.

pub(super) fn update53(cur: &mut [i32], a: &[i32], b: &[i32]) {
    for ((c, &a), &b) in cur.iter_mut().zip(a).zip(b) {
        *c = c.wrapping_sub(a.wrapping_add(b).wrapping_add(2) >> 2);
    }
}

pub(super) fn predict53(cur: &mut [i32], a: &[i32], b: &[i32]) {
    for ((c, &a), &b) in cur.iter_mut().zip(a).zip(b) {
        *c = c.wrapping_add(a.wrapping_add(b) >> 1);
    }
}

pub(super) fn lift97(cur: &mut [f32], a: &[f32], b: &[f32], k: f32) {
    for ((c, &a), &b) in cur.iter_mut().zip(a).zip(b) {
        *c -= k * (a + b);
    }
}

pub(super) fn scale97(cur: &mut [f32], k: f32) {
    for c in cur {
        *c *= k;
    }
}
