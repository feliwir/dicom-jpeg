//! Portable level shift: the reference (`*_exact`, in the decoder's
//! original form) and the clamp-first form of the SIMD kernels.

use dicom_jpeg_common::float::round_i32;

use super::Range;

pub(super) fn shift_int_exact(data: &mut [i32], r: Range) {
    for v in data {
        *v = v.saturating_add(r.shift).clamp(r.lo, r.hi);
    }
}

pub(super) fn shift_float_exact(src: &[f32], dst: &mut [i32], r: Range) {
    for (d, &v) in dst.iter_mut().zip(src) {
        *d = round_i32(v).saturating_add(r.shift).clamp(r.lo, r.hi);
    }
}

/// `clamp(v, a, b) + shift`, where `a + shift` and `b + shift` do not
/// overflow.
pub(super) fn shift_int(data: &mut [i32], a: i32, b: i32, shift: i32) {
    for v in data {
        *v = (*v).clamp(a, b) + shift;
    }
}

/// `round(clamp(v, a, b)) + shift` (NaN as 0), with `a` and `b` exact in
/// f32.
pub(super) fn shift_float(src: &[f32], dst: &mut [i32], a: i32, b: i32, shift: i32) {
    let (fa, fb) = (a as f32, b as f32);
    for (d, &v) in dst.iter_mut().zip(src) {
        let v = if v.is_nan() { 0.0 } else { v.clamp(fa, fb) };
        *d = round_i32(v) + shift;
    }
}
