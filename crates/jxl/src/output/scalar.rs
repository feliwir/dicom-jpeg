//! Portable float to integer conversion.

pub(super) fn to_u16(src: &[f32], scale: f32, top: u16, dst: &mut [u16]) {
    for (d, &v) in dst.iter_mut().zip(src) {
        *d = super::convert(v, scale, top);
    }
}
