//! Portable XYB to linear RGB.

use super::Xyb;

pub(super) fn to_linear(planes: [&mut [f32]; 3], k: &Xyb) {
    let [p0, p1, p2] = planes;
    let m = &k.matrix;
    for ((x, y), b) in p0.iter_mut().zip(p1.iter_mut()).zip(p2.iter_mut()) {
        let l = *y + *x - k.cbrt_bias[0];
        let mm = *y - *x - k.cbrt_bias[1];
        let s = *b - k.cbrt_bias[2];
        let mixed = [
            (l * l * l + k.bias[0]) * k.itscale,
            (mm * mm * mm + k.bias[1]) * k.itscale,
            (s * s * s + k.bias[2]) * k.itscale,
        ];
        let rgb: [f32; 3] = std::array::from_fn(|c| {
            m[c * 3] * mixed[0] + m[c * 3 + 1] * mixed[1] + m[c * 3 + 2] * mixed[2]
        });
        (*x, *y, *b) = (rgb[0], rgb[1], rgb[2]);
    }
}
