//! AVX2 XYB to linear RGB: 8 pixels at a time (FMA is not used, to match the scalar rounding).
#![allow(unsafe_code)]

use std::arch::x86_64::*;

use super::Xyb;

/// Lanes per vector.
const L: usize = 8;

/// `scalar::to_linear`, L pixels at a time.
#[target_feature(enable = "avx2,fma")]
pub(super) fn to_linear(planes: [&mut [f32]; 3], k: &Xyb) {
    let [p0, p1, p2] = planes;
    let n = p0.len().min(p1.len()).min(p2.len()) / L * L;
    let cb = k.cbrt_bias.map(|v| _mm256_set1_ps(v));
    let bias = k.bias.map(|v| _mm256_set1_ps(v));
    let it = _mm256_set1_ps(k.itscale);
    let m = k.matrix.map(|v| _mm256_set1_ps(v));
    for i in (0..n).step_by(L) {
        // SAFETY: i + L <= n <= the length of each plane.
        unsafe {
            let (x, y, b) = (
                _mm256_loadu_ps(p0.as_ptr().add(i)),
                _mm256_loadu_ps(p1.as_ptr().add(i)),
                _mm256_loadu_ps(p2.as_ptr().add(i)),
            );
            let l = _mm256_sub_ps(_mm256_add_ps(y, x), cb[0]);
            let mm = _mm256_sub_ps(_mm256_sub_ps(y, x), cb[1]);
            let s = _mm256_sub_ps(b, cb[2]);
            let mixed = [
                _mm256_mul_ps(
                    _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(l, l), l), bias[0]),
                    it,
                ),
                _mm256_mul_ps(
                    _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(mm, mm), mm), bias[1]),
                    it,
                ),
                _mm256_mul_ps(
                    _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(s, s), s), bias[2]),
                    it,
                ),
            ];
            let out = [p0.as_mut_ptr(), p1.as_mut_ptr(), p2.as_mut_ptr()];
            for (c, o) in out.into_iter().enumerate() {
                let v = _mm256_add_ps(
                    _mm256_add_ps(
                        _mm256_mul_ps(m[c * 3], mixed[0]),
                        _mm256_mul_ps(m[c * 3 + 1], mixed[1]),
                    ),
                    _mm256_mul_ps(m[c * 3 + 2], mixed[2]),
                );
                _mm256_storeu_ps(o.add(i), v);
            }
        }
    }
    super::scalar::to_linear([&mut p0[n..], &mut p1[n..], &mut p2[n..]], k);
}
