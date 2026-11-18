//! SSE4.1 Gaborish rows: 4 lanes.
#![allow(unsafe_code)]

use std::arch::x86_64::*;

/// Lanes per vector.
const L: usize = 4;

/// `scalar::row`, L outputs at a time.
#[target_feature(enable = "sse4.1")]
pub(super) fn row(up: &[f32], mid: &[f32], down: &[f32], dst: &mut [f32], k: [f32; 3]) {
    let n = dst.len() / L * L;
    assert!(up.len() >= n + 2 && mid.len() >= n + 2 && down.len() >= n + 2);
    let (w0, w1, norm) = (_mm_set1_ps(k[0]), _mm_set1_ps(k[1]), _mm_set1_ps(k[2]));
    for x in (0..n).step_by(L) {
        // SAFETY: x + 2 + L <= n + 2 <= the length of each input row
        // (asserted), and x + L <= n <= dst.len().
        unsafe {
            let (u, m, d) = (
                up.as_ptr().add(x),
                mid.as_ptr().add(x),
                down.as_ptr().add(x),
            );
            let side = _mm_add_ps(
                _mm_add_ps(
                    _mm_add_ps(_mm_loadu_ps(m), _mm_loadu_ps(m.add(2))),
                    _mm_loadu_ps(u.add(1)),
                ),
                _mm_loadu_ps(d.add(1)),
            );
            let diag = _mm_add_ps(
                _mm_add_ps(
                    _mm_add_ps(_mm_loadu_ps(u), _mm_loadu_ps(u.add(2))),
                    _mm_loadu_ps(d),
                ),
                _mm_loadu_ps(d.add(2)),
            );
            let v = _mm_add_ps(
                _mm_add_ps(_mm_loadu_ps(m.add(1)), _mm_mul_ps(side, w0)),
                _mm_mul_ps(diag, w1),
            );
            _mm_storeu_ps(dst.as_mut_ptr().add(x), _mm_mul_ps(v, norm));
        }
    }
    super::scalar::row(&up[n..], &mid[n..], &down[n..], &mut dst[n..], k);
}
