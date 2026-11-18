//! Portable Gaborish row.

pub(super) fn row(up: &[f32], mid: &[f32], down: &[f32], dst: &mut [f32], k: [f32; 3]) {
    let [w0, w1, norm] = k;
    let (up, mid, down) = (
        &up[..dst.len() + 2],
        &mid[..dst.len() + 2],
        &down[..dst.len() + 2],
    );
    for (x, d) in dst.iter_mut().enumerate() {
        let side = mid[x] + mid[x + 2] + up[x + 1] + down[x + 1];
        let diag = up[x] + up[x + 2] + down[x] + down[x + 2];
        *d = (mid[x + 1] + side * w0 + diag * w1) * norm;
    }
}
