//! Portable fancy upsampling.

pub(super) fn h2v1(src: &[u16], dst: &mut [u16]) {
    h2v1_edges(src, dst);
    h2v1_inner(src, &mut dst[2..2 * src.len() - 2]);
}

/// The first and last two outputs.
pub(super) fn h2v1_edges(src: &[u16], dst: &mut [u16]) {
    let w = src.len();
    let (first, last) = (src[0], src[w - 1]);
    dst[0] = first;
    dst[1] = ((first as u32 * 3 + src[1] as u32 + 2) >> 2) as u16;
    dst[2 * w - 2] = ((last as u32 * 3 + src[w - 2] as u32 + 1) >> 2) as u16;
    dst[2 * w - 1] = last;
}

/// Outputs 2i and 2i+1 for the inputs i with both neighbours: `dst` holds
/// `2 * (src.len() - 2)` samples, starting at output 2.
pub(super) fn h2v1_inner(src: &[u16], dst: &mut [u16]) {
    for (d, s) in dst.as_chunks_mut::<2>().0.iter_mut().zip(src.windows(3)) {
        let c = s[1] as u32 * 3;
        d[0] = ((c + s[0] as u32 + 1) >> 2) as u16;
        d[1] = ((c + s[2] as u32 + 2) >> 2) as u16;
    }
}

pub(super) fn h2v2(near: &[u16], far: &[u16], dst: &mut [u16]) {
    h2v2_edges(near, far, dst);
    h2v2_inner(near, far, &mut dst[2..2 * near.len() - 2]);
}

/// The first and last two outputs.
pub(super) fn h2v2_edges(near: &[u16], far: &[u16], dst: &mut [u16]) {
    let w = near.len();
    let cs = |i: usize| near[i] as u32 * 3 + far[i] as u32;
    dst[0] = ((cs(0) * 4 + 8) >> 4) as u16;
    dst[1] = ((cs(0) * 3 + cs(1) + 7) >> 4) as u16;
    dst[2 * w - 2] = ((cs(w - 1) * 3 + cs(w - 2) + 8) >> 4) as u16;
    dst[2 * w - 1] = ((cs(w - 1) * 4 + 7) >> 4) as u16;
}

/// As `h2v1_inner` for h2v2.
pub(super) fn h2v2_inner(near: &[u16], far: &[u16], dst: &mut [u16]) {
    let cs = |i: usize| near[i] as u32 * 3 + far[i] as u32;
    for (i, d) in dst.as_chunks_mut::<2>().0.iter_mut().enumerate() {
        let c = cs(i + 1) * 3;
        d[0] = ((c + cs(i) + 8) >> 4) as u16;
        d[1] = ((c + cs(i + 2) + 7) >> 4) as u16;
    }
}
