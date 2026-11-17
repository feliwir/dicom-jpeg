//! Portable downsampling.

pub(super) fn h2v1(row: &[u16], out: &mut [u16]) {
    for (j, (o, p)) in out.iter_mut().zip(row.as_chunks::<2>().0).enumerate() {
        *o = ((p[0] as u32 + p[1] as u32 + (j % 2) as u32) >> 1) as u16;
    }
}

pub(super) fn h2v2(a: &[u16], b: &[u16], out: &mut [u16]) {
    let pairs = a.as_chunks::<2>().0.iter().zip(b.as_chunks::<2>().0);
    for (j, (o, (a, b))) in out.iter_mut().zip(pairs).enumerate() {
        let s = a[0] as u32 + a[1] as u32 + b[0] as u32 + b[1] as u32;
        *o = ((s + 1 + (j % 2) as u32) >> 2) as u16;
    }
}
