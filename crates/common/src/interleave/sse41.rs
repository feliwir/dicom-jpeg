//! SSE4.1 (SSSE3 `pshufb`) interleaving: 16 pixels of bytes or 8 pixels of
//! 16-bit samples (48 bytes) per step, merged from three byte shuffles.
#![allow(unsafe_code)]

use std::arch::x86_64::*;

/// Byte shuffle placing byte channel `chan` into chunk `chunk` (16 bytes)
/// of 3-channel interleaved bytes; -128 clears a byte.
const fn mask8(chunk: usize, chan: usize) -> [i8; 16] {
    let mut m = [-128i8; 16];
    let mut j = 0;
    while j < 16 {
        let p = chunk * 16 + j;
        if p % 3 == chan {
            m[j] = (p / 3) as i8;
        }
        j += 1;
    }
    m
}

/// As `mask8` for 16-bit samples.
const fn mask16(chunk: usize, chan: usize) -> [i8; 16] {
    let mut m = [-128i8; 16];
    let mut j = 0;
    while j < 16 {
        let bp = chunk * 16 + j;
        let e = bp / 2;
        if e % 3 == chan {
            m[j] = (2 * (e / 3) + bp % 2) as i8;
        }
        j += 1;
    }
    m
}

/// Byte shuffle gathering channel `chan` of 16 byte pixels from chunk
/// `chunk` of the interleaved data.
const fn unmask8(chunk: usize, chan: usize) -> [i8; 16] {
    let mut m = [-128i8; 16];
    let mut i = 0;
    while i < 16 {
        let p = 3 * i + chan;
        if p / 16 == chunk {
            m[i] = (p % 16) as i8;
        }
        i += 1;
    }
    m
}

/// As `unmask8` for 8 pixels of 16-bit samples.
const fn unmask16(chunk: usize, chan: usize) -> [i8; 16] {
    let mut m = [-128i8; 16];
    let mut j = 0;
    while j < 16 {
        let bp = 2 * (3 * (j / 2) + chan) + j % 2;
        if bp / 16 == chunk {
            m[j] = (bp % 16) as i8;
        }
        j += 1;
    }
    m
}

/// The shuffles of `$f` for every (chunk, channel).
macro_rules! masks {
    ($f:ident) => {
        [
            [$f(0, 0), $f(0, 1), $f(0, 2)],
            [$f(1, 0), $f(1, 1), $f(1, 2)],
            [$f(2, 0), $f(2, 1), $f(2, 2)],
        ]
    };
}

static M8: [[[i8; 16]; 3]; 3] = masks!(mask8);
static M16: [[[i8; 16]; 3]; 3] = masks!(mask16);
static U8: [[[i8; 16]; 3]; 3] = masks!(unmask8);
static U16: [[[i8; 16]; 3]; 3] = masks!(unmask16);

#[inline]
#[target_feature(enable = "sse4.1")]
fn load_masks(m: &[[[i8; 16]; 3]; 3]) -> [[__m128i; 3]; 3] {
    // SAFETY: each mask is 16 bytes.
    m.map(|c| c.map(|v| unsafe { _mm_loadu_si128(v.as_ptr().cast()) }))
}

/// Interleave three vectors of 16 bytes (or 8 u16) into 48 bytes.
#[inline]
#[target_feature(enable = "sse4.1")]
fn merge(m: &[[__m128i; 3]; 3], x: [__m128i; 3]) -> [__m128i; 3] {
    m.map(|c| {
        _mm_or_si128(
            _mm_or_si128(_mm_shuffle_epi8(x[0], c[0]), _mm_shuffle_epi8(x[1], c[1])),
            _mm_shuffle_epi8(x[2], c[2]),
        )
    })
}

/// Split 48 interleaved bytes into three vectors of 16 bytes (or 8 u16).
#[inline]
#[target_feature(enable = "sse4.1")]
fn split(m: &[[__m128i; 3]; 3], x: [__m128i; 3]) -> [__m128i; 3] {
    [0, 1, 2].map(|k| {
        _mm_or_si128(
            _mm_or_si128(
                _mm_shuffle_epi8(x[0], m[0][k]),
                _mm_shuffle_epi8(x[1], m[1][k]),
            ),
            _mm_shuffle_epi8(x[2], m[2][k]),
        )
    })
}

#[target_feature(enable = "sse4.1")]
pub(super) fn interleave3_u8(a: &[u16], b: &[u16], c: &[u16], out: &mut [u8]) {
    let m = load_masks(&M8);
    let low = _mm_set1_epi16(0xFF);
    let n = out.len() / 3 / 16 * 16;
    let (a16, b16, c16) = (
        a[..n].as_chunks::<16>().0,
        b[..n].as_chunks::<16>().0,
        c[..n].as_chunks::<16>().0,
    );
    for (o, ((x, y), z)) in out[..3 * n]
        .as_chunks_mut::<48>()
        .0
        .iter_mut()
        .zip(a16.iter().zip(b16).zip(c16))
    {
        let bytes = |p: &[u16; 16]| {
            // SAFETY: `p` holds 16 u16 (32 bytes).
            let (lo, hi) = unsafe {
                (
                    _mm_loadu_si128(p.as_ptr().cast()),
                    _mm_loadu_si128(p[8..].as_ptr().cast()),
                )
            };
            // Truncate to 8 bits, then pack (no saturation left).
            _mm_packus_epi16(_mm_and_si128(lo, low), _mm_and_si128(hi, low))
        };
        let v = merge(&m, [bytes(x), bytes(y), bytes(z)]);
        for (k, v) in v.into_iter().enumerate() {
            // SAFETY: `o` holds 48 bytes.
            unsafe { _mm_storeu_si128(o[16 * k..].as_mut_ptr().cast(), v) };
        }
    }
    super::scalar::interleave3_u8(&a[n..], &b[n..], &c[n..], &mut out[3 * n..]);
}

#[target_feature(enable = "sse4.1")]
pub(super) fn interleave3_u16(a: &[u16], b: &[u16], c: &[u16], out: &mut [u16]) {
    let m = load_masks(&M16);
    let n = out.len() / 3 / 8 * 8;
    let (a8, b8, c8) = (
        a[..n].as_chunks::<8>().0,
        b[..n].as_chunks::<8>().0,
        c[..n].as_chunks::<8>().0,
    );
    for (o, ((x, y), z)) in out[..3 * n]
        .as_chunks_mut::<24>()
        .0
        .iter_mut()
        .zip(a8.iter().zip(b8).zip(c8))
    {
        // SAFETY: each input holds 8 u16 (16 bytes).
        let v = unsafe {
            [
                _mm_loadu_si128(x.as_ptr().cast()),
                _mm_loadu_si128(y.as_ptr().cast()),
                _mm_loadu_si128(z.as_ptr().cast()),
            ]
        };
        let v = merge(&m, v);
        for (k, v) in v.into_iter().enumerate() {
            // SAFETY: `o` holds 24 u16 (48 bytes).
            unsafe { _mm_storeu_si128(o[8 * k..].as_mut_ptr().cast(), v) };
        }
    }
    super::scalar::interleave3_u16(&a[n..], &b[n..], &c[n..], &mut out[3 * n..]);
}

#[target_feature(enable = "sse4.1")]
pub(super) fn deinterleave3_u8(src: &[u8], a: &mut [u16], b: &mut [u16], c: &mut [u16]) {
    let m = load_masks(&U8);
    let n = src.len() / 3 / 16 * 16;
    let (a16, b16, c16) = (
        a[..n].as_chunks_mut::<16>().0,
        b[..n].as_chunks_mut::<16>().0,
        c[..n].as_chunks_mut::<16>().0,
    );
    for (s, ((x, y), z)) in src[..3 * n]
        .as_chunks::<48>()
        .0
        .iter()
        .zip(a16.iter_mut().zip(b16).zip(c16))
    {
        // SAFETY: `s` holds 48 bytes.
        let v = unsafe {
            [
                _mm_loadu_si128(s.as_ptr().cast()),
                _mm_loadu_si128(s[16..].as_ptr().cast()),
                _mm_loadu_si128(s[32..].as_ptr().cast()),
            ]
        };
        for (d, v) in [x, y, z].into_iter().zip(split(&m, v)) {
            let (lo, hi) = (
                _mm_cvtepu8_epi16(v),
                _mm_cvtepu8_epi16(_mm_srli_si128::<8>(v)),
            );
            // SAFETY: `d` holds 16 u16 (32 bytes).
            unsafe {
                _mm_storeu_si128(d.as_mut_ptr().cast(), lo);
                _mm_storeu_si128(d[8..].as_mut_ptr().cast(), hi);
            }
        }
    }
    super::scalar::deinterleave3_u8(&src[3 * n..], &mut a[n..], &mut b[n..], &mut c[n..]);
}

#[target_feature(enable = "sse4.1")]
pub(super) fn deinterleave3_u16(src: &[u16], a: &mut [u16], b: &mut [u16], c: &mut [u16]) {
    let m = load_masks(&U16);
    let n = src.len() / 3 / 8 * 8;
    let (a8, b8, c8) = (
        a[..n].as_chunks_mut::<8>().0,
        b[..n].as_chunks_mut::<8>().0,
        c[..n].as_chunks_mut::<8>().0,
    );
    for (s, ((x, y), z)) in src[..3 * n]
        .as_chunks::<24>()
        .0
        .iter()
        .zip(a8.iter_mut().zip(b8).zip(c8))
    {
        // SAFETY: `s` holds 24 u16 (48 bytes).
        let v = unsafe {
            [
                _mm_loadu_si128(s.as_ptr().cast()),
                _mm_loadu_si128(s[8..].as_ptr().cast()),
                _mm_loadu_si128(s[16..].as_ptr().cast()),
            ]
        };
        for (d, v) in [x, y, z].into_iter().zip(split(&m, v)) {
            // SAFETY: `d` holds 8 u16 (16 bytes).
            unsafe { _mm_storeu_si128(d.as_mut_ptr().cast(), v) };
        }
    }
    super::scalar::deinterleave3_u16(&src[3 * n..], &mut a[n..], &mut b[n..], &mut c[n..]);
}
