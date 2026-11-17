//! JPEG-LS scan decoding and encoding (T.87 Annex A), for all interleave modes.
//!
//! Lines are processed in buffers of `width + 2` pixels (one edge pixel on each
//! side), each pixel holding `n` samples (`n` = 1 except for sample
//! interleaving). Edge handling, restart behaviour and multi-component run
//! interruption follow T.87 as implemented by the reference CharLS codec.

use super::CODEC;
use super::bitio::BitReader;
#[cfg(feature = "encode")]
use super::bitio::BitWriter;
use super::coding::{J, Params, Quantizer, State, context, predict};
use dicom_jpeg_common::{Error, Result};

/// Interleave mode of a scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Interleave {
    None = 0,
    Line = 1,
    Sample = 2,
}

/// Geometry of a scan; `planes[c]` holds component `c` of the scan.
#[derive(Debug)]
pub(crate) struct ScanLayout {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) interleave: Interleave,
    pub(crate) restart_interval: usize,
}

#[inline]
fn sign(v: i32) -> i32 {
    if v < 0 { -1 } else { 1 }
}

/// Line buffers for one "line component group": `n` samples per pixel.
struct Lines {
    prev: Vec<i32>,
    cur: Vec<i32>,
    n: usize,
}

impl Lines {
    fn new(width: usize, n: usize) -> Self {
        Lines {
            prev: vec![0; (width + 2) * n],
            cur: vec![0; (width + 2) * n],
            n,
        }
    }

    fn reset(&mut self) {
        self.prev.fill(0);
        self.cur.fill(0);
    }

    /// Edge pixels: prev[w+1] = prev[w], cur[0] = prev[1].
    fn init_edges(&mut self, width: usize) {
        let n = self.n;
        for c in 0..n {
            self.prev[(width + 1) * n + c] = self.prev[width * n + c];
            self.cur[c] = self.prev[n + c];
        }
    }

    fn swap(&mut self) {
        std::mem::swap(&mut self.prev, &mut self.cur);
    }
}

/// Iterate the line structure of a scan: for every line, for every line group
/// (component for line interleave, otherwise one group), call `f`. Handles
/// restart intervals (reset of lines, contexts and run indices).
fn for_each_line<F, R>(
    layout: &ScanLayout,
    p: &Params,
    groups: usize,
    n: usize,
    mut on_restart: R,
    mut f: F,
) -> Result<()>
where
    F: FnMut(&mut State, &mut Lines, usize, usize) -> Result<()>,
    R: FnMut() -> Result<()>,
{
    let mut lines: Vec<Lines> = (0..groups).map(|_| Lines::new(layout.width, n)).collect();
    let mut state = State::new(p);
    let mut run_index = vec![0usize; groups];
    let interval = if layout.restart_interval == 0 {
        layout.height
    } else {
        layout.restart_interval
    };
    for y in 0..layout.height {
        if y > 0 && y % interval == 0 {
            on_restart()?;
            state = State::new(p);
            run_index.fill(0);
            lines.iter_mut().for_each(Lines::reset);
        }
        for (g, l) in lines.iter_mut().enumerate() {
            state.run_index = run_index[g];
            l.init_edges(layout.width);
            f(&mut state, l, g, y)?;
            run_index[g] = state.run_index;
            l.swap();
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Decoding

struct Decoder<'r, 'a> {
    r: &'r mut BitReader<'a>,
    p: Params,
    q: &'r Quantizer,
}

impl Decoder<'_, '_> {
    fn regular(&mut self, s: &mut State, q: (i32, i32, i32), pred: i32) -> Result<i32> {
        let p = &self.p;
        let (ci, sg) = context(q.0, q.1, q.2);
        let ctx = &mut s.regular[ci];
        let px = p.clamp(pred + sg * ctx.c);
        let k = ctx.golomb_k()?;
        let m = self.r.golomb(k, p.limit, p.qbpp)?;
        let mut e = (m >> 1) ^ -(m & 1);
        if ctx.special_mapping(k, p.near) {
            e = -(e + 1);
        }
        if e.abs() > 65535 {
            return Err(Error::invalid(CODEC, "prediction error out of range"));
        }
        ctx.update(e, p.near, p.reset);
        Ok(p.reconstruct(px, sg * e))
    }

    fn ri_error(&mut self, s: &mut State, which: usize) -> Result<i32> {
        let p = &self.p;
        let limit = p.limit - J[s.run_index] - 1;
        let ctx = &mut s.run[which];
        let k = ctx.golomb_k()?;
        let em = self.r.golomb(k, limit, p.qbpp)?;
        let e = ctx.unmap(em + ctx.ritype, k);
        ctx.update(e, em, p.reset);
        Ok(e)
    }

    fn run_pixels(&mut self, s: &mut State, remaining: usize) -> Result<usize> {
        let mut index = 0usize;
        while self.r.bit()? {
            let full = 1usize << J[s.run_index];
            let count = full.min(remaining - index);
            index += count;
            if count == full && s.run_index < 31 {
                s.run_index += 1;
            }
            if index == remaining {
                break;
            }
        }
        if index != remaining {
            index += self.r.bits(J[s.run_index])? as usize;
        }
        if index > remaining {
            return Err(Error::invalid(CODEC, "run length exceeds line"));
        }
        Ok(index)
    }

    /// Decode one line of `N`-sample pixels.
    fn line<const N: usize>(&mut self, s: &mut State, l: &mut Lines, width: usize) -> Result<()> {
        let near = self.p.near;
        let (prev, cur) = (&l.prev[..(width + 2) * N], &mut l.cur[..(width + 2) * N]);
        let mut x = 1usize;
        while x <= width {
            // Neighbours a (left), b (above), c (above left), d (above right).
            let nb = |cur: &[i32], c: usize| {
                (
                    cur[(x - 1) * N + c],
                    prev[x * N + c],
                    prev[(x - 1) * N + c],
                    prev[(x + 1) * N + c],
                )
            };
            // Run mode iff every gradient quantizes to 0, i.e. lies within NEAR.
            let flat = (0..N).all(|c| {
                let (ra, rb, rc, rd) = nb(cur, c);
                (rd - rb).abs() <= near && (rb - rc).abs() <= near && (rc - ra).abs() <= near
            });
            if !flat {
                for c in 0..N {
                    let (ra, rb, rc, rd) = nb(cur, c);
                    let q = (self.q.q(rd - rb), self.q.q(rb - rc), self.q.q(rc - ra));
                    cur[x * N + c] = self.regular(s, q, predict(ra, rb, rc))?;
                }
                x += 1;
                continue;
            }
            // Run mode.
            let count = self.run_pixels(s, width - (x - 1))?;
            for i in x..x + count {
                for c in 0..N {
                    cur[i * N + c] = cur[(x - 1) * N + c];
                }
            }
            let end = x + count;
            if end > width {
                break;
            }
            for c in 0..N {
                let ra = cur[(x - 1) * N + c];
                let rb = prev[end * N + c];
                cur[end * N + c] = if N == 1 && (ra - rb).abs() <= near {
                    let e = self.ri_error(s, 1)?;
                    self.p.reconstruct(ra, e)
                } else {
                    let e = self.ri_error(s, 0)?;
                    self.p.reconstruct(rb, e * sign(rb - ra))
                };
            }
            s.run_index = s.run_index.saturating_sub(1);
            x = end + 1;
        }
        Ok(())
    }
}

/// Decode one scan into `planes` (one per scan component, `width * height`).
pub(crate) fn decode_scan(
    r: &mut BitReader<'_>,
    p: &Params,
    layout: &ScanLayout,
    planes: &mut [&mut [u16]],
) -> Result<()> {
    let ns = planes.len();
    let (groups, n) = match layout.interleave {
        Interleave::Sample => (1, ns),
        _ => (ns, 1),
    };
    let width = layout.width;
    let quantizer = Quantizer::new(p);
    let r_cell = std::cell::RefCell::new(r);
    for_each_line(
        layout,
        p,
        groups,
        n,
        || r_cell.borrow_mut().restart(),
        |state, lines, g, y| {
            let mut rb = r_cell.borrow_mut();
            let mut d = Decoder {
                r: &mut rb,
                p: *p,
                q: &quantizer,
            };
            match n {
                1 => d.line::<1>(state, lines, width)?,
                2 => d.line::<2>(state, lines, width)?,
                3 => d.line::<3>(state, lines, width)?,
                _ => d.line::<4>(state, lines, width)?,
            }
            for c in 0..n {
                let row = &mut planes[g + c][y * width..(y + 1) * width];
                for (d, px) in row.iter_mut().zip(lines.cur[n..].chunks_exact(n)) {
                    *d = px[c] as u16;
                }
            }
            Ok(())
        },
    )
}

// ---------------------------------------------------------------------------
// Encoding

#[cfg(feature = "encode")]
struct Encoder<'w> {
    w: &'w mut BitWriter,
    p: Params,
    q: &'w Quantizer,
}

#[cfg(feature = "encode")]
impl Encoder<'_> {
    fn regular(&mut self, s: &mut State, q: (i32, i32, i32), pred: i32, x: i32) -> i32 {
        let p = &self.p;
        let (ci, sg) = context(q.0, q.1, q.2);
        let ctx = &mut s.regular[ci];
        let px = p.clamp(pred + sg * ctx.c);
        let k = ctx.golomb_k().expect("encoder contexts stay in range");
        let e = p.modulo_range(p.quantize_error(sg * (x - px)));
        let e2 = if ctx.special_mapping(k, p.near) {
            -(e + 1)
        } else {
            e
        };
        let m = if e2 >= 0 { 2 * e2 } else { -2 * e2 - 1 };
        self.w.golomb(m, k, p.limit, p.qbpp);
        ctx.update(e, p.near, p.reset);
        p.reconstruct(px, sg * e)
    }

    fn ri_error(&mut self, s: &mut State, which: usize, e: i32) {
        let p = &self.p;
        let limit = p.limit - J[s.run_index] - 1;
        let ctx = &mut s.run[which];
        let k = ctx.golomb_k().expect("encoder contexts stay in range");
        let map = ctx.map(e, k) as i32;
        let em = 2 * e.abs() - ctx.ritype - map;
        self.w.golomb(em, k, limit, p.qbpp);
        ctx.update(e, em, p.reset);
    }

    fn run_pixels(&mut self, s: &mut State, mut len: usize, end_of_line: bool) {
        while len >= 1 << J[s.run_index] {
            self.w.write(1, 1);
            len -= 1 << J[s.run_index];
            if s.run_index < 31 {
                s.run_index += 1;
            }
        }
        if end_of_line {
            if len > 0 {
                self.w.write(1, 1);
            }
        } else {
            self.w.write(len as u32, J[s.run_index] + 1);
        }
    }

    /// Encode one line of `N`-sample pixels; `src[c]` is the line of
    /// component `c`. The current line buffer receives the reconstructed
    /// samples.
    fn line<const N: usize>(
        &mut self,
        s: &mut State,
        l: &mut Lines,
        width: usize,
        src: [&[u16]; N],
    ) {
        let near = self.p.near;
        let (prev, cur) = (&l.prev[..(width + 2) * N], &mut l.cur[..(width + 2) * N]);
        let src = src.map(|r| &r[..width]);
        let mut x = 1usize;
        while x <= width {
            let nb = |cur: &[i32], c: usize| {
                (
                    cur[(x - 1) * N + c],
                    prev[x * N + c],
                    prev[(x - 1) * N + c],
                    prev[(x + 1) * N + c],
                )
            };
            // Run mode iff every gradient quantizes to 0, i.e. lies within NEAR.
            let flat = (0..N).all(|c| {
                let (ra, rb, rc, rd) = nb(cur, c);
                (rd - rb).abs() <= near && (rb - rc).abs() <= near && (rc - ra).abs() <= near
            });
            if !flat {
                for c in 0..N {
                    let (ra, rb, rc, rd) = nb(cur, c);
                    let q = (self.q.q(rd - rb), self.q.q(rb - rc), self.q.q(rc - ra));
                    let v = src[c][x - 1] as i32;
                    cur[x * N + c] = self.regular(s, q, predict(ra, rb, rc), v);
                }
                x += 1;
                continue;
            }
            // Run mode: extend while every sample is within NEAR of Ra.
            let mut count = 0;
            while x + count <= width
                && (0..N)
                    .all(|c| (src[c][x + count - 1] as i32 - cur[(x - 1) * N + c]).abs() <= near)
            {
                for c in 0..N {
                    cur[(x + count) * N + c] = cur[(x - 1) * N + c];
                }
                count += 1;
            }
            let end = x + count;
            self.run_pixels(s, count, end > width);
            if end > width {
                break;
            }
            for c in 0..N {
                let ra = cur[(x - 1) * N + c];
                let rb = prev[end * N + c];
                let v = src[c][end - 1] as i32;
                cur[end * N + c] = if N == 1 && (ra - rb).abs() <= near {
                    let e = self.p.modulo_range(self.p.quantize_error(v - ra));
                    self.ri_error(s, 1, e);
                    self.p.reconstruct(ra, e)
                } else {
                    let sg = sign(rb - ra);
                    let e = self.p.modulo_range(self.p.quantize_error(sg * (v - rb)));
                    self.ri_error(s, 0, e);
                    self.p.reconstruct(rb, e * sg)
                };
            }
            s.run_index = s.run_index.saturating_sub(1);
            x = end + 1;
        }
    }
}

/// Encode one scan from `planes` (one per scan component, `width * height`).
#[cfg(feature = "encode")]
pub(crate) fn encode_scan(w: &mut BitWriter, p: &Params, layout: &ScanLayout, planes: &[&[u16]]) {
    let ns = planes.len();
    let (groups, n) = match layout.interleave {
        Interleave::Sample => (1, ns),
        _ => (ns, 1),
    };
    let width = layout.width;
    let quantizer = Quantizer::new(p);
    let w_cell = std::cell::RefCell::new(w);
    let mut rst = 0u8;
    for_each_line(
        layout,
        p,
        groups,
        n,
        || {
            let mut w = w_cell.borrow_mut();
            w.restart(rst);
            rst = (rst + 1) & 7;
            Ok(())
        },
        |state, lines, g, y| {
            let mut wb = w_cell.borrow_mut();
            let mut e = Encoder {
                w: &mut wb,
                p: *p,
                q: &quantizer,
            };
            let row = |c: usize| &planes[g + c][y * width..(y + 1) * width];
            match n {
                1 => e.line::<1>(state, lines, width, [row(0)]),
                2 => e.line::<2>(state, lines, width, [row(0), row(1)]),
                3 => e.line::<3>(state, lines, width, [row(0), row(1), row(2)]),
                _ => e.line::<4>(state, lines, width, [row(0), row(1), row(2), row(3)]),
            }
            Ok(())
        },
    )
    .expect("encoding cannot fail");
}
