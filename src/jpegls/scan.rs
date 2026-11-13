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
use super::coding::{J, Params, State, context, predict};
use crate::error::{Error, Result};

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

    fn line(&mut self, s: &mut State, l: &mut Lines, width: usize) -> Result<()> {
        let n = l.n;
        let near = self.p.near;
        let mut x = 1usize;
        while x <= width {
            let mut q = [(0, 0, 0); 4];
            let mut all_zero = true;
            for (c, qslot) in q.iter_mut().enumerate().take(n) {
                let ra = l.cur[(x - 1) * n + c];
                let rc = l.prev[(x - 1) * n + c];
                let rb = l.prev[x * n + c];
                let rd = l.prev[(x + 1) * n + c];
                let qc = (
                    self.p.quantize_gradient(rd - rb),
                    self.p.quantize_gradient(rb - rc),
                    self.p.quantize_gradient(rc - ra),
                );
                all_zero &= qc == (0, 0, 0);
                *qslot = qc;
            }
            if !all_zero {
                for (c, &qc) in q.iter().enumerate().take(n) {
                    let ra = l.cur[(x - 1) * n + c];
                    let rc = l.prev[(x - 1) * n + c];
                    let rb = l.prev[x * n + c];
                    l.cur[x * n + c] = self.regular(s, qc, predict(ra, rb, rc))?;
                }
                x += 1;
                continue;
            }
            // Run mode.
            let count = self.run_pixels(s, width - (x - 1))?;
            for i in x..x + count {
                for c in 0..n {
                    l.cur[i * n + c] = l.cur[(x - 1) * n + c];
                }
            }
            let end = x + count;
            if end > width {
                break;
            }
            for c in 0..n {
                let ra = l.cur[(x - 1) * n + c];
                let rb = l.prev[end * n + c];
                l.cur[end * n + c] = if n == 1 && (ra - rb).abs() <= near {
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
    let r_cell = std::cell::RefCell::new(r);
    for_each_line(
        layout,
        p,
        groups,
        n,
        || r_cell.borrow_mut().restart(),
        |state, lines, g, y| {
            let mut rb = r_cell.borrow_mut();
            let mut d = Decoder { r: &mut rb, p: *p };
            d.line(state, lines, width)?;
            for c in 0..n {
                let plane = &mut planes[g + c];
                for x in 0..width {
                    plane[y * width + x] = lines.cur[(x + 1) * n + c] as u16;
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

    /// Encode one line; `src(x, c)` gives the original sample. The current
    /// line buffer receives the reconstructed samples.
    fn line(
        &mut self,
        s: &mut State,
        l: &mut Lines,
        width: usize,
        src: &dyn Fn(usize, usize) -> i32,
    ) {
        let n = l.n;
        let near = self.p.near;
        let mut x = 1usize;
        while x <= width {
            let mut q = [(0, 0, 0); 4];
            let mut all_zero = true;
            for (c, qslot) in q.iter_mut().enumerate().take(n) {
                let ra = l.cur[(x - 1) * n + c];
                let rc = l.prev[(x - 1) * n + c];
                let rb = l.prev[x * n + c];
                let rd = l.prev[(x + 1) * n + c];
                let qc = (
                    self.p.quantize_gradient(rd - rb),
                    self.p.quantize_gradient(rb - rc),
                    self.p.quantize_gradient(rc - ra),
                );
                all_zero &= qc == (0, 0, 0);
                *qslot = qc;
            }
            if !all_zero {
                for (c, &qc) in q.iter().enumerate().take(n) {
                    let ra = l.cur[(x - 1) * n + c];
                    let rc = l.prev[(x - 1) * n + c];
                    let rb = l.prev[x * n + c];
                    l.cur[x * n + c] = self.regular(s, qc, predict(ra, rb, rc), src(x - 1, c));
                }
                x += 1;
                continue;
            }
            // Run mode: extend while every sample is within NEAR of Ra.
            let mut count = 0;
            while x + count <= width
                && (0..n).all(|c| (src(x + count - 1, c) - l.cur[(x - 1) * n + c]).abs() <= near)
            {
                for c in 0..n {
                    l.cur[(x + count) * n + c] = l.cur[(x - 1) * n + c];
                }
                count += 1;
            }
            let end = x + count;
            self.run_pixels(s, count, end > width);
            if end > width {
                break;
            }
            for c in 0..n {
                let ra = l.cur[(x - 1) * n + c];
                let rb = l.prev[end * n + c];
                let v = src(end - 1, c);
                l.cur[end * n + c] = if n == 1 && (ra - rb).abs() <= near {
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
    let w_cell = std::cell::RefCell::new(w);
    let mut rst = 0u8;
    for_each_line(
        layout,
        p,
        groups,
        n,
        || {
            let mut w = w_cell.borrow_mut();
            w.flush();
            w.out.extend_from_slice(&[0xFF, 0xD0 + rst]);
            rst = (rst + 1) & 7;
            Ok(())
        },
        |state, lines, g, y| {
            let mut wb = w_cell.borrow_mut();
            let mut e = Encoder { w: &mut wb, p: *p };
            let src = |x: usize, c: usize| planes[g + c][y * width + x] as i32;
            e.line(state, lines, width, &src);
            Ok(())
        },
    )
    .expect("encoding cannot fail");
}
