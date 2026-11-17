//! Modular image sub-bitstream (ISO/IEC 18181-1 Annex H): MA trees,
//! predictors (including the self-correcting weighted predictor) and the
//! RCT, palette and squeeze transforms. Arithmetic follows the reference
//! decoder exactly, including its edge-case conventions.

use super::bits::{BitReader, bits, invalid, unpack_signed, val};
use super::entropy::{Histograms, SymbolReader};
use dicom_jpeg_common::Result;

// ---------------------------------------------------------------------------
// Channels and images

#[derive(Debug, Clone)]
pub(crate) struct Channel {
    pub(crate) w: usize,
    pub(crate) h: usize,
    pub(crate) hshift: i32,
    pub(crate) vshift: i32,
    pub(crate) data: Vec<i32>,
}

impl Channel {
    pub(crate) fn new(w: usize, h: usize, hshift: i32, vshift: i32) -> Self {
        Channel {
            w,
            h,
            hshift,
            vshift,
            data: Vec::new(),
        }
    }

    fn alloc(&mut self) {
        if self.data.len() != self.w * self.h {
            self.data = vec![0; self.w * self.h];
        }
    }

    #[inline]
    pub(crate) fn row(&self, y: usize) -> &[i32] {
        &self.data[y * self.w..(y + 1) * self.w]
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct WpHeader {
    p1: i64,
    p2: i64,
    p3: [i64; 5],
    w: [u32; 4],
}

impl Default for WpHeader {
    fn default() -> Self {
        WpHeader {
            p1: 16,
            p2: 10,
            p3: [7, 7, 7, 0, 0],
            w: [13, 12, 12, 12],
        }
    }
}

impl WpHeader {
    fn read(br: &mut BitReader) -> Result<Self> {
        if br.read_bool()? {
            return Ok(Self::default());
        }
        let p1 = br.read(5)? as i64;
        let p2 = br.read(5)? as i64;
        let mut p3 = [0i64; 5];
        for p in &mut p3 {
            *p = br.read(5)? as i64;
        }
        let mut w = [0u32; 4];
        for x in &mut w {
            *x = br.read(4)?;
        }
        Ok(WpHeader { p1, p2, p3, w })
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct SqueezeParams {
    horizontal: bool,
    in_place: bool,
    begin_c: u32,
    num_c: u32,
}

#[derive(Debug, Clone)]
pub(crate) enum Transform {
    Rct {
        begin_c: u32,
        rct_type: u32,
    },
    Palette {
        begin_c: u32,
        num_c: u32,
        nb_colours: u32,
        nb_deltas: u32,
        d_pred: u32,
    },
    Squeeze(Vec<SqueezeParams>),
}

const BEGIN_C: [super::bits::U32D; 4] = [bits(3, 0), bits(6, 8), bits(10, 72), bits(13, 1096)];

impl Transform {
    fn read(br: &mut BitReader) -> Result<Self> {
        Ok(match br.read(2)? {
            0 => {
                let begin_c = br.u32(BEGIN_C)?;
                let rct_type = br.u32([val(6), bits(2, 0), bits(4, 2), bits(6, 10)])?;
                if rct_type >= 42 {
                    return Err(invalid("invalid RCT type"));
                }
                Transform::Rct { begin_c, rct_type }
            }
            1 => {
                let begin_c = br.u32(BEGIN_C)?;
                let num_c = br.u32([val(1), val(3), val(4), bits(13, 1)])?;
                let nb_colours =
                    br.u32([bits(8, 0), bits(10, 256), bits(12, 1280), bits(16, 5376)])?;
                let nb_deltas = br.u32([val(0), bits(8, 1), bits(10, 257), bits(16, 1281)])?;
                let d_pred = br.read(4)?;
                if d_pred >= 14 {
                    return Err(invalid("invalid palette predictor"));
                }
                Transform::Palette {
                    begin_c,
                    num_c,
                    nb_colours,
                    nb_deltas,
                    d_pred,
                }
            }
            2 => {
                let num_sq = br.u32([val(0), bits(4, 1), bits(6, 9), bits(8, 41)])?;
                let mut params = Vec::with_capacity(num_sq as usize);
                for _ in 0..num_sq {
                    let horizontal = br.read_bool()?;
                    let in_place = br.read_bool()?;
                    let begin_c = br.u32(BEGIN_C)?;
                    let num_c = br.u32([val(1), val(2), val(3), bits(4, 4)])?;
                    params.push(SqueezeParams {
                        horizontal,
                        in_place,
                        begin_c,
                        num_c,
                    });
                }
                Transform::Squeeze(params)
            }
            _ => return Err(invalid("unknown modular transform")),
        })
    }
}

/// A modular image: channel geometry plus the transforms that produced it.
#[derive(Debug, Clone, Default)]
pub(crate) struct Image {
    pub(crate) channels: Vec<Channel>,
    pub(crate) nb_meta: usize,
    pub(crate) transforms: Vec<Transform>,
    pub(crate) wp: WpHeader,
    pub(crate) bit_depth: u32,
}

impl Image {
    /// Apply the channel bookkeeping of `transforms[i]` (the "meta" step).
    fn meta_apply(&mut self, i: usize) -> Result<()> {
        match &mut self.transforms[i] {
            Transform::Rct { begin_c, .. } => {
                let b = *begin_c as usize;
                if b + 3 > self.channels.len() {
                    return Err(invalid("RCT channels out of range"));
                }
                let c0 = &self.channels[b];
                for c in &self.channels[b + 1..b + 3] {
                    if (c.w, c.h, c.hshift, c.vshift) != (c0.w, c0.h, c0.hshift, c0.vshift) {
                        return Err(invalid("RCT on differently sized channels"));
                    }
                }
            }
            Transform::Palette {
                begin_c,
                num_c,
                nb_colours,
                ..
            } => {
                let (b, n) = (*begin_c as usize, *num_c as usize);
                let end = b + n;
                if end > self.channels.len() || n == 0 {
                    return Err(invalid("palette channels out of range"));
                }
                if b < self.nb_meta {
                    if end > self.nb_meta {
                        return Err(invalid("palette mixes meta and image channels"));
                    }
                    self.nb_meta = self.nb_meta + 2 - n;
                } else {
                    self.nb_meta += 1;
                }
                let c0 = &self.channels[b];
                for c in &self.channels[b + 1..end] {
                    if (c.w, c.h, c.hshift, c.vshift) != (c0.w, c0.h, c0.hshift, c0.vshift) {
                        return Err(invalid("palette on differently sized channels"));
                    }
                }
                self.channels.drain(b + 1..end);
                self.channels
                    .insert(0, Channel::new(*nb_colours as usize, n, -1, -1));
            }
            Transform::Squeeze(params) => {
                if params.is_empty() {
                    *params = default_squeeze(&self.channels, self.nb_meta)?;
                }
                for p in params.iter() {
                    let b = p.begin_c as usize;
                    let n = p.num_c as usize;
                    let end = b + n;
                    if end > self.channels.len() || n == 0 {
                        return Err(invalid("squeeze channels out of range"));
                    }
                    if b < self.nb_meta {
                        if end > self.nb_meta || !p.in_place {
                            return Err(invalid("invalid squeeze of meta channels"));
                        }
                        self.nb_meta += n;
                    }
                    let offset = if p.in_place { end } else { self.channels.len() };
                    for (k, c) in (b..end).enumerate() {
                        let ch = &mut self.channels[c];
                        if ch.hshift > 30 || ch.vshift > 30 {
                            return Err(invalid("too many squeezes"));
                        }
                        if ch.w == 0 || ch.h == 0 {
                            return Err(invalid("squeezing an empty channel"));
                        }
                        let (mut rw, mut rh) = (ch.w, ch.h);
                        if p.horizontal {
                            rw = ch.w / 2;
                            ch.w = ch.w.div_ceil(2);
                            if ch.hshift >= 0 {
                                ch.hshift += 1;
                            }
                        } else {
                            rh = ch.h / 2;
                            ch.h = ch.h.div_ceil(2);
                            if ch.vshift >= 0 {
                                ch.vshift += 1;
                            }
                        }
                        let res = Channel::new(rw, rh, ch.hshift, ch.vshift);
                        self.channels.insert(offset + k, res);
                    }
                }
            }
        }
        if self.channels.len() > 1 << 16 {
            return Err(invalid("too many modular channels"));
        }
        Ok(())
    }

    /// Undo all transforms (in reverse order).
    pub(crate) fn undo_transforms(&mut self) -> Result<()> {
        while let Some(t) = self.transforms.pop() {
            match t {
                Transform::Rct { begin_c, rct_type } => self.inv_rct(begin_c as usize, rct_type),
                Transform::Palette {
                    begin_c,
                    num_c,
                    nb_colours,
                    nb_deltas,
                    d_pred,
                } => self.inv_palette(
                    begin_c as usize,
                    num_c as usize,
                    nb_colours,
                    nb_deltas,
                    d_pred,
                )?,
                Transform::Squeeze(params) => {
                    for p in params.iter().rev() {
                        self.inv_squeeze(p)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn inv_rct(&mut self, b: usize, rct_type: u32) {
        let perm = (rct_type / 7) as usize;
        let custom = rct_type % 7;
        let n = self.channels[b].data.len();
        if custom != 0 {
            let (second, third) = (custom >> 1, custom & 1);
            for i in 0..n {
                let a = self.channels[b].data[i];
                let mut s = self.channels[b + 1].data[i];
                let mut t = self.channels[b + 2].data[i];
                let (o0, o1, o2);
                if custom == 6 {
                    let tmp = a.wrapping_sub(t >> 1);
                    let g = t.wrapping_add(tmp);
                    let bb = tmp.wrapping_sub(s >> 1);
                    let r = bb.wrapping_add(s);
                    (o0, o1, o2) = (r, g, bb);
                } else {
                    if third == 1 {
                        t = t.wrapping_add(a);
                    }
                    if second == 1 {
                        s = s.wrapping_add(a);
                    } else if second == 2 {
                        s = s.wrapping_add(a.wrapping_add(t) >> 1);
                    }
                    (o0, o1, o2) = (a, s, t);
                }
                self.channels[b].data[i] = o0;
                self.channels[b + 1].data[i] = o1;
                self.channels[b + 2].data[i] = o2;
            }
        }
        // Permute: input k goes to position pos[k].
        let pos = [
            perm % 3,
            (perm + 1 + perm / 3) % 3,
            (perm + 2 - perm / 3) % 3,
        ];
        let olds: Vec<Channel> = (0..3)
            .map(|k| std::mem::replace(&mut self.channels[b + k], Channel::new(0, 0, 0, 0)))
            .collect();
        for (k, ch) in olds.into_iter().enumerate() {
            self.channels[b + pos[k]] = ch;
        }
    }

    fn inv_palette(
        &mut self,
        b: usize,
        nb: usize,
        nb_colours: u32,
        nb_deltas: u32,
        d_pred: u32,
    ) -> Result<()> {
        if self.nb_meta < 1 {
            return Err(invalid("palette transform without palette"));
        }
        let c0 = b + 1;
        if c0 >= self.channels.len() {
            return Err(invalid("palette index channel out of range"));
        }
        let palette = self.channels[0].clone();
        if palette.h != nb || palette.w != nb_colours as usize {
            return Err(invalid("palette channel has the wrong size"));
        }
        let (w, h, hs, vs) = {
            let c = &self.channels[c0];
            (c.w, c.h, c.hshift, c.vshift)
        };
        let indices = std::mem::take(&mut self.channels[c0].data);
        for _ in 1..nb {
            self.channels.insert(c0 + 1, Channel::new(w, h, hs, vs));
        }
        let bit_depth = self.bit_depth.min(24);
        let pal_w = palette.w as i64;
        for c in 0..nb {
            let ch = &mut self.channels[c0 + c];
            ch.data = vec![0; w * h];
            if w == 0 {
                continue;
            }
            if nb_deltas == 0 && d_pred == 0 {
                for (o, &idx) in ch.data.iter_mut().zip(&indices) {
                    let idx = if nb == 1 {
                        (idx as i64).clamp(0, pal_w - 1)
                    } else {
                        idx as i64
                    };
                    *o = palette_value(&palette, idx, c, bit_depth);
                }
                continue;
            }
            let mut wp = (d_pred == 6).then(|| WpState::new(self.wp, w));
            for y in 0..h {
                for x in 0..w {
                    let idx = indices[y * w + x] as i64;
                    let entry = palette_value(&palette, idx, c, bit_depth) as i64;
                    let nb_ = Neighbours::at(&ch.data, w, x, y);
                    let wp_pred = match &mut wp {
                        Some(s) => s.predict(x, y, &nb_, None),
                        None => 0,
                    };
                    let v = if idx < nb_deltas as i64 {
                        entry + predict(d_pred, &nb_, wp_pred)
                    } else {
                        entry
                    };
                    let v = v as i32;
                    ch.data[y * w + x] = v;
                    if let Some(s) = &mut wp {
                        s.update(v as i64, x, y);
                    }
                }
            }
        }
        if c0 >= self.nb_meta {
            self.nb_meta -= 1;
        } else {
            self.nb_meta -= 2 - nb.min(2);
            if nb > 2 {
                self.nb_meta += nb - 2;
            }
        }
        self.channels.remove(0);
        Ok(())
    }

    fn inv_squeeze(&mut self, p: &SqueezeParams) -> Result<()> {
        let b = p.begin_c as usize;
        let n = p.num_c as usize;
        let end = b + n - 1;
        if end >= self.channels.len() {
            return Err(invalid("squeeze channels out of range"));
        }
        let offset = if p.in_place {
            end + 1
        } else {
            self.channels
                .len()
                .checked_sub(n)
                .ok_or_else(|| invalid("squeeze residuals missing"))?
        };
        if b < self.nb_meta {
            if !p.in_place || end >= self.nb_meta {
                return Err(invalid("invalid squeeze of meta channels"));
            }
            self.nb_meta -= n;
        }
        if offset + n > self.channels.len() {
            return Err(invalid("squeeze residuals missing"));
        }
        for c in b..=end {
            let rc = offset + c - b;
            let res = std::mem::replace(&mut self.channels[rc], Channel::new(0, 0, 0, 0));
            let ch = &mut self.channels[c];
            if p.horizontal {
                inv_hsqueeze(ch, &res)?;
            } else {
                inv_vsqueeze(ch, &res)?;
            }
        }
        self.channels.drain(offset..offset + n);
        Ok(())
    }
}

const DELTA_PALETTE: [[i32; 3]; 72] = [
    [0, 0, 0],
    [4, 4, 4],
    [11, 0, 0],
    [0, 0, -13],
    [0, -12, 0],
    [-10, -10, -10],
    [-18, -18, -18],
    [-27, -27, -27],
    [-18, -18, 0],
    [0, 0, -32],
    [-32, 0, 0],
    [-37, -37, -37],
    [0, -32, -32],
    [24, 24, 45],
    [50, 50, 50],
    [-45, -24, -24],
    [-24, -45, -45],
    [0, -24, -24],
    [-34, -34, 0],
    [-24, 0, -24],
    [-45, -45, -24],
    [64, 64, 64],
    [-32, 0, -32],
    [0, -32, 0],
    [-32, 0, 32],
    [-24, -45, -24],
    [45, 24, 45],
    [24, -24, -45],
    [-45, -24, 24],
    [80, 80, 80],
    [64, 0, 0],
    [0, 0, -64],
    [0, -64, -64],
    [-24, -24, 45],
    [96, 96, 96],
    [64, 64, 0],
    [45, -24, -24],
    [34, -34, 0],
    [112, 112, 112],
    [24, -45, -45],
    [45, 45, -24],
    [0, -32, 32],
    [24, -24, 45],
    [0, 96, 96],
    [45, -24, 24],
    [24, -45, -24],
    [-24, -45, 24],
    [0, -64, 0],
    [96, 0, 0],
    [128, 128, 128],
    [64, 0, 64],
    [144, 144, 144],
    [96, 96, 0],
    [-36, -36, 36],
    [45, -24, -45],
    [45, -45, -24],
    [0, 0, -96],
    [0, 128, 128],
    [0, 96, 0],
    [45, 24, -45],
    [-128, 0, 0],
    [24, -45, 24],
    [-45, 24, -45],
    [64, 0, -64],
    [64, -64, -64],
    [96, 0, 96],
    [45, -45, 24],
    [24, 45, -45],
    [64, 64, -64],
    [128, 128, 0],
    [0, 0, -128],
    [-24, 45, -45],
];

/// Palette entry or implicit colour for `index` in channel `c` (H.6.4).
fn palette_value(palette: &Channel, index: i64, c: usize, bit_depth: u32) -> i32 {
    let size = palette.w as i64;
    const SMALL_OFFSET: i64 = 64;
    if index < 0 {
        if c >= 3 {
            return 0;
        }
        let mut i = -(index + 1);
        i %= 1 + 2 * (DELTA_PALETTE.len() as i64 - 1);
        let mut v = DELTA_PALETTE[((i + 1) >> 1) as usize][c];
        if i & 1 == 0 {
            v = -v;
        }
        if bit_depth > 8 {
            v = v.wrapping_mul(1 << (bit_depth - 8));
        }
        v
    } else if index >= size && index < size + SMALL_OFFSET {
        if c >= 3 {
            return 0;
        }
        let i = (index - size) >> (2 * c);
        let scaled = (((i % 4) as u64) * ((1u64 << bit_depth) - 1)) >> 2;
        scaled as i32 + (1i32 << bit_depth.saturating_sub(3))
    } else if index >= size + SMALL_OFFSET {
        if c >= 3 {
            return 0;
        }
        let mut i = index - size - SMALL_OFFSET;
        i /= [1, 5, 25][c];
        ((((i % 5) as u64) * ((1u64 << bit_depth) - 1)) >> 2) as i32
    } else {
        palette.data[c * palette.w + index as usize]
    }
}

fn smooth_tendency(b: i64, a: i64, n: i64) -> i64 {
    let mut diff = 0;
    if b >= a && a >= n {
        diff = (4 * b - 3 * n - a + 6) / 12;
        if diff - (diff & 1) > 2 * (b - a) {
            diff = 2 * (b - a) + 1;
        }
        if diff + (diff & 1) > 2 * (a - n) {
            diff = 2 * (a - n);
        }
    } else if b <= a && a <= n {
        diff = (4 * b - 3 * n - a - 6) / 12;
        if diff + (diff & 1) < 2 * (b - a) {
            diff = 2 * (b - a) - 1;
        }
        if diff - (diff & 1) < 2 * (a - n) {
            diff = 2 * (a - n);
        }
    }
    diff
}

fn inv_hsqueeze(ch: &mut Channel, res: &Channel) -> Result<()> {
    if ch.w != (ch.w + res.w).div_ceil(2) || ch.h != res.h || ch.hshift == 0 {
        return Err(invalid("inconsistent horizontal squeeze"));
    }
    let new_shift = if ch.hshift > 0 {
        ch.hshift - 1
    } else {
        ch.hshift
    };
    if res.w == 0 {
        ch.hshift = new_shift;
        return Ok(());
    }
    let ow = ch.w + res.w;
    let mut out = vec![0i32; ow * ch.h];
    for y in 0..ch.h {
        let avg_row = ch.row(y);
        let res_row = res.row(y);
        let out_row = &mut out[y * ow..(y + 1) * ow];
        for x in 0..res.w {
            let avg = avg_row[x] as i64;
            let next = if x + 1 < ch.w {
                avg_row[x + 1] as i64
            } else {
                avg
            };
            let left = if x > 0 {
                out_row[2 * x - 1] as i64
            } else {
                avg
            };
            let diff = res_row[x] as i64 + smooth_tendency(left, avg, next);
            let a = avg + diff / 2;
            out_row[2 * x] = a as i32;
            out_row[2 * x + 1] = (a - diff) as i32;
        }
        if ow & 1 == 1 {
            out_row[ow - 1] = avg_row[ch.w - 1];
        }
    }
    ch.w = ow;
    ch.hshift = new_shift;
    ch.data = out;
    Ok(())
}

fn inv_vsqueeze(ch: &mut Channel, res: &Channel) -> Result<()> {
    if ch.h != (ch.h + res.h).div_ceil(2) || ch.w != res.w || ch.vshift == 0 {
        return Err(invalid("inconsistent vertical squeeze"));
    }
    let new_shift = if ch.vshift > 0 {
        ch.vshift - 1
    } else {
        ch.vshift
    };
    if res.h == 0 {
        ch.vshift = new_shift;
        return Ok(());
    }
    let oh = ch.h + res.h;
    let w = ch.w;
    let mut out = vec![0i32; w * oh];
    for y in 0..res.h {
        for x in 0..w {
            let avg = ch.data[y * w + x] as i64;
            let next = if y + 1 < ch.h {
                ch.data[(y + 1) * w + x] as i64
            } else {
                avg
            };
            let top = if y > 0 {
                out[(2 * y - 1) * w + x] as i64
            } else {
                avg
            };
            let diff = res.data[y * w + x] as i64 + smooth_tendency(top, avg, next);
            let a = avg + diff / 2;
            out[2 * y * w + x] = a as i32;
            out[(2 * y + 1) * w + x] = (a - diff) as i32;
        }
    }
    if oh & 1 == 1 {
        let last = (ch.h - 1) * w;
        out[(oh - 1) * w..].copy_from_slice(&ch.data[last..last + w]);
    }
    ch.h = oh;
    ch.vshift = new_shift;
    ch.data = out;
    Ok(())
}

fn default_squeeze(channels: &[Channel], nb_meta: usize) -> Result<Vec<SqueezeParams>> {
    const MAX_FIRST_PREVIEW: usize = 8;
    let first = channels
        .get(nb_meta)
        .ok_or_else(|| invalid("squeeze without image channels"))?;
    let nb_channels = channels.len() - nb_meta;
    let (mut w, mut h) = (first.w, first.h);
    let mut out = Vec::new();
    if nb_channels > 2 && channels[nb_meta + 1].w == w && channels[nb_meta + 1].h == h {
        let base = SqueezeParams {
            horizontal: true,
            in_place: false,
            begin_c: nb_meta as u32 + 1,
            num_c: 2,
        };
        out.push(base);
        out.push(SqueezeParams {
            horizontal: false,
            ..base
        });
    }
    let base = SqueezeParams {
        horizontal: false,
        in_place: true,
        begin_c: nb_meta as u32,
        num_c: nb_channels as u32,
    };
    if w <= h && h > MAX_FIRST_PREVIEW {
        out.push(base);
        h = h.div_ceil(2);
    }
    while w > MAX_FIRST_PREVIEW || h > MAX_FIRST_PREVIEW {
        if w > MAX_FIRST_PREVIEW {
            out.push(SqueezeParams {
                horizontal: true,
                ..base
            });
            w = w.div_ceil(2);
        }
        if h > MAX_FIRST_PREVIEW {
            out.push(base);
            h = h.div_ceil(2);
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// MA tree

#[derive(Debug, Clone, Copy)]
enum Node {
    Split {
        prop: u32,
        value: i32,
        left: u32,
        right: u32,
    },
    Leaf {
        ctx: u32,
        predictor: u32,
        offset: i64,
        multiplier: i64,
    },
}

/// A decoded MA tree with the entropy code of its leaves.
#[derive(Debug, Clone)]
pub(crate) struct MaConfig {
    nodes: Vec<Node>,
    histograms: Histograms,
}

const NUM_PREDICTORS: u32 = 14;

impl MaConfig {
    pub(crate) fn read(br: &mut BitReader, size_limit: usize) -> Result<Self> {
        let h = Histograms::decode(br, 6)?;
        let mut reader = SymbolReader::new(&h, br, 0)?;
        let mut nodes = Vec::new();
        let mut to_decode = 1usize;
        let mut leaves = 0u32;
        while to_decode > 0 {
            if nodes.len() > size_limit.min(1 << 26) {
                return Err(invalid("MA tree too large"));
            }
            to_decode -= 1;
            let prop1 = reader.read(&h, br, 1)?;
            if prop1 > 256 {
                return Err(invalid("invalid MA tree property"));
            }
            if prop1 == 0 {
                let predictor = reader.read(&h, br, 2)?;
                if predictor >= NUM_PREDICTORS {
                    return Err(invalid("invalid predictor"));
                }
                let offset = unpack_signed(reader.read(&h, br, 3)?) as i64;
                let mul_log = reader.read(&h, br, 4)?;
                if mul_log >= 31 {
                    return Err(invalid("invalid multiplier"));
                }
                let mul_bits = reader.read(&h, br, 5)?;
                if mul_bits as u64 >= (1u64 << (31 - mul_log)) - 1 {
                    return Err(invalid("invalid multiplier"));
                }
                nodes.push(Node::Leaf {
                    ctx: leaves,
                    predictor,
                    offset,
                    multiplier: ((mul_bits as i64) + 1) << mul_log,
                });
                leaves += 1;
                continue;
            }
            let value = unpack_signed(reader.read(&h, br, 0)?);
            let base = (nodes.len() + to_decode) as u32;
            nodes.push(Node::Split {
                prop: prop1 - 1,
                value,
                left: base + 1,
                right: base + 2,
            });
            to_decode += 2;
        }
        reader.finish()?;
        let histograms = Histograms::decode(br, leaves as usize)?;
        Ok(MaConfig { nodes, histograms })
    }

    /// Highest property index used by the tree, whether the weighted
    /// predictor (or its error property) is used, and a single leaf if the
    /// tree is trivial.
    fn analyse(&self) -> (u32, bool) {
        let mut max_prop = 0;
        let mut wp = false;
        for n in &self.nodes {
            match *n {
                Node::Split { prop, .. } => {
                    max_prop = max_prop.max(prop);
                    wp |= prop == 15;
                }
                Node::Leaf { predictor, .. } => wp |= predictor == 6,
            }
        }
        (max_prop, wp)
    }
}

// ---------------------------------------------------------------------------
// Prediction

#[derive(Debug, Clone, Copy)]
struct Neighbours {
    w: i64,
    n: i64,
    nw: i64,
    ne: i64,
    nn: i64,
    ww: i64,
    nee: i64,
}

impl Neighbours {
    /// The neighbours of (x, y) in a channel of width `width` whose rows
    /// before `y` and samples before `x` are decoded.
    #[inline(always)]
    fn get(data: &[i32], width: usize, x: usize, y: usize) -> Self {
        if y >= 2 && x >= 2 && x + 2 < width {
            // Interior: no edge rules apply.
            let i = y * width + x;
            let p = |j: usize| data[j] as i64;
            Neighbours {
                w: p(i - 1),
                n: p(i - width),
                nw: p(i - width - 1),
                ne: p(i - width + 1),
                nn: p(i - 2 * width),
                ww: p(i - 2),
                nee: p(i - width + 2),
            }
        } else {
            Self::at(data, width, x, y)
        }
    }

    #[cold]
    fn at(data: &[i32], width: usize, x: usize, y: usize) -> Self {
        let p = |xx: usize, yy: usize| data[yy * width + xx] as i64;
        let w = if x > 0 {
            p(x - 1, y)
        } else if y > 0 {
            p(x, y - 1)
        } else {
            0
        };
        let n = if y > 0 { p(x, y - 1) } else { w };
        let nw = if x > 0 && y > 0 { p(x - 1, y - 1) } else { w };
        let ne = if x + 1 < width && y > 0 {
            p(x + 1, y - 1)
        } else {
            n
        };
        let ww = if x > 1 { p(x - 2, y) } else { w };
        let nn = if y > 1 { p(x, y - 2) } else { n };
        let nee = if x + 2 < width && y > 0 {
            p(x + 2, y - 1)
        } else {
            ne
        };
        Neighbours {
            w,
            n,
            nw,
            ne,
            nn,
            ww,
            nee,
        }
    }
}

#[inline]
fn clamped_gradient(w: i64, n: i64, nw: i64) -> i64 {
    let (lo, hi) = (w.min(n), w.max(n));
    let g = w + n - nw;
    if nw < lo {
        hi
    } else if nw > hi {
        lo
    } else {
        g
    }
}

#[inline]
fn predict(p: u32, nb: &Neighbours, wp_pred: i64) -> i64 {
    match p {
        0 => 0,
        1 => nb.w,
        2 => nb.n,
        3 => (nb.w + nb.n) / 2,
        4 => {
            // Select.
            let s = nb.w + nb.n - nb.nw;
            if (s - nb.w).abs() < (s - nb.n).abs() {
                nb.w
            } else {
                nb.n
            }
        }
        5 => clamped_gradient(nb.w, nb.n, nb.nw),
        6 => wp_pred,
        7 => nb.ne,
        8 => nb.nw,
        9 => nb.ww,
        10 => (nb.w + nb.nw) / 2,
        11 => (nb.n + nb.nw) / 2,
        12 => (nb.n + nb.ne) / 2,
        _ => (6 * nb.n - 2 * nb.nn + 7 * nb.w + nb.ww + nb.nee + 3 * nb.ne + 8) / 16,
    }
}

const DIV_LOOKUP: [u32; 64] = {
    let mut t = [0u32; 64];
    let mut i = 0;
    while i < 64 {
        t[i] = (1u32 << 24) / (i as u32 + 1);
        i += 1;
    }
    t
};

/// Self-correcting (weighted) predictor state for one channel.
struct WpState {
    header: WpHeader,
    width: usize,
    prediction: [i64; 4],
    pred: i64,
    /// Per position: the errors of the four sub-predictors.
    pred_errors: Vec<[u32; 4]>,
    error: Vec<i32>,
}

const PRED_EXTRA_BITS: u32 = 3;
const PRED_ROUND: i64 = ((1 << PRED_EXTRA_BITS) >> 1) - 1;

impl WpState {
    fn new(header: WpHeader, width: usize) -> Self {
        let n = (width + 2) * 2;
        WpState {
            header,
            width,
            prediction: [0; 4],
            pred: 0,
            pred_errors: vec![[0; 4]; n],
            error: vec![0; n],
        }
    }

    #[inline]
    fn error_weight(x: u64, maxweight: u32) -> u32 {
        let shift = (63 - (x + 1).leading_zeros() as i32 - 5).max(0);
        4 + ((maxweight as u64 * DIV_LOOKUP[(x >> shift) as usize] as u64) >> shift) as u32
    }

    /// Returns the prediction; `prop` receives the max-error property.
    #[inline]
    fn predict(&mut self, x: usize, y: usize, nb: &Neighbours, prop: Option<&mut i64>) -> i64 {
        let xsize = self.width;
        let cur_row = if y & 1 == 1 { 0 } else { xsize + 2 };
        let prev_row = if y & 1 == 1 { xsize + 2 } else { 0 };
        let pos_n = prev_row + x;
        let pos_ne = if x < xsize - 1 { pos_n + 1 } else { pos_n };
        let pos_nw = if x > 0 { pos_n - 1 } else { pos_n };
        let (en, ene, enw) = (
            self.pred_errors[pos_n],
            self.pred_errors[pos_ne],
            self.pred_errors[pos_nw],
        );
        let weights: [u32; 4] = std::array::from_fn(|i| {
            let sum = en[i] as u64 + ene[i] as u64 + enw[i] as u64;
            Self::error_weight(sum, self.header.w[i])
        });
        let mut weights = weights;
        let n = nb.n << PRED_EXTRA_BITS;
        let w = nb.w << PRED_EXTRA_BITS;
        let ne = nb.ne << PRED_EXTRA_BITS;
        let nw = nb.nw << PRED_EXTRA_BITS;
        let nn = nb.nn << PRED_EXTRA_BITS;
        let te_w = if x == 0 {
            0
        } else {
            self.error[cur_row + x - 1] as i64
        };
        let te_n = self.error[pos_n] as i64;
        let te_nw = self.error[pos_nw] as i64;
        let te_ne = self.error[pos_ne] as i64;
        let sum_wn = te_n + te_w;
        if let Some(p) = prop {
            let mut v = te_w;
            if te_n.abs() > v.abs() {
                v = te_n;
            }
            if te_nw.abs() > v.abs() {
                v = te_nw;
            }
            if te_ne.abs() > v.abs() {
                v = te_ne;
            }
            *p = v;
        }
        let h = &self.header;
        self.prediction[0] = w + ne - n;
        self.prediction[1] = n - (((sum_wn + te_ne) * h.p1) >> 5);
        self.prediction[2] = w - (((sum_wn + te_nw) * h.p2) >> 5);
        self.prediction[3] = n
            - ((te_nw * h.p3[0]
                + te_n * h.p3[1]
                + te_ne * h.p3[2]
                + (nn - n) * h.p3[3]
                + (nw - w) * h.p3[4])
                >> 5);
        // Weighted average.
        let total: u32 = weights.iter().sum();
        let log_weight = 31 - total.leading_zeros();
        let mut wsum = 0u32;
        for wt in &mut weights {
            *wt >>= log_weight - 4;
            wsum += *wt;
        }
        let mut sum = (wsum >> 1) as i64 - 1;
        for (p, &wt) in self.prediction.iter().zip(&weights) {
            sum += p * wt as i64;
        }
        let mut pred = (sum * DIV_LOOKUP[(wsum - 1) as usize] as i64) >> 24;
        if ((te_n ^ te_w) | (te_n ^ te_nw)) <= 0 {
            let mx = w.max(ne.max(n));
            let mn = w.min(ne.min(n));
            pred = pred.clamp(mn, mx);
        }
        self.pred = pred;
        (pred + PRED_ROUND) >> PRED_EXTRA_BITS
    }

    #[inline]
    fn update(&mut self, val: i64, x: usize, y: usize) {
        let xsize = self.width;
        let cur_row = if y & 1 == 1 { 0 } else { xsize + 2 };
        let prev_row = if y & 1 == 1 { xsize + 2 } else { 0 };
        let v = val << PRED_EXTRA_BITS;
        self.error[cur_row + x] = (self.pred - v) as i32;
        let err: [u32; 4] = std::array::from_fn(|i| {
            (((self.prediction[i] - v).abs() + PRED_ROUND) >> PRED_EXTRA_BITS) as u32
        });
        self.pred_errors[cur_row + x] = err;
        let above = &mut self.pred_errors[prev_row + x + 1];
        for (a, e) in above.iter_mut().zip(err) {
            *a = a.wrapping_add(e);
        }
    }
}

// ---------------------------------------------------------------------------
// Channel decoding

/// Decode one channel of `img` with the MA tree.
#[allow(clippy::too_many_arguments)]
fn decode_channel(
    br: &mut BitReader,
    reader: &mut SymbolReader,
    cfg: &MaConfig,
    wp_header: WpHeader,
    channels: &mut [Channel],
    idx: usize,
    stream_id: u32,
    max_prop: u32,
    uses_wp: bool,
) -> Result<()> {
    let (w, h, hs, vs) = {
        let c = &channels[idx];
        (c.w, c.h, c.hshift, c.vshift)
    };
    // Reference channels for properties 16+ (most recent first).
    let num_refs = if max_prop >= 16 {
        ((max_prop - 16) / 4 + 1) as usize
    } else {
        0
    };
    let refs: Vec<usize> = (0..idx)
        .rev()
        .filter(|&j| {
            let c = &channels[j];
            (c.w, c.h, c.hshift, c.vshift) == (w, h, hs, vs)
        })
        .take(num_refs)
        .collect();
    let (before, rest) = channels.split_at_mut(idx);
    let ch = &mut rest[0];
    ch.alloc();
    let hist = &cfg.histograms;
    let mut wp = uses_wp.then(|| WpState::new(wp_header, w));

    // Trivial tree: one leaf.
    if let [
        Node::Leaf {
            ctx,
            predictor,
            offset,
            multiplier,
        },
    ] = cfg.nodes[..]
    {
        for y in 0..h {
            for x in 0..w {
                let nb = Neighbours::get(&ch.data, w, x, y);
                let wp_pred = match &mut wp {
                    Some(s) => s.predict(x, y, &nb, None),
                    None => 0,
                };
                let v = unpack_signed(reader.read(hist, br, ctx as usize)?) as i64;
                let val = v
                    .wrapping_mul(multiplier)
                    .wrapping_add(offset)
                    .wrapping_add(predict(predictor, &nb, wp_pred));
                let val = val as i32;
                ch.data[y * w + x] = val;
                if let Some(s) = &mut wp {
                    s.update(val as i64, x, y);
                }
            }
        }
        return Ok(());
    }

    // Child indices are checked once here instead of at every step.
    if cfg.nodes.iter().any(|node| {
        matches!(node, Node::Split { left, right, .. }
            if *left as usize >= cfg.nodes.len() || *right as usize >= cfg.nodes.len())
    }) {
        return Err(invalid("MA tree child out of range"));
    }
    let nprops = 16 + 4 * num_refs.max(1);
    let mut props = vec![0i64; nprops.max(max_prop as usize + 1)];
    props[0] = idx as i64;
    props[1] = stream_id as i64;
    for y in 0..h {
        props[2] = y as i64;
        // Splits on the channel, stream and row (properties 0..=2) are
        // the same for the whole row: resolve them once.
        let mut root = 0usize;
        while let Node::Split {
            prop: prop @ 0..=2,
            value,
            left,
            right,
        } = cfg.nodes[root]
        {
            root = if props[prop as usize] as i32 > value {
                left
            } else {
                right
            } as usize;
        }
        if let Node::Leaf {
            ctx,
            predictor,
            offset,
            multiplier,
        } = cfg.nodes[root]
        {
            // One leaf for the row: no properties needed.
            for x in 0..w {
                let nb = Neighbours::get(&ch.data, w, x, y);
                let wp_pred = match &mut wp {
                    Some(s) => s.predict(x, y, &nb, None),
                    None => 0,
                };
                let v = unpack_signed(reader.read(hist, br, ctx as usize)?) as i64;
                let val = v
                    .wrapping_mul(multiplier)
                    .wrapping_add(offset)
                    .wrapping_add(predict(predictor, &nb, wp_pred));
                let val = val as i32;
                ch.data[y * w + x] = val;
                if let Some(s) = &mut wp {
                    s.update(val as i64, x, y);
                }
            }
            continue;
        }
        let mut prev_prop9 = 0i64;
        for x in 0..w {
            let nb = Neighbours::get(&ch.data, w, x, y);
            props[3] = x as i64;
            props[4] = nb.n.abs();
            props[5] = nb.w.abs();
            props[6] = nb.n;
            props[7] = nb.w;
            props[8] = nb.w - prev_prop9;
            let p9 = nb.w + nb.n - nb.nw;
            props[9] = p9;
            prev_prop9 = p9;
            props[10] = nb.w - nb.nw;
            props[11] = nb.nw - nb.n;
            props[12] = nb.n - nb.ne;
            props[13] = nb.n - nb.nn;
            props[14] = nb.w - nb.ww;
            let mut maxerr = 0i64;
            let wp_pred = match &mut wp {
                Some(s) => s.predict(x, y, &nb, Some(&mut maxerr)),
                None => 0,
            };
            props[15] = maxerr;
            for (k, &j) in refs.iter().enumerate() {
                let rc = &before[j];
                let v = rc.data[y * w + x] as i64;
                let left = if x > 0 {
                    rc.data[y * w + x - 1] as i64
                } else {
                    0
                };
                let top = if y > 0 {
                    rc.data[(y - 1) * w + x] as i64
                } else {
                    left
                };
                let topleft = if x > 0 && y > 0 {
                    rc.data[(y - 1) * w + x - 1] as i64
                } else {
                    left
                };
                let pred = clamped_gradient(left, top, topleft);
                let base = 16 + 4 * k;
                if base + 3 < props.len() {
                    props[base] = v.abs();
                    props[base + 1] = v;
                    props[base + 2] = (v - pred).abs();
                    props[base + 3] = v - pred;
                }
            }
            // Walk the tree.
            let mut n = root;
            let (ctx, predictor, offset, multiplier) = loop {
                match cfg.nodes[n] {
                    Node::Split {
                        prop,
                        value,
                        left,
                        right,
                    } => {
                        let pv = props.get(prop as usize).copied().unwrap_or(0) as i32;
                        n = if pv > value { left } else { right } as usize;
                    }
                    Node::Leaf {
                        ctx,
                        predictor,
                        offset,
                        multiplier,
                    } => break (ctx, predictor, offset, multiplier),
                }
            };
            let v = unpack_signed(reader.read(hist, br, ctx as usize)?) as i64;
            let val = v
                .wrapping_mul(multiplier)
                .wrapping_add(offset)
                .wrapping_add(predict(predictor, &nb, wp_pred));
            let val = val as i32;
            ch.data[y * w + x] = val;
            if let Some(s) = &mut wp {
                s.update(val as i64, x, y);
            }
        }
    }
    Ok(())
}

/// Decode a modular sub-bitstream into `img` (whose channels describe the
/// untransformed geometry). Channels larger than `max_chan_size` (after the
/// meta channels) are left for the group streams. With `undo`, the
/// stream's own transforms are inverted afterwards.
pub(crate) fn decode_stream(
    br: &mut BitReader,
    img: &mut Image,
    stream_id: u32,
    global: Option<&MaConfig>,
    max_chan_size: usize,
    undo: bool,
    alloc_limit: u64,
) -> Result<()> {
    if img.channels.is_empty() {
        return Ok(());
    }
    let use_global_tree = br.read_bool()?;
    img.wp = WpHeader::read(br)?;
    let nb_transforms = br.u32([val(0), val(1), bits(4, 2), bits(8, 18)])?;
    for _ in 0..nb_transforms {
        img.transforms.push(Transform::read(br)?);
    }
    for i in 0..img.transforms.len() {
        img.meta_apply(i)?;
    }
    // Every channel is allocated eventually; bound the total up front.
    let all: u64 = img
        .channels
        .iter()
        .map(|c| c.w as u64 * c.h as u64)
        .fold(0u64, u64::saturating_add);
    if all.saturating_mul(4) > alloc_limit {
        return Err(dicom_jpeg_common::Error::LimitExceeded {
            what: "modular channel bytes",
            requested: all.saturating_mul(4),
            limit: alloc_limit,
        });
    }
    // Which channels this stream codes.
    let mut coded = Vec::new();
    let mut total = 0u64;
    let mut dist_multiplier = 0usize;
    for (i, c) in img.channels.iter().enumerate() {
        if i >= img.nb_meta && (c.w > max_chan_size || c.h > max_chan_size) {
            break;
        }
        if c.w == 0 || c.h == 0 {
            continue;
        }
        total = total.saturating_add(c.w as u64 * c.h as u64);
        dist_multiplier = dist_multiplier.max(c.w);
        coded.push(i);
    }
    if total.saturating_mul(4) > alloc_limit {
        return Err(dicom_jpeg_common::Error::LimitExceeded {
            what: "modular channel bytes",
            requested: total.saturating_mul(4),
            limit: alloc_limit,
        });
    }
    if !coded.is_empty() {
        let local;
        let cfg = if use_global_tree {
            global.ok_or_else(|| invalid("global MA tree requested but absent"))?
        } else {
            let limit = (1024 + total).min(1 << 20) as usize;
            local = MaConfig::read(br, limit)?;
            &local
        };
        let (max_prop, uses_wp) = cfg.analyse();
        let mut reader = SymbolReader::new(&cfg.histograms, br, dist_multiplier as u32)?;
        for &i in &coded {
            decode_channel(
                br,
                &mut reader,
                cfg,
                img.wp,
                &mut img.channels,
                i,
                stream_id,
                max_prop,
                uses_wp,
            )?;
        }
        reader.finish()?;
    }
    for c in &mut img.channels {
        c.alloc();
    }
    if undo {
        img.undo_transforms()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn squeeze_round_trip_on_constant() {
        // A constant channel squeezed horizontally has zero residuals.
        let mut ch = Channel::new(3, 2, 1, 0);
        ch.data = vec![5; 6];
        let mut res = Channel::new(2, 2, 1, 0);
        res.data = vec![0; 4];
        inv_hsqueeze(&mut ch, &res).unwrap();
        assert_eq!(ch.w, 5);
        assert_eq!(ch.hshift, 0);
        assert!(ch.data.iter().all(|&v| v == 5));
    }

    #[test]
    fn palette_implicit_values() {
        let pal = Channel {
            w: 2,
            h: 3,
            hshift: -1,
            vshift: -1,
            data: vec![10, 11, 20, 21, 30, 31],
        };
        assert_eq!(palette_value(&pal, 1, 2, 8), 31);
        // Delta entries: index -1 -> -(first non-zero delta) per channel.
        assert_eq!(palette_value(&pal, -1, 0, 8), 0);
        assert_eq!(palette_value(&pal, -2, 0, 8), 4);
        assert_eq!(palette_value(&pal, -3, 0, 8), -4);
    }
}
