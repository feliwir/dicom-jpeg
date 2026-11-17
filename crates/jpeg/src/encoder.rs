//! T.81 encoder: baseline / extended sequential DCT (8 and 12 bit) and
//! lossless process 14, all with per-image optimal Huffman tables.

use super::CODEC;
use super::decoder::ZIGZAG;
use super::fdct::fdct_block;
use super::huffman_encode::{BitWriter, EncodingTable, category, magnitude_bits};
use dicom_jpeg_common::{AppliedTransform, ColorSpace, Encoded, Image, Samples};
use dicom_jpeg_common::{Error, Result};

/// Chroma subsampling used when encoding colour images lossily.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ChromaSubsampling {
    /// No subsampling.
    S444,
    /// Horizontal 2:1 (DICOM `YBR_FULL_422`).
    #[default]
    S422,
    /// Horizontal and vertical 2:1.
    S420,
}

/// The T.81 coding process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Process {
    /// Baseline sequential DCT (process 1): 8 bits, standard Huffman tables.
    #[default]
    Baseline,
    /// Extended sequential DCT (processes 2 and 4): 8 or 12 bits.
    Extended,
    /// Lossless (process 14) with [`EncoderOptions::predictor`].
    Lossless,
}

/// JPEG specific encoder parameters.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct EncoderOptions {
    /// Coding process.
    pub process: Process,
    /// Quality 1..=100 for DCT-based coding (IJG scaling of the Annex K tables).
    pub quality: u8,
    /// Chroma subsampling for RGB / YCbCr input in DCT-based coding.
    pub subsampling: ChromaSubsampling,
    /// Restart interval in MCUs (0 = none).
    pub restart_interval: u16,
    /// Lossless predictor 1..=7 (1 is "selection value 1").
    pub predictor: u8,
    /// Lossless point transform (0 = mathematically lossless).
    pub point_transform: u8,
}

impl Default for EncoderOptions {
    fn default() -> Self {
        EncoderOptions {
            process: Process::Baseline,
            quality: 90,
            subsampling: ChromaSubsampling::default(),
            restart_interval: 0,
            predictor: 1,
            point_transform: 0,
        }
    }
}

/// Annex K.1 luminance quantization table (natural order).
const STD_LUMA_Q: [u16; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
    92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];

/// Annex K.1 chrominance quantization table (natural order).
const STD_CHROMA_Q: [u16; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
];

/// IJG quality scaling; returns the table in natural order.
fn scaled_table(base: &[u16; 64], quality: u8, max: u32) -> [u16; 64] {
    let q = quality.clamp(1, 100) as u32;
    let scale = if q < 50 { 5000 / q } else { 200 - 2 * q };
    base.map(|v| ((v as u32 * scale + 50) / 100).clamp(1, max) as u16)
}

/// Callback receiving (pending restart marker, component index, block).
type BlockVisitor<'a> = dyn FnMut(Option<u8>, usize, &Block) + 'a;

/// Quantized coefficients of a block (zig-zag order) with a bit set for
/// each non-zero AC coefficient.
struct Block {
    zz: [i16; 64],
    ac: u64,
}

struct Component {
    id: u8,
    h: usize,
    v: usize,
    tq: u8,
    table: u8,
    /// Quantized blocks, row-major over the padded block grid.
    blocks: Vec<Block>,
    blocks_w: usize,
}

/// Copy the samples of row `y` into the full-resolution planes (row stride
/// `pw`), converting RGB to YCbCr with libjpeg's fixed-point arithmetic
/// (jccolor.c) when `rgb`.
fn fill_row<S: Copy + Into<u32>>(
    src: &[S],
    planes: &mut [Vec<u16>],
    y: usize,
    pw: usize,
    rgb: bool,
    precision: u8,
) {
    let nc = planes.len();
    let w = src.len() / nc;
    let off = y * pw;
    if rgb {
        const SCALEBITS: u32 = 16;
        let fix = |x: f64| (x * 65536.0 + 0.5) as i32;
        let half = 1i32 << (SCALEBITS - 1);
        let offset = (1i32 << (precision - 1)) << SCALEBITS;
        let (yr, yg, yb) = (fix(0.29900), fix(0.58700), fix(0.11400));
        let (cbr, cbg, cbb) = (fix(0.16874), fix(0.33126), fix(0.5));
        let (crr, crg, crb) = (fix(0.5), fix(0.41869), fix(0.08131));
        let [p0, p1, p2] = planes else {
            unreachable!("three components")
        };
        let rows = p0[off..off + w]
            .iter_mut()
            .zip(&mut p1[off..off + w])
            .zip(&mut p2[off..off + w]);
        for (((y, cb), cr), px) in rows.zip(src.as_chunks::<3>().0) {
            let (r, g, b) = (
                px[0].into() as i32,
                px[1].into() as i32,
                px[2].into() as i32,
            );
            *y = ((yr * r + yg * g + yb * b + half) >> SCALEBITS) as u16;
            *cb = ((-cbr * r - cbg * g + cbb * b + offset + half - 1) >> SCALEBITS) as u16;
            *cr = ((crr * r - crg * g - crb * b + offset + half - 1) >> SCALEBITS) as u16;
        }
    } else {
        for (c, plane) in planes.iter_mut().enumerate() {
            for (d, &s) in plane[off..off + w]
                .iter_mut()
                .zip(src.iter().skip(c).step_by(nc))
            {
                *d = s.into() as u16;
            }
        }
    }
}

/// The image as full-resolution planes in the coded colour space, padded to
/// `pw` x `ph` by edge replication.
fn full_planes(image: &Image, precision: u8, rgb: bool, pw: usize, ph: usize) -> Vec<Vec<u16>> {
    let (w, h) = (image.info.width as usize, image.info.height as usize);
    let nc = image.info.components as usize;
    let mut planes = vec![vec![0u16; pw * ph]; nc];
    for y in 0..h {
        let r = y * w * nc..(y + 1) * w * nc;
        match &image.samples {
            Samples::U8(s) => fill_row(&s[r], &mut planes, y, pw, rgb, precision),
            Samples::U16(s) => fill_row(&s[r], &mut planes, y, pw, rgb, precision),
        }
    }
    for plane in &mut planes {
        for y in 0..h {
            let row = &mut plane[y * pw..(y + 1) * pw];
            let last = row[w - 1];
            row[w..].fill(last);
        }
        let (done, rest) = plane.split_at_mut(h * pw);
        let last = &done[(h - 1) * pw..];
        for row in rest.chunks_exact_mut(pw) {
            row.copy_from_slice(last);
        }
    }
    planes
}

/// Downsample a padded `pw`-wide plane by (`fx`, `fy`) with IJG's
/// alternating-bias averaging (jcsample.c: bias 0,1 for 2 samples, 1,2 for 4).
fn downsample(full: Vec<u16>, pw: usize, fx: usize, fy: usize) -> Vec<u16> {
    let mut out = Vec::with_capacity(full.len() / (fx * fy));
    match (fx, fy) {
        (1, 1) => return full,
        (2, 1) => {
            for row in full.chunks_exact(pw) {
                out.extend(row.as_chunks::<4>().0.iter().flat_map(|p| {
                    [
                        ((p[0] as u32 + p[1] as u32) >> 1) as u16,
                        ((p[2] as u32 + p[3] as u32 + 1) >> 1) as u16,
                    ]
                }));
            }
        }
        (2, 2) => {
            for rows in full.chunks_exact(2 * pw) {
                let (a, b) = rows.split_at(pw);
                let pairs = a.as_chunks::<4>().0.iter().zip(b.as_chunks::<4>().0);
                out.extend(pairs.flat_map(|(a, b)| {
                    let s0 = a[0] as u32 + a[1] as u32 + b[0] as u32 + b[1] as u32;
                    let s1 = a[2] as u32 + a[3] as u32 + b[2] as u32 + b[3] as u32;
                    [((s0 + 1) >> 2) as u16, ((s1 + 2) >> 2) as u16]
                }));
            }
        }
        _ => unreachable!("only 4:2:2 and 4:2:0 subsampling"),
    }
    out
}

/// A divisor with its reciprocal for exact division by multiplication
/// (Lemire et al., "Faster remainder by direct computation"): for any
/// 32-bit `n`, `n / d == (m * n) >> 64` with `m = 2^64 / d + 1`.
#[derive(Clone, Copy)]
struct Divisor {
    half: u32,
    m: u64,
}

impl Divisor {
    fn new(d: u32) -> Self {
        Divisor {
            half: d >> 1,
            m: u64::MAX / d as u64 + 1,
        }
    }

    /// libjpeg quantisation: round half away from zero.
    #[inline(always)]
    fn quantize(self, c: i32) -> i32 {
        let a = c.unsigned_abs() + self.half;
        let v = ((self.m as u128 * a as u128) >> 64) as i32;
        if c < 0 { -v } else { v }
    }
}

fn write_segment(out: &mut Vec<u8>, marker: u8, payload: &[u8]) {
    out.extend_from_slice(&[0xFF, marker]);
    out.extend_from_slice(&((payload.len() + 2) as u16).to_be_bytes());
    out.extend_from_slice(payload);
}

fn write_sof(
    out: &mut Vec<u8>,
    marker: u8,
    precision: u8,
    w: usize,
    h: usize,
    comps: &[(u8, usize, usize, u8)],
) {
    let mut p = vec![precision];
    p.extend_from_slice(&(h as u16).to_be_bytes());
    p.extend_from_slice(&(w as u16).to_be_bytes());
    p.push(comps.len() as u8);
    for &(id, ch, cv, tq) in comps {
        p.extend_from_slice(&[id, ((ch as u8) << 4) | cv as u8, tq]);
    }
    write_segment(out, marker, &p);
}

fn check_dimensions(image: &Image) -> Result<(usize, usize)> {
    let (w, h) = (image.info.width as usize, image.info.height as usize);
    if w > 65535 || h > 65535 {
        return Err(Error::unsupported(CODEC, "dimensions above 65535"));
    }
    Ok((w, h))
}

/// Encode with JPEG specific parameters.
pub fn encode_with(image: &Image, opts: &EncoderOptions) -> Result<Encoded> {
    match opts.process {
        Process::Baseline | Process::Extended => encode_dct(opts.process, image, opts),
        Process::Lossless => encode_lossless(
            image,
            opts.predictor,
            opts.point_transform,
            opts.restart_interval,
        ),
    }
}

fn encode_dct(process: Process, image: &Image, opts: &EncoderOptions) -> Result<Encoded> {
    let (w, h) = check_dimensions(image)?;
    let bits = image.info.bits_per_sample;
    let precision = match (process, bits) {
        (_, 1..=8) => 8,
        (Process::Extended, 9..=12) => 12,
        (Process::Baseline, _) => {
            return Err(Error::unsupported(
                CODEC,
                "baseline JPEG requires at most 8 bits per sample",
            ));
        }
        _ => {
            return Err(Error::unsupported(
                CODEC,
                "lossy JPEG supports at most 12 bits per sample",
            ));
        }
    };
    if image.info.signed {
        return Err(Error::unsupported(CODEC, "signed samples in lossy JPEG"));
    }
    let nc = image.info.components as usize;
    if nc > 4 {
        return Err(Error::unsupported(CODEC, "more than 4 components"));
    }
    let color = image.info.color_space;
    let is_color = nc == 3 && matches!(color, ColorSpace::Rgb | ColorSpace::YCbCr);
    let (hmax, vmax) = match (is_color, opts.subsampling) {
        (true, ChromaSubsampling::S422) => (2, 1),
        (true, ChromaSubsampling::S420) => (2, 2),
        _ => (1, 1),
    };
    let (mcus_x, mcus_y) = (w.div_ceil(8 * hmax), h.div_ceil(8 * vmax));
    let (pw, ph) = (mcus_x * 8 * hmax, mcus_y * 8 * vmax);
    // Padded full-resolution planes in the coded colour space.
    let planes = full_planes(
        image,
        precision,
        nc == 3 && color == ColorSpace::Rgb,
        pw,
        ph,
    );

    let qmax = if precision == 8 { 255 } else { 32767 };
    let quality = opts.quality;
    let qtables = [
        scaled_table(&STD_LUMA_Q, quality, qmax),
        scaled_table(&STD_CHROMA_Q, quality, qmax),
    ];
    let coef_max: i32 = if precision == 8 { 1023 } else { 16383 };

    let mut comps: Vec<Component> = Vec::with_capacity(nc);
    for (ci, plane) in planes.into_iter().enumerate() {
        let chroma = is_color && ci > 0;
        let (ch, cv) = if chroma { (1, 1) } else { (hmax, vmax) };
        let (fx, fy) = (hmax / ch, vmax / cv);
        let data = downsample(plane, pw, fx, fy);
        let (cw, chh) = (pw / fx, ph / fy);
        let (bw, bh) = (cw / 8, chh / 8);
        let table = if chroma { 1 } else { 0 };
        // libjpeg quantization: divisor 8*q (the FDCT output is scaled by
        // 8), in zig-zag order.
        let q = &qtables[table as usize];
        let div: [Divisor; 64] = std::array::from_fn(|k| Divisor::new(q[ZIGZAG[k]] as u32 * 8));
        let mut blocks = Vec::with_capacity(bw * bh);
        let mut coef = [0i32; 64];
        for by in 0..bh {
            for bx in 0..bw {
                fdct_block(&data[by * 8 * cw + bx * 8..], cw, precision, &mut coef);
                let mut zz = [0i16; 64];
                let mut nonzero = 0u64;
                for k in 0..64 {
                    let v = div[k].quantize(coef[ZIGZAG[k]]);
                    zz[k] = v.clamp(-coef_max, coef_max) as i16;
                    nonzero |= ((v != 0) as u64) << k;
                }
                blocks.push(Block {
                    zz,
                    ac: nonzero & !1,
                });
            }
        }
        comps.push(Component {
            id: ci as u8 + 1,
            h: ch,
            v: cv,
            tq: table,
            table,
            blocks,
            blocks_w: bw,
        });
    }

    // MCU visiting order shared by both passes.
    let restart = opts.restart_interval as usize;
    let for_each_mcu = |f: &mut BlockVisitor<'_>| {
        let mut n = 0usize;
        let mut rst = 0u8;
        for my in 0..mcus_y {
            for mx in 0..mcus_x {
                let marker = if restart > 0 && n > 0 && n.is_multiple_of(restart) {
                    let r = rst;
                    rst = (rst + 1) & 7;
                    Some(r)
                } else {
                    None
                };
                let mut first = true;
                for (ci, c) in comps.iter().enumerate() {
                    for v in 0..c.v {
                        for hh in 0..c.h {
                            let b = &c.blocks[(my * c.v + v) * c.blocks_w + mx * c.h + hh];
                            f(if first { marker } else { None }, ci, b);
                            first = false;
                        }
                    }
                }
                n += 1;
            }
        }
    };

    // Pass 1: statistics.
    let ntables = if is_color { 2 } else { 1 };
    let mut dc_freq = vec![[0u64; 256]; ntables];
    let mut ac_freq = vec![[0u64; 256]; ntables];
    {
        let mut pred = vec![0i32; nc];
        for_each_mcu(&mut |marker, ci, b| {
            if marker.is_some() {
                pred.fill(0);
            }
            let t = comps[ci].table as usize;
            let diff = b.zz[0] as i32 - pred[ci];
            pred[ci] = b.zz[0] as i32;
            dc_freq[t][category(diff) as usize] += 1;
            let (mut mask, mut last) = (b.ac, 0);
            while mask != 0 {
                let k = mask.trailing_zeros();
                let mut run = k - last - 1;
                while run > 15 {
                    ac_freq[t][0xF0] += 1;
                    run -= 16;
                }
                ac_freq[t][(run << 4) as usize | category(b.zz[k as usize] as i32) as usize] += 1;
                last = k;
                mask &= mask - 1;
            }
            if last < 63 {
                ac_freq[t][0] += 1;
            }
        });
    }
    let dc_tables: Vec<EncodingTable> = dc_freq.iter().map(EncodingTable::optimal).collect();
    let ac_tables: Vec<EncodingTable> = ac_freq.iter().map(EncodingTable::optimal).collect();

    // Headers.
    let mut out = vec![0xFF, 0xD8];
    if is_color {
        write_segment(
            &mut out,
            0xE0,
            b"JFIF\0\x01\x01\x00\x00\x01\x00\x01\x00\x00",
        );
    }
    for (t, q) in qtables.iter().enumerate().take(ntables) {
        let mut p = Vec::new();
        if q.iter().all(|&v| v <= 255) {
            p.push(t as u8);
            p.extend(ZIGZAG.iter().map(|&n| q[n] as u8));
        } else {
            p.push(0x10 | t as u8);
            for &n in &ZIGZAG {
                p.extend_from_slice(&q[n].to_be_bytes());
            }
        }
        write_segment(&mut out, 0xDB, &p);
    }
    let sof = if process == Process::Baseline {
        0xC0
    } else {
        0xC1
    };
    let spec: Vec<_> = comps.iter().map(|c| (c.id, c.h, c.v, c.tq)).collect();
    write_sof(&mut out, sof, precision, w, h, &spec);
    let mut dht = Vec::new();
    for t in 0..ntables {
        dc_tables[t].write_dht(0, t as u8, &mut dht);
        ac_tables[t].write_dht(1, t as u8, &mut dht);
    }
    write_segment(&mut out, 0xC4, &dht);
    if restart > 0 {
        write_segment(&mut out, 0xDD, &(restart as u16).to_be_bytes());
    }
    let mut sos = vec![nc as u8];
    for c in &comps {
        sos.extend_from_slice(&[c.id, (c.table << 4) | c.table]);
    }
    sos.extend_from_slice(&[0, 63, 0]);
    write_segment(&mut out, 0xDA, &sos);

    // Pass 2: entropy coding.
    let mut wr = BitWriter::new(out);
    let mut pred = vec![0i32; nc];
    for_each_mcu(&mut |marker, ci, b| {
        if let Some(r) = marker {
            wr.restart(r);
            pred.fill(0);
        }
        let t = comps[ci].table as usize;
        let (dc, ac) = (&dc_tables[t], &ac_tables[t]);
        let diff = b.zz[0] as i32 - pred[ci];
        pred[ci] = b.zz[0] as i32;
        let s = category(diff);
        dc.emit(&mut wr, s as u8);
        wr.write(magnitude_bits(diff, s), s);
        let (mut mask, mut last) = (b.ac, 0);
        while mask != 0 {
            let k = mask.trailing_zeros();
            let mut run = k - last - 1;
            while run > 15 {
                ac.emit(&mut wr, 0xF0);
                run -= 16;
            }
            let c = b.zz[k as usize] as i32;
            let s = category(c);
            ac.emit(&mut wr, ((run << 4) | s) as u8);
            wr.write(magnitude_bits(c, s), s);
            last = k;
            mask &= mask - 1;
        }
        if last < 63 {
            ac.emit(&mut wr, 0);
        }
    });
    wr.flush();
    let mut out = wr.out;
    out.extend_from_slice(&[0xFF, 0xD9]);

    let transform = if nc == 3 && color == ColorSpace::Rgb {
        AppliedTransform::YCbCr {
            h: hmax as u8,
            v: vmax as u8,
        }
    } else {
        AppliedTransform::None
    };
    Ok(Encoded {
        data: out,
        transform,
        lossy: true,
    })
}

/// Lossless (process 14) encoding.
fn encode_lossless(image: &Image, predictor: u8, pt: u8, restart: u16) -> Result<Encoded> {
    let (w, h) = check_dimensions(image)?;
    if !(1..=7).contains(&predictor) {
        return Err(Error::input("lossless predictor must be within 1..=7"));
    }
    let precision = image.info.bits_per_sample.max(2);
    if pt >= precision {
        return Err(Error::input("point transform must be below the precision"));
    }
    let nc = image.info.components as usize;
    if nc > 4 {
        return Err(Error::unsupported(CODEC, "more than 4 components"));
    }
    let mask = ((1u32 << precision) - 1) as u16;
    let plane = |c: usize| -> Vec<u16> {
        match &image.samples {
            Samples::U8(s) => s
                .iter()
                .skip(c)
                .step_by(nc)
                .map(|&v| (v as u16 & mask) >> pt)
                .collect(),
            Samples::U16(s) => s
                .iter()
                .skip(c)
                .step_by(nc)
                .map(|&v| (v & mask) >> pt)
                .collect(),
        }
    };
    let planes: Vec<Vec<u16>> = (0..nc).map(plane).collect();
    let initial = 1i32 << (precision - pt - 1);
    let restart = restart as usize;

    // Differences in coding order (one interleaved scan, H = V = 1).
    let n = w * h;
    let mut diffs = vec![0u16; n * nc];
    for (ci, p) in planes.iter().enumerate() {
        let mut first_line = 0usize;
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let restarted = restart > 0 && i > 0 && i % restart == 0;
                if restarted {
                    first_line = y;
                }
                let pred = if i == 0 || restarted {
                    initial
                } else if y == first_line {
                    p[i - 1] as i32
                } else if x == 0 {
                    p[i - w] as i32
                } else {
                    let (ra, rb, rc) = (p[i - 1] as i32, p[i - w] as i32, p[i - w - 1] as i32);
                    match predictor {
                        1 => ra,
                        2 => rb,
                        3 => rc,
                        4 => ra + rb - rc,
                        5 => ra + ((rb - rc) >> 1),
                        6 => rb + ((ra - rc) >> 1),
                        _ => (ra + rb) >> 1,
                    }
                };
                diffs[i * nc + ci] = (p[i] as i32).wrapping_sub(pred) as u16;
            }
        }
    }
    let ssss = |d: u16| -> (u32, i32) {
        if d == 0x8000 {
            (16, 0)
        } else {
            let sd = d as i16 as i32;
            (category(sd), sd)
        }
    };
    let mut freq = [0u64; 256];
    for &d in &diffs {
        freq[ssss(d).0 as usize] += 1;
    }
    let table = EncodingTable::optimal(&freq);

    let mut out = vec![0xFF, 0xD8];
    let rgb = nc == 3 && image.info.color_space == ColorSpace::Rgb;
    let ids: Vec<u8> = if rgb {
        b"RGB".to_vec()
    } else {
        (1..=nc as u8).collect()
    };
    let spec: Vec<_> = ids.iter().map(|&id| (id, 1, 1, 0)).collect();
    write_sof(&mut out, 0xC3, precision, w, h, &spec);
    let mut dht = Vec::new();
    table.write_dht(0, 0, &mut dht);
    write_segment(&mut out, 0xC4, &dht);
    if restart > 0 {
        write_segment(&mut out, 0xDD, &(restart as u16).to_be_bytes());
    }
    let mut sos = vec![nc as u8];
    for &id in &ids {
        sos.extend_from_slice(&[id, 0x00]);
    }
    sos.extend_from_slice(&[predictor, 0, pt]);
    write_segment(&mut out, 0xDA, &sos);

    let mut wr = BitWriter::new(out);
    let mut rst = 0u8;
    for i in 0..n {
        if restart > 0 && i > 0 && i % restart == 0 {
            wr.restart(rst);
            rst = (rst + 1) & 7;
        }
        for ci in 0..nc {
            let (s, v) = ssss(diffs[i * nc + ci]);
            table.emit(&mut wr, s as u8);
            if s < 16 {
                wr.write(magnitude_bits(v, s), s);
            }
        }
    }
    wr.flush();
    let mut out = wr.out;
    out.extend_from_slice(&[0xFF, 0xD9]);
    Ok(Encoded {
        data: out,
        transform: AppliedTransform::None,
        lossy: pt > 0,
    })
}
