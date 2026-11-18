//! ITU-T T.87 | ISO/IEC 14495-1: JPEG-LS lossless and near-lossless.
//!
//! Supports 2..=16 bit samples, NEAR 0..=255, all three interleave modes,
//! custom LSE preset parameters, restart intervals and point transform.
//! The non-standard HP colour transforms written by CharLS (APP8 "mrfx")
//! are decoded. Mapping tables (palettes, LSE types 2/3) are not supported.

#![forbid(unsafe_code)]

mod bitio;
mod coding;
mod scan;

/// The internal kernels, exposed only for benchmarking them one by one.
/// Not a stable API.
#[cfg(feature = "kernels")]
#[doc(hidden)]
pub mod kernels {
    /// Golomb codes of 8-bit lossless coding (LIMIT 32, qbpp 8).
    const LIMIT: u32 = 32;
    const QBPP: u32 = 8;

    /// Write `values[i]` (mapped errors) as Golomb codes with parameters
    /// `ks[i]`.
    pub fn golomb_encode(values: &[i32], ks: &[u32]) -> Vec<u8> {
        let mut w = crate::bitio::BitWriter::new(Vec::new());
        for (&v, &k) in values.iter().zip(ks) {
            w.golomb(v, k, LIMIT, QBPP);
        }
        w.flush();
        w.into_out()
    }

    /// Read `out.len()` Golomb codes with parameters `ks` from `data`.
    pub fn golomb_decode(data: &[u8], ks: &[u32], out: &mut [i32]) {
        let mut r = crate::bitio::BitReader::new(data, 0);
        for (o, &k) in out.iter_mut().zip(ks) {
            *o = r.golomb(k, LIMIT, QBPP).expect("valid stream");
        }
    }
}

use bitio::BitReader;
#[cfg(feature = "encode")]
use bitio::BitWriter;
use coding::{Params, Preset};
use scan::{Interleave, ScanLayout};

#[cfg(feature = "encode")]
use dicom_jpeg_common::{AppliedTransform, Encoded};
use dicom_jpeg_common::{Codec, ColorSpace, DecodeOptions, Image, ImageInfo, Limits};
use dicom_jpeg_common::{Error, Result};

pub(crate) const CODEC: Codec = Codec::JpegLs;

fn truncated() -> Error {
    Error::Truncated { codec: CODEC }
}

#[derive(Debug, Clone)]
struct Frame {
    precision: u8,
    width: usize,
    height: usize,
    ids: Vec<u8>,
}

/// Parsed header state.
#[derive(Debug)]
struct Parser<'a> {
    data: &'a [u8],
    pos: usize,
    frame: Option<Frame>,
    preset: Preset,
    restart_interval: usize,
    /// CharLS HP colour transform (1, 2, 3), 0 = none.
    hp_transform: u8,
}

impl<'a> Parser<'a> {
    fn new(data: &'a [u8]) -> Result<Self> {
        if !data.starts_with(&[0xFF, 0xD8]) {
            return Err(Error::invalid(CODEC, "missing SOI marker"));
        }
        Ok(Parser {
            data,
            pos: 2,
            frame: None,
            preset: Preset::default(),
            restart_interval: 0,
            hp_transform: 0,
        })
    }

    fn next_marker(&mut self) -> Result<u8> {
        loop {
            let b = *self.data.get(self.pos).ok_or_else(truncated)?;
            let m = *self.data.get(self.pos + 1).ok_or_else(truncated)?;
            if b == 0xFF && m != 0xFF && m != 0x00 {
                self.pos += 2;
                return Ok(m);
            }
            self.pos += 1;
        }
    }

    fn segment(&mut self) -> Result<&'a [u8]> {
        let l = self
            .data
            .get(self.pos..self.pos + 2)
            .ok_or_else(truncated)?;
        let len = u16::from_be_bytes([l[0], l[1]]) as usize;
        if len < 2 {
            return Err(Error::invalid(CODEC, "marker segment length < 2"));
        }
        let seg = self
            .data
            .get(self.pos + 2..self.pos + len)
            .ok_or_else(truncated)?;
        self.pos += len;
        Ok(seg)
    }

    /// Handle a non-SOS marker. Returns `true` at EOI.
    fn handle(&mut self, m: u8, limits: &Limits) -> Result<bool> {
        match m {
            0xF7 => {
                let seg = self.segment()?;
                if self.frame.is_some() {
                    return Err(Error::invalid(CODEC, "multiple frame headers"));
                }
                if seg.len() < 6 {
                    return Err(truncated());
                }
                let precision = seg[0];
                let height = u16::from_be_bytes([seg[1], seg[2]]) as usize;
                let width = u16::from_be_bytes([seg[3], seg[4]]) as usize;
                let nf = seg[5] as usize;
                if !(2..=16).contains(&precision) {
                    return Err(Error::invalid(
                        CODEC,
                        format!("invalid precision {precision}"),
                    ));
                }
                if nf == 0 || seg.len() < 6 + 3 * nf {
                    return Err(Error::invalid(CODEC, "invalid component count"));
                }
                let mut ids = Vec::with_capacity(nf);
                for i in 0..nf {
                    let c = &seg[6 + 3 * i..9 + 3 * i];
                    if c[1] != 0x11 {
                        return Err(Error::unsupported(CODEC, "component subsampling"));
                    }
                    ids.push(c[0]);
                }
                let (mut width, mut height) = (width, height);
                if let Some(f) = &self.frame {
                    // Dimensions from an LSE type 4 segment seen earlier.
                    width = width.max(f.width);
                    height = height.max(f.height);
                }
                if width == 0 || height == 0 {
                    return Err(Error::unsupported(
                        CODEC,
                        "zero dimension (DNL not supported)",
                    ));
                }
                let info = ImageInfo {
                    width: width as u32,
                    height: height as u32,
                    components: nf as u16,
                    bits_per_sample: precision,
                    signed: false,
                    color_space: ColorSpace::Unknown,
                };
                info.validate(limits)?;
                self.frame = Some(Frame {
                    precision,
                    width,
                    height,
                    ids,
                });
            }
            0xF8 => {
                let seg = self.segment()?;
                match seg.first() {
                    Some(1) if seg.len() >= 11 => {
                        let v = |i: usize| u16::from_be_bytes([seg[i], seg[i + 1]]) as i32;
                        self.preset = Preset {
                            maxval: v(1),
                            t1: v(3),
                            t2: v(5),
                            t3: v(7),
                            reset: v(9),
                        };
                    }
                    Some(2 | 3) => return Err(Error::unsupported(CODEC, "mapping tables")),
                    Some(4) => return Err(Error::unsupported(CODEC, "oversize image dimensions")),
                    _ => return Err(Error::invalid(CODEC, "invalid LSE segment")),
                }
            }
            0xDD => {
                let seg = self.segment()?;
                if seg.is_empty() || seg.len() > 4 {
                    return Err(Error::invalid(CODEC, "invalid DRI segment"));
                }
                self.restart_interval = seg.iter().fold(0usize, |a, &b| (a << 8) | b as usize);
            }
            0xE8 => {
                let seg = self.segment()?;
                if seg.len() == 5 && seg.starts_with(b"mrfx") {
                    self.hp_transform = seg[4];
                    if self.hp_transform > 3 {
                        return Err(Error::unsupported(CODEC, "unknown colour transform"));
                    }
                }
            }
            0xD9 => return Ok(true),
            0xC0..=0xCF if m != 0xC4 && m != 0xC8 && m != 0xCC => {
                return Err(Error::invalid(CODEC, "T.81 JPEG codestream, not JPEG-LS"));
            }
            _ => {
                self.segment()?;
            }
        }
        Ok(false)
    }
}

/// Parse the codestream headers and describe the image.
pub fn read_info(data: &[u8]) -> Result<ImageInfo> {
    let mut p = Parser::new(data)?;
    while p.frame.is_none() {
        let m = p.next_marker()?;
        if m == 0xDA || p.handle(m, &Limits::UNLIMITED)? {
            return Err(Error::invalid(CODEC, "no frame header"));
        }
    }
    let f = p.frame.expect("frame");
    Ok(ImageInfo {
        width: f.width as u32,
        height: f.height as u32,
        components: f.ids.len() as u16,
        bits_per_sample: f.precision,
        signed: false,
        color_space: if f.ids.len() == 1 {
            ColorSpace::Gray
        } else if f.ids.len() == 3 {
            ColorSpace::Rgb
        } else {
            ColorSpace::Unknown
        },
    })
}

/// Decode a complete codestream.
pub fn decode(data: &[u8], opts: &DecodeOptions) -> Result<Image> {
    let mut p = Parser::new(data)?;
    let mut planes: Vec<Vec<u16>> = Vec::new();
    let mut done: Vec<bool> = Vec::new();
    loop {
        let m = match p.next_marker() {
            Ok(m) => m,
            Err(Error::Truncated { .. }) if !done.is_empty() && done.iter().all(|&d| d) => break,
            Err(e) => return Err(e),
        };
        if m != 0xDA {
            if p.handle(m, &opts.limits)? {
                break;
            }
            continue;
        }
        let frame = p
            .frame
            .clone()
            .ok_or_else(|| Error::invalid(CODEC, "SOS before frame header"))?;
        if planes.is_empty() {
            let n = frame.width * frame.height;
            opts.limits.check_alloc((n * frame.ids.len() * 2) as u64)?;
            planes = vec![vec![0u16; n]; frame.ids.len()];
            done = vec![false; frame.ids.len()];
        }
        let seg = p.segment()?;
        let ns = *seg.first().ok_or_else(truncated)? as usize;
        if ns == 0 || seg.len() < 1 + 2 * ns + 3 {
            return Err(Error::invalid(CODEC, "invalid scan header"));
        }
        let mut comps = Vec::with_capacity(ns);
        for i in 0..ns {
            let id = seg[1 + 2 * i];
            if seg[2 + 2 * i] != 0 {
                return Err(Error::unsupported(CODEC, "mapping tables"));
            }
            let ci = frame
                .ids
                .iter()
                .position(|&x| x == id)
                .ok_or_else(|| Error::invalid(CODEC, "scan references unknown component"))?;
            if comps.contains(&ci) {
                return Err(Error::invalid(CODEC, "component repeated in scan"));
            }
            comps.push(ci);
        }
        let near = seg[1 + 2 * ns] as i32;
        let interleave = match seg[2 + 2 * ns] {
            0 => Interleave::None,
            1 => Interleave::Line,
            2 => Interleave::Sample,
            _ => return Err(Error::invalid(CODEC, "invalid interleave mode")),
        };
        if (interleave == Interleave::None) != (ns == 1) || ns > 4 {
            return Err(Error::invalid(
                CODEC,
                "interleave mode does not match component count",
            ));
        }
        let pt = (seg[3 + 2 * ns] & 0x0F) as u32;
        if pt as u8 >= frame.precision {
            return Err(Error::invalid(CODEC, "point transform >= precision"));
        }
        let mut preset = p.preset;
        if preset.maxval == 0 && pt > 0 {
            preset.maxval = ((1u32 << (frame.precision as u32 - pt)) - 1) as i32;
        }
        let params = Params::new(frame.precision - pt as u8, near, preset)?;
        let layout = ScanLayout {
            width: frame.width,
            height: frame.height,
            interleave,
            restart_interval: p.restart_interval,
        };
        let mut reader = BitReader::new(p.data, p.pos);
        {
            let mut refs: Vec<&mut [u16]> = Vec::with_capacity(ns);
            // Collect disjoint mutable plane references in scan order.
            let mut rest: Vec<Option<&mut Vec<u16>>> = planes.iter_mut().map(Some).collect();
            for &ci in &comps {
                refs.push(rest[ci].take().expect("unique components").as_mut_slice());
            }
            scan::decode_scan(&mut reader, &params, &layout, &mut refs)?;
            if pt > 0 {
                for r in refs.iter_mut() {
                    r.iter_mut().for_each(|v| *v <<= pt);
                }
            }
        }
        for &ci in &comps {
            done[ci] = true;
        }
        p.pos = reader.end_position();
    }
    let frame = p
        .frame
        .ok_or_else(|| Error::invalid(CODEC, "no frame header"))?;
    if !done.iter().all(|&d| d) {
        return Err(Error::invalid(CODEC, "not all components were coded"));
    }
    let nc = planes.len();
    if p.hp_transform != 0 {
        if nc < 3 {
            return Err(Error::invalid(CODEC, "colour transform needs 3 components"));
        }
        inverse_hp(p.hp_transform, frame.precision, &mut planes);
    }
    let color_space = match nc {
        1 => ColorSpace::Gray,
        3 if p.hp_transform == 0 && opts.color_hint == Some(ColorSpace::YCbCr) => ColorSpace::YCbCr,
        3 => ColorSpace::Rgb,
        _ => ColorSpace::Unknown,
    };
    let info = ImageInfo {
        width: frame.width as u32,
        height: frame.height as u32,
        components: nc as u16,
        bits_per_sample: frame.precision,
        signed: false,
        color_space,
    };
    let planes: Vec<&[u16]> = planes.iter().map(Vec::as_slice).collect();
    Image::from_planes_with(info, &planes, |v| v)
}

#[allow(clippy::needless_range_loop)] // indexes three planes in lockstep
/// Inverse of the CharLS HP1/HP2/HP3 colour transforms (modulo 2^precision).
fn inverse_hp(kind: u8, precision: u8, planes: &mut [Vec<u16>]) {
    let range = 1i32 << precision;
    let mask = range - 1;
    let (half, quarter) = (range / 2, range / 4);
    let n = planes[0].len();
    for i in 0..n {
        let (v1, v2, v3) = (
            planes[0][i] as i32,
            planes[1][i] as i32,
            planes[2][i] as i32,
        );
        let (r, g, b) = match kind {
            1 => (v1 + v2 - half, v2, v3 + v2 - half),
            2 => {
                let r = (v1 + v2 - half) & mask;
                (r, v2, v3 + ((r + v2) >> 1) - half)
            }
            _ => {
                let g = v1 - ((v3 + v2) >> 2) + quarter;
                let g = g & mask;
                (v3 + g - half, g, v2 + g - half)
            }
        };
        planes[0][i] = (r & mask) as u16;
        planes[1][i] = (g & mask) as u16;
        planes[2][i] = (b & mask) as u16;
    }
}

/// JPEG-LS specific encoder parameters.
#[cfg(feature = "encode")]
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct EncoderOptions {
    /// Maximum absolute reconstruction error (0 = lossless).
    pub near: u16,
    /// Interleave mode for multi-component images: 0 = none, 1 = line, 2 = sample.
    pub interleave: u8,
    /// Restart interval in lines (0 = none).
    pub restart_interval: u16,
}

#[cfg(feature = "encode")]
impl Default for EncoderOptions {
    fn default() -> Self {
        EncoderOptions {
            near: 0,
            interleave: 2,
            restart_interval: 0,
        }
    }
}

#[cfg(feature = "encode")]
fn write_segment(out: &mut Vec<u8>, marker: u8, payload: &[u8]) {
    out.extend_from_slice(&[0xFF, marker]);
    out.extend_from_slice(&((payload.len() + 2) as u16).to_be_bytes());
    out.extend_from_slice(payload);
}

/// Encode with JPEG-LS specific parameters.
#[cfg(feature = "encode")]
pub fn encode_with(image: &Image, opts: &EncoderOptions) -> Result<Encoded> {
    let info = &image.info;
    let (w, h) = (info.width as usize, info.height as usize);
    if w > 65535 || h > 65535 {
        return Err(Error::unsupported(CODEC, "dimensions above 65535"));
    }
    let nc = info.components as usize;
    if nc > 255 {
        return Err(Error::unsupported(CODEC, "more than 255 components"));
    }
    let precision = info.bits_per_sample.max(2);
    let mask = ((1u32 << precision) - 1) as u16;
    let near = opts.near as i32;
    let params = Params::new(precision, near, Preset::default())?;
    let planes = image.planes_with(|s| s & mask);
    let interleave = match (nc, opts.interleave) {
        (1, _) | (_, 0) => Interleave::None,
        (2..=4, 2) => Interleave::Sample,
        (_, 1 | 2) => Interleave::Line,
        _ => return Err(Error::input("interleave mode must be 0, 1 or 2")),
    };

    let mut out = vec![0xFF, 0xD8];
    let mut sof = vec![precision];
    sof.extend_from_slice(&(h as u16).to_be_bytes());
    sof.extend_from_slice(&(w as u16).to_be_bytes());
    sof.push(nc as u8);
    for c in 0..nc {
        sof.extend_from_slice(&[c as u8 + 1, 0x11, 0]);
    }
    write_segment(&mut out, 0xF7, &sof);
    if opts.restart_interval > 0 {
        write_segment(&mut out, 0xDD, &opts.restart_interval.to_be_bytes());
    }
    let layout = ScanLayout {
        width: w,
        height: h,
        interleave,
        restart_interval: opts.restart_interval as usize,
    };
    let scans: Vec<Vec<usize>> = if interleave == Interleave::None {
        (0..nc).map(|c| vec![c]).collect()
    } else {
        vec![(0..nc).collect()]
    };
    let mut writer = BitWriter::new(out);
    for comps in scans {
        let mut sos = vec![comps.len() as u8];
        for &c in &comps {
            sos.extend_from_slice(&[c as u8 + 1, 0]);
        }
        sos.extend_from_slice(&[opts.near as u8, interleave as u8, 0]);
        write_segment(writer.out(), 0xDA, &sos);
        let refs: Vec<&[u16]> = comps.iter().map(|&c| planes[c].as_slice()).collect();
        scan::encode_scan(&mut writer, &params, &layout, &refs);
        writer.flush();
    }
    let mut out = writer.into_out();
    out.extend_from_slice(&[0xFF, 0xD9]);
    Ok(Encoded {
        data: out,
        transform: AppliedTransform::None,
        lossy: near > 0,
    })
}
