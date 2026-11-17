//! T.81 decoder: sequential DCT (processes 1, 2, 4 with Huffman coding) and
//! lossless (process 14).

use super::CODEC;
use super::bitreader::BitReader;
use super::color::{Plane, Sample, Upsampler};
use super::huffman::HuffmanTable;
use super::idct::idct_blocks;
use super::ycbcr::ycbcr_to_rgb;
use dicom_jpeg_common::simd::{self, Level};
use dicom_jpeg_common::{
    ColorOutput, ColorSpace, DecodeOptions, Image, ImageInfo, Limits, Samples,
};
use dicom_jpeg_common::{Error, Result};

/// Zig-zag scan position -> natural (row-major) coefficient index.
pub(crate) const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Process {
    Baseline,
    Extended,
    Lossless,
}

#[derive(Debug, Clone)]
pub(crate) struct Component {
    pub(crate) id: u8,
    pub(crate) h: usize,
    pub(crate) v: usize,
    pub(crate) tq: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct Frame {
    pub(crate) process: Process,
    pub(crate) precision: u8,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) components: Vec<Component>,
    pub(crate) hmax: usize,
    pub(crate) vmax: usize,
}

impl Frame {
    fn mcus_x(&self) -> usize {
        let unit = if self.process == Process::Lossless {
            1
        } else {
            8
        };
        self.width.div_ceil(unit * self.hmax)
    }

    fn mcus_y(&self) -> usize {
        let unit = if self.process == Process::Lossless {
            1
        } else {
            8
        };
        self.height.div_ceil(unit * self.vmax)
    }

    /// Real (downsampled) dimensions of a component.
    fn comp_size(&self, c: &Component) -> (usize, usize) {
        (
            (self.width * c.h).div_ceil(self.hmax),
            (self.height * c.v).div_ceil(self.vmax),
        )
    }
}

#[derive(Debug, Clone, Copy)]
struct ScanComponent {
    index: usize,
    td: usize,
    ta: usize,
}

#[derive(Debug)]
struct Scan {
    components: Vec<ScanComponent>,
    ss: u8,
    al: u8,
}

type TablePair<'t> = (&'t HuffmanTable, Option<&'t HuffmanTable>);

fn scan_tables<'t>(
    dc: &'t [Option<HuffmanTable>; 4],
    ac: &'t [Option<HuffmanTable>; 4],
    scan: &Scan,
    need_ac: bool,
) -> Result<Vec<TablePair<'t>>> {
    scan.components
        .iter()
        .map(|c| {
            let d = dc[c.td]
                .as_ref()
                .ok_or_else(|| Error::invalid(CODEC, "missing DC Huffman table"))?;
            let a = if need_ac {
                Some(
                    ac[c.ta]
                        .as_ref()
                        .ok_or_else(|| Error::invalid(CODEC, "missing AC Huffman table"))?,
                )
            } else {
                None
            };
            Ok((d, a))
        })
        .collect()
}

/// Parsed headers and decoding state.
#[derive(Debug)]
pub(crate) struct Decoder<'a> {
    data: &'a [u8],
    pos: usize,
    qt: [Option<[u16; 64]>; 4],
    dc: [Option<HuffmanTable>; 4],
    ac: [Option<HuffmanTable>; 4],
    restart_interval: usize,
    pub(crate) frame: Option<Frame>,
    jfif: bool,
    adobe_transform: Option<u8>,
    planes: Vec<Plane>,
    decoded: Vec<bool>,
    limits: Limits,
}

fn truncated() -> Error {
    Error::Truncated { codec: CODEC }
}

impl<'a> Decoder<'a> {
    pub(crate) fn new(data: &'a [u8], limits: Limits) -> Result<Self> {
        if !data.starts_with(&[0xFF, 0xD8]) {
            return Err(Error::invalid(CODEC, "missing SOI marker"));
        }
        Ok(Decoder {
            data,
            pos: 2,
            qt: [None; 4],
            dc: Default::default(),
            ac: Default::default(),
            restart_interval: 0,
            frame: None,
            jfif: false,
            adobe_transform: None,
            planes: Vec::new(),
            decoded: Vec::new(),
            limits,
        })
    }

    /// Find the next marker, skipping fill bytes and (leniently) garbage.
    fn next_marker(&mut self) -> Result<u8> {
        loop {
            let b = *self.data.get(self.pos).ok_or_else(truncated)?;
            if b != 0xFF {
                self.pos += 1;
                continue;
            }
            let m = *self.data.get(self.pos + 1).ok_or_else(truncated)?;
            if m == 0xFF {
                self.pos += 1;
                continue;
            }
            self.pos += 2;
            if m == 0x00 || (0xD0..=0xD7).contains(&m) {
                continue; // stray stuffing / restart marker outside a scan
            }
            return Ok(m);
        }
    }

    /// Read a marker segment payload (without the length field).
    fn segment(&mut self) -> Result<&'a [u8]> {
        let len_bytes = self
            .data
            .get(self.pos..self.pos + 2)
            .ok_or_else(truncated)?;
        let len = u16::from_be_bytes([len_bytes[0], len_bytes[1]]) as usize;
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

    /// Parse markers up to and including the frame header.
    pub(crate) fn read_headers(&mut self) -> Result<()> {
        while self.frame.is_none() {
            let m = self.next_marker()?;
            self.handle_marker(m)?;
            if m == 0xD9 {
                return Err(Error::invalid(CODEC, "no frame header before EOI"));
            }
        }
        Ok(())
    }

    fn handle_marker(&mut self, m: u8) -> Result<()> {
        match m {
            0xC0 | 0xC1 | 0xC3 => {
                let seg = self.segment()?;
                let process = match m {
                    0xC0 => Process::Baseline,
                    0xC1 => Process::Extended,
                    _ => Process::Lossless,
                };
                self.read_sof(process, seg)?;
            }
            0xC2 => return Err(Error::unsupported(CODEC, "progressive DCT (process 10/12)")),
            0xC5..=0xC7 => return Err(Error::unsupported(CODEC, "hierarchical JPEG")),
            0xC9..=0xCB | 0xCD..=0xCF => {
                return Err(Error::unsupported(CODEC, "arithmetic coding"));
            }
            0xF7 => return Err(Error::invalid(CODEC, "JPEG-LS codestream, not JPEG")),
            0xC4 => {
                let seg = self.segment()?;
                self.read_dht(seg)?;
            }
            0xDB => {
                let seg = self.segment()?;
                self.read_dqt(seg)?;
            }
            0xDD => {
                let seg = self.segment()?;
                if seg.len() < 2 {
                    return Err(truncated());
                }
                self.restart_interval = u16::from_be_bytes([seg[0], seg[1]]) as usize;
            }
            0xE0 => {
                let seg = self.segment()?;
                if seg.starts_with(b"JFIF\0") {
                    self.jfif = true;
                }
            }
            0xEE => {
                let seg = self.segment()?;
                if seg.starts_with(b"Adobe") && seg.len() >= 12 {
                    self.adobe_transform = Some(seg[11]);
                }
            }
            0xDC => {
                // DNL: height already resolved by `resolve_dnl`.
                self.segment()?;
            }
            0xD9 => {}
            0xDA => return Err(Error::invalid(CODEC, "SOS before frame header")),
            _ => {
                // APPn, COM, DHP/EXP etc.: skip.
                self.segment()?;
            }
        }
        Ok(())
    }

    fn read_sof(&mut self, process: Process, seg: &[u8]) -> Result<()> {
        if self.frame.is_some() {
            return Err(Error::invalid(CODEC, "multiple frame headers"));
        }
        if seg.len() < 6 {
            return Err(truncated());
        }
        let precision = seg[0];
        let mut height = u16::from_be_bytes([seg[1], seg[2]]) as usize;
        let width = u16::from_be_bytes([seg[3], seg[4]]) as usize;
        let nf = seg[5] as usize;
        let valid_precision = match process {
            Process::Baseline => precision == 8,
            Process::Extended => precision == 8 || precision == 12,
            Process::Lossless => (2..=16).contains(&precision),
        };
        if !valid_precision {
            return Err(Error::invalid(
                CODEC,
                format!("sample precision {precision} not allowed for {process:?}"),
            ));
        }
        if width == 0 || nf == 0 {
            return Err(Error::invalid(CODEC, "zero width or component count"));
        }
        if seg.len() < 6 + 3 * nf {
            return Err(truncated());
        }
        let mut components = Vec::with_capacity(nf);
        for i in 0..nf {
            let c = &seg[6 + 3 * i..9 + 3 * i];
            let (h, v, tq) = ((c[1] >> 4) as usize, (c[1] & 15) as usize, c[2] as usize);
            if !(1..=4).contains(&h) || !(1..=4).contains(&v) || tq > 3 {
                return Err(Error::invalid(CODEC, "invalid component specification"));
            }
            if components.iter().any(|x: &Component| x.id == c[0]) {
                return Err(Error::invalid(CODEC, "duplicate component identifier"));
            }
            components.push(Component { id: c[0], h, v, tq });
        }
        if height == 0 {
            height = self.resolve_dnl()?;
        }
        let hmax = components.iter().map(|c| c.h).max().unwrap_or(1);
        let vmax = components.iter().map(|c| c.v).max().unwrap_or(1);
        let frame = Frame {
            process,
            precision,
            width,
            height,
            components,
            hmax,
            vmax,
        };
        let info = ImageInfo {
            width: width as u32,
            height: height as u32,
            components: nf as u16,
            bits_per_sample: precision,
            signed: false,
            color_space: ColorSpace::Unknown,
        };
        info.validate(&self.limits)?;
        self.frame = Some(frame);
        Ok(())
    }

    /// Height defined by a DNL marker (B.2.5); DNL cannot occur inside
    /// entropy-coded data, so a byte search is sufficient.
    fn resolve_dnl(&self) -> Result<usize> {
        let d = self.data;
        (self.pos..d.len().saturating_sub(5))
            .find(|&i| d[i] == 0xFF && d[i + 1] == 0xDC)
            .map(|i| u16::from_be_bytes([d[i + 4], d[i + 5]]) as usize)
            .filter(|&h| h > 0)
            .ok_or_else(|| Error::invalid(CODEC, "frame height is 0 and no DNL marker found"))
    }

    fn read_dht(&mut self, mut seg: &[u8]) -> Result<()> {
        while !seg.is_empty() {
            if seg.len() < 17 {
                return Err(truncated());
            }
            let (tc, th) = (seg[0] >> 4, (seg[0] & 15) as usize);
            if tc > 1 || th > 3 {
                return Err(Error::invalid(
                    CODEC,
                    "invalid Huffman table class/destination",
                ));
            }
            let counts: [u8; 16] = seg[1..17].try_into().expect("16 bytes");
            let total: usize = counts.iter().map(|&c| c as usize).sum();
            let values = seg.get(17..17 + total).ok_or_else(truncated)?;
            let table = HuffmanTable::new(&counts, values)?;
            if tc == 0 {
                self.dc[th] = Some(table);
            } else {
                self.ac[th] = Some(table);
            }
            seg = &seg[17 + total..];
        }
        Ok(())
    }

    fn read_dqt(&mut self, mut seg: &[u8]) -> Result<()> {
        while !seg.is_empty() {
            let (pq, tq) = (seg[0] >> 4, (seg[0] & 15) as usize);
            if pq > 1 || tq > 3 {
                return Err(Error::invalid(CODEC, "invalid quantization table"));
            }
            let mut table = [0u16; 64];
            if pq == 0 {
                let v = seg.get(1..65).ok_or_else(truncated)?;
                for (t, &b) in table.iter_mut().zip(v) {
                    *t = b as u16;
                }
                seg = &seg[65..];
            } else {
                let v = seg.get(1..129).ok_or_else(truncated)?;
                for (t, b) in table.iter_mut().zip(v.as_chunks::<2>().0) {
                    *t = u16::from_be_bytes(*b);
                }
                seg = &seg[129..];
            }
            self.qt[tq] = Some(table);
        }
        Ok(())
    }

    fn read_sos(&self, seg: &[u8]) -> Result<Scan> {
        let frame = self.frame.as_ref().expect("frame parsed");
        let ns = *seg.first().ok_or_else(truncated)? as usize;
        if ns == 0 || ns > 4 || seg.len() < 1 + 2 * ns + 3 {
            return Err(Error::invalid(CODEC, "invalid scan header"));
        }
        let mut components = Vec::with_capacity(ns);
        for i in 0..ns {
            let (cs, t) = (seg[1 + 2 * i], seg[2 + 2 * i]);
            let index = frame
                .components
                .iter()
                .position(|c| c.id == cs)
                .ok_or_else(|| Error::invalid(CODEC, "scan references unknown component"))?;
            let (td, ta) = ((t >> 4) as usize, (t & 15) as usize);
            if td > 3 || ta > 3 {
                return Err(Error::invalid(CODEC, "invalid Huffman table selector"));
            }
            if components.iter().any(|c: &ScanComponent| c.index == index) {
                return Err(Error::invalid(CODEC, "component repeated in scan"));
            }
            components.push(ScanComponent { index, td, ta });
        }
        let blocks: usize = components
            .iter()
            .map(|c| frame.components[c.index].h * frame.components[c.index].v)
            .sum();
        if ns > 1 && blocks > 10 {
            return Err(Error::invalid(CODEC, "too many blocks in MCU"));
        }
        let p = 1 + 2 * ns;
        Ok(Scan {
            components,
            ss: seg[p],
            al: seg[p + 2] & 15,
        })
    }

    fn allocate_planes(&mut self) -> Result<()> {
        let frame = self.frame.as_ref().expect("frame parsed");
        let unit = if frame.process == Process::Lossless {
            1
        } else {
            8
        };
        let (mx, my) = (frame.mcus_x(), frame.mcus_y());
        let mut planes = Vec::with_capacity(frame.components.len());
        for c in &frame.components {
            let stride = mx * c.h * unit;
            let rows = my * c.v * unit;
            let bytes = (stride as u64) * (rows as u64) * 2;
            self.limits.check_alloc(bytes)?;
            let (width, height) = frame.comp_size(c);
            planes.push(Plane {
                data: vec![0; stride * rows],
                stride,
                width,
                height,
            });
        }
        self.decoded = vec![false; planes.len()];
        self.planes = planes;
        Ok(())
    }

    /// Decode all scans until EOI.
    pub(crate) fn decode_scans(&mut self) -> Result<()> {
        self.allocate_planes()?;
        loop {
            let m = match self.next_marker() {
                Ok(m) => m,
                // Missing EOI after at least one scan: be lenient like libjpeg.
                Err(Error::Truncated { .. }) if self.decoded.iter().any(|&d| d) => break,
                Err(e) => return Err(e),
            };
            match m {
                0xD9 => break,
                0xDA => {
                    let seg = self.segment()?;
                    let scan = self.read_sos(seg)?;
                    let end = if self.frame.as_ref().expect("frame").process == Process::Lossless {
                        self.decode_lossless_scan(&scan)?
                    } else {
                        self.decode_dct_scan(&scan)?
                    };
                    for c in &scan.components {
                        self.decoded[c.index] = true;
                    }
                    self.pos = end;
                }
                _ => self.handle_marker(m)?,
            }
        }
        if !self.decoded.iter().any(|&d| d) {
            return Err(Error::invalid(CODEC, "no scan data"));
        }
        Ok(())
    }

    fn decode_dct_scan(&mut self, scan: &Scan) -> Result<usize> {
        // Ss/Se/Ah/Al must be 0/63/0/0 for sequential scans, but some encoders
        // (e.g. the PVRG codec used for the NEMA WG04 set) write other values.
        // libjpeg only warns; the fields carry no information here.
        let frame = self.frame.as_ref().expect("frame").clone();
        let tables = scan_tables(&self.dc, &self.ac, scan, true)?;
        let quant: Vec<[u16; 64]> = scan
            .components
            .iter()
            .map(|c| {
                self.qt[frame.components[c.index].tq]
                    .ok_or_else(|| Error::invalid(CODEC, "missing quantization table"))
            })
            .collect::<Result<_>>()?;
        let mut reader = BitReader::new(self.data, self.pos);
        let mut pred = vec![0i32; scan.components.len()];
        let planes = &mut self.planes;
        let precision = frame.precision;
        let mut batch = IdctBatch::new(precision);

        let mut decode_block = |reader: &mut BitReader<'_>,
                                si: usize,
                                pred: &mut [i32],
                                bx: usize,
                                by: usize|
         -> Result<()> {
            let (dc, ac) = tables[si];
            let ac = ac.expect("AC table");
            let q = &quant[si];
            let coef = batch.slot();
            let s = dc.decode(reader)? as u32;
            if s > 16 {
                return Err(Error::invalid(CODEC, "DC magnitude category > 16"));
            }
            pred[si] = pred[si].wrapping_add(reader.receive_extend(s));
            coef[0] = pred[si].wrapping_mul(q[0] as i32);
            let mut k = 1;
            while k < 64 {
                if let Some((r, v)) = ac.decode_fast_ac(reader) {
                    k += r;
                    if k > 63 {
                        return Err(Error::invalid(CODEC, "AC coefficient index out of range"));
                    }
                    coef[ZIGZAG[k]] = v.wrapping_mul(q[k] as i32);
                    k += 1;
                    continue;
                }
                let rs = ac.decode(reader)?;
                let (r, s) = ((rs >> 4) as usize, (rs & 15) as u32);
                if s == 0 {
                    if r == 15 {
                        k += 16;
                        continue;
                    }
                    break;
                }
                k += r;
                if k > 63 {
                    return Err(Error::invalid(CODEC, "AC coefficient index out of range"));
                }
                coef[ZIGZAG[k]] = reader.receive_extend(s).wrapping_mul(q[k] as i32);
                k += 1;
            }
            let index = scan.components[si].index;
            let off = by * 8 * planes[index].stride + bx * 8;
            batch.commit(index, off, planes);
            Ok(())
        };

        let restart = self.restart_interval;
        let mut rst_index = 0u8;
        let mut handle_restart = |reader: &mut BitReader<'_>, pred: &mut [i32]| {
            reader.restart();
            rst_index = (rst_index + 1) & 7;
            pred.fill(0);
        };

        if scan.components.len() == 1 {
            let c = &frame.components[scan.components[0].index];
            let (cw, ch) = frame.comp_size(c);
            let (bw, bh) = (cw.div_ceil(8), ch.div_ceil(8));
            let mut n = 0usize;
            for by in 0..bh {
                for bx in 0..bw {
                    if restart > 0 && n > 0 && n.is_multiple_of(restart) {
                        handle_restart(&mut reader, &mut pred);
                    }
                    decode_block(&mut reader, 0, &mut pred, bx, by)?;
                    n += 1;
                }
            }
        } else {
            let (mx, my) = (frame.mcus_x(), frame.mcus_y());
            let mut n = 0usize;
            for my_ in 0..my {
                for mx_ in 0..mx {
                    if restart > 0 && n > 0 && n.is_multiple_of(restart) {
                        handle_restart(&mut reader, &mut pred);
                    }
                    for (si, sc) in scan.components.iter().enumerate() {
                        let c = &frame.components[sc.index];
                        for v in 0..c.v {
                            for h in 0..c.h {
                                decode_block(
                                    &mut reader,
                                    si,
                                    &mut pred,
                                    mx_ * c.h + h,
                                    my_ * c.v + v,
                                )?;
                            }
                        }
                    }
                    n += 1;
                }
            }
        }
        batch.flush(planes);
        Ok(reader.end_position())
    }

    fn decode_lossless_scan(&mut self, scan: &Scan) -> Result<usize> {
        let frame = self.frame.as_ref().expect("frame").clone();
        let predictor = scan.ss;
        let pt = scan.al as u32;
        if !(1..=7).contains(&predictor) {
            return Err(Error::invalid(
                CODEC,
                format!("invalid lossless predictor {predictor}"),
            ));
        }
        if pt as u8 >= frame.precision {
            return Err(Error::invalid(CODEC, "point transform >= precision"));
        }
        let tables = scan_tables(&self.dc, &self.ac, scan, false)?;
        let initial = 1i32 << (frame.precision as u32 - pt - 1);
        let mut reader = BitReader::new(self.data, self.pos);

        // Per scan component: row index of the first line after a (re)start,
        // and whether the next sample uses the initial prediction.
        let ns = scan.components.len();
        let mut first_line = vec![0usize; ns];
        let mut need_init = vec![true; ns];

        let planes = &mut self.planes;
        let mut decode_sample = |reader: &mut BitReader<'_>,
                                 si: usize,
                                 x: usize,
                                 y: usize,
                                 first_line: &[usize],
                                 need_init: &mut [bool]|
         -> Result<()> {
            let plane = &mut planes[scan.components[si].index];
            let s = plane.stride;
            let d = &mut plane.data;
            let pred = if need_init[si] {
                need_init[si] = false;
                initial
            } else if y == first_line[si] {
                d[y * s + x - 1] as i32
            } else if x == 0 {
                d[(y - 1) * s] as i32
            } else {
                let ra = d[y * s + x - 1] as i32;
                let rb = d[(y - 1) * s + x] as i32;
                let rc = d[(y - 1) * s + x - 1] as i32;
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
            let diff = tables[si].0.decode_diff(reader)?;
            d[y * s + x] = (pred.wrapping_add(diff) & 0xFFFF) as u16;
            Ok(())
        };

        let restart = self.restart_interval;
        let mut rst_index = 0u8;
        if ns == 1 {
            // One sample per MCU: decode each run of a row up to the next
            // restart boundary as differences first, then reconstruct it.
            let c = &frame.components[scan.components[0].index];
            let (cw, ch) = frame.comp_size(c);
            let table = tables[0].0;
            let plane = &mut planes[scan.components[0].index];
            let s = plane.stride;
            let mut diff = vec![0i32; cw];
            let (mut need_init, mut first_line) = (true, 0usize);
            let mut n = 0usize;
            for y in 0..ch {
                let mut x0 = 0;
                while x0 < cw {
                    if restart > 0 && n > 0 && n.is_multiple_of(restart) {
                        reader.restart();
                        rst_index = (rst_index + 1) & 7;
                        need_init = true;
                        first_line = y;
                    }
                    let x1 = if restart > 0 {
                        (x0 + restart - n % restart).min(cw)
                    } else {
                        cw
                    };
                    for d in &mut diff[x0..x1] {
                        *d = table.decode_diff(&mut reader)?;
                    }
                    let (above, rest) = plane.data.split_at_mut(y * s);
                    let row = &mut rest[..cw];
                    let above = if y > 0 {
                        &above[(y - 1) * s..][..cw]
                    } else {
                        &[][..]
                    };
                    // The first sample of the run: initial value after a
                    // (re)start, left neighbour on the first line, above at
                    // the start of a line.
                    let pred = if need_init {
                        need_init = false;
                        initial
                    } else if y == first_line {
                        row[x0 - 1] as i32
                    } else if x0 == 0 {
                        above[0] as i32
                    } else {
                        predict(predictor, row[x0 - 1], above[x0], above[x0 - 1])
                    };
                    row[x0] = (pred.wrapping_add(diff[x0]) & 0xFFFF) as u16;
                    let rest = (x0 + 1, x1);
                    if y == first_line {
                        undifference::<1>(row, above, &diff, rest);
                    } else {
                        match predictor {
                            1 => undifference::<1>(row, above, &diff, rest),
                            2 => undifference::<2>(row, above, &diff, rest),
                            3 => undifference::<3>(row, above, &diff, rest),
                            4 => undifference::<4>(row, above, &diff, rest),
                            5 => undifference::<5>(row, above, &diff, rest),
                            6 => undifference::<6>(row, above, &diff, rest),
                            _ => undifference::<7>(row, above, &diff, rest),
                        }
                    }
                    n += x1 - x0;
                    x0 = x1;
                }
            }
        } else {
            let (mx, my) = (frame.mcus_x(), frame.mcus_y());
            let mut n = 0usize;
            for my_ in 0..my {
                for mx_ in 0..mx {
                    if restart > 0 && n > 0 && n.is_multiple_of(restart) {
                        reader.restart();
                        rst_index = (rst_index + 1) & 7;
                        for (si, sc) in scan.components.iter().enumerate() {
                            need_init[si] = true;
                            first_line[si] = my_ * frame.components[sc.index].v;
                        }
                    }
                    for (si, sc) in scan.components.iter().enumerate() {
                        let c = &frame.components[sc.index];
                        for v in 0..c.v {
                            for h in 0..c.h {
                                let (x, y) = (mx_ * c.h + h, my_ * c.v + v);
                                decode_sample(&mut reader, si, x, y, &first_line, &mut need_init)?;
                            }
                        }
                    }
                    n += 1;
                }
            }
        }
        // Undo the point transform.
        if pt > 0 {
            let mask = ((1u32 << frame.precision) - 1) as u16;
            for sc in &scan.components {
                for v in &mut planes[sc.index].data {
                    *v = ((*v as u32) << pt) as u16 & mask;
                }
            }
        }
        Ok(reader.end_position())
    }

    /// Colour model of the stored samples.
    pub(crate) fn native_color(&self, hint: Option<ColorSpace>) -> ColorSpace {
        let frame = self.frame.as_ref().expect("frame");
        match frame.components.len() {
            1 => ColorSpace::Gray,
            3 => {
                if let Some(t) = self.adobe_transform {
                    return if t == 0 {
                        ColorSpace::Rgb
                    } else {
                        ColorSpace::YCbCr
                    };
                }
                if self.jfif {
                    return ColorSpace::YCbCr;
                }
                let ids: Vec<u8> = frame.components.iter().map(|c| c.id).collect();
                if ids == b"RGB" {
                    return ColorSpace::Rgb;
                }
                // Subsampled chroma only makes sense for YCbCr, whatever the
                // (DICOM) hint claims.
                let c = &frame.components;
                if c[1..].iter().any(|x| (x.h, x.v) != (c[0].h, c[0].v)) {
                    return ColorSpace::YCbCr;
                }
                match hint {
                    Some(h @ (ColorSpace::Rgb | ColorSpace::YCbCr)) => h,
                    _ if frame.process == Process::Lossless => ColorSpace::Rgb,
                    _ => ColorSpace::YCbCr,
                }
            }
            _ => ColorSpace::Unknown,
        }
    }

    /// Assemble the decoded planes into an interleaved image.
    pub(crate) fn into_image(self, opts: &DecodeOptions) -> Result<Image> {
        let color = self.native_color(opts.color_hint);
        let frame = self.frame.as_ref().expect("frame");
        let nc = frame.components.len();
        let convert =
            nc == 3 && color == ColorSpace::YCbCr && opts.color_output == ColorOutput::Rgb;
        let info = ImageInfo {
            width: frame.width as u32,
            height: frame.height as u32,
            components: nc as u16,
            bits_per_sample: frame.precision,
            signed: false,
            color_space: if convert { ColorSpace::Rgb } else { color },
        };
        let samples = if frame.precision <= 8 {
            Samples::U8(self.assemble(convert))
        } else {
            Samples::U16(self.assemble(convert))
        };
        Image::new(info, samples)
    }

    /// One pass over the planes: per output row, upsample each component,
    /// then convert to RGB or interleave straight into the output.
    fn assemble<S: Sample>(&self, convert: bool) -> Vec<S> {
        let frame = self.frame.as_ref().expect("frame");
        let (w, h) = (frame.width, frame.height);
        let nc = frame.components.len();
        let level = simd::level();
        let mut ups: Vec<Upsampler> = frame
            .components
            .iter()
            .zip(&self.planes)
            .map(|(c, p)| {
                let factors = (frame.hmax, frame.vmax);
                Upsampler::new(p, (c.h, c.v), factors, w, level, frame.precision)
            })
            .collect();
        let mut out = vec![S::default(); w * h * nc];
        // Converted RGB rows.
        let mut rgb = if convert {
            vec![vec![0u16; w]; 3]
        } else {
            Vec::new()
        };
        for (y, dst) in out.chunks_exact_mut(w * nc).enumerate() {
            if nc == 1 {
                let src = ups[0].row(&self.planes[0], y);
                for (d, &s) in dst.iter_mut().zip(src) {
                    *d = S::from_u16(s);
                }
                continue;
            }
            if nc == 3 {
                let [u0, u1, u2] = &mut ups[..] else {
                    unreachable!("three components")
                };
                let (p0, p1, p2) = (&self.planes[0], &self.planes[1], &self.planes[2]);
                let rows = [
                    &u0.row(p0, y)[..w],
                    &u1.row(p1, y)[..w],
                    &u2.row(p2, y)[..w],
                ];
                let [r, g, b] = if convert {
                    let [r, g, b] = &mut rgb[..] else {
                        unreachable!("three rows")
                    };
                    ycbcr_to_rgb(level, rows, frame.precision, [r, g, b]);
                    [&r[..], &g[..], &b[..]]
                } else {
                    rows
                };
                S::interleave3(level, r, g, b, dst);
                continue;
            }
            for (ci, (up, plane)) in ups.iter_mut().zip(&self.planes).enumerate() {
                let src = up.row(plane, y);
                for (d, &s) in dst.iter_mut().skip(ci).step_by(nc).zip(src) {
                    *d = S::from_u16(s);
                }
            }
        }
        out
    }
}

/// Lossless prediction (H.1.2.1, Table H.1) from the left (`ra`), above
/// (`rb`) and above-left (`rc`) samples.
#[inline(always)]
fn predict(predictor: u8, ra: u16, rb: u16, rc: u16) -> i32 {
    let (ra, rb, rc) = (ra as i32, rb as i32, rc as i32);
    match predictor {
        1 => ra,
        2 => rb,
        3 => rc,
        4 => ra + rb - rc,
        5 => ra + ((rb - rc) >> 1),
        6 => rb + ((ra - rc) >> 1),
        _ => (ra + rb) >> 1,
    }
}

/// Reconstruct `row[x0..x1]` (`x0 >= 1`) from the differences with
/// predictor `P`; `above` is the previous row (unused by predictor 1).
#[inline(always)]
fn undifference<const P: u8>(
    row: &mut [u16],
    above: &[u16],
    diff: &[i32],
    (x0, x1): (usize, usize),
) {
    for x in x0..x1 {
        let (rb, rc) = if P == 1 {
            (0, 0)
        } else {
            (above[x], above[x - 1])
        };
        let pred = predict(P, row[x - 1], rb, rc);
        row[x] = (pred.wrapping_add(diff[x]) & 0xFFFF) as u16;
    }
}

/// Dequantised blocks waiting for the inverse DCT, which runs on batches
/// (for SIMD) and copies the results into the planes.
struct IdctBatch {
    level: Level,
    precision: u8,
    len: usize,
    coefs: Vec<[i32; 64]>,
    out: Vec<[u16; 64]>,
    /// Plane index and sample offset of each block.
    dst: Vec<(usize, usize)>,
}

impl IdctBatch {
    const SIZE: usize = 16;

    fn new(precision: u8) -> Self {
        IdctBatch {
            level: simd::level(),
            precision,
            len: 0,
            coefs: vec![[0; 64]; Self::SIZE],
            out: vec![[0; 64]; Self::SIZE],
            dst: vec![(0, 0); Self::SIZE],
        }
    }

    /// The zeroed coefficients of the next block.
    fn slot(&mut self) -> &mut [i32; 64] {
        let c = &mut self.coefs[self.len];
        c.fill(0);
        c
    }

    /// Queue the block in `slot()` for the 8 x 8 area at `off` in `planes[plane]`.
    fn commit(&mut self, plane: usize, off: usize, planes: &mut [Plane]) {
        self.dst[self.len] = (plane, off);
        self.len += 1;
        if self.len == Self::SIZE {
            self.flush(planes);
        }
    }

    fn flush(&mut self, planes: &mut [Plane]) {
        let n = self.len;
        idct_blocks(
            self.level,
            &self.coefs[..n],
            self.precision,
            &mut self.out[..n],
        );
        for (block, &(p, off)) in self.out[..n].iter().zip(&self.dst[..n]) {
            let plane = &mut planes[p];
            let stride = plane.stride;
            for (r, row) in block.as_chunks::<8>().0.iter().enumerate() {
                plane.data[off + r * stride..][..8].copy_from_slice(row);
            }
        }
        self.len = 0;
    }
}
