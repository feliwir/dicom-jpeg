//! ISO/IEC 18181: JPEG XL.
//!
//! The decoder handles modular (lossless and lossy) and VarDCT frames with
//! XYB, the restoration filters and sRGB-family output. ICC profiles,
//! premultiplied alpha, patches, splines, noise, upsampling, blending and
//! JPEG reconstruction are reported as unsupported. The encoder writes
//! lossless modular codestreams.

#![forbid(unsafe_code)]

mod bits;
#[cfg(feature = "encode")]
mod encode;

#[cfg(feature = "encode")]
pub use encode::encode_lossless;
mod entropy;
mod frame;
mod headers;
mod modular;
mod render;
mod vardct;

use dicom_jpeg_common::{Codec, ColorSpace, DecodeOptions, Image, ImageInfo};
use dicom_jpeg_common::{Error, Result};

use bits::{BitReader, invalid};
use frame::{FrameHeader, Toc};
use headers::ImageHeader;
use modular::{Channel, MaConfig};

pub(crate) const CODEC: Codec = Codec::JpegXl;

/// Extract the codestream from a bare codestream or an ISO BMFF container
/// (ISO/IEC 18181-2): the concatenated `jxlc` / `jxlp` payloads.
pub(crate) fn codestream(data: &[u8]) -> Result<std::borrow::Cow<'_, [u8]>> {
    if data.starts_with(&[0xFF, 0x0A]) {
        return Ok(std::borrow::Cow::Borrowed(data));
    }
    const SIG: [u8; 12] = [
        0, 0, 0, 0x0C, b'J', b'X', b'L', b' ', 0x0D, 0x0A, 0x87, 0x0A,
    ];
    if !data.starts_with(&SIG) {
        return Err(invalid("not a JPEG XL codestream or container"));
    }
    let mut pos = SIG.len();
    let mut out: Option<Vec<u8>> = None;
    while pos + 8 <= data.len() {
        let size32 = u32::from_be_bytes(data[pos..pos + 4].try_into().expect("4 bytes"));
        let ty = &data[pos + 4..pos + 8];
        let (header, size) = match size32 {
            0 => (8usize, (data.len() - pos) as u64),
            1 => {
                if pos + 16 > data.len() {
                    return Err(Error::Truncated { codec: CODEC });
                }
                let s = u64::from_be_bytes(data[pos + 8..pos + 16].try_into().expect("8 bytes"));
                (16, s)
            }
            s => (8, s as u64),
        };
        if size < header as u64 {
            return Err(invalid("invalid box size"));
        }
        let end = (pos as u64).saturating_add(size).min(data.len() as u64) as usize;
        let payload = &data[pos + header..end];
        match ty {
            b"jxlc" => out.get_or_insert_with(Vec::new).extend_from_slice(payload),
            b"jxlp" => {
                if payload.len() < 4 {
                    return Err(invalid("short jxlp box"));
                }
                out.get_or_insert_with(Vec::new)
                    .extend_from_slice(&payload[4..]);
            }
            _ => {}
        }
        pos = end;
    }
    out.map(std::borrow::Cow::Owned)
        .ok_or_else(|| invalid("container without a codestream box"))
}

/// Parse the codestream headers and describe the image.
pub fn read_info(data: &[u8]) -> Result<ImageInfo> {
    let cs = codestream(data)?;
    let mut br = BitReader::new(&cs);
    let ih = ImageHeader::read(&mut br)?;
    Ok(image_info(&ih))
}

fn image_info(ih: &ImageHeader) -> ImageInfo {
    let gray = ih.color.color_space == headers::CS_GRAY;
    let (width, height) = if ih.orientation > 4 {
        (ih.height, ih.width)
    } else {
        (ih.width, ih.height)
    };
    ImageInfo {
        width,
        height,
        components: if gray { 1 } else { 3 },
        bits_per_sample: ih.bit_depth.bits.min(16) as u8,
        signed: false,
        color_space: if gray {
            ColorSpace::Gray
        } else {
            ColorSpace::Rgb
        },
    }
}

/// Decoded colour planes of the displayed image.
struct Canvas {
    width: usize,
    height: usize,
    frame: FrameOut,
}

/// Colour planes of a decoded frame.
enum FrameOut {
    /// Integer samples (modular, no XYB, no filters).
    Int(Vec<Vec<i32>>),
    /// Samples normalised to [0, 1] (modular without XYB, filtered).
    Normalised([Vec<f32>; 3]),
    /// Linear-light sRGB (from XYB).
    Linear([Vec<f32>; 3]),
}

/// Decode a complete codestream or container.
pub fn decode(data: &[u8], opts: &DecodeOptions) -> Result<Image> {
    let cs = codestream(data)?;
    let mut br = BitReader::new(&cs);
    let ih = ImageHeader::read(&mut br)?;
    let info = image_info(&ih);
    opts.limits.check_image(&info)?;
    if ih.bit_depth.float {
        return Err(Error::unsupported(CODEC, "floating point samples"));
    }
    if ih.bit_depth.bits > 16 {
        return Err(Error::unsupported(CODEC, "more than 16 bits per sample"));
    }
    let alloc_limit = opts.limits.max_alloc_bytes;
    let mut pos = br.byte_pos();
    if ih.preview.is_some() {
        // Skip the preview frame.
        let mut fbr = BitReader::new(&cs[pos..]);
        let fh = FrameHeader::read(&mut fbr, &ih)?;
        let toc = Toc::read(&mut fbr, &fh)?;
        pos += toc.end;
    }
    let mut canvas: Option<Canvas> = None;
    loop {
        if pos >= cs.len() {
            return Err(Error::Truncated { codec: CODEC });
        }
        let mut fbr = BitReader::new(&cs[pos..]);
        let fh = FrameHeader::read(&mut fbr, &ih)?;
        let toc = Toc::read(&mut fbr, &fh)?;
        if toc.end > cs.len() - pos {
            return Err(Error::Truncated { codec: CODEC });
        }
        let frame_data = &cs[pos..pos + toc.end];
        let displayed =
            fh.frame_type == frame::FRAME_REGULAR || fh.frame_type == frame::FRAME_SKIP_PROGRESSIVE;
        if displayed {
            if fh.blending.mode != frame::BLEND_REPLACE
                || fh.have_crop
                || fh.upsampling != 1
                || fh.lf_level != 0
            {
                return Err(Error::unsupported(
                    CODEC,
                    "frame blending, cropping or upsampling",
                ));
            }
            let frame = decode_frame(&ih, &fh, &toc, frame_data, alloc_limit)?;
            canvas = Some(Canvas {
                width: ih.width as usize,
                height: ih.height as usize,
                frame,
            });
        } else if fh.frame_type == frame::FRAME_LF || fh.save_as_reference != 0 {
            return Err(Error::unsupported(CODEC, "LF and reference frames"));
        }
        pos += toc.end;
        if fh.is_last {
            break;
        }
    }
    let canvas = canvas.ok_or_else(|| invalid("no displayed frame"))?;
    to_image(&ih, canvas)
}

/// Decode the colour channels of one frame.
fn decode_frame(
    ih: &ImageHeader,
    fh: &FrameHeader,
    toc: &Toc,
    data: &[u8],
    alloc_limit: u64,
) -> Result<FrameOut> {
    if fh.do_ycbcr {
        return Err(Error::unsupported(
            CODEC,
            "YCbCr frames (JPEG recompression)",
        ));
    }
    if !fh.modular && !ih.xyb_encoded {
        return Err(Error::unsupported(CODEC, "VarDCT frames without XYB"));
    }
    if fh.flags & (frame::FLAG_PATCHES | frame::FLAG_SPLINES | frame::FLAG_NOISE) != 0 {
        return Err(Error::unsupported(CODEC, "patches, splines and noise"));
    }
    if fh.flags & frame::FLAG_USE_LF_FRAME != 0 {
        return Err(Error::unsupported(CODEC, "LF frames"));
    }
    if fh.ec_upsampling.iter().any(|&u| u != 1)
        || ih.extra_channels.iter().any(|e| e.dim_shift != 0)
    {
        return Err(Error::unsupported(CODEC, "upsampled extra channels"));
    }
    if ih
        .extra_channels
        .iter()
        .any(|e| e.kind == headers::EC_ALPHA && e.alpha_associated)
    {
        return Err(Error::unsupported(CODEC, "premultiplied alpha"));
    }
    if ih.xyb_encoded && (ih.color.want_icc || !ih.color.srgb_gamut) {
        return Err(Error::unsupported(
            CODEC,
            "XYB images with ICC profiles or non-sRGB primaries",
        ));
    }
    let (w, h) = fh.sample_size();
    let (w, h) = (w as usize, h as usize);
    let n_color = if fh.modular { fh.color_channels } else { 0 };
    let n_ec = ih.extra_channels.len();
    let group_dim = fh.group_dim() as usize;
    let num_lf_groups = fh.num_lf_groups();
    let num_groups = fh.num_groups();
    let single = toc.sections.len() == 1;

    // Section readers: one shared reader for a single-section frame.
    fn section<'a>(data: &'a [u8], toc: &Toc, i: usize) -> Result<BitReader<'a>> {
        let (off, size) = toc.sections[i];
        data.get(off..off + size)
            .map(BitReader::new)
            .ok_or(Error::Truncated { codec: CODEC })
    }
    let mut shared = if single {
        Some(section(data, toc, 0)?)
    } else {
        None
    };
    fn take_section<'a>(
        shared: &mut Option<BitReader<'a>>,
        data: &'a [u8],
        toc: &Toc,
        i: usize,
    ) -> Result<BitReader<'a>> {
        match shared.take() {
            Some(b) => Ok(b),
            None => section(data, toc, i),
        }
    }

    // LfGlobal.
    let mut br = take_section(&mut shared, data, toc, 0)?;
    let mut m_lf = [1.0 / 32.0, 1.0 / 4.0, 1.0 / 2.0];
    if !br.read_bool()? {
        for m in &mut m_lf {
            *m = br.f16()?;
        }
        if m_lf.iter().any(|&m| m / 128.0 < 1e-8) {
            return Err(invalid("LF dequantisation weight too small"));
        }
    }
    let vg = if fh.modular {
        None
    } else {
        Some(vardct::LfGlobalVarDct::read(&mut br)?)
    };
    let global_cfg = if br.read_bool()? {
        let limit = (1024 + w as u64 * h as u64 * (n_color.max(3) + n_ec) as u64 / 16).min(1 << 22);
        Some(MaConfig::read(&mut br, limit as usize)?)
    } else {
        None
    };
    let mut full = modular::Image {
        channels: (0..n_color + n_ec)
            .map(|_| Channel::new(w, h, 0, 0))
            .collect(),
        bit_depth: ih.bit_depth.bits,
        ..Default::default()
    };
    modular::decode_stream(
        &mut br,
        &mut full,
        0,
        global_cfg.as_ref(),
        group_dim,
        false,
        alloc_limit,
    )?;
    let beginc = (full.nb_meta..full.channels.len())
        .find(|&c| full.channels[c].w > group_dim || full.channels[c].h > group_dim)
        .unwrap_or(full.channels.len());
    if single {
        shared = Some(br);
    }
    let (gx_count, _) = fh.groups();
    let (lgx_count, _) = fh.lf_groups();
    let lf_dim = group_dim * 8;

    // LF groups.
    let mut lf_groups = Vec::new();
    for i in 0..num_lf_groups {
        let mut br = take_section(&mut shared, data, toc, 1 + i as usize)?;
        let gx = (i % lgx_count) as usize;
        let gy = (i / lgx_count) as usize;
        let (x0, y0) = (gx * lf_dim, gy * lf_dim);
        let stream = 1 + num_lf_groups + i;
        let mut modular_part = |br: &mut BitReader| {
            decode_group(
                br,
                &mut full,
                beginc,
                (x0, y0, lf_dim),
                (3, i32::MAX),
                stream,
                global_cfg.as_ref(),
                alloc_limit,
            )
        };
        if let Some(vg) = &vg {
            let rect = (x0, y0, lf_dim.min(w - x0), lf_dim.min(h - y0));
            lf_groups.push(vardct::read_lf_group(
                &mut br,
                fh,
                vg,
                m_lf,
                i,
                rect,
                global_cfg.as_ref(),
                alloc_limit,
                &mut modular_part,
            )?);
        } else {
            modular_part(&mut br)?;
        }
        if single {
            shared = Some(br);
        }
    }
    // HfGlobal.
    let hg = match &vg {
        Some(vg) => {
            let mut br = take_section(&mut shared, data, toc, 1 + num_lf_groups as usize)?;
            let hg = vardct::read_hf_global(&mut br, fh, vg, global_cfg.as_ref(), alloc_limit)?;
            if single {
                shared = Some(br);
            }
            Some(hg)
        }
        None => None,
    };
    let (bw, bh) = (w.div_ceil(8), h.div_ceil(8));
    let mut coeffs = vardct::Coefficients {
        stride: bw * 8,
        data: std::array::from_fn(|_| {
            if vg.is_some() {
                vec![0; bw * 8 * bh * 8]
            } else {
                Vec::new()
            }
        }),
    };
    // Pass groups.
    for pass in 0..fh.passes.num_passes {
        let (min_shift, max_shift) = shift_bracket(fh, pass);
        for g in 0..num_groups {
            let mut br = take_section(
                &mut shared,
                data,
                toc,
                2 + num_lf_groups as usize + (pass * num_groups + g) as usize,
            )?;
            let gx = (g % gx_count) as usize;
            let gy = (g / gx_count) as usize;
            if let (Some(vg), Some(hg)) = (&vg, &hg) {
                vardct::read_hf_coefficients(
                    &mut br,
                    fh,
                    vg,
                    hg,
                    &lf_groups,
                    pass as usize,
                    (gx, gy),
                    &mut coeffs,
                )?;
            }
            decode_group(
                &mut br,
                &mut full,
                beginc,
                (gx * group_dim, gy * group_dim, group_dim),
                (min_shift, max_shift),
                1 + 3 * num_lf_groups + 17 + num_groups * pass + g,
                global_cfg.as_ref(),
                alloc_limit,
            )?;
            if single {
                shared = Some(br);
            }
        }
    }
    full.undo_transforms()?;

    // Colour planes before filtering.
    let mut planes: [Vec<f32>; 3];
    let mut sigma = Vec::new();
    if let (Some(vg), Some(hg)) = (&vg, &hg) {
        let rec = vardct::reconstruct(
            fh,
            vg,
            hg,
            &lf_groups,
            coeffs,
            m_lf,
            ih.opsin.quant_bias,
            ih.opsin.quant_bias_numerator,
        )?;
        let stride = bw * 8;
        planes = rec.map(|p| {
            (0..h)
                .flat_map(|y| p[y * stride..y * stride + w].to_vec())
                .collect()
        });
        if fh.epf_iters > 0 {
            sigma = vec![0f32; bw * bh];
            for g in &lf_groups {
                g.copy_sigma(&mut sigma, bw);
            }
        }
    } else {
        if full.channels.len() < n_color {
            return Err(invalid("modular image lost its colour channels"));
        }
        for c in &full.channels[..n_color] {
            if c.w != w || c.h != h {
                return Err(invalid("colour channel has the wrong size"));
            }
        }
        let filtered = fh.gaborish || fh.epf_iters > 0;
        if ih.xyb_encoded {
            // Channels are Y, X, B - Y; scaled by the LF dequant weights.
            let [y, x, b] = [
                &full.channels[0].data,
                &full.channels[1].data,
                &full.channels[2].data,
            ];
            let (mx, my, mb) = (m_lf[0] / 128.0, m_lf[1] / 128.0, m_lf[2] / 128.0);
            planes = [
                x.iter().map(|&v| v as f32 * mx).collect(),
                y.iter().map(|&v| v as f32 * my).collect(),
                b.iter()
                    .zip(y)
                    .map(|(&b, &y)| b.saturating_add(y) as f32 * mb)
                    .collect(),
            ];
        } else if filtered {
            let max = ((1u64 << ih.bit_depth.bits) - 1) as f32;
            let norm = |c: usize| -> Vec<f32> {
                full.channels[c]
                    .data
                    .iter()
                    .map(|&v| v as f32 / max)
                    .collect()
            };
            planes = if n_color == 1 {
                let p = norm(0);
                [p.clone(), p.clone(), p]
            } else {
                [norm(0), norm(1), norm(2)]
            };
        } else {
            return Ok(FrameOut::Int(
                full.channels
                    .into_iter()
                    .take(n_color)
                    .map(|c| c.data)
                    .collect(),
            ));
        }
        if fh.epf_iters > 0 {
            sigma = vec![fh.epf.sigma_for_modular; bw * bh];
        }
    }
    if fh.gaborish {
        render::gaborish(&mut planes, w, h, fh.gab_weights);
    }
    if fh.epf_iters > 0 {
        render::epf(&mut planes, w, h, &sigma, bw, fh);
    }
    if ih.xyb_encoded {
        render::xyb_to_linear(&mut planes, &ih.opsin, ih.intensity_target);
        Ok(FrameOut::Linear(planes))
    } else {
        Ok(FrameOut::Normalised(planes))
    }
}

/// Inclusive downsampling shift bracket of a pass (GetDownsamplingBracket).
fn shift_bracket(fh: &FrameHeader, pass: u32) -> (i32, i32) {
    let mut max_shift = 2;
    let mut min_shift = 3;
    let mut i = 0;
    loop {
        for &(ds, last) in &fh.passes.downsample {
            if i == last {
                min_shift = match ds {
                    8 => 3,
                    4 => 2,
                    2 => 1,
                    _ => 0,
                };
            }
        }
        if i == fh.passes.num_passes - 1 {
            min_shift = 0;
        }
        if i == pass {
            return (min_shift, max_shift);
        }
        max_shift = min_shift - 1;
        i += 1;
    }
}

/// Decode one LF or pass group stream into `full`.
#[allow(clippy::too_many_arguments)]
fn decode_group(
    br: &mut BitReader,
    full: &mut modular::Image,
    beginc: usize,
    (x0, y0, dim): (usize, usize, usize),
    (min_shift, max_shift): (i32, i32),
    stream_id: u32,
    global: Option<&MaConfig>,
    alloc_limit: u64,
) -> Result<()> {
    let mut sub = modular::Image {
        bit_depth: full.bit_depth,
        ..Default::default()
    };
    let mut targets = Vec::new();
    for (c, fc) in full.channels.iter().enumerate().skip(beginc) {
        let shift = fc.hshift.min(fc.vshift);
        if shift < min_shift || shift > max_shift || fc.hshift < 0 || fc.vshift < 0 {
            continue;
        }
        let rx = x0 >> fc.hshift;
        let ry = y0 >> fc.vshift;
        let rw = (dim >> fc.hshift).min(fc.w.saturating_sub(rx));
        let rh = (dim >> fc.vshift).min(fc.h.saturating_sub(ry));
        if rw == 0 || rh == 0 {
            continue;
        }
        sub.channels
            .push(Channel::new(rw, rh, fc.hshift, fc.vshift));
        targets.push((c, rx, ry));
    }
    if sub.channels.is_empty() {
        return Ok(());
    }
    modular::decode_stream(
        br,
        &mut sub,
        stream_id,
        global,
        usize::MAX,
        true,
        alloc_limit,
    )?;
    if sub.channels.len() != targets.len() {
        return Err(invalid("group transforms changed the channel count"));
    }
    for (sc, &(c, rx, ry)) in sub.channels.iter().zip(&targets) {
        let fc = &mut full.channels[c];
        if fc.data.len() != fc.w * fc.h {
            fc.data = vec![0; fc.w * fc.h];
        }
        for y in 0..sc.h {
            let dst = (ry + y) * fc.w + rx;
            fc.data[dst..dst + sc.w].copy_from_slice(sc.row(y));
        }
    }
    Ok(())
}

fn to_image(ih: &ImageHeader, canvas: Canvas) -> Result<Image> {
    let info = image_info(ih);
    let bits = ih.bit_depth.bits;
    let max = ((1u64 << bits) - 1) as i64;
    let nc = info.components as usize;
    let n = canvas.width * canvas.height;
    if ih.orientation != 1 {
        return Err(Error::unsupported(CODEC, "image orientation"));
    }
    let maxf = max as f32;
    match canvas.frame {
        FrameOut::Int(planes) => {
            let planes: Vec<&[i32]> = (0..nc)
                .map(|c| &planes[c.min(planes.len() - 1)][..n])
                .collect();
            Image::from_planes_with(info, &planes, |v| (v as i64).clamp(0, max) as u16)
        }
        FrameOut::Normalised(planes) => {
            let planes: Vec<&[f32]> = planes[..nc].iter().map(|p| &p[..n]).collect();
            Image::from_planes_with(info, &planes, |v| {
                (v * maxf).round().clamp(0.0, maxf) as u16
            })
        }
        FrameOut::Linear(planes) => {
            let tf = match ih.color.transfer {
                Some(13) => render::Transfer::Srgb,
                Some(8) => render::Transfer::Linear,
                Some(1) => render::Transfer::Bt709,
                Some(17) => render::Transfer::Dci,
                None => render::Transfer::Gamma(ih.color.gamma as f32 * 1e-7),
                Some(_) => {
                    return Err(Error::unsupported(CODEC, "PQ / HLG transfer functions"));
                }
            };
            let luma: Vec<f32>;
            let planes: Vec<&[f32]> = if nc == 1 {
                luma = (0..n)
                    .map(|i| 0.2126 * planes[0][i] + 0.7152 * planes[1][i] + 0.0722 * planes[2][i])
                    .collect();
                vec![&luma]
            } else {
                planes.iter().map(|p| &p[..n]).collect()
            };
            Image::from_planes_with(info, &planes, |v| (tf.encode(v) * maxf).round() as u16)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fuzzer input (CI): a cropped frame whose group count overflowed u32.
    #[test]
    fn huge_cropped_frame_is_rejected() {
        let data = [
            255, 10, 255, 32, 32, 32, 32, 0, 4, 255, 84, 10, 4, 255, 0, 10, 3, 58, 0, 1, 0, 0, 3,
            85, 105, 32, 32, 160, 255, 0, 49, 255,
        ];
        assert!(decode(&data, &DecodeOptions::default()).is_err());
    }
}
