//! ISO/IEC 18181: JPEG XL.
//!
//! The decoder currently handles codestreams whose frames are coded in
//! modular mode without XYB (the lossless mode used by the JPEG XL Lossless
//! transfer syntax).

mod bits;
#[cfg(feature = "encode")]
mod encode;
mod entropy;
mod frame;
mod headers;
mod modular;

use crate::error::{Error, Result};
use crate::{Codec, ColorSpace, DecodeOptions, Image, ImageInfo, Samples};
#[cfg(feature = "encode")]
use crate::{EncodeOptions, Encoded, TransferSyntax};

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
    planes: Vec<Vec<i32>>,
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
    if ih.xyb_encoded {
        return Err(Error::unsupported(CODEC, "XYB-encoded (lossy) images"));
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
            let planes = decode_frame(&ih, &fh, &toc, frame_data, alloc_limit)?;
            canvas = Some(Canvas {
                width: ih.width as usize,
                height: ih.height as usize,
                planes,
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
) -> Result<Vec<Vec<i32>>> {
    if !fh.modular {
        return Err(Error::unsupported(CODEC, "VarDCT frames"));
    }
    if fh.do_ycbcr {
        return Err(Error::unsupported(CODEC, "YCbCr modular frames"));
    }
    if fh.flags & (frame::FLAG_PATCHES | frame::FLAG_SPLINES | frame::FLAG_NOISE) != 0 {
        return Err(Error::unsupported(CODEC, "patches, splines and noise"));
    }
    if fh.gaborish || fh.epf_iters > 0 {
        return Err(Error::unsupported(
            CODEC,
            "restoration filters (Gaborish, EPF)",
        ));
    }
    if fh.flags & frame::FLAG_USE_LF_FRAME != 0 {
        return Err(Error::unsupported(CODEC, "LF frames"));
    }
    if fh.ec_upsampling.iter().any(|&u| u != 1)
        || ih.extra_channels.iter().any(|e| e.dim_shift != 0)
    {
        return Err(Error::unsupported(CODEC, "upsampled extra channels"));
    }
    let (w, h) = fh.sample_size();
    let (w, h) = (w as usize, h as usize);
    let n_color = fh.color_channels;
    let n_ec = ih.extra_channels.len();
    let group_dim = fh.group_dim() as usize;
    let num_lf_groups = fh.num_lf_groups();
    let num_groups = fh.num_groups();
    let single = toc.sections.len() == 1;

    // Section readers: one shared reader for a single-section frame.
    let section = |i: usize| -> Result<BitReader<'_>> {
        let (off, size) = toc.sections[i];
        data.get(off..off + size)
            .map(BitReader::new)
            .ok_or(Error::Truncated { codec: CODEC })
    };
    let mut shared = if single { Some(section(0)?) } else { None };

    // LfGlobal.
    let mut br = match shared.take() {
        Some(b) => b,
        None => section(0)?,
    };
    // LfChannelDequantization.
    if !br.read_bool()? {
        for _ in 0..3 {
            br.f16()?;
        }
    }
    let global_cfg = if br.read_bool()? {
        let limit = (1024 + w as u64 * h as u64 * (n_color + n_ec) as u64 / 16).min(1 << 22);
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

    // First channel left for the groups.
    let beginc = (full.nb_meta..full.channels.len())
        .find(|&c| full.channels[c].w > group_dim || full.channels[c].h > group_dim)
        .unwrap_or(full.channels.len());

    if single {
        shared = Some(br);
    }
    let (gx_count, _) = fh.groups();
    let (lgx_count, _) = fh.lf_groups();
    let lf_dim = group_dim * 8;

    // LfGroups: channels with shift >= 3.
    for i in 0..num_lf_groups {
        let mut br = match shared.take() {
            Some(b) => b,
            None => section(1 + i as usize)?,
        };
        let gx = (i % lgx_count) as usize;
        let gy = (i / lgx_count) as usize;
        decode_group(
            &mut br,
            &mut full,
            beginc,
            (gx * lf_dim, gy * lf_dim, lf_dim),
            (3, i32::MAX),
            1 + num_lf_groups + i,
            global_cfg.as_ref(),
            alloc_limit,
        )?;
        if single {
            shared = Some(br);
        }
    }
    // HfGlobal is empty for modular frames.
    for pass in 0..fh.passes.num_passes {
        let (min_shift, max_shift) = shift_bracket(fh, pass);
        for g in 0..num_groups {
            let mut br = match shared.take() {
                Some(b) => b,
                None => section(2 + num_lf_groups as usize + (pass * num_groups + g) as usize)?,
            };
            let gx = (g % gx_count) as usize;
            let gy = (g / gx_count) as usize;
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
    if full.channels.len() < n_color {
        return Err(invalid("modular image lost its colour channels"));
    }
    let mut planes = Vec::with_capacity(n_color);
    for c in full.channels.into_iter().take(n_color) {
        if c.w != w || c.h != h {
            return Err(invalid("colour channel has the wrong size"));
        }
        planes.push(c.data);
    }
    Ok(planes)
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
    let (w, h) = (canvas.width, canvas.height);
    let planes: Vec<&Vec<i32>> = if canvas.planes.len() == 1 && nc == 3 {
        vec![&canvas.planes[0]; 3]
    } else {
        canvas.planes.iter().take(nc).collect()
    };
    if ih.orientation != 1 {
        return Err(Error::unsupported(CODEC, "image orientation"));
    }
    let mut out = Vec::with_capacity(w * h * nc);
    for i in 0..w * h {
        for p in &planes {
            out.push((p[i] as i64).clamp(0, max) as u16);
        }
    }
    let samples = if bits <= 8 {
        Samples::U8(out.into_iter().map(|v| v as u8).collect())
    } else {
        Samples::U16(out)
    };
    Image::new(info, samples)
}

/// Encode an image for the given transfer syntax.
///
/// `JpegXlLossless`, and `JpegXl` without a quality below 100, produce a
/// lossless modular codestream. Lossy coding and JPEG recompression are
/// not implemented yet.
#[cfg(feature = "encode")]
pub fn encode(ts: TransferSyntax, image: &Image, opts: &EncodeOptions) -> Result<Encoded> {
    match ts {
        TransferSyntax::JpegXlLossless => encode::encode_lossless(image),
        TransferSyntax::JpegXl if matches!(opts.quality, None | Some(100..)) => {
            encode::encode_lossless(image)
        }
        TransferSyntax::JpegXl => Err(Error::unsupported(CODEC, "lossy encoding")),
        TransferSyntax::JpegXlJpegRecompression => {
            Err(Error::unsupported(CODEC, "JPEG recompression"))
        }
        _ => Err(Error::input(format!(
            "{ts} is not a JPEG XL transfer syntax"
        ))),
    }
}
