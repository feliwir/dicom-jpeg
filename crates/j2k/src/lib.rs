//! ITU-T T.800 | ISO/IEC 15444-1 JPEG 2000 (with Part 2 multi-component and Part 15 HTJ2K).
//!
//! The decoder supports raw codestreams and JP2/JPH files, all progression
//! orders and POC, tiles, precincts, all code-block styles, ROI (maxshift),
//! PPM/PPT packed headers, SOP/EPH markers, the 5/3 and 9/7 wavelets, RCT/ICT
//! and reduced-resolution decoding. Truncated codestreams decode as far as
//! the data goes.
//!
//! Features: `htj2k` (ISO/IEC 15444-15 block coder), `part2` (ISO/IEC
//! 15444-2 multiple component transforms) and `encode`.

#![forbid(unsafe_code)]

mod codestream;
mod decoder;
mod dwt;
#[cfg(feature = "encode")]
mod encoder;
mod ht;
mod jp2;
mod mct;
mod mq;
mod t1;
#[cfg(feature = "encode")]
mod t1_enc;
mod tagtree;
mod tile;

pub use decoder::{Component, DecoderOptions};
#[cfg(feature = "encode")]
pub use encoder::{ComponentWavelet, EncoderOptions, encode_with};

use dicom_jpeg_common::{Codec, ColorSpace, DecodeOptions, Image, ImageInfo};
use dicom_jpeg_common::{Error, Result};

pub(crate) const CODEC: Codec = Codec::Jpeg2000;

fn codestream(data: &[u8]) -> Result<&[u8]> {
    if data.starts_with(&[0xFF, 0x4F]) {
        Ok(data)
    } else if data.get(4..8) == Some(b"jP  ") {
        jp2::codestream(data)
    } else {
        Err(Error::invalid(
            CODEC,
            "neither a J2K codestream nor a JP2 file",
        ))
    }
}

/// Decode every component at its own resolution (no up-sampling, no
/// conversion), as needed for conformance testing.
pub fn decode_components(data: &[u8], opts: &DecoderOptions) -> Result<Vec<Component>> {
    let cs = codestream::parse(codestream(data)?)?;
    decoder::decode_codestream(&cs, opts)
}

/// Parse the codestream headers and describe the image.
pub fn read_info(data: &[u8]) -> Result<ImageInfo> {
    let cs = codestream::parse(codestream(data)?)?;
    let s = &cs.siz;
    let out = cs.output_comps();
    let c0 = out[0];
    Ok(ImageInfo {
        width: (s.x1 - s.x0).div_ceil(c0.dx),
        height: (s.y1 - s.y0).div_ceil(c0.dy),
        components: out.len() as u16,
        bits_per_sample: out.iter().map(|c| c.precision).max().unwrap_or(8),
        signed: c0.signed,
        color_space: match out.len() {
            1 => ColorSpace::Gray,
            3 => ColorSpace::Rgb,
            _ => ColorSpace::Unknown,
        },
    })
}

/// Decode a complete codestream (or JP2 file) into an interleaved image.
/// Components with sub-sampling are up-sampled by replication.
pub fn decode(data: &[u8], opts: &DecodeOptions) -> Result<Image> {
    let comps = decode_components(
        data,
        &DecoderOptions {
            reduce: 0,
            limits: opts.limits,
        },
    )?;
    let precision = comps.iter().map(|c| c.precision).max().unwrap_or(8);
    if precision > 16 {
        return Err(Error::unsupported(CODEC, "more than 16 bits per sample"));
    }
    let width = comps.iter().map(|c| c.width).max().unwrap_or(0);
    let height = comps.iter().map(|c| c.height).max().unwrap_or(0);
    let nc = comps.len();
    let signed = comps[0].signed;
    let info = ImageInfo {
        width,
        height,
        components: nc as u16,
        bits_per_sample: precision,
        signed,
        color_space: match nc {
            1 => ColorSpace::Gray,
            3 => ColorSpace::Rgb,
            _ => ColorSpace::Unknown,
        },
    };
    let mask = ((1u32 << precision) - 1) as i32;
    let (w, h) = (width as usize, height as usize);
    // Up-sample sub-sampled components by replication.
    let full: Vec<Vec<i32>> = comps
        .iter()
        .filter(|c| (c.width, c.height) != (width, height))
        .map(|c| {
            let (cw, ch) = (c.width as usize, c.height as usize);
            let sx: Vec<usize> = (0..w).map(|x| (x * cw / w).min(cw - 1)).collect();
            let mut out = Vec::with_capacity(w * h);
            for y in 0..h {
                let row = &c.data[(y * ch / h).min(ch - 1) * cw..][..cw];
                out.extend(sx.iter().map(|&x| row[x]));
            }
            out
        })
        .collect();
    let mut full = full.iter();
    let planes: Vec<&[i32]> = comps
        .iter()
        .map(|c| {
            if (c.width, c.height) == (width, height) {
                c.data.as_slice()
            } else {
                full.next().map_or(&[][..], Vec::as_slice)
            }
        })
        .collect();
    Image::from_planes_with(info, &planes, |v| (v & mask) as u16)
}

/// Whether a Part 2 encode of `info` benefits from the component-axis
/// wavelet. A single RGB image is better served by the Part 1 RCT / ICT
/// (the Part 2 syntaxes do not require a multiple component transform),
/// and a single component has nothing to transform.
#[cfg(feature = "encode")]
pub fn part2_uses_wavelet(info: &ImageInfo) -> bool {
    info.components > 1 && !(info.components == 3 && info.color_space == ColorSpace::Rgb)
}
