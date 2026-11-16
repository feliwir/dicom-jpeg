//! ITU-T T.800 | ISO/IEC 15444-1 JPEG 2000 (with Part 2 multi-component and Part 15 HTJ2K).
//!
//! The decoder supports raw codestreams and JP2/JPH files, all progression
//! orders and POC, tiles, precincts, all code-block styles, ROI (maxshift),
//! PPM/PPT packed headers, SOP/EPH markers, the 5/3 and 9/7 wavelets, RCT/ICT
//! and reduced-resolution decoding. Truncated codestreams decode as far as
//! the data goes.

mod codestream;
mod decoder;
mod dwt;
#[cfg(feature = "encode")]
mod encoder;
mod ht;
mod jp2;
mod mq;
mod t1;
#[cfg(feature = "encode")]
mod t1_enc;
mod tagtree;
mod tile;

pub use decoder::{Component, DecoderOptions};
#[cfg(feature = "encode")]
pub use encoder::{EncoderOptions, encode_with};

use crate::error::{Error, Result};
use crate::{Codec, ColorSpace, DecodeOptions, Image, ImageInfo, Samples};
#[cfg(feature = "encode")]
use crate::{EncodeOptions, Encoded, TransferSyntax};

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
    let c0 = s.comps[0];
    Ok(ImageInfo {
        width: (s.x1 - s.x0).div_ceil(c0.dx),
        height: (s.y1 - s.y0).div_ceil(c0.dy),
        components: s.comps.len() as u16,
        bits_per_sample: s.comps.iter().map(|c| c.precision).max().unwrap_or(8),
        signed: c0.signed,
        color_space: match s.comps.len() {
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
    let mut out = vec![0u16; width as usize * height as usize * nc];
    for (ci, c) in comps.iter().enumerate() {
        let mask = if precision >= 16 {
            0xFFFF
        } else {
            (1u32 << precision) - 1
        };
        for y in 0..height as usize {
            let sy = (y * c.height as usize / height as usize).min(c.height as usize - 1);
            for x in 0..width as usize {
                let sx = (x * c.width as usize / width as usize).min(c.width as usize - 1);
                let v = c.data[sy * c.width as usize + sx];
                out[(y * width as usize + x) * nc + ci] = (v as u32 & mask) as u16;
            }
        }
    }
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
    let samples = if precision <= 8 {
        Samples::U8(out.into_iter().map(|v| v as u8).collect())
    } else {
        Samples::U16(out)
    };
    Image::new(info, samples)
}

/// Encode an image for the given transfer syntax.
///
/// `Jpeg2000Lossless` is always lossless. `Jpeg2000` is lossless when
/// `opts.quality` is `None` or 100, otherwise irreversible (9/7) with that
/// quality. Use [`encode_with`] for full control.
#[cfg(feature = "encode")]
pub fn encode(ts: TransferSyntax, image: &Image, opts: &EncodeOptions) -> Result<Encoded> {
    let reversible = match ts {
        TransferSyntax::Jpeg2000Lossless => true,
        TransferSyntax::Jpeg2000 => matches!(opts.quality, None | Some(100..)),
        TransferSyntax::Jpeg2000Part2Lossless | TransferSyntax::Jpeg2000Part2 => {
            return Err(Error::unsupported(CODEC, "Part 2 multi-component encoding"));
        }
        TransferSyntax::HtJ2kLossless
        | TransferSyntax::HtJ2kLosslessRpcl
        | TransferSyntax::HtJ2k => {
            return Err(Error::unsupported(Codec::HtJ2k, "HTJ2K encoding"));
        }
        _ => {
            return Err(Error::input(format!(
                "{ts} is not a JPEG 2000 transfer syntax"
            )));
        }
    };
    let params = EncoderOptions {
        reversible,
        quality: opts.quality.unwrap_or(90),
        ..EncoderOptions::default()
    };
    encode_with(image, &params)
}
