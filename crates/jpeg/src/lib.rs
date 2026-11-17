//! ITU-T T.81 | ISO/IEC 10918-1: baseline, extended (8/12 bit) and lossless (process 14) JPEG.
//!
//! Decoding is bit-exact with the IJG libjpeg 6b "ISLOW" integer IDCT,
//! "fancy" chroma upsampling and fixed-point YCbCr conversion (as used by
//! DCMTK). Progressive, hierarchical and arithmetic-coded streams (retired
//! DICOM transfer syntaxes) are reported as unsupported.

#![deny(unsafe_code)]

mod bitreader;
mod color;
mod decoder;
#[cfg(feature = "encode")]
mod downsample;
#[cfg(feature = "encode")]
mod encoder;
#[cfg(feature = "encode")]
mod fdct;
mod huffman;
#[cfg(feature = "encode")]
mod huffman_encode;
mod idct;
#[cfg(feature = "encode")]
mod rgb_ycbcr;
mod upsample;
mod ycbcr;

#[cfg(feature = "encode")]
pub use encoder::{ChromaSubsampling, EncoderOptions, Process, encode_with};

use dicom_jpeg_common::Result;
use dicom_jpeg_common::{Codec, DecodeOptions, Image, ImageInfo, Limits};

use decoder::Decoder;

pub(crate) const CODEC: Codec = Codec::Jpeg;

/// Parse the codestream headers and describe the image.
///
/// The colour space reported is the one stored in the codestream (as far as
/// it can be determined from JFIF / Adobe markers and component identifiers).
pub fn read_info(data: &[u8]) -> Result<ImageInfo> {
    let mut dec = Decoder::new(data, Limits::UNLIMITED)?;
    dec.read_headers()?;
    let color_space = dec.native_color(None);
    let frame = dec.frame.as_ref().expect("frame parsed");
    Ok(ImageInfo {
        width: frame.width as u32,
        height: frame.height as u32,
        components: frame.components.len() as u16,
        bits_per_sample: frame.precision,
        signed: false,
        color_space,
    })
}

/// Decode a complete codestream.
pub fn decode(data: &[u8], opts: &DecodeOptions) -> Result<Image> {
    let mut dec = Decoder::new(data, opts.limits)?;
    dec.read_headers()?;
    dec.decode_scans()?;
    dec.into_image(opts)
}
