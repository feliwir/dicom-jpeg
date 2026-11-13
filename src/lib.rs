//! Pure-Rust JPEG-family codecs for every active DICOM transfer syntax.
//!
//! | Transfer syntax | Codec | Cargo feature |
//! |---|---|---|
//! | 1.2.840.10008.1.2.4.50 / .51 / .57 / .70 | JPEG (T.81) | `jpeg` |
//! | 1.2.840.10008.1.2.4.80 / .81 | JPEG-LS (T.87) | `jpeg-ls` |
//! | 1.2.840.10008.1.2.4.90 / .91 | JPEG 2000 (15444-1) | `jpeg2000` |
//! | 1.2.840.10008.1.2.4.92 / .93 | JPEG 2000 Part 2 MCT | `jpeg2000-part2` |
//! | 1.2.840.10008.1.2.4.201 / .202 / .203 | HTJ2K (15444-15) | `htj2k` |
//! | 1.2.840.10008.1.2.4.110 / .111 / .112 | JPEG XL (18181) | `jpeg-xl` |
//!
//! Encoders are enabled with the `encode` feature, and the `dicom` feature
//! provides [`dicom_rs::PixelAdapter`], an implementation of the dicom-rs
//! `PixelDataReader` / `PixelDataWriter` traits.
//!
//! ```no_run
//! use dicom_jpeg::{decode, DecodeOptions, TransferSyntax};
//! # let codestream: &[u8] = &[];
//! let ts = TransferSyntax::from_uid("1.2.840.10008.1.2.4.70").unwrap();
//! let image = decode(ts, codestream, &DecodeOptions::default())?;
//! println!("{}x{}", image.info.width, image.info.height);
//! # Ok::<(), dicom_jpeg::Error>(())
//! ```

mod error;
mod image;
mod options;
mod transfer_syntax;

#[cfg(feature = "jpeg2000")]
pub mod j2k;
#[cfg(feature = "jpeg")]
pub mod jpeg;
#[cfg(feature = "jpeg-ls")]
pub mod jpegls;
#[cfg(feature = "jpeg-xl")]
pub mod jxl;

#[cfg(feature = "dicom")]
pub mod dicom_rs;

pub use error::{Error, Result};
pub use image::{ColorSpace, Image, ImageInfo, Samples};
pub use options::{AppliedTransform, ColorOutput, DecodeOptions, EncodeOptions, Encoded, Limits};
pub use transfer_syntax::{Codec, TransferSyntax};

/// Read the image description from a codestream without decoding pixel data.
pub fn read_info(ts: TransferSyntax, data: &[u8]) -> Result<ImageInfo> {
    match ts.codec() {
        #[cfg(feature = "jpeg")]
        Codec::Jpeg => jpeg::read_info(data),
        #[cfg(feature = "jpeg-ls")]
        Codec::JpegLs => jpegls::read_info(data),
        #[cfg(feature = "jpeg2000")]
        Codec::Jpeg2000 => j2k::read_info(data),
        #[cfg(feature = "htj2k")]
        Codec::HtJ2k => j2k::read_info(data),
        #[cfg(feature = "jpeg-xl")]
        Codec::JpegXl => jxl::read_info(data),
        #[allow(unreachable_patterns)]
        codec => {
            let _ = data;
            Err(Error::CodecDisabled(codec))
        }
    }
}

/// Decode one frame (a complete codestream) of the given transfer syntax.
pub fn decode(ts: TransferSyntax, data: &[u8], opts: &DecodeOptions) -> Result<Image> {
    match ts.codec() {
        #[cfg(feature = "jpeg")]
        Codec::Jpeg => jpeg::decode(data, opts),
        #[cfg(feature = "jpeg-ls")]
        Codec::JpegLs => jpegls::decode(data, opts),
        #[cfg(feature = "jpeg2000")]
        Codec::Jpeg2000 => j2k::decode(data, opts),
        #[cfg(feature = "htj2k")]
        Codec::HtJ2k => j2k::decode(data, opts),
        #[cfg(feature = "jpeg-xl")]
        Codec::JpegXl => jxl::decode(data, opts),
        #[allow(unreachable_patterns)]
        codec => {
            let _ = (data, opts);
            Err(Error::CodecDisabled(codec))
        }
    }
}

/// Encode an image into a codestream of the given transfer syntax.
///
/// Lossless-only syntaxes always encode losslessly; for syntaxes that allow
/// both, `opts.quality == Some(100)` (or `None` for the lossless-capable
/// JPEG 2000 / JPEG XL / JPEG-LS syntaxes) selects lossless coding.
#[cfg(feature = "encode")]
pub fn encode(ts: TransferSyntax, image: &Image, opts: &EncodeOptions) -> Result<Encoded> {
    image.info.validate(&Limits::UNLIMITED)?;
    match ts.codec() {
        #[cfg(feature = "jpeg")]
        Codec::Jpeg => jpeg::encode(ts, image, opts),
        #[cfg(feature = "jpeg-ls")]
        Codec::JpegLs => jpegls::encode(ts, image, opts),
        #[cfg(feature = "jpeg2000")]
        Codec::Jpeg2000 => j2k::encode(ts, image, opts),
        #[cfg(feature = "htj2k")]
        Codec::HtJ2k => j2k::encode(ts, image, opts),
        #[cfg(feature = "jpeg-xl")]
        Codec::JpegXl => jxl::encode(ts, image, opts),
        #[allow(unreachable_patterns)]
        codec => {
            let _ = opts;
            Err(Error::CodecDisabled(codec))
        }
    }
}
