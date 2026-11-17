//! Pure-Rust JPEG-family codecs for every active DICOM transfer syntax.
//!
//! | Transfer syntax | Codec | Cargo feature | Crate |
//! |---|---|---|---|
//! | 1.2.840.10008.1.2.4.50 / .51 / .57 / .70 | JPEG (T.81) | `jpeg` | `dicom-jpeg-jpeg` |
//! | 1.2.840.10008.1.2.4.80 / .81 | JPEG-LS (T.87) | `jpeg-ls` | `dicom-jpeg-jpegls` |
//! | 1.2.840.10008.1.2.4.90 / .91 | JPEG 2000 (15444-1) | `jpeg2000` | `dicom-jpeg-j2k` |
//! | 1.2.840.10008.1.2.4.92 / .93 | JPEG 2000 Part 2 MCT | `jpeg2000-part2` | `dicom-jpeg-j2k` |
//! | 1.2.840.10008.1.2.4.201 / .202 / .203 | HTJ2K (15444-15) | `htj2k` | `dicom-jpeg-j2k` |
//! | 1.2.840.10008.1.2.4.110 / .111 / .112 | JPEG XL (18181) | `jpeg-xl` | `dicom-jpeg-jxl` |
//!
//! The codec crates know nothing about DICOM: they take codec-specific
//! options. The `dicom` feature adds [`dicom_rs::PixelAdapter`], which maps
//! a transfer syntax UID to a codec and its parameters and implements the
//! dicom-rs `PixelDataReader` / `PixelDataWriter` traits. Encoders are
//! enabled with the `encode` feature.

#![forbid(unsafe_code)]

pub use dicom_jpeg_common::{
    AppliedTransform, Codec, ColorOutput, ColorSpace, DecodeOptions, EncodeOptions, Encoded, Error,
    Image, ImageInfo, Limits, Result, Samples, color, huffman, simd,
};

#[cfg(feature = "jpeg2000")]
pub use dicom_jpeg_j2k as j2k;
#[cfg(feature = "jpeg")]
pub use dicom_jpeg_jpeg as jpeg;
#[cfg(feature = "jpeg-ls")]
pub use dicom_jpeg_jpegls as jpegls;
#[cfg(feature = "jpeg-xl")]
pub use dicom_jpeg_jxl as jxl;

#[cfg(feature = "dicom")]
pub use dicom_jpeg_dicom as dicom_rs;

/// Whether support for `codec` is compiled in.
#[allow(clippy::match_like_matches_macro)] // the arms depend on the enabled features
pub const fn codec_enabled(codec: Codec) -> bool {
    match codec {
        Codec::Jpeg => cfg!(feature = "jpeg"),
        Codec::JpegLs => cfg!(feature = "jpeg-ls"),
        Codec::Jpeg2000 => cfg!(feature = "jpeg2000"),
        Codec::HtJ2k => cfg!(feature = "htj2k"),
        Codec::JpegXl => cfg!(feature = "jpeg-xl"),
        _ => false,
    }
}

/// Read the image description from a codestream without decoding pixel data.
pub fn read_info(codec: Codec, data: &[u8]) -> Result<ImageInfo> {
    match codec {
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

/// Decode a complete codestream of `codec`.
pub fn decode(codec: Codec, data: &[u8], opts: &DecodeOptions) -> Result<Image> {
    match codec {
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
