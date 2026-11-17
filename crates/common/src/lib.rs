//! Types and helpers shared by the dicom-jpeg codec crates: images and
//! sample buffers, options and resource limits, errors, and the coding
//! tools used by more than one codec (canonical prefix codes, colour
//! transforms).

#![deny(unsafe_code)]

mod error;
mod image;
mod options;

pub mod bitio;
pub mod color;
pub mod float;
pub mod huffman;
pub mod simd;

pub use error::{Error, Result};
pub use image::{ColorSpace, Image, ImageInfo, Samples};
pub use options::{AppliedTransform, ColorOutput, DecodeOptions, EncodeOptions, Encoded, Limits};

use std::fmt;

/// Codec families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Codec {
    /// ITU-T T.81 | ISO/IEC 10918-1 (DCT sequential and lossless).
    Jpeg,
    /// ITU-T T.87 | ISO/IEC 14495-1.
    JpegLs,
    /// ITU-T T.800 | ISO/IEC 15444-1 (and Part 2 multi-component).
    Jpeg2000,
    /// ITU-T T.814 | ISO/IEC 15444-15.
    HtJ2k,
    /// ISO/IEC 18181.
    JpegXl,
}

impl Codec {
    /// Short human readable name.
    pub const fn name(self) -> &'static str {
        match self {
            Codec::Jpeg => "JPEG",
            Codec::JpegLs => "JPEG-LS",
            Codec::Jpeg2000 => "JPEG 2000",
            Codec::HtJ2k => "HTJ2K",
            Codec::JpegXl => "JPEG XL",
        }
    }
}

impl fmt::Display for Codec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}
