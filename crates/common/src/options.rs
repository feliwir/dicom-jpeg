//! Decoder / encoder options and resource limits.

use crate::error::{Error, Result};
use crate::image::{ColorSpace, ImageInfo};

/// Resource limits enforced *before* any large allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Limits {
    /// Maximum number of samples (`width * height * components`).
    pub max_samples: u64,
    /// Maximum size in bytes of any single allocation the codec makes.
    pub max_alloc_bytes: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            // 2^31 samples, e.g. 46340 x 46340 x 1 or 26754 x 26754 x 3
            max_samples: 1 << 31,
            max_alloc_bytes: 1 << 33,
        }
    }
}

impl Limits {
    /// No limits at all (only arithmetic overflow is still checked).
    pub const UNLIMITED: Limits = Limits {
        max_samples: u64::MAX,
        max_alloc_bytes: u64::MAX,
    };

    pub fn check_image(&self, info: &ImageInfo) -> Result<()> {
        let samples = info.width as u64 * info.height as u64 * info.components as u64;
        if samples > self.max_samples {
            return Err(Error::LimitExceeded {
                what: "samples",
                requested: samples,
                limit: self.max_samples,
            });
        }
        self.check_alloc(samples.saturating_mul(info.bytes_per_sample() as u64))
    }

    pub fn check_alloc(&self, bytes: u64) -> Result<()> {
        if bytes > self.max_alloc_bytes {
            return Err(Error::LimitExceeded {
                what: "allocation bytes",
                requested: bytes,
                limit: self.max_alloc_bytes,
            });
        }
        Ok(())
    }
}

/// What colour model the decoded samples should be delivered in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ColorOutput {
    /// Deliver samples as stored in the codestream, after any codestream-internal
    /// transform that is part of the format (J2K RCT/ICT, JPEG XL XYB) has been
    /// inverted. JPEG YCbCr stays YCbCr.
    #[default]
    Native,
    /// Convert YCbCr to RGB where applicable.
    Rgb,
}

/// Options for decoding.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct DecodeOptions {
    /// Requested output colour model.
    pub color_output: ColorOutput,
    /// Colour model to assume for 3-component JPEG data that carries no
    /// explicit marker (JFIF / Adobe APP14 / component ids). In DICOM this
    /// comes from the Photometric Interpretation.
    pub color_hint: Option<ColorSpace>,
    /// Resource limits.
    pub limits: Limits,
}

impl DecodeOptions {
    /// Default options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the requested output colour model.
    pub fn with_color_output(mut self, color_output: ColorOutput) -> Self {
        self.color_output = color_output;
        self
    }

    /// Set the colour model hint.
    pub fn with_color_hint(mut self, hint: Option<ColorSpace>) -> Self {
        self.color_hint = hint;
        self
    }

    /// Set resource limits.
    pub fn with_limits(mut self, limits: Limits) -> Self {
        self.limits = limits;
        self
    }
}

/// Generic encoding options; codec modules offer finer-grained parameters.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct EncodeOptions {
    /// Quality 1..=100 for lossy transfer syntaxes (100 = best).
    /// `None` lets the codec choose a sensible default. Ignored by lossless syntaxes.
    pub quality: Option<u8>,
    /// Encoder effort 0..=100 (higher = slower, smaller). `None` = codec default.
    pub effort: Option<u8>,
    /// JPEG-LS near-lossless bound (NEAR). `None` = codec default for the syntax.
    pub near_lossless: Option<u16>,
}

impl EncodeOptions {
    /// Default options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the quality (1..=100).
    pub fn with_quality(mut self, quality: u8) -> Self {
        self.quality = Some(quality);
        self
    }

    /// Set the effort (0..=100).
    pub fn with_effort(mut self, effort: u8) -> Self {
        self.effort = Some(effort);
        self
    }

    /// Set the JPEG-LS NEAR parameter.
    pub fn with_near_lossless(mut self, near: u16) -> Self {
        self.near_lossless = Some(near);
        self
    }
}

/// Colour transform that the encoder applied inside the codestream.
///
/// The DICOM adapter maps this to the Photometric Interpretation of the
/// encoded object (e.g. `YBR_RCT`, `YBR_ICT`, `YBR_FULL_422`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AppliedTransform {
    /// Samples were stored without colour transform.
    None,
    /// RGB was converted to YCbCr, with the given chroma subsampling (h, v).
    YCbCr {
        /// Horizontal chroma subsampling factor (1 or 2).
        h: u8,
        /// Vertical chroma subsampling factor (1 or 2).
        v: u8,
    },
    /// JPEG 2000 reversible colour transform.
    Rct,
    /// JPEG 2000 irreversible colour transform.
    Ict,
    /// JPEG XL XYB (or other internal transform that the decoder always inverts).
    Internal,
}

/// Output of an encoder.
#[derive(Debug, Clone)]
pub struct Encoded {
    /// The codestream.
    pub data: Vec<u8>,
    /// Colour transform applied inside the codestream.
    pub transform: AppliedTransform,
    /// Whether the encoding lost information.
    pub lossy: bool,
}
