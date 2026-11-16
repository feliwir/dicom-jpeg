//! [dicom-rs](https://github.com/Enet4/dicom-rs) integration.
//!
//! [`PixelAdapter`] implements `PixelDataReader` and `PixelDataWriter` from
//! `dicom-encoding` for every transfer syntax in [`TransferSyntax`].
//!
//! Notes on the dicom-rs adapter contract:
//! * decoded 3-sample images are delivered as interleaved RGB
//!   (YCbCr data is converted, J2K RCT/ICT is inverted);
//! * samples are written little-endian with the object's Bits Allocated;
//! * `PixelDataObject` does not expose Pixel Representation or Planar
//!   Configuration, so native input to the writer is assumed to be
//!   interleaved (planar configuration 0) and the codestream's own signedness
//!   is used when decoding.

use std::borrow::Cow;

use dicom_encoding::adapters::{DecodeError, DecodeResult, PixelDataObject, PixelDataReader};

use crate::{Codec, ColorOutput, ColorSpace, DecodeOptions, Image, Limits, TransferSyntax};

#[cfg(feature = "encode")]
mod writer;

/// dicom-rs pixel data adapter for one transfer syntax.
///
/// ```no_run
/// use dicom_jpeg::{dicom_rs::PixelAdapter, TransferSyntax};
/// use dicom_encoding::adapters::PixelDataReader;
/// # fn f(obj: &dyn dicom_encoding::adapters::PixelDataObject) {
/// let adapter = PixelAdapter::new(TransferSyntax::JpegLosslessSv1);
/// let mut pixels = Vec::new();
/// adapter.decode_frame(obj, 0, &mut pixels).unwrap();
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PixelAdapter {
    ts: TransferSyntax,
    limits: Limits,
}

impl PixelAdapter {
    /// Adapter for the given transfer syntax with default [`Limits`].
    pub const fn new(ts: TransferSyntax) -> Self {
        PixelAdapter {
            ts,
            limits: Limits {
                max_samples: 1 << 31,
                max_alloc_bytes: 1 << 33,
            },
        }
    }

    /// Adapter for the transfer syntax identified by `uid`.
    pub fn from_uid(uid: &str) -> Option<Self> {
        TransferSyntax::from_uid(uid).map(Self::new)
    }

    /// Replace the resource limits.
    pub const fn with_limits(mut self, limits: Limits) -> Self {
        self.limits = limits;
        self
    }

    /// The transfer syntax handled by this adapter.
    pub const fn transfer_syntax(&self) -> TransferSyntax {
        self.ts
    }
}

fn custom_decode(message: impl Into<String>) -> DecodeError {
    DecodeError::Custom {
        message: message.into(),
        source: None,
    }
}

fn codec_decode_error(err: crate::Error) -> DecodeError {
    DecodeError::Custom {
        message: err.to_string(),
        source: Some(Box::new(err)),
    }
}

/// Does this fragment start a new codestream of the given codec?
fn starts_codestream(codec: Codec, data: &[u8]) -> bool {
    match codec {
        Codec::Jpeg | Codec::JpegLs => data.starts_with(&[0xFF, 0xD8]),
        Codec::Jpeg2000 | Codec::HtJ2k => {
            data.starts_with(&[0xFF, 0x4F, 0xFF, 0x51])
                || data.starts_with(&[0, 0, 0, 0x0C, b'j', b'P', b' ', b' '])
        }
        Codec::JpegXl => {
            data.starts_with(&[0xFF, 0x0A])
                || data.starts_with(&[0, 0, 0, 0x0C, b'J', b'X', b'L', b' '])
        }
    }
}

/// Collect the encoded bytes of one frame.
///
/// Handles the 1:1 fragment/frame case, the Basic Offset Table, and as a
/// fallback splits the fragment list at codestream start signatures
/// (frames always begin at a fragment boundary).
pub fn frame_codestream<'a>(
    src: &'a dyn PixelDataObject,
    codec: Codec,
    frame: u32,
) -> DecodeResult<Cow<'a, [u8]>> {
    let frames = src.number_of_frames().unwrap_or(1).max(1);
    if frame >= frames {
        return Err(DecodeError::FrameRangeOutOfBounds);
    }
    let fragments = src
        .number_of_fragments()
        .ok_or(DecodeError::NotEncapsulated)?;
    if fragments == 0 {
        return Err(DecodeError::FrameRangeOutOfBounds);
    }
    if fragments == frames {
        return src
            .fragment(frame as usize)
            .ok_or(DecodeError::FrameRangeOutOfBounds);
    }
    let has_bot = src.offset_table().is_some_and(|t| !t.is_empty());
    if has_bot {
        return src
            .frame_pixel_data(frame)
            .ok_or(DecodeError::FrameRangeOutOfBounds);
    }
    if frames == 1 {
        let mut data = Vec::new();
        for i in 0..fragments as usize {
            data.extend_from_slice(&src.fragment(i).ok_or(DecodeError::FrameRangeOutOfBounds)?);
        }
        return Ok(Cow::Owned(data));
    }
    // No offset table, multiple fragments per frame: split at codestream starts.
    let mut current: i64 = -1;
    let mut data = Vec::new();
    for i in 0..fragments as usize {
        let fragment = src.fragment(i).ok_or(DecodeError::FrameRangeOutOfBounds)?;
        if starts_codestream(codec, &fragment) {
            current += 1;
            if current > frame as i64 {
                break;
            }
        }
        if current == frame as i64 {
            data.extend_from_slice(&fragment);
        }
    }
    if data.is_empty() {
        return Err(DecodeError::FrameRangeOutOfBounds);
    }
    Ok(Cow::Owned(data))
}

fn color_hint(pi: Option<&str>) -> Option<ColorSpace> {
    match pi? {
        "MONOCHROME1" | "MONOCHROME2" | "PALETTE COLOR" => Some(ColorSpace::Gray),
        "RGB" => Some(ColorSpace::Rgb),
        "YBR_FULL" | "YBR_FULL_422" => Some(ColorSpace::YCbCr),
        _ => None,
    }
}

impl PixelDataReader for PixelAdapter {
    fn decode_frame(
        &self,
        src: &dyn PixelDataObject,
        frame: u32,
        dst: &mut Vec<u8>,
    ) -> DecodeResult<()> {
        let rows = src
            .rows()
            .ok_or(DecodeError::MissingAttribute { name: "Rows" })?;
        let cols = src
            .cols()
            .ok_or(DecodeError::MissingAttribute { name: "Columns" })?;
        let spp = src
            .samples_per_pixel()
            .ok_or(DecodeError::MissingAttribute {
                name: "SamplesPerPixel",
            })?;
        let bits_allocated = src.bits_allocated().ok_or(DecodeError::MissingAttribute {
            name: "BitsAllocated",
        })?;
        if bits_allocated != 8 && bits_allocated != 16 {
            return Err(custom_decode(format!(
                "unsupported Bits Allocated {bits_allocated}"
            )));
        }

        let data = frame_codestream(src, self.ts.codec(), frame)?;
        let opts = DecodeOptions::new()
            .with_color_output(if spp == 3 {
                ColorOutput::Rgb
            } else {
                ColorOutput::Native
            })
            .with_color_hint(color_hint(src.photometric_interpretation()))
            .with_limits(self.limits);
        let mut image: Image = crate::decode(self.ts, &data, &opts).map_err(codec_decode_error)?;
        // Lossy codestreams may carry more bits than the object stores (e.g. 10-bit
        // data in a 12-bit JPEG); clamp overshoot so it cannot wrap when masked.
        if let Some(bits_stored) = src.bits_stored() {
            if !image.info.signed
                && (bits_stored as u8) < image.info.bits_per_sample
                && bits_stored > 0
            {
                clamp_unsigned(&mut image, bits_stored as u8);
            }
        }

        let info = &image.info;
        if info.width != u32::from(cols) || info.height != u32::from(rows) {
            return Err(custom_decode(format!(
                "codestream is {}x{}, DICOM object declares {cols}x{rows}",
                info.width, info.height
            )));
        }
        if info.components != spp {
            return Err(custom_decode(format!(
                "codestream has {} components, DICOM object declares {spp} samples per pixel",
                info.components
            )));
        }
        let bytes = usize::from(bits_allocated / 8);
        let needed = info
            .sample_count()
            .unwrap_or(usize::MAX)
            .saturating_mul(bytes);
        if dst.try_reserve(needed).is_err() {
            return Err(DecodeError::Allocate {
                name: "frame",
                size: Some(needed),
            });
        }
        image
            .write_le_bytes(bytes, dst)
            .map_err(codec_decode_error)?;
        Ok(())
    }
}

fn clamp_unsigned(image: &mut Image, bits: u8) {
    let max = (1u32 << bits) - 1;
    match &mut image.samples {
        crate::Samples::U8(v) => v.iter_mut().for_each(|s| *s = (*s as u32).min(max) as u8),
        crate::Samples::U16(v) => v.iter_mut().for_each(|s| *s = (*s as u32).min(max) as u16),
    }
}
