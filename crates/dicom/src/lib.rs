//! [dicom-rs](https://github.com/Enet4/dicom-rs) integration.
//!
//! [`PixelAdapter`] implements `PixelDataReader` and `PixelDataWriter` from
//! `dicom-encoding` for the active JPEG-family transfer syntaxes (UIDs as in
//! `dicom_dictionary_std::uids`); it maps each to a codec crate and its
//! parameters.
//!
//! With the `inventory-registry` feature the adapter is also registered in
//! the dicom-rs transfer syntax registry, so that dicom-rs itself (e.g.
//! `dicom_pixeldata::PixelDecoder::decode_pixel_data`) uses it. dicom-rs
//! keeps a built-in codec that already decodes and encodes a transfer
//! syntax; in dicom-rs 0.10 that is only JPEG baseline with the `jpeg`
//! feature of dicom-pixeldata, which its default `native` feature enables.
//! Make sure the crate is linked, e.g. with `use dicom_jpeg_dicom as _;`.
//! The crate's README is a complete guide to using it with dicom-rs.
//!
//! Notes on the dicom-rs adapter contract:
//! * decoded 3-sample images are delivered as interleaved RGB
//!   (YCbCr data is converted, J2K RCT/ICT is inverted);
//! * samples are written little-endian with the object's Bits Allocated;
//! * JPEG 2000 Part 2 (.92 / .93) codes all frames as the components of one
//!   codestream (PS3.5 A.4.4), transformed along the frames: use
//!   `PixelDataWriter::encode` (per-frame `encode_frame` only works for
//!   single-frame objects), and `decode_frame` decodes the whole codestream;
//! * `PixelDataObject` does not expose Pixel Representation or Planar
//!   Configuration, so native input to the writer is assumed to be
//!   interleaved (planar configuration 0) and the codestream's own signedness
//!   is used when decoding.

#![forbid(unsafe_code)]

use std::borrow::Cow;

use dicom_encoding::adapters::{DecodeError, DecodeResult, PixelDataObject, PixelDataReader};

use dicom_jpeg_common::{Codec, ColorOutput, ColorSpace, DecodeOptions, Image, Limits};
// Re-imported at the crate root for the submodules.
#[cfg(feature = "encode")]
use dicom_jpeg_common::{AppliedTransform, EncodeOptions, Encoded};
use dicom_jpeg_common::{Error, Result};
use dicom_jpeg_common::{ImageInfo, Samples};

#[cfg(feature = "jpeg2000")]
use dicom_jpeg_j2k as j2k;
#[cfg(feature = "jpeg")]
use dicom_jpeg_jpeg as jpeg;
#[cfg(feature = "jpeg-ls")]
use dicom_jpeg_jpegls as jpegls;
#[cfg(feature = "jpeg-xl")]
use dicom_jpeg_jxl as jxl;

/// Decode a codestream with the codec of a transfer syntax.
fn decode(codec: Codec, data: &[u8], opts: &DecodeOptions) -> Result<Image> {
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

// The dependency only enables entry collection in the registry crate (its
// `inventory-registry` feature); nothing in it is called.
#[cfg(feature = "inventory-registry")]
use dicom_transfer_syntax_registry as _;

// Without any codec there is nothing to register.
#[cfg(all(
    feature = "inventory-registry",
    any(
        feature = "jpeg",
        feature = "jpeg-ls",
        feature = "jpeg2000",
        feature = "jpeg-xl"
    )
))]
mod registry;
mod syntax;
#[cfg(feature = "encode")]
mod writer;

use syntax::{SYNTAXES, Syntax};

/// dicom-rs pixel data adapter for one transfer syntax.
///
/// ```no_run
/// use dicom_jpeg_dicom::PixelAdapter;
/// use dicom_dictionary_std::uids;
/// use dicom_encoding::adapters::PixelDataReader;
/// # fn f(obj: &dyn dicom_encoding::adapters::PixelDataObject) {
/// let adapter = PixelAdapter::from_uid(uids::JPEG_LOSSLESS_SV1).unwrap();
/// let mut pixels = Vec::new();
/// adapter.decode_frame(obj, 0, &mut pixels).unwrap();
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PixelAdapter {
    uid: &'static str,
    syntax: Syntax,
    limits: Limits,
}

impl PixelAdapter {
    /// Adapter for the transfer syntax identified by `uid` (trailing NUL /
    /// space padding is ignored), with default [`Limits`]. `None` for UIDs
    /// outside the JPEG family.
    pub fn from_uid(uid: &str) -> Option<Self> {
        Syntax::from_uid(uid).map(|(uid, syntax)| PixelAdapter {
            uid,
            syntax,
            limits: Limits {
                max_samples: 1 << 31,
                max_alloc_bytes: 1 << 33,
            },
        })
    }

    /// UIDs of all transfer syntaxes the adapter handles.
    pub fn supported_uids() -> impl Iterator<Item = &'static str> {
        SYNTAXES.iter().map(|(uid, _)| *uid)
    }

    /// Replace the resource limits.
    pub const fn with_limits(mut self, limits: Limits) -> Self {
        self.limits = limits;
        self
    }

    /// The transfer syntax UID handled by this adapter.
    pub const fn uid(&self) -> &'static str {
        self.uid
    }

    /// The codec used for this transfer syntax.
    pub fn codec(&self) -> Codec {
        self.syntax.codec()
    }

    /// Whether this transfer syntax admits lossy coding.
    pub fn allows_lossy(&self) -> bool {
        self.syntax.allows_lossy()
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
        _ => false,
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

/// Image attributes the reader needs from the object.
struct FrameLayout {
    rows: u16,
    cols: u16,
    spp: u16,
    bits_allocated: u16,
}

fn frame_layout(src: &dyn PixelDataObject) -> DecodeResult<FrameLayout> {
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
    Ok(FrameLayout {
        rows,
        cols,
        spp,
        bits_allocated,
    })
}

impl PixelAdapter {
    fn decode_options(&self, src: &dyn PixelDataObject, spp: u16) -> DecodeOptions {
        DecodeOptions::new()
            .with_color_output(if spp == 3 {
                ColorOutput::Rgb
            } else {
                ColorOutput::Native
            })
            .with_color_hint(color_hint(src.photometric_interpretation()))
            .with_limits(self.limits)
    }

    /// Decode the codestreams of a Part 2 object: the components of all
    /// codestreams, in order, are the samples of frame 0, frame 1, ...
    fn decode_part2(
        &self,
        src: &dyn PixelDataObject,
        layout: &FrameLayout,
    ) -> DecodeResult<Vec<Image>> {
        let fragments = src
            .number_of_fragments()
            .ok_or(DecodeError::NotEncapsulated)?;
        let mut streams: Vec<Vec<u8>> = Vec::new();
        for i in 0..fragments as usize {
            let fragment = src.fragment(i).ok_or(DecodeError::FrameRangeOutOfBounds)?;
            if streams.is_empty() || starts_codestream(self.syntax.codec(), &fragment) {
                streams.push(Vec::new());
            }
            streams
                .last_mut()
                .expect("pushed above")
                .extend_from_slice(&fragment);
        }
        let opts = self.decode_options(src, 1);
        let spp = usize::from(layout.spp);
        let mut frames = Vec::new();
        let mut pending: Vec<Image> = Vec::new();
        for data in &streams {
            let image =
                crate::decode(self.syntax.codec(), data, &opts).map_err(codec_decode_error)?;
            pending.extend(split_components(&image).map_err(codec_decode_error)?);
            while pending.len() >= spp {
                let samples: Vec<Image> = pending.drain(..spp).collect();
                frames.push(join_components(&samples, layout.spp).map_err(codec_decode_error)?);
            }
        }
        if !pending.is_empty() {
            return Err(custom_decode(
                "component count is not a multiple of Samples per Pixel",
            ));
        }
        Ok(frames)
    }

    /// Clamp, validate and append one decoded frame to `dst`.
    fn finish_frame(
        &self,
        src: &dyn PixelDataObject,
        layout: &FrameLayout,
        mut image: Image,
        dst: &mut Vec<u8>,
    ) -> DecodeResult<()> {
        // Lossy codestreams may carry more bits than the object stores (e.g. 10-bit
        // data in a 12-bit JPEG); clamp overshoot so it cannot wrap when masked.
        if let Some(bits_stored) = src.bits_stored()
            && !image.info.signed
            && (bits_stored as u8) < image.info.bits_per_sample
            && bits_stored > 0
        {
            clamp_unsigned(&mut image, bits_stored as u8);
        }
        let (rows, cols, spp) = (layout.rows, layout.cols, layout.spp);
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
        let bytes = usize::from(layout.bits_allocated / 8);
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

/// Split an interleaved image into single-component images.
fn split_components(image: &Image) -> crate::Result<Vec<Image>> {
    let nc = usize::from(image.info.components);
    (0..nc)
        .map(|c| {
            let info = crate::ImageInfo {
                components: 1,
                color_space: ColorSpace::Gray,
                ..image.info
            };
            let samples = match &image.samples {
                crate::Samples::U8(v) => {
                    crate::Samples::U8(v.iter().skip(c).step_by(nc).copied().collect())
                }
                crate::Samples::U16(v) => {
                    crate::Samples::U16(v.iter().skip(c).step_by(nc).copied().collect())
                }
            };
            Image::new(info, samples)
        })
        .collect()
}

/// Interleave single-component images into one image.
fn join_components(planes: &[Image], spp: u16) -> crate::Result<Image> {
    let first = &planes[0].info;
    let n = planes[0].samples.len();
    if planes
        .iter()
        .any(|p| (p.info.width, p.info.height) != (first.width, first.height))
    {
        return Err(crate::Error::input(
            "components of one frame differ in size",
        ));
    }
    let bits = planes
        .iter()
        .map(|p| p.info.bits_per_sample)
        .max()
        .unwrap_or(8);
    let info = crate::ImageInfo {
        components: spp,
        bits_per_sample: bits,
        color_space: if spp == 3 {
            ColorSpace::Rgb
        } else {
            ColorSpace::Gray
        },
        ..*first
    };
    let mut out = vec![0u16; n * planes.len()];
    for (c, p) in planes.iter().enumerate() {
        let vals: Box<dyn Iterator<Item = u16>> = match &p.samples {
            crate::Samples::U8(v) => Box::new(v.iter().map(|&x| x as u16)),
            crate::Samples::U16(v) => Box::new(v.iter().copied()),
        };
        for (i, x) in vals.enumerate() {
            out[i * planes.len() + c] = x;
        }
    }
    let samples = if bits <= 8 {
        crate::Samples::U8(out.into_iter().map(|x| x as u8).collect())
    } else {
        crate::Samples::U16(out)
    };
    Image::new(info, samples)
}

impl PixelDataReader for PixelAdapter {
    fn decode(&self, src: &dyn PixelDataObject, dst: &mut Vec<u8>) -> DecodeResult<()> {
        if !self.syntax.is_part2() {
            let frames = src.number_of_frames().unwrap_or(1);
            for frame in 0..frames {
                self.decode_frame(src, frame, dst)?;
            }
            return Ok(());
        }
        let layout = frame_layout(src)?;
        let frames = self.decode_part2(src, &layout)?;
        let declared = src.number_of_frames().unwrap_or(1).max(1) as usize;
        if frames.len() < declared {
            return Err(custom_decode(format!(
                "codestreams hold {} frames, DICOM object declares {declared}",
                frames.len()
            )));
        }
        for image in frames.into_iter().take(declared) {
            self.finish_frame(src, &layout, image, dst)?;
        }
        Ok(())
    }

    fn decode_frame(
        &self,
        src: &dyn PixelDataObject,
        frame: u32,
        dst: &mut Vec<u8>,
    ) -> DecodeResult<()> {
        let layout = frame_layout(src)?;
        let image = if self.syntax.is_part2() {
            if frame >= src.number_of_frames().unwrap_or(1).max(1) {
                return Err(DecodeError::FrameRangeOutOfBounds);
            }
            self.decode_part2(src, &layout)?
                .into_iter()
                .nth(frame as usize)
                .ok_or(DecodeError::FrameRangeOutOfBounds)?
        } else {
            let data = frame_codestream(src, self.syntax.codec(), frame)?;
            let opts = self.decode_options(src, layout.spp);
            crate::decode(self.syntax.codec(), &data, &opts).map_err(codec_decode_error)?
        };
        self.finish_frame(src, &layout, image, dst)
    }
}

fn clamp_unsigned(image: &mut Image, bits: u8) {
    let max = (1u32 << bits) - 1;
    match &mut image.samples {
        crate::Samples::U8(v) => v.iter_mut().for_each(|s| *s = (*s as u32).min(max) as u8),
        crate::Samples::U16(v) => v.iter_mut().for_each(|s| *s = (*s as u32).min(max) as u16),
    }
}
