//! `PixelDataWriter` implementation (requires the `encode` feature).

use dicom_core::PrimitiveValue;
use dicom_core::ops::{AttributeAction, AttributeOp};
use dicom_dictionary_std::tags::{
    LOSSY_IMAGE_COMPRESSION, LOSSY_IMAGE_COMPRESSION_METHOD, LOSSY_IMAGE_COMPRESSION_RATIO,
    PHOTOMETRIC_INTERPRETATION, PLANAR_CONFIGURATION,
};
use dicom_encoding::adapters::{EncodeError, EncodeResult, PixelDataObject, PixelDataWriter};

use super::{PixelAdapter, is_part2};
use crate::{AppliedTransform, Codec, ColorSpace, Image, TransferSyntax};

fn custom_encode(message: impl Into<String>) -> EncodeError {
    EncodeError::Custom {
        message: message.into(),
        source: None,
    }
}

fn codec_encode_error(err: crate::Error) -> EncodeError {
    EncodeError::Custom {
        message: err.to_string(),
        source: Some(Box::new(err)),
    }
}

/// Defined terms of Lossy Image Compression Method (PS3.3 C.7.6.1.1.5.1).
/// There is none for Part 2, whose core coding system is Part 1.
fn lossy_method(ts: TransferSyntax) -> &'static str {
    match ts.codec() {
        Codec::Jpeg => "ISO_10918_1",
        Codec::JpegLs => "ISO_14495_1",
        Codec::Jpeg2000 => "ISO_15444_1",
        Codec::HtJ2k => "ISO_15444_15",
        Codec::JpegXl => "ISO_18181_1",
    }
}

/// Native image attributes of the object being encoded.
struct Source {
    info: crate::ImageInfo,
    bits_allocated: u16,
    pi: String,
    frames: u32,
}

fn source(src: &dyn PixelDataObject) -> EncodeResult<Source> {
    if src.number_of_fragments().is_some() {
        return Err(EncodeError::NotNative);
    }
    let rows = src
        .rows()
        .ok_or(EncodeError::MissingAttribute { name: "Rows" })?;
    let cols = src
        .cols()
        .ok_or(EncodeError::MissingAttribute { name: "Columns" })?;
    let spp = src
        .samples_per_pixel()
        .ok_or(EncodeError::MissingAttribute {
            name: "SamplesPerPixel",
        })?;
    let bits_allocated = src.bits_allocated().ok_or(EncodeError::MissingAttribute {
        name: "BitsAllocated",
    })?;
    let bits_stored = src
        .bits_stored()
        .ok_or(EncodeError::MissingAttribute { name: "BitsStored" })?;
    let pi = src.photometric_interpretation().unwrap_or("").to_string();
    if bits_allocated != 8 && bits_allocated != 16 {
        return Err(custom_encode(format!(
            "unsupported Bits Allocated {bits_allocated}"
        )));
    }
    if bits_stored == 0 || bits_stored > bits_allocated {
        return Err(custom_encode(format!("invalid Bits Stored {bits_stored}")));
    }
    let color_space = match (spp, pi.as_str()) {
        (1, _) => ColorSpace::Gray,
        (3, "RGB") => ColorSpace::Rgb,
        (3, "YBR_FULL") => ColorSpace::YCbCr,
        (3, other) => {
            return Err(custom_encode(format!(
                "cannot encode native data with Photometric Interpretation {other:?}"
            )));
        }
        _ => ColorSpace::Unknown,
    };
    Ok(Source {
        info: crate::ImageInfo {
            width: u32::from(cols),
            height: u32::from(rows),
            components: spp,
            bits_per_sample: bits_stored as u8,
            signed: false,
            color_space,
        },
        bits_allocated,
        pi,
        frames: src.number_of_frames().unwrap_or(1).max(1),
    })
}

impl PixelAdapter {
    fn frame_image(
        &self,
        src: &dyn PixelDataObject,
        s: &Source,
        frame: u32,
    ) -> EncodeResult<(Image, usize)> {
        if frame >= s.frames {
            return Err(EncodeError::FrameRangeOutOfBounds);
        }
        let native = src
            .frame_pixel_data(frame)
            .ok_or(EncodeError::FrameRangeOutOfBounds)?;
        let image = Image::from_le_bytes(s.info, &native, usize::from(s.bits_allocated / 8))
            .map_err(codec_encode_error)?;
        Ok((image, native.len()))
    }

    /// Attribute changes for the encoded data.
    fn attribute_ops(
        &self,
        s: &Source,
        encoded: &crate::Encoded,
        raw_len: usize,
    ) -> Vec<AttributeOp> {
        let mut ops = Vec::new();
        if s.info.components == 3 {
            let new_pi = match (encoded.transform, s.info.color_space) {
                (AppliedTransform::Rct, _) => "YBR_RCT",
                (AppliedTransform::Ict, _) => "YBR_ICT",
                (AppliedTransform::YCbCr { h: 2, .. }, _) => "YBR_FULL_422",
                (AppliedTransform::YCbCr { .. }, _) => "YBR_FULL",
                (_, ColorSpace::YCbCr) => "YBR_FULL",
                _ => "RGB",
            };
            if new_pi != s.pi {
                ops.push(AttributeOp::new(
                    PHOTOMETRIC_INTERPRETATION,
                    AttributeAction::SetStr(new_pi.into()),
                ));
            }
            ops.push(AttributeOp::new(
                PLANAR_CONFIGURATION,
                AttributeAction::Set(PrimitiveValue::from(0_u16)),
            ));
        }
        if encoded.lossy {
            let ratio = raw_len as f64 / encoded.data.len().max(1) as f64;
            ops.push(AttributeOp::new(
                LOSSY_IMAGE_COMPRESSION,
                AttributeAction::SetStr("01".into()),
            ));
            ops.push(AttributeOp::new(
                LOSSY_IMAGE_COMPRESSION_RATIO,
                AttributeAction::PushStr(format!("{ratio:.6}").into()),
            ));
            ops.push(AttributeOp::new(
                LOSSY_IMAGE_COMPRESSION_METHOD,
                AttributeAction::PushStr(lossy_method(self.ts).into()),
            ));
        }
        ops
    }

    /// JPEG 2000 Part 2: all frames become the components of one codestream
    /// (frame-major, `spp` samples per frame), transformed along the frames.
    #[cfg(feature = "jpeg2000-part2")]
    fn encode_part2(
        &self,
        src: &dyn PixelDataObject,
        options: dicom_encoding::adapters::EncodeOptions,
        dst: &mut Vec<Vec<u8>>,
    ) -> EncodeResult<Vec<AttributeOp>> {
        use crate::j2k::{ComponentWavelet, EncoderOptions};
        let s = source(src)?;
        let spp = usize::from(s.info.components);
        let nc = spp * s.frames as usize;
        if nc > 16384 {
            return Err(custom_encode(
                "JPEG 2000 Part 2 supports at most 16384 components (frames x samples)",
            ));
        }
        let mut frames = Vec::with_capacity(s.frames as usize);
        let mut raw_len = 0;
        for f in 0..s.frames {
            let (image, len) = self.frame_image(src, &s, f)?;
            raw_len += len;
            frames.push(image);
        }
        let mut info = s.info;
        info.components = nc as u16;
        info.color_space = ColorSpace::Unknown;
        // Interleave the frames' samples: component f * spp + c.
        let per_frame = frames[0].samples.len();
        let samples = match &frames[0].samples {
            crate::Samples::U8(_) => {
                crate::Samples::U8(interleave(&frames, per_frame, spp, |s| match s {
                    crate::Samples::U8(v) => v.as_slice(),
                    crate::Samples::U16(_) => unreachable!("all frames share one sample type"),
                }))
            }
            crate::Samples::U16(_) => {
                crate::Samples::U16(interleave(&frames, per_frame, spp, |s| match s {
                    crate::Samples::U16(v) => v.as_slice(),
                    crate::Samples::U8(_) => unreachable!("all frames share one sample type"),
                }))
            }
        };
        let mut image = Image::new(info, samples).map_err(codec_encode_error)?;
        if spp == 1 && looks_signed(&frames[0]) {
            image.info.signed = true;
        }
        let lossy = self.ts == TransferSyntax::Jpeg2000Part2
            && !matches!(options.quality, None | Some(100..));
        // Multi-frame colour data: transform each sample along the frames.
        let interleave = if s.frames > 1 { spp as u16 } else { 1 };
        if s.frames == 1 {
            image.info.color_space = s.info.color_space;
        }
        let opts = EncoderOptions {
            reversible: !lossy,
            quality: options.quality.unwrap_or(90),
            component_wavelet: crate::j2k::part2_uses_wavelet(&image.info)
                .then(|| ComponentWavelet::new(None, interleave)),
            ..EncoderOptions::default()
        };
        let encoded = crate::j2k::encode_with(&image, &opts).map_err(codec_encode_error)?;
        let ops = self.attribute_ops(&s, &encoded, raw_len);
        dst.push(encoded.data);
        Ok(ops)
    }
}

/// Interleave `frames` (each `per_frame` samples with `spp` per pixel) into
/// one image with `frames * spp` components per pixel.
#[cfg(feature = "jpeg2000-part2")]
fn interleave<T: Copy + Default>(
    frames: &[Image],
    per_frame: usize,
    spp: usize,
    get: impl Fn(&crate::Samples) -> &[T],
) -> Vec<T> {
    let nc = spp * frames.len();
    let mut out = vec![T::default(); per_frame * frames.len()];
    for (f, frame) in frames.iter().enumerate() {
        for (i, &v) in get(&frame.samples).iter().enumerate() {
            let (pixel, c) = (i / spp, i % spp);
            out[pixel * nc + f * spp + c] = v;
        }
    }
    out
}

impl PixelDataWriter for PixelAdapter {
    fn encode(
        &self,
        src: &dyn PixelDataObject,
        options: dicom_encoding::adapters::EncodeOptions,
        dst: &mut Vec<Vec<u8>>,
        offset_table: &mut Vec<u32>,
    ) -> EncodeResult<Vec<AttributeOp>> {
        #[cfg(feature = "jpeg2000-part2")]
        if is_part2(self.ts) {
            // One codestream for all frames: no per-frame offsets.
            return self.encode_part2(src, options, dst);
        }
        let frames = src.number_of_frames().unwrap_or(1).max(1);
        let mut ops = Vec::new();
        let mut offset = 0u32;
        let mut offsets = Vec::with_capacity(frames as usize);
        for frame in 0..frames {
            let mut data = Vec::new();
            ops = self.encode_frame(src, frame, options.clone(), &mut data)?;
            offsets.push(offset);
            offset = u32::try_from(data.len())
                .ok()
                .and_then(|d| d.checked_add(8))
                .and_then(|d| offset.checked_add(d))
                .unwrap_or(u32::MAX);
            dst.push(data);
        }
        // A Basic Offset Table is only valid if every offset fits.
        if offsets.last().is_some_and(|&o| o != u32::MAX) {
            offset_table.extend(offsets);
        }
        Ok(ops)
    }

    fn encode_frame(
        &self,
        src: &dyn PixelDataObject,
        frame: u32,
        options: dicom_encoding::adapters::EncodeOptions,
        dst: &mut Vec<u8>,
    ) -> EncodeResult<Vec<AttributeOp>> {
        let s = source(src)?;
        if is_part2(self.ts) && s.frames > 1 {
            return Err(custom_encode(
                "JPEG 2000 Part 2 codes all frames in one codestream; use encode()",
            ));
        }
        let (mut image, raw_len) = self.frame_image(src, &s, frame)?;
        // `PixelDataObject` does not expose Pixel Representation. JPEG 2000
        // signals signedness in the codestream and lossy coding needs it (a
        // two's complement image read as unsigned has huge artificial edges),
        // so infer it from the data.
        if matches!(self.ts.codec(), Codec::Jpeg2000 | Codec::HtJ2k) && looks_signed(&image) {
            image.info.signed = true;
        }
        let mut opts = crate::EncodeOptions::new();
        opts.quality = options.quality;
        opts.effort = options.effort;
        let encoded = crate::encode(self.ts, &image, &opts).map_err(codec_encode_error)?;
        dst.extend_from_slice(&encoded.data);
        Ok(self.attribute_ops(&s, &encoded, raw_len))
    }
}

/// Whether a single-component image is much smoother when its samples are
/// read as two's complement (i.e. it is most likely signed data).
fn looks_signed(image: &Image) -> bool {
    let info = &image.info;
    if info.components != 1 || info.bits_per_sample < 2 {
        return false;
    }
    let bits = info.bits_per_sample as u32;
    let half = 1i64 << (bits - 1);
    let values: Vec<i64> = match &image.samples {
        crate::Samples::U8(v) => v.iter().map(|&x| x as i64).collect(),
        crate::Samples::U16(v) => v.iter().map(|&x| x as i64).collect(),
    };
    let as_signed = |v: i64| if v >= half { v - 2 * half } else { v };
    let w = info.width as usize;
    let (mut tv_unsigned, mut tv_signed) = (0i64, 0i64);
    for row in values.chunks(w).step_by(4) {
        for p in row.windows(2) {
            tv_unsigned += (p[1] - p[0]).abs();
            tv_signed += (as_signed(p[1]) - as_signed(p[0])).abs();
        }
    }
    tv_signed * 2 < tv_unsigned
}
