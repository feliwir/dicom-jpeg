//! `PixelDataWriter` implementation (requires the `encode` feature).

use dicom_core::PrimitiveValue;
use dicom_core::ops::{AttributeAction, AttributeOp};
use dicom_dictionary_std::tags::{
    LOSSY_IMAGE_COMPRESSION, LOSSY_IMAGE_COMPRESSION_METHOD, LOSSY_IMAGE_COMPRESSION_RATIO,
    PHOTOMETRIC_INTERPRETATION, PLANAR_CONFIGURATION,
};
use dicom_encoding::adapters::{EncodeError, EncodeResult, PixelDataObject, PixelDataWriter};

use super::PixelAdapter;
use crate::{AppliedTransform, Codec, ColorSpace, Image};

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

fn lossy_method(codec: Codec) -> &'static str {
    match codec {
        Codec::Jpeg => "ISO_10918_1",
        Codec::JpegLs => "ISO_14495_1",
        Codec::Jpeg2000 => "ISO_15444_1",
        Codec::HtJ2k => "ISO_15444_15",
        Codec::JpegXl => "ISO_18181_1",
    }
}

impl PixelDataWriter for PixelAdapter {
    fn encode_frame(
        &self,
        src: &dyn PixelDataObject,
        frame: u32,
        options: dicom_encoding::adapters::EncodeOptions,
        dst: &mut Vec<u8>,
    ) -> EncodeResult<Vec<AttributeOp>> {
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
        let pi = src.photometric_interpretation().unwrap_or("");
        if bits_allocated != 8 && bits_allocated != 16 {
            return Err(custom_encode(format!(
                "unsupported Bits Allocated {bits_allocated}"
            )));
        }
        if bits_stored == 0 || bits_stored > bits_allocated {
            return Err(custom_encode(format!("invalid Bits Stored {bits_stored}")));
        }
        let color_space = match (spp, pi) {
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

        let frames = src.number_of_frames().unwrap_or(1).max(1);
        if frame >= frames {
            return Err(EncodeError::FrameRangeOutOfBounds);
        }
        let native = src
            .frame_pixel_data(frame)
            .ok_or(EncodeError::FrameRangeOutOfBounds)?;

        let info = crate::ImageInfo {
            width: u32::from(cols),
            height: u32::from(rows),
            components: spp,
            bits_per_sample: bits_stored as u8,
            signed: false,
            color_space,
        };
        let image = Image::from_le_bytes(info, &native, usize::from(bits_allocated / 8))
            .map_err(codec_encode_error)?;
        let mut opts = crate::EncodeOptions::new();
        opts.quality = options.quality;
        opts.effort = options.effort;
        let encoded = crate::encode(self.ts, &image, &opts).map_err(codec_encode_error)?;

        let raw_len = native.len();
        let encoded_len = encoded.data.len();
        dst.extend_from_slice(&encoded.data);

        let mut ops = Vec::new();
        if spp == 3 {
            let new_pi = match (encoded.transform, color_space) {
                (AppliedTransform::Rct, _) => "YBR_RCT",
                (AppliedTransform::Ict, _) => "YBR_ICT",
                (AppliedTransform::YCbCr { h: 2, .. }, _) => "YBR_FULL_422",
                (AppliedTransform::YCbCr { .. }, _) => "YBR_FULL",
                (_, ColorSpace::YCbCr) => "YBR_FULL",
                _ => "RGB",
            };
            if new_pi != pi {
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
            let ratio = raw_len as f64 / encoded_len.max(1) as f64;
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
                AttributeAction::PushStr(lossy_method(self.ts.codec()).into()),
            ));
        }
        Ok(ops)
    }
}
