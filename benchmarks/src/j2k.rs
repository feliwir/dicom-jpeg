//! JPEG 2000 reference implementations shared by the J2K and HTJ2K benches.

use dicom_jpeg::Image;

use crate::{DecodeFn, EncodeFn, openjpeg};

/// OpenJPEG (C).
pub fn openjpeg_decode<'a>() -> DecodeFn<'a> {
    Box::new(openjpeg::decode)
}

/// openjp2, the pure-Rust port of OpenJPEG (built without rayon).
pub fn openjp2_decode<'a>() -> DecodeFn<'a> {
    Box::new(|d| {
        let img = openjp2::Image::from_bytes_with(d, openjp2::DecodeParameters::new().threads(1))
            .map_err(|e| e.to_string())?;
        Ok(img
            .components()
            .iter()
            .map(|c| c.w as usize * c.h as usize)
            .sum())
    })
}

/// hayro-jpeg2000 (pure Rust).
pub fn hayro_decode<'a>() -> DecodeFn<'a> {
    Box::new(|d| {
        let settings = hayro_jpeg2000::DecodeSettings::default();
        let img = hayro_jpeg2000::Image::new(d, &settings).map_err(|e| format!("{e:?}"))?;
        let n = img.width() as usize
            * img.height() as usize
            * img.color_space().num_channels() as usize;
        let mut ctx = hayro_jpeg2000::DecoderContext::default();
        img.decode(&mut ctx).map_err(|e| format!("{e:?}"))?;
        Ok(n)
    })
}

/// OpenJPEG (C) encoder.
pub fn openjpeg_encode<'a>(mode: openjpeg::Mode) -> EncodeFn<'a> {
    Box::new(move |img| openjpeg::encode(img, mode))
}

/// openjp2 encoder, lossless or with one layer at `ratio`:1.
pub fn openjp2_encode<'a>(ratio: Option<f32>) -> EncodeFn<'a> {
    Box::new(move |img: &Image| {
        let info = &img.info;
        let comps = img
            .planes()
            .into_iter()
            .map(|data| openjp2::CoreImageComp {
                dx: 1,
                dy: 1,
                w: info.width,
                h: info.height,
                prec: info.bits_per_sample as u32,
                bpp: info.bits_per_sample as u32,
                sgnd: info.signed as u32,
                data,
                ..Default::default()
            })
            .collect();
        let core = openjp2::CoreImage {
            x1: info.width,
            y1: info.height,
            numcomps: info.components as u32,
            color_space: if info.components == 3 {
                openjp2::ColorSpace::Srgb
            } else {
                openjp2::ColorSpace::Gray
            },
            comps,
            ..Default::default()
        };
        let mut p = openjp2::EncodeParameters::new()
            .irreversible(ratio.is_some())
            .layer_rate(ratio.unwrap_or(0.0));
        p.low_level().tcp_mct = (info.components == 3) as _;
        openjp2::Image::from_core_image(core)
            .encode(p)
            .map_err(|e| e.to_string())
    })
}
