//! Mapping of the JPEG-family transfer syntax UIDs (from
//! `dicom_dictionary_std::uids`) to codecs and their encoder parameters.

use dicom_dictionary_std::uids;

use crate::Codec;
#[cfg(feature = "encode")]
use crate::{EncodeOptions, Encoded, Error, Image, Result};

/// The transfer syntaxes handled by the adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Syntax {
    JpegBaseline,
    JpegExtended,
    JpegLossless,
    JpegLosslessSv1,
    JpegLsLossless,
    JpegLsNearLossless,
    J2kLossless,
    J2k,
    J2kPart2Lossless,
    J2kPart2,
    HtJ2kLossless,
    HtJ2kLosslessRpcl,
    HtJ2k,
    JxlLossless,
    JxlJpegRecompression,
    Jxl,
}

/// (UID, syntax) for every supported transfer syntax.
pub(crate) const SYNTAXES: [(&str, Syntax); 16] = [
    (uids::JPEG_BASELINE8_BIT, Syntax::JpegBaseline),
    (uids::JPEG_EXTENDED12_BIT, Syntax::JpegExtended),
    (uids::JPEG_LOSSLESS, Syntax::JpegLossless),
    (uids::JPEG_LOSSLESS_SV1, Syntax::JpegLosslessSv1),
    (uids::JPEGLS_LOSSLESS, Syntax::JpegLsLossless),
    (uids::JPEGLS_NEAR_LOSSLESS, Syntax::JpegLsNearLossless),
    (uids::JPEG2000_LOSSLESS, Syntax::J2kLossless),
    (uids::JPEG2000, Syntax::J2k),
    (uids::JPEG2000MC_LOSSLESS, Syntax::J2kPart2Lossless),
    (uids::JPEG2000MC, Syntax::J2kPart2),
    (uids::HTJ2K_LOSSLESS, Syntax::HtJ2kLossless),
    (uids::HTJ2K_LOSSLESS_RPCL, Syntax::HtJ2kLosslessRpcl),
    (uids::HTJ2K, Syntax::HtJ2k),
    (uids::JPEGXL_LOSSLESS, Syntax::JxlLossless),
    (uids::JPEGXLJPEG_RECOMPRESSION, Syntax::JxlJpegRecompression),
    (uids::JPEGXL, Syntax::Jxl),
];

impl Syntax {
    pub(crate) fn from_uid(uid: &str) -> Option<(&'static str, Syntax)> {
        let uid = uid.trim_end_matches(['\0', ' ']);
        SYNTAXES.iter().copied().find(|(u, _)| *u == uid)
    }

    pub(crate) fn codec(self) -> Codec {
        use Syntax::*;
        match self {
            JpegBaseline | JpegExtended | JpegLossless | JpegLosslessSv1 => Codec::Jpeg,
            JpegLsLossless | JpegLsNearLossless => Codec::JpegLs,
            J2kLossless | J2k | J2kPart2Lossless | J2kPart2 => Codec::Jpeg2000,
            HtJ2kLossless | HtJ2kLosslessRpcl | HtJ2k => Codec::HtJ2k,
            JxlLossless | JxlJpegRecompression | Jxl => Codec::JpegXl,
        }
    }

    /// JPEG 2000 Part 2: all frames are the components of one codestream.
    pub(crate) fn is_part2(self) -> bool {
        matches!(self, Syntax::J2kPart2Lossless | Syntax::J2kPart2)
    }

    /// Whether the syntax admits lossy coding.
    pub(crate) fn allows_lossy(self) -> bool {
        use Syntax::*;
        matches!(
            self,
            JpegBaseline
                | JpegExtended
                | JpegLsNearLossless
                | J2k
                | J2kPart2
                | HtJ2k
                | JxlJpegRecompression
                | Jxl
        )
    }
}

/// Whether `opts` asks for lossy coding (a quality below 100).
#[cfg(all(feature = "encode", any(feature = "jpeg2000", feature = "jpeg-xl")))]
fn lossy_requested(opts: &EncodeOptions) -> bool {
    !matches!(opts.quality, None | Some(100..))
}

/// Encode one frame for `syntax`.
///
/// Lossless-only syntaxes always encode losslessly. For the others a
/// quality below 100 selects lossy coding: JPEG DCT quality, JPEG-LS NEAR
/// (`near_lossless`, or 100 -> 0, 90 -> 1, ..., default 2), the 9/7 JPEG
/// 2000 path. HTJ2K RPCL uses RPCL order, TLM and one tile-part per
/// resolution; Part 2 transforms single images along their components.
#[cfg(feature = "encode")]
pub(crate) fn encode(syntax: Syntax, image: &Image, opts: &EncodeOptions) -> Result<Encoded> {
    image.info.validate(&crate::Limits::UNLIMITED)?;
    match syntax.codec() {
        Codec::Jpeg => encode_jpeg(syntax, image, opts),
        Codec::JpegLs => encode_jpegls(syntax, image, opts),
        Codec::Jpeg2000 | Codec::HtJ2k => encode_j2k(syntax, image, opts),
        _ => encode_jxl(syntax, image, opts),
    }
}

#[cfg(all(feature = "encode", feature = "jpeg"))]
fn encode_jpeg(syntax: Syntax, image: &Image, opts: &EncodeOptions) -> Result<Encoded> {
    use crate::jpeg::{EncoderOptions, Process};
    let mut o = EncoderOptions::default();
    o.quality = opts.quality.unwrap_or(90);
    o.process = match syntax {
        Syntax::JpegBaseline => Process::Baseline,
        Syntax::JpegExtended => Process::Extended,
        _ => Process::Lossless,
    };
    o.predictor = 1;
    crate::jpeg::encode_with(image, &o)
}

#[cfg(all(feature = "encode", feature = "jpeg-ls"))]
fn encode_jpegls(syntax: Syntax, image: &Image, opts: &EncodeOptions) -> Result<Encoded> {
    let near = if syntax == Syntax::JpegLsLossless {
        0
    } else {
        match (opts.near_lossless, opts.quality) {
            (Some(n), _) => n,
            (None, Some(q)) => (100u16.saturating_sub(q.min(100) as u16) + 5) / 10,
            (None, None) => 2,
        }
    };
    let max_near = (((1u32 << image.info.bits_per_sample.max(2)) - 1) / 2).min(255) as u16;
    let mut o = crate::jpegls::EncoderOptions::default();
    o.near = near.min(max_near);
    crate::jpegls::encode_with(image, &o)
}

#[cfg(all(feature = "encode", feature = "jpeg2000"))]
fn encode_j2k(syntax: Syntax, image: &Image, opts: &EncodeOptions) -> Result<Encoded> {
    use crate::j2k::{ComponentWavelet, EncoderOptions};
    let ht = syntax.codec() == Codec::HtJ2k;
    if ht && !cfg!(feature = "htj2k") {
        return Err(Error::CodecDisabled(Codec::HtJ2k));
    }
    if syntax.is_part2() && !cfg!(feature = "jpeg2000-part2") {
        return Err(Error::unsupported(
            Codec::Jpeg2000,
            "Part 2 multiple component transforms (enable the jpeg2000-part2 feature)",
        ));
    }
    let rpcl = syntax == Syntax::HtJ2kLosslessRpcl;
    let mut o = EncoderOptions::default();
    o.reversible = !(syntax.allows_lossy() && lossy_requested(opts));
    o.quality = opts.quality.unwrap_or(90);
    o.high_throughput = ht;
    o.progression = if rpcl { 2 } else { 0 };
    o.tlm = rpcl;
    o.component_wavelet = (syntax.is_part2() && crate::j2k::part2_uses_wavelet(&image.info))
        .then(|| ComponentWavelet::new(None, 1));
    crate::j2k::encode_with(image, &o)
}

#[cfg(all(feature = "encode", feature = "jpeg-xl"))]
fn encode_jxl(syntax: Syntax, image: &Image, opts: &EncodeOptions) -> Result<Encoded> {
    match syntax {
        Syntax::JxlJpegRecompression => {
            Err(Error::unsupported(Codec::JpegXl, "JPEG recompression"))
        }
        Syntax::Jxl if lossy_requested(opts) => {
            Err(Error::unsupported(Codec::JpegXl, "lossy encoding"))
        }
        _ => crate::jxl::encode_lossless(image),
    }
}

/// Fallbacks for codecs that are not compiled in.
macro_rules! disabled {
    ($($name:ident: $feature:literal),*) => {$(
        #[cfg(all(feature = "encode", not(feature = $feature)))]
        fn $name(syntax: Syntax, _: &Image, _: &EncodeOptions) -> Result<Encoded> {
            Err(Error::CodecDisabled(syntax.codec()))
        }
    )*};
}
disabled!(encode_jpeg: "jpeg", encode_jpegls: "jpeg-ls", encode_j2k: "jpeg2000", encode_jxl: "jpeg-xl");

#[cfg(all(test, feature = "encode", feature = "jpeg-ls"))]
mod tests {
    use super::*;
    use crate::{ColorSpace, DecodeOptions, ImageInfo, Samples};

    fn image() -> Image {
        let v: Vec<u16> = (0..32 * 32).map(|i| ((i * 37) % 4096) as u16).collect();
        let info = ImageInfo {
            width: 32,
            height: 32,
            components: 1,
            bits_per_sample: 12,
            signed: false,
            color_space: ColorSpace::Gray,
        };
        Image::new(info, Samples::U16(v)).unwrap()
    }

    fn max_error(a: &Image, b: &Image) -> i32 {
        let (Samples::U16(a), Samples::U16(b)) = (&a.samples, &b.samples) else {
            panic!("16-bit samples expected")
        };
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x as i32 - y as i32).abs())
            .max()
            .unwrap()
    }

    #[test]
    fn quality_maps_to_near() {
        let img = image();
        for (quality, near) in [(100u8, 0), (90, 1), (50, 5), (1, 10)] {
            let opts = EncodeOptions::new().with_quality(quality);
            let enc = encode(Syntax::JpegLsNearLossless, &img, &opts).unwrap();
            let dec = crate::jpegls::decode(&enc.data, &DecodeOptions::new()).unwrap();
            assert!(max_error(&dec, &img) <= near, "quality {quality}");
        }
        // .80 is always lossless, whatever the quality.
        let opts = EncodeOptions::new().with_quality(10);
        let enc = encode(Syntax::JpegLsLossless, &img, &opts).unwrap();
        assert!(!enc.lossy);
        let dec = crate::jpegls::decode(&enc.data, &DecodeOptions::new()).unwrap();
        assert_eq!(max_error(&dec, &img), 0);
    }

    #[test]
    fn uids_round_trip() {
        for (uid, syntax) in SYNTAXES {
            assert_eq!(Syntax::from_uid(uid), Some((uid, syntax)));
            assert_eq!(
                Syntax::from_uid(&format!("{uid}\0")).map(|s| s.1),
                Some(syntax)
            );
        }
        assert!(Syntax::from_uid("1.2.840.10008.1.2.1").is_none());
    }
}
