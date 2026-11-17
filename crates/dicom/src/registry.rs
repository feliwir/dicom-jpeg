//! Registration of [`PixelAdapter`] in the dicom-rs transfer syntax
//! registry (`inventory-registry` feature).
//!
//! dicom-rs adds these entries after its built-in ones and lets an entry
//! replace a built-in one only if it does strictly more (a stub, or a
//! decoder without encoder): dicom-rs's own JPEG baseline codec (feature
//! `jpeg` of dicom-pixeldata / dicom-transfer-syntax-registry) stays in
//! place unless that feature is disabled.

use dicom_dictionary_std::uids;
#[cfg(not(feature = "encode"))]
use dicom_encoding::NeverPixelAdapter;
use dicom_encoding::{Codec, NeverAdapter, submit_ele_transfer_syntax};

use crate::PixelAdapter;

/// The registry entry of `uid`: decoder, and encoder with the `encode`
/// feature.
macro_rules! register {
    ($feature:literal, $uid:expr, $name:literal) => {
        #[cfg(feature = $feature)]
        submit_ele_transfer_syntax!($uid, $name, {
            let adapter = PixelAdapter::from_uid($uid);
            #[cfg(feature = "encode")]
            let codec =
                Codec::<NeverAdapter, _, PixelAdapter>::EncapsulatedPixelData(adapter, adapter);
            #[cfg(not(feature = "encode"))]
            let codec =
                Codec::<NeverAdapter, _, NeverPixelAdapter>::EncapsulatedPixelData(adapter, None);
            codec
        });
    };
}

register!(
    "jpeg",
    uids::JPEG_BASELINE8_BIT,
    "JPEG Baseline (Process 1)"
);
register!(
    "jpeg",
    uids::JPEG_EXTENDED12_BIT,
    "JPEG Extended (Process 2 & 4)"
);
register!(
    "jpeg",
    uids::JPEG_LOSSLESS,
    "JPEG Lossless, Non-Hierarchical (Process 14)"
);
register!(
    "jpeg",
    uids::JPEG_LOSSLESS_SV1,
    "JPEG Lossless, Non-Hierarchical, First-Order Prediction"
);
register!(
    "jpeg-ls",
    uids::JPEGLS_LOSSLESS,
    "JPEG-LS Lossless Image Compression"
);
register!(
    "jpeg-ls",
    uids::JPEGLS_NEAR_LOSSLESS,
    "JPEG-LS Lossy (Near-Lossless) Image Compression"
);
register!(
    "jpeg2000",
    uids::JPEG2000_LOSSLESS,
    "JPEG 2000 Image Compression (Lossless Only)"
);
register!("jpeg2000", uids::JPEG2000, "JPEG 2000 Image Compression");
register!(
    "jpeg2000-part2",
    uids::JPEG2000MC_LOSSLESS,
    "JPEG 2000 Part 2 Multi-component Image Compression (Lossless Only)"
);
register!(
    "jpeg2000-part2",
    uids::JPEG2000MC,
    "JPEG 2000 Part 2 Multi-component Image Compression"
);
register!(
    "htj2k",
    uids::HTJ2K_LOSSLESS,
    "High-Throughput JPEG 2000 Image Compression (Lossless Only)"
);
register!(
    "htj2k",
    uids::HTJ2K_LOSSLESS_RPCL,
    "High-Throughput JPEG 2000 with RPCL Options Image Compression (Lossless Only)"
);
register!(
    "htj2k",
    uids::HTJ2K,
    "High-Throughput JPEG 2000 Image Compression"
);
register!("jpeg-xl", uids::JPEGXL_LOSSLESS, "JPEG XL Lossless");
register!(
    "jpeg-xl",
    uids::JPEGXLJPEG_RECOMPRESSION,
    "JPEG XL JPEG Recompression"
);
register!("jpeg-xl", uids::JPEGXL, "JPEG XL");
