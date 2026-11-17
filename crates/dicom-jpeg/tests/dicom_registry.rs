//! The `inventory-registry` feature: dicom-rs finds the adapter through its
//! transfer syntax registry, so `dicom_pixeldata::PixelDecoder` decodes the
//! JPEG family with it. dicom-pixeldata is built without its own codecs
//! here, so every entry checked must come from dicom-jpeg.

mod common;

use dicom_encoding::adapters::PixelDataReader;
use dicom_encoding::{Codec, TransferSyntaxIndex};
use dicom_jpeg::dicom_rs::PixelAdapter;
use dicom_pixeldata::PixelDecoder;
use dicom_transfer_syntax_registry::TransferSyntaxRegistry;

/// Whether the codec behind `uid` is compiled in.
fn enabled(uid: &str) -> bool {
    let adapter = PixelAdapter::from_uid(uid).expect("supported UID");
    let part2 = uid.ends_with(".92") || uid.ends_with(".93");
    dicom_jpeg::codec_enabled(adapter.codec()) && (!part2 || cfg!(feature = "jpeg2000-part2"))
}

#[test]
fn every_enabled_syntax_is_registered() {
    for uid in PixelAdapter::supported_uids().filter(|u| enabled(u)) {
        let ts = TransferSyntaxRegistry
            .get(uid)
            .unwrap_or_else(|| panic!("{uid} not registered"));
        assert!(ts.can_decode_all(), "{uid}: no decoder");
        let encodes = matches!(ts.codec(), Codec::EncapsulatedPixelData(_, Some(_)));
        assert_eq!(encodes, cfg!(feature = "encode"), "{uid}: encoder");
    }
}

#[test]
fn pixeldata_decodes_through_the_registry() {
    let files = [
        "dicom/wg04/JPLL/CT1_JPLL",
        "dicom/wg04/JPLY/MR1_JPLY",
        "dicom/pydicom/SC_rgb_jpeg_gdcm.dcm",
        "dicom/wg04/JLSL/CT1_JLSL",
        "dicom/wg04/JLSN/CT1_JLSN",
        "dicom/wg04/J2KR/CT1_J2KR",
        "dicom/wg04/J2KI/CT1_J2KI",
        "dicom/pydicom/HTJ2KLossless_08_RGB.dcm",
        "dicom/pydicom/HTJ2K_08_RGB.dcm",
    ];
    let mut checked = 0;
    for file in files {
        let obj = common::open(file);
        let uid = obj.meta().transfer_syntax().trim_end_matches('\0');
        if !enabled(uid) {
            continue;
        }
        let decoded = obj
            .decode_pixel_data()
            .unwrap_or_else(|e| panic!("{file}: {e}"));
        let mut direct = Vec::new();
        PixelAdapter::from_uid(uid)
            .unwrap()
            .decode(&obj, &mut direct)
            .unwrap();
        assert_eq!(decoded.data(), &direct[..], "{file}");
        checked += 1;
    }
    assert!(checked > 0 || !cfg!(feature = "jpeg"));
}

/// dicom-pixeldata's `Transcode` encodes through the registered encoder:
/// native references, transcoded to each lossless syntax and decoded again,
/// come back bit-identical.
#[cfg(feature = "encode")]
#[test]
fn pixeldata_transcodes_through_the_registry() {
    use dicom_dictionary_std::uids;
    use dicom_pixeldata::Transcode;

    let targets = [
        uids::JPEG_LOSSLESS_SV1,
        uids::JPEGLS_LOSSLESS,
        uids::JPEG2000_LOSSLESS,
        uids::HTJ2K_LOSSLESS,
        uids::JPEGXL_LOSSLESS,
    ];
    for file in ["dicom/wg04/REF/CT1_UNC", "dicom/wg04/REF/VL1_UNC"] {
        let original = common::open(file);
        let native = original.decode_pixel_data().unwrap().data().to_vec();
        for uid in targets.into_iter().filter(|u| enabled(u)) {
            let ts = TransferSyntaxRegistry.get(uid).unwrap();
            let mut obj = original.clone();
            obj.transcode(ts)
                .unwrap_or_else(|e| panic!("{file} -> {uid}: {e}"));
            assert_eq!(obj.meta().transfer_syntax().trim_end_matches('\0'), uid);
            let decoded = obj.decode_pixel_data().unwrap();
            assert_eq!(decoded.data(), &native[..], "{file} -> {uid}");
        }
    }
}
