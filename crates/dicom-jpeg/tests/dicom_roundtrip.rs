//! Encode native DICOM reference images through the dicom-rs adapter, write a
//! complete DICOM file, read it back, decode it and compare with the original.
//!
//! Each codec block is `pending` (tests `#[ignore]`d) until its encoder lands.

// Imports are unused when every codec feature is disabled.
#![allow(unused_imports)]

mod common;

use common::Expect::*;
use common::check_roundtrip;
use dicom_dictionary_std::uids::*;

macro_rules! cases {
    ($state:ident, $feature:literal, { $($name:ident: $file:literal => $ts:expr, $quality:expr, $expect:expr;)* }) => {
        $( gate!($state, $feature, $name, { check_roundtrip($file, $ts, $quality, $expect) }); )*
    };
}

// Reference images used below (all native, little endian explicit):
//   CT1  512x512   16 bit signed     MR2  1024x1024 12 bit       RG1 1955x1841 15 bit
//   XA1  1024x1024 10 bit            US1  640x480   RGB 8 bit    VL1 756x486   RGB 8 bit
//   SC_rgb 100x100 RGB 8 bit         SC_rgb_small_odd 3x3 RGB    emri_small 64x64x10 frames 12 bit

// ---- JPEG (T.81) -------------------------------------------------------------
cases!(ready, "jpeg", {
    jpeg_baseline_us1: "dicom/wg04/REF/US1_UNC" => JPEG_BASELINE8_BIT, Some(90), Psnr(30.0);
    jpeg_baseline_vl1: "dicom/wg04/REF/VL1_UNC" => JPEG_BASELINE8_BIT, Some(90), Psnr(30.0);
    jpeg_baseline_sc_rgb: "dicom/pydicom/SC_rgb.dcm" => JPEG_BASELINE8_BIT, Some(90), Psnr(30.0);
    jpeg_baseline_small_odd: "dicom/pydicom/SC_rgb_small_odd.dcm" => JPEG_BASELINE8_BIT, Some(90), Psnr(25.0);
    jpeg_extended_mr2: "dicom/wg04/REF/MR2_UNC" => JPEG_EXTENDED12_BIT, Some(90), Psnr(40.0);
    jpeg_extended_xa1: "dicom/wg04/REF/XA1_UNC" => JPEG_EXTENDED12_BIT, Some(90), Psnr(40.0);
    jpeg_extended_emri: "dicom/pydicom/emri_small.dcm" => JPEG_EXTENDED12_BIT, Some(90), Psnr(35.0);
    jpeg_extended_us1: "dicom/wg04/REF/US1_UNC" => JPEG_EXTENDED12_BIT, Some(90), Psnr(30.0);
    jpeg_lossless_ct1: "dicom/wg04/REF/CT1_UNC" => JPEG_LOSSLESS, None, Exact;
    jpeg_lossless_rg1: "dicom/wg04/REF/RG1_UNC" => JPEG_LOSSLESS, None, Exact;
    jpeg_lossless_us1: "dicom/wg04/REF/US1_UNC" => JPEG_LOSSLESS, None, Exact;
    jpeg_lossless_sv1_ct1: "dicom/wg04/REF/CT1_UNC" => JPEG_LOSSLESS_SV1, None, Exact;
    jpeg_lossless_sv1_mr2: "dicom/wg04/REF/MR2_UNC" => JPEG_LOSSLESS_SV1, None, Exact;
    jpeg_lossless_sv1_xa1: "dicom/wg04/REF/XA1_UNC" => JPEG_LOSSLESS_SV1, None, Exact;
    jpeg_lossless_sv1_vl1: "dicom/wg04/REF/VL1_UNC" => JPEG_LOSSLESS_SV1, None, Exact;
    jpeg_lossless_sv1_emri: "dicom/pydicom/emri_small.dcm" => JPEG_LOSSLESS_SV1, None, Exact;
    jpeg_lossless_sv1_small_odd: "dicom/pydicom/SC_rgb_small_odd.dcm" => JPEG_LOSSLESS_SV1, None, Exact;
});

// ---- JPEG-LS (T.87) ------------------------------------------------------------
cases!(ready, "jpeg-ls", {
    jls_lossless_ct1: "dicom/wg04/REF/CT1_UNC" => JPEGLS_LOSSLESS, None, Exact;
    jls_lossless_mr2: "dicom/wg04/REF/MR2_UNC" => JPEGLS_LOSSLESS, None, Exact;
    jls_lossless_rg1: "dicom/wg04/REF/RG1_UNC" => JPEGLS_LOSSLESS, None, Exact;
    jls_lossless_xa1: "dicom/wg04/REF/XA1_UNC" => JPEGLS_LOSSLESS, None, Exact;
    jls_lossless_us1: "dicom/wg04/REF/US1_UNC" => JPEGLS_LOSSLESS, None, Exact;
    jls_lossless_emri: "dicom/pydicom/emri_small.dcm" => JPEGLS_LOSSLESS, None, Exact;
    jls_lossless_small_odd: "dicom/pydicom/SC_rgb_small_odd.dcm" => JPEGLS_LOSSLESS, None, Exact;
    jls_near_ct1: "dicom/wg04/REF/CT1_UNC" => JPEGLS_NEAR_LOSSLESS, Some(90), JlsNear;
    jls_near_mr2: "dicom/wg04/REF/MR2_UNC" => JPEGLS_NEAR_LOSSLESS, Some(90), JlsNear;
    jls_near_us1: "dicom/wg04/REF/US1_UNC" => JPEGLS_NEAR_LOSSLESS, Some(90), JlsNear;
    jls_near_q100_is_lossless: "dicom/wg04/REF/MR2_UNC" => JPEGLS_NEAR_LOSSLESS, Some(100), Exact;
});

// ---- JPEG 2000 (15444-1 / -2) ------------------------------------------------------
cases!(ready, "jpeg2000", {
    j2k_lossless_ct1: "dicom/wg04/REF/CT1_UNC" => JPEG2000_LOSSLESS, None, Exact;
    j2k_lossless_mr2: "dicom/wg04/REF/MR2_UNC" => JPEG2000_LOSSLESS, None, Exact;
    j2k_lossless_rg1: "dicom/wg04/REF/RG1_UNC" => JPEG2000_LOSSLESS, None, Exact;
    j2k_lossless_xa1: "dicom/wg04/REF/XA1_UNC" => JPEG2000_LOSSLESS, None, Exact;
    j2k_lossless_us1: "dicom/wg04/REF/US1_UNC" => JPEG2000_LOSSLESS, None, Exact;
    j2k_lossless_emri: "dicom/pydicom/emri_small.dcm" => JPEG2000_LOSSLESS, None, Exact;
    j2k_lossless_small_odd: "dicom/pydicom/SC_rgb_small_odd.dcm" => JPEG2000_LOSSLESS, None, Exact;
    j2k_lossy_ct1: "dicom/wg04/REF/CT1_UNC" => JPEG2000, Some(90), Psnr(45.0);
    j2k_lossy_mr2: "dicom/wg04/REF/MR2_UNC" => JPEG2000, Some(90), Psnr(45.0);
    j2k_lossy_us1: "dicom/wg04/REF/US1_UNC" => JPEG2000, Some(90), Psnr(35.0);
    j2k_q100_is_lossless: "dicom/wg04/REF/MR2_UNC" => JPEG2000, Some(100), Exact;
});

// Part 2 codes all frames as the components of one codestream.
cases!(ready, "jpeg2000-part2", {
    j2k_part2_lossless_us1: "dicom/wg04/REF/US1_UNC" => JPEG2000MC_LOSSLESS, None, Exact;
    j2k_part2_lossless_ct1: "dicom/wg04/REF/CT1_UNC" => JPEG2000MC_LOSSLESS, None, Exact;
    j2k_part2_lossless_emri: "dicom/pydicom/emri_small.dcm" => JPEG2000MC_LOSSLESS, None, Exact;
    j2k_part2_lossless_small_odd: "dicom/pydicom/SC_rgb_small_odd.dcm" => JPEG2000MC_LOSSLESS, None, Exact;
    j2k_part2_lossy_us1: "dicom/wg04/REF/US1_UNC" => JPEG2000MC, Some(90), Psnr(35.0);
    j2k_part2_lossy_ct1: "dicom/wg04/REF/CT1_UNC" => JPEG2000MC, Some(90), Psnr(45.0);
    j2k_part2_lossy_emri: "dicom/pydicom/emri_small.dcm" => JPEG2000MC, Some(90), Psnr(40.0);
    j2k_part2_q100_is_lossless: "dicom/pydicom/emri_small.dcm" => JPEG2000MC, Some(100), Exact;
});

// ---- HTJ2K (15444-15) ----------------------------------------------------------------
cases!(ready, "htj2k", {
    htj2k_lossless_ct1: "dicom/wg04/REF/CT1_UNC" => HTJ2K_LOSSLESS, None, Exact;
    htj2k_lossless_mr2: "dicom/wg04/REF/MR2_UNC" => HTJ2K_LOSSLESS, None, Exact;
    htj2k_lossless_rg1: "dicom/wg04/REF/RG1_UNC" => HTJ2K_LOSSLESS, None, Exact;
    htj2k_lossless_us1: "dicom/wg04/REF/US1_UNC" => HTJ2K_LOSSLESS, None, Exact;
    htj2k_lossless_emri: "dicom/pydicom/emri_small.dcm" => HTJ2K_LOSSLESS, None, Exact;
    htj2k_rpcl_ct1: "dicom/wg04/REF/CT1_UNC" => HTJ2K_LOSSLESS_RPCL, None, Exact;
    htj2k_rpcl_us1: "dicom/wg04/REF/US1_UNC" => HTJ2K_LOSSLESS_RPCL, None, Exact;
    htj2k_lossy_mr2: "dicom/wg04/REF/MR2_UNC" => HTJ2K, Some(90), Psnr(45.0);
    htj2k_lossy_us1: "dicom/wg04/REF/US1_UNC" => HTJ2K, Some(90), Psnr(35.0);
});

// ---- JPEG XL (18181) -------------------------------------------------------------------
cases!(ready, "jpeg-xl", {
    jxl_lossless_ct1: "dicom/wg04/REF/CT1_UNC" => JPEGXL_LOSSLESS, None, Exact;
    jxl_lossless_mr2: "dicom/wg04/REF/MR2_UNC" => JPEGXL_LOSSLESS, None, Exact;
    jxl_lossless_rg1: "dicom/wg04/REF/RG1_UNC" => JPEGXL_LOSSLESS, None, Exact;
    jxl_lossless_xa1: "dicom/wg04/REF/XA1_UNC" => JPEGXL_LOSSLESS, None, Exact;
    jxl_lossless_us1: "dicom/wg04/REF/US1_UNC" => JPEGXL_LOSSLESS, None, Exact;
    jxl_lossless_vl1: "dicom/wg04/REF/VL1_UNC" => JPEGXL_LOSSLESS, None, Exact;
    jxl_lossless_emri: "dicom/pydicom/emri_small.dcm" => JPEGXL_LOSSLESS, None, Exact;
    jxl_lossless_small_odd: "dicom/pydicom/SC_rgb_small_odd.dcm" => JPEGXL_LOSSLESS, None, Exact;
    jxl_q100_is_lossless: "dicom/wg04/REF/US1_UNC" => JPEGXL, Some(100), Exact;
});

cases!(pending, "jpeg-xl", {
    jxl_lossy_mr2: "dicom/wg04/REF/MR2_UNC" => JPEGXL, Some(90), Psnr(40.0);
    jxl_lossy_us1: "dicom/wg04/REF/US1_UNC" => JPEGXL, Some(90), Psnr(35.0);
});
