//! Encode native DICOM reference images through the dicom-rs adapter, write a
//! complete DICOM file, read it back, decode it and compare with the original.
//!
//! Each codec block is `pending` (tests `#[ignore]`d) until its encoder lands.

// Imports are unused when every codec feature is disabled.
#![allow(unused_imports)]

mod common;

use common::Expect::*;
use common::check_roundtrip;
use dicom_jpeg::TransferSyntax::*;

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
    jpeg_baseline_us1: "dicom/wg04/REF/US1_UNC" => JpegBaseline, Some(90), Psnr(30.0);
    jpeg_baseline_vl1: "dicom/wg04/REF/VL1_UNC" => JpegBaseline, Some(90), Psnr(30.0);
    jpeg_baseline_sc_rgb: "dicom/pydicom/SC_rgb.dcm" => JpegBaseline, Some(90), Psnr(30.0);
    jpeg_baseline_small_odd: "dicom/pydicom/SC_rgb_small_odd.dcm" => JpegBaseline, Some(90), Psnr(25.0);
    jpeg_extended_mr2: "dicom/wg04/REF/MR2_UNC" => JpegExtended, Some(90), Psnr(40.0);
    jpeg_extended_xa1: "dicom/wg04/REF/XA1_UNC" => JpegExtended, Some(90), Psnr(40.0);
    jpeg_extended_emri: "dicom/pydicom/emri_small.dcm" => JpegExtended, Some(90), Psnr(35.0);
    jpeg_extended_us1: "dicom/wg04/REF/US1_UNC" => JpegExtended, Some(90), Psnr(30.0);
    jpeg_lossless_ct1: "dicom/wg04/REF/CT1_UNC" => JpegLossless, None, Exact;
    jpeg_lossless_rg1: "dicom/wg04/REF/RG1_UNC" => JpegLossless, None, Exact;
    jpeg_lossless_us1: "dicom/wg04/REF/US1_UNC" => JpegLossless, None, Exact;
    jpeg_lossless_sv1_ct1: "dicom/wg04/REF/CT1_UNC" => JpegLosslessSv1, None, Exact;
    jpeg_lossless_sv1_mr2: "dicom/wg04/REF/MR2_UNC" => JpegLosslessSv1, None, Exact;
    jpeg_lossless_sv1_xa1: "dicom/wg04/REF/XA1_UNC" => JpegLosslessSv1, None, Exact;
    jpeg_lossless_sv1_vl1: "dicom/wg04/REF/VL1_UNC" => JpegLosslessSv1, None, Exact;
    jpeg_lossless_sv1_emri: "dicom/pydicom/emri_small.dcm" => JpegLosslessSv1, None, Exact;
    jpeg_lossless_sv1_small_odd: "dicom/pydicom/SC_rgb_small_odd.dcm" => JpegLosslessSv1, None, Exact;
});

// ---- JPEG-LS (T.87) ------------------------------------------------------------
cases!(ready, "jpeg-ls", {
    jls_lossless_ct1: "dicom/wg04/REF/CT1_UNC" => JpegLsLossless, None, Exact;
    jls_lossless_mr2: "dicom/wg04/REF/MR2_UNC" => JpegLsLossless, None, Exact;
    jls_lossless_rg1: "dicom/wg04/REF/RG1_UNC" => JpegLsLossless, None, Exact;
    jls_lossless_xa1: "dicom/wg04/REF/XA1_UNC" => JpegLsLossless, None, Exact;
    jls_lossless_us1: "dicom/wg04/REF/US1_UNC" => JpegLsLossless, None, Exact;
    jls_lossless_emri: "dicom/pydicom/emri_small.dcm" => JpegLsLossless, None, Exact;
    jls_lossless_small_odd: "dicom/pydicom/SC_rgb_small_odd.dcm" => JpegLsLossless, None, Exact;
    jls_near_ct1: "dicom/wg04/REF/CT1_UNC" => JpegLsNearLossless, Some(90), JlsNear;
    jls_near_mr2: "dicom/wg04/REF/MR2_UNC" => JpegLsNearLossless, Some(90), JlsNear;
    jls_near_us1: "dicom/wg04/REF/US1_UNC" => JpegLsNearLossless, Some(90), JlsNear;
    jls_near_q100_is_lossless: "dicom/wg04/REF/MR2_UNC" => JpegLsNearLossless, Some(100), Exact;
});

// ---- JPEG 2000 (15444-1 / -2) ------------------------------------------------------
cases!(ready, "jpeg2000", {
    j2k_lossless_ct1: "dicom/wg04/REF/CT1_UNC" => Jpeg2000Lossless, None, Exact;
    j2k_lossless_mr2: "dicom/wg04/REF/MR2_UNC" => Jpeg2000Lossless, None, Exact;
    j2k_lossless_rg1: "dicom/wg04/REF/RG1_UNC" => Jpeg2000Lossless, None, Exact;
    j2k_lossless_xa1: "dicom/wg04/REF/XA1_UNC" => Jpeg2000Lossless, None, Exact;
    j2k_lossless_us1: "dicom/wg04/REF/US1_UNC" => Jpeg2000Lossless, None, Exact;
    j2k_lossless_emri: "dicom/pydicom/emri_small.dcm" => Jpeg2000Lossless, None, Exact;
    j2k_lossless_small_odd: "dicom/pydicom/SC_rgb_small_odd.dcm" => Jpeg2000Lossless, None, Exact;
    j2k_lossy_ct1: "dicom/wg04/REF/CT1_UNC" => Jpeg2000, Some(90), Psnr(45.0);
    j2k_lossy_mr2: "dicom/wg04/REF/MR2_UNC" => Jpeg2000, Some(90), Psnr(45.0);
    j2k_lossy_us1: "dicom/wg04/REF/US1_UNC" => Jpeg2000, Some(90), Psnr(35.0);
    j2k_q100_is_lossless: "dicom/wg04/REF/MR2_UNC" => Jpeg2000, Some(100), Exact;
});

cases!(pending, "jpeg2000-part2", {
    j2k_part2_lossless_us1: "dicom/wg04/REF/US1_UNC" => Jpeg2000Part2Lossless, None, Exact;
    j2k_part2_lossless_ct1: "dicom/wg04/REF/CT1_UNC" => Jpeg2000Part2Lossless, None, Exact;
    j2k_part2_lossy_us1: "dicom/wg04/REF/US1_UNC" => Jpeg2000Part2, Some(90), Psnr(35.0);
});

// ---- HTJ2K (15444-15) ----------------------------------------------------------------
cases!(pending, "htj2k", {
    htj2k_lossless_ct1: "dicom/wg04/REF/CT1_UNC" => HtJ2kLossless, None, Exact;
    htj2k_lossless_mr2: "dicom/wg04/REF/MR2_UNC" => HtJ2kLossless, None, Exact;
    htj2k_lossless_rg1: "dicom/wg04/REF/RG1_UNC" => HtJ2kLossless, None, Exact;
    htj2k_lossless_us1: "dicom/wg04/REF/US1_UNC" => HtJ2kLossless, None, Exact;
    htj2k_lossless_emri: "dicom/pydicom/emri_small.dcm" => HtJ2kLossless, None, Exact;
    htj2k_rpcl_ct1: "dicom/wg04/REF/CT1_UNC" => HtJ2kLosslessRpcl, None, Exact;
    htj2k_rpcl_us1: "dicom/wg04/REF/US1_UNC" => HtJ2kLosslessRpcl, None, Exact;
    htj2k_lossy_mr2: "dicom/wg04/REF/MR2_UNC" => HtJ2k, Some(90), Psnr(45.0);
    htj2k_lossy_us1: "dicom/wg04/REF/US1_UNC" => HtJ2k, Some(90), Psnr(35.0);
});

// ---- JPEG XL (18181) -------------------------------------------------------------------
cases!(pending, "jpeg-xl", {
    jxl_lossless_ct1: "dicom/wg04/REF/CT1_UNC" => JpegXlLossless, None, Exact;
    jxl_lossless_mr2: "dicom/wg04/REF/MR2_UNC" => JpegXlLossless, None, Exact;
    jxl_lossless_xa1: "dicom/wg04/REF/XA1_UNC" => JpegXlLossless, None, Exact;
    jxl_lossless_us1: "dicom/wg04/REF/US1_UNC" => JpegXlLossless, None, Exact;
    jxl_lossless_emri: "dicom/pydicom/emri_small.dcm" => JpegXlLossless, None, Exact;
    jxl_lossy_mr2: "dicom/wg04/REF/MR2_UNC" => JpegXl, Some(90), Psnr(40.0);
    jxl_lossy_us1: "dicom/wg04/REF/US1_UNC" => JpegXl, Some(90), Psnr(35.0);
    jxl_q100_is_lossless: "dicom/wg04/REF/US1_UNC" => JpegXl, Some(100), Exact;
});
