//! Decode every JPEG-family DICOM test file through the dicom-rs adapter.
//!
//! Generated from the contents of `testdata/dicom`; expectations are tuned by hand.
//! Each codec block is `pending` (tests `#[ignore]`d) until that codec is implemented.

// Imports are unused when every codec feature is disabled.
#![allow(unused_imports)]

mod common;

use common::Expect::*;
use common::check_decode;

macro_rules! cases {
    ($state:ident, $feature:literal, { $($name:ident: $file:literal => $reference:expr, $expect:expr;)* }) => {
        $( gate!($state, $feature, $name, { check_decode($file, $reference, $expect) }); )*
    };
}

// ---- jpeg ------------------------------------------------------------
cases!(ready, "jpeg", {
    pydicom_color3d_jpeg_baseline: "dicom/pydicom/color3d_jpeg_baseline.dcm" => None, Decodes;
    pydicom_jpglosslessp14sv1_1s_1f_8b: "dicom/pydicom/JPGLosslessP14SV1_1s_1f_8b.dcm" => None, Decodes;
    pydicom_sc_rgb_dcmtk_eb_cr: "dicom/pydicom/SC_rgb_dcmtk_+eb+cr.dcm" => Some("dicom/pydicom/SC_rgb_dcmtk_ebcr_dcmd.dcm"), MaxAbs(1);
    pydicom_sc_rgb_dcmtk_eb_cy_n1: "dicom/pydicom/SC_rgb_dcmtk_+eb+cy+n1.dcm" => Some("dicom/pydicom/SC_rgb_dcmtk_ebcyn1_dcmd.dcm"), MaxAbs(1);
    pydicom_sc_rgb_dcmtk_eb_cy_n2: "dicom/pydicom/SC_rgb_dcmtk_+eb+cy+n2.dcm" => Some("dicom/pydicom/SC_rgb_dcmtk_ebcyn2_dcmd.dcm"), MaxAbs(1);
    pydicom_sc_rgb_dcmtk_eb_cy_np: "dicom/pydicom/SC_rgb_dcmtk_+eb+cy+np.dcm" => Some("dicom/pydicom/SC_rgb_dcmtk_ebcynp_dcmd.dcm"), MaxAbs(1);
    pydicom_sc_rgb_dcmtk_eb_cy_s2: "dicom/pydicom/SC_rgb_dcmtk_+eb+cy+s2.dcm" => Some("dicom/pydicom/SC_rgb_dcmtk_ebcys2_dcmd.dcm"), MaxAbs(1);
    pydicom_sc_rgb_dcmtk_eb_cy_s4: "dicom/pydicom/SC_rgb_dcmtk_+eb+cy+s4.dcm" => Some("dicom/pydicom/SC_rgb_dcmtk_ebcys4_dcmd.dcm"), MaxAbs(1);
    pydicom_sc_rgb_jpeg_dcmtk: "dicom/pydicom/SC_rgb_jpeg_dcmtk.dcm" => Some("dicom/pydicom/SC_rgb.dcm"), Psnr(30.0);
    pydicom_sc_rgb_jpeg_gdcm: "dicom/pydicom/SC_rgb_jpeg_gdcm.dcm" => Some("dicom/pydicom/SC_rgb.dcm"), Exact;
        // Labelled RGB but stored as 4:2:0 YCbCr; strongly compressed.
    pydicom_sc_rgb_jpeg_lossy_gdcm: "dicom/pydicom/SC_rgb_jpeg_lossy_gdcm.dcm" => Some("dicom/pydicom/SC_rgb.dcm"), Psnr(25.0);
    pydicom_sc_rgb_small_odd_jpeg: "dicom/pydicom/SC_rgb_small_odd_jpeg.dcm" => Some("dicom/pydicom/SC_rgb_small_odd.dcm"), Psnr(30.0);
    pylibjpeg_jpegbaseline_jpegbaseline_1s_1f_u_08_08: "dicom/pylibjpeg/JPEGBaseline/JPEGBaseline_1s_1f_u_08_08.dcm" => None, Decodes;
    pylibjpeg_jpegbaseline_sc_jpeg_no_color_transform_2: "dicom/pylibjpeg/JPEGBaseline/SC_jpeg_no_color_transform_2.dcm" => None, Decodes;
    pylibjpeg_jpegextended_jpegextended_1s_1f_u_16_12: "dicom/pylibjpeg/JPEGExtended/JPEGExtended_1s_1f_u_16_12.dcm" => None, Decodes;
    pylibjpeg_jpegextended_jpegextended_3s_1f_u_08_08: "dicom/pylibjpeg/JPEGExtended/JPEGExtended_3s_1f_u_08_08.dcm" => None, Decodes;
    pylibjpeg_jpegextended_rg2_jply_fixed: "dicom/pylibjpeg/JPEGExtended/RG2_JPLY_fixed.dcm" => None, Decodes;
    pylibjpeg_jpeglossless_jpeglossless_1s_1f_i_16_16_bad: "dicom/pylibjpeg/JPEGLossless/JPEGLossless_1s_1f_i_16_16_bad.dcm" => None, Decodes;
    pylibjpeg_jpeglossless_jpeglossless_1s_1f_u_16_12: "dicom/pylibjpeg/JPEGLossless/JPEGLossless_1s_1f_u_16_12.dcm" => None, Decodes;
    pylibjpeg_jpeglosslesssv1_532_jpeglossless_voi: "dicom/pylibjpeg/JPEGLosslessSV1/532_JPEGLossless_VOI.dcm" => None, Decodes;
    pylibjpeg_jpeglosslesssv1_jpeglosslessp14sv1_1s_1f_u_16_16: "dicom/pylibjpeg/JPEGLosslessSV1/JPEGLosslessP14SV1_1s_1f_u_16_16.dcm" => None, Decodes;
    pylibjpeg_jpeglosslesssv1_jpeglosslessp14sv1_3s_2f_u_08_08: "dicom/pylibjpeg/JPEGLosslessSV1/JPEGLosslessP14SV1_3s_2f_u_08_08.dcm" => None, Decodes;
    wg04_jpll_ct1_jpll: "dicom/wg04/JPLL/CT1_JPLL" => Some("dicom/wg04/REF/CT1_UNC"), Exact;
    wg04_jpll_ct2_jpll: "dicom/wg04/JPLL/CT2_JPLL" => Some("dicom/wg04/REF/CT2_UNC"), Exact;
    wg04_jpll_mg1_jpll: "dicom/wg04/JPLL/MG1_JPLL" => Some("dicom/wg04/REF/MG1_UNC"), Exact;
    wg04_jpll_mr1_jpll: "dicom/wg04/JPLL/MR1_JPLL" => Some("dicom/wg04/REF/MR1_UNC"), Exact;
    wg04_jpll_mr2_jpll: "dicom/wg04/JPLL/MR2_JPLL" => Some("dicom/wg04/REF/MR2_UNC"), Exact;
    wg04_jpll_mr3_jpll: "dicom/wg04/JPLL/MR3_JPLL" => Some("dicom/wg04/REF/MR3_UNC"), Exact;
    wg04_jpll_mr4_jpll: "dicom/wg04/JPLL/MR4_JPLL" => Some("dicom/wg04/REF/MR4_UNC"), Exact;
    wg04_jpll_nm1_jpll: "dicom/wg04/JPLL/NM1_JPLL" => Some("dicom/wg04/REF/NM1_UNC"), Exact;
    wg04_jpll_rg1_jpll: "dicom/wg04/JPLL/RG1_JPLL" => Some("dicom/wg04/REF/RG1_UNC"), Exact;
    wg04_jpll_rg2_jpll: "dicom/wg04/JPLL/RG2_JPLL" => Some("dicom/wg04/REF/RG2_UNC"), Exact;
    wg04_jpll_rg3_jpll: "dicom/wg04/JPLL/RG3_JPLL" => Some("dicom/wg04/REF/RG3_UNC"), Exact;
    wg04_jpll_sc1_jpll: "dicom/wg04/JPLL/SC1_JPLL" => Some("dicom/wg04/REF/SC1_UNC"), Exact;
    wg04_jpll_xa1_jpll: "dicom/wg04/JPLL/XA1_JPLL" => Some("dicom/wg04/REF/XA1_UNC"), Exact;
    wg04_jply_mg1_jply: "dicom/wg04/JPLY/MG1_JPLY" => Some("dicom/wg04/REF/MG1_UNC"), Psnr(44.0);
    // MR1 is 16 bit; the 12-bit lossy version cannot be compared with the original.
    wg04_jply_mr1_jply: "dicom/wg04/JPLY/MR1_JPLY" => None, Decodes;
    wg04_jply_mr2_jply: "dicom/wg04/JPLY/MR2_JPLY" => Some("dicom/wg04/REF/MR2_UNC"), Psnr(44.0);
    wg04_jply_mr3_jply: "dicom/wg04/JPLY/MR3_JPLY" => Some("dicom/wg04/REF/MR3_UNC"), Psnr(44.0);
    wg04_jply_mr4_jply: "dicom/wg04/JPLY/MR4_JPLY" => Some("dicom/wg04/REF/MR4_UNC"), Psnr(44.0);
    wg04_jply_nm1_jply: "dicom/wg04/JPLY/NM1_JPLY" => Some("dicom/wg04/REF/NM1_UNC"), Psnr(44.0);
    wg04_jply_rg2_jply: "dicom/wg04/JPLY/RG2_JPLY" => Some("dicom/wg04/REF/RG2_UNC"), Psnr(44.0);
    wg04_jply_rg3_jply: "dicom/wg04/JPLY/RG3_JPLY" => Some("dicom/wg04/REF/RG3_UNC"), Psnr(44.0);
    wg04_jply_sc1_jply: "dicom/wg04/JPLY/SC1_JPLY" => Some("dicom/wg04/REF/SC1_UNC"), Psnr(44.0);
    wg04_jply_xa1_jply: "dicom/wg04/JPLY/XA1_JPLY" => Some("dicom/wg04/REF/XA1_UNC"), Psnr(44.0);
});

// ---- jpegls ------------------------------------------------------------
cases!(pending, "jpeg-ls", {
    pydicom_emri_small_jpeg_ls_lossless: "dicom/pydicom/emri_small_jpeg_ls_lossless.dcm" => Some("dicom/pydicom/emri_small.dcm"), Exact;
    pydicom_jlsl_08_07_0_1f: "dicom/pydicom/JLSL_08_07_0_1F.dcm" => None, Decodes;
    pydicom_jlsl_16_15_1_1f: "dicom/pydicom/JLSL_16_15_1_1F.dcm" => None, Decodes;
    pydicom_jlsl_rgb_ilv0: "dicom/pydicom/JLSL_RGB_ILV0.dcm" => None, Decodes;
    pydicom_jlsl_rgb_ilv1: "dicom/pydicom/JLSL_RGB_ILV1.dcm" => None, Decodes;
    pydicom_jlsl_rgb_ilv2: "dicom/pydicom/JLSL_RGB_ILV2.dcm" => None, Decodes;
    pydicom_jlsn_rgb_ilv0: "dicom/pydicom/JLSN_RGB_ILV0.dcm" => None, Decodes;
    pydicom_mr_small_jpeg_ls_lossless: "dicom/pydicom/MR_small_jpeg_ls_lossless.dcm" => Some("dicom/pydicom/MR_small.dcm"), Exact;
    wg04_jlsl_ct1_jlsl: "dicom/wg04/JLSL/CT1_JLSL" => Some("dicom/wg04/REF/CT1_UNC"), Exact;
    wg04_jlsl_ct2_jlsl: "dicom/wg04/JLSL/CT2_JLSL" => Some("dicom/wg04/REF/CT2_UNC"), Exact;
    wg04_jlsl_mg1_jlsl: "dicom/wg04/JLSL/MG1_JLSL" => Some("dicom/wg04/REF/MG1_UNC"), Exact;
    wg04_jlsl_mr1_jlsl: "dicom/wg04/JLSL/MR1_JLSL" => Some("dicom/wg04/REF/MR1_UNC"), Exact;
    wg04_jlsl_mr2_jlsl: "dicom/wg04/JLSL/MR2_JLSL" => Some("dicom/wg04/REF/MR2_UNC"), Exact;
    wg04_jlsl_mr3_jlsl: "dicom/wg04/JLSL/MR3_JLSL" => Some("dicom/wg04/REF/MR3_UNC"), Exact;
    wg04_jlsl_mr4_jlsl: "dicom/wg04/JLSL/MR4_JLSL" => Some("dicom/wg04/REF/MR4_UNC"), Exact;
    wg04_jlsl_nm1_jlsl: "dicom/wg04/JLSL/NM1_JLSL" => Some("dicom/wg04/REF/NM1_UNC"), Exact;
    wg04_jlsl_rg1_jlsl: "dicom/wg04/JLSL/RG1_JLSL" => Some("dicom/wg04/REF/RG1_UNC"), Exact;
    wg04_jlsl_rg2_jlsl: "dicom/wg04/JLSL/RG2_JLSL" => Some("dicom/wg04/REF/RG2_UNC"), Exact;
    wg04_jlsl_rg3_jlsl: "dicom/wg04/JLSL/RG3_JLSL" => Some("dicom/wg04/REF/RG3_UNC"), Exact;
    wg04_jlsl_sc1_jlsl: "dicom/wg04/JLSL/SC1_JLSL" => Some("dicom/wg04/REF/SC1_UNC"), Exact;
    wg04_jlsl_xa1_jlsl: "dicom/wg04/JLSL/XA1_JLSL" => Some("dicom/wg04/REF/XA1_UNC"), Exact;
    wg04_jlsn_ct1_jlsn: "dicom/wg04/JLSN/CT1_JLSN" => Some("dicom/wg04/REF/CT1_UNC"), JlsNear;
    wg04_jlsn_ct2_jlsn: "dicom/wg04/JLSN/CT2_JLSN" => Some("dicom/wg04/REF/CT2_UNC"), JlsNear;
    wg04_jlsn_mg1_jlsn: "dicom/wg04/JLSN/MG1_JLSN" => Some("dicom/wg04/REF/MG1_UNC"), JlsNear;
    wg04_jlsn_mr1_jlsn: "dicom/wg04/JLSN/MR1_JLSN" => Some("dicom/wg04/REF/MR1_UNC"), JlsNear;
    wg04_jlsn_mr2_jlsn: "dicom/wg04/JLSN/MR2_JLSN" => Some("dicom/wg04/REF/MR2_UNC"), JlsNear;
    wg04_jlsn_mr3_jlsn: "dicom/wg04/JLSN/MR3_JLSN" => Some("dicom/wg04/REF/MR3_UNC"), JlsNear;
    wg04_jlsn_mr4_jlsn: "dicom/wg04/JLSN/MR4_JLSN" => Some("dicom/wg04/REF/MR4_UNC"), JlsNear;
    wg04_jlsn_nm1_jlsn: "dicom/wg04/JLSN/NM1_JLSN" => Some("dicom/wg04/REF/NM1_UNC"), JlsNear;
    wg04_jlsn_rg1_jlsn: "dicom/wg04/JLSN/RG1_JLSN" => Some("dicom/wg04/REF/RG1_UNC"), JlsNear;
    wg04_jlsn_rg2_jlsn: "dicom/wg04/JLSN/RG2_JLSN" => Some("dicom/wg04/REF/RG2_UNC"), JlsNear;
    wg04_jlsn_rg3_jlsn: "dicom/wg04/JLSN/RG3_JLSN" => Some("dicom/wg04/REF/RG3_UNC"), JlsNear;
    wg04_jlsn_sc1_jlsn: "dicom/wg04/JLSN/SC1_JLSN" => Some("dicom/wg04/REF/SC1_UNC"), JlsNear;
    wg04_jlsn_xa1_jlsn: "dicom/wg04/JLSN/XA1_JLSN" => Some("dicom/wg04/REF/XA1_UNC"), JlsNear;
});

// ---- j2k ------------------------------------------------------------
cases!(pending, "jpeg2000", {
    pydicom_693_j2ki: "dicom/pydicom/693_J2KI.dcm" => Some("dicom/pydicom/693_UNCI.dcm"), MaxAbs(1);
    pydicom_693_j2kr: "dicom/pydicom/693_J2KR.dcm" => Some("dicom/pydicom/693_UNCR.dcm"), Exact;
    pydicom_emri_small_jpeg_2k_lossless: "dicom/pydicom/emri_small_jpeg_2k_lossless.dcm" => Some("dicom/pydicom/emri_small.dcm"), Exact;
    pydicom_emri_small_jpeg_2k_lossless_too_short: "dicom/pydicom/emri_small_jpeg_2k_lossless_too_short.dcm" => None, Decodes;
    pydicom_jpeg2000: "dicom/pydicom/JPEG2000.dcm" => Some("dicom/pydicom/JPEG2000_UNC.dcm"), Psnr(40.0);
    pydicom_mr_small_jp2klossless: "dicom/pydicom/MR_small_jp2klossless.dcm" => Some("dicom/pydicom/MR_small.dcm"), Exact;
    pydicom_sc_rgb_gdcm_ky: "dicom/pydicom/SC_rgb_gdcm_KY.dcm" => Some("dicom/pydicom/SC_rgb_gdcm2k_uncompressed.dcm"), MaxAbs(1);
    pylibjpeg_jpeg2000_elscint1_jp2vsj2k: "dicom/pylibjpeg/JPEG2000/ELSCINT1_JP2vsJ2K.dcm" => None, Decodes;
    pylibjpeg_jpeg2000_marotech_ct_jp2lossy: "dicom/pylibjpeg/JPEG2000/MAROTECH_CT_JP2Lossy.dcm" => None, Decodes;
    pylibjpeg_jpeg2000_osirixfake16bitsstoredfakespacing: "dicom/pylibjpeg/JPEG2000/OsirixFake16BitsStoredFakeSpacing.dcm" => None, Decodes;
    pylibjpeg_jpeg2000lossless_966: "dicom/pylibjpeg/JPEG2000Lossless/966.dcm" => None, Decodes;
    pylibjpeg_jpeg2000lossless_966_fixed: "dicom/pylibjpeg/JPEG2000Lossless/966_fixed.dcm" => None, Decodes;
    pylibjpeg_jpeg2000lossless_explicit_vr_un: "dicom/pylibjpeg/JPEG2000Lossless/explicit_VR-UN.dcm" => None, Decodes;
    pylibjpeg_jpeg2000lossless_gdcmj2k_textgbr: "dicom/pylibjpeg/JPEG2000Lossless/GDCMJ2K_TextGBR.dcm" => None, Decodes;
    pylibjpeg_jpeg2000lossless_j2k_pixelrep_mismatch: "dicom/pylibjpeg/JPEG2000Lossless/J2K_pixelrep_mismatch.dcm" => None, Decodes;
    pylibjpeg_jpeg2000lossless_jpeg2klossless_1s_1f_u_16_16: "dicom/pylibjpeg/JPEG2000Lossless/JPEG2KLossless_1s_1f_u_16_16.dcm" => None, Decodes;
    pylibjpeg_jpeg2000lossless_nm_kakadu44_sotmarkerincons: "dicom/pylibjpeg/JPEG2000Lossless/NM_Kakadu44_SOTmarkerincons.dcm" => None, Decodes;
    pylibjpeg_jpeg2000lossless_toshiba_j2k_openjpegv2regression: "dicom/pylibjpeg/JPEG2000Lossless/TOSHIBA_J2K_OpenJPEGv2Regression.dcm" => None, Decodes;
    pylibjpeg_jpeg2000lossless_toshiba_j2k_siz0_pixrep1: "dicom/pylibjpeg/JPEG2000Lossless/TOSHIBA_J2K_SIZ0_PixRep1.dcm" => None, Decodes;
    pylibjpeg_jpeg2000lossless_toshiba_j2k_siz1_pixrep0: "dicom/pylibjpeg/JPEG2000Lossless/TOSHIBA_J2K_SIZ1_PixRep0.dcm" => None, Decodes;
    wg04_j2ki_ct1_j2ki: "dicom/wg04/J2KI/CT1_J2KI" => Some("dicom/wg04/REF/CT1_UNC"), Psnr(30.0);
    wg04_j2ki_ct2_j2ki: "dicom/wg04/J2KI/CT2_J2KI" => Some("dicom/wg04/REF/CT2_UNC"), Psnr(30.0);
    wg04_j2ki_mg1_j2ki: "dicom/wg04/J2KI/MG1_J2KI" => Some("dicom/wg04/REF/MG1_UNC"), Psnr(30.0);
    wg04_j2ki_mr1_j2ki: "dicom/wg04/J2KI/MR1_J2KI" => Some("dicom/wg04/REF/MR1_UNC"), Psnr(30.0);
    wg04_j2ki_mr2_j2ki: "dicom/wg04/J2KI/MR2_J2KI" => Some("dicom/pydicom/MR2_UNCI.dcm"), MaxAbs(1);
    wg04_j2ki_mr3_j2ki: "dicom/wg04/J2KI/MR3_J2KI" => Some("dicom/wg04/REF/MR3_UNC"), Psnr(30.0);
    wg04_j2ki_mr4_j2ki: "dicom/wg04/J2KI/MR4_J2KI" => Some("dicom/wg04/REF/MR4_UNC"), Psnr(30.0);
    wg04_j2ki_nm1_j2ki: "dicom/wg04/J2KI/NM1_J2KI" => Some("dicom/wg04/REF/NM1_UNC"), Psnr(30.0);
    wg04_j2ki_rg1_j2ki: "dicom/wg04/J2KI/RG1_J2KI" => Some("dicom/pydicom/RG1_UNCI.dcm"), MaxAbs(1);
    wg04_j2ki_rg2_j2ki: "dicom/wg04/J2KI/RG2_J2KI" => Some("dicom/wg04/REF/RG2_UNC"), Psnr(30.0);
    wg04_j2ki_rg3_j2ki: "dicom/wg04/J2KI/RG3_J2KI" => Some("dicom/pydicom/RG3_UNCI.dcm"), MaxAbs(1);
    wg04_j2ki_sc1_j2ki: "dicom/wg04/J2KI/SC1_J2KI" => Some("dicom/wg04/REF/SC1_UNC"), Psnr(30.0);
    wg04_j2ki_us1_j2ki: "dicom/wg04/J2KI/US1_J2KI" => Some("dicom/wg04/REF/US1_UNC"), Psnr(30.0);
    wg04_j2ki_vl1_j2ki: "dicom/wg04/J2KI/VL1_J2KI" => Some("dicom/wg04/REF/VL1_UNC"), Psnr(30.0);
    wg04_j2ki_vl2_j2ki: "dicom/wg04/J2KI/VL2_J2KI" => Some("dicom/wg04/REF/VL2_UNC"), Psnr(30.0);
    wg04_j2ki_vl3_j2ki: "dicom/wg04/J2KI/VL3_J2KI" => Some("dicom/wg04/REF/VL3_UNC"), Psnr(30.0);
    wg04_j2ki_vl4_j2ki: "dicom/wg04/J2KI/VL4_J2KI" => Some("dicom/wg04/REF/VL4_UNC"), Psnr(30.0);
    wg04_j2ki_vl5_j2ki: "dicom/wg04/J2KI/VL5_J2KI" => Some("dicom/wg04/REF/VL5_UNC"), Psnr(30.0);
    wg04_j2ki_vl6_j2ki: "dicom/wg04/J2KI/VL6_J2KI" => Some("dicom/wg04/REF/VL6_UNC"), Psnr(30.0);
    wg04_j2ki_xa1_j2ki: "dicom/wg04/J2KI/XA1_J2KI" => Some("dicom/wg04/REF/XA1_UNC"), Psnr(30.0);
    wg04_j2kr_ct1_j2kr: "dicom/wg04/J2KR/CT1_J2KR" => Some("dicom/wg04/REF/CT1_UNC"), Exact;
    wg04_j2kr_ct2_j2kr: "dicom/wg04/J2KR/CT2_J2KR" => Some("dicom/wg04/REF/CT2_UNC"), Exact;
    wg04_j2kr_mg1_j2kr: "dicom/wg04/J2KR/MG1_J2KR" => Some("dicom/wg04/REF/MG1_UNC"), Exact;
    wg04_j2kr_mr1_j2kr: "dicom/wg04/J2KR/MR1_J2KR" => Some("dicom/wg04/REF/MR1_UNC"), Exact;
    wg04_j2kr_mr2_j2kr: "dicom/wg04/J2KR/MR2_J2KR" => Some("dicom/wg04/REF/MR2_UNC"), Exact;
    wg04_j2kr_mr3_j2kr: "dicom/wg04/J2KR/MR3_J2KR" => Some("dicom/wg04/REF/MR3_UNC"), Exact;
    wg04_j2kr_mr4_j2kr: "dicom/wg04/J2KR/MR4_J2KR" => Some("dicom/wg04/REF/MR4_UNC"), Exact;
    wg04_j2kr_nm1_j2kr: "dicom/wg04/J2KR/NM1_J2KR" => Some("dicom/wg04/REF/NM1_UNC"), Exact;
    wg04_j2kr_rg1_j2kr: "dicom/wg04/J2KR/RG1_J2KR" => Some("dicom/wg04/REF/RG1_UNC"), Exact;
    wg04_j2kr_rg2_j2kr: "dicom/wg04/J2KR/RG2_J2KR" => Some("dicom/wg04/REF/RG2_UNC"), Exact;
    wg04_j2kr_rg3_j2kr: "dicom/wg04/J2KR/RG3_J2KR" => Some("dicom/wg04/REF/RG3_UNC"), Exact;
    wg04_j2kr_sc1_j2kr: "dicom/wg04/J2KR/SC1_J2KR" => Some("dicom/wg04/REF/SC1_UNC"), Exact;
    wg04_j2kr_us1_j2kr: "dicom/wg04/J2KR/US1_J2KR" => Some("dicom/wg04/REF/US1_UNC"), Exact;
    wg04_j2kr_vl1_j2kr: "dicom/wg04/J2KR/VL1_J2KR" => Some("dicom/wg04/REF/VL1_UNC"), Exact;
    wg04_j2kr_vl2_j2kr: "dicom/wg04/J2KR/VL2_J2KR" => Some("dicom/wg04/REF/VL2_UNC"), Exact;
    wg04_j2kr_vl3_j2kr: "dicom/wg04/J2KR/VL3_J2KR" => Some("dicom/wg04/REF/VL3_UNC"), Exact;
    wg04_j2kr_vl4_j2kr: "dicom/wg04/J2KR/VL4_J2KR" => Some("dicom/wg04/REF/VL4_UNC"), Exact;
    wg04_j2kr_vl5_j2kr: "dicom/wg04/J2KR/VL5_J2KR" => Some("dicom/wg04/REF/VL5_UNC"), Exact;
    wg04_j2kr_vl6_j2kr: "dicom/wg04/J2KR/VL6_J2KR" => Some("dicom/wg04/REF/VL6_UNC"), Exact;
    wg04_j2kr_xa1_j2kr: "dicom/wg04/J2KR/XA1_J2KR" => Some("dicom/wg04/REF/XA1_UNC"), Exact;
});

// ---- htj2k ------------------------------------------------------------
cases!(pending, "htj2k", {
    pydicom_htj2k_08_rgb: "dicom/pydicom/HTJ2K_08_RGB.dcm" => None, Decodes;
    pydicom_htj2klossless_08_rgb: "dicom/pydicom/HTJ2KLossless_08_RGB.dcm" => None, Decodes;
});
