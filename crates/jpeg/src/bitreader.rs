//! Entropy-coded segment bit reader (T.81 F.2.2.5): removes stuffed zero
//! bytes and stops at markers, supplying zero bits from then on as libjpeg
//! does for truncated data.

pub(crate) type BitReader<'a> =
    dicom_jpeg_common::bitio::BitReader<'a, dicom_jpeg_common::bitio::T81>;
