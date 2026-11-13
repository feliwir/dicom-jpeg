//! DICOM transfer syntaxes covered by this crate.

use std::fmt;

/// Codec families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Codec {
    /// ITU-T T.81 | ISO/IEC 10918-1 (DCT sequential and lossless).
    Jpeg,
    /// ITU-T T.87 | ISO/IEC 14495-1.
    JpegLs,
    /// ITU-T T.800 | ISO/IEC 15444-1 (and Part 2 multi-component).
    Jpeg2000,
    /// ITU-T T.814 | ISO/IEC 15444-15.
    HtJ2k,
    /// ISO/IEC 18181.
    JpegXl,
}

impl Codec {
    /// Whether support for this codec is compiled in.
    pub const fn is_enabled(self) -> bool {
        match self {
            Codec::Jpeg => cfg!(feature = "jpeg"),
            Codec::JpegLs => cfg!(feature = "jpeg-ls"),
            Codec::Jpeg2000 => cfg!(feature = "jpeg2000"),
            Codec::HtJ2k => cfg!(feature = "htj2k"),
            Codec::JpegXl => cfg!(feature = "jpeg-xl"),
        }
    }

    /// Short human readable name.
    pub const fn name(self) -> &'static str {
        match self {
            Codec::Jpeg => "JPEG",
            Codec::JpegLs => "JPEG-LS",
            Codec::Jpeg2000 => "JPEG 2000",
            Codec::HtJ2k => "HTJ2K",
            Codec::JpegXl => "JPEG XL",
        }
    }
}

impl fmt::Display for Codec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

macro_rules! transfer_syntaxes {
    ($( $(#[$doc:meta])* $variant:ident = $uid:literal, $name:literal, $codec:ident, lossless: $lossless:literal, lossy: $lossy:literal; )*) => {
        /// The active DICOM transfer syntaxes of the JPEG family.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[non_exhaustive]
        pub enum TransferSyntax {
            $( $(#[$doc])* $variant, )*
        }

        impl TransferSyntax {
            /// All supported transfer syntaxes.
            pub const ALL: &'static [TransferSyntax] = &[$(TransferSyntax::$variant),*];

            /// The transfer syntax UID.
            pub const fn uid(self) -> &'static str {
                match self { $(TransferSyntax::$variant => $uid,)* }
            }

            /// The DICOM name of the transfer syntax.
            pub const fn name(self) -> &'static str {
                match self { $(TransferSyntax::$variant => $name,)* }
            }

            /// The codec family used by the transfer syntax.
            pub const fn codec(self) -> Codec {
                match self { $(TransferSyntax::$variant => Codec::$codec,)* }
            }

            /// Whether this syntax can carry mathematically lossless data.
            pub const fn supports_lossless(self) -> bool {
                match self { $(TransferSyntax::$variant => $lossless,)* }
            }

            /// Whether this syntax can carry lossy data.
            pub const fn supports_lossy(self) -> bool {
                match self { $(TransferSyntax::$variant => $lossy,)* }
            }

            /// Look up a transfer syntax by UID.
            ///
            /// Trailing NUL / space padding (as found in DICOM UI values) is ignored.
            pub fn from_uid(uid: &str) -> Option<Self> {
                match uid.trim_end_matches(['\0', ' ']) {
                    $($uid => Some(TransferSyntax::$variant),)*
                    _ => None,
                }
            }
        }
    };
}

transfer_syntaxes! {
    /// JPEG Baseline (Process 1), 8 bit lossy.
    JpegBaseline = "1.2.840.10008.1.2.4.50", "JPEG Baseline (Process 1)", Jpeg, lossless: false, lossy: true;
    /// JPEG Extended (Process 2 & 4), 8/12 bit lossy.
    JpegExtended = "1.2.840.10008.1.2.4.51", "JPEG Extended (Process 2 & 4)", Jpeg, lossless: false, lossy: true;
    /// JPEG Lossless, Non-Hierarchical (Process 14), any predictor.
    JpegLossless = "1.2.840.10008.1.2.4.57", "JPEG Lossless, Non-Hierarchical (Process 14)", Jpeg, lossless: true, lossy: false;
    /// JPEG Lossless, Non-Hierarchical, First-Order Prediction (Process 14, Selection Value 1).
    JpegLosslessSv1 = "1.2.840.10008.1.2.4.70", "JPEG Lossless, Non-Hierarchical, First-Order Prediction (Process 14 [Selection Value 1])", Jpeg, lossless: true, lossy: false;
    /// JPEG-LS Lossless.
    JpegLsLossless = "1.2.840.10008.1.2.4.80", "JPEG-LS Lossless Image Compression", JpegLs, lossless: true, lossy: false;
    /// JPEG-LS Lossy (Near-Lossless).
    JpegLsNearLossless = "1.2.840.10008.1.2.4.81", "JPEG-LS Lossy (Near-Lossless) Image Compression", JpegLs, lossless: true, lossy: true;
    /// JPEG 2000 Lossless Only.
    Jpeg2000Lossless = "1.2.840.10008.1.2.4.90", "JPEG 2000 Image Compression (Lossless Only)", Jpeg2000, lossless: true, lossy: false;
    /// JPEG 2000 (lossless or lossy).
    Jpeg2000 = "1.2.840.10008.1.2.4.91", "JPEG 2000 Image Compression", Jpeg2000, lossless: true, lossy: true;
    /// JPEG 2000 Part 2 Multi-component Lossless Only.
    Jpeg2000Part2Lossless = "1.2.840.10008.1.2.4.92", "JPEG 2000 Part 2 Multi-component Image Compression (Lossless Only)", Jpeg2000, lossless: true, lossy: false;
    /// JPEG 2000 Part 2 Multi-component.
    Jpeg2000Part2 = "1.2.840.10008.1.2.4.93", "JPEG 2000 Part 2 Multi-component Image Compression", Jpeg2000, lossless: true, lossy: true;
    /// JPEG XL Lossless.
    JpegXlLossless = "1.2.840.10008.1.2.4.110", "JPEG XL Lossless", JpegXl, lossless: true, lossy: false;
    /// JPEG XL JPEG Recompression.
    JpegXlJpegRecompression = "1.2.840.10008.1.2.4.111", "JPEG XL JPEG Recompression", JpegXl, lossless: false, lossy: true;
    /// JPEG XL (lossless or lossy).
    JpegXl = "1.2.840.10008.1.2.4.112", "JPEG XL", JpegXl, lossless: true, lossy: true;
    /// High-Throughput JPEG 2000 Lossless Only.
    HtJ2kLossless = "1.2.840.10008.1.2.4.201", "High-Throughput JPEG 2000 Image Compression (Lossless Only)", HtJ2k, lossless: true, lossy: false;
    /// High-Throughput JPEG 2000 with RPCL Options Lossless Only.
    HtJ2kLosslessRpcl = "1.2.840.10008.1.2.4.202", "High-Throughput JPEG 2000 with RPCL Options Image Compression (Lossless Only)", HtJ2k, lossless: true, lossy: false;
    /// High-Throughput JPEG 2000 (lossless or lossy).
    HtJ2k = "1.2.840.10008.1.2.4.203", "High-Throughput JPEG 2000 Image Compression", HtJ2k, lossless: true, lossy: true;
}

impl fmt::Display for TransferSyntax {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uid_roundtrip() {
        for &ts in TransferSyntax::ALL {
            assert_eq!(TransferSyntax::from_uid(ts.uid()), Some(ts));
        }
        assert_eq!(
            TransferSyntax::from_uid("1.2.840.10008.1.2.4.50\0"),
            Some(TransferSyntax::JpegBaseline)
        );
        assert_eq!(TransferSyntax::from_uid("1.2.840.10008.1.2.1"), None);
    }
}
