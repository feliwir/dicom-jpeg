//! Colour transforms shared by the codecs.

/// Full-range ITU-R BT.601 YCbCr, as used by JFIF / DICOM `YBR_FULL` and the
/// JPEG 2000 irreversible component transform (ICT). Chroma is centred on 0.
pub mod ycbcr {
    /// Luma weights.
    pub const KR: f64 = 0.299;
    pub const KG: f64 = 0.587;
    pub const KB: f64 = 0.114;
    /// Forward chroma weights.
    pub const CB_R: f64 = -0.168_736;
    pub const CB_G: f64 = -0.331_264;
    pub const CR_G: f64 = -0.418_688;
    pub const CR_B: f64 = -0.081_312;
    /// Inverse weights: R = Y + CR_TO_R Cr, G = Y - CB_TO_G Cb - CR_TO_G Cr,
    /// B = Y + CB_TO_B Cb.
    pub const CR_TO_R: f64 = 1.402;
    pub const CB_TO_G: f64 = 0.344_136;
    pub const CR_TO_G: f64 = 0.714_136;
    pub const CB_TO_B: f64 = 1.772;

    /// RGB -> YCbCr (centred chroma).
    #[inline]
    pub fn forward(r: f32, g: f32, b: f32) -> [f32; 3] {
        [
            KR as f32 * r + KG as f32 * g + KB as f32 * b,
            CB_R as f32 * r + CB_G as f32 * g + 0.5 * b,
            0.5 * r + CR_G as f32 * g + CR_B as f32 * b,
        ]
    }

    /// YCbCr (centred chroma) -> RGB.
    #[inline]
    pub fn inverse(y: f32, cb: f32, cr: f32) -> [f32; 3] {
        [
            y + CR_TO_R as f32 * cr,
            y - CB_TO_G as f32 * cb - CR_TO_G as f32 * cr,
            y + CB_TO_B as f32 * cb,
        ]
    }

    /// Squared L2 norm of each column of the inverse transform: how much
    /// error energy in Y, Cb and Cr reaches R, G and B together.
    pub const fn inverse_energy() -> [f64; 3] {
        [
            3.0,
            CB_TO_G * CB_TO_G + CB_TO_B * CB_TO_B,
            CR_TO_R * CR_TO_R + CR_TO_G * CR_TO_G,
        ]
    }
}

/// The JPEG 2000 reversible component transform (RCT).
pub mod rct {
    /// RGB -> (Y, Cb, Cr) = (floor((R + 2G + B) / 4), B - G, R - G).
    #[inline]
    pub fn forward(r: i32, g: i32, b: i32) -> [i32; 3] {
        [(r + 2 * g + b) >> 2, b - g, r - g]
    }

    /// Exact inverse of [`forward`].
    #[inline]
    pub fn inverse(y: i32, cb: i32, cr: i32) -> [i32; 3] {
        let g = y - ((cb + cr) >> 2);
        [cr + g, g, cb + g]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rct_round_trip() {
        for r in (-300..300).step_by(37) {
            for g in (-300..300).step_by(41) {
                for b in (-300..300).step_by(43) {
                    let [y, cb, cr] = rct::forward(r, g, b);
                    assert_eq!(rct::inverse(y, cb, cr), [r, g, b]);
                }
            }
        }
    }

    #[test]
    fn ycbcr_round_trip() {
        let [y, cb, cr] = ycbcr::forward(200.0, 30.0, 90.0);
        let [r, g, b] = ycbcr::inverse(y, cb, cr);
        assert!((r - 200.0).abs() < 0.01 && (g - 30.0).abs() < 0.01 && (b - 90.0).abs() < 0.01);
    }
}
