//! Image header, metadata and ICC profile (ISO/IEC 18181-1 Annex A / B).

use super::bits::{BitReader, U32D, bits, invalid, val};
use super::entropy::{Histograms, SymbolReader};
use crate::error::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BitDepth {
    pub(crate) float: bool,
    pub(crate) bits: u32,
    pub(crate) exp_bits: u32,
}

impl Default for BitDepth {
    fn default() -> Self {
        BitDepth {
            float: false,
            bits: 8,
            exp_bits: 0,
        }
    }
}

impl BitDepth {
    fn read(br: &mut BitReader) -> Result<Self> {
        if br.read_bool()? {
            let bits = br.u32([val(32), val(16), val(24), bits(6, 1)])?;
            let exp_bits = br.read(4)? + 1;
            let mantissa = bits as i32 - exp_bits as i32 - 1;
            if !(2..=8).contains(&exp_bits) || !(2..=23).contains(&mantissa) {
                return Err(invalid("invalid floating point bit depth"));
            }
            Ok(BitDepth {
                float: true,
                bits,
                exp_bits,
            })
        } else {
            let bits = br.u32([val(8), val(10), val(12), bits(6, 1)])?;
            if bits > 31 {
                return Err(invalid("invalid bit depth"));
            }
            Ok(BitDepth {
                float: false,
                bits,
                exp_bits: 0,
            })
        }
    }
}

/// Extra channel types (Table A.20).
pub(crate) const EC_ALPHA: u32 = 0;
pub(crate) const EC_SPOT: u32 = 2;
pub(crate) const EC_CFA: u32 = 5;

#[derive(Debug, Clone)]
#[allow(dead_code)] // kind / depth are used once extra channels are rendered
pub(crate) struct ExtraChannel {
    pub(crate) kind: u32,
    pub(crate) bit_depth: BitDepth,
    pub(crate) dim_shift: u32,
    pub(crate) alpha_associated: bool,
}

impl ExtraChannel {
    fn read(br: &mut BitReader) -> Result<Self> {
        if br.read_bool()? {
            return Ok(ExtraChannel {
                kind: EC_ALPHA,
                bit_depth: BitDepth::default(),
                dim_shift: 0,
                alpha_associated: false,
            });
        }
        let kind = br.enumeration()?;
        if !(kind <= 6 || kind == 15 || kind == 16) {
            return Err(invalid("unknown extra channel type"));
        }
        let bit_depth = BitDepth::read(br)?;
        let dim_shift = br.u32([val(0), val(3), val(4), bits(3, 1)])?;
        if dim_shift > 3 {
            return Err(invalid("extra channel dim_shift too large"));
        }
        read_name(br)?;
        let mut alpha_associated = false;
        match kind {
            EC_ALPHA => alpha_associated = br.read_bool()?,
            EC_SPOT => {
                for _ in 0..4 {
                    br.f16()?;
                }
            }
            EC_CFA => {
                br.u32([val(1), bits(2, 0), bits(4, 3), bits(8, 19)])?;
            }
            _ => {}
        }
        Ok(ExtraChannel {
            kind,
            bit_depth,
            dim_shift,
            alpha_associated,
        })
    }
}

/// Skip a `Name` field (length-prefixed UTF-8).
pub(crate) fn read_name(br: &mut BitReader) -> Result<()> {
    let len = br.u32([val(0), bits(4, 0), bits(5, 16), bits(10, 48)])?;
    br.skip(len as u64 * 8)
}

/// Skip an `Extensions` bundle.
pub(crate) fn read_extensions(br: &mut BitReader) -> Result<()> {
    let mask = br.u64()?;
    let mut total = 0u64;
    for i in 0..64 {
        if mask >> i & 1 == 1 {
            total = total
                .checked_add(br.u64()?)
                .ok_or_else(|| invalid("extension size overflows"))?;
        }
    }
    br.skip(total)
}

/// Colour spaces (Table A.15).
pub(crate) const CS_RGB: u32 = 0;
pub(crate) const CS_GRAY: u32 = 1;
pub(crate) const CS_XYB: u32 = 2;

#[derive(Debug, Clone)]
pub(crate) struct ColorEncoding {
    pub(crate) want_icc: bool,
    /// sRGB primaries with a D65 white point.
    pub(crate) srgb_gamut: bool,
    pub(crate) color_space: u32,
    /// Transfer function enum, or `None` for a gamma value.
    pub(crate) transfer: Option<u32>,
    pub(crate) gamma: u32,
}

impl Default for ColorEncoding {
    fn default() -> Self {
        ColorEncoding {
            want_icc: false,
            srgb_gamut: true,
            color_space: CS_RGB,
            transfer: Some(13),
            gamma: 0,
        }
    }
}

const XY: [U32D; 4] = [
    bits(19, 0),
    bits(19, 524_288),
    bits(20, 1_048_576),
    bits(21, 2_097_152),
];

impl ColorEncoding {
    fn read(br: &mut BitReader) -> Result<Self> {
        if br.read_bool()? {
            return Ok(Self::default());
        }
        let want_icc = br.read_bool()?;
        let color_space = br.enumeration()?;
        if color_space > 3 {
            return Err(invalid("unknown colour space"));
        }
        let mut ce = ColorEncoding {
            want_icc,
            color_space,
            ..Self::default()
        };
        if !want_icc && color_space != CS_XYB {
            let wp = br.enumeration()?;
            ce.srgb_gamut &= wp == 1;
            match wp {
                1 | 10 | 11 => {}
                2 => {
                    br.u32(XY)?;
                    br.u32(XY)?;
                }
                _ => return Err(invalid("unknown white point")),
            }
            if color_space != CS_GRAY {
                let pr = br.enumeration()?;
                ce.srgb_gamut &= pr == 1;
                match pr {
                    1 | 9 | 11 => {}
                    2 => {
                        for _ in 0..6 {
                            br.u32(XY)?;
                        }
                    }
                    _ => return Err(invalid("unknown primaries")),
                }
            }
        }
        if !want_icc {
            let have_gamma = color_space != CS_XYB && br.read_bool()?;
            if have_gamma {
                ce.gamma = br.read(24)?;
                ce.transfer = None;
                if ce.gamma == 0 || ce.gamma > 10_000_000 {
                    return Err(invalid("invalid gamma"));
                }
            } else if color_space != CS_XYB {
                let tf = br.enumeration()?;
                if ![1, 2, 8, 13, 16, 17, 18].contains(&tf) {
                    return Err(invalid("unknown transfer function"));
                }
                ce.transfer = Some(tf);
            }
            if br.enumeration()? > 3 {
                return Err(invalid("unknown rendering intent"));
            }
        }
        Ok(ce)
    }
}

/// Inverse opsin parameters for XYB images (Annex L).
#[derive(Debug, Clone)]
pub(crate) struct OpsinInverse {
    pub(crate) matrix: [f32; 9],
    pub(crate) bias: [f32; 3],
    pub(crate) quant_bias: [f32; 3],
    pub(crate) quant_bias_numerator: f32,
}

impl Default for OpsinInverse {
    #[allow(clippy::excessive_precision)]
    fn default() -> Self {
        OpsinInverse {
            matrix: [
                11.031566901960783,
                -9.866943921568629,
                -0.16462299647058826,
                -3.254147380392157,
                4.418770392156863,
                -0.16462299647058826,
                -3.6588512862745097,
                2.7129230470588235,
                1.9459282392156863,
            ],
            bias: [-0.0037930732552754493; 3],
            quant_bias: [
                1.0 - 0.05465007330715401,
                1.0 - 0.07005449891748593,
                1.0 - 0.049935103337343655,
            ],
            quant_bias_numerator: 0.145,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ImageHeader {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) orientation: u32,
    pub(crate) preview: Option<(u32, u32)>,
    pub(crate) animation: Option<Animation>,
    pub(crate) bit_depth: BitDepth,
    pub(crate) extra_channels: Vec<ExtraChannel>,
    pub(crate) xyb_encoded: bool,
    pub(crate) color: ColorEncoding,
    pub(crate) opsin: OpsinInverse,
    pub(crate) intensity_target: f32,
    /// Custom upsampling weights for 2x, 4x and 8x (`None` = defaults).
    pub(crate) up_weights: [Option<Vec<f32>>; 3],
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Animation {
    pub(crate) have_timecodes: bool,
}

const SIZE: [U32D; 4] = [bits(9, 1), bits(13, 1), bits(18, 1), bits(30, 1)];

fn aspect(ratio: u32, y: u32) -> u64 {
    let y = y as u64;
    match ratio {
        1 => y,
        2 => y * 12 / 10,
        3 => y * 4 / 3,
        4 => y * 3 / 2,
        5 => y * 16 / 9,
        6 => y * 5 / 4,
        _ => y * 2,
    }
}

/// `SizeHeader`.
fn read_size(br: &mut BitReader) -> Result<(u32, u32)> {
    let small = br.read_bool()?;
    let h = if small {
        (br.read(5)? + 1) * 8
    } else {
        br.u32(SIZE)?
    };
    let ratio = br.read(3)?;
    let w = if ratio == 0 {
        if small {
            ((br.read(5)? + 1) * 8) as u64
        } else {
            br.u32(SIZE)? as u64
        }
    } else {
        aspect(ratio, h)
    };
    if w > 1 << 30 || h > 1 << 30 || w * h as u64 > 1 << 40 {
        return Err(invalid("image dimensions too large"));
    }
    Ok((w as u32, h))
}

/// `PreviewHeader`.
fn read_preview(br: &mut BitReader) -> Result<(u32, u32)> {
    const DIV8: [U32D; 4] = [val(16), val(32), bits(5, 1), bits(9, 33)];
    const FULL: [U32D; 4] = [bits(6, 1), bits(8, 65), bits(10, 321), bits(12, 1345)];
    let div8 = br.read_bool()?;
    let h = if div8 {
        br.u32(DIV8)? * 8
    } else {
        br.u32(FULL)?
    };
    let ratio = br.read(3)?;
    let w = if ratio == 0 {
        (if div8 {
            br.u32(DIV8)? * 8
        } else {
            br.u32(FULL)?
        }) as u64
    } else {
        aspect(ratio, h)
    };
    if w > 4096 || h > 4096 {
        return Err(invalid("preview too large"));
    }
    Ok((w as u32, h))
}

impl ImageHeader {
    /// Parse the signature, `SizeHeader`, `ImageMetadata`, the transform
    /// data and the ICC profile; leaves the reader byte aligned.
    pub(crate) fn read(br: &mut BitReader) -> Result<Self> {
        if br.read(16)? != 0x0AFF {
            return Err(invalid("missing JPEG XL signature"));
        }
        let (width, height) = read_size(br)?;
        let mut h = ImageHeader {
            width,
            height,
            orientation: 1,
            preview: None,
            animation: None,
            bit_depth: BitDepth::default(),
            extra_channels: Vec::new(),
            xyb_encoded: true,
            color: ColorEncoding::default(),
            opsin: OpsinInverse::default(),
            intensity_target: 255.0,
            up_weights: [None, None, None],
        };
        if !br.read_bool()? {
            let extra_fields = br.read_bool()?;
            if extra_fields {
                h.orientation = br.read(3)? + 1;
                if br.read_bool()? {
                    read_size(br)?; // intrinsic size
                }
                if br.read_bool()? {
                    h.preview = Some(read_preview(br)?);
                }
                if br.read_bool()? {
                    br.u32([val(100), val(1000), bits(10, 1), bits(30, 1)])?;
                    br.u32([val(1), val(1001), bits(8, 1), bits(10, 1)])?;
                    br.u32([val(0), bits(3, 0), bits(16, 0), bits(32, 0)])?;
                    h.animation = Some(Animation {
                        have_timecodes: br.read_bool()?,
                    });
                }
            }
            h.bit_depth = BitDepth::read(br)?;
            br.read_bool()?; // modular_16bit_buffers
            let num_ec = br.u32([val(0), val(1), bits(4, 2), bits(12, 1)])?;
            for _ in 0..num_ec {
                h.extra_channels.push(ExtraChannel::read(br)?);
            }
            h.xyb_encoded = br.read_bool()?;
            h.color = ColorEncoding::read(br)?;
            if extra_fields && !br.read_bool()? {
                // Tone mapping.
                h.intensity_target = br.f16()?;
                if h.intensity_target <= 0.0 {
                    return Err(invalid("invalid intensity target"));
                }
                br.f16()?;
                br.read_bool()?;
                br.f16()?;
            }
            read_extensions(br)?;
        }
        // CustomTransformData.
        if !br.read_bool()? {
            if h.xyb_encoded && !br.read_bool()? {
                for m in &mut h.opsin.matrix {
                    *m = br.f16()?;
                }
                for b in &mut h.opsin.bias {
                    *b = br.f16()?;
                }
                for q in &mut h.opsin.quant_bias {
                    *q = br.f16()?;
                }
                h.opsin.quant_bias_numerator = br.f16()?;
            }
            let mask = br.read(3)?;
            for (i, n) in [15usize, 55, 210].into_iter().enumerate() {
                if mask >> i & 1 == 1 {
                    let w = (0..n).map(|_| br.f16()).collect::<Result<Vec<_>>>()?;
                    h.up_weights[i] = Some(w);
                }
            }
        }
        if h.color.want_icc {
            skip_icc(br)?;
        }
        br.zero_pad_to_byte()?;
        Ok(h)
    }
}

/// Decode the entropy-coded ICC profile stream (B.3) and discard it.
fn skip_icc(br: &mut BitReader) -> Result<()> {
    let enc_size = br.u64()?;
    // Real profiles are at most a few megabytes; symbols can be zero bits
    // long, so the size must be bounded independently of the input length.
    if enc_size > 1 << 22 {
        return Err(invalid("ICC profile too large"));
    }
    let h = Histograms::decode(br, 41)?;
    let mut reader = SymbolReader::new(&h, br, 0)?;
    let (mut b1, mut b2) = (0u32, 0u32);
    for i in 0..enc_size {
        let ctx = icc_context(i, b1, b2);
        let b = reader.read(&h, br, ctx)?;
        if b > 255 {
            return Err(invalid("ICC byte out of range"));
        }
        b2 = b1;
        b1 = b;
    }
    reader.finish()
}

fn icc_context(i: u64, b1: u32, b2: u32) -> usize {
    if i <= 128 {
        return 0;
    }
    let class = |b: u32| -> u32 {
        if (b'a' as u32..=b'z' as u32).contains(&b) || (b'A' as u32..=b'Z' as u32).contains(&b) {
            0
        } else if b == b'.' as u32 || b == b',' as u32 || (b'0' as u32..=b'9' as u32).contains(&b) {
            1
        } else if b <= 1 {
            2 + b
        } else if b <= 15 {
            4
        } else if b > 240 {
            5 + (b == 255) as u32
        } else {
            7
        }
    };
    let p1 = class(b1);
    let p2 = match b2 {
        b if (b'a' as u32..=b'z' as u32).contains(&b)
            || (b'A' as u32..=b'Z' as u32).contains(&b) =>
        {
            0
        }
        b if b == b'.' as u32 || b == b',' as u32 || (b'0' as u32..=b'9' as u32).contains(&b) => 1,
        b if b <= 15 => 2,
        b if b > 240 => 3,
        _ => 4,
    };
    (1 + p1 + p2 * 8) as usize
}
