//! Frame header and table of contents (ISO/IEC 18181-1 clause C / F).

use super::bits::{BitReader, U32D, bits, invalid, val};
use super::entropy::{Histograms, SymbolReader, read_permutation};
use super::headers::{ImageHeader, read_extensions, read_name};
use dicom_jpeg_common::Result;

pub(crate) const FRAME_REGULAR: u32 = 0;
pub(crate) const FRAME_LF: u32 = 1;
pub(crate) const FRAME_REFERENCE_ONLY: u32 = 2;
pub(crate) const FRAME_SKIP_PROGRESSIVE: u32 = 3;

pub(crate) const BLEND_REPLACE: u32 = 0;

pub(crate) const FLAG_NOISE: u64 = 0x1;
pub(crate) const FLAG_PATCHES: u64 = 0x2;
pub(crate) const FLAG_SPLINES: u64 = 0x10;
pub(crate) const FLAG_USE_LF_FRAME: u64 = 0x20;
pub(crate) const FLAG_SKIP_ADAPTIVE_LF_SMOOTHING: u64 = 0x80;

#[derive(Debug, Clone)]
pub(crate) struct Passes {
    pub(crate) num_passes: u32,
    /// Coefficient shift of each pass (the last pass has none).
    pub(crate) shift: Vec<u32>,
    /// (downsample, last_pass) pairs.
    pub(crate) downsample: Vec<(u32, u32)>,
}

/// Edge-preserving filter parameters.
#[derive(Debug, Clone, Copy)]
pub(crate) struct EpfParams {
    pub(crate) sharp_lut: [f32; 8],
    pub(crate) channel_scale: [f32; 3],
    pub(crate) quant_mul: f32,
    pub(crate) pass0_sigma_scale: f32,
    pub(crate) pass2_sigma_scale: f32,
    pub(crate) border_sad_mul: f32,
    pub(crate) sigma_for_modular: f32,
}

impl Default for EpfParams {
    fn default() -> Self {
        EpfParams {
            sharp_lut: [
                0.0,
                1.0 / 7.0,
                2.0 / 7.0,
                3.0 / 7.0,
                4.0 / 7.0,
                5.0 / 7.0,
                6.0 / 7.0,
                1.0,
            ],
            channel_scale: [40.0, 5.0, 3.5],
            quant_mul: 0.46,
            pass0_sigma_scale: 0.9,
            pass2_sigma_scale: 6.5,
            border_sad_mul: 2.0 / 3.0,
            sigma_for_modular: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)] // source is used once blending is implemented
pub(crate) struct Blending {
    pub(crate) mode: u32,
    pub(crate) source: u32,
}

#[derive(Debug, Clone)]
pub(crate) struct FrameHeader {
    pub(crate) frame_type: u32,
    pub(crate) modular: bool,
    pub(crate) flags: u64,
    pub(crate) do_ycbcr: bool,
    pub(crate) jpeg_upsampling: [u32; 3],
    pub(crate) upsampling: u32,
    pub(crate) ec_upsampling: Vec<u32>,
    pub(crate) group_size_shift: u32,
    pub(crate) x_qm_scale: u32,
    pub(crate) b_qm_scale: u32,
    pub(crate) passes: Passes,
    pub(crate) lf_level: u32,
    pub(crate) have_crop: bool,
    pub(crate) x0: i32,
    pub(crate) y0: i32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) blending: Blending,
    pub(crate) is_last: bool,
    pub(crate) save_as_reference: u32,
    pub(crate) save_before_ct: bool,
    pub(crate) gaborish: bool,
    /// Gaborish weights (side, diagonal) per XYB channel.
    pub(crate) gab_weights: [[f32; 2]; 3],
    pub(crate) epf_iters: u32,
    pub(crate) epf: EpfParams,
    /// Number of colour channels coded in the frame (1 or 3).
    pub(crate) color_channels: usize,
}

const CROP: [U32D; 4] = [bits(8, 0), bits(11, 256), bits(14, 2304), bits(30, 18688)];

fn read_blending(br: &mut BitReader, have_ec: bool, full_frame: bool) -> Result<Blending> {
    let mode = br.u32([val(0), val(1), val(2), bits(2, 3)])?;
    if mode > 4 {
        return Err(invalid("unknown blend mode"));
    }
    if have_ec && (mode == 2 || mode == 3) {
        br.u32([val(0), val(1), val(2), bits(3, 3)])?; // alpha channel
    }
    if (have_ec && (mode == 2 || mode == 3)) || mode == 4 {
        br.read_bool()?; // clamp
    }
    let source = if !(mode == BLEND_REPLACE && full_frame) {
        br.read(2)?
    } else {
        0
    };
    Ok(Blending { mode, source })
}

impl FrameHeader {
    pub(crate) fn read(br: &mut BitReader, ih: &ImageHeader) -> Result<Self> {
        let all_default = br.read_bool()?;
        let n_ec = ih.extra_channels.len();
        let mut fh = FrameHeader {
            frame_type: FRAME_REGULAR,
            modular: false,
            flags: 0,
            do_ycbcr: false,
            jpeg_upsampling: [0; 3],
            upsampling: 1,
            ec_upsampling: vec![1; n_ec],
            group_size_shift: 1,
            x_qm_scale: if ih.xyb_encoded { 3 } else { 2 },
            b_qm_scale: 2,
            passes: Passes {
                num_passes: 1,
                shift: Vec::new(),
                downsample: Vec::new(),
            },
            lf_level: 0,
            have_crop: false,
            x0: 0,
            y0: 0,
            width: ih.width,
            height: ih.height,
            blending: Blending {
                mode: BLEND_REPLACE,
                source: 0,
            },
            is_last: true,
            save_as_reference: 0,
            save_before_ct: false,
            gaborish: true,
            gab_weights: [[0.115_169_525, 0.061_248_592]; 3],
            epf_iters: 2,
            epf: EpfParams::default(),
            color_channels: 3,
        };
        if !all_default {
            fh.frame_type = br.read(2)?;
            fh.modular = br.read_bool()?;
            fh.flags = br.u64()?;
            if !ih.xyb_encoded {
                fh.do_ycbcr = br.read_bool()?;
            }
            let use_lf_frame = fh.flags & FLAG_USE_LF_FRAME != 0;
            if !use_lf_frame {
                if fh.do_ycbcr {
                    for j in &mut fh.jpeg_upsampling {
                        *j = br.read(2)?;
                    }
                }
                fh.upsampling = br.u32([val(1), val(2), val(4), val(8)])?;
                for e in &mut fh.ec_upsampling {
                    *e = br.u32([val(1), val(2), val(4), val(8)])?;
                }
            }
            if fh.modular {
                fh.group_size_shift = br.read(2)?;
            }
            if ih.xyb_encoded && !fh.modular {
                fh.x_qm_scale = br.read(3)?;
                fh.b_qm_scale = br.read(3)?;
            }
            if fh.frame_type != FRAME_REFERENCE_ONLY {
                let num_passes = br.u32([val(1), val(2), val(3), bits(3, 4)])?;
                let mut downsample = Vec::new();
                let mut shift = Vec::new();
                if num_passes != 1 {
                    let num_ds = br.u32([val(0), val(1), val(2), bits(1, 3)])?;
                    if num_ds >= num_passes {
                        return Err(invalid("too many downsampled passes"));
                    }
                    for _ in 0..num_passes - 1 {
                        shift.push(br.read(2)?);
                    }
                    let ds = (0..num_ds)
                        .map(|_| br.u32([val(1), val(2), val(4), val(8)]))
                        .collect::<Result<Vec<_>>>()?;
                    for d in ds {
                        let last = br.u32([val(0), val(1), val(2), bits(3, 0)])?;
                        if last >= num_passes {
                            return Err(invalid("invalid last pass"));
                        }
                        downsample.push((d, last));
                    }
                }
                fh.passes = Passes {
                    num_passes,
                    shift,
                    downsample,
                };
            }
            if fh.frame_type == FRAME_LF {
                fh.lf_level = br.read(2)? + 1;
            } else {
                fh.have_crop = br.read_bool()?;
            }
            if fh.have_crop {
                if fh.frame_type != FRAME_REFERENCE_ONLY {
                    fh.x0 = super::bits::unpack_signed(br.u32(CROP)?);
                    fh.y0 = super::bits::unpack_signed(br.u32(CROP)?);
                }
                fh.width = br.u32(CROP)?;
                fh.height = br.u32(CROP)?;
            }
            let normal = fh.frame_type == FRAME_REGULAR || fh.frame_type == FRAME_SKIP_PROGRESSIVE;
            let full_frame = !fh.have_crop
                || (fh.x0 <= 0
                    && fh.y0 <= 0
                    && fh.x0 as i64 + fh.width as i64 >= ih.width as i64
                    && fh.y0 as i64 + fh.height as i64 >= ih.height as i64);
            let mut duration = 0;
            if normal {
                fh.blending = read_blending(br, n_ec > 0, full_frame)?;
                for _ in 0..n_ec {
                    read_blending(br, true, full_frame)?;
                }
                if ih.animation.is_some() {
                    duration = br.u32([val(0), val(1), bits(8, 0), bits(32, 0)])?;
                }
                if ih.animation.is_some_and(|a| a.have_timecodes) {
                    br.read(32)?;
                }
                fh.is_last = br.read_bool()?;
            } else {
                fh.is_last = false;
            }
            if fh.frame_type != FRAME_LF && !fh.is_last {
                fh.save_as_reference = br.read(2)?;
            }
            let resets_canvas = fh.blending.mode == BLEND_REPLACE && full_frame;
            let can_reference = !fh.is_last
                && (duration == 0 || fh.save_as_reference != 0)
                && fh.frame_type != FRAME_LF;
            fh.save_before_ct = !normal;
            if fh.frame_type == FRAME_REFERENCE_ONLY || (resets_canvas && can_reference) {
                fh.save_before_ct = br.read_bool()?;
            }
            read_name(br)?;
            // RestorationFilter.
            if !br.read_bool()? {
                fh.gaborish = br.read_bool()?;
                if fh.gaborish && br.read_bool()? {
                    for weights in &mut fh.gab_weights {
                        for wt in weights.iter_mut() {
                            *wt = br.f16()?;
                        }
                        let [w1, w2] = *weights;
                        if (1.0 + (w1 + w2) * 4.0).abs() < 1e-8 {
                            return Err(invalid("degenerate Gaborish weights"));
                        }
                    }
                }
                fh.epf_iters = br.read(2)?;
                if fh.epf_iters > 0 {
                    let epf = &mut fh.epf;
                    if !fh.modular && br.read_bool()? {
                        for v in &mut epf.sharp_lut {
                            *v = br.f16()?;
                        }
                    }
                    if br.read_bool()? {
                        for v in &mut epf.channel_scale {
                            *v = br.f16()?;
                        }
                        br.read(32)?; // zero-flush parameters (unused)
                    }
                    if br.read_bool()? {
                        if !fh.modular {
                            epf.quant_mul = br.f16()?;
                        }
                        epf.pass0_sigma_scale = br.f16()?;
                        epf.pass2_sigma_scale = br.f16()?;
                        epf.border_sad_mul = br.f16()?;
                    }
                    if fh.modular {
                        epf.sigma_for_modular = br.f16()?;
                        if epf.sigma_for_modular < 1e-8 {
                            return Err(invalid("EPF sigma for modular too small"));
                        }
                    }
                }
                read_extensions(br)?;
            }
            read_extensions(br)?;
        }
        if fh.upsampling > 1 || fh.ec_upsampling.iter().any(|&u| u > 1) {
            // Validated when rendering.
        }
        fh.color_channels = if fh.modular
            && !fh.do_ycbcr
            && !ih.xyb_encoded
            && ih.color.color_space == super::headers::CS_GRAY
        {
            1
        } else {
            3
        };
        if fh.width == 0 || fh.height == 0 {
            return Err(invalid("empty frame"));
        }
        if fh.width > 1 << 30 || fh.height > 1 << 30 {
            return Err(invalid("frame dimensions too large"));
        }
        // Group counts must fit the TOC (and u32 arithmetic).
        let (gx, gy) = fh.groups();
        if gx as u64 * gy as u64 * fh.passes.num_passes as u64 > 1 << 22 {
            return Err(invalid("too many groups"));
        }
        Ok(fh)
    }

    pub(crate) fn group_dim(&self) -> u32 {
        128 << self.group_size_shift
    }

    /// Size of the colour sample grid (after upsampling and LF level).
    pub(crate) fn sample_size(&self) -> (u32, u32) {
        let mut w = self.width.div_ceil(self.upsampling);
        let mut h = self.height.div_ceil(self.upsampling);
        if self.lf_level > 0 {
            let s = 3 * self.lf_level;
            w = (w + (1 << s) - 1) >> s;
            h = (h + (1 << s) - 1) >> s;
        }
        (w, h)
    }

    pub(crate) fn groups(&self) -> (u32, u32) {
        let (w, h) = self.sample_size();
        (w.div_ceil(self.group_dim()), h.div_ceil(self.group_dim()))
    }

    pub(crate) fn lf_groups(&self) -> (u32, u32) {
        let (w, h) = self.sample_size();
        let d = self.group_dim() * 8;
        (w.div_ceil(d), h.div_ceil(d))
    }

    pub(crate) fn num_groups(&self) -> u32 {
        let (x, y) = self.groups();
        x * y
    }

    pub(crate) fn num_lf_groups(&self) -> u32 {
        let (x, y) = self.lf_groups();
        x * y
    }
}

/// Byte ranges of the frame sections, in logical order
/// (LfGlobal, LfGroups, HfGlobal, PassGroups) or a single entry.
pub(crate) struct Toc {
    pub(crate) sections: Vec<(usize, usize)>,
    /// Offset (relative to the TOC start reader) where the frame ends.
    pub(crate) end: usize,
}

impl Toc {
    /// Read the TOC; `base` is the byte offset of `br` within the data.
    pub(crate) fn read(br: &mut BitReader, fh: &FrameHeader) -> Result<Self> {
        let num_groups = fh.num_groups() as usize;
        let num_passes = fh.passes.num_passes as usize;
        let entries = if num_groups == 1 && num_passes == 1 {
            1
        } else {
            1 + fh.num_lf_groups() as usize + 1 + num_groups * num_passes
        };
        if entries > 1 << 22 {
            return Err(invalid("too many TOC entries"));
        }
        let permutation = if br.read_bool()? {
            let h = Histograms::decode(br, 8)?;
            let mut reader = SymbolReader::new(&h, br, 0)?;
            let p = read_permutation(br, &h, &mut reader, entries as u32, 0)?;
            reader.finish()?;
            Some(p)
        } else {
            None
        };
        br.zero_pad_to_byte()?;
        let sizes = (0..entries)
            .map(|_| {
                br.u32([
                    bits(10, 0),
                    bits(14, 1024),
                    bits(22, 17408),
                    bits(30, 4_211_712),
                ])
            })
            .collect::<Result<Vec<_>>>()?;
        br.zero_pad_to_byte()?;
        let start = br.byte_pos();
        let mut offsets = Vec::with_capacity(entries);
        let mut acc = start;
        for &s in &sizes {
            offsets.push(acc);
            acc = acc
                .checked_add(s as usize)
                .ok_or_else(|| invalid("TOC sizes overflow"))?;
        }
        let sections = match permutation {
            None => offsets
                .into_iter()
                .zip(sizes)
                .map(|(o, s)| (o, s as usize))
                .collect(),
            Some(p) => p
                .iter()
                .map(|&i| (offsets[i as usize], sizes[i as usize] as usize))
                .collect(),
        };
        Ok(Toc { sections, end: acc })
    }
}
