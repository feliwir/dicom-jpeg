//! JPEG 2000 encoder sweeps on synthetic images.

use dicom_jpeg::j2k::{DecoderOptions, EncoderOptions, decode_components, encode_with};
use dicom_jpeg::{ColorSpace, Image, ImageInfo, Samples};

fn synthetic(w: u32, h: u32, nc: u16, bits: u8, signed: bool, seed: u32) -> Image {
    let max = (1u32 << bits) - 1;
    let mut s = seed;
    let mut v = Vec::new();
    for y in 0..h {
        for x in 0..w {
            for c in 0..nc as u32 {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                let smooth = (x * 3 + y * 5 + c * 17) % (max + 1);
                let edge = if (x + y) % 23 < 4 { max / 2 } else { 0 };
                v.push(((smooth + edge + (s >> 28)) % (max + 1)) as u16);
            }
        }
    }
    let info = ImageInfo {
        width: w,
        height: h,
        components: nc,
        bits_per_sample: bits,
        signed,
        color_space: if nc == 3 {
            ColorSpace::Rgb
        } else {
            ColorSpace::Unknown
        },
    };
    let samples = if bits <= 8 {
        Samples::U8(v.into_iter().map(|x| x as u8).collect())
    } else {
        Samples::U16(v)
    };
    Image::new(info, samples).unwrap()
}

/// Expected sample values as the decoder reports them (sign-extended if signed).
fn expected(img: &Image) -> Vec<Vec<i32>> {
    let nc = img.info.components as usize;
    let bits = img.info.bits_per_sample as u32;
    let vals: Vec<i32> = match &img.samples {
        Samples::U8(v) => v.iter().map(|&x| x as i32).collect(),
        Samples::U16(v) => v.iter().map(|&x| x as i32).collect(),
    };
    (0..nc)
        .map(|c| {
            vals.iter()
                .skip(c)
                .step_by(nc)
                .map(|&x| {
                    if img.info.signed && x >= 1 << (bits - 1) {
                        x - (1 << bits)
                    } else {
                        x
                    }
                })
                .collect()
        })
        .collect()
}

#[test]
fn lossless_sweep() {
    lossless_sweep_with(false);
}

#[test]
#[cfg(feature = "htj2k")]
fn ht_lossless_sweep() {
    lossless_sweep_with(true);
}

fn lossless_sweep_with(ht: bool) {
    for &(w, h) in &[(1u32, 1u32), (2, 3), (13, 7), (65, 33), (130, 70)] {
        for (nc, bits, signed) in [
            (1u16, 1u8, false),
            (1, 8, false),
            (1, 12, true),
            (1, 16, false),
            (3, 8, false),
            (3, 10, false),
            (4, 8, false),
        ] {
            let img = synthetic(w, h, nc, bits, signed, w * 7 + h);
            for levels in [0u8, 1, 3, 5] {
                for progression in 0..=4u8 {
                    let mut opts = EncoderOptions::default();
                    opts.levels = Some(levels);
                    opts.progression = progression;
                    opts.high_throughput = ht;
                    opts.tlm = ht;
                    let enc = encode_with(&img, &opts)
                        .unwrap_or_else(|e| panic!("{w}x{h} x{nc} {bits}b: {e}"));
                    assert!(!enc.lossy);
                    let comps = decode_components(&enc.data, &DecoderOptions::default())
                        .unwrap_or_else(|e| {
                            panic!("{w}x{h} x{nc} {bits}b L{levels} P{progression} ht={ht}: {e}")
                        });
                    let got: Vec<Vec<i32>> = comps.into_iter().map(|c| c.data).collect();
                    assert_eq!(
                        got,
                        expected(&img),
                        "{w}x{h} x{nc} {bits}b signed={signed} L{levels} P{progression} ht={ht}"
                    );
                }
            }
        }
    }
}

#[test]
fn lossy_quality_is_monotonic_and_on_target() {
    let img = synthetic(200, 150, 3, 8, false, 3);
    let reference = expected(&img);
    let mut last_size = 0;
    for quality in [10u8, 50, 80, 95] {
        let mut opts = EncoderOptions::default();
        opts.reversible = false;
        opts.quality = quality;
        let enc = encode_with(&img, &opts).unwrap();
        assert!(enc.lossy);
        assert!(enc.data.len() > last_size, "size should grow with quality");
        last_size = enc.data.len();
        let comps = decode_components(&enc.data, &DecoderOptions::default()).unwrap();
        let mse = comps
            .iter()
            .zip(&reference)
            .flat_map(|(c, r)| c.data.iter().zip(r).map(|(a, b)| ((a - b) as f64).powi(2)))
            .sum::<f64>()
            / (200.0 * 150.0 * 3.0);
        let psnr = 10.0 * (255f64.powi(2) / mse).log10();
        let target = 26.0 + 0.23 * quality as f64;
        assert!(
            (psnr - target).abs() < 2.5,
            "quality {quality}: PSNR {psnr:.2} vs target {target:.2}"
        );
    }
}

/// The HT block coder is lossless with respect to the quantized coefficients,
/// so an HT lossy stream must decode to exactly the same samples as the
/// EBCOT stream with the same quantization.
#[test]
#[cfg(feature = "htj2k")]
fn ht_lossy_matches_part1() {
    for (w, h, nc, bits) in [
        (200u32, 150u32, 3u16, 8u8),
        (97, 61, 1, 12),
        (64, 64, 1, 16),
    ] {
        let img = synthetic(w, h, nc, bits, false, w + h);
        for quality in [20u8, 70, 95] {
            let mut opts = EncoderOptions::default();
            opts.reversible = false;
            opts.quality = quality;
            let part1 = encode_with(&img, &opts).unwrap();
            opts.high_throughput = true;
            let ht = encode_with(&img, &opts).unwrap();
            assert!(ht.lossy);
            let a = decode_components(&part1.data, &DecoderOptions::default()).unwrap();
            let b = decode_components(&ht.data, &DecoderOptions::default()).unwrap();
            for (x, y) in a.iter().zip(&b) {
                assert_eq!(x.data, y.data, "{w}x{h} x{nc} {bits}b q{quality}");
            }
        }
    }
}

/// RPCL + TLM streams split into one tile-part per resolution decode at
/// every reduced resolution.
#[test]
#[cfg(feature = "htj2k")]
fn ht_rpcl_tile_parts_reduce() {
    let img = synthetic(130, 70, 1, 12, false, 9);
    let mut opts = EncoderOptions::default();
    opts.high_throughput = true;
    opts.progression = 2;
    opts.tlm = true;
    opts.levels = Some(3);
    let enc = encode_with(&img, &opts).unwrap();
    // SOT markers: one per resolution.
    let sots = enc.data.windows(2).filter(|m| m == &[0xFF, 0x90]).count();
    assert!(
        sots >= 4,
        "expected 4 tile-parts, found {sots} SOT candidates"
    );
    assert!(
        enc.data.windows(2).any(|m| m == [0xFF, 0x55]),
        "TLM missing"
    );
    for reduce in 0..=3 {
        let mut dopts = DecoderOptions::default();
        dopts.reduce = reduce;
        let comps = decode_components(&enc.data, &dopts).unwrap();
        assert_eq!(comps[0].width, 130u32.div_ceil(1 << reduce));
    }
}

/// Part 2 wavelet-based multiple component transform: lossless round trips
/// over component counts, interleave factors, levels and sample formats.
#[test]
#[cfg(feature = "jpeg2000-part2")]
fn part2_lossless_sweep() {
    use dicom_jpeg::j2k::ComponentWavelet;
    for (nc, interleave) in [
        (1u16, 1u16),
        (2, 1),
        (5, 1),
        (9, 3),
        (16, 1),
        (30, 3),
        (300, 1),
    ] {
        for (bits, signed) in [
            (1u8, false),
            (8, false),
            (12, true),
            (16, false),
            (16, true),
        ] {
            let (w, h) = if nc > 100 { (9, 5) } else { (37, 21) };
            let img = synthetic(w, h, nc, bits, signed, nc as u32 * 31 + bits as u32);
            let max_levels = 31 - (nc as u32).leading_zeros();
            for cl in [None, Some(0u8), Some(max_levels as u8)] {
                let mut opts = EncoderOptions::default();
                opts.levels = Some(2);
                opts.component_wavelet = Some(ComponentWavelet::new(cl, interleave));
                let enc = encode_with(&img, &opts)
                    .unwrap_or_else(|e| panic!("x{nc}/{interleave} {bits}b {cl:?}: {e}"));
                assert!(!enc.lossy);
                let comps = decode_components(&enc.data, &DecoderOptions::default())
                    .unwrap_or_else(|e| panic!("x{nc}/{interleave} {bits}b {cl:?}: {e}"));
                assert!(
                    comps
                        .iter()
                        .all(|c| c.precision == bits && c.signed == signed)
                );
                let got: Vec<Vec<i32>> = comps.into_iter().map(|c| c.data).collect();
                assert!(
                    got == expected(&img),
                    "x{nc}/{interleave} {bits}b signed={signed} {cl:?}"
                );
            }
        }
    }
}

/// Lossy Part 2 (9/7 along the components) meets the quality target and
/// compresses correlated components better than coding them independently.
#[test]
#[cfg(feature = "jpeg2000-part2")]
fn part2_lossy_quality() {
    use dicom_jpeg::j2k::ComponentWavelet;
    let (w, h, nc) = (96u32, 80u32, 12u16);
    // A slowly changing "volume": every component is a shifted version of a
    // textured slice.
    let mut v = Vec::new();
    for y in 0..h {
        for x in 0..w {
            for c in 0..nc as u32 {
                let t = ((x * 7 + y * 3 + c) % 64) as f64;
                let val =
                    2000.0 + 900.0 * (t / 10.0).sin() + 40.0 * c as f64 + ((x ^ y) & 15) as f64;
                v.push(val as u16);
            }
        }
    }
    let info = ImageInfo {
        width: w,
        height: h,
        components: nc,
        bits_per_sample: 12,
        signed: false,
        color_space: ColorSpace::Unknown,
    };
    let img = Image::new(info, Samples::U16(v)).unwrap();
    let reference = expected(&img);
    let encode = |part2: bool| {
        let mut opts = EncoderOptions::default();
        opts.reversible = false;
        opts.quality = 80;
        opts.component_wavelet = part2.then(|| ComponentWavelet::new(None, 1));
        encode_with(&img, &opts).unwrap()
    };
    let (p1, p2) = (encode(false), encode(true));
    assert!(p2.lossy);
    let comps = decode_components(&p2.data, &DecoderOptions::default()).unwrap();
    let mse = comps
        .iter()
        .zip(&reference)
        .flat_map(|(c, r)| c.data.iter().zip(r).map(|(a, b)| ((a - b) as f64).powi(2)))
        .sum::<f64>()
        / (w * h * nc as u32) as f64;
    let (lo, hi) = reference
        .iter()
        .flatten()
        .fold((i32::MAX, i32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    let psnr = 10.0 * (((hi - lo) as f64).powi(2) / mse).log10();
    let target = 26.0 + 0.23 * 80.0;
    assert!(
        (psnr - target).abs() < 2.5,
        "PSNR {psnr:.2} vs target {target:.2}"
    );
    assert!(
        p2.data.len() < p1.data.len(),
        "part 2 {} bytes vs part 1 {} bytes",
        p2.data.len(),
        p1.data.len()
    );
}
