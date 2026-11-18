//! The SIMD kernels one by one, at every SIMD level the CPU supports, on
//! synthetic data of 64 K samples per iteration (small enough to stay in
//! the L2 cache, so the numbers reflect the kernels rather than memory).
//! Throughput is in samples.

use std::hint::black_box;

use criterion::{Criterion, SamplingMode, Throughput, criterion_group, criterion_main};
use dicom_jpeg_benchmarks::criterion;
use dicom_jpeg_common::interleave;
use dicom_jpeg_common::simd::Level;
use dicom_jpeg_j2k::kernels as j2k;
use dicom_jpeg_jpeg::kernels;
use dicom_jpeg_jxl::kernels as jxl;

/// Row width and number of rows of the test planes.
const W: usize = 1024;
const H: usize = 64;
const N: usize = W * H;

/// Pseudo-random 8-bit samples.
fn samples(n: usize, seed: u32) -> Vec<u16> {
    let mut s = seed | 1;
    (0..n)
        .map(|i| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            // Smooth-ish image content plus noise.
            (((i % W) / 4 + (i / W) * 2 + (s % 32) as usize) % 256) as u16
        })
        .collect()
}

/// Benchmark `f` (one iteration over `elements` samples) at every level.
fn group(c: &mut Criterion, name: &str, elements: usize, mut f: impl FnMut(Level)) {
    let mut g = c.benchmark_group(format!("kernel/{name}"));
    g.sampling_mode(SamplingMode::Flat);
    g.throughput(Throughput::Elements(elements as u64));
    for level in Level::supported() {
        g.bench_function(level.name(), |b| b.iter(|| f(black_box(level))));
    }
    g.finish();
}

fn jpeg_kernels(c: &mut Criterion) {
    // IDCT: dequantised blocks as from a q90 image (DC plus a few low
    // frequencies, most AC zero).
    let blocks = N / 64;
    let mut s = 7u32;
    let coefs: Vec<[i32; 64]> = (0..blocks)
        .map(|_| {
            let mut b = [0i32; 64];
            for (k, v) in b.iter_mut().enumerate() {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                let (u, v_) = (k % 8, k / 8);
                if k == 0 {
                    *v = (s % 2048) as i32 - 1024;
                } else if u + v_ < 4 && !s.is_multiple_of(3) {
                    *v = (s % 200) as i32 - 100;
                }
            }
            b
        })
        .collect();
    let mut out = vec![[0u16; 64]; blocks];
    group(c, "jpeg/idct", N, |l| {
        kernels::idct_blocks(l, &coefs, 8, &mut out);
        black_box(&out);
    });
    // 12-bit: the same blocks scaled by 16 (64-bit arithmetic).
    let coefs12: Vec<[i32; 64]> = coefs.iter().map(|b| b.map(|v| v * 16)).collect();
    group(c, "jpeg/idct 12-bit", N, |l| {
        kernels::idct_blocks(l, &coefs12, 12, &mut out);
        black_box(&out);
    });

    // FDCT + quantisation of a W x H plane, a block row at a time.
    let plane = samples(N, 1);
    let q: [u16; 64] = std::array::from_fn(|k| 2 + (k as u16 % 8) + (k as u16 / 8));
    let table = kernels::QuantTable::new(&q, 1023);
    let mut row = vec![[0i32; 64]; W / 8];
    group(c, "jpeg/fdct+quantize", N, |l| {
        for by in 0..H / 8 {
            kernels::fdct_quantize(l, &plane[by * 8 * W..], W, 8, &table, &mut row);
        }
        black_box(&row);
    });
    let plane12: Vec<u16> = plane.iter().map(|&v| v * 16 + (v >> 4)).collect();
    let table12 = kernels::QuantTable::new(&q, 16383);
    group(c, "jpeg/fdct+quantize 12-bit", N, |l| {
        for by in 0..H / 8 {
            kernels::fdct_quantize(l, &plane12[by * 8 * W..], W, 12, &table12, &mut row);
        }
        black_box(&row);
    });

    // Upsampling: W/2-wide chroma rows to W-wide rows (N output samples).
    let chroma = samples(N / 2, 2);
    let mut up = vec![0u16; W];
    group(c, "jpeg/upsample h2v1", N, |l| {
        for r in chroma.as_chunks::<{ W / 2 }>().0.iter() {
            kernels::upsample_h2v1(l, 8, r, &mut up);
        }
        black_box(&up);
    });
    group(c, "jpeg/upsample h2v2", N, |l| {
        for (i, r) in chroma.as_chunks::<{ W / 2 }>().0.iter().enumerate() {
            let far = &chroma[(i ^ 1).min(H - 1) * W / 2..][..W / 2];
            kernels::upsample_h2v2(l, 8, r, far, &mut up);
        }
        black_box(&up);
    });

    // Downsampling: W-wide rows to W/2 (N input samples).
    let mut down = vec![0u16; W / 2];
    group(c, "jpeg/downsample h2v1", N, |l| {
        for r in plane.as_chunks::<W>().0.iter() {
            kernels::downsample_h2v1(l, 8, r, &mut down);
        }
        black_box(&down);
    });
    group(c, "jpeg/downsample h2v2", N, |l| {
        for r in plane.as_chunks::<{ 2 * W }>().0.iter() {
            let (a, b) = r.split_at(W);
            kernels::downsample_h2v2(l, 8, a, b, &mut down);
        }
        black_box(&down);
    });

    // Colour conversion of W-wide rows (N pixels).
    let ycc = [samples(N, 3), samples(N, 4), samples(N, 5)];
    let mut rgb = [vec![0u16; W], vec![0u16; W], vec![0u16; W]];
    group(c, "jpeg/ycbcr->rgb", N, |l| {
        for i in 0..H {
            let src = [
                &ycc[0][i * W..][..W],
                &ycc[1][i * W..][..W],
                &ycc[2][i * W..][..W],
            ];
            let [r, g, b] = &mut rgb;
            kernels::ycbcr_to_rgb(l, src, 8, [r, g, b]);
        }
        black_box(&rgb);
    });
    let mut work = ycc.clone();
    group(c, "jpeg/rgb->ycbcr", N, |l| {
        let [a, b, c] = &mut work;
        for ((a, b), c) in a
            .as_chunks_mut::<W>()
            .0
            .iter_mut()
            .zip(b.as_chunks_mut::<W>().0.iter_mut())
            .zip(c.as_chunks_mut::<W>().0.iter_mut())
        {
            kernels::rgb_to_ycbcr(l, 8, [a, b, c]);
        }
        black_box(&work);
    });
}

fn j2k_kernels(c: &mut Criterion) {
    // One lifting step over the N / 2 samples of one kind, as in a row or
    // column pass (throughput in updated samples).
    let h = N / 2;
    let ints = |seed| -> Vec<i32> { samples(h, seed).iter().map(|&v| v as i32 - 128).collect() };
    let (a, b) = (ints(9), ints(10));
    let mut cur = ints(11);
    group(c, "j2k/dwt update53", h, |l| {
        j2k::update53(l, &mut cur, &a, &b);
        black_box(&cur);
    });
    group(c, "j2k/dwt predict53", h, |l| {
        j2k::predict53(l, &mut cur, &a, &b);
        black_box(&cur);
    });
    group(c, "j2k/dwt apredict53", h, |l| {
        j2k::apredict53(l, &mut cur, &a, &b);
        black_box(&cur);
    });
    group(c, "j2k/dwt aupdate53", h, |l| {
        j2k::aupdate53(l, &mut cur, &a, &b);
        black_box(&cur);
    });
    let floats = |v: &[i32]| -> Vec<f32> { v.iter().map(|&x| x as f32).collect() };
    let (fa, fb) = (floats(&a), floats(&b));
    let mut fcur = floats(&cur);
    group(c, "j2k/dwt lift97", h, |l| {
        // A small k keeps the values bounded over the iterations.
        j2k::lift97(l, &mut fcur, &fa, &fb, 0.0625);
        black_box(&fcur);
    });
    group(c, "j2k/dwt scale97", h, |l| {
        j2k::scale97(l, &mut fcur, 1.0);
        black_box(&fcur);
    });
}

fn jxl_kernels(c: &mut Criterion) {
    // Inverse VarDCT of N coefficients as 8 x 8 and as 32 x 32 blocks,
    // with magnitudes falling off with frequency.
    let coefs: Vec<f32> = samples(N, 12)
        .iter()
        .enumerate()
        .map(|(i, &v)| (v as f32 - 128.0) / (1 + i % 64) as f32)
        .collect();
    for side in [8, 32] {
        let mut work = coefs.clone();
        let mut levels: Vec<_> = Level::supported().map(|l| (l, jxl::Idct::new(l))).collect();
        group(c, &format!("jxl/idct {side}x{side}"), N, |l| {
            let idct = &mut levels.iter_mut().find(|(k, _)| *k == l).expect("level").1;
            work.copy_from_slice(&coefs);
            for b in work.chunks_exact_mut(side * side) {
                idct.inverse(b, side, side);
            }
            black_box(&work);
        });
    }
}

fn common_kernels(c: &mut Criterion) {
    let p = [samples(N, 6), samples(N, 7), samples(N, 8)];
    let mut px8 = vec![0u8; 3 * N];
    let mut px16 = vec![0u16; 3 * N];
    group(c, "common/interleave3 u8", N, |l| {
        interleave::interleave3_u8(l, &p[0], &p[1], &p[2], &mut px8);
        black_box(&px8);
    });
    group(c, "common/interleave3 u16", N, |l| {
        interleave::interleave3_u16(l, &p[0], &p[1], &p[2], &mut px16);
        black_box(&px16);
    });
    let mut q = [vec![0u16; N], vec![0u16; N], vec![0u16; N]];
    group(c, "common/deinterleave3 u8", N, |l| {
        let [a, b, c] = &mut q;
        interleave::deinterleave3_u8(l, &px8, a, b, c);
        black_box(&q);
    });
    group(c, "common/deinterleave3 u16", N, |l| {
        let [a, b, c] = &mut q;
        interleave::deinterleave3_u16(l, &px16, a, b, c);
        black_box(&q);
    });
}

criterion_group! {
    name = benches;
    config = criterion();
    targets = jpeg_kernels, j2k_kernels, jxl_kernels, common_kernels
}
criterion_main!(benches);
