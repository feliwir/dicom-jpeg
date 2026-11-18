//! The bit-level entropy-coding primitives one by one: the T.81 bit reader
//! and writer (JPEG), JPEG-LS Golomb codes and the JPEG 2000 MQ coder.
//! They do not dispatch on the SIMD level; what changes their speed is the
//! scalar instruction set the crate is compiled for (BMI1/BMI2, LZCNT), so
//! compare builds with different `RUSTFLAGS` (see the README). Throughput
//! is in coded symbols.

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use dicom_jpeg_benchmarks::criterion;
use dicom_jpeg_common::bitio::{BitReader, BitWriter, T81};
use dicom_jpeg_j2k::kernels as j2k;
use dicom_jpeg_jpegls::kernels as jpegls;

/// Symbols per iteration.
const N: usize = 1 << 16;

/// Pseudo-random numbers.
fn rng(seed: u32) -> impl FnMut() -> u32 {
    let mut s = seed | 1;
    move || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s
    }
}

fn bench(c: &mut Criterion, name: &str, mut f: impl FnMut()) {
    let mut g = c.benchmark_group(format!("bits/{name}"));
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function("dicom-jpeg", |b| b.iter(&mut f));
    g.finish();
}

fn t81(c: &mut Criterion) {
    // Codes of 1..=16 bits, short ones more frequent (as Huffman codes and
    // coefficient magnitudes are), including 0xFF bytes to stuff.
    let mut r = rng(1);
    let codes: Vec<(u32, u32)> = (0..N)
        .map(|_| {
            let x = r();
            let n = 1 + (x % 16).min(x >> 4 & 7);
            (x >> 8 & ((1 << n) - 1), n)
        })
        .collect();
    let mut stream = Vec::new();
    bench(c, "T.81 write", || {
        let mut w = BitWriter::<T81>::new(std::mem::take(&mut stream));
        for &(v, n) in &codes {
            w.write(v, n);
        }
        w.flush();
        stream = black_box(w.out);
        stream.clear();
    });
    let mut w = BitWriter::<T81>::new(Vec::new());
    for &(v, n) in &codes {
        w.write(v, n);
    }
    w.flush();
    let data = w.out;
    let mut rd = BitReader::<T81>::new(&data, 0);
    for &(v, n) in &codes {
        assert_eq!(rd.bits(n), v, "T.81 round trip");
    }
    bench(c, "T.81 read", || {
        let mut rd = BitReader::<T81>::new(&data, 0);
        let mut acc = 0i32;
        for &(_, n) in &codes {
            acc = acc.wrapping_add(rd.receive_extend(n));
        }
        black_box(acc);
    });
}

fn golomb(c: &mut Criterion) {
    // Mapped errors of a smooth image: mostly small, k around 2.
    let mut r = rng(2);
    let ks: Vec<u32> = (0..N).map(|_| 1 + r() % 4).collect();
    let values: Vec<i32> = ks
        .iter()
        .map(|&k| {
            let x = r();
            ((x % (8 << k)) >> (x >> 28 & 3)) as i32
        })
        .collect();
    bench(c, "JPEG-LS Golomb write", || {
        black_box(jpegls::golomb_encode(&values, &ks));
    });
    let data = jpegls::golomb_encode(&values, &ks);
    let mut out = vec![0i32; N];
    jpegls::golomb_decode(&data, &ks, &mut out);
    assert_eq!(out, values, "Golomb round trip");
    bench(c, "JPEG-LS Golomb read", || {
        jpegls::golomb_decode(&data, &ks, &mut out);
        black_box(&out);
    });
}

fn mq(c: &mut Criterion) {
    // Skewed decisions in a few contexts, as in the significance passes.
    let mut r = rng(3);
    let decisions: Vec<(u8, u8)> = (0..N)
        .map(|_| {
            let x = r();
            let cx = (x % 9) as u8;
            (cx, (x >> 8 & 0xFF < 32 + 16 * cx as u32) as u8)
        })
        .collect();
    bench(c, "MQ encode", || {
        black_box(j2k::mq_encode(&decisions));
    });
    let data = j2k::mq_encode(&decisions);
    let contexts: Vec<u8> = decisions.iter().map(|&(cx, _)| cx).collect();
    let mut out = vec![0u8; N];
    j2k::mq_decode(&data, &contexts, &mut out);
    let bits: Vec<u8> = decisions.iter().map(|&(_, d)| d).collect();
    assert_eq!(out, bits, "MQ round trip");
    bench(c, "MQ decode", || {
        j2k::mq_decode(&data, &contexts, &mut out);
        black_box(&out);
    });
}

criterion_group! {
    name = benches;
    config = criterion();
    targets = t81, golomb, mq
}
criterion_main!(benches);
