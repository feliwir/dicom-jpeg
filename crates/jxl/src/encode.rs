//! Lossless JPEG XL encoder: modular mode with the clamped gradient
//! predictor, a fixed MA tree over local gradients, the YCoCg RCT for RGB
//! input, 256x256 groups and prefix-coded (Huffman) entropy coding.

use super::CODEC;
use dicom_jpeg_common::huffman::{canonical_codes, code_lengths as huffman_lengths};
use dicom_jpeg_common::{AppliedTransform, ColorSpace, Encoded, Image};
use dicom_jpeg_common::{Error, Result};

// ---------------------------------------------------------------------------
// Bit writer

#[derive(Default)]
pub(crate) struct BitWriter {
    out: Vec<u8>,
    acc: u64,
    n: u32,
}

impl BitWriter {
    pub(crate) fn write(&mut self, n: u32, v: u64) {
        debug_assert!(n <= 56 && (n == 64 || v >> n == 0));
        if n == 0 {
            return;
        }
        self.acc |= v << self.n;
        self.n += n;
        while self.n >= 8 {
            self.out.push(self.acc as u8);
            self.acc >>= 8;
            self.n -= 8;
        }
    }

    fn bool(&mut self, b: bool) {
        self.write(1, b as u64);
    }

    pub(crate) fn zero_pad(&mut self) {
        if self.n > 0 {
            self.write(8 - self.n, 0);
        }
    }

    pub(crate) fn finish(mut self) -> Vec<u8> {
        self.zero_pad();
        self.out
    }

    fn bits_written(&self) -> u64 {
        self.out.len() as u64 * 8 + self.n as u64
    }

    /// `U32` with distribution `[(offset, bits); 4]`: picks the first fitting selector.
    fn u32(&mut self, d: [(u32, u32); 4], v: u32) {
        for (sel, &(off, bits)) in d.iter().enumerate() {
            if v >= off && ((v - off) as u64) < (1u64 << bits) {
                self.write(2, sel as u64);
                self.write(bits, (v - off) as u64);
                return;
            }
        }
        unreachable!("value {v} not representable");
    }

    /// `U64` (only small values are needed).
    fn u64(&mut self, v: u64) {
        match v {
            0 => self.write(2, 0),
            1..=16 => {
                self.write(2, 1);
                self.write(4, v - 1);
            }
            17..=272 => {
                self.write(2, 2);
                self.write(8, v - 17);
            }
            _ => unreachable!("large U64 values are not written"),
        }
    }

    /// `Enum()` for values < 18.
    fn enumeration(&mut self, v: u32) {
        self.u32([(0, 0), (1, 0), (2, 4), (18, 6)], v);
    }
}

fn pack_signed(v: i32) -> u32 {
    ((v << 1) ^ (v >> 31)) as u32
}

fn ceil_log2(x: u32) -> u32 {
    x.next_power_of_two().trailing_zeros()
}

// ---------------------------------------------------------------------------
// Prefix codes

/// A prefix code ready for writing symbols.
struct PrefixEnc {
    lengths: Vec<u8>,
    /// Canonical codes, bit-reversed: the stream is MSB first per code and
    /// the writer LSB first.
    rev_codes: Vec<u32>,
    /// Single used symbol: coded with zero bits.
    single: bool,
}

impl PrefixEnc {
    fn new(lengths: Vec<u8>, single: bool) -> Self {
        let rev_codes = if single {
            vec![0; lengths.len()]
        } else {
            canonical_codes(&lengths)
                .iter()
                .zip(&lengths)
                .map(|(&c, &l)| {
                    if l == 0 {
                        0
                    } else {
                        c.reverse_bits() >> (32 - l as u32)
                    }
                })
                .collect()
        };
        PrefixEnc {
            lengths,
            rev_codes,
            single,
        }
    }

    #[inline]
    fn write_symbol(&self, w: &mut BitWriter, s: u32) {
        if self.single {
            return;
        }
        let len = self.lengths[s as usize] as u32;
        debug_assert!(len > 0, "symbol without a code");
        w.write(len, self.rev_codes[s as usize] as u64);
    }
}

/// Write a prefix code over `freq.len()` symbols (RFC 7932 format) and
/// return the encoder. `freq.len()` must be >= 2.
fn write_prefix_code(w: &mut BitWriter, freq: &[u64]) -> PrefixEnc {
    let alphabet = freq.len() as u32;
    let used: Vec<usize> = (0..freq.len()).filter(|&i| freq[i] > 0).collect();
    let nbits = ceil_log2(alphabet);
    if used.len() <= 4 {
        // Simple prefix code.
        let mut syms = used.clone();
        if syms.is_empty() {
            syms.push(0);
        }
        // Most frequent first (it gets the shortest code for nsym 3).
        syms.sort_by_key(|&s| std::cmp::Reverse(freq[s]));
        w.write(2, 1);
        w.write(2, syms.len() as u64 - 1);
        let mut lengths = vec![0u8; freq.len()];
        let lens: Vec<u8> = match syms.len() {
            1 => vec![0],
            2 => vec![1, 1],
            3 => vec![1, 2, 2],
            _ => {
                let f: Vec<u64> = syms.iter().map(|&s| freq[s]).collect();
                // Tree-select 1 (lengths 1,2,3,3) when it is cheaper.
                let skew = f[0] + 2 * f[1] + 3 * (f[2] + f[3]);
                let flat = 2 * (f[0] + f[1] + f[2] + f[3]);
                if skew < flat {
                    vec![1, 2, 3, 3]
                } else {
                    vec![2, 2, 2, 2]
                }
            }
        };
        for &s in &syms {
            w.write(nbits, s as u64);
        }
        if syms.len() == 4 {
            w.bool(lens[0] == 1);
        }
        if syms.len() == 1 {
            return PrefixEnc::new(lengths, true);
        }
        for (k, &s) in syms.iter().enumerate() {
            lengths[s] = lens[k];
        }
        return PrefixEnc::new(lengths, false);
    }

    let lengths = huffman_lengths(freq, 15);
    // Number of symbol lengths to send: up to the last non-zero length.
    let last = lengths.iter().rposition(|&l| l > 0).expect("used symbols") + 1;
    let mut cl_freq = [0u64; 18];
    for &l in &lengths[..last] {
        cl_freq[l as usize] += 1;
    }
    let mut cl_lengths = huffman_lengths(&cl_freq, 5);
    let cl_used = cl_freq.iter().filter(|&&f| f > 0).count();
    if cl_used == 1 {
        // A single code length symbol: any non-zero length, read with 0 bits.
        for l in cl_lengths.iter_mut() {
            if *l > 0 {
                *l = 1;
            }
        }
    }
    const ORDER: [usize; 18] = [1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    // Fixed code for code length code lengths 0..=5: (value, bits), LSB first.
    const FIXED: [(u64, u32); 6] = [(0, 2), (7, 4), (3, 3), (2, 2), (1, 2), (15, 4)];
    w.write(2, 0); // hskip = 0
    let mut space = 32i32;
    let mut written = 0;
    for &o in &ORDER {
        let l = cl_lengths[o] as usize;
        let (v, b) = FIXED[l];
        w.write(b, v);
        written += 1;
        if l != 0 {
            space -= 32 >> l;
            if cl_used > 1 && space <= 0 {
                break;
            }
        }
    }
    let _ = written;
    let cl = PrefixEnc::new(cl_lengths, cl_used == 1);
    for &l in &lengths[..last] {
        cl.write_symbol(w, l as u32);
    }
    PrefixEnc::new(lengths, false)
}

// ---------------------------------------------------------------------------
// Entropy-coded streams

/// Hybrid uint configuration 4/2/0 for all clusters.
const SPLIT_EXP: u32 = 4;
const MSB: u32 = 2;

fn tokenize(v: u32) -> (u32, u32, u64) {
    if v < 1 << SPLIT_EXP {
        return (v, 0, 0);
    }
    let n = 31 - v.leading_zeros();
    let m = v - (1 << n);
    let token = (1 << SPLIT_EXP) + ((n - SPLIT_EXP) << MSB) + (m >> (n - MSB));
    let nbits = n - MSB;
    (token, nbits, (m & ((1 << nbits) - 1)) as u64)
}

/// Entropy code for a set of contexts, built from collected symbols.
pub(crate) struct EntropyCode {
    context_map: Vec<u8>,
    codes: Vec<PrefixEnc>,
}

impl EntropyCode {
    /// Build and write the histograms for `num_contexts` contexts from the
    /// (context, value) symbols of all streams that will use the code.
    pub(crate) fn write(
        w: &mut BitWriter,
        num_contexts: usize,
        symbols: impl IntoIterator<Item = (u32, u32)>,
    ) -> Self {
        // Token histograms per context.
        let mut hist: Vec<Vec<u64>> = vec![Vec::new(); num_contexts];
        for (ctx, v) in symbols {
            let (t, _, _) = tokenize(v);
            let h = &mut hist[ctx as usize];
            if h.len() <= t as usize {
                h.resize(t as usize + 1, 0);
            }
            h[t as usize] += 1;
        }
        // Clustering: identical histograms share a cluster (no merging).
        let mut clusters: Vec<Vec<u64>> = Vec::new();
        let mut context_map = Vec::with_capacity(num_contexts);
        for h in &hist {
            let key: Vec<u64> = {
                let mut k = h.clone();
                while k.last() == Some(&0) {
                    k.pop();
                }
                k
            };
            let idx = match clusters.iter().position(|c| *c == key) {
                Some(i) => i,
                None => {
                    clusters.push(key);
                    clusters.len() - 1
                }
            };
            context_map.push(idx as u8);
        }
        if clusters.len() > 256 {
            // Merge the tail into the last cluster (rare).
            for m in &mut context_map {
                if *m as usize >= 255 {
                    *m = 255;
                }
            }
            let tail: Vec<Vec<u64>> = clusters.drain(255..).collect();
            let mut merged = Vec::new();
            for c in tail {
                if merged.len() < c.len() {
                    merged.resize(c.len(), 0);
                }
                for (m, x) in merged.iter_mut().zip(c) {
                    *m += x;
                }
            }
            clusters.push(merged);
        }
        w.bool(false); // no LZ77
        write_context_map(w, &context_map, clusters.len());
        w.bool(true); // prefix codes
        for _ in &clusters {
            // HybridUint(log_alpha_size 15): split 4, msb 2, lsb 0.
            w.write(ceil_log2(16), SPLIT_EXP as u64);
            w.write(ceil_log2(SPLIT_EXP + 1), MSB as u64);
            w.write(ceil_log2(SPLIT_EXP - MSB + 1), 0);
        }
        for c in &clusters {
            let size = c.len().max(1) as u32;
            if size == 1 {
                w.bool(false);
            } else {
                w.bool(true);
                let n = 31 - (size - 1).leading_zeros();
                w.write(4, n as u64);
                w.write(n, ((size - 1) - (1 << n)) as u64);
            }
        }
        let codes = clusters
            .iter()
            .map(|c| {
                if c.len() <= 1 {
                    // Alphabet of one symbol: nothing is written.
                    PrefixEnc::new(vec![0], true)
                } else {
                    write_prefix_code(w, c)
                }
            })
            .collect();
        EntropyCode { context_map, codes }
    }

    pub(crate) fn write_symbol(&self, w: &mut BitWriter, ctx: u32, v: u32) {
        let (t, nbits, extra) = tokenize(v);
        self.codes[self.context_map[ctx as usize] as usize].write_symbol(w, t);
        w.write(nbits, extra);
    }
}

fn write_context_map(w: &mut BitWriter, map: &[u8], num_clusters: usize) {
    if map.len() == 1 {
        return;
    }
    let nbits = ceil_log2(num_clusters as u32);
    if nbits <= 3 {
        w.bool(true);
        w.write(2, nbits as u64);
        for &m in map {
            w.write(nbits, m as u64);
        }
    } else {
        w.bool(false);
        w.bool(false); // no MTF
        let symbols: Vec<(u32, u32)> = map.iter().map(|&m| (0, m as u32)).collect();
        let code = EntropyCode::write(w, 1, symbols.iter().copied());
        for &(c, v) in &symbols {
            code.write_symbol(w, c, v);
        }
    }
}

// ---------------------------------------------------------------------------
// Headers

const SIZE: [(u32, u32); 4] = [(1, 9), (1, 13), (1, 18), (1, 30)];

fn write_size(w: &mut BitWriter, width: u32, height: u32) {
    let small =
        width.is_multiple_of(8) && height.is_multiple_of(8) && width <= 256 && height <= 256;
    w.bool(small);
    if small {
        w.write(5, (height / 8 - 1) as u64);
    } else {
        w.u32(SIZE, height);
    }
    w.write(3, 0); // explicit width
    if small {
        w.write(5, (width / 8 - 1) as u64);
    } else {
        w.u32(SIZE, width);
    }
}

fn write_image_header(w: &mut BitWriter, width: u32, height: u32, bits: u32, gray: bool) {
    w.write(16, 0x0AFF);
    write_size(w, width, height);
    w.bool(false); // ImageMetadata.all_default
    w.bool(false); // extra_fields
    w.bool(false); // integer samples
    w.u32([(8, 0), (10, 0), (12, 0), (1, 6)], bits);
    w.bool(bits <= 12); // modular_16bit_buffers
    w.u32([(0, 0), (1, 0), (2, 4), (1, 12)], 0); // no extra channels
    w.bool(false); // xyb_encoded
    if gray {
        w.bool(false); // ColourEncoding.all_default
        w.bool(false); // want_icc
        w.enumeration(1); // grey
        w.enumeration(1); // D65
        w.bool(false); // have_gamma
        w.enumeration(13); // sRGB transfer
        w.enumeration(1); // relative intent
    } else {
        w.bool(true); // sRGB
    }
    w.u64(0); // extensions
    w.bool(true); // CustomTransformData.all_default
    w.zero_pad();
}

fn write_frame_header(w: &mut BitWriter) {
    w.bool(false); // all_default
    w.write(2, 0); // regular frame
    w.bool(true); // modular
    w.u64(0); // flags
    w.bool(false); // do_YCbCr
    w.write(2, 0); // upsampling 1
    w.write(2, 1); // group_size_shift: 256
    w.write(2, 0); // one pass
    w.bool(false); // no crop
    w.write(2, 0); // blend mode replace
    w.bool(true); // is_last
    w.write(2, 0); // empty name
    w.bool(false); // RestorationFilter.all_default
    w.bool(false); // no Gaborish
    w.write(2, 0); // no EPF
    w.u64(0); // restoration filter extensions
    w.u64(0); // frame extensions
}

// ---------------------------------------------------------------------------
// Modular encoding

/// Leaf contexts of the fixed tree: channel x quantised (W - NW) x (N - NW).
const BUCKETS: usize = 5;

fn thresholds(bits: u32) -> [i32; 4] {
    let s = 1i32 << bits.saturating_sub(8).min(8);
    [15 * s, 3 * s, -4 * s, -16 * s]
}

/// Bucket index of `v` for thresholds `t` (descending): matches the tree
/// walk `v > t ? left : right`.
fn bucket(v: i64, t: &[i32; 4]) -> usize {
    t.iter().position(|&th| v > th as i64).unwrap_or(4)
}

/// One tree node in symbolic form.
enum TreeNode {
    Split {
        prop: u32,
        value: i32,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
    Leaf(u32),
}

/// Tree on a property: a right-leaning chain over the thresholds whose
/// leaves are produced by `leaf(bucket)`.
fn chain(prop: u32, t: &[i32; 4], leaf: &mut dyn FnMut(usize) -> TreeNode) -> TreeNode {
    fn go(prop: u32, t: &[i32; 4], i: usize, leaf: &mut dyn FnMut(usize) -> TreeNode) -> TreeNode {
        if i == 4 {
            return leaf(4);
        }
        let left = Box::new(leaf(i));
        let right = Box::new(go(prop, t, i + 1, leaf));
        TreeNode::Split {
            prop,
            value: t[i],
            left,
            right,
        }
    }
    go(prop, t, 0, leaf)
}

/// Build the fixed tree; leaf payload = logical context (channel, b1, b2).
fn build_tree(num_channels: usize, t: &[i32; 4]) -> TreeNode {
    fn per_channel(c: usize, n: usize, t: &[i32; 4]) -> TreeNode {
        let leaf = |b1: usize| {
            chain(11, t, &mut |b2| {
                TreeNode::Leaf(((c * BUCKETS + b1) * BUCKETS + b2) as u32)
            })
        };
        if c + 1 == n {
            return chain(10, t, &mut |b1| leaf(b1));
        }
        // property 0 (channel) > c ? rest : this channel
        TreeNode::Split {
            prop: 0,
            value: c as i32,
            left: Box::new(per_channel(c + 1, n, t)),
            right: Box::new(chain(10, t, &mut |b1| leaf(b1))),
        }
    }
    per_channel(0, num_channels, t)
}

/// Serialise the tree breadth first; returns (tree symbols, logical
/// context -> leaf index map).
fn serialise_tree(root: &TreeNode, num_logical: usize) -> (Vec<(u32, u32)>, Vec<u32>) {
    let mut syms = Vec::new();
    let mut leaf_of = vec![0u32; num_logical];
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(root);
    let mut leaves = 0;
    while let Some(n) = queue.pop_front() {
        match n {
            TreeNode::Split {
                prop,
                value,
                left,
                right,
            } => {
                syms.push((1, prop + 1));
                syms.push((0, pack_signed(*value)));
                queue.push_back(left);
                queue.push_back(right);
            }
            TreeNode::Leaf(logical) => {
                syms.push((1, 0));
                syms.push((2, 5)); // gradient predictor
                syms.push((3, 0)); // offset
                syms.push((4, 0)); // multiplier log
                syms.push((5, 0)); // multiplier bits
                leaf_of[*logical as usize] = leaves;
                leaves += 1;
            }
        }
    }
    (syms, leaf_of)
}

#[inline]
fn gradient(w: i64, n: i64, nw: i64) -> i64 {
    let (lo, hi) = (w.min(n), w.max(n));
    if nw < lo {
        hi
    } else if nw > hi {
        lo
    } else {
        w + n - nw
    }
}

/// (context, value) symbols of one channel region.
fn encode_plane(
    data: &[i32],
    w: usize,
    h: usize,
    chan: usize,
    t: &[i32; 4],
    leaf_of: &[u32],
    out: &mut Vec<(u32, u32)>,
) {
    let p = |x: usize, y: usize| data[y * w + x] as i64;
    for y in 0..h {
        for x in 0..w {
            let wv = if x > 0 {
                p(x - 1, y)
            } else if y > 0 {
                p(x, y - 1)
            } else {
                0
            };
            let n = if y > 0 { p(x, y - 1) } else { wv };
            let nw = if x > 0 && y > 0 { p(x - 1, y - 1) } else { wv };
            let ne = if x + 1 < w && y > 0 {
                p(x + 1, y - 1)
            } else {
                n
            };
            let _ = ne;
            let b1 = bucket(wv - nw, t);
            let b2 = bucket(nw - n, t);
            let ctx = leaf_of[(chan * BUCKETS + b1) * BUCKETS + b2];
            let res = p(x, y) - gradient(wv, n, nw);
            out.push((ctx, pack_signed(res as i32)));
        }
    }
}

/// Encode an image losslessly as a bare JPEG XL codestream (modular mode,
/// 1 or 3 components, 1 to 16 bits). Signed samples are coded as their
/// unsigned bit patterns.
pub fn encode_lossless(image: &Image) -> Result<Encoded> {
    let info = &image.info;
    let (width, height) = (info.width as usize, info.height as usize);
    let nc = info.components as usize;
    let bits = info.bits_per_sample as u32;
    if !(1..=16).contains(&bits) {
        return Err(Error::unsupported(CODEC, "bit depth outside 1..=16"));
    }
    if nc != 1 && nc != 3 {
        return Err(Error::unsupported(CODEC, "only 1 or 3 components"));
    }
    if width == 0 || height == 0 {
        return Err(Error::input("empty image"));
    }
    if info.color_space == ColorSpace::YCbCr {
        return Err(Error::unsupported(
            CODEC,
            "YCbCr input (convert to RGB first)",
        ));
    }
    // Signed samples are coded as their unsigned bit patterns.
    let mask = ((1u32 << bits) - 1) as i32;
    let mut planes = image.planes();
    for v in planes.iter_mut().flatten() {
        *v &= mask;
    }
    let rct = nc == 3;
    if rct {
        // Forward YCoCg (RCT type 6): Y, Co, Cg.
        for i in 0..width * height {
            let (r, g, b) = (planes[0][i], planes[1][i], planes[2][i]);
            let co = r - b;
            let tmp = b + (co >> 1);
            let cg = g - tmp;
            let y = tmp + (cg >> 1);
            planes[0][i] = y;
            planes[1][i] = co;
            planes[2][i] = cg;
        }
    }

    // Context tree.
    let t = thresholds(bits);
    let tree = build_tree(nc, &t);
    let (tree_syms, leaf_of) = serialise_tree(&tree, nc * BUCKETS * BUCKETS);
    let num_leaves = leaf_of.iter().max().map_or(1, |&m| m as usize + 1);

    // Group layout (256 x 256).
    const GROUP: usize = 256;
    let gx = width.div_ceil(GROUP);
    let gy = height.div_ceil(GROUP);
    let single_group = gx == 1 && gy == 1;
    let mut streams: Vec<Vec<(u32, u32)>> = Vec::new();
    if single_group {
        let mut s = Vec::new();
        for (c, p) in planes.iter().enumerate() {
            encode_plane(p, width, height, c, &t, &leaf_of, &mut s);
        }
        streams.push(s);
    } else {
        for g in 0..gx * gy {
            let (x0, y0) = ((g % gx) * GROUP, (g / gx) * GROUP);
            let (w, h) = (GROUP.min(width - x0), GROUP.min(height - y0));
            let mut s = Vec::new();
            for (c, p) in planes.iter().enumerate() {
                let mut region: Vec<i32> = Vec::with_capacity(w * h);
                for y in y0..y0 + h {
                    region.extend_from_slice(&p[y * width + x0..][..w]);
                }
                encode_plane(&region, w, h, c, &t, &leaf_of, &mut s);
            }
            streams.push(s);
        }
    }

    // LfGlobal: LfChannelDequantization, global tree, global modular header.
    let mut lf = BitWriter::default();
    lf.bool(true); // LfChannelDequantization.all_default
    lf.bool(true); // global tree present
    let tree_code = EntropyCode::write(&mut lf, 6, tree_syms.iter().copied());
    for &(c, v) in &tree_syms {
        tree_code.write_symbol(&mut lf, c, v);
    }
    let code = EntropyCode::write(&mut lf, num_leaves, streams.iter().flatten().copied());
    let write_group_header = |w: &mut BitWriter, transforms: bool| {
        w.bool(true); // use_global_tree
        w.bool(true); // default weighted predictor parameters
        if transforms {
            w.u32([(0, 0), (1, 0), (2, 4), (18, 8)], 1);
            w.write(2, 0); // RCT
            w.u32([(0, 3), (8, 6), (72, 10), (1096, 13)], 0);
            w.u32([(6, 0), (0, 2), (2, 4), (10, 6)], 6);
        } else {
            w.write(2, 0); // no transforms
        }
    };
    write_group_header(&mut lf, rct);

    let mut sections: Vec<Vec<u8>> = Vec::new();
    if single_group {
        for &(c, v) in &streams[0] {
            code.write_symbol(&mut lf, c, v);
        }
        sections.push(lf.finish());
    } else {
        sections.push(lf.finish());
        let num_lf_groups = width.div_ceil(GROUP * 8) * height.div_ceil(GROUP * 8);
        for _ in 0..num_lf_groups {
            sections.push(Vec::new());
        }
        sections.push(Vec::new()); // HfGlobal
        for s in &streams {
            let mut gw = BitWriter::default();
            write_group_header(&mut gw, false);
            for &(c, v) in s {
                code.write_symbol(&mut gw, c, v);
            }
            sections.push(gw.finish());
        }
    }

    let mut out = BitWriter::default();
    write_image_header(&mut out, width as u32, height as u32, bits, nc == 1);
    write_frame_header(&mut out);
    // TOC.
    out.bool(false); // not permuted
    out.zero_pad();
    for s in &sections {
        out.u32(
            [(0, 10), (1024, 14), (17408, 22), (4_211_712, 30)],
            s.len() as u32,
        );
    }
    out.zero_pad();
    debug_assert_eq!(out.bits_written() % 8, 0);
    let mut data = out.finish();
    for s in sections {
        data.extend_from_slice(&s);
    }
    Ok(Encoded {
        data,
        transform: AppliedTransform::Internal,
        lossy: false,
    })
}
