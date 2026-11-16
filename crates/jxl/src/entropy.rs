//! Entropy decoding (ISO/IEC 18181-1 Annex C): hybrid integers, prefix
//! codes (RFC 7932 code layout), the rANS coder with alias tables, context
//! clustering and LZ77.

use super::bits::{BitReader, bits, invalid, val};
use dicom_jpeg_common::Result;

/// `ceil(log2(x))` for `x >= 1`.
fn ceil_log2(x: u32) -> u32 {
    x.next_power_of_two().trailing_zeros()
}

/// Hybrid integer configuration (C.2.4).
#[derive(Debug, Clone, Copy)]
pub(crate) struct HybridUint {
    split_exp: u32,
    msb: u32,
    lsb: u32,
}

impl HybridUint {
    pub(crate) fn decode(br: &mut BitReader, log_alpha_size: u32) -> Result<Self> {
        let split_exp = br.read(ceil_log2(log_alpha_size + 1))?;
        if split_exp > log_alpha_size {
            return Err(invalid("hybrid integer split exponent out of range"));
        }
        let (mut msb, mut lsb) = (0, 0);
        if split_exp != log_alpha_size {
            msb = br.read(ceil_log2(split_exp + 1))?;
            if msb > split_exp {
                return Err(invalid("hybrid integer msb_in_token out of range"));
            }
            lsb = br.read(ceil_log2(split_exp - msb + 1))?;
            if msb + lsb > split_exp {
                return Err(invalid("hybrid integer lsb_in_token out of range"));
            }
        }
        Ok(HybridUint {
            split_exp,
            msb,
            lsb,
        })
    }

    #[inline]
    fn read(&self, token: u32, br: &mut BitReader) -> Result<u32> {
        let split = 1u32 << self.split_exp;
        if token < split {
            return Ok(token);
        }
        let in_token = self.msb + self.lsb;
        let nbits = (self.split_exp - in_token) as u64 + ((token - split) >> in_token) as u64;
        if nbits > 31 {
            return Err(invalid("hybrid integer too large"));
        }
        let low = token & ((1 << self.lsb) - 1);
        let t = token >> self.lsb;
        let hi = (t & ((1 << self.msb) - 1)) | (1 << self.msb);
        let rest = br.read(nbits as u32)? as u64;
        let v = ((((hi as u64) << nbits) | rest) << self.lsb) | low as u64;
        u32::try_from(v).map_err(|_| invalid("hybrid integer too large"))
    }
}

// ---------------------------------------------------------------------------
// Prefix codes

/// A canonical prefix code; codes are read most significant bit first.
#[derive(Debug, Clone)]
struct PrefixCode {
    /// Direct lookup on `FAST_BITS` stream bits: (symbol, length), length 0 = slow path.
    fast: Vec<(u16, u8)>,
    /// Number of codes of each length 1..=15.
    counts: [u16; 16],
    /// Symbols sorted by (length, symbol).
    sorted: Vec<u16>,
    /// Single symbol coded with zero bits.
    single: Option<u16>,
}

const FAST_BITS: u32 = 10;

impl PrefixCode {
    fn from_lengths(lengths: &[u8]) -> Result<Self> {
        let nonzero: Vec<usize> = (0..lengths.len()).filter(|&i| lengths[i] > 0).collect();
        if nonzero.is_empty() {
            return Err(invalid("empty prefix code"));
        }
        if nonzero.len() == 1 {
            return Ok(PrefixCode {
                fast: Vec::new(),
                counts: [0; 16],
                sorted: Vec::new(),
                single: Some(nonzero[0] as u16),
            });
        }
        let mut counts = [0u16; 16];
        for &i in &nonzero {
            counts[lengths[i] as usize] += 1;
        }
        // Kraft equality: a complete code.
        let space: u32 = (1..16).map(|l| (counts[l] as u32) << (15 - l)).sum();
        if space != 1 << 15 {
            return Err(invalid("incomplete or oversubscribed prefix code"));
        }
        let mut sorted = Vec::with_capacity(nonzero.len());
        for len in 1..16u8 {
            sorted.extend(
                nonzero
                    .iter()
                    .filter(|&&i| lengths[i] == len)
                    .map(|&i| i as u16),
            );
        }
        // Fast table: enumerate canonical codes of length <= FAST_BITS.
        let mut fast = vec![(0u16, 0u8); 1 << FAST_BITS];
        let mut code = 0u32;
        let mut idx = 0usize;
        for len in 1..16u32 {
            for _ in 0..counts[len as usize] {
                if len <= FAST_BITS {
                    // Stream order is the bit-reversed code.
                    let rev = code.reverse_bits() >> (32 - len);
                    let mut k = rev;
                    while k < 1 << FAST_BITS {
                        fast[k as usize] = (sorted[idx], len as u8);
                        k += 1 << len;
                    }
                }
                code += 1;
                idx += 1;
            }
            code <<= 1;
        }
        Ok(PrefixCode {
            fast,
            counts,
            sorted,
            single: None,
        })
    }

    #[inline]
    fn read(&self, br: &mut BitReader) -> Result<u32> {
        if let Some(s) = self.single {
            return Ok(s as u32);
        }
        let (sym, len) = self.fast[br.peek(FAST_BITS) as usize];
        if len > 0 {
            br.consume(len as u32)?;
            return Ok(sym as u32);
        }
        // Slow path: canonical decoding bit by bit.
        let (mut code, mut first, mut index) = (0i32, 0i32, 0i32);
        for len in 1..16 {
            code |= br.read(1)? as i32;
            let count = self.counts[len] as i32;
            if code - first < count {
                return Ok(self.sorted[(index + code - first) as usize] as u32);
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        Err(invalid("invalid prefix code"))
    }

    /// Read a prefix code over `alphabet_size` symbols (RFC 7932 3.4 / 3.5).
    fn decode(br: &mut BitReader, alphabet_size: u32) -> Result<Self> {
        if alphabet_size == 1 {
            return Self::from_lengths(&[1]);
        }
        let hskip = br.read(2)?;
        let mut lengths = vec![0u8; alphabet_size as usize];
        if hskip == 1 {
            // Simple prefix code.
            let nsym = br.read(2)? + 1;
            let nbits = ceil_log2(alphabet_size);
            let mut syms = [0usize; 4];
            for i in 0..nsym as usize {
                let s = br.read(nbits)?;
                if s >= alphabet_size || syms[..i].contains(&(s as usize)) {
                    return Err(invalid("invalid simple prefix code"));
                }
                syms[i] = s as usize;
            }
            let lens: &[u8] = match nsym {
                1 => &[0],
                2 => &[1, 1],
                3 => &[1, 2, 2],
                _ => {
                    if br.read_bool()? {
                        &[1, 2, 3, 3]
                    } else {
                        &[2, 2, 2, 2]
                    }
                }
            };
            if nsym == 1 {
                return Ok(PrefixCode {
                    fast: Vec::new(),
                    counts: [0; 16],
                    sorted: Vec::new(),
                    single: Some(syms[0] as u16),
                });
            }
            for (i, &l) in lens.iter().enumerate() {
                lengths[syms[i]] = l;
            }
            return Self::from_lengths(&lengths);
        }

        // Complex prefix code: code length code lengths with a fixed code.
        const ORDER: [usize; 18] = [1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        // (symbol, length) for 4 peeked bits (RFC 7932 3.5).
        const FIXED: [(u8, u8); 16] = [
            (0, 2),
            (4, 2),
            (3, 2),
            (2, 3),
            (0, 2),
            (4, 2),
            (3, 2),
            (1, 4),
            (0, 2),
            (4, 2),
            (3, 2),
            (2, 3),
            (0, 2),
            (4, 2),
            (3, 2),
            (5, 4),
        ];
        let mut cl_lengths = [0u8; 18];
        let mut space = 32i32;
        let mut num_codes = 0;
        for &o in &ORDER[hskip as usize..] {
            let (sym, len) = FIXED[br.peek(4) as usize];
            br.consume(len as u32)?;
            cl_lengths[o] = sym;
            if sym != 0 {
                space -= 32 >> sym;
                num_codes += 1;
                if space <= 0 {
                    break;
                }
            }
        }
        if !(num_codes == 1 || space == 0) {
            return Err(invalid("invalid code length code"));
        }
        let cl_code = Self::from_lengths(&cl_lengths)?;

        // Symbol code lengths with run-length codes 16 and 17.
        let mut symbol = 0usize;
        let mut prev_len = 8u8;
        let mut repeat = 0u32;
        let mut repeat_len = 0u8;
        let mut space = 1i32 << 15;
        while symbol < alphabet_size as usize && space > 0 {
            let c = cl_code.read(br)? as u8;
            if c < 16 {
                repeat = 0;
                lengths[symbol] = c;
                symbol += 1;
                if c != 0 {
                    prev_len = c;
                    space -= 32768 >> c;
                }
            } else {
                let (extra, new_len) = if c == 16 { (2, prev_len) } else { (3, 0) };
                if repeat_len != new_len {
                    repeat = 0;
                    repeat_len = new_len;
                }
                let old = repeat;
                if repeat > 0 {
                    repeat = (repeat - 2) << extra;
                }
                repeat += br.read(extra)? + 3;
                let delta = (repeat - old) as usize;
                if symbol + delta > alphabet_size as usize {
                    return Err(invalid("prefix code length repeat overflows the alphabet"));
                }
                lengths[symbol..symbol + delta].fill(repeat_len);
                symbol += delta;
                if repeat_len != 0 {
                    space -= (delta as i32) << (15 - repeat_len);
                }
            }
        }
        if space != 0 {
            return Err(invalid("incomplete prefix code"));
        }
        Self::from_lengths(&lengths)
    }
}

// ---------------------------------------------------------------------------
// ANS

const LOG_SUM_PROBS: u32 = 12;
const SUM_PROBS: u32 = 1 << LOG_SUM_PROBS;
const ANS_SIGNATURE: u32 = 0x13 << 16;

#[derive(Debug, Clone, Copy, Default)]
struct AliasBucket {
    dist: u16,
    alias_symbol: u16,
    /// Offset added for the aliased part, already minus the cutoff.
    alias_offset: u16,
    cutoff: u16,
    alias_dist: u16,
}

#[derive(Debug, Clone)]
struct AnsDist {
    buckets: Vec<AliasBucket>,
}

fn read_u8_ans(br: &mut BitReader) -> Result<u32> {
    if br.read_bool()? {
        let n = br.read(3)?;
        Ok((1 << n) + br.read(n)?)
    } else {
        Ok(0)
    }
}

impl AnsDist {
    fn decode(br: &mut BitReader, log_alpha_size: u32) -> Result<Self> {
        let table_size = 1usize << log_alpha_size;
        let mut dist = vec![0u32; table_size];
        let alphabet_size;
        if br.read_bool()? {
            if br.read_bool()? {
                let v0 = read_u8_ans(br)? as usize;
                let v1 = read_u8_ans(br)? as usize;
                if v0 == v1 || v0 >= table_size || v1 >= table_size {
                    return Err(invalid("invalid two-symbol ANS distribution"));
                }
                let p = br.read(LOG_SUM_PROBS)?;
                dist[v0] = p;
                dist[v1] = SUM_PROBS - p;
                alphabet_size = v0.max(v1) + 1;
            } else {
                let v = read_u8_ans(br)? as usize;
                if v >= table_size {
                    return Err(invalid("invalid single-symbol ANS distribution"));
                }
                dist[v] = SUM_PROBS;
                alphabet_size = v + 1;
            }
        } else if br.read_bool()? {
            let n = read_u8_ans(br)? as usize + 1;
            if n > table_size {
                return Err(invalid("invalid flat ANS distribution"));
            }
            let base = SUM_PROBS / n as u32;
            let rem = (SUM_PROBS % n as u32) as usize;
            for (i, d) in dist[..n].iter_mut().enumerate() {
                *d = base + (i < rem) as u32;
            }
            alphabet_size = n;
        } else {
            alphabet_size = Self::decode_complex(br, &mut dist)?;
        }
        Ok(AnsDist {
            buckets: build_alias(&dist, alphabet_size, log_alpha_size)?,
        })
    }

    fn decode_complex(br: &mut BitReader, dist: &mut [u32]) -> Result<usize> {
        let table_size = dist.len();
        let mut len = 0;
        while len < 3 && br.read_bool()? {
            len += 1;
        }
        let shift = (br.read(len)? + (1 << len) - 1) as i32;
        if shift > 13 {
            return Err(invalid("invalid ANS distribution shift"));
        }
        let alphabet_size = read_u8_ans(br)? as usize + 3;
        if alphabet_size > table_size {
            return Err(invalid("ANS alphabet larger than the table"));
        }
        // Log-count prefix code (kLogCountLut): (symbol, length) for 7 peeked bits.
        #[rustfmt::skip]
        const LOGCOUNT: [(u8, u8); 128] = [
            (10, 3), (12, 7), (7, 3), (3, 4), (6, 3), (8, 3), (9, 3), (5, 4),
            (10, 3), ( 4, 4), (7, 3), (1, 4), (6, 3), (8, 3), (9, 3), (2, 4),
            (10, 3), ( 0, 5), (7, 3), (3, 4), (6, 3), (8, 3), (9, 3), (5, 4),
            (10, 3), ( 4, 4), (7, 3), (1, 4), (6, 3), (8, 3), (9, 3), (2, 4),
            (10, 3), (11, 6), (7, 3), (3, 4), (6, 3), (8, 3), (9, 3), (5, 4),
            (10, 3), ( 4, 4), (7, 3), (1, 4), (6, 3), (8, 3), (9, 3), (2, 4),
            (10, 3), ( 0, 5), (7, 3), (3, 4), (6, 3), (8, 3), (9, 3), (5, 4),
            (10, 3), ( 4, 4), (7, 3), (1, 4), (6, 3), (8, 3), (9, 3), (2, 4),
            (10, 3), (13, 7), (7, 3), (3, 4), (6, 3), (8, 3), (9, 3), (5, 4),
            (10, 3), ( 4, 4), (7, 3), (1, 4), (6, 3), (8, 3), (9, 3), (2, 4),
            (10, 3), ( 0, 5), (7, 3), (3, 4), (6, 3), (8, 3), (9, 3), (5, 4),
            (10, 3), ( 4, 4), (7, 3), (1, 4), (6, 3), (8, 3), (9, 3), (2, 4),
            (10, 3), (11, 6), (7, 3), (3, 4), (6, 3), (8, 3), (9, 3), (5, 4),
            (10, 3), ( 4, 4), (7, 3), (1, 4), (6, 3), (8, 3), (9, 3), (2, 4),
            (10, 3), ( 0, 5), (7, 3), (3, 4), (6, 3), (8, 3), (9, 3), (5, 4),
            (10, 3), ( 4, 4), (7, 3), (1, 4), (6, 3), (8, 3), (9, 3), (2, 4),
        ];
        const RLE: u32 = 13;
        let mut codes = vec![0u32; alphabet_size];
        let mut repeat_runs: Vec<(usize, usize)> = Vec::new();
        let mut omit: Option<(u32, usize)> = None;
        let mut i = 0;
        while i < alphabet_size {
            let (sym, l) = LOGCOUNT[br.peek(7) as usize];
            br.consume(l as u32)?;
            let sym = sym as u32;
            if sym == RLE {
                let n = read_u8_ans(br)? as usize + 4;
                if i + n > alphabet_size {
                    return Err(invalid("ANS distribution repeat overflows the alphabet"));
                }
                repeat_runs.push((i, i + n));
                i += n;
                continue;
            }
            codes[i] = sym;
            match omit {
                Some((log, _)) if sym <= log => {}
                _ => omit = Some((sym, i)),
            }
            i += 1;
        }
        let Some((_, omit_pos)) = omit else {
            return Err(invalid("ANS distribution without an explicit entry"));
        };
        if repeat_runs.iter().any(|&(s, _)| s == omit_pos + 1) {
            return Err(invalid("ANS distribution repeats the omitted entry"));
        }
        let mut total = 0u32;
        let mut prev = 0u32;
        let mut run = 0usize;
        let mut idx = 0;
        while idx < alphabet_size {
            if run < repeat_runs.len() && repeat_runs[run].0 == idx {
                let (s, e) = repeat_runs[run];
                for d in &mut dist[s..e] {
                    *d = prev;
                    total += prev;
                }
                run += 1;
                idx = e;
                continue;
            }
            let code = codes[idx];
            if idx == omit_pos {
                prev = 0;
                idx += 1;
                continue;
            }
            let v = if code > 1 {
                let zeros = code as i32 - 1;
                let bitcount = (shift - ((LOG_SUM_PROBS as i32 - zeros) >> 1)).clamp(0, zeros);
                (1u32 << zeros) + (br.read(bitcount as u32)? << (zeros - bitcount))
            } else {
                code
            };
            dist[idx] = v;
            prev = v;
            total += v;
            idx += 1;
        }
        if total >= SUM_PROBS {
            return Err(invalid("ANS distribution sums past 4096"));
        }
        dist[omit_pos] = SUM_PROBS - total;
        Ok(alphabet_size)
    }

    #[inline]
    fn read(&self, br: &mut BitReader, state: &mut u32, log_bucket: u32) -> Result<u32> {
        let idx = *state & (SUM_PROBS - 1);
        let i = (idx >> log_bucket) as usize;
        let pos = idx & ((1 << log_bucket) - 1);
        let b = &self.buckets[i];
        let (symbol, dist, offset) = if pos < b.cutoff as u32 {
            (i as u32, b.dist as u32, pos)
        } else {
            (
                b.alias_symbol as u32,
                b.alias_dist as u32,
                b.alias_offset as u32 + pos,
            )
        };
        let mut s = (*state >> LOG_SUM_PROBS) * dist + offset;
        if s < 1 << 16 {
            s = (s << 16) | br.read(16)?;
        }
        *state = s;
        Ok(symbol)
    }
}

/// Alias table construction (C.2.6), identical to the reference decoder.
fn build_alias(
    dist: &[u32],
    alphabet_size: usize,
    log_alpha_size: u32,
) -> Result<Vec<AliasBucket>> {
    let table_size = dist.len();
    let log_bucket = LOG_SUM_PROBS - log_alpha_size;
    let bucket_size = 1u32 << log_bucket;
    if let Some(s) = dist.iter().position(|&d| d == SUM_PROBS) {
        return Ok((0..table_size)
            .map(|i| AliasBucket {
                dist: dist[i] as u16,
                alias_symbol: s as u16,
                alias_offset: (bucket_size * i as u32) as u16,
                cutoff: 0,
                alias_dist: SUM_PROBS as u16,
            })
            .collect());
    }
    // (cutoff, alias symbol, alias offset)
    let mut cutoff: Vec<u32> = dist.to_vec();
    let mut alias_sym: Vec<u32> = (0..table_size)
        .map(|i| if i < alphabet_size { i as u32 } else { 0 })
        .collect();
    let mut alias_off = vec![0u32; table_size];
    let mut under = Vec::new();
    let mut over = Vec::new();
    for (i, &d) in dist.iter().enumerate() {
        if d < bucket_size {
            under.push(i);
        } else if d > bucket_size {
            over.push(i);
        }
    }
    while let (Some(&o), Some(&u)) = (over.last(), under.last()) {
        over.pop();
        under.pop();
        let by = bucket_size - cutoff[u];
        cutoff[o] -= by;
        alias_sym[u] = o as u32;
        alias_off[u] = cutoff[o];
        if cutoff[o] < bucket_size {
            under.push(o);
        } else if cutoff[o] > bucket_size {
            over.push(o);
        }
    }
    if !over.is_empty() || !under.is_empty() {
        return Err(invalid("ANS distribution does not sum to 4096"));
    }
    Ok((0..table_size)
        .map(|i| {
            if cutoff[i] == bucket_size {
                AliasBucket {
                    dist: dist[i] as u16,
                    alias_symbol: i as u16,
                    alias_offset: 0,
                    cutoff: bucket_size as u16,
                    alias_dist: dist[i] as u16,
                }
            } else {
                AliasBucket {
                    dist: dist[i] as u16,
                    alias_symbol: alias_sym[i] as u16,
                    alias_offset: (alias_off[i] - cutoff[i]) as u16,
                    cutoff: cutoff[i] as u16,
                    alias_dist: dist[alias_sym[i] as usize] as u16,
                }
            }
        })
        .collect())
}

// ---------------------------------------------------------------------------
// Histograms and symbol reading

#[derive(Debug, Clone)]
enum Codes {
    Prefix(Vec<PrefixCode>),
    Ans {
        dists: Vec<AnsDist>,
        log_bucket: u32,
    },
}

#[derive(Debug, Clone, Copy)]
struct Lz77 {
    min_symbol: u32,
    min_length: u32,
    length: HybridUint,
}

/// The entropy code of a stream: clustered distributions with their
/// hybrid integer configurations (C.2.1).
#[derive(Debug, Clone)]
pub(crate) struct Histograms {
    lz77: Option<Lz77>,
    context_map: Vec<u8>,
    configs: Vec<HybridUint>,
    codes: Codes,
}

impl Histograms {
    pub(crate) fn decode(br: &mut BitReader, num_contexts: usize) -> Result<Self> {
        Self::decode_inner(br, num_contexts, true)
    }

    fn decode_inner(br: &mut BitReader, mut num_contexts: usize, allow_lz77: bool) -> Result<Self> {
        let lz77 = if br.read_bool()? {
            if !allow_lz77 {
                return Err(invalid("LZ77 is not allowed here"));
            }
            let min_symbol = br.u32([val(224), val(512), val(4096), bits(15, 8)])?;
            let min_length = br.u32([val(3), val(4), bits(2, 5), bits(8, 9)])?;
            let length = HybridUint::decode(br, 8)?;
            num_contexts += 1;
            Some(Lz77 {
                min_symbol,
                min_length,
                length,
            })
        } else {
            None
        };
        let context_map = decode_context_map(br, num_contexts)?;
        let num_clusters = context_map.iter().copied().max().unwrap_or(0) as usize + 1;
        let use_prefix = br.read_bool()?;
        let log_alpha_size = if use_prefix { 15 } else { 5 + br.read(2)? };
        let configs = (0..num_clusters)
            .map(|_| HybridUint::decode(br, log_alpha_size))
            .collect::<Result<Vec<_>>>()?;
        let codes = if use_prefix {
            let mut sizes = Vec::with_capacity(num_clusters);
            for _ in 0..num_clusters {
                sizes.push(if br.read_bool()? {
                    let n = br.read(4)?;
                    1 + (1 << n) + br.read(n)?
                } else {
                    1
                });
            }
            if sizes.iter().any(|&s| s > 1 << 15) {
                return Err(invalid("prefix code alphabet too large"));
            }
            Codes::Prefix(
                sizes
                    .into_iter()
                    .map(|s| PrefixCode::decode(br, s))
                    .collect::<Result<_>>()?,
            )
        } else {
            Codes::Ans {
                dists: (0..num_clusters)
                    .map(|_| AnsDist::decode(br, log_alpha_size))
                    .collect::<Result<_>>()?,
                log_bucket: LOG_SUM_PROBS - log_alpha_size,
            }
        };
        Ok(Histograms {
            lz77,
            context_map,
            configs,
            codes,
        })
    }
}

pub(crate) fn decode_context_map(br: &mut BitReader, num_contexts: usize) -> Result<Vec<u8>> {
    if num_contexts == 1 {
        return Ok(vec![0]);
    }
    let mut map = vec![0u8; num_contexts];
    if br.read_bool()? {
        let nbits = br.read(2)?;
        for m in &mut map {
            *m = br.read(nbits)? as u8;
        }
    } else {
        let use_mtf = br.read_bool()?;
        let h = Histograms::decode_inner(br, 1, num_contexts > 2)?;
        let mut reader = SymbolReader::new(&h, br, 0)?;
        for m in &mut map {
            let v = reader.read(&h, br, 0)?;
            if v >= 256 {
                return Err(invalid("context map entry out of range"));
            }
            *m = v as u8;
        }
        reader.finish()?;
        if use_mtf {
            let mut mtf: Vec<u8> = (0..=255).collect();
            for m in &mut map {
                let idx = *m as usize;
                let v = mtf[idx];
                mtf.copy_within(0..idx, 1);
                mtf[0] = v;
                *m = v;
            }
        }
    }
    let max = map.iter().copied().max().unwrap_or(0) as usize;
    let mut seen = vec![false; max + 1];
    for &m in &map {
        seen[m as usize] = true;
    }
    if seen.contains(&false) {
        return Err(invalid("incomplete context map"));
    }
    Ok(map)
}

const WINDOW_SIZE: usize = 1 << 20;

/// Decoding state of one entropy-coded stream (C.3).
pub(crate) struct SymbolReader {
    ans_state: u32,
    dist_multiplier: u32,
    window: Vec<u32>,
    num_to_copy: u32,
    copy_pos: u32,
    num_decoded: u32,
}

#[rustfmt::skip]
const SPECIAL_DISTANCES: [(i8, u8); 120] = [
    (0, 1), (1, 0), (1, 1), (-1, 1), (0, 2), (2, 0), (1, 2), (-1, 2), (2, 1), (-2, 1),
    (2, 2), (-2, 2), (0, 3), (3, 0), (1, 3), (-1, 3), (3, 1), (-3, 1), (2, 3), (-2, 3),
    (3, 2), (-3, 2), (0, 4), (4, 0), (1, 4), (-1, 4), (4, 1), (-4, 1), (3, 3), (-3, 3),
    (2, 4), (-2, 4), (4, 2), (-4, 2), (0, 5), (3, 4), (-3, 4), (4, 3), (-4, 3), (5, 0),
    (1, 5), (-1, 5), (5, 1), (-5, 1), (2, 5), (-2, 5), (5, 2), (-5, 2), (4, 4), (-4, 4),
    (3, 5), (-3, 5), (5, 3), (-5, 3), (0, 6), (6, 0), (1, 6), (-1, 6), (6, 1), (-6, 1),
    (2, 6), (-2, 6), (6, 2), (-6, 2), (4, 5), (-4, 5), (5, 4), (-5, 4), (3, 6), (-3, 6),
    (6, 3), (-6, 3), (0, 7), (7, 0), (1, 7), (-1, 7), (5, 5), (-5, 5), (7, 1), (-7, 1),
    (4, 6), (-4, 6), (6, 4), (-6, 4), (2, 7), (-2, 7), (7, 2), (-7, 2), (3, 7), (-3, 7),
    (7, 3), (-7, 3), (5, 6), (-5, 6), (6, 5), (-6, 5), (8, 0), (4, 7), (-4, 7), (7, 4),
    (-7, 4), (8, 1), (8, 2), (6, 6), (-6, 6), (8, 3), (5, 7), (-5, 7), (7, 5), (-7, 5),
    (8, 4), (6, 7), (-6, 7), (7, 6), (-7, 6), (8, 5), (7, 7), (-7, 7), (8, 6), (8, 7),
];

impl SymbolReader {
    /// Start reading a stream; the ANS state is read immediately.
    pub(crate) fn new(h: &Histograms, br: &mut BitReader, dist_multiplier: u32) -> Result<Self> {
        let ans_state = match h.codes {
            Codes::Ans { .. } => br.read(32)?,
            Codes::Prefix(_) => ANS_SIGNATURE,
        };
        Ok(SymbolReader {
            ans_state,
            dist_multiplier,
            window: Vec::new(),
            num_to_copy: 0,
            copy_pos: 0,
            num_decoded: 0,
        })
    }

    #[inline]
    fn token(&mut self, h: &Histograms, br: &mut BitReader, cluster: usize) -> Result<u32> {
        match &h.codes {
            Codes::Prefix(codes) => codes[cluster].read(br),
            Codes::Ans { dists, log_bucket } => {
                dists[cluster].read(br, &mut self.ans_state, *log_bucket)
            }
        }
    }

    #[inline]
    fn push(&mut self, v: u32) {
        let i = self.num_decoded as usize & (WINDOW_SIZE - 1);
        if i < self.window.len() {
            self.window[i] = v;
        } else {
            self.window.push(v);
        }
        self.num_decoded = self.num_decoded.wrapping_add(1);
    }

    #[inline]
    fn copy_one(&mut self) -> u32 {
        let v = self.window[self.copy_pos as usize & (WINDOW_SIZE - 1)];
        self.copy_pos = self.copy_pos.wrapping_add(1);
        self.num_to_copy -= 1;
        self.push(v);
        v
    }

    /// Read one integer in context `ctx` (DecodeHybridVarLenUint).
    pub(crate) fn read(&mut self, h: &Histograms, br: &mut BitReader, ctx: usize) -> Result<u32> {
        let Some(lz) = h.lz77 else {
            let cluster = h.context_map[ctx] as usize;
            let token = self.token(h, br, cluster)?;
            return h.configs[cluster].read(token, br);
        };
        if self.num_to_copy > 0 {
            return Ok(self.copy_one());
        }
        let cluster = h.context_map[ctx] as usize;
        let token = self.token(h, br, cluster)?;
        if token < lz.min_symbol {
            let v = h.configs[cluster].read(token, br)?;
            self.push(v);
            return Ok(v);
        }
        if self.num_decoded == 0 {
            return Err(invalid("LZ77 copy before any symbol"));
        }
        let count = lz
            .length
            .read(token - lz.min_symbol, br)?
            .checked_add(lz.min_length)
            .ok_or_else(|| invalid("LZ77 length overflows"))?;
        let dist_cluster = *h.context_map.last().expect("LZ77 context") as usize;
        let dtoken = self.token(h, br, dist_cluster)?;
        let dsym = h.configs[dist_cluster].read(dtoken, br)?;
        let dist_minus_1 = if self.dist_multiplier == 0 {
            dsym
        } else if dsym >= 120 {
            dsym - 120
        } else {
            let (offset, mul) = SPECIAL_DISTANCES[dsym as usize];
            (self.dist_multiplier as i64 * mul as i64 + offset as i64 - 1).max(0) as u32
        };
        let distance = (dist_minus_1.min(WINDOW_SIZE as u32 - 1) + 1).min(self.num_decoded);
        self.copy_pos = self.num_decoded - distance;
        self.num_to_copy = count;
        Ok(self.copy_one())
    }

    /// Check the final ANS state of the stream.
    pub(crate) fn finish(self) -> Result<()> {
        if self.ans_state != ANS_SIGNATURE {
            return Err(invalid("ANS final state mismatch"));
        }
        Ok(())
    }
}

/// Read an entropy-coded permutation (Lehmer code, C.3.2 / F.3).
pub(crate) fn read_permutation(
    br: &mut BitReader,
    h: &Histograms,
    reader: &mut SymbolReader,
    size: u32,
    skip: u32,
) -> Result<Vec<u32>> {
    let ctx = |x: u32| (ceil_log2(x + 1)).min(7) as usize;
    let end = reader.read(h, br, ctx(size))?;
    if end > size - skip {
        return Err(invalid("invalid permutation length"));
    }
    let mut lehmer = Vec::with_capacity(end as usize);
    let mut prev = 0;
    for i in 0..end {
        let v = reader.read(h, br, ctx(prev))?;
        if v >= size - skip - i {
            return Err(invalid("invalid permutation entry"));
        }
        lehmer.push(v);
        prev = v;
    }
    let mut temp: Vec<u32> = (skip..size).collect();
    let mut perm: Vec<u32> = (0..skip).collect();
    for l in lehmer {
        perm.push(temp.remove(l as usize));
    }
    perm.extend(temp);
    Ok(perm)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_code_canonical() {
        // Lengths 1, 2, 3, 3 -> codes 0, 10, 110, 111 (MSB first).
        let code = PrefixCode::from_lengths(&[1, 2, 3, 3]).unwrap();
        // Stream (LSB first): symbol 2 = "110" -> bits 1,1,0; symbol 0 = "0".
        let data = [0b0000_0011u8];
        let mut br = BitReader::new(&data);
        assert_eq!(code.read(&mut br).unwrap(), 2);
        assert_eq!(code.read(&mut br).unwrap(), 0);
    }
}
