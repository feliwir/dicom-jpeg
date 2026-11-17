//! Float to integer conversions that compile to a few instructions on every
//! x86-64 level.
//!
//! `f32::round` and `f64::floor` only become single instructions with
//! SSE4.1; on the baseline target they are calls into libm. These helpers
//! use the truncating `as` cast (always one instruction) and correct the
//! result, giving exactly the same values as `x.round() as i32` and so on,
//! including saturation and `NaN -> 0`.

/// `x.round() as i32`: round half away from zero, saturating.
#[inline(always)]
pub fn round_i32(x: f32) -> i32 {
    let t = x as i32;
    // Exact: for |x| < 2^24, x and trunc(x) are within one binade (or t = 0);
    // larger values are integers and give 0.
    let f = x - t as f32;
    t.saturating_add((f >= 0.5) as i32)
        .saturating_sub((f <= -0.5) as i32)
}

/// `x.round() as i64`: round half away from zero, saturating.
#[inline(always)]
pub fn round_i64(x: f64) -> i64 {
    let t = x as i64;
    let f = x - t as f64;
    t.saturating_add((f >= 0.5) as i64)
        .saturating_sub((f <= -0.5) as i64)
}

/// `x.floor() as i64`, saturating.
#[inline(always)]
pub fn floor_i64(x: f64) -> i64 {
    let t = x as i64;
    t.saturating_sub((x < t as f64) as i64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cases_f64() -> Vec<f64> {
        let mut v = vec![
            0.0,
            -0.0,
            0.5,
            -0.5,
            1.5,
            -1.5,
            2.5,
            0.49999999999999994,
            -0.49999999999999994,
            4503599627370495.5,
            -4503599627370495.5,
            9007199254740993.0,
            1e300,
            -1e300,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            i64::MAX as f64,
            i64::MIN as f64,
        ];
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..100_000 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let scale = [1.0, 100.0, 70000.0, 1e12][(s & 3) as usize];
            let x = (s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0;
            v.push(x * scale);
            v.push((x * scale * 2.0).round() / 2.0);
        }
        v
    }

    #[test]
    fn matches_std() {
        for x in cases_f64() {
            assert_eq!(round_i64(x), x.round() as i64, "round_i64({x})");
            assert_eq!(floor_i64(x), x.floor() as i64, "floor_i64({x})");
            let y = x as f32;
            assert_eq!(round_i32(y), y.round() as i32, "round_i32({y})");
        }
        for y in [
            0.49999997f32,
            -0.49999997,
            8388607.5,
            -8388607.5,
            2147483520.0,
            3e9,
            -3e9,
        ] {
            assert_eq!(round_i32(y), y.round() as i32, "round_i32({y})");
        }
    }
}
