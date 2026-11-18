//! Portable squared quantisation errors (the encoder's original form).

pub(super) fn error_terms(a: &[f64], d: f64, out: &mut [f64]) {
    for (o, &a) in out.iter_mut().zip(a) {
        // a / d >= 0: truncation is floor (and no libm call).
        let q = (a / d) as i64 as f64;
        let rec = if q == 0.0 { 0.0 } else { (q + 0.5) * d };
        *o = (a - rec) * (a - rec);
    }
}
