/// Tracks pitch for mono samples at 22,050 Hz. Unvoiced frequencies are NaN.
pub fn track(samples: &[f64], frame: usize, hop: usize) -> (Vec<f64>, Vec<bool>, Vec<f64>) {
    let mut tracker = pyin::PYINExecutor::new(
        65.406_391_325_149_66,
        1046.502_261_202_394_5,
        22050,
        frame,
        None,
        Some(hop),
        None,
    );
    let (_, frequencies, voiced, probability) = tracker.pyin(
        samples,
        f64::NAN,
        pyin::Framing::Center(pyin::PadMode::Constant(0.0)),
    );
    (frequencies, voiced, probability)
}
