use std::path::Path;

pub(crate) fn detect(path: &Path) -> Result<f64, String> {
    let mut audio = hound::WavReader::open(path).map_err(|error| error.to_string())?;
    let spec = audio.spec();
    if spec.bits_per_sample != 16 || spec.channels == 0 {
        return Err("Tempo detection requires 16-bit PCM".into());
    }
    let samples_per_hop = (spec.sample_rate / 100) * u32::from(spec.channels);
    if samples_per_hop == 0 {
        return Err("Invalid tempo sample rate".into());
    }
    let mut onset = Vec::new();
    let mut previous = 0.0;
    let mut energy = 0.0;
    for (index, sample) in audio.samples::<i16>().enumerate() {
        energy += f64::from(sample.map_err(|error| error.to_string())?).abs();
        if (index + 1) % samples_per_hop as usize == 0 {
            let level = (1.0 + energy / f64::from(samples_per_hop)).ln();
            onset.push((level - previous).max(0.0));
            previous = level;
            energy = 0.0;
        }
    }
    Ok(tempo_from_onsets(&onset).unwrap_or(120.0))
}

fn tempo_from_onsets(onsets: &[f64]) -> Option<f64> {
    if onsets.len() < 300 {
        return None;
    }
    let energy = onsets.iter().map(|onset| onset * onset).sum::<f64>();
    if energy <= f64::EPSILON {
        return None;
    }

    let score = |lag: usize| {
        onsets[..onsets.len() - lag]
            .iter()
            .zip(&onsets[lag..])
            .map(|(left, right)| left * right)
            .sum::<f64>()
            / energy
    };
    let lag = (25..=150).max_by(|left, right| {
        let weight = |lag| {
            let bpm = 6000.0 / lag as f64;
            score(lag) * (-0.5 * (bpm / 120.0).ln().powi(2)).exp()
        };
        weight(*left).total_cmp(&weight(*right))
    })?;
    (score(lag) >= 0.15).then_some(6000.0 / lag as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_click_track_period_and_skips_silence() {
        let mut onsets = vec![0.0; 1200];
        assert_eq!(tempo_from_onsets(&onsets), None);
        for click in onsets.iter_mut().step_by(50) {
            *click = 1.0;
        }
        assert_eq!(tempo_from_onsets(&onsets), Some(120.0));
    }
}
