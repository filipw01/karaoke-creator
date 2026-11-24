use super::{Track, merge_beams, setting};

fn enabled(name: &str, default: bool) -> bool {
    std::env::var(name).map_or(default, |value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn number(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(default)
}

fn median(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable_by(f64::total_cmp);
    Some((values[(values.len() - 1) / 2] + values[values.len() / 2]) / 2.0)
}

fn midi(hz: f64) -> Option<f64> {
    (hz.is_finite() && hz > 0.0).then(|| 69.0 + 12.0 * (hz.max(1e-6) / 440.0).log2())
}

fn needs_rescue(track: &Track) -> bool {
    let confidence = track
        .frequencies
        .iter()
        .zip(&track.confidences)
        .filter_map(|(&hz, &score)| midi(hz).map(|_| score))
        .fold((0.0, 0), |(sum, count), score| (sum + score, count + 1));
    let mean = confidence.0 / f64::from(confidence.1.max(1));
    let register = median(
        track
            .frequencies
            .iter()
            .filter_map(|&hz| midi(hz))
            .filter(|note| (45.0..=90.0).contains(note))
            .collect(),
    )
    .unwrap_or(0.0);
    mean < number("BASIC_PITCH_HIGHPASS_MAX_BASE_CONFIDENCE", 0.45)
        || (mean < number("BASIC_PITCH_HIGHPASS_HIGH_REGISTER_MAX_CONFIDENCE", 0.60)
            && register >= number("BASIC_PITCH_HIGHPASS_HIGH_REGISTER_MIN_MIDI", 66.0))
}

fn replacements(base: &Track, highpass: &Track, delta: f64) -> Vec<bool> {
    if base.frequencies.len() != highpass.frequencies.len() {
        return Vec::new();
    }
    base.frequencies
        .iter()
        .zip(&highpass.frequencies)
        .map(|(&low, &high)| {
            midi(high).is_some_and(|high| {
                midi(low).is_none_or(|low| high.round_ties_even() >= low.round_ties_even() + delta)
            })
        })
        .collect()
}

fn resolve(
    mut base: Track,
    mut highpass: Track,
    replace: &[bool],
    minimum_fraction: f64,
    partial_context: Option<f64>,
) -> Track {
    let changed = replace.iter().filter(|&&value| value).count();
    if changed == 0 {
        return base;
    }
    if changed as f64 / replace.len() as f64 >= minimum_fraction {
        highpass.energies = base.energies;
        return highpass;
    }
    let Some(context) = partial_context else {
        return base;
    };
    let radius = median(
        base.times
            .windows(2)
            .map(|pair| pair[1] - pair[0])
            .collect(),
    )
    .filter(|&step| step.is_finite() && step > 0.0)
    .map_or(0, |step| {
        ((context / step).round_ties_even() as usize).max(1)
    });
    let matching_onsets = base
        .onset_scores
        .as_ref()
        .zip(highpass.onset_scores.as_ref())
        .is_some_and(|(base, highpass)| base.len() == highpass.len());
    for index in 0..replace.len() {
        let start = index.saturating_sub(radius);
        let end = index
            .saturating_add(radius)
            .saturating_add(1)
            .min(replace.len());
        if midi(highpass.frequencies[index]).is_none() || !replace[start..end].contains(&true) {
            continue;
        }
        base.frequencies[index] = highpass.frequencies[index];
        base.confidences[index] = highpass.confidences[index];
        if base.candidate_midis[index].len() == highpass.candidate_midis[index].len() {
            base.candidate_midis[index].clone_from(&highpass.candidate_midis[index]);
            base.candidate_scores[index].clone_from(&highpass.candidate_scores[index]);
        }
        if matching_onsets
            && let Some((base, filtered)) = base
                .onset_scores
                .as_mut()
                .zip(highpass.onset_scores.as_ref())
        {
            base[index] = filtered[index];
        }
    }
    base
}

pub(super) fn with_rescue(
    base: Track,
    mut analyze: impl FnMut(f64) -> Result<Track, Box<dyn std::error::Error>>,
) -> Track {
    if !enabled("BASIC_PITCH_HIGHPASS_RESCUE", true)
        || (!enabled("BASIC_PITCH_HIGHPASS_FORCE", false) && !needs_rescue(&base))
    {
        return base;
    }
    let cutoff = number("BASIC_PITCH_HIGHPASS_CUTOFF_HZ", 300.0);
    let delta = setting("BASIC_PITCH_HIGHPASS_MIN_DELTA", 6).clamp(1, 24) as f64;
    let fraction = number("BASIC_PITCH_HIGHPASS_FULL_PASS_FRACTION", 0.075).clamp(0.0, 1.0);
    let systematic = number("BASIC_PITCH_HIGHPASS_SYSTEMATIC_CUTOFF_HZ", 750.0).clamp(0.0, 8000.0);
    let highpass = match analyze(cutoff.max(0.0)) {
        Ok(track) => track,
        Err(error) => {
            eprintln!("Basic Pitch high-pass pass unavailable: {error}");
            return base;
        }
    };
    let replace = replacements(&base, &highpass, delta);
    let rescue_fraction =
        replace.iter().filter(|&&value| value).count() as f64 / replace.len().max(1) as f64;
    if rescue_fraction >= fraction && systematic > cutoff {
        match analyze(systematic) {
            Ok(mut track) => {
                if enabled("BASIC_PITCH_HIGHPASS_MERGE_CANDIDATES", true)
                    && let Some((midis, scores)) = merge_beams(&base, &track)
                {
                    track.candidate_midis = midis;
                    track.candidate_scores = scores;
                }
                track.energies = base.energies;
                return track;
            }
            Err(error) => eprintln!("Basic Pitch systematic high-pass pass unavailable: {error}"),
        }
    }
    let partial = enabled("BASIC_PITCH_HIGHPASS_PARTIAL_RESCUE", false)
        .then(|| number("BASIC_PITCH_HIGHPASS_CONTEXT_S", 0.25).clamp(0.0, 0.75));
    resolve(base, highpass, &replace, fraction, partial)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(frequencies: Vec<f64>, energy: f64) -> Track {
        let length = frequencies.len();
        Track {
            frequencies,
            times: (0..length).map(|frame| frame as f64 * 0.01).collect(),
            confidences: vec![1.0; length],
            energies: vec![energy; length],
            candidate_midis: vec![vec![60]; length],
            candidate_scores: vec![vec![0.5]; length],
            onset_scores: None,
        }
    }

    #[test]
    fn systematic_rescue_replaces_all_pitches_and_preserves_base_energies() {
        let base = fixture(vec![261.6; 4], 0.01);
        let highpass = fixture(vec![523.25; 4], 0.0);
        let replace = replacements(&base, &highpass, 6.0);
        let merged = resolve(base, highpass, &replace, 0.5, None);
        assert_eq!(merged.frequencies, [523.25; 4]);
        assert_eq!(merged.energies, [0.01; 4]);
    }

    #[test]
    fn sparse_rescue_only_replaces_frames_when_partial_is_enabled() {
        for context in [None, Some(0.01)] {
            let base = fixture(vec![261.6; 4], 0.01);
            let highpass = fixture(vec![523.25, 300.0, 261.6, 261.6], 0.0);
            let replace = replacements(&base, &highpass, 6.0);
            let merged = resolve(base, highpass, &replace, 0.5, context);
            assert_eq!(
                merged.frequencies,
                if context.is_some() {
                    [523.25, 300.0, 261.6, 261.6]
                } else {
                    [261.6; 4]
                }
            );
        }
    }
}
