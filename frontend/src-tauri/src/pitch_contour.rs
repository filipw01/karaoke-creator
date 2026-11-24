use std::collections::HashMap;

const SMOOTHING_FRAMES: usize = 13;
pub(crate) const CORE_CLASSES: usize = 4;

fn midi(frequency: f64) -> Option<i32> {
    (frequency.is_finite() && frequency > 0.0)
        .then(|| (69.0 + 12.0 * (frequency / 440.0).log2()).round_ties_even() as i32)
}

fn frequency(midi: i32) -> f64 {
    440.0 * ((midi as f64 - 69.0) / 12.0).exp2()
}

fn smooth(values: &[f64], window: usize) -> Vec<f64> {
    if window <= 1 || values.len() < 2 {
        return values.to_vec();
    }
    let notes: Vec<_> = values.iter().map(|&value| midi(value)).collect();
    let radius = window / 2;
    let mut counts = HashMap::with_capacity(window);
    notes
        .iter()
        .enumerate()
        .map(|(index, &note)| {
            let Some(note) = note else { return 0.0 };
            let neighborhood =
                &notes[index.saturating_sub(radius)..(index + radius + 1).min(notes.len())];
            let voiced = || {
                neighborhood
                    .iter()
                    .flatten()
                    .copied()
                    .filter(|&note| note > 0)
            };
            counts.clear();
            for candidate in voiced() {
                *counts.entry(candidate).or_insert(0) += 1;
            }
            // Preserve the first occurrence on ties so note onsets do not favor lower pitches.
            let mode = counts
                .values()
                .max()
                .and_then(|best| voiced().find(|key| counts[key] == *best));
            frequency(mode.unwrap_or(note))
        })
        .collect()
}

fn class_counts(values: &[f64]) -> [usize; 12] {
    let mut counts = [0; 12];
    for note in values.iter().filter_map(|&value| midi(value)) {
        counts[note.rem_euclid(12) as usize] += 1;
    }
    counts
}

fn dominant_classes(values: &[f64]) -> Vec<usize> {
    let counts = class_counts(values);
    let total = counts.iter().sum::<usize>();
    if total == 0 {
        return Vec::new();
    }
    let mut ranked = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    ranked.sort_by_key(|&class| std::cmp::Reverse(counts[class]));
    let support = |classes: &[usize]| {
        classes.iter().map(|&class| counts[class]).sum::<usize>() as f64 / total as f64
    };
    let core = &ranked[..CORE_CLASSES];
    let full = &ranked[..6];
    let core_support = support(core);
    let full_support = support(full);
    let selected = if core_support >= 0.60 && full_support < 0.90 {
        core
    } else if full_support >= 0.80 {
        full
    } else if core_support >= 0.45 {
        core
    } else {
        return Vec::new();
    };
    let mut classes = selected.to_vec();
    classes.sort_unstable();
    classes
}

pub(crate) struct Contour {
    pub(crate) frequencies: Vec<f64>,
    pub(crate) classes: Vec<usize>,
    fast_profile: bool,
}

pub(crate) fn prepare(frequencies: Vec<f64>) -> Contour {
    let fast_profile = dominant_classes(&frequencies) == [0, 3, 7, 10];
    let frequencies = if fast_profile {
        frequencies
    } else {
        smooth(&frequencies, SMOOTHING_FRAMES)
    };
    let classes = dominant_classes(&frequencies);
    Contour {
        frequencies,
        classes,
        fast_profile,
    }
}

fn leading_tone(values: &[f64], classes: &[usize]) -> Option<usize> {
    let counts = class_counts(values);
    let tonic = classes
        .iter()
        .copied()
        .min_by_key(|&class| (std::cmp::Reverse(counts[class]), class))?;
    let tone = (tonic + 11) % 12;
    let minimum = (0.015 * counts.iter().sum::<usize>() as f64).max(8.0);
    (counts[tone] as f64 >= minimum).then_some(tone)
}

pub(crate) fn harmonic_class(scores: &[f32; 12], excluded: &[usize]) -> Option<usize> {
    let peak = scores.iter().copied().fold(0.0, f32::max);
    if peak <= 0.0 {
        return None;
    }
    (0..12)
        .filter(|class| !excluded.contains(class) && scores[*class] >= 0.65 * peak)
        .min_by(|&a, &b| scores[b].total_cmp(&scores[a]).then_with(|| a.cmp(&b)))
}

pub(crate) fn finish(
    mut contour: Contour,
    add_accompaniment: bool,
    harmonic_class: Option<usize>,
) -> Vec<f64> {
    if add_accompaniment && contour.classes.len() == CORE_CLASSES {
        contour.classes.extend(harmonic_class);
        contour
            .classes
            .extend(leading_tone(&contour.frequencies, &contour.classes));
    }
    let mut allowed = [false; 12];
    for class in contour.classes {
        allowed[class] = true;
    }
    for value in &mut contour.frequencies {
        let Some(note) = midi(*value) else { continue };
        // At the same distance, prefer the lower note.
        let nearest = (0..=6)
            .flat_map(|distance| [-distance, distance])
            .find(|&delta| allowed[(note + delta).rem_euclid(12) as usize]);
        if let Some(delta) = nearest
            && (!contour.fast_profile || delta.abs() <= 1)
        {
            *value = frequency(note + delta);
        }
    }
    contour.frequencies
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_filter_removes_isolated_flips_and_preserves_gaps_and_ties() {
        let values = [261.6, 261.6, 392.0, 261.6, 261.6];
        assert_eq!(smooth(&values, 5), vec![frequency(60); 5]);
        assert_eq!(
            smooth(&[392.0, 0.0, 261.6], 3),
            [frequency(67), 0.0, frequency(60)]
        );
        assert_eq!(smooth(&[392.0, 261.6], 3), [frequency(67); 2]);
    }

    #[test]
    fn snaps_nearby_notes_and_retains_a_compact_tonal_core() {
        let mut voiced = Vec::new();
        for note in [60, 62, 64, 65, 67, 69] {
            voiced.extend([frequency(note); 10]);
        }
        voiced.push(frequency(61));
        let classes = dominant_classes(&voiced);
        assert_eq!(classes, [0, 2, 4, 5, 7, 9]);
        let contour = Contour {
            frequencies: voiced,
            classes,
            fast_profile: false,
        };
        assert_eq!(finish(contour, false, None).last(), Some(&frequency(60)));

        let mut voiced = Vec::new();
        for (note, count) in [
            (60, 34),
            (67, 28),
            (70, 20),
            (51, 18),
            (63, 8),
            (65, 6),
            (68, 4),
            (69, 10),
            (61, 10),
            (66, 10),
        ] {
            voiced.extend(std::iter::repeat_n(frequency(note), count));
        }
        assert_eq!(dominant_classes(&voiced), [0, 3, 7, 10]);
    }
}
