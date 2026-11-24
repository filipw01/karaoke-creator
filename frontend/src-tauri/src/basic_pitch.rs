use std::{
    collections::{BTreeMap, HashSet},
    io,
    path::Path,
};

use ort::{session::Session, value::TensorRef};
use serde::Serialize;

mod rescue;

use crate::native_pitch::{centered_rms, decode_mono, pcm_samples};

const SAMPLE_RATE: u32 = 22_050;
const WINDOW_SAMPLES: usize = 43_844;
const HOP: usize = 256;
const NOTES: usize = 88;
const FIRST_MIDI: usize = 21;
const MIN_CONFIDENCE: f64 = 0.15;

type Activations = [f64; NOTES];

#[derive(Default, Serialize)]
pub(crate) struct Track {
    pub(crate) times: Vec<f64>,
    pub(crate) frequencies: Vec<f64>,
    pub(crate) confidences: Vec<f64>,
    energies: Vec<f64>,
    candidate_midis: Vec<Vec<usize>>,
    candidate_scores: Vec<Vec<f32>>,
    onset_scores: Option<Vec<f32>>,
}

type CandidateBeam = (Vec<Vec<usize>>, Vec<Vec<f32>>);

fn merge_beams(base: &Track, rescue: &Track) -> Option<CandidateBeam> {
    if base.candidate_midis.len() != rescue.candidate_midis.len() {
        return None;
    }
    let width = base
        .candidate_midis
        .first()?
        .len()
        .max(rescue.candidate_midis.first()?.len());
    let mut midis = Vec::new();
    let mut scores = Vec::new();
    let insert = |scores: &mut BTreeMap<usize, f32>, notes: &[usize], values: &[f32]| {
        for (&midi, &score) in notes.iter().zip(values) {
            if midi > 0 && score.is_finite() {
                let best = scores.entry(midi).or_default();
                *best = best.max(score);
            }
        }
    };
    let by_score =
        |a: &(usize, f32), b: &(usize, f32)| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0));
    for row in 0..base.candidate_midis.len() {
        let mut base_values = BTreeMap::new();
        insert(
            &mut base_values,
            &base.candidate_midis[row],
            &base.candidate_scores[row],
        );
        let mut values = base_values.clone();
        insert(
            &mut values,
            &rescue.candidate_midis[row],
            &rescue.candidate_scores[row],
        );
        let mut selected: Vec<_> = values.into_iter().collect();
        selected.sort_by(by_score);
        selected.truncate(width);

        // Reserve some of the beam for fundamentals from the unfiltered pass.
        let mut reserved: Vec<_> = base_values.into_iter().collect();
        reserved.sort_by(by_score);
        reserved.truncate((width / 3).max(1));
        let mut selected_keys: HashSet<_> = selected.iter().map(|(midi, _)| *midi).collect();
        for candidate in reserved {
            if !selected_keys.insert(candidate.0) {
                continue;
            }
            if selected.len() < width {
                selected.push(candidate);
            } else if let Some(lowest) = selected.iter_mut().max_by(|a, b| by_score(a, b)) {
                *lowest = candidate;
            }
        }
        selected.sort_by(by_score);
        let (mut row_midis, mut row_scores): (Vec<_>, Vec<_>) = selected.into_iter().unzip();
        row_midis.resize(width, 0);
        row_scores.resize(width, 0.0);
        midis.push(row_midis);
        scores.push(row_scores);
    }
    Some((midis, scores))
}

fn setting(key: &str, default: i64) -> i64 {
    std::env::var(key)
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(default)
}

fn strongest(values: &Activations) -> usize {
    values.iter().enumerate().fold(
        0,
        |best, (index, &score)| {
            if score > values[best] { index } else { best }
        },
    )
}

fn average(notes: &[Activations], frame: usize, width: usize) -> Activations {
    if width == 1 {
        return notes[frame];
    }
    let mut averaged = [0.0; NOTES];
    for offset in 0..width {
        let index = (frame + offset)
            .saturating_sub(width / 2)
            .min(notes.len() - 1);
        for (total, &score) in averaged.iter_mut().zip(&notes[index]) {
            *total += score / width as f64;
        }
    }
    averaged
}

fn candidates(activations: &Activations, count: usize) -> (Vec<usize>, Vec<f32>) {
    let mut ranks: [usize; NOTES] = std::array::from_fn(|index| index);
    // An equal activation has no preference in the model. Rank by MIDI for
    // reproducible output, including where a tie crosses the beam cutoff.
    ranks.sort_by(|&a, &b| {
        activations[b]
            .total_cmp(&activations[a])
            .then_with(|| a.cmp(&b))
    });
    ranks[..count]
        .iter()
        .map(|&index| (index + FIRST_MIDI, activations[index] as f32))
        .unzip()
}

fn analyze(samples: &[f32], session: &mut Session) -> Result<Track, Box<dyn std::error::Error>> {
    let overlap = setting("BASIC_PITCH_OVERLAP_FRAMES", 30).clamp(0, 120) as usize;
    let trim = overlap / 2;
    let padding = trim * HOP;
    let stride = WINDOW_SAMPLES - 2 * padding;
    let frame_limit = samples.len() * (SAMPLE_RATE as usize / HOP) / SAMPLE_RATE as usize;
    let mut notes = Vec::with_capacity(frame_limit);
    let mut track = Track::default();
    let mut window = vec![0.0_f32; WINDOW_SAMPLES];
    for start in (0..samples.len() + padding).step_by(stride) {
        if notes.len() >= frame_limit {
            break;
        }
        let source = start.saturating_sub(padding);
        let target = padding.saturating_sub(start);
        let length = (WINDOW_SAMPLES - target).min(samples.len() - source);
        window.fill(0.0);
        window[target..target + length].copy_from_slice(&samples[source..source + length]);
        let tensor = TensorRef::from_array_view(([1usize, WINDOW_SAMPLES, 1], window.as_slice()))?;
        let outputs = session.run(ort::inputs![tensor])?;
        let mut note_outputs = Vec::new();
        for name in outputs.keys() {
            let value = &outputs[name];
            let (shape, values) = value.try_extract_tensor::<f32>()?;
            if matches!(shape.as_ref(), [1, _, 88]) {
                note_outputs.push((name, values));
            }
        }
        let named = |suffix: &str| {
            note_outputs
                .iter()
                .find(|(name, _)| name.ends_with(suffix))
                .map(|(_, values)| *values)
        };
        let notes_output = named(":1")
            .or_else(|| (note_outputs.len() == 2).then(|| note_outputs[1].1))
            .ok_or("Basic Pitch ONNX output has no 88-bin note activation")?;
        let onset = named(":2").or_else(|| (note_outputs.len() == 2).then(|| note_outputs[0].1));
        let size = notes_output.len() / NOTES;
        if size <= 2 * trim || onset.is_some_and(|onset| onset.len() != notes_output.len()) {
            return Err("Basic Pitch ONNX returned invalid frame dimensions".into());
        }
        let count = (size - 2 * trim).min(frame_limit - notes.len());
        let start_time = (start as f64 - padding as f64) / f64::from(SAMPLE_RATE);
        let step = HOP as f64 / f64::from(SAMPLE_RATE);
        for frame in trim..trim + count {
            let row: Activations =
                std::array::from_fn(|index| f64::from(notes_output[frame * NOTES + index]));
            if let Some(onset) = onset {
                track
                    .onset_scores
                    .get_or_insert_with(Vec::new)
                    .push(onset[frame * NOTES + strongest(&row)]);
            }
            notes.push(row);
            track.times.push(start_time + frame as f64 * step);
        }
    }

    let beam = setting("BASIC_PITCH_CANDIDATE_COUNT", 12).clamp(1, NOTES as i64) as usize;
    let smooth = setting("BASIC_PITCH_CANDIDATE_SMOOTH_FRAMES", 1).clamp(1, 15) as usize;
    let rms = centered_rms(samples, 2048, HOP);
    for (frame, &time) in track.times.iter().enumerate() {
        let (midis, scores) = candidates(&average(&notes, frame, smooth), beam);
        track.candidate_midis.push(midis);
        track.candidate_scores.push(scores);
        let confidence = notes[frame][strongest(&notes[frame])];
        track.confidences.push(confidence);
        let frequency = if confidence >= MIN_CONFIDENCE {
            let midi = strongest(&average(&notes, frame, 5)) + FIRST_MIDI;
            440.0 * ((midi as f64 - 69.0) / 12.0).exp2()
        } else {
            0.0
        };
        track.frequencies.push(frequency);
        let position = time * f64::from(SAMPLE_RATE) / HOP as f64;
        let left = position.floor() as usize;
        let energy = if position < 0.0 || position > (rms.len() - 1) as f64 {
            0.0
        } else {
            let right = (left + 1).min(rms.len() - 1);
            rms[left] + (rms[right] - rms[left]) * position.fract()
        };
        track.energies.push(energy);
    }
    Ok(track)
}

pub(crate) fn detect(
    model: &Path,
    source: &Path,
    cutoff: Option<f64>,
) -> Result<Track, Box<dyn std::error::Error>> {
    let samples = pcm_samples(&decode_mono(source, SAMPLE_RATE, cutoff.unwrap_or(0.0))?)?;
    let track = if samples.is_empty() {
        Track::default()
    } else {
        if let Some(runtime) = std::env::var_os("ORT_DYLIB_PATH") {
            ort::init_from(runtime)?.commit();
        }
        let mut session = Session::builder()?
            .with_intra_threads(setting("BASIC_PITCH_INTRA_THREADS", 1).max(1) as usize)?
            .with_inter_threads(setting("BASIC_PITCH_INTER_THREADS", 1).max(1) as usize)?
            .commit_from_file(model)?;
        let base = analyze(&samples, &mut session)?;
        if cutoff.is_some() {
            base
        } else {
            rescue::with_rescue(base, |cutoff| {
                let filtered = pcm_samples(&decode_mono(source, SAMPLE_RATE, cutoff)?)?;
                analyze(&filtered, &mut session)
            })
        }
    };
    Ok(track)
}

pub(crate) fn run(
    model: &Path,
    source: &Path,
    cutoff: Option<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    serde_json::to_writer(io::stdout().lock(), &detect(model, source, cutoff)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beam_merge_reserves_a_base_note_and_keeps_the_higher_duplicate_score() {
        let base = Track {
            candidate_midis: vec![vec![60, 64]],
            candidate_scores: vec![vec![0.3, 0.2]],
            ..Track::default()
        };
        let filtered = Track {
            candidate_midis: vec![vec![64, 67]],
            candidate_scores: vec![vec![0.6, 0.5]],
            ..Track::default()
        };
        assert_eq!(
            merge_beams(&base, &filtered),
            Some((vec![vec![64, 60]], vec![vec![0.6, 0.3]]))
        );
    }

    #[test]
    fn candidates_break_ties_by_midi_and_keep_the_requested_beam() {
        let mut row = [0.0; NOTES];
        row[5] = 0.75;
        row[9] = 0.75;
        row[4] = 0.5;
        assert_eq!(strongest(&row), 5);
        assert_eq!(
            candidates(&row, 3),
            (vec![26, 30, 25], vec![0.75, 0.75, 0.5])
        );
    }
}
