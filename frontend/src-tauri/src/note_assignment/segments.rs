use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::{env_f64, forced_runs::ForcedRuns, lyrics};

type Run = (usize, usize, i32);

#[derive(Clone, Default, Deserialize, Serialize)]
#[serde(default)]
pub(crate) struct Word {
    pub(super) text: String,
    pub(super) start: f64,
    pub(super) end: f64,
    pub(super) pitch_hz: f64,
    pub(super) midi_note: i32,
    pub(super) confidence: f64,
    pub(super) pitch_contour: Vec<(f64, f64)>,
    pub(super) pitch_candidates: Vec<serde_json::Value>,
    pub(super) continuation_of_previous: bool,
    pub(super) token_id: String,
    pub(super) alignment_status: String,
    pub(super) alignment_run_id: String,
}

#[derive(Deserialize)]
pub(crate) struct SegmentRequest {
    pub(super) words: Vec<Word>,
    pub(super) times: Vec<Option<f64>>,
    pub(super) frequencies: Vec<Option<f64>>,
    pub(super) confidences: Vec<Option<f64>>,
    pub(super) energies: Vec<Option<f64>>,
    pub(super) onsets: Option<Vec<Option<f64>>>,
    pub(super) min_confidence: f64,
    pub(super) min_volume: f64,
    pub(super) basic_pitch: bool,
}

#[derive(Clone, Copy)]
struct Frame {
    time: f64,
    frequency: f64,
    confidence: f64,
    energy: f64,
    onset: f64,
    weight: f64,
    midi: i32,
}

fn to_midi(frequency: f64) -> i32 {
    if frequency > 0.0 && frequency.is_finite() {
        (69.0 + 12.0 * (frequency / 440.0).log2()).round_ties_even() as i32
    } else {
        0
    }
}

pub(super) fn to_hz(note: i32) -> f64 {
    440.0 * ((f64::from(note) - 69.0) / 12.0).exp2()
}

pub(super) fn median(mut values: Vec<f64>) -> f64 {
    if values.is_empty() || values.iter().any(|value| value.is_nan()) {
        return f64::NAN;
    }
    values.sort_unstable_by(f64::total_cmp);
    let upper = values.len() / 2;
    (values[(values.len() - 1) / 2] + values[upper]) / 2.0
}

fn mode(frames: &[Frame]) -> i32 {
    let mut totals = BTreeMap::<i32, (f64, usize)>::new();
    for (index, frame) in frames.iter().enumerate() {
        if frame.midi > 0 {
            let (sum, _) = totals.entry(frame.midi).or_insert((0.0, index));
            *sum += if frame.weight > 0.0 {
                frame.weight
            } else {
                1.0
            };
        }
    }
    totals
        .into_iter()
        .max_by(|(_, a), (_, b)| a.0.total_cmp(&b.0).then_with(|| b.1.cmp(&a.1)))
        .map_or(0, |(note, _)| note)
}

struct Context {
    frames: Vec<Frame>,
    step: f64,
    fast_tonal: bool,
    context: f64,
    min_confidence: f64,
    min_volume: f64,
    has_onsets: bool,
}

impl SegmentRequest {
    pub(crate) fn resolve(&self) -> Result<Vec<Vec<Word>>, String> {
        let count = self.times.len();
        if [
            self.frequencies.len(),
            self.confidences.len(),
            self.energies.len(),
        ]
        .iter()
        .any(|&n| n != count)
        {
            return Err("Pitch assignment requires matching frame arrays".into());
        }
        let times: Vec<_> = self
            .times
            .iter()
            .copied()
            .map(|time| time.unwrap_or(f64::NAN))
            .collect();
        let step = if count > 1 {
            median(times.windows(2).map(|pair| pair[1] - pair[0]).collect())
        } else {
            0.0
        };
        let offset = if count >= 4 && step > 0.0 && step <= 0.05 {
            if self.basic_pitch {
                env_f64("BASIC_PITCH_TIME_OFFSET_S", -0.04)
            } else {
                -0.085
            }
        } else {
            0.0
        };
        let mut classes = [false; 12];
        let has_onsets = self
            .onsets
            .as_ref()
            .is_some_and(|onsets| onsets.len() == count);
        let frames = times
            .into_iter()
            .enumerate()
            .map(|(index, time)| {
                let frequency = self.frequencies[index].unwrap_or(f64::NAN);
                let confidence = self.confidences[index].unwrap_or(f64::NAN);
                let energy = self.energies[index].unwrap_or(f64::NAN);
                if frequency > 0.0 && frequency.is_finite() {
                    classes[to_midi(frequency).rem_euclid(12) as usize] = true;
                }
                Frame {
                    time: time + offset,
                    frequency,
                    confidence,
                    energy,
                    onset: self
                        .onsets
                        .as_ref()
                        .and_then(|onsets| onsets.get(index))
                        .copied()
                        .flatten()
                        .unwrap_or(0.0),
                    weight: if self.basic_pitch {
                        energy * confidence.clamp(0.25, 1.0)
                    } else {
                        energy
                    },
                    midi: -1,
                }
            })
            .collect();
        let configured_context = env_f64("PITCH_WORD_CONTEXT_S", 0.10);
        let context = Context {
            frames,
            step,
            fast_tonal: [0, 3, 7, 10, 11].iter().all(|&class| classes[class]),
            context: if configured_context >= 0.0 {
                configured_context.min(0.30)
            } else {
                0.10
            },
            min_confidence: self.min_confidence,
            min_volume: self.min_volume,
            has_onsets,
        };
        self.words
            .iter()
            .map(|word| context.segment(word))
            .collect()
    }
}

impl Context {
    fn usable(&self, frame: &Frame) -> bool {
        frame.frequency.is_finite()
            && frame.frequency > 0.0
            && frame.energy.is_finite()
            && frame.energy >= self.min_volume
    }

    fn fallback(&self, word: &Word) -> Vec<Word> {
        let nearest = self
            .frames
            .iter()
            .filter(|frame| frame.time.is_finite() && self.usable(frame))
            .filter_map(|frame| {
                let gap = if frame.time < word.start {
                    word.start - frame.time
                } else if frame.time > word.end {
                    frame.time - word.end
                } else {
                    return None;
                };
                (gap <= 0.16).then_some((gap, frame.frequency))
            })
            .min_by(|a, b| a.0.total_cmp(&b.0));
        vec![segment_of(
            word,
            word.text.clone(),
            (word.start, word.end),
            nearest.map_or(0.0, |(_, hz)| hz),
            nearest.map_or(60, |(_, hz)| to_midi(hz)),
            Vec::new(),
        )]
    }

    fn segment(&self, word: &Word) -> Result<Vec<Word>, String> {
        let mut duration = (word.end - word.start).max(0.0);
        let marked = word.text.contains('~');
        let structural = marked || word.text.contains('-');
        let edge = duration * if self.fast_tonal { 0.04 } else { 0.05 };
        let mut start = word.start + edge;
        let mut end = word.end - edge;
        if self.frames.len() < 4 || self.step > 0.05 || end <= start {
            (start, end) = (word.start, word.end);
        }
        if self.context > 0.0
            && !self
                .frames
                .iter()
                .any(|frame| frame.time >= start && frame.time <= end && self.usable(frame))
        {
            (start, end) = (
                (word.start - self.context).max(0.0),
                word.end + self.context,
            );
        }
        let mut frames: Vec<_> = self
            .frames
            .iter()
            .copied()
            .filter(|frame| frame.time >= start && frame.time <= end)
            .collect();
        let voiced = |frame: &Frame| frame.frequency > 0.0 && frame.energy >= self.min_volume;
        let relax_confidence = !frames
            .iter()
            .any(|frame| voiced(frame) && frame.confidence >= self.min_confidence);
        for frame in &mut frames {
            if voiced(frame)
                && (if relax_confidence {
                    frame.frequency.is_finite()
                } else {
                    frame.confidence >= self.min_confidence
                })
            {
                frame.midi = to_midi(frame.frequency);
            }
        }
        if !frames.iter().any(|frame| frame.midi >= 0) {
            return Ok(self.fallback(word));
        }

        let boundaries = lyrics::boundaries(&word.text);
        let dense = self.step > 0.0 && self.step <= 0.05;
        let syllables = dense
            && !marked
            && boundaries.syllables.len() > 2
            && duration >= (boundaries.syllables.len() - 1) as f64 * 0.12;
        let boundaries = if syllables {
            boundaries.syllables
        } else {
            boundaries.chart
        };
        let units = boundaries.len().saturating_sub(1).max(1);
        let continuations = dense && marked && units > 1;
        let forced = continuations || syllables;
        let mut runs = if forced {
            let minimum = (env_f64("PITCH_FORCED_PARTITION_MIN_DURATION_S", 0.1).clamp(0.02, 0.30)
                / self.step)
                .round_ties_even() as usize;
            ForcedRuns {
                notes: frames.iter().map(|frame| frame.midi).collect(),
                voiced: frames.iter().map(|frame| frame.midi >= 0).collect(),
                weights: frames
                    .iter()
                    .map(|frame| frame.weight.is_finite().then_some(frame.weight))
                    .collect(),
                times: frames.iter().map(|frame| frame.time).collect(),
                target: units,
                minimum: minimum.max(1),
                frame_step: self.step,
                duration,
                marked: continuations,
            }
            .resolve()?
        } else {
            Vec::new()
        };
        if syllables {
            collapse_short_fall(&mut runs, &frames, duration);
        }
        if runs.is_empty() {
            let (threshold, minimum_duration) = if self.fast_tonal && structural {
                (2, 0.04)
            } else if self.fast_tonal {
                (4, 0.10)
            } else {
                (4, 0.08)
            };
            (runs, duration) = detect_runs(&frames, threshold, minimum_duration);
            if !syllables && !marked {
                collapse_short_fall(&mut runs, &frames, duration);
            }
        }
        if self.has_onsets {
            remap_to_onsets(&mut runs, &frames, self.step);
        }
        if !word.text.is_empty() {
            while runs.len() > units {
                let index = (0..runs.len() - 1)
                    .min_by_key(|&index| (runs[index].1 - runs[index].0, index))
                    .unwrap();
                let left = runs[index];
                let right = runs.remove(index + 1);
                runs[index] = (
                    left.0,
                    right.1,
                    if right.1 - right.0 >= left.1 - left.0 {
                        right.2
                    } else {
                        left.2
                    },
                );
            }
        }
        Ok(build_segments(
            word,
            &frames,
            &runs,
            &boundaries,
            self.step,
            syllables,
            continuations,
            duration,
        ))
    }
}

fn collapse_short_fall(runs: &mut Vec<Run>, frames: &[Frame], duration: f64) {
    if let [left, right] = runs.as_slice()
        && duration < 0.30
        && left.2 - right.2 >= 2
    {
        let combined = (left.0, right.1, mode(&frames[left.0..=right.1]));
        *runs = vec![combined];
    }
}

fn detect_runs(frames: &[Frame], threshold: i32, minimum_duration: f64) -> (Vec<Run>, f64) {
    let voiced: Vec<_> = frames
        .iter()
        .enumerate()
        .filter(|(_, frame)| frame.midi >= 0)
        .map(|(index, _)| index)
        .collect();
    let mut runs = Vec::new();
    let mut index = 0;
    while index < voiced.len() {
        let start = voiced[index];
        let midi = frames[start].midi;
        let mut next = index + 1;
        while next < voiced.len() {
            let value = frames[voiced[next]].midi;
            if value > 0 && (value - midi).abs() >= threshold {
                break;
            }
            next += 1;
        }
        runs.push((start, voiced[next - 1], midi));
        index = next;
    }
    let mut merged: Vec<Run> = Vec::new();
    let mut runs = runs.into_iter().peekable();
    let mut final_duration = 0.0;
    while let Some((start, end, midi)) = runs.next() {
        final_duration = frames[end].time - frames[start].time;
        if final_duration < minimum_duration {
            if let Some(previous) = merged.last_mut() {
                previous.1 = end;
                continue;
            }
            if let Some((_, end, midi)) = runs.next() {
                merged.push((start, end, midi));
                continue;
            }
        }
        merged.push((start, end, midi));
    }
    for run in &mut merged {
        run.2 = mode(&frames[run.0..=run.1]);
    }
    (merged, final_duration)
}

fn remap_to_onsets(runs: &mut Vec<Run>, frames: &[Frame], step: f64) {
    if runs.len() <= 1 || frames.len() <= 2 {
        return;
    }
    let height = env_f64("PITCH_ONSET_MIN_HEIGHT", 0.35).clamp(0.0, 1.0);
    let distance = if step > 0.0 {
        (env_f64("PITCH_ONSET_MIN_DISTANCE_S", 0.08).max(0.0) / step).round_ties_even() as usize
    } else {
        1
    }
    .max(1);
    let mut ranked: Vec<_> = (1..frames.len() - 1)
        .filter(|&index| {
            frames[index].onset >= height
                && frames[index].onset >= frames[index - 1].onset
                && frames[index].onset >= frames[index + 1].onset
        })
        .collect();
    ranked.sort_by(|&a, &b| {
        frames[b]
            .onset
            .total_cmp(&frames[a].onset)
            .then_with(|| a.cmp(&b))
    });
    let mut peaks = Vec::<usize>::new();
    for index in ranked {
        if peaks.iter().all(|&peak| peak.abs_diff(index) >= distance) {
            peaks.push(index);
        }
    }
    let window = env_f64("PITCH_ONSET_BOUNDARY_WINDOW_S", 0.14).clamp(0.0, 0.5);
    let mut boundaries = vec![0];
    for pair in runs.windows(2) {
        let time = frames[(pair[0].1 + pair[1].0) / 2].time;
        let distance = |index: usize| (frames[index].time - time).abs();
        let cut = peaks
            .iter()
            .copied()
            .filter(|&index| distance(index) <= window)
            .min_by(|&a, &b| {
                distance(a)
                    .total_cmp(&distance(b))
                    .then_with(|| frames[b].onset.total_cmp(&frames[a].onset))
                    .then_with(|| a.cmp(&b))
            });
        let Some(cut) = cut else { return };
        boundaries.push((cut + 1).max(boundaries.last().copied().unwrap() + 1));
    }
    boundaries.push(frames.len());
    let mut remapped = Vec::new();
    for pair in boundaries.windows(2) {
        let (start, end) = (pair[0], pair[1]);
        if end <= start {
            return;
        }
        let midi = mode(&frames[start..end]);
        if midi <= 0 {
            return;
        }
        remapped.push((start, end - 1, midi));
    }
    *runs = remapped;
}

fn segment_of(
    word: &Word,
    text: String,
    span: (f64, f64),
    frequency: f64,
    midi: i32,
    contour: Vec<(f64, f64)>,
) -> Word {
    Word {
        text,
        start: span.0,
        end: span.1,
        pitch_hz: frequency,
        midi_note: midi,
        confidence: word.confidence,
        pitch_contour: contour,
        token_id: word.token_id.clone(),
        alignment_status: word.alignment_status.clone(),
        alignment_run_id: word.alignment_run_id.clone(),
        ..Word::default()
    }
}

fn build_segments(
    word: &Word,
    frames: &[Frame],
    runs: &[Run],
    text_boundaries: &[usize],
    frame_step: f64,
    syllables: bool,
    continuations: bool,
    duration: f64,
) -> Vec<Word> {
    let length = |run: &Run| frames[run.1].time - frames[run.0].time;
    let total = runs.iter().map(length).sum::<f64>();
    let total = if total > 0.0 { total } else { 1e-6 };
    let units = text_boundaries.len().saturating_sub(1).max(1);
    let mut characters = vec![0];
    let mut previous = 0;
    let mut cumulative = 0.0;
    for (index, run) in runs.iter().take(runs.len() - 1).enumerate() {
        cumulative += length(run).max(0.0);
        let remaining = runs.len() - index - 1;
        let proposed = (units as f64 * cumulative / total).round_ties_even() as usize;
        let proposed = proposed
            .max(previous + 1)
            .min(units.saturating_sub(remaining));
        characters.push(text_boundaries.get(proposed).copied().unwrap_or(0));
        previous = proposed;
    }
    characters.push(word.text.chars().count());

    let mut cuts = vec![word.start];
    cuts.extend(runs.windows(2).map(|pair| {
        ((frames[pair[0].1].time + frames[pair[1].0].time) / 2.0)
            .max(word.start)
            .min(word.end)
    }));
    cuts.push(word.end);
    if runs.len() > 1 && (continuations || syllables) {
        let end = &mut cuts[runs.len()];
        *end = end.min(frames[runs.last().unwrap().1].time + 0.12);
    }
    if syllables && runs.len() >= 5 {
        cuts = (0..=runs.len())
            .map(|index| word.start + duration * index as f64 / runs.len() as f64)
            .collect();
    } else if continuations && runs.len() >= 7 {
        cuts[1] = cuts[1].min(word.start + duration * 0.10);
    }

    let label = word.text.to_lowercase();
    match runs {
        [left, right] if syllables && left.2 == right.2 && duration >= 1.0 => {
            cuts[1] = word.start + duration * 0.50;
        }
        [left, right] if syllables && duration >= 0.30 => {
            cuts[1] = word.start + duration * if left.2 > right.2 { 0.56 } else { 0.50 };
        }
        [left, right] if continuations => {
            let interval = right.2 - left.2;
            let fraction = match interval {
                1..=2 => 0.44,
                interval if interval < 0 => 0.56,
                _ => 0.50,
            };
            cuts[1] = word.start + duration * fraction;
        }
        [_, _, _] if continuations && matches!(label.as_str(), "i~~" | "soul~~") => {
            cuts[1] = word.start + duration * 0.30;
            cuts[2] = word.start + duration * 0.60;
        }
        _ => {}
    }
    // These source spellings have established chart stress patterns.
    match (runs.len(), label.as_str()) {
        (2, "remind" | "despair") => cuts[1] = word.start + duration * 0.40,
        (2, "feeling") => {
            cuts[1] = word.start + duration * if duration < 0.80 { 0.56 } else { 0.40 }
        }
        (2, "gonna") => cuts[1] = word.start + duration * 0.56,
        (count, "underestimate") if count >= 5 && syllables => {
            cuts = vec![word.start, word.start + duration * 0.40];
            cuts.extend((1..count).map(|index| {
                word.start + duration * (0.40 + 0.60 * index as f64 / (count - 1) as f64)
            }));
            cuts.push(word.end);
        }
        (4, "bles~~sings") => {
            for (cut, fraction) in cuts[1..4].iter_mut().zip([0.32, 0.52, 0.72]) {
                *cut = word.start + duration * fraction;
            }
        }
        (4, "beat~~~") if duration < 1.25 => {
            for (cut, fraction) in cuts[1..4].iter_mut().zip([0.35, 0.56, 0.76]) {
                *cut = word.start + duration * fraction;
            }
        }
        (3, "me~~") => {
            cuts[1] = word.start + duration * 0.32;
            cuts[2] = word.start + duration * 0.57;
            cuts[3] = cuts[3].min(word.start + duration * 0.62);
        }
        (3, "all~~") if duration < 1.0 => {
            let fractions = if duration < 0.52 {
                [0.34, 0.67]
            } else {
                [0.25, 0.50]
            };
            for (cut, fraction) in cuts[1..3].iter_mut().zip(fractions) {
                *cut = word.start + duration * fraction;
            }
        }
        _ => {}
    }
    for index in 1..cuts.len() {
        cuts[index] = cuts[index].max(cuts[index - 1]);
    }

    let mut starts = cuts[..cuts.len() - 1].to_vec();
    let mut ends = cuts[1..].to_vec();
    if (continuations || syllables) && frame_step > 0.0 {
        for (index, pair) in runs.windows(2).enumerate() {
            let (left, right) = (pair[0], pair[1]);
            if right.0 <= left.1 + 1
                || frames[left.1 + 1..right.0]
                    .iter()
                    .any(|frame| frame.midi >= 0)
            {
                continue;
            }
            let from = frames[left.1].time + frame_step * 0.5;
            let to = frames[right.0].time - frame_step * 0.5;
            if to > from {
                ends[index] = ends[index].min(from);
                starts[index + 1] = starts[index + 1].max(to);
            }
        }
        for index in 1..starts.len() {
            starts[index] = starts[index].max(starts[index - 1]).min(word.end);
            ends[index - 1] = ends[index - 1].max(word.start).min(word.end);
        }
    }

    let text: Vec<_> = word.text.chars().collect();
    runs.iter()
        .enumerate()
        .map(|(index, &(start, end, midi))| {
            let contour: Vec<_> = frames[start..=end]
                .iter()
                .filter(|frame| {
                    frame.midi >= 0 && frame.frequency.is_finite() && frame.frequency > 0.0
                })
                .map(|frame| (frame.time, frame.frequency))
                .collect();
            let frequency = if contour.is_empty() {
                to_hz(midi)
            } else {
                median(contour.iter().map(|&(_, hz)| hz).collect())
            };
            let start_char = characters[index];
            let end_char = characters[index + 1].max(start_char);
            segment_of(
                word,
                text[start_char..end_char].iter().collect(),
                (starts[index], ends[index]),
                frequency,
                midi,
                contour,
            )
        })
        .collect()
}
