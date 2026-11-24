use std::collections::{BTreeMap, HashSet};

use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
pub(crate) struct Beam {
    midis: Vec<Vec<i32>>,
    scores: Vec<Vec<Option<f64>>>,
}

impl Beam {
    fn valid(&self) -> bool {
        self.scores.len() == self.midis.len()
            && self
                .scores
                .iter()
                .zip(&self.midis)
                .all(|(scores, midis)| scores.len() == midis.len())
    }

    fn rows(&self) -> impl Iterator<Item = (&[i32], &[Option<f64>])> {
        self.midis
            .iter()
            .map(Vec::as_slice)
            .zip(self.scores.iter().map(Vec::as_slice))
    }
}

#[derive(Serialize)]
pub(crate) struct KeyPrior {
    classes: Vec<usize>,
    margin: f64,
}

pub(crate) fn key_prior(beam: &Beam) -> KeyPrior {
    let mut histogram = [0.0; 12];
    if beam.valid() {
        for (midis, scores) in beam.rows() {
            for (&midi, &score) in midis.iter().zip(scores) {
                if (45..=78).contains(&midi)
                    && let Some(score) = score.filter(|&score| score > 0.0)
                {
                    histogram[midi as usize % 12] += score;
                }
            }
        }
    }
    let total = histogram.iter().sum::<f64>();
    if total <= 0.0 {
        return KeyPrior {
            classes: Vec::new(),
            margin: 0.0,
        };
    }
    // Krumhansl-Schmuckler major and minor profiles.
    let profiles = [
        [
            6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
        ],
        [
            6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
        ],
    ];
    let centered = histogram.map(|weight| weight / total - 1.0 / 12.0);
    let norm_squared = centered.iter().map(|value| value * value).sum::<f64>();
    let mut ranked = Vec::with_capacity(24);
    for (mode, profile) in profiles.into_iter().enumerate() {
        let mean = profile.iter().sum::<f64>() / 12.0;
        let profile = profile.map(|value| value - mean);
        let denominator =
            (norm_squared * profile.iter().map(|value| value * value).sum::<f64>()).sqrt();
        for tonic in 0..12 {
            let dot: f64 = (0..12)
                .map(|pitch| centered[pitch] * profile[(pitch + 12 - tonic) % 12])
                .sum();
            let score = if denominator > 0.0 {
                dot / denominator
            } else {
                0.0
            };
            ranked.push((score, mode, tonic));
        }
    }
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
    let (score, mode, tonic) = ranked[0];
    let intervals = if mode == 0 {
        [0, 2, 4, 5, 7, 9, 11]
    } else {
        [0, 2, 3, 5, 7, 8, 10]
    };
    KeyPrior {
        classes: intervals
            .into_iter()
            .map(|interval| (tonic + interval) % 12)
            .collect(),
        margin: score - ranked[1].0,
    }
}

#[derive(Serialize)]
pub(crate) struct Evidence {
    midi: i32,
    score: f64,
    support: usize,
}

pub(crate) fn evidence_for_spans(
    beam: &Beam,
    times: &[Option<f64>],
    spans: &[(f64, f64)],
    limit: usize,
) -> Vec<Vec<Evidence>> {
    if !beam.valid() || beam.midis.len() != times.len() {
        return spans.iter().map(|_| Vec::new()).collect();
    }
    spans
        .iter()
        .map(|&(start, end)| {
            let mut totals: BTreeMap<i32, (f64, usize)> = BTreeMap::new();
            let mut seen = HashSet::new();
            for ((midis, scores), &time) in beam.rows().zip(times) {
                if !time.is_some_and(|time| time >= start && time <= end) {
                    continue;
                }
                seen.clear();
                for (&midi, &score) in midis.iter().zip(scores) {
                    if !(21..=108).contains(&midi) {
                        continue;
                    }
                    if let Some(score) = score.filter(|&score| score > 0.0) {
                        let (total, support) = totals.entry(midi).or_default();
                        *total += score;
                        *support += usize::from(seen.insert(midi));
                    }
                }
            }
            let mut ranked: Vec<_> = totals.into_iter().collect();
            ranked.sort_by(|a, b| {
                b.1.0
                    .total_cmp(&a.1.0)
                    .then_with(|| b.1.1.cmp(&a.1.1))
                    .then_with(|| a.0.cmp(&b.0))
            });
            let Some(&(_, (maximum, _))) = ranked.first() else {
                return Vec::new();
            };
            ranked
                .into_iter()
                .take(limit)
                .map(|(midi, (total, support))| Evidence {
                    midi,
                    score: (total / maximum * 1_000_000.0).round_ties_even() / 1_000_000.0,
                    support,
                })
                .collect()
        })
        .collect()
}

#[derive(Deserialize)]
pub(crate) struct Span {
    pub(super) start: f64,
    pub(super) end: f64,
    pub(super) anchor: i32,
    pub(super) trim: bool,
}

#[derive(Deserialize)]
pub(crate) struct CandidateSelection {
    pub(super) beam: Beam,
    pub(super) times: Vec<Option<f64>>,
    pub(super) energies: Vec<Option<f64>>,
    pub(super) frequencies: Vec<Option<f64>>,
    pub(super) confidences: Vec<Option<f64>>,
    pub(super) basic_pitch: bool,
    pub(super) spans: Vec<Span>,
}

#[derive(Copy, Clone)]
enum Aggregation {
    Sum,
    Mean,
    Quantile(f64),
}

fn quantile(values: &mut [f64], fraction: f64) -> f64 {
    values.sort_unstable_by(f64::total_cmp);
    let index = fraction * (values.len() - 1) as f64;
    let lower = values[index.floor() as usize];
    let upper = values[index.ceil() as usize];
    lower + (upper - lower) * index.fract()
}

struct Scoring {
    key_prior: KeyPrior,
    key_penalty: f64,
    power: f64,
    floor: f64,
    distance: f64,
    energy: f64,
    min_note: i32,
    max_note: i32,
    range_penalty: f64,
    support_power: f64,
    support_reward: f64,
    aggregation: Aggregation,
    edge: f64,
    offset: f64,
}

impl Scoring {
    fn for_audio(input: &CandidateSelection) -> Self {
        use super::{env_bool, env_f64, env_i32};
        let key_prior = if env_bool("PITCH_WORD_KEY_PRIOR", true) {
            key_prior(&input.beam)
        } else {
            KeyPrior {
                classes: Vec::new(),
                margin: 0.0,
            }
        };
        let key_penalty = if key_prior.margin > 0.0 && !key_prior.classes.is_empty() {
            env_f64("PITCH_WORD_KEY_PENALTY", 0.1).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let mut min_note = env_i32("PITCH_WORD_RANGE_MIN_MIDI", 45).clamp(21, 108);
        let mut max_note = env_i32("PITCH_WORD_RANGE_MAX_MIDI", 78).clamp(min_note, 108);
        let mut distance = env_f64("PITCH_WORD_CANDIDATE_DISTANCE_PENALTY", 0.0).clamp(0.0, 1.0);
        if input.basic_pitch && env_bool("PITCH_WORD_ADAPTIVE_RANGE", true) {
            let mut register: Vec<_> = input
                .frequencies
                .iter()
                .flatten()
                .filter(|&&hz| hz > 0.0)
                .map(|&hz| 69.0 + 12.0 * (hz.max(1e-6) / 440.0).log2())
                .filter(|value| (45.0..=90.0).contains(value))
                .collect();
            if !register.is_empty() {
                let median = quantile(&mut register, 0.5);
                if median >= 66.0 {
                    max_note = max_note.min(((median + 6.0).round_ties_even() as i32).max(72));
                }
                if median >= 72.0 {
                    min_note = min_note.max(((median - 8.0).round_ties_even() as i32).min(72));
                }
                let (confidence, count) = input
                    .confidences
                    .iter()
                    .flatten()
                    .filter(|&&value| value > 0.0)
                    .fold((0.0, 0), |(sum, count), value| (sum + value, count + 1));
                if std::env::var_os("PITCH_WORD_CANDIDATE_DISTANCE_PENALTY").is_none()
                    && distance <= 0.0
                    && median <= 65.0
                    && confidence / f64::from(count.max(1)) >= 0.44
                {
                    distance = 0.15;
                }
            }
        }
        let aggregation = match std::env::var("PITCH_WORD_CANDIDATE_AGGREGATION")
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "mean" => Aggregation::Mean,
            "median" => Aggregation::Quantile(0.5),
            "p75" => Aggregation::Quantile(0.75),
            "p90" => Aggregation::Quantile(0.90),
            _ => Aggregation::Sum,
        };
        Self {
            key_prior,
            key_penalty,
            power: env_f64("PITCH_WORD_CANDIDATE_POWER", 1.0).clamp(0.1, 4.0),
            floor: env_f64("PITCH_WORD_CANDIDATE_FLOOR", 0.0).clamp(0.0, 0.5),
            distance,
            energy: env_f64("PITCH_WORD_ENERGY_POWER", 0.0).clamp(0.0, 2.0),
            min_note,
            max_note,
            range_penalty: env_f64("PITCH_WORD_RANGE_PENALTY", 0.05).clamp(0.0, 1.0),
            support_power: env_f64("PITCH_WORD_CANDIDATE_SUPPORT_POWER", 0.0).clamp(0.0, 1.0),
            support_reward: env_f64("PITCH_WORD_CANDIDATE_SUPPORT_REWARD", 0.0).clamp(0.0, 1.0),
            aggregation,
            edge: env_f64("PITCH_WORD_CANDIDATE_EDGE_FRACTION", 0.0).clamp(0.0, 0.2),
            offset: if input.basic_pitch {
                env_f64("BASIC_PITCH_TIME_OFFSET_S", -0.04)
            } else {
                0.0
            },
        }
    }

    fn best(&self, input: &CandidateSelection, span: &Span) -> Option<i32> {
        let (mut start, mut end) = (span.start, span.end);
        if span.trim {
            let edge = ((end - start) * self.edge).clamp(0.0, 0.05);
            if end - start > 2.0 * edge {
                start += edge;
                end -= edge;
            }
        }
        let frames: Vec<_> = input
            .times
            .iter()
            .enumerate()
            .filter_map(|(index, &time)| {
                let time = time? + self.offset;
                (time >= start && time <= end).then_some(index)
            })
            .collect();
        let scale = if self.energy > 0.0 {
            let mut positive: Vec<_> = frames
                .iter()
                .filter_map(|&index| input.energies[index].filter(|&value| value > 0.0))
                .collect();
            if positive.is_empty() {
                1.0
            } else {
                quantile(&mut positive, 0.5).max(1e-6)
            }
        } else {
            1.0
        };
        let mut observations: BTreeMap<i32, Vec<f64>> = BTreeMap::new();
        for index in frames {
            let weight = if self.energy > 0.0 {
                (input.energies[index].unwrap_or(0.0) / scale)
                    .max(0.0)
                    .powf(self.energy)
                    .min(4.0)
            } else {
                1.0
            };
            for (&midi, &score) in input.beam.midis[index]
                .iter()
                .zip(&input.beam.scores[index])
            {
                let Some(score) = score.filter(|_| midi > 0) else {
                    continue;
                };
                let class = if !self.key_prior.classes.is_empty()
                    && !self.key_prior.classes.contains(&(midi as usize % 12))
                {
                    self.key_penalty
                } else {
                    1.0
                };
                let range = if (self.min_note..=self.max_note).contains(&midi) {
                    1.0
                } else {
                    self.range_penalty
                };
                let distance = (-self.distance * f64::from((midi - span.anchor).abs())).exp();
                let contribution = (score - self.floor).max(0.0).powf(self.power) * distance;
                observations
                    .entry(midi)
                    .or_default()
                    .push(contribution * weight * class * range);
            }
        }
        let minimum = if end - start < 0.08 { 1 } else { 2 };
        let has_sustained = observations.values().any(|values| values.len() >= minimum);
        observations
            .into_iter()
            .filter(|(_, values)| !has_sustained || values.len() >= minimum)
            .map(|(midi, mut values)| {
                let support = values.len() as f64;
                let total = match self.aggregation {
                    Aggregation::Sum => values.iter().sum::<f64>(),
                    Aggregation::Mean => values.iter().sum::<f64>() / support,
                    Aggregation::Quantile(percentile) => quantile(&mut values, percentile),
                };
                (
                    midi,
                    total * support.powf(self.support_reward) / support.powf(self.support_power),
                )
            })
            .max_by(|a, b| a.1.total_cmp(&b.1).then_with(|| b.0.cmp(&a.0)))
            .map(|(midi, _)| midi)
    }
}

impl CandidateSelection {
    pub(crate) fn resolve(&self) -> Vec<Option<i32>> {
        if !super::env_bool("PITCH_WORD_CANDIDATES", true)
            || !self.beam.valid()
            || self.beam.midis.len() != self.times.len()
            || self.energies.len() != self.times.len()
        {
            return vec![None; self.spans.len()];
        }
        let scoring = Scoring::for_audio(&self);
        self.spans
            .iter()
            .map(|span| scoring.best(&self, span))
            .collect()
    }
}
