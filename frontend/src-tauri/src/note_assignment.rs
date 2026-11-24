pub(crate) mod candidates;
pub(crate) mod forced_runs;
pub(crate) mod lyrics;
pub(crate) mod segments;

fn env_bool(key: &str, default: bool) -> bool {
    std::env::var(key).map_or(default, |value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|value| value.trim().parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(default)
}

fn env_i32(key: &str, default: i32) -> i32 {
    std::env::var(key)
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(default)
}

mod repairs;

use std::collections::HashSet;

use serde::Deserialize;

use candidates::{Beam, CandidateSelection, Span, evidence_for_spans};
use segments::{SegmentRequest, Word, to_hz};

#[derive(Deserialize)]
pub(crate) struct AssignmentRequest {
    #[serde(flatten)]
    segment: SegmentRequest,
    beam: Option<Beam>,
    #[serde(default)]
    line_breaks: Vec<usize>,
}

fn evidence_span(word: &Word) -> Option<(f64, f64)> {
    word.pitch_contour
        .iter()
        .filter_map(|&(time, _)| time.is_finite().then_some(time))
        .fold(None, |bounds, time| {
            Some(bounds.map_or((time, time), |(start, end): (f64, f64)| {
                (start.min(time), end.max(time))
            }))
        })
}

impl AssignmentRequest {
    pub(crate) fn resolve(self) -> Result<(Vec<Word>, Vec<usize>), String> {
        let mut parts = self.segment.resolve()?;
        for group in &mut parts {
            for part in group.iter_mut().skip(1) {
                part.continuation_of_previous = true;
            }
        }
        let words = &self.segment.words;
        let selection = self.beam.map(|beam| CandidateSelection {
            beam,
            times: self.segment.times,
            energies: self.segment.energies,
            frequencies: self.segment.frequencies,
            confidences: self.segment.confidences,
            basic_pitch: self.segment.basic_pitch,
            spans: words
                .iter()
                .zip(&parts)
                .flat_map(|(word, parts)| {
                    let marked = word.text.contains('~');
                    parts.iter().map(move |part| {
                        let trim = !marked && parts.len() == 1;
                        let (start, end) = if trim {
                            (word.start, word.end)
                        } else if marked {
                            evidence_span(part).unwrap_or((part.start, part.end))
                        } else {
                            (part.start, part.end)
                        };
                        Span {
                            start,
                            end,
                            trim,
                            anchor: part.midi_note,
                        }
                    })
                })
                .collect(),
        });
        if let Some(selection) = &selection {
            for (part, candidate) in parts.iter_mut().flatten().zip(selection.resolve()) {
                if let Some(note) = candidate {
                    part.midi_note = note;
                    part.pitch_hz = to_hz(note);
                }
            }
        }
        repairs::repair(words, &mut parts);
        if let Some(selection) = &selection {
            let spans: Vec<_> = parts
                .iter()
                .flatten()
                .map(|part| {
                    let (start, end) = evidence_span(part).unwrap_or((part.start, part.end));
                    (start.min(end), start.max(end))
                })
                .collect();
            let evidence = evidence_for_spans(&selection.beam, &selection.times, &spans, 12);
            for (part, evidence) in parts.iter_mut().flatten().zip(evidence) {
                part.pitch_candidates = evidence
                    .into_iter()
                    .map(|candidate| serde_json::json!(candidate))
                    .collect();
            }
        }

        let requested: HashSet<_> = self.line_breaks.into_iter().collect();
        let mut segments = Vec::new();
        let mut breaks = Vec::new();
        for (index, group) in parts.into_iter().enumerate() {
            if !group.is_empty() && requested.contains(&index) {
                breaks.push(segments.len());
            }
            segments.extend(group);
        }
        Ok((segments, breaks))
    }
}
