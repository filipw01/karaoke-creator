use std::collections::{HashMap, HashSet};

use serde_json::{Value, json};

use super::{array, field_text, render_result};
use crate::{api::ApiError, chart_output, external_chart::Phrase};

pub(super) fn span(value: &Value) -> Option<(f64, f64)> {
    let start = value["start"].as_f64()?;
    let end = value["end"].as_f64()?;
    (start.is_finite() && end.is_finite() && end >= start).then_some((start, end))
}

fn by_token(words: &[Value]) -> HashMap<&str, Vec<&Value>> {
    let mut grouped = HashMap::<&str, Vec<&Value>>::new();
    for word in words {
        let token_id = field_text(word, "token_id");
        if !token_id.is_empty() {
            grouped.entry(token_id).or_default().push(word);
        }
    }
    grouped
}

pub(crate) fn pitch_locks(source: &[Value], merged: &[Value]) -> Vec<Value> {
    let final_by_token: HashMap<_, _> = merged
        .iter()
        .map(|alignment| (field_text(alignment, "token_id"), alignment))
        .collect();
    source
        .iter()
        .filter_map(|source| {
            let final_alignment = final_by_token.get(field_text(source, "token_id"))?;
            let mut alignment = (*final_alignment).clone();
            let reasons = array(&final_alignment["evidence"]["reasons"]);
            for (boundary, reason) in [
                (
                    "start",
                    "previous start retained to keep the locked end paired",
                ),
                (
                    "end",
                    "previous end retained to keep the locked start paired",
                ),
            ] {
                if final_alignment[format!("{boundary}_locked")] != true
                    && !reasons.iter().any(|value| value == reason)
                {
                    alignment[boundary] = source[boundary].clone();
                }
            }
            if span(&alignment).is_none() && span(final_alignment).is_some() {
                alignment["start"] = final_alignment["start"].clone();
                alignment["end"] = final_alignment["end"].clone();
            }
            Some(alignment)
        })
        .collect()
}

pub(crate) fn source_spans(alignments: &[Value], words: &[Value]) -> Vec<Value> {
    let by_token = by_token(words);
    alignments
        .iter()
        .filter_map(|alignment| {
            let words = by_token.get(field_text(alignment, "token_id"))?;
            let combined = words.iter().filter_map(|word| span(word)).reduce(
                |(start, end), (other_start, other_end)| {
                    (start.min(other_start), end.max(other_end))
                },
            )?;
            let mut source = alignment.clone();
            source["start"] = json!(combined.0);
            source["end"] = json!(combined.1);
            source["start_locked"] = json!(false);
            source["end_locked"] = json!(false);
            Some(source)
        })
        .collect()
}

pub(crate) fn words(tokens: &[Value], alignments: &[Value], existing: &[Value]) -> Vec<Value> {
    let alignments: HashMap<_, _> = alignments
        .iter()
        .map(|alignment| (field_text(alignment, "token_id"), alignment))
        .collect();
    let existing = by_token(existing);
    let mut projected = Vec::new();
    for token in tokens {
        let id = field_text(token, "id");
        let Some(alignment) = alignments.get(id) else {
            continue;
        };
        let Some((start, end)) = span(alignment).filter(|(start, _)| *start >= 0.0) else {
            continue;
        };
        let fallback = json!({"text":token["display_text"],"start":start,"end":end});
        let fallback_words = [&fallback];
        let pitch_words = existing.get(id).map_or(&fallback_words[..], Vec::as_slice);
        let source_start = pitch_words
            .iter()
            .filter_map(|word| word["start"].as_f64())
            .reduce(f64::min)
            .unwrap_or(start);
        let source_end = pitch_words
            .iter()
            .filter_map(|word| word["end"].as_f64())
            .reduce(f64::max)
            .unwrap_or(end);
        let source_duration = source_end - source_start;
        let target_duration = end - start;
        for (index, word) in pitch_words.iter().enumerate() {
            let (projected_start, projected_end) = if source_duration > 0.0 {
                (
                    start
                        + (word["start"].as_f64().unwrap_or(source_start) - source_start)
                            * target_duration
                            / source_duration,
                    start
                        + (word["end"].as_f64().unwrap_or(source_end) - source_start)
                            * target_duration
                            / source_duration,
                )
            } else {
                let slot = target_duration / pitch_words.len() as f64;
                (
                    start + index as f64 * slot,
                    start + (index + 1) as f64 * slot,
                )
            };
            projected.push(json!({
                "text": if pitch_words.len() == 1 { &token["display_text"] } else { &word["text"] },
                "start":projected_start, "end":projected_start.max(projected_end.min(end)),
                "pitch_hz":word["pitch_hz"].as_f64().unwrap_or_default(),
                "midi_note":word["midi_note"].as_i64().unwrap_or_default(),
                "confidence":alignment["confidence"].as_f64().unwrap_or_default(),
                "pitch_contour":array(&word["pitch_contour"]), "pitch_candidates":[],
                "continuation_of_previous":word["continuation_of_previous"].as_bool().unwrap_or_default(),
                "token_id":id, "alignment_status":alignment["status"], "alignment_run_id":alignment["alignment_run_id"],
            }));
        }
    }
    projected
}

pub(crate) fn line_breaks(tokens: &[Value], words: &[Value]) -> Vec<usize> {
    let lines: HashMap<_, _> = tokens
        .iter()
        .map(|token| (field_text(token, "id"), &token["line_index"]))
        .collect();
    words
        .windows(2)
        .enumerate()
        .filter_map(|(index, pair)| {
            (lines.get(field_text(&pair[0], "token_id"))
                != lines.get(field_text(&pair[1], "token_id")))
            .then_some(index + 1)
        })
        .collect()
}

pub(crate) fn apply_overrides(phrases: &mut [Phrase], overrides: &Value) -> Result<(), ApiError> {
    let Some(overrides) = overrides.as_object() else {
        return Ok(());
    };
    for (key, values) in overrides {
        let mut parts = key.split(':');
        let kind = parts.next();
        let phrase = parts
            .next()
            .and_then(|part| part.parse::<usize>().ok())
            .and_then(|index| phrases.get_mut(index));
        let Some(phrase) = phrase else {
            continue;
        };
        match (kind, parts.next(), parts.next()) {
            (Some("phrase"), None, None) => {
                if let Some(end) = values["end_beat"].as_i64() {
                    phrase.end_beat = end;
                }
            }
            (Some("note"), Some(index), None) => {
                let Some(note) = index
                    .parse::<usize>()
                    .ok()
                    .and_then(|index| phrase.notes.get_mut(index))
                else {
                    continue;
                };
                let mut new_note = json!(note);
                for field in ["note_type", "start_beat", "duration_beats", "pitch", "text"] {
                    if let Some(value) = values.get(field) {
                        new_note[field] = value.clone();
                    }
                }
                *note = serde_json::from_value(new_note)
                    .map_err(|error| ApiError::internal(error.to_string()))?;
            }
            _ => {}
        }
    }
    Ok(())
}

pub(crate) fn build_variant(
    job: &Value,
    revision: &Value,
    run: &Value,
    alignments: &[Value],
    canonical_words: &[Value],
    pitch_words: &[Value],
    line_breaks: &[usize],
    use_overrides: bool,
) -> Result<Value, ApiError> {
    let bpm = job["real_bpm"].as_f64().unwrap_or_default();
    let words = serde_json::from_value(json!(pitch_words))
        .map_err(|error| ApiError::internal(error.to_string()))?;
    let (mut phrases, gap_ms) =
        chart_output::phrases(words, bpm, 0.8, line_breaks).map_err(ApiError::internal)?;
    if use_overrides {
        apply_overrides(&mut phrases, &job["result"]["note_overrides"])?;
    }
    let mut result = job["result"].clone();
    result["metadata"] = job["metadata"].clone();
    result["metadata"]["bpm"] = json!(bpm);
    result["metadata"]["gap_ms"] = json!(gap_ms);
    result["words"] = json!(canonical_words);
    result["pitch_words"] = json!(pitch_words);
    result["phrases"] = json!(phrases);
    result["ultrastar_txt"] = json!(render_result(&result, &phrases)?);
    result["word_alignments"] = json!(alignments);
    result["alignment_run"] = run.clone();
    result["lyric_revision"] = revision.clone();
    Ok(result)
}

pub(crate) fn as_variant(result: &Value) -> Value {
    json!({
        "words":result["words"],"pitch_words":result["pitch_words"],"phrases":result["phrases"],
        "ultrastar_txt":result["ultrastar_txt"],"gap_ms":result["metadata"]["gap_ms"],
        "word_alignments":result["word_alignments"],"alignment_run":result["alignment_run"],
    })
}

fn duration(word: &Value) -> f64 {
    let start = word["start"].as_f64().unwrap_or_default();
    let end = word["end"].as_f64().unwrap_or_default();
    (end - start).max(0.0)
}

fn midi(word: &Value) -> i64 {
    word["midi_note"].as_i64().unwrap_or_default()
}

fn pitch_by_token(variant: &Value) -> HashMap<String, i64> {
    let pitch = array(&variant["pitch_words"]);
    let words = if pitch.is_empty() {
        array(&variant["words"])
    } else {
        pitch
    };
    let mut selected = HashMap::<String, &Value>::new();
    for (index, word) in words.iter().enumerate() {
        let token = field_text(word, "token_id");
        let key = if token.is_empty() {
            format!("index:{index}")
        } else {
            token.to_owned()
        };
        selected
            .entry(key)
            .and_modify(|old| {
                let voiced = midi(word) > 0;
                let old_voiced = midi(old) > 0;
                if (voiced && !old_voiced)
                    || (voiced == old_voiced && duration(word) > duration(old))
                {
                    *old = word;
                }
            })
            .or_insert(word);
    }
    selected
        .into_iter()
        .map(|(key, word)| (key, midi(word)))
        .collect()
}

pub(crate) fn compare(job: &Value, sofa: &Value, legacy: &Value) -> Value {
    let sofa_words = array(&sofa["words"]);
    let legacy_words = array(&legacy["words"]);
    let keyed = |word: &Value| !field_text(word, "token_id").is_empty();
    let left: HashMap<_, _> = sofa_words
        .iter()
        .filter(|word| keyed(word))
        .map(|word| (field_text(word, "token_id"), word))
        .collect();
    let right: HashMap<_, _> = legacy_words
        .iter()
        .filter(|word| keyed(word))
        .map(|word| (field_text(word, "token_id"), word))
        .collect();
    let mut seen = HashSet::new();
    let mut pairs: Vec<_> = sofa_words
        .iter()
        .filter_map(|word| {
            let id = field_text(word, "token_id");
            if !seen.insert(id) {
                return None;
            }
            Some((*left.get(id)?, *right.get(id)?))
        })
        .collect();
    if pairs.is_empty() {
        pairs.extend(sofa_words.iter().zip(legacy_words));
    }
    let bpm = job["metadata"]["bpm"]
        .as_f64()
        .filter(|value| *value != 0.0)
        .or_else(|| job["real_bpm"].as_f64())
        .unwrap_or_default()
        .max(1.0);
    let one_beat = 60.0 / (bpm * 16.0);
    let sofa_pitch = pitch_by_token(sofa);
    let legacy_pitch = pitch_by_token(legacy);
    let mut timing = Vec::new();
    let mut error = Vec::new();
    let mut pitch_matches = 0;
    let mut pitch_comparisons = 0;
    for (left, right) in &pairs {
        let Some((ls, le)) = span(left) else {
            continue;
        };
        let Some((rs, re)) = span(right) else {
            continue;
        };
        let delta = (ls - rs).abs().max((le - re).abs());
        error.push(delta);
        timing.push((1.0 - delta / one_beat.max(1e-6)).max(0.0));
        let left_pitch = sofa_pitch
            .get(field_text(left, "token_id"))
            .copied()
            .unwrap_or_else(|| midi(left));
        let right_pitch = legacy_pitch
            .get(field_text(right, "token_id"))
            .copied()
            .unwrap_or_else(|| midi(right));
        if left_pitch > 0 && right_pitch > 0 {
            pitch_comparisons += 1;
            pitch_matches += i32::from(left_pitch == right_pitch);
        }
    }
    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len().max(1) as f64;
    let timing = mean(&timing);
    let pitch = f64::from(pitch_matches) / f64::from(pitch_comparisons.max(1));
    let confidence = 0.65 * timing + 0.35 * pitch;
    let round = |value: f64, precision: usize| {
        format!("{value:.precision$}")
            .parse::<f64>()
            .unwrap_or(value)
    };
    let updated_at = time::OffsetDateTime::now_utc()
        .replace_nanosecond(0)
        .ok()
        .and_then(|instant| {
            instant
                .format(&time::format_description::well_known::Rfc3339)
                .ok()
        })
        .unwrap_or_default();
    json!({
        "status":"completed","sofa_available":true,"legacy_available":true,"compared_word_count":pairs.len(),
        "timing_agreement":round(timing,4),"pitch_agreement":round(pitch,4),"pitch_exact_fraction":round(pitch,4),
        "timing_mean_abs_error_ms":round(mean(&error)*1000.0,2),"one_chart_beat_ms":round(one_beat*1000.0,2),
        "confidence":round(confidence,4),"preferred_mode":if confidence >= 0.5 {"sofa"} else {"legacy"},
        "updated_at":updated_at,
    })
}

pub(crate) fn apply_canonical(
    job: &mut Value,
    revision: &Value,
    run: &Value,
    alignments: &[Value],
    canonical_words: &[Value],
    pitch_words: &[Value],
    breaks: &[usize],
) -> Result<(), ApiError> {
    if job["result"].is_null() {
        return Ok(());
    }
    let mut result = build_variant(
        job,
        revision,
        run,
        alignments,
        canonical_words,
        pitch_words,
        breaks,
        true,
    )?;
    let canonical = as_variant(&result);
    result["alignment_results"] = json!({"canonical":canonical,"sofa":canonical});
    result["current_alignment_mode"] = json!("canonical");
    job["metadata"] = result["metadata"].clone();
    job["result"] = result;
    Ok(())
}

pub(crate) fn refresh(job: &mut Value) -> Result<bool, ApiError> {
    let previous = &job["result"];
    let revision = &previous["lyric_revision"];
    let run = &previous["alignment_run"];
    if revision.is_null() || run.is_null() {
        return Ok(false);
    }
    let tokens = array(&revision["tokens"]);
    let alignments = array(&previous["word_alignments"]);
    let canonical = words(tokens, alignments, &[]);
    let pitch = array(&previous["pitch_words"]);
    let seed = if pitch.is_empty() {
        array(&previous["words"])
    } else {
        pitch
    };
    let sources = source_spans(alignments, seed);
    let locked_sources = if sources.is_empty() {
        alignments.to_vec()
    } else {
        pitch_locks(&sources, alignments)
    };
    let pitch_words = words(tokens, &locked_sources, seed);
    let breaks = line_breaks(tokens, &pitch_words);
    let mut result = build_variant(
        job,
        revision,
        run,
        alignments,
        &canonical,
        &pitch_words,
        &breaks,
        true,
    )?;
    let mode = field_text(previous, "current_alignment_mode");
    let variant = as_variant(&result);
    result["alignment_results"] = json!({"canonical":variant,"sofa":variant});
    result["current_alignment_mode"] = json!("canonical");
    let variants = &previous["alignment_results"];
    match mode {
        "legacy" if !variants["sofa"].is_null() => {
            result["alignment_results"]["legacy"] = variant.clone();
            result["alignment_results"]["sofa"] = variants["sofa"].clone();
            result["alignment_comparison"] = compare(job, &variants["sofa"], &variant);
            result["current_alignment_mode"] = json!("legacy");
        }
        "sofa" if !variants["legacy"].is_null() => {
            result["alignment_results"]["legacy"] = variants["legacy"].clone();
            result["alignment_comparison"] = compare(job, &variant, &variants["legacy"]);
            result["current_alignment_mode"] = json!("sofa");
        }
        _ => {}
    }
    job["metadata"] = result["metadata"].clone();
    job["result"] = result;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn applies_retained_companion_boundaries_to_pitch_source() {
        let source = json!({
            "token_id":"token", "start":0.8, "end":1.0,
            "start_locked":false,"end_locked":false
        });
        let start_lock = json!({
            "token_id":"token","start":1.1,"end":1.4,"start_locked":true,"end_locked":false,
            "evidence":{"reasons":["previous end retained to keep the locked start paired"]}
        });
        let end_lock = json!({
            "token_id":"token","start":0.3,"end":0.6,"start_locked":false,"end_locked":true,
            "evidence":{"reasons":["previous start retained to keep the locked end paired"]}
        });
        assert_eq!(
            span(&pitch_locks(std::slice::from_ref(&source), &[start_lock])[0]),
            Some((1.1, 1.4))
        );
        assert_eq!(
            span(&pitch_locks(&[source], &[end_lock])[0]),
            Some((0.3, 0.6))
        );
    }

    #[test]
    fn editor_rebuild_preserves_the_separate_pitch_clock() {
        let metadata = json!({"title":"Song","artist":"Artist","genre":"","year":"","language":"en","bpm":120,"gap_ms":0});
        let mut job = json!({
            "metadata":metadata,"real_bpm":120,
            "result":{
                "metadata":metadata,"audio_file":"audio.wav","lyric_revision":{
                    "tokens":[{"id":"a","display_text":"hello","line_index":0},{"id":"b","display_text":"world","line_index":1}]
                },
                "alignment_run":{"id":"run","status":"completed"},
                "word_alignments":[
                    {"token_id":"a","start":0.0,"end":2.0,"status":"estimated","alignment_run_id":"run"},
                    {"token_id":"b","start":2.0,"end":4.0,"status":"estimated","alignment_run_id":"run"}
                ],
                "pitch_words":[
                    {"token_id":"a","text":"hello","start":0.0,"end":0.5,"midi_note":69},
                    {"token_id":"b","text":"world","start":1.0,"end":1.5,"midi_note":69}
                ]
            }
        });
        assert!(refresh(&mut job).ok().unwrap());
        let timing = array(&job["result"]["words"])
            .iter()
            .filter_map(span)
            .collect::<Vec<_>>();
        let pitch = array(&job["result"]["pitch_words"])
            .iter()
            .filter_map(span)
            .collect::<Vec<_>>();
        assert_eq!(timing, [(0.0, 2.0), (2.0, 4.0)]);
        assert_eq!(pitch, [(0.0, 0.5), (1.0, 1.5)]);
    }
}
