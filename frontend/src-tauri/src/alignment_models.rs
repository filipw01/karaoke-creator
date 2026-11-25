//! Native CTC inference for the stock MMS_FA and ForceAlign acoustic models.

use std::{collections::HashMap, path::PathBuf};

use ort::{session::Session, value::TensorRef};
use serde::Deserialize;
use serde_json::{Value, json};
use unicode_general_category::{GeneralCategory, get_general_category};
use unicode_normalization::UnicodeNormalization;

use crate::native_pitch::{decode_mono, pcm_samples};

const RATE: u32 = 16_000;
const WINDOW: usize = RATE as usize * 30;

#[derive(Deserialize)]
struct ModelVocab {
    blank: usize,
    labels: Vec<String>,
}

#[derive(Deserialize)]
struct VocabFile {
    mms_fa: ModelVocab,
    forcealign_en: ModelVocab,
}

fn paths() -> Result<(PathBuf, PathBuf, PathBuf), String> {
    let root = std::env::var_os("ALIGNMENT_MODELS_DIR")
        .map(PathBuf::from)
        .or_else(|| {
            let current = std::env::current_dir().ok()?;
            current.ancestors().find_map(|directory| {
                [
                    directory.join("backend/models/alignment"),
                    directory.join("backend/benchmarks/local-four/rust-onnx"),
                    directory.join("models/alignment"),
                ]
                .into_iter()
                .find(|candidate| candidate.join("alignment_vocab.json").is_file())
            })
        });
    let required = |name: &str, file: &str| {
        std::env::var_os(name)
            .map(PathBuf::from)
            .or_else(|| root.as_ref().map(|path| path.join(file)))
            .ok_or_else(|| format!("Set {name} or ALIGNMENT_MODELS_DIR to enable native alignment"))
    };
    Ok((
        required("ALIGNMENT_MMS_ONNX_PATH", "mms_fa.onnx")?,
        required("ALIGNMENT_FORCEALIGN_ONNX_PATH", "forcealign_en.onnx")?,
        required("ALIGNMENT_VOCAB_PATH", "alignment_vocab.json")?,
    ))
}

fn tokens_for_word(word: &str, vocab: &ModelVocab, model: &str) -> Vec<usize> {
    let mut chars = Vec::new();
    for ch in word.chars() {
        let decomposed = if ch == 'ł' || ch == 'Ł' {
            "l".to_owned()
        } else {
            ch.to_lowercase()
                .collect::<String>()
                .nfd()
                .filter(|c| {
                    !matches!(
                        get_general_category(*c),
                        GeneralCategory::NonspacingMark
                            | GeneralCategory::SpacingMark
                            | GeneralCategory::EnclosingMark
                    )
                })
                .collect()
        };
        for normalized in decomposed.chars() {
            let symbol = match model {
                "forcealign" if normalized.is_ascii_alphabetic() => {
                    normalized.to_ascii_uppercase().to_string()
                }
                "forcealign" => continue,
                "mms" if normalized.is_ascii_alphanumeric() || normalized == '\'' => {
                    normalized.to_string()
                }
                _ => continue,
            };
            let Some(index) = vocab
                .labels
                .iter()
                .position(|label| label.eq_ignore_ascii_case(&symbol))
            else {
                continue;
            };
            chars.push(index);
        }
    }
    chars
}

fn ctc_spans(
    logp: &[f32],
    frames: usize,
    labels: usize,
    blank: usize,
    target: &[usize],
) -> Vec<(usize, usize, f32)> {
    if frames == 0 || target.is_empty() || labels == 0 {
        return Vec::new();
    }
    let mut states = Vec::with_capacity(target.len() * 2 + 1);
    for &label in target {
        states.push(blank);
        states.push(label);
    }
    states.push(blank);
    let n = states.len();
    let mut prev = vec![f32::NEG_INFINITY; n];
    prev[0] = logp[blank];
    if n > 1 {
        prev[1] = logp[target[0]];
    }
    let mut backs = vec![0u8; frames * n];
    for frame in 1..frames {
        let mut next = vec![f32::NEG_INFINITY; n];
        for s in 0..n {
            let mut best = prev[s];
            let mut from = s;
            if s > 0 && prev[s - 1] > best {
                best = prev[s - 1];
                from = s - 1;
            }
            if s > 1 && states[s] != blank && states[s] != states[s - 2] && prev[s - 2] > best {
                best = prev[s - 2];
                from = s - 2;
            }
            next[s] = best + logp[frame * labels + states[s]];
            backs[frame * n + s] = (s - from) as u8;
        }
        prev = next;
    }
    let mut state = if n > 1 && prev[n - 2] > prev[n - 1] {
        n - 2
    } else {
        n - 1
    };
    let mut aligned = vec![0usize; frames];
    for frame in (0..frames).rev() {
        aligned[frame] = state;
        if frame > 0 {
            state = state.saturating_sub(backs[frame * n + state] as usize);
        }
    }
    let mut output = Vec::with_capacity(target.len());
    for (index, _) in target.iter().enumerate() {
        let state = 2 * index + 1;
        let hit: Vec<_> = aligned
            .iter()
            .enumerate()
            .filter_map(|(f, &s)| (s == state).then_some(f))
            .collect();
        if let (Some(&first), Some(&last)) = (hit.first(), hit.last()) {
            let confidence = (first..=last)
                .map(|f| logp[f * labels + target[index]].exp())
                .sum::<f32>()
                / hit.len() as f32;
            output.push((first, last + 1, confidence.clamp(0.0, 1.0)));
        } else {
            output.push((0, 0, 0.0));
        }
    }
    output
}

fn run(model: &str, audio: &std::path::Path, words: &[String]) -> Result<Vec<Value>, String> {
    let (mms_path, force_path, vocab_path) = paths()?;
    if let Some(runtime) = std::env::var_os("ORT_DYLIB_PATH") {
        ort::init_from(runtime).map_err(|e| e.to_string())?.commit();
    }
    let vocab_file: VocabFile =
        serde_json::from_slice(&std::fs::read(vocab_path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let vocab = if model == "mms" {
        vocab_file.mms_fa
    } else {
        vocab_file.forcealign_en
    };
    let label_to_id: HashMap<String, usize> = vocab
        .labels
        .iter()
        .enumerate()
        .map(|(i, x)| (x.to_uppercase(), i))
        .collect();
    let path = if model == "mms" { mms_path } else { force_path };
    let bytes = decode_mono(audio, RATE, 0.0).map_err(|e| e.to_string())?;
    let samples = pcm_samples(&bytes).map_err(|e| e.to_string())?;
    if samples.is_empty() {
        return Err("No audio samples for alignment".into());
    }
    let mut session = Session::builder()
        .map_err(|e| e.to_string())?
        .with_intra_threads(std::thread::available_parallelism().map_or(1, |x| x.get().min(4)))
        .map_err(|e| e.to_string())?
        .with_inter_threads(1)
        .map_err(|e| e.to_string())?
        .commit_from_file(path)
        .map_err(|e| e.to_string())?;
    let step = WINDOW - RATE as usize * 2;
    let mut emissions = Vec::<Vec<f32>>::new();
    let mut frame_times = Vec::<f64>::new();
    for start in (0..samples.len()).step_by(step) {
        let mut chunk = vec![0.0f32; WINDOW];
        let count = (samples.len() - start).min(WINDOW);
        chunk[..count].copy_from_slice(&samples[start..start + count]);
        let tensor = TensorRef::from_array_view(([1usize, WINDOW], chunk.as_slice()))
            .map_err(|e| e.to_string())?;
        let out = session
            .run(ort::inputs![tensor])
            .map_err(|e| e.to_string())?;
        let (shape, data) = out[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| e.to_string())?;
        let dims = shape.as_ref();
        if dims.len() != 3 || dims[0] != 1 || dims[2] != vocab.labels.len() as i64 {
            return Err("Alignment ONNX output has unexpected shape".into());
        }
        let frames = dims[1] as usize;
        let trim = if start == 0 { 0 } else { frames / 30 };
        let valid_frames = frames * count / WINDOW;
        let last = if count < WINDOW {
            valid_frames
        } else {
            frames.saturating_sub(frames / 30)
        };
        let frame_seconds = WINDOW as f64 / frames as f64 / f64::from(RATE);
        for (frame, row) in data
            .chunks_exact(vocab.labels.len())
            .enumerate()
            .take(last)
            .skip(trim)
        {
            emissions.push(row.to_vec());
            frame_times.push(start as f64 / f64::from(RATE) + frame as f64 * frame_seconds);
        }
        if count < WINDOW {
            break;
        }
    }
    let frame_count = emissions.len();
    let mut result = Vec::new();
    let mut sequence = Vec::new();
    let mut owners = Vec::new();
    for (word_index, word) in words.iter().enumerate() {
        for ch in tokens_for_word(word, &vocab, model) {
            sequence.push(ch);
            owners.push(word_index);
        }
        if model == "forcealign" {
            if let Some(&bar) = label_to_id.get("|") {
                sequence.push(bar);
                owners.push(usize::MAX);
            }
        }
    }
    if sequence.is_empty() {
        return Err("Lyrics contain no symbols supported by this model".into());
    }
    let flat: Vec<f32> = emissions[..frame_count]
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect();
    let spans = ctc_spans(
        &flat,
        frame_count,
        vocab.labels.len(),
        vocab.blank,
        &sequence,
    );
    for (word_index, word) in words.iter().enumerate() {
        let word_spans: Vec<_> = spans
            .iter()
            .zip(owners.iter())
            .filter_map(|(span, &owner)| (owner == word_index && span.1 > span.0).then_some(*span))
            .collect();
        if let (Some(first), Some(last)) = (word_spans.first(), word_spans.last()) {
            let start = frame_times[first.0];
            let frame_step = frame_times
                .get(last.1)
                .zip(frame_times.get(last.1 - 1))
                .map_or_else(
                    || {
                        last.1
                            .checked_sub(2)
                            .and_then(|index| frame_times.get(index))
                            .map_or(0.02, |previous| frame_times[last.1 - 1] - previous)
                    },
                    |(next, previous)| next - previous,
                );
            let end = frame_times
                .get(last.1)
                .copied()
                .unwrap_or(frame_times[last.1 - 1] + frame_step);
            if end > start {
                result.push(json!({"text":word,"start":start,"end":end,"confidence":word_spans.iter().map(|x| x.2).sum::<f32>() / word_spans.len() as f32,"model":if model=="mms" {"mms_fa"} else {"forcealign_en"},"audio_source":"vocal_stem","evidence_type":"forced","sequence_index":result.len()}));
            }
        }
    }
    Ok(result)
}

pub(crate) fn align_forcealign(
    audio: &std::path::Path,
    words: &[String],
) -> Result<Vec<Value>, String> {
    run("forcealign", audio, words)
}
pub(crate) fn align_mms(audio: &std::path::Path, words: &[String]) -> Result<Vec<Value>, String> {
    run("mms", audio, words)
}
