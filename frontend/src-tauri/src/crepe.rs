use std::{io, path::Path};

use ort::{session::Session, value::TensorRef};
use serde::Serialize;

use crate::native_pitch::{centered_rms, decode_mono, pcm_samples};

const RATE: u32 = 16_000;
const HOP: usize = 320;
const WINDOW: usize = 1024;
const BINS: usize = 360;
const MIN_CENTS: f64 = 1997.3794084376191;
const CENTS_PER_BIN: f64 = 20.0;
const TRANSITION_WIDTH: usize = 12;

#[derive(Default, Serialize)]
pub(crate) struct Track {
    times: Vec<f64>,
    frequencies: Vec<f64>,
    confidences: Vec<f64>,
    energies: Vec<f64>,
}

fn frequency(bin: usize) -> f64 {
    10.0 * ((CENTS_PER_BIN * bin as f64 + MIN_CENTS) / 1200.0).exp2()
}

fn pitch_bin(frequency: f64) -> f64 {
    (1200.0 * (frequency / 10.0).log2() - MIN_CENTS) / CENTS_PER_BIN
}

fn normalize(samples: &[f32], center: usize, window: &mut [f32]) {
    let start = center.saturating_sub(WINDOW / 2);
    let target = (WINDOW / 2).saturating_sub(center);
    let length = (WINDOW - target).min(samples.len() - start);
    window.fill(0.0);
    window[target..target + length].copy_from_slice(&samples[start..start + length]);
    let mean = (window.iter().map(|&sample| f64::from(sample)).sum::<f64>() / WINDOW as f64) as f32;
    for value in window.iter_mut() {
        *value -= mean;
    }
    // The model was trained on windows normalized with sample variance.
    let power = window
        .iter()
        .map(|&value| f64::from(value).powi(2))
        .sum::<f64>()
        / (WINDOW - 1) as f64;
    let deviation = (power.sqrt() as f32).max(1e-10);
    for value in window {
        *value /= deviation;
    }
}

fn decode(outputs: &[f32], low: usize, high: usize) -> Vec<usize> {
    let frames = outputs.len() / BINS;
    let width = high - low;
    let mut choices = vec![0; frames * width];
    let mut previous: Vec<_> = outputs[low..high].iter().copied().map(f64::from).collect();
    let mut current = vec![0.0; width];
    // Normalize the triangular transition from each source across all bins,
    // including the edge bins outside the requested singing range.
    let totals: Vec<_> = (low..high)
        .map(|source| {
            let start = source.saturating_sub(TRANSITION_WIDTH - 1);
            let end = (source + TRANSITION_WIDTH).min(BINS);
            ((start..end)
                .map(|target| TRANSITION_WIDTH - target.abs_diff(source))
                .sum::<usize>() as f64)
                .ln()
        })
        .collect();
    let transition: Vec<_> = (1..=TRANSITION_WIDTH)
        .rev()
        .map(|weight| (weight as f64).ln())
        .collect();
    for frame in 1..frames {
        for target in 0..width {
            let start = target.saturating_sub(TRANSITION_WIDTH - 1);
            let end = (target + TRANSITION_WIDTH).min(width);
            let mut best = f64::NEG_INFINITY;
            let mut source = start;
            for candidate in start..end {
                let score = previous[candidate] - totals[candidate]
                    + transition[target.abs_diff(candidate)];
                if score > best {
                    best = score;
                    source = candidate;
                }
            }
            current[target] = best + f64::from(outputs[frame * BINS + low + target]);
            choices[frame * width + target] = source;
        }
        std::mem::swap(&mut current, &mut previous);
    }
    let mut chosen = (0..width)
        .max_by(|&a, &b| previous[a].total_cmp(&previous[b]).then_with(|| b.cmp(&a)))
        .unwrap_or(0);
    let mut result = vec![0; frames];
    for frame in (0..frames).rev() {
        result[frame] = low + chosen;
        chosen = choices[frame * width + chosen];
    }
    result
}

fn setting<T: std::str::FromStr>(name: &str) -> Option<T> {
    std::env::var(name).ok()?.trim().parse().ok()
}

pub(crate) fn detect(source: &Path) -> Result<Track, Box<dyn std::error::Error>> {
    let model = std::env::var("TORCHCREPE_MODEL").unwrap_or_default();
    let capacity = if model.trim().eq_ignore_ascii_case("full") {
        "full"
    } else {
        "tiny"
    };
    let weights = std::env::var_os("TORCHCREPE_ONNX_PATH").map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("TORCHCREPE_ONNX_DIR").map(|dir| std::path::PathBuf::from(dir).join(format!("{capacity}.onnx"))))
        .ok_or("Set TORCHCREPE_ONNX_PATH to a float32 CREPE ONNX model or TORCHCREPE_ONNX_DIR to its directory")?;
    let batch_size = setting("TORCHCREPE_BATCH_SIZE").unwrap_or(512_i64).max(32) as usize;
    let min_frequency = setting::<f64>("TORCHCREPE_MIN_FREQUENCY")
        .filter(|n| n.is_finite())
        .unwrap_or(55.0)
        .clamp(20.0, 1000.0);
    let max_frequency = setting::<f64>("TORCHCREPE_MAX_FREQUENCY")
        .filter(|n| n.is_finite())
        .unwrap_or(1760.0)
        .clamp(min_frequency + 1.0, 4000.0);
    let minimum_periodicity = setting::<f64>("TORCHCREPE_MIN_PERIODICITY")
        .filter(|n| n.is_finite())
        .unwrap_or(0.22)
        .clamp(0.0, 1.0);
    let low = (pitch_bin(min_frequency).floor() as usize).min(BINS - 1);
    let high = (pitch_bin(max_frequency).ceil() as usize).clamp(low + 1, BINS);

    let samples = pcm_samples(&decode_mono(source, RATE, 0.0)?)?;
    if samples.is_empty() {
        return Ok(Track::default());
    }
    if let Some(runtime) = std::env::var_os("ORT_DYLIB_PATH") {
        ort::init_from(runtime)?.commit();
    }
    let default_threads =
        std::thread::available_parallelism().map_or(1, |threads| threads.get().min(4)) as i64;
    let mut session = Session::builder()?
        .with_intra_threads(
            setting("TORCHCREPE_THREADS")
                .unwrap_or(default_threads)
                .max(1) as usize,
        )?
        .with_inter_threads(1)?
        .commit_from_file(weights)?;

    let mut track = Track {
        energies: centered_rms(&samples, 2048, HOP),
        ..Track::default()
    };
    let frames = track.energies.len();
    let mut windows = Vec::new();
    for first in (0..frames).step_by(batch_size) {
        let count = (frames - first).min(batch_size);
        windows.resize(count * WINDOW, 0.0);
        for (offset, window) in windows.chunks_exact_mut(WINDOW).enumerate() {
            normalize(&samples, (first + offset) * HOP, window);
        }
        let tensor = TensorRef::from_array_view(([count, WINDOW], windows.as_slice()))?;
        let outputs = session.run(ort::inputs![tensor])?;
        let (shape, values) = outputs[0].try_extract_tensor::<f32>()?;
        if shape.as_ref() != [count as i64, BINS as i64]
            || values.iter().any(|value| !value.is_finite())
        {
            return Err("CREPE ONNX must produce a finite [frames, 360] activation matrix".into());
        }
        for (offset, bin) in decode(values, low, high).into_iter().enumerate() {
            let hz = frequency(bin);
            let confidence = f64::from(values[offset * BINS + bin]);
            track
                .times
                .push((first + offset) as f64 * (HOP as f64 / f64::from(RATE)));
            track.confidences.push(confidence);
            track.frequencies.push(
                if confidence >= minimum_periodicity
                    && (min_frequency..=max_frequency).contains(&hz)
                {
                    hz
                } else {
                    0.0
                },
            );
        }
    }
    Ok(track)
}

pub(crate) fn run(source: &Path) -> Result<(), Box<dyn std::error::Error>> {
    serde_json::to_writer(io::stdout().lock(), &detect(source)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sustained_pitch_survives_an_isolated_off_range_activation() {
        let mut values = vec![0.0; 4 * BINS];
        for row in values.chunks_exact_mut(BINS) {
            row[30] = 0.9;
        }
        values[BINS + 45] = 1.0;
        assert_eq!(decode(&values, 25, 50), [30; 4]);
    }
}
