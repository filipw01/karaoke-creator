use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{
    basic_pitch, crepe, melodia, native_pitch,
    note_assignment::{AssignmentRequest, segments::Word},
    worker,
};

const BASIC_PITCH_WEIGHTS: &str = "basic_pitch/saved_models/icassp_2022/nmp.onnx";

fn basic_pitch_model(backend: &Path) -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("BASIC_PITCH_MODEL_PATH").filter(|value| !value.is_empty())
    {
        let path = PathBuf::from(path);
        let path = if let Ok(relative) = path.strip_prefix("~") {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .map(|home| home.join(relative))
                .unwrap_or(path)
        } else {
            path
        };
        return path.is_file().then_some(path);
    }
    if let Some(paths) = std::env::var_os("PYTHONPATH") {
        if let Some(weights) = std::env::split_paths(&paths)
            .map(|directory| directory.join(BASIC_PITCH_WEIGHTS))
            .find(|path| path.is_file())
        {
            return Some(weights);
        }
    }
    let python = worker::python_executable(backend);
    let venv = python.parent()?.parent()?;
    let windows = venv.join("Lib/site-packages").join(BASIC_PITCH_WEIGHTS);
    if windows.is_file() {
        return Some(windows);
    }
    std::fs::read_dir(venv.join("lib"))
        .ok()?
        .flatten()
        .filter(|entry| entry.file_name().to_string_lossy().starts_with("python"))
        .map(|entry| entry.path().join("site-packages").join(BASIC_PITCH_WEIGHTS))
        .find(|path| path.is_file())
}

pub(crate) fn basic_pitch_clock(backend: &Path) -> bool {
    match detector_name().as_str() {
        "" => basic_pitch_model(backend).is_some(),
        "basic_pitch"
        | "basic-pitch"
        | "basicpitch"
        | "basic_pitch_melodia"
        | "basic-pitch-melodia"
        | "basicpitch-melodia" => true,
        _ => false,
    }
}

fn detector_name() -> String {
    std::env::var("PITCH_DETECTOR")
        .unwrap_or_default()
        .trim()
        .to_lowercase()
}

#[derive(Deserialize, Serialize)]
pub(crate) struct Detection {
    pub(crate) times: Vec<f64>,
    pub(crate) frequencies: Vec<f64>,
    pub(crate) confidences: Vec<f64>,
    pub(crate) energies: Vec<f64>,
    pub(crate) candidate_midis: Option<Vec<Vec<i32>>>,
    pub(crate) candidate_scores: Option<Vec<Vec<f64>>>,
    pub(crate) onset_scores: Option<Vec<f64>>,
}

fn convert<T: Serialize>(frames: T) -> Result<Detection, Box<dyn std::error::Error>> {
    Ok(serde_json::from_value(serde_json::to_value(frames)?)?)
}

pub(crate) fn detect(
    backend: &Path,
    vocals: &Path,
    mix: Option<&Path>,
) -> Result<Detection, String> {
    let model = basic_pitch_model(backend);
    let requested = detector_name();
    let detector = if requested.is_empty() && model.is_some() {
        "basic_pitch"
    } else {
        &requested
    };
    let basic = |cutoff| -> Result<Detection, Box<dyn std::error::Error>> {
        let model = model
            .as_ref()
            .ok_or("Basic Pitch ONNX weights are unavailable; set BASIC_PITCH_MODEL_PATH")?;
        convert(basic_pitch::detect(model, vocals, cutoff)?)
    };
    let attempt = match detector {
        "basic_pitch_melodia" | "basic-pitch-melodia" | "basicpitch-melodia" => {
            let fusion = || -> Result<Detection, Box<dyn std::error::Error>> {
                let model = model.as_ref().ok_or(
                    "Basic Pitch ONNX weights are unavailable; set BASIC_PITCH_MODEL_PATH",
                )?;
                convert(melodia::fused_track(model, vocals)?)
            };
            Some(fusion().or_else(|error| {
                log::warn!("Basic Pitch/MELODIA unavailable, trying Basic Pitch: {error}");
                basic(Some(0.0))
            }))
        }
        "basic_pitch" | "basic-pitch" | "basicpitch" => Some(basic(None)),
        "torchcrepe" | "crepe" | "torch-crepe" => Some(crepe::detect(vocals).and_then(convert)),
        "melodia" | "essentia" | "predominant_melody" => {
            Some(melodia::detect(vocals).and_then(convert))
        }
        _ => None,
    };
    if let Some(attempt) = attempt {
        match attempt {
            Ok(track) => return Ok(track),
            Err(error) => log::warn!("{detector} unavailable, using pYIN: {error}"),
        }
    }
    native_pitch::detect_pyin(vocals, mix)
        .and_then(convert)
        .map_err(|error| error.to_string())
}

pub(crate) fn assign(
    backend: &Path,
    detection: &Detection,
    words: Vec<Value>,
    breaks: Vec<usize>,
) -> Result<(Vec<Word>, Vec<usize>), String> {
    let beam = match (&detection.candidate_midis, &detection.candidate_scores) {
        (Some(midis), Some(scores)) => json!({"midis":midis,"scores":scores}),
        _ => Value::Null,
    };
    let request: AssignmentRequest = serde_json::from_value(json!({
        "words":words,"times":detection.times,"frequencies":detection.frequencies,
        "confidences":detection.confidences,"energies":detection.energies,"onsets":detection.onset_scores,
        "min_confidence":0.0,"min_volume":0.005,"basic_pitch":basic_pitch_clock(backend),
        "beam":beam,"line_breaks":breaks,
    })).map_err(|error| error.to_string())?;
    request.resolve()
}
