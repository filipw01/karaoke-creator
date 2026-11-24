use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use tokio::task::spawn_blocking;

use crate::{
    api::Api,
    managed_api::field_text,
    managed_editor::{array, projection},
    media, pitch_detection, whisper_draft, worker,
};

pub(super) struct Inputs {
    pub(super) vocals: PathBuf,
    pub(super) original: PathBuf,
    pub(super) transcription: PathBuf,
    pub(super) detection: pitch_detection::Detection,
    pub(super) duration: f64,
}

pub(crate) fn audio_duration(path: &Path) -> f64 {
    hound::WavReader::open(path).map_or(0.0, |wav| {
        f64::from(wav.duration()) / f64::from(wav.spec().sample_rate)
    })
}

pub(super) async fn load(
    api: &Api,
    job: &Value,
    normalize: Option<PathBuf>,
) -> Result<Inputs, String> {
    let backend = api.backend.clone();
    let vocals = PathBuf::from(field_text(job, "vocals_path"));
    let original = PathBuf::from(field_text(job, "audio_path"));
    spawn_blocking(move || {
        let transcription = if let Some(path) = normalize {
            media::normalize_audio(&vocals, &path)?;
            path
        } else {
            vocals.clone()
        };
        let detection = pitch_detection::detect(&backend, &vocals, Some(&original))?;
        let duration = audio_duration(&original);
        Ok(Inputs {
            vocals,
            original,
            transcription,
            detection,
            duration,
        })
    })
    .await
    .map_err(|error| error.to_string())?
}

fn activity(detection: &pitch_detection::Detection) -> (Vec<f64>, f64, Vec<(f64, f64)>) {
    let activity: Vec<_> = detection
        .frequencies
        .iter()
        .zip(&detection.energies)
        .map(|(frequency, energy)| {
            if frequency.is_finite() && *frequency > 0.0 && energy.is_finite() {
                energy.max(0.0)
            } else {
                0.0
            }
        })
        .collect();
    let mut positives: Vec<_> = activity
        .iter()
        .copied()
        .filter(|value| *value > 0.0)
        .collect();
    positives.sort_unstable_by(f64::total_cmp);
    let maximum = std::env::var("ALIGNMENT_ACTIVITY_MAX_THRESHOLD")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(0.06)
        .max(0.005);
    let threshold = if positives.is_empty() {
        0.005
    } else {
        let position = (positives.len() - 1) as f64 * 0.25;
        let lower = positives[position.floor() as usize];
        let upper = positives[position.ceil() as usize];
        let weight = position.fract();
        let quartile = if weight >= 0.5 {
            upper - (upper - lower) * (1.0 - weight)
        } else {
            lower + (upper - lower) * weight
        };
        (quartile * 1.25).clamp(0.005, maximum)
    };
    let mut steps: Vec<_> = detection
        .times
        .windows(2)
        .filter_map(|times| {
            let step = times[1] - times[0];
            (step.is_finite() && step > 0.0).then_some(step)
        })
        .collect();
    steps.sort_unstable_by(f64::total_cmp);
    let step = if steps.is_empty() {
        0.05
    } else {
        ((steps[(steps.len() - 1) / 2] + steps[steps.len() / 2]) / 2.0).clamp(0.01, 0.2)
    };
    let mut voiced = detection
        .times
        .iter()
        .zip(&activity)
        .filter_map(|(&time, &value)| (time.is_finite() && value >= threshold).then_some(time));
    let mut spans = Vec::new();
    if let Some(mut start) = voiced.next() {
        let mut previous = start;
        for current in voiced {
            if current - previous > 0.18 {
                spans.push((start, previous + step));
                start = current;
            }
            previous = current;
        }
        spans.push((start, previous + step));
    }
    (activity, threshold, spans)
}

pub(super) fn resolve_backend(language: &str, requested: &str) -> String {
    let primary = std::env::var("ALIGNMENT_PRIMARY_BACKEND").unwrap_or_default();
    let non_english = std::env::var("SOFA_ALLOW_NON_ENGLISH")
        .ok()
        .is_some_and(|value| {
            !matches!(
                value.trim().to_lowercase().as_str(),
                "0" | "false" | "off" | "no" | "disabled"
            )
        });
    if requested == "sofa"
        && (matches!(
            primary.trim().to_lowercase().as_str(),
            "legacy" | "current" | "asr" | "off"
        ) || (language != "en" && !non_english))
    {
        "legacy".into()
    } else {
        requested.into()
    }
}

pub(super) fn strategy(backend: &str) -> &'static str {
    if backend == "legacy" {
        "legacy-global-match-and-candidate-score"
    } else {
        "sofa-primary-global-match-and-candidate-score"
    }
}

fn worker_request(
    job: &Value,
    revision: &Value,
    run: &Value,
    inputs: &Inputs,
    transcript: &Path,
    lane: &str,
    spans: &[(f64, f64)],
) -> Value {
    json!({
        "job_id":job["id"],"alignment_run_id":run["id"],"lyric_revision_id":revision["id"],
        "language":revision["language"],"original_audio_ref":inputs.original,"vocal_stem_audio_ref":transcript,
        "alignment_backend":lane,"tokens":array(&revision["tokens"]),"activity_spans":spans,
        "timing_grid_step":null,"timing_grid_offset":null,
    })
}

fn remote_url() -> Option<String> {
    std::env::var("ALIGNMENT_WORKER_URL")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(|url| url.trim().to_owned())
}

fn resolve_existing_prefix(path: &Path) -> Result<PathBuf, String> {
    let absolute = std::path::absolute(path).map_err(|error| error.to_string())?;
    let mut resolved = PathBuf::new();
    for part in absolute.components() {
        match part {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                resolved.pop();
            }
            part => {
                resolved.push(part);
                if let Ok(canonical) = dunce::canonicalize(&resolved) {
                    resolved = canonical;
                }
            }
        }
    }
    Ok(resolved)
}

async fn remote_dispatch(api: &Api, url: String, mut request: Value) -> Result<Value, String> {
    let token = std::env::var("ALIGNMENT_WORKER_TOKEN").unwrap_or_default();
    let token = token.trim().to_owned();
    if token.is_empty() {
        return Err(
            "ALIGNMENT_WORKER_TOKEN is required when ALIGNMENT_WORKER_URL is configured".into(),
        );
    }
    let root = std::env::var("ALIGNMENT_WORKER_AUDIO_ROOT").unwrap_or_default();
    if root.trim().is_empty() {
        return Err(
            "ALIGNMENT_WORKER_AUDIO_ROOT must map API audio files into the remote worker".into(),
        );
    }
    let root = resolve_existing_prefix(Path::new(root.trim()))?;
    let api_root = std::env::var_os("ALIGNMENT_API_AUDIO_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| api.backend.clone());
    let api_root = resolve_existing_prefix(&api_root)?;
    for name in ["original_audio_ref", "vocal_stem_audio_ref"] {
        let source = resolve_existing_prefix(Path::new(field_text(&request, name)))?;
        let relative = source
            .strip_prefix(&api_root)
            .map_err(|_| "Audio path is outside the configured API audio root")?;
        request[name] = json!(root.join(relative));
    }
    spawn_blocking(move || worker::dispatch(request, &url, Some(&token), 900.0))
        .await
        .map_err(|error| error.to_string())?
}

pub(super) async fn draft(
    api: &Api,
    job: &Value,
    run: &Value,
    language: &str,
    inputs: &Inputs,
) -> Result<(String, Vec<Value>), String> {
    if let Some(url) = remote_url() {
        let revision = json!({"id":format!("{}:draft-request:{}",field_text(job,"id"),uuid::Uuid::new_v4()),"language":language});
        let request = worker_request(job, &revision, run, inputs, &inputs.vocals, "sofa", &[]);
        let response = remote_dispatch(api, url, request).await?;
        let text = field_text(&response, "draft_text").trim();
        let matches = response["job_id"] == job["id"]
            && response["alignment_run_id"] == run["id"]
            && response["lyric_revision_id"] == revision["id"];
        if !matches {
            return Err("Remote lyric draft did not match its originating request".into());
        }
        if response["status"] != "completed" || text.is_empty() {
            return Err(format!(
                "Remote worker could not create an ASR lyric draft: {}",
                response["diagnostics"]["reason"]
                    .as_str()
                    .unwrap_or("no draft text returned")
            ));
        }
        Ok((text.into(), Vec::new()))
    } else {
        let source = inputs.transcription.clone();
        let language = language.to_owned();
        let words = spawn_blocking(move || whisper_draft::transcribe(&source, &language))
            .await
            .map_err(|error| error.to_string())??;
        let text = words
            .iter()
            .filter_map(|word| word["text"].as_str())
            .filter(|word| !word.trim().is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        Ok((text, words))
    }
}

pub(super) struct Aligned {
    pub(super) alignments: Vec<Value>,
    pub(super) canonical: Vec<Value>,
    pub(super) pitch: Vec<Value>,
    pub(super) breaks: Vec<usize>,
    pub(super) diagnostics: Value,
    pub(super) lane: String,
}

pub(super) struct Failure {
    pub(super) error: String,
    pub(super) diagnostics: Value,
}

impl From<String> for Failure {
    fn from(error: String) -> Self {
        Self {
            diagnostics: json!({"error":error}),
            error,
        }
    }
}

pub(super) async fn align(
    api: &Api,
    job: &Value,
    revision: &Value,
    run: &Value,
    mut inputs: Inputs,
    lane: &str,
    fallback: &[Value],
    resets: &[String],
    use_locks: bool,
) -> Result<Aligned, Failure> {
    if let Some(duration) = job["result"]["audio_duration"]
        .as_f64()
        .filter(|value| *value > 0.0)
    {
        inputs.duration = duration;
    }
    let (activity, threshold, spans) = activity(&inputs.detection);
    let request = worker_request(
        job,
        revision,
        run,
        &inputs,
        &inputs.transcription,
        lane,
        &spans,
    );
    let remote = if let Some(url) = remote_url() {
        Some(remote_dispatch(api, url, request.clone()).await?)
    } else {
        None
    };
    let alignments = if use_locks {
        array(&job["result"]["word_alignments"])
    } else {
        &[]
    };
    let response = api.worker.run(&api.backend,"managed_alignment",json!({
        "request":request,"remote_response":remote,"times":inputs.detection.times,"activity":activity,
        "activity_threshold":threshold,"audio_duration":inputs.duration,"fallback_words":fallback,
        "previous_tokens":array(&job["result"]["lyric_revision"]["tokens"]),
        "previous_alignments":alignments,"reset_token_ids":resets,
    })).await?;
    if let Some(error) = response["error"].as_str() {
        return Err(Failure {
            error: error.into(),
            diagnostics: response["diagnostics"].clone(),
        });
    }
    let alignments = array(&response["alignments"]);
    let pitch_alignments =
        projection::pitch_locks(array(&response["pitch_source_alignments"]), alignments);
    let tokens = array(&revision["tokens"]);
    let canonical = projection::words(tokens, alignments, &[]);
    let source_words = projection::words(tokens, &pitch_alignments, &[]);
    let breaks = projection::line_breaks(tokens, &canonical);
    let (pitch, breaks) =
        pitch_detection::assign(&api.backend, &inputs.detection, source_words, breaks)?;
    let pitch = serde_json::to_value(pitch).map_err(|error| error.to_string())?;
    Ok(Aligned {
        alignments: alignments.to_vec(),
        canonical,
        pitch: array(&pitch).to_vec(),
        breaks,
        diagnostics: response["diagnostics"].clone(),
        lane: field_text(&response, "alignment_backend").to_owned(),
    })
}
