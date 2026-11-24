use std::{
    fs,
    path::{Component, Path, PathBuf},
};

use axum::{
    Json, Router,
    extract::{State, rejection::JsonRejection},
    http::StatusCode,
    routing::post,
};
use serde_json::{Value, json};
use uuid::Uuid;

use crate::{
    api::{self, Api, ApiError, Owner},
    external_chart::{self, Note, Phrase},
    managed_api, managed_editor, managed_ingest,
};

const CACHE_PREFIX: &str = "benchmark-";

fn asset(api: &Api, value: &Value, required: bool) -> Result<Option<PathBuf>, ApiError> {
    let value = value.as_str().unwrap_or_default();
    if value.trim().is_empty() {
        return if required {
            Err(ApiError::invalid("Benchmark asset path is required"))
        } else {
            Ok(None)
        };
    }
    let path = PathBuf::from(value);
    let path = if let Ok(relative) = path.strip_prefix("~") {
        std::env::var_os("HOME")
            .map(PathBuf::from)
            .map(|home| home.join(relative))
            .unwrap_or(path)
    } else {
        path
    };
    let full = dunce::canonicalize(&path).map_err(|_| {
        ApiError(
            StatusCode::NOT_FOUND,
            format!(
                "Benchmark asset not found: {}",
                path.file_name().unwrap_or_default().to_string_lossy()
            ),
        )
    })?;
    let cache = ["benchmarks", "uploads"]
        .into_iter()
        .filter_map(|directory| dunce::canonicalize(api.backend.join(directory)).ok());
    if !cache.into_iter().any(|root| full.starts_with(root)) {
        return Err(ApiError::invalid(
            "Benchmark asset must be inside the local benchmark cache",
        ));
    }
    if !full.is_file() {
        return Err(ApiError(
            StatusCode::NOT_FOUND,
            format!(
                "Benchmark asset not found: {}",
                full.file_name().unwrap_or_default().to_string_lossy()
            ),
        ));
    }
    Ok(Some(full))
}

#[derive(Default)]
struct TimedWord {
    text: String,
    start: i64,
    end: i64,
    line: usize,
    best_pitch: i64,
    best_duration: i64,
}

fn words(phrases: &[Phrase]) -> Vec<TimedWord> {
    let mut words = Vec::new();
    for (line, phrase) in phrases.iter().enumerate() {
        let mut current = None::<TimedWord>;
        for note in &phrase.notes {
            // One separator is present between the pitch and the first lyric
            // character. Extra leading whitespace marks a new source word.
            let lyric = note
                .text
                .strip_prefix(char::is_whitespace)
                .unwrap_or(&note.text);
            let text = lyric.trim();
            if text.is_empty() {
                continue;
            }
            if lyric.starts_with(char::is_whitespace) || text.starts_with('(') {
                if let Some(word) = current.take() {
                    words.push(word);
                }
            }
            let word = current.get_or_insert_with(|| TimedWord {
                start: note.start_beat,
                line,
                best_duration: -1,
                ..Default::default()
            });
            word.text.push_str(text);
            word.end = note.start_beat + note.duration_beats;
            if note.duration_beats > word.best_duration && note.duration_beats > 0 {
                word.best_duration = note.duration_beats;
                word.best_pitch = note.pitch.saturating_add(60);
            }
        }
        if let Some(word) = current {
            words.push(word);
        }
    }
    words.retain(|word| !managed_editor::tokenize_lyrics(&word.text, "temp").is_empty());
    words
}

fn regroup(phrases: Vec<Phrase>) -> Vec<Phrase> {
    let mut result = Vec::new();
    let mut notes = Vec::<Note>::new();
    let mut last_end = -1i64;
    for mut note in phrases.into_iter().flat_map(|phrase| phrase.notes) {
        if !notes.is_empty() && note.start_beat.saturating_sub(last_end) > 4 {
            result.push(Phrase {
                notes: std::mem::take(&mut notes),
                end_beat: last_end.saturating_add(1),
            });
        }
        last_end = note.start_beat.saturating_add(note.duration_beats);
        note.text = note.text.trim().into();
        notes.push(note);
    }
    if !notes.is_empty() {
        result.push(Phrase {
            notes,
            end_beat: last_end.saturating_add(1),
        });
    }
    result
}

async fn import(
    State(api): State<Api>,
    owner: Owner,
    body: Result<Json<Value>, JsonRejection>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    let body = api::parse_body(body)?;
    let result = tokio::task::spawn_blocking(move || import_local(&api, &owner, &body))
        .await
        .map_err(|error| ApiError::internal(error.to_string()))??;
    Ok((StatusCode::CREATED, Json(result)))
}

fn import_local(api: &Api, owner: &Owner, body: &Value) -> Result<Value, ApiError> {
    let pipeline = body
        .get("pipeline")
        .map_or(Some("current"), Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_lowercase();
    if !matches!(pipeline.as_str(), "current" | "sofa") {
        return Err(ApiError::invalid("pipeline must be current or sofa"));
    }
    let chart = asset(api, &body["chart_path"], true)?.expect("required asset");
    let audio = asset(api, &body["audio_path"], true)?.expect("required asset");
    let vocals = asset(
        api,
        body.get("vocal_path").unwrap_or(&body["vocals_path"]),
        true,
    )?
    .expect("required asset");
    let instrumental = asset(api, &body["instrumental_path"], false)?;
    let requested = body["job_id"].as_str().unwrap_or_default().trim();
    if !requested.is_empty()
        && (!requested.starts_with(CACHE_PREFIX)
            || requested.contains(['/', '\\', '\0'])
            || !matches!(
                Path::new(requested).components().next(),
                Some(Component::Normal(_))
            ))
    {
        return Err(ApiError::invalid("job_id must use the benchmark- prefix"));
    }
    let id = if requested.is_empty() {
        format!(
            "{CACHE_PREFIX}{}",
            &Uuid::new_v4().simple().to_string()[..16]
        )
    } else {
        requested.to_owned()
    };
    let content = fs::read_to_string(&chart)
        .map_err(|error| ApiError::invalid(format!("Invalid benchmark chart: {error}")))?;
    let fallback = chart.file_stem().unwrap_or_default().to_string_lossy();
    let mut parsed = external_chart::parse(&content, &fallback)
        .map_err(|error| ApiError::invalid(format!("Invalid benchmark chart: {error}")))?;
    let headers = external_chart::headers(&content);
    parsed.metadata.language = headers.get("LANGUAGE").copied().unwrap_or("en").into();
    if let Some(title) = body["title"].as_str().filter(|title| !title.is_empty()) {
        parsed.metadata.title = title.into();
    }
    if let Some(artist) = body["artist"].as_str().filter(|artist| !artist.is_empty()) {
        parsed.metadata.artist = artist.into();
    }
    let words = words(&parsed.phrases);
    if words.is_empty() {
        return Err(ApiError::invalid("Reference contains no timed lyric words"));
    }
    let mut lines = Vec::<String>::new();
    let mut previous = None;
    for word in &words {
        if previous == Some(word.line) {
            let line = lines.last_mut().unwrap();
            line.push(' ');
            line.push_str(&word.text);
        } else {
            lines.push(word.text.clone());
            previous = Some(word.line);
        }
    }
    let text = lines.join("\n");
    let revision_id = format!("revision-{}", Uuid::new_v4().simple());
    let mut revision = managed_editor::create_revision(
        &id,
        Some(&revision_id),
        &text,
        &parsed.metadata.language,
        "manual",
        None,
    )?;
    revision["language"] = json!(parsed.metadata.language);
    let tokens = managed_editor::array(&revision["tokens"]);
    if tokens.len() != words.len() {
        return Err(ApiError::invalid(
            "Benchmark chart lyrics do not tokenize one-to-one",
        ));
    }
    let run_id = format!("run-{}", Uuid::new_v4().simple());
    let run = json!({"id":run_id,"lyric_revision_id":revision_id,"status":"completed","selected_strategy":pipeline,
        "diagnostics":{"source":"cached benchmark chart","pipeline":pipeline},"created_at":managed_ingest::queued_now()?});
    let step = 60.0 / (parsed.metadata.bpm * 16.0);
    let gap = parsed.metadata.gap_ms / 1000.0;
    let mut projected = Vec::with_capacity(words.len());
    let mut alignments = Vec::with_capacity(words.len());
    for (word, token) in words.into_iter().zip(tokens) {
        let start = gap + word.start as f64 * step;
        let end = gap + word.end as f64 * step;
        projected.push(json!({
            "text":word.text,"start":start,"end":end,"midi_note":word.best_pitch,"pitch_hz":0.0,"confidence":1.0,
            "token_id":token["id"],"alignment_run_id":run_id,"alignment_status":"aligned",
            "continuation_of_previous":false,"pitch_contour":[],"pitch_candidates":[],
        }));
        alignments.push(json!({
            "start":start,"end":end,"status":"aligned","confidence":1.0,"token_id":token["id"],
            "alignment_run_id":run_id,"start_locked":false,"end_locked":false,
            "evidence":{"selected_model":pipeline,"selected_audio":"vocal_stem","candidate_scores":[],"reasons":[]},
        }));
    }
    let txt = content
        .lines()
        .map(|line| {
            if line.starts_with("#MP3:") {
                "#MP3:audio.wav"
            } else if line.starts_with("#VOCALS:") {
                "#VOCALS:vocals.wav"
            } else if line.starts_with("#INSTRUMENTAL:") {
                "#INSTRUMENTAL:instrumental.wav"
            } else {
                line
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    let phrases = regroup(parsed.phrases);
    let variant = json!({"words":projected,"pitch_words":projected,"phrases":phrases,"ultrastar_txt":txt,
        "gap_ms":parsed.metadata.gap_ms,"word_alignments":[],"alignment_run":null});
    let uploads = api.backend.join("uploads");
    fs::create_dir_all(&uploads).map_err(|error| ApiError::internal(error.to_string()))?;
    let work = uploads.join(&id);
    let staging =
        tempfile::tempdir_in(&uploads).map_err(|error| ApiError::internal(error.to_string()))?;
    for (source, name) in [
        (Some(audio.as_path()), "audio.wav"),
        (Some(vocals.as_path()), "vocals.wav"),
        (instrumental.as_deref(), "instrumental.wav"),
    ] {
        if let Some(source) = source {
            let path = staging.path().join(name);
            if fs::hard_link(source, &path).is_err() {
                fs::copy(source, &path).map_err(|error| ApiError::internal(error.to_string()))?;
            }
        }
    }
    let duration = super::managed_alignment::engine::audio_duration(&audio);
    let result = json!({"metadata":parsed.metadata,"words":projected,"pitch_words":projected,"phrases":phrases,
        "ultrastar_txt":txt,"audio_file":"audio.wav","vocals_file":"vocals.wav","video_file":"",
        "instrumental_file":if instrumental.is_some() {"instrumental.wav"} else {""},
        "alignment_results":{pipeline.clone():variant},"current_alignment_mode":pipeline,
        "lyric_revision":revision,"word_alignments":alignments,"alignment_run":run,
        "alignment_comparison":{},"note_overrides":{},"audio_duration":duration});
    let job = json!({"id":id,"owner_id":owner.0,"status":"done","progress_message":format!("Imported {pipeline} benchmark result"),
        "progress_percent":100.0,"original_filename":format!("{} - {} [{pipeline}]",parsed.metadata.artist,parsed.metadata.title),
        "metadata":parsed.metadata,"work_dir":work,"upload_path":work.join("audio.wav"),"audio_path":work.join("audio.wav"),
        "vocals_path":work.join("vocals.wav"),
        "instrumental_path":instrumental.as_ref().map_or_else(PathBuf::new, |_|work.join("instrumental.wav")),
        "real_bpm":parsed.metadata.bpm,"lyric_revisions":[revision],"selected_lyric_revision_id":revision_id,
        "alignment_runs":[run],"result":result});
    let mut installed = false;
    let receipt = managed_api::jobs(api)?.mutate(&id, |existing| {
        if existing.is_some() || work.exists() {
            return Err(ApiError(
                StatusCode::CONFLICT,
                "Benchmark job already exists".into(),
            ));
        }
        fs::rename(staging.path(), &work).map_err(|error| ApiError::internal(error.to_string()))?;
        installed = true;
        let result = managed_api::result_payload(&job)
            .ok_or_else(|| ApiError::internal("Benchmark result is unavailable"))?;
        Ok((
            Some(job),
            json!({"job_id":id,"pipeline":pipeline,"result":result}),
        ))
    });
    if receipt.is_err() && installed {
        let _ = fs::remove_dir_all(work);
    }
    receipt
}

pub(crate) fn routes() -> Router<Api> {
    Router::new().route("/api/benchmark/import", post(import))
}
