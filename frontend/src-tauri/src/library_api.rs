use std::{
    collections::HashMap,
    fs::{self, File, OpenOptions},
    path::{Component, Path, PathBuf},
    sync::Mutex,
};

use axum::{
    Json, Router,
    extract::{Query, Request, State, rejection::JsonRejection},
    http::{HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{delete, get, post},
};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tower::ServiceExt;
use tower_http::services::ServeFile;

use crate::{
    api::{self, Api, ApiError, ApiResult, Owner},
    external_chart::{self, Phrase},
    library::{self, Library},
    managed_api::{self, field_text},
    worker,
};

pub(crate) type StemStates = Mutex<HashMap<PathBuf, Separation>>;

#[derive(Clone, Default)]
pub(crate) enum Separation {
    #[default]
    Idle,
    InProgress,
    Done,
    Error(String),
}

impl Separation {
    fn status(&self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::InProgress => "separating",
            Self::Done => "done",
            Self::Error(_) => "error",
        }
    }

    fn error(&self) -> &str {
        match self {
            Self::Error(error) => error,
            _ => "",
        }
    }

    fn payload(&self) -> Value {
        if self.error().is_empty() {
            json!({"status":self.status()})
        } else {
            json!({"status":self.status(),"error":self.error()})
        }
    }
}

#[derive(Deserialize)]
struct SongQuery {
    path: String,
}

#[derive(Deserialize)]
struct TrackQuery {
    path: String,
    #[serde(default = "default_track")]
    track: String,
}

fn default_track() -> String {
    "original".into()
}

pub(crate) fn library(api: &Api) -> Library {
    let outputs = api.backend.join("outputs");
    Library::new(
        outputs.join("working-directory.json"),
        outputs.join("playback-audio"),
    )
}

async fn get_directory(State(api): State<Api>) -> ApiResult {
    Ok(Json(
        json!({"directory":library(&api).directory().ok().flatten()}),
    ))
}

async fn set_directory(
    State(api): State<Api>,
    body: Result<Json<Value>, JsonRejection>,
) -> ApiResult {
    let body = api::parse_body(body)?;
    let directory = if body["directory"].is_null() {
        None
    } else {
        let directory = body["directory"]
            .as_str()
            .ok_or_else(|| ApiError::invalid("directory must be a path or null"))?
            .trim();
        if directory.is_empty() {
            return Err(ApiError::invalid("directory must not be empty"));
        }
        let path = if let Some(relative) = directory.strip_prefix("~/") {
            home_directory().map_or_else(|| PathBuf::from(directory), |home| home.join(relative))
        } else if directory == "~" {
            home_directory().unwrap_or_else(|| PathBuf::from(directory))
        } else {
            PathBuf::from(directory)
        };
        Some(
            dunce::canonicalize(&path)
                .ok()
                .filter(|path| path.is_dir())
                .ok_or_else(|| ApiError::invalid("Working directory must be an existing folder"))?,
        )
    };
    library(&api)
        .write_directory(directory.clone())
        .map_err(ApiError::internal)?;
    Ok(Json(json!({"directory":directory})))
}

fn home_directory() -> Option<PathBuf> {
    std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" }).map(PathBuf::from)
}

fn chart(api: &Api, path: &str) -> Result<(PathBuf, PathBuf), ApiError> {
    library(api).chart_path(path).map_err(|error| {
        let status = if error == "Working directory is not configured" {
            StatusCode::CONFLICT
        } else {
            StatusCode::NOT_FOUND
        };
        ApiError(status, error)
    })
}

fn stem_directory(api: &Api, chart: &Path) -> PathBuf {
    let key = hex::encode(Sha256::digest(chart.to_string_lossy().as_bytes()));
    api.backend.join("outputs/external-stems").join(key)
}

fn both_stems(directory: &Path) -> bool {
    directory.join("vocals.wav").is_file() && directory.join("instrumental.wav").is_file()
}

fn stem_state(api: &Api, directory: &Path) -> Result<Separation, ApiError> {
    if both_stems(directory) {
        return Ok(Separation::Done);
    }
    let states = api
        .stem_states
        .lock()
        .map_err(|error| ApiError::internal(error.to_string()))?;
    Ok(states.get(directory).cloned().unwrap_or_default())
}

fn chart_text(path: &Path) -> Result<String, ApiError> {
    library::read_chart(path)
        .map_err(|_| ApiError(StatusCode::NOT_FOUND, "Song file is unavailable".into()))
}

async fn get_song(
    State(api): State<Api>,
    Query(SongQuery { path }): Query<SongQuery>,
) -> ApiResult {
    let (root, chart) = chart(&api, &path)?;
    let content = chart_text(&chart)?;
    let name = chart
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("Song");
    let parsed = external_chart::parse(&content, name).map_err(ApiError::invalid)?;
    let state = stem_state(&api, &stem_directory(&api, &chart))?;
    Ok(Json(json!({
        "path":path,"filename":chart.file_name(),"content":content,"metadata":parsed.metadata,
        "phrases":parsed.phrases,"audio_available":library::resolve_audio(&root, &chart, &content).is_some(),
        "stems_available":matches!(state, Separation::Done),"separation_status":state.status(),
        "separation_error":state.error(),
    })))
}

async fn save_song(
    State(api): State<Api>,
    Query(SongQuery { path }): Query<SongQuery>,
    body: Result<Json<Value>, JsonRejection>,
) -> ApiResult {
    let (_, chart) = chart(&api, &path)?;
    let body = api::parse_body(body)?;
    if !body["phrases"].is_array() {
        return Err(ApiError::invalid("phrases must be an array"));
    }
    let invalid_edit = |error: String| ApiError::invalid(format!("Invalid chart edit: {error}"));
    let phrases: Vec<Phrase> = serde_json::from_value(body["phrases"].clone())
        .map_err(|error| invalid_edit(error.to_string()))?;
    let content = tokio::task::spawn_blocking(move || library::save_chart_file(&chart, &phrases))
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?
        .map_err(invalid_edit)?;
    Ok(Json(json!({"status":"updated","content":content})))
}

pub(crate) async fn send_file(path: PathBuf, request: Request) -> Result<Response, ApiError> {
    ServeFile::new(path)
        .oneshot(request)
        .await
        .map(IntoResponse::into_response)
        .map_err(|never| match never {})
}

pub(crate) fn download_name(filename: &str) -> Result<HeaderValue, ApiError> {
    let mut encoded = String::new();
    for byte in filename.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_') {
            encoded.push(byte as char);
        } else {
            use std::fmt::Write;
            write!(&mut encoded, "%{byte:02X}").unwrap();
        }
    }
    let value = if encoded == filename {
        format!("attachment; filename=\"{filename}\"")
    } else {
        format!("attachment; filename*=UTF-8''{encoded}")
    };
    value
        .parse()
        .map_err(|error: axum::http::header::InvalidHeaderValue| {
            ApiError::internal(error.to_string())
        })
}

async fn download_song(
    State(api): State<Api>,
    Query(SongQuery { path }): Query<SongQuery>,
    request: Request,
) -> Result<Response, ApiError> {
    let (_, chart) = chart(&api, &path)?;
    let disposition = download_name(&chart.file_name().unwrap_or_default().to_string_lossy())?;
    let mut response = send_file(chart, request).await?;
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/plain; charset=utf-8"),
    );
    response
        .headers_mut()
        .insert(header::CONTENT_DISPOSITION, disposition);
    Ok(response)
}

async fn audio(
    State(api): State<Api>,
    Query(TrackQuery { path, track }): Query<TrackQuery>,
    request: Request,
) -> Result<Response, ApiError> {
    let (root, chart) = chart(&api, &path)?;
    let source = match track.as_str() {
        "original" => library::resolve_audio(&root, &chart, &chart_text(&chart)?),
        "vocals" | "instrumental" => {
            Some(stem_directory(&api, &chart).join(format!("{track}.wav")))
        }
        _ => return Err(ApiError::invalid("Invalid audio track")),
    }
    .filter(|source| source.is_file())
    .ok_or_else(|| ApiError(StatusCode::NOT_FOUND, "Audio file is unavailable".into()))?;
    let cache = api.backend.join("outputs/playback-audio");
    let path = tokio::task::spawn_blocking(move || library::playback_path(&source, &cache))
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?;
    send_file(path, request).await
}

async fn separation_status(
    State(api): State<Api>,
    Query(SongQuery { path }): Query<SongQuery>,
) -> ApiResult {
    let (_, chart) = chart(&api, &path)?;
    Ok(Json(
        stem_state(&api, &stem_directory(&api, &chart))?.payload(),
    ))
}

fn separate(backend: &Path, audio: &Path, output: &Path) -> Result<(), String> {
    let parent = output.parent().ok_or("Missing stem output directory")?;
    fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    let _slot = acquire_processing_slot(backend)?;
    if both_stems(output) {
        return Ok(());
    }
    let staging = tempfile::tempdir_in(parent).map_err(|error| error.to_string())?;
    let result: Value = worker::infer(
        backend,
        json!({
            "operation":"separate","audio":audio,"output_directory":staging.path(),"require_demucs":true,
        }),
    )?;
    if result["vocals"].as_str() != staging.path().join("vocals.wav").to_str()
        || result["instrumental"].as_str() != staging.path().join("instrumental.wav").to_str()
        || !both_stems(staging.path())
    {
        return Err("Vocal separation did not produce both audio stems.".into());
    }
    if output.exists() {
        fs::remove_dir_all(output).map_err(|error| error.to_string())?;
    }
    fs::rename(staging.path(), output).map_err(|error| error.to_string())
}

pub(crate) fn acquire_processing_slot(backend: &Path) -> Result<File, String> {
    let slot = OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(backend.join("outputs/processing.lock"))
        .map_err(|error| error.to_string())?;
    slot.lock().map_err(|error| error.to_string())?;
    Ok(slot)
}

async fn start_separation(
    State(api): State<Api>,
    Query(SongQuery { path }): Query<SongQuery>,
) -> ApiResult {
    let (root, chart) = chart(&api, &path)?;
    let source = library::resolve_audio(&root, &chart, &chart_text(&chart)?)
        .ok_or_else(|| ApiError(StatusCode::NOT_FOUND, "Source audio is unavailable".into()))?;
    let directory = stem_directory(&api, &chart);
    if both_stems(&directory) {
        return Ok(Json(Separation::Done.payload()));
    }
    {
        let mut states = api
            .stem_states
            .lock()
            .map_err(|error| ApiError::internal(error.to_string()))?;
        if !matches!(states.get(&directory), Some(Separation::InProgress)) {
            states.insert(directory.clone(), Separation::InProgress);
            let api = api.clone();
            tokio::task::spawn_blocking(move || {
                let state = match separate(&api.backend, &source, &directory) {
                    Ok(()) => Separation::Done,
                    Err(error) => Separation::Error(error),
                };
                if let Ok(mut states) = api.stem_states.lock() {
                    states.insert(directory, state);
                }
            });
        }
    }
    Ok(Json(Separation::InProgress.payload()))
}

#[derive(Deserialize)]
struct RemoveQuery {
    job_id: Option<String>,
    path: Option<String>,
}

fn managed_directory(root: &Path, target: &Path) -> Result<Option<PathBuf>, String> {
    let metadata = match fs::symlink_metadata(target) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.to_string()),
    };
    if metadata.is_symlink() {
        return Err("Refusing to remove a symlinked song directory".into());
    }
    let root = dunce::canonicalize(root).map_err(|error| error.to_string())?;
    let target = dunce::canonicalize(target).map_err(|error| error.to_string())?;
    if target.parent() != Some(root.as_path()) {
        return Err("Managed song directory must be directly inside its storage root".into());
    }
    if !metadata.is_dir() {
        return Err("Managed song storage path is not a directory".into());
    }
    Ok(Some(target))
}

fn delete_managed(api: &Api, owner: &Owner, id: &str) -> Result<(), ApiError> {
    let failure = |error| {
        ApiError::internal(format!(
            "Song record was kept because its files could not all be removed: {error}"
        ))
    };
    managed_api::jobs(api)?.mutate(id, |job| {
        let job = job
            .filter(|job| managed_api::belongs_to(job, owner))
            .ok_or_else(|| ApiError(StatusCode::NOT_FOUND, "Song not found".into()))?;
        if !matches!(
            field_text(job, "status"),
            "pending" | "awaiting_lyrics" | "done" | "error"
        ) {
            return Err(ApiError(
                StatusCode::CONFLICT,
                "Cannot remove a song while it is processing".into(),
            ));
        }
        if id.contains(['/', '\\', '\0'])
            || !matches!(
                Path::new(id).components().next(),
                Some(Component::Normal(_))
            )
        {
            return Err(failure("Invalid managed song identifier".into()));
        }

        let outputs = api.backend.join("outputs");
        let uploads = api.backend.join("uploads");
        let work = field_text(job, "work_dir");
        let work = if work.is_empty() {
            outputs.join(id)
        } else {
            PathBuf::from(work)
        };
        let imported_benchmark = id.starts_with("benchmark-") && work == uploads.join(id);
        let workspace = managed_directory(
            if imported_benchmark {
                &uploads
            } else {
                &outputs
            },
            &work,
        )
        .map_err(failure)?;

        let upload = Path::new(field_text(job, "upload_path"));
        let upload = upload
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .map_or_else(|| uploads.join(id), Path::to_path_buf);
        let uploaded = if upload == uploads.join(id) {
            managed_directory(&uploads, &upload).map_err(failure)?
        } else {
            log::warn!(
                "Skipping upload cleanup for {id} because its path is outside its managed directory"
            );
            None
        };

        let published = field_text(job, "library_path");
        let configured = field_text(job, "library_root");
        let root = if configured.is_empty() {
            library(api).directory().ok().flatten()
        } else {
            Some(PathBuf::from(configured))
        };
        let published = if let Some(root) = root.filter(|_| !published.is_empty()) {
            managed_directory(&root, Path::new(published)).map_err(failure)?
        } else {
            None
        };
        let mut removed = std::collections::HashSet::new();
        for target in [workspace, uploaded, published].into_iter().flatten() {
            if removed.insert(target.clone()) {
                fs::remove_dir_all(target).map_err(|error| failure(error.to_string()))?;
            }
        }
        Ok((None, ()))
    })
}

async fn remove_song(
    State(api): State<Api>,
    owner: Owner,
    Query(query): Query<RemoveQuery>,
) -> Result<Response, ApiError> {
    let job_id = query.job_id.filter(|value| !value.is_empty());
    let path = query.path.filter(|value| !value.is_empty());
    match (job_id, path) {
        (Some(id), None) => {
            let removed = id.clone();
            tokio::task::spawn_blocking(move || delete_managed(&api, &owner, &id))
                .await
                .map_err(|error| ApiError::internal(error.to_string()))??;
            Ok(Json(json!({"removed":removed})).into_response())
        }
        (None, Some(path)) => {
            let (root, chart) = chart(&api, &path)?;
            tokio::task::spawn_blocking(move || library::remove_chart_file(&root, &chart))
                .await
                .map_err(|error| ApiError::internal(error.to_string()))?
                .map_err(ApiError::internal)?;
            Ok(Json(json!({"removed":path})).into_response())
        }
        _ => Err(ApiError::invalid("Provide either job_id or path")),
    }
}

pub(crate) fn routes() -> Router<Api> {
    Router::new()
        .route(
            "/api/library/working-directory",
            get(get_directory).put(set_directory),
        )
        .route("/api/library/external-song", get(get_song).put(save_song))
        .route("/api/library/external-song/audio", get(audio))
        .route("/api/library/external-song/download", get(download_song))
        .route("/api/library/external-song/split", post(start_separation))
        .route(
            "/api/library/external-song/split-status",
            get(separation_status),
        )
        .route("/api/library/song", delete(remove_song))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removing_chart_at_root_and_nested_song_have_distinct_scopes() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path();
        let nested = root.join("Artist/Song");
        fs::create_dir_all(&nested).unwrap();
        let chart = nested.join("Song.txt");
        fs::write(&chart, "#TITLE:Song").unwrap();
        fs::write(nested.join("Song.mp3"), "audio").unwrap();
        let first = root.join("Other.txt");
        fs::write(&first, "#TITLE:Other").unwrap();
        library::remove_chart_file(root, &chart).unwrap();
        assert!(!nested.exists());
        assert!(first.exists());
        library::remove_chart_file(root, &first).unwrap();
        assert!(root.exists());
        assert!(!first.exists());
    }

    #[tokio::test]
    async fn http_audio_preserves_byte_range_responses() {
        let temporary = tempfile::tempdir().unwrap();
        let audio = temporary.path().join("Song.wav");
        fs::write(&audio, "RIFF-wave").unwrap();
        let request = Request::builder()
            .header(header::RANGE, "bytes=0-3")
            .body(axum::body::Body::empty())
            .unwrap();
        let response = send_file(audio, request).await.ok().unwrap();
        assert_eq!(response.status(), StatusCode::PARTIAL_CONTENT);
        let bytes = axum::body::to_bytes(response.into_body(), 4).await.unwrap();
        assert_eq!(&bytes[..], b"RIFF");
    }
}
