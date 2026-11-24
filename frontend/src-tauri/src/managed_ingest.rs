use std::{
    fs,
    net::SocketAddr,
    path::{Component, Path, PathBuf},
    sync::Arc,
};

use axum::{
    Json, Router,
    extract::{
        ConnectInfo, DefaultBodyLimit, Multipart, Path as RoutePath, State,
        multipart::MultipartRejection, rejection::JsonRejection,
    },
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
};
use serde::Deserialize;
use serde_json::{Value, json};
use time::{OffsetDateTime, format_description::well_known::Rfc3339};
use tokio::io::AsyncWriteExt;
use uuid::Uuid;

use crate::{
    api::{self, Api, ApiError, ApiResult, Owner},
    managed_api::{self, ManagedJobs, field_text},
    preparation::{self, BrowserMedia},
    youtube,
};

const DEFAULT_UPLOAD_BYTES: u64 = 2 * 1024 * 1024 * 1024;

fn max_upload_bytes() -> u64 {
    std::env::var("KARAOKE_MAX_UPLOAD_BYTES")
        .ok()
        .and_then(|bytes| bytes.parse().ok())
        .unwrap_or(DEFAULT_UPLOAD_BYTES)
}

pub(crate) fn language(language: &str) -> Result<&'static str, ApiError> {
    let language = language.trim().to_lowercase();
    if language.starts_with("pl") || matches!(language.as_str(), "polish" | "polski" | "po") {
        return Ok("pl");
    }
    if language.starts_with("en") || matches!(language.as_str(), "" | "english") {
        return Ok("en");
    }
    Err(ApiError::invalid(
        "Word alignment supports English (en) or Polish (pl)",
    ))
}

fn text_or(value: &str, fallback: impl FnOnce() -> String) -> String {
    if value.is_empty() {
        fallback()
    } else {
        value.to_owned()
    }
}

fn stem(path: &str) -> String {
    Path::new(path)
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or(path)
        .into()
}

#[derive(Default, Deserialize)]
#[serde(default)]
struct Metadata {
    title: String,
    artist: String,
    language: String,
    genre: String,
    year: String,
}

impl Metadata {
    fn from_required(body: &Value) -> Result<Self, ApiError> {
        for key in ["title", "artist", "language", "genre", "year"] {
            if !body[key].is_string() {
                return Err(ApiError::invalid(format!("{key} must be a string")));
            }
        }
        serde_json::from_value(body.clone()).map_err(|error| ApiError::invalid(error.to_string()))
    }

    fn payload(&self, title: String) -> Result<Value, ApiError> {
        Ok(json!({
            "title": title, "artist":self.artist, "language":language(&self.language)?,
            "genre":self.genre, "year":self.year, "bpm":0.0, "gap_ms":0.0
        }))
    }
}

fn pending(id: Uuid, owner: &Owner, filename: &str, path: &Path, metadata: Value) -> Value {
    json!({
        "id":id, "owner_id":owner.0, "original_filename":filename, "upload_path":path,
        "metadata":metadata, "status":"pending", "progress_percent":0.0
    })
}

fn bad_request(message: &str) -> ApiError {
    ApiError(StatusCode::BAD_REQUEST, message.into())
}

fn conflict(message: &str) -> ApiError {
    ApiError(StatusCode::CONFLICT, message.into())
}

fn missing_job() -> ApiError {
    ApiError(StatusCode::NOT_FOUND, "Job not found".into())
}

fn save_new(jobs: &ManagedJobs, id: Uuid, job: Value) -> Result<(), ApiError> {
    jobs.mutate(&id.to_string(), |existing| {
        if existing.is_some() {
            return Err(conflict("Job already exists"));
        }
        Ok((Some(job), ()))
    })
}

async fn upload(
    State(api): State<Api>,
    owner: Owner,
    multipart: Result<Multipart, MultipartRejection>,
) -> ApiResult {
    let mut multipart = multipart.map_err(|error| ApiError::invalid(error.body_text()))?;
    let id = Uuid::new_v4();
    let uploads = api.backend.join("uploads");
    fs::create_dir_all(&uploads).map_err(|error| ApiError::internal(error.to_string()))?;
    let staging =
        tempfile::tempdir_in(&uploads).map_err(|error| ApiError::internal(error.to_string()))?;
    let mut file_info = None;
    let mut metadata = Metadata::default();
    let max_bytes = max_upload_bytes();
    while let Some(mut field) = multipart
        .next_field()
        .await
        .map_err(|error| ApiError::invalid(error.to_string()))?
    {
        match field.name() {
            Some("file") => {
                let filename = field.file_name().unwrap_or("input").to_owned();
                if file_info.is_some()
                    || filename.contains(['/', '\\', '\0'])
                    || !matches!(
                        Path::new(&filename).components().next(),
                        Some(Component::Normal(_))
                    )
                {
                    return Err(ApiError::invalid("Invalid upload filename"));
                }
                let stored = Path::new(&filename)
                    .extension()
                    .and_then(|suffix| suffix.to_str())
                    .filter(|suffix| !suffix.is_empty())
                    .map_or_else(
                        || "input.bin".to_owned(),
                        |suffix| format!("input.{}", suffix.to_lowercase()),
                    );
                let mut output = tokio::fs::File::create(staging.path().join(&stored))
                    .await
                    .map_err(|error| ApiError::internal(error.to_string()))?;
                let mut len = 0u64;
                while let Some(bytes) = field
                    .chunk()
                    .await
                    .map_err(|error| ApiError::invalid(error.to_string()))?
                {
                    len += bytes.len() as u64;
                    if len > max_bytes {
                        return Err(ApiError(
                            StatusCode::PAYLOAD_TOO_LARGE,
                            "Uploaded file is too large".into(),
                        ));
                    }
                    output
                        .write_all(&bytes)
                        .await
                        .map_err(|error| ApiError::internal(error.to_string()))?;
                }
                output
                    .flush()
                    .await
                    .map_err(|error| ApiError::internal(error.to_string()))?;
                file_info = Some((filename, stored));
            }
            field_name => {
                let target = match field_name {
                    Some("title") => Some(&mut metadata.title),
                    Some("artist") => Some(&mut metadata.artist),
                    Some("language") => Some(&mut metadata.language),
                    Some("genre") => Some(&mut metadata.genre),
                    Some("year") => Some(&mut metadata.year),
                    _ => None,
                };
                if let Some(target) = target {
                    let mut text = Vec::new();
                    while let Some(bytes) = field
                        .chunk()
                        .await
                        .map_err(|error| ApiError::invalid(error.to_string()))?
                    {
                        if text.len() + bytes.len() > 1024 * 1024 {
                            return Err(ApiError::invalid("Upload metadata is too large"));
                        }
                        text.extend_from_slice(&bytes);
                    }
                    *target = String::from_utf8(text)
                        .map_err(|error| ApiError::invalid(error.to_string()))?;
                }
            }
        }
    }
    let (filename, stored) = file_info.ok_or_else(|| ApiError::invalid("file is required"))?;
    let metadata = metadata.payload(text_or(&metadata.title, || stem(&filename)))?;
    let directory = uploads.join(id.to_string());
    fs::rename(staging.path(), &directory)
        .map_err(|error| ApiError::internal(error.to_string()))?;
    let job = pending(id, &owner, &filename, &directory.join(stored), metadata);
    if let Err(error) = save_new(managed_api::jobs(&api)?, id, job) {
        let _ = fs::remove_dir_all(directory);
        return Err(error);
    }
    Ok(Json(json!({"job_id":id})))
}

#[derive(Deserialize)]
struct Import {
    import_id: Uuid,
    #[serde(flatten)]
    metadata: Metadata,
}

#[derive(Deserialize)]
struct Manifest {
    original_filename: String,
    has_video: bool,
}

fn invalid_import(message: impl std::fmt::Display) -> ApiError {
    ApiError::invalid(format!("Invalid prepared media: {message}"))
}

fn prepared_file(root: &Path, parts: &[&str]) -> Result<PathBuf, ApiError> {
    let mut path = root.to_path_buf();
    for part in parts {
        path.push(part);
    }
    let path = dunce::canonicalize(path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            ApiError(
                StatusCode::NOT_FOUND,
                "Prepared media not found in this backend".into(),
            )
        } else {
            invalid_import(error)
        }
    })?;
    if !path.starts_with(root) || !path.is_file() {
        return Err(invalid_import("Stored file escapes the job directory"));
    }
    Ok(path)
}

async fn register(
    State(api): State<Api>,
    owner: Owner,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    body: Result<Json<Import>, JsonRejection>,
) -> ApiResult {
    if !peer.ip().is_loopback() {
        return Err(ApiError(
            StatusCode::FORBIDDEN,
            "Prepared media import requires a local backend".into(),
        ));
    }
    let Json(body) = body.map_err(|error| ApiError::invalid(error.body_text()))?;
    let id = body.import_id;
    let job_id = id.to_string();
    let jobs = managed_api::jobs(&api)?;
    let uploads = api.backend.join("uploads");
    let outputs = api.backend.join("outputs");
    let upload = prepared_file(&uploads, &[&job_id, "input"])?;
    if jobs
        .job(&owner, &job_id, |job| {
            Ok(field_text(job, "upload_path") == upload.to_string_lossy())
        })
        .unwrap_or(false)
    {
        return Ok(Json(json!({"job_id":id})));
    }
    let manifest = prepared_file(&uploads, &[&job_id, "prepared-media.json"])?;
    let manifest: Manifest = serde_json::from_slice(&fs::read(manifest).map_err(invalid_import)?)
        .map_err(invalid_import)?;
    let original_size = fs::metadata(&upload).map_err(invalid_import)?.len();
    if original_size > max_upload_bytes() {
        return Err(ApiError(
            StatusCode::PAYLOAD_TOO_LARGE,
            "Selected file is too large".into(),
        ));
    }
    if original_size == 0 || manifest.original_filename.is_empty() {
        return Err(invalid_import(
            "Prepared import is missing the original media",
        ));
    }
    let audio = prepared_file(&outputs, &[&job_id, "audio.wav"])?;
    let wav = hound::WavReader::open(&audio).map_err(invalid_import)?;
    let format = wav.spec();
    if wav.duration() == 0
        || (format.channels, format.bits_per_sample, format.sample_rate) != (2, 16, 44100)
        || format.sample_format != hound::SampleFormat::Int
    {
        return Err(invalid_import(
            "Expected a nonempty 44.1 kHz stereo PCM WAV",
        ));
    }
    let record = prepared_file(&uploads, &[&job_id, "desktop-preparation.json"])?;
    let preparation: Value = serde_json::from_slice(&fs::read(record).map_err(invalid_import)?)
        .map_err(invalid_import)?;
    if preparation["status"] != "awaiting_lyrics" {
        return Err(invalid_import(
            "Desktop preparation has not reached lyrics review",
        ));
    }
    let bpm = preparation["real_bpm"]
        .as_f64()
        .filter(|bpm| bpm.is_finite() && *bpm >= 0.0)
        .ok_or_else(|| invalid_import("Invalid prepared tempo"))?;
    let vocals = prepared_file(&outputs, &[&job_id, "vocals.wav"])?;
    let instrumental = prepared_file(&outputs, &[&job_id, "instrumental.wav"])?;
    for stem in [&vocals, &instrumental] {
        if hound::WavReader::open(stem)
            .map_err(invalid_import)?
            .duration()
            == 0
        {
            return Err(invalid_import(format!(
                "Prepared {} is empty",
                stem.file_name().unwrap_or_default().to_string_lossy()
            )));
        }
    }
    let metadata = body.metadata.payload(text_or(&body.metadata.title, || {
        stem(&manifest.original_filename)
    }))?;
    let mut job = pending(id, &owner, &manifest.original_filename, &upload, metadata);
    job["status"] = json!("awaiting_lyrics");
    job["progress_message"] = json!("Waiting for lyrics approval.");
    job["work_dir"] = json!(outputs.join(&job_id));
    job["audio_path"] = json!(audio);
    job["has_video"] = json!(manifest.has_video);
    job["vocals_path"] = json!(vocals);
    job["instrumental_path"] = json!(instrumental);
    job["real_bpm"] = json!(bpm);
    job["found_lyrics"] = json!(field_text(&preparation, "found_lyrics"));
    jobs.mutate(&job_id, |current| match current {
        Some(old)
            if managed_api::belongs_to(old, &owner) && old["upload_path"] == job["upload_path"] =>
        {
            Ok((Some(old.clone()), ()))
        }
        Some(_) => Err(ApiError(
            StatusCode::NOT_FOUND,
            "Prepared media not found".into(),
        )),
        None => Ok((Some(job), ())),
    })?;
    Ok(Json(json!({"job_id":id})))
}

fn update_job(
    jobs: &ManagedJobs,
    job_id: &str,
    update: impl FnOnce(&mut Value),
) -> Result<Value, ApiError> {
    jobs.mutate(job_id, |job| {
        let mut job = job.cloned().ok_or_else(missing_job)?;
        update(&mut job);
        Ok((Some(job.clone()), job))
    })
}

fn set_status(job: &mut Value, status: &str, message: &str) {
    job["status"] = json!(status);
    job["progress_message"] = json!(message);
}

fn event(job: &Value) -> Value {
    let mut event = managed_api::progress_snapshot(job);
    let error = field_text(job, "error");
    if !error.is_empty() {
        event["error"] = json!(error);
    }
    event
}

fn run_download(api: Api, jobs: Arc<ManagedJobs>, id: Uuid, url: String) {
    tokio::task::spawn_blocking(move || {
        let id_text = id.to_string();
        let download = youtube::download(&api.backend, id, &url, |progress| {
            jobs.update_download(&id_text, progress)
        });
        let updated = update_job(&jobs, &id_text, |stored| match download {
            Ok(media) => {
                let media = json!(media);
                for field in [
                    "original_filename",
                    "upload_path",
                    "audio_path",
                    "work_dir",
                    "has_video",
                ] {
                    stored[field] = media[field].clone();
                }
                if field_text(&stored["metadata"], "title").is_empty() {
                    stored["metadata"]["title"] = media["title"].clone();
                }
                if field_text(&stored["metadata"], "artist").is_empty() {
                    stored["metadata"]["artist"] = media["artist"].clone();
                }
                stored["failed_phase"] = json!("");
                set_status(stored, "pending", "Download complete");
            }
            Err(error) => {
                stored["failed_phase"] = json!("download");
                stored["error"] = json!(error);
                set_status(stored, "error", &format!("Download failed: {error}"));
            }
        });
        match updated {
            Ok(value) => jobs.emit(&id_text, event(&value)),
            Err(error) => log::error!(
                "Unable to store YouTube download result for {id}: {}",
                error.1
            ),
        }
    });
}

async fn download(
    State(api): State<Api>,
    owner: Owner,
    body: Result<Json<Value>, JsonRejection>,
) -> ApiResult {
    let body = api::parse_body(body)?;
    let url = body["url"]
        .as_str()
        .ok_or_else(|| ApiError::invalid("url must be a string"))?
        .to_owned();
    let metadata = Metadata::from_required(&body)?;
    let metadata = metadata
        .payload(metadata.title.clone())
        .map_err(|_| ApiError::invalid("language must be 'en' or 'pl'"))?;
    let id = Uuid::new_v4();
    let mut job = pending(id, &owner, "", Path::new(""), metadata);
    job["source_url"] = json!(url);
    set_status(&mut job, "downloading", "Starting download...");
    let jobs = managed_api::jobs(&api)?.clone();
    save_new(&jobs, id, job)?;
    run_download(api, jobs, id, url);
    Ok(Json(json!({"job_id":id})))
}

pub(crate) fn queued_now() -> Result<String, ApiError> {
    OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .map(|value| value.replace('Z', "+00:00"))
        .map_err(|error| ApiError::internal(error.to_string()))
}

fn safe_job_path(backend: &Path, job_id: &str) -> Result<PathBuf, ApiError> {
    if job_id.is_empty()
        || job_id.contains(['/', '\\', '\0'])
        || !matches!(
            Path::new(job_id).components().next(),
            Some(Component::Normal(_))
        )
    {
        return Err(missing_job());
    }
    Ok(backend.join("outputs").join(job_id))
}

fn run_preparation(api: Api, jobs: Arc<ManagedJobs>, job_id: String) {
    tokio::spawn(async move {
        let Ok(_slot) = api.prep_queue.clone().acquire_owned().await else {
            return;
        };
        match tokio::task::spawn_blocking(move || prepare(&api, &jobs, &job_id)).await {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                log::error!("Unable to store browser preparation result: {}", error.1)
            }
            Err(error) => log::error!("Browser preparation worker failed: {error}"),
        }
    });
}

fn prepare(api: &Api, jobs: &ManagedJobs, job_id: &str) -> Result<(), ApiError> {
    let work = safe_job_path(&api.backend, job_id)?;
    let job = update_job(jobs, job_id, |job| {
        job["queued_at"] = json!("");
        job["work_dir"] = json!(work);
        job["failed_phase"] = json!("phase1");
    })?;
    let source = field_text(&job, "upload_path");
    let prior_audio = field_text(&job, "audio_path");
    let request = json!({
        "source":source,"work":work,"prior_audio":(!prior_audio.is_empty()).then_some(prior_audio),
        "has_video":job["has_video"].as_bool().unwrap_or_default(),
        "real_bpm":job["real_bpm"].as_f64().unwrap_or_default(),
        "metadata":job["metadata"],
    });
    let media: BrowserMedia =
        serde_json::from_value(request).map_err(|error| ApiError::internal(error.to_string()))?;
    let prepared = preparation::prepare_browser_media(&api.backend, media, |progress| {
        if let Err(error) = update_job(jobs, job_id, |job| {
            set_status(
                job,
                field_text(&progress, "status"),
                field_text(&progress, "message"),
            );
        }) {
            log::error!(
                "Unable to store preparation progress for {job_id}: {}",
                error.1
            );
        }
        jobs.emit(job_id, progress);
    });
    let updated = update_job(jobs, job_id, |job| match prepared {
        Ok(media) => {
            for field in [
                "audio_path",
                "vocals_path",
                "instrumental_path",
                "has_video",
                "real_bpm",
                "found_lyrics",
            ] {
                job[field] = media[field].clone();
            }
            job["metadata"]["genre"] = media["metadata"]["genre"].clone();
            job["metadata"]["year"] = media["metadata"]["year"].clone();
            job["failed_phase"] = json!("");
            set_status(job, "awaiting_lyrics", "Review lyrics before transcription");
        }
        Err(error) => {
            job["error"] = json!(error);
            set_status(job, "error", &format!("Error: {error}"));
        }
    })?;
    jobs.emit(job_id, event(&updated));
    Ok(())
}

async fn start(
    State(api): State<Api>,
    owner: Owner,
    RoutePath(job_id): RoutePath<String>,
) -> ApiResult {
    safe_job_path(&api.backend, &job_id)?;
    let jobs = managed_api::jobs(&api)?.clone();
    jobs.mutate(&job_id, |job| {
        let mut job = job
            .filter(|job| managed_api::belongs_to(job, &owner))
            .cloned()
            .ok_or_else(missing_job)?;
        if job["status"] != "pending" {
            return Err(bad_request("Job already started"));
        }
        set_status(&mut job, "extracting_audio", "Queued for processing...");
        job["queued_at"] = json!(queued_now()?);
        Ok((Some(job), ()))
    })?;
    run_preparation(api, jobs, job_id);
    Ok(Json(json!({"status":"started"})))
}

async fn retry(
    State(api): State<Api>,
    owner: Owner,
    RoutePath(job_id): RoutePath<String>,
) -> Result<Response, ApiError> {
    let jobs = managed_api::jobs(&api)?.clone();
    let previous = jobs.job(&owner, &job_id, |job| Ok(job.clone()))?;
    if previous["status"] != "error" {
        return Err(conflict("Only failed songs can be retried"));
    }
    let phase = field_text(&previous, "failed_phase");
    let usable = |field| Path::new(field_text(&previous, field)).is_file();
    if (phase == "publish" && !previous["result"].is_null())
        || (phase == "phase2" && usable("audio_path") && usable("vocals_path"))
    {
        crate::managed_alignment::retry(api, jobs, &owner, job_id, phase)?;
        return Ok(Json(json!({"status":"retrying","phase":phase})).into_response());
    }
    let download = phase == "download";
    let url = field_text(&previous, "source_url").to_owned();
    let youtube_id = if download {
        if url.is_empty() {
            return Err(conflict(
                "The original YouTube URL is unavailable; upload or select the song again",
            ));
        }
        Some(Uuid::parse_str(&job_id).map_err(|_| missing_job())?)
    } else {
        if !usable("upload_path") {
            return Err(conflict(
                "The original audio file is unavailable; upload the song again",
            ));
        }
        safe_job_path(&api.backend, &job_id)?;
        None
    };
    jobs.mutate(&job_id, |job| {
        let mut job = job
            .filter(|job| managed_api::belongs_to(job, &owner))
            .cloned()
            .ok_or_else(missing_job)?;
        if job["status"] != "error" {
            return Err(conflict("Only failed songs can be retried"));
        }
        job["error"] = json!("");
        job["progress_percent"] = json!(0.0);
        if download {
            set_status(&mut job, "downloading", "Retrying YouTube download...");
        } else {
            set_status(
                &mut job,
                "extracting_audio",
                "Retrying from the last completed processing step...",
            );
            job["queued_at"] = json!(queued_now()?);
        }
        Ok((Some(job), ()))
    })?;
    if let Some(id) = youtube_id {
        run_download(api, jobs, id, url);
    } else {
        run_preparation(api, jobs, job_id);
    }
    Ok(
        Json(json!({"status":"retrying","phase":if phase.is_empty() {"phase1"} else {phase}}))
            .into_response(),
    )
}

pub(crate) fn routes() -> Router<Api> {
    Router::new()
        .route(
            "/api/upload",
            post(upload).layer(DefaultBodyLimit::disable()),
        )
        .route("/api/desktop/import", post(register))
        .route("/api/youtube/download", post(download))
        .route("/api/process/{job_id}/start", post(start))
        .route("/api/jobs/{job_id}/retry", post(retry))
}
