use std::{
    collections::{HashMap, HashSet},
    convert::Infallible,
    fs::{File, OpenOptions},
    path::{Path, PathBuf},
    pin::Pin,
    sync::{Arc, RwLock},
    task::{Context, Poll},
    time::Duration,
};

use axum::{
    Json, Router,
    body::{Body, Bytes, HttpBody},
    extract::{Path as RoutePath, Query, Request, State},
    http::{HeaderValue, StatusCode, header},
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::get,
};
use futures_util::stream;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::broadcast;

use crate::{
    api::{Api, ApiError, ApiResult, Owner},
    chart_output::{CompletedChart, chart_name},
    library, library_api,
    managed_state::{self, Snapshot},
};

#[derive(Clone)]
struct Progress {
    job_id: String,
    message: Value,
}

pub(crate) struct ManagedJobs {
    path: PathBuf,
    _writer: File,
    snapshot: RwLock<Snapshot>,
    updates: broadcast::Sender<Progress>,
}

impl ManagedJobs {
    pub(crate) fn load(path: PathBuf) -> Result<Self, String> {
        let writer = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(path.with_extension("lock"))
            .map_err(|error| error.to_string())?;
        writer
            .try_lock()
            .map_err(|error| format!("Unable to lock karaoke job storage: {error}"))?;
        let snapshot = managed_state::restore(&path)?;
        Ok(Self {
            path,
            _writer: writer,
            snapshot: RwLock::new(snapshot),
            updates: broadcast::channel(128).0,
        })
    }

    pub(crate) fn mutate<T>(
        &self,
        id: &str,
        update: impl FnOnce(Option<&Value>) -> Result<(Option<Value>, T), ApiError>,
    ) -> Result<T, ApiError> {
        let mut current = self
            .snapshot
            .write()
            .map_err(|error| ApiError::internal(error.to_string()))?;
        let (job, receipt) = update(current.get(id))?;
        let mut next = current.clone();
        if let Some(job) = job {
            next.insert(id.into(), job);
        } else {
            next.remove(id);
        }
        managed_state::store(&self.path, &next).map_err(ApiError::internal)?;
        *current = next;
        Ok(receipt)
    }

    pub(crate) fn emit(&self, job_id: &str, message: Value) {
        let _ = self.updates.send(Progress {
            job_id: job_id.into(),
            message,
        });
    }

    pub(crate) fn show_unpersisted_error(&self, id: &str, error: &str, message: &str) {
        if let Ok(mut snapshot) = self.snapshot.write()
            && let Some(job) = snapshot.get_mut(id)
        {
            job["status"] = json!("error");
            job["error"] = json!(error);
            job["progress_message"] = json!(message);
            self.emit(
                id,
                json!({"status":"error","error":error,"message":message}),
            );
        }
    }

    pub(crate) fn update_download(&self, job_id: &str, percent: f64) {
        let message = format!("Downloading... {percent:.1}%");
        if let Ok(mut snapshot) = self.snapshot.write() {
            if let Some(job) = snapshot.get_mut(job_id) {
                let previous = job["progress_percent"].as_f64().unwrap_or_default();
                if percent > previous {
                    job["progress_percent"] = json!(percent);
                    job["progress_message"] = json!(message);
                    self.emit(job_id, json!({"status":"downloading","message":message,"progress_percent":percent}));
                }
            }
        }
    }

    pub(crate) fn job<T>(
        &self,
        owner: &Owner,
        job_id: &str,
        read: impl FnOnce(&Value) -> Result<T, ApiError>,
    ) -> Result<T, ApiError> {
        let snapshot = self
            .snapshot
            .read()
            .map_err(|error| ApiError::internal(error.to_string()))?;
        let job = snapshot
            .get(job_id)
            .filter(|job| belongs_to(job, owner))
            .ok_or_else(|| ApiError(StatusCode::NOT_FOUND, "Job not found".into()))?;
        read(job)
    }
}

pub(crate) fn belongs_to(job: &Value, owner: &Owner) -> bool {
    job["owner_id"].as_str().unwrap_or("local") == owner.0
}

pub(crate) fn jobs(api: &Api) -> Result<&Arc<ManagedJobs>, ApiError> {
    api.jobs
        .as_ref()
        .ok_or_else(|| ApiError::internal("Managed state is unavailable"))
}

pub(crate) fn field_text<'a>(job: &'a Value, field: &str) -> &'a str {
    job[field].as_str().unwrap_or_default()
}

fn list_item(job: &Value) -> Value {
    let filename = field_text(job, "original_filename");
    let title = field_text(&job["metadata"], "title");
    let title = if !title.is_empty() {
        title
    } else {
        Path::new(filename)
            .file_stem()
            .and_then(|stem| stem.to_str())
            .filter(|title| !title.is_empty())
            .unwrap_or("Untitled song")
    };
    json!({
        "id": job["id"], "title":title, "artist":field_text(&job["metadata"], "artist"),
        "filename":filename, "status":job["status"], "message":field_text(job, "progress_message"),
        "progress_percent":job["progress_percent"].as_f64().unwrap_or_default(), "error":field_text(job, "error"),
    })
}

async fn list(State(api): State<Api>, owner: Owner) -> ApiResult {
    let snapshot = jobs(&api)?
        .snapshot
        .read()
        .map_err(|error| ApiError::internal(error.to_string()))?;
    let songs: Vec<_> = snapshot
        .values()
        .rev()
        .filter(|job| belongs_to(job, &owner))
        .map(list_item)
        .collect();
    Ok(Json(json!({"songs":songs})))
}

#[derive(Deserialize)]
struct LibraryQuery {
    #[serde(default)]
    managed_only: bool,
}

pub(crate) fn library_payload(
    snapshot: &Snapshot,
    owner: &Owner,
    directory: Option<PathBuf>,
    managed_only: bool,
) -> Value {
    let owned: Vec<_> = snapshot
        .values()
        .filter(|job| belongs_to(job, owner))
        .collect();
    let mut queued: Vec<_> = owned
        .iter()
        .copied()
        .filter(|job| !field_text(job, "queued_at").is_empty())
        .collect();
    queued.sort_by_key(|job| field_text(job, "queued_at"));
    let positions: HashMap<_, _> = queued
        .into_iter()
        .enumerate()
        .map(|(index, job)| (field_text(job, "id"), index + 1))
        .collect();
    let mut published = HashSet::new();
    let mut songs = Vec::with_capacity(owned.len());
    for job in owned {
        let mut song = list_item(job);
        song["kind"] = json!("managed");
        song["failed_phase"] = json!(field_text(job, "failed_phase"));
        song["queue_position"] = json!(positions.get(field_text(job, "id")));
        let library_path = field_text(job, "library_path");
        if !library_path.is_empty() {
            let name = chart_name(
                field_text(&job["metadata"], "artist"),
                field_text(&job["metadata"], "title"),
            );
            let path = Path::new(library_path).join(format!("{name}.txt"));
            published.insert(dunce::canonicalize(&path).unwrap_or(path));
        }
        songs.push(song);
    }
    if !managed_only {
        if let Some(directory) = directory.as_ref().filter(|directory| directory.is_dir()) {
            let root = dunce::canonicalize(directory).unwrap_or_else(|_| directory.clone());
            if let Value::Array(external) =
                json!(library::scan_songs(Some(root.clone())))["songs"].take()
            {
                songs.extend(external.into_iter().filter(|song| {
                    let chart = root.join(field_text(song, "path"));
                    !published.contains(&dunce::canonicalize(&chart).unwrap_or(chart))
                }));
            }
        }
    }
    songs.sort_by_cached_key(|song| {
        (
            song["kind"] != "managed",
            field_text(song, "title").to_lowercase(),
            field_text(song, "artist").to_lowercase(),
        )
    });
    let mut result = json!({"working_directory":directory,"songs":songs});
    if managed_only {
        let root = directory.map(|path| dunce::canonicalize(&path).unwrap_or(path));
        let mut paths: Vec<_> = published
            .iter()
            .filter_map(|path| {
                let relative = path.strip_prefix(root.as_ref()?).ok()?;
                Some(relative.to_string_lossy().replace('\\', "/"))
            })
            .collect();
        paths.sort_unstable();
        result["published_paths"] = json!(paths);
    }
    result
}

async fn list_library(
    State(api): State<Api>,
    owner: Owner,
    Query(query): Query<LibraryQuery>,
) -> ApiResult {
    let directory = library_api::library(&api).directory().ok().flatten();
    let snapshot = jobs(&api)?
        .snapshot
        .read()
        .map_err(|error| ApiError::internal(error.to_string()))?
        .clone();
    let result = tokio::task::spawn_blocking(move || {
        library_payload(&snapshot, &owner, directory, query.managed_only)
    })
    .await
    .map_err(|error| ApiError::internal(error.to_string()))?;
    Ok(Json(result))
}

pub(crate) fn status_payload(job: &Value) -> Value {
    let mut status = json!({
        "status":job["status"],"message":field_text(job, "progress_message"),"metadata":{
            "title":field_text(&job["metadata"], "title"),"artist":field_text(&job["metadata"], "artist"),
        },
    });
    if !field_text(job, "error").is_empty() {
        status["error"] = job["error"].clone();
        status["failed_phase"] = json!(field_text(job, "failed_phase"));
    }
    match job["status"].as_str() {
        Some("awaiting_lyrics") => status["found_lyrics"] = json!(field_text(job, "found_lyrics")),
        Some("downloading") => {
            status["progress_percent"] = json!(job["progress_percent"].as_f64().unwrap_or_default())
        }
        _ => {}
    }
    if let Some(run) = latest_run(job) {
        status["alignment_run"] = run.clone();
    }
    if !field_text(job, "selected_lyric_revision_id").is_empty() {
        status["lyric_revision_id"] = job["selected_lyric_revision_id"].clone();
    }
    let comparison = &job["result"]["alignment_comparison"];
    if comparison
        .as_object()
        .is_some_and(|value| !value.is_empty())
    {
        status["alignment_comparison_status"] = comparison["status"].clone();
    }
    status
}

fn latest_run(job: &Value) -> Option<&Value> {
    job["alignment_runs"]
        .as_array()
        .and_then(|runs| runs.last())
}

async fn status(
    State(api): State<Api>,
    owner: Owner,
    RoutePath(job_id): RoutePath<String>,
) -> ApiResult {
    jobs(&api)?.job(&owner, &job_id, |job| Ok(Json(status_payload(job))))
}

async fn lyrics(
    State(api): State<Api>,
    owner: Owner,
    RoutePath(job_id): RoutePath<String>,
) -> ApiResult {
    jobs(&api)?.job(&owner, &job_id, |job| {
        Ok(Json(json!({"lyrics": field_text(job, "found_lyrics")})))
    })
}

pub(crate) fn result_payload(job: &Value) -> Option<Value> {
    let result = &job["result"];
    if result.is_null() {
        return None;
    }
    let mut payload = result.clone();
    let revision = &result["lyric_revision"];
    let selected = field_text(job, "selected_lyric_revision_id");
    let exported = !revision.is_null() && revision["id"].as_str() == Some(selected);
    payload["selected_lyric_revision_id"] = json!(selected);
    payload["lyric_revisions"] = job["lyric_revisions"]
        .as_array()
        .map_or_else(|| json!([]), |items| json!(items));
    payload["export_ready"] = json!(
        exported
            && job["status"] != "aligning_words"
            && result["alignment_run"]["status"] == "completed"
    );
    payload["pending_revision_id"] = if exported {
        Value::Null
    } else {
        json!(selected)
    };
    Some(payload)
}

async fn result(
    State(api): State<Api>,
    owner: Owner,
    RoutePath(job_id): RoutePath<String>,
) -> ApiResult {
    jobs(&api)?.job(&owner, &job_id, |job| {
        result_payload(job).map(Json).ok_or_else(|| {
            ApiError(
                StatusCode::BAD_REQUEST,
                "Job has no completed result yet".into(),
            )
        })
    })
}

async fn alignment(
    State(api): State<Api>,
    owner: Owner,
    RoutePath(job_id): RoutePath<String>,
) -> ApiResult {
    jobs(&api)?.job(&owner, &job_id, |job| {
        let result = &job["result"];
        if result["lyric_revision"].is_null() {
            return Err(ApiError(StatusCode::BAD_REQUEST, "No alignment result yet".into()));
        }
        let run = &result["alignment_run"];
        Ok(Json(json!({
            "job_id":job["id"],"status":job["status"],"lyric_revision":result["lyric_revision"],
            "word_alignments":result["word_alignments"],"alignment_run":run,
            "last_completed_alignment_run":run,"latest_alignment_run":latest_run(job).unwrap_or(run),
        })))
    })
}

pub(crate) fn progress_snapshot(job: &Value) -> Value {
    json!({"status":job["status"],"message":field_text(job, "progress_message")})
}

async fn events(
    State(api): State<Api>,
    owner: Owner,
    RoutePath(job_id): RoutePath<String>,
) -> Result<Response, ApiError> {
    let jobs = jobs(&api)?.clone();
    let updates = jobs.updates.subscribe();
    let initial = jobs.job(&owner, &job_id, |job| Ok(progress_snapshot(job)))?;
    let events = stream::unfold(
        (
            jobs,
            owner,
            job_id,
            updates,
            api.stopping,
            Some(initial),
            false,
        ),
        |(jobs, owner, job_id, mut updates, mut stopping, initial, finished)| async move {
            if finished || *stopping.borrow() {
                return None;
            }
            let payload = if let Some(initial) = initial {
                initial
            } else {
                loop {
                    let update = tokio::select! {
                        update = updates.recv() => update,
                        _ = stopping.changed() => return None,
                    };
                    match update {
                        Ok(update) if update.job_id == job_id => break update.message,
                        Ok(_) => {}
                        Err(broadcast::error::RecvError::Lagged(_)) => {
                            break jobs
                                .job(&owner, &job_id, |job| Ok(progress_snapshot(job)))
                                .ok()?;
                        }
                        Err(broadcast::error::RecvError::Closed) => return None,
                    }
                }
            };
            let finished = matches!(payload["status"].as_str(), Some("done" | "error"));
            let event = Event::default().event("progress").data(payload.to_string());
            Some((
                Ok::<_, Infallible>(event),
                (jobs, owner, job_id, updates, stopping, None, finished),
            ))
        },
    );
    let mut response = Sse::new(events)
        .keep_alive(
            KeepAlive::new()
                .interval(Duration::from_secs(15))
                .text("keepalive"),
        )
        .into_response();
    response
        .headers_mut()
        .insert("x-accel-buffering", HeaderValue::from_static("no"));
    Ok(response)
}

pub(crate) fn playback_source(job: &Value, track: &str) -> Result<PathBuf, ApiError> {
    if job["result"].is_null() {
        return Err(ApiError(StatusCode::BAD_REQUEST, "No result yet".into()));
    }
    let (asset, pcm) = match track {
        "original" => ("audio_file", "audio.wav"),
        "vocals" => ("vocals_file", "vocals.wav"),
        "instrumental" => ("instrumental_file", "instrumental.wav"),
        _ => {
            return Err(ApiError(
                StatusCode::BAD_REQUEST,
                "Invalid track name".into(),
            ));
        }
    };
    let filename = field_text(&job["result"], asset);
    if filename.is_empty() {
        return Err(ApiError(
            StatusCode::BAD_REQUEST,
            "Invalid track name".into(),
        ));
    }
    let directory = Path::new(field_text(job, "work_dir"));
    let source = safe_asset(directory, filename)?;
    if source
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("mp3"))
    {
        if let Ok(wav) = safe_asset(directory, pcm) {
            return Ok(wav);
        }
    }
    Ok(source)
}

fn safe_asset(directory: &Path, filename: &str) -> Result<PathBuf, ApiError> {
    let missing = || ApiError(StatusCode::NOT_FOUND, "Audio file not found".into());
    if filename.contains(['/', '\\', '\0'])
        || !matches!(
            Path::new(filename).components().next(),
            Some(std::path::Component::Normal(_))
        )
    {
        return Err(ApiError::invalid("Invalid stored filename"));
    }
    let directory = dunce::canonicalize(directory).map_err(|_| missing())?;
    let path = dunce::canonicalize(directory.join(filename)).map_err(|_| missing())?;
    if !path.starts_with(directory) {
        return Err(ApiError::invalid("Stored file escapes the job directory"));
    }
    if !path.is_file() {
        return Err(missing());
    }
    Ok(path)
}

async fn audio(
    State(api): State<Api>,
    owner: Owner,
    RoutePath((job_id, track)): RoutePath<(String, String)>,
    request: Request,
) -> Result<Response, ApiError> {
    let (source, cache) = jobs(&api)?.job(&owner, &job_id, |job| {
        Ok((
            playback_source(job, &track)?,
            Path::new(field_text(job, "work_dir")).join("playback-audio"),
        ))
    })?;
    let path = tokio::task::spawn_blocking(move || library::playback_path(&source, &cache))
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?;
    library_api::send_file(path, request).await
}

#[derive(Deserialize)]
struct ArchiveQuery {
    alignment_mode: Option<String>,
}

fn export_chart(job: &Value, alignment_mode: Option<&str>) -> Result<CompletedChart, ApiError> {
    let result = &job["result"];
    if result.is_null() {
        return Err(ApiError(StatusCode::BAD_REQUEST, "No result yet".into()));
    }
    if job["status"] == "aligning_words" {
        return Err(ApiError(
            StatusCode::CONFLICT,
            "A new alignment is still running; export is temporarily unavailable".into(),
        ));
    }
    if result["lyric_revision"].is_null()
        || result["lyric_revision"]["id"] != job["selected_lyric_revision_id"]
    {
        return Err(ApiError(
            StatusCode::CONFLICT,
            "The selected lyric revision has no completed alignment to export".into(),
        ));
    }
    if result["alignment_run"]["status"] != "completed" {
        return Err(ApiError(
            StatusCode::CONFLICT,
            "The selected result has no completed alignment run to export".into(),
        ));
    }
    let mut chart = result.clone();
    if let Some(text) =
        alignment_mode.and_then(|mode| result["alignment_results"][mode]["ultrastar_txt"].as_str())
    {
        chart["ultrastar_txt"] = json!(text);
    }
    serde_json::from_value(chart).map_err(|error| ApiError::internal(error.to_string()))
}

struct StagedArchive {
    body: Pin<Box<Body>>,
    _directory: tempfile::TempDir,
}

impl HttpBody for StagedArchive {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<Option<Result<http_body::Frame<Bytes>, Self::Error>>> {
        self.body.as_mut().poll_frame(context)
    }

    fn size_hint(&self) -> http_body::SizeHint {
        self.body.size_hint()
    }
}

async fn archive(
    State(api): State<Api>,
    owner: Owner,
    RoutePath(job_id): RoutePath<String>,
    Query(query): Query<ArchiveQuery>,
    request: Request,
) -> Result<Response, ApiError> {
    let (chart, work) = jobs(&api)?.job(&owner, &job_id, |job| {
        Ok((
            export_chart(job, query.alignment_mode.as_deref())?,
            PathBuf::from(field_text(job, "work_dir")),
        ))
    })?;
    let filename = format!("{}.zip", chart.name());
    let (staging, path) = tokio::task::spawn_blocking(move || {
        let staging = tempfile::tempdir().map_err(|error| error.to_string())?;
        let path = staging.path().join("export.zip");
        chart.zip(&work, &path)?;
        Ok::<_, String>((staging, path))
    })
    .await
    .map_err(|error| ApiError::internal(error.to_string()))?
    .map_err(ApiError::internal)?;
    let (mut parts, body) = library_api::send_file(path, request).await?.into_parts();
    parts.headers.insert(
        header::CONTENT_DISPOSITION,
        library_api::download_name(&filename)?,
    );
    Ok(Response::from_parts(
        parts,
        Body::new(StagedArchive {
            body: Box::pin(body),
            _directory: staging,
        }),
    ))
}

pub(crate) fn routes() -> Router<Api> {
    Router::new()
        .route("/api/jobs", get(list))
        .route("/api/library", get(list_library))
        .route("/api/jobs/{job_id}/audio/{track}", get(audio))
        .route("/api/jobs/{job_id}/download", get(archive))
        .route("/api/jobs/{job_id}/status", get(status))
        .route("/api/jobs/{job_id}/found-lyrics", get(lyrics))
        .route("/api/jobs/{job_id}/result", get(result))
        .route("/api/jobs/{job_id}/alignment", get(alignment))
        .route("/api/process/{job_id}", get(events))
}
