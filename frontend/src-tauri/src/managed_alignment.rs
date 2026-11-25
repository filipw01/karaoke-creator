pub(crate) mod engine;

use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use axum::{
    Json, Router,
    extract::{Path as RoutePath, State, rejection::JsonRejection},
    http::StatusCode,
    routing::post,
};
use serde_json::{Value, json};
use tokio::task::spawn_blocking;
use uuid::Uuid;

use crate::{
    api::{self, Api, ApiError, ApiResult, Owner},
    chart_output,
    library::write_atomically,
    library_api,
    managed_api::{self, ManagedJobs, field_text},
    managed_editor::{self, array, projection},
    managed_ingest, media,
};

use engine::{Aligned, Failure};

const DEFAULT_PROMPT: &str = "These are song lyrics with singing.";

fn bad(message: &str) -> ApiError {
    ApiError(StatusCode::BAD_REQUEST, message.into())
}
fn conflict(message: &str) -> ApiError {
    ApiError(StatusCode::CONFLICT, message.into())
}
fn missing() -> ApiError {
    ApiError(StatusCode::NOT_FOUND, "Job not found".into())
}

impl From<ApiError> for Failure {
    fn from(error: ApiError) -> Self {
        error.1.into()
    }
}

fn claim<T>(
    jobs: &ManagedJobs,
    owner: &Owner,
    id: &str,
    update: impl FnOnce(&mut Value) -> Result<T, ApiError>,
) -> Result<T, ApiError> {
    jobs.mutate(id, |job| {
        let mut job = job
            .filter(|job| managed_api::belongs_to(job, owner))
            .cloned()
            .ok_or_else(missing)?;
        let result = update(&mut job)?;
        Ok((Some(job), result))
    })
}

fn update<T>(
    jobs: &ManagedJobs,
    id: &str,
    change: impl FnOnce(&mut Value) -> Result<T, ApiError>,
) -> Result<T, ApiError> {
    jobs.mutate(id, |job| {
        let mut job = job.cloned().ok_or_else(missing)?;
        let receipt = change(&mut job)?;
        Ok((Some(job), receipt))
    })
}

fn selected(job: &Value) -> Option<&Value> {
    revision(job, field_text(job, "selected_lyric_revision_id"))
}

fn revision<'a>(job: &'a Value, id: &str) -> Option<&'a Value> {
    array(&job["lyric_revisions"])
        .iter()
        .find(|revision| revision["id"] == id)
}

fn add_revision(
    job: &mut Value,
    text: &str,
    source: &str,
    parent: Option<&str>,
    language: &str,
) -> Result<Value, ApiError> {
    let revision = managed_editor::create_revision(
        field_text(job, "id"),
        None,
        text,
        language,
        source,
        parent,
    )?;
    let mut revisions = array(&job["lyric_revisions"]).to_vec();
    revisions.push(revision.clone());
    job["lyric_revisions"] = json!(revisions);
    job["selected_lyric_revision_id"] = revision["id"].clone();
    Ok(revision)
}

fn register_run(job: &mut Value, revision_id: &str, lane: &str) -> Result<Value, ApiError> {
    let run = json!({
        "id":Uuid::new_v4(), "lyric_revision_id":revision_id, "status":"running",
        "selected_strategy":engine::strategy(lane), "diagnostics":{},"created_at":managed_ingest::queued_now()?,
    });
    set_run(job, &run);
    Ok(run)
}

fn set_run(job: &mut Value, run: &Value) {
    let mut runs = array(&job["alignment_runs"]).to_vec();
    if let Some(previous) = runs.iter_mut().find(|previous| previous["id"] == run["id"]) {
        *previous = run.clone();
    } else {
        runs.push(run.clone());
    }
    job["alignment_runs"] = json!(runs);
}

fn status(job: &mut Value, status: &str, message: &str) {
    job["status"] = json!(status);
    job["progress_message"] = json!(message);
}

fn emit(jobs: &ManagedJobs, id: &str, job: &Value) {
    let mut event = managed_api::progress_snapshot(job);
    if !field_text(job, "error").is_empty() {
        event["error"] = job["error"].clone();
    }
    jobs.emit(id, event);
}

fn advance(jobs: &ManagedJobs, id: &str, state: &str, message: &str) -> Result<Value, ApiError> {
    let job = update(jobs, id, |job| {
        status(job, state, message);
        Ok(job.clone())
    })?;
    emit(jobs, id, &job);
    Ok(job)
}

fn finish_run(run: &mut Value, aligned: &Aligned) {
    run["status"] = json!("completed");
    run["selected_strategy"] = json!(engine::strategy(&aligned.lane));
    run["diagnostics"] = aligned.diagnostics.clone();
}

fn report_failure(
    jobs: &ManagedJobs,
    id: &str,
    run: Option<&Value>,
    failure: Failure,
    label: &str,
) {
    let message = format!("{label}: {}", failure.error);
    let result = update(jobs, id, |job| {
        if let Some(run) = run.filter(|run| run["status"] == "running") {
            let mut run = run.clone();
            run["status"] = json!("failed");
            run["diagnostics"] = failure.diagnostics;
            set_run(job, &run);
        }
        job["error"] = json!(failure.error);
        status(job, "error", &message);
        Ok(job.clone())
    });
    match result {
        Ok(job) => emit(jobs, id, &job),
        Err(error) => {
            log::error!("Unable to store alignment failure for {id}: {}", error.1);
            jobs.show_unpersisted_error(id, &failure.error, &message);
        }
    }
}

async fn initial(api: Api, jobs: Arc<ManagedJobs>, id: String, lyrics: String) {
    let Ok(_permit) = api.prep_queue.clone().acquire_owned().await else {
        return;
    };
    let backend = api.backend.clone();
    let slot = spawn_blocking(move || library_api::acquire_processing_slot(&backend)).await;
    let _slot = match slot {
        Ok(Ok(slot)) => slot,
        Ok(Err(error)) => {
            report_failure(&jobs, &id, None, error.into(), "Error");
            return;
        }
        Err(error) => {
            report_failure(&jobs, &id, None, error.to_string().into(), "Error");
            return;
        }
    };
    let mut run = Value::Null;
    let result = initial_alignment(&api, &jobs, &id, &lyrics, &mut run).await;
    match result {
        Ok(()) => {}
        Err(error) => report_failure(&jobs, &id, (!run.is_null()).then_some(&run), error, "Error"),
    }
}

async fn initial_alignment(
    api: &Api,
    jobs: &Arc<ManagedJobs>,
    id: &str,
    lyrics: &str,
    run: &mut Value,
) -> Result<(), Failure> {
    let has_lyrics = !lyrics.trim().is_empty() && lyrics.trim() != DEFAULT_PROMPT;
    let message = if has_lyrics {
        "Aligning provided lyrics to audio..."
    } else {
        "Transcribing lyrics with word timestamps..."
    };
    let job = update(jobs, id, |job| {
        job["queued_at"] = json!("");
        job["failed_phase"] = json!("phase2");
        status(job, "extracting_lyrics", message);
        let revision_id = selected(job).map_or_else(String::new, |revision| {
            field_text(revision, "id").to_owned()
        });
        *run = register_run(job, &revision_id, "sofa")?;
        Ok(job.clone())
    })?;
    emit(jobs, id, &job);
    let work = PathBuf::from(field_text(&job, "work_dir"));
    let vocals = PathBuf::from(field_text(&job, "vocals_path"));
    let normalized = work.join("vocals_normalized.wav");
    let normalization = normalized.clone();
    spawn_blocking(move || media::normalize_audio(&vocals, &normalization))
        .await
        .map_err(|error| error.to_string())??;
    let job = advance(jobs, id, "detecting_pitch", "Analyzing vocal pitch...")?;
    let mut inputs = engine::load(api, &job, None).await?;
    inputs.transcription = normalized;
    let language =
        managed_ingest::language(field_text(&job["metadata"], "language")).unwrap_or("en");
    let (text, fallback) = if has_lyrics {
        (lyrics.trim().to_owned(), Vec::new())
    } else {
        engine::draft(api, &job, run, language, &inputs).await?
    };
    let duration = inputs.duration;
    let (job, revision) = update(jobs, id, |job| {
        let current = selected(job).cloned();
        let revision = if has_lyrics
            && current
                .as_ref()
                .is_some_and(|revision| revision["raw_text"] == text)
        {
            current.unwrap()
        } else {
            let parent = current
                .as_ref()
                .and_then(|revision| revision["id"].as_str());
            let revision = add_revision(
                job,
                &text,
                if has_lyrics { "manual" } else { "asr_draft" },
                if has_lyrics { parent } else { None },
                language,
            )?;
            if !has_lyrics
                && std::env::var("ALIGNMENT_WORKER_URL")
                    .ok()
                    .is_some_and(|url| !url.trim().is_empty())
            {
                job["confirmed_lyrics"] = json!(text);
            }
            revision
        };
        run["lyric_revision_id"] = revision["id"].clone();
        let lane = engine::resolve_backend(field_text(&revision, "language"), "sofa");
        run["selected_strategy"] = json!(engine::strategy(&lane));
        set_run(job, run);
        status(
            job,
            "aligning_words",
            "Aligning canonical lyric words to both audio sources...",
        );
        Ok((job.clone(), revision))
    })?;
    emit(jobs, id, &job);
    let lane = engine::resolve_backend(field_text(&revision, "language"), "sofa");
    let aligned = engine::align(
        api,
        &job,
        &revision,
        run,
        inputs,
        &lane,
        &fallback,
        &[],
        true,
    )
    .await?;
    finish_run(run, &aligned);
    let generating = update(jobs, id, |job| {
        set_run(job, run);
        status(job, "generating", "Generating UltraStar song file...");
        Ok(job.clone())
    })?;
    emit(jobs, id, &generating);
    let generated =
        generate_files(job, revision.clone(), run.clone(), aligned, work, duration).await?;
    let job = update(jobs, id, |job| {
        set_run(job, run);
        job["metadata"] = generated["metadata"].clone();
        job["result"] = generated["result"].clone();
        job["failed_phase"] = json!("publish");
        Ok(job.clone())
    })?;
    let api = api.clone();
    finish_publication(api, jobs.clone(), id.to_owned(), job).await?;
    Ok(())
}

async fn generate_files(
    mut job: Value,
    revision: Value,
    run: Value,
    aligned: Aligned,
    work: PathBuf,
    duration: f64,
) -> Result<Value, Failure> {
    spawn_blocking(move || {
        let name = chart_output::chart_name(field_text(&job["metadata"],"artist"),field_text(&job["metadata"],"title"));
        let audio = format!("{name}.mp3");
        let vocals = format!("{name} [VOC].mp3");
        let instrumental = format!("{name} [INSTR].mp3");
        let video = if job["has_video"] == true { format!("{name}.mp4") } else { String::new() };
        job["result"] = json!({
            "audio_file":audio,"vocals_file":vocals,"instrumental_file":instrumental,"video_file":video,
            "note_overrides":{},"alignment_comparison":{},"audio_duration":duration,
        });
        projection::apply_canonical(&mut job,&revision,&run,&aligned.alignments,&aligned.canonical,&aligned.pitch,&aligned.breaks)?;
        let instrumental_source = PathBuf::from(field_text(&job,"instrumental_path"));
        let instrumental_source = if instrumental_source.as_os_str().is_empty() { work.join("instrumental.wav") } else { instrumental_source };
        media::encode_output(Path::new(field_text(&job,"audio_path")),&work.join(audio),false)?;
        media::encode_output(Path::new(field_text(&job,"vocals_path")),&work.join(vocals),false)?;
        media::encode_output(&instrumental_source,&work.join(instrumental),false)?;
        if !video.is_empty() {
            media::encode_output(Path::new(field_text(&job,"upload_path")),&work.join(video),true)?;
        }
        write_atomically(&work.join(format!("{name}.txt")),field_text(&job["result"],"ultrastar_txt").as_bytes())
            .map_err(|error| error.to_string())?;
        Ok(job)
    }).await.map_err(|error| error.to_string())?
}

async fn finish_publication(
    api: Api,
    jobs: Arc<ManagedJobs>,
    id: String,
    mut job: Value,
) -> Result<(), Failure> {
    let started = spawn_blocking({
        let api = api.clone();
        move || {
            managed_editor::write_chart_and_publish(&api, &mut job, false)?;
            Ok::<_, ApiError>(job)
        }
    })
    .await
    .map_err(|error| error.to_string())??;
    let (job, comparison) = update(&jobs, &id, |job| {
        job["library_path"] = started["library_path"].clone();
        job["library_root"] = started["library_root"].clone();
        job["error"] = json!("");
        job["failed_phase"] = json!("");
        status(job, "done", "Done! Your karaoke song is ready.");
        let result = &mut job["result"];
        let comparison = result["lyric_revision"]["language"] == "en"
            && field_text(&result["alignment_run"], "selected_strategy").contains("sofa")
            && array_or_object_empty(&result["alignment_comparison"]);
        let revision = if comparison {
            result["alignment_comparison"] = comparing();
            Some(result["lyric_revision"].clone())
        } else {
            None
        };
        Ok((job.clone(), revision))
    })?;
    emit(&jobs, &id, &job);
    if let Some(revision) = comparison {
        schedule_comparison(
            api,
            jobs,
            id,
            revision,
            job["result"]["alignment_run"]["id"].clone(),
        );
    }
    Ok(())
}

fn array_or_object_empty(value: &Value) -> bool {
    value.is_null() || value.as_object().is_some_and(|obj| obj.is_empty())
}

pub(crate) fn retry(
    api: Api,
    jobs: Arc<ManagedJobs>,
    owner: &Owner,
    id: String,
    phase: &str,
) -> Result<(), ApiError> {
    let job = claim(&jobs, owner, &id, |job| {
        if job["status"] != "error" {
            return Err(conflict("Only failed songs can be retried"));
        }
        job["error"] = json!("");
        job["progress_percent"] = json!(0.0);
        if phase == "publish" {
            status(job, "generating", "Publishing the finished song…");
        } else {
            status(
                job,
                "extracting_lyrics",
                "Retrying from lyric extraction...",
            );
            job["queued_at"] = json!(managed_ingest::queued_now()?);
        }
        Ok(job.clone())
    })?;
    if phase == "publish" {
        tokio::spawn(async move {
            if let Err(error) = finish_publication(api, jobs.clone(), id.clone(), job).await {
                report_failure(&jobs, &id, None, error, "Unable to publish song");
            }
        });
    } else {
        let lyrics = field_text(&job, "confirmed_lyrics").to_owned();
        tokio::spawn(initial(api, jobs, id, lyrics));
    }
    Ok(())
}

struct Realign {
    revision: Value,
    run: Value,
    lane: String,
    resets: Vec<String>,
    normalize: bool,
}

fn check_realign(job: &Value) -> Result<(), ApiError> {
    if job["result"]["lyric_revision"].is_null() {
        return Err(bad("No initial alignment result"));
    }
    if !Path::new(field_text(job, "vocals_path")).is_file() {
        return Err(bad("Vocals file not available for alignment"));
    }
    if !matches!(field_text(job, "status"), "done" | "error") {
        return Err(conflict("Another alignment is already running"));
    }
    Ok(())
}

fn reset_tokens(body: &Value, revision: &Value) -> Result<Vec<String>, ApiError> {
    let Some(values) = body.get("reset_token_ids") else {
        return Ok(Vec::new());
    };
    let values: Vec<String> = serde_json::from_value(values.clone())
        .map_err(|_| ApiError::invalid("reset_token_ids must be an array of strings"))?;
    let tokens = array(&revision["tokens"]);
    if values
        .iter()
        .any(|value| !tokens.iter().any(|token| token["id"] == *value))
    {
        return Err(ApiError::invalid("Reset contains unknown token IDs"));
    }
    Ok(values)
}

fn claim_realign(
    job: &mut Value,
    revision: Value,
    lane: &str,
    resets: Vec<String>,
    normalize: bool,
    message: &str,
) -> Result<Realign, ApiError> {
    let lane = engine::resolve_backend(field_text(&revision, "language"), lane);
    let run = register_run(job, field_text(&revision, "id"), &lane)?;
    job["selected_lyric_revision_id"] = revision["id"].clone();
    status(job, "aligning_words", message);
    Ok(Realign {
        revision,
        run,
        lane,
        resets,
        normalize,
    })
}

async fn realign(
    api: &Api,
    jobs: &Arc<ManagedJobs>,
    id: &str,
    request: Realign,
) -> Result<Value, ApiError> {
    let Realign {
        revision,
        run,
        lane,
        resets,
        normalize,
    } = request;
    let owner_job = update(jobs, id, |job| Ok(job.clone()))?;
    emit(jobs, id, &owner_job);
    let result = async {
        let normalization = normalize.then(|| {
            Path::new(field_text(&owner_job, "work_dir")).join(format!(
                "alignment-{}-normalized.wav",
                field_text(&run, "id")
            ))
        });
        let inputs = engine::load(api, &owner_job, normalization).await?;
        let aligned = engine::align(
            api,
            &owner_job,
            &revision,
            &run,
            inputs,
            &lane,
            &[],
            &resets,
            true,
        )
        .await?;
        let api = api.clone();
        let stored = jobs.clone();
        let id = id.to_owned();
        let mut completed = run.clone();
        finish_run(&mut completed, &aligned);
        let job = spawn_blocking(move || {
            update(&stored, &id, |job| {
                projection::apply_canonical(
                    job,
                    &revision,
                    &completed,
                    &aligned.alignments,
                    &aligned.canonical,
                    &aligned.pitch,
                    &aligned.breaks,
                )?;
                managed_editor::write_chart_and_publish(&api, job, true)?;
                set_run(job, &completed);
                job["error"] = json!("");
                status(job, "done", "Re-alignment complete.");
                Ok(job.clone())
            })
        })
        .await
        .map_err(|error| error.to_string())??;
        Ok::<_, Failure>(job)
    }
    .await;
    match result {
        Ok(job) => {
            emit(jobs, id, &job);
            Ok(job)
        }
        Err(error) => {
            let detail = error.error.clone();
            report_failure(jobs, id, Some(&run), error, "Alignment error");
            Err(ApiError::internal(detail))
        }
    }
}

async fn confirm(
    State(api): State<Api>,
    owner: Owner,
    RoutePath(id): RoutePath<String>,
    body: Result<Json<Value>, JsonRejection>,
) -> ApiResult {
    let body = api::parse_body(body)?;
    let lyrics = body
        .get("lyrics")
        .map_or(Some(""), Value::as_str)
        .ok_or_else(|| ApiError::invalid("lyrics must be a string"))?
        .trim()
        .to_owned();
    let jobs = managed_api::jobs(&api)?.clone();
    let result = claim(&jobs, &owner, &id, |job| {
        if job["status"] == "awaiting_lyrics" {
            if !lyrics.is_empty() {
                let current = selected(job).cloned();
                if !current
                    .as_ref()
                    .is_some_and(|revision| revision["raw_text"] == lyrics)
                {
                    let parent = current
                        .as_ref()
                        .and_then(|revision| revision["id"].as_str());
                    let source = if lyrics == field_text(job, "found_lyrics").trim() {
                        "online"
                    } else {
                        "manual"
                    };
                    let language = field_text(&job["metadata"], "language").to_owned();
                    add_revision(job, &lyrics, source, parent, &language)?;
                }
            }
            job["confirmed_lyrics"] = json!(lyrics);
            status(job, "extracting_lyrics", "Preparing lyric evidence...");
            job["queued_at"] = json!(managed_ingest::queued_now()?);
            return Ok(None);
        }
        if lyrics.is_empty() {
            return Err(ApiError::invalid("lyrics cannot be empty"));
        }
        if job["result"].is_null() || !matches!(field_text(job, "status"), "done" | "error") {
            return Err(bad("Job is not ready for lyrics confirmation"));
        }
        let current = job["result"]["lyric_revision"].clone();
        let revision = if current["raw_text"] == lyrics {
            current
        } else {
            let language = if current.is_null() {
                field_text(&job["metadata"], "language").to_owned()
            } else {
                field_text(&current, "language").to_owned()
            };
            add_revision(job, &lyrics, "manual", current["id"].as_str(), &language)?
        };
        check_realign(job)?;
        Ok(Some(claim_realign(
            job,
            revision,
            "sofa",
            Vec::new(),
            false,
            "Re-aligning the selected lyric revision...",
        )?))
    })?;
    if let Some(request) = result {
        let run = request.run["id"].clone();
        tokio::spawn(async move {
            let _ = realign(&api, &jobs, &id, request).await;
        });
        Ok(Json(json!({"status":"started","alignment_run_id":run})))
    } else {
        tokio::spawn(initial(api, jobs, id, lyrics));
        Ok(Json(json!({"status":"started"})))
    }
}

async fn start_run(
    State(api): State<Api>,
    owner: Owner,
    RoutePath(id): RoutePath<String>,
    body: Result<Json<Value>, JsonRejection>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    let body = api::parse_body(body)?;
    let jobs = managed_api::jobs(&api)?.clone();
    let request = claim(&jobs, &owner, &id, |job| {
        check_realign(job)?;
        let lane = body
            .get("alignment_backend")
            .map_or(Some("sofa"), Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_lowercase();
        if !matches!(lane.as_str(), "sofa" | "legacy") {
            return Err(ApiError::invalid(
                "alignment_backend must be sofa or legacy",
            ));
        }
        let revision_id = body["lyric_revision_id"]
            .as_str()
            .filter(|id| !id.trim().is_empty())
            .ok_or_else(|| ApiError::invalid("lyric_revision_id must be a non-empty string"))?;
        let revision = revision(job, revision_id)
            .cloned()
            .ok_or_else(|| conflict("Lyric revision is stale or unknown"))?;
        let resets = reset_tokens(&body, &revision)?;
        claim_realign(
            job,
            revision,
            &lane,
            resets,
            false,
            "Re-aligning the selected lyric revision...",
        )
    })?;
    let receipt = json!({"alignment_run_id":request.run["id"],"lyric_revision_id":request.revision["id"],"status":"running"});
    tokio::spawn(async move {
        let _ = realign(&api, &jobs, &id, request).await;
    });
    Ok((StatusCode::ACCEPTED, Json(receipt)))
}

async fn regenerate(
    State(api): State<Api>,
    owner: Owner,
    RoutePath(id): RoutePath<String>,
    body: Result<Json<Value>, JsonRejection>,
) -> ApiResult {
    let body = api::parse_body(body)?;
    let jobs = managed_api::jobs(&api)?.clone();
    let request = claim(&jobs, &owner, &id, |job| {
        if job["result"]["lyric_revision"].is_null() {
            return Err(bad("No alignment result to regenerate"));
        }
        check_realign(job)?;
        let current = job["result"]["lyric_revision"].clone();
        let prompt = body
            .get("initial_prompt")
            .map_or(Some(DEFAULT_PROMPT), Value::as_str)
            .ok_or_else(|| ApiError::invalid("initial_prompt must be a string"))?
            .trim();
        let revision =
            if !prompt.is_empty() && prompt != DEFAULT_PROMPT && current["raw_text"] != prompt {
                add_revision(
                    job,
                    prompt,
                    "manual",
                    current["id"].as_str(),
                    field_text(&current, "language"),
                )?
            } else {
                current
            };
        let requested = body
            .get("language")
            .filter(|language| !language.is_null())
            .map_or(Some(""), Value::as_str)
            .ok_or_else(|| ApiError::invalid("language must be 'en' or 'pl'"))?;
        managed_ingest::language(if requested.is_empty() {
            field_text(&revision, "language")
        } else {
            requested
        })
        .map_err(|_| ApiError::invalid("language must be 'en' or 'pl'"))?;
        let audio = body
            .get("audio_source")
            .map_or(Some("original"), Value::as_str)
            .unwrap_or_default();
        if !matches!(audio, "original" | "vocals") {
            return Err(ApiError::invalid("audio_source must be original or vocals"));
        }
        let resets = reset_tokens(&body, &revision)?;
        claim_realign(
            job,
            revision,
            "sofa",
            resets,
            body.get("normalize")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            "Regenerating canonical lyric alignment...",
        )
    })?;
    // Own the task even if the client disconnects while awaiting a result.
    let result = tokio::spawn(async move { realign(&api, &jobs, &id, request).await })
        .await
        .map_err(|error| ApiError::internal(error.to_string()))??;
    Ok(Json(managed_api::result_payload(&result).ok_or_else(
        || ApiError::internal("Lyrics regeneration failed"),
    )?))
}

fn comparing() -> Value {
    json!({"status":"running","sofa_available":true,"legacy_available":false})
}

fn schedule_comparison(api: Api, jobs: Arc<ManagedJobs>, id: String, revision: Value, base: Value) {
    tokio::spawn(async move {
        let attempt = async {
            let (started, mut run) = update(&jobs, &id, |job| {
                if job["result"]["alignment_run"]["id"] != base {
                    return Err(conflict("Comparison was superseded"));
                }
                let run = register_run(job, field_text(&revision, "id"), "legacy")?;
                Ok((job.clone(), run))
            })?;
            let aligned = async {
                let inputs = engine::load(&api, &started, None).await?;
                engine::align(
                    &api,
                    &started,
                    &revision,
                    &run,
                    inputs,
                    "legacy",
                    &[],
                    &[],
                    false,
                )
                .await
            }
            .await;
            match aligned {
                Ok(aligned) => {
                    finish_run(&mut run, &aligned);
                    let job = update(&jobs, &id, |job| {
                        set_run(job, &run);
                        if job["result"]["alignment_run"]["id"] != base {
                            return Ok(job.clone());
                        }
                        let legacy = projection::as_variant(&projection::build_variant(
                            job,
                            &revision,
                            &run,
                            &aligned.alignments,
                            &aligned.canonical,
                            &aligned.pitch,
                            &aligned.breaks,
                            true,
                        )?);
                        let key = managed_editor::variant_key(&job["result"], "sofa");
                        let sofa = job["result"]["alignment_results"][key].clone();
                        if sofa.is_null() {
                            return Err(conflict("SOFA result is unavailable for comparison"));
                        }
                        let comparison = projection::compare(job, &sofa, &legacy);
                        job["result"]["alignment_results"]["sofa"] = sofa;
                        job["result"]["alignment_results"]["legacy"] = legacy;
                        job["result"]["alignment_comparison"] = comparison;
                        Ok(job.clone())
                    })?;
                    emit(&jobs, &id, &job);
                }
                Err(failure) => {
                    run["status"] = json!("failed");
                    run["diagnostics"] = failure.diagnostics;
                    comparison_failure(&jobs, &id, &base, &failure.error, Some(&run));
                }
            }
            Ok::<_, ApiError>(())
        }
        .await;
        if let Err(error) = attempt {
            comparison_failure(&jobs, &id, &base, &error.1, None);
        }
    });
}

fn comparison_failure(
    jobs: &ManagedJobs,
    id: &str,
    base: &Value,
    error: &str,
    run: Option<&Value>,
) {
    if let Ok(job) = update(jobs, id, |job| {
        if let Some(run) = run {
            set_run(job, run);
        }
        if job["result"]["alignment_run"]["id"] == *base {
            let variants = &job["result"]["alignment_results"];
            job["result"]["alignment_comparison"] = json!({
                "status":"failed","sofa_available":!variants["sofa"].is_null()||!variants["canonical"].is_null(),
                "legacy_available":false,"error":error,
            });
        }
        Ok(job.clone())
    }) {
        emit(jobs, id, &job);
    }
}

async fn compare(
    State(api): State<Api>,
    owner: Owner,
    RoutePath(id): RoutePath<String>,
    body: Option<Json<Value>>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    let body = body.map_or(Value::Null, |Json(body)| body);
    let jobs = managed_api::jobs(&api)?.clone();
    let (comparison, task) = claim(&jobs, &owner, &id, |job| {
        let result = &job["result"];
        let current = result["lyric_revision"].clone();
        if current.is_null() {
            return Err(bad("No initial alignment result"));
        }
        if job["status"] == "aligning_words" {
            return Err(conflict("Another alignment is already running"));
        }
        let comparison = result["alignment_comparison"].clone();
        if comparison["status"] == "running" {
            return Ok((
                json!({"status":"running","alignment_comparison":comparison}),
                None,
            ));
        }
        if !result["alignment_results"]["legacy"].is_null() {
            return Ok((
                json!({"status":"completed","alignment_comparison":comparison}),
                None,
            ));
        }
        let run = &result["alignment_run"];
        let language = managed_ingest::language(field_text(&current, "language")).unwrap_or("en");
        if language != "en" || !field_text(run, "selected_strategy").contains("sofa") {
            let unsupported = json!({"status":"not_supported","sofa_available":false,"legacy_available":true,
                "preferred_mode":"legacy","reason":"SOFA comparison is only enabled for English lyrics"});
            job["result"]["alignment_comparison"] = unsupported.clone();
            return Ok((
                json!({"status":"not_supported","alignment_comparison":unsupported}),
                None,
            ));
        }
        let key = managed_editor::variant_key(result, "sofa");
        let sofa = result["alignment_results"][key].clone();
        if sofa.is_null() {
            return Err(conflict("SOFA result is unavailable for comparison"));
        }
        let base = run["id"].clone();
        let revision = revision(job, field_text(&body, "lyric_revision_id"))
            .cloned()
            .unwrap_or(current);
        job["result"]["alignment_results"]["sofa"] = sofa;
        let comparison = comparing();
        job["result"]["alignment_comparison"] = comparison.clone();
        Ok((
            json!({"status":"running","alignment_comparison":comparison}),
            Some((revision, base)),
        ))
    })?;
    if let Some((revision, base)) = task {
        schedule_comparison(api, jobs, id, revision, base);
    }
    Ok((StatusCode::ACCEPTED, Json(comparison)))
}

pub(crate) fn routes() -> Router<Api> {
    Router::new()
        .route("/api/jobs/{job_id}/confirm-lyrics", post(confirm))
        .route("/api/jobs/{job_id}/alignment-runs", post(start_run))
        .route("/api/jobs/{job_id}/alignment-comparison", post(compare))
        .route("/api/jobs/{job_id}/regenerate-lyrics", post(regenerate))
}
