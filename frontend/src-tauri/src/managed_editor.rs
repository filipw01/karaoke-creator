pub(crate) mod projection;

use std::{collections::HashSet, path::Path};

use axum::{
    Json, Router,
    extract::{Path as RoutePath, State, rejection::JsonRejection},
    http::StatusCode,
    routing::{patch, post, put},
};
use serde_json::{Map, Value, json};
use unicode_general_category::{GeneralCategory, get_general_category};
use unicode_normalization::UnicodeNormalization;
use uuid::Uuid;

use crate::{
    api::{self, Api, ApiError, ApiResult, Owner},
    chart_output::{self, Assets, CompletedChart},
    external_chart::{Metadata, Phrase},
    library::write_atomically,
    library_api,
    managed_api::{self, field_text},
    managed_ingest,
};

pub(crate) fn array(value: &Value) -> &[Value] {
    value.as_array().map_or(&[], Vec::as_slice)
}

fn invalid(message: &str) -> ApiError {
    ApiError::invalid(message)
}

fn bad_request(message: &str) -> ApiError {
    ApiError(StatusCode::BAD_REQUEST, message.into())
}

fn conflict(message: &str) -> ApiError {
    ApiError(StatusCode::CONFLICT, message.into())
}

fn find_revision<'a>(job: &'a Value, id: &str) -> Option<&'a Value> {
    array(&job["lyric_revisions"])
        .iter()
        .find(|revision| revision["id"] == id)
}

fn normalize_word(text: &str) -> String {
    let normalized: String = text.nfkc().collect();
    caseless::default_case_fold_str(&normalized)
        .chars()
        .filter(|character| {
            matches!(
                get_general_category(*character),
                GeneralCategory::UppercaseLetter
                    | GeneralCategory::LowercaseLetter
                    | GeneralCategory::TitlecaseLetter
                    | GeneralCategory::ModifierLetter
                    | GeneralCategory::OtherLetter
                    | GeneralCategory::DecimalNumber
                    | GeneralCategory::LetterNumber
                    | GeneralCategory::OtherNumber
                    | GeneralCategory::NonspacingMark
                    | GeneralCategory::SpacingMark
                    | GeneralCategory::EnclosingMark
            )
        })
        .collect()
}

pub(crate) fn tokenize_lyrics(raw_text: &str, revision_id: &str) -> Vec<Value> {
    let lines = raw_text.replace("\r\n", "\n");
    let mut source_index = 0;
    let mut tokens = Vec::new();
    for (line_index, line) in lines
        .split([
            '\n', '\r', '\x0b', '\x0c', '\x1c', '\x1d', '\x1e', '\u{85}', '\u{2028}', '\u{2029}',
        ])
        .enumerate()
    {
        for (token_index, display_text) in line
            .split(|character: char| {
                character.is_whitespace() || matches!(character, '\x1c'..='\x1f')
            })
            .filter(|text| !text.is_empty())
            .enumerate()
        {
            let normalized_text = normalize_word(display_text);
            if !normalized_text.is_empty() {
                tokens.push(json!({
                    "id":format!("{revision_id}:token:{source_index}"),"lyric_revision_id":revision_id,
                    "display_text":display_text,"normalized_text":normalized_text,
                    "line_index":line_index,"token_index":token_index,"source_index":source_index,
                }));
            }
            source_index += 1;
        }
    }
    tokens
}

pub(crate) fn create_revision(
    job_id: &str,
    revision_id: Option<&str>,
    raw_text: &str,
    language: &str,
    source: &str,
    parent: Option<&str>,
) -> Result<Value, ApiError> {
    let revision_id = revision_id.map_or_else(|| Uuid::new_v4().to_string(), str::to_owned);
    let language = managed_ingest::language(language)?;
    Ok(json!({
        "id":revision_id,"song_id":job_id,"language":language,"source":source,"raw_text":raw_text,
        "parent_revision_id":parent,"created_at":managed_ingest::queued_now()?,
        "tokens":tokenize_lyrics(raw_text,&revision_id),
    }))
}

async fn mutate(
    api: Api,
    owner: Owner,
    job_id: String,
    update: impl FnOnce(&Api, &mut Value) -> Result<Value, ApiError> + Send + 'static,
) -> ApiResult {
    let jobs = managed_api::jobs(&api)?.clone();
    tokio::task::spawn_blocking(move || {
        jobs.mutate(&job_id, |job| {
            let mut job = job
                .filter(|job| managed_api::belongs_to(job, &owner))
                .cloned()
                .ok_or_else(|| ApiError(StatusCode::NOT_FOUND, "Job not found".into()))?;
            let receipt = update(&api, &mut job)?;
            Ok((Some(job), Json(receipt)))
        })
    })
    .await
    .map_err(|error| ApiError::internal(error.to_string()))?
}

async fn save_revision(
    State(api): State<Api>,
    owner: Owner,
    RoutePath(id): RoutePath<String>,
    body: Result<Json<Value>, JsonRejection>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    let body = api::parse_body(body)?;
    let response = mutate(api, owner, id, move |_, job| {
        if job["status"] == "aligning_words" {
            return Err(conflict("Cannot revise lyrics during alignment"));
        }
        let current = &job["result"]["lyric_revision"];
        if current.is_null() {
            return Err(bad_request(
                "Lyrics can only be revised after an alignment result",
            ));
        }
        let raw_text = body
            .get("raw_text")
            .or_else(|| body.get("lyrics"))
            .map_or(Some(""), Value::as_str)
            .ok_or_else(|| invalid("raw_text must be a string"))?;
        if raw_text.trim().is_empty() {
            return Err(invalid("Lyric revision cannot be empty"));
        }
        let language = match body.get("language") {
            None | Some(Value::Null) => field_text(current, "language"),
            Some(Value::String(language)) if language.is_empty() => field_text(current, "language"),
            Some(Value::String(language)) => language,
            _ => return Err(invalid("Word alignment language must be a string")),
        };
        let parent =
            find_revision(job, field_text(job, "selected_lyric_revision_id")).unwrap_or(current);
        let revision = create_revision(
            field_text(job, "id"),
            None,
            raw_text,
            language,
            "manual",
            parent["id"].as_str(),
        )?;
        let mut revisions = array(&job["lyric_revisions"]).to_vec();
        revisions.push(revision.clone());
        job["lyric_revisions"] = json!(revisions);
        job["selected_lyric_revision_id"] = revision["id"].clone();
        Ok(json!({"status":"saved","alignment_started":false,"revision":revision}))
    })
    .await?;
    Ok((StatusCode::CREATED, response))
}

pub(crate) fn variant_key(result: &Value, mode: &str) -> String {
    let variants = &result["alignment_results"];
    match mode {
        "canonical" if variants.get("canonical").is_none() => "sofa",
        "sofa" if variants.get("sofa").is_none() => "canonical",
        _ => mode,
    }
    .into()
}

pub(crate) fn apply_variant(job: &mut Value, mode: &str) -> Result<String, ApiError> {
    let result = &mut job["result"];
    if result.is_null() {
        return Err(bad_request("No alignment result"));
    }
    let variant = &result["alignment_results"][variant_key(result, mode)];
    if variant.is_null() {
        return Err(ApiError(
            StatusCode::CONFLICT,
            format!("Alignment mode is not available: {mode}"),
        ));
    }
    let resolved = if mode == "canonical" && result["alignment_results"].get("canonical").is_none()
    {
        "sofa"
    } else {
        mode
    };
    let variant = variant.clone();
    for field in [
        "words",
        "pitch_words",
        "phrases",
        "ultrastar_txt",
        "word_alignments",
        "alignment_run",
    ] {
        result[field] = variant[field].clone();
    }
    result["current_alignment_mode"] = json!(resolved);
    if variant["gap_ms"].as_f64().is_some_and(|value| value != 0.0) {
        result["metadata"]["gap_ms"] = variant["gap_ms"].clone();
        job["metadata"]["gap_ms"] = variant["gap_ms"].clone();
    }
    Ok(resolved.into())
}

async fn switch_mode(
    State(api): State<Api>,
    owner: Owner,
    RoutePath(id): RoutePath<String>,
    body: Result<Json<Value>, JsonRejection>,
) -> ApiResult {
    let body = api::parse_body(body)?;
    let mode = field_text(&body, "mode").trim().to_lowercase();
    mutate(api, owner, id, move |_, job| {
        if job["result"].is_null() {
            return Err(bad_request("No alignment result"));
        }
        if !matches!(mode.as_str(), "sofa" | "legacy" | "canonical") {
            return Err(invalid("mode must be sofa, legacy, or canonical"));
        }
        let resolved_mode = apply_variant(job, &mode)?;
        let mut result =
            managed_api::result_payload(job).ok_or_else(|| bad_request("No alignment result"))?;
        result["switched_to"] = json!(resolved_mode);
        Ok(result)
    })
    .await
}

pub(crate) fn note_overrides(phrases: &[Phrase]) -> Value {
    let mut overrides = Map::new();
    for (phrase_index, phrase) in phrases.iter().enumerate() {
        overrides.insert(
            format!("phrase:{phrase_index}"),
            json!({"end_beat":phrase.end_beat}),
        );
        for (note_index, note) in phrase.notes.iter().enumerate() {
            overrides.insert(format!("note:{phrase_index}:{note_index}"), json!(note));
        }
    }
    Value::Object(overrides)
}

fn validated_phrases(body: &Value) -> Result<Vec<Phrase>, ApiError> {
    let phrases = body.get("phrases").map_or_else(
        || Ok(Vec::new()),
        |value| {
            if !value.is_array() {
                return Err(invalid("phrases must be an array"));
            }
            serde_json::from_value::<Vec<Phrase>>(value.clone())
                .map_err(|error| ApiError::invalid(format!("Invalid note payload: {error}")))
        },
    )?;
    if phrases.iter().any(|phrase| phrase.end_beat < 0) {
        return Err(invalid(
            "Invalid note payload: phrase end_beat must be a non-negative integer",
        ));
    }
    if phrases
        .iter()
        .flat_map(|phrase| &phrase.notes)
        .any(|note| note.start_beat < 0 || note.duration_beats < 0)
    {
        return Err(invalid(
            "Invalid note payload: note timing values must be non-negative",
        ));
    }
    Ok(phrases)
}

pub(crate) fn render_result(result: &Value, phrases: &[Phrase]) -> Result<String, ApiError> {
    let metadata: Metadata = serde_json::from_value(result["metadata"].clone())
        .map_err(|error| ApiError::internal(error.to_string()))?;
    let assets: Assets = serde_json::from_value(result.clone())
        .map_err(|error| ApiError::internal(error.to_string()))?;
    Ok(chart_output::render(&metadata, phrases, &assets))
}

pub(crate) fn write_chart_and_publish(
    api: &Api,
    job: &mut Value,
    write_local: bool,
) -> Result<(), ApiError> {
    let result = &job["result"];
    let work_dir = field_text(job, "work_dir");
    let chart = serde_json::from_value::<CompletedChart>(result.clone())
        .map_err(|error| ApiError::internal(error.to_string()))?;
    if write_local && !work_dir.is_empty() {
        let file = Path::new(work_dir).join(format!("{}.txt", chart.name()));
        write_atomically(&file, field_text(result, "ultrastar_txt").as_bytes())
            .map_err(|error| ApiError::internal(error.to_string()))?;
    }
    if let Some(directory) = library_api::library(api).directory().ok().flatten() {
        let root = dunce::canonicalize(directory)
            .map_err(|error| ApiError::internal(error.to_string()))?;
        let existing = field_text(job, "library_path");
        let published = chart
            .publish(
                field_text(job, "id"),
                Path::new(work_dir),
                &root,
                (!existing.is_empty()).then(|| Path::new(existing)),
            )
            .map_err(ApiError::internal)?;
        job["library_path"] = json!(published);
        job["library_root"] = json!(root);
    }
    Ok(())
}

async fn save_notes(
    State(api): State<Api>,
    owner: Owner,
    RoutePath(id): RoutePath<String>,
    body: Result<Json<Value>, JsonRejection>,
) -> ApiResult {
    let body = api::parse_body(body)?;
    mutate(api, owner, id, move |api, job| {
        if job["result"].is_null() {
            return Err(bad_request("No result to update"));
        }
        if job["status"] == "aligning_words" {
            return Err(conflict("Cannot edit notes during alignment"));
        }
        let phrases = validated_phrases(&body)?;
        let result = &mut job["result"];
        let text = render_result(result, &phrases)?;
        let overrides = note_overrides(&phrases);
        result["phrases"] = json!(phrases);
        result["note_overrides"] = overrides;
        result["ultrastar_txt"] = json!(text);
        let key = variant_key(result, field_text(result, "current_alignment_mode"));
        let variants = &mut result["alignment_results"];
        let before = variants[&key].clone();
        if !before.is_null() {
            let alias = match key.as_str() {
                "canonical" => Some("sofa"),
                "sofa" => Some("canonical"),
                _ => None,
            };
            let mut updated = before.clone();
            updated["phrases"] = json!(phrases);
            updated["ultrastar_txt"] = json!(text);
            variants[&key] = updated.clone();
            if let Some(alias) = alias.filter(|alias| variants[*alias] == before) {
                variants[alias] = updated;
            }
        }
        write_chart_and_publish(api, job, true)?;
        Ok(json!({"status":"updated"}))
    })
    .await
}

fn parse_boundary(value: &Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_str()?.trim().parse().ok())
        .or_else(|| value.as_bool().map(|value| if value { 1.0 } else { 0.0 }))
        .filter(|value| value.is_finite())
}

fn edit_alignment(alignment: &mut Value, body: &Value) -> Result<(), ApiError> {
    let start = body.get("start").unwrap_or(&alignment["start"]);
    let end = body.get("end").unwrap_or(&alignment["end"]);
    if start.is_null() || end.is_null() {
        return Err(invalid("Word boundaries must be supplied as a finite pair"));
    }
    let number = || invalid("Word boundaries must be finite numbers");
    let start = parse_boundary(start).ok_or_else(number)?;
    let end = parse_boundary(end).ok_or_else(number)?;
    if start < 0.0 || end < start {
        return Err(invalid("Word end must be at or after its start"));
    }
    alignment["start"] = json!(start);
    alignment["end"] = json!(end);
    let mut reasons = array(&alignment["evidence"]["reasons"]).to_vec();
    for field in ["start", "end"]
        .into_iter()
        .filter(|field| body.get(*field).is_some())
    {
        alignment[format!("{field}_locked")] = json!(true);
        reasons.push(json!(format!(
            "{field} boundary manually edited and locked"
        )));
    }
    let mut seen = HashSet::new();
    reasons.retain(|reason| seen.insert(reason.clone()));
    alignment["evidence"]["reasons"] = json!(reasons);
    if alignment["status"] == "unmatched" {
        alignment["status"] = json!("estimated");
    }
    Ok(())
}

fn reset_alignment(alignment: &mut Value, boundaries: &HashSet<&str>) {
    let mut reasons = array(&alignment["evidence"]["reasons"]).to_vec();
    for field in ["start", "end"]
        .into_iter()
        .filter(|field| boundaries.contains(field))
    {
        alignment[format!("{field}_locked")] = json!(false);
        reasons.push(json!(format!("{field} boundary lock reset")));
    }
    alignment["evidence"]["reasons"] = json!(reasons);
}

fn validate_sequence(items: &[Value]) -> Result<(), ApiError> {
    let mut previous_end: Option<f64> = None;
    for item in items {
        if item["start"].is_null() && item["end"].is_null() {
            continue;
        }
        let invalid_sequence = || invalid("Word boundaries must remain finite and monotonic");
        let (start, end) = projection::span(item).ok_or_else(invalid_sequence)?;
        if start < 0.0 || previous_end.is_some_and(|previous| start < previous - 0.02) {
            return Err(invalid_sequence());
        }
        previous_end = Some(previous_end.map_or(end, |previous| previous.max(end)));
    }
    Ok(())
}

fn boundaries_to_reset(body: &Value) -> Result<HashSet<&str>, ApiError> {
    let Some(values) = body.get("boundaries") else {
        return Ok(HashSet::from(["start", "end"]));
    };
    let invalid_type = || invalid("boundaries must be an array of strings");
    let values = values.as_array().ok_or_else(invalid_type)?;
    let values = values
        .iter()
        .map(|value| value.as_str().ok_or_else(invalid_type))
        .collect::<Result<HashSet<_>, _>>()?;
    if !values
        .iter()
        .all(|boundary| matches!(*boundary, "start" | "end"))
    {
        return Err(invalid("Boundaries must be start and/or end"));
    }
    Ok(values)
}

fn edit_boundaries(
    api: &Api,
    job: &mut Value,
    token: &str,
    body: &Value,
    reset: bool,
) -> Result<Value, ApiError> {
    if job["status"] == "aligning_words" {
        return Err(conflict("Cannot edit word boundaries during alignment"));
    }
    if job["result"].is_null() || (!reset && job["result"]["lyric_revision"].is_null()) {
        return Err(bad_request("No alignment result"));
    }
    let reset_fields = if reset {
        boundaries_to_reset(body)?
    } else {
        if body.get("start").is_none() && body.get("end").is_none() {
            return Err(invalid("Provide start and/or end"));
        }
        if body.as_object().is_some_and(|body| {
            body.keys()
                .any(|field| !matches!(field.as_str(), "start" | "end"))
        }) {
            return Err(invalid("Only start and end may be edited"));
        }
        HashSet::new()
    };
    let mut alignments = array(&job["result"]["word_alignments"]).to_vec();
    let index = alignments
        .iter()
        .position(|alignment| alignment["token_id"] == token)
        .ok_or_else(|| ApiError(StatusCode::NOT_FOUND, "Word token not found".into()))?;
    if reset {
        reset_alignment(&mut alignments[index], &reset_fields);
    } else {
        edit_alignment(&mut alignments[index], body)?;
        validate_sequence(&alignments)?;
    }
    let updated = alignments[index].clone();
    job["result"]["word_alignments"] = json!(alignments);
    if projection::refresh(job)? {
        write_chart_and_publish(api, job, true)?;
    }
    Ok(json!({"status":"updated","alignment":updated}))
}

async fn update_boundary(
    State(api): State<Api>,
    owner: Owner,
    RoutePath((id, token)): RoutePath<(String, String)>,
    body: Result<Json<Value>, JsonRejection>,
) -> ApiResult {
    let body = api::parse_body(body)?;
    mutate(api, owner, id, move |api, job| {
        edit_boundaries(api, job, &token, &body, false)
    })
    .await
}

async fn reset_boundary(
    State(api): State<Api>,
    owner: Owner,
    RoutePath((id, token)): RoutePath<(String, String)>,
    body: Option<Json<Value>>,
) -> ApiResult {
    let body = body.map_or(Value::Null, |Json(body)| body);
    if !body.is_null() && !body.is_object() {
        return Err(invalid("Expected a JSON object"));
    }
    mutate(api, owner, id, move |api, job| {
        edit_boundaries(api, job, &token, &body, true)
    })
    .await
}

pub(crate) fn routes() -> Router<Api> {
    Router::new()
        .route("/api/jobs/{job_id}/notes", put(save_notes))
        .route("/api/jobs/{job_id}/lyric-revisions", post(save_revision))
        .route("/api/jobs/{job_id}/alignment-mode", patch(switch_mode))
        .route(
            "/api/jobs/{job_id}/word-alignments/{token_id}",
            patch(update_boundary),
        )
        .route(
            "/api/jobs/{job_id}/word-alignments/{token_id}/reset",
            post(reset_boundary),
        )
}
