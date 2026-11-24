use std::{
    net::{IpAddr, SocketAddr},
    path::{Path, PathBuf},
    sync::Arc,
};

use axum::{
    Extension, Json, Router,
    extract::{ConnectInfo, Query, Request, State, rejection::JsonRejection},
    http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode, header},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio::{net::TcpListener, sync::watch};
use tower_http::cors::{AllowOrigin, CorsLayer};

use crate::{
    catalog,
    managed_api::{self, ManagedJobs},
    worker, youtube,
};

const WORKER_TOKEN_HEADER: &str = "x-alignment-worker-token";
const DEFAULT_ORIGINS: &str =
    "http://localhost:5173,http://127.0.0.1:5173,tauri://localhost,https://tauri.localhost";

#[derive(Clone)]
pub(crate) struct Api {
    pub(crate) backend: PathBuf,
    api_token: String,
    worker_token: String,
    worker_audio_root: PathBuf,
    pub(crate) worker: Arc<worker::AlignmentWorker>,
    pub(crate) jobs: Option<Arc<ManagedJobs>>,
    pub(crate) stopping: watch::Receiver<bool>,
    pub(crate) prep_queue: Arc<tokio::sync::Semaphore>,
    pub(crate) stem_states: Arc<crate::library_api::StemStates>,
}

pub(crate) struct ApiError(pub(crate) StatusCode, pub(crate) String);

#[derive(Clone)]
pub(crate) struct Owner(pub(crate) String);

impl<S: Send + Sync> axum::extract::FromRequestParts<S> for Owner {
    type Rejection = ApiError;

    async fn from_request_parts(
        parts: &mut axum::http::request::Parts,
        state: &S,
    ) -> Result<Self, Self::Rejection> {
        let Extension(owner) = Extension::<Self>::from_request_parts(parts, state)
            .await
            .map_err(|_| ApiError::internal("Request owner is unavailable"))?;
        Ok(owner)
    }
}

impl ApiError {
    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self(StatusCode::UNPROCESSABLE_ENTITY, message.into())
    }

    pub(crate) fn internal(message: impl Into<String>) -> Self {
        Self(StatusCode::INTERNAL_SERVER_ERROR, message.into())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({"detail": self.1}))).into_response()
    }
}

pub(crate) type ApiResult = Result<Json<Value>, ApiError>;

pub(crate) fn parse_body(body: Result<Json<Value>, JsonRejection>) -> Result<Value, ApiError> {
    let Json(value) = body.map_err(|error| ApiError::invalid(error.body_text()))?;
    if !value.is_object() {
        return Err(ApiError::invalid("Expected a JSON object"));
    }
    Ok(value)
}

fn token<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers.get(name)?.to_str().ok()
}

fn authorize(
    api_token: &str,
    ip: IpAddr,
    headers: &HeaderMap,
    query: Option<&str>,
) -> Result<Owner, ApiError> {
    let authorization = token(headers, "authorization").unwrap_or_default();
    let mut supplied = authorization
        .strip_prefix("Bearer ")
        .unwrap_or(authorization)
        .trim()
        .to_owned();
    if supplied.is_empty() {
        supplied = reqwest::Url::parse(&format!("http://localhost/?{}", query.unwrap_or_default()))
            .ok()
            .and_then(|url| {
                url.query_pairs()
                    .filter(|(name, _)| name == "access_token")
                    .last()
                    .map(|(_, value)| value.into_owned())
            })
            .unwrap_or_default();
    }
    if api_token.is_empty() {
        if !ip.is_loopback() {
            return Err(ApiError(
                StatusCode::UNAUTHORIZED,
                "Configure KARAOKE_API_TOKEN for non-local access".into(),
            ));
        }
    } else if supplied != api_token {
        return Err(ApiError(
            StatusCode::UNAUTHORIZED,
            "Authentication required".into(),
        ));
    }
    let owner = if supplied.is_empty() {
        token(headers, "x-karaoke-owner")
            .unwrap_or("local")
            .to_owned()
    } else {
        hex::encode(&Sha256::digest(supplied.as_bytes())[..16])
    };
    Ok(Owner(owner))
}

async fn authenticate(
    State(api): State<Api>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    mut request: Request,
    next: Next,
) -> Response {
    if request.method() != Method::OPTIONS {
        match authorize(
            &api.api_token,
            peer.ip(),
            request.headers(),
            request.uri().query(),
        ) {
            Ok(owner) => {
                request.extensions_mut().insert(owner);
            }
            Err(error) => return error.into_response(),
        }
    }
    next.run(request).await
}

async fn parse_filename(body: Result<Json<Value>, JsonRejection>) -> ApiResult {
    let body = parse_body(body)?;
    let filename = match body.get("filename") {
        None => "",
        Some(value) => value
            .as_str()
            .ok_or_else(|| ApiError::invalid("filename must be a string"))?,
    };
    Ok(Json(catalog::parse_local_filename(filename.into())))
}

#[derive(Deserialize)]
struct LyricsQuery {
    #[serde(default)]
    q: String,
}

async fn search_lyrics(Query(LyricsQuery { q }): Query<LyricsQuery>) -> ApiResult {
    let results = tokio::task::spawn_blocking(move || {
        if q.trim().is_empty() {
            return Ok(Vec::new());
        }
        Ok::<_, String>(catalog::search(&catalog::client()?, q.trim(), 20))
    })
    .await
    .map_err(|error| ApiError::internal(error.to_string()))?
    .map_err(ApiError::internal)?;
    Ok(Json(json!({"results": results})))
}

async fn search_youtube(
    State(api): State<Api>,
    body: Result<Json<Value>, JsonRejection>,
) -> ApiResult {
    let body = parse_body(body)?;
    let query = body["query"]
        .as_str()
        .ok_or_else(|| ApiError::invalid("query must be a string"))?
        .to_owned();
    let results = tokio::task::spawn_blocking(move || youtube::search(&api.backend, &query))
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?
        .map_err(|error| ApiError::internal(format!("Search failed: {error}")))?;
    Ok(Json(json!({"results": results})))
}

fn check_audio_reference(reference: &str, root: &Path) -> Result<(), ApiError> {
    let path = Path::new(reference);
    if reference.contains("://") || !path.is_absolute() {
        return Err(ApiError::invalid(
            "Worker audio references must be absolute local paths",
        ));
    }
    let root = root.canonicalize().map_err(|_| {
        ApiError(
            StatusCode::NOT_FOUND,
            "Worker audio reference is unavailable".into(),
        )
    })?;
    let mut parent = path;
    while !parent.exists() {
        parent = parent.parent().ok_or_else(|| {
            ApiError(
                StatusCode::NOT_FOUND,
                "Worker audio reference is unavailable".into(),
            )
        })?;
    }
    let canonical_parent = parent.canonicalize().map_err(|_| {
        ApiError(
            StatusCode::NOT_FOUND,
            "Worker audio reference is unavailable".into(),
        )
    })?;
    if !canonical_parent.starts_with(&root) {
        return Err(ApiError::invalid(
            "Worker audio reference is outside the worker audio root",
        ));
    }
    let path = path.canonicalize().map_err(|_| {
        ApiError(
            StatusCode::NOT_FOUND,
            "Worker audio reference is unavailable".into(),
        )
    })?;
    if !path.starts_with(root) {
        return Err(ApiError::invalid(
            "Worker audio reference is outside the worker audio root",
        ));
    }
    if !path.is_file() {
        return Err(ApiError(
            StatusCode::NOT_FOUND,
            "Worker audio reference is unavailable".into(),
        ));
    }
    Ok(())
}

fn worker_draft(
    mut response: Value,
    locale: &str,
    sources: [(&str, PathBuf); 2],
    transcribe: impl Fn(&Path, &str) -> Result<Vec<Value>, String>,
) -> Value {
    let mut errors = serde_json::Map::new();
    let mut first_error = None;
    for (name, path) in sources {
        match transcribe(&path, locale) {
            Ok(words) => {
                let text = words
                    .iter()
                    .filter_map(|word| word["text"].as_str())
                    .collect::<Vec<_>>()
                    .join(" ");
                let text = text.trim();
                if !text.is_empty() {
                    response["status"] = json!("completed");
                    response["draft_text"] = json!(text);
                    response["diagnostics"] = json!({"source_errors": errors, "draft_text": text});
                    return response;
                }
            }
            Err(error) => {
                first_error.get_or_insert_with(|| error.clone());
                errors.insert(name.into(), json!(error));
            }
        }
    }
    response["diagnostics"] = json!({"source_errors": errors, "reason": first_error.unwrap_or_else(|| "No timestamped lyrics were found".into())});
    response
}

async fn worker_alignment(
    State(api): State<Api>,
    headers: HeaderMap,
    body: Result<Json<Value>, JsonRejection>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    let body = parse_body(body)?;
    if api.worker_token.is_empty() {
        return Err(ApiError(
            StatusCode::SERVICE_UNAVAILABLE,
            "Alignment worker ingress is not configured".into(),
        ));
    }
    if token(&headers, WORKER_TOKEN_HEADER) != Some(api.worker_token.as_str()) {
        return Err(ApiError(
            StatusCode::UNAUTHORIZED,
            "Invalid alignment worker credentials".into(),
        ));
    }
    for key in ["original_audio_ref", "vocal_stem_audio_ref"] {
        let reference = body[key]
            .as_str()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| ApiError::invalid("Both audio references are required"))?;
        check_audio_reference(reference, &api.worker_audio_root)?;
    }
    let failed = json!({
        "job_id": body["job_id"], "alignment_run_id": body["alignment_run_id"],
        "lyric_revision_id": body["lyric_revision_id"], "status": "failed",
        "alignments": [], "diagnostics": {}, "draft_text": null
    });
    // An empty token list asks for a draft rather than alignment evidence.
    if body.get("tokens").is_none() || body["tokens"].as_array().is_some_and(Vec::is_empty) {
        for field in ["job_id", "alignment_run_id", "lyric_revision_id"] {
            if !body[field]
                .as_str()
                .is_some_and(|value| !value.trim().is_empty())
            {
                return Err(ApiError::invalid(format!(
                    "Invalid alignment request: {field} is required"
                )));
            }
        }
        let locale = body["language"]
            .as_str()
            .ok_or_else(|| ApiError::invalid("Invalid alignment request: language is required"))
            .and_then(crate::managed_ingest::language)?;
        if let Some(backend) = body.get("alignment_backend") {
            if !backend.as_str().is_some_and(|name| {
                matches!(name.trim().to_lowercase().as_str(), "" | "sofa" | "legacy")
            }) {
                return Err(ApiError::invalid(
                    "Invalid alignment request: alignment_backend must be sofa or legacy",
                ));
            }
        }
        let sources = [
            (
                "vocal_stem",
                PathBuf::from(body["vocal_stem_audio_ref"].as_str().unwrap()),
            ),
            (
                "original_mix",
                PathBuf::from(body["original_audio_ref"].as_str().unwrap()),
            ),
        ];
        let draft = tokio::task::spawn_blocking(move || {
            worker_draft(failed, locale, sources, crate::whisper_draft::transcribe)
        })
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?;
        return Ok((StatusCode::ACCEPTED, Json(draft)));
    }
    let result = tokio::spawn(async move { api.worker.run(&api.backend, "align", body).await })
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?;
    let response = match result {
        Ok(result) => {
            if let Some(error) = result["validation_error"].as_str() {
                return Err(ApiError::invalid(format!(
                    "Invalid alignment request: {error}"
                )));
            }
            result["response"].clone()
        }
        Err(error) => {
            let mut response = failed;
            response["diagnostics"]["error"] = Value::String(error);
            response
        }
    };
    Ok((StatusCode::ACCEPTED, Json(response)))
}

pub(crate) async fn serve(
    backend: PathBuf,
    address: SocketAddr,
    worker_only: bool,
) -> Result<(), String> {
    let backend = backend
        .canonicalize()
        .map_err(|error| format!("Cannot open backend directory: {error}"))?;
    let listener = TcpListener::bind(address)
        .await
        .map_err(|error| error.to_string())?;
    let jobs = if worker_only {
        None
    } else {
        let outputs = backend.join("outputs");
        std::fs::create_dir_all(&outputs).map_err(|error| error.to_string())?;
        Some(Arc::new(ManagedJobs::load(outputs.join("jobs.json"))?))
    };
    let (stop, stopping) = watch::channel(false);
    let api = Api {
        api_token: std::env::var("KARAOKE_API_TOKEN")
            .unwrap_or_default()
            .trim()
            .to_owned(),
        worker_token: std::env::var("ALIGNMENT_WORKER_TOKEN")
            .unwrap_or_default()
            .trim()
            .to_owned(),
        worker_audio_root: std::env::var_os("ALIGNMENT_WORKER_AUDIO_ROOT")
            .map_or_else(|| backend.clone(), PathBuf::from),
        worker: Arc::default(),
        backend,
        jobs,
        stopping,
        prep_queue: Arc::new(tokio::sync::Semaphore::new(1)),
        stem_states: Arc::default(),
    };
    let origins: Vec<HeaderValue> = std::env::var("KARAOKE_CORS_ORIGINS")
        .unwrap_or_else(|_| DEFAULT_ORIGINS.to_owned())
        .split(',')
        .filter_map(|origin| origin.trim().parse().ok())
        .filter(|origin: &HeaderValue| !origin.is_empty())
        .collect();
    let origins = if origins.iter().any(|origin| origin == "*") {
        AllowOrigin::mirror_request()
    } else {
        AllowOrigin::list(origins)
    };
    let browser = Router::new()
        .route("/api/parse-filename", post(parse_filename))
        .route("/api/lyrics/search", get(search_lyrics))
        .route("/api/youtube/search", post(search_youtube))
        .merge(managed_api::routes())
        .merge(crate::library_api::routes())
        .merge(crate::managed_ingest::routes())
        .merge(crate::managed_editor::routes())
        .merge(crate::managed_alignment::routes())
        .merge(crate::managed_benchmark::routes())
        .route_layer(middleware::from_fn_with_state(api.clone(), authenticate))
        .layer(
            CorsLayer::new()
                .allow_origin(origins)
                .allow_credentials(true)
                .allow_methods([
                    Method::GET,
                    Method::POST,
                    Method::PUT,
                    Method::PATCH,
                    Method::DELETE,
                    Method::OPTIONS,
                ])
                .allow_headers([
                    header::AUTHORIZATION,
                    header::CONTENT_TYPE,
                    HeaderName::from_static("x-karaoke-owner"),
                    HeaderName::from_static(WORKER_TOKEN_HEADER),
                ]),
        );
    let mut router = Router::new().route("/v1/alignments", post(worker_alignment));
    if !worker_only {
        router = router.merge(browser);
    }
    eprintln!(
        "Karaoke API listening on {}",
        listener.local_addr().map_err(|error| error.to_string())?
    );
    axum::serve(
        listener,
        router
            .with_state(api)
            .into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(async move {
        shutdown().await;
        let _ = stop.send(true);
    })
    .await
    .map_err(|error| error.to_string())
}

async fn shutdown() {
    #[cfg(unix)]
    if let Ok(mut sigterm) =
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
    {
        tokio::select! {
            _ = sigterm.recv() => {},
            _ = tokio::signal::ctrl_c() => {},
        }
        return;
    }
    let _ = tokio::signal::ctrl_c().await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn worker_ingress_fails_closed_without_credentials() {
        let api = Api {
            backend: PathBuf::new(),
            api_token: String::new(),
            worker_token: String::new(),
            worker_audio_root: PathBuf::new(),
            worker: Arc::default(),
            jobs: None,
            stopping: watch::channel(false).1,
            prep_queue: Arc::new(tokio::sync::Semaphore::new(1)),
            stem_states: Arc::default(),
        };
        let request = json!({"job_id": "job", "alignment_run_id": "run", "lyric_revision_id": "revision",
            "language": "en", "original_audio_ref": "/audio/mix.wav", "vocal_stem_audio_ref": "/audio/vocals.wav"});
        let result = worker_alignment(State(api), HeaderMap::new(), Ok(Json(request))).await;
        assert_eq!(result.err().unwrap().0, StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn empty_alignment_request_can_draft_from_the_original_mix() {
        let sources = [
            ("vocal_stem", PathBuf::from("voice")),
            ("original_mix", PathBuf::from("mix")),
        ];
        let result = worker_draft(
            json!({"status":"failed","draft_text":null}),
            "pl",
            sources,
            |path, language| {
                assert_eq!(language, "pl");
                if path == Path::new("voice") {
                    return Err("No timestamped words".into());
                }
                Ok(vec![json!({"text":"Żółć"}), json!({"text":"ma"})])
            },
        );
        assert_eq!(result["status"], "completed");
        assert_eq!(result["draft_text"], "Żółć ma");
        assert_eq!(
            result["diagnostics"]["source_errors"]["vocal_stem"],
            "No timestamped words"
        );
    }

    #[test]
    fn worker_audio_references_stay_inside_configured_root() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("audio");
        std::fs::create_dir(&root).unwrap();
        let inside = root.join("in.wav");
        std::fs::write(&inside, "").unwrap();
        assert!(check_audio_reference(inside.to_str().unwrap(), &root).is_ok());
        let outside = temporary.path().join("out.wav");
        let error = check_audio_reference(outside.to_str().unwrap(), &root)
            .err()
            .unwrap();
        assert_eq!(error.0, StatusCode::UNPROCESSABLE_ENTITY);
        let missing = root.join("missing.wav");
        let error = check_audio_reference(missing.to_str().unwrap(), &root)
            .err()
            .unwrap();
        assert_eq!(error.0, StatusCode::NOT_FOUND);
    }

    #[test]
    fn browser_requires_token_for_non_local_requests() {
        let ip = "192.0.2.5".parse().unwrap();
        assert!(authorize("", ip, &HeaderMap::new(), None).is_err());
        assert!(
            authorize(
                "required",
                ip,
                &HeaderMap::new(),
                Some("access_token=required")
            )
            .is_ok()
        );
        let mut headers = HeaderMap::new();
        headers.insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("Bearer wrong"),
        );
        assert!(authorize("required", ip, &headers, Some("access_token=required")).is_err());
    }
}
