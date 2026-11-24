use std::{
    fs::{self, File, OpenOptions, TryLockError},
    path::{Path, PathBuf},
    sync::Mutex,
};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tauri::{AppHandle, Manager, State};
use uuid::Uuid;

use crate::{catalog, library::write_atomically, media::LocalImport, worker};

const RECORD: &str = "desktop-preparation.json";

#[derive(Default)]
pub(crate) struct PreparationQueue(Mutex<()>);

#[derive(Deserialize, Serialize)]
pub(crate) struct ImportMetadata {
    title: String,
    artist: String,
    language: String,
    genre: String,
    year: String,
}

#[derive(Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Status {
    Pending,
    Downloading,
    ExtractingAudio,
    SeparatingVocals,
    DetectingBpm,
    SearchingLyrics,
    AwaitingLyrics,
    Error,
}

impl Status {
    fn running(self) -> bool {
        matches!(
            self,
            Self::Downloading
                | Self::ExtractingAudio
                | Self::SeparatingVocals
                | Self::DetectingBpm
                | Self::SearchingLyrics
        )
    }
}

#[derive(Deserialize, Serialize)]
struct Preparation {
    metadata: ImportMetadata,
    original_filename: String,
    #[serde(default)]
    source_url: String,
    status: Status,
    message: String,
    error: String,
    #[serde(default)]
    progress_percent: f64,
    real_bpm: Option<f64>,
    found_lyrics: String,
}

#[derive(Deserialize)]
struct Manifest {
    original_filename: String,
}

#[derive(Deserialize)]
struct Stems {
    vocals: PathBuf,
    instrumental: PathBuf,
}

fn path(root: &Path, id: Uuid) -> PathBuf {
    root.join("uploads").join(id.to_string()).join(RECORD)
}

fn read(path: &Path) -> Result<Option<Preparation>, String> {
    match fs::read(path) {
        Ok(bytes) => serde_json::from_slice(&bytes)
            .map(Some)
            .map_err(|error| error.to_string()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.to_string()),
    }
}

fn save(path: &Path, job: &Preparation) -> Result<(), String> {
    write_atomically(
        path,
        &serde_json::to_vec(job).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())
}

fn transition(
    path: &Path,
    job: &mut Preparation,
    status: Status,
    message: &str,
) -> Result<(), String> {
    job.status = status;
    job.message = message.into();
    save(path, job)
}

fn try_lock(record: &Path) -> Result<Option<File>, String> {
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(record.with_extension("lock"))
        .map_err(|error| error.to_string())?;
    match lock.try_lock() {
        Ok(()) => Ok(Some(lock)),
        Err(TryLockError::WouldBlock) => Ok(None),
        Err(TryLockError::Error(error)) => Err(error.to_string()),
    }
}

fn load(record: &Path) -> Result<Option<Preparation>, String> {
    let Some(job) = read(record)? else {
        return Ok(None);
    };
    if !job.status.running() {
        return Ok(Some(job));
    }
    let Some(_lock) = try_lock(record)? else {
        return Ok(Some(job));
    };
    let Some(mut job) = read(record)? else {
        return Ok(None);
    };
    if job.status.running() {
        job.error.clear();
        transition(
            record,
            &mut job,
            Status::Pending,
            "Preparation interrupted. Ready to resume.",
        )?;
    }
    Ok(Some(job))
}

fn wav(path: &Path) -> bool {
    hound::WavReader::open(path).is_ok_and(|audio| audio.duration() > 0)
}

fn prepare(root: &Path, id: Uuid, record: &Path, job: &mut Preparation) -> Result<(), String> {
    let output = root.join("outputs").join(id.to_string());
    let audio = output.join("audio.wav");
    if !job.source_url.is_empty() {
        let manifest_path = record.with_file_name("prepared-media.json");
        if !wav(&audio) || !manifest_path.is_file() || !record.with_file_name("input").is_file() {
            let url = job.source_url.clone();
            job.progress_percent = 0.0;
            transition(record, job, Status::Downloading, "Downloading video…")?;
            let downloaded = crate::youtube::download(root, id, &url, |progress| {
                if progress >= job.progress_percent + 1.0 {
                    job.progress_percent = progress;
                    job.message = format!("Downloading video… {progress:.0}%");
                    if let Err(error) = save(record, job) {
                        log::warn!("Unable to record progress for desktop download {id}: {error}");
                    }
                }
            })?;
            if job.metadata.title.is_empty() {
                job.metadata.title = downloaded.title;
            }
            if job.metadata.artist.is_empty() {
                job.metadata.artist = downloaded.artist;
            }
        }
        let manifest = fs::read(record.with_file_name("prepared-media.json"))
            .map_err(|error| error.to_string())?;
        job.original_filename = serde_json::from_slice::<Manifest>(&manifest)
            .map_err(|error| error.to_string())?
            .original_filename;
        job.progress_percent = 0.0;
    }
    if !wav(&audio) {
        return Err("Prepared audio is missing. Import the original media again.".into());
    }

    analyze(root, &audio, &output, job, |job, status, message| {
        transition(record, job, status, message)
    })?;
    transition(
        record,
        job,
        Status::AwaitingLyrics,
        "Waiting for lyrics approval.",
    )
}

pub(crate) fn prepare_browser_media(
    root: &Path,
    media: BrowserMedia,
    mut progress: impl FnMut(Value),
) -> Result<Value, String> {
    let BrowserMedia {
        source,
        work,
        prior_audio,
        mut has_video,
        real_bpm,
        metadata,
    } = media;
    fs::create_dir_all(&work).map_err(|error| error.to_string())?;
    let audio = if let Some(audio) = prior_audio.filter(|path| wav(path)) {
        audio
    } else {
        progress(json!({"status": Status::ExtractingAudio, "message": "Extracting audio…"}));
        has_video = crate::media::has_video(&source)?;
        let cached = work.join("audio.wav");
        if !wav(&cached) {
            let staging =
                tempfile::NamedTempFile::new_in(&work).map_err(|error| error.to_string())?;
            crate::media::write_pcm_wav(&source, staging.path())
                .map_err(|error| error.to_string())?;
            staging
                .persist(&cached)
                .map_err(|error| error.to_string())?;
        }
        cached
    };
    let mut job = initialize(metadata, String::new(), String::new());
    job.real_bpm = (real_bpm > 0.0).then_some(real_bpm);
    analyze(root, &audio, &work, &mut job, |_, status, message| {
        progress(json!({"status": status, "message": message}));
        Ok(())
    })?;
    Ok(json!({
        "metadata": job.metadata, "audio_path": audio, "has_video": has_video,
        "vocals_path": work.join("vocals.wav"), "instrumental_path": work.join("instrumental.wav"),
        "real_bpm": job.real_bpm, "found_lyrics": job.found_lyrics,
    }))
}

#[derive(Deserialize)]
pub(crate) struct BrowserMedia {
    source: PathBuf,
    work: PathBuf,
    prior_audio: Option<PathBuf>,
    has_video: bool,
    real_bpm: f64,
    metadata: ImportMetadata,
}

fn analyze(
    root: &Path,
    audio: &Path,
    output: &Path,
    job: &mut Preparation,
    mut notify: impl FnMut(&mut Preparation, Status, &str) -> Result<(), String>,
) -> Result<(), String> {
    let model_slot = OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(
            output
                .parent()
                .ok_or("Missing output directory")?
                .join("processing.lock"),
        )
        .map_err(|error| error.to_string())?;
    model_slot.lock().map_err(|error| error.to_string())?;
    let vocals = output.join("vocals.wav");
    let instrumental = output.join("instrumental.wav");
    if !wav(&vocals) || !wav(&instrumental) {
        notify(
            job,
            Status::SeparatingVocals,
            "Separating vocals from instruments…",
        )?;
        let staging = tempfile::tempdir_in(&output).map_err(|error| error.to_string())?;
        let stems: Stems = worker::infer(
            root,
            json!({
                "operation": "separate", "audio": audio, "output_directory": staging.path(),
            }),
        )?;
        if stems.vocals != staging.path().join("vocals.wav")
            || stems.instrumental != staging.path().join("instrumental.wav")
            || !wav(&stems.vocals)
            || !wav(&stems.instrumental)
        {
            return Err("Inference worker returned invalid stems".into());
        }
        fs::rename(stems.vocals, &vocals).map_err(|error| error.to_string())?;
        fs::rename(stems.instrumental, &instrumental).map_err(|error| error.to_string())?;
    }
    if job.real_bpm.is_none() {
        notify(job, Status::DetectingBpm, "Detecting song tempo…")?;
        job.real_bpm = Some(crate::tempo::detect(audio)?);
    }
    drop(model_slot);

    notify(job, Status::SearchingLyrics, "Searching for lyrics online…")?;
    if let Ok(client) = catalog::client() {
        let metadata = &mut job.metadata;
        if metadata.genre.is_empty() || metadata.year.is_empty() {
            let (genre, year) =
                catalog::release_metadata(&client, &metadata.title, &metadata.artist);
            if metadata.genre.is_empty() {
                metadata.genre = genre;
            }
            if metadata.year.is_empty() {
                metadata.year = year;
            }
        }
        job.found_lyrics = catalog::find_lyrics(&client, &metadata.artist, &metadata.title);
        if job.found_lyrics == "These are song lyrics with singing." {
            job.found_lyrics.clear();
        }
    }
    Ok(())
}

fn initialize(
    metadata: ImportMetadata,
    original_filename: String,
    source_url: String,
) -> Preparation {
    Preparation {
        metadata,
        original_filename,
        source_url,
        status: Status::Pending,
        message: "Ready to process".into(),
        error: String::new(),
        progress_percent: 0.0,
        real_bpm: None,
        found_lyrics: String::new(),
    }
}

#[tauri::command]
pub(crate) fn register_native_media(
    state: State<'_, LocalImport>,
    import_id: Uuid,
    metadata: ImportMetadata,
) -> Result<String, String> {
    let root = state
        .0
        .as_deref()
        .ok_or("Local processing requires the desktop development backend")?;
    let record = path(root, import_id);
    let manifest = fs::read(record.with_file_name("prepared-media.json"))
        .map_err(|error| error.to_string())?;
    let manifest: Manifest =
        serde_json::from_slice(&manifest).map_err(|error| error.to_string())?;
    let _lock = try_lock(&record)?.ok_or("Media is already being registered")?;
    if !record.is_file() {
        save(
            &record,
            &initialize(metadata, manifest.original_filename, String::new()),
        )?;
    }
    Ok(import_id.to_string())
}

#[tauri::command]
pub(crate) fn register_native_download(
    app: AppHandle,
    url: String,
    metadata: ImportMetadata,
) -> Result<String, String> {
    crate::youtube::validate_url(&url)?;
    let root = app
        .state::<LocalImport>()
        .0
        .clone()
        .ok_or("Local processing requires the desktop development backend")?;
    let id = Uuid::new_v4();
    let record = path(&root, id);
    fs::create_dir_all(record.parent().unwrap()).map_err(|error| error.to_string())?;
    save(&record, &initialize(metadata, String::new(), url))?;
    start_native_job(app, id, false)?;
    Ok(id.to_string())
}

#[tauri::command]
pub(crate) fn native_job_status(
    state: State<'_, LocalImport>,
    job_id: String,
) -> Result<Option<Value>, String> {
    let (Some(root), Ok(id)) = (state.0.as_deref(), Uuid::parse_str(&job_id)) else {
        return Ok(None);
    };
    Ok(load(&path(root, id))?.map(|job| json!({
        "status": job.status, "message": job.message, "error": job.error,
        "metadata": job.metadata, "found_lyrics": job.found_lyrics, "progress_percent": job.progress_percent,
    })))
}

#[tauri::command]
pub(crate) fn list_native_jobs(state: State<'_, LocalImport>) -> Result<Vec<Value>, String> {
    let Some(root) = state.0.as_deref() else {
        return Ok(Vec::new());
    };
    let directory = root.join("uploads");
    if !directory.is_dir() {
        return Ok(Vec::new());
    }
    let mut jobs = Vec::new();
    for entry in fs::read_dir(directory)
        .map_err(|error| error.to_string())?
        .flatten()
    {
        let Ok(id) = Uuid::parse_str(&entry.file_name().to_string_lossy()) else {
            continue;
        };
        let job = match load(&entry.path().join(RECORD)) {
            Ok(Some(job)) => job,
            Ok(None) => continue,
            Err(error) => {
                log::warn!("Unable to read desktop job {id}: {error}");
                continue;
            }
        };
        jobs.push(json!({
            "id": id.to_string(), "kind": "managed", "title": job.metadata.title,
            "artist": job.metadata.artist, "filename": job.original_filename,
            "status": job.status, "message": job.message, "error": job.error,
            "failed_phase": if job.status == Status::Error { "phase1" } else { "" },
            "progress_percent": job.progress_percent,
        }));
    }
    jobs.sort_by_cached_key(|job| job["title"].as_str().unwrap_or_default().to_lowercase());
    Ok(jobs)
}

#[tauri::command]
pub(crate) fn start_native_job(app: AppHandle, job_id: Uuid, retry: bool) -> Result<bool, String> {
    let Some(root) = app.state::<LocalImport>().0.clone() else {
        return Ok(false);
    };
    let record = path(&root, job_id);
    if !record.is_file() {
        return Ok(false);
    }
    let Some(lock) = try_lock(&record)? else {
        return Ok(true);
    };
    let Some(mut job) = read(&record)? else {
        return Ok(false);
    };
    if job.status == Status::AwaitingLyrics || (job.status == Status::Error && !retry) {
        return Ok(true);
    }
    job.error.clear();
    let status = if job.source_url.is_empty()
        || root
            .join("outputs")
            .join(job_id.to_string())
            .join("audio.wav")
            .is_file()
    {
        Status::ExtractingAudio
    } else {
        Status::Downloading
    };
    transition(&record, &mut job, status, "Queued for processing…")?;
    tauri::async_runtime::spawn_blocking(move || {
        let _lock = lock;
        let queue = app.state::<PreparationQueue>();
        let _engine = queue
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Err(error) = prepare(&root, job_id, &record, &mut job) {
            job.error = error.clone();
            if let Err(save_error) = transition(&record, &mut job, Status::Error, &error) {
                log::error!("Unable to persist desktop job failure: {save_error}");
            }
        }
    });
    Ok(true)
}

#[tauri::command]
pub(crate) fn retire_native_job(state: State<'_, LocalImport>, job_id: Uuid) -> Result<(), String> {
    let Some(root) = state.0.as_deref() else {
        return Ok(());
    };
    let record = path(root, job_id);
    if read(&record)?.is_some_and(|job| job.status != Status::AwaitingLyrics) {
        return Err("Only a job at lyrics review can be transferred".into());
    }
    match fs::remove_file(record) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.to_string()),
    }
}

#[tauri::command]
pub(crate) fn remove_native_job(
    state: State<'_, LocalImport>,
    job_id: Uuid,
) -> Result<bool, String> {
    let Some(root) = state.0.as_deref() else {
        return Ok(false);
    };
    let record = path(root, job_id);
    if !record.is_file() {
        return Ok(false);
    }
    let _lock = try_lock(&record)?.ok_or("Cannot remove a song while it is processing")?;
    let output = root.join("outputs").join(job_id.to_string());
    if output.is_dir() {
        fs::remove_dir_all(&output).map_err(|error| error.to_string())?;
    }
    fs::remove_dir_all(record.parent().unwrap()).map_err(|error| error.to_string())?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interrupted_preparation_returns_to_pending_and_preserves_tempo() {
        let storage = tempfile::tempdir().unwrap();
        let record = storage.path().join(RECORD);
        let mut job = initialize(
            ImportMetadata {
                title: "Song".into(),
                artist: "".into(),
                language: "en".into(),
                year: "".into(),
                genre: "".into(),
            },
            "Song.mp3".into(),
            String::new(),
        );
        job.real_bpm = Some(121.0);
        transition(&record, &mut job, Status::SearchingLyrics, "Searching").unwrap();
        let running = try_lock(&record).unwrap().unwrap();
        assert!(load(&record).unwrap().unwrap().status == Status::SearchingLyrics);
        drop(running);
        let recovered = load(&record).unwrap().unwrap();
        assert!(recovered.status == Status::Pending);
        assert_eq!(recovered.real_bpm, Some(121.0));
    }
}
