use std::{
    fs::{self, File},
    io::{self, Read},
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use serde::Serialize;
use tauri::{AppHandle, Manager};
use tauri_plugin_dialog::DialogExt;
use uuid::Uuid;

const DEFAULT_MAX_UPLOAD_BYTES: u64 = 2 * 1024 * 1024 * 1024;

pub(crate) struct LocalImport(pub(crate) Option<PathBuf>);

#[derive(Serialize)]
pub(crate) struct SelectedMedia {
    path: PathBuf,
    name: String,
    size: u64,
}

#[derive(Serialize)]
pub(crate) struct PreparedMedia {
    pub(crate) original_filename: String,
    pub(crate) has_video: bool,
}

pub(crate) fn write_pcm_wav(source: &Path, destination: &Path) -> io::Result<()> {
    let output = Command::new("ffmpeg")
        .args(["-nostdin", "-loglevel", "error", "-y", "-i"])
        .arg(source)
        .args([
            "-vn",
            "-c:a",
            "pcm_s16le",
            "-ar",
            "44100",
            "-ac",
            "2",
            "-f",
            "wav",
        ])
        .arg(destination)
        .stdout(Stdio::null())
        .output()?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "FFmpeg could not extract audio: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    Ok(())
}

pub(crate) fn encode_output(source: &Path, destination: &Path, video: bool) -> Result<(), String> {
    let options: &[&str] = if video {
        &["-c", "copy", "-movflags", "+faststart"]
    } else {
        &["-codec:a", "libmp3lame", "-qscale:a", "2"]
    };
    let output = Command::new("ffmpeg")
        .args(["-nostdin", "-loglevel", "error", "-y", "-i"])
        .arg(source)
        .args(options)
        .arg(destination)
        .output()
        .map_err(|error| format!("Unable to run FFmpeg: {error}"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_owned());
    }
    Ok(())
}

pub(crate) fn has_video(source: &Path) -> Result<bool, String> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "V",
            "-show_entries",
            "stream=index",
            "-of",
            "json",
        ])
        .arg(source)
        .output()
        .map_err(|error| format!("Unable to run ffprobe: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "ffprobe could not read the media file: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let data: serde_json::Value = serde_json::from_slice(&output.stdout)
        .map_err(|error| format!("Invalid ffprobe output: {error}"))?;
    data["streams"]
        .as_array()
        .map(|streams| !streams.is_empty())
        .ok_or_else(|| "ffprobe did not return media streams".into())
}

fn inspect_media(path: PathBuf) -> Result<SelectedMedia, String> {
    let path =
        dunce::canonicalize(path).map_err(|error| format!("Unable to open media: {error}"))?;
    let metadata = path
        .metadata()
        .map_err(|error| format!("Unable to open media: {error}"))?;
    if !metadata.is_file() || metadata.len() == 0 {
        return Err("Choose a nonempty audio or video file".into());
    }
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("Media filename must be valid Unicode")?
        .to_owned();
    Ok(SelectedMedia {
        path,
        name,
        size: metadata.len(),
    })
}

fn prepare_media(backend: &Path, source: PathBuf, max_bytes: u64) -> Result<String, String> {
    let source = inspect_media(source)?;
    if source.size > max_bytes {
        return Err("Selected file is too large".into());
    }
    let uploads = backend.join("uploads");
    let outputs = backend.join("outputs");
    fs::create_dir_all(&uploads).map_err(|error| error.to_string())?;
    fs::create_dir_all(&outputs).map_err(|error| error.to_string())?;

    let original = tempfile::tempdir_in(&uploads).map_err(|error| error.to_string())?;
    let prepared = tempfile::tempdir_in(&outputs).map_err(|error| error.to_string())?;
    let input_path = original.path().join("input");
    let bytes = io::copy(
        &mut File::open(&source.path)
            .map_err(|error| error.to_string())?
            .take(max_bytes.saturating_add(1)),
        &mut File::create(&input_path).map_err(|error| error.to_string())?,
    )
    .map_err(|error| format!("Unable to copy media: {error}"))?;
    if bytes == 0 || bytes > max_bytes {
        return Err("Selected file is empty or too large".into());
    }

    prepare_audio(
        &input_path,
        &prepared.path().join("audio.wav"),
        original.path().join("prepared-media.json"),
        source.name,
    )?;

    let id = Uuid::new_v4().to_string();
    let output = outputs.join(&id);
    fs::rename(prepared.path(), &output).map_err(|error| error.to_string())?;
    // The upload directory is published last, so registration never observes half an import.
    if let Err(error) = fs::rename(original.path(), uploads.join(&id)) {
        if let Err(cleanup_error) = fs::remove_dir_all(&output) {
            log::warn!("Unable to remove unregistered audio output: {cleanup_error}");
        }
        return Err(error.to_string());
    }
    Ok(id)
}

fn prepare_audio(
    input: &Path,
    audio: &Path,
    manifest: PathBuf,
    original_filename: String,
) -> Result<PreparedMedia, String> {
    let has_video = has_video(input)?;
    write_pcm_wav(input, audio)
        .map_err(|error| format!("Unable to prepare audio using FFmpeg: {error}"))?;
    let metadata = PreparedMedia {
        original_filename,
        has_video,
    };
    fs::write(
        manifest,
        serde_json::to_vec(&metadata).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    Ok(metadata)
}

pub(crate) fn stage_download(
    source: &Path,
    upload: &Path,
    output: &Path,
) -> Result<PreparedMedia, String> {
    let selected = inspect_media(source.to_path_buf())?;
    if selected.size > upload_limit() {
        return Err("Downloaded media is too large".into());
    }
    let target = tempfile::tempdir_in(output).map_err(|error| error.to_string())?;
    let audio = target.path().join("audio.wav");
    let manifest = target.path().join("prepared-media.json");
    let prepared = prepare_audio(&selected.path, &audio, manifest.clone(), selected.name)?;
    // Preserve the original container for the Python editor/export pipeline.
    fs::rename(selected.path, upload.join("input")).map_err(|error| error.to_string())?;
    fs::rename(audio, output.join("audio.wav")).map_err(|error| error.to_string())?;
    fs::rename(manifest, upload.join("prepared-media.json")).map_err(|error| error.to_string())?;
    Ok(prepared)
}

fn upload_limit() -> u64 {
    std::env::var("KARAOKE_MAX_UPLOAD_BYTES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_MAX_UPLOAD_BYTES)
}

#[tauri::command]
pub(crate) async fn pick_local_media(app: AppHandle) -> Result<Option<SelectedMedia>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        app.dialog()
            .file()
            .set_title("Choose audio or video")
            .blocking_pick_file()
            .map(|file| {
                let path = file
                    .simplified()
                    .into_path()
                    .map_err(|error| error.to_string())?;
                inspect_media(path)
            })
            .transpose()
    })
    .await
    .map_err(|error| error.to_string())?
}

#[tauri::command]
pub(crate) async fn inspect_local_media(path: PathBuf) -> Result<SelectedMedia, String> {
    tauri::async_runtime::spawn_blocking(move || inspect_media(path))
        .await
        .map_err(|error| error.to_string())?
}

#[tauri::command]
pub(crate) async fn prepare_local_media(app: AppHandle, path: PathBuf) -> Result<String, String> {
    let backend = app
        .state::<LocalImport>()
        .0
        .clone()
        .ok_or("Local import requires the desktop development backend")?;
    tauri::async_runtime::spawn_blocking(move || prepare_media(&backend, path, upload_limit()))
        .await
        .map_err(|error| error.to_string())?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires ffmpeg and ffprobe"]
    fn prepares_audio_only_and_video_inputs_for_python() {
        let sandbox = tempfile::tempdir().unwrap();
        for video in [false, true] {
            let source = sandbox
                .path()
                .join(if video { "video.mp4" } else { "audio.mp4" });
            let mut encoder = Command::new("ffmpeg");
            encoder.args([
                "-nostdin",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "sine=d=0.2",
            ]);
            if video {
                encoder.args(["-f", "lavfi", "-i", "color=s=64x64:d=0.2", "-c:v", "mpeg4"]);
            }
            assert!(
                encoder
                    .args(["-c:a", "aac"])
                    .arg(&source)
                    .status()
                    .unwrap()
                    .success()
            );

            let id = prepare_media(sandbox.path(), source.clone(), 1024 * 1024).unwrap();
            let upload = sandbox.path().join("uploads").join(&id);
            assert_eq!(
                fs::read(upload.join("input")).unwrap(),
                fs::read(&source).unwrap()
            );
            let manifest: serde_json::Value =
                serde_json::from_slice(&fs::read(upload.join("prepared-media.json")).unwrap())
                    .unwrap();
            assert_eq!(manifest["has_video"], video);
            assert_eq!(
                manifest["original_filename"],
                source.file_name().unwrap().to_str().unwrap()
            );

            let probe = Command::new("ffprobe")
                .args([
                    "-v",
                    "error",
                    "-show_entries",
                    "stream=codec_name,sample_rate,channels",
                    "-of",
                    "json",
                ])
                .arg(sandbox.path().join("outputs").join(&id).join("audio.wav"))
                .output()
                .unwrap();
            assert!(probe.status.success());
            let data: serde_json::Value = serde_json::from_slice(&probe.stdout).unwrap();
            assert_eq!(
                data["streams"],
                serde_json::json!([
                    {"codec_name": "pcm_s16le", "sample_rate": "44100", "channels": 2}
                ])
            );
        }
    }

    #[test]
    fn failed_preparation_does_not_publish_job_directories() {
        let sandbox = tempfile::tempdir().unwrap();
        let backend = sandbox.path().join("backend");
        let source = sandbox.path().join("broken.wav");
        fs::write(&source, b"invalid media").unwrap();
        assert!(prepare_media(&backend, source, 1024).is_err());
        assert_eq!(fs::read_dir(backend.join("uploads")).unwrap().count(), 0);
        assert_eq!(fs::read_dir(backend.join("outputs")).unwrap().count(), 0);
    }
}
