use std::{
    fs,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use reqwest::Url;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tauri::{AppHandle, Manager};
use uuid::Uuid;

use crate::{media, worker};

pub(crate) fn validate_url(url: &str) -> Result<(), String> {
    let url = Url::parse(url).map_err(|_| "Invalid YouTube URL")?;
    if url.scheme() == "https"
        && url.username().is_empty()
        && url.password().is_none()
        && matches!(
            url.host_str(),
            Some(
                "www.youtube.com"
                    | "youtube.com"
                    | "m.youtube.com"
                    | "music.youtube.com"
                    | "youtu.be"
            )
        )
    {
        Ok(())
    } else {
        Err("Enter a YouTube HTTPS URL".into())
    }
}

fn command(backend: &Path) -> Command {
    let mut cmd = if let Some(path) = std::env::var_os("KARAOKE_YTDLP_PATH") {
        Command::new(path)
    } else {
        let mut cmd = Command::new(worker::python_executable(backend));
        cmd.args(["-m", "yt_dlp"]);
        cmd
    };
    cmd.args(["--ignore-config", "--no-playlist", "--socket-timeout", "30"])
        .arg("--cache-dir")
        .arg(backend.join("outputs/yt-dlp-cache"));
    if let Some(path) =
        std::env::var_os("KARAOKE_YTDLP_NODE_PATH").or_else(|| executable_on_path("node"))
    {
        cmd.arg("--js-runtimes")
            .arg(format!("node:{}", path.to_string_lossy()))
            .args(["--remote-components", "ejs:github"]);
    } else if let Some(path) =
        std::env::var_os("KARAOKE_YTDLP_DENO_PATH").or_else(|| executable_on_path("deno"))
    {
        cmd.arg("--js-runtimes")
            .arg(format!("deno:{}", path.to_string_lossy()))
            .args(["--remote-components", "ejs:github"]);
    }
    cmd
}

fn executable_on_path(name: &str) -> Option<std::ffi::OsString> {
    let name = if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.into()
    };
    std::env::var_os("PATH")
        .into_iter()
        .flat_map(|path| std::env::split_paths(&path).collect::<Vec<_>>())
        .map(|directory| directory.join(&name))
        .find(|path| path.is_file())
        .map(PathBuf::into_os_string)
}

pub(crate) fn search(backend: &Path, query: &str) -> Result<Vec<Value>, String> {
    let output = command(backend)
        .args([
            "--flat-playlist",
            "--skip-download",
            "--ignore-errors",
            "--dump-single-json",
            "--",
        ])
        .arg(format!("ytsearch10:{query}"))
        .output()
        .map_err(|error| error.to_string())?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().into());
    }
    let payload: Value =
        serde_json::from_slice(&output.stdout).map_err(|error| error.to_string())?;
    Ok(payload["entries"].as_array().into_iter().flatten().filter_map(|entry| {
        let id = entry["id"].as_str().filter(|id| !id.is_empty())?;
        let thumbnail = entry["thumbnail"].as_str().or_else(|| {
            entry["thumbnails"].as_array()?.last()?["url"].as_str()
        }).map(str::to_owned).unwrap_or_else(|| format!("https://i.ytimg.com/vi/{id}/maxresdefault.jpg"));
        Some(json!({
            "id": id, "title": entry["title"].as_str().unwrap_or_default(),
            "channel": entry["channel"].as_str().or(entry["uploader"].as_str()).unwrap_or_default(),
            "duration": entry["duration"].as_f64(), "thumbnail": thumbnail,
            "url": format!("https://www.youtube.com/watch?v={id}"),
        }))
    }).collect())
}

#[derive(Deserialize)]
struct DownloadResult {
    path: PathBuf,
    title: Option<String>,
    artist: Option<String>,
}

#[derive(Serialize)]
pub(crate) struct DownloadedMedia {
    #[serde(flatten)]
    pub(crate) media: media::PreparedMedia,
    pub(crate) title: String,
    pub(crate) artist: String,
    upload_path: PathBuf,
    audio_path: PathBuf,
    work_dir: PathBuf,
}

pub(crate) fn download(
    backend: &Path,
    id: Uuid,
    url: &str,
    mut progress: impl FnMut(f64),
) -> Result<DownloadedMedia, String> {
    validate_url(url)?;
    let upload = backend.join("uploads").join(id.to_string());
    let output = backend.join("outputs").join(id.to_string());
    fs::create_dir_all(&upload).map_err(|error| error.to_string())?;
    fs::create_dir_all(&output).map_err(|error| error.to_string())?;
    let staging = tempfile::tempdir_in(&upload).map_err(|error| error.to_string())?;
    let format = "bestvideo[height>=720][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height>=480][ext=mp4]+bestaudio[ext=m4a]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best";
    let mut child = command(backend)
        .args([
            "--format",
            format,
            "--retries",
            "10",
            "--fragment-retries",
            "10",
            "--newline",
            "--progress",
            "--progress-template",
            "download:progress:%(progress._percent_str)s",
            // The explicit template below prints the final output path as JSON.
            "--print",
            "after_move:{\"path\":%(filepath)j,\"title\":%(title)j,\"artist\":%(channel,uploader)j}",
            "--output",
        ])
        .arg(staging.path().join("%(title).180B.%(ext)s"))
        .arg("--")
        .arg(url)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|error| format!("Unable to start yt-dlp: {error}"))?;
    let mut downloaded = None;
    for line in BufReader::new(child.stdout.take().unwrap())
        .lines()
        .map_while(Result::ok)
    {
        if let Some(percent) = line
            .strip_prefix("progress:")
            .and_then(|percent| percent.trim().trim_end_matches('%').parse::<f64>().ok())
        {
            progress(percent.clamp(0.0, 99.0));
        } else if let Ok(value) = serde_json::from_str::<DownloadResult>(&line) {
            downloaded = Some(value);
        }
    }
    if !child.wait().map_err(|error| error.to_string())?.success() {
        return Err("yt-dlp download failed. Check the URL and try again.".into());
    }
    let downloaded = downloaded.ok_or("yt-dlp did not report a downloaded file")?;
    let source = dunce::canonicalize(downloaded.path).map_err(|error| error.to_string())?;
    let staging_root = dunce::canonicalize(staging.path()).map_err(|error| error.to_string())?;
    if !source.starts_with(staging_root) {
        return Err("yt-dlp returned a path outside its staging directory".into());
    }
    let media = media::stage_download(&source, &upload, &output)?;
    Ok(DownloadedMedia {
        media,
        title: downloaded.title.unwrap_or_default(),
        artist: downloaded.artist.unwrap_or_default(),
        upload_path: upload.join("input"),
        audio_path: output.join("audio.wav"),
        work_dir: output,
    })
}

#[tauri::command]
pub(crate) async fn search_local_youtube(
    app: AppHandle,
    query: String,
) -> Result<Vec<Value>, String> {
    let backend = app
        .state::<media::LocalImport>()
        .0
        .clone()
        .ok_or("YouTube search requires the desktop development backend")?;
    tauri::async_runtime::spawn_blocking(move || search(&backend, query.trim()))
        .await
        .map_err(|error| error.to_string())?
}
