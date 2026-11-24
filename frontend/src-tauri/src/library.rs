use std::{
    fs::{self, File},
    io::{self, BufRead, BufReader, Write},
    path::{Component, Path, PathBuf},
    time::UNIX_EPOCH,
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tauri::{AppHandle, Manager, State};
use tauri_plugin_dialog::DialogExt;

use crate::{
    external_chart::{self, Metadata, Phrase},
    media::write_pcm_wav,
};

pub(crate) struct Library {
    settings_file: PathBuf,
    playback_cache: PathBuf,
}

#[derive(Serialize, Deserialize)]
struct Settings {
    directory: Option<PathBuf>,
}

#[derive(Serialize)]
pub(crate) struct SongList {
    working_directory: Option<PathBuf>,
    songs: Vec<SongListItem>,
}

#[derive(Serialize)]
struct SongListItem {
    id: String,
    kind: &'static str,
    title: String,
    artist: String,
    filename: String,
    folder: String,
    path: String,
    status: &'static str,
    message: &'static str,
    progress_percent: u32,
}

#[derive(Serialize)]
pub(crate) struct ExternalSong {
    path: String,
    filename: String,
    content: String,
    metadata: Metadata,
    phrases: Vec<Phrase>,
    audio_available: bool,
    playback_path: Option<PathBuf>,
}

impl Library {
    pub(crate) fn new(settings_file: PathBuf, playback_cache: PathBuf) -> Self {
        Self {
            settings_file,
            playback_cache,
        }
    }

    fn directory(&self) -> Result<Option<PathBuf>, String> {
        match fs::read(&self.settings_file) {
            Ok(bytes) => serde_json::from_slice::<Settings>(&bytes)
                .map(|settings| settings.directory)
                .map_err(|error| format!("Unable to read working directory: {error}")),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(format!("Unable to read working directory: {error}")),
        }
    }

    fn set_directory(&self, directory: &Path) -> Result<PathBuf, String> {
        let directory = dunce::canonicalize(directory)
            .ok()
            .filter(|directory| directory.is_dir() && directory.to_str().is_some())
            .ok_or("Working directory must be an existing folder")?;
        let settings = Settings {
            directory: Some(directory.clone()),
        };
        let encoded = serde_json::to_vec(&settings).map_err(|error| error.to_string())?;
        if let Some(parent) = self.settings_file.parent() {
            fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        write_atomically(&self.settings_file, &encoded)
            .map_err(|error| format!("Unable to set working directory: {error}"))?;
        Ok(directory)
    }

    fn list(&self) -> Result<SongList, String> {
        Ok(scan_songs(self.directory()?))
    }

    fn chart_path(&self, path: &str) -> Result<(PathBuf, PathBuf), String> {
        let configured = self
            .directory()?
            .ok_or("Working directory is not configured")?;
        let root =
            dunce::canonicalize(configured).map_err(|_| "Working directory is unavailable")?;
        if !Path::new(path)
            .components()
            .all(|part| matches!(part, Component::Normal(_)))
        {
            return Err("Song not found".into());
        }
        let chart = dunce::canonicalize(root.join(path))
            .ok()
            .filter(|chart| chart.starts_with(&root) && chart.is_file() && is_chart(chart))
            .ok_or("Song not found")?;
        Ok((root, chart))
    }

    fn open_chart(&self, path: String) -> Result<ExternalSong, String> {
        let (root, chart) = self.chart_path(&path)?;
        let content = read_chart(&chart)?;
        let parsed = external_chart::parse(&content, chart_stem(&chart))?;
        let playback_path = resolve_audio(&root, &chart, &content)
            .map(|source| playback_path(&source, &self.playback_cache));
        Ok(ExternalSong {
            path,
            filename: chart
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned(),
            content,
            metadata: parsed.metadata,
            phrases: parsed.phrases,
            audio_available: playback_path.is_some(),
            playback_path,
        })
    }

    fn save_chart(&self, path: &str, phrases: &[Phrase]) -> Result<String, String> {
        let (_, chart) = self.chart_path(path)?;
        save_chart_file(&chart, phrases)
    }

    fn remove_chart(&self, path: &str) -> Result<(), String> {
        let (root, chart) = self.chart_path(path)?;
        remove_chart_file(&root, &chart)
    }
}

pub(crate) fn remove_chart_file(root: &Path, chart: &Path) -> Result<(), String> {
    let parent = chart
        .parent()
        .filter(|path| path.starts_with(root))
        .ok_or("Song not found")?;
    let result = if parent == root {
        fs::remove_file(chart)
    } else {
        fs::remove_dir_all(parent)
    };
    result.map_err(|error| format!("Unable to remove song: {error}"))
}

pub(crate) fn save_chart_file(chart: &Path, phrases: &[Phrase]) -> Result<String, String> {
    let content = read_chart(chart)?;
    let parsed = external_chart::parse(&content, chart_stem(chart))?;
    let updated = external_chart::update(&content, &parsed, phrases)?;
    write_atomically(chart, updated.as_bytes())
        .map_err(|error| format!("Failed to save existing song chart: {error}"))?;
    Ok(updated)
}

pub(crate) fn scan_songs(working_directory: Option<PathBuf>) -> SongList {
    let mut songs = Vec::new();
    if let Some(root) = working_directory.as_ref().filter(|root| root.is_dir()) {
        let mut directories = vec![root.clone()];
        while let Some(directory) = directories.pop() {
            let Ok(entries) = fs::read_dir(directory) else {
                continue;
            };
            for entry in entries.flatten() {
                let Ok(file_type) = entry.file_type() else {
                    continue;
                };
                if file_type.is_dir() {
                    directories.push(entry.path());
                } else if file_type.is_file() && is_chart(&entry.path()) {
                    if let Some(song) = list_item(root, &entry.path()) {
                        songs.push(song);
                    }
                }
            }
        }
    }
    songs.sort_by_cached_key(|song| (song.title.to_lowercase(), song.artist.to_lowercase()));
    SongList {
        working_directory,
        songs,
    }
}

fn is_chart(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("txt"))
}

fn chart_stem(path: &Path) -> &str {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or_default()
}

fn list_item(root: &Path, chart: &Path) -> Option<SongListItem> {
    let relative = chart.strip_prefix(root).ok()?;
    let path = relative
        .iter()
        .map(|part| part.to_str())
        .collect::<Option<Vec<_>>>()?
        .join("/");
    let mut bytes = Vec::new();
    let mut file = BufReader::new(File::open(chart).ok()?);
    for _ in 0..80 {
        if file.read_until(b'\n', &mut bytes).ok()? == 0 {
            break;
        }
    }
    let content = external_chart::decode(&bytes);
    let headers = external_chart::headers(&content);
    let (folder, filename) = path.rsplit_once('/').unwrap_or(("", &path));
    Some(SongListItem {
        id: format!("external:{path}"),
        kind: "external",
        title: headers
            .get("TITLE")
            .copied()
            .filter(|title| !title.is_empty())
            .unwrap_or_else(|| chart_stem(chart))
            .to_owned(),
        artist: headers
            .get("ARTIST")
            .copied()
            .unwrap_or_default()
            .to_owned(),
        filename: filename.to_owned(),
        folder: folder.to_owned(),
        path,
        status: "external",
        message: "Existing UltraStar song",
        progress_percent: 0,
    })
}

pub(crate) fn read_chart(path: &Path) -> Result<String, String> {
    fs::read(path)
        .map(|bytes| external_chart::decode(&bytes))
        .map_err(|error| format!("Song file is unavailable: {error}"))
}

pub(crate) fn resolve_audio(root: &Path, chart: &Path, content: &str) -> Option<PathBuf> {
    let headers = external_chart::headers(content);
    let reference = external_chart::audio_reference(&headers)?;
    let source = dunce::canonicalize(chart.parent()?.join(reference)).ok()?;
    (source.starts_with(root) && source.is_file()).then_some(source)
}

pub(crate) fn playback_path(source: &Path, cache: &Path) -> PathBuf {
    let is_mp3 = source
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("mp3"));
    if is_mp3 {
        match cache_pcm(source, cache) {
            Ok(path) => return path,
            Err(error) => log::debug!("Using original audio for playback: {error}"),
        }
    }
    source.to_path_buf()
}

fn cache_pcm(source: &Path, cache: &Path) -> io::Result<PathBuf> {
    let metadata = source.metadata()?;
    let modified = metadata
        .modified()?
        .duration_since(UNIX_EPOCH)
        .map_err(io::Error::other)?;
    let mut key = Sha256::new();
    key.update(source.as_os_str().as_encoded_bytes());
    key.update(metadata.len().to_le_bytes());
    key.update(modified.as_nanos().to_le_bytes());
    let path = cache.join(format!("{}.wav", hex::encode(key.finalize())));
    if path.is_file() {
        return dunce::canonicalize(path);
    }

    fs::create_dir_all(cache)?;
    let temporary = tempfile::NamedTempFile::new_in(cache)?;
    write_pcm_wav(source, temporary.path())?;
    temporary.persist(&path)?;
    dunce::canonicalize(path)
}

pub(crate) fn write_atomically(destination: &Path, contents: &[u8]) -> io::Result<()> {
    let parent = destination
        .parent()
        .ok_or_else(|| io::Error::other("Missing parent folder"))?;
    let mut staging = tempfile::NamedTempFile::new_in(parent)?;
    staging.write_all(contents)?;
    if let Ok(metadata) = destination.metadata() {
        staging.as_file().set_permissions(metadata.permissions())?;
    }
    staging.as_file().sync_all()?;
    staging.persist(destination)?;
    Ok(())
}

async fn run<T: Send + 'static>(
    action: impl FnOnce() -> Result<T, String> + Send + 'static,
) -> Result<T, String> {
    tauri::async_runtime::spawn_blocking(action)
        .await
        .map_err(|error| error.to_string())?
}

#[tauri::command]
pub(crate) async fn list_native_songs(app: AppHandle) -> Result<SongList, String> {
    run(move || app.state::<Library>().list()).await
}

#[tauri::command]
pub(crate) fn set_native_library_directory(
    library: State<'_, Library>,
    directory: PathBuf,
) -> Result<PathBuf, String> {
    library.set_directory(&directory)
}

#[tauri::command]
pub(crate) async fn pick_native_library_directory(
    app: AppHandle,
) -> Result<Option<PathBuf>, String> {
    #[cfg(desktop)]
    return run(move || {
        app.dialog()
            .file()
            .set_title("Choose your songs folder")
            .blocking_pick_folder()
            .map(|folder| {
                folder
                    .simplified()
                    .into_path()
                    .map_err(|error| error.to_string())
            })
            .transpose()
    })
    .await;

    #[cfg(mobile)]
    {
        let _ = app;
        Err("Folder selection is available on desktop".into())
    }
}

#[tauri::command]
pub(crate) async fn open_external_chart(
    app: AppHandle,
    path: String,
) -> Result<ExternalSong, String> {
    run(move || {
        let song = app.state::<Library>().open_chart(path)?;
        if let Some(audio) = &song.playback_path {
            app.asset_protocol_scope()
                .allow_file(audio)
                .map_err(|error| error.to_string())?;
        }
        Ok(song)
    })
    .await
}

#[tauri::command]
pub(crate) async fn save_external_chart(
    app: AppHandle,
    path: String,
    phrases: Vec<Phrase>,
) -> Result<String, String> {
    run(move || app.state::<Library>().save_chart(&path, &phrases)).await
}

#[tauri::command]
pub(crate) async fn remove_external_chart(app: AppHandle, path: String) -> Result<(), String> {
    run(move || app.state::<Library>().remove_chart(&path)).await
}

#[tauri::command]
pub(crate) async fn export_external_chart(app: AppHandle, path: String) -> Result<(), String> {
    run(move || {
        let (_, chart) = app.state::<Library>().chart_path(&path)?;
        let filename = chart.file_name().unwrap_or_default().to_string_lossy();
        let selected = app
            .dialog()
            .file()
            .set_file_name(filename)
            .add_filter("UltraStar chart", &["txt"])
            .blocking_save_file();
        if let Some(selected) = selected {
            let destination = selected.into_path().map_err(|error| error.to_string())?;
            let bytes = fs::read(&chart).map_err(|error| error.to_string())?;
            write_atomically(&destination, &bytes)
                .map_err(|error| format!("Failed to save a copy of the chart: {error}"))?;
        }
        Ok(())
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn library_persists_settings_and_edits_external_chart_in_place() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("songs");
        let folder = root.join("Singer");
        fs::create_dir_all(&folder).unwrap();
        fs::write(folder.join("Track.wav"), b"audio").unwrap();
        let chart = folder.join("Track.TXT");
        fs::write(
            &chart,
            "#TITLE: Track\n#MP3: \"Track.wav\"\n#BPM: 400\n: 0 1 0 hey\nE\n",
        )
        .unwrap();
        let settings = temporary.path().join("app/working-directory.json");
        let library = Library::new(settings.clone(), temporary.path().join("cache"));
        assert!(library.list().unwrap().songs.is_empty());
        let selected = library.set_directory(&root).unwrap();

        let reopened = Library::new(settings, temporary.path().join("cache"));
        let list = reopened.list().unwrap();
        assert_eq!(list.working_directory, Some(selected));
        assert_eq!(list.songs.len(), 1);
        let path = &list.songs[0].path;
        assert_eq!(path, "Singer/Track.TXT");

        let song = reopened.open_chart(path.clone()).unwrap();
        assert!(song.audio_available);
        assert!(song.playback_path.unwrap().ends_with("Singer/Track.wav"));
        let mut phrases = serde_json::to_value(song.phrases).unwrap();
        phrases[0]["notes"][0]["text"] = serde_json::json!(" changed");
        let updated = reopened
            .save_chart(
                path,
                &serde_json::from_value::<Vec<Phrase>>(phrases).unwrap(),
            )
            .unwrap();
        assert!(updated.contains(": 0 1 0 changed"));
        assert_eq!(fs::read_to_string(&chart).unwrap(), updated);
    }

    #[test]
    fn reports_no_audio_for_references_outside_the_library() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("songs");
        fs::create_dir(&root).unwrap();
        fs::write(temporary.path().join("outside.wav"), b"audio").unwrap();
        fs::write(temporary.path().join("outside.txt"), "unrelated").unwrap();
        fs::write(
            root.join("Chart.txt"),
            "#MP3: ../outside.wav\n#BPM: 400\n: 0 1 0 hey\nE\n",
        )
        .unwrap();
        let library = Library::new(
            temporary.path().join("settings.json"),
            temporary.path().join("cache"),
        );
        library.set_directory(&root).unwrap();

        assert!(
            !library
                .open_chart("Chart.txt".into())
                .unwrap()
                .audio_available
        );
        assert!(library.remove_chart("../outside.txt").is_err());
        library.remove_chart("Chart.txt").unwrap();
        assert!(root.is_dir());
        assert!(temporary.path().join("outside.txt").is_file());
    }
}
