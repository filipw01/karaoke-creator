use std::{
    collections::HashSet,
    fmt::Write as _,
    fs::{self, File},
    io::{self, Write as _},
    path::{Component, Path, PathBuf},
};

use serde::Deserialize;
use serde_json::{Value, json};
use tauri::{AppHandle, Manager};
use tauri_plugin_dialog::DialogExt;
use uuid::Uuid;
use zip::{CompressionMethod, ZipWriter, write::SimpleFileOptions};

use crate::{
    external_chart::{Metadata, Note, Phrase},
    library::write_atomically,
    media::LocalImport,
};

#[derive(Deserialize)]
struct Word {
    text: String,
    start: f64,
    end: f64,
    midi_note: i64,
    #[serde(default)]
    continuation_of_previous: bool,
    #[serde(default)]
    alignment_status: String,
}

fn phrases(
    words: Vec<Word>,
    bpm: f64,
    gap_threshold: f64,
    line_breaks: &[usize],
) -> Result<(Vec<Phrase>, f64), String> {
    let boundary_set: HashSet<_> = line_breaks.iter().copied().collect();
    let mut filtered = Vec::new();
    let mut breaks = HashSet::new();
    let mut pending_break = false;
    for (index, word) in words.into_iter().enumerate() {
        pending_break |= boundary_set.contains(&index);
        if matches!(
            word.alignment_status.as_str(),
            "unmatched" | "AlignmentStatus.UNMATCHED"
        ) {
            continue;
        }
        if pending_break && !filtered.is_empty() {
            breaks.insert(filtered.len());
        }
        pending_break = false;
        if !word.start.is_finite() || !word.end.is_finite() {
            return Err("Chart notes require finite word timings".into());
        }
        filtered.push(word);
    }
    let Some(first) = filtered.first() else {
        return Ok((Vec::new(), 0.0));
    };
    if !bpm.is_finite() || bpm <= 0.0 {
        return Err("Chart BPM must be positive".into());
    }
    let gap = first.start;
    let beat =
        |time: f64| (((time - gap).max(0.0) * (bpm * 4.0) * 4.0) / 60.0).round_ties_even() as i64;
    let mut phrases = Vec::new();
    let mut notes = Vec::new();
    for (index, word) in filtered.iter().enumerate() {
        if breaks.contains(&index) {
            finish_phrase(&mut phrases, &mut notes, None);
        }
        let start_beat = beat(word.start);
        let end_beat = beat(word.end);
        let text = if word.continuation_of_previous || notes.is_empty() {
            word.text.clone()
        } else {
            format!(" {}", word.text)
        };
        notes.push(Note {
            note_type: ":".into(),
            start_beat,
            duration_beats: (end_beat - start_beat).max(1),
            pitch: word.midi_note - 60,
            text,
        });
        if let Some(next) = filtered.get(index + 1)
            && !breaks.contains(&(index + 1))
            && next.start - word.end >= gap_threshold
        {
            finish_phrase(&mut phrases, &mut notes, Some(end_beat));
        }
    }
    finish_phrase(&mut phrases, &mut notes, None);
    Ok((phrases, gap * 1000.0))
}

fn finish_phrase(phrases: &mut Vec<Phrase>, notes: &mut Vec<Note>, end_beat: Option<i64>) {
    if let Some(last) = notes.last() {
        let end_beat = end_beat.unwrap_or(last.start_beat + last.duration_beats) + 1;
        phrases.push(Phrase {
            notes: std::mem::take(notes),
            end_beat,
        });
    }
}

fn render(metadata: &Metadata, phrases: &[Phrase], assets: &Assets) -> String {
    let mut chart = format!(
        "#TITLE:{}\n#ARTIST:{}\n#MP3:{}\n",
        metadata.title, metadata.artist, assets.audio_file
    );
    let headers = [
        ("VIDEO", &assets.video_file),
        ("VOCALS", &assets.vocals_file),
        ("INSTRUMENTAL", &assets.instrumental_file),
    ];
    for (tag, value) in headers {
        if !value.is_empty() {
            let _ = writeln!(chart, "#{tag}:{value}");
        }
    }
    let _ = writeln!(
        chart,
        "#BPM:{:.2}\n#GAP:{:.0}",
        metadata.bpm * 4.0,
        metadata.gap_ms
    );
    for (tag, value) in [
        ("LANGUAGE", &metadata.language),
        ("GENRE", &metadata.genre),
        ("YEAR", &metadata.year),
    ] {
        if !value.is_empty() {
            let _ = writeln!(chart, "#{tag}:{value}");
        }
    }
    chart.push_str("#CREATOR:Karaoke Creator\n\n");
    for (index, phrase) in phrases.iter().enumerate() {
        for note in &phrase.notes {
            let _ = writeln!(
                chart,
                "{} {} {} {} {}",
                note.note_type, note.start_beat, note.duration_beats, note.pitch, note.text
            );
        }
        if index + 1 < phrases.len() {
            let _ = writeln!(chart, "- {}", phrase.end_beat);
        }
    }
    chart.push_str("E\n");
    chart
}

#[derive(Deserialize)]
struct Assets {
    audio_file: String,
    #[serde(default)]
    vocals_file: String,
    #[serde(default)]
    instrumental_file: String,
    #[serde(default)]
    video_file: String,
}

impl Assets {
    fn names(&self) -> impl Iterator<Item = &str> {
        [
            &self.audio_file,
            &self.vocals_file,
            &self.instrumental_file,
            &self.video_file,
        ]
        .into_iter()
        .map(String::as_str)
        .filter(|name| !name.is_empty())
    }
}

#[derive(Deserialize)]
pub(crate) struct CompletedChart {
    metadata: Metadata,
    ultrastar_txt: String,
    #[serde(flatten)]
    assets: Assets,
}

impl CompletedChart {
    fn name(&self) -> String {
        let name: String = format!("{} - {}", self.metadata.artist, self.metadata.title)
            .chars()
            .map(|part| {
                if part.is_alphanumeric() || matches!(part, ' ' | '-' | '_' | '(' | ')') {
                    part
                } else {
                    '_'
                }
            })
            .collect();
        let name = name.trim();
        if name.is_empty() {
            "song".into()
        } else {
            name.into()
        }
    }

    fn sources(&self, work: &Path) -> Result<Vec<(&str, PathBuf)>, String> {
        let root = dunce::canonicalize(work).map_err(|error| error.to_string())?;
        self.assets
            .names()
            .map(|name| {
                if name.contains(['/', '\\'])
                    || !matches!(
                        Path::new(name).components().next(),
                        Some(Component::Normal(_))
                    )
                {
                    return Err("Invalid chart asset name".into());
                }
                let file = dunce::canonicalize(root.join(name))
                    .map_err(|error| format!("{name}: {error}"))?;
                if !file.starts_with(&root) || !file.is_file() {
                    return Err(format!("Chart asset is unavailable: {name}"));
                }
                Ok((name, file))
            })
            .collect()
    }

    fn zip(&self, work: &Path, destination: &Path) -> Result<(), String> {
        let sources = self.sources(work)?;
        let parent = destination.parent().ok_or("Choose a ZIP destination")?;
        let mut temp =
            tempfile::NamedTempFile::new_in(parent).map_err(|error| error.to_string())?;
        let name = self.name();
        let create = || -> Result<(), Box<dyn std::error::Error>> {
            let mut archive = ZipWriter::new(temp.as_file_mut());
            let options =
                SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
            archive.start_file(format!("{name}/{name}.txt"), options)?;
            archive.write_all(self.ultrastar_txt.as_bytes())?;
            for (filename, source) in sources {
                // Encoded media is already compressed.
                archive.start_file(
                    format!("{name}/{filename}"),
                    SimpleFileOptions::default()
                        .compression_method(CompressionMethod::Stored)
                        .large_file(true),
                )?;
                io::copy(&mut File::open(source)?, &mut archive)?;
            }
            archive.finish()?.sync_all()?;
            Ok(())
        };
        create().map_err(|error| error.to_string())?;
        temp.persist(destination)
            .map_err(|error| error.to_string())?;
        Ok(())
    }

    fn publish(
        &self,
        job_id: &str,
        work: &Path,
        library: &Path,
        existing: Option<&Path>,
    ) -> Result<PathBuf, String> {
        let sources = self.sources(work)?;
        let root = dunce::canonicalize(library).map_err(|error| error.to_string())?;
        let name = self.name();
        if let Some(existing) = existing
            .and_then(|path| dunce::canonicalize(path).ok())
            .filter(|path| path != &root && path.starts_with(&root) && path.is_dir())
            .or_else(|| recover_publication(&root, &name, job_id))
        {
            for (filename, source) in sources {
                let mut staging = tempfile::NamedTempFile::new_in(&existing)
                    .map_err(|error| error.to_string())?;
                io::copy(
                    &mut File::open(source).map_err(|error| error.to_string())?,
                    &mut staging,
                )
                .map_err(|error| error.to_string())?;
                staging
                    .persist(existing.join(filename))
                    .map_err(|error| error.to_string())?;
            }
            write_atomically(
                &existing.join(format!("{name}.txt")),
                self.ultrastar_txt.as_bytes(),
            )
            .map_err(|error| error.to_string())?;
            return Ok(existing);
        }

        let staging = tempfile::tempdir_in(&root).map_err(|error| error.to_string())?;
        for (filename, source) in sources {
            fs::copy(source, staging.path().join(filename)).map_err(|error| error.to_string())?;
        }
        write_atomically(
            &staging.path().join(format!("{name}.txt")),
            self.ultrastar_txt.as_bytes(),
        )
        .map_err(|error| error.to_string())?;
        fs::write(staging.path().join(".karaoke-creator-job"), job_id)
            .map_err(|error| error.to_string())?;
        for suffix in 1.. {
            let destination = root.join(if suffix == 1 {
                name.clone()
            } else {
                format!("{name} ({suffix})")
            });
            if destination.exists() {
                continue;
            }
            match fs::rename(staging.path(), &destination) {
                Ok(()) => return Ok(destination),
                Err(_) if destination.exists() => continue,
                Err(error) => return Err(error.to_string()),
            }
        }
        unreachable!();
    }
}

fn recover_publication(root: &Path, name: &str, job_id: &str) -> Option<PathBuf> {
    fs::read_dir(root).ok()?.flatten().find_map(|entry| {
        if !entry.file_type().ok()?.is_dir() {
            return None;
        }
        let filename = entry.file_name();
        let folder = filename.to_str()?;
        let suffix = folder.strip_prefix(name)?;
        if !suffix.is_empty()
            && !suffix
                .strip_prefix(" (")
                .and_then(|value| value.strip_suffix(')'))
                .is_some_and(|digits| {
                    !digits.is_empty() && digits.bytes().all(|digit| digit.is_ascii_digit())
                })
        {
            return None;
        }
        let path = entry.path();
        let owner = fs::read_to_string(path.join(".karaoke-creator-job")).ok()?;
        (owner == job_id).then_some(path)
    })
}

#[derive(Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case")]
enum Utility {
    PrepareBrowserMedia {
        backend: PathBuf,
        media: crate::preparation::BrowserMedia,
    },
    ParseFilename {
        filename: String,
    },
    CleanTitle {
        title: String,
    },
    SearchLyrics {
        query: String,
    },
    SearchYoutube {
        backend: PathBuf,
        query: String,
    },
    DownloadYoutube {
        backend: PathBuf,
        job_id: Uuid,
        url: String,
    },
    ExtractAudio {
        source: PathBuf,
        destination: PathBuf,
    },
    PlaybackAudio {
        source: PathBuf,
        cache: PathBuf,
    },
    ScanLibrary {
        directory: Option<PathBuf>,
    },
    ReadExternalChart {
        path: PathBuf,
    },
    ParseExternalChart {
        content: String,
        fallback_title: String,
    },
    UpdateExternalChart {
        content: String,
        phrases: Vec<Phrase>,
        pitch_offset: i64,
    },
    SaveExternalChart {
        path: PathBuf,
        phrases: Vec<Phrase>,
    },
    ExternalAudio {
        root: PathBuf,
        path: PathBuf,
    },
    RemoveExternalChart {
        root: PathBuf,
        path: PathBuf,
    },
    Archive {
        chart: CompletedChart,
        work: PathBuf,
        destination: PathBuf,
    },
    Phrases {
        words: Vec<Word>,
        bpm: f64,
        gap_threshold: f64,
        line_breaks: Vec<usize>,
    },
    Chart {
        metadata: Metadata,
        phrases: Vec<Phrase>,
        assets: Assets,
    },
    WhisperDraft {
        audio: PathBuf,
        language: String,
    },
    EncodeMedia {
        source: PathBuf,
        destination: PathBuf,
        video: bool,
    },
    Publish {
        job_id: String,
        chart: CompletedChart,
        work: PathBuf,
        library: PathBuf,
        existing: Option<PathBuf>,
    },
}

fn execute(request: Utility, mut progress: impl FnMut(Value)) -> Result<Value, String> {
    match request {
        Utility::PrepareBrowserMedia { backend, media } => {
            crate::preparation::prepare_browser_media(&backend, media, progress)
        }
        Utility::ParseFilename { filename } => Ok(crate::catalog::parse_local_filename(filename)),
        Utility::CleanTitle { title } => Ok(json!(crate::catalog::clean_title(&title))),
        Utility::SearchLyrics { query } => Ok(json!(crate::catalog::search(
            &crate::catalog::client()?,
            query.trim(),
            20
        ))),
        Utility::SearchYoutube { backend, query } => {
            Ok(json!(crate::youtube::search(&backend, &query)?))
        }
        Utility::DownloadYoutube {
            backend,
            job_id,
            url,
        } => Ok(json!(crate::youtube::download(
            &backend,
            job_id,
            &url,
            |percent| progress(json!({"progress_percent": percent}))
        )?)),
        Utility::ExtractAudio {
            source,
            destination,
        } => {
            let parent = destination.parent().ok_or("Missing output directory")?;
            fs::create_dir_all(parent).map_err(|error| error.to_string())?;
            let staging =
                tempfile::NamedTempFile::new_in(parent).map_err(|error| error.to_string())?;
            crate::media::write_pcm_wav(&source, staging.path())
                .map_err(|error| error.to_string())?;
            staging
                .persist(&destination)
                .map_err(|error| error.to_string())?;
            Ok(json!(destination))
        }
        Utility::PlaybackAudio { source, cache } => {
            Ok(json!(crate::library::playback_path(&source, &cache)))
        }
        Utility::ScanLibrary { directory } => Ok(json!(crate::library::scan_songs(directory))),
        Utility::ReadExternalChart { path } => Ok(json!(crate::library::read_chart(&path)?)),
        Utility::ParseExternalChart {
            content,
            fallback_title,
        } => {
            let mut parsed = json!(crate::external_chart::parse(&content, &fallback_title)?);
            let headers = crate::external_chart::headers(&content);
            parsed["audio_file"] =
                json!(crate::external_chart::audio_reference(&headers).unwrap_or_default());
            Ok(parsed)
        }
        Utility::UpdateExternalChart {
            content,
            phrases,
            pitch_offset,
        } => {
            let mut parsed = crate::external_chart::parse(&content, "")?;
            parsed.pitch_offset = pitch_offset;
            Ok(json!(crate::external_chart::update(
                &content, &parsed, &phrases
            )?))
        }
        Utility::SaveExternalChart { path, phrases } => {
            Ok(json!(crate::library::save_chart_file(&path, &phrases)?))
        }
        Utility::ExternalAudio { root, path } => {
            let content = crate::library::read_chart(&path)?;
            Ok(json!(crate::library::resolve_audio(&root, &path, &content)))
        }
        Utility::RemoveExternalChart { root, path } => {
            crate::library::remove_chart_file(&root, &path)?;
            Ok(Value::Null)
        }
        Utility::Archive {
            chart,
            work,
            destination,
        } => {
            chart.zip(&work, &destination)?;
            Ok(json!(destination))
        }
        Utility::EncodeMedia {
            source,
            destination,
            video,
        } => {
            crate::media::encode_output(&source, &destination, video)?;
            Ok(json!(destination))
        }
        Utility::WhisperDraft { audio, language } => {
            Ok(json!(crate::whisper_draft::transcribe(&audio, &language)?))
        }
        Utility::Phrases {
            words,
            bpm,
            gap_threshold,
            line_breaks,
        } => {
            let (phrases, gap_ms) = phrases(words, bpm, gap_threshold, &line_breaks)?;
            Ok(json!({"phrases": phrases, "gap_ms": gap_ms}))
        }
        Utility::Chart {
            metadata,
            phrases,
            assets,
        } => Ok(Value::String(render(&metadata, &phrases, &assets))),
        Utility::Publish {
            job_id,
            chart,
            work,
            library,
            existing,
        } => Ok(json!(chart.publish(
            &job_id,
            &work,
            &library,
            existing.as_deref()
        )?)),
    }
}

pub(crate) fn utility() -> Result<(), String> {
    let envelope: Value =
        serde_json::from_reader(io::stdin().lock()).map_err(|error| error.to_string())?;
    if envelope["version"].as_u64() != Some(1) {
        return Err("Unsupported chart utility protocol".into());
    }
    let request =
        serde_json::from_value(envelope["request"].clone()).map_err(|error| error.to_string())?;
    let mut output = io::stdout().lock();
    let result = execute(request, |event| {
        let _ = writeln!(output, "{}", json!({"progress": event}));
        let _ = output.flush();
    })?;
    writeln!(output, "{result}").map_err(|error| error.to_string())
}

#[tauri::command]
pub(crate) async fn save_managed_zip(
    app: AppHandle,
    job_id: Uuid,
    chart: CompletedChart,
) -> Result<(), String> {
    let backend = app
        .state::<LocalImport>()
        .0
        .clone()
        .ok_or("Export requires the local managed-song backend")?;
    tauri::async_runtime::spawn_blocking(move || {
        let output = app
            .dialog()
            .file()
            .set_file_name(format!("{}.zip", chart.name()))
            .add_filter("Karaoke ZIP", &["zip"])
            .blocking_save_file();
        if let Some(output) = output {
            let destination = output.into_path().map_err(|error| error.to_string())?;
            chart.zip(
                &backend.join("outputs").join(job_id.to_string()),
                &destination,
            )?;
        }
        Ok(())
    })
    .await
    .map_err(|error| error.to_string())?
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    #[test]
    fn stages_library_output_and_exports_matching_zip_entries() {
        let temp = tempfile::tempdir().unwrap();
        let work = temp.path().join("work");
        let library = temp.path().join("library");
        fs::create_dir(&work).unwrap();
        fs::create_dir(&library).unwrap();
        fs::write(work.join("Song.mp3"), b"encoded audio").unwrap();
        let chart: CompletedChart = serde_json::from_value(json!({
            "metadata": {"artist": "Artist", "title": "Song", "language": "en", "year": "", "genre": "", "bpm": 120, "gap_ms": 0},
            "ultrastar_txt": "#TITLE:Song\nE\n", "audio_file": "Song.mp3",
        })).unwrap();
        let first = chart.publish("job-1", &work, &library, None).unwrap();
        let second = chart.publish("job-2", &work, &library, None).unwrap();
        assert_ne!(first, second);
        assert_eq!(
            chart.publish("job-1", &work, &library, None).unwrap(),
            first
        );
        assert_eq!(
            chart
                .publish("job-1", &work, &library, Some(&first))
                .unwrap(),
            first
        );
        let name = "Artist - Song/Artist - Song.txt";
        let destination = temp.path().join("song.zip");
        chart.zip(&work, &destination).unwrap();
        let mut archive = zip::ZipArchive::new(File::open(destination).unwrap()).unwrap();
        let mut content = String::new();
        archive
            .by_name(name)
            .unwrap()
            .read_to_string(&mut content)
            .unwrap();
        assert_eq!(
            content,
            fs::read_to_string(first.join("Artist - Song.txt")).unwrap()
        );
        let mut audio = Vec::new();
        archive
            .by_name("Artist - Song/Song.mp3")
            .unwrap()
            .read_to_end(&mut audio)
            .unwrap();
        assert_eq!(audio, b"encoded audio");
    }
}
