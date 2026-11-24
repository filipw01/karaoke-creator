use std::{path::Path, sync::LazyLock, time::Duration};

use regex::Regex;
use reqwest::{Url, blocking::Client};
use serde_json::{Value, json};

static TITLE_NOISE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?ix)
        \s*[\[(][^)\]]*(?:official|lyrics?|visualizer|remaster(?:ed)?|\b(?:audio|mv|hq|hd|4k|8k)\b)[^)\]]*[\])]\s*
        |\s+(?:[-–—]\s*)?(?:official\s+(?:(?:music|lyric)\s+)?(?:video|audio|mv|visualizer)|lyrics?|lyric\s+video|visualizer)\s*$
        |\s*[-–—]\s*(?:topic|vevo)\s*$",
    )
    .unwrap()
});

pub(crate) fn clean_title(title: &str) -> String {
    let title = Path::new(title)
        .extension()
        .and_then(|ext| ext.to_str())
        .filter(|ext| {
            matches!(
                ext.to_ascii_lowercase().as_str(),
                "mp3" | "wav" | "flac" | "m4a" | "mp4" | "mkv" | "webm" | "avi"
            )
        })
        .and_then(|_| title.rsplit_once('.').map(|(name, _)| name))
        .unwrap_or(title);
    TITLE_NOISE
        .replace_all(title.split('|').next().unwrap_or(title), " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim_matches([' ', '.', '-', '_'])
        .to_owned()
}

#[tauri::command]
pub(crate) fn parse_local_filename(filename: String) -> Value {
    let stem = Path::new(&filename)
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or(&filename);
    let (artist, title) = stem.split_once(" - ").unwrap_or(("", stem));
    json!({"title": clean_title(title), "artist": artist.trim(), "language": "", "genre": "", "year": ""})
}

pub(crate) fn client() -> Result<Client, String> {
    Client::builder()
        .timeout(Duration::from_secs(8))
        .user_agent("KaraokeCreator/1.0")
        .build()
        .map_err(|error| error.to_string())
}

fn get(client: &Client, base: &str, segments: &[&str], query: &[(&str, &str)]) -> Option<Value> {
    let mut url = Url::parse(base).ok()?;
    url.path_segments_mut()
        .ok()?
        .pop_if_empty()
        .extend(segments.iter().copied());
    if !query.is_empty() {
        url.query_pairs_mut().extend_pairs(query);
    }
    client
        .get(url)
        .send()
        .ok()?
        .error_for_status()
        .ok()?
        .json()
        .ok()
}

fn text<'a>(value: &'a Value, key: &str) -> &'a str {
    value[key].as_str().unwrap_or_default().trim()
}

pub(crate) fn release_metadata(client: &Client, title: &str, artist: &str) -> (String, String) {
    let query = format!("{artist} {title}");
    let result = get(
        client,
        "https://itunes.apple.com/search",
        &[],
        &[("term", query.trim()), ("entity", "song"), ("limit", "1")],
    )
    .unwrap_or_default();
    let release = &result["results"][0];
    let year = text(release, "releaseDate")
        .get(..4)
        .filter(|year| year.bytes().all(|byte| byte.is_ascii_digit()))
        .unwrap_or_default();
    (
        text(release, "primaryGenreName").to_owned(),
        year.to_owned(),
    )
}

fn lyrics_ovh(client: &Client, artist: &str, title: &str) -> String {
    let result =
        get(client, "https://api.lyrics.ovh/v1", &[artist, title], &[]).unwrap_or_default();
    text(&result, "lyrics").to_owned()
}

fn lyric_entry(
    title: &str,
    artist: &str,
    album: &str,
    duration: &Value,
    lyrics: &str,
    source: &str,
) -> Value {
    json!({
        "track_name": title, "artist_name": artist, "album_name": album,
        "duration": duration.as_f64().unwrap_or_default(), "lyrics": lyrics, "source": source,
    })
}

pub(crate) fn search(client: &Client, query: &str, limit: usize) -> Vec<Value> {
    use std::collections::HashSet;
    let mut results = Vec::new();
    let mut seen = HashSet::new();
    let lrclib = get(
        client,
        "https://lrclib.net/api/search",
        &[],
        &[("q", query)],
    )
    .unwrap_or_default();
    for item in lrclib.as_array().into_iter().flatten() {
        let lyrics = text(item, "plainLyrics");
        if !lyrics.is_empty() && seen.insert(lyrics.chars().take(200).collect::<String>()) {
            results.push(lyric_entry(
                text(item, "trackName"),
                text(item, "artistName"),
                text(item, "albumName"),
                &item["duration"],
                lyrics,
                "lrclib",
            ));
        }
        if results.len() == limit {
            return results;
        }
    }
    let suggestions =
        get(client, "https://api.lyrics.ovh/suggest", &[query], &[]).unwrap_or_default();
    for item in suggestions["data"].as_array().into_iter().flatten().take(5) {
        let title = text(item, "title_short");
        let title = if title.is_empty() {
            text(item, "title")
        } else {
            title
        };
        let artist = text(&item["artist"], "name");
        if title.is_empty() || artist.is_empty() {
            continue;
        }
        let lyrics = lyrics_ovh(client, artist, title);
        if !lyrics.is_empty() && seen.insert(lyrics.chars().take(200).collect()) {
            results.push(lyric_entry(
                title,
                artist,
                text(&item["album"], "title"),
                &item["duration"],
                &lyrics,
                "lyrics.ovh",
            ));
        }
        if results.len() == limit {
            break;
        }
    }
    results
}

pub(crate) fn find_lyrics(client: &Client, artist: &str, title: &str) -> String {
    let cleaned = clean_title(title);
    for candidate in [&cleaned, title] {
        let result = get(
            client,
            "https://lrclib.net/api/get",
            &[],
            &[("artist_name", artist), ("track_name", candidate)],
        )
        .unwrap_or_default();
        let lyrics = text(&result, "plainLyrics");
        if !lyrics.is_empty() {
            return lyrics.to_owned();
        }
        let lyrics = lyrics_ovh(client, artist, candidate);
        if !lyrics.is_empty() {
            return lyrics;
        }
        if cleaned == title {
            break;
        }
    }
    let query = format!("{artist} {cleaned}");
    search(client, query.trim(), 1)
        .first()
        .map(|result| text(result, "lyrics").to_owned())
        .unwrap_or_default()
}

#[tauri::command]
pub(crate) async fn search_local_lyrics(query: String) -> Result<Vec<Value>, String> {
    if query.trim().is_empty() {
        return Ok(Vec::new());
    }
    tauri::async_runtime::spawn_blocking(move || Ok(search(&client()?, query.trim(), 20)))
        .await
        .map_err(|error| error.to_string())?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_filename_without_presentation_suffixes() {
        let parsed = parse_local_filename("Artist - Song (Official Music Video).mp4".into());
        assert_eq!(parsed["title"], "Song");
        assert_eq!(parsed["artist"], "Artist");
    }
}
