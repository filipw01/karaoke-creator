use std::{collections::HashMap, sync::LazyLock};

use encoding_rs::{ISO_8859_2, WINDOWS_1250};
use regex::Regex;
use serde::{Deserialize, Serialize};

static NOTE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^([:*FRG])\s+(-?[0-9]+)\s+(-?[0-9]+)\s+(-?[0-9]+)(.*)$").unwrap()
});
static VOICE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^P([0-9]+)$").unwrap());
static BREAK: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^-\s*(-?[0-9]+)").unwrap());

#[derive(Serialize, Deserialize)]
pub(crate) struct Note {
    pub(crate) note_type: String,
    pub(crate) start_beat: i64,
    pub(crate) duration_beats: i64,
    pub(crate) pitch: i64,
    pub(crate) text: String,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct Phrase {
    #[serde(default)]
    pub(crate) notes: Vec<Note>,
    #[serde(default)]
    pub(crate) end_beat: i64,
}

#[derive(Deserialize, Serialize)]
pub(crate) struct Metadata {
    pub(crate) title: String,
    pub(crate) artist: String,
    pub(crate) language: String,
    pub(crate) genre: String,
    pub(crate) year: String,
    pub(crate) bpm: f64,
    pub(crate) gap_ms: f64,
}

#[derive(Serialize)]
pub(crate) struct Chart {
    pub(crate) metadata: Metadata,
    pub(crate) phrases: Vec<Phrase>,
    pub(crate) pitch_offset: i64,
}

pub(crate) fn decode(raw: &[u8]) -> String {
    if let Ok(text) = std::str::from_utf8(raw.strip_prefix(b"\xef\xbb\xbf").unwrap_or(raw)) {
        return text.to_owned();
    }

    // Python's strict CP1250 decoder falls back to ISO-8859-2 on these bytes.
    let encoding = if raw
        .iter()
        .any(|byte| matches!(byte, 0x81 | 0x83 | 0x88 | 0x90 | 0x98))
    {
        ISO_8859_2
    } else {
        WINDOWS_1250
    };
    encoding.decode_without_bom_handling(raw).0.into_owned()
}

fn lines(content: &str) -> impl Iterator<Item = &str> {
    content
        .split_inclusive(['\r', '\n'])
        .scan(false, |previous_cr, part| {
            let skip = *previous_cr && part == "\n";
            *previous_cr = part.ends_with('\r');
            Some((!skip).then_some(part.trim_end_matches(['\r', '\n'])))
        })
        .flatten()
}

pub(crate) fn headers(content: &str) -> HashMap<String, &str> {
    lines(content)
        .filter_map(|line| line.strip_prefix('#')?.split_once(':'))
        .map(|(key, value)| (key.trim().to_ascii_uppercase(), value.trim()))
        .collect()
}

pub(crate) fn audio_reference<'a>(headers: &HashMap<String, &'a str>) -> Option<&'a str> {
    ["MP3", "AUDIO", "OGG"]
        .into_iter()
        .filter_map(|key| headers.get(key))
        .map(|value| value.trim_matches(['"', '\'']))
        .find(|value| !value.is_empty())
}

fn is_relative(headers: &HashMap<String, &str>) -> bool {
    headers.get("RELATIVE").is_some_and(|value| {
        matches!(
            value.to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn get_number(headers: &HashMap<String, &str>, key: &str) -> Result<f64, String> {
    headers
        .get(key)
        .unwrap_or(&"0")
        .replace(',', ".")
        .parse::<f64>()
        .ok()
        .filter(|value| value.is_finite())
        .ok_or_else(|| format!("Invalid #{key} value"))
}

fn finish_phrase(phrases: &mut Vec<Phrase>, notes: &mut Vec<Note>, marker: i64) {
    if let Some(last) = notes.last() {
        let end_beat = (last.start_beat + last.duration_beats).max(marker);
        phrases.push(Phrase {
            notes: std::mem::take(notes),
            end_beat,
        });
    }
}

pub(crate) fn parse(content: &str, fallback_title: &str) -> Result<Chart, String> {
    let headers = headers(content);
    let bpm = get_number(&headers, "BPM")?;
    let gap_ms = get_number(&headers, "GAP")?;
    if bpm <= 0.0 {
        return Err("Chart must have a positive #BPM value".into());
    }

    let relative = is_relative(&headers);
    let mut voice_one = true;
    let mut previous_start = 0i64;
    let mut phrases = Vec::new();
    let mut notes = Vec::new();
    let mut pitches = Vec::new();

    for (index, line) in lines(content).enumerate() {
        let trimmed = line.trim();
        if let Some(voice) = VOICE.captures(trimmed) {
            finish_phrase(&mut phrases, &mut notes, 0);
            voice_one = voice[1].trim_start_matches('0') == "1";
            if voice_one {
                previous_start = 0;
            }
            continue;
        }
        if trimmed == "E" {
            break;
        }
        if !voice_one {
            continue;
        }
        if line.starts_with('-') {
            let end_beat = BREAK
                .captures(line)
                .and_then(|marker| marker[1].parse().ok());
            finish_phrase(&mut phrases, &mut notes, end_beat.unwrap_or(0));
            continue;
        }
        let Some(note) = NOTE.captures(line) else {
            continue;
        };

        let invalid_note = || format!("Invalid note timing or pitch on line {}", index + 1);
        let mut start_beat = note[2].parse::<i64>().map_err(|_| invalid_note())?;
        let duration_beats = note[3].parse::<i64>().map_err(|_| invalid_note())?;
        let pitch = note[4].parse::<i64>().map_err(|_| invalid_note())?;
        if start_beat < 0 || duration_beats < 0 {
            return Err(invalid_note());
        }
        if relative {
            start_beat = start_beat
                .checked_add(previous_start)
                .ok_or_else(invalid_note)?;
        }
        start_beat
            .checked_add(duration_beats)
            .ok_or_else(invalid_note)?;
        previous_start = start_beat;
        pitches.push(pitch);
        notes.push(Note {
            note_type: note[1].to_owned(),
            start_beat,
            duration_beats,
            pitch,
            text: note[5].to_owned(),
        });
    }

    finish_phrase(&mut phrases, &mut notes, 0);
    if pitches.is_empty() {
        return Err("Chart contains no editable notes in voice 1".into());
    }
    pitches.sort_unstable();
    let middle = pitches.len() / 2;
    let pitch_offset =
        if (i128::from(pitches[middle]) + i128::from(pitches[(pitches.len() - 1) / 2])) >= 100 {
            60
        } else {
            0
        };
    for note in phrases.iter_mut().flat_map(|phrase| &mut phrase.notes) {
        note.pitch = note
            .pitch
            .checked_sub(pitch_offset)
            .ok_or("Note pitch is out of range")?;
    }

    let get = |key, default| {
        headers
            .get(key)
            .copied()
            .filter(|value| !value.is_empty())
            .unwrap_or(default)
            .to_owned()
    };
    Ok(Chart {
        metadata: Metadata {
            title: get("TITLE", fallback_title),
            artist: get("ARTIST", ""),
            language: get("LANGUAGE", ""),
            genre: get("GENRE", ""),
            year: get("YEAR", ""),
            bpm: bpm / 4.0,
            gap_ms,
        },
        phrases,
        pitch_offset,
    })
}

pub(crate) fn update(content: &str, chart: &Chart, phrases: &[Phrase]) -> Result<String, String> {
    if !phrases.iter().any(|phrase| !phrase.notes.is_empty()) {
        return Err("A chart must contain at least one note".into());
    }
    let relative = is_relative(&headers(content));
    let mut generated = Vec::new();
    let mut previous_start = 0;
    for (index, phrase) in phrases.iter().enumerate() {
        if phrase.end_beat < 0 {
            return Err("Phrase end beat must be non-negative".into());
        }
        for note in &phrase.notes {
            if !matches!(note.note_type.as_str(), ":" | "*" | "F" | "R" | "G") {
                return Err(format!("Unsupported note type: {}", note.note_type));
            }
            if note.text.contains(['\n', '\r']) {
                return Err("Note text cannot contain line breaks".into());
            }
            if note.start_beat < 0
                || note.duration_beats < 0
                || note.start_beat.checked_add(note.duration_beats).is_none()
            {
                return Err("Invalid note timing".into());
            }

            let start = if relative {
                note.start_beat - previous_start
            } else {
                note.start_beat
            };
            if start < 0 {
                return Err(
                    "Relative charts require notes to remain in chronological order".into(),
                );
            }
            previous_start = note.start_beat;
            let pitch = note
                .pitch
                .checked_add(chart.pitch_offset)
                .ok_or("Note pitch is out of range")?;
            let space = if note.text.starts_with(char::is_whitespace) {
                ""
            } else {
                " "
            };
            generated.push(format!(
                "{} {start} {} {pitch}{space}{}",
                note.note_type, note.duration_beats, note.text
            ));
        }
        if index + 1 < phrases.len() {
            generated.push(format!("- {}", phrase.end_beat));
        }
    }

    let mut output = Vec::new();
    let mut voice_one = true;
    let mut inserted = false;
    for line in lines(content) {
        if let Some(voice) = VOICE.captures(line.trim()) {
            voice_one = voice[1].trim_start_matches('0') == "1";
        }
        if voice_one && line.starts_with('-') {
            continue;
        }
        if voice_one && !inserted && (NOTE.is_match(line) || line.trim() == "E") {
            output.extend(generated.iter().map(String::as_str));
            inserted = true;
        }
        if !voice_one || !NOTE.is_match(line) {
            output.push(line);
        }
    }
    Ok(output.join("\n") + "\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_duet_round_trips_pitch_offset_and_second_voice() {
        let content = "#TITLE: Test\r\n#BPM: 360,5\r\n#RELATIVE: yes\r\nP1\r\n: 2 3 60 first\r\n- 6\r\n* 5 2 62 second\r\nP2\r\n: 1 2 50 other\r\nE\r\n";
        let mut chart = parse(content, "fallback").unwrap();
        assert_eq!(chart.metadata.bpm, 90.125);
        assert_eq!(chart.phrases[0].end_beat, 6);
        assert_eq!(chart.phrases[1].notes[0].start_beat, 7);
        assert_eq!(chart.phrases[1].notes[0].pitch, 2);
        chart.phrases[0].notes[0].pitch = 4;
        let saved = update(content, &chart, &chart.phrases).unwrap();
        assert!(
            saved.contains("P1\n: 2 3 64 first\n- 6\n* 5 2 62 second\nP2\n: 1 2 50 other\nE\n")
        );
    }

    #[test]
    fn accepts_bom_and_legacy_encoding_but_rejects_invalid_timing() {
        assert_eq!(decode(b"\xef\xbb\xbfTitle"), "Title");
        assert_eq!(decode(b"\xa3\xf3d\x9f"), "Łódź");
        assert_eq!(decode(b"\x81\xa3"), "\u{81}Ł");
        assert!(parse("#BPM:120\r: 1 -2 0 hello\rE\r", "fallback").is_err());
        assert!(parse("#BPM:120\r: 1 2 0 hello\rE\r", "fallback").is_ok());
    }
}
