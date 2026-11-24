use std::{
    fs,
    path::Path,
    process::{Command, Stdio},
    time::Duration,
};

use serde_json::{Value, json};
use wait_timeout::ChildExt;

fn timed_words(transcript: &Value) -> Vec<Value> {
    transcript["transcription"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|segment| {
            let text = segment["text"].as_str()?.trim();
            let start = segment["offsets"]["from"].as_f64()? / 1000.0;
            let end = segment["offsets"]["to"].as_f64()? / 1000.0;
            if end <= start
                || start < 0.0
                || text.split_whitespace().count() != 1
                || !text.chars().any(char::is_alphanumeric)
            {
                return None;
            }
            let (sum, count) = segment["tokens"]
                .as_array()
                .into_iter()
                .flatten()
                .filter(|token| {
                    token["text"].as_str().is_some_and(|text| {
                        !text.starts_with("[_") && text.chars().any(char::is_alphanumeric)
                    })
                })
                .filter_map(|token| token["p"].as_f64())
                .fold((0.0, 0), |(sum, count), p| (sum + p, count + 1));
            let confidence = if count == 0 {
                0.0
            } else {
                sum / f64::from(count)
            };
            Some(json!({"text": text, "start": start, "end": end, "confidence": confidence}))
        })
        .collect()
}

pub(crate) fn transcribe(audio: &Path, language: &str) -> Result<Vec<Value>, String> {
    if !matches!(language, "en" | "pl") {
        return Err("Whisper draft language must be English or Polish".into());
    }
    let model = std::env::var_os("KARAOKE_WHISPER_CPP_MODEL")
        .ok_or("Set KARAOKE_WHISPER_CPP_MODEL to an installed multilingual Whisper model")?;
    let executable =
        std::env::var_os("KARAOKE_WHISPER_CPP_PATH").unwrap_or_else(|| "whisper-cli".into());
    let staging = tempfile::tempdir().map_err(|error| error.to_string())?;
    let input = staging.path().join("audio.wav");
    let decoded = Command::new("ffmpeg")
        .args(["-nostdin", "-loglevel", "error", "-i"])
        .arg(audio)
        .args(["-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le"])
        .arg(&input)
        .output()
        .map_err(|error| error.to_string())?;
    if !decoded.status.success() {
        return Err(String::from_utf8_lossy(&decoded.stderr).trim().into());
    }

    let output = staging.path().join("draft");
    let mut child = Command::new(executable)
        .arg("--model")
        .arg(model)
        .arg("--file")
        .arg(input)
        .args([
            "--language",
            language,
            "--max-len",
            "1",
            "--split-on-word",
            "--output-json-full",
            "--output-file",
        ])
        .arg(&output)
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|error| format!("Unable to run whisper.cpp: {error}"))?;
    let Some(status) = child
        .wait_timeout(Duration::from_secs(900))
        .map_err(|error| error.to_string())?
    else {
        let _ = child.kill();
        let _ = child.wait();
        return Err("whisper.cpp timed out".into());
    };
    if !status.success() {
        return Err(format!("whisper.cpp exited with {status}"));
    }
    let encoded = fs::read(output.with_extension("json")).map_err(|error| error.to_string())?;
    let transcript = serde_json::from_slice(&encoded).map_err(|error| error.to_string())?;
    let words = timed_words(&transcript);
    if words.is_empty() {
        return Err("whisper.cpp returned no timestamped words".into());
    }
    Ok(words)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_valid_word_offsets_from_milliseconds() {
        let words = timed_words(&json!({"transcription": [
            {"text": " cześć", "offsets": {"from": 100, "to": 350}, "tokens":[
                {"text":" cze", "p":0.75}, {"text":"ść", "p":0.5}, {"text":"[_EOT_]", "p":1.0}
            ]},
            {"text": "!", "offsets": {"from": 350, "to": 400}},
            {"text": "missing time", "offsets": {"from": 500, "to": 450}},
        ]}));
        assert_eq!(
            words,
            vec![json!({"text": "cześć", "start": 0.1, "end": 0.35, "confidence":0.625})]
        );
    }
}
