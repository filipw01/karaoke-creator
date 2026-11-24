use std::{fs, io, path::Path};

use serde_json::{Map, Value, json};

use crate::library::write_atomically;

pub(crate) type Snapshot = Map<String, Value>;

fn recover(mut job: Snapshot) -> Snapshot {
    let status = job
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("pending");
    match status {
        "downloading" => {
            job.insert("queued_at".into(), json!(""));
            if job
                .get("upload_path")
                .and_then(Value::as_str)
                .is_some_and(|path| Path::new(path).is_file())
            {
                job.insert("status".into(), json!("pending"));
                job.insert("failed_phase".into(), json!(""));
                job.insert("error".into(), json!(""));
                job.insert(
                    "progress_message".into(),
                    json!("Download complete; ready to process."),
                );
            } else {
                job.insert("status".into(), json!("error"));
                job.insert("failed_phase".into(), json!("download"));
                job.insert(
                    "error".into(),
                    json!("The YouTube download was interrupted before it completed."),
                );
                job.insert(
                    "progress_message".into(),
                    json!("Download interrupted; retry is available."),
                );
            }
            job.insert("progress_percent".into(), json!(0.0));
        }
        "awaiting_lyrics" => {
            job.insert("queued_at".into(), json!(""));
            job.insert("error".into(), json!(""));
            job.insert(
                "progress_message".into(),
                json!("Waiting for lyrics approval."),
            );
        }
        "generating"
            if job.get("failed_phase").and_then(Value::as_str) == Some("publish")
                && job
                    .get("result")
                    .and_then(Value::as_object)
                    .is_some_and(|result| !result.is_empty()) =>
        {
            job.insert("queued_at".into(), json!(""));
            job.insert("status".into(), json!("error"));
            job.insert(
                "error".into(),
                json!("Library publication was interrupted."),
            );
            job.insert(
                "progress_message".into(),
                json!("Retry publishing to reuse the finished chart and media."),
            );
        }
        "extracting_audio" | "separating_vocals" | "detecting_bpm" | "searching_lyrics"
        | "aligning_words" | "extracting_lyrics" | "detecting_pitch" | "generating" => {
            job.insert("queued_at".into(), json!(""));
            job.insert("status".into(), json!("pending"));
            job.insert("progress_percent".into(), json!(0.0));
            job.insert("error".into(), json!(""));
            job.insert(
                "progress_message".into(),
                json!("Processing paused after API restart; ready to resume."),
            );
            if let Some(runs) = job.get_mut("alignment_runs").and_then(Value::as_array_mut) {
                for run in runs.iter_mut().filter_map(Value::as_object_mut) {
                    if run.get("status").and_then(Value::as_str) == Some("running") {
                        run.insert("status".into(), json!("failed"));
                        run.insert(
                            "diagnostics".into(),
                            json!({"error":"API restarted during alignment"}),
                        );
                    }
                }
            }
        }
        _ => {}
    }
    if let Some(result) = job.get_mut("result").and_then(Value::as_object_mut)
        && result
            .get("alignment_comparison")
            .and_then(|value| value["status"].as_str())
            == Some("running")
    {
        result.insert(
            "alignment_comparison".into(),
            json!({
                "status": "failed",
                "sofa_available": true,
                "legacy_available": false,
                "error": "Background comparison was interrupted by an API restart"
            }),
        );
    }
    job
}

pub(crate) fn restore(path: &Path) -> Result<Snapshot, String> {
    let data = match fs::read(path) {
        Ok(data) => data,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Snapshot::new()),
        Err(error) => return Err(format!("Could not read saved karaoke jobs: {error}")),
    };
    let jobs: Snapshot = serde_json::from_slice(&data).map_err(|error| error.to_string())?;
    Ok(jobs
        .into_iter()
        .filter_map(|(id, value)| {
            let Value::Object(job) = value else {
                return None;
            };
            let mut job = job;
            let stored_id = job.get("id")?;
            if stored_id.as_str() == Some(&id)
                || (stored_id.is_number() && stored_id.to_string() == id)
            {
                job.insert("id".into(), json!(id));
                Some((id, Value::Object(recover(job))))
            } else {
                None
            }
        })
        .collect())
}

pub(crate) fn store(path: &Path, jobs: &Snapshot) -> Result<(), String> {
    let parent = path.parent().ok_or("Missing karaoke job state directory")?;
    fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    let contents = serde_json::to_vec_pretty(jobs).map_err(|error| error.to_string())?;
    write_atomically(path, &contents).map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restart_recovers_active_states_without_losing_chart_data() {
        let temp = tempfile::tempdir().unwrap();
        let upload = temp.path().join("upload.wav");
        fs::write(&upload, "existing upload").unwrap();
        let path = temp.path().join("nested/jobs.json");
        let snapshot = json!({
            "download": {"id":"download","status":"downloading","upload_path":upload},
            "interrupted": {"id":"interrupted","status":"downloading"},
            "review": {"id":"review","status":"awaiting_lyrics","found_lyrics":"Words"},
            "analysis": {"id":"analysis","status":"aligning_words","alignment_runs":[{"id":"run","status":"running"}]},
            "publish": {"id":"publish","status":"generating","failed_phase":"publish","result":{"ultrastar_txt":"#TITLE:Song"}},
            "published": {"id":"published","status":"done","result":{
                "ultrastar_txt":"#TITLE:Other","alignment_comparison":{"status":"running"},"note_overrides":{"0":{"pitch":8}}
            }},
        });
        assert!(restore(&path).unwrap().is_empty());
        store(&path, snapshot.as_object().unwrap()).unwrap();
        let restored = restore(&path).unwrap();
        assert_eq!(restored["download"]["status"], "pending");
        assert_eq!(restored["interrupted"]["failed_phase"], "download");
        assert_eq!(restored["review"]["found_lyrics"], "Words");
        assert_eq!(
            restored["analysis"]["alignment_runs"][0]["status"],
            "failed"
        );
        assert_eq!(restored["publish"]["status"], "error");
        assert_eq!(
            restored["publish"]["result"]["ultrastar_txt"],
            "#TITLE:Song"
        );
        assert_eq!(
            restored["published"]["result"]["alignment_comparison"]["status"],
            "failed"
        );
        assert_eq!(
            restored["published"]["result"]["note_overrides"]["0"]["pitch"],
            8
        );
    }
}
