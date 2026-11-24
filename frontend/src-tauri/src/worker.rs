use std::{
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::Duration,
};

use serde::de::DeserializeOwned;
use serde_json::Value;
use wait_timeout::ChildExt;

pub(crate) fn python_executable(backend: &Path) -> PathBuf {
    if let Some(path) = std::env::var_os("KARAOKE_PYTHON") {
        return PathBuf::from(path);
    }
    let relative = if cfg!(windows) {
        "venv/Scripts/python.exe"
    } else {
        "venv/bin/python"
    };
    let local = backend.join(relative);
    if local.is_file() {
        return local;
    }
    // A linked project worktree can reuse the packages in the main checkout.
    if let Some(path) = backend
        .ancestors()
        .find(|directory| {
            directory
                .file_name()
                .is_some_and(|name| name == ".worktrees")
        })
        .and_then(Path::parent)
        .map(|root| root.join("backend").join(relative))
        .filter(|path| path.is_file())
    {
        return path;
    }
    PathBuf::from("python3")
}

pub(crate) fn infer<T: DeserializeOwned>(backend: &Path, mut request: Value) -> Result<T, String> {
    request["version"] = 1.into();
    let mut child = Command::new(python_executable(backend))
        .arg(backend.join("desktop_worker.py"))
        .current_dir(backend)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|error| format!("Unable to start Python inference worker: {error}"))?;

    let encoded = serde_json::to_vec(&request).map_err(|error| error.to_string())?;
    if let Err(error) = child.stdin.take().unwrap().write_all(&encoded) {
        let _ = child.kill();
        let _ = child.wait();
        return Err(format!("Unable to send inference request: {error}"));
    }
    let timeout = std::env::var("KARAOKE_WORKER_TIMEOUT_SECONDS")
        .ok()
        .and_then(|seconds| seconds.parse().ok())
        .unwrap_or(1800);
    if child
        .wait_timeout(Duration::from_secs(timeout))
        .map_err(|error| error.to_string())?
        .is_none()
    {
        let _ = child.kill();
        let _ = child.wait();
        return Err("Python inference worker timed out".into());
    }
    let output = child
        .wait_with_output()
        .map_err(|error| error.to_string())?;
    if !output.status.success() {
        return Err(format!(
            "Python inference worker exited with {}",
            output.status
        ));
    }
    let response: Value = serde_json::from_slice(&output.stdout)
        .map_err(|error| format!("Invalid inference worker response: {error}"))?;
    if response["version"].as_u64() != Some(1) {
        return Err("Unsupported inference worker response version".into());
    }
    if let Some(error) = response["error"].as_str() {
        return Err(error.to_owned());
    }
    serde_json::from_value(response["result"].clone()).map_err(|error| error.to_string())
}
