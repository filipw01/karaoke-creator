use std::{
    io::{Seek, SeekFrom},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::Duration,
};

use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::{Child, ChildStdout},
    sync::Mutex,
};
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
    let mut input = tempfile::tempfile().map_err(|error| error.to_string())?;
    serde_json::to_writer(&mut input, &request).map_err(|error| error.to_string())?;
    input.rewind().map_err(|error| error.to_string())?;
    let mut output = tempfile::tempfile().map_err(|error| error.to_string())?;
    let mut child = Command::new(python_executable(backend))
        .arg(backend.join("desktop_worker.py"))
        .current_dir(backend)
        .stdin(input)
        .stdout(output.try_clone().map_err(|error| error.to_string())?)
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|error| format!("Unable to start Python inference worker: {error}"))?;

    let Some(status) = child
        .wait_timeout(worker_timeout())
        .map_err(|error| error.to_string())?
    else {
        let _ = child.kill();
        let _ = child.wait();
        return Err("Python inference worker timed out".into());
    };
    if !status.success() {
        return Err(format!("Python inference worker exited with {status}"));
    }
    output
        .seek(SeekFrom::Start(0))
        .map_err(|error| error.to_string())?;
    let response: Value = serde_json::from_reader(output)
        .map_err(|error| format!("Invalid inference worker response: {error}"))?;
    parse_response(response)
}

fn worker_timeout() -> Duration {
    Duration::from_secs(
        std::env::var("KARAOKE_WORKER_TIMEOUT_SECONDS")
            .ok()
            .and_then(|seconds| seconds.parse().ok())
            .unwrap_or(1800),
    )
}

fn parse_response<T: DeserializeOwned>(mut response: Value) -> Result<T, String> {
    if response["version"].as_u64() != Some(1) {
        return Err("Unsupported inference worker response version".into());
    }
    if let Some(error) = response["error"].as_str() {
        return Err(error.to_owned());
    }
    serde_json::from_value(response["result"].take()).map_err(|error| error.to_string())
}

struct AlignmentProcess {
    child: Child,
    stdout: BufReader<ChildStdout>,
}

impl AlignmentProcess {
    fn start(backend: &Path) -> Result<Self, String> {
        let mut command = tokio::process::Command::new(python_executable(backend));
        if std::env::var_os("KARAOKE_RUST_UTILITY").is_none() {
            command.env(
                "KARAOKE_RUST_UTILITY",
                std::env::current_exe().map_err(|error| error.to_string())?,
            );
        }
        let mut child = command
            .arg(backend.join("desktop_worker.py"))
            .arg("--serve")
            .current_dir(backend)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .spawn()
            .map_err(|error| format!("Unable to start Python alignment worker: {error}"))?;
        let stdout = BufReader::new(
            child
                .stdout
                .take()
                .ok_or("Alignment worker stdout unavailable")?,
        );
        Ok(Self { child, stdout })
    }

    async fn exchange(&mut self, request: Value) -> Result<Value, String> {
        let mut payload = serde_json::to_vec(&request).map_err(|error| error.to_string())?;
        payload.push(b'\n');
        let input = self
            .child
            .stdin
            .as_mut()
            .ok_or("Alignment worker stdin unavailable")?;
        input
            .write_all(&payload)
            .await
            .map_err(|error| error.to_string())?;
        let mut line = String::new();
        if self
            .stdout
            .read_line(&mut line)
            .await
            .map_err(|error| error.to_string())?
            == 0
        {
            return Err("Python alignment worker exited before returning a result".into());
        }
        serde_json::from_str(&line)
            .map_err(|error| format!("Invalid inference worker response: {error}"))
    }
}

#[derive(Default)]
pub(crate) struct AlignmentWorker(Mutex<Option<AlignmentProcess>>);

impl AlignmentWorker {
    pub(crate) async fn run(
        &self,
        backend: &Path,
        operation: &str,
        request: Value,
    ) -> Result<Value, String> {
        let mut process = self.0.lock().await;
        if process.is_none() {
            *process = Some(AlignmentProcess::start(backend)?);
        }
        let request = json!({"version": 1, "operation": operation, "request": request});
        match tokio::time::timeout(
            worker_timeout(),
            process.as_mut().unwrap().exchange(request),
        )
        .await
        {
            Ok(Ok(response)) => parse_response(response),
            Ok(Err(error)) => {
                process.take();
                Err(error)
            }
            Err(_) => {
                process.take();
                Err("Python alignment worker timed out".into())
            }
        }
    }
}

pub(crate) fn dispatch(
    request: Value,
    url: &str,
    token: Option<&str>,
    timeout: f64,
) -> Result<Value, String> {
    let timeout = Duration::try_from_secs_f64(timeout).map_err(|error| error.to_string())?;
    let client = reqwest::blocking::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(timeout)
        .build()
        .map_err(|error| error.to_string())?;
    let mut outgoing = client.post(url).json(&request);
    if let Some(token) = token.filter(|value| !value.is_empty()) {
        outgoing = outgoing.header("X-Alignment-Worker-Token", token);
    }
    outgoing
        .send()
        .and_then(reqwest::blocking::Response::error_for_status)
        .and_then(|response| response.json())
        .map_err(|error| error.without_url().to_string())
}
