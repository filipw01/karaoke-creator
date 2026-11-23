#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
  tauri::Builder::default()
    .plugin(tauri_plugin_dialog::init())
    .setup(|app| {
      if cfg!(debug_assertions) {
        app.handle().plugin(
          tauri_plugin_log::Builder::default()
            .level(log::LevelFilter::Info)
            .build(),
        )?;
      }

      if std::net::TcpStream::connect_timeout(
        &"127.0.0.1:8000".parse().unwrap(),
        std::time::Duration::from_millis(300),
      )
      .is_err()
      {
        let backend_dir = std::env::current_dir()
          .ok()
          .and_then(|start| {
            let mut dir = start;
            loop {
              let candidate = dir.join("backend");
              if candidate.join("main.py").is_file() {
                return candidate.canonicalize().ok();
              }
              if !dir.pop() {
                break;
              }
            }
            None
          })
          .or_else(|| {
            std::env::current_exe()
              .ok()
              .and_then(|exe| {
                let mut dir = exe.parent()?.to_path_buf();
                loop {
                  let candidate = dir.join("backend");
                  if candidate.join("main.py").is_file() {
                    return candidate.canonicalize().ok();
                  }
                  if !dir.pop() {
                    break;
                  }
                }
                None
              })
          });

        if let Some(backend_dir) = backend_dir {
          let python = backend_dir.join("venv/bin/python");
          let python = if python.exists() {
            python
          } else {
            std::path::PathBuf::from("python3")
          };

          let _ = std::process::Command::new(python)
            .args(["-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"])
            .current_dir(&backend_dir)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn();
        }
      }

      Ok(())
    })
    .run(tauri::generate_context!())
    .expect("error while building tauri application");
}
