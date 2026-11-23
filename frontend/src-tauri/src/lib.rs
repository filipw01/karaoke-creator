mod external_chart;
mod library;

use tauri::Manager;

#[cfg(debug_assertions)]
fn backend_directory() -> Option<std::path::PathBuf> {
    let current = std::env::current_dir().ok();
    let executable = std::env::current_exe().ok();
    [
        current.as_deref(),
        executable.as_deref().and_then(std::path::Path::parent),
    ]
    .into_iter()
    .flatten()
    .flat_map(std::path::Path::ancestors)
    .map(|directory| directory.join("backend"))
    .find(|directory| directory.join("main.py").is_file())
}

#[cfg(debug_assertions)]
fn start_dev_backend(directory: &std::path::Path) {
    if std::net::TcpStream::connect_timeout(
        &std::net::SocketAddr::from(([127, 0, 0, 1], 8000)),
        std::time::Duration::from_millis(300),
    )
    .is_ok()
    {
        return;
    }

    let venv = if cfg!(windows) {
        "venv/Scripts/python.exe"
    } else {
        "venv/bin/python"
    };
    let python = directory.join(venv);
    let python = if python.is_file() {
        python
    } else {
        std::path::PathBuf::from("python3")
    };
    if let Err(error) = std::process::Command::new(python)
        .args([
            "-m",
            "uvicorn",
            "main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ])
        .current_dir(directory)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
    {
        log::warn!("Unable to start development Python backend: {error}");
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            library::list_native_songs,
            library::set_native_library_directory,
            library::pick_native_library_directory,
            library::open_external_chart,
            library::save_external_chart,
            library::remove_external_chart,
            library::export_external_chart,
        ])
        .setup(|app| {
            let data = app.path().app_data_dir()?;
            let settings = data.join("working-directory.json");

            #[cfg(debug_assertions)]
            let settings = {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
                if let Some(backend) = backend_directory() {
                    start_dev_backend(&backend);
                    backend.join("outputs/working-directory.json")
                } else {
                    settings
                }
            };

            app.manage(library::Library::new(settings, data.join("playback-audio")));
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while building tauri application");
}
