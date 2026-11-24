mod catalog;
mod chart_output;
mod external_chart;
mod library;
mod media;
mod native_pitch;
mod preparation;
mod tempo;
mod whisper_draft;
mod worker;
mod youtube;

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

    let mut command = std::process::Command::new(worker::python_executable(directory));
    if let Ok(executable) = std::env::current_exe() {
        command.env("KARAOKE_RUST_UTILITY", executable);
    }
    if let Err(error) = command
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
    if std::env::args_os().nth(1).as_deref() == Some(std::ffi::OsStr::new("--desktop-pyin")) {
        let dimensions = std::env::args()
            .skip(2)
            .take(2)
            .map(|argument| argument.parse::<usize>())
            .collect::<Result<Vec<_>, _>>();
        let result = match dimensions.as_deref() {
            Ok(&[frame, hop]) => native_pitch::run_pyin(frame, hop),
            _ => Err("pYIN requires a frame and hop length".into()),
        };
        if let Err(error) = result {
            eprintln!("{error}");
            std::process::exit(1);
        }
        return;
    }
    if std::env::args_os().nth(1).as_deref() == Some(std::ffi::OsStr::new("--desktop-pitch")) {
        let model = std::env::args_os().nth(2);
        let result = model
            .as_deref()
            .ok_or("Missing ONNX model".into())
            .and_then(|path| native_pitch::run(std::path::Path::new(path)));
        if let Err(error) = result {
            eprintln!("{error}");
            std::process::exit(1);
        }
        return;
    }
    if std::env::args().nth(1).as_deref() == Some("--desktop-utility") {
        if let Err(error) = chart_output::utility() {
            eprintln!("{error}");
            std::process::exit(1);
        }
        return;
    }
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
            media::pick_local_media,
            media::inspect_local_media,
            media::prepare_local_media,
            preparation::register_native_media,
            preparation::register_native_download,
            preparation::list_native_jobs,
            preparation::native_job_status,
            preparation::start_native_job,
            preparation::retire_native_job,
            preparation::remove_native_job,
            catalog::parse_local_filename,
            catalog::search_local_lyrics,
            youtube::search_local_youtube,
            chart_output::save_managed_zip,
        ])
        .setup(|app| {
            let data = app.path().app_data_dir()?;
            let settings = data.join("working-directory.json");
            #[cfg(debug_assertions)]
            let backend = backend_directory();
            #[cfg(not(debug_assertions))]
            let backend = None;

            #[cfg(debug_assertions)]
            let settings = {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
                if let Some(backend) = &backend {
                    start_dev_backend(backend);
                    backend.join("outputs/working-directory.json")
                } else {
                    settings
                }
            };

            app.manage(library::Library::new(settings, data.join("playback-audio")));
            app.manage(media::LocalImport(backend));
            app.manage(preparation::PreparationQueue::default());
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while building tauri application");
}
