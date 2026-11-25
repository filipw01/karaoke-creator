mod alignment_models;
mod api;
mod basic_pitch;
mod catalog;
mod chart_output;
mod crepe;
mod external_chart;
mod library;
mod library_api;
mod managed_alignment;
mod managed_api;
mod managed_benchmark;
mod managed_editor;
mod managed_ingest;
mod managed_state;
mod media;
mod melodia;
mod native_pitch;
mod note_assignment;
mod pitch_contour;
mod pitch_detection;
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
    .find(|directory| directory.join("desktop_worker.py").is_file())
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

    let directory = directory.to_path_buf();
    tauri::async_runtime::spawn(async move {
        let address = std::net::SocketAddr::from(([127, 0, 0, 1], 8000));
        if let Err(error) = api::serve(directory, address, false).await {
            log::warn!("Unable to start development API: {error}");
        }
    });
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let arguments: Vec<_> = std::env::args_os().skip(1).collect();
    if let Some(flag) = arguments.first() {
        if flag == "--serve-api" || flag == "--serve-alignment-worker" {
            let result = (|| -> Result<(), String> {
                let [_, directory, address] = arguments.as_slice() else {
                    return Err("Usage: app --serve-api <backend directory> <address:port>".into());
                };
                let address = address
                    .to_str()
                    .ok_or("Invalid API listen address")?
                    .parse()
                    .map_err(|error: std::net::AddrParseError| error.to_string())?;
                tokio::runtime::Runtime::new()
                    .map_err(|error| error.to_string())?
                    .block_on(api::serve(
                        directory.into(),
                        address,
                        flag == "--serve-alignment-worker",
                    ))
            })();
            if let Err(error) = result {
                eprintln!("{error}");
                std::process::exit(1);
            }
            return;
        }
    }
    let utility = match arguments.as_slice() {
        [flag] if flag == "--desktop-utility" => Some(chart_output::utility().map_err(Into::into)),
        [flag, source] if flag == "--desktop-crepe" => {
            Some(crepe::run(std::path::Path::new(source)))
        }
        [flag, source] if flag == "--desktop-melodia" => {
            Some(melodia::run(std::path::Path::new(source)))
        }
        [flag, model, source] if flag == "--desktop-pitch-melodia" => Some(melodia::fuse(
            std::path::Path::new(model),
            std::path::Path::new(source),
        )),
        [flag, model, source] if flag == "--desktop-pitch" => Some(basic_pitch::run(
            std::path::Path::new(model),
            std::path::Path::new(source),
            None,
        )),
        [flag, model, source, cutoff] if flag == "--desktop-pitch" => Some((|| {
            let cutoff = cutoff
                .to_str()
                .ok_or("Invalid high-pass frequency")?
                .parse()?;
            basic_pitch::run(
                std::path::Path::new(model),
                std::path::Path::new(source),
                Some(cutoff),
            )
        })()),
        [flag, source] if flag == "--desktop-pyin" => {
            Some(native_pitch::run_pyin(std::path::Path::new(source), None))
        }
        [flag, source, mix] if flag == "--desktop-pyin" => Some(native_pitch::run_pyin(
            std::path::Path::new(source),
            Some(std::path::Path::new(mix)),
        )),
        [flag, source, rate, highpass] if flag == "--desktop-pitch-audio" => Some((|| {
            let rate = rate.to_str().ok_or("Invalid pitch sample rate")?.parse()?;
            let highpass = highpass
                .to_str()
                .ok_or("Invalid high-pass frequency")?
                .parse()?;
            native_pitch::run_decode(std::path::Path::new(source), rate, highpass)
        })()),
        [flag, frame, hop] if flag == "--desktop-pitch-rms" => Some((|| {
            let frame = frame.to_str().ok_or("Invalid RMS frame length")?.parse()?;
            let hop = hop.to_str().ok_or("Invalid RMS hop length")?.parse()?;
            native_pitch::run_rms(frame, hop)
        })()),
        [flag, ..]
            if matches!(
                flag.to_str(),
                Some(
                    "--desktop-pyin"
                        | "--desktop-pitch"
                        | "--desktop-crepe"
                        | "--desktop-melodia"
                        | "--desktop-pitch-melodia"
                        | "--desktop-pitch-audio"
                        | "--desktop-pitch-rms"
                        | "--desktop-utility"
                )
            ) =>
        {
            Some(Err("Invalid Rust utility arguments".into()))
        }
        _ => None,
    };
    if let Some(result) = utility {
        if let Err(error) = result {
            eprintln!("{error}");
            std::process::exit(1);
        }
        return;
    }
    #[cfg(debug_assertions)]
    let backend = backend_directory();
    #[cfg(not(debug_assertions))]
    let backend = None;
    #[cfg(debug_assertions)]
    if let Some(backend) = &backend {
        start_dev_backend(backend);
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
        .setup(move |app| {
            let data = app.path().app_data_dir()?;
            let settings = data.join("working-directory.json");
            #[cfg(debug_assertions)]
            let settings = {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
                if let Some(backend) = &backend {
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
