# Moving the desktop backend to Rust

Rust now owns the browser API, remote worker HTTP ingress, desktop import
queue, managed job lifecycle, editor, library, media operations, and chart
generation. Pitch detection, high-pass rescue, candidate selection, and
pitch-to-word and syllable assignment run in Rust. Rust supervises FFmpeg,
yt-dlp, whisper.cpp, and the optional Essentia MELODIA helper.

The production Python entry point is a versioned stdin/stdout worker for
Demucs and canonical lyric reconciliation. ForceAlign and MMS_FA acoustic
inference runs through ONNX Runtime in Rust. The Python benchmark harness
remains available for offline comparisons.
[API](frontend/src-tauri/src/api.rs) · [Managed workflow](frontend/src-tauri/src/managed_alignment.rs) · [Inference worker](backend/desktop_worker.py)

The Tauri development shell starts the Rust API on loopback and shares
`backend/outputs/jobs.json` with the browser entry point. The desktop library
commands can open and edit existing charts independently. Packaging the
managed API, inference worker, and native model assets for clean machines
remains a separate distribution task. [Tauri startup](frontend/src-tauri/src/lib.rs) · [Vite proxy](frontend/vite.config.ts)

## Component trade-offs

| Component | Implementation | Pros | Cons and parity risks |
| --- | --- | --- | --- |
| Tauri shell, frontend transport, and progress | Tauri commands handle desktop file access. Axum serves managed operations, progress events, and browser clients. | Keeps the UI flows and TypeScript contracts while sharing a single job engine. | Two transports need contract coverage during rollout. Audio playback and downloads need their own file transport. [Frontend API](frontend/src/api.ts) · [Tauri commands](https://v2.tauri.app/develop/calling-rust/) |
| Uploads, queue, retries, and persistence | Rust owns one JSON state writer and schedules preparation and processing. It recovers existing job snapshots. | Desktop owns restart recovery and worker lifecycle. Native picking can copy media directly without a multipart round trip. | Needs parity for in-flight jobs, failed-phase retries, lyrics approval, missing or relocated files, and reuse of existing artifacts. Keep the current serialized processing limit initially. [Job state and recovery](frontend/src-tauri/src/managed_state.rs) |
| Song library and external charts | Port library discovery, chart parse/edit, publishing, and deletion with filesystem tests. | Enables opening and editing existing charts with no inference worker. A good first desktop feature to migrate. | Preserve encoding fallbacks, relative asset lookup, and path containment. The existing code and UI delete an external chart's containing folder when it is nested, and only the chart at the library root. The README describes chart-only deletion. Preserve the confirmed behavior and fix the documentation discrepancy at cutover. [Library routes](frontend/src-tauri/src/library_api.rs) · [External chart codec](backend/processing/external_chart.py) · [Delete confirmation](frontend/src/App.tsx) |
| Media processing and playback | Invoke the same FFmpeg and ffprobe commands from Rust. Use a scoped Tauri asset or custom protocol for seekable audio. | Reuses the tested codec pipeline and removes Python wrappers. Large media stays on disk. | Bundle target-specific binaries and verify the selected FFmpeg build's distribution terms. Test accurate seeking with the existing PCM preference/cache, range reads, conversion failures, and video remux compatibility. Expect similar codec runtime. [Audio helpers](backend/processing/audio.py) · [Tauri asset scope](https://v2.tauri.app/security/asset-protocol/) · [FFmpeg licensing](https://ffmpeg.org/legal.html) |
| Filename metadata and online lyrics | Move filename heuristics, iTunes release lookup, and LRCLIB/lyrics.ovh requests to Rust. Keep the optional llama.cpp metadata parser in the worker if users rely on it. | Straightforward HTTP and text processing. Offline chart work and manual lyrics can work without Python. | Match title cleanup, provider ranking and fallbacks, timeouts, and empty results. Heuristic output can differ from the optional model. [Catalog and lyrics providers](frontend/src-tauri/src/catalog.rs) |
| YouTube search and download | Have Rust supervise the upstream yt-dlp executable and map its structured output into the existing job contract. | Keeps extraction updates independent from app logic and can eliminate a separate Python installation for this feature. | A standalone build may embed Python. Provision the required JavaScript runtime and challenge components, and preserve format selection, progress, file discovery, retries, and cancellation. [Rust yt-dlp integration](frontend/src-tauri/src/youtube.rs) · [yt-dlp](https://github.com/yt-dlp/yt-dlp) |
| Vocal separation | Keep Demucs in the worker. Trial a native model only with representative A/B separation and chart results. | Preserves `htdemucs`, higher-quality model options, and existing stem behavior. Worker isolation lets the app recover from inference failure. | PyTorch, model weights, and hardware support remain distribution costs. A native implementation also needs equivalent segmentation, shifts, stitching, and stem levels. The existing worker path can return copies of the mix when Demucs is unavailable, so use a real separation run for acceptance. [Demucs wrapper](backend/processing/vocals.py) · [HTDemucs upstream](https://github.com/facebookresearch/demucs/blob/main/demucs/htdemucs.py) |
| BPM detection | Rust estimates tempo from audio. | Small DSP surface and a candidate for eliminating one librosa use. | Half- and double-tempo choices or small grid shifts affect beat quantization. Librosa is also used elsewhere, so a BPM port alone does little for package size. [Tempo detector](frontend/src-tauri/src/tempo.rs) · [UltraStar grid](backend/processing/ultrastar.py) |
| Draft lyrics and optional ASR anchors | Rust invokes whisper.cpp for lyric drafts. Optional Qwen can supply ASR anchors to forced alignment. | A native implementation has a plausible offline deployment path, and the worker contract can accommodate alternative sources. | Singing accuracy, English and Polish output, timing, and downstream alignment need comparison. Canonical forced timing remains its own stage. [whisper.cpp](https://github.com/ggml-org/whisper.cpp) |
| Forced alignment and evidence selection | Run stock MMS_FA and ForceAlign acoustic inference in Rust through ONNX Runtime. Keep canonical token reconciliation and manual-lock merging in the Python worker during this stage. Qwen remains an optional worker-side transcript anchor. | Reuses the app's existing ONNX runtime and supports English plus Polish routing. | Model output parity, CTC decoding, chunk stitching, and large model distribution need verification. Canonical word matching and lock behavior still cross the Python worker boundary. [Rust alignment inference](frontend/src-tauri/src/alignment_models.rs) · [Model exporter](backend/scripts/export_alignment_models.py) |
| Pitch detection and note assignment | Rust runs Basic Pitch ONNX, native pYIN, and optional CREPE ONNX or Essentia MELODIA. Rust assigns lyric notes and split timings. | The existing ONNX model makes inference a good native candidate. Intermediate comparisons can isolate model differences from note-assignment differences. | Resampling, frame offsets, overlap, activity, high-pass rescue, candidate ranking, split notes, and canonical pitch-source spans affect audible results. Native runtime packaging and accelerator support vary by target. [Pitch implementation](backend/processing/pitch.py) · [Basic Pitch](https://github.com/spotify/basic-pitch) · [Rust ort](https://github.com/pykeio/ort) · [Execution providers](https://onnxruntime.ai/docs/execution-providers/index.html) |
| Editor rules, UltraStar generation, and export | Move deterministic chart generation and ZIP writing early. Move result rebuilds, revision selection, alignment switches, boundary locks, and note overrides when Rust takes over managed state. Use a native save dialog and write the archive to disk. | Easy to compare outputs on fixed inputs. Removes the current in-memory ZIP buffer and lets users export existing work without inference. | Keep beat and GAP units, rounding, continuations, line breaks, filenames, optional video, variant selection, and export eligibility. Manual edits must survive restart and realignment as they do today. [UltraStar generator](backend/processing/ultrastar.py) · [Editor routes](frontend/src-tauri/src/managed_editor.rs) |
| Browser API, GPU service, and benchmark tooling | Axum serves browser jobs and authenticated APIs. The Python benchmark harness remains an independent reference. | Keeps existing browser workflows and baselines available. | The canonical reconciliation worker still needs Python. [Worker integration](backend/processing/alignment_worker.py) · [Benchmark suite](backend/benchmark_suite.py) |

## Remaining Python production components

| Component | Complexity | Why |
| --- | --- | --- |
| Demucs separation | Very high | A replacement needs matching stems, model weights, chunking, overlap, resampling, and hardware support. |
| Canonical alignment reconciliation | High | Python maps Rust model spans to saved lyric tokens, applies activity and melody repairs, and preserves manual timing locks. Porting that behavior is a separate step from model inference. |

## Acceptance gates

- **Deterministic outputs.** Compare chart text on fixed inputs and ZIP entry names and contents. Round-trip older jobs and verify job IDs, selected revisions, run IDs, locks, note overrides, and stem paths after restart. Check nested and root chart handling, path boundaries, and deletion scope.
- **Model quality.** Reuse `benchmark_alignment.py` and `benchmark_suite.py` with reference charts and matching audio. Compare canonical coverage, timing within one chart beat, boundary errors, exact MIDI agreement, and split-note accuracy per song and language. Use cached evidence or contours for component tests and real stems for end-to-end comparisons. Review outliers per song before accepting a replacement. [Benchmark implementation](backend/benchmark_alignment.py)
- **Deployment cost.** Compare installer and model-cache sizes, cold and warm stage time, peak memory, and import-to-export time on each intended platform. Rust reduces deployment or orchestration overhead when those are measured bottlenecks. FFmpeg and model inference need their own benchmarks.

## Verification

The Rust Basic Pitch rescue modes matched the prior contour and candidate
arrays on a saved vocal clip. Native note assignment matched all 523 generated
segments and their line breaks from a saved 486-word alignment. CREPE matched
the reference's voicing and pitch bins with dithering disabled. MELODIA and
Basic Pitch fusion matched the standalone and fused frame outputs on the
tested fixtures.

The HTTP integration suite covers import, lyrics confirmation, generated
charts, editing, realignment, ZIP export, recovery, deletion, and token
handling. Its managed flow uses a deterministic inference worker. Real-model smoke tests
processed a vocal clip through Demucs, whisper.cpp drafting, local and
remote legacy alignment, chart generation, and ZIP export. Tauri watch mode
started the Rust API and served existing jobs through Vite. The existing
adaptive pitch-range test expects MIDI 68 and receives 76, which was also the
result before this migration. Packaged deployment and broad song-by-song
inference parity remain unverified.
