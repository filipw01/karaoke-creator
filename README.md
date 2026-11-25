# Karaoke Creator

Create UltraStar Deluxe karaoke songs from any audio or video file.

## What it does

1. **Audio extraction** — extracts audio from video files using ffmpeg
2. **Vocal separation** — isolates vocals from instruments using Demucs (Meta)
3. **Lyrics alignment** — preserves a canonical lyric revision, compares original-mix and vocal-stem evidence, and records aligned, estimated, or unmatched word spans
4. **Pitch detection** — analyzes the separated vocal stem with the bundled
   Basic Pitch ONNX tracker (falling back to centered pYIN), confidence-weighted
   note tracking, and continuation-aware split handling
5. **BPM detection** — estimates song tempo in Rust
6. **UltraStar generation** — combines everything into a valid UltraStar Deluxe TXT file

## Prerequisites

- Python 3.11+ for the Demucs and forced-alignment workers
- Rust 1.90+
- Node.js 18+
- ffmpeg installed and available in PATH

## Setup

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

The desktop app runs stock torchaudio MMS_FA and ForceAlign Wav2Vec2 acoustic models through ONNX Runtime in Rust. English starts with ForceAlign and schedules MMS_FA as a background comparison; Polish uses MMS_FA. Python remains the canonical lyric reconciliation and pitch projection worker. Neither SOFA nor WhisperX is part of the active alignment path.

Local API state is stored in `backend/outputs/jobs.json` and recovered after a
restart. For a network deployment, set `KARAOKE_API_TOKEN` and the matching
`VITE_KARAOKE_API_TOKEN` in the frontend environment. Without a token, the API
accepts requests only from loopback clients. Set `KARAOKE_CORS_ORIGINS` to the
exact frontend origins used by the deployment.

### Frontend

```bash
cd frontend
npm install
```

## Running

### Desktop app (Tauri)

```bash
cd frontend
npm install
npm run tauri:dev
```

In development, the Tauri shell starts the Rust API on `127.0.0.1:8000`.
Vite proxies managed-job and playback requests to it. The browser uses the
same Rust API when started with `backend/run.sh`.

Rust handles local imports, yt-dlp downloads, FFmpeg processing, tempo and
pitch tracking, lyrics approval, job state, editor changes, library
publication, and ZIP export. Rust starts a Python subprocess on demand for
Demucs or forced alignment and its model evidence. **Retry publish** reuses
the chart and media from the completed analysis.

Set `KARAOKE_PYTHON` to select the worker interpreter. The Rust backend
also looks for a Python virtual environment under `backend/venv`.
For YouTube search and downloads, install standalone yt-dlp on PATH
or point `KARAOKE_YTDLP_PATH` to it. [yt-dlp release files](https://github.com/yt-dlp/yt-dlp#release-files)

#### Audio backends

Rust detects tempo and uses 120 BPM when the audio has no reliable tempo
estimate. Rust owns loudness normalization, pitch audio resampling and
high-pass filtering, frame energy, and pYIN contour cleanup and accompaniment
harmony. With the Basic Pitch graph available, Rust prepares overlapping
windows, runs inference through ONNX Runtime, and ranks pitch candidates. Set
`BASIC_PITCH_MODEL_PATH` to the ONNX graph and `ORT_DYLIB_PATH` to the
ONNX Runtime 1.24 shared library. Rust assigns notes and syllables after
the worker supplies canonical alignment evidence.

Draft lyrics use whisper.cpp on the machine running the draft. Install `whisper-cli` and set
`KARAOKE_WHISPER_CPP_MODEL` to a multilingual GGML Whisper checkpoint.
Set `KARAOKE_WHISPER_CPP_PATH` if `whisper-cli` is outside PATH. [whisper.cpp and model setup](https://github.com/ggml-org/whisper.cpp#quick-start)

On first launch, choose a working directory for your song library. You can
open, play, edit, and save existing UltraStar charts in the desktop app without
Python. When FFmpeg is on PATH, the app caches a PCM copy of MP3 tracks for
precise seeking. In development, managed jobs appear alongside existing
charts and the selected folder is shared with the Rust API through
`backend/outputs/working-directory.json`. Release builds store the selection
in the app-data directory. Managed job processing and stem separation still
need the development or browser API setup.

Removing an existing chart at the library root deletes its chart file.
Removing a nested chart deletes the containing song folder, as shown in the
confirmation dialog.

### Backend (port 8000)

```bash
cd backend
./run.sh
```

Alignment models are exported once as ONNX files and loaded by the Rust desktop app. From an environment with PyTorch and torchaudio installed:

```bash
python backend/scripts/export_alignment_models.py --output-dir backend/models/alignment
```

Set `ALIGNMENT_MODELS_DIR` to that directory when launching the app, or set
`ALIGNMENT_MMS_ONNX_PATH`, `ALIGNMENT_FORCEALIGN_ONNX_PATH`, and
`ALIGNMENT_VOCAB_PATH` individually. The exporter downloads stock torchaudio
weights. These large ONNX files are local model assets and are not committed to
the repository. Set `ORT_DYLIB_PATH` if ONNX Runtime's dynamic library is not
discoverable on the system.

English uses ForceAlign as the fast first pass and schedules stock MMS_FA in
the background after the chart is published. The editor can also request MMS_FA
for a new alignment run. Polish routes directly to MMS_FA because the
ForceAlign acoustic model is English-only. Both models use the separated vocal
stem. Python maps their spans to canonical lyric tokens and applies pitch,
phrase, and manual-lock behavior.

Pitch detection uses the separated vocal stem with Basic Pitch when its
optional ONNX model/runtime is installed; minimal installations fall back to
pYIN automatically. Basic Pitch keeps a small posterior beam through word
assignment, applies an audio-only key/scale prior, and selects candidates per
detected word/split span so sustained harmonics do not replace the vocal
fundamental. Select a detector explicitly with `PITCH_DETECTOR=basic_pitch` or
`PITCH_DETECTOR=pyin`. The candidate/scale pass can be disabled for diagnostics
with `PITCH_WORD_CANDIDATES=0`; by default, candidates above MIDI 78 receive a
small vocal-range penalty, as do candidates below MIDI 45 (spurious low
subharmonics are common in separated stems). Override the limits and penalty
with `PITCH_WORD_RANGE_MIN_MIDI`, `PITCH_WORD_RANGE_MAX_MIDI`, and
`PITCH_WORD_RANGE_PENALTY` for unusual registers. A conservative audio-derived
register adjustment is enabled by
default; disable it with `PITCH_WORD_ADAPTIVE_RANGE=0` when processing unusual
registers. Basic Pitch runs in a sequential,
single-threaded ONNX session by default; `BASIC_PITCH_INTRA_THREADS` and
`BASIC_PITCH_INTER_THREADS` can raise that deliberately.
For controlled benchmark experiments, `BASIC_PITCH_CANDIDATE_COUNT`,
`BASIC_PITCH_CANDIDATE_SMOOTH_FRAMES`, and `BASIC_PITCH_OVERLAP_FRAMES` expose
the posterior-beam size, candidate smoothing, and window overlap; their
defaults are the RAM-safe production settings.

The optional `PITCH_DETECTOR=torchcrepe` mode uses a float32 CREPE ONNX model.
Set `TORCHCREPE_ONNX_PATH` to the weights or `TORCHCREPE_ONNX_DIR` to a directory
with `tiny.onnx` or `full.onnx`. `TORCHCREPE_MODEL` chooses the model in that directory.
The native decoder emits the center of each decoded pitch bin for repeatable
results. [CREPE ONNX models](https://huggingface.co/FredrikKarlssonSpeech/torchcrepe-onnx)

The optional `PITCH_DETECTOR=melodia` and `PITCH_DETECTOR=basic_pitch_melodia`
modes use the native Essentia library through `karaoke-melodia`. Build the helper
with Essentia's C++ development files, then add it to PATH or set
`KARAOKE_MELODIA_PATH` to the executable. [Essentia installation](https://essentia.upf.edu/installing.html)

```bash
c++ -std=c++20 -O2 $(pkg-config --cflags essentia) \
  frontend/src-tauri/native/melodia.cpp -o karaoke-melodia $(pkg-config --libs essentia)
```

Repeated lyric contexts use a conservative three-semitone disagreement
threshold (`PITCH_CONTEXT_REPAIR_MIN_DELTA`) to correct isolated harmonic
outliers while preserving genuinely transposed repetitions.
Basic Pitch also runs a sequential 300 Hz high-pass rescue by default; it is
only promoted when the filtered track wins by at least six semitones across a
sustained portion of the recording. Disable it with
`BASIC_PITCH_HIGHPASS_RESCUE=0`, or tune the cutoff and gate with
`BASIC_PITCH_HIGHPASS_CUTOFF_HZ` and `BASIC_PITCH_HIGHPASS_MIN_DELTA`.
When that sustained-register trigger fires, a second 750 Hz pass is used to
resolve the octave; tune it with
`BASIC_PITCH_HIGHPASS_SYSTEMATIC_CUTOFF_HZ` and
`BASIC_PITCH_HIGHPASS_FULL_PASS_FRACTION`.
For a deliberate high-quality, serial Demucs run, use `DEMUCS_MODEL=htdemucs_ft`
with `DEMUCS_SHIFTS=1` and `DEMUCS_JOBS=0`; the default remains `htdemucs` to
keep local RAM and processing time predictable.

Use the **Lyrics & alignment** dialog to select MMS_FA for a re-alignment run.
English defaults to ForceAlign, then evaluates MMS_FA in the background. Polish
always routes to MMS_FA. Both models require the ONNX assets described above.

### Word alignment worker

The application keeps canonical lyric revisions and word alignments separate
from pitch/note data. Rust runs acoustic inference with stock MMS_FA or
ForceAlign ONNX models on the separated vocal stem. Python maps the returned
word spans onto saved lyric-token order, handles manual timing locks, and
projects pitch and phrase data. The app does not use faster-whisper, WhisperX,
or SOFA for alignment. Optional Qwen3 support remains a separate worker-side
transcript anchor feature when explicitly enabled.

```bash
pip install -U qwen-asr
export ALIGNMENT_ASR_BACKEND=qwen
export QWEN_DEVICE=cuda:0
```

Qwen model weights are loaded lazily by the optional worker-side transcript
feature. Alignment model inference runs in the Rust app on the local machine.

The configured worker URL must return a complete `AlignmentWorkerResponse`;
the API validates its job, revision, run, token coverage, finite boundaries,
and source-order monotonicity before applying it. Without
`ALIGNMENT_WORKER_URL`, the same contract runs in explicit local mode. The
authenticated worker ingress is `POST /v1/alignments`; it performs model
inference on the worker host instead of acknowledging the request. Canonical
results are available at `GET /api/jobs/{job_id}/alignment`.

### Private alignment benchmark

UltraStar Deluxe TXT files can provide a repeatable timing reference for a
known song. Keep the downloaded reference and matching audio under
`backend/benchmarks/private/` (that directory is ignored), then run:

```bash
cd backend
./venv/bin/python benchmark_alignment.py \
  --reference benchmarks/private/rolling-in-the-deep.txt \
  --audio benchmarks/private/rolling-in-the-deep.mp3 \
  --vocal-audio benchmarks/private/rolling-in-the-deep-vocals.wav \
  --activity-audio benchmarks/private/rolling-in-the-deep-vocals.wav \
  --repeat 3 \
  --json
```

The benchmark converts UltraStar beats to seconds from `#BPM` and `#GAP`,
groups note fragments into words, runs the local alignment worker, and reports
coverage, tolerance pass rate, mean errors, and the worst boundaries. The
optional `--activity-audio` flag also exercises the same vocal-activity and
sustain post-processing used by the application. The reference remains local;
do not commit copyrighted song files.

To score generated pitches, provide a pitch-bearing UltraStar chart:

```bash
./venv/bin/python benchmark_alignment.py \
  --reference benchmarks/private/rolling-in-the-deep.txt \
  --pitch-reference benchmarks/private/rolling-in-the-deep.txt \
  --audio benchmarks/private/rolling-in-the-deep.mp3 \
  --vocal-audio benchmarks/private/rolling-in-the-deep-vocals.wav \
  --activity-audio benchmarks/private/rolling-in-the-deep-vocals.wav \
  --pitch-octave-offset 12 \
  --json
```

The pitch score is exact MIDI agreement for aligned lyric words. Continuation
notes (`~`) remain part of the scored note sequence, and the JSON also
includes per-note and split-word metrics (`pitch_note_accuracy`,
`pitch_split_word_accuracy`). Unvoiced
predictions count as misses, while `pitch_coverage` reports how often a voiced
prediction was available. `--pitch-octave-offset` is a diagnostic for charts
whose register differs from the audio; it does not change production output.
Because chart authors may join or split lyric text differently, the suite also
reports `pitch_global_note_accuracy` and
`pitch_global_note_timing_within_tolerance`: these pair generated note events
to chart notes by time, independently of word-token boundaries. The global
note timing value now measures detected note onset against the UltraStar start
beat, allowing ±1 chart beat (using the chart's `#BPM` and `#GAP`) instead of a
fixed 50 ms window. It is therefore invariant to the song tempo. The timing
score from an explicit `--timing-reference` is intentionally chart-backed; the
global note timing value is the useful detector-only timing diagnostic.
When `--activity-audio` is supplied, JSON also reports
`pitch_canonical_accuracy` for the pre-expansion word spans; the regular
`pitch_accuracy` remains the activity-expanded comparison path. Production
pitch assignment uses the canonical spans so sustain timing repair cannot
steal the next word's onset.
Use `--min-pitch-accuracy .9` as an exact-pitch acceptance gate once the
detector reaches the target; with activity audio, the gate uses the canonical
pre-expansion score that production pitch assignment uses.

To measure the explicit-chart path against the same fixture, add
`--timing-reference benchmarks/private/rolling-in-the-deep.txt`
and `--min-within-tolerance .9`. This is a reference-backed upper-bound
measurement; omit `--timing-reference` for the honest model-only score.

### Historical model benchmarks

Files under `backend/benchmarks/local-four/` preserve earlier local model
comparisons, including SOFA and WhisperX results. Those models are no longer
used by the desktop app. Current production alignment uses Rust ONNX inference
for ForceAlign and MMS_FA.

## Output format

The app generates a ZIP containing:
- `Artist - Title.txt` — UltraStar Deluxe song file
- `Artist - Title.mp3` — full audio mix
- `Artist - Title [VOC].mp3` — isolated vocals
- `Artist - Title [INSTR].mp3` — instrumental (karaoke) track

Place the extracted folder into your UltraStar Deluxe songs directory.

## Tech stack

- **Backend**: Rust and Axum for HTTP, jobs, media, pitch, forced-alignment inference, and charts. Python workers for Demucs and canonical lyric reconciliation.
- **Frontend**: SolidJS, TypeScript, Vite, Tailwind CSS
