# Karaoke Creator

Create UltraStar Deluxe karaoke songs from any audio or video file.

## What it does

1. **Audio extraction** — extracts audio from video files using ffmpeg
2. **Vocal separation** — isolates vocals from instruments using Demucs (Meta)
3. **Lyrics alignment** — preserves a canonical lyric revision, compares original-mix and vocal-stem evidence, and records aligned, estimated, or unmatched word spans
4. **Pitch detection** — analyzes the separated vocal stem with the bundled
   Basic Pitch ONNX tracker (falling back to centered pYIN), confidence-weighted
   note tracking, and continuation-aware split handling
5. **BPM detection** — detects song tempo using librosa
6. **UltraStar generation** — combines everything into a valid UltraStar Deluxe TXT file

## Prerequisites

- Python 3.11+
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

The local path uses WhisperX CTC forced alignment for canonical word timing and automatically selects CPU when no CUDA device is available. Faster-whisper remains a local anchor source, not a canonical timing source. Optional Qwen model backends are loaded only when enabled; they never send audio or lyrics to a third-party service.

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

In development, the desktop shell starts the Python API on `127.0.0.1:8000` when it is not already running. The Vite frontend uses it for managed jobs and stem separation.

On first launch, choose a working directory for your song library. You can
open, play, edit, and save existing UltraStar charts in the desktop app without
Python. When FFmpeg is on PATH, the app caches a PCM copy of MP3 tracks for
precise seeking. In development, managed jobs appear alongside existing
charts and the selected folder is shared with the Python API through
`backend/outputs/working-directory.json`. Release builds store the selection
in the app-data directory. Managed job processing and stem separation still
need the development or browser API setup.

Removing an existing chart at the library root deletes its chart file.
Removing a nested chart deletes the containing song folder, as shown in the
confirmation dialog.

### Backend (port 8000)

```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000 --reload-exclude venv
```

WhisperX CTC is enabled by default. Set the device explicitly only when needed:

```bash
cd backend
source venv/bin/activate
export WHISPERX_DEVICE=auto                         # cuda when available, otherwise cpu
uvicorn main:app --reload --host 0.0.0.0 --port 8000 --reload-exclude venv
```

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
registers. The optional MELODIA fusion remains available
for a deliberate comparison via `PITCH_DETECTOR=basic_pitch_melodia`; it is not
the default on RAM-constrained machines. Basic Pitch runs in a sequential,
single-threaded ONNX session by default; `BASIC_PITCH_INTRA_THREADS` and
`BASIC_PITCH_INTER_THREADS` can raise that deliberately.
For controlled benchmark experiments, `BASIC_PITCH_CANDIDATE_COUNT`,
`BASIC_PITCH_CANDIDATE_SMOOTH_FRAMES`, and `BASIC_PITCH_OVERLAP_FRAMES` expose
the posterior-beam size, candidate smoothing, and window overlap; their
defaults are the RAM-safe production settings.
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

For a diagnostic run that intentionally disables CTC, set `ALIGNMENT_CTC_BACKEND=off`.
If CTC cannot load and the English transcript is sparse, the worker automatically
falls back to the installed ForceAlign backend so it does not fabricate a full
song of timings from activity bleed. Disable that fallback on a RAM-constrained
worker with `ALIGNMENT_AUTO_FORCEALIGN=0`.
Qwen3-ASR remains an optional GPU evidence source:

```bash
export ALIGNMENT_ASR_BACKEND=qwen
export QWEN_DEVICE=cuda:0
```

For an additional, English-only Wav2Vec2 timing source, install `forcealign`.
It is CPU-heavy and runs one serialized model call at a time; use the explicit
setting below to force it even when CTC is available:

```bash
pip install forcealign
export ALIGNMENT_WAV2VEC_BACKEND=forcealign
```

When Basic Pitch is enabled, the worker also analyzes a temporary 60% mix /
40% vocal blend and merges those note boundaries into the vocal evidence
lane. Disable that supplemental pass, or tune its mix ratio, with:

```bash
export ALIGNMENT_MELODY_BACKEND=basic_pitch
export ALIGNMENT_MELODY_BLEND_ENABLED=0
export ALIGNMENT_MELODY_BLEND_MIX=0.6
```

### Frontend (port 5173)

```bash
cd frontend
npm run dev
```

Open http://localhost:5173 in your browser.

### Word alignment worker

The application keeps canonical lyric revisions and word alignments separate
from pitch/note data. By default it uses local WhisperX CTC forced alignment
for canonical timing, with faster-whisper available as matching evidence. A
GPU worker can enable the self-hosted Qwen3 adapter in its own environment
with:

```bash
pip install -U qwen-asr
export ALIGNMENT_ASR_BACKEND=qwen
export QWEN_DEVICE=cuda:0
```

Qwen model weights are loaded lazily by the worker; the FastAPI process only
orchestrates the revision/run contract when a remote worker is configured. To
dispatch model work to a separate GPU service, set the endpoint, shared token,
and the two sides of the audio-volume mapping:

```bash
# API process
export ALIGNMENT_WORKER_URL=http://gpu-worker:8001/v1/alignments
export ALIGNMENT_WORKER_TOKEN=shared-secret
export ALIGNMENT_API_AUDIO_ROOT=/srv/karaoke/backend
export ALIGNMENT_WORKER_AUDIO_ROOT=/srv/karaoke-worker/audio

# GPU worker process
export ALIGNMENT_WORKER_TOKEN=shared-secret
export ALIGNMENT_WORKER_AUDIO_ROOT=/srv/karaoke-worker/audio
```

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

### SOFA comparison pipeline

`benchmark_alternatives.py` compares the normal pipeline with the optional
[SOFA-combined](https://huggingface.co/Silasimo/SOFA-combined) singing forced
aligner (implementation: [qiuqiao/SOFA](https://github.com/qiuqiao/SOFA)). Run
the two pipelines sequentially to keep RAM bounded:

```bash
cd backend
NLTK_DATA=nltk_data ./venv/bin/python benchmark_alternatives.py \
  --pipeline current --json-output benchmarks/private/current-suite-score.json
NLTK_DATA=nltk_data TORCH_NUM_THREADS=1 ./venv/bin/python benchmark_alternatives.py \
  --pipeline sofa --json-output benchmarks/private/sofa-suite-score.json
```

The cached six-song comparison is a proxy when the original private TXT/MP3
pairs are unavailable. It uses the cached generated chart as the reference and
scores word identity, timing within one UltraStar beat, pitch, and split-note
accuracy. Restore the original pairs under `~/Downloads` and use
`benchmark_suite.py` for ground-truth scores. A completed chart can be opened
in the editor with `POST /api/benchmark/import`; the request accepts the chart,
full mix, vocal stem, and optional instrumental-stem paths and returns a normal
local `job_id`.

Production alignment now uses SOFA first when the local checkpoint is present.
The previous ASR/CTC selector remains available as a compatibility fallback:
set `ALIGNMENT_PRIMARY_BACKEND=legacy` or `ALIGNMENT_SOFA_BACKEND=off`. SOFA
supplies word boundaries; pitch is assigned afterward from the separated vocal
contour, with a small boundary context fallback for detector frames that land
just outside a phonetic word span. The SOFA checkpoint currently targets
English, so Polish requests automatically resolve to the preserved ASR/CTC
path. Set `SOFA_ALLOW_NON_ENGLISH=1` only when deliberately testing SOFA on a
non-English job.

The editor's **Use legacy non-SOFA alignment** toggle sends the same choice for
an explicit re-alignment run, so the archived timing behavior can be compared
without restarting the server.

## Output format

The app generates a ZIP containing:
- `Artist - Title.txt` — UltraStar Deluxe song file
- `Artist - Title.mp3` — full audio mix
- `Artist - Title [VOC].mp3` — isolated vocals
- `Artist - Title [INSTR].mp3` — instrumental (karaoke) track

Place the extracted folder into your UltraStar Deluxe songs directory.

## Tech stack

- **Backend**: Python, FastAPI, Demucs, WhisperX, faster-whisper, Basic Pitch ONNX, librosa/pYIN
- **Frontend**: React, TypeScript, Vite, TailwindCSS
