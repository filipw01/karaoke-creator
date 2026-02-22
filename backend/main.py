"""Karaoke Creator backend - FastAPI application."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import shutil
import time
import uuid
import zipfile
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from models import AlignmentVariant, Job, JobResult, JobStatus, SongMetadata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

app = FastAPI(title="Karaoke Creator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# In-memory job store
jobs: dict[str, Job] = {}
# SSE event queues per job
job_queues: dict[str, list[asyncio.Queue[dict[str, Any]]]] = {}


def _notify(job: Job) -> None:
    """Push a status update to all SSE listeners for a job."""
    msg = {"status": job.status.value, "message": job.progress_message}
    if job.error:
        msg["error"] = job.error
    if job.status == JobStatus.DOWNLOADING:
        msg["progress_percent"] = job.progress_percent
    for q in job_queues.get(job.id, []):
        q.put_nowait(msg)


def _sanitize_filename(name: str) -> str:
    """Create a safe filename from artist - title."""
    return "".join(c if c.isalnum() or c in " -_()" else "_" for c in name).strip()


def _step(job: Job, status: JobStatus, message: str) -> float:
    """Update job status and notify listeners."""
    job.status = status
    job.progress_message = message
    _notify(job)
    logging.getLogger("pipeline").info("[%s] %s", job.id[:8], message)
    return time.time()


async def _process_phase1(job: Job) -> None:
    """Phase 1: Extract audio, separate vocals, detect BPM, search lyrics online.

    Pauses at AWAITING_LYRICS so the user can review/edit lyrics before transcription.
    """
    log = logging.getLogger("pipeline")
    loop = asyncio.get_event_loop()
    pipeline_start = time.time()

    try:
        work_dir = os.path.join(OUTPUT_DIR, job.id)
        os.makedirs(work_dir, exist_ok=True)
        job.work_dir = work_dir
        log.info("[%s] Phase 1 starting for '%s - %s'", job.id[:8], job.metadata.artist, job.metadata.title)

        # --- Step 1: Extract audio (+ detect video) ---
        t = _step(job, JobStatus.EXTRACTING_AUDIO, "Extracting audio from input file...")

        from processing.audio import extract_audio, is_video_file
        job.has_video = is_video_file(job.upload_path)
        log.info("[%s] Input is %s", job.id[:8], "video" if job.has_video else "audio-only")

        audio_path = await loop.run_in_executor(
            None, extract_audio, job.upload_path, work_dir
        )
        job.audio_path = audio_path
        log.info("[%s] Audio extracted in %.1fs -> %s", job.id[:8], time.time() - t, audio_path)

        # --- Step 2: Separate vocals ---
        t = _step(job, JobStatus.SEPARATING_VOCALS, "Separating vocals from instruments (this may take a few minutes)...")

        from processing.vocals import separate_vocals
        vocals_path, instrumental_path = await loop.run_in_executor(
            None, separate_vocals, audio_path, work_dir
        )
        job.vocals_path = vocals_path
        job.instrumental_path = instrumental_path
        log.info("[%s] Vocals separated in %.1fs", job.id[:8], time.time() - t)

        # --- Step 3: Detect BPM ---
        t = _step(job, JobStatus.DETECTING_BPM, "Detecting song tempo (BPM)...")

        from processing.bpm import detect_bpm
        real_bpm = await loop.run_in_executor(None, detect_bpm, audio_path)
        job.real_bpm = real_bpm
        log.info("[%s] BPM detected in %.1fs -> %.1f BPM", job.id[:8], time.time() - t, real_bpm)

        # --- Step 4: Search lyrics online ---
        _step(job, JobStatus.SEARCHING_LYRICS, "Searching for lyrics online...")

        from processing.lyrics_search import search_lyrics
        found = await loop.run_in_executor(
            None, search_lyrics, job.metadata.artist, job.metadata.title
        )
        job.found_lyrics = found

        # Pause for user to review lyrics
        total = time.time() - pipeline_start
        job.status = JobStatus.AWAITING_LYRICS
        job.progress_message = "Review lyrics before transcription"
        _notify(job)
        log.info("[%s] Phase 1 complete in %.1fs — awaiting lyrics confirmation", job.id[:8], total)

    except Exception as e:
        log.exception("[%s] Phase 1 failed after %.1fs", job.id[:8], time.time() - pipeline_start)
        job.status = JobStatus.ERROR
        job.error = str(e)
        job.progress_message = f"Error: {e}"
        _notify(job)


async def _process_phase2(job: Job, lyrics_text: str) -> None:
    """Phase 2: Transcribe lyrics (using confirmed text as prompt), detect pitch, generate output."""
    log = logging.getLogger("pipeline")
    loop = asyncio.get_event_loop()
    phase_start = time.time()

    try:
        work_dir = job.work_dir
        vocals_path = job.vocals_path
        instrumental_path = job.instrumental_path or os.path.join(work_dir, "instrumental.wav")
        has_video = job.has_video

        # --- Step 5: Extract lyrics ---
        has_lyrics = bool(lyrics_text.strip()) and lyrics_text.strip() != "These are song lyrics with singing."
        step_msg = "Aligning provided lyrics to audio..." if has_lyrics else "Transcribing lyrics with word timestamps..."
        t = _step(job, JobStatus.EXTRACTING_LYRICS, step_msg)

        from processing.lyrics import (
            LyricsParams,
            extract_lyrics,
            extract_lyrics_aeneas,
            extract_lyrics_autolyrixalign,
            extract_lyrics_forcealign,
            extract_lyrics_gentle,
            extract_lyrics_mfa,
            extract_lyrics_nemo,
            extract_lyrics_whisperx,
        )
        from processing.lyrics_aligner import extract_lyrics_lyrics_aligner
        from processing.audio import normalize_audio

        params = LyricsParams()
        if has_lyrics:
            params.initial_prompt = lyrics_text.strip()
        lang = job.metadata.language.lower()[:2] if job.metadata.language else None

        # Normalize vocals for transcription/alignment
        transcribe_path = vocals_path
        if params.normalize:
            norm_path = os.path.join(work_dir, "vocals_normalized.wav")
            log.info("[%s] Normalizing vocals...", job.id[:8])
            await loop.run_in_executor(None, normalize_audio, vocals_path, norm_path)
            transcribe_path = norm_path

        # --- Step 6: Detect pitch (once) ---
        t = _step(job, JobStatus.DETECTING_PITCH, "Analyzing vocal pitch...")

        from processing.pitch import detect_pitch, assign_pitch_to_words
        times, frequencies, confidences = await loop.run_in_executor(
            None, detect_pitch, vocals_path
        )
        log.info("[%s] Pitch detected in %.1fs -> %d frames", job.id[:8], time.time() - t, len(times))

        # --- Run all aligners (when lyrics provided) and build alignment_results ---
        alignment_results: dict[str, AlignmentVariant] = {}
        preferred_order = ("forcealign", "lyrics_aligner", "nemo", "whisperx", "aeneas", "mfa", "gentle", "autolyrixalign")
        current_mode = preferred_order[0]
        words_for_download = None
        line_breaks_for_download: list[int] = []

        if has_lyrics:
            extractors: list[tuple[str, Any]] = [
                ("lyrics_aligner", extract_lyrics_lyrics_aligner),
                ("nemo", extract_lyrics_nemo),
                ("forcealign", extract_lyrics_forcealign),
                ("aeneas", extract_lyrics_aeneas),
                ("whisperx", extract_lyrics_whisperx),
                ("mfa", extract_lyrics_mfa),
                ("gentle", extract_lyrics_gentle),
                ("autolyrixalign", extract_lyrics_autolyrixalign),
            ]
            for mode, ext in extractors:
                try:
                    res = await loop.run_in_executor(None, ext, transcribe_path, lyrics_text.strip(), lang)
                    if res[0] is None:
                        log.debug("[%s] %s skipped: %s", job.id[:8], mode, res[1])
                        continue
                    w, lb = res[0], res[1]
                    w = assign_pitch_to_words(w, times, frequencies, confidences)
                    alignment_results[mode] = (w, lb)
                    if words_for_download is None:
                        words_for_download, line_breaks_for_download = w, lb
                        current_mode = mode
                except Exception as e:
                    log.info("[%s] %s failed: %s", job.id[:8], mode, e)
            if words_for_download is None and alignment_results:
                for pref in preferred_order:
                    if pref in alignment_results:
                        current_mode = pref
                        break
                else:
                    current_mode = next(iter(alignment_results))
                words_for_download, line_breaks_for_download = alignment_results[current_mode]
            elif words_for_download is None:
                # All aligners failed; fall back to Whisper transcription
                words_for_download = await loop.run_in_executor(
                    None, extract_lyrics, transcribe_path, lang, params
                )
                line_breaks_for_download = []
                # Map Whisper words to user's lyrics so we preserve case and diacritics (e.g. Polish)
                lyrics_stripped = lyrics_text.strip()
                if lyrics_stripped:
                    from processing.lyrics import preprocess_lyrics_for_alignment, _aligner_words_to_lyrics_order
                    _, all_words, line_breaks_pre = preprocess_lyrics_for_alignment(lyrics_stripped)
                    if all_words:
                        aligned = [
                            type("W", (), {"text": w.text, "start_time": w.start, "end_time": w.end})()
                            for w in words_for_download
                        ]
                        words_for_download = _aligner_words_to_lyrics_order(aligned, all_words)
                        line_breaks_for_download = line_breaks_pre
                words_for_download = assign_pitch_to_words(words_for_download, times, frequencies, confidences)
                alignment_results["whisper"] = (words_for_download, line_breaks_for_download)
                current_mode = "whisper"
        else:
            words_for_download = await loop.run_in_executor(
                None, extract_lyrics, transcribe_path, lang, params
            )
            words_for_download = assign_pitch_to_words(words_for_download, times, frequencies, confidences)
            current_mode = "whisper"

        # --- Step 7: Generate UltraStar format for each variant ---
        t = _step(job, JobStatus.GENERATING, "Generating UltraStar song file...")

        from processing.ultrastar import words_to_phrases, generate_ultrastar_txt
        from processing.audio import convert_to_mp3, copy_video

        safe_name = _sanitize_filename(
            f"{job.metadata.artist} - {job.metadata.title}"
        ) or "song"
        audio_mp3 = f"{safe_name}.mp3"
        vocals_mp3 = f"{safe_name} [VOC].mp3"
        instrumental_mp3 = f"{safe_name} [INSTR].mp3"
        video_mp4 = f"{safe_name}.mp4" if has_video else ""

        def build_variant(w: Any, lb: list[int]) -> AlignmentVariant:
            ph, gap_ms = words_to_phrases(w, job.real_bpm, line_breaks=lb)
            job.metadata.bpm = job.real_bpm
            job.metadata.gap_ms = gap_ms
            txt = generate_ultrastar_txt(
                metadata=job.metadata,
                phrases=ph,
                audio_filename=audio_mp3,
                vocals_filename=vocals_mp3,
                instrumental_filename=instrumental_mp3,
                video_filename=video_mp4,
            )
            return AlignmentVariant(words=w, phrases=ph, ultrastar_txt=txt, gap_ms=gap_ms)

        variants_for_result: dict[str, AlignmentVariant] = {}
        for mode, (w, lb) in alignment_results.items():
            variants_for_result[mode] = build_variant(w, lb)
        phrases, gap_ms = words_to_phrases(words_for_download, job.real_bpm, line_breaks=line_breaks_for_download)
        job.metadata.bpm = job.real_bpm
        job.metadata.gap_ms = gap_ms
        log.info("[%s] Generated %d phrases from %d words (GAP=%.0fms)", job.id[:8], len(phrases), len(words_for_download), gap_ms)

        ultrastar_txt = generate_ultrastar_txt(
            metadata=job.metadata,
            phrases=phrases,
            audio_filename=audio_mp3,
            vocals_filename=vocals_mp3,
            instrumental_filename=instrumental_mp3,
            video_filename=video_mp4,
        )

        log.info("[%s] Converting to MP3...", job.id[:8])
        await loop.run_in_executor(
            None, convert_to_mp3, job.audio_path,
            os.path.join(work_dir, audio_mp3),
        )
        await loop.run_in_executor(
            None, convert_to_mp3, vocals_path,
            os.path.join(work_dir, vocals_mp3),
        )
        await loop.run_in_executor(
            None, convert_to_mp3, instrumental_path,
            os.path.join(work_dir, instrumental_mp3),
        )

        if has_video:
            log.info("[%s] Copying video...", job.id[:8])
            await loop.run_in_executor(
                None, copy_video, job.upload_path,
                os.path.join(work_dir, video_mp4),
            )

        log.info("[%s] File conversion done in %.1fs", job.id[:8], time.time() - t)

        txt_path = os.path.join(work_dir, f"{safe_name}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(ultrastar_txt)

        job.result = JobResult(
            metadata=job.metadata,
            words=words_for_download,
            phrases=phrases,
            ultrastar_txt=ultrastar_txt,
            audio_file=audio_mp3,
            vocals_file=vocals_mp3,
            instrumental_file=instrumental_mp3,
            video_file=video_mp4,
            alignment_results=variants_for_result,
            current_alignment_mode=current_mode,
        )

        total = time.time() - phase_start
        job.status = JobStatus.DONE
        job.progress_message = "Done! Your karaoke song is ready."
        _notify(job)
        modes_str = ", ".join(sorted(variants_for_result.keys()))
        log.info("[%s] Phase 2 complete in %.1fs. Alignment modes: %s", job.id[:8], total, modes_str)

    except Exception as e:
        log.exception("[%s] Phase 2 failed after %.1fs", job.id[:8], time.time() - phase_start)
        job.status = JobStatus.ERROR
        job.error = str(e)
        job.progress_message = f"Error: {e}"
        _notify(job)


# ── API Endpoints ──────────────────────────────────────────────────────────


@app.post("/api/parse-filename")
async def parse_filename_endpoint(body: dict[str, Any]) -> dict[str, str]:
    """Use an LLM to extract artist/title/etc. from a filename."""
    filename = body.get("filename", "")
    if not filename:
        return {"title": "", "artist": "", "language": "", "year": "", "genre": ""}

    loop = asyncio.get_event_loop()
    from processing.metadata import parse_filename

    return await loop.run_in_executor(None, parse_filename, filename)


@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    title: str = Form(""),
    artist: str = Form(""),
    language: str = Form(""),
    genre: str = Form(""),
    year: str = Form(""),
) -> dict[str, str]:
    """Upload an audio/video file and create a processing job."""
    job_id = str(uuid.uuid4())

    # Save uploaded file
    job_upload_dir = os.path.join(UPLOAD_DIR, job_id)
    os.makedirs(job_upload_dir, exist_ok=True)

    filename = file.filename or "input"
    upload_path = os.path.join(job_upload_dir, filename)
    with open(upload_path, "wb") as f:
        content = await file.read()
        f.write(content)

    job = Job(
        id=job_id,
        original_filename=filename,
        upload_path=upload_path,
        metadata=SongMetadata(
            title=title or os.path.splitext(filename)[0],
            artist=artist,
            language=language,
            genre=genre,
            year=year,
        ),
    )
    jobs[job_id] = job

    return {"job_id": job_id}


@app.post("/api/process/{job_id}/start")
async def start_processing(job_id: str) -> dict[str, str]:
    """Start Phase 1 of the processing pipeline."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.PENDING:
        raise HTTPException(status_code=400, detail="Job already started")

    asyncio.create_task(_process_phase1(job))
    return {"status": "started"}


@app.post("/api/jobs/{job_id}/confirm-lyrics")
async def confirm_lyrics(job_id: str, body: dict[str, Any]) -> dict[str, str]:
    """Confirm lyrics and start Phase 2 (transcription + generation)."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.AWAITING_LYRICS:
        raise HTTPException(status_code=400, detail="Job is not awaiting lyrics confirmation")

    lyrics_text = body.get("lyrics", "")
    job.confirmed_lyrics = lyrics_text
    asyncio.create_task(_process_phase2(job, lyrics_text))
    return {"status": "started"}


@app.get("/api/jobs/{job_id}/found-lyrics")
async def get_found_lyrics(job_id: str) -> dict[str, str]:
    """Get the lyrics found online for a job."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"lyrics": job.found_lyrics}


@app.get("/api/lyrics/search")
async def lyrics_search(q: str = "") -> dict[str, Any]:
    """Search for lyrics using a single query string."""
    if not q.strip():
        return {"results": []}

    loop = asyncio.get_event_loop()
    from processing.lyrics_search import search_lyrics_query

    results = await loop.run_in_executor(None, search_lyrics_query, q.strip())
    return {"results": results}


@app.get("/api/process/{job_id}")
async def process_events(job_id: str, request: Request):
    """SSE stream of processing progress for a job."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    if job_id not in job_queues:
        job_queues[job_id] = []
    job_queues[job_id].append(queue)

    # Send current status immediately
    await queue.put({
        "status": job.status.value,
        "message": job.progress_message,
    })

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                yield f"event: progress\ndata: {json.dumps(msg)}\n\n"
                if msg["status"] in (JobStatus.DONE.value, JobStatus.ERROR.value):
                    break
        finally:
            listeners = job_queues.get(job_id, [])
            if queue in listeners:
                listeners.remove(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/jobs/{job_id}/result")
async def get_result(job_id: str) -> dict[str, Any]:
    """Get the processing result as JSON."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.DONE or not job.result:
        raise HTTPException(status_code=400, detail="Job not done yet")

    alignment_results = getattr(job.result, "alignment_results", {}) or {}
    current_mode = getattr(job.result, "current_alignment_mode", "forcealign")
    serialized_alignments: dict[str, dict[str, Any]] = {}
    for k, v in alignment_results.items():
        serialized_alignments[k] = {
            "words": [asdict(w) for w in v.words],
            "phrases": [{"notes": [asdict(n) for n in p.notes], "end_beat": p.end_beat} for p in v.phrases],
            "ultrastar_txt": v.ultrastar_txt,
            "gap_ms": getattr(v, "gap_ms", 0.0),
        }

    return {
        "metadata": asdict(job.result.metadata),
        "words": [asdict(w) for w in job.result.words],
        "phrases": [
            {"notes": [asdict(n) for n in p.notes], "end_beat": p.end_beat}
            for p in job.result.phrases
        ],
        "ultrastar_txt": job.result.ultrastar_txt,
        "audio_file": job.result.audio_file,
        "vocals_file": job.result.vocals_file,
        "instrumental_file": job.result.instrumental_file,
        "video_file": job.result.video_file,
        "alignment_results": serialized_alignments,
        "current_alignment_mode": current_mode,
    }


@app.put("/api/jobs/{job_id}/notes")
async def update_notes(job_id: str, body: dict[str, Any]) -> dict[str, str]:
    """Update the notes/phrases and regenerate the UltraStar TXT."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.result:
        raise HTTPException(status_code=400, detail="No result to update")

    from models import UltraStarNote, Phrase
    from processing.ultrastar import generate_ultrastar_txt

    phrases = []
    for p_data in body.get("phrases", []):
        notes = [
            UltraStarNote(**n) for n in p_data.get("notes", [])
        ]
        phrases.append(Phrase(notes=notes, end_beat=p_data.get("end_beat", 0)))

    job.result.phrases = phrases
    job.result.ultrastar_txt = generate_ultrastar_txt(
        metadata=job.result.metadata,
        phrases=phrases,
        audio_filename=job.result.audio_file,
        vocals_filename=job.result.vocals_file,
        instrumental_filename=job.result.instrumental_file,
        video_filename=job.result.video_file,
    )

    # Rewrite TXT file
    safe_name = _sanitize_filename(
        f"{job.result.metadata.artist} - {job.result.metadata.title}"
    ) or "song"
    txt_path = os.path.join(job.work_dir, f"{safe_name}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(job.result.ultrastar_txt)

    return {"status": "updated"}


@app.post("/api/jobs/{job_id}/regenerate-lyrics")
async def regenerate_lyrics(job_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """Re-run lyrics extraction with custom parameters, then regenerate phrases + TXT."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.result:
        raise HTTPException(status_code=400, detail="No result to update")
    if not job.vocals_path or not os.path.exists(job.vocals_path):
        raise HTTPException(status_code=400, detail="Vocals file not available for re-processing")

    log = logging.getLogger("pipeline")
    loop = asyncio.get_event_loop()

    from processing.lyrics import (
        LyricsParams,
        extract_lyrics,
        extract_lyrics_aeneas,
        extract_lyrics_autolyrixalign,
        extract_lyrics_forcealign,
        extract_lyrics_gentle,
        extract_lyrics_mfa,
        extract_lyrics_nemo,
        extract_lyrics_whisperx,
    )
    from processing.lyrics_aligner import extract_lyrics_lyrics_aligner
    from processing.pitch import detect_pitch, assign_pitch_to_words
    from processing.ultrastar import words_to_phrases, generate_ultrastar_txt

    from processing.audio import normalize_audio

    # Build params from request body (defaults mirror LyricsParams dataclass)
    initial_prompt = body.get("initial_prompt", "These are song lyrics with singing.")
    if (not initial_prompt or initial_prompt == "These are song lyrics with singing." or len(initial_prompt) < 100) and getattr(job, "confirmed_lyrics", ""):
        initial_prompt = job.confirmed_lyrics
    has_lyrics = bool(initial_prompt) and "\n" in initial_prompt and len(initial_prompt) > 100
    params = LyricsParams(
        model_size=body.get("model_size", "large-v3"),
        language=body.get("language") or None,
        vad_filter=body.get("vad_filter", False),
        vad_threshold=body.get("vad_threshold", 0.25),
        vad_min_silence_ms=body.get("vad_min_silence_ms", 200),
        speech_pad_ms=body.get("speech_pad_ms", 250),
        no_speech_threshold=body.get("no_speech_threshold", 0.4),
        beam_size=body.get("beam_size", 5),
        initial_prompt=initial_prompt,
        min_word_confidence=body.get("min_word_confidence", 0.15),
        audio_source=body.get("audio_source", "original"),
        normalize=body.get("normalize", True),
        alignment_mode=body.get("alignment_mode", "forcealign"),
    )

    # Resolve the audio file to feed to Whisper
    if params.audio_source == "original":
        transcribe_path = job.audio_path
    else:
        transcribe_path = job.vocals_path

    # Optionally normalize volume before transcription
    if params.normalize:
        norm_path = os.path.join(job.work_dir, "normalized_for_whisper.wav")
        log.info("[%s] Normalizing audio for Whisper...", job.id[:8])
        await loop.run_in_executor(None, normalize_audio, transcribe_path, norm_path)
        transcribe_path = norm_path

    log.info("[%s] Regenerating lyrics (all aligners) (source=%s, normalize=%s)",
             job.id[:8], params.audio_source, params.normalize)

    times, frequencies, confidences = await loop.run_in_executor(None, detect_pitch, job.vocals_path)
    alignment_results: dict[str, AlignmentVariant] = {}
    reg_preferred = ("forcealign", "lyrics_aligner", "nemo", "whisperx", "aeneas", "mfa", "gentle", "autolyrixalign")
    current_mode = reg_preferred[0]
    words_for_download = None
    line_breaks_for_download: list[int] = []

    if has_lyrics:
        reg_extractors: list[tuple[str, Any]] = [
            ("lyrics_aligner", extract_lyrics_lyrics_aligner),
            ("nemo", extract_lyrics_nemo),
            ("forcealign", extract_lyrics_forcealign),
            ("aeneas", extract_lyrics_aeneas),
            ("whisperx", extract_lyrics_whisperx),
            ("mfa", extract_lyrics_mfa),
            ("gentle", extract_lyrics_gentle),
            ("autolyrixalign", extract_lyrics_autolyrixalign),
        ]
        for mode, ext in reg_extractors:
            try:
                res = await loop.run_in_executor(None, ext, transcribe_path, params.initial_prompt, params.language)
                if res[0] is None:
                    log.debug("[%s] %s skipped: %s", job.id[:8], mode, res[1])
                    continue
                w, lb = res[0], res[1]
                w = assign_pitch_to_words(w, times, frequencies, confidences)
                ph, gap_ms_var = words_to_phrases(w, job.real_bpm, line_breaks=lb)
                txt = generate_ultrastar_txt(
                    metadata=job.metadata,
                    phrases=ph,
                    audio_filename=job.result.audio_file,
                    vocals_filename=job.result.vocals_file,
                    instrumental_filename=job.result.instrumental_file,
                    video_filename=job.result.video_file,
                )
                alignment_results[mode] = AlignmentVariant(words=w, phrases=ph, ultrastar_txt=txt, gap_ms=gap_ms_var)
                if words_for_download is None:
                    words_for_download, line_breaks_for_download = w, lb
                    current_mode = mode
            except Exception as e:
                log.info("[%s] %s failed: %s", job.id[:8], mode, e)
        if words_for_download is None and alignment_results:
            for pref in reg_preferred:
                if pref in alignment_results:
                    current_mode = pref
                    break
            else:
                current_mode = next(iter(alignment_results))
            v = alignment_results[current_mode]
            words_for_download, line_breaks_for_download = v.words, []
        if words_for_download is None:
            words_for_download = await loop.run_in_executor(
                None, extract_lyrics, transcribe_path, params.language, params
            )
            line_breaks_for_download = []
            lyrics_stripped = (params.initial_prompt or "").strip()
            if lyrics_stripped and len(lyrics_stripped) > 100:
                from processing.lyrics import preprocess_lyrics_for_alignment, _aligner_words_to_lyrics_order
                _, all_words, line_breaks_pre = preprocess_lyrics_for_alignment(lyrics_stripped)
                if all_words:
                    aligned = [
                        type("W", (), {"text": w.text, "start_time": w.start, "end_time": w.end})()
                        for w in words_for_download
                    ]
                    words_for_download = _aligner_words_to_lyrics_order(aligned, all_words)
                    line_breaks_for_download = line_breaks_pre
            words_for_download = assign_pitch_to_words(words_for_download, times, frequencies, confidences)
            ph, gap_ms_var = words_to_phrases(words_for_download, job.real_bpm, line_breaks=line_breaks_for_download)
            txt = generate_ultrastar_txt(
                metadata=job.metadata,
                phrases=ph,
                audio_filename=job.result.audio_file,
                vocals_filename=job.result.vocals_file,
                instrumental_filename=job.result.instrumental_file,
                video_filename=job.result.video_file,
            )
            alignment_results["whisper"] = AlignmentVariant(words=words_for_download, phrases=ph, ultrastar_txt=txt, gap_ms=gap_ms_var)
            current_mode = "whisper"
    else:
        words_for_download = await loop.run_in_executor(None, extract_lyrics, transcribe_path, params.language, params)
        words_for_download = assign_pitch_to_words(words_for_download, times, frequencies, confidences)
        ph, _ = words_to_phrases(words_for_download, job.real_bpm, line_breaks=[])
        current_mode = "whisper"

    real_bpm = job.real_bpm
    job.metadata.bpm = real_bpm
    if alignment_results and current_mode in alignment_results:
        v = alignment_results[current_mode]
        words_for_download = v.words
        phrases = v.phrases
        ultrastar_txt = v.ultrastar_txt
        job.metadata.gap_ms = job.metadata.gap_ms or (v.words[0].start * 1000.0 if v.words else 0.0)
    elif words_for_download is not None:
        phrases, gap_ms = words_to_phrases(words_for_download, real_bpm, line_breaks=line_breaks_for_download)
        job.metadata.gap_ms = gap_ms
        ultrastar_txt = generate_ultrastar_txt(
            metadata=job.metadata,
            phrases=phrases,
            audio_filename=job.result.audio_file,
            vocals_filename=job.result.vocals_file,
            instrumental_filename=job.result.instrumental_file,
            video_filename=job.result.video_file,
        )
    else:
        phrases, job.metadata.gap_ms = [], 0.0
        ultrastar_txt = ""

    job.result.words = words_for_download or []
    job.result.phrases = phrases
    job.result.ultrastar_txt = ultrastar_txt
    job.result.metadata = job.metadata
    job.result.alignment_results = alignment_results
    job.result.current_alignment_mode = current_mode

    safe_name = _sanitize_filename(
        f"{job.metadata.artist} - {job.metadata.title}"
    ) or "song"
    txt_path = os.path.join(job.work_dir, f"{safe_name}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(ultrastar_txt)

    modes_str = ", ".join(sorted(alignment_results.keys()))
    log.info("[%s] Lyrics regenerated: %d alignments, current=%s. Modes: %s", job.id[:8], len(alignment_results), current_mode, modes_str)

    def _serialize_alignment_results(d: dict[str, AlignmentVariant]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for k, v in d.items():
            out[k] = {
                "words": [asdict(w) for w in v.words],
                "phrases": [{"notes": [asdict(n) for n in p.notes], "end_beat": p.end_beat} for p in v.phrases],
                "ultrastar_txt": v.ultrastar_txt,
                "gap_ms": v.gap_ms,
            }
        return out

    return {
        "metadata": asdict(job.result.metadata),
        "words": [asdict(w) for w in job.result.words],
        "phrases": [
            {"notes": [asdict(n) for n in p.notes], "end_beat": p.end_beat}
            for p in job.result.phrases
        ],
        "ultrastar_txt": ultrastar_txt,
        "audio_file": job.result.audio_file,
        "vocals_file": job.result.vocals_file,
        "instrumental_file": job.result.instrumental_file,
        "video_file": job.result.video_file,
        "alignment_results": _serialize_alignment_results(alignment_results),
        "current_alignment_mode": current_mode,
    }


@app.get("/api/jobs/{job_id}/audio/{track}")
async def get_audio(job_id: str, track: str) -> FileResponse:
    """Stream an audio file (original, vocals, instrumental)."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.result:
        raise HTTPException(status_code=400, detail="No result yet")

    filename_map = {
        "original": job.result.audio_file,
        "vocals": job.result.vocals_file,
        "instrumental": job.result.instrumental_file,
    }
    filename = filename_map.get(track)
    if not filename:
        raise HTTPException(status_code=400, detail="Invalid track name")

    filepath = os.path.join(job.work_dir, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(filepath, media_type="audio/mpeg")


@app.get("/api/jobs/{job_id}/download")
async def download_zip(job_id: str, alignment_mode: str | None = None) -> StreamingResponse:
    """Download a ZIP containing TXT + all audio files. Optional alignment_mode uses that variant's TXT."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.result:
        raise HTTPException(status_code=400, detail="No result yet")

    txt_content = job.result.ultrastar_txt
    if alignment_mode:
        ar = getattr(job.result, "alignment_results", {}) or {}
        if alignment_mode in ar:
            txt_content = ar[alignment_mode].ultrastar_txt

    safe_name = _sanitize_filename(
        f"{job.result.metadata.artist} - {job.result.metadata.title}"
    ) or "song"

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        folder = safe_name

        # TXT file
        zf.writestr(f"{folder}/{safe_name}.txt", txt_content)

        # Audio + video files
        for filename in [
            job.result.audio_file,
            job.result.vocals_file,
            job.result.instrumental_file,
            job.result.video_file,
        ]:
            if not filename:
                continue
            filepath = os.path.join(job.work_dir, filename)
            if os.path.exists(filepath):
                zf.write(filepath, f"{folder}/{filename}")

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_name}.zip"'
        },
    )


@app.get("/api/jobs/{job_id}/status")
async def get_status(job_id: str) -> dict[str, Any]:
    """Get current job status."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    result: dict[str, Any] = {
        "status": job.status.value,
        "message": job.progress_message,
    }
    if job.error:
        result["error"] = job.error
    if job.status == JobStatus.AWAITING_LYRICS:
        result["found_lyrics"] = job.found_lyrics
    if job.status == JobStatus.DOWNLOADING:
        result["progress_percent"] = job.progress_percent
    return result


class YouTubeSearchRequest(BaseModel):
    query: str


class YouTubeSearchResult(BaseModel):
    id: str
    title: str
    channel: str
    duration: int | None
    thumbnail: str | None
    url: str


@app.post("/api/youtube/search")
async def search_youtube(req: YouTubeSearchRequest) -> dict[str, list[dict[str, Any]]]:
    """Search YouTube for videos."""
    try:
        import yt_dlp
        
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,  # Get full info for thumbnails
            "default_search": "ytsearch10",
            "ignoreerrors": True,  # Skip unavailable/private videos in search results
        }
        
        results = []
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            search_query = f"ytsearch10:{req.query}"
            info = ydl.extract_info(search_query, download=False)
            
            if "entries" in info:
                for entry in info["entries"]:
                    if entry:
                        # Get best thumbnail (prefer maxresdefault, fallback to high/default)
                        thumbnail = (
                            entry.get("thumbnail") or
                            entry.get("thumbnails", [{}])[-1].get("url") if entry.get("thumbnails") else None
                        )
                        # If no thumbnail, construct YouTube thumbnail URL
                        video_id = entry.get("id", "")
                        if not thumbnail and video_id:
                            thumbnail = f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg"
                        
                        results.append({
                            "id": video_id,
                            "title": entry.get("title", ""),
                            "channel": entry.get("channel", entry.get("uploader", "")),
                            "duration": entry.get("duration"),
                            "thumbnail": thumbnail,
                            "url": f"https://www.youtube.com/watch?v={video_id}",
                        })
        
        return {"results": results}
    except Exception as e:
        logging.error(f"YouTube search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


class YouTubeDownloadRequest(BaseModel):
    url: str
    title: str
    artist: str
    language: str
    genre: str
    year: str


@app.post("/api/youtube/download")
async def download_youtube(req: YouTubeDownloadRequest) -> dict[str, str]:
    """Download a YouTube video and create a processing job."""
    job_id = str(uuid.uuid4())
    
    # Create job immediately with DOWNLOADING status
    job = Job(
        id=job_id,
        original_filename="",
        upload_path="",
        status=JobStatus.DOWNLOADING,
        progress_message="Starting download...",
        metadata=SongMetadata(
            title=req.title,
            artist=req.artist,
            language=req.language,
            genre=req.genre,
            year=req.year,
        ),
    )
    jobs[job_id] = job
    
    # Start download in background
    asyncio.create_task(_download_youtube_video(job_id, req))
    
    return {"job_id": job_id}


async def _download_youtube_video(job_id: str, req: YouTubeDownloadRequest) -> None:
    """Download YouTube video with progress updates."""
    job = jobs.get(job_id)
    if not job:
        return
    
    try:
        import yt_dlp
        
        job_upload_dir = os.path.join(UPLOAD_DIR, job_id)
        os.makedirs(job_upload_dir, exist_ok=True)
        
        def progress_hook(d: dict[str, Any]) -> None:
            """Update job progress from yt-dlp download progress."""
            if d.get("status") == "downloading":
                total = d.get("total_bytes") or d.get("total_bytes_estimate")
                downloaded = d.get("downloaded_bytes", 0)
                if total and total > 0:
                    percent = (downloaded / total) * 100
                    job.progress_percent = percent
                    speed = d.get("speed", 0)
                    speed_mb = speed / (1024 * 1024) if speed else 0
                    job.progress_message = f"Downloading... {percent:.1f}% ({speed_mb:.1f} MB/s)"
                else:
                    # Estimate progress if total unknown
                    downloaded_mb = downloaded / (1024 * 1024)
                    job.progress_message = f"Downloading... {downloaded_mb:.1f} MB"
                    # Set a small progress to show activity
                    if downloaded > 0:
                        job.progress_percent = min(90.0, downloaded_mb / 10.0)  # Rough estimate
                _notify(job)
            elif d.get("status") == "finished":
                job.progress_percent = 100.0
                job.progress_message = "Download complete, processing..."
                _notify(job)
        
        # Download video - prioritize HD (720p, 1080p, etc.)
        ydl_opts = {
            "format": "bestvideo[height>=720][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height>=480][ext=mp4]+bestaudio[ext=m4a]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "outtmpl": os.path.join(job_upload_dir, "%(title)s.%(ext)s"),
            "quiet": False,
            "no_warnings": False,
            "progress_hooks": [progress_hook],
        }
        
        loop = asyncio.get_event_loop()
        upload_path = None
        
        def download_sync():
            nonlocal upload_path
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(req.url, download=True)
                # yt-dlp saves the file, find it
                video_title = info.get("title", "video")
                # Sanitize filename
                safe_title = "".join(c for c in video_title if c.isalnum() or c in (" ", "-", "_")).strip()[:100]
                # Check for downloaded file with various extensions
                for ext in [".mp4", ".webm", ".mkv", ".m4a"]:
                    candidate = os.path.join(job_upload_dir, f"{safe_title}{ext}")
                    if os.path.exists(candidate):
                        upload_path = candidate
                        break
                # If not found, search for any file in the directory
                if not upload_path:
                    files = [f for f in os.listdir(job_upload_dir) if os.path.isfile(os.path.join(job_upload_dir, f))]
                    if files:
                        upload_path = os.path.join(job_upload_dir, files[0])
                return info
        
        info = await loop.run_in_executor(None, download_sync)
        
        if not upload_path or not os.path.exists(upload_path):
            job.status = JobStatus.ERROR
            job.error = "Downloaded file not found"
            job.progress_message = "Download failed"
            _notify(job)
            return
        
        # Extract title/artist from video info if not provided
        video_title = req.title or info.get("title", "")
        video_artist = req.artist or info.get("uploader", info.get("channel", ""))
        
        # Update job with file info and set to PENDING for processing
        job.original_filename = os.path.basename(upload_path)
        job.upload_path = upload_path
        job.status = JobStatus.PENDING
        job.progress_message = "Download complete"
        job.metadata.title = video_title or job.metadata.title
        job.metadata.artist = video_artist or job.metadata.artist
        _notify(job)
        
    except Exception as e:
        logging.error(f"YouTube download failed: {e}")
        job.status = JobStatus.ERROR
        job.error = str(e)
        job.progress_message = f"Download failed: {str(e)}"
        _notify(job)
