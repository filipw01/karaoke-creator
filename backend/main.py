"""Karaoke Creator backend - FastAPI application."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import mimetypes
import os
import shutil
import tempfile
import threading
import time
import uuid
import wave
from dataclasses import asdict, replace
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

from models import (
    AlignmentEvidence,
    AlignmentRun,
    AlignmentRunStatus,
    AlignmentVariant,
    Job,
    JobResult,
    JobStatus,
    LyricRevision,
    LyricSource,
    Phrase,
    SongMetadata,
    UltraStarNote,
    Word,
    WordAlignment,
)
from processing.audio import playback_audio_path
from processing.desktop_lock import processing_slot
from processing.external_chart import parse_chart, read_chart_text
from processing.rust_utility import RustUtilityUnavailable, call as rust_utility, call_with_progress

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

app = FastAPI(title="Karaoke Creator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.environ.get("KARAOKE_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,tauri://localhost,https://tauri.localhost").split(",") if origin.strip()],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Karaoke-Owner", "X-Alignment-Worker-Token"],
)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Worktrees can share a read-only environment. Keep librosa's Numba cache
# beside the backend outputs unless the caller has configured another location.
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(OUTPUT_DIR, "numba-cache"))

# The JSON store keeps local deployments recoverable without introducing a
# database dependency. Deployments with multiple API processes should point
# this at shared durable storage or replace it with their database adapter.
STATE_FILE = os.path.join(OUTPUT_DIR, "jobs.json")
WORKING_DIRECTORY_FILE = os.path.join(OUTPUT_DIR, "working-directory.json")
MAX_UPLOAD_BYTES = int(os.environ.get("KARAOKE_MAX_UPLOAD_BYTES", str(2 * 1024 * 1024 * 1024)))
_state_lock = threading.RLock()
_working_directory_lock = threading.RLock()

# In-memory cache backed by STATE_FILE.
jobs: dict[str, Job] = {}
# SSE event queues per job
job_queues: dict[str, list[asyncio.Queue[dict[str, Any]]]] = {}
processing_semaphore = asyncio.Semaphore(1)
external_song_separation_states: dict[str, dict[str, str]] = {}
external_song_separation_tasks: dict[str, asyncio.Task[None]] = {}
_external_song_separation_lock = threading.Lock()


def _owner_from_authorization(value: str | None) -> str:
    if not value:
        return "local"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


def _request_owner(request: Request) -> str:
    return getattr(request.state, "owner_id", _owner_from_authorization(request.headers.get("Authorization")))


def _job_id_from_path(path: str) -> str | None:
    parts = [part for part in path.split("/") if part]
    for marker in ("jobs", "process"):
        if marker in parts:
            index = parts.index(marker)
            if index + 1 < len(parts):
                return parts[index + 1]
    return None


@app.middleware("http")
async def protect_api_routes(request: Request, call_next: Any) -> Any:
    """Require configured API credentials and prevent cross-owner job access."""

    if request.method == "OPTIONS":
        return await call_next(request)
    if request.url.path.startswith("/api/"):
        configured_token = os.environ.get("KARAOKE_API_TOKEN", "").strip()
        authorization = request.headers.get("Authorization", "")
        supplied_token = authorization.removeprefix("Bearer ").strip() or request.query_params.get("access_token", "")
        if configured_token and supplied_token != configured_token:
            return JSONResponse(status_code=401, content={"detail": "Authentication required"})
        if not configured_token and request.client and request.client.host not in {"127.0.0.1", "::1", "localhost", "testclient"}:
            return JSONResponse(status_code=401, content={"detail": "Configure KARAOKE_API_TOKEN for non-local access"})
        owner_id = _owner_from_authorization(supplied_token) if supplied_token else request.headers.get("X-Karaoke-Owner", "local")
        request.state.owner_id = owner_id
        job_id = _job_id_from_path(request.url.path)
        if job_id:
            job = jobs.get(job_id)
            if job is not None and job.owner_id != owner_id:
                return JSONResponse(status_code=404, content={"detail": "Job not found"})
    return await call_next(request)


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


def _safe_child_path(root: str, filename: str) -> str:
    if not filename or Path(filename).name != filename or "\x00" in filename:
        raise HTTPException(status_code=422, detail="Invalid stored filename")
    root_path = os.path.realpath(root)
    candidate = os.path.realpath(os.path.join(root_path, filename))
    if candidate != root_path and not candidate.startswith(root_path + os.sep):
        raise HTTPException(status_code=422, detail="Stored file escapes the job directory")
    return candidate


def _chart_payload(result: JobResult, *, content: str | None = None) -> dict[str, Any]:
    return {
        "metadata": asdict(result.metadata),
        "ultrastar_txt": result.ultrastar_txt if content is None else content,
        "audio_file": result.audio_file,
        "vocals_file": result.vocals_file,
        "instrumental_file": result.instrumental_file,
        "video_file": result.video_file,
    }


def _publish_song_to_library(job: Job) -> None:
    """Publish a complete UltraStar song folder to the configured library."""
    directory = _working_directory()
    if not job.result or not directory:
        return
    root = Path(directory).expanduser().resolve()
    job.library_path = rust_utility(
        "publish", job_id=job.id, work=job.work_dir, library=str(root),
        existing=job.library_path or None, chart=_chart_payload(job.result),
    )
    job.library_root = str(root)


def _step(job: Job, status: JobStatus, message: str) -> float:
    """Update job status and notify listeners."""
    job.status = status
    job.progress_message = message
    _persist_jobs()
    _notify(job)
    logging.getLogger("pipeline").info("[%s] %s", job.id[:8], message)
    return time.time()


def _enum_value(value: object) -> str:
    return str(getattr(value, "value", value))


def _worker_audio_reference(path: str) -> str:
    """Map an API-local file into the worker's configured shared volume."""

    if not path:
        return ""
    worker_root = os.environ.get("ALIGNMENT_WORKER_AUDIO_ROOT", "").strip()
    if not worker_root:
        raise RuntimeError(
            "ALIGNMENT_WORKER_AUDIO_ROOT must map API audio files into the remote worker"
        )
    api_root = os.path.realpath(
        os.environ.get("ALIGNMENT_API_AUDIO_ROOT", os.path.dirname(__file__))
    )
    source = os.path.realpath(path)
    try:
        relative = os.path.relpath(source, api_root)
    except ValueError as exc:
        raise RuntimeError("Audio path cannot be mapped to the worker volume") from exc
    if relative == os.pardir or relative.startswith(os.pardir + os.sep):
        raise RuntimeError("Audio path is outside the configured API audio root")
    return os.path.join(os.path.realpath(worker_root), relative)


def _create_lyric_revision(
    job: Job,
    raw_text: str,
    *,
    source: LyricSource | str,
    parent_revision_id: str | None = None,
    revision_id: str | None = None,
    language: str | None = None,
) -> LyricRevision:
    """Create an immutable revision and its stable source-order tokens."""

    from processing.alignment import normalize_language, tokenize_lyrics

    language = normalize_language(language or job.metadata.language)
    revision = LyricRevision(
        id=revision_id or str(uuid.uuid4()),
        song_id=job.id,
        language=language,
        source=source,
        raw_text=raw_text,
        parent_revision_id=parent_revision_id,
    )
    revision.tokens = tokenize_lyrics(raw_text, revision.id, language)
    job.lyric_revisions.append(revision)
    job.selected_lyric_revision_id = revision.id
    return revision


def _selected_revision(job: Job) -> LyricRevision | None:
    if not job.selected_lyric_revision_id:
        return None
    return next(
        (revision for revision in job.lyric_revisions if revision.id == job.selected_lyric_revision_id),
        None,
    )


def _serialize_alignment_run(run: AlignmentRun | None) -> dict[str, Any] | None:
    if run is None:
        return None
    return {
        "id": run.id,
        "lyric_revision_id": run.lyric_revision_id,
        "status": _enum_value(run.status),
        "selected_strategy": run.selected_strategy,
        "diagnostics": run.diagnostics,
        "created_at": run.created_at,
    }


def _serialize_revision(revision: LyricRevision | None) -> dict[str, Any] | None:
    if revision is None:
        return None
    return {
        "id": revision.id,
        "song_id": revision.song_id,
        "language": _enum_value(revision.language),
        "source": _enum_value(revision.source),
        "raw_text": revision.raw_text,
        "created_at": revision.created_at,
        "parent_revision_id": revision.parent_revision_id,
        "tokens": [
            {
                "id": token.id,
                "lyric_revision_id": token.lyric_revision_id,
                "display_text": token.display_text,
                "normalized_text": token.normalized_text,
                "line_index": token.line_index,
                "token_index": token.token_index,
                "source_index": token.source_index,
            }
            for token in revision.tokens
        ],
    }


def _serialize_word_alignment(alignment: Any) -> dict[str, Any]:
    evidence = alignment.evidence
    return {
        "token_id": alignment.token_id,
        "alignment_run_id": alignment.alignment_run_id,
        "start": alignment.start,
        "end": alignment.end,
        "status": _enum_value(alignment.status),
        "confidence": alignment.confidence,
        "evidence": {
            "selected_model": evidence.selected_model,
            "selected_audio": evidence.selected_audio,
            "reasons": list(evidence.reasons),
            "candidate_scores": list(evidence.candidate_scores),
        },
        "start_locked": alignment.start_locked,
        "end_locked": alignment.end_locked,
    }


def _serialize_alignment_variant(variant: AlignmentVariant) -> dict[str, Any]:
    return {
        "words": [asdict(word) for word in variant.words],
        "pitch_words": [asdict(word) for word in variant.pitch_words],
        "phrases": [
            {"notes": [asdict(note) for note in phrase.notes], "end_beat": phrase.end_beat}
            for phrase in variant.phrases
        ],
        "ultrastar_txt": variant.ultrastar_txt,
        "gap_ms": variant.gap_ms,
        "word_alignments": [_serialize_word_alignment(item) for item in variant.word_alignments],
        "alignment_run": _serialize_alignment_run(variant.alignment_run),
    }


def _serialize_result(job: Job) -> dict[str, Any]:
    if not job.result:
        raise HTTPException(status_code=400, detail="No result yet")
    result = job.result
    completed_alignment = (
        result.alignment_run is not None
        and _enum_value(result.alignment_run.status) == AlignmentRunStatus.COMPLETED.value
    )
    current_revision_is_exported = (
        result.lyric_revision is not None
        and result.lyric_revision.id == job.selected_lyric_revision_id
    )
    return {
        "metadata": asdict(result.metadata),
        "words": [asdict(word) for word in result.words],
        "pitch_words": [asdict(word) for word in result.pitch_words],
        "phrases": [
            {"notes": [asdict(note) for note in phrase.notes], "end_beat": phrase.end_beat}
            for phrase in result.phrases
        ],
        "ultrastar_txt": result.ultrastar_txt,
        "audio_file": result.audio_file,
        "vocals_file": result.vocals_file,
        "instrumental_file": result.instrumental_file,
        "video_file": result.video_file,
        "alignment_results": {
            key: _serialize_alignment_variant(value)
            for key, value in (result.alignment_results or {}).items()
        },
        "current_alignment_mode": result.current_alignment_mode,
        "lyric_revision": _serialize_revision(result.lyric_revision),
        "lyric_revisions": [_serialize_revision(revision) for revision in job.lyric_revisions],
        "selected_lyric_revision_id": job.selected_lyric_revision_id,
        "word_alignments": [_serialize_word_alignment(item) for item in result.word_alignments],
        "alignment_run": _serialize_alignment_run(result.alignment_run),
        "alignment_comparison": result.alignment_comparison,
        "note_overrides": result.note_overrides,
        "audio_duration": result.audio_duration,
        "export_ready": bool(
            job.status != JobStatus.ALIGNING_WORDS
            and completed_alignment
            and current_revision_is_exported
        ),
        "pending_revision_id": (
            job.selected_lyric_revision_id
            if result.lyric_revision is None or result.lyric_revision.id != job.selected_lyric_revision_id
            else None
        ),
    }


def _json_default(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _revision_from_dict(data: dict[str, Any]) -> LyricRevision:
    from models import LyricToken

    revision = LyricRevision(
        id=str(data["id"]),
        song_id=str(data.get("song_id", "")),
        language=data.get("language", "en"),
        source=data.get("source", LyricSource.MANUAL.value),
        raw_text=str(data.get("raw_text", "")),
        created_at=str(data.get("created_at", "")),
        parent_revision_id=data.get("parent_revision_id"),
        tokens=[LyricToken(**token) for token in data.get("tokens", [])],
    )
    return revision


def _run_from_dict(data: dict[str, Any] | None) -> AlignmentRun | None:
    if not data:
        return None
    return AlignmentRun(
        id=str(data["id"]),
        lyric_revision_id=str(data.get("lyric_revision_id", "")),
        status=data.get("status", AlignmentRunStatus.FAILED.value),
        selected_strategy=str(data.get("selected_strategy", "")),
        diagnostics=data.get("diagnostics", {}),
        created_at=str(data.get("created_at", "")),
    )


def _alignment_from_dict(data: dict[str, Any]) -> Any:
    from models import AlignmentEvidence, WordAlignment

    evidence = data.get("evidence", {}) or {}
    return WordAlignment(
        token_id=str(data["token_id"]),
        alignment_run_id=str(data.get("alignment_run_id", "")),
        start=data.get("start"),
        end=data.get("end"),
        status=data.get("status", "unmatched"),
        confidence=float(data.get("confidence", 0.0)),
        evidence=AlignmentEvidence(
            selected_model=str(evidence.get("selected_model", "")),
            selected_audio=str(evidence.get("selected_audio", "")),
            reasons=list(evidence.get("reasons", [])),
            candidate_scores=list(evidence.get("candidate_scores", [])),
        ),
        start_locked=bool(data.get("start_locked", False)),
        end_locked=bool(data.get("end_locked", False)),
    )


def _variant_from_dict(data: dict[str, Any]) -> AlignmentVariant:
    from models import Phrase, UltraStarNote, Word

    return AlignmentVariant(
        words=[Word(**word) for word in data.get("words", [])],
        pitch_words=[Word(**word) for word in data.get("pitch_words", [])],
        phrases=[
            Phrase(
                notes=[UltraStarNote(**note) for note in phrase.get("notes", [])],
                end_beat=int(phrase.get("end_beat", 0)),
            )
            for phrase in data.get("phrases", [])
        ],
        ultrastar_txt=str(data.get("ultrastar_txt", "")),
        gap_ms=float(data.get("gap_ms", 0.0)),
        word_alignments=[_alignment_from_dict(item) for item in data.get("word_alignments", [])],
        alignment_run=_run_from_dict(data.get("alignment_run")),
    )


def _result_from_dict(data: dict[str, Any] | None) -> JobResult | None:
    if not data:
        return None
    from models import Phrase, SongMetadata, UltraStarNote, Word

    return JobResult(
        metadata=SongMetadata(**(data.get("metadata", {}) or {})),
        words=[Word(**word) for word in data.get("words", [])],
        pitch_words=[Word(**word) for word in data.get("pitch_words", [])],
        phrases=[
            Phrase(
                notes=[UltraStarNote(**note) for note in phrase.get("notes", [])],
                end_beat=int(phrase.get("end_beat", 0)),
            )
            for phrase in data.get("phrases", [])
        ],
        ultrastar_txt=str(data.get("ultrastar_txt", "")),
        audio_file=str(data.get("audio_file", "")),
        vocals_file=str(data.get("vocals_file", "")),
        instrumental_file=str(data.get("instrumental_file", "")),
        video_file=str(data.get("video_file", "")),
        alignment_results={
            key: _variant_from_dict(value)
            for key, value in (data.get("alignment_results", {}) or {}).items()
        },
        current_alignment_mode=str(data.get("current_alignment_mode", "canonical")),
        lyric_revision=_revision_from_dict(data["lyric_revision"]) if data.get("lyric_revision") else None,
        word_alignments=[_alignment_from_dict(item) for item in data.get("word_alignments", [])],
        alignment_run=_run_from_dict(data.get("alignment_run")),
        alignment_comparison=data.get("alignment_comparison", {}) or {},
        note_overrides=data.get("note_overrides", {}) or {},
        audio_duration=float(data.get("audio_duration", 0.0)),
    )


def _job_from_dict(data: dict[str, Any]) -> Job:
    metadata = data.get("metadata", {}) or {}
    job = Job(
        id=str(data["id"]),
        owner_id=str(data.get("owner_id", "local")),
        status=JobStatus(data.get("status", JobStatus.PENDING.value)),
        progress_message=str(data.get("progress_message", "")),
        progress_percent=float(data.get("progress_percent", 0.0)),
        queued_at=str(data.get("queued_at", "")),
        library_path=str(data.get("library_path", "")),
        library_root=str(data.get("library_root", "")),
        error=str(data.get("error", "")),
        failed_phase=str(data.get("failed_phase", "")),
        original_filename=str(data.get("original_filename", "")),
        metadata=SongMetadata(**metadata),
        result=_result_from_dict(data.get("result")),
        upload_path=str(data.get("upload_path", "")),
        source_url=str(data.get("source_url", "")),
        work_dir=str(data.get("work_dir", "")),
        audio_path=str(data.get("audio_path", "")),
        vocals_path=str(data.get("vocals_path", "")),
        instrumental_path=str(data.get("instrumental_path", "")),
        has_video=bool(data.get("has_video", False)),
        real_bpm=float(data.get("real_bpm", 0.0)),
        found_lyrics=str(data.get("found_lyrics", "")),
        confirmed_lyrics=str(data.get("confirmed_lyrics", "")),
        lyric_revisions=[_revision_from_dict(item) for item in data.get("lyric_revisions", [])],
        selected_lyric_revision_id=str(data.get("selected_lyric_revision_id", "")),
        alignment_runs=[run for item in data.get("alignment_runs", []) if (run := _run_from_dict(item)) is not None],
    )
    return job


def _persist_jobs() -> None:
    """Atomically persist the job cache so completed work survives restart."""

    state_path = Path(STATE_FILE)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = state_path.with_suffix(".json.tmp")
    with _state_lock:
        payload = {job_id: asdict(job) for job_id, job in jobs.items()}
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2, default=_json_default)
        os.replace(temporary_path, state_path)


def _load_persisted_jobs() -> None:
    state_path = Path(STATE_FILE)
    if not state_path.exists():
        return
    try:
        with state_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        for job_id, data in payload.items():
            job = _job_from_dict(data)
            if job.id == job_id:
                if job.status == JobStatus.DOWNLOADING:
                    job.queued_at = ""
                    if job.upload_path and os.path.isfile(job.upload_path):
                        job.status = JobStatus.PENDING
                        job.error = ""
                        job.failed_phase = ""
                        job.progress_message = "Download complete; ready to process."
                    else:
                        job.status = JobStatus.ERROR
                        job.failed_phase = "download"
                        job.error = "The YouTube download was interrupted before it completed."
                        job.progress_message = "Download interrupted; retry is available."
                    job.progress_percent = 0.0
                elif job.status == JobStatus.AWAITING_LYRICS:
                    job.queued_at = ""
                    job.error = ""
                    job.progress_message = "Waiting for lyrics approval."
                elif job.failed_phase == "publish" and job.status == JobStatus.GENERATING and job.result:
                    job.queued_at = ""
                    job.status = JobStatus.ERROR
                    job.error = "Library publication was interrupted."
                    job.progress_message = "Retry publishing to reuse the finished chart and media."
                elif job.status not in {JobStatus.DONE, JobStatus.ERROR, JobStatus.PENDING}:
                    job.queued_at = ""
                    job.status = JobStatus.PENDING
                    job.progress_percent = 0.0
                    job.error = ""
                    job.progress_message = "Processing paused after API restart; ready to resume."
                    for run in job.alignment_runs:
                        if _enum_value(run.status) == AlignmentRunStatus.RUNNING.value:
                            run.status = AlignmentRunStatus.FAILED
                            run.diagnostics = {"error": "API restarted during alignment"}
                if job.result and job.result.alignment_comparison.get("status") == "running":
                    job.result.alignment_comparison = {
                        "status": "failed",
                        "sofa_available": True,
                        "legacy_available": False,
                        "error": "Background comparison was interrupted by an API restart",
                    }
                jobs[job_id] = job
        if jobs:
            _persist_jobs()
    except Exception:
        logging.getLogger("persistence").exception("Unable to restore persisted karaoke jobs")


_load_persisted_jobs()


def _working_directory() -> str | None:
    try:
        with open(WORKING_DIRECTORY_FILE, encoding="utf-8") as file:
            value = json.load(file).get("directory")
    except (OSError, ValueError, AttributeError):
        return None
    return value if isinstance(value, str) and value else None


def _save_working_directory(directory: str | None) -> None:
    state_path = Path(WORKING_DIRECTORY_FILE)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = state_path.with_suffix(".json.tmp")
    with _working_directory_lock:
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump({"directory": directory}, file, ensure_ascii=False)
        os.replace(temporary_path, state_path)


def _external_song_path(relative_path: str) -> Path:
    directory = _working_directory()
    if not directory:
        raise HTTPException(status_code=409, detail="Working directory is not configured")
    root = Path(directory).resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Song not found") from exc
    if not candidate.is_file() or candidate.suffix.lower() != ".txt":
        raise HTTPException(status_code=404, detail="Song not found")
    return candidate


def _external_song_audio_path(chart: Path) -> Path | None:
    directory = _working_directory()
    if not directory:
        return None
    path = rust_utility("external_audio", root=str(Path(directory).resolve()), path=str(chart))
    return Path(path) if path else None


def _external_song_stem_paths(chart: Path) -> tuple[str, Path, Path, Path]:
    key = hashlib.sha256(str(chart.resolve()).encode("utf-8")).hexdigest()
    directory = Path(OUTPUT_DIR) / "external-stems" / key
    return key, directory, directory / "vocals.wav", directory / "instrumental.wav"


def _external_song_stems_available(chart: Path) -> bool:
    _key, _directory, vocals, instrumental = _external_song_stem_paths(chart)
    return vocals.is_file() and instrumental.is_file()


def _external_song_separation_status(chart: Path) -> dict[str, str]:
    key, _directory, _vocals, _instrumental = _external_song_stem_paths(chart)
    if _external_song_stems_available(chart):
        return {"status": "done"}
    with _external_song_separation_lock:
        return external_song_separation_states.get(key, {"status": "idle"}).copy()


async def _run_external_song_separation(
    key: str,
    audio_path: str,
    output_directory: Path,
) -> None:
    try:
        async with processing_slot(OUTPUT_DIR, processing_semaphore):
            from processing.vocals import _demucs_available, separate_vocals

            if not await asyncio.to_thread(_demucs_available):
                raise RuntimeError("Demucs is not installed; vocals cannot be separated.")

            output_directory.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                dir=output_directory.parent,
                prefix=f".{key}.",
            ) as temporary_directory:
                vocals, instrumental = await asyncio.to_thread(
                    separate_vocals,
                    audio_path,
                    temporary_directory,
                )
                if not Path(vocals).is_file() or not Path(instrumental).is_file():
                    raise RuntimeError("Vocal separation did not produce both audio stems.")
                if output_directory.exists():
                    shutil.rmtree(output_directory)
                output_directory.mkdir(parents=True)
                os.replace(vocals, output_directory / "vocals.wav")
                os.replace(instrumental, output_directory / "instrumental.wav")
        state = {"status": "done"}
    except Exception as exc:
        logging.getLogger("pipeline").exception("Existing song vocal separation failed")
        state = {"status": "error", "error": str(exc)}
    finally:
        with _external_song_separation_lock:
            external_song_separation_states[key] = state
            external_song_separation_tasks.pop(key, None)


def _remove_managed_directory(path: str, root: str) -> None:
    if not path or not root:
        return
    root_path = Path(root).resolve()
    target_path = Path(path).expanduser()
    if target_path.is_symlink():
        raise ValueError("Refusing to remove a symlinked song directory")
    candidate = target_path.resolve()
    if candidate == root_path:
        raise ValueError("Refusing to remove a storage root")
    if candidate.parent != root_path:
        raise ValueError("Managed song directory must be directly inside its storage root")
    if candidate.exists():
        if not candidate.is_dir():
            raise ValueError("Managed song storage path is not a directory")
        shutil.rmtree(candidate)


def _capture_note_overrides(phrases: list[Any]) -> dict[str, dict[str, Any]]:
    """Persist user-edited note fields independently from regenerated timing."""

    overrides: dict[str, dict[str, Any]] = {}
    for phrase_index, phrase in enumerate(phrases):
        overrides[f"phrase:{phrase_index}"] = {"end_beat": phrase.end_beat}
        for note_index, note in enumerate(phrase.notes):
            overrides[f"note:{phrase_index}:{note_index}"] = asdict(note)
    return overrides


def _apply_note_overrides(phrases: list[Any], overrides: dict[str, dict[str, Any]]) -> None:
    """Reapply note edits after canonical timing is projected again."""

    for key, values in overrides.items():
        parts = key.split(":")
        if parts[0] == "phrase" and len(parts) == 2:
            phrase_index = int(parts[1])
            if 0 <= phrase_index < len(phrases) and "end_beat" in values:
                phrases[phrase_index].end_beat = int(values["end_beat"])
        elif parts[0] == "note" and len(parts) == 3:
            phrase_index, note_index = int(parts[1]), int(parts[2])
            if not (0 <= phrase_index < len(phrases) and 0 <= note_index < len(phrases[phrase_index].notes)):
                continue
            note = phrases[phrase_index].notes[note_index]
            for field_name in ("note_type", "start_beat", "duration_beats", "pitch", "text"):
                if field_name in values:
                    setattr(note, field_name, values[field_name])


def _apply_canonical_result(
    job: Job,
    revision: LyricRevision,
    run: AlignmentRun,
    alignments: list[Any],
    canonical_words: list[Any],
    pitch_words: list[Any],
    line_breaks: list[int],
) -> None:
    """Project a canonical run into the legacy export result."""

    if not job.result:
        return
    from processing.ultrastar import generate_ultrastar_txt, words_to_phrases

    phrases, gap_ms = words_to_phrases(pitch_words, job.real_bpm, line_breaks=line_breaks)
    _apply_note_overrides(phrases, job.result.note_overrides)
    job.metadata.bpm = job.real_bpm
    job.metadata.gap_ms = gap_ms
    safe_name = _sanitize_filename(f"{job.metadata.artist} - {job.metadata.title}") or "song"
    audio_filename = job.result.audio_file
    vocals_filename = job.result.vocals_file
    instrumental_filename = job.result.instrumental_file
    video_filename = job.result.video_file
    ultrastar_txt = generate_ultrastar_txt(
        metadata=job.metadata,
        phrases=phrases,
        audio_filename=audio_filename,
        vocals_filename=vocals_filename,
        instrumental_filename=instrumental_filename,
        video_filename=video_filename,
    )
    job.result.metadata = job.metadata
    job.result.words = canonical_words
    job.result.pitch_words = pitch_words
    job.result.phrases = phrases
    job.result.ultrastar_txt = ultrastar_txt
    canonical_variant = AlignmentVariant(
        words=canonical_words,
        pitch_words=pitch_words,
        phrases=phrases,
        ultrastar_txt=ultrastar_txt,
        gap_ms=gap_ms,
        word_alignments=alignments,
        alignment_run=run,
    )
    job.result.alignment_results = {"canonical": canonical_variant, "sofa": canonical_variant}
    job.result.current_alignment_mode = "canonical"
    job.result.lyric_revision = revision
    job.result.word_alignments = alignments
    job.result.alignment_run = run
    if job.work_dir:
        txt_path = os.path.join(job.work_dir, f"{safe_name}.txt")
        with open(txt_path, "w", encoding="utf-8") as file:
            file.write(ultrastar_txt)
    _publish_song_to_library(job)


def _refresh_legacy_result_from_canonical(job: Job) -> None:
    """Rebuild export data without allowing it to rewrite canonical spans."""

    if not job.result or not job.result.lyric_revision or not job.result.alignment_run:
        return
    previous_variants = dict(job.result.alignment_results or {})
    previous_mode = job.result.current_alignment_mode
    from processing.alignment import line_breaks_for_projected_words, project_alignments_to_words

    canonical_words = project_alignments_to_words(
        job.result.lyric_revision.tokens,
        job.result.word_alignments,
        existing_words=(),
    )
    pitch_seed = job.result.pitch_words or job.result.words
    pitch_source_alignments = _pitch_source_alignments_from_words(
        job.result.word_alignments,
        pitch_seed,
    )
    pitch_alignments = (
        _pitch_alignments_with_boundary_locks(
            pitch_source_alignments,
            job.result.word_alignments,
        )
        if pitch_source_alignments
        else job.result.word_alignments
    )
    pitch_words = project_alignments_to_words(
        job.result.lyric_revision.tokens,
        pitch_alignments,
        existing_words=pitch_seed,
    )
    _apply_canonical_result(
        job,
        job.result.lyric_revision,
        job.result.alignment_run,
        job.result.word_alignments,
        canonical_words,
        pitch_words,
        line_breaks_for_projected_words(job.result.lyric_revision.tokens, pitch_words),
    )
    # Boundary edits apply to the selected lane only. Keep the other cached
    # lane intact and refresh the agreement score instead of forcing a model
    # rerun or silently discarding the comparison.
    edited_variant = job.result.alignment_results.get("canonical")
    if edited_variant is not None and previous_mode == "legacy" and previous_variants.get("sofa"):
        job.result.alignment_results["legacy"] = edited_variant
        job.result.alignment_results["sofa"] = previous_variants["sofa"]
        job.result.current_alignment_mode = "legacy"
        job.result.alignment_comparison = _compare_alignment_variants(
            job,
            previous_variants["sofa"],
            edited_variant,
        )
    elif edited_variant is not None and previous_mode == "sofa" and previous_variants.get("legacy"):
        job.result.alignment_results["sofa"] = edited_variant
        job.result.alignment_results["canonical"] = edited_variant
        job.result.alignment_results["legacy"] = previous_variants["legacy"]
        job.result.current_alignment_mode = "sofa"
        job.result.alignment_comparison = _compare_alignment_variants(
            job,
            edited_variant,
            previous_variants["legacy"],
        )


def _alignment_variant_for_mode(result: JobResult, mode: str) -> AlignmentVariant | None:
    """Resolve the public mode names while retaining the old canonical alias."""

    variants = result.alignment_results or {}
    if mode == "sofa" and "sofa" not in variants:
        return variants.get("canonical")
    if mode == "canonical" and "canonical" not in variants:
        return variants.get("sofa")
    return variants.get(mode)


def _apply_alignment_variant(job: Job, mode: str) -> str:
    """Project a cached alignment variant into the fields consumed by the editor/exporter."""

    if not job.result:
        raise ValueError("No alignment result")
    result = job.result
    variant = _alignment_variant_for_mode(result, mode)
    if variant is None:
        raise ValueError(f"Alignment mode is not available: {mode}")
    resolved_mode = mode
    if mode == "canonical" and "canonical" not in (result.alignment_results or {}):
        resolved_mode = "sofa"
    result.words = list(variant.words)
    result.pitch_words = list(variant.pitch_words)
    result.phrases = list(variant.phrases)
    result.ultrastar_txt = variant.ultrastar_txt
    result.word_alignments = list(variant.word_alignments)
    result.alignment_run = variant.alignment_run
    result.current_alignment_mode = resolved_mode
    if variant.gap_ms:
        result.metadata.gap_ms = variant.gap_ms
        job.metadata.gap_ms = variant.gap_ms
    return resolved_mode


def _pitch_by_token(words: list[Any]) -> dict[str, int]:
    """Pick the longest voiced MIDI segment for each lyric token."""

    grouped: dict[str, list[Any]] = {}
    for index, word in enumerate(words):
        token_id = getattr(word, "token_id", None) or f"index:{index}"
        grouped.setdefault(str(token_id), []).append(word)
    result: dict[str, int] = {}
    for token_id, segments in grouped.items():
        voiced = [word for word in segments if int(getattr(word, "midi_note", 0) or 0) > 0]
        candidates = voiced or segments
        if not candidates:
            continue
        selected = max(
            candidates,
            key=lambda word: max(0.0, float(getattr(word, "end", 0.0) or 0.0) - float(getattr(word, "start", 0.0) or 0.0)),
        )
        result[token_id] = int(getattr(selected, "midi_note", 0) or 0)
    return result


def _compare_alignment_variants(job: Job, sofa: AlignmentVariant, legacy: AlignmentVariant) -> dict[str, Any]:
    """Calculate an explainable agreement score between two cached alignments."""

    sofa_words = {str(word.token_id): word for word in sofa.words if getattr(word, "token_id", None)}
    legacy_words = {str(word.token_id): word for word in legacy.words if getattr(word, "token_id", None)}
    token_ids = [token_id for token_id in sofa_words if token_id in legacy_words]
    if not token_ids:
        count = min(len(sofa.words), len(legacy.words))
        pairs = list(zip(sofa.words[:count], legacy.words[:count]))
    else:
        pairs = [(sofa_words[token_id], legacy_words[token_id]) for token_id in token_ids]

    bpm = max(float(job.metadata.bpm or job.real_bpm or 0.0), 1.0)
    one_chart_beat_s = 60.0 / (bpm * 16.0)
    timing_scores: list[float] = []
    timing_errors: list[float] = []
    valid_pairs: list[tuple[Any, Any]] = []
    for left, right in pairs:
        if left.start is None or left.end is None or right.start is None or right.end is None:
            continue
        try:
            error = max(abs(float(left.start) - float(right.start)), abs(float(left.end) - float(right.end)))
        except (TypeError, ValueError):
            continue
        timing_errors.append(error)
        timing_scores.append(max(0.0, 1.0 - error / max(one_chart_beat_s, 1e-6)))
        valid_pairs.append((left, right))

    sofa_pitch = _pitch_by_token(sofa.pitch_words or sofa.words)
    legacy_pitch = _pitch_by_token(legacy.pitch_words or legacy.words)
    pitch_matches = 0
    pitch_comparisons = 0
    for left, right in valid_pairs:
        token_id = str(getattr(left, "token_id", ""))
        right_token_id = str(getattr(right, "token_id", ""))
        left_pitch = sofa_pitch.get(token_id, int(getattr(left, "midi_note", 0) or 0))
        right_pitch = legacy_pitch.get(right_token_id, int(getattr(right, "midi_note", 0) or 0))
        if left_pitch > 0 and right_pitch > 0:
            pitch_comparisons += 1
            pitch_matches += int(left_pitch == right_pitch)

    timing_agreement = sum(timing_scores) / len(timing_scores) if timing_scores else 0.0
    pitch_agreement = pitch_matches / pitch_comparisons if pitch_comparisons else 0.0
    confidence = 0.65 * timing_agreement + 0.35 * pitch_agreement
    return {
        "status": "completed",
        "sofa_available": True,
        "legacy_available": True,
        "compared_word_count": len(pairs),
        "timing_agreement": round(timing_agreement, 4),
        "pitch_agreement": round(pitch_agreement, 4),
        "pitch_exact_fraction": round(pitch_agreement, 4),
        "timing_mean_abs_error_ms": round((sum(timing_errors) / len(timing_errors) * 1000.0) if timing_errors else 0.0, 2),
        "one_chart_beat_ms": round(one_chart_beat_s * 1000.0, 2),
        "confidence": round(confidence, 4),
        "preferred_mode": "sofa" if confidence >= 0.5 else "legacy",
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _build_alignment_variant(
    job: Job,
    canonical_words: list[Any],
    pitch_words: list[Any],
    line_breaks: list[int],
    *,
    word_alignments: list[WordAlignment] | None = None,
    alignment_run: AlignmentRun | None = None,
    filenames: tuple[str, str, str, str] | None = None,
) -> AlignmentVariant:
    """Build an exportable cached variant without changing the active result."""

    from processing.ultrastar import generate_ultrastar_txt, words_to_phrases

    audio_filename, vocals_filename, instrumental_filename, video_filename = filenames or (
        "audio.wav",
        "vocals.wav",
        "instrumental.wav",
        "",
    )
    phrases, gap_ms = words_to_phrases(pitch_words, job.real_bpm, line_breaks=line_breaks)
    _apply_note_overrides(phrases, job.result.note_overrides if job.result else {})
    metadata = replace(job.metadata, bpm=job.real_bpm, gap_ms=gap_ms)
    ultrastar_txt = generate_ultrastar_txt(
        metadata=metadata,
        phrases=phrases,
        audio_filename=audio_filename,
        vocals_filename=vocals_filename,
        instrumental_filename=instrumental_filename,
        video_filename=video_filename,
    )
    return AlignmentVariant(
        words=list(canonical_words),
        pitch_words=list(pitch_words),
        phrases=phrases,
        ultrastar_txt=ultrastar_txt,
        gap_ms=gap_ms,
        word_alignments=list(word_alignments or []),
        alignment_run=alignment_run,
    )


async def _run_legacy_comparison(job: Job, revision: LyricRevision) -> None:
    """Run the archived alignment lane after SOFA, preserving the SOFA result."""

    log = logging.getLogger("pipeline")
    if not job.result:
        return
    result = job.result
    try:
        from processing.alignment import line_breaks_for_projected_words
        from processing.lyrics import LyricsParams
        from processing.pitch import detect_pitch

        loop = asyncio.get_running_loop()
        detection = await loop.run_in_executor(None, detect_pitch, job.vocals_path, job.audio_path)
        times, frequencies, confidences, energies = detection
        run = AlignmentRun(
            id=str(uuid.uuid4()),
            lyric_revision_id=revision.id,
            status=AlignmentRunStatus.RUNNING,
            selected_strategy="legacy-global-match-and-candidate-score",
        )
        job.alignment_runs.append(run)
        params = LyricsParams(
            language=_enum_value(revision.language),
            initial_prompt=revision.raw_text,
            normalize=False,
            audio_source="vocals",
        )
        selected, completed_run, alignments, words, pitch_words, line_breaks = await _run_canonical_alignment(
            job,
            revision.raw_text,
            job.vocals_path,
            _enum_value(revision.language),
            params,
            [],
            times,
            frequencies,
            confidences,
            energies,
            getattr(detection, "candidate_midis", None),
            getattr(detection, "candidate_scores", None),
            getattr(detection, "onset_scores", None),
            audio_duration=result.audio_duration or _audio_duration(job.audio_path),
            alignment_run=run,
            alignment_backend="legacy",
            update_job_status=False,
            use_existing_alignment_locks=False,
        )
        filenames = (result.audio_file, result.vocals_file, result.instrumental_file, result.video_file)
        legacy_variant = _build_alignment_variant(
            job,
            words,
            pitch_words,
            line_breaks,
            word_alignments=alignments,
            alignment_run=completed_run,
            filenames=filenames,
        )
        sofa_variant = _alignment_variant_for_mode(result, "sofa")
        if sofa_variant is None:
            raise RuntimeError("SOFA result is unavailable for comparison")
        # Keep the historical canonical key and add explicit public lanes.
        result.alignment_results["sofa"] = sofa_variant
        result.alignment_results["legacy"] = legacy_variant
        result.alignment_comparison = _compare_alignment_variants(job, sofa_variant, legacy_variant)
        _persist_jobs()
        _notify(job)
        log.info("[%s] Background SOFA/legacy comparison complete: %.1f%% confidence", job.id[:8], result.alignment_comparison["confidence"] * 100)
    except Exception as exc:
        log.exception("[%s] Background legacy comparison failed", job.id[:8])
        result.alignment_comparison = {
            "status": "failed",
            "sofa_available": bool(_alignment_variant_for_mode(result, "sofa")),
            "legacy_available": False,
            "error": str(exc),
        }
        _persist_jobs()
        _notify(job)


def _draft_words_from_audio(audio_path: str, language: str, params: Any) -> list[Any]:
    """Create an ASR draft through the configured self-hosted backend first."""

    from models import Word
    from processing.lyrics import extract_lyrics
    from processing.model_adapters import qwen_asr_enabled, run_serialized_model_call, transcribe_with_qwen

    if qwen_asr_enabled():
        try:
            tokens = run_serialized_model_call(transcribe_with_qwen, audio_path, language)
            return [
                Word(
                    text=token.text,
                    start=token.start,
                    end=token.end,
                    confidence=token.confidence,
                )
                for token in tokens
            ]
        except Exception:
            log = logging.getLogger("pipeline")
            log.exception("Qwen ASR draft generation failed; trying the local compatibility fallback")
    if os.environ.get("KARAOKE_WHISPER_CPP_MODEL"):
        words = rust_utility("whisper_draft", audio=audio_path, language=language)
        return [Word(**word) for word in words]
    return list(run_serialized_model_call(extract_lyrics, audio_path, language, params))


def _audio_duration(audio_path: str) -> float:
    try:
        import soundfile as sf

        duration = float(sf.info(audio_path).duration)
        return duration if duration > 0 else 0.0
    except Exception:
        return 0.0


def _draft_text_from_remote_worker(job: Job, alignment_run_id: str, language: str) -> str:
    """Ask the configured worker for an ASR draft without running models in the API."""

    from processing.alignment import normalize_language
    from processing.alignment_worker import build_worker_request, dispatch_worker_alignment

    worker_url = os.environ.get("ALIGNMENT_WORKER_URL", "").strip()
    worker_token = os.environ.get("ALIGNMENT_WORKER_TOKEN", "").strip()
    if not worker_url or not worker_token:
        raise RuntimeError("A worker URL and token are required for remote lyric drafting")
    draft_revision = LyricRevision(
        id=f"{job.id}:draft-request:{uuid.uuid4()}",
        song_id=job.id,
        language=normalize_language(language),
        source=LyricSource.ASR_DRAFT,
        raw_text="",
    )
    request = build_worker_request(
        job_id=job.id,
        revision=draft_revision,
        alignment_run_id=alignment_run_id,
        original_audio_ref=_worker_audio_reference(job.audio_path),
        vocal_stem_audio_ref=_worker_audio_reference(job.vocals_path),
    )
    response = dispatch_worker_alignment(
        request,
        worker_url,
        worker_token=worker_token,
    )
    draft_text = (response.draft_text or "").strip()
    if response.status != "completed" or not draft_text:
        diagnostics = response.diagnostics or {}
        raise RuntimeError(
            f"Remote worker could not create an ASR lyric draft: "
            f"{diagnostics.get('reason', 'no draft text returned')}"
        )
    return draft_text


def _pitch_alignments_with_boundary_locks(
    source_alignments: list[Any],
    merged_alignments: list[Any],
) -> list[Any]:
    """Keep canonical pitch spans while applying every effective lock boundary.

    ``merge_alignment_locks`` can retain an unlocked companion boundary when a
    one-sided manual lock would otherwise make the span invalid. That retained
    value is evidence in the merged alignment, not an explicit lock, so copying
    only ``*_locked`` fields would silently drop it from pitch assignment.
    """

    final_by_token_id = {alignment.token_id: alignment for alignment in merged_alignments}
    repaired_end_reason = "previous end retained to keep the locked start paired"
    repaired_start_reason = "previous start retained to keep the locked end paired"
    result: list[Any] = []
    for source in source_alignments:
        final = final_by_token_id.get(source.token_id)
        if final is None:
            continue
        reasons = set(getattr(final.evidence, "reasons", ()))
        pitch_start = final.start if final.start_locked or repaired_start_reason in reasons else source.start
        pitch_end = final.end if final.end_locked or repaired_end_reason in reasons else source.end

        def valid_span(start: Any, end: Any) -> bool:
            try:
                return (
                    start is not None
                    and end is not None
                    and math.isfinite(float(start))
                    and math.isfinite(float(end))
                    and float(end) >= float(start)
                )
            except (TypeError, ValueError):
                return False

        # A merge may repair an incomplete source span through a valid final
        # pair. Use that pair as a last-resort safeguard for pitch extraction.
        if not valid_span(pitch_start, pitch_end) and valid_span(final.start, final.end):
            pitch_start, pitch_end = final.start, final.end
        result.append(
            replace(
                source,
                start=pitch_start,
                end=pitch_end,
                status=final.status,
                confidence=final.confidence,
                evidence=final.evidence,
                start_locked=final.start_locked,
                end_locked=final.end_locked,
            )
        )
    return result


def _pitch_source_alignments_from_words(
    alignments: list[Any],
    pitch_words: list[Any],
) -> list[Any]:
    """Recover one source span per token from persisted pitch segments."""

    words_by_token: dict[str, list[Any]] = {}
    for word in pitch_words:
        if word.token_id:
            words_by_token.setdefault(word.token_id, []).append(word)
    result: list[Any] = []
    for alignment in alignments:
        words = words_by_token.get(alignment.token_id, ())
        spans = [
            (float(word.start), float(word.end))
            for word in words
            if word.start is not None
            and word.end is not None
            and math.isfinite(float(word.start))
            and math.isfinite(float(word.end))
            and float(word.end) >= float(word.start)
        ]
        if not spans:
            continue
        result.append(
            replace(
                alignment,
                start=min(start for start, _ in spans),
                end=max(end for _, end in spans),
                start_locked=False,
                end_locked=False,
            )
        )
    return result


async def _run_canonical_alignment(
    job: Job,
    lyrics_text: str,
    transcribe_path: str,
    language: str | None,
    params: Any,
    fallback_words: list[Any],
    pitch_times: Any,
    pitch_frequencies: Any,
    pitch_confidences: Any,
    pitch_energies: Any,
    pitch_candidate_midis: Any = None,
    pitch_candidate_scores: Any = None,
    pitch_onset_scores: Any = None,
    audio_duration: float = 0.0,
    reset_token_ids: set[str] | None = None,
    alignment_run: AlignmentRun | None = None,
    alignment_run_id: str | None = None,
    alignment_backend: str = "sofa",
    update_job_status: bool = True,
    use_existing_alignment_locks: bool = True,
) -> tuple[LyricRevision, AlignmentRun, list[Any], list[Any], list[Any], list[int]]:
    """Run the local adapter behind the stable worker contract."""

    from processing.alignment import (
        activity_spans_from_signal,
        asr_tokens_from_words,
        expand_alignment_sustains,
        line_breaks_for_projected_words,
        map_token_id_aliases,
        merge_alignment_locks,
        project_alignments_to_words,
        repair_melody_handoffs,
        reset_word_boundaries,
    )
    from processing.alignment_worker import (
        alignment_response_to_models,
        build_worker_request,
        collect_model_evidence,
        dispatch_worker_alignment,
        resolve_alignment_backend,
        run_local_alignment,
    )
    from processing.pitch import (
        alignment_activity_threshold,
        assign_pitch_to_words,
        DEFAULT_MIN_VOLUME_RMS,
        voiced_activity_signal,
    )

    loop = asyncio.get_running_loop()
    selected = _selected_revision(job)
    if selected is None or (lyrics_text.strip() and selected.raw_text != lyrics_text.strip()):
        if lyrics_text.strip():
            selected = _create_lyric_revision(
                job,
                lyrics_text.strip(),
                source=LyricSource.MANUAL,
                parent_revision_id=selected.id if selected else None,
                language=language,
            )
        else:
            draft_text = " ".join(word.text for word in fallback_words if getattr(word, "text", "").strip())
            selected = _create_lyric_revision(
                job,
                draft_text,
                source=LyricSource.ASR_DRAFT,
                language=language,
            )

    run = alignment_run or AlignmentRun(
        id=alignment_run_id or str(uuid.uuid4()),
        lyric_revision_id=selected.id,
        status=AlignmentRunStatus.RUNNING,
        selected_strategy="sofa-primary-global-match-and-candidate-score",
    )
    run.lyric_revision_id = selected.id
    if run not in job.alignment_runs:
        job.alignment_runs.append(run)
    if update_job_status:
        _step(job, JobStatus.ALIGNING_WORDS, "Aligning canonical lyric words to both audio sources...")

    try:
        alignment_activity = voiced_activity_signal(
            pitch_frequencies if pitch_frequencies is not None else (),
            pitch_energies if pitch_energies is not None else (),
        )
        activity_threshold = alignment_activity_threshold(alignment_activity)
        activity_spans = activity_spans_from_signal(
            pitch_times if pitch_times is not None else (),
            alignment_activity,
            threshold=activity_threshold,
        )
        request = build_worker_request(
            job_id=job.id,
            revision=selected,
            alignment_run_id=run.id,
            original_audio_ref=job.audio_path,
            vocal_stem_audio_ref=transcribe_path or job.vocals_path,
            activity_spans=activity_spans,
            alignment_backend=alignment_backend,
        )
        # The worker contract normalizes non-English SOFA requests to the
        # archived ASR/CTC backend unless SOFA_ALLOW_NON_ENGLISH is explicit.
        # Persist the resolved lane so the editor and status API reflect what
        # actually ran, rather than the caller's default request.
        resolved_backend = resolve_alignment_backend(
            request.language,
            request.alignment_backend,
        )
        run.selected_strategy = (
            "legacy-global-match-and-candidate-score"
            if resolved_backend == "legacy"
            else "sofa-primary-global-match-and-candidate-score"
        )

        worker_url = os.environ.get("ALIGNMENT_WORKER_URL", "").strip()
        worker_token = os.environ.get("ALIGNMENT_WORKER_TOKEN", "").strip()
        source_errors: dict[str, str] = {}
        response: Any
        if worker_url:
            if not worker_token:
                raise RuntimeError("ALIGNMENT_WORKER_TOKEN is required when ALIGNMENT_WORKER_URL is configured")
            remote_request = request.model_copy(
                update={
                    "original_audio_ref": _worker_audio_reference(request.original_audio_ref),
                    "vocal_stem_audio_ref": _worker_audio_reference(request.vocal_stem_audio_ref),
                }
            )
            response = await loop.run_in_executor(
                None,
                partial(
                    dispatch_worker_alignment,
                    remote_request,
                    worker_url,
                    worker_token=worker_token,
                ),
            )
            if response.status != "completed":
                diagnostics = response.diagnostics or {}
                raise RuntimeError(
                    f"Remote alignment worker reported failure: {diagnostics.get('reason', 'unknown error')}"
                )
            worker_name = "remote"
        else:
            fallback_evidence = asr_tokens_from_words(
                fallback_words,
                model="legacy-alignment-fallback",
                audio_source="vocal_stem",
                evidence_type="anchor",
            ) if fallback_words else ()
            evidence, source_errors = await loop.run_in_executor(
                None,
                partial(collect_model_evidence, request, fallback_words=fallback_evidence),
            )
            if not any(evidence.values()):
                raise RuntimeError(f"No usable model evidence remained: {source_errors or 'all sources were empty'}")
            response = await loop.run_in_executor(
                None,
                partial(
                    run_local_alignment,
                    request,
                    evidence,
                    activity_spans=activity_spans,
                    activity_times=pitch_times if pitch_times is not None else (),
                    activity_values=alignment_activity,
                ),
            )
            worker_name = "local"

        # Keep the worker's canonical spans separate from the
        # activity-expanded spans used by the alignment/export layer.
        # Sustain expansion is useful for timing, but sampling pitch from
        # those longer windows can pull the next word's onset into the
        # current word's note.
        canonical_pitch_alignments = alignment_response_to_models(response)
        alignments = list(canonical_pitch_alignments)
        if pitch_times is not None and pitch_energies is not None:
            alignments = expand_alignment_sustains(
                alignments,
                pitch_times,
                alignment_activity,
                clear_gap_s=0.12,
                activity_threshold=activity_threshold,
                window_end=audio_duration if audio_duration > 0 else None,
                melody_outputs=evidence if worker_name == "local" else None,
                melody_start_gap_threshold_s=0.08,
                melody_start_tolerance_s=0.08,
                melody_start_offset_s=0.08,
                melody_start_min_confidence=0.55,
                melody_handoff_min_gap_s=0.08,
                melody_handoff_max_gap_s=0.25,
                melody_handoff_offset_s=0.08,
            )
        if worker_name == "local":
            alignments = repair_melody_handoffs(
                alignments,
                evidence,
                tokens=selected.tokens,
                anchor_outputs=evidence,
            )

        previous = job.result.word_alignments if job.result and use_existing_alignment_locks else []
        reset_ids = reset_token_ids or set()
        aliases = map_token_id_aliases(
            job.result.lyric_revision.tokens if job.result and job.result.lyric_revision else (),
            selected.tokens,
        )
        old_reset_ids = {aliases.get(token_id, token_id) for token_id in reset_ids}
        previous_for_merge = [
            reset_word_boundaries(
                alignment,
                reset_start=True,
                reset_end=True,
            )
            if alignment.token_id in reset_ids or alignment.token_id in old_reset_ids
            else alignment
            for alignment in previous
        ]
        alignments = merge_alignment_locks(
            previous_for_merge,
            alignments,
            reset_token_ids=reset_ids,
            token_id_aliases=aliases,
        )

        # Apply explicit boundary locks to the canonical pitch spans without
        # reintroducing unlocked sustain extensions. This keeps user-edited
        # starts/ends authoritative while preserving the detector's clean
        # per-token windows for ordinary spans.
        canonical_pitch_alignments = _pitch_alignments_with_boundary_locks(
            canonical_pitch_alignments,
            alignments,
        )
        run.status = AlignmentRunStatus.COMPLETED
        run.diagnostics = {
            **response.diagnostics,
            "worker": worker_name,
            "source_errors": source_errors,
            "audio_references": {
                "original_mix": bool(job.audio_path),
                "vocal_stem": bool(job.vocals_path),
            },
        }

        canonical_words = project_alignments_to_words(selected.tokens, alignments)
        pitch_source_words = project_alignments_to_words(
            selected.tokens,
            canonical_pitch_alignments,
        )
        projected_line_breaks = line_breaks_for_projected_words(selected.tokens, canonical_words)
        if pitch_times is None:
            pitch_words = list(pitch_source_words)
            line_breaks = projected_line_breaks
        else:
            pitch_words, line_breaks = assign_pitch_to_words(
                pitch_source_words,
                pitch_times,
                pitch_frequencies,
                pitch_confidences,
                pitch_energies,
                line_breaks=projected_line_breaks,
                min_volume=DEFAULT_MIN_VOLUME_RMS,
                candidate_midis=pitch_candidate_midis,
                candidate_scores=pitch_candidate_scores,
                onset_scores=pitch_onset_scores,
            )
        return selected, run, alignments, canonical_words, pitch_words, line_breaks
    except Exception as exc:
        run.status = AlignmentRunStatus.FAILED
        run.diagnostics = {
            **(getattr(response, "diagnostics", {}) if "response" in locals() else {}),
            "error": str(exc),
            "worker": "remote" if os.environ.get("ALIGNMENT_WORKER_URL") else "local",
            "source_errors": source_errors if "source_errors" in locals() else {},
        }
        _persist_jobs()
        raise


async def _execute_realignment(
    job: Job,
    revision: LyricRevision,
    run: AlignmentRun,
    reset_token_ids: set[str],
    alignment_params: Any | None = None,
    transcribe_path: str | None = None,
    alignment_backend: str = "sofa",
) -> None:
    """Run an explicit revision alignment without silently changing locks."""

    log = logging.getLogger("pipeline")
    loop = asyncio.get_running_loop()
    try:
        from processing.lyrics import LyricsParams
        from processing.pitch import detect_pitch
        from processing.audio import normalize_audio

        params = alignment_params or LyricsParams(
            language=_enum_value(revision.language),
            initial_prompt=revision.raw_text,
            normalize=False,
            audio_source="vocals",
        )
        # Canonical alignment always labels the vocal stem correctly; the
        # worker independently evaluates the original mix as its second source.
        source_path = transcribe_path or job.vocals_path
        if params.normalize:
            normalized_path = os.path.join(job.work_dir, f"alignment-{run.id}-normalized.wav")
            await loop.run_in_executor(None, normalize_audio, source_path, normalized_path)
            source_path = normalized_path
        detection = await loop.run_in_executor(
            None,
            detect_pitch,
            job.vocals_path,
            job.audio_path,
        )
        times, frequencies, confidences, energies = detection
        pitch_candidate_midis = getattr(detection, "candidate_midis", None)
        pitch_candidate_scores = getattr(detection, "candidate_scores", None)
        pitch_onset_scores = getattr(detection, "onset_scores", None)
        selected, run, alignments, canonical_words, pitch_words, line_breaks = await _run_canonical_alignment(
            job,
            revision.raw_text,
            source_path,
            _enum_value(revision.language),
            params,
            [],
            times,
            frequencies,
            confidences,
            energies,
            pitch_candidate_midis,
            pitch_candidate_scores,
            pitch_onset_scores,
            audio_duration=job.result.audio_duration if job.result else _audio_duration(job.audio_path),
            reset_token_ids=reset_token_ids,
            alignment_run=run,
            alignment_backend=alignment_backend,
        )
        _apply_canonical_result(job, selected, run, alignments, canonical_words, pitch_words, line_breaks)
        job.error = ""
        job.status = JobStatus.DONE
        job.progress_message = "Re-alignment complete."
        _persist_jobs()
        _notify(job)
    except Exception as exc:
        log.exception("[%s] Explicit re-alignment failed", job.id[:8])
        run.status = AlignmentRunStatus.FAILED
        prior_diagnostics = run.diagnostics if isinstance(run.diagnostics, dict) else {}
        run.diagnostics = {**prior_diagnostics, "error": str(exc)}
        job.status = JobStatus.ERROR
        job.error = str(exc)
        job.progress_message = f"Alignment error: {exc}"
        _persist_jobs()
        _notify(job)


async def _process_phase1(job: Job) -> None:
    """Drive the shared Rust preparation pipeline and expose its status over HTTP."""
    job.failed_phase = "phase1"
    _persist_jobs()

    # Rust owns the cross-process model lock. The API semaphore keeps browser
    # job scheduling ordered without trying to acquire the same lock twice.
    async with processing_semaphore:
        job.queued_at = ""
        job.work_dir = os.path.join(OUTPUT_DIR, job.id)
        _persist_jobs()

        def progress(event: dict[str, Any]) -> None:
            _step(job, JobStatus(event["status"]), event["message"])

        try:
            media = await call_with_progress(
                "prepare_browser_media", progress,
                backend=str(Path(__file__).resolve().parent),
                media={
                    "source": job.upload_path, "work": job.work_dir, "prior_audio": job.audio_path or None,
                    "has_video": job.has_video, "real_bpm": job.real_bpm, "metadata": asdict(job.metadata),
                },
            )
            job.audio_path = media["audio_path"]
            job.vocals_path = media["vocals_path"]
            job.instrumental_path = media["instrumental_path"]
            job.has_video = media["has_video"]
            job.real_bpm = media["real_bpm"]
            job.found_lyrics = media["found_lyrics"]
            job.metadata.genre = media["metadata"]["genre"]
            job.metadata.year = media["metadata"]["year"]
            job.status = JobStatus.AWAITING_LYRICS
            job.failed_phase = ""
            job.progress_message = "Review lyrics before transcription"
        except Exception as exc:
            logging.getLogger("pipeline").exception("[%s] Preparation failed", job.id[:8])
            job.status = JobStatus.ERROR
            job.error = str(exc)
            job.progress_message = f"Error: {exc}"
        _persist_jobs()
        _notify(job)


async def _process_phase2(job: Job, lyrics_text: str) -> None:
    job.failed_phase = "phase2"
    _persist_jobs()
    async with processing_slot(OUTPUT_DIR, processing_semaphore):
        job.queued_at = ""
        _persist_jobs()
        await _process_phase2_impl(job, lyrics_text)


def _complete_published_job(job: Job) -> None:
    job.status = JobStatus.DONE
    job.failed_phase = ""
    job.error = ""
    job.progress_message = "Done! Your karaoke song is ready."
    result = job.result
    if (
        result and result.lyric_revision and result.alignment_run
        and _enum_value(result.lyric_revision.language) == "en"
        and "sofa" in result.alignment_run.selected_strategy
        and not result.alignment_comparison
    ):
        result.alignment_comparison = {
            "status": "running", "sofa_available": True, "legacy_available": False,
        }
        asyncio.create_task(_run_legacy_comparison(job, result.lyric_revision))
    _persist_jobs()
    _notify(job)


async def _retry_library_publication(job: Job) -> None:
    try:
        await asyncio.to_thread(_publish_song_to_library, job)
    except Exception as exc:
        job.status = JobStatus.ERROR
        job.error = str(exc)
        job.progress_message = f"Unable to publish song: {exc}"
        _persist_jobs()
        _notify(job)
        return
    _complete_published_job(job)


async def _process_phase2_impl(job: Job, lyrics_text: str) -> None:
    """Phase 2: Transcribe lyrics (using confirmed text as prompt), detect pitch, generate output."""
    log = logging.getLogger("pipeline")
    loop = asyncio.get_event_loop()
    phase_start = time.time()
    initial_alignment_run: AlignmentRun | None = None

    try:
        work_dir = job.work_dir
        vocals_path = job.vocals_path
        audio_path = job.audio_path
        instrumental_path = job.instrumental_path or os.path.join(work_dir, "instrumental.wav")
        has_video = job.has_video

        # --- Step 5: Extract lyrics ---
        has_lyrics = bool(lyrics_text.strip()) and lyrics_text.strip() != "These are song lyrics with singing."
        step_msg = "Aligning provided lyrics to audio..." if has_lyrics else "Transcribing lyrics with word timestamps..."
        t = _step(job, JobStatus.EXTRACTING_LYRICS, step_msg)

        existing_revision = _selected_revision(job)
        initial_alignment_run = AlignmentRun(
            id=str(uuid.uuid4()),
            lyric_revision_id=existing_revision.id if existing_revision else "",
            status=AlignmentRunStatus.RUNNING,
            selected_strategy="sofa-primary-global-match-and-candidate-score",
        )
        job.alignment_runs.append(initial_alignment_run)

        from processing.lyrics import LyricsParams
        from processing.audio import normalize_audio

        params = LyricsParams()
        if has_lyrics:
            params.initial_prompt = lyrics_text.strip()
        from processing.alignment import normalize_language
        try:
            lang = normalize_language(job.metadata.language)
        except ValueError:
            lang = "en"

        # Normalize vocals for transcription/alignment
        transcribe_path = vocals_path
        if params.normalize:
            norm_path = os.path.join(work_dir, "vocals_normalized.wav")
            log.info("[%s] Normalizing vocals...", job.id[:8])
            await loop.run_in_executor(None, normalize_audio, vocals_path, norm_path)
            transcribe_path = norm_path

        # --- Step 6: Detect pitch (once) ---
        t = _step(job, JobStatus.DETECTING_PITCH, "Analyzing vocal pitch...")

        from processing.pitch import detect_pitch
        detection = await loop.run_in_executor(
            None, detect_pitch, vocals_path, audio_path
        )
        times, frequencies, confidences, energies = detection
        pitch_candidate_midis = getattr(detection, "candidate_midis", None)
        pitch_candidate_scores = getattr(detection, "candidate_scores", None)
        pitch_onset_scores = getattr(detection, "onset_scores", None)
        log.info("[%s] Pitch detected in %.1fs -> %d frames", job.id[:8], time.time() - t, len(times))

        raw_asr_words_for_draft: list[Any] = []
        alignment_lyrics_text = lyrics_text
        if not has_lyrics:
            if os.environ.get("ALIGNMENT_WORKER_URL", "").strip():
                alignment_lyrics_text = await loop.run_in_executor(
                    None,
                    _draft_text_from_remote_worker,
                    job,
                    initial_alignment_run.id,
                    lang,
                )
                draft_revision = _create_lyric_revision(
                    job,
                    alignment_lyrics_text,
                    source=LyricSource.ASR_DRAFT,
                    language=lang,
                )
                initial_alignment_run.lyric_revision_id = draft_revision.id
                job.confirmed_lyrics = draft_revision.raw_text
                words_for_download = []
            else:
                words_for_download = await loop.run_in_executor(
                    None, _draft_words_from_audio, transcribe_path, lang, params
                )
                raw_asr_words_for_draft = list(words_for_download)
        else:
            words_for_download = []

        # Canonical timing is selected from dual-source evidence. Pitch reads
        # the resulting spans later, but never supplies canonical timing.
        (
            canonical_revision,
            canonical_run,
            canonical_alignments,
            canonical_words,
            pitch_words,
            canonical_line_breaks,
        ) = await _run_canonical_alignment(
            job,
            alignment_lyrics_text,
            transcribe_path,
            lang,
            params,
            raw_asr_words_for_draft or words_for_download or [],
            times,
            frequencies,
            confidences,
            energies,
            pitch_candidate_midis,
            pitch_candidate_scores,
            pitch_onset_scores,
            audio_duration=_audio_duration(job.audio_path),
            alignment_run=initial_alignment_run,
        )
        words_for_download = canonical_words
        line_breaks_for_download = canonical_line_breaks

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

        def build_variant(canonical: Any, pitch: Any, lb: list[int]) -> AlignmentVariant:
            ph, gap_ms = words_to_phrases(pitch, job.real_bpm, line_breaks=lb)
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
            return AlignmentVariant(
                words=canonical,
                pitch_words=pitch,
                phrases=ph,
                ultrastar_txt=txt,
                gap_ms=gap_ms,
                word_alignments=canonical_alignments,
                alignment_run=canonical_run,
            )

        canonical_variant = build_variant(canonical_words, pitch_words, line_breaks_for_download)
        # Keep canonical for older clients, while exposing an explicit SOFA
        # lane so the legacy comparison can be switched to without rerunning.
        variants_for_result = {"canonical": canonical_variant, "sofa": canonical_variant}
        phrases = canonical_variant.phrases
        ultrastar_txt = canonical_variant.ultrastar_txt
        log.info(
            "[%s] Generated %d phrases from %d words (GAP=%.0fms)",
            job.id[:8],
            len(phrases),
            len(words_for_download),
            canonical_variant.gap_ms,
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
            words=canonical_words,
            pitch_words=pitch_words,
            phrases=phrases,
            ultrastar_txt=ultrastar_txt,
            audio_file=audio_mp3,
            vocals_file=vocals_mp3,
            instrumental_file=instrumental_mp3,
            video_file=video_mp4,
            alignment_results=variants_for_result,
            current_alignment_mode="canonical",
            lyric_revision=canonical_revision,
            word_alignments=canonical_alignments,
            alignment_run=canonical_run,
            audio_duration=_audio_duration(job.audio_path),
        )
        job.failed_phase = "publish"
        _persist_jobs()
        await asyncio.to_thread(_publish_song_to_library, job)

        total = time.time() - phase_start
        _complete_published_job(job)
        modes_str = ", ".join(sorted(variants_for_result.keys()))
        log.info("[%s] Phase 2 complete in %.1fs. Alignment modes: %s", job.id[:8], total, modes_str)

    except Exception as e:
        log.exception("[%s] Phase 2 failed after %.1fs", job.id[:8], time.time() - phase_start)
        if initial_alignment_run is not None and _enum_value(initial_alignment_run.status) == AlignmentRunStatus.RUNNING.value:
            initial_alignment_run.status = AlignmentRunStatus.FAILED
            prior_diagnostics = initial_alignment_run.diagnostics if isinstance(initial_alignment_run.diagnostics, dict) else {}
            initial_alignment_run.diagnostics = {**prior_diagnostics, "error": str(e)}
        job.status = JobStatus.ERROR
        if job.failed_phase != "publish":
            job.failed_phase = "phase2"
        job.error = str(e)
        job.progress_message = f"Error: {e}"
        _persist_jobs()
        _notify(job)


# ── API Endpoints ──────────────────────────────────────────────────────────


def _benchmark_asset(path_value: Any, *, required: bool = True) -> Path | None:
    """Resolve a cached benchmark asset without allowing arbitrary file reads."""

    if not isinstance(path_value, str) or not path_value.strip():
        if required:
            raise HTTPException(status_code=422, detail="Benchmark asset path is required")
        return None
    candidate = Path(path_value).expanduser().resolve()
    allowed_roots = (
        (Path(__file__).resolve().parent / "benchmarks").resolve(),
        Path(UPLOAD_DIR).resolve(),
    )
    if not any(candidate == root or root in candidate.parents for root in allowed_roots):
        raise HTTPException(status_code=422, detail="Benchmark asset must be inside the local benchmark cache")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"Benchmark asset not found: {candidate.name}")
    return candidate


def _copy_benchmark_asset(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            if os.path.samefile(source, destination):
                return
        except OSError:
            pass
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _benchmark_phrases(notes: list[Any]) -> list[Phrase]:
    """Turn parsed chart notes into editor phrases while retaining note timing."""

    phrases: list[Phrase] = []
    current: list[UltraStarNote] = []
    previous_end = -1
    for note in notes:
        if current and note.start_beat - previous_end > 4:
            phrases.append(Phrase(notes=current, end_beat=previous_end + 1))
            current = []
        current.append(UltraStarNote(
            note_type=note.note_type,
            start_beat=note.start_beat,
            duration_beats=note.duration_beats,
            pitch=note.pitch,
            text=note.text,
        ))
        previous_end = note.start_beat + note.duration_beats
    if current:
        phrases.append(Phrase(notes=current, end_beat=previous_end + 1))
    return phrases


@app.post("/api/benchmark/import", status_code=201)
async def import_benchmark_result(body: dict[str, Any], request: Request) -> dict[str, Any]:
    """Create a reviewable local job from a cached pipeline chart and stems."""

    pipeline = str(body.get("pipeline", "current")).strip().casefold()
    if pipeline not in {"current", "sofa"}:
        raise HTTPException(status_code=422, detail="pipeline must be current or sofa")
    chart = _benchmark_asset(body.get("chart_path"))
    audio = _benchmark_asset(body.get("audio_path"))
    vocals = _benchmark_asset(body.get("vocal_path", body.get("vocals_path")))
    instrumental = _benchmark_asset(body.get("instrumental_path"), required=False)

    import benchmark_alignment as benchmark
    from processing.alignment import tokenize_lyrics

    try:
        reference = benchmark.parse_ultrastar_reference(chart)
        _headers, pitch_notes = benchmark.parse_ultrastar_pitch_notes(chart)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid benchmark chart: {exc}") from exc
    requested_job_id = str(body.get("job_id", "")).strip()
    if requested_job_id and not requested_job_id.startswith("benchmark-"):
        raise HTTPException(status_code=422, detail="job_id must use the benchmark- prefix")
    job_id = requested_job_id or f"benchmark-{uuid.uuid4().hex[:16]}"
    if job_id in jobs:
        raise HTTPException(status_code=409, detail="Benchmark job already exists")
    work_dir = Path(UPLOAD_DIR) / job_id
    _copy_benchmark_asset(audio, work_dir / "audio.wav")
    _copy_benchmark_asset(vocals, work_dir / "vocals.wav")
    if instrumental:
        _copy_benchmark_asset(instrumental, work_dir / "instrumental.wav")

    headers = reference.headers
    try:
        chart_bpm = float(headers.get("BPM", "0").replace(",", "."))
        gap_ms = float(headers.get("GAP", "0").replace(",", "."))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Benchmark chart has invalid BPM/GAP") from exc
    metadata = SongMetadata(
        title=str(body.get("title") or headers.get("TITLE", reference.path.stem)),
        artist=str(body.get("artist") or headers.get("ARTIST", "")),
        language=headers.get("LANGUAGE", "en"),
        bpm=chart_bpm / 4.0,
        gap_ms=gap_ms,
    )
    job = Job(
        id=job_id,
        status=JobStatus.DONE,
        progress_message=f"Imported {pipeline} benchmark result",
        progress_percent=100.0,
        original_filename=f"{metadata.artist} - {metadata.title} [{pipeline}]",
        metadata=metadata,
        work_dir=str(work_dir),
        upload_path=str(work_dir / "audio.wav"),
        audio_path=str(work_dir / "audio.wav"),
        vocals_path=str(work_dir / "vocals.wav"),
        instrumental_path=str(work_dir / "instrumental.wav") if instrumental else "",
        real_bpm=metadata.bpm,
        owner_id=getattr(request.state, "owner_id", "local"),
    )
    revision = LyricRevision(
        id=f"revision-{uuid.uuid4().hex}",
        song_id=job.id,
        language=metadata.language,
        source=LyricSource.MANUAL,
        raw_text=reference.lyrics,
    )
    revision.tokens = tokenize_lyrics(reference.lyrics, revision.id, metadata.language)
    if len(revision.tokens) != len(reference.words):
        raise HTTPException(status_code=422, detail="Benchmark chart lyrics do not tokenize one-to-one")
    job.lyric_revisions = [revision]
    job.selected_lyric_revision_id = revision.id
    run = AlignmentRun(
        id=f"run-{uuid.uuid4().hex}",
        lyric_revision_id=revision.id,
        status=AlignmentRunStatus.COMPLETED,
        selected_strategy=pipeline,
        diagnostics={"source": "cached benchmark chart", "pipeline": pipeline},
    )
    job.alignment_runs = [run]
    request_for_pitch = benchmark._build_request(reference, audio, vocals)
    notes_by_word = benchmark._pitch_notes_by_word(reference, pitch_notes, request_for_pitch.language)
    words: list[Word] = []
    alignments: list[WordAlignment] = []
    for index, (token, expected, expected_notes) in enumerate(zip(revision.tokens, reference.words, notes_by_word)):
        midi = (max(expected_notes, key=lambda note: note.duration_beats).pitch + 60) if expected_notes else 0
        word = Word(
            text=expected.text,
            start=expected.start,
            end=expected.end,
            midi_note=midi,
            confidence=1.0,
            token_id=token.id,
            alignment_status="aligned",
            alignment_run_id=run.id,
        )
        words.append(word)
        alignments.append(WordAlignment(
            token_id=token.id,
            alignment_run_id=run.id,
            start=word.start,
            end=word.end,
            status="aligned",
            confidence=1.0,
            evidence=AlignmentEvidence(selected_model=pipeline, selected_audio="vocal_stem"),
        ))
    phrases = _benchmark_phrases(list(pitch_notes))
    chart_text = chart.read_text(encoding="utf-8")
    chart_lines = []
    for line in chart_text.splitlines():
        if line.startswith("#MP3:"):
            line = "#MP3:audio.wav"
        elif line.startswith("#VOCALS:"):
            line = "#VOCALS:vocals.wav"
        elif line.startswith("#INSTRUMENTAL:"):
            line = "#INSTRUMENTAL:instrumental.wav"
        chart_lines.append(line)
    result = JobResult(
        metadata=metadata,
        words=words,
        pitch_words=list(words),
        phrases=phrases,
        ultrastar_txt="\n".join(chart_lines) + "\n",
        audio_file="audio.wav",
        vocals_file="vocals.wav",
        instrumental_file="instrumental.wav" if instrumental else "",
        alignment_results={pipeline: AlignmentVariant(words=words, pitch_words=list(words), phrases=phrases, ultrastar_txt="\n".join(chart_lines) + "\n", gap_ms=gap_ms)},
        current_alignment_mode=pipeline,
        lyric_revision=revision,
        word_alignments=alignments,
        alignment_run=run,
        audio_duration=_audio_duration(str(work_dir / "audio.wav")),
    )
    job.result = result
    jobs[job.id] = job
    _persist_jobs()
    return {"job_id": job.id, "pipeline": pipeline, "result": _serialize_result(job)}


@app.post("/v1/alignments", status_code=202)
async def request_worker_alignment(
    body: dict[str, Any],
    worker_token: str | None = Header(default=None, alias="X-Alignment-Worker-Token"),
) -> dict[str, Any]:
    """Execute an authenticated alignment request on the worker host."""

    from processing.alignment import normalize_language
    from processing.alignment_worker import AlignmentWorkerRequest, AlignmentWorkerResponse, run_worker_alignment

    try:
        parsed = AlignmentWorkerRequest(**body)
        normalize_language(parsed.language)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid alignment request: {exc}") from exc
    configured_token = os.environ.get("ALIGNMENT_WORKER_TOKEN", "").strip()
    if not configured_token:
        raise HTTPException(status_code=503, detail="Alignment worker ingress is not configured")
    if worker_token != configured_token:
        raise HTTPException(status_code=401, detail="Invalid alignment worker credentials")
    if not parsed.original_audio_ref or not parsed.vocal_stem_audio_ref:
        raise HTTPException(status_code=422, detail="Both audio references are required")
    worker_root = os.path.realpath(
        os.environ.get("ALIGNMENT_WORKER_AUDIO_ROOT", os.path.dirname(__file__))
    )
    for reference in (parsed.original_audio_ref, parsed.vocal_stem_audio_ref):
        if "://" in reference or not os.path.isabs(reference):
            raise HTTPException(status_code=422, detail="Worker audio references must be absolute local paths")
        resolved = os.path.realpath(reference)
        if resolved != worker_root and not resolved.startswith(worker_root + os.sep):
            raise HTTPException(status_code=422, detail="Worker audio reference is outside the worker audio root")
        if not os.path.isfile(resolved):
            raise HTTPException(status_code=404, detail="Worker audio reference is unavailable")

    loop = asyncio.get_running_loop()
    try:
        response = await loop.run_in_executor(None, run_worker_alignment, parsed)
    except Exception as exc:
        response = AlignmentWorkerResponse(
            job_id=parsed.job_id,
            alignment_run_id=parsed.alignment_run_id,
            lyric_revision_id=parsed.lyric_revision_id,
            status="failed",
            diagnostics={"error": str(exc)},
        )
    return response.model_dump()


@app.post("/api/parse-filename")
async def parse_filename_endpoint(body: dict[str, Any]) -> dict[str, str]:
    """Use an LLM to extract artist/title/etc. from a filename."""
    filename = body.get("filename", "")
    if not isinstance(filename, str):
        raise HTTPException(status_code=422, detail="filename must be a string")
    if not filename:
        return {"title": "", "artist": "", "language": "", "year": "", "genre": ""}

    loop = asyncio.get_event_loop()
    from processing.metadata import parse_filename

    return await loop.run_in_executor(None, parse_filename, filename)


@app.post("/api/upload")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    title: str = Form(""),
    artist: str = Form(""),
    language: str = Form(""),
    genre: str = Form(""),
    year: str = Form(""),
) -> dict[str, str]:
    """Upload an audio/video file and create a processing job."""
    from processing.alignment import normalize_language

    job_id = str(uuid.uuid4())

    job_upload_dir = os.path.join(UPLOAD_DIR, job_id)
    os.makedirs(job_upload_dir, exist_ok=True)

    original_filename = (file.filename or "input").replace("\\", "/")
    filename = Path(original_filename).name
    if not filename or filename in {".", ".."} or filename != original_filename or "\x00" in filename:
        shutil.rmtree(job_upload_dir, ignore_errors=True)
        raise HTTPException(status_code=422, detail="Invalid upload filename")
    suffix = Path(filename).suffix.lower()
    stored_name = f"input{suffix}" if suffix else "input.bin"
    upload_path = os.path.join(job_upload_dir, stored_name)
    total_bytes = 0
    try:
        with open(upload_path, "wb") as output:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Uploaded file is too large")
                output.write(chunk)
    except Exception:
        shutil.rmtree(job_upload_dir, ignore_errors=True)
        raise
    try:
        normalized_language = normalize_language(language or "en")
    except ValueError as exc:
        shutil.rmtree(job_upload_dir, ignore_errors=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    job = Job(
        id=job_id,
        owner_id=_request_owner(request),
        original_filename=filename,
        upload_path=upload_path,
        metadata=SongMetadata(
            title=title or os.path.splitext(filename)[0],
            artist=artist,
            language=normalized_language,
            genre=genre,
            year=year,
        ),
    )
    jobs[job_id] = job
    _persist_jobs()

    return {"job_id": job_id}


class PreparedMediaImport(BaseModel):
    import_id: uuid.UUID
    title: str = ""
    artist: str = ""
    language: str = ""
    genre: str = ""
    year: str = ""


class PreparedMediaManifest(BaseModel):
    original_filename: str
    has_video: bool


@app.post("/api/desktop/import")
async def register_prepared_media(request: Request, body: PreparedMediaImport) -> dict[str, str]:
    """Register the files prepared by the local desktop process."""
    from processing.alignment import normalize_language

    if not request.client or request.client.host not in {"127.0.0.1", "::1", "localhost", "testclient"}:
        raise HTTPException(status_code=403, detail="Prepared media import requires a local backend")
    job_id = str(body.import_id)
    owner_id = _request_owner(request)
    upload_dir = _safe_child_path(UPLOAD_DIR, job_id)
    work_dir = _safe_child_path(OUTPUT_DIR, job_id)
    upload_path = _safe_child_path(upload_dir, "input")
    audio_path = _safe_child_path(work_dir, "audio.wav")

    existing = jobs.get(job_id)
    if existing:
        if existing.owner_id == owner_id and existing.upload_path == upload_path:
            return {"job_id": job_id}
        raise HTTPException(status_code=404, detail="Prepared media not found")

    manifest_path = Path(_safe_child_path(upload_dir, "prepared-media.json"))
    try:
        manifest = PreparedMediaManifest.model_validate_json(manifest_path.read_bytes())
        original_size = Path(upload_path).stat().st_size
        if original_size > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Selected file is too large")
        with wave.open(audio_path, "rb") as audio:
            pcm_format = (audio.getnchannels(), audio.getsampwidth(), audio.getframerate())
            if pcm_format != (2, 2, 44100) or not audio.getnframes():
                raise ValueError("Expected a nonempty 44.1 kHz stereo PCM WAV")
        if not original_size or not Path(upload_path).is_file() or not manifest.original_filename:
            raise ValueError("Prepared import is missing the original media")
        language = normalize_language(body.language or "en")
        record = _safe_child_path(upload_dir, "desktop-preparation.json")
        prepared = json.loads(Path(record).read_text(encoding="utf-8"))
        if not isinstance(prepared, dict) or prepared.get("status") != "awaiting_lyrics":
            raise ValueError("Desktop preparation has not reached lyrics review")
        bpm = prepared.get("real_bpm")
        if isinstance(bpm, bool) or not isinstance(bpm, (int, float)) or not math.isfinite(bpm) or bpm < 0:
            raise ValueError("Invalid prepared tempo")
        for track in ("vocals.wav", "instrumental.wav"):
            with wave.open(_safe_child_path(work_dir, track), "rb") as stem:
                if not stem.getnframes():
                    raise ValueError(f"Prepared {track} is empty")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Prepared media not found in this backend") from exc
    except (ValueError, wave.Error, EOFError, OSError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid prepared media: {exc}") from exc

    jobs[job_id] = Job(
        id=job_id,
        owner_id=owner_id,
        original_filename=manifest.original_filename,
        upload_path=upload_path,
        work_dir=work_dir,
        audio_path=audio_path,
        has_video=manifest.has_video,
        status=JobStatus.AWAITING_LYRICS,
        progress_message="Waiting for lyrics approval.",
        vocals_path=_safe_child_path(work_dir, "vocals.wav"),
        instrumental_path=_safe_child_path(work_dir, "instrumental.wav"),
        real_bpm=float(prepared.get("real_bpm", 0.0)),
        found_lyrics=str(prepared.get("found_lyrics", "")),
        metadata=SongMetadata(
            title=body.title or Path(manifest.original_filename).stem,
            artist=body.artist,
            language=language,
            genre=body.genre,
            year=body.year,
        ),
    )
    try:
        _persist_jobs()
    except Exception:
        jobs.pop(job_id)
        raise
    return {"job_id": job_id}


@app.post("/api/process/{job_id}/start")
async def start_processing(job_id: str) -> dict[str, str]:
    """Start Phase 1 of the processing pipeline."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.PENDING:
        raise HTTPException(status_code=400, detail="Job already started")
    # Claim synchronously before yielding to the background task so duplicate
    # HTTP requests cannot launch two phase-one pipelines.
    job.status = JobStatus.EXTRACTING_AUDIO
    job.progress_message = "Queued for processing..."
    job.queued_at = datetime.now(timezone.utc).isoformat()
    _persist_jobs()
    asyncio.create_task(_process_phase1(job))
    return {"status": "started"}


@app.post("/api/jobs/{job_id}/retry")
async def retry_job(job_id: str) -> dict[str, str]:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.ERROR:
        raise HTTPException(status_code=409, detail="Only failed songs can be retried")

    phase = job.failed_phase
    job.error = ""
    job.progress_percent = 0.0
    if phase == "publish" and job.result:
        job.status = JobStatus.GENERATING
        job.progress_message = "Publishing the finished song…"
        _persist_jobs()
        asyncio.create_task(_retry_library_publication(job))
    elif phase == "download":
        if not job.source_url:
            raise HTTPException(status_code=409, detail="The original YouTube URL is unavailable; upload or select the song again")
        req = YouTubeDownloadRequest(
            url=job.source_url,
            title=job.metadata.title,
            artist=job.metadata.artist,
            language=job.metadata.language,
            genre=job.metadata.genre,
            year=job.metadata.year,
        )
        job.status = JobStatus.DOWNLOADING
        job.progress_message = "Retrying YouTube download..."
        _persist_jobs()
        asyncio.create_task(_download_youtube_video(job_id, req))
    elif phase == "phase2" and job.audio_path and os.path.isfile(job.audio_path) and job.vocals_path and os.path.isfile(job.vocals_path):
        job.status = JobStatus.EXTRACTING_LYRICS
        job.progress_message = "Retrying from lyric extraction..."
        job.queued_at = datetime.now(timezone.utc).isoformat()
        _persist_jobs()
        asyncio.create_task(_process_phase2(job, job.confirmed_lyrics))
    else:
        if not job.upload_path or not os.path.isfile(job.upload_path):
            raise HTTPException(status_code=409, detail="The original audio file is unavailable; upload the song again")
        job.status = JobStatus.EXTRACTING_AUDIO
        job.progress_message = "Retrying from the last completed processing step..."
        job.queued_at = datetime.now(timezone.utc).isoformat()
        _persist_jobs()
        asyncio.create_task(_process_phase1(job))
    return {"status": "retrying", "phase": phase or "phase1"}


@app.post("/api/jobs/{job_id}/confirm-lyrics")
async def confirm_lyrics(job_id: str, body: dict[str, Any]) -> dict[str, str]:
    """Confirm lyrics and start Phase 2 (transcription + generation)."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    lyrics_value = body.get("lyrics", "")
    if not isinstance(lyrics_value, str):
        raise HTTPException(status_code=422, detail="lyrics must be a string")

    if job.status == JobStatus.AWAITING_LYRICS:
        lyrics_text = lyrics_value.strip()
        if lyrics_text:
            current_revision = _selected_revision(job)
            if current_revision is None or current_revision.raw_text != lyrics_text:
                source = (
                    LyricSource.ONLINE
                    if lyrics_text == job.found_lyrics.strip()
                    else LyricSource.MANUAL
                )
                revision = _create_lyric_revision(
                    job,
                    lyrics_text,
                    source=source,
                    parent_revision_id=current_revision.id if current_revision else None,
                    language=job.metadata.language,
                )
                job.selected_lyric_revision_id = revision.id
            job.confirmed_lyrics = lyrics_text
        else:
            # An empty confirmation deliberately selects the ASR draft path.
            job.confirmed_lyrics = ""
        job.status = JobStatus.EXTRACTING_LYRICS
        job.progress_message = "Preparing lyric evidence..."
        job.queued_at = datetime.now(timezone.utc).isoformat()
        _persist_jobs()
        asyncio.create_task(_process_phase2(job, lyrics_text))
        return {"status": "started"}

    lyrics_text = lyrics_value.strip()
    if not lyrics_text:
        raise HTTPException(status_code=422, detail="lyrics cannot be empty")
    if job.result and job.status in {JobStatus.DONE, JobStatus.ERROR}:
        revision = job.result.lyric_revision
        if revision is None or revision.raw_text != lyrics_text.strip():
            revision = _create_lyric_revision(
                job,
                lyrics_text.strip(),
                source=LyricSource.MANUAL,
                parent_revision_id=revision.id if revision else None,
                language=_enum_value(revision.language) if revision else job.metadata.language,
            )
        run_response = await start_alignment_run(
            job_id,
            {"lyric_revision_id": revision.id, "reset_token_ids": []},
        )
        return {"status": "started", "alignment_run_id": run_response["alignment_run_id"]}
    raise HTTPException(status_code=400, detail="Job is not ready for lyrics confirmation")


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
    if not job.result:
        raise HTTPException(status_code=400, detail="Job has no completed result yet")
    return _serialize_result(job)


@app.get("/api/jobs/{job_id}/alignment")
async def get_alignment(job_id: str) -> dict[str, Any]:
    """Return the canonical alignment layer and safe source metadata."""

    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.result or not job.result.lyric_revision:
        raise HTTPException(status_code=400, detail="No alignment result yet")
    completed_run = job.result.alignment_run
    latest_run = job.alignment_runs[-1] if job.alignment_runs else completed_run
    return {
        "job_id": job.id,
        "status": job.status.value,
        "lyric_revision": _serialize_revision(job.result.lyric_revision),
        "word_alignments": [_serialize_word_alignment(item) for item in job.result.word_alignments],
        # These words and this run are one internally consistent snapshot.
        "alignment_run": _serialize_alignment_run(completed_run),
        "last_completed_alignment_run": _serialize_alignment_run(completed_run),
        "latest_alignment_run": _serialize_alignment_run(latest_run),
    }


@app.post("/api/jobs/{job_id}/lyric-revisions", status_code=201)
async def save_lyric_revision(job_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """Save edited source lyrics without automatically starting alignment."""

    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status == JobStatus.ALIGNING_WORDS:
        raise HTTPException(status_code=409, detail="Cannot revise lyrics during alignment")
    if not job.result or not job.result.lyric_revision:
        raise HTTPException(status_code=400, detail="Lyrics can only be revised after an alignment result")
    raw_text_value = body.get("raw_text", body.get("lyrics", ""))
    if not isinstance(raw_text_value, str):
        raise HTTPException(status_code=422, detail="raw_text must be a string")
    raw_text = raw_text_value
    if not raw_text.strip():
        raise HTTPException(status_code=422, detail="Lyric revision cannot be empty")
    from processing.alignment import normalize_language

    try:
        language = normalize_language(body.get("language") or _enum_value(job.result.lyric_revision.language))
    except (AttributeError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    parent_revision = _selected_revision(job) or job.result.lyric_revision
    revision = _create_lyric_revision(
        job,
        raw_text,
        source=LyricSource.MANUAL,
        parent_revision_id=parent_revision.id,
        language=language,
    )
    _persist_jobs()
    return {
        "status": "saved",
        "alignment_started": False,
        "revision": _serialize_revision(revision),
    }


@app.post("/api/jobs/{job_id}/alignment-runs", status_code=202)
async def start_alignment_run(job_id: str, body: dict[str, Any]) -> dict[str, str]:
    """Explicitly re-align a selected revision, preserving locks by default."""

    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.result or not job.result.lyric_revision:
        raise HTTPException(status_code=400, detail="No initial alignment result")
    if not job.vocals_path or not os.path.exists(job.vocals_path):
        raise HTTPException(status_code=400, detail="Vocals file not available for alignment")
    if job.status not in {JobStatus.DONE, JobStatus.ERROR}:
        raise HTTPException(status_code=409, detail="Another alignment is already running")

    alignment_backend = str(body.get("alignment_backend", "sofa")).strip().casefold()
    if alignment_backend not in {"sofa", "legacy"}:
        raise HTTPException(status_code=422, detail="alignment_backend must be sofa or legacy")

    revision_id_value = body.get("lyric_revision_id", "")
    if not isinstance(revision_id_value, str) or not revision_id_value.strip():
        raise HTTPException(status_code=422, detail="lyric_revision_id must be a non-empty string")
    revision_id = revision_id_value
    revision = next((item for item in job.lyric_revisions if item.id == revision_id), None)
    if revision is None:
        raise HTTPException(status_code=409, detail="Lyric revision is stale or unknown")
    from processing.alignment_worker import resolve_alignment_backend

    alignment_backend = resolve_alignment_backend(
        _enum_value(revision.language),
        alignment_backend,
    )
    reset_values = body.get("reset_token_ids", [])
    if reset_values is None or not isinstance(reset_values, list) or not all(isinstance(token_id, str) for token_id in reset_values):
        raise HTTPException(status_code=422, detail="reset_token_ids must be an array of strings")
    reset_token_ids = set(reset_values)
    unknown_resets = reset_token_ids - {token.id for token in revision.tokens}
    if unknown_resets:
        raise HTTPException(status_code=422, detail="Reset contains unknown token IDs")

    run_id = str(uuid.uuid4())
    run = AlignmentRun(
        id=run_id,
        lyric_revision_id=revision.id,
        status=AlignmentRunStatus.RUNNING,
        selected_strategy=(
            "legacy-global-match-and-candidate-score"
            if alignment_backend == "legacy"
            else "sofa-primary-global-match-and-candidate-score"
        ),
    )
    job.alignment_runs.append(run)
    job.selected_lyric_revision_id = revision.id
    _step(job, JobStatus.ALIGNING_WORDS, "Re-aligning the selected lyric revision...")
    if alignment_backend == "legacy":
        alignment_task = _execute_realignment(
            job,
            revision,
            run,
            reset_token_ids,
            alignment_backend="legacy",
        )
    else:
        # Keep the default call shape compatible with older worker adapters
        # that monkeypatch/forward the realignment coroutine positionally.
        alignment_task = _execute_realignment(job, revision, run, reset_token_ids)
    asyncio.create_task(alignment_task)
    return {"alignment_run_id": run_id, "lyric_revision_id": revision.id, "status": "running"}


@app.post("/api/jobs/{job_id}/alignment-comparison", status_code=202)
async def start_alignment_comparison(job_id: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    """Start the archived lane in the background while keeping SOFA usable."""

    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.result or not job.result.lyric_revision:
        raise HTTPException(status_code=400, detail="No initial alignment result")
    if job.status == JobStatus.ALIGNING_WORDS:
        raise HTTPException(status_code=409, detail="Another alignment is already running")
    result = job.result
    comparison = result.alignment_comparison or {}
    if comparison.get("status") == "running":
        return {"status": "running", "alignment_comparison": comparison}
    if "legacy" in (result.alignment_results or {}):
        return {"status": "completed", "alignment_comparison": comparison}

    from processing.alignment import normalize_language

    try:
        language = normalize_language(_enum_value(result.lyric_revision.language))
    except ValueError:
        language = "en"
    selected_strategy = _enum_value(result.alignment_run.selected_strategy) if result.alignment_run else ""
    sofa_variant = _alignment_variant_for_mode(result, "sofa")
    # SOFA is intentionally not attempted for Polish; the resolved initial
    # run is already the archived lane and remains the only cached version.
    if language != "en" or "sofa" not in selected_strategy:
        result.alignment_comparison = {
            "status": "not_supported",
            "sofa_available": False,
            "legacy_available": True,
            "preferred_mode": "legacy",
            "reason": "SOFA comparison is only enabled for English lyrics",
        }
        _persist_jobs()
        return {"status": "not_supported", "alignment_comparison": result.alignment_comparison}
    if sofa_variant is None:
        raise HTTPException(status_code=409, detail="SOFA result is unavailable for comparison")
    result.alignment_results["sofa"] = sofa_variant
    result.alignment_comparison = {
        "status": "running",
        "sofa_available": True,
        "legacy_available": False,
    }
    _persist_jobs()
    revision_id = str((body or {}).get("lyric_revision_id") or result.lyric_revision.id)
    revision = next((item for item in job.lyric_revisions if item.id == revision_id), None)
    if revision is None:
        revision = result.lyric_revision
    asyncio.create_task(_run_legacy_comparison(job, revision))
    return {"status": "running", "alignment_comparison": result.alignment_comparison}


@app.patch("/api/jobs/{job_id}/alignment-mode")
async def switch_alignment_mode(job_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """Switch between already-computed SOFA/legacy exports without rerunning models."""

    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.result:
        raise HTTPException(status_code=400, detail="No alignment result")
    mode = str(body.get("mode", "")).strip().casefold()
    if mode not in {"sofa", "legacy", "canonical"}:
        raise HTTPException(status_code=422, detail="mode must be sofa, legacy, or canonical")
    try:
        resolved_mode = _apply_alignment_variant(job, mode)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    _persist_jobs()
    payload = _serialize_result(job)
    payload["switched_to"] = resolved_mode
    return payload


@app.patch("/api/jobs/{job_id}/word-alignments/{token_id}")
async def update_word_alignment(job_id: str, token_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """Move one or both canonical boundaries and hard-lock changed fields."""

    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status == JobStatus.ALIGNING_WORDS:
        raise HTTPException(status_code=409, detail="Cannot edit word boundaries during alignment")
    if not job.result or not job.result.lyric_revision:
        raise HTTPException(status_code=400, detail="No alignment result")
    if "start" not in body and "end" not in body:
        raise HTTPException(status_code=422, detail="Provide start and/or end")
    unknown_fields = set(body) - {"start", "end"}
    if unknown_fields:
        raise HTTPException(status_code=422, detail="Only start and end may be edited")

    from processing.alignment import edit_word_boundary

    alignments = list(job.result.word_alignments)
    index = next((i for i, item in enumerate(alignments) if item.token_id == token_id), -1)
    if index < 0:
        raise HTTPException(status_code=404, detail="Word token not found")
    kwargs: dict[str, Any] = {}
    if "start" in body:
        kwargs["start"] = body["start"]
    if "end" in body:
        kwargs["end"] = body["end"]
    try:
        updated = edit_word_boundary(alignments[index], **kwargs)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    alignments[index] = updated
    try:
        from processing.alignment import validate_alignment_sequence

        validate_alignment_sequence(alignments)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Word boundaries must remain finite and monotonic") from exc
    job.result.word_alignments = alignments
    _refresh_legacy_result_from_canonical(job)
    _persist_jobs()
    return {
        "status": "updated",
        "alignment": _serialize_word_alignment(updated),
    }


@app.post("/api/jobs/{job_id}/word-alignments/{token_id}/reset")
async def reset_word_alignment(job_id: str, token_id: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    """Explicitly remove selected locks; it does not start a new run."""

    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status == JobStatus.ALIGNING_WORDS:
        raise HTTPException(status_code=409, detail="Cannot reset word boundaries during alignment")
    if not job.result:
        raise HTTPException(status_code=400, detail="No alignment result")
    from processing.alignment import reset_word_boundaries

    body = body or {}
    boundary_values = body.get("boundaries", ["start", "end"])
    if boundary_values is None or not isinstance(boundary_values, list) or not all(isinstance(value, str) for value in boundary_values):
        raise HTTPException(status_code=422, detail="boundaries must be an array of strings")
    boundaries = set(boundary_values)
    unknown_boundaries = boundaries - {"start", "end"}
    if unknown_boundaries:
        raise HTTPException(status_code=422, detail="Boundaries must be start and/or end")
    alignments = list(job.result.word_alignments)
    index = next((i for i, item in enumerate(alignments) if item.token_id == token_id), -1)
    if index < 0:
        raise HTTPException(status_code=404, detail="Word token not found")
    alignments[index] = reset_word_boundaries(
        alignments[index],
        reset_start="start" in boundaries,
        reset_end="end" in boundaries,
    )
    job.result.word_alignments = alignments
    _refresh_legacy_result_from_canonical(job)
    _persist_jobs()
    return {
        "status": "updated",
        "alignment": _serialize_word_alignment(alignments[index]),
    }


@app.put("/api/jobs/{job_id}/notes")
async def update_notes(job_id: str, body: dict[str, Any]) -> dict[str, str]:
    """Update the notes/phrases and regenerate the UltraStar TXT."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.result:
        raise HTTPException(status_code=400, detail="No result to update")
    if job.status == JobStatus.ALIGNING_WORDS:
        raise HTTPException(status_code=409, detail="Cannot edit notes during alignment")

    from models import UltraStarNote, Phrase
    from processing.ultrastar import generate_ultrastar_txt

    phrase_values = body.get("phrases", [])
    if not isinstance(phrase_values, list):
        raise HTTPException(status_code=422, detail="phrases must be an array")
    phrases = []
    try:
        for p_data in phrase_values:
            if not isinstance(p_data, dict) or not isinstance(p_data.get("notes", []), list):
                raise ValueError("Each phrase must contain a notes array")
            end_beat = p_data.get("end_beat", 0)
            if isinstance(end_beat, bool) or not isinstance(end_beat, int) or end_beat < 0:
                raise ValueError("phrase end_beat must be a non-negative integer")
            notes = []
            for note_data in p_data.get("notes", []):
                if not isinstance(note_data, dict):
                    raise ValueError("Each note must be an object")
                note_type = note_data.get("note_type")
                text = note_data.get("text")
                start_beat = note_data.get("start_beat")
                duration_beats = note_data.get("duration_beats")
                pitch = note_data.get("pitch")
                if not isinstance(note_type, str) or not isinstance(text, str):
                    raise ValueError("note_type and text must be strings")
                integer_values = (start_beat, duration_beats, pitch)
                if any(isinstance(value, bool) or not isinstance(value, int) for value in integer_values):
                    raise ValueError("note timing and pitch values must be integers")
                if start_beat < 0 or duration_beats < 0:
                    raise ValueError("note timing values must be non-negative")
                notes.append(
                    UltraStarNote(
                        note_type=note_type,
                        start_beat=start_beat,
                        duration_beats=duration_beats,
                        pitch=pitch,
                        text=text,
                    )
                )
            phrases.append(Phrase(notes=notes, end_beat=end_beat))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid note payload: {exc}") from exc

    job.result.phrases = phrases
    job.result.note_overrides = _capture_note_overrides(phrases)
    job.result.ultrastar_txt = generate_ultrastar_txt(
        metadata=job.result.metadata,
        phrases=phrases,
        audio_filename=job.result.audio_file,
        vocals_filename=job.result.vocals_file,
        instrumental_filename=job.result.instrumental_file,
        video_filename=job.result.video_file,
    )
    active_variant = _alignment_variant_for_mode(job.result, job.result.current_alignment_mode)
    if active_variant is not None:
        active_variant.phrases = list(phrases)
        active_variant.ultrastar_txt = job.result.ultrastar_txt

    # Rewrite TXT file
    safe_name = _sanitize_filename(
        f"{job.result.metadata.artist} - {job.result.metadata.title}"
    ) or "song"
    txt_path = os.path.join(job.work_dir, f"{safe_name}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(job.result.ultrastar_txt)
    _publish_song_to_library(job)
    _persist_jobs()

    return {"status": "updated"}


@app.post("/api/jobs/{job_id}/regenerate-lyrics")
async def regenerate_lyrics(job_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """Compatibility endpoint retaining the historical synchronous result contract."""

    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.result or not job.result.lyric_revision:
        raise HTTPException(status_code=400, detail="No alignment result to regenerate")
    if job.status not in {JobStatus.DONE, JobStatus.ERROR}:
        raise HTTPException(status_code=409, detail="Another processing run is already active")
    if not job.vocals_path or not os.path.exists(job.vocals_path):
        raise HTTPException(status_code=400, detail="Vocals file not available for alignment")
    revision = job.result.lyric_revision
    initial_prompt_value = body.get("initial_prompt", "These are song lyrics with singing.")
    if not isinstance(initial_prompt_value, str):
        raise HTTPException(status_code=422, detail="initial_prompt must be a string")
    initial_prompt = initial_prompt_value.strip()
    if (
        initial_prompt
        and initial_prompt != "These are song lyrics with singing."
        and initial_prompt != revision.raw_text
    ):
        revision = _create_lyric_revision(
            job,
            initial_prompt,
            source=LyricSource.MANUAL,
            parent_revision_id=revision.id,
            language=_enum_value(revision.language),
        )
    from processing.lyrics import LyricsParams
    from processing.alignment import normalize_language

    try:
        requested_language = normalize_language(body.get("language") or _enum_value(revision.language))
    except (AttributeError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="language must be 'en' or 'pl'") from exc

    params = LyricsParams(
        model_size=body.get("model_size", "large-v3"),
        language=requested_language,
        vad_filter=body.get("vad_filter", False),
        vad_threshold=body.get("vad_threshold", 0.25),
        vad_min_silence_ms=body.get("vad_min_silence_ms", 200),
        speech_pad_ms=body.get("speech_pad_ms", 250),
        no_speech_threshold=body.get("no_speech_threshold", 0.4),
        beam_size=body.get("beam_size", 5),
        initial_prompt=initial_prompt or job.confirmed_lyrics or revision.raw_text,
        min_word_confidence=body.get("min_word_confidence", 0.15),
        audio_source=body.get("audio_source", "original"),
        normalize=body.get("normalize", True),
    )
    if params.audio_source not in {"original", "vocals"}:
        raise HTTPException(status_code=422, detail="audio_source must be original or vocals")
    source_path = job.vocals_path
    reset_values = body.get("reset_token_ids", [])
    if reset_values is None or not isinstance(reset_values, list) or not all(isinstance(token_id, str) for token_id in reset_values):
        raise HTTPException(status_code=422, detail="reset_token_ids must be an array of strings")
    unknown_resets = set(reset_values) - {token.id for token in revision.tokens}
    if unknown_resets:
        raise HTTPException(status_code=422, detail="Reset contains unknown token IDs")
    run = AlignmentRun(
        id=str(uuid.uuid4()),
        lyric_revision_id=revision.id,
        status=AlignmentRunStatus.RUNNING,
        selected_strategy="sofa-primary-global-match-and-candidate-score",
    )
    job.alignment_runs.append(run)
    job.selected_lyric_revision_id = revision.id
    _step(job, JobStatus.ALIGNING_WORDS, "Regenerating canonical lyric alignment...")
    await _execute_realignment(
        job,
        revision,
        run,
        set(reset_values),
        alignment_params=params,
        transcribe_path=source_path,
    )
    if _enum_value(run.status) != AlignmentRunStatus.COMPLETED.value or job.status != JobStatus.DONE:
        raise HTTPException(status_code=500, detail=job.error or "Lyrics regeneration failed")
    _persist_jobs()
    return _serialize_result(job)

async def _playback_response(source: Path, cache_dir: Path) -> FileResponse:
    audio = await asyncio.to_thread(playback_audio_path, source, cache_dir)
    media_type = "audio/wav" if audio.suffix.lower() == ".wav" else mimetypes.guess_type(audio.name)[0]
    return FileResponse(audio, media_type=media_type or "application/octet-stream")


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

    filepath = Path(_safe_child_path(job.work_dir, filename))
    if not filepath.is_file():
        raise HTTPException(status_code=404, detail="Audio file not found")

    # Reuse the PCM files created by the pipeline when they are still present.
    wav_name = {"original": "audio.wav", "vocals": "vocals.wav", "instrumental": "instrumental.wav"}[track]
    if filepath.suffix.lower() == ".mp3":
        wav = Path(_safe_child_path(job.work_dir, wav_name))
        if wav.is_file():
            filepath = wav
    return await _playback_response(filepath, Path(job.work_dir) / "playback-audio")


@app.get("/api/jobs/{job_id}/download")
async def download_zip(job_id: str, alignment_mode: str | None = None) -> FileResponse:
    """Download a ZIP containing TXT + all audio files. Optional alignment_mode uses that variant's TXT."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.result:
        raise HTTPException(status_code=400, detail="No result yet")
    if job.status == JobStatus.ALIGNING_WORDS:
        raise HTTPException(status_code=409, detail="A new alignment is still running; export is temporarily unavailable")
    if not job.result.lyric_revision or job.result.lyric_revision.id != job.selected_lyric_revision_id:
        raise HTTPException(status_code=409, detail="The selected lyric revision has no completed alignment to export")
    if (
        job.result.alignment_run is None
        or _enum_value(job.result.alignment_run.status) != AlignmentRunStatus.COMPLETED.value
    ):
        raise HTTPException(status_code=409, detail="The selected result has no completed alignment run to export")

    txt_content = job.result.ultrastar_txt
    if alignment_mode:
        ar = getattr(job.result, "alignment_results", {}) or {}
        if alignment_mode in ar:
            txt_content = ar[alignment_mode].ultrastar_txt

    safe_name = _sanitize_filename(
        f"{job.result.metadata.artist} - {job.result.metadata.title}"
    ) or "song"

    staging = tempfile.TemporaryDirectory(prefix="karaoke-export-")
    archive = Path(staging.name) / "export.zip"
    try:
        await asyncio.to_thread(
            rust_utility, "archive", chart=_chart_payload(job.result, content=txt_content),
            work=job.work_dir, destination=str(archive),
        )
    except BaseException:
        staging.cleanup()
        raise
    return FileResponse(
        archive, media_type="application/zip", filename=f"{safe_name}.zip",
        background=BackgroundTask(staging.cleanup),
    )


@app.get("/api/jobs")
async def list_jobs(request: Request) -> dict[str, Any]:
    """List songs known to this local API, newest first."""
    owner_id = _request_owner(request)
    items: list[dict[str, Any]] = []
    for job in jobs.values():
        if job.owner_id != owner_id:
            continue
        items.append({
            "id": job.id,
            "title": job.metadata.title or os.path.splitext(job.original_filename)[0] or "Untitled song",
            "artist": job.metadata.artist,
            "filename": job.original_filename,
            "status": job.status.value,
            "message": job.progress_message,
            "progress_percent": job.progress_percent,
            "error": job.error,
        })
    items.sort(key=lambda item: item["id"], reverse=True)
    return {"songs": items}


@app.get("/api/library/working-directory")
async def get_working_directory() -> dict[str, str | None]:
    return {"directory": _working_directory()}


@app.put("/api/library/working-directory")
async def set_working_directory(body: dict[str, Any]) -> dict[str, str | None]:
    directory = body.get("directory")
    if directory is not None and not isinstance(directory, str):
        raise HTTPException(status_code=422, detail="directory must be a path or null")
    if directory is not None:
        directory = directory.strip()
        if not directory:
            raise HTTPException(status_code=422, detail="directory must not be empty")
        candidate = Path(directory).expanduser()
        if not candidate.is_dir():
            raise HTTPException(status_code=422, detail="Working directory must be an existing folder")
        directory = str(candidate.resolve())
    _save_working_directory(directory)
    return {"directory": directory}


def _list_external_charts(directory: str | None, published_charts: set[Path]) -> list[dict[str, Any]]:
    if not directory or not os.path.isdir(directory):
        return []
    root = Path(directory).resolve()
    published = {chart.resolve() for chart in published_charts}
    return [
        song for song in rust_utility("scan_library", directory=str(root))["songs"]
        if (root / song["path"]).resolve() not in published
    ]


@app.get("/api/library")
async def list_library(request: Request, managed_only: bool = False) -> dict[str, Any]:
    owner_id = _request_owner(request)
    items: list[dict[str, Any]] = []
    queued_jobs = sorted(
        (
            job for job in jobs.values()
            if job.owner_id == owner_id and job.queued_at
        ),
        key=lambda job: job.queued_at,
    )
    queue_positions = {job.id: index for index, job in enumerate(queued_jobs, start=1)}
    published_charts: set[Path] = set()
    for job in jobs.values():
        if job.owner_id != owner_id:
            continue
        if job.library_path:
            published_charts.add(Path(job.library_path) / f"{_sanitize_filename(f'{job.metadata.artist} - {job.metadata.title}') or 'song'}.txt")
        items.append({
            "id": job.id,
            "kind": "managed",
            "title": job.metadata.title or os.path.splitext(job.original_filename)[0] or "Untitled song",
            "artist": job.metadata.artist,
            "filename": job.original_filename,
            "status": job.status.value,
            "message": job.progress_message,
            "progress_percent": job.progress_percent,
            "error": job.error,
            "failed_phase": job.failed_phase,
            "queue_position": queue_positions.get(job.id),
        })

    directory = _working_directory()
    if not managed_only:
        items.extend(await asyncio.to_thread(_list_external_charts, directory, published_charts))
    items.sort(key=lambda item: (item["kind"] != "managed", item["title"].casefold(), item["artist"].casefold()))
    if managed_only:
        root = Path(directory).resolve() if directory else None
        return {
            "working_directory": directory,
            "songs": items,
            "published_paths": sorted(
                chart.relative_to(root).as_posix()
                for chart in (path.resolve() for path in published_charts)
                if root is not None and chart.is_relative_to(root)
            ),
        }
    return {"working_directory": directory, "songs": items}


@app.get("/api/library/external-song")
async def get_external_song(path: str) -> dict[str, Any]:
    chart = _external_song_path(path)
    try:
        content = read_chart_text(chart)
    except OSError as exc:
        raise HTTPException(status_code=404, detail="Song file is unavailable") from exc
    try:
        parsed = parse_chart(content, chart.stem)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    phrases = parsed.pop("phrases")
    separation_status = _external_song_separation_status(chart)
    return {
        "path": path,
        "filename": chart.name,
        "content": content,
        "metadata": parsed["metadata"],
        "phrases": [asdict(phrase) for phrase in phrases],
        "audio_available": _external_song_audio_path(chart) is not None,
        "stems_available": separation_status["status"] == "done",
        "separation_status": separation_status["status"],
        "separation_error": separation_status.get("error", ""),
    }


@app.get("/api/library/external-song/audio")
async def get_external_song_audio(path: str, track: str = "original") -> FileResponse:
    chart = _external_song_path(path)
    if track == "original":
        audio = _external_song_audio_path(chart)
    elif track in {"vocals", "instrumental"}:
        _key, _directory, vocals, instrumental = _external_song_stem_paths(chart)
        audio = vocals if track == "vocals" else instrumental
    else:
        raise HTTPException(status_code=422, detail="Invalid audio track")
    if audio is None or not audio.is_file():
        raise HTTPException(status_code=404, detail="Audio file is unavailable")
    return await _playback_response(audio, Path(OUTPUT_DIR) / "playback-audio")


@app.post("/api/library/external-song/split")
async def start_external_song_separation(path: str) -> dict[str, str]:
    chart = _external_song_path(path)
    audio = _external_song_audio_path(chart)
    if audio is None:
        raise HTTPException(status_code=404, detail="Source audio is unavailable")
    key, output_directory, _vocals, _instrumental = _external_song_stem_paths(chart)
    if _external_song_stems_available(chart):
        return {"status": "done"}

    with _external_song_separation_lock:
        task = external_song_separation_tasks.get(key)
        if task is not None and not task.done():
            return {"status": "separating"}
        external_song_separation_states[key] = {"status": "separating"}
        external_song_separation_tasks[key] = asyncio.create_task(
            _run_external_song_separation(key, str(audio), output_directory),
        )
    return {"status": "separating"}


@app.get("/api/library/external-song/split-status")
async def get_external_song_separation_status(path: str) -> dict[str, str]:
    chart = _external_song_path(path)
    return _external_song_separation_status(chart)


@app.put("/api/library/external-song")
async def save_external_song(path: str, body: dict[str, Any]) -> dict[str, str]:
    chart = _external_song_path(path)
    if not isinstance(body.get("phrases"), list):
        raise HTTPException(status_code=422, detail="phrases must be an array")
    try:
        updated = await asyncio.to_thread(
            rust_utility, "save_external_chart", path=str(chart), phrases=body["phrases"],
        )
    except RustUtilityUnavailable:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid chart edit: {exc}") from exc
    return {"status": "updated", "content": updated}


@app.get("/api/library/external-song/download")
async def download_external_song(path: str) -> FileResponse:
    chart = _external_song_path(path)
    return FileResponse(chart, media_type="text/plain; charset=utf-8", filename=chart.name)


@app.delete("/api/library/song")
async def remove_library_song(
    request: Request,
    job_id: str | None = None,
    path: str | None = None,
) -> dict[str, str]:
    if bool(job_id) == bool(path):
        raise HTTPException(status_code=422, detail="Provide either job_id or path")
    if job_id:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Song not found")
        if job.status not in {
            JobStatus.PENDING,
            JobStatus.AWAITING_LYRICS,
            JobStatus.DONE,
            JobStatus.ERROR,
        }:
            raise HTTPException(status_code=409, detail="Cannot remove a song while it is processing")
        owner_id = _request_owner(request)
        if job.owner_id != owner_id:
            raise HTTPException(status_code=404, detail="Song not found")
        try:
            if Path(job_id).name != job_id or job_id in {".", ".."}:
                raise ValueError("Invalid managed song identifier")
            _remove_managed_directory(
                job.work_dir or str(Path(OUTPUT_DIR).resolve() / job_id),
                OUTPUT_DIR,
            )
            upload_directory = (
                str(Path(job.upload_path).parent)
                if job.upload_path
                else str(Path(UPLOAD_DIR).resolve() / job_id)
            )
            try:
                _remove_managed_directory(upload_directory, UPLOAD_DIR)
            except ValueError as exc:
                logging.getLogger("library").warning(
                    "Skipping upload cleanup for song %s because its path is not inside the managed upload directory: %s",
                    job_id,
                    exc,
                )
            if job.library_path:
                library_root = job.library_root or _working_directory() or ""
                _remove_managed_directory(job.library_path, library_root)
        except (OSError, ValueError) as exc:
            logging.getLogger("library").exception("Unable to remove all files for song %s", job_id)
            raise HTTPException(
                status_code=500,
                detail=f"Song record was kept because its files could not all be removed: {exc}",
            ) from exc
        with _state_lock:
            jobs.pop(job_id, None)
            _persist_jobs()
        return {"removed": job_id}

    assert path is not None
    chart = _external_song_path(path)
    directory = _working_directory()
    if not directory:
        raise HTTPException(status_code=409, detail="Working directory is not configured")
    root = Path(directory).resolve()
    try:
        await asyncio.to_thread(rust_utility, "remove_external_chart", root=str(root), path=str(chart))
    except (OSError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"removed": path}


@app.get("/api/jobs/{job_id}/status")
async def get_status(job_id: str) -> dict[str, Any]:
    """Get current job status."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    result: dict[str, Any] = {
        "status": job.status.value,
        "message": job.progress_message,
        "metadata": {
            "title": job.metadata.title,
            "artist": job.metadata.artist,
        },
    }
    if job.error:
        result["error"] = job.error
        result["failed_phase"] = job.failed_phase
    if job.status == JobStatus.AWAITING_LYRICS:
        result["found_lyrics"] = job.found_lyrics
    if job.status == JobStatus.DOWNLOADING:
        result["progress_percent"] = job.progress_percent
    if job.alignment_runs:
        result["alignment_run"] = _serialize_alignment_run(job.alignment_runs[-1])
    if job.selected_lyric_revision_id:
        result["lyric_revision_id"] = job.selected_lyric_revision_id
    if job.result and job.result.alignment_comparison:
        result["alignment_comparison_status"] = job.result.alignment_comparison.get("status")
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
        results = await asyncio.to_thread(
            rust_utility, "search_youtube", backend=str(Path(__file__).resolve().parent),
            query=req.query,
        )
    except (RuntimeError, OSError) as exc:
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}") from exc
    return {"results": results}


class YouTubeDownloadRequest(BaseModel):
    url: str
    title: str
    artist: str
    language: str
    genre: str
    year: str


@app.post("/api/youtube/download")
async def download_youtube(request: Request, req: YouTubeDownloadRequest) -> dict[str, str]:
    """Download a YouTube video and create a processing job."""
    from processing.alignment import normalize_language

    try:
        language = normalize_language(req.language)
    except (AttributeError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="language must be 'en' or 'pl'") from exc
    job_id = str(uuid.uuid4())
    
    # Create job immediately with DOWNLOADING status
    job = Job(
        id=job_id,
        owner_id=_request_owner(request),
        original_filename="",
        upload_path="",
        source_url=req.url,
        status=JobStatus.DOWNLOADING,
        progress_message="Starting download...",
        metadata=SongMetadata(
            title=req.title,
            artist=req.artist,
            language=language,
            genre=req.genre,
            year=req.year,
        ),
    )
    jobs[job_id] = job
    _persist_jobs()
    
    # Start download in background
    asyncio.create_task(_download_youtube_video(job_id, req))
    
    return {"job_id": job_id}


async def _download_youtube_video(job_id: str, req: YouTubeDownloadRequest) -> None:
    """Download YouTube video with progress updates."""
    job = jobs.get(job_id)
    if not job:
        return

    def progress(event: dict[str, Any]) -> None:
        percent = float(event["progress_percent"])
        if percent > job.progress_percent:
            job.progress_percent = percent
            job.progress_message = f"Downloading... {percent:.1f}%"
            _notify(job)

    try:
        media = await call_with_progress(
            "download_youtube", progress, backend=str(Path(__file__).resolve().parent), job_id=job_id, url=req.url,
        )
        job.original_filename = media["original_filename"]
        job.upload_path = media["upload_path"]
        job.audio_path = media["audio_path"]
        job.work_dir = media["work_dir"]
        job.has_video = media["has_video"]
        job.metadata.title = req.title or media["title"]
        job.metadata.artist = req.artist or media["artist"]
        job.status = JobStatus.PENDING
        job.failed_phase = ""
        job.progress_message = "Download complete"
        _persist_jobs()
        _notify(job)
    except Exception as exc:
        logging.getLogger("pipeline").exception("YouTube download failed")
        job.status = JobStatus.ERROR
        job.failed_phase = "download"
        job.error = str(exc)
        job.progress_message = f"Download failed: {exc}"
        _persist_jobs()
        _notify(job)
