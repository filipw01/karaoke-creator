"""Browser API and managed workflow tests against the Rust server."""

from __future__ import annotations

import io
import json
import math
import os
import queue
import shutil
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import wave
import zipfile
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import pytest

backend = Path(__file__).resolve().parents[1]
if str(backend) not in sys.path:
    sys.path.insert(0, str(backend))

from processing.rust_utility import RustUtilityUnavailable, executable_path

OWNER = {"X-Karaoke-Owner": "test-owner"}

WORKER = """
import json
import sys
for line in sys.stdin:
    payload = json.loads(line)
    assert payload["operation"] == "managed_alignment"
    request = payload["request"]["request"]
    alignments = [
        {
            "token_id": token["id"], "alignment_run_id": request["alignment_run_id"],
            "start": 0.15 + index * 0.30, "end": 0.30 + index * 0.30,
            "status": "aligned", "confidence": 0.8,
            "evidence": {"selected_model": "test", "selected_audio": "vocal_stem",
                         "reasons": [], "candidate_scores": []},
            "start_locked": False, "end_locked": False,
        }
        for index, token in enumerate(request["tokens"])
    ]
    print(json.dumps({"version": 1, "result": {
        "alignments": alignments, "pitch_source_alignments": alignments,
        "alignment_backend": request["alignment_backend"], "diagnostics": {"worker": "test"},
    }}), flush=True)
"""


@contextmanager
def server(root: Path, worker_only: bool = False, **runtime: str):
    try:
        executable = executable_path()
    except RustUtilityUnavailable:
        pytest.skip("Requires the Rust backend executable")
    environment = os.environ.copy()
    for key in (
        "KARAOKE_API_TOKEN", "ALIGNMENT_WORKER_URL", "ALIGNMENT_WORKER_TOKEN",
        "ALIGNMENT_WORKER_AUDIO_ROOT", "KARAOKE_MAX_UPLOAD_BYTES",
    ):
        environment.pop(key, None)
    environment.update(
        KARAOKE_PYTHON=sys.executable,
        PITCH_DETECTOR="pyin",
        ALIGNMENT_PRIMARY_BACKEND="legacy",
    )
    environment.update(runtime)
    flag = "--serve-alignment-worker" if worker_only else "--serve-api"
    with subprocess.Popen(
        [executable, flag, str(root), "127.0.0.1:0"],
        stderr=subprocess.PIPE, text=True, env=environment,
    ) as process:
        startup: queue.Queue[str] = queue.Queue()
        diagnostics: list[str] = []

        def drain() -> None:
            assert process.stderr is not None
            for line in process.stderr:
                diagnostics.append(line)
                if line.startswith("Karaoke API listening on "):
                    startup.put("http://" + line.split()[-1])
            startup.put("")

        reader = threading.Thread(target=drain, daemon=True)
        reader.start()
        try:
            address = startup.get(timeout=15)
            assert address, "".join(diagnostics)

            def request(path, body=None, *, method=None, headers=None, raw=False):
                headers = dict(headers or {})
                if body is not None and not isinstance(body, bytes):
                    headers["Content-Type"] = "application/json"
                    body = json.dumps(body).encode()
                request = urllib.request.Request(address + path, data=body, headers=headers, method=method)
                try:
                    with urllib.request.urlopen(request, timeout=15) as response:
                        content = response.read()
                        return response.status, content if raw else json.loads(content)
                except urllib.error.HTTPError as error:
                    with error:
                        content = error.read()
                    return error.code, json.loads(content) if content else None

            request.address = address
            yield request
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            reader.join(timeout=5)


def write_audio(path: Path) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setparams((2, 2, 44100, 0, "NONE", "not compressed"))
        frames = (
            round(8192 * math.sin(2 * math.pi * 220 * index / 44100))
            for index in range(44100)
        )
        audio.writeframes(b"".join(struct.pack("<hh", value, value) for value in frames))


def await_result(request, job_id):
    cutoff = time.monotonic() + 20
    while time.monotonic() < cutoff:
        code, state = request(f"/api/jobs/{job_id}/status", headers=OWNER)
        assert code == 200, state
        assert state["status"] != "error", state
        if state["status"] == "done":
            code, result = request(f"/api/jobs/{job_id}/result", headers=OWNER)
            assert code == 200, result
            return result
        time.sleep(0.05)
    pytest.fail("Timed out waiting for the managed alignment result")


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="Managed output requires FFmpeg")
def test_import_alignment_edit_export_restart_and_delete(tmp_path: Path):
    (tmp_path / "desktop_worker.py").write_text(WORKER, encoding="utf-8")
    job_id = str(uuid4())
    upload = tmp_path / "uploads" / job_id
    work = tmp_path / "outputs" / job_id
    songs = tmp_path / "songs"
    for directory in (upload, work, songs):
        directory.mkdir(parents=True)
    (upload / "input").write_bytes(b"original media")
    (upload / "prepared-media.json").write_text(
        json.dumps({"original_filename": "Artist - Song.mp3", "has_video": False}),
    )
    write_audio(work / "audio.wav")
    for name in ("vocals.wav", "instrumental.wav"):
        shutil.copyfile(work / "audio.wav", work / name)

    jobs = f"/api/jobs/{job_id}"
    with server(tmp_path) as request:
        assert request("/api/library/working-directory", {"directory": str(songs)}, method="PUT")[0] == 200
        registration = {"import_id": job_id, "title": "Song", "artist": "Artist", "language": "en"}
        assert request("/api/desktop/import", registration, headers=OWNER)[0] == 404
        (upload / "desktop-preparation.json").write_text(json.dumps({
            "status": "awaiting_lyrics", "found_lyrics": "hello world", "real_bpm": 120,
        }))
        for _ in range(2):
            assert request("/api/desktop/import", registration, headers=OWNER) == (200, {"job_id": job_id})
        assert request("/api/desktop/import", registration)[0] == 404
        assert request(f"{jobs}/found-lyrics", headers=OWNER) == (200, {"lyrics": "hello world"})
        assert request(f"{jobs}/status")[0] == 404
        assert request(f"{jobs}/confirm-lyrics", {"lyrics": "hello world"}, headers=OWNER) == (200, {"status": "started"})

        result = await_result(request, job_id)
        assert result["export_ready"]
        assert result["lyric_revision"]["source"] == "online"
        assert len(result["word_alignments"]) == 2
        assert [word["text"] for word in result["words"]] == ["hello", "world"]
        assert "hello" in result["ultrastar_txt"]
        assert (songs / "Artist - Song" / "Artist - Song.txt").is_file()
        code, listed = request("/api/library?managed_only=true", headers=OWNER)
        assert code == 200
        assert listed["published_paths"] == ["Artist - Song/Artist - Song.txt"]

        notes = {
            "phrases": [{
                "notes": [{"note_type": ":", "text": " edited", "start_beat": 0, "duration_beats": 4, "pitch": 9}],
                "end_beat": 5,
            }],
        }
        assert request(f"{jobs}/notes", notes, headers=OWNER, method="PUT")[0] == 200
        token = result["word_alignments"][0]["token_id"]
        boundary = f"{jobs}/word-alignments/{token}"
        code, edited = request(boundary, {"start": 0.13}, headers=OWNER, method="PATCH")
        assert code == 200 and edited["alignment"]["start_locked"]
        code, reset = request(f"{boundary}/reset", {"boundaries": ["start"]}, headers=OWNER)
        assert code == 200 and not reset["alignment"]["start_locked"]
        assert request(f"{jobs}/audio/vocals", headers={**OWNER, "Range": "bytes=0-3"}, raw=True) == (206, b"RIFF")

        code, archive = request(f"{jobs}/download", headers=OWNER, raw=True)
        assert code == 200
        with zipfile.ZipFile(io.BytesIO(archive)) as file:
            assert b" edited" in file.read("Artist - Song/Artist - Song.txt")

        code, saved = request(f"{jobs}/lyric-revisions", {"raw_text": "Cześć -- świecie", "language": "pl"}, headers=OWNER)
        assert code == 201
        revision = saved["revision"]
        assert [word["source_index"] for word in revision["tokens"]] == [0, 2]
        assert request(f"{jobs}/download", headers=OWNER)[0] == 409
        assert request(f"{jobs}/alignment-runs", {"lyric_revision_id": revision["id"], "reset_token_ids": None}, headers=OWNER)[0] == 422
        code, run = request(f"{jobs}/alignment-runs", {"lyric_revision_id": revision["id"]}, headers=OWNER)
        assert code == 202 and run["status"] == "running"
        result = await_result(request, job_id)
        assert result["export_ready"]
        assert result["lyric_revision"]["language"] == "pl"
        assert result["note_overrides"]["note:0:0"]["pitch"] == 9

    with server(tmp_path, KARAOKE_PYTHON="nonexistent-worker") as request:
        code, restored = request(f"{jobs}/result", headers=OWNER)
        assert code == 200 and restored == result
        assert request(f"/api/library/song?job_id={job_id}", method="DELETE")[0] == 404
        assert request(f"/api/library/song?job_id={job_id}", method="DELETE", headers=OWNER) == (200, {"removed": job_id})
        assert not work.exists()
        assert not upload.exists()
        assert not (songs / "Artist - Song").exists()


def test_rust_ingress_requires_tokens_and_bounds_uploads(tmp_path: Path):
    with server(tmp_path, KARAOKE_PYTHON="nonexistent-worker", KARAOKE_API_TOKEN="demo", KARAOKE_MAX_UPLOAD_BYTES="4") as request:
        assert request("/api/jobs")[0] == 401
        assert request("/api/jobs?access_token=demo") == (200, {"songs": []})
        assert request("/v1/alignments", {})[0] == 503
        upload = b'--test\r\nContent-Disposition: form-data; name="file"; filename="Song.mp3"\r\n\r\naudio\r\n--test--\r\n'
        headers = {"Authorization": "Bearer demo", "Content-Type": "multipart/form-data; boundary=test"}
        assert request("/api/upload", upload, headers=headers)[0] == 413
        assert request("/api/jobs?access_token=demo") == (200, {"songs": []})
        assert not list((tmp_path / "uploads").iterdir())


def test_benchmark_import_and_archive(tmp_path: Path):
    cache = tmp_path / "benchmarks"
    cache.mkdir()
    chart = cache / "Song.txt"
    chart.write_text(
        "#TITLE:Song\n#ARTIST:Artist\n#BPM:480\n#MP3:Song.wav\n"
        ": 0 4 3 Hel\n: 4 4 3 lo\n: 8 4 4  world\nE\n",
    )
    audio = cache / "Song.wav"
    write_audio(audio)
    with server(tmp_path, KARAOKE_PYTHON="nonexistent-worker") as request:
        data = {"chart_path": str(chart), "audio_path": str(audio), "vocal_path": str(audio)}
        code, imported = request("/api/benchmark/import", data, headers=OWNER)
        assert code == 201, imported
        job_id = imported["job_id"]
        assert imported["result"]["lyric_revision"]["raw_text"] == "Hello world"
        assert [word["midi_note"] for word in imported["result"]["words"]] == [63, 64]
        assert request(f"/api/jobs/{job_id}/download", headers=OWNER, raw=True)[0] == 200
        assert request(f"/api/library/song?job_id={job_id}", method="DELETE", headers=OWNER)[0] == 200
        assert chart.is_file() and audio.is_file()


def test_alignment_worker_deployment_is_independent_of_the_browser_api(tmp_path: Path):
    with server(tmp_path, worker_only=True, ALIGNMENT_WORKER_TOKEN="demo") as request:
        assert request("/api/jobs")[0] == 404
        assert request("/v1/alignments", {})[0] == 401
        assert not (tmp_path / "outputs").exists()


def test_worker_alignment_accepts_only_authenticated_contained_audio(tmp_path: Path):
    audio = tmp_path / "audio"
    audio.mkdir()
    vocal = audio / "vocal.wav"
    original = audio / "mix.wav"
    outside = tmp_path / "outside.wav"
    for path in (vocal, original, outside):
        path.touch()
    (tmp_path / "desktop_worker.py").write_text("""
import json
import sys
for line in sys.stdin:
    request = json.loads(line)["request"]
    response = {key: request[key] for key in ("job_id", "alignment_run_id", "lyric_revision_id")}
    response.update(status="completed", alignments=[], diagnostics={"checked": True}, draft_text=None)
    print(json.dumps({"version": 1, "result": {"response": response}}), flush=True)
""")
    with server(tmp_path, worker_only=True, ALIGNMENT_WORKER_TOKEN="demo", ALIGNMENT_WORKER_AUDIO_ROOT=str(audio)) as request:
        headers = {"X-Alignment-Worker-Token": "demo"}
        body = {
            "job_id": "job", "alignment_run_id": "run", "lyric_revision_id": "revision",
            "original_audio_ref": str(original), "vocal_stem_audio_ref": str(vocal),
            "language": "en", "tokens": [{"id": "revision:token:0"}],
        }
        assert request("/v1/alignments", body)[0] == 401
        assert request("/v1/alignments", {**body, "original_audio_ref": str(outside)}, headers=headers)[0] == 422
        code, aligned = request("/v1/alignments", body, headers=headers)
        assert code == 202 and aligned["diagnostics"]["checked"]
        assert aligned["job_id"] == "job" and aligned["alignment_run_id"] == "run"


def test_retry_publish_uses_saved_result_without_starting_inference(tmp_path: Path):
    output = tmp_path / "outputs"
    work = output / "job"
    work.mkdir(parents=True)
    songs = tmp_path / "songs"
    songs.mkdir()
    write_audio(work / "audio.wav")
    (output / "working-directory.json").write_text(json.dumps({"directory": str(songs)}))
    metadata = {"artist": "Artist", "title": "Song", "language": "en", "genre": "", "year": "", "bpm": 120, "gap_ms": 0}
    chart = "#TITLE:Song\n#ARTIST:Artist\n#MP3:audio.wav\n#BPM:480\n: 0 4 0 Hello\nE\n"
    run = {"id": "run", "status": "completed", "selected_strategy": "legacy"}
    (output / "jobs.json").write_text(json.dumps({
        "job": {
            "id": "job", "owner_id": OWNER["X-Karaoke-Owner"], "work_dir": str(work), "status": "error",
            "error": "permission denied", "failed_phase": "publish", "metadata": metadata,
            "selected_lyric_revision_id": "revision",
            "result": {"metadata": metadata, "audio_file": "audio.wav", "ultrastar_txt": chart,
                       "lyric_revision": {"id": "revision", "language": "en"}, "alignment_run": run},
        },
    }))
    with server(tmp_path, KARAOKE_PYTHON="nonexistent-worker") as request:
        assert request("/api/jobs/job/retry", method="POST", headers=OWNER) == (200, {"status": "retrying", "phase": "publish"})
        result = await_result(request, "job")
        assert result["ultrastar_txt"] == chart
        assert result["alignment_run"]["id"] == "run"
        assert (songs / "Artist - Song" / "Artist - Song.txt").read_text() == chart
