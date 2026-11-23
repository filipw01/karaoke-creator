"""API lifecycle and compatibility contract tests."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import pytest
from fastapi import HTTPException
from starlette.requests import Request

backend = Path(__file__).resolve().parent.parent
if str(backend) not in sys.path:
    sys.path.insert(0, str(backend))

import main
from models import (
    AlignmentEvidence,
    AlignmentRun,
    AlignmentRunStatus,
    AlignmentStatus,
    Job,
    JobResult,
    JobStatus,
    LyricRevision,
    LyricSource,
    SongMetadata,
    Word,
    WordAlignment,
)
from processing.alignment import edit_word_boundary, merge_alignment_locks, tokenize_lyrics
from processing.alignment_worker import AlignmentWorkerResponse
from processing.metadata import clean_music_title


@pytest.fixture(autouse=True)
def prevent_contract_tests_from_persisting_jobs(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(main, "_persist_jobs", lambda: None)


def test_clean_music_title_removes_video_noise():
    assert clean_music_title("Numb (Official Video)") == "Numb"
    assert clean_music_title("Dynamite Official MV") == "Dynamite"
    assert clean_music_title("Hello - Official Lyric Video") == "Hello"


def _job(tmp_path: Path) -> Job:
    revision = LyricRevision(
        id="revision",
        song_id="job",
        language="en",
        source="manual",
        raw_text="hello world",
    )
    revision.tokens = tokenize_lyrics(revision.raw_text, revision.id, "en")
    vocals = tmp_path / "vocals.wav"
    vocals.touch()
    return Job(
        id="job",
        status=JobStatus.DONE,
        result=JobResult(lyric_revision=revision),
        lyric_revisions=[revision],
        selected_lyric_revision_id=revision.id,
        vocals_path=str(vocals),
    )


def _awaiting_lyrics_job() -> Job:
    return Job(
        id="job",
        status=JobStatus.AWAITING_LYRICS,
        found_lyrics="wrong lyrics",
        metadata=SongMetadata(title="Song", artist="Artist", language="en"),
    )


def test_library_reports_queue_positions_in_fifo_order(monkeypatch: pytest.MonkeyPatch):
    main.jobs.clear()
    main.jobs["later"] = Job(
        id="later",
        status=JobStatus.EXTRACTING_AUDIO,
        queued_at="2025-01-02T00:00:00+00:00",
    )
    main.jobs["sooner"] = Job(
        id="sooner",
        status=JobStatus.EXTRACTING_AUDIO,
        queued_at="2025-01-01T00:00:00+00:00",
    )
    monkeypatch.setattr(main, "_working_directory", lambda: None)
    request = Request({
        "type": "http",
        "method": "GET",
        "path": "/api/library",
        "headers": [],
        "query_string": b"",
        "client": ("127.0.0.1", 8000),
        "server": ("127.0.0.1", 8000),
        "scheme": "http",
    })

    response = asyncio.run(main.list_library(request))
    positions = {song["id"]: song["queue_position"] for song in response["songs"]}

    assert positions == {"later": 2, "sooner": 1}


def test_completed_song_is_published_into_selected_library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    library = tmp_path / "songs"
    library.mkdir()
    work_dir = tmp_path / "job-output"
    work_dir.mkdir()
    audio = work_dir / "Artist - Song.mp3"
    audio.write_bytes(b"audio")
    job = Job(
        id="job",
        status=JobStatus.DONE,
        work_dir=str(work_dir),
        metadata=SongMetadata(title="Song", artist="Artist"),
        result=JobResult(
            metadata=SongMetadata(title="Song", artist="Artist"),
            audio_file=audio.name,
            ultrastar_txt="#TITLE:Song\n#ARTIST:Artist\n",
        ),
    )
    monkeypatch.setattr(main, "_working_directory", lambda: str(library))

    main._publish_song_to_library(job)

    chart = library / "Artist - Song" / "Artist - Song.txt"
    assert chart.read_text(encoding="utf-8") == job.result.ultrastar_txt
    assert (chart.parent / audio.name).read_bytes() == b"audio"
    assert job.library_path == str(chart.parent)

    main.jobs.clear()
    main.jobs[job.id] = job
    request = Request({
        "type": "http",
        "method": "GET",
        "path": "/api/library",
        "headers": [],
        "query_string": b"",
        "client": ("127.0.0.1", 8000),
        "server": ("127.0.0.1", 8000),
        "scheme": "http",
    })
    response = asyncio.run(main.list_library(request))
    assert [song["id"] for song in response["songs"]] == ["job"]


def test_managed_playback_uses_existing_pcm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    export = tmp_path / "Song [VOC].mp3"
    export.write_bytes(b"export")
    wav = tmp_path / "vocals.wav"
    wav.write_bytes(b"RIFF" + b"\0" * 40)
    job = Job(id="job", work_dir=str(tmp_path), result=JobResult(vocals_file=export.name))
    monkeypatch.setattr(main, "jobs", {job.id: job})

    response = asyncio.run(main.get_audio(job.id, "vocals"))

    assert response.media_type == "audio/wav"
    assert response.path == wav
    assert export.read_bytes() == b"export"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="MP3 playback conversion requires ffmpeg")
def test_external_mp3_playback_caches_a_fresh_pcm_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    library = tmp_path / "songs"
    library.mkdir()
    source = library / "Song.mp3"
    (library / "Song.txt").write_text("#MP3:Song.mp3\n")
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(main, "OUTPUT_DIR", str(outputs))
    monkeypatch.setattr(main, "_working_directory", lambda: str(library))

    def write_source(duration: float) -> bytes:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"sine=duration={duration}", str(source)],
            check=True,
        )
        return source.read_bytes()

    for duration, sample_count in ((0.25, 11025), (0.5, 22050)):
        mp3 = write_source(duration)
        response = asyncio.run(main.get_external_song_audio("Song.txt"))
        assert response.media_type == "audio/wav"
        assert source.read_bytes() == mp3
        with wave.open(str(response.path)) as audio:
            assert audio.getframerate() == 44100
            assert audio.getnframes() == sample_count
        cached = asyncio.run(main.get_external_song_audio("Song.txt"))
        assert cached.path == response.path

    assert len(list((outputs / "playback-audio").glob("*.wav"))) == 2


def test_removing_managed_song_deletes_upload_output_and_published_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    uploads = tmp_path / "uploads"
    outputs = tmp_path / "outputs"
    library = tmp_path / "selected-songs"
    different_current_directory = tmp_path / "new-current-songs"
    upload_dir = uploads / "job"
    output_dir = outputs / "job"
    published_dir = library / "Artist - Song"
    for directory in (upload_dir, output_dir, published_dir, different_current_directory):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "asset.bin").write_bytes(b"song")
    job = Job(
        id="job",
        status=JobStatus.DONE,
        upload_path=str(upload_dir / "input.mp3"),
        work_dir=str(output_dir),
        library_path=str(published_dir),
        library_root=str(library),
    )
    monkeypatch.setattr(main, "UPLOAD_DIR", str(uploads))
    monkeypatch.setattr(main, "OUTPUT_DIR", str(outputs))
    monkeypatch.setattr(main, "_working_directory", lambda: str(different_current_directory))
    monkeypatch.setattr(main, "_persist_jobs", lambda: None)
    monkeypatch.setattr(main, "jobs", {"job": job})
    request = Request({
        "type": "http",
        "method": "DELETE",
        "path": "/api/library/song",
        "headers": [],
        "query_string": b"",
        "client": ("127.0.0.1", 8000),
        "server": ("127.0.0.1", 8000),
        "scheme": "http",
    })

    response = asyncio.run(main.remove_library_song(request, job_id="job", path=None))

    assert response == {"removed": "job"}
    assert not upload_dir.exists()
    assert not output_dir.exists()
    assert not published_dir.exists()
    assert "job" not in main.jobs


def test_managed_directory_removal_rejects_paths_outside_storage_root(tmp_path: Path):
    root = tmp_path / "owned"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    marker = outside / "keep.txt"
    marker.write_text("keep")

    with pytest.raises(ValueError, match="directly inside"):
        main._remove_managed_directory(str(outside), str(root))

    assert marker.exists()


def test_removing_stale_test_job_skips_upload_path_outside_managed_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    uploads = tmp_path / "uploads"
    outputs = tmp_path / "outputs"
    output_dir = outputs / "job"
    external_upload = tmp_path / "pytest-temp" / "input.mp3"
    output_dir.mkdir(parents=True)
    external_upload.parent.mkdir()
    external_upload.touch()
    job = Job(
        id="job",
        status=JobStatus.ERROR,
        upload_path=str(external_upload),
        work_dir=str(output_dir),
    )
    monkeypatch.setattr(main, "UPLOAD_DIR", str(uploads))
    monkeypatch.setattr(main, "OUTPUT_DIR", str(outputs))
    monkeypatch.setattr(main, "_working_directory", lambda: None)
    monkeypatch.setattr(main, "jobs", {"job": job})
    request = Request({
        "type": "http",
        "method": "DELETE",
        "path": "/api/library/song",
        "headers": [],
        "query_string": b"",
        "client": ("127.0.0.1", 8000),
        "server": ("127.0.0.1", 8000),
        "scheme": "http",
    })

    response = asyncio.run(main.remove_library_song(request, job_id="job", path=None))

    assert response == {"removed": "job"}
    assert not output_dir.exists()
    assert external_upload.exists()
    assert "job" not in main.jobs


def test_removing_external_song_deletes_its_subdirectory_and_contents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "songs"
    song_directory = root / "Artist" / "Song"
    song_directory.mkdir(parents=True)
    (song_directory / "Song.txt").write_text("#TITLE:Song\n")
    (song_directory / "Song.mp3").write_bytes(b"audio")
    monkeypatch.setattr(main, "_working_directory", lambda: str(root))
    request = Request({
        "type": "http",
        "method": "DELETE",
        "path": "/api/library/song",
        "headers": [],
        "query_string": b"",
        "client": ("127.0.0.1", 8000),
        "server": ("127.0.0.1", 8000),
        "scheme": "http",
    })

    response = asyncio.run(
        main.remove_library_song(request, job_id=None, path="Artist/Song/Song.txt"),
    )

    assert response == {"removed": "Artist/Song/Song.txt"}
    assert not song_directory.exists()
    assert (root / "Artist").is_dir()
    assert root.is_dir()


def test_removing_external_song_at_library_root_preserves_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "songs"
    root.mkdir()
    chart = root / "Song.txt"
    chart.write_text("#TITLE:Song\n")
    monkeypatch.setattr(main, "_working_directory", lambda: str(root))
    request = Request({
        "type": "http",
        "method": "DELETE",
        "path": "/api/library/song",
        "headers": [],
        "query_string": b"",
        "client": ("127.0.0.1", 8000),
        "server": ("127.0.0.1", 8000),
        "scheme": "http",
    })

    response = asyncio.run(main.remove_library_song(request, job_id=None, path="Song.txt"))

    assert response == {"removed": "Song.txt"}
    assert not chart.exists()
    assert root.is_dir()


def test_previous_result_remains_readable_after_failed_run(tmp_path: Path):
    job = _job(tmp_path)
    job.status = JobStatus.ERROR
    main.jobs.clear()
    main.jobs[job.id] = job

    result = asyncio.run(main.get_result(job.id))

    assert result["lyric_revision"]["id"] == "revision"


def test_realign_rejects_null_reset_token_payload(tmp_path: Path):
    job = _job(tmp_path)
    main.jobs.clear()
    main.jobs[job.id] = job

    with pytest.raises(HTTPException) as error:
        asyncio.run(
            main.start_alignment_run(
                job.id,
                {"lyric_revision_id": "revision", "reset_token_ids": None},
            )
        )

    assert error.value.status_code == 422


def test_save_revision_rejects_non_string_text(tmp_path: Path):
    job = _job(tmp_path)
    main.jobs.clear()
    main.jobs[job.id] = job

    with pytest.raises(HTTPException) as error:
        asyncio.run(main.save_lyric_revision(job.id, {"raw_text": None}))

    assert error.value.status_code == 422


def test_worker_ingress_is_fail_closed_without_configured_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ALIGNMENT_WORKER_TOKEN", raising=False)
    body = {
        "job_id": "job",
        "alignment_run_id": "run",
        "lyric_revision_id": "revision",
        "language": "en",
        "original_audio_ref": "/audio/mix.wav",
        "vocal_stem_audio_ref": "/audio/vocals.wav",
        "tokens": [],
    }

    with pytest.raises(HTTPException) as error:
        asyncio.run(main.request_worker_alignment(body, None))

    assert error.value.status_code == 503


def test_alignment_run_is_registered_before_async_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    job = _job(tmp_path)
    main.jobs.clear()
    main.jobs[job.id] = job
    scheduled: list[object] = []

    async def fake_execute(*_: object) -> None:
        return None

    monkeypatch.setattr(main, "_execute_realignment", fake_execute)
    monkeypatch.setattr(asyncio, "create_task", lambda coroutine: scheduled.append(coroutine))

    response = asyncio.run(
        main.start_alignment_run(job.id, {"lyric_revision_id": "revision", "reset_token_ids": []})
    )

    assert response["status"] == "running"
    assert len(job.alignment_runs) == 1
    assert job.alignment_runs[0].status == AlignmentRunStatus.RUNNING
    assert scheduled
    scheduled[0].close()


def test_canonical_result_keeps_one_word_per_token_separate_from_pitch_segments(tmp_path: Path):
    job = _job(tmp_path)
    job.real_bpm = 120.0
    revision = job.result.lyric_revision
    assert revision is not None
    run = job.alignment_runs[0] if job.alignment_runs else None
    if run is None:
        from models import AlignmentRun

        run = AlignmentRun("run", revision.id, "completed", "test")
    canonical = [
        Word(text="hello", start=1.0, end=2.0, token_id=revision.tokens[0].id),
        Word(text="world", start=2.0, end=3.0, token_id=revision.tokens[1].id),
    ]
    pitch = [
        Word(text="hel", start=1.0, end=1.5, token_id=revision.tokens[0].id),
        Word(text="lo", start=1.5, end=2.0, token_id=revision.tokens[0].id),
    ]
    alignments = [
        WordAlignment(token.id, run.id, word.start, word.end, AlignmentStatus.ALIGNED, 1.0, AlignmentEvidence())
        for token, word in zip(revision.tokens, canonical)
    ]

    main._apply_canonical_result(job, revision, run, alignments, canonical, pitch, [])

    assert [word.text for word in job.result.words] == ["hello", "world"]
    assert [word.text for word in job.result.pitch_words] == ["hel", "lo"]
    assert len(job.result.alignment_results["canonical"].words) == 2


def test_boundary_refresh_rebuilds_export_layer_from_canonical_spans(tmp_path: Path):
    job = _job(tmp_path)
    job.real_bpm = 120.0
    revision = job.result.lyric_revision
    assert revision is not None
    run = AlignmentRun("run", revision.id, AlignmentRunStatus.COMPLETED, "test")
    job.result.alignment_run = run
    job.result.word_alignments = [
        WordAlignment(
            token_id=token.id,
            alignment_run_id=run.id,
            start=float(index + 2),
            end=float(index + 2.5),
            status=AlignmentStatus.ESTIMATED,
            confidence=0.3,
            evidence=AlignmentEvidence(),
        )
        for index, token in enumerate(revision.tokens)
    ]

    main._refresh_legacy_result_from_canonical(job)

    assert [word.start for word in job.result.words] == [2.0, 3.0]
    assert [word.start for word in job.result.alignment_results["canonical"].words] == [2.0, 3.0]


def test_boundary_refresh_keeps_persisted_pitch_source_spans_separate(tmp_path: Path):
    job = _job(tmp_path)
    job.real_bpm = 120.0
    assert job.result is not None
    revision = job.result.lyric_revision
    assert revision is not None
    run = AlignmentRun("run", revision.id, AlignmentRunStatus.COMPLETED, "test")
    job.result.alignment_run = run
    job.result.word_alignments = [
        WordAlignment(
            token_id=token.id,
            alignment_run_id=run.id,
            start=float(index * 2),
            end=float(index * 2 + 2),
            status=AlignmentStatus.ESTIMATED,
            confidence=0.3,
            evidence=AlignmentEvidence(),
        )
        for index, token in enumerate(revision.tokens)
    ]
    job.result.pitch_words = [
        Word(
            text=token.display_text,
            start=float(index),
            end=float(index + 0.5),
            pitch_hz=440.0,
            midi_note=69,
            token_id=token.id,
        )
        for index, token in enumerate(revision.tokens)
    ]

    main._refresh_legacy_result_from_canonical(job)

    assert [(word.start, word.end) for word in job.result.words] == [(0.0, 2.0), (2.0, 4.0)]
    assert [(word.start, word.end) for word in job.result.pitch_words] == [(0.0, 0.5), (1.0, 1.5)]


def test_pitch_source_keeps_repaired_companion_boundary_from_partial_lock():
    source = WordAlignment(
        "t",
        "canonical",
        0.8,
        1.0,
        AlignmentStatus.ALIGNED,
        0.9,
        AlignmentEvidence(),
    )

    start_locked = edit_word_boundary(
        WordAlignment("t", "old", 1.0, 1.4, AlignmentStatus.ALIGNED, 0.9, AlignmentEvidence()),
        start=1.1,
    )
    incoming_end_before_locked_start = WordAlignment(
        "t", "new", 0.7, 1.0, AlignmentStatus.ALIGNED, 0.8, AlignmentEvidence()
    )
    merged_start = merge_alignment_locks([start_locked], [incoming_end_before_locked_start])
    pitch_start = main._pitch_alignments_with_boundary_locks([source], merged_start)[0]

    end_locked = edit_word_boundary(
        WordAlignment("t", "old", 1.0, 1.4, AlignmentStatus.ALIGNED, 0.9, AlignmentEvidence()),
        end=1.1,
    )
    incoming_start_after_locked_end = WordAlignment(
        "t", "new", 1.2, 1.3, AlignmentStatus.ALIGNED, 0.8, AlignmentEvidence()
    )
    merged_end = merge_alignment_locks([end_locked], [incoming_start_after_locked_end])
    pitch_end = main._pitch_alignments_with_boundary_locks([source], merged_end)[0]

    assert (pitch_start.start, pitch_start.end) == (1.1, 1.4)
    assert (pitch_end.start, pitch_end.end) == (1.0, 1.1)


def test_failed_realignment_keeps_a_completed_result_exportable(tmp_path: Path):
    job = _job(tmp_path)
    revision = job.result.lyric_revision
    assert revision is not None
    job.result.alignment_run = AlignmentRun("run", revision.id, AlignmentRunStatus.COMPLETED, "test")
    job.status = JobStatus.ERROR

    assert main._serialize_result(job)["export_ready"] is True
    job.status = JobStatus.ALIGNING_WORDS
    assert main._serialize_result(job)["export_ready"] is False


def test_remote_draft_helper_uses_worker_audio_mapping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    job = _job(tmp_path)
    mix = tmp_path / "mix.wav"
    mix.touch()
    job.audio_path = str(mix)
    monkeypatch.setenv("ALIGNMENT_WORKER_URL", "http://worker/v1/alignments")
    monkeypatch.setenv("ALIGNMENT_WORKER_TOKEN", "secret")
    monkeypatch.setenv("ALIGNMENT_API_AUDIO_ROOT", str(tmp_path))
    monkeypatch.setenv("ALIGNMENT_WORKER_AUDIO_ROOT", str(tmp_path / "worker"))

    def fake_dispatch(request, worker_url, *, worker_token):
        assert worker_url.endswith("/v1/alignments")
        assert worker_token == "secret"
        assert request.tokens == []
        return AlignmentWorkerResponse(
            job_id=request.job_id,
            alignment_run_id=request.alignment_run_id,
            lyric_revision_id=request.lyric_revision_id,
            status="completed",
            draft_text="hello world",
        )

    monkeypatch.setattr("processing.alignment_worker.dispatch_worker_alignment", fake_dispatch)

    assert main._draft_text_from_remote_worker(job, "run", "en") == "hello world"


def test_phase1_pauses_for_lyrics_review(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    input_path = tmp_path / "input.mp3"
    audio_path = tmp_path / "audio.wav"
    vocals_path = tmp_path / "vocals.wav"
    instrumental_path = tmp_path / "instrumental.wav"
    for path in (input_path, audio_path, vocals_path, instrumental_path):
        path.touch()

    job = Job(
        id="job",
        upload_path=str(input_path),
        metadata=SongMetadata(title="Song", artist="Artist", language="en"),
    )
    main.jobs.clear()
    main.jobs[job.id] = job
    scheduled: list[object] = []

    monkeypatch.setattr("processing.audio.extract_audio", lambda *_: str(audio_path))
    monkeypatch.setattr("processing.audio.is_video_file", lambda *_: False)
    monkeypatch.setattr("processing.vocals.separate_vocals", lambda *_: (str(vocals_path), str(instrumental_path)))
    monkeypatch.setattr("processing.bpm.detect_bpm", lambda *_: 120.0)
    monkeypatch.setattr("processing.lyrics_search.search_lyrics", lambda *_: "online lyrics")
    monkeypatch.setattr(asyncio, "create_task", lambda coroutine: scheduled.append(coroutine))

    asyncio.run(main._process_phase1(job))

    assert job.status == JobStatus.AWAITING_LYRICS
    assert job.found_lyrics == "online lyrics"
    assert job.lyric_revisions == []
    assert scheduled == []


def test_confirm_lyrics_creates_revision_and_resumes_phase2(monkeypatch: pytest.MonkeyPatch):
    job = _awaiting_lyrics_job()
    main.jobs.clear()
    main.jobs[job.id] = job
    scheduled: list[object] = []
    monkeypatch.setattr(asyncio, "create_task", lambda coroutine: scheduled.append(coroutine))

    response = asyncio.run(main.confirm_lyrics(job.id, {"lyrics": "correct lyrics"}))

    assert response["status"] == "started"
    assert job.status == JobStatus.EXTRACTING_LYRICS
    assert job.confirmed_lyrics == "correct lyrics"
    assert len(job.lyric_revisions) == 1
    assert job.lyric_revisions[0].source == LyricSource.MANUAL
    assert scheduled
    scheduled[0].close()


def test_confirm_lyrics_empty_value_selects_asr_path(monkeypatch: pytest.MonkeyPatch):
    job = _awaiting_lyrics_job()
    main.jobs.clear()
    main.jobs[job.id] = job
    scheduled: list[object] = []
    monkeypatch.setattr(asyncio, "create_task", lambda coroutine: scheduled.append(coroutine))

    response = asyncio.run(main.confirm_lyrics(job.id, {"lyrics": ""}))

    assert response["status"] == "started"
    assert job.status == JobStatus.EXTRACTING_LYRICS
    assert job.confirmed_lyrics == ""
    assert job.lyric_revisions == []
    assert scheduled
    scheduled[0].close()


def test_status_is_available_for_pre_result_lyrics_review():
    job = _awaiting_lyrics_job()
    main.jobs.clear()
    main.jobs[job.id] = job

    status = asyncio.run(main.get_status(job.id))

    assert status["metadata"] == {"title": "Song", "artist": "Artist"}
    assert status["found_lyrics"] == "wrong lyrics"


def test_retry_phase1_reuses_existing_audio_and_starts_background(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    source = tmp_path / "input.mp3"
    audio = tmp_path / "audio.wav"
    vocals = tmp_path / "vocals.wav"
    source.touch()
    audio.touch()
    vocals.touch()
    job = Job(
        id="job",
        status=JobStatus.ERROR,
        failed_phase="phase1",
        upload_path=str(source),
        audio_path=str(audio),
        vocals_path=str(vocals),
        confirmed_lyrics="lyrics",
    )
    main.jobs.clear()
    main.jobs[job.id] = job
    scheduled: list[object] = []
    monkeypatch.setattr(asyncio, "create_task", lambda coroutine: scheduled.append(coroutine))

    response = asyncio.run(main.retry_job(job.id))

    assert response == {"status": "retrying", "phase": "phase1"}
    assert job.status == JobStatus.EXTRACTING_AUDIO
    assert job.queued_at
    assert scheduled
    scheduled[0].close()


def test_youtube_download_options_enable_js_challenge_solver(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("KARAOKE_YTDLP_NODE_PATH", "/usr/local/bin/node")

    options = main._youtube_download_options(str(tmp_path), lambda _: None)

    assert options["js_runtimes"] == {"node": {"path": "/usr/local/bin/node"}}
    assert options["remote_components"] == ["ejs:github"]
    assert options["cachedir"] == main.YTDLP_CACHE_DIR
    assert options["outtmpl"] == str(tmp_path / "%(title)s.%(ext)s")
    assert options["retries"] >= 1
    assert options["fragment_retries"] >= 1
