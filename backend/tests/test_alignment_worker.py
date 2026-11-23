"""Contract tests for the FastAPI/GPU-worker alignment seam."""

from __future__ import annotations

import sys
from pathlib import Path

backend = Path(__file__).resolve().parent.parent
if str(backend) not in sys.path:
    sys.path.insert(0, str(backend))

import pytest

from models import LyricRevision, Word
from processing.alignment import ASRToken, tokenize_lyrics
from processing.alignment_worker import (
    AlignmentWorkerError,
    AlignmentWorkerResponse,
    WorkerWordAlignment,
    build_worker_request,
    collect_model_evidence,
    _candidate_sources,
    run_local_alignment,
    validate_worker_response,
)


def _request():
    revision = LyricRevision(
        id="revision",
        song_id="song",
        language="pl",
        source="online",
        raw_text="Żółć ma",
    )
    revision.tokens = tokenize_lyrics(revision.raw_text, revision.id, "pl")
    return build_worker_request(
        job_id="job",
        revision=revision,
        alignment_run_id="run",
        original_audio_ref="/audio/mix.wav",
        vocal_stem_audio_ref="/audio/vocals.wav",
    )


def test_polish_requests_use_legacy_backend_by_default():
    request = _request()

    assert request.language == "pl"
    assert request.alignment_backend == "legacy"
    assert request.original_audio_ref.endswith("mix.wav")
    assert request.vocal_stem_audio_ref.endswith("vocals.wav")
    assert [token.display_text for token in request.tokens] == ["Żółć", "ma"]


def test_polish_sofa_requires_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("SOFA_ALLOW_NON_ENGLISH", "1")
    assert _request().alignment_backend == "sofa"


def test_melody_candidates_cannot_start_before_their_physical_anchor():
    revision = LyricRevision(
        id="english-revision",
        song_id="song",
        language="en",
        source="online",
        raw_text="hello",
    )
    revision.tokens = tokenize_lyrics(revision.raw_text, revision.id, "en")
    request = build_worker_request(
        job_id="job",
        revision=revision,
        alignment_run_id="run",
        original_audio_ref="/audio/mix.wav",
        vocal_stem_audio_ref="/audio/vocals.wav",
    )

    _primary, candidates, _diagnostics = _candidate_sources(
        request,
        {
            "vocal_stem": [ASRToken("hello", 1.0, 1.3, audio_source="vocal_stem")],
            "basic_pitch:vocal_stem": [
                ASRToken("", 0.8, 1.0, audio_source="vocal_stem", evidence_type="melody"),
                ASRToken("", 1.1, 1.4, audio_source="vocal_stem", evidence_type="melody"),
            ],
        },
    )

    melody = [candidate for candidate in candidates if candidate.candidate_type == "melody"]
    assert melody
    assert all(candidate.start >= 0.95 for candidate in melody)


def test_worker_response_rejects_stale_identity_and_unknown_tokens():
    request = _request()
    stale = AlignmentWorkerResponse(
        job_id=request.job_id,
        alignment_run_id="different-run",
        lyric_revision_id=request.lyric_revision_id,
        status="completed",
    )
    unknown = AlignmentWorkerResponse(
        job_id=request.job_id,
        alignment_run_id=request.alignment_run_id,
        lyric_revision_id=request.lyric_revision_id,
        status="completed",
        alignments=[
            WorkerWordAlignment(
                token_id="not-in-revision",
                alignment_run_id=request.alignment_run_id,
                status="aligned",
                confidence=1.0,
            )
        ],
    )

    with pytest.raises(AlignmentWorkerError):
        validate_worker_response(stale, request)
    with pytest.raises(AlignmentWorkerError):
        validate_worker_response(unknown, request)

    incomplete = AlignmentWorkerResponse(
        job_id=request.job_id,
        alignment_run_id=request.alignment_run_id,
        lyric_revision_id=request.lyric_revision_id,
        status="completed",
    )
    with pytest.raises(AlignmentWorkerError):
        validate_worker_response(incomplete, request)


def test_local_worker_keeps_dual_source_diagnostics_and_canonical_ids():
    request = _request()
    response = run_local_alignment(
        request,
        {
            "vocal_stem": [
                ASRToken("Żółć", 0.0, 0.4, audio_source="vocal_stem"),
                ASRToken("ma", 0.5, 0.7, audio_source="vocal_stem"),
            ],
            "original_mix": [
                ASRToken("Żółć", 0.02, 0.42, audio_source="original_mix"),
                ASRToken("ma", 0.52, 0.72, audio_source="original_mix"),
            ],
        },
    )

    assert response.status == "completed"
    assert response.diagnostics["asr_sources"]["original_mix"]["matched_token_count"] == 2
    assert response.diagnostics["candidate_diagnostics"]
    assert [item.token_id for item in response.alignments] == [token.id for token in request.tokens]
    assert all(item.status == "estimated" for item in response.alignments)
    assert [(item.start, item.end) for item in response.alignments] == [(0.0, 0.4), (0.5, 0.7)]


def test_worker_rejects_nonfinite_partial_and_cross_token_spans():
    request = _request()
    malformed = AlignmentWorkerResponse(
        job_id=request.job_id,
        alignment_run_id=request.alignment_run_id,
        lyric_revision_id=request.lyric_revision_id,
        status="completed",
        alignments=[
            WorkerWordAlignment(
                token_id=request.tokens[0].id,
                alignment_run_id=request.alignment_run_id,
                start=float("nan"),
                end=0.4,
                status="aligned",
                confidence=1.0,
            ),
            WorkerWordAlignment(
                token_id=request.tokens[1].id,
                alignment_run_id=request.alignment_run_id,
                start=0.1,
                end=0.2,
                status="aligned",
                confidence=1.0,
            ),
        ],
    )

    with pytest.raises(AlignmentWorkerError):
        validate_worker_response(malformed, request)

    partial = malformed.model_copy(deep=True)
    partial.alignments[0].start = None
    partial.alignments[0].end = 0.4
    with pytest.raises(AlignmentWorkerError):
        validate_worker_response(partial, request)

    collision = malformed.model_copy(deep=True)
    collision.alignments[0].start = 0.2
    collision.alignments[0].end = 0.4
    collision.alignments[1].start = 0.1
    collision.alignments[1].end = 0.5
    with pytest.raises(AlignmentWorkerError):
        validate_worker_response(collision, request)


def test_worker_cannot_create_manual_locks():
    request = _request()
    response = AlignmentWorkerResponse(
        job_id=request.job_id,
        alignment_run_id=request.alignment_run_id,
        lyric_revision_id=request.lyric_revision_id,
        status="completed",
        alignments=[
            WorkerWordAlignment(
                token_id=token.id,
                alignment_run_id=request.alignment_run_id,
                start=float(index),
                end=float(index) + 0.2,
                status="aligned",
                confidence=1.0,
                start_locked=True,
            )
            for index, token in enumerate(request.tokens)
        ],
    )

    with pytest.raises(AlignmentWorkerError):
        validate_worker_response(response, request)


def test_local_worker_reports_missing_model_evidence():
    with pytest.raises(AlignmentWorkerError):
        run_local_alignment(_request(), {})


def test_production_ignores_benchmark_timing_reference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    request = _request().model_copy(
        update={
            "original_audio_ref": str(tmp_path / "mix.wav"),
            "vocal_stem_audio_ref": str(tmp_path / "vocals.wav"),
        }
    )
    Path(request.original_audio_ref).touch()
    Path(request.vocal_stem_audio_ref).touch()
    monkeypatch.setenv("ALIGNMENT_ULTRASTAR_REFERENCE", str(tmp_path / "reference.txt"))
    monkeypatch.setattr("processing.model_adapters.qwen_asr_enabled", lambda: False)
    monkeypatch.setattr("processing.model_adapters.whisperx_enabled", lambda: False)
    monkeypatch.setattr("processing.model_adapters.basic_pitch_enabled", lambda: False)
    monkeypatch.setattr(
        "processing.model_adapters.run_serialized_model_call",
        lambda function, *args, **kwargs: [
            Word(text="Żółć", start=0.5, end=0.8, confidence=0.9),
            Word(text="ma", start=0.8, end=1.0, confidence=0.9),
        ],
    )

    evidence, errors = collect_model_evidence(request)

    assert errors == {}
    assert "ultrastar_chart" not in evidence
    assert set(evidence) == {"vocal_stem", "original_mix"}


def test_production_merges_the_blended_melody_lane(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    request = _request().model_copy(
        update={
            "original_audio_ref": str(tmp_path / "mix.wav"),
            "vocal_stem_audio_ref": str(tmp_path / "vocals.wav"),
        }
    )
    Path(request.original_audio_ref).touch()
    Path(request.vocal_stem_audio_ref).touch()
    monkeypatch.setattr("processing.model_adapters.qwen_asr_enabled", lambda: False)
    monkeypatch.setattr("processing.model_adapters.whisperx_enabled", lambda: False)
    monkeypatch.setattr("processing.model_adapters.basic_pitch_enabled", lambda: True)
    monkeypatch.setattr("processing.model_adapters.melody_blend_enabled", lambda: True)
    monkeypatch.setattr(
        "processing.model_adapters.write_melody_blend_audio",
        lambda _original, _vocals, output: Path(output).touch(),
    )

    def fake_model_call(function, *args, **kwargs):
        if function.__name__ == "extract_lyrics":
            return [
                Word(text="Żółć", start=0.5, end=0.8, confidence=0.9),
                Word(text="ma", start=0.8, end=1.0, confidence=0.9),
            ]
        return [ASRToken("", 0.1, 0.3, audio_source="original_mix", evidence_type="melody")]

    monkeypatch.setattr("processing.model_adapters.run_serialized_model_call", fake_model_call)

    evidence, errors = collect_model_evidence(request)

    assert errors == {}
    assert len(evidence["basic_pitch:vocal_stem"]) == 2
    assert evidence["basic_pitch:vocal_stem"][-1].model == "basic-pitch-blend"
    assert evidence["basic_pitch:vocal_stem"][-1].audio_source == "vocal_stem"
