"""Tests for the private UltraStar alignment benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

backend = Path(__file__).resolve().parent.parent
if str(backend) not in sys.path:
    sys.path.insert(0, str(backend))

import pytest

from benchmark_alignment import (
    ReferenceNote,
    ReferenceWord,
    UltraStarReference,
    _compare_pitch_note_events,
    _compare_global_pitch_note_events,
    _pitch_notes_by_word,
    compare_reference,
    normalize_ultrastar_word_text,
    parse_ultrastar_pitch_notes,
    parse_ultrastar_reference,
)
from models import LyricRevision, Word
from processing.alignment import tokenize_lyrics
from processing.alignment_worker import (
    AlignmentWorkerResponse,
    WorkerWordAlignment,
    build_worker_request,
)
from processing.ultrastar import parse_ultrastar_word_timings


def _reference_file(tmp_path: Path) -> Path:
    path = tmp_path / "reference.txt"
    path.write_text(
        "\n".join(
            [
                "#TITLE:Song",
                "#ARTIST:Artist",
                "#BPM:100",
                "#GAP:500",
                ": 0 4 10 Hel",
                ": 4 4 10 lo",
                ": 10 4 12  world",
                "- 16",
                ": 20 2 10 again",
                "E",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_parse_ultrastar_groups_note_fragments_and_converts_beats(tmp_path: Path):
    reference = parse_ultrastar_reference(_reference_file(tmp_path))

    assert reference.lyrics == "Hello world\nagain"
    assert [word.text for word in reference.words] == ["Hello", "world", "again"]
    assert [(word.start, word.end) for word in reference.words] == [
        (0.5, 1.7),
        (2.0, 2.6),
        (3.5, 3.8),
    ]


def test_parse_ultrastar_pitch_notes_keeps_individual_pitches(tmp_path: Path):
    headers, notes = parse_ultrastar_pitch_notes(_reference_file(tmp_path))

    assert headers["BPM"] == "100"
    assert [(note.text, note.pitch, note.start_beat, note.duration_beats) for note in notes] == [
        ("Hel", 10, 0, 4),
        ("lo", 10, 4, 4),
        ("world", 12, 10, 4),
        ("again", 10, 20, 2),
    ]


def test_pitch_word_grouping_keeps_marker_note_pitch(tmp_path: Path):
    path = tmp_path / "continuation.txt"
    path.write_text(
        "\n".join(
            [
                "#BPM:100",
                "#GAP:0",
                ": 0 2 10 held",
                ": 2 2 12 ~",
                "E",
            ]
        ),
        encoding="utf-8",
    )

    reference = parse_ultrastar_reference(path)
    _headers, notes = parse_ultrastar_pitch_notes(path)
    grouped = _pitch_notes_by_word(reference, notes, "en")

    assert len(grouped) == 1
    assert [(note.text, note.pitch) for note in grouped[0]] == [("held", 10), ("~", 12)]


def test_ultrastar_word_normalization_ignores_fragment_and_continuation_markers():
    assert normalize_ultrastar_word_text("Some~one") == "someone"
    assert normalize_ultrastar_word_text("~?") == ""
    assert normalize_ultrastar_word_text("well-being") == "wellbeing"


def test_split_pitch_score_allows_extra_same_pitch_syllable_fragments():
    """A different text partition is equivalent when its pitch events agree."""

    reference = UltraStarReference(
        path=Path("reference.txt"),
        headers={"BPM": "100", "GAP": "0"},
        words=(ReferenceWord("hello", 0.0, 0.6),),
        lyrics="hello",
    )
    token = tokenize_lyrics("hello", "revision", "en")[0]
    notes = [
        ReferenceNote("hel", 0.0, 0.3, 10, ":", 0, 2),
        ReferenceNote("lo", 0.3, 0.6, 10, ":", 2, 2),
    ]
    alignment = type("Alignment", (), {"status": "aligned"})()
    generated = {
        token.id: [
            Word("he", 0.0, 0.15, midi_note=70, pitch_hz=466.16),
            Word("l", 0.15, 0.3, midi_note=70, pitch_hz=466.16),
            Word("o", 0.3, 0.6, midi_note=70, pitch_hz=466.16),
        ]
    }

    report = _compare_pitch_note_events(
        reference,
        [notes],
        [token],
        {token.id: alignment},
        generated,
        language="en",
        pitch_octave_offset=0,
    )

    assert report["pitch_split_word_count"] == 1
    assert report["pitch_split_word_correct_count"] == 1


def test_ultrastar_word_parser_matches_normalized_parenthetical_and_marker_words(tmp_path: Path):
    path = tmp_path / "special.txt"
    path.write_text(
        "\n".join(
            [
                "#BPM:100",
                "#GAP:0",
                ": 0 2 10 (hello",
                ": 2 2 10 )",
                ": 4 2 10  world",
                ": 6 2 10 ~",
                ": 8 2 10  well-being",
                "E",
            ]
        ),
        encoding="utf-8",
    )

    words = parse_ultrastar_word_timings(path)

    assert [word.text for word in words] == ["(hello)", "world~", "well-being"]
    reference = parse_ultrastar_reference(path)
    assert [word.text for word in reference.words] == ["(hello)", "world~", "well-being"]


def test_compare_reference_reports_boundary_metrics(tmp_path: Path):
    reference = parse_ultrastar_reference(_reference_file(tmp_path))
    revision = LyricRevision(
        id="benchmark-revision",
        song_id="song",
        language="en",
        source="manual",
        raw_text=reference.lyrics,
    )
    revision.tokens = tokenize_lyrics(reference.lyrics, revision.id, revision.language)
    request = build_worker_request(
        job_id="benchmark-job",
        revision=revision,
        alignment_run_id="benchmark-run",
        original_audio_ref="/private/mix.wav",
        vocal_stem_audio_ref="/private/vocals.wav",
    )
    response = AlignmentWorkerResponse(
        job_id=request.job_id,
        alignment_run_id=request.alignment_run_id,
        lyric_revision_id=request.lyric_revision_id,
        status="completed",
        alignments=[
            WorkerWordAlignment(
                token_id=token.id,
                alignment_run_id=request.alignment_run_id,
                start=expected.start + 0.02,
                end=expected.end - 0.02,
                status="aligned",
                confidence=0.9,
            )
            for token, expected in zip(request.tokens, reference.words)
        ],
    )

    report = compare_reference(reference, response, request, tolerance_ms=50)

    assert report["reference_word_count"] == 3
    assert report["timed_word_count"] == 3
    assert report["coverage"] == 1.0
    assert report["within_tolerance"] == 1.0
    assert report["start_mae_ms"] == pytest.approx(20.0)
    assert report["end_mae_ms"] == pytest.approx(20.0)

    beat_report = compare_reference(
        reference,
        response,
        request,
        tolerance_ms=50,
        timing_tolerance_beats=1.0,
    )
    # UltraStar's integer chart beat is 60 / (BPM * 4) seconds. At 100 BPM
    # one chart beat is 150 ms, independent of the legacy tolerance-ms value.
    assert beat_report["timing_tolerance_beats"] == 1.0
    assert beat_report["timing_tolerance_ms"] == pytest.approx(150.0)

    late_response = AlignmentWorkerResponse(
        job_id=request.job_id,
        alignment_run_id=request.alignment_run_id,
        lyric_revision_id=request.lyric_revision_id,
        status="completed",
        alignments=[
            WorkerWordAlignment(
                token_id=token.id,
                alignment_run_id=request.alignment_run_id,
                start=expected.start + 0.12,
                end=expected.end - 0.12,
                status="aligned",
                confidence=0.9,
            )
            for token, expected in zip(request.tokens, reference.words)
        ],
    )
    assert compare_reference(
        reference,
        late_response,
        request,
        tolerance_ms=100,
    )["within_tolerance"] == 0.0
    assert compare_reference(
        reference,
        late_response,
        request,
        tolerance_ms=100,
        timing_tolerance_beats=1.0,
    )["within_tolerance"] == 1.0


def test_compare_reference_scores_exact_pitch_and_unvoiced_as_wrong(tmp_path: Path):
    reference = parse_ultrastar_reference(_reference_file(tmp_path))
    _headers, pitch_notes = parse_ultrastar_pitch_notes(_reference_file(tmp_path))
    revision = LyricRevision(
        id="pitch-benchmark-revision",
        song_id="song",
        language="en",
        source="manual",
        raw_text=reference.lyrics,
    )
    revision.tokens = tokenize_lyrics(reference.lyrics, revision.id, revision.language)
    request = build_worker_request(
        job_id="pitch-benchmark-job",
        revision=revision,
        alignment_run_id="pitch-benchmark-run",
        original_audio_ref="/private/mix.wav",
        vocal_stem_audio_ref="/private/vocals.wav",
    )
    response = AlignmentWorkerResponse(
        job_id=request.job_id,
        alignment_run_id=request.alignment_run_id,
        lyric_revision_id=request.lyric_revision_id,
        status="completed",
        alignments=[
            WorkerWordAlignment(
                token_id=token.id,
                alignment_run_id=request.alignment_run_id,
                start=expected.start,
                end=expected.end,
                status="aligned",
                confidence=0.9,
            )
            for token, expected in zip(request.tokens, reference.words)
        ],
    )
    pitch_words = [
        Word(
            text=token.display_text,
            start=expected.start,
            end=expected.end,
            pitch_hz=440.0 if index != 1 else 0.0,
            midi_note=70 if index != 1 else 60,
            token_id=token.id,
        )
        for index, (token, expected) in enumerate(zip(request.tokens, reference.words))
    ]

    report = compare_reference(
        reference,
        response,
        request,
        pitch_notes=pitch_notes,
        pitch_words=pitch_words,
    )

    assert report["pitch_scorable_word_count"] == 3
    assert report["pitch_predicted_word_count"] == 2
    assert report["pitch_correct_word_count"] == 2
    assert report["pitch_accuracy"] == pytest.approx(2 / 3)
    assert report["pitch_coverage"] == pytest.approx(2 / 3)
    assert report["pitch_missing_prediction_count"] == 1


def test_global_pitch_events_do_not_require_matching_word_boundaries():
    notes = (
        ReferenceNote("hel", 0.0, 0.4, 10, ":", 0, 4),
        ReferenceNote("lo", 0.4, 0.8, 12, ":", 4, 4),
    )
    generated = [
        Word("hello", 0.0, 0.42, midi_note=70, pitch_hz=466.16),
        Word("", 0.42, 0.8, midi_note=72, pitch_hz=523.25),
    ]

    report = _compare_global_pitch_note_events(
        notes,
        generated,
        pitch_octave_offset=0,
    )

    assert report["pitch_global_note_scored_count"] == 2
    assert report["pitch_global_note_predicted_count"] == 2
    assert report["pitch_global_note_correct_count"] == 2
    assert report["pitch_global_note_timing_within_tolerance"] == 1.0


def test_global_pitch_onset_timing_uses_ultrastar_beats():
    notes = (ReferenceNote("hello", 1.5, 1.9, 10, ":", 10, 4),)
    generated = [
        # At 100 BPM, one UltraStar chart beat is 150 ms.  This onset is one
        # chart beat late and should still pass the inclusive tolerance.
        Word("hello", 1.65, 2.05, midi_note=70, pitch_hz=466.16),
    ]

    report = _compare_global_pitch_note_events(
        notes,
        generated,
        pitch_octave_offset=0,
        chart_headers={"BPM": "100", "GAP": "0"},
    )

    assert report["pitch_global_note_timing_tolerance_beats"] == 1.0
    assert report["pitch_global_note_timing_within_tolerance"] == 1.0

    too_late = [Word("hello", 1.81, 2.21, midi_note=70, pitch_hz=466.16)]
    report = _compare_global_pitch_note_events(
        notes,
        too_late,
        pitch_octave_offset=0,
        chart_headers={"BPM": "100", "GAP": "0"},
    )
    assert report["pitch_global_note_timing_within_tolerance"] == 0.0
