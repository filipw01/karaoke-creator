"""Tests for pitch detection and assign_pitch_to_words (split by pitch, less chunked)."""

from __future__ import annotations

import numpy as np
import pytest

import sys
from pathlib import Path

backend = Path(__file__).resolve().parent.parent
if str(backend) not in sys.path:
    sys.path.insert(0, str(backend))

from models import Word
from processing.pitch import (
    alignment_activity_threshold,
    assign_pitch_to_words,
    _candidate_evidence_for_span,
    hz_to_midi,
    _infer_candidate_scale,
    _pitch_text_boundaries,
    voiced_activity_signal,
    word_to_segments,
)


def test_candidate_evidence_for_span_aggregates_ranked_midi_support():
    times = np.array([0.0, 0.1, 0.2])
    candidate_midis = np.array([[60, 64], [60, 67], [64, 67]])
    candidate_scores = np.array([[0.8, 0.4], [0.7, 0.3], [0.5, 0.6]])

    evidence = _candidate_evidence_for_span(
        0.0,
        0.2,
        times,
        candidate_midis,
        candidate_scores,
    )

    assert evidence
    assert evidence[0]["midi"] == 60
    assert evidence[0]["support"] == 2
    assert evidence[0]["score"] == 1.0
    assert {item["midi"] for item in evidence} == {60, 64, 67}


def test_alignment_activity_uses_voiced_frames_and_track_relative_threshold():
    frequencies = np.array([0.0, 220.0, 220.0, 0.0, 220.0])
    energies = np.array([0.3, 0.02, 0.04, 0.5, 0.08])

    activity = voiced_activity_signal(frequencies, energies)

    np.testing.assert_array_equal(activity, np.array([0.0, 0.02, 0.04, 0.0, 0.08]))
    assert alignment_activity_threshold(activity) == pytest.approx(0.0375)


def test_assign_pitch_to_words_returns_tuple_segments_line_breaks():
    """assign_pitch_to_words returns (list[Word], list[int])."""
    words = [
        Word(text="hi", start=0.0, end=0.5, pitch_hz=0.0, midi_note=0, confidence=0.0),
        Word(text="bye", start=0.6, end=1.0, pitch_hz=0.0, midi_note=0, confidence=0.0),
    ]
    times = np.array([0.0, 0.25, 0.5, 0.6, 0.8, 1.0])
    frequencies = np.array([261.6, 261.6, 261.6, 261.6, 261.6, 261.6])
    confidences = np.array([0.9] * 6)
    energies = np.ones(6) * 0.01

    segments, seg_lb = assign_pitch_to_words(
        words, times, frequencies, confidences, energies, min_volume=0.0
    )

    assert isinstance(segments, list)
    assert isinstance(seg_lb, list)
    assert len(segments) == 2
    assert segments[0].pitch_hz > 0 and segments[0].midi_note == 60


def test_word_to_segments_splits_only_when_semitones_large():
    """Split when pitch changes by >= MIN_SEMITONES_TO_SPLIT (C4 to F4 = 5 semitones)."""
    word = Word(text="hello", start=0.0, end=1.0, pitch_hz=0.0, midi_note=0, confidence=0.0)
    # C4=60, F4=65 (5 semitones). First run >= 0.25s so it isn't merged away.
    times = np.array([0.0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0])
    frequencies = np.array([261.6, 261.6, 261.6, 349.2, 349.2, 349.2, 349.2])  # C4 then F4
    confidences = np.array([0.9] * 7)
    energies = np.ones(7) * 0.01

    segments = word_to_segments(
        word, times, frequencies, confidences, energies, min_confidence=0.4
    )

    assert len(segments) >= 2
    assert segments[0].midi_note == 60
    assert segments[1].midi_note == 65
    concatenated = "".join(s.text for s in segments)
    assert concatenated == "hello"


def test_word_to_segments_no_split_for_small_change():
    """C4 to D4 (2 semitones) does not split with MIN_SEMITONES_TO_SPLIT=5."""
    word = Word(text="hi", start=0.0, end=0.5, pitch_hz=0.0, midi_note=0, confidence=0.0)
    times = np.array([0.0, 0.2, 0.4])
    frequencies = np.array([261.6, 261.6, 293.7])  # C4, D4
    confidences = np.array([0.9] * 3)
    energies = np.ones(3) * 0.01

    segments = word_to_segments(
        word,
        times,
        frequencies,
        confidences,
        energies,
        min_confidence=0.4,
    )

    assert len(segments) == 1
    assert segments[0].text == "hi"


def test_word_to_segments_never_duplicates_text_when_pitch_has_more_runs_than_chars():
    word = Word(text="a", start=0.0, end=1.0, pitch_hz=0.0, midi_note=0, confidence=0.0)
    times = np.array([0.0, 0.3, 0.6, 1.0])
    frequencies = np.array([261.6, 261.6, 392.0, 392.0])
    confidences = np.array([0.9] * 4)
    energies = np.ones(4) * 0.01

    segments = word_to_segments(word, times, frequencies, confidences, energies, min_confidence=0.4)

    assert "".join(segment.text for segment in segments) == "a"
    assert segments[0].start == word.start
    assert segments[-1].end == word.end


def test_assign_pitch_to_words_no_voiced_frames():
    """Word with no voiced frames yields one segment with pitch 0."""
    words = [Word(text="x", start=0.0, end=0.5, pitch_hz=0.0, midi_note=0, confidence=0.0)]
    times = np.array([0.0, 0.25, 0.5])
    frequencies = np.array([0.0, 0.0, 0.0])
    confidences = np.array([0.1, 0.1, 0.1])
    energies = np.ones(3) * 0.01

    segments, _ = assign_pitch_to_words(
        words, times, frequencies, confidences, energies, min_volume=0.0
    )

    assert len(segments) == 1
    assert segments[0].pitch_hz == 0.0
    assert segments[0].midi_note == 60
    assert segments[0].text == "x"


def test_candidate_range_prior_rejects_an_upper_harmonic(monkeypatch):
    """A sustained low candidate wins over a louder high harmonic by default."""
    monkeypatch.setenv("PITCH_WORD_KEY_PRIOR", "0")
    monkeypatch.setenv("PITCH_WORD_RANGE_MAX_MIDI", "78")
    monkeypatch.setenv("PITCH_WORD_RANGE_PENALTY", "0.05")
    words = [Word(text="hello", start=0.0, end=0.3, pitch_hz=0.0, midi_note=0, confidence=0.0)]
    times = np.array([0.0, 0.1, 0.2])
    frequencies = np.array([261.6, 261.6, 261.6])
    confidences = np.ones(3)
    energies = np.ones(3) * 0.01
    candidates = np.array([[60, 84], [60, 84], [60, 84]])
    scores = np.array([[0.6, 0.9], [0.6, 0.9], [0.6, 0.9]])

    segments, _ = assign_pitch_to_words(
        words,
        times,
        frequencies,
        confidences,
        energies,
        min_volume=0.0,
        candidate_midis=candidates,
        candidate_scores=scores,
    )

    assert segments[0].midi_note == 60


def test_candidate_range_prior_rejects_a_low_subharmonic(monkeypatch):
    """A sustained vocal candidate wins over a spurious A0 subharmonic."""
    monkeypatch.setenv("PITCH_WORD_KEY_PRIOR", "0")
    monkeypatch.setenv("PITCH_WORD_RANGE_MIN_MIDI", "45")
    monkeypatch.setenv("PITCH_WORD_RANGE_MAX_MIDI", "78")
    monkeypatch.setenv("PITCH_WORD_RANGE_PENALTY", "0.05")
    words = [Word(text="hello", start=0.0, end=0.3, pitch_hz=0.0, midi_note=0, confidence=0.0)]
    times = np.array([0.0, 0.1, 0.2])
    frequencies = np.array([261.6, 261.6, 261.6])
    confidences = np.ones(3)
    energies = np.ones(3) * 0.01
    candidates = np.array([[60, 21], [60, 21], [60, 21]])
    scores = np.array([[0.6, 0.9], [0.6, 0.9], [0.6, 0.9]])

    segments, _ = assign_pitch_to_words(
        words,
        times,
        frequencies,
        confidences,
        energies,
        min_volume=0.0,
        candidate_midis=candidates,
        candidate_scores=scores,
    )

    assert segments[0].midi_note == 60


def test_candidate_range_prior_adapts_to_a_high_register(monkeypatch):
    """A high-register contour narrows the harmonic cap from audio evidence."""
    monkeypatch.setenv("PITCH_WORD_KEY_PRIOR", "0")
    monkeypatch.setenv("PITCH_WORD_RANGE_MAX_MIDI", "78")
    monkeypatch.setenv("PITCH_WORD_RANGE_PENALTY", "0.05")
    monkeypatch.setenv("PITCH_WORD_ADAPTIVE_RANGE", "1")
    words = [Word(text="hello", start=0.0, end=0.3, pitch_hz=0.0, midi_note=0, confidence=0.0)]
    times = np.array([0.0, 0.1, 0.2])
    frequencies = np.array([440.0, 440.0, 440.0])
    confidences = np.ones(3)
    energies = np.ones(3) * 0.01
    candidates = np.array([[68, 76], [68, 76], [68, 76]])
    scores = np.array([[0.6, 0.9], [0.6, 0.9], [0.6, 0.9]])

    segments, _ = assign_pitch_to_words(
        words,
        times,
        frequencies,
        confidences,
        energies,
        min_volume=0.0,
        candidate_midis=candidates,
        candidate_scores=scores,
    )

    assert segments[0].midi_note == 68


def test_min_volume_filters_quiet_frames():
    """Frames below min_volume are excluded from pitch (treated as unvoiced)."""
    word = Word(text="hi", start=0.0, end=0.5, pitch_hz=0.0, midi_note=0, confidence=0.0)
    times = np.array([0.0, 0.25, 0.5])
    frequencies = np.array([261.6, 261.6, 261.6])
    confidences = np.array([0.9, 0.9, 0.9])
    # Middle frame below threshold -> only 2 frames count, still one segment
    energies = np.array([0.01, 0.001, 0.01])  # 0.001 < 0.005 default

    segments = word_to_segments(
        word, times, frequencies, confidences, energies,
        min_confidence=0.4, min_volume=0.005,
    )
    assert len(segments) == 1
    assert segments[0].pitch_hz > 0


def test_word_to_segments_median_ignores_unvoiced_frames():
    """A silent frame must not pull a voiced segment's exported Hz toward zero."""
    word = Word(text="hello", start=0.0, end=0.5)
    times = np.array([0.0, 0.12, 0.24, 0.36, 0.5])
    frequencies = np.array([261.6, 0.0, 261.6, 261.6, 261.6])
    confidences = np.ones(5)
    energies = np.ones(5) * 0.01

    segments = word_to_segments(word, times, frequencies, confidences, energies)

    assert segments[0].midi_note == 60
    assert segments[0].pitch_hz == pytest.approx(261.6)


def test_hz_to_midi():
    assert hz_to_midi(440.0) == 69
    assert hz_to_midi(261.6) == 60
    assert hz_to_midi(0) == 0


def test_infer_candidate_scale_uses_pitch_beam_without_reference_labels():
    profile = {9: 3.66, 11: 2.88, 1: 2.23, 2: 3.48, 4: 4.38, 6: 2.52, 8: 2.39}
    classes = set(profile)  # A major pitch classes
    midis = np.array([[60 + pitch_class] for pitch_class in profile])
    scores = np.array([[weight] for weight in profile.values()], dtype=np.float32)

    inferred, margin = _infer_candidate_scale(midis, scores)

    assert inferred == classes
    assert margin > 0.0


def test_word_to_segments_keeps_low_confidence_frequency_as_pitch():
    word = Word(text="hello", start=0.0, end=0.5)
    times = np.array([0.0, 0.25, 0.5])
    frequencies = np.array([261.6, 261.6, 261.6])
    confidences = np.array([0.1, 0.2, 0.1])
    energies = np.ones(3) * 0.01

    segments = word_to_segments(word, times, frequencies, confidences, energies)

    assert segments[0].pitch_hz > 0.0
    assert segments[0].midi_note == 60


def test_word_to_segments_borrows_neighbor_pitch_for_short_unvoiced_gap():
    word = Word(text="gap", start=0.20, end=0.30)
    times = np.array([0.10, 0.25, 0.40])
    frequencies = np.array([293.7, 0.0, 293.7])
    confidences = np.array([0.9, 0.0, 0.9])
    energies = np.array([0.01, 0.01, 0.01])

    segments = word_to_segments(word, times, frequencies, confidences, energies)

    assert segments[0].pitch_hz > 0.0
    assert segments[0].midi_note == 62


def test_word_to_segments_partitions_the_full_word_span_at_pitch_boundaries():
    word = Word(text="hello", start=0.0, end=1.0)
    times = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    frequencies = np.array([261.6, 261.6, 261.6, 349.2, 349.2, 349.2])
    confidences = np.ones(6)
    energies = np.ones(6) * 0.01

    segments = word_to_segments(word, times, frequencies, confidences, energies)

    assert segments[0].start == pytest.approx(word.start)
    assert segments[-1].end == pytest.approx(word.end)
    assert all(left.end == pytest.approx(right.start) for left, right in zip(segments, segments[1:]))
    assert "".join(segment.text for segment in segments) == word.text


def test_word_to_segments_preserves_trailing_continuation_pitch_fragments():
    word = Word(text="all~~~", start=0.0, end=1.0)
    times = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    frequencies = np.array([261.6, 261.6, 392.0, 392.0, 392.0])
    confidences = np.ones(5)
    energies = np.ones(5) * 0.01

    segments = word_to_segments(word, times, frequencies, confidences, energies)

    assert len(segments) == 2
    assert segments[0].text == "all"
    assert segments[1].text == "~~~"
    assert "".join(segment.text for segment in segments) == word.text


def test_pitch_text_boundaries_keep_an_internal_hyphen_with_the_following_syllable():
    word = Word(text="well-being", start=0.0, end=1.0)
    times = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    frequencies = np.array([261.6, 261.6, 392.0, 392.0, 392.0])
    confidences = np.ones(5)
    energies = np.ones(5) * 0.01

    segments = word_to_segments(word, times, frequencies, confidences, energies)

    assert [segment.text for segment in segments] == ["well", "-being"]


def test_pitch_text_boundaries_split_each_internal_continuation_marker():
    assert _pitch_text_boundaries("fi~re") == [0, 2, 3, 5]
    assert _pitch_text_boundaries("breath~~less") == [0, 6, 7, 8, 12]


def test_pitch_text_boundaries_keep_common_chart_morphemes_intact():
    """Do not move onset consonants into the next lyric fragment."""

    assert _pitch_text_boundaries("Having") == [0, 3, 6]
    assert _pitch_text_boundaries("Without") == [0, 4, 7]
    assert _pitch_text_boundaries("Someone") == [0, 4, 7]
    assert _pitch_text_boundaries("nothing") == [0, 2, 7]
    assert _pitch_text_boundaries("everything") == [0, 1, 5, 10]
    assert _pitch_text_boundaries("uninvited") == [0, 2, 4, 6, 9]


def test_pitch_text_boundaries_cover_chart_contractions_and_irregular_splits():
    """Keep recurring UltraStar contraction and syllable annotations stable."""

    assert _pitch_text_boundaries("Getting") == [0, 3, 7]
    assert _pitch_text_boundaries("Settled") == [0, 5, 7]
    assert _pitch_text_boundaries("didn't") == [0, 3, 6]
    assert _pitch_text_boundaries("reminded") == [0, 2, 5, 8]
    assert _pitch_text_boundaries("nevermind") == [0, 2, 5, 9]
    assert _pitch_text_boundaries("careful") == [0, 4, 7]
    assert _pitch_text_boundaries("champions") == [0, 4, 9]
    assert _pitch_text_boundaries("seventeen") == [0, 2, 5, 9]


def test_pitch_text_boundaries_attach_punctuation_to_continuation_marker():
    assert _pitch_text_boundaries("doors~?") == [0, 5, 7]
    assert _pitch_text_boundaries("every~thing") == [0, 3, 5, 6, 11]
