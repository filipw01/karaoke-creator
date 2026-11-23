"""Deterministic tests for canonical word alignment semantics."""

from __future__ import annotations

from dataclasses import replace
import sys
from pathlib import Path

import pytest

backend = Path(__file__).resolve().parent.parent
if str(backend) not in sys.path:
    sys.path.insert(0, str(backend))

from models import AlignmentStatus, LyricToken, Word, WordAlignment
import processing.alignment as alignment_module
from processing.alignment import (
    ASRToken,
    AlignmentCandidate,
    AlignmentEvidence,
    ScoredCandidate,
    activity_spans_from_signal,
    build_word_alignments,
    edit_word_boundary,
    expand_alignment_sustains,
    expand_sustained_end,
    global_monotonic_match,
    line_breaks_for_projected_words,
    merge_alignment_locks,
    map_token_id_aliases,
    normalize_lyric_token,
    normalize_language,
    project_alignments_to_words,
    repair_melody_handoffs,
    reset_word_boundaries,
    score_alignment_candidate,
    select_best_candidates,
    repair_unmatched_alignment_gaps,
    trim_selected_melody_gaps,
    trim_alignment_tails_to_activity,
    snap_alignment_starts_to_activity_onsets,
    trim_alignment_tails_to_strong_activity,
    tokenize_lyrics,
    _candidate_activity_onset_score,
    _same_line_gap_penalty,
    _prepare_activity_profile,
)


def test_tokenization_preserves_source_text_lines_and_polish_diacritics():
    tokens = tokenize_lyrics(
        "Nie ma\nŻadnych, słów!\n\nDon't go",
        "revision-1",
        "pl",
    )

    assert [token.display_text for token in tokens] == ["Nie", "ma", "Żadnych,", "słów!", "Don't", "go"]
    assert [token.line_index for token in tokens] == [0, 0, 1, 1, 3, 3]
    assert tokens[2].normalized_text == "żadnych"
    assert tokens[3].normalized_text == "słów"
    assert normalize_lyric_token("DON'T") == "dont"


def test_invalid_language_types_are_rejected_at_the_alignment_boundary():
    try:
        normalize_language(123)  # type: ignore[arg-type]
    except ValueError as error:
        assert "language" in str(error)
    else:
        raise AssertionError("invalid language type was accepted")


def test_global_match_handles_repeated_words_and_inserted_adlib_monotonically():
    tokens = tokenize_lyrics("we go home\nwe go home", "r", "en")
    asr = [
        ASRToken("we", 0.0, 0.2),
        ASRToken("yeah", 0.2, 0.3),
        ASRToken("go", 0.3, 0.5),
        ASRToken("home", 0.5, 0.7),
        ASRToken("we", 1.0, 1.2),
        ASRToken("go", 1.2, 1.4),
        ASRToken("home", 1.4, 1.6),
    ]

    result = global_monotonic_match(tokens, asr)
    matched = [(item.lyric_index, item.asr_index) for item in result.matches if item.matched]

    assert matched == [(0, 0), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
    assert result.inserted_asr_indices == [1]


def test_global_match_merges_asr_pieces_for_hyphenated_source_words():
    tokens = tokenize_lyrics("hello well-being stays", "r", "en")
    asr = [
        ASRToken("hello", 0.0, 0.2),
        ASRToken("well", 0.3, 0.55),
        ASRToken("being", 0.55, 0.8),
        ASRToken("stays", 0.9, 1.0),
    ]

    result = global_monotonic_match(tokens, asr)
    matched = [(item.lyric_index, item.asr_index) for item in result.matches if item.matched]

    assert matched == [(0, 0), (1, 1), (2, 2)]
    assert result.prepared_asr_tokens[1].start == 0.3
    assert result.prepared_asr_tokens[1].end == 0.8


def test_standalone_punctuation_is_formatting_not_an_alignment_token():
    tokens = tokenize_lyrics("hello — world", "r", "en")

    assert [token.display_text for token in tokens] == ["hello", "world"]
    assert [token.source_index for token in tokens] == [0, 2]


def test_zero_evidence_and_zero_confidence_cannot_score_as_supported():
    candidate = AlignmentCandidate(
        token_id="t",
        start=0.0,
        end=0.4,
        model="forced",
        audio_source="vocal_stem",
        text_score=1.0,
        model_confidence=0.0,
        anchor_score=0.0,
        source_agreement=0.0,
        quality=1.0,
    )

    scored = score_alignment_candidate(candidate)

    assert scored.score == 0.0


def test_silent_candidate_is_rejected_when_activity_is_available():
    tokens = tokenize_lyrics("hello", "r", "en")
    candidate = AlignmentCandidate(tokens[0].id, 4.0, 4.2, "forced", "vocal_stem")

    selected = select_best_candidates(tokens, {tokens[0].id: [candidate]}, activity_spans=[(0.0, 1.0)])

    assert selected[tokens[0].id] is None


def test_activity_onset_score_prefers_a_rising_boundary():
    times = [index / 100.0 for index in range(101)]
    values = [1.0 if time >= 0.5 else 0.0 for time in times]
    profile = _prepare_activity_profile(times, values)

    assert profile is not None
    rising = AlignmentCandidate("rising", 0.5, 0.7, "pitch", "vocal_stem", candidate_type="melody")
    flat = AlignmentCandidate("flat", 0.8, 1.0, "pitch", "vocal_stem", candidate_type="melody")

    assert _candidate_activity_onset_score(rising, profile) > _candidate_activity_onset_score(flat, profile)


def test_gap_at_a_fresh_activity_island_is_not_penalized():
    tokens = tokenize_lyrics("one two", "r", "en")

    assert _same_line_gap_penalty(
        tokens,
        current_index=1,
        previous_index=0,
        previous_end=0.2,
        candidate_start=0.82,
        activity_spans=[(0.0, 0.5), (0.8, 1.2)],
    ) == 0.0


def test_fallback_allocates_only_inside_activity_and_remains_monotonic():
    tokens = tokenize_lyrics("a b c d", "r", "en")
    alignments, _ = build_word_alignments(
        tokens,
        [ASRToken("d", 3.0, 3.2)],
        "run",
        activity_spans=[(0.0, 3.5)],
    )

    spans = [(alignment.start, alignment.end) for alignment in alignments]
    assert all(start is not None and end is not None for start, end in spans)
    assert all(0.0 <= start <= end <= 3.5 for start, end in spans if start is not None and end is not None)
    assert all(left[1] <= right[0] for left, right in zip(spans, spans[1:]) if left[1] is not None and right[0] is not None)


def test_fallback_trims_one_overlapping_trusted_span_without_shifting_later_words():
    tokens = tokenize_lyrics("first second third", "r", "en")
    alignments = [
        WordAlignment(tokens[0].id, "run", 1.0, 2.0, AlignmentStatus.ALIGNED, 0.9),
        WordAlignment(
            tokens[1].id,
            "run",
            1.8,
            2.5,
            AlignmentStatus.ALIGNED,
            0.9,
            start_locked=True,
            end_locked=True,
        ),
        WordAlignment(tokens[2].id, "run", 2.4, 2.9, AlignmentStatus.ALIGNED, 0.9),
    ]

    repaired = alignment_module.fallback_word_alignments(tokens, alignments)

    assert [(item.start, item.end) for item in repaired] == [(1.0, 2.0), (2.0, 2.4), (2.4, 2.9)]
    assert repaired[1].status == AlignmentStatus.ESTIMATED
    assert repaired[1].start_locked is False
    assert repaired[1].end_locked is False
    assert repaired[1].confidence == 0.35
    assert "overlapping trusted span trimmed" in repaired[1].evidence.reasons[-1]
    alignment_module.validate_alignment_sequence(repaired)


def test_fallback_discards_only_fully_overlapped_trusted_span():
    tokens = tokenize_lyrics("first second third", "r", "en")
    alignments = [
        WordAlignment(tokens[0].id, "run", 1.0, 2.0, AlignmentStatus.ALIGNED, 0.9),
        WordAlignment(tokens[1].id, "run", 1.5, 1.9, AlignmentStatus.ALIGNED, 0.9),
        WordAlignment(tokens[2].id, "run", 2.1, 2.5, AlignmentStatus.ALIGNED, 0.9),
    ]

    repaired = alignment_module.fallback_word_alignments(tokens, alignments)

    assert (repaired[1].start, repaired[1].end) == (None, None)
    assert repaired[1].status == AlignmentStatus.UNMATCHED
    assert repaired[2] == alignments[2]
    assert "no non-overlapping duration remained" in repaired[1].evidence.reasons[-1]
    alignment_module.validate_alignment_sequence(repaired)


def test_fallback_between_anchors_stays_unmatched_without_detected_activity():
    tokens = tokenize_lyrics("a b c", "r", "en")
    alignments, _ = build_word_alignments(
        tokens,
        [ASRToken("a", 0.0, 0.2), ASRToken("c", 1.0, 1.2)],
        "run",
    )

    assert alignments[1].status == AlignmentStatus.UNMATCHED
    assert alignments[1].start is None
    assert alignments[1].end is None


def test_weak_match_becomes_unmatched_or_estimated_never_aligned():
    tokens = tokenize_lyrics("hello world", "r", "en")
    asr = [ASRToken("completely", 0.0, 0.4), ASRToken("different", 0.5, 0.9)]

    alignments, _ = build_word_alignments(tokens, asr, "run")

    assert all(item.status != AlignmentStatus.ALIGNED for item in alignments)
    assert all(item.status in {AlignmentStatus.ESTIMATED, AlignmentStatus.UNMATCHED} for item in alignments)


def test_asr_anchor_timestamps_are_retained_as_estimates_without_forced_evidence():
    tokens = tokenize_lyrics("hello world", "r", "en")
    alignments, _ = build_word_alignments(
        tokens,
        [ASRToken("hello", 4.0, 4.4), ASRToken("world", 4.5, 4.9)],
        "run",
    )

    assert all(item.status == AlignmentStatus.ESTIMATED for item in alignments)
    assert [(item.start, item.end) for item in alignments] == [(4.0, 4.4), (4.5, 4.9)]


def test_activity_fallback_is_bounded_by_forced_spans_not_asr_anchors():
    tokens = tokenize_lyrics("a b c", "r", "en")
    forced = [
        AlignmentCandidate(tokens[0].id, 1.0, 1.3, "ctc", "vocal_stem", candidate_type="ctc"),
        AlignmentCandidate(tokens[2].id, 3.0, 3.3, "ctc", "vocal_stem", candidate_type="ctc"),
    ]
    alignments, _ = build_word_alignments(
        tokens,
        [ASRToken("a", 0.0, 0.2), ASRToken("c", 8.0, 8.2)],
        "run",
        candidates=forced,
        activity_spans=[(0.0, 4.0)],
    )

    assert alignments[0].status == AlignmentStatus.ALIGNED
    assert alignments[2].status == AlignmentStatus.ALIGNED
    assert alignments[1].status == AlignmentStatus.ESTIMATED
    assert alignments[1].start >= alignments[0].end
    assert alignments[1].end <= alignments[2].start


def test_forced_evidence_beats_an_asr_anchor_and_is_the_only_aligned_status():
    tokens = tokenize_lyrics("hello", "r", "en")
    forced = AlignmentCandidate(
        token_id=tokens[0].id,
        start=0.1,
        end=0.5,
        model="forced-model",
        audio_source="vocal_stem",
        text_score=1.0,
        model_confidence=0.9,
        candidate_type="forced",
    )

    alignments, _ = build_word_alignments(tokens, [ASRToken("hello", 0.0, 0.8)], "run", candidates=[forced])

    assert alignments[0].status == AlignmentStatus.ALIGNED
    assert alignments[0].evidence.selected_model == "forced-model"


def test_candidate_selection_rejects_colliding_out_of_order_candidates():
    tokens = tokenize_lyrics("one two", "r", "en")
    candidates = {
        tokens[0].id: [AlignmentCandidate(tokens[0].id, 0.8, 1.0, "a", "vocal_stem")],
        tokens[1].id: [
            AlignmentCandidate(tokens[1].id, 0.7, 0.9, "bad", "original_mix", quality=1.0),
            AlignmentCandidate(tokens[1].id, 1.1, 1.4, "good", "original_mix", quality=0.8),
        ],
    }

    selected = select_best_candidates(tokens, candidates)

    assert selected[tokens[0].id] is not None
    assert selected[tokens[1].id] is not None
    assert selected[tokens[1].id].candidate.model == "good"  # type: ignore[union-attr]


def test_candidate_selection_keeps_an_earlier_path_when_a_skipped_token_has_later_options():
    tokens = tokenize_lyrics("one two three", "r", "en")
    candidates = {
        tokens[0].id: [AlignmentCandidate(tokens[0].id, 0.0, 0.2, "first", "vocal_stem", quality=1.0)],
        tokens[1].id: [AlignmentCandidate(tokens[1].id, 0.3, 0.5, "second", "vocal_stem", quality=1.0)],
        tokens[2].id: [
            AlignmentCandidate(tokens[2].id, 0.25, 0.4, "early", "vocal_stem", quality=1.0),
            AlignmentCandidate(tokens[2].id, 0.6, 0.8, "late", "vocal_stem", quality=1.0),
        ],
    }

    selected = select_best_candidates(tokens, candidates)

    assert selected[tokens[0].id] is not None
    assert selected[tokens[1].id] is not None
    assert selected[tokens[2].id].candidate.model == "late"  # type: ignore[union-attr]


def test_sustain_expansion_stops_at_next_word():
    end = expand_sustained_end(
        0.3,
        0.4,
        [0.45, 0.55, 0.65, 0.9, 1.0],
        [1, 1, 1, 1, 1],
        next_word_start=0.7,
    )

    assert 0.6 <= end <= 0.7


def test_sustain_expansion_does_not_cross_an_unmatched_token_or_locked_end():
    alignments = [
        WordAlignment("a", "run", 0.0, 0.4, AlignmentStatus.ALIGNED, 0.9, AlignmentEvidence()),
        WordAlignment("b", "run", None, None, AlignmentStatus.UNMATCHED, 0.0, AlignmentEvidence()),
        WordAlignment("c", "run", 0.9, 1.1, AlignmentStatus.ALIGNED, 0.9, AlignmentEvidence()),
    ]

    expanded = expand_alignment_sustains(alignments, [0.45, 0.55, 0.65, 0.9], [1, 1, 1, 1])

    assert expanded[0].end == 0.4
    locked = [replace(expanded[0], end=0.45, end_locked=True)]
    assert expand_alignment_sustains(locked, [0.5, 0.6], [1, 1])[0].end == 0.45


def test_final_sustain_respects_max_extension_even_with_audio_window_end():
    alignment = WordAlignment(
        "final",
        "run",
        0.0,
        0.4,
        AlignmentStatus.ALIGNED,
        0.9,
        AlignmentEvidence(),
    )

    expanded = expand_alignment_sustains(
        [alignment],
        [0.45, 0.8, 2.3, 2.5, 9.0],
        [1, 1, 1, 1, 1],
        window_end=10.0,
        max_final_extension_s=2.0,
    )

    assert expanded[0].end <= 2.4


def test_melody_selected_span_stops_at_a_substantial_detector_gap():
    alignment = WordAlignment(
        "word",
        "run",
        0.0,
        1.2,
        AlignmentStatus.ALIGNED,
        0.9,
        AlignmentEvidence(selected_model="basic-pitch", selected_audio="vocal_stem"),
    )
    outputs = {
        "basic_pitch:vocal_stem": [
            ASRToken("", 0.05, 0.45, audio_source="vocal_stem", evidence_type="melody"),
            ASRToken("", 0.55, 0.80, audio_source="vocal_stem", evidence_type="melody"),
            ASRToken("", 1.10, 1.20, audio_source="vocal_stem", evidence_type="melody"),
        ]
    }

    trimmed = trim_selected_melody_gaps(
        [alignment],
        outputs,
        gap_threshold_s=0.18,
        extension_s=0.05,
    )

    assert trimmed[0].end is not None and abs(trimmed[0].end - 0.85) < 1e-9
    assert "melody-detector gap" in trimmed[0].evidence.reasons[-1]


def test_melody_gap_trim_preserves_short_and_locked_spans():
    short_gap = WordAlignment(
        "short",
        "run",
        0.0,
        1.0,
        AlignmentStatus.ALIGNED,
        0.9,
        AlignmentEvidence(selected_model="basic-pitch", selected_audio="vocal_stem"),
    )
    locked = replace(short_gap, token_id="locked", start=1.2, end=1.5, end_locked=True)
    outputs = {
        "basic_pitch:vocal_stem": [
            ASRToken("", 0.05, 0.40, audio_source="vocal_stem", evidence_type="melody"),
            ASRToken("", 0.55, 0.75, audio_source="vocal_stem", evidence_type="melody"),
            ASRToken("", 0.90, 1.00, audio_source="vocal_stem", evidence_type="melody"),
        ]
    }

    trimmed = trim_selected_melody_gaps(
        [short_gap, locked],
        outputs,
        gap_threshold_s=0.18,
        extension_s=0.05,
    )

    assert trimmed[0].end == 1.0
    assert trimmed[1] == locked


def test_activity_tail_trim_stops_before_next_island_but_preserves_continuous_activity():
    long_tail = WordAlignment(
        "long",
        "run",
        1.0,
        4.0,
        AlignmentStatus.ALIGNED,
        0.9,
        AlignmentEvidence(selected_model="forcealign-wav2vec2", selected_audio="vocal_stem"),
    )
    continuous = WordAlignment(
        "continuous",
        "run",
        5.0,
        7.0,
        AlignmentStatus.ALIGNED,
        0.9,
        AlignmentEvidence(selected_model="forcealign-wav2vec2", selected_audio="vocal_stem"),
    )

    trimmed = trim_alignment_tails_to_activity(
        [long_tail, continuous],
        [(1.0, 2.0), (4.5, 8.0)],
        extension_s=0.05,
    )

    assert trimmed[0].end == pytest.approx(2.05)
    assert trimmed[1].end == 7.0
    assert "isolated vocal-activity boundary" in trimmed[0].evidence.reasons[-1]


def test_activity_onset_refines_only_an_early_quiet_start():
    alignment = WordAlignment(
        "word",
        "run",
        0.0,
        1.0,
        AlignmentStatus.ALIGNED,
        0.9,
        AlignmentEvidence(selected_model="forcealign-wav2vec2", selected_audio="vocal_stem"),
    )
    refined = snap_alignment_starts_to_activity_onsets(
        [alignment],
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        [0.01, 0.01, 0.01, 0.01, 0.01, 0.20, 0.24, 0.24, 0.24, 0.24, 0.24],
    )

    assert refined[0].start == pytest.approx(0.5)
    assert "onset refined" in refined[0].evidence.reasons[-1]


def test_strong_activity_handoff_trims_only_a_long_crossing_tail():
    alignment = WordAlignment(
        "word",
        "run",
        0.3,
        1.5,
        AlignmentStatus.ALIGNED,
        0.9,
            AlignmentEvidence(selected_model="forcealign-wav2vec2", selected_audio="vocal_stem"),
        )
    successor = WordAlignment(
        "next",
        "run-next",
        1.6,
        2.0,
        AlignmentStatus.ALIGNED,
        0.9,
        AlignmentEvidence(selected_model="forcealign-wav2vec2", selected_audio="vocal_stem"),
    )
    trimmed = trim_alignment_tails_to_strong_activity(
        [alignment, successor],
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        [0.01, 0.01, 0.01, 0.10, 0.10, 0.10, 0.01, 0.01, 0.10, 0.10, 0.10, 0.10, 0.10],
        minimum_gap_s=0.15,
        minimum_source_duration_s=1.0,
    )

    assert trimmed[0].end == pytest.approx(0.64)
    assert "strong vocal handoff" in trimmed[0].evidence.reasons[-1]


def test_melody_gap_caps_activity_sustain_expansion():
    alignment = WordAlignment(
        "word",
        "run",
        0.0,
        0.4,
        AlignmentStatus.ALIGNED,
        0.9,
        AlignmentEvidence(selected_model="basic-pitch", selected_audio="vocal_stem"),
    )
    outputs = {
        "basic_pitch:vocal_stem": [
            ASRToken("", 0.3, 0.5, audio_source="vocal_stem", evidence_type="melody"),
            ASRToken("", 0.7, 0.9, audio_source="vocal_stem", evidence_type="melody"),
        ]
    }

    expanded = expand_alignment_sustains(
        [alignment],
        [0.45, 0.55, 0.65, 0.75, 0.85, 1.0],
        [1, 1, 1, 1, 1, 1],
        melody_outputs=outputs,
    )

    assert expanded[0].end == 0.58


def test_melody_gap_refines_nearby_unlocked_start():
    alignment = WordAlignment(
        "word",
        "run",
        0.58,
        0.9,
        AlignmentStatus.ALIGNED,
        0.9,
        AlignmentEvidence(selected_model="basic-pitch", selected_audio="vocal_stem"),
    )
    outputs = {
        "basic_pitch:vocal_stem": [
            ASRToken("", 0.2, 0.4, audio_source="vocal_stem", evidence_type="melody"),
            ASRToken("", 0.6, 0.8, audio_source="vocal_stem", evidence_type="melody"),
        ]
    }

    expanded = expand_alignment_sustains(
        [alignment],
        [0.95, 1.05],
        [1, 1],
        melody_outputs=outputs,
    )

    assert expanded[0].start == 0.64


def test_confidence_gated_melody_onset_lane_ignores_harmonic_decoy():
    alignment = WordAlignment(
        "word",
        "run",
        0.58,
        0.9,
        AlignmentStatus.ALIGNED,
        0.9,
        AlignmentEvidence(selected_model="basic-pitch", selected_audio="vocal_stem"),
    )
    outputs = {
        "basic_pitch:vocal_stem": [
            ASRToken("", 0.2, 0.4, confidence=0.8, audio_source="vocal_stem", evidence_type="melody"),
            ASRToken("", 0.6, 0.65, confidence=0.35, audio_source="vocal_stem", evidence_type="melody"),
            ASRToken("", 0.64, 0.9, confidence=0.8, audio_source="vocal_stem", evidence_type="melody"),
        ]
    }

    expanded = expand_alignment_sustains(
        [alignment],
        [0.95, 1.05],
        [1, 1],
        melody_outputs=outputs,
        melody_start_gap_threshold_s=0.08,
        melody_start_tolerance_s=0.08,
        melody_start_offset_s=0.08,
        melody_start_min_confidence=0.55,
    )

    assert expanded[0].start == 0.72


def test_voiced_island_onset_refines_nearby_melody_start():
    alignment = WordAlignment(
        "word",
        "run",
        0.52,
        0.8,
        AlignmentStatus.ALIGNED,
        0.9,
        AlignmentEvidence(selected_model="basic-pitch", selected_audio="vocal_stem"),
    )
    outputs = {
        "basic_pitch:vocal_stem": [
            ASRToken("", 0.4, 0.8, audio_source="vocal_stem", evidence_type="melody"),
        ]
    }

    expanded = expand_alignment_sustains(
        [alignment],
        [0.5, 0.6, 0.7],
        [1, 1, 1],
        melody_outputs=outputs,
    )

    assert expanded[0].start == 0.54


def test_melody_handoff_fills_a_short_voiced_gap_between_words():
    alignments = [
        WordAlignment(
            "first",
            "run",
            0.0,
            0.5,
            AlignmentStatus.ALIGNED,
            0.9,
            AlignmentEvidence(selected_model="basic-pitch", selected_audio="vocal_stem"),
        ),
        WordAlignment(
            "second",
            "run",
            0.68,
            0.9,
            AlignmentStatus.ALIGNED,
            0.9,
            AlignmentEvidence(selected_model="basic-pitch", selected_audio="vocal_stem"),
        ),
    ]
    outputs = {
        "basic_pitch:vocal_stem": [
            ASRToken("", 0.1, 0.5, confidence=0.8, audio_source="vocal_stem", evidence_type="melody"),
            ASRToken("", 0.56, 0.9, confidence=0.8, audio_source="vocal_stem", evidence_type="melody"),
        ]
    }

    expanded = expand_alignment_sustains(
        alignments,
        [0.5, 0.58, 0.66, 0.74, 0.82],
        [1, 1, 1, 1, 1],
        clear_gap_s=0.01,
        melody_outputs=outputs,
        melody_handoff_min_gap_s=0.08,
        melody_handoff_max_gap_s=0.25,
        melody_handoff_offset_s=0.08,
    )

    assert expanded[1].start == 0.58


def test_melody_handoff_repairs_internal_same_pitch_boundary():
    tokens = tokenize_lyrics("I just", "revision", "en")
    alignments = [
        WordAlignment(
            tokens[0].id,
            "run",
            0.01,
            0.60,
            AlignmentStatus.ALIGNED,
            0.9,
            AlignmentEvidence(selected_model="basic-pitch", selected_audio="vocal_stem"),
        ),
        WordAlignment(
            tokens[1].id,
            "run",
            0.61,
            1.05,
            AlignmentStatus.ALIGNED,
            0.9,
            AlignmentEvidence(selected_model="basic-pitch", selected_audio="vocal_stem"),
        ),
    ]
    evidence = {
        "vocal_stem": [ASRToken("I", 0.0, 0.3), ASRToken("just", 0.8, 1.7)],
        "basic_pitch:vocal_stem": [
            ASRToken("", 0.0, 0.25, confidence=0.8, midi_note=60, evidence_type="melody"),
            ASRToken("", 0.25, 0.8, confidence=0.8, midi_note=60, evidence_type="melody"),
            ASRToken("", 0.82, 1.0, confidence=0.8, midi_note=62, evidence_type="melody"),
        ],
    }

    repaired = repair_melody_handoffs(
        alignments,
        evidence,
        tokens=tokens,
        anchor_outputs=evidence,
    )

    assert repaired[0].end is not None and repaired[0].end > 0.78
    assert repaired[1].start is not None and repaired[1].start > 0.86
    assert repaired[0].end <= repaired[1].start
    assert "confident melody handoff" in repaired[0].evidence.reasons[-1]


def test_melody_handoff_does_not_change_locked_or_short_successors():
    tokens = tokenize_lyrics("I a", "revision", "en")
    first = WordAlignment(
        tokens[0].id,
        "run",
        0.01,
        0.75,
        AlignmentStatus.ALIGNED,
        0.9,
        AlignmentEvidence(selected_model="basic-pitch", selected_audio="vocal_stem"),
    )
    second = WordAlignment(
        tokens[1].id,
        "run",
        0.76,
        1.05,
        AlignmentStatus.ALIGNED,
        0.9,
        AlignmentEvidence(selected_model="basic-pitch", selected_audio="vocal_stem"),
        start_locked=True,
    )
    evidence = {
        "vocal_stem": [ASRToken("I", 0.0, 0.3), ASRToken("a", 0.8, 0.9)],
        "basic_pitch:vocal_stem": [
            ASRToken("", 0.0, 0.25, confidence=0.8, midi_note=60, evidence_type="melody"),
            ASRToken("", 0.25, 0.8, confidence=0.8, midi_note=60, evidence_type="melody"),
            ASRToken("", 0.82, 1.0, confidence=0.8, midi_note=62, evidence_type="melody"),
        ],
    }

    repaired = repair_melody_handoffs(
        [first, second],
        evidence,
        tokens=tokens,
        anchor_outputs=evidence,
    )

    assert repaired == [first, second]


def test_unmatched_gap_repair_requires_candidates_to_fit_both_neighbors():
    alignments = [
        WordAlignment("before", "run", 0.0, 0.5, AlignmentStatus.ALIGNED, 0.9),
        WordAlignment("missing", "run", None, None, AlignmentStatus.UNMATCHED, 0.0),
        WordAlignment("after", "run", 1.0, 1.5, AlignmentStatus.ALIGNED, 0.9),
    ]
    candidate = AlignmentCandidate(
        "missing",
        0.62,
        0.88,
        "basic-pitch",
        "vocal_stem",
        text_score=1.0,
        model_confidence=0.9,
        candidate_type="melody",
    )

    repaired = repair_unmatched_alignment_gaps(alignments, [candidate])

    assert repaired[1].status == AlignmentStatus.ALIGNED
    assert repaired[1].start == 0.62
    assert repaired[1].end == 0.88


def test_activity_signal_becomes_bounded_fallback_spans():
    spans = activity_spans_from_signal(
        [0.0, 0.05, 0.1, 0.5, 0.55],
        [0.0, 0.01, 0.02, 0.0, 0.03],
    )

    assert [(round(start, 6), round(end, 6)) for start, end in spans] == [
        (0.05, 0.15),
        (0.55, 0.6),
    ]


def test_rejected_overlapping_candidate_is_not_emitted_as_trusted(monkeypatch: pytest.MonkeyPatch):
    tokens = tokenize_lyrics("hello world", "revision", "en")
    candidates = [
        AlignmentCandidate(tokens[0].id, 1.0, 1.4, "forced-aligner", "vocal_stem"),
        AlignmentCandidate(tokens[1].id, 1.3, 1.6, "forced-aligner", "vocal_stem"),
    ]
    monkeypatch.setattr(
        alignment_module,
        "select_best_candidates",
        lambda *_args, **_kwargs: {
            candidate.token_id: ScoredCandidate(candidate, 0.9, ())
            for candidate in candidates
        },
    )

    alignments, _ = build_word_alignments(tokens, [], "run", candidates=candidates)

    assert alignments[0].status == AlignmentStatus.UNMATCHED
    assert alignments[0].start is None
    assert alignments[1].status == AlignmentStatus.ALIGNED
    assert alignments[1].start == 1.3
    assert alignments[1].end == 1.6
    assert "conflicts with neighboring" in alignments[0].evidence.reasons[0]


def test_boundary_edit_lock_survives_realign_and_explicit_reset():
    original = WordAlignment(
        token_id="t",
        alignment_run_id="old",
        start=1.0,
        end=1.4,
        status=AlignmentStatus.ALIGNED,
        confidence=0.9,
        evidence=AlignmentEvidence(),
    )
    edited = edit_word_boundary(original, start=1.1)
    incoming = WordAlignment(
        token_id="t",
        alignment_run_id="new",
        start=0.8,
        end=1.5,
        status=AlignmentStatus.ALIGNED,
        confidence=0.8,
        evidence=AlignmentEvidence(),
    )

    merged = merge_alignment_locks([edited], [incoming])[0]
    reset = reset_word_boundaries(edited, reset_start=True, reset_end=False)

    assert merged.start == 1.1
    assert merged.start_locked is True
    assert merged.end == 1.5
    assert reset.start_locked is False
    assert reset.end_locked is False


def test_manual_locks_follow_unchanged_words_into_a_new_revision():
    previous_tokens = tokenize_lyrics("hello world", "old", "en")
    new_tokens = tokenize_lyrics("hey hello world", "new", "en")
    aliases = map_token_id_aliases(previous_tokens, new_tokens)
    old_alignment = edit_word_boundary(
        WordAlignment(previous_tokens[0].id, "old-run", 1.0, 1.3, AlignmentStatus.ALIGNED, 0.9, AlignmentEvidence()),
        start=1.1,
    )
    incoming = [
        WordAlignment(new_tokens[0].id, "new-run", None, None, AlignmentStatus.UNMATCHED, 0.0, AlignmentEvidence()),
        WordAlignment(new_tokens[1].id, "new-run", 1.4, 1.7, AlignmentStatus.UNMATCHED, 0.0, AlignmentEvidence()),
        WordAlignment(new_tokens[2].id, "new-run", None, None, AlignmentStatus.UNMATCHED, 0.0, AlignmentEvidence()),
    ]

    merged = merge_alignment_locks([old_alignment], incoming, token_id_aliases=aliases)

    assert merged[1].start == 1.1
    assert merged[1].start_locked is True
    assert merged[1].status == AlignmentStatus.ESTIMATED


def test_manual_span_turns_unmatched_word_into_explicit_estimate():
    original = WordAlignment(
        token_id="t",
        alignment_run_id="run",
        start=None,
        end=None,
        status=AlignmentStatus.UNMATCHED,
        confidence=0.0,
        evidence=AlignmentEvidence(),
    )

    edited = edit_word_boundary(original, start=1.0, end=1.3)

    assert edited.status == AlignmentStatus.ESTIMATED
    assert edited.start_locked is True
    assert edited.end_locked is True


def test_partial_manual_lock_keeps_a_paired_span_when_new_evidence_is_missing():
    edited = edit_word_boundary(
        WordAlignment("t", "old", 1.0, 1.4, AlignmentStatus.ALIGNED, 0.9, AlignmentEvidence()),
        start=1.1,
    )
    incoming = WordAlignment("t", "new", None, None, AlignmentStatus.UNMATCHED, 0.0, AlignmentEvidence())

    merged = merge_alignment_locks([edited], [incoming])[0]

    assert (merged.start, merged.end) == (1.1, 1.4)
    assert merged.start_locked is True
    assert merged.status == AlignmentStatus.ESTIMATED


def test_projection_uses_canonical_boundaries_not_legacy_word_timestamps():
    tokens = tokenize_lyrics("hello", "r", "en")
    alignment = WordAlignment(
        token_id=tokens[0].id,
        alignment_run_id="run",
        start=2.0,
        end=2.5,
        status=AlignmentStatus.ESTIMATED,
        confidence=0.3,
        evidence=AlignmentEvidence(),
    )
    legacy = project_alignments_to_words(
        tokens,
        [alignment],
        existing_words=[],
    )

    assert legacy[0].start == 2.0
    assert legacy[0].end == 2.5
    assert legacy[0].text == "hello"
    assert legacy[0].alignment_status == "estimated"


def test_projection_preserves_pitch_segments_while_rescaling_canonical_span():
    tokens = tokenize_lyrics("hello", "r", "en")
    alignment = WordAlignment(
        token_id=tokens[0].id,
        alignment_run_id="run",
        start=2.0,
        end=4.0,
        status=AlignmentStatus.ALIGNED,
        confidence=0.9,
        evidence=AlignmentEvidence(),
    )
    projected = project_alignments_to_words(
        tokens,
        [alignment],
        existing_words=[
            Word(text="hel", start=10.0, end=11.0, midi_note=60, token_id=tokens[0].id),
            Word(text="lo", start=11.0, end=13.0, midi_note=64, token_id=tokens[0].id),
        ],
    )

    assert [(round(word.start, 6), round(word.end, 6), word.text, word.midi_note) for word in projected] == [
        (2.0, round(2.0 + 2.0 / 3.0, 6), "hel", 60),
        (round(2.0 + 2.0 / 3.0, 6), 4.0, "lo", 64),
    ]
    assert line_breaks_for_projected_words(tokens, projected) == []
