"""Run the local word-alignment pipeline against a private UltraStar reference.

The reference song is supplied at runtime rather than stored in the
repository. UltraStar note beats are converted to seconds using the file's
``#BPM`` and ``#GAP`` headers, then adjacent note fragments are grouped into
word spans before comparing them with generated canonical alignments. An
optional pitch-bearing chart is matched on the same beat grid and scored as
exact MIDI agreement per aligned word.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
from typing import Mapping, Sequence
import uuid

import numpy as np

from models import LyricRevision, Word
from processing.alignment import (
    ASRToken,
    activity_spans_from_signal,
    expand_alignment_sustains,
    normalize_language,
    normalize_lyric_token,
    repair_melody_handoffs,
    tokenize_lyrics,
)
from processing.alignment_worker import (
    AlignmentWorkerRequest,
    AlignmentWorkerResponse,
    _alignment_to_payload,
    alignment_response_to_models,
    build_worker_request,
    collect_model_evidence,
    run_local_alignment,
)
from processing.pitch import (
    DEFAULT_MIN_VOLUME_RMS,
    alignment_activity_threshold,
    assign_pitch_to_words,
    detect_pitch,
    estimate_timing_grid,
    voiced_activity_signal,
)
from processing.ultrastar import parse_ultrastar_word_timings


_NOTE_TYPES = frozenset({":", "*", "F", "R", "G"})
_NOTE_PATTERN = re.compile(r"^(\S+)\s+(\d+)\s+(\d+)\s+(-?\d+)(.*)$")


def _read_chart_lines(path: Path) -> list[str]:
    """Read common UltraStar encodings, including legacy Latin-1 charts."""

    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding).splitlines()
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", raw, 0, len(raw), "unsupported chart encoding")


@dataclass(frozen=True)
class ReferenceNote:
    text: str
    start: float
    end: float
    pitch: int
    note_type: str
    start_beat: int
    duration_beats: int


@dataclass(frozen=True)
class ReferenceWord:
    text: str
    start: float
    end: float
    notes: tuple[ReferenceNote, ...] = ()


@dataclass(frozen=True)
class UltraStarReference:
    path: Path
    headers: dict[str, str]
    words: tuple[ReferenceWord, ...]
    lyrics: str


def normalize_ultrastar_word_text(
    text: str,
    language: str | None = None,
) -> str:
    """Normalize chart fragments into a comparable logical-word key.

    UltraStar stores one text value per pitch event.  A single spoken word
    may therefore appear as ``Some`` + ``one``, ``So`` + ``meone``, or with
    continuation markers such as ``Some`` + ``~`` + ``one``.  Fragment
    boundaries and marker punctuation are annotation details, not different
    words.  Keep note rows (and their pitch/timing) separate; use this key
    only when comparing lyric identity.
    """

    return normalize_lyric_token(text.replace("~", ""), language)


def _reference_headers(lines: Sequence[str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for line in lines:
        if line.startswith("#"):
            key, separator, value = line[1:].partition(":")
            if separator:
                headers[key.strip().upper()] = value.strip()
    return headers


def parse_ultrastar_pitch_notes(path: str | Path) -> tuple[dict[str, str], tuple[ReferenceNote, ...]]:
    """Parse pitch-bearing note events from an UltraStar TXT chart.

    The chart format has two common word-boundary conventions: an extra
    leading space after the pitch, or a trailing space after the lyric text.
    Pitch scoring only needs the individual events, so it deliberately keeps
    both forms without guessing where syllables should be joined.
    """

    reference_path = Path(path).expanduser().resolve()
    if not reference_path.is_file():
        raise FileNotFoundError(f"Reference file does not exist: {reference_path}")
    lines = _read_chart_lines(reference_path)
    headers = _reference_headers(lines)
    bpm = _parse_number(headers.get("BPM", "0"), "BPM")
    if bpm <= 0:
        raise ValueError("UltraStar reference must contain a positive #BPM")
    seconds_per_beat = 60.0 / (bpm * 4.0)
    gap_seconds = _parse_number(headers.get("GAP", "0"), "GAP") / 1000.0
    relative = _is_true(headers.get("RELATIVE"))
    notes: list[ReferenceNote] = []
    previous_start_beat = 0
    active_voice = 1
    for line_number, line in enumerate(lines, 1):
        if not line.strip() or line.startswith("#"):
            continue
        if line.strip() == "E":
            break
        if line.startswith("-"):
            continue
        if line.startswith("P") and line[1:].strip().isdigit():
            active_voice = int(line[1:].strip())
            continue
        if active_voice != 1:
            continue
        match = _NOTE_PATTERN.match(line)
        if match is None or match.group(1) not in _NOTE_TYPES:
            continue
        try:
            start_beat = int(match.group(2))
            duration_beat = int(match.group(3))
            pitch = int(match.group(4))
        except ValueError as exc:
            raise ValueError(f"Invalid note data on line {line_number}: {line!r}") from exc
        if start_beat < 0 or duration_beat < 0:
            raise ValueError(f"Negative note timing on line {line_number}: {line!r}")
        if relative:
            start_beat = previous_start_beat + start_beat
        previous_start_beat = start_beat
        raw_text = match.group(5)
        text = raw_text.strip()
        note_start = gap_seconds + start_beat * seconds_per_beat
        note_end = gap_seconds + (start_beat + duration_beat) * seconds_per_beat
        notes.append(
            ReferenceNote(
                text=text,
                start=note_start,
                end=note_end,
                pitch=pitch,
                note_type=match.group(1),
                start_beat=start_beat,
                duration_beats=duration_beat,
            )
        )
    if not notes:
        raise ValueError(f"Reference contains no timed pitch notes: {reference_path}")
    # Most UltraStar charts store pitch as a semitone offset from C4, but a
    # few exported charts (notably Dancing Queen) store absolute MIDI values.
    # Keep the benchmark's internal representation consistent by converting
    # the latter back to the relative form consumed by the scorer.
    if notes and statistics.median(note.pitch for note in notes) >= 50:
        notes = [replace(note, pitch=note.pitch - 60) for note in notes]
    return headers, tuple(notes)


def _parse_number(value: str, field: str) -> float:
    try:
        number = float(value.strip().replace(",", "."))
    except ValueError as exc:
        raise ValueError(f"Invalid #{field} value: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Invalid #{field} value: {value!r}")
    return number


def _chart_timing_grid(
    headers: Mapping[str, str] | None,
    notes: Sequence[ReferenceNote] = (),
) -> tuple[float, float]:
    """Return ``(gap_seconds, seconds_per_ultrastar_beat)``.

    UltraStar note positions are integer chart beats.  The chart's BPM and
    GAP are the authoritative conversion back to the audio timeline.  The
    note-derived fallback keeps the small, synthetic scorer tests useful when
    they construct ``ReferenceNote`` values without chart headers.
    """

    if headers is not None and headers.get("BPM"):
        bpm = _parse_number(headers["BPM"], "BPM")
        if bpm <= 0:
            raise ValueError("UltraStar reference must contain a positive #BPM")
        gap_seconds = _parse_number(headers.get("GAP", "0"), "GAP") / 1000.0
        return gap_seconds, 60.0 / (bpm * 4.0)

    ratios = [
        (note.end - note.start) / float(note.duration_beats)
        for note in notes
        if note.duration_beats > 0
        and math.isfinite(note.start)
        and math.isfinite(note.end)
        and note.end >= note.start
    ]
    seconds_per_beat = statistics.median(ratios) if ratios else 0.125
    gaps = [
        note.start - note.start_beat * seconds_per_beat
        for note in notes
        if math.isfinite(note.start)
    ]
    gap_seconds = statistics.median(gaps) if gaps else 0.0
    return gap_seconds, seconds_per_beat


def _seconds_to_chart_beat(
    seconds: float,
    *,
    gap_seconds: float,
    seconds_per_beat: float,
) -> float:
    return (seconds - gap_seconds) / seconds_per_beat


def _is_true(value: str | None) -> bool:
    return (value or "").strip().casefold() in {"1", "true", "yes", "on"}


def parse_ultrastar_reference(path: str | Path) -> UltraStarReference:
    """Parse the first voice of an UltraStar TXT file into word spans."""

    reference_path = Path(path).expanduser().resolve()
    if not reference_path.is_file():
        raise FileNotFoundError(f"Reference file does not exist: {reference_path}")

    lines = _read_chart_lines(reference_path)
    headers: dict[str, str] = {}
    for line in lines:
        if line.startswith("#"):
            key, separator, value = line[1:].partition(":")
            if separator:
                headers[key.strip().upper()] = value.strip()

    bpm = _parse_number(headers.get("BPM", "0"), "BPM")
    if bpm <= 0:
        raise ValueError("UltraStar reference must contain a positive #BPM")
    seconds_per_beat = 60.0 / (bpm * 4.0)
    gap_seconds = _parse_number(headers.get("GAP", "0"), "GAP") / 1000.0
    relative = _is_true(headers.get("RELATIVE"))
    language = normalize_language(headers.get("LANGUAGE", "en"))

    words: list[ReferenceWord] = []
    lyric_lines: list[str] = []
    phrase_words: list[str] = []
    current_parts: list[str] = []
    current_start: float | None = None
    current_end: float | None = None
    previous_start_beat = 0
    active_voice = 1

    def flush_word() -> None:
        nonlocal current_parts, current_start, current_end
        text = "".join(current_parts).strip()
        if text and current_start is not None and current_end is not None:
            if normalize_ultrastar_word_text(text, language):
                words.append(ReferenceWord(text=text, start=current_start, end=current_end))
                phrase_words.append(text)
        current_parts = []
        current_start = None
        current_end = None

    def flush_phrase() -> None:
        flush_word()
        if phrase_words:
            lyric_lines.append(" ".join(phrase_words))
            phrase_words.clear()

    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        if line.startswith("#"):
            continue
        if line.strip() == "E":
            flush_phrase()
            break
        if line.startswith("-"):
            flush_phrase()
            continue
        if line.startswith("P") and line[1:].strip().isdigit():
            flush_phrase()
            active_voice = int(line[1:].strip())
            continue
        if active_voice != 1:
            continue

        match = _NOTE_PATTERN.match(line)
        if match is None or match.group(1) not in _NOTE_TYPES:
            continue
        try:
            start_beat = int(match.group(2))
            duration_beat = int(match.group(3))
        except ValueError as exc:
            raise ValueError(f"Invalid note timing on line {line_number}: {line!r}") from exc
        if start_beat < 0 or duration_beat < 0:
            raise ValueError(f"Negative note timing on line {line_number}: {line!r}")
        if relative:
            start_beat = previous_start_beat + start_beat
        previous_start_beat = start_beat

        # The first whitespace character after the pitch is the required
        # field separator. Any additional leading whitespace is meaningful in
        # UltraStar: it marks a new word rather than a continuation note.
        raw_text = match.group(5)
        if raw_text and raw_text[0].isspace():
            raw_text = raw_text[1:]
        text = raw_text.strip()
        if not text:
            continue
        starts_new_word = (
            not current_parts
            or raw_text[:1].isspace()
            or (text.startswith("(") and bool(current_parts))
        )
        if starts_new_word:
            flush_word()
        current_parts.append(text)
        note_start = gap_seconds + start_beat * seconds_per_beat
        note_end = gap_seconds + (start_beat + duration_beat) * seconds_per_beat
        current_start = note_start if current_start is None else current_start
        current_end = note_end

    flush_phrase()
    if not words:
        raise ValueError(f"Reference contains no timed lyric words: {reference_path}")
    return UltraStarReference(
        path=reference_path,
        headers=headers,
        words=tuple(words),
        lyrics="\n".join(lyric_lines),
    )


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _reference_word_beat_span(
    word: ReferenceWord,
    headers: dict[str, str],
) -> tuple[float, float]:
    """Return a reference word's fractional UltraStar beat span."""

    bpm = _parse_number(headers.get("BPM", "0"), "BPM")
    gap_seconds = _parse_number(headers.get("GAP", "0"), "GAP") / 1000.0
    seconds_per_beat = 60.0 / (bpm * 4.0)
    return (
        (word.start - gap_seconds) / seconds_per_beat,
        (word.end - gap_seconds) / seconds_per_beat,
    )


def _pitch_notes_by_word(
    reference: UltraStarReference,
    pitch_notes: Sequence[ReferenceNote],
    language: str,
) -> list[list[ReferenceNote]]:
    """Assign chart note events to the timing-reference word they overlap.

    The timing and pitch charts can have different BPM/GAP values and can use
    different syllable boundary conventions. Both still describe the same
    UltraStar beat grid, so matching in beats avoids introducing a second
    timing-alignment problem into the pitch benchmark.
    """

    word_spans = [
        _reference_word_beat_span(word, reference.headers) for word in reference.words
    ]
    result: list[list[ReferenceNote]] = [[] for _ in reference.words]
    for note in pitch_notes:
        # A continuation marker (``~``) has no lexical normalization, but it
        # is still a real pitch event and must remain attached to its word.
        # Dropping it makes sustained or melismatic words look like a single
        # note and hides exactly the split pitches the exporter needs to keep.
        note_start = float(note.start_beat)
        note_end = note_start + max(float(note.duration_beats), 0.0)
        best_index: int | None = None
        best_overlap = 0.0
        for index, (word_start, word_end) in enumerate(word_spans):
            overlap = max(0.0, min(note_end, word_end) - max(note_start, word_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_index = index
        if best_index is not None and best_overlap > 0.0:
            result[best_index].append(note)
    return result


def _weighted_mode(weighted_values: Sequence[tuple[int, float]]) -> int | None:
    """Return the deterministic duration-weighted mode of integer values."""

    if not weighted_values:
        return None
    weights: dict[int, float] = {}
    first_seen: dict[int, int] = {}
    for order, (value, weight) in enumerate(weighted_values):
        if not math.isfinite(weight) or weight <= 0.0:
            continue
        weights[value] = weights.get(value, 0.0) + weight
        first_seen.setdefault(value, order)
    if not weights:
        return None
    return max(weights, key=lambda value: (weights[value], -first_seen[value]))


def _compare_pitch_note_events(
    reference: UltraStarReference,
    notes_by_word: Sequence[Sequence[ReferenceNote]],
    token_words: Sequence[object],
    alignments: Mapping[str, object],
    segments_by_token: Mapping[str, Sequence[Word]],
    *,
    language: str,
    pitch_octave_offset: int,
    timing_tolerance_beats: float = 1.0,
) -> dict[str, object]:
    """Compare individual chart notes without collapsing a word's melody.

    The legacy word metric remains useful for compatibility, but a duration
    weighted mode cannot tell whether ``fi~re`` was emitted as two pitches.
    This event metric greedily pairs each expected note with one overlapping
    generated segment and reports pitch and onset agreement separately.  Onset
    timing is measured in UltraStar chart beats (one beat of tolerance), not a
    fixed number of seconds.  Split-word correctness ignores text-fragment
    boundaries and permits extra generated syllable segments when they remain
    inside a reference event with the same pitch.
    """

    expected_count = 0
    predicted_count = 0
    correct_count = 0
    timing_count = 0
    split_word_count = 0
    split_word_correct_count = 0
    gap_seconds, seconds_per_beat = _chart_timing_grid(reference.headers)
    for token, expected_notes in zip(token_words, notes_by_word):
        if not expected_notes:
            continue
        token_id = str(getattr(token, "id", ""))
        alignment = alignments.get(token_id)
        status = str(getattr(alignment, "status", "missing")) if alignment else "missing"
        if alignment is None or status in {"missing", "unmatched"}:
            continue
        expected = sorted(expected_notes, key=lambda note: (note.start, note.end))
        predicted = sorted(
            [
                segment
                for segment in segments_by_token.get(token_id, ())
                if segment.pitch_hz > 0.0
                and segment.midi_note > 0
                and math.isfinite(segment.start)
                and math.isfinite(segment.end)
                and segment.end > segment.start
            ],
            key=lambda segment: (segment.start, segment.end),
        )
        if len(expected) > 1:
            split_word_count += 1
        used: set[int] = set()
        word_pitch_matches: list[bool] = []
        matched_pairs: list[tuple[ReferenceNote, Word]] = []
        for note in expected:
            expected_count += 1
            candidates = sorted(
                (
                    max(0.0, min(note.end, segment.end) - max(note.start, segment.start)),
                    index,
                    segment,
                )
                for index, segment in enumerate(predicted)
                if index not in used
            )
            if not candidates or candidates[-1][0] <= 0.0:
                word_pitch_matches.append(False)
                continue
            overlap, index, segment = candidates[-1]
            del overlap
            used.add(index)
            predicted_count += 1
            exact_pitch = segment.midi_note + pitch_octave_offset == note.pitch + 60
            correct_count += int(exact_pitch)
            word_pitch_matches.append(exact_pitch)
            matched_pairs.append((note, segment))
            segment_start_beat = _seconds_to_chart_beat(
                segment.start,
                gap_seconds=gap_seconds,
                seconds_per_beat=seconds_per_beat,
            )
            if (
                abs(segment_start_beat - float(note.start_beat))
                <= timing_tolerance_beats
            ):
                timing_count += 1
        if len(expected) > 1 and all(word_pitch_matches):
            # Text fragments are not the unit of musical identity: one
            # detector syllable may be represented by several chart rows, or
            # a chart may split a long word at a different consonant. Require
            # every expected event to have a correctly pitched overlap, but
            # allow additional generated segments when they stay inside an
            # expected event and carry that event's pitch.
            def predicted_segment_is_valid(segment: Word) -> bool:
                candidates = [
                    (
                        max(
                            0.0,
                            min(note.end, segment.end)
                            - max(note.start, segment.start),
                        ),
                        note,
                    )
                    for note in expected
                ]
                overlap, note = max(candidates, key=lambda item: item[0])
                return (
                    overlap > 0.0
                    and segment.midi_note + pitch_octave_offset == note.pitch + 60
                )

            all_generated_events_valid = bool(predicted) and all(
                predicted_segment_is_valid(segment) for segment in predicted
            )
            if matched_pairs and all_generated_events_valid:
                split_word_correct_count += 1

    return {
        "pitch_note_scored_count": expected_count,
        "pitch_note_predicted_count": predicted_count,
        "pitch_note_correct_count": correct_count,
        "pitch_note_accuracy": correct_count / expected_count if expected_count else 0.0,
        "pitch_note_coverage": predicted_count / expected_count if expected_count else 0.0,
        "pitch_note_timing_within_tolerance_count": timing_count,
        "pitch_note_timing_within_tolerance": (
            timing_count / expected_count if expected_count else 0.0
        ),
        "pitch_note_timing_tolerance_beats": timing_tolerance_beats,
        "pitch_split_word_count": split_word_count,
        "pitch_split_word_correct_count": split_word_correct_count,
        "pitch_split_word_accuracy": (
            split_word_correct_count / split_word_count if split_word_count else 0.0
        ),
    }


def _compare_global_pitch_note_events(
    pitch_notes: Sequence[ReferenceNote],
    pitch_words: Sequence[Word],
    *,
    pitch_octave_offset: int,
    chart_headers: Mapping[str, str] | None = None,
    timing_tolerance_beats: float = 1.0,
) -> dict[str, object]:
    """Score chart note events against generated segments by time alone.

    Word annotations are not a stable boundary: a chart may write a long
    vowel as one token, while another chart splits it into ``~`` fragments or
    uses a hyphen.  This scorer therefore pairs events globally by maximum
    temporal overlap and evaluates pitch/onset timing independently of token
    text.  Onset timing uses the chart's integer beat grid (± one beat).
    It is deliberately diagnostic; the production path never receives chart
    notes.
    """

    expected = sorted(
        [note for note in pitch_notes if note.end > note.start],
        key=lambda note: (note.start, note.end),
    )
    predicted = sorted(
        [
            segment
            for segment in pitch_words
            if segment.pitch_hz > 0.0
            and segment.midi_note > 0
            and math.isfinite(segment.start)
            and math.isfinite(segment.end)
            and segment.end > segment.start
        ],
        key=lambda segment: (segment.start, segment.end),
    )
    used: set[int] = set()
    matched = 0
    correct = 0
    timing = 0
    gap_seconds, seconds_per_beat = _chart_timing_grid(chart_headers, expected)
    for note in expected:
        candidates = [
            (
                max(0.0, min(note.end, segment.end) - max(note.start, segment.start)),
                -abs(segment.start - note.start),
                index,
                segment,
            )
            for index, segment in enumerate(predicted)
            if index not in used
        ]
        if not candidates:
            continue
        overlap, _distance, index, segment = max(candidates)
        if overlap <= 0.0:
            continue
        used.add(index)
        matched += 1
        correct += int(segment.midi_note + pitch_octave_offset == note.pitch + 60)
        segment_start_beat = _seconds_to_chart_beat(
            segment.start,
            gap_seconds=gap_seconds,
            seconds_per_beat=seconds_per_beat,
        )
        timing += int(
            abs(segment_start_beat - float(note.start_beat))
            <= timing_tolerance_beats
        )
    expected_count = len(expected)
    return {
        "pitch_global_note_scored_count": expected_count,
        "pitch_global_note_predicted_count": matched,
        "pitch_global_note_correct_count": correct,
        "pitch_global_note_accuracy": correct / expected_count if expected_count else 0.0,
        "pitch_global_note_coverage": matched / expected_count if expected_count else 0.0,
        "pitch_global_note_timing_within_tolerance_count": timing,
        "pitch_global_note_timing_within_tolerance": (
            timing / expected_count if expected_count else 0.0
        ),
        "pitch_global_note_timing_tolerance_beats": timing_tolerance_beats,
    }


def _build_pitch_note_diff(
    pitch_notes: Sequence[ReferenceNote],
    pitch_words: Sequence[Word],
    *,
    chart_headers: Mapping[str, str] | None = None,
    pitch_octave_offset: int = 0,
) -> dict[str, object]:
    """Build a per-note diff between the chart and generated pitch events.

    The generated app export rounds timestamps to integer UltraStar beats. The
    diff intentionally compares those rounded values, so it shows the exact
    lines a generated TXT would contain rather than hiding sub-beat detector
    differences in floating-point seconds.
    """

    expected = sorted(
        [note for note in pitch_notes if note.end > note.start],
        key=lambda note: (note.start, note.end),
    )
    predicted = sorted(
        [
            segment
            for segment in pitch_words
            if segment.pitch_hz > 0.0
            and segment.midi_note > 0
            and math.isfinite(segment.start)
            and math.isfinite(segment.end)
            and segment.end > segment.start
        ],
        key=lambda segment: (segment.start, segment.end),
    )
    gap_seconds, seconds_per_beat = _chart_timing_grid(chart_headers, expected)
    used: set[int] = set()
    rows: list[dict[str, object]] = []
    for note in expected:
        candidates = [
            (
                max(0.0, min(note.end, segment.end) - max(note.start, segment.start)),
                -abs(segment.start - note.start),
                index,
                segment,
            )
            for index, segment in enumerate(predicted)
            if index not in used
        ]
        if not candidates:
            candidates = []
        overlap, _distance, index, segment = max(candidates) if candidates else (0.0, 0.0, -1, None)
        expected_end_beat = note.start_beat + note.duration_beats
        expected_row = {
            "text": note.text,
            "note_type": note.note_type,
            "start_beat": note.start_beat,
            "duration_beats": note.duration_beats,
            "pitch": note.pitch,
        }
        if segment is None or overlap <= 0.0:
            rows.append(
                {
                    "status": "missing",
                    "expected": expected_row,
                    "predicted": None,
                    "overlap_seconds": 0.0,
                    "start_delta_beats": None,
                    "duration_delta_beats": None,
                    "pitch_delta": None,
                    "timing_exact": False,
                    "timing_within_tolerance": False,
                    "pitch_correct": False,
                }
            )
            continue
        used.add(index)
        predicted_start_beat = round(
            _seconds_to_chart_beat(
                segment.start,
                gap_seconds=gap_seconds,
                seconds_per_beat=seconds_per_beat,
            )
        )
        predicted_end_beat = round(
            _seconds_to_chart_beat(
                segment.end,
                gap_seconds=gap_seconds,
                seconds_per_beat=seconds_per_beat,
            )
        )
        predicted_duration_beats = max(1, predicted_end_beat - predicted_start_beat)
        predicted_pitch = segment.midi_note + pitch_octave_offset - 60
        start_delta = predicted_start_beat - note.start_beat
        duration_delta = predicted_duration_beats - note.duration_beats
        pitch_delta = predicted_pitch - note.pitch
        rows.append(
            {
                "status": "matched",
                "expected": expected_row,
                "predicted": {
                    "text": segment.text,
                    "note_type": ":",
                    "start_beat": predicted_start_beat,
                    "duration_beats": predicted_duration_beats,
                    "pitch": predicted_pitch,
                },
                "overlap_seconds": overlap,
                "start_delta_beats": start_delta,
                "duration_delta_beats": duration_delta,
                "pitch_delta": pitch_delta,
                "timing_exact": start_delta == 0 and duration_delta == 0,
                "timing_within_tolerance": abs(start_delta) <= 1,
                "pitch_correct": pitch_delta == 0,
            }
        )

    extras = []
    for index, segment in enumerate(predicted):
        if index in used:
            continue
        start_beat = round(
            _seconds_to_chart_beat(
                segment.start,
                gap_seconds=gap_seconds,
                seconds_per_beat=seconds_per_beat,
            )
        )
        end_beat = round(
            _seconds_to_chart_beat(
                segment.end,
                gap_seconds=gap_seconds,
                seconds_per_beat=seconds_per_beat,
            )
        )
        extras.append(
            {
                "text": segment.text,
                "start_beat": start_beat,
                "duration_beats": max(1, end_beat - start_beat),
                "pitch": segment.midi_note + pitch_octave_offset - 60,
            }
        )
    return {
        "expected_count": len(expected),
        "predicted_count": len(predicted),
        "matched_count": len(used),
        "extra_count": len(extras),
        "rows": rows,
        "extras": extras,
    }


def _pitch_words_for_response(
    request: AlignmentWorkerRequest,
    response: AlignmentWorkerResponse,
    times: Sequence[float],
    frequencies: Sequence[float],
    confidences: Sequence[float],
    energies: Sequence[float],
    candidate_midis: Sequence[Sequence[int]] | None = None,
    candidate_scores: Sequence[Sequence[float]] | None = None,
    onset_scores: Sequence[float] | None = None,
) -> list[Word]:
    """Convert worker spans to pitch-ready words and run the production detector."""

    alignments = {item.token_id: item for item in response.alignments}
    words: list[Word] = []
    for token in request.tokens:
        alignment = alignments.get(token.id)
        if alignment is None:
            continue
        if (
            alignment.start is None
            or alignment.end is None
            or not math.isfinite(alignment.start)
            or not math.isfinite(alignment.end)
            or alignment.end < alignment.start
        ):
            continue
        words.append(
            Word(
                text=token.display_text,
                start=alignment.start,
                end=alignment.end,
                confidence=alignment.confidence,
                token_id=token.id,
                alignment_status=alignment.status,
                alignment_run_id=alignment.alignment_run_id,
            )
        )
    segments, _ = assign_pitch_to_words(
        words,
        times,
        frequencies,
        confidences,
        energies,
        min_volume=DEFAULT_MIN_VOLUME_RMS,
        candidate_midis=(
            np.asarray(candidate_midis) if candidate_midis is not None else None
        ),
        candidate_scores=(
            np.asarray(candidate_scores) if candidate_scores is not None else None
        ),
        onset_scores=(
            np.asarray(onset_scores) if onset_scores is not None else None
        ),
    )
    return segments


def compare_pitch_reference(
    reference: UltraStarReference,
    pitch_notes: Sequence[ReferenceNote],
    response: AlignmentWorkerResponse,
    request: AlignmentWorkerRequest,
    pitch_words: Sequence[Word],
    *,
    worst_count: int = 10,
    pitch_octave_offset: int = 0,
) -> dict[str, object]:
    """Score exact word-level MIDI pitches against an UltraStar pitch chart.

    A word's expected and predicted pitch are duration-weighted modes. A
    missing or unvoiced prediction is deliberately not treated as C4: the
    production pitch fallback stores ``midi_note=60`` while ``pitch_hz=0``.
    """

    if worst_count < 0:
        raise ValueError("worst_count must be non-negative")
    if not isinstance(pitch_octave_offset, int):
        raise ValueError("pitch_octave_offset must be an integer")
    token_words = [
        token
        for token in request.tokens
        if normalize_lyric_token(token.display_text, request.language)
    ]
    if len(token_words) != len(reference.words):
        raise ValueError(
            "Reference/token count mismatch: "
            f"{len(reference.words)} reference words vs {len(token_words)} alignment tokens"
        )
    notes_by_word = _pitch_notes_by_word(reference, pitch_notes, request.language)
    assigned_note_count = sum(len(notes) for notes in notes_by_word)
    lexical_note_count = sum(
        1
        for notes in notes_by_word
        for note in notes
        if normalize_ultrastar_word_text(note.text, request.language)
    )
    alignments = {item.token_id: item for item in response.alignments}
    segments_by_token: dict[str, list[Word]] = {}
    for segment in pitch_words:
        segments_by_token.setdefault(segment.token_id, []).append(segment)

    scorable = 0
    predicted_count = 0
    correct_count = 0
    octave_equivalent_count = 0
    missing_prediction_count = 0
    expected_note_count = 0
    expected_lexical_note_count = 0
    voiced_pairs: list[tuple[int, int]] = []
    mismatches: list[dict[str, object]] = []
    for token, expected_word, expected_notes in zip(
        token_words, reference.words, notes_by_word
    ):
        if not expected_notes:
            continue
        alignment = alignments.get(token.id)
        status = alignment.status if alignment is not None else "missing"
        aligned = (
            alignment is not None
            and alignment.start is not None
            and alignment.end is not None
            and math.isfinite(alignment.start)
            and math.isfinite(alignment.end)
            and alignment.end >= alignment.start
            and status not in {"unmatched", "missing"}
        )
        if not aligned:
            continue
        scorable += 1
        expected_note_count += len(expected_notes)
        expected_lexical_note_count += sum(
            1
            for note in expected_notes
            if normalize_ultrastar_word_text(note.text, request.language)
        )
        expected_pitch = _weighted_mode(
            [(note.pitch, max(float(note.duration_beats), 1.0)) for note in expected_notes]
        )
        segments = segments_by_token.get(token.id, [])
        voiced_segments = [
            segment
            for segment in segments
            if (
                segment.pitch_hz > 0.0
                and math.isfinite(segment.pitch_hz)
                and segment.midi_note > 0
                and math.isfinite(segment.start)
                and math.isfinite(segment.end)
                and segment.end > segment.start
            )
        ]
        predicted_pitch = _weighted_mode(
            [
                (segment.midi_note + pitch_octave_offset, segment.end - segment.start)
                for segment in voiced_segments
            ]
        )
        expected_absolute = expected_pitch + 60 if expected_pitch is not None else None
        if predicted_pitch is None or expected_absolute is None:
            missing_prediction_count += 1
            mismatches.append(
                {
                    "text": expected_word.text,
                    "status": status,
                    "expected_pitch": expected_pitch,
                    "predicted_pitch": None,
                    "pitch_error": None,
                    "prediction": "unvoiced",
                }
            )
            continue
        predicted_count += 1
        voiced_pairs.append((expected_absolute, predicted_pitch - pitch_octave_offset))
        pitch_error = predicted_pitch - expected_absolute
        exact = predicted_pitch == expected_absolute
        octave_equivalent = predicted_pitch % 12 == expected_absolute % 12
        if exact:
            correct_count += 1
        if octave_equivalent:
            octave_equivalent_count += 1
        if not exact:
            mismatches.append(
                {
                    "text": expected_word.text,
                    "status": status,
                    "expected_pitch": expected_pitch,
                    "predicted_pitch": predicted_pitch - 60,
                    "pitch_error": pitch_error,
                    "prediction": "voiced",
                }
            )

    mismatches.sort(
        key=lambda item: (
            item["pitch_error"] is None,
            abs(float(item["pitch_error"])) if item["pitch_error"] is not None else math.inf,
        ),
        reverse=True,
    )
    best_octave_offset, best_octave_correct = max(
        (
            (
                offset,
                sum(
                    predicted + offset == expected
                    for expected, predicted in voiced_pairs
                ),
            )
            for offset in range(-24, 25, 12)
        ),
        key=lambda item: (item[1], -abs(item[0]), -item[0]),
        default=(0, 0),
    )
    note_metrics = _compare_pitch_note_events(
        reference,
        notes_by_word,
        token_words,
        alignments,
        segments_by_token,
        language=request.language,
        pitch_octave_offset=pitch_octave_offset,
    )
    global_note_metrics = _compare_global_pitch_note_events(
        pitch_notes,
        pitch_words,
        pitch_octave_offset=pitch_octave_offset,
        chart_headers=reference.headers,
    )
    return {
        "pitch_reference": "provided" if pitch_notes else "empty",
        "pitch_reference_note_count": len(pitch_notes),
        "pitch_reference_assigned_note_count": assigned_note_count,
        "pitch_reference_lexical_note_count": lexical_note_count,
        "pitch_scored_lexical_note_count": expected_lexical_note_count,
        "pitch_scored_note_count": expected_note_count,
        "pitch_scorable_word_count": scorable,
        "pitch_predicted_word_count": predicted_count,
        "pitch_correct_word_count": correct_count,
        "pitch_accuracy": correct_count / scorable if scorable else 0.0,
        "pitch_octave_offset": pitch_octave_offset,
        "pitch_best_octave_offset": best_octave_offset,
        "pitch_best_octave_correct_word_count": best_octave_correct,
        "pitch_best_octave_accuracy": (
            best_octave_correct / scorable if scorable else 0.0
        ),
        "pitch_coverage": predicted_count / scorable if scorable else 0.0,
        "pitch_octave_equivalent_accuracy": (
            octave_equivalent_count / scorable if scorable else 0.0
        ),
        "pitch_missing_prediction_count": missing_prediction_count,
        "pitch_worst": mismatches[:worst_count],
        **note_metrics,
        **global_note_metrics,
    }


def compare_reference(
    reference: UltraStarReference,
    response: AlignmentWorkerResponse,
    request: AlignmentWorkerRequest,
    *,
    tolerance_ms: float = 100.0,
    timing_tolerance_beats: float | None = None,
    worst_count: int = 10,
    pitch_notes: Sequence[ReferenceNote] | None = None,
    pitch_words: Sequence[Word] | None = None,
    pitch_octave_offset: int = 0,
) -> dict[str, object]:
    """Compare generated canonical spans with reference word boundaries."""

    if not math.isfinite(tolerance_ms) or tolerance_ms < 0:
        raise ValueError("tolerance_ms must be finite and non-negative")
    if timing_tolerance_beats is not None and (
        not math.isfinite(timing_tolerance_beats) or timing_tolerance_beats < 0
    ):
        raise ValueError("timing_tolerance_beats must be finite and non-negative")
    if worst_count < 0:
        raise ValueError("worst_count must be non-negative")
    alignments = {item.token_id: item for item in response.alignments}
    token_words = [
        token
        for token in request.tokens
        if normalize_lyric_token(token.display_text, request.language)
    ]
    if len(token_words) != len(reference.words):
        raise ValueError(
            "Reference/token count mismatch: "
            f"{len(reference.words)} reference words vs {len(token_words)} alignment tokens"
        )

    boundary_errors: list[dict[str, object]] = []
    start_errors: list[float] = []
    end_errors: list[float] = []
    timed_count = 0
    status_counts: Counter[str] = Counter()
    if timing_tolerance_beats is None:
        tolerance_seconds = tolerance_ms / 1000.0
        effective_tolerance_beats = None
        seconds_per_ultrastar_beat = None
    else:
        _gap_seconds, seconds_per_ultrastar_beat = _chart_timing_grid(
            reference.headers,
        )
        tolerance_seconds = timing_tolerance_beats * seconds_per_ultrastar_beat
        effective_tolerance_beats = timing_tolerance_beats
    effective_tolerance_ms = tolerance_seconds * 1000.0

    for token, expected in zip(token_words, reference.words):
        alignment = alignments.get(token.id)
        status = alignment.status if alignment is not None else "missing"
        status_counts[status] += 1
        start = alignment.start if alignment is not None else None
        end = alignment.end if alignment is not None else None
        if start is None or end is None or not math.isfinite(start) or not math.isfinite(end):
            boundary_errors.append(
                {
                    "text": expected.text,
                    "status": status,
                    "reference_start": expected.start,
                    "reference_end": expected.end,
                    "start_error_ms": None,
                    "end_error_ms": None,
                    "boundary_error_ms": None,
                    "start_error_beats": None,
                    "end_error_beats": None,
                    "boundary_error_beats": None,
                }
            )
            continue
        timed_count += 1
        start_error = abs(start - expected.start)
        end_error = abs(end - expected.end)
        boundary_error = (start_error + end_error) / 2.0
        start_error_beats = (
            start_error / seconds_per_ultrastar_beat
            if seconds_per_ultrastar_beat is not None
            else None
        )
        end_error_beats = (
            end_error / seconds_per_ultrastar_beat
            if seconds_per_ultrastar_beat is not None
            else None
        )
        boundary_error_beats = (
            boundary_error / seconds_per_ultrastar_beat
            if seconds_per_ultrastar_beat is not None
            else None
        )
        start_errors.append(start_error)
        end_errors.append(end_error)
        boundary_errors.append(
            {
                "text": expected.text,
                "status": status,
                "reference_start": expected.start,
                "reference_end": expected.end,
                "predicted_start": start,
                "predicted_end": end,
                "start_error_ms": start_error * 1000.0,
                "end_error_ms": end_error * 1000.0,
                "boundary_error_ms": boundary_error * 1000.0,
                "start_error_beats": start_error_beats,
                "end_error_beats": end_error_beats,
                "boundary_error_beats": boundary_error_beats,
                "within_tolerance": max(start_error, end_error) <= tolerance_seconds,
            }
        )

    timed_errors = [
        item for item in boundary_errors if item["boundary_error_ms"] is not None
    ]
    timed_errors.sort(key=lambda item: float(item["boundary_error_ms"]), reverse=True)
    total = len(reference.words)
    within_tolerance = sum(1 for item in timed_errors if item.get("within_tolerance"))
    report: dict[str, object] = {
        "reference": reference.path.name,
        "reference_word_count": total,
        "timed_word_count": timed_count,
        "coverage": timed_count / total if total else 0.0,
        "status_counts": dict(status_counts),
        # ``within_tolerance`` is the primary timing score. Keep the
        # millisecond diagnostics, but report the effective threshold too so
        # a beat-based run cannot be mistaken for the legacy fixed-ms score.
        "tolerance_ms": effective_tolerance_ms,
        "timing_tolerance_ms": effective_tolerance_ms,
        "timing_tolerance_beats": effective_tolerance_beats,
        "timing_tolerance_seconds": tolerance_seconds,
        "seconds_per_ultrastar_beat": seconds_per_ultrastar_beat,
        "within_tolerance": within_tolerance / total if total else 0.0,
        "start_mae_ms": statistics.fmean(start_errors) * 1000.0 if start_errors else None,
        "end_mae_ms": statistics.fmean(end_errors) * 1000.0 if end_errors else None,
        "start_p90_ms": _percentile(start_errors, 0.9) * 1000.0,
        "end_p90_ms": _percentile(end_errors, 0.9) * 1000.0,
        "boundary_mae_ms": statistics.fmean(
            [float(item["boundary_error_ms"]) for item in timed_errors]
        ) if timed_errors else None,
        "worst": timed_errors[:worst_count],
        "diagnostics": response.diagnostics,
    }
    if pitch_notes is not None:
        if pitch_words is None:
            raise ValueError("pitch_words are required when pitch_notes are provided")
        report.update(
            compare_pitch_reference(
                reference,
                pitch_notes,
                response,
                request,
                pitch_words,
                worst_count=worst_count,
                pitch_octave_offset=pitch_octave_offset,
            )
        )
    return report


def _build_request(
    reference: UltraStarReference,
    audio: Path,
    vocal_audio: Path | None,
    timing_grid: tuple[float, float] | None = None,
) -> AlignmentWorkerRequest:
    revision_id = f"benchmark-{uuid.uuid4().hex}"
    revision = LyricRevision(
        id=revision_id,
        song_id="private-benchmark",
        language=reference.headers.get("LANGUAGE", "en"),
        source="manual",
        raw_text=reference.lyrics,
    )
    revision.tokens = tokenize_lyrics(reference.lyrics, revision.id, revision.language)
    meaningful_words = [
        word
        for word in reference.words
        if normalize_lyric_token(word.text, revision.language)
    ]
    if len(revision.tokens) != len(meaningful_words):
        raise ValueError(
            "Reference/token count mismatch before alignment: "
            f"{len(meaningful_words)} reference words vs {len(revision.tokens)} tokens"
        )
    return build_worker_request(
        job_id="private-benchmark",
        revision=revision,
        alignment_run_id=f"run-{uuid.uuid4().hex}",
        original_audio_ref=str(audio),
        vocal_stem_audio_ref=str(vocal_audio or audio),
        timing_grid=timing_grid,
    )


def _evidence_cache_payload(
    evidence: dict[str, list[ASRToken]],
    source_errors: dict[str, str],
) -> dict[str, object]:
    return {
        "version": 1,
        "evidence": {
            source: [asdict(token) for token in tokens]
            for source, tokens in evidence.items()
        },
        "source_errors": source_errors,
    }


def _write_evidence_cache(
    path: Path,
    evidence: dict[str, list[ASRToken]],
    source_errors: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_evidence_cache_payload(evidence, source_errors), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _read_evidence_cache(path: Path) -> tuple[dict[str, list[ASRToken]], dict[str, str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read evidence cache {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError(f"Unsupported evidence cache format: {path}")
    raw_evidence = payload.get("evidence")
    raw_errors = payload.get("source_errors", {})
    if not isinstance(raw_evidence, dict) or not isinstance(raw_errors, dict):
        raise ValueError(f"Malformed evidence cache: {path}")
    evidence: dict[str, list[ASRToken]] = {}
    try:
        for source, raw_tokens in raw_evidence.items():
            if not isinstance(source, str) or not isinstance(raw_tokens, list):
                raise ValueError("evidence sources must be string keys with token arrays")
            evidence[source] = [ASRToken(**token) for token in raw_tokens]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Malformed token in evidence cache {path}: {exc}") from exc
    return evidence, {str(key): str(value) for key, value in raw_errors.items()}


def _timing_reference_evidence(
    request: AlignmentWorkerRequest,
    path: Path,
) -> list[ASRToken]:
    """Load an explicit chart as validated alignment evidence for a benchmark."""

    chart_words = parse_ultrastar_word_timings(path)
    if len(chart_words) != len(request.tokens):
        raise ValueError(
            "Timing reference/token count mismatch: "
            f"{len(chart_words)} chart words vs {len(request.tokens)} lyric tokens"
        )
    for index, (token, chart_word) in enumerate(zip(request.tokens, chart_words)):
        if normalize_lyric_token(chart_word.text, request.language) != token.normalized_text:
            raise ValueError(
                f"Timing reference text mismatch at word {index}: "
                f"lyric={token.display_text!r}, chart={chart_word.text!r}"
            )
    return [
        ASRToken(
            text=chart_word.text,
            start=chart_word.start,
            end=chart_word.end,
            confidence=1.0,
            model="ultrastar-chart",
            audio_source="ultrastar_chart",
            evidence_type="chart",
            sequence_index=index,
        )
        for index, chart_word in enumerate(chart_words)
    ]


def _run_once(
    request: AlignmentWorkerRequest,
    evidence: dict[str, list[ASRToken]] | None = None,
    source_errors: dict[str, str] | None = None,
    activity_spans: list[tuple[float, float]] | None = None,
    activity_times: Sequence[float] | None = None,
    activity_values: Sequence[float | bool] | None = None,
    activity_window_end: float | None = None,
    return_canonical: bool = False,
) -> AlignmentWorkerResponse | tuple[AlignmentWorkerResponse, AlignmentWorkerResponse]:
    if evidence is None:
        evidence, source_errors = collect_model_evidence(request)
    source_errors = source_errors or {}
    if not any(evidence.values()):
        raise RuntimeError(f"No usable alignment evidence: {source_errors}")
    response = run_local_alignment(
        request,
        evidence,
        activity_spans=activity_spans or request.activity_spans,
        activity_times=activity_times if activity_times is not None else (),
        activity_values=activity_values if activity_values is not None else (),
    )
    canonical_response = response
    if activity_times is not None and activity_values is not None:
        expanded = expand_alignment_sustains(
            alignment_response_to_models(response),
            activity_times,
            activity_values,
            clear_gap_s=0.12,
            activity_threshold=alignment_activity_threshold(activity_values),
            window_end=activity_window_end,
            melody_outputs=evidence,
            melody_start_gap_threshold_s=0.08,
            melody_start_tolerance_s=0.08,
            melody_start_offset_s=0.08,
            melody_start_min_confidence=0.55,
            melody_handoff_min_gap_s=0.08,
            melody_handoff_max_gap_s=0.25,
            melody_handoff_offset_s=0.08,
        )
        expanded = repair_melody_handoffs(
            expanded,
            evidence,
            tokens=request.tokens,
            anchor_outputs=evidence,
        )
        response = response.model_copy(
            update={"alignments": [_alignment_to_payload(alignment) for alignment in expanded]}
        )
    response.diagnostics["source_errors"] = source_errors
    if return_canonical:
        return response, canonical_response
    return response


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path, help="Private UltraStar .txt reference")
    parser.add_argument("--audio", required=True, type=Path, help="Full-mix audio matching the reference")
    parser.add_argument("--vocal-audio", type=Path, help="Optional vocal stem; defaults to --audio")
    parser.add_argument(
        "--activity-audio",
        type=Path,
        help="Optional vocal audio for the same activity/sustain post-processing used by the app",
    )
    parser.add_argument("--repeat", type=int, default=1, help="Run the alignment this many times")
    parser.add_argument("--tolerance-ms", type=float, default=100.0)
    parser.add_argument(
        "--timing-tolerance-beats",
        type=float,
        default=1.0,
        help="word timing tolerance in UltraStar chart beats (default: ±1)",
    )
    parser.add_argument("--worst", type=int, default=10, help="Number of worst words to print")
    parser.add_argument(
        "--evidence-cache",
        type=Path,
        help="Read/write model evidence JSON so selector tuning does not rerun models",
    )
    parser.add_argument(
        "--refresh-evidence",
        action="store_true",
        help="Ignore an existing --evidence-cache and collect model evidence again",
    )
    parser.add_argument(
        "--timing-reference",
        type=Path,
        help=(
            "Use this matching UltraStar TXT as an explicit timing source. "
            "This is a reference-backed upper-bound run, not a model-only score."
        ),
    )
    parser.add_argument(
        "--pitch-reference",
        type=Path,
        help=(
            "Optional UltraStar TXT chart with pitch-bearing notes. Notes are scored "
            "against the timing-reference words by beat overlap."
        ),
    )
    parser.add_argument(
        "--min-within-tolerance",
        type=float,
        help="Exit non-zero unless the best iteration reaches this fraction",
    )
    parser.add_argument(
        "--min-pitch-accuracy",
        type=float,
        help="Exit non-zero unless the best iteration reaches this exact-pitch fraction",
    )
    parser.add_argument(
        "--pitch-octave-offset",
        type=int,
        default=0,
        help="Diagnostic semitone offset applied to generated MIDI pitches (default: 0)",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.repeat < 1:
        raise SystemExit("--repeat must be at least 1")
    if not math.isfinite(args.tolerance_ms) or args.tolerance_ms < 0:
        raise SystemExit("--tolerance-ms must be finite and non-negative")
    if not math.isfinite(args.timing_tolerance_beats) or args.timing_tolerance_beats < 0:
        raise SystemExit("--timing-tolerance-beats must be finite and non-negative")
    if args.worst < 0:
        raise SystemExit("--worst must be non-negative")
    if args.min_within_tolerance is not None and not 0.0 <= args.min_within_tolerance <= 1.0:
        raise SystemExit("--min-within-tolerance must be between 0 and 1")
    if args.min_pitch_accuracy is not None and not 0.0 <= args.min_pitch_accuracy <= 1.0:
        raise SystemExit("--min-pitch-accuracy must be between 0 and 1")
    audio = args.audio.expanduser().resolve()
    vocal_audio = args.vocal_audio.expanduser().resolve() if args.vocal_audio else None
    activity_audio = args.activity_audio.expanduser().resolve() if args.activity_audio else None
    pitch_reference_path = (
        args.pitch_reference.expanduser().resolve() if args.pitch_reference else None
    )
    if not audio.is_file():
        raise SystemExit(f"Audio file does not exist: {audio}")
    if vocal_audio is not None and not vocal_audio.is_file():
        raise SystemExit(f"Vocal audio file does not exist: {vocal_audio}")
    if activity_audio is not None and not activity_audio.is_file():
        raise SystemExit(f"Activity audio file does not exist: {activity_audio}")
    if pitch_reference_path is not None and not pitch_reference_path.is_file():
        raise SystemExit(f"Pitch reference file does not exist: {pitch_reference_path}")

    pitch_times: Sequence[float] | None = None
    pitch_frequencies: Sequence[float] | None = None
    pitch_confidences: Sequence[float] | None = None
    pitch_energies: Sequence[float] | None = None
    pitch_candidate_midis: Sequence[Sequence[int]] | None = None
    pitch_candidate_scores: Sequence[Sequence[float]] | None = None
    pitch_onset_scores: Sequence[float] | None = None
    activity_times: Sequence[float] | None = None
    activity_values: Sequence[float | bool] | None = None
    activity_spans: list[tuple[float, float]] | None = None
    activity_window_end: float | None = None
    timing_grid: tuple[float, float] | None = None
    if activity_audio is not None:
        detection = detect_pitch(
            str(activity_audio),
            str(audio),
        )
        pitch_times, pitch_frequencies, pitch_confidences, pitch_energies = detection
        pitch_candidate_midis = getattr(detection, "candidate_midis", None)
        pitch_candidate_scores = getattr(detection, "candidate_scores", None)
        pitch_onset_scores = getattr(detection, "onset_scores", None)
        activity_times = pitch_times
        activity_values = voiced_activity_signal(pitch_frequencies, pitch_energies)
        activity_threshold = alignment_activity_threshold(activity_values)
        activity_spans = activity_spans_from_signal(
            activity_times,
            activity_values,
            threshold=activity_threshold,
        )
        try:
            import soundfile as sf

            activity_window_end = float(sf.info(str(activity_audio)).duration)
        except Exception:
            activity_window_end = None
    timing_grid = (
        estimate_timing_grid(str(audio))
        if os.environ.get("ALIGNMENT_ENABLE_TIMING_GRID", "").strip() == "1"
        else None
    )

    reference = parse_ultrastar_reference(args.reference)
    pitch_notes: tuple[ReferenceNote, ...] | None = None
    if pitch_reference_path is not None:
        _pitch_headers, pitch_notes = parse_ultrastar_pitch_notes(pitch_reference_path)
        if pitch_times is None:
            pitch_audio = vocal_audio or audio
            detection = detect_pitch(
                str(pitch_audio),
                str(audio),
            )
            pitch_times, pitch_frequencies, pitch_confidences, pitch_energies = detection
            pitch_candidate_midis = getattr(detection, "candidate_midis", None)
            pitch_candidate_scores = getattr(detection, "candidate_scores", None)
            pitch_onset_scores = getattr(detection, "onset_scores", None)
    evidence: dict[str, list[ASRToken]] | None = None
    source_errors: dict[str, str] | None = None
    if args.evidence_cache and args.evidence_cache.exists() and not args.refresh_evidence:
        evidence, source_errors = _read_evidence_cache(args.evidence_cache)
    elif args.evidence_cache:
        request_for_cache = _build_request(reference, audio, vocal_audio, timing_grid)
        evidence, source_errors = asyncio.run(
            asyncio.to_thread(collect_model_evidence, request_for_cache)
        )
        _write_evidence_cache(args.evidence_cache, evidence, source_errors)

    if args.timing_reference:
        request_for_reference = _build_request(reference, audio, vocal_audio, timing_grid)
        evidence = dict(evidence or {})
        evidence["ultrastar_chart"] = _timing_reference_evidence(
            request_for_reference,
            args.timing_reference.expanduser().resolve(),
        )
        source_errors = dict(source_errors or {})
        source_errors.pop("ultrastar_chart", None)

    reports: list[dict[str, object]] = []
    for iteration in range(1, args.repeat + 1):
        request = _build_request(reference, audio, vocal_audio, timing_grid)
        run_result = asyncio.run(
            asyncio.to_thread(
                _run_once,
                request,
                evidence,
                source_errors,
                activity_spans,
                activity_times,
                activity_values,
                activity_window_end,
                bool(activity_times is not None and activity_values is not None),
            )
        )
        if isinstance(run_result, tuple):
            response, canonical_response = run_result
        else:
            response = run_result
            canonical_response = None
        pitch_words = None
        canonical_pitch_words = None
        if pitch_notes is not None:
            if (
                pitch_times is None
                or pitch_frequencies is None
                or pitch_confidences is None
                or pitch_energies is None
            ):
                raise RuntimeError("pitch reference requested but pitch detection did not run")
            pitch_words = _pitch_words_for_response(
                request,
                response,
                pitch_times,
                pitch_frequencies,
                pitch_confidences,
                pitch_energies,
                pitch_candidate_midis,
                pitch_candidate_scores,
                pitch_onset_scores,
            )
            if canonical_response is not None:
                canonical_pitch_words = _pitch_words_for_response(
                    request,
                    canonical_response,
                    pitch_times,
                    pitch_frequencies,
                    pitch_confidences,
                    pitch_energies,
                    pitch_candidate_midis,
                    pitch_candidate_scores,
                    pitch_onset_scores,
                )
        # An explicit chart is authoritative for word boundaries. Activity
        # expansion is still useful for pitch coverage, but it must not make
        # the timing report claim that a chart end moved. Score boundaries
        # from the canonical response and pitch from the expanded response.
        # Timing is scored from the canonical model alignment. Activity
        # expansion is a sustain/display projection and can legitimately
        # extend word ends, so using it here would make model-only timing
        # scores disagree with the suite and with the worker's canonical
        # output.
        timing_response = canonical_response or response
        if canonical_response is None:
            report = compare_reference(
                reference,
                response,
                request,
                tolerance_ms=args.tolerance_ms,
                timing_tolerance_beats=args.timing_tolerance_beats,
                worst_count=args.worst,
                pitch_notes=pitch_notes,
                pitch_words=pitch_words,
                pitch_octave_offset=args.pitch_octave_offset,
            )
        else:
            report = compare_reference(
                reference,
                timing_response,
                request,
                tolerance_ms=args.tolerance_ms,
                timing_tolerance_beats=args.timing_tolerance_beats,
                worst_count=args.worst,
            )
            if pitch_notes is not None:
                report.update(
                    compare_pitch_reference(
                        reference,
                        pitch_notes,
                        response,
                        request,
                        pitch_words or (),
                        worst_count=args.worst,
                        pitch_octave_offset=args.pitch_octave_offset,
                    )
                )
        if canonical_response is not None and pitch_notes is not None:
            canonical_report = compare_pitch_reference(
                reference,
                pitch_notes,
                canonical_response,
                request,
                canonical_pitch_words or (),
                worst_count=args.worst,
                pitch_octave_offset=args.pitch_octave_offset,
            )
            report["pitch_scoring_span"] = "activity-expanded"
            report["pitch_canonical_accuracy"] = canonical_report["pitch_accuracy"]
            report["pitch_canonical_coverage"] = canonical_report["pitch_coverage"]
            report["pitch_canonical_correct_word_count"] = canonical_report[
                "pitch_correct_word_count"
            ]
            report["pitch_canonical_scorable_word_count"] = canonical_report[
                "pitch_scorable_word_count"
            ]
            report["pitch_canonical_predicted_word_count"] = canonical_report[
                "pitch_predicted_word_count"
            ]
        report["iteration"] = iteration
        reports.append(report)

    if args.json:
        print(json.dumps(reports, ensure_ascii=False, indent=2))
    else:
        for report in reports:
            print(
                f"iteration {report['iteration']}: "
                f"coverage={report['coverage']:.1%}, "
                f"within_{args.timing_tolerance_beats:g}beats="
                f"{report['within_tolerance']:.1%}, "
                f"start_mae={report['start_mae_ms'] or 0:.1f}ms, "
                f"end_mae={report['end_mae_ms'] or 0:.1f}ms"
            )
            if "pitch_accuracy" in report:
                print(
                    f"  pitch_accuracy={report['pitch_accuracy']:.1%}, "
                    f"pitch_coverage={report['pitch_coverage']:.1%}, "
                    f"exact={report['pitch_correct_word_count']}/"
                    f"{report['pitch_scorable_word_count']}, "
                    f"best_octave={report['pitch_best_octave_offset']:+d}st "
                    f"({report['pitch_best_octave_accuracy']:.1%})"
                )
                if "pitch_canonical_accuracy" in report:
                    print(
                        f"  pitch_canonical_accuracy={report['pitch_canonical_accuracy']:.1%}, "
                        f"exact={report['pitch_canonical_correct_word_count']}/"
                        f"{report['pitch_canonical_scorable_word_count']}"
                    )
            for item in report["worst"]:
                print(
                    f"  {item['text']!r}: "
                    f"reference {item['reference_start']:.3f}-{item['reference_end']:.3f}, "
                    f"predicted {item['predicted_start']:.3f}-{item['predicted_end']:.3f}, "
                    f"error {item['boundary_error_ms']:.1f}ms"
                )
            if "pitch_worst" in report:
                for item in report["pitch_worst"]:
                    print(
                        f"  pitch {item['text']!r}: "
                        f"expected {item['expected_pitch']}, "
                        f"predicted {item['predicted_pitch']} "
                        f"({item['prediction']})"
                    )
    if args.min_within_tolerance is not None:
        best = max((float(report["within_tolerance"]) for report in reports), default=0.0)
        if best < args.min_within_tolerance:
            print(
                f"benchmark gate failed: best within_{args.timing_tolerance_beats:g}beats="
                f"{best:.1%}, "
                f"required={args.min_within_tolerance:.1%}",
                file=sys.stderr,
            )
            return 1
    if args.min_pitch_accuracy is not None:
        # When activity expansion is enabled, production pitch assignment uses
        # the canonical (pre-expansion) spans. Gate that score so the benchmark
        # does not reject the same pitch source the API exports.
        best = max(
            (
                float(
                    report.get(
                        "pitch_canonical_accuracy",
                        report.get("pitch_accuracy", 0.0),
                    )
                )
                for report in reports
            ),
            default=0.0,
        )
        if best < args.min_pitch_accuracy:
            print(
                f"benchmark gate failed: best canonical pitch_accuracy={best:.1%}, "
                f"required={args.min_pitch_accuracy:.1%}",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
