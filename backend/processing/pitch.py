"""Pitch detection using Basic Pitch with a pYIN fallback for vocal analysis."""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np

from models import Word
from processing.native_pitch import load_mono
from processing.rust_utility import call as rust_utility


log = logging.getLogger("pitch")

# Default frame gates shared with the alignment worker and benchmarks.
DEFAULT_MIN_VOLUME_RMS = 0.005
DEFAULT_MIN_CONFIDENCE = 0.0


@dataclass(slots=True)
class PitchDetection:
    """Pitch contour plus optional per-frame Basic Pitch alternatives.

    The first four fields intentionally remain iterable as the historical
    ``(times, frequencies, confidences, energies)`` result.  Candidate notes
    are kept private to the assignment stage, where aligned lyric spans make
    a word-level choice possible without changing the public API used by
    callers that only need the contour.
    """

    times: np.ndarray
    frequencies: np.ndarray
    confidences: np.ndarray
    energies: np.ndarray
    candidate_midis: np.ndarray | None = None
    candidate_scores: np.ndarray | None = None
    # Basic Pitch's onset lane is reduced to one strength value per frame
    # after inference.  Keeping this compact signal (rather than the full
    # 88-bin onset matrix) lets assignment refine internal note boundaries
    # without retaining another model-sized tensor in the API process.
    onset_scores: np.ndarray | None = None

    def __iter__(self):
        yield self.times
        yield self.frequencies
        yield self.confidences
        yield self.energies


def detect_pitch(
    vocals_path: str,
    accompaniment_path: str | None = None,
) -> PitchDetection:
    frames = rust_utility(
        "detect_pitch", backend=str(Path(__file__).resolve().parents[1]),
        vocals=vocals_path, accompaniment=accompaniment_path,
    )
    for key in ("times", "frequencies", "confidences", "energies"):
        frames[key] = np.asarray(frames[key], dtype=np.float64)
    for key, dtype in (("candidate_midis", np.int16), ("candidate_scores", np.float32), ("onset_scores", np.float32)):
        if frames[key] is not None:
            frames[key] = np.asarray(frames[key], dtype=dtype)
    return PitchDetection(**frames)


def voiced_activity_signal(
    frequencies: np.ndarray,
    energies: np.ndarray,
) -> np.ndarray:
    """Return RMS activity only for frames that pYIN marked as voiced.

    RMS alone remains non-zero through bleed, breaths, and instrumental
    sections of a vocal stem.  Alignment uses this signal for sustain
    expansion and bounded fallback; pitch assignment continues to receive
    the original RMS array so its existing volume threshold is unchanged.
    """

    frequency_values = np.asarray(frequencies, dtype=np.float64)
    energy_values = np.asarray(energies, dtype=np.float64)
    length = min(len(frequency_values), len(energy_values))
    result = np.zeros(length, dtype=np.float64)
    voiced = np.isfinite(frequency_values[:length]) & (frequency_values[:length] > 0.0)
    finite_energy = np.nan_to_num(energy_values[:length], nan=0.0, posinf=0.0, neginf=0.0)
    result[voiced] = np.maximum(0.0, finite_energy[voiced])
    return result


def alignment_activity_threshold(activity: np.ndarray) -> float:
    """Choose a conservative, track-relative threshold for alignment activity."""

    positive = np.asarray(activity, dtype=np.float64)
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    if len(positive) == 0:
        return DEFAULT_MIN_VOLUME_RMS
    # The lower quartile of voiced RMS is a useful noise floor for separated
    # stems, but sitting exactly on that quartile lets low-level bleed bridge
    # neighboring words. A modest margin keeps quiet recordings usable while
    # making detected vocal islands more distinct.
    try:
        maximum_threshold = float(
            os.environ.get("ALIGNMENT_ACTIVITY_MAX_THRESHOLD", "0.06")
        )
    except (TypeError, ValueError):
        maximum_threshold = 0.06
    if not np.isfinite(maximum_threshold) or maximum_threshold <= 0.0:
        maximum_threshold = 0.06
    maximum_threshold = max(DEFAULT_MIN_VOLUME_RMS, maximum_threshold)
    return float(
        np.clip(
            np.percentile(positive, 25) * 1.25,
            DEFAULT_MIN_VOLUME_RMS,
            maximum_threshold,
        )
    )


def estimate_timing_grid(audio_path: str) -> tuple[float, float] | None:
    """Estimate a beat subdivision from audio for boundary tie-breaking.

    The grid is an audio-derived prior, not a lyric or chart source. A
    conservative eighth-note subdivision is useful for word boundaries while
    avoiding the over-fitting that comes from snapping directly to every
    detected onset. Callers should treat a missing grid as normal for short,
    noisy, or untrackable recordings.
    """

    try:
        sample_rate = 22050
        audio = load_mono(audio_path, sample_rate)
        if len(audio) < sample_rate:
            return None
        tempo, beat_frames = librosa.beat.beat_track(
            y=audio,
            sr=sample_rate,
            trim=False,
        )
        tempo_values = np.asarray(tempo, dtype=np.float64).reshape(-1)
        beats = librosa.frames_to_time(
            np.asarray(beat_frames, dtype=np.int64).reshape(-1),
            sr=sample_rate,
        )
        if len(tempo_values) == 0 or len(beats) < 4:
            return None
        bpm = float(tempo_values[0])
        if not np.isfinite(bpm) or bpm <= 0.0:
            return None
        beat_period = 60.0 / bpm
        step = beat_period / 8.0
        offset = float(beats[0])
        if (
            not np.isfinite(step)
            or not np.isfinite(offset)
            or step <= 0.02
            or step > 1.0
            or offset < 0.0
        ):
            return None
        log.info(
            "Timing grid: bpm=%.2f subdivision=1/8 step=%.4fs offset=%.4fs",
            bpm,
            step,
            offset,
        )
        return step, offset
    except (OSError, ValueError, RuntimeError) as exc:
        log.info("Timing grid unavailable for %s: %s", audio_path, exc)
        return None


def hz_to_midi(hz: float) -> int:
    if hz <= 0:
        return 0
    return int(round(69 + 12 * np.log2(hz / 440.0)))


def _lyric_boundaries(texts: list[str]) -> list[dict[str, list[int]]]:
    return rust_utility("pitch_text_boundaries", texts=texts)


def _pitch_text_boundaries(text: str) -> list[int]:
    return _lyric_boundaries([text])[0]["chart"]


def _finite_json(values: np.ndarray) -> list:
    finite = np.isfinite(values)
    return (values if finite.all() else np.where(finite, values, None)).tolist()


def _candidate_beam(midis: np.ndarray, scores: np.ndarray) -> dict | None:
    midis = np.asarray(midis, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    if midis.ndim != 2 or scores.shape != midis.shape:
        return None
    return {"midis": midis.tolist(), "scores": _finite_json(scores)}


def _infer_candidate_scale(
    candidate_midis: np.ndarray,
    candidate_scores: np.ndarray,
) -> tuple[set[int], float]:
    beam = _candidate_beam(candidate_midis, candidate_scores)
    if beam is None:
        return set(), 0.0
    response = rust_utility("pitch_key_prior", beam=beam)
    return set(response["classes"]), response["margin"]


def _segment_request(
    words: list[Word],
    times: np.ndarray,
    frequencies: np.ndarray,
    confidences: np.ndarray,
    energies: np.ndarray,
    min_confidence: float,
    min_volume: float,
    onset_scores: np.ndarray | None,
) -> dict:
    return {
        "words": [asdict(word) for word in words],
        "times": _finite_json(np.asarray(times, dtype=np.float64)),
        "frequencies": _finite_json(np.asarray(frequencies, dtype=np.float64)),
        "confidences": _finite_json(np.asarray(confidences, dtype=np.float64)),
        "energies": _finite_json(np.asarray(energies, dtype=np.float64)),
        "onsets": None if onset_scores is None else _finite_json(np.asarray(onset_scores, dtype=np.float64)),
        "min_confidence": min_confidence,
        "min_volume": min_volume,
        "basic_pitch": rust_utility("uses_basic_pitch_clock", backend=str(Path(__file__).resolve().parents[1])),
    }


def _segment_from_native(value: dict) -> Word:
    return Word(**{**value, "pitch_contour": [tuple(point) for point in value["pitch_contour"]]})


def word_to_segments(
    word: Word,
    times: np.ndarray,
    frequencies: np.ndarray,
    confidences: np.ndarray,
    energies: np.ndarray,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_volume: float = 0.0,
    onset_scores: np.ndarray | None = None,
) -> list[Word]:
    request = _segment_request([word], times, frequencies, confidences, energies, min_confidence, min_volume, onset_scores)
    return [_segment_from_native(part) for part in rust_utility("segment_pitch_words", request=request)[0]]


def _candidate_evidence_for_spans(
    spans: list[tuple[float, float]],
    times: np.ndarray,
    candidate_midis: np.ndarray,
    candidate_scores: np.ndarray,
    limit: int = 12,
) -> list[list[dict[str, float | int]]]:
    beam = _candidate_beam(candidate_midis, candidate_scores)
    if beam is None:
        return [[] for _ in spans]
    return rust_utility(
        "pitch_candidate_evidence", beam=beam,
        times=_finite_json(np.asarray(times, dtype=np.float64)), spans=spans, limit=limit,
    )


def _candidate_evidence_for_span(
    start: float, end: float, times: np.ndarray,
    candidate_midis: np.ndarray, candidate_scores: np.ndarray, limit: int = 12,
) -> list[dict[str, float | int]]:
    return _candidate_evidence_for_spans([(start, end)], times, candidate_midis, candidate_scores, limit)[0]


def assign_pitch_to_words(
    words: list[Word],
    times: np.ndarray,
    frequencies: np.ndarray,
    confidences: np.ndarray,
    energies: np.ndarray,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_volume: float = DEFAULT_MIN_VOLUME_RMS,
    line_breaks: list[int] | None = None,
    candidate_midis: np.ndarray | None = None,
    candidate_scores: np.ndarray | None = None,
    onset_scores: np.ndarray | None = None,
) -> tuple[list[Word], list[int]]:
    """Assign and repair pitch notes for the complete lyric in Rust."""
    request = _segment_request(words, times, frequencies, confidences, energies, min_confidence, min_volume, onset_scores)
    request["beam"] = (
        _candidate_beam(candidate_midis, candidate_scores)
        if candidate_midis is not None and candidate_scores is not None else None
    )
    request["line_breaks"] = [int(index) for index in line_breaks if index >= 0] if line_breaks is not None else []
    segments, breaks = rust_utility("assign_pitch_to_words", request=request)
    return [_segment_from_native(part) for part in segments], breaks
