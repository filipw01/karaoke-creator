"""Pitch detection using librosa pyin for monophonic vocal analysis."""

from __future__ import annotations

import logging

import librosa
import numpy as np

from models import Word


log = logging.getLogger("pitch")


def detect_pitch(vocals_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run pyin pitch detection on the vocal track.

    Returns (times, frequencies, confidences) arrays.
    """
    # Load at original sample rate for best pitch accuracy
    y, sr = librosa.load(vocals_path, sr=22050, mono=True)

    log.info("Running pyin on %d samples (sr=%d, duration=%.1fs)", len(y), sr, len(y) / sr)

    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, sr=sr,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C6"),
        frame_length=2048,
        hop_length=512,
    )

    times = librosa.times_like(f0, sr=sr, hop_length=512)

    frequencies = np.nan_to_num(f0, nan=0.0)
    confidences = voiced_prob if voiced_prob is not None else voiced_flag.astype(float)

    voiced_count = np.sum(frequencies > 0)
    log.info("Pitch detection: %d frames, %d voiced (%.0f%%)",
             len(frequencies), voiced_count,
             100 * voiced_count / max(len(frequencies), 1))

    return times, frequencies, confidences


def hz_to_midi(hz: float) -> int:
    """Convert frequency in Hz to the nearest MIDI note number."""
    if hz <= 0:
        return 0
    return int(round(69 + 12 * np.log2(hz / 440.0)))


def assign_pitch_to_words(
    words: list[Word],
    times: np.ndarray,
    frequencies: np.ndarray,
    confidences: np.ndarray,
    min_confidence: float = 0.4,
) -> list[Word]:
    """For each word, compute the median pitch over its time range."""
    for word in words:
        mask = (times >= word.start) & (times <= word.end)
        word_freqs = frequencies[mask]
        word_confs = confidences[mask]

        if len(word_freqs) == 0:
            word.pitch_hz = 0.0
            word.midi_note = 60
            continue

        # Filter by confidence and non-zero frequency
        voiced_mask = (word_confs >= min_confidence) & (word_freqs > 0)
        voiced_freqs = word_freqs[voiced_mask]

        if len(voiced_freqs) == 0:
            voiced_freqs = word_freqs[word_freqs > 0]

        if len(voiced_freqs) == 0:
            word.pitch_hz = 0.0
            word.midi_note = 60
            continue

        median_hz = float(np.median(voiced_freqs))
        word.pitch_hz = median_hz
        word.midi_note = hz_to_midi(median_hz)

    # Log a summary of assigned pitches
    pitched = [w for w in words if w.pitch_hz > 0]
    log.info("Assigned pitch to %d/%d words", len(pitched), len(words))
    if pitched:
        midi_notes = [w.midi_note for w in pitched]
        log.info("Pitch range: MIDI %d-%d (%.0f-%.0f Hz)",
                 min(midi_notes), max(midi_notes),
                 min(w.pitch_hz for w in pitched),
                 max(w.pitch_hz for w in pitched))

    return words
