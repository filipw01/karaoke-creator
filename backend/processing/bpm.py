"""BPM detection using librosa beat tracking."""

from __future__ import annotations

import librosa
import numpy as np


def detect_bpm(audio_path: str) -> float:
    """Detect the BPM of a song using librosa.

    Returns the real BPM (not the UltraStar-multiplied value).
    """
    y, sr = librosa.load(audio_path, sr=None)
    tempo, _beats = librosa.beat.beat_track(y=y, sr=sr)

    # librosa may return an ndarray; extract scalar
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo[0])

    return float(tempo)
