"""Transport for Rust pitch inference and audio analysis."""

from __future__ import annotations

import subprocess

import numpy as np

from processing.rust_utility import executable_path


def _run_analysis(command: str, *arguments: str, timeout: int = 300) -> bytes:
    try:
        completed = subprocess.run(
            [executable_path(), command, *arguments],
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Rust pitch analysis exceeded its {timeout}-second limit") from exc
    if completed.returncode:
        raise RuntimeError(completed.stderr.decode(errors="replace").strip() or "Rust pitch analysis failed")
    return completed.stdout


def load_mono(path: str, sample_rate: int, *, highpass_hz: float = 0.0) -> np.ndarray:
    pcm = _run_analysis("--desktop-pitch-audio", str(path), str(sample_rate), str(highpass_hz))
    return np.frombuffer(pcm, dtype="<f4")
