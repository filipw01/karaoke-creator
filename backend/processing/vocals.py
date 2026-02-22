"""Vocal separation using Demucs Python API. Falls back to passthrough when demucs is not installed (e.g. Python 3.14)."""

from __future__ import annotations

import logging
import os
import shutil
import sys
import subprocess


log = logging.getLogger("demucs")


def _demucs_available() -> bool:
    result = subprocess.run(
        [sys.executable, "-c", "import demucs"],
        capture_output=True,
        timeout=5,
    )
    return result.returncode == 0


def separate_vocals(audio_path: str, output_dir: str) -> tuple[str, str]:
    """Separate vocals from instrumental using Demucs htdemucs model.

    If demucs is not installed (e.g. on Python 3.14), copies the original audio to both outputs (no separation).
    Returns (vocals_path, instrumental_path).
    """
    if not _demucs_available():
        log.warning("demucs not installed; using original audio for both vocals and instrumental")
        vocals_dst = os.path.join(output_dir, "vocals.wav")
        instrumental_dst = os.path.join(output_dir, "instrumental.wav")
        shutil.copy2(audio_path, vocals_dst)
        shutil.copy2(audio_path, instrumental_dst)
        return vocals_dst, instrumental_dst

    demucs_out = os.path.join(output_dir, "demucs_raw")
    os.makedirs(demucs_out, exist_ok=True)

    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems", "vocals",
        "-o", demucs_out,
        "--name", "htdemucs",
        audio_path,
    ]
    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        log.error("Demucs stderr:\n%s", result.stderr)
        raise RuntimeError(f"Demucs failed (exit {result.returncode}): {result.stderr[-500:]}")

    track_name = os.path.splitext(os.path.basename(audio_path))[0]
    demucs_track_dir = os.path.join(demucs_out, "htdemucs", track_name)

    vocals_src = os.path.join(demucs_track_dir, "vocals.wav")
    instrumental_src = os.path.join(demucs_track_dir, "no_vocals.wav")

    vocals_dst = os.path.join(output_dir, "vocals.wav")
    instrumental_dst = os.path.join(output_dir, "instrumental.wav")

    shutil.move(vocals_src, vocals_dst)
    shutil.move(instrumental_src, instrumental_dst)

    shutil.rmtree(demucs_out, ignore_errors=True)

    return vocals_dst, instrumental_dst
