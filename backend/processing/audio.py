"""Shared audio adapters and model-specific FFmpeg preprocessing."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from processing.rust_utility import call as rust_utility


def copy_video(input_path: str, output_path: str) -> str:
    return rust_utility("encode_media", source=input_path, destination=output_path, video=True)


def extract_audio(input_path: str, output_dir: str) -> str:
    return rust_utility("extract_audio", source=input_path, destination=os.path.join(output_dir, "audio.wav"))


def playback_audio_path(source: Path, cache_dir: Path) -> Path:
    if source.suffix.lower() != ".mp3":
        return source
    return Path(rust_utility("playback_audio", source=str(source), cache=str(cache_dir)))


def convert_to_mono_16k(input_path: str, output_path: str) -> str:
    """Convert audio to 16kHz mono WAV (for Whisper / CREPE)."""
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


def normalize_audio(input_path: str, output_path: str) -> str:
    """Normalize audio loudness via ffmpeg loudnorm (two-pass EBU R128)."""
    # First pass: measure loudness
    measure_cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json",
        "-f", "null", "-",
    ]
    result = subprocess.run(measure_cmd, capture_output=True, text=True)

    # Parse measured values from stderr (loudnorm prints JSON at the end)
    import re
    json_match = re.search(r"\{[^}]+\}", result.stderr, re.DOTALL)
    if json_match:
        stats = json.loads(json_match.group())
        measured_i = stats.get("input_i", "-16")
        measured_tp = stats.get("input_tp", "-1.5")
        measured_lra = stats.get("input_lra", "11")
        measured_thresh = stats.get("input_thresh", "-26")

        # Second pass: apply normalization with measured values
        af_filter = (
            f"loudnorm=I=-16:TP=-1.5:LRA=11:"
            f"measured_I={measured_i}:measured_TP={measured_tp}:"
            f"measured_LRA={measured_lra}:measured_thresh={measured_thresh}:linear=true"
        )
    else:
        af_filter = "loudnorm=I=-16:TP=-1.5:LRA=11"

    normalize_cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-af", af_filter,
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        "-ac", "1",
        output_path,
    ]
    subprocess.run(normalize_cmd, check=True, capture_output=True)
    return output_path


def convert_to_mp3(input_path: str, output_path: str) -> str:
    return rust_utility("encode_media", source=input_path, destination=output_path, video=False)
