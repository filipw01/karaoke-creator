"""Extract and convert audio from video/audio files using ffmpeg."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import threading
from hashlib import sha256
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".webm", ".mov", ".flv", ".wmv", ".m4v"}
_playback_cache_lock = threading.Lock()


def is_video_file(input_path: str) -> bool:
    """Check if the input file contains a video stream."""
    ext = os.path.splitext(input_path)[1].lower()
    if ext in VIDEO_EXTENSIONS:
        return True
    # Also probe with ffprobe in case extension is misleading
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", input_path],
            capture_output=True, text=True,
        )
        data = json.loads(result.stdout)
        return any(s.get("codec_type") == "video" for s in data.get("streams", []))
    except Exception:
        return False


def copy_video(input_path: str, output_path: str) -> str:
    """Re-mux the video to MP4 (no re-encoding) for UltraStar compatibility."""
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-c", "copy",
        "-movflags", "+faststart",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


def extract_audio(input_path: str, output_dir: str) -> str:
    """Extract audio from a video/audio file and convert to WAV.

    Returns the path to the output WAV file (44100 Hz stereo for Demucs).
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "audio.wav")

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        "-ac", "2",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


def playback_audio_path(source: Path, cache_dir: Path) -> Path:
    """Use PCM for MP3 playback so the seek lands on the correct samples."""
    if source.suffix.lower() != ".mp3":
        return source

    try:
        info = source.stat()
        key = sha256(os.fsencode(source))
        key.update(f":{info.st_ino}:{info.st_size}:{info.st_mtime_ns}:{info.st_ctime_ns}".encode())
        cached = cache_dir / f"{key.hexdigest()}.wav"
        if cached.is_file():
            return cached

        with _playback_cache_lock:
            if not cached.is_file():
                cache_dir.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(dir=cache_dir) as staging:
                    wav = extract_audio(str(source), staging)
                    os.replace(wav, cached)
        return cached
    except (OSError, subprocess.CalledProcessError):
        logging.getLogger(__name__).exception("Could not prepare seekable audio for %s", source)
        return source


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
    """Convert audio to MP3 for the final output package."""
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-codec:a", "libmp3lame",
        "-qscale:a", "2",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path
