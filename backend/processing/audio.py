"""Python adapters for audio operations owned by the Rust backend."""

from __future__ import annotations

import os

from processing.rust_utility import call as rust_utility


def copy_video(input_path: str, output_path: str) -> str:
    return rust_utility("encode_media", source=input_path, destination=output_path, video=True)


def extract_audio(input_path: str, output_dir: str) -> str:
    return rust_utility("extract_audio", source=input_path, destination=os.path.join(output_dir, "audio.wav"))


def normalize_audio(input_path: str, output_path: str) -> str:
    return rust_utility("normalize_audio", source=input_path, destination=output_path)


def convert_to_mp3(input_path: str, output_path: str) -> str:
    return rust_utility("encode_media", source=input_path, destination=output_path, video=False)
