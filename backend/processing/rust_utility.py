"""Transport for the shared Rust backend used by the model worker and benchmarks."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable


class RustUtilityUnavailable(RuntimeError):
    """The required Rust executable is absent or could not be started."""


def executable_path() -> str:
    configured = os.environ.get("KARAOKE_RUST_UTILITY")
    if configured:
        return configured
    target = Path(__file__).resolve().parents[2] / "frontend" / "src-tauri" / "target"
    name = "app.exe" if os.name == "nt" else "app"
    for build in ("debug", "release"):
        candidate = target / build / name
        if candidate.is_file():
            return str(candidate)
    raise RustUtilityUnavailable(
        "Build the Rust backend with cargo build --manifest-path frontend/src-tauri/Cargo.toml "
        "or set KARAOKE_RUST_UTILITY to the compiled app executable."
    )


def _request(operation: str, arguments: dict[str, Any], json_default: Callable[[Any], Any] | None = None) -> str:
    return json.dumps(
        {"version": 1, "request": {"operation": operation, **arguments}},
        ensure_ascii=False, default=json_default,
    )


def call(
    operation: str,
    *,
    json_default: Callable[[Any], Any] | None = None,
    timeout_seconds: float | None = None,
    **arguments: Any,
) -> Any:
    executable = executable_path()
    try:
        result = subprocess.run(
            [executable, "--desktop-utility"],
            input=_request(operation, arguments, json_default),
            encoding="utf-8",
            capture_output=True,
            timeout=timeout_seconds if timeout_seconds is not None else {
                "publish": 300, "archive": 300, "encode_media": 300, "extract_audio": 300,
                "playback_audio": 300, "normalize_audio": 300, "whisper_draft": 960,
                "search_lyrics": 75, "search_youtube": 60, "detect_pitch": 2800,
            }.get(operation, 20),
        )
    except OSError as exc:
        raise RustUtilityUnavailable(f"Unable to start the Rust backend: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"Rust {operation} timed out") from exc
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"Rust {operation} exited with {result.returncode}")
    return json.loads(result.stdout)
