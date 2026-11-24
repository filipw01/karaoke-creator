"""Versioned stdin/stdout inference protocol for the Tauri development app."""

from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path

PROTOCOL_VERSION = 1


def execute(request: dict) -> dict:
    if request.get("version") != PROTOCOL_VERSION:
        raise ValueError("Unsupported desktop worker protocol")
    audio = Path(request["audio"])
    if not audio.is_file():
        raise ValueError("Prepared audio file is missing")
    operation = request.get("operation")
    if operation == "separate":
        from processing.vocals import separate_vocals
        directory = Path(request["output_directory"])
        directory.mkdir(parents=True, exist_ok=True)
        vocals, instrumental = separate_vocals(str(audio), str(directory))
        return {"vocals": str(vocals), "instrumental": str(instrumental)}
    raise ValueError("Unsupported desktop worker operation")


def main() -> None:
    os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(__file__).resolve().parent / "outputs" / "numba-cache"))
    try:
        request = json.load(sys.stdin)
        with contextlib.redirect_stdout(sys.stderr):
            result = execute(request)
        response = {"version": PROTOCOL_VERSION, "result": result}
    except Exception as exc:
        response = {"version": PROTOCOL_VERSION, "error": str(exc)}
    print(json.dumps(response, allow_nan=False))


if __name__ == "__main__":
    main()
