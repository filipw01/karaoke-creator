"""Transport for the shared Rust backend used by FastAPI and benchmarks."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tempfile
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


def _request(operation: str, arguments: dict[str, Any]) -> str:
    return json.dumps({"version": 1, "request": {"operation": operation, **arguments}}, ensure_ascii=False)


def call(operation: str, **arguments: Any) -> Any:
    executable = executable_path()
    try:
        result = subprocess.run(
            [executable, "--desktop-utility"],
            input=_request(operation, arguments),
            encoding="utf-8",
            capture_output=True,
            timeout={
                "publish": 300, "archive": 300, "encode_media": 300, "extract_audio": 300,
                "playback_audio": 300, "whisper_draft": 960,
                "search_lyrics": 75, "search_youtube": 60,
            }.get(operation, 20),
        )
    except OSError as exc:
        raise RustUtilityUnavailable(f"Unable to start the Rust backend: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"Rust {operation} timed out") from exc
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"Rust {operation} exited with {result.returncode}")
    return json.loads(result.stdout)


async def call_with_progress(
    operation: str, operation_progress: Callable[[dict[str, Any]], None], **arguments: Any,
) -> dict[str, Any]:
    executable = executable_path()
    with tempfile.TemporaryFile() as diagnostics:
        process = await asyncio.create_subprocess_exec(
            executable, "--desktop-utility",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=diagnostics,
        )

        async def exchange() -> dict[str, Any]:
            assert process.stdin is not None and process.stdout is not None
            process.stdin.write(_request(operation, arguments).encode())
            await process.stdin.drain()
            process.stdin.close()
            receipt = None
            async for line in process.stdout:
                payload = json.loads(line)
                if "progress" in payload:
                    operation_progress(payload["progress"])
                else:
                    receipt = payload
            if await process.wait() or receipt is None:
                diagnostics.seek(0)
                detail = diagnostics.read().decode(errors="replace").strip()
                raise RuntimeError(detail or f"Rust {operation} did not return a receipt")
            return receipt

        try:
            return await asyncio.wait_for(exchange(), timeout=1800)
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
