"""Versioned stdin/stdout inference protocol for the Rust backend."""

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
    operation = request.get("operation")
    if operation == "separate":
        from processing.vocals import separate_vocals
        audio = Path(request["audio"])
        if not audio.is_file():
            raise ValueError("Prepared audio file is missing")
        directory = Path(request["output_directory"])
        directory.mkdir(parents=True, exist_ok=True)
        vocals, instrumental = separate_vocals(str(audio), str(directory), require_model=request.get("require_demucs", False))
        return {"vocals": str(vocals), "instrumental": str(instrumental)}
    if operation == "managed_alignment":
        from processing.alignment_job import managed_alignment
        return managed_alignment(request["request"])
    if operation == "align":
        from processing.alignment_worker import AlignmentWorkerRequest, AlignmentWorkerResponse, run_worker_alignment
        try:
            alignment = AlignmentWorkerRequest(**request["request"])
        except (TypeError, ValueError) as exc:
            return {"validation_error": str(exc)}
        try:
            response = run_worker_alignment(alignment)
        except Exception as exc:
            response = AlignmentWorkerResponse(
                job_id=alignment.job_id,
                alignment_run_id=alignment.alignment_run_id,
                lyric_revision_id=alignment.lyric_revision_id,
                status="failed",
                diagnostics={"error": str(exc)},
            )
        return {"response": response.model_dump()}
    raise ValueError("Unsupported desktop worker operation")


def respond(payload: str) -> str:
    try:
        request = json.loads(payload)
        result = execute(request)
        response = {"version": PROTOCOL_VERSION, "result": result}
        return json.dumps(response, allow_nan=False)
    except Exception as exc:
        return json.dumps({"version": PROTOCOL_VERSION, "error": str(exc)})


def main() -> None:
    os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(__file__).resolve().parent / "outputs" / "numba-cache"))
    with os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8") as protocol:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        with contextlib.redirect_stdout(sys.stderr):
            if sys.argv[1:] == ["--serve"]:
                for line in sys.stdin:
                    print(respond(line), file=protocol, flush=True)
            else:
                print(respond(sys.stdin.read()), file=protocol)


if __name__ == "__main__":
    main()
