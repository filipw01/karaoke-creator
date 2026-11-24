"""Optional adapter for running the Basic Pitch ONNX graph in the Rust utility."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class NativePitchSession:
    def __init__(self, executable: str, model: Path):
        self._process = subprocess.Popen(
            [executable, "--desktop-pitch", str(model)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        self._outputs = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._process.stdin.close()
        self._process.stdout.close()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()

    def get_inputs(self):
        return [SimpleNamespace(name="audio")]

    def get_outputs(self):
        return self._outputs

    def run(self, _, inputs):
        pcm = np.asarray(inputs["audio"], dtype="<f4")
        if pcm.shape != (1, 43844, 1):
            raise ValueError("The native pitch worker expects a single 43844-sample window")
        try:
            self._process.stdin.write(pcm.tobytes())
            self._process.stdin.flush()
            response = self._process.stdout.readline()
        except BrokenPipeError as exc:
            raise RuntimeError("The native Basic Pitch inference process exited") from exc
        if not response:
            raise RuntimeError("The native Basic Pitch inference process returned no output")
        tensors = json.loads(response)
        self._outputs = [SimpleNamespace(name=tensor["name"]) for tensor in tensors]
        return [np.asarray(item["values"], dtype=np.float32).reshape(item["shape"]) for item in tensors]


def native_pyin(executable: str, samples: np.ndarray, frame: int, hop: int):
    completed = subprocess.run(
        [executable, "--desktop-pyin", str(frame), str(hop)],
        input=np.asarray(samples, dtype="<f4").tobytes(),
        capture_output=True,
        timeout=300,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.decode(errors="replace").strip() or "Native pYIN failed")
    frequency, voiced, probability = json.loads(completed.stdout)
    expected = 1 + len(samples) // hop
    if len(frequency) != expected or len(voiced) != expected or len(probability) != expected:
        raise ValueError("Native pYIN returned an unexpected number of frames")
    return np.asarray(frequency, dtype=np.float64), np.asarray(voiced, dtype=bool), np.asarray(probability, dtype=np.float64)
