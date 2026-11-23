"""Shape-contract tests for optional model adapters without loading weights."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

backend = Path(__file__).resolve().parent.parent
if str(backend) not in sys.path:
    sys.path.insert(0, str(backend))

from processing.model_adapters import (
    ForceAlignAdapter,
    Qwen3ASRAdapter,
    Qwen3ForcedAlignerAdapter,
    WhisperXCTCAdapter,
    _anchored_segments,
    forcealign_enabled,
    melody_blend_enabled,
    write_melody_blend_audio,
    whisperx_enabled,
)
from processing.alignment import ASRToken


class _Stamp:
    def __init__(self, text: str, start: float, end: float, confidence: float = 1.0) -> None:
        self.text = text
        self.start_time = start
        self.end_time = end
        self.confidence = confidence


class _ASRResult:
    def __init__(self) -> None:
        self.time_stamps = [[_Stamp("hello", 0.1, 0.4, 0.0)]]


class _ASRModel:
    def transcribe(self, **_: object):
        return [_ASRResult()]


class _ForcedItem:
    text = "hello"
    start_time = 0.1
    end_time = 0.4
    confidence = 0.8


class _ForcedModel:
    def align(self, **_: object):
        # Current Qwen returns a batch of per-audio result lists.
        return [[_ForcedItem()]]


class _OfficialForcedAlignResult:
    def __init__(self) -> None:
        self.items = [_ForcedItem()]


class _OfficialForcedModel:
    def align(self, **_: object):
        return _OfficialForcedAlignResult()


class _Shape:
    def __init__(self, *shape: int) -> None:
        self._shape = shape

    def size(self, dimension: int) -> int:
        return self._shape[dimension]


class _ForceAlignModel:
    def __init__(self, _audio: str, _text: str) -> None:
        self.dictionary = {character: index for index, character in enumerate("|EHLO")}
        self.waveform = _Shape(1, 1600)
        self.emission = _Shape(10, 5)
        self.bundle = types.SimpleNamespace(sample_rate=16000)
        self.transcript = ""
        self.tokens = []

    def get_trellis(self):
        return object()

    def backtrack(self, _trellis):
        return object()

    def merge_repeats(self, _path):
        return object()

    def merge_words(self, _segments):
        return [types.SimpleNamespace(label="HELLO", start=1, end=3, score=0.8)]


def test_qwen_asr_unwraps_current_timestamp_batch_and_preserves_zero_confidence():
    adapter = Qwen3ASRAdapter()
    adapter._model = _ASRModel()

    tokens = adapter.transcribe("vocals.wav", "en")

    assert len(tokens) == 1
    assert tokens[0].text == "hello"
    assert tokens[0].confidence == 0.0
    assert tokens[0].evidence_type == "anchor"


def test_qwen_forced_aligner_unwraps_current_nested_result():
    adapter = Qwen3ForcedAlignerAdapter()
    adapter._model = _ForcedModel()

    tokens = adapter.align("vocals.wav", "hello", "en")

    assert len(tokens) == 1
    assert tokens[0].evidence_type == "forced"
    assert tokens[0].start == 0.1


def test_qwen_forced_aligner_unwraps_official_items_wrapper():
    adapter = Qwen3ForcedAlignerAdapter()
    adapter._model = _OfficialForcedModel()

    tokens = adapter.align("vocals.wav", "hello", "en")

    assert [token.text for token in tokens] == ["hello"]


def test_whisperx_adapter_preserves_ctc_evidence_shape(monkeypatch: pytest.MonkeyPatch):
    captured_segments: list[dict[str, object]] = []

    def fake_align(segments, *_):
        captured_segments.extend(segments)
        return {"segments": [{"words": [{"word": "hello", "start": 18.1, "end": 18.4, "score": 0.7}]}]}

    fake_whisperx = types.SimpleNamespace(
        load_align_model=lambda **_: (object(), {"language": "en"}),
        align=fake_align,
    )
    fake_soundfile = types.SimpleNamespace(info=lambda _: types.SimpleNamespace(duration=30.0))
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)
    monkeypatch.setitem(sys.modules, "soundfile", fake_soundfile)

    tokens = WhisperXCTCAdapter().align(
        "vocals.wav",
        "hello",
        "en",
        anchor_tokens=[types.SimpleNamespace(start=18.1, end=18.4)],
    )

    assert len(tokens) == 1
    assert tokens[0].evidence_type == "ctc"
    assert tokens[0].confidence == 0.7
    assert tokens[0].sequence_index == 0
    assert captured_segments[0]["start"] == pytest.approx(17.35)
    assert captured_segments[0]["end"] == pytest.approx(19.15)


def test_whisperx_auto_device_uses_cpu_without_cuda(monkeypatch: pytest.MonkeyPatch):
    import torch

    monkeypatch.delenv("WHISPERX_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert WhisperXCTCAdapter(device=None).device == "cpu"


def test_whisperx_ctc_is_enabled_by_default_and_can_be_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ALIGNMENT_CTC_BACKEND", raising=False)
    assert whisperx_enabled()

    monkeypatch.setenv("ALIGNMENT_CTC_BACKEND", "off")
    assert not whisperx_enabled()


def test_melody_blend_is_enabled_by_default_and_can_be_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ALIGNMENT_MELODY_BLEND_ENABLED", raising=False)
    assert melody_blend_enabled()

    monkeypatch.setenv("ALIGNMENT_MELODY_BLEND_ENABLED", "off")
    assert not melody_blend_enabled()


def test_write_melody_blend_audio_mixes_and_normalizes(tmp_path: Path):
    import numpy as np
    import soundfile as sf

    original_path = tmp_path / "original.wav"
    vocal_path = tmp_path / "vocals.wav"
    output_path = tmp_path / "blend.wav"
    sf.write(original_path, np.ones(32, dtype=np.float32) * 0.5, 16000)
    sf.write(vocal_path, np.zeros(32, dtype=np.float32), 16000)

    write_melody_blend_audio(
        str(original_path),
        str(vocal_path),
        str(output_path),
        accompaniment_ratio=0.6,
    )

    blended, sample_rate = sf.read(output_path)
    assert sample_rate == 16000
    assert blended.shape == (32,)
    assert np.allclose(blended, 0.294, atol=1e-4)


def test_whisperx_uses_sixteen_word_anchored_windows_by_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("WHISPERX_WINDOW_WORDS", raising=False)
    words = [f"word{index}" for index in range(20)]
    anchors = [
        ASRToken(text=word, start=float(index), end=float(index) + 0.4)
        for index, word in enumerate(words)
    ]

    segments = _anchored_segments(" ".join(words), anchors, duration=30.0)

    assert [segment["sequence_start"] for segment in segments] == [0, 16]
    assert segments[0]["text"] == " ".join(words[:16])
    assert segments[1]["text"] == " ".join(words[16:])


def test_whisperx_keeps_unanchored_leading_window(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("WHISPERX_WINDOW_WORDS", raising=False)
    words = [f"word{index}" for index in range(20)]
    anchors = [
        ASRToken(text=word, start=float(index) + 16.0, end=float(index) + 16.4)
        for index, word in enumerate(words[16:])
    ]

    segments = _anchored_segments(" ".join(words), anchors, duration=60.0)

    assert [segment["sequence_start"] for segment in segments] == [0, 16]
    assert segments[0]["start"] == 0.0
    assert segments[0]["end"] > 16.0


def test_forcealign_uses_trellis_word_spans_without_phonemizer(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "forcealign", types.SimpleNamespace(ForceAlign=_ForceAlignModel))

    tokens = ForceAlignAdapter().align("vocals.wav", "hello", "en")

    assert len(tokens) == 1
    assert tokens[0].model == "forcealign-wav2vec2"
    assert tokens[0].evidence_type == "forced"
    assert tokens[0].start == pytest.approx(0.01)
    assert tokens[0].end == pytest.approx(0.03)


def test_forcealign_is_explicitly_opt_in(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ALIGNMENT_WAV2VEC_BACKEND", raising=False)
    assert not forcealign_enabled()

    monkeypatch.setenv("ALIGNMENT_WAV2VEC_BACKEND", "forcealign")
    assert forcealign_enabled()
