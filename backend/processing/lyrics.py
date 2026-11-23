"""Lyrics extraction with word-level timestamps using Qwen3/NeMo forced alignment and faster-whisper."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from faster_whisper import WhisperModel

from models import Word


log = logging.getLogger("lyrics")

_model_cache: dict[str, WhisperModel] = {}

_JUNK_PATTERN = re.compile(r"^[\s.,!?…\-–—:;\"'()\[\]♪♫#*]+$")
_PROMPT_MAX_CHARS = 200  # Whisper prompt truncation (≈50 tokens)


@dataclass
class LyricsParams:
    """Configurable parameters for lyrics extraction."""
    model_size: str = "large-v3"
    language: str | None = None
    vad_filter: bool = False
    vad_threshold: float = 0.25
    vad_min_silence_ms: int = 200
    speech_pad_ms: int = 250
    no_speech_threshold: float = 0.4
    # Keep this explicit for sung audio.  Conditioning a later segment on a
    # hallucinated word from an instrumental gap is one of the most common
    # causes of a permanently shifted word clock.
    condition_on_previous_text: bool | None = None
    beam_size: int = 5
    initial_prompt: str = "These are song lyrics with singing."
    min_word_confidence: float = 0.15
    audio_source: str = "original"  # "vocals" | "original"
    normalize: bool = True


def _get_model(model_size: str = "medium") -> WhisperModel:
    if model_size not in _model_cache:
        log.info("Loading Whisper model (%s, int8)...", model_size)
        _model_cache[model_size] = WhisperModel(model_size, device="cpu", compute_type="int8")
        log.info("Whisper model loaded (%s)", model_size)
    return _model_cache[model_size]


def extract_lyrics(
    vocals_path: str,
    language: str | None = None,
    params: LyricsParams | None = None,
) -> list[Word]:
    """Transcribe vocals and return words with timestamps.

    Uses faster-whisper with word_timestamps=True.
    Pass a {@link LyricsParams} to override default settings.
    """
    p = params or LyricsParams()
    if language:
        p.language = language

    model = _get_model(p.model_size)

    prompt = p.initial_prompt
    if len(prompt) > _PROMPT_MAX_CHARS:
        prompt = prompt[: _PROMPT_MAX_CHARS].rsplit(maxsplit=1)[0]
        log.debug("Truncated prompt to %d chars", len(prompt))
    has_lyrics_prompt = len(p.initial_prompt) > 100 and "\n" in p.initial_prompt

    transcribe_kwargs: dict = {
        "word_timestamps": True,
        "language": p.language if p.language else None,
        "beam_size": p.beam_size,
        "best_of": p.beam_size,
        "condition_on_previous_text": (
            p.condition_on_previous_text
            if p.condition_on_previous_text is not None
            else not has_lyrics_prompt
        ),
        "initial_prompt": prompt,
        "no_speech_threshold": p.no_speech_threshold,
        "temperature": 0.0 if has_lyrics_prompt else [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    }

    if p.vad_filter:
        transcribe_kwargs["vad_filter"] = True
        transcribe_kwargs["vad_parameters"] = {
            "min_silence_duration_ms": p.vad_min_silence_ms,
            "speech_pad_ms": p.speech_pad_ms,
            "threshold": p.vad_threshold,
        }

    log.info("Transcribing with params: model=%s, vad=%s, vad_thresh=%.2f, pad=%dms, beam=%d, no_speech=%.2f",
             p.model_size, p.vad_filter, p.vad_threshold, p.speech_pad_ms, p.beam_size, p.no_speech_threshold)

    segments, info = model.transcribe(vocals_path, **transcribe_kwargs)

    log.info("Detected language: %s (prob=%.2f)", info.language, info.language_probability)

    words: list[Word] = []
    for segment in segments:
        if segment.words is None:
            continue
        for w in segment.words:
            text = w.word.strip()

            if not text or _JUNK_PATTERN.match(text):
                continue

            if w.probability < p.min_word_confidence:
                log.debug("Skipping low-confidence word: '%s' (p=%.2f)", text, w.probability)
                continue

            words.append(Word(
                text=text,
                start=w.start,
                end=w.end,
                confidence=w.probability,
            ))

    log.info("Extracted %d words from vocals", len(words))
    if words:
        log.info("First 15 words: %s", " ".join(f"{w.text}@{w.start:.2f}s" for w in words[:15]))
        log.info("Last 5 words: %s", " ".join(f"{w.text}@{w.start:.2f}s" for w in words[-5:]))

    return words
