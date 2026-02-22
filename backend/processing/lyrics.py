"""Lyrics extraction with word-level timestamps using Qwen3/NeMo forced alignment and faster-whisper."""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass

import librosa
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
        "condition_on_previous_text": not has_lyrics_prompt,
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


def _clean_lyrics_line(t: str) -> str:
    """Strip section markers like [Verse 1], [Chorus], (x2) that aren't sung."""
    return re.sub(r"\[[^\]]*\]|\(x\d+\)", " ", t, flags=re.IGNORECASE)


_PUNCTUATION_STRIP = re.compile(r"^[\s.,!?…\-–—:;\"'()\[\]♪♫#*]+|[\s.,!?…\-–—:;\"'()\[\]♪♫#*]+$")


def _normalize_word_for_alignment(w: str) -> str:
    """Strip leading/trailing punctuation for aligner tokenization."""
    return _PUNCTUATION_STRIP.sub("", w).strip()


def preprocess_lyrics_for_alignment(lyrics_text: str) -> tuple[list[list[str]], list[str], list[int]]:
    """Single preprocessing pipeline for all aligners. Returns (lines, all_words, line_breaks)."""
    lines: list[list[str]] = []
    all_words: list[str] = []
    line_breaks: list[int] = []
    idx = 0
    for raw_line in lyrics_text.splitlines():
        if not raw_line.strip():
            continue
        cleaned = _clean_lyrics_line(raw_line)
        words = [_normalize_word_for_alignment(w) for w in cleaned.split() if w]
        words = [w for w in words if w]
        if not words:
            continue
        if idx > 0:
            line_breaks.append(idx)
        lines.append(words)
        all_words.extend(words)
        idx += len(words)
    return (lines, all_words, line_breaks)


def _normalize_lang_code(language: str | None) -> str:
    """Normalize to ISO 639-1 two-letter code; fix common mistakes (e.g. po -> pl for Polish)."""
    if not language or not language.strip():
        return "en"
    code = language.lower().strip()
    if code.startswith("polish") or code == "polski" or code == "po":
        return "pl"
    return code[:2] if len(code) >= 2 else "en"


def _aligner_words_to_lyrics_order(aligned: list, lyrics_words: list[str]) -> list[Word]:
    """Map aligner output to lyrics word order; use normalized match when counts differ."""
    if not aligned:
        return [Word(text=w, start=0.0, end=0.0, confidence=0.0) for w in lyrics_words]
    if len(aligned) == len(lyrics_words):
        return [
            Word(text=lw, start=float(w.start_time), end=float(w.end_time), confidence=1.0)
            for lw, w in zip(lyrics_words, aligned)
        ]
    # Match by normalized text in a small window
    def norm(s: str) -> str:
        return _normalize_word_for_alignment(s).lower()

    out: list[Word] = []
    ai = 0
    window = 3
    for i, lw in enumerate(lyrics_words):
        best_j = -1
        best_score = -1
        for j in range(ai, min(ai + window, len(aligned))):
            aw = getattr(aligned[j], "text", str(aligned[j]))
            if norm(aw) == norm(lw):
                best_j = j
                best_score = 2
                break
            if norm(aw) in norm(lw) or norm(lw) in norm(aw):
                if best_score < 1:
                    best_score = 1
                    best_j = j
        if best_j >= 0:
            w = aligned[best_j]
            out.append(Word(text=lw, start=float(w.start_time), end=float(w.end_time), confidence=1.0))
            ai = best_j + 1
        elif ai < len(aligned):
            w = aligned[ai]
            out.append(Word(text=lw, start=float(w.start_time), end=float(w.end_time), confidence=0.7))
            ai += 1
        else:
            t = out[-1].end + 0.2 if out else 0.0
            out.append(Word(text=lw, start=t, end=t + 0.3, confidence=0.0))
    return out


# NeMo 2 ASR CTC model by language (fallback when Qwen3 does not support the language; supports song lyrics)
def extract_lyrics_forcealign(
    audio_path: str,
    lyrics_text: str,
    language: str | None = None,
) -> tuple[list[Word], list[int]] | tuple[None, str]:
    """ForceAlign (pip install forcealign): English, word-level. Returns (words, line_breaks) or (None, reason)."""
    try:
        from forcealign import ForceAlign
        import nltk
        import os
        nltk_data_dir = os.path.join(os.path.dirname(__file__), "..", "nltk_data")
        os.makedirs(nltk_data_dir, exist_ok=True)
        if nltk_data_dir not in nltk.data.path:
            nltk.data.path.insert(0, nltk_data_dir)
        nltk.download("averaged_perceptron_tagger_eng", quiet=True, download_dir=nltk_data_dir)
    except ImportError as e:
        return None, f"forcealign not installed (pip install forcealign): {e}"

    _, all_words, line_breaks = preprocess_lyrics_for_alignment(lyrics_text)
    if not all_words:
        return None, "No lyrics words after cleaning"
    lang = _normalize_lang_code(language)
    if lang != "en":
        return None, "ForceAlign supports English only"

    transcript = " ".join(all_words)
    try:
        align = ForceAlign(audio_file=audio_path, transcript=transcript)
        raw_words = align.inference()
    except Exception as e:
        log.warning("ForceAlign failed: %s", e)
        return None, str(e)
    if not raw_words:
        return None, "ForceAlign returned no words"

    class _AlignerWord:
        def __init__(self, w: object) -> None:
            self.text = getattr(w, "word", str(w))
            self.start_time = float(getattr(w, "time_start", 0))
            self.end_time = float(getattr(w, "time_end", 0))

    words = [Word(text=_AlignerWord(w).text, start=_AlignerWord(w).start_time, end=_AlignerWord(w).end_time, confidence=1.0) for w in raw_words]
    if len(words) == len(all_words):
        words = [
            Word(text=all_words[i], start=w.start, end=w.end, confidence=w.confidence)
            for i, w in enumerate(words)
        ]
    elif len(words) != len(all_words):
        log.warning("ForceAlign word count %d != lyrics %d; mapping", len(words), len(all_words))
        words = _aligner_words_to_lyrics_order([_AlignerWord(w) for w in raw_words], all_words)
    log.info("ForceAlign produced %d words, %d line breaks", len(words), len(line_breaks))
    return (words, line_breaks)
