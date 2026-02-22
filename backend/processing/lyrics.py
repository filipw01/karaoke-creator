"""Lyrics extraction with word-level timestamps using Qwen3/NeMo forced alignment and faster-whisper."""

from __future__ import annotations

import logging
import re
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
    # "qwen3" = Qwen3-ForcedAligner (11 langs) with NeMo fallback; "proportional" = spread evenly
    alignment_mode: str = "qwen3"
    # Seconds to skip at start when using proportional (intro)
    lyrics_start_offset: float = 2.0


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


def _normalize_lang_code(language: str | None) -> str:
    """Normalize to ISO 639-1 two-letter code; fix common mistakes (e.g. po -> pl for Polish)."""
    if not language or not language.strip():
        return "en"
    code = language.lower().strip()
    if code.startswith("polish") or code == "polski" or code == "po":
        return "pl"
    return code[:2] if len(code) >= 2 else "en"


# Qwen3-ForcedAligner supports 11 languages; ISO-2 code -> Qwen3 language name
QWEN3_LANGUAGES: dict[str, str] = {
    "en": "English",
    "zh": "Chinese",
    "yue": "Cantonese",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
    "es": "Spanish",
}
QWEN3_MAX_DURATION_SEC = 300.0  # 5 minutes per align() call

_qwen3_aligner_cache: object | None = None


def extract_lyrics_qwen3(
    audio_path: str,
    lyrics_text: str,
    language: str | None = None,
) -> tuple[list[Word], list[int]] | tuple[None, str]:
    """Use Qwen3-ForcedAligner for precise word-level forced alignment. 11 languages.

    Returns (words, line_breaks) on success, or (None, reason) if unavailable/fails.
    Chunks at line breaks when audio exceeds 5 minutes.
    """
    try:
        import torch
        from qwen_asr import Qwen3ForcedAligner
    except ImportError as e:
        reason = f"qwen-asr not installed (pip install qwen-asr): {e}"
        log.info("%s", reason)
        return None, reason

    lines = [[w for w in _clean_lyrics_line(ln).split() if w] for ln in lyrics_text.splitlines() if ln.strip()]
    all_words = [w for line in lines for w in line]
    if not all_words:
        return None, "No lyrics words after cleaning"

    lang_code = _normalize_lang_code(language)
    qwen3_lang = QWEN3_LANGUAGES.get(lang_code)
    if not qwen3_lang:
        return None, f"Qwen3 does not support language {lang_code}"

    global _qwen3_aligner_cache
    if _qwen3_aligner_cache is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        log.info("Loading Qwen3-ForcedAligner on %s", device)
        _qwen3_aligner_cache = Qwen3ForcedAligner.from_pretrained(
            "Qwen/Qwen3-ForcedAligner-0.6B",
            dtype=torch.bfloat16,
            device_map=device,
        )
        log.info("Qwen3-ForcedAligner loaded")
    model = _qwen3_aligner_cache

    duration = librosa.get_duration(path=audio_path)
    line_breaks: list[int] = []
    idx = 0
    for line in lines:
        if idx > 0:
            line_breaks.append(idx)
        idx += len(line)

    if duration <= QWEN3_MAX_DURATION_SEC:
        transcript = " ".join(all_words)
        try:
            results = model.align(audio=audio_path, text=transcript, language=qwen3_lang)
        except Exception as e:
            reason = str(e)
            log.warning("Qwen3 align failed (%s)", reason)
            return None, reason
        if not results or not results[0]:
            return None, "Qwen3 returned no alignments"
        words = [
            Word(text=w.text, start=float(w.start_time), end=float(w.end_time), confidence=1.0)
            for w in results[0]
        ]
        if len(words) != len(all_words):
            log.warning("Qwen3 word count %d != lyrics %d; using lyrics text", len(words), len(all_words))
            words = _qwen3_words_to_lyrics_order(results[0], all_words)
        log.info("Qwen3 produced %d words with %d line breaks", len(words), len(line_breaks))
        return (words, line_breaks)

    # Chunk by time: proportional word ranges per ~5 min segment
    n_chunks = max(1, int(duration / QWEN3_MAX_DURATION_SEC) + (1 if duration % QWEN3_MAX_DURATION_SEC > 0 else 0))
    chunk_dur = duration / n_chunks
    words_out: list[Word] = []
    for i in range(n_chunks):
        t_start = i * chunk_dur
        t_end = min((i + 1) * chunk_dur, duration)
        w_start = int(i * len(all_words) / n_chunks)
        w_end = int((i + 1) * len(all_words) / n_chunks) if i < n_chunks - 1 else len(all_words)
        if w_start >= w_end:
            continue
        chunk_text = " ".join(all_words[w_start:w_end])
        overlap = 1.0
        load_start = max(0.0, t_start - overlap)
        load_dur = (t_end - load_start) + overlap
        try:
            y, sr = librosa.load(audio_path, sr=16000, mono=True, offset=load_start, duration=load_dur)
        except Exception as e:
            log.warning("Qwen3 chunk load failed (%s)", e)
            return None, str(e)
        try:
            results = model.align(audio=(y, sr), text=chunk_text, language=qwen3_lang)
        except Exception as e:
            reason = str(e)
            log.warning("Qwen3 chunk align failed (%s)", reason)
            return None, reason
        if not results or not results[0]:
            return None, "Qwen3 returned no alignments for chunk"
        chunk_lyrics = all_words[w_start:w_end]
        for j, w in enumerate(results[0]):
            lw = chunk_lyrics[j] if j < len(chunk_lyrics) else w.text
            words_out.append(Word(text=lw, start=float(w.start_time) + load_start, end=float(w.end_time) + load_start, confidence=1.0))
    if len(words_out) != len(all_words):
        log.warning("Qwen3 chunked word count %d != lyrics %d", len(words_out), len(all_words))
    log.info("Qwen3 (chunked) produced %d words with %d line breaks", len(words_out), len(line_breaks))
    return (words_out, line_breaks)


def _qwen3_words_to_lyrics_order(aligned: list, lyrics_words: list[str]) -> list[Word]:
    """Map Qwen3 output to lyrics word order when counts differ."""
    if not aligned:
        return [Word(text=w, start=0.0, end=0.0, confidence=0.0) for w in lyrics_words]
    out: list[Word] = []
    ai = 0
    for i, lw in enumerate(lyrics_words):
        if ai < len(aligned):
            w = aligned[ai]
            out.append(Word(text=lw, start=float(w.start_time), end=float(w.end_time), confidence=1.0))
            ai += 1
        else:
            t = out[-1].end + 0.2 if out else 0.0
            out.append(Word(text=lw, start=t, end=t + 0.3, confidence=0.0))
    return out


# NeMo 2 ASR CTC model by language (fallback when Qwen3 does not support the language; supports song lyrics)
NEMO_CTC_MODELS: dict[str, str] = {
    "pl": "stt_pl_quartznet15x5",
    "en": "stt_en_fastconformer_hybrid_large_pc",
}

_nemo_model_cache: dict[str, object] = {}


def extract_lyrics_nemo(
    audio_path: str,
    lyrics_text: str,
    language: str | None = None,
) -> tuple[list[Word], list[int]] | tuple[None, str]:
    """Use NeMo 2 ASR (nemo-toolkit-asr) with word timestamps for song lyrics alignment.

    Fallback for languages Qwen3 does not support (e.g. Polish). Maps ASR output to lyrics by order.
    Returns (words, line_breaks) on success, or (None, reason) if unavailable/fails.
    """
    try:
        import nemo.collections.asr as nemo_asr
    except Exception as e:
        reason = f"NeMo ASR unavailable (pip install nemo-toolkit-asr): {e}"
        log.info("%s", reason)
        return None, reason

    lines = [[w for w in _clean_lyrics_line(ln).split() if w] for ln in lyrics_text.splitlines() if ln.strip()]
    all_words = [w for line in lines for w in line]
    if not all_words:
        return None, "No lyrics words after cleaning"

    lang_code = _normalize_lang_code(language)
    model_name = NEMO_CTC_MODELS.get(lang_code) or NEMO_CTC_MODELS["en"]

    global _nemo_model_cache
    if model_name not in _nemo_model_cache:
        try:
            log.info("Loading NeMo ASR model %s", model_name)
            model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
            if hasattr(model, "change_decoding_strategy"):
                try:
                    model.change_decoding_strategy(decoder_type="ctc")
                except Exception:
                    pass
            _nemo_model_cache[model_name] = model
        except Exception as e:
            reason = str(e)
            log.warning("NeMo model load failed (%s)", reason)
            return None, reason
    model = _nemo_model_cache[model_name]

    try:
        hypotheses = model.transcribe([audio_path], return_hypotheses=True)
    except Exception as e:
        reason = str(e)
        log.warning("NeMo transcribe failed (%s)", reason)
        return None, reason
    if not hypotheses or not hypotheses[0]:
        return None, "NeMo returned no hypothesis"

    hyp = hypotheses[0]
    if not hasattr(hyp, "timestamp") or not hyp.timestamp or "word" not in hyp.timestamp:
        return None, "NeMo model did not return word timestamps"
    word_ts = hyp.timestamp["word"]
    asr_words = (hyp.text or "").split()

    line_breaks = []
    idx = 0
    for line in lines:
        if idx > 0:
            line_breaks.append(idx)
        idx += len(line)

    if not asr_words or not word_ts:
        return None, "NeMo returned empty words or timestamps"
    if len(asr_words) != len(word_ts):
        word_ts = word_ts[: len(asr_words)] if len(word_ts) > len(asr_words) else word_ts + [(word_ts[-1][1], word_ts[-1][1] + 0.2)] * (len(asr_words) - len(word_ts))

    words: list[Word] = []
    for i, lw in enumerate(all_words):
        if i < len(word_ts):
            start_sec, end_sec = _nemo_ts_to_sec(word_ts[i], model)
            words.append(Word(text=lw, start=start_sec, end=end_sec, confidence=1.0))
        else:
            t = words[-1].end + 0.2 if words else 0.0
            words.append(Word(text=lw, start=t, end=t + 0.3, confidence=0.0))
    log.info("NeMo produced %d words with %d line breaks", len(words), len(line_breaks))
    return (words, line_breaks)


def _nemo_ts_to_sec(ts: object, model: object) -> tuple[float, float]:
    """Convert NeMo timestamp (frames or sec) to (start_sec, end_sec)."""
    if isinstance(ts, (list, tuple)) and len(ts) >= 2:
        a, b = float(ts[0]), float(ts[1])
        if b < 100 and hasattr(model, "cfg"):
            stride = getattr(getattr(model, "cfg", None), "preprocessor", None)
            if stride and hasattr(stride, "window_stride_in_secs"):
                a *= stride.window_stride_in_secs
                b *= stride.window_stride_in_secs
        return (a, b)
    return (0.0, 0.2)


def extract_lyrics_proportional(
    audio_path: str,
    lyrics_text: str,
    start_offset: float = 2.0,
) -> tuple[list[Word], list[int]]:
    """Distribute lyrics proportionally across audio duration. No Whisper.

    Words are spread evenly from start_offset to end. Line breaks preserved.
    """
    duration = librosa.get_duration(path=audio_path)
    t_start = min(start_offset, duration * 0.1)
    t_end = duration
    effective = max(0.1, t_end - t_start)

    lines = [[w for w in _clean_lyrics_line(ln).split() if w] for ln in lyrics_text.splitlines() if ln.strip()]
    all_words = [w for line in lines for w in line]
    if not all_words:
        return [], []

    n_total = len(all_words)
    words: list[Word] = []
    line_breaks: list[int] = []
    idx = 0
    for line in lines:
        if idx > 0:
            line_breaks.append(idx)
        n_line = len(line)
        for i in range(n_line):
            frac = (idx + i) / n_total
            next_frac = (idx + i + 1) / n_total
            t_s = t_start + frac * effective
            t_e = t_start + next_frac * effective
            words.append(Word(text=line[i], start=t_s, end=t_e, confidence=0))
        idx += n_line

    log.info("Proportional: %d words across %.1f–%.1fs (%d line breaks)", len(words), t_start, t_end, len(line_breaks))
    return words, line_breaks
