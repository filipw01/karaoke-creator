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
    alignment_mode: str = "forcealign"


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

    lines, all_words, _line_breaks_pre = preprocess_lyrics_for_alignment(lyrics_text)
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

    line_breaks = _line_breaks_pre

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

    lines, all_words, line_breaks = preprocess_lyrics_for_alignment(lyrics_text)
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


def extract_lyrics_aeneas(
    audio_path: str,
    lyrics_text: str,
    language: str | None = None,
) -> tuple[list[Word], list[int]] | tuple[None, str]:
    """aeneas (pip install aeneas): word-level sync. Returns (words, line_breaks) or (None, reason)."""
    try:
        from aeneas.executetask import ExecuteTask
        from aeneas.task import Task
    except ImportError as e:
        return None, f"aeneas not installed (pip install aeneas): {e}"

    lines, all_words, line_breaks = preprocess_lyrics_for_alignment(lyrics_text)
    if not all_words:
        return None, "No lyrics words after cleaning"

    try:
        import json
        import os
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8") as f:
            for w in all_words:
                f.write(w + "\n")
            txt_path = f.name
        task = Task(config_string="task_language=eng|is_text_type=plain|os_task_file_format=json")
        task.audio_file_path_absolute = audio_path
        task.text_file_path_absolute = txt_path
        task.sync_map_file_path_absolute = txt_path + ".json"
        ExecuteTask(task).execute()
        with open(task.sync_map_file_path_absolute, encoding="utf-8") as fp:
            data = json.load(fp)
        fragments = data.get("fragments", [])
        words_out: list[Word] = []
        for i, frag in enumerate(fragments):
            if i >= len(all_words):
                break
            begin = float(frag.get("begin", 0))
            end = float(frag.get("end", 0))
            words_out.append(Word(text=all_words[i], start=begin, end=end, confidence=1.0))
        while len(words_out) < len(all_words):
            t = words_out[-1].end + 0.2 if words_out else 0.0
            words_out.append(Word(text=all_words[len(words_out)], start=t, end=t + 0.3, confidence=0.0))
        try:
            os.unlink(txt_path)
            os.unlink(task.sync_map_file_path_absolute)
        except Exception:
            pass
        log.info("aeneas produced %d words, %d line breaks", len(words_out), len(line_breaks))
        return (words_out, line_breaks)
    except Exception as e:
        log.warning("aeneas failed: %s", e)
        return None, str(e)


def extract_lyrics_whisperx(
    audio_path: str,
    lyrics_text: str,
    language: str | None = None,
) -> tuple[list[Word], list[int]] | tuple[None, str]:
    """WhisperX: fixed transcript via dummy segment then align. Returns (words, line_breaks) or (None, reason)."""
    try:
        import whisperx
        import torch
    except ImportError as e:
        return None, f"whisperx not installed (pip install whisperx): {e}"

    lines, all_words, line_breaks = preprocess_lyrics_for_alignment(lyrics_text)
    if not all_words:
        return None, "No lyrics words after cleaning"

    lang_code = _normalize_lang_code(language)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        audio = whisperx.load_audio(audio_path)
        duration_sec = len(audio) / 16000
        transcript = " ".join(all_words)
        segments = [{"start": 0.0, "end": duration_sec, "text": transcript}]
        model_a, metadata = whisperx.load_align_model(language_code=lang_code, device=device)
        aligned = whisperx.align(segments, model_a, metadata, audio, device, return_char_alignments=False)
    except Exception as e:
        log.warning("WhisperX failed: %s", e)
        return None, str(e)
    if not aligned or "segments" not in aligned:
        return None, "WhisperX align returned no segments"
    words_out: list[Word] = []
    for seg in aligned["segments"]:
        for w in seg.get("words", []):
            words_out.append(Word(text=w.get("word", ""), start=float(w.get("start", 0)), end=float(w.get("end", 0)), confidence=1.0))
    if len(words_out) == len(all_words):
        words_out = [
            Word(text=all_words[i], start=w.start, end=w.end, confidence=w.confidence)
            for i, w in enumerate(words_out)
        ]
    elif len(words_out) != len(all_words):
        class _W:
            def __init__(self, text: str, start_time: float, end_time: float):
                self.text = text
                self.start_time = start_time
                self.end_time = end_time
        words_out = _aligner_words_to_lyrics_order(
            [_W(w.text, w.start, w.end) for w in words_out],
            all_words,
        )
    log.info("WhisperX produced %d words, %d line breaks", len(words_out), len(line_breaks))
    return (words_out, line_breaks)


def extract_lyrics_mfa(
    audio_path: str,
    lyrics_text: str,
    language: str | None = None,
) -> tuple[list[Word], list[int]] | tuple[None, str]:
    """Montreal Forced Aligner (pip install Montreal-Forced-Aligner): CLI align. Returns (words, line_breaks) or (None, reason)."""
    try:
        import os
        import subprocess
        import tempfile
        from pathlib import Path
    except ImportError:
        return None, "Montreal-Forced-Aligner not available"

    lines, all_words, line_breaks = preprocess_lyrics_for_alignment(lyrics_text)
    if not all_words:
        return None, "No lyrics words after cleaning"

    with tempfile.TemporaryDirectory(prefix="mfa_") as tmp:
        corpus = Path(tmp) / "corpus"
        corpus.mkdir()
        utt_dir = corpus / "1"
        utt_dir.mkdir()
        audio_link = utt_dir / "utt.wav"
        try:
            import shutil
            shutil.copy2(audio_path, audio_link)
        except Exception as e:
            return None, str(e)
        with open(utt_dir / "utt.lab", "w", encoding="utf-8") as f:
            f.write(" ".join(all_words))
        out_dir = Path(tmp) / "out"
        out_dir.mkdir()
        try:
            subprocess.run(
                ["mfa", "align", str(corpus), "english_us_arpa", "english_us_arpa", str(out_dir), "--clean"],
                capture_output=True,
                text=True,
                timeout=600,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            return None, f"MFA align failed: {e.stderr or e.stdout}"
        except FileNotFoundError:
            return None, "mfa CLI not found (pip install Montreal-Forced-Aligner)"
        except subprocess.TimeoutExpired:
            return None, "MFA timed out"
        textgrid = out_dir / "1" / "utt.TextGrid"
        if not textgrid.exists():
            return None, "MFA produced no TextGrid"
        try:
            with open(textgrid, encoding="utf-8") as f:
                content = f.read()
            words_out = _parse_mfa_textgrid(content, all_words)
        except Exception as e:
            return None, str(e)
    log.info("MFA produced %d words, %d line breaks", len(words_out), len(line_breaks))
    return (words_out, line_breaks)


def _parse_mfa_textgrid(content: str, lyrics_words: list[str]) -> list[Word]:
    """Parse Praat TextGrid word tier into Word list."""
    import re
    intervals = re.findall(r"intervals \[.*?\]:\s*xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = \"([^\"]*)\"", content, re.DOTALL)
    words_out: list[Word] = []
    for i, (xmin, xmax, text) in enumerate(intervals):
        if not text.strip():
            continue
        lw = lyrics_words[len(words_out)] if len(words_out) < len(lyrics_words) else text.strip()
        words_out.append(Word(text=lw, start=float(xmin), end=float(xmax), confidence=1.0))
    while len(words_out) < len(lyrics_words):
        t = words_out[-1].end + 0.2 if words_out else 0.0
        words_out.append(Word(text=lyrics_words[len(words_out)], start=t, end=t + 0.3, confidence=0.0))
    return words_out[: len(lyrics_words)]


def extract_lyrics_gentle(
    audio_path: str,
    lyrics_text: str,
    language: str | None = None,
) -> tuple[list[Word], list[int]] | tuple[None, str]:
    """Gentle (Docker/API): POST to Gentle service. Returns (words, line_breaks) or (None, reason)."""
    try:
        import urllib.request
        import urllib.error
        import json
    except ImportError:
        return None, "urllib not available"

    lines, all_words, line_breaks = preprocess_lyrics_for_alignment(lyrics_text)
    if not all_words:
        return None, "No lyrics words after cleaning"

    url = "http://localhost:8765/transcribe"
    transcript = " ".join(all_words)
    try:
        with open(audio_path, "rb") as f:
            audio_data = f.read()
        boundary = "----FormBoundaryKaraokeCreator"
        body_parts = [
            b"--" + boundary.encode(),
            b'Content-Disposition: form-data; name="audio"; filename="audio.wav"',
            b"Content-Type: audio/wav",
            b"",
            audio_data,
            b"--" + boundary.encode(),
            b'Content-Disposition: form-data; name="transcript"',
            b"",
            transcript.encode("utf-8"),
            b"--" + boundary.encode() + b"--",
        ]
        req = urllib.request.Request(
            url,
            data=b"\r\n".join(body_parts),
            method="POST",
        )
        req.add_header("Content-Type", "multipart/form-data; boundary=" + boundary)
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        return None, f"Gentle not reachable (run Docker: lowerquality/gentle): {e}"
    except Exception as e:
        return None, str(e)
    words_json = data.get("words", [])
    words_out: list[Word] = []
    for i, w in enumerate(words_json):
        lw = all_words[i] if i < len(all_words) else w.get("word", "")
        words_out.append(Word(text=lw, start=float(w.get("start", 0)), end=float(w.get("end", 0)), confidence=1.0))
    while len(words_out) < len(all_words):
        t = words_out[-1].end + 0.2 if words_out else 0.0
        words_out.append(Word(text=all_words[len(words_out)], start=t, end=t + 0.3, confidence=0.0))
    log.info("Gentle produced %d words, %d line breaks", len(words_out), len(line_breaks))
    return (words_out[: len(all_words)], line_breaks)


def extract_lyrics_autolyrixalign(
    audio_path: str,
    lyrics_text: str,
    language: str | None = None,
) -> tuple[list[Word], list[int]] | tuple[None, str]:
    """AutoLyrixAlign (Karaokeer Docker or script): optional. Returns (words, line_breaks) or (None, reason)."""
    import os
    import subprocess
    import tempfile
    from pathlib import Path

    lines, all_words, line_breaks = preprocess_lyrics_for_alignment(lyrics_text)
    if not all_words:
        return None, "No lyrics words after cleaning"

    root = os.environ.get("AUTOLYRIXALIGN_ROOT")
    if not root or not os.path.isdir(root):
        return None, "AUTOLYRIXALIGN_ROOT not set (download from NUS AutoLyrixAlign, extract, set env)"
    script = os.path.join(root, "align.py")
    if not os.path.isfile(script):
        return None, "AUTOLYRIXALIGN_ROOT has no align.py"
    with tempfile.TemporaryDirectory(prefix="ala_") as tmp:
        txt_path = os.path.join(tmp, "lyrics.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(" ".join(line) + "\n")
        out_path = os.path.join(tmp, "out.txt")
        try:
            subprocess.run(
                [sys.executable, script, audio_path, txt_path, out_path],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=600,
                check=True,
            )
        except Exception as e:
            return None, str(e)
        if not os.path.isfile(out_path):
            return None, "AutoLyrixAlign produced no output"
        words_out = []
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    words_out.append(Word(text=parts[2], start=float(parts[0]), end=float(parts[1]), confidence=1.0))
        # Always use our lyrics words for text so Unicode (e.g. Polish ą, ę, ł) is preserved
        if len(words_out) == len(all_words):
            words_out = [
                Word(text=all_words[i], start=w.start, end=w.end, confidence=w.confidence)
                for i, w in enumerate(words_out)
            ]
        else:
            words_out = _aligner_words_to_lyrics_order(
                [type("W", (), {"text": w.text, "start_time": w.start, "end_time": w.end})() for w in words_out],
                all_words,
            )
    return (words_out, line_breaks)
