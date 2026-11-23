"""Optional self-hosted model adapters used by the alignment worker.

Imports and model loading are lazy so the deterministic alignment core and
the existing CPU fallback remain available when GPU weights are not present.
Set ``ALIGNMENT_ASR_BACKEND=qwen`` to enable the Qwen3-ASR adapter.
"""

from __future__ import annotations

import os
import math
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from models import LyricToken
from processing.alignment import ASRToken, global_monotonic_match, normalize_lyric_token


class ModelAdapterError(RuntimeError):
    """Raised when an optional local model cannot produce usable evidence."""


@contextmanager
def _trim_leading_false_activity(
    audio_path: str,
    activity_spans: Sequence[tuple[float, float]],
    anchor_tokens: Sequence[ASRToken] = (),
):
    """Temporarily remove a false vocal island before a long intro gap.

    A full-track forced path can spend its first few seconds matching the
    lyric transcript to piano/drum energy.  If the separated-vocal activity
    has a short leading island followed by a genuinely long silence, align the
    transcript on the first post-gap island instead.  This is deliberately
    conservative: ordinary phrase gaps and songs that start with vocals are
    left on the original timeline.
    """

    spans = sorted(
        (float(start), float(end))
        for start, end in activity_spans
        if math.isfinite(float(start))
        and math.isfinite(float(end))
        and float(end) > float(start)
    )
    leading_gap_index = next(
        (
            index
            for index, (_left, right) in enumerate(spans[:-1])
            if spans[index + 1][0] - right >= 8.0
        ),
        None,
    )
    trim_anchor: float | None = None
    if (
        leading_gap_index is not None
        and spans[0][0] <= 4.0
        and spans[0][1] - spans[0][0] <= 5.0
    ):
        trim_anchor = spans[leading_gap_index + 1][0]
    # When the activity detector sees no short false island, a clean vocal
    # stem can still contain a long instrumental lead-in.  Use the first
    # lexical ASR clock as a confirmation before trimming it: this avoids the
    # known separator failure mode where the stem misses an early vocal and
    # reports its first island much later than the song's true onset.
    if trim_anchor is None and spans and anchor_tokens:
        first_anchor = min(
            (
                token
                for token in anchor_tokens
                if _valid_times(token.start, token.end) is not None
            ),
            key=lambda token: float(token.start),
            default=None,
        )
        if first_anchor is not None and 4.0 <= float(first_anchor.start) <= 40.0:
            matching_span = next(
                (
                    (index, span)
                    for index, span in enumerate(spans)
                    if span[0] >= 4.0
                    and abs(float(first_anchor.start) - span[0]) <= 2.0
                    and (index == 0 or span[0] - spans[index - 1][1] >= 1.0)
                ),
                None,
            )
            if matching_span is not None:
                trim_anchor = matching_span[1][0]
    if trim_anchor is None:
        yield audio_path, 0.0
        return

    try:
        leading_pad = float(os.environ.get("ALIGNMENT_FORCEALIGN_LEADING_PAD_S", "0.25"))
    except ValueError:
        leading_pad = 0.25
    if not math.isfinite(leading_pad) or leading_pad < 0.0:
        leading_pad = 0.25
    trim_start = max(0.0, trim_anchor - min(2.0, leading_pad))
    try:
        import soundfile as sf

        info = sf.info(audio_path)
        start_frame = min(info.frames, max(0, int(round(trim_start * info.samplerate))))
        if start_frame <= 0 or start_frame >= info.frames - info.samplerate:
            yield audio_path, 0.0
            return
    except Exception:
        yield audio_path, 0.0
        return

    try:
        with tempfile.TemporaryDirectory(prefix="forcealign-trim-") as temp_dir:
            trimmed_path = os.path.join(temp_dir, "audio.wav")
            with sf.SoundFile(audio_path) as source:
                source.seek(start_frame)
                with sf.SoundFile(
                    trimmed_path,
                    mode="w",
                    samplerate=source.samplerate,
                    channels=source.channels,
                    subtype=source.subtype,
                ) as destination:
                    while True:
                        block = source.read(262144, dtype="float32", always_2d=True)
                        if len(block) == 0:
                            break
                        destination.write(block)
            yield trimmed_path, start_frame / float(info.samplerate)
    except Exception:
        # A trim is an optimization, never a reason to lose the normal
        # full-track alignment path on an unusual audio container.
        yield audio_path, 0.0


def _ctc_safe_text(text: str) -> str:
    """Map typographic punctuation to the English CTC alphabet."""

    return text.translate(str.maketrans({
        "’": "'",
        "‘": "'",
        "‛": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "–": "-",
        "—": "-",
        "‑": "-",
    }))


def _resolve_whisperx_device(device: str | None) -> str:
    """Choose an explicit WhisperX device or the best local device."""

    configured = (device or os.environ.get("WHISPERX_DEVICE", "")).strip()
    if configured and configured.casefold() != "auto":
        return configured
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _flatten_items(value: Any) -> list[Any]:
    """Normalize Qwen batches, result wrappers, and older flat results."""

    if value is None:
        return []
    if isinstance(value, dict):
        if "items" in value:
            return _flatten_items(value["items"])
        return [value]
    if isinstance(value, (list, tuple)):
        flattened: list[Any] = []
        for item in value:
            flattened.extend(_flatten_items(item))
        return flattened
    items = getattr(value, "items", None)
    if items is not None and not callable(items):
        return _flatten_items(items)
    return [value]


def _confidence(value: Any) -> float:
    if value is None:
        return 1.0
    numeric = float(value)
    return 0.0 if not math.isfinite(numeric) else max(0.0, min(1.0, numeric))


def _valid_times(start: Any, end: Any) -> tuple[float, float] | None:
    if start is None or end is None:
        return None
    try:
        start_value, end_value = float(start), float(end)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(start_value) or not math.isfinite(end_value) or start_value < 0.0 or end_value < start_value:
        return None
    return start_value, end_value


def _language_name(language: str) -> str:
    return "Polish" if language == "pl" else "English"


def _anchored_segments(
    text: str,
    anchor_tokens: Sequence[ASRToken],
    duration: float,
    *,
    max_words: int | None = None,
    padding_seconds: float = 0.75,
) -> list[dict[str, object]]:
    """Split CTC text into small anchor-bounded alignment windows.

    WhisperX is much more stable when repeated lyric sections are aligned in
    local windows. A single full-song segment lets a missed syllable shift all
    later words, especially in a repeated chorus. ASR timestamps only choose
    the search windows here; the returned word boundaries still come from CTC.
    """

    words = text.split()
    if max_words is None:
        try:
            max_words = max(1, int(os.environ.get("WHISPERX_WINDOW_WORDS", "16")))
        except ValueError:
            max_words = 8
    if not words or duration <= 0.0 or not anchor_tokens:
        return [{
            "start": 0.0,
            "end": max(duration, 0.0),
            "text": _ctc_safe_text(text),
            "sequence_start": 0,
        }]
    if not all(hasattr(anchor, "text") for anchor in anchor_tokens):
        start, end = _anchor_window(anchor_tokens, duration)
        return [{
            "start": start,
            "end": end,
            "text": _ctc_safe_text(text),
            "sequence_start": 0,
        }]
    lyric_tokens = [
        LyricToken(
            id=f"ctc-window:{index}",
            lyric_revision_id="ctc-window",
            display_text=word,
            normalized_text=normalize_lyric_token(word),
            line_index=0,
            token_index=index,
            source_index=index,
        )
        for index, word in enumerate(words)
        if normalize_lyric_token(word)
    ]
    if len(lyric_tokens) != len(words):
        return [{
            "start": 0.0,
            "end": duration,
            "text": _ctc_safe_text(text),
            "sequence_start": 0,
        }]
    match = global_monotonic_match(lyric_tokens, anchor_tokens)
    prepared_anchors = match.prepared_asr_tokens or tuple(anchor_tokens)
    anchor_by_word = {
        item.lyric_index: prepared_anchors[item.asr_index]
        for item in match.matches
        if item.matched and item.asr_index is not None
    }
    if len(anchor_by_word) < max(4, len(words) // 10):
        return [{"start": 0.0, "end": duration, "text": _ctc_safe_text(text)}]

    segments: list[dict[str, object]] = []
    for word_start in range(0, len(words), max_words):
        word_end = min(len(words), word_start + max_words)
        batch_anchors = [
            anchor_by_word[index]
            for index in range(word_start, word_end)
            if index in anchor_by_word
        ]
        if batch_anchors:
            start = max(0.0, min(anchor.start for anchor in batch_anchors) - padding_seconds)
            end = min(duration, max(anchor.end for anchor in batch_anchors) + padding_seconds)
        else:
            # Do not silently drop a lyric window just because Whisper missed
            # every word in it.  That is especially damaging at the start of
            # songs with an instrumental intro: the first anchored window can
            # begin many seconds into the recording, after which every
            # earlier canonical word is forced into a fabricated fallback
            # span. Bound an unanchored window by its nearest lexical clocks
            # and let CTC decide whether there is actually sung audio there.
            previous = [
                anchor
                for index, anchor in anchor_by_word.items()
                if index < word_start
            ]
            following = [
                anchor
                for index, anchor in anchor_by_word.items()
                if index >= word_end
            ]
            start = (
                0.0
                if not previous
                else max(0.0, max(anchor.end for anchor in previous) - padding_seconds)
            )
            end = (
                duration
                if not following
                else min(duration, min(anchor.start for anchor in following) + padding_seconds)
            )
        if end <= start:
            continue
        segments.append({
            "start": start,
            "end": end,
            "text": _ctc_safe_text(" ".join(words[word_start:word_end])),
            "sequence_start": word_start,
        })
    return segments or [{
        "start": 0.0,
        "end": duration,
        "text": _ctc_safe_text(text),
        "sequence_start": 0,
    }]


class Qwen3ASRAdapter:
    """Lazy wrapper around the official ``qwen-asr`` Transformers API."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-ASR-1.7B",
        forced_aligner_name: str | None = "Qwen/Qwen3-ForcedAligner-0.6B",
        device: str = "cuda:0",
    ) -> None:
        self.model_name = model_name
        self.forced_aligner_name = forced_aligner_name
        self.device = device
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            import torch
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:
            raise ModelAdapterError("qwen-asr and torch are required for ALIGNMENT_ASR_BACKEND=qwen") from exc
        kwargs: dict[str, Any] = {"device_map": self.device, "dtype": torch.bfloat16}
        if self.forced_aligner_name:
            kwargs["forced_aligner"] = self.forced_aligner_name
            kwargs["forced_aligner_kwargs"] = {"device_map": self.device, "dtype": torch.bfloat16}
        try:
            self._model = Qwen3ASRModel.from_pretrained(self.model_name, **kwargs)
        except Exception as exc:
            raise ModelAdapterError(f"Unable to load {self.model_name}") from exc
        return self._model

    def transcribe(self, audio_path: str, language: str) -> list[ASRToken]:
        model = self._load()
        try:
            results = model.transcribe(
                audio=audio_path,
                language=_language_name(language),
                return_time_stamps=True,
            )
        except Exception as exc:
            raise ModelAdapterError(f"Qwen3-ASR failed for {audio_path}") from exc

        tokens: list[ASRToken] = []
        for result in _flatten_items(results):
            for stamp in _flatten_items(_value(result, "time_stamps", [])):
                text = str(_value(stamp, "text", "")).strip()
                start = _value(stamp, "start_time")
                end = _value(stamp, "end_time")
                times = _valid_times(start, end)
                if not text or times is None:
                    continue
                tokens.append(
                    ASRToken(
                        text=text,
                        start=times[0],
                        end=times[1],
                        confidence=_confidence(_value(stamp, "confidence", 1.0)),
                        model=self.model_name,
                        audio_source="",
                        evidence_type="anchor",
                    )
                )
        if not tokens:
            raise ModelAdapterError("Qwen3-ASR returned no word timestamps")
        return tokens


class Qwen3ForcedAlignerAdapter:
    """Lazy wrapper for Qwen3-ForcedAligner candidate generation."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-ForcedAligner-0.6B",
        device: str = "cuda:0",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            import torch
            from qwen_asr import Qwen3ForcedAligner
        except ImportError as exc:
            raise ModelAdapterError("qwen-asr and torch are required for Qwen forced alignment") from exc
        try:
            self._model = Qwen3ForcedAligner.from_pretrained(
                self.model_name,
                device_map=self.device,
                dtype=torch.bfloat16,
            )
        except Exception as exc:
            raise ModelAdapterError(f"Unable to load {self.model_name}") from exc
        return self._model

    def align(self, audio_path: str, text: str, language: str) -> list[ASRToken]:
        if language != "en":
            raise ModelAdapterError("Qwen3-ForcedAligner is not a Polish candidate in version 1")
        return self._align_once(audio_path, text, language)

    def _align_once(self, audio_path: str, text: str, language: str) -> list[ASRToken]:
        try:
            results = self._load().align(
                audio=audio_path,
                text=text,
                language=_language_name(language),
            )
        except Exception as exc:
            raise ModelAdapterError(f"Qwen forced alignment failed for {audio_path}") from exc
        tokens: list[ASRToken] = []
        for item in _flatten_items(results):
            word = str(_value(item, "text", _value(item, "word", ""))).strip()
            start = _value(item, "start_time")
            end = _value(item, "end_time")
            times = _valid_times(start, end)
            if word and times is not None:
                tokens.append(
                    ASRToken(
                        text=word,
                        start=times[0],
                        end=times[1],
                        confidence=_confidence(_value(item, "confidence", 1.0)),
                        model=self.model_name,
                        audio_source="vocal_stem",
                        evidence_type="forced",
                    )
                )
        if not tokens:
            raise ModelAdapterError("Qwen forced aligner returned no word timestamps")
        return tokens

    def align_chunked(
        self,
        audio_path: str,
        text: str,
        language: str,
        *,
        chunk_seconds: float = 240.0,
        overlap_seconds: float = 15.0,
        anchor_tokens: Sequence[ASRToken] = (),
    ) -> list[ASRToken]:
        """Align long songs in bounded overlapping windows.

        Qwen's forced aligner has a finite context window. Text is partitioned
        proportionally to each audio window and duplicate overlap words are
        removed by source order and timestamp.
        """

        try:
            import soundfile as sf

            info = sf.info(audio_path)
            duration = float(info.duration)
            sample_rate = int(info.samplerate)
        except Exception as exc:
            raise ModelAdapterError(f"Unable to inspect audio for chunked alignment: {audio_path}") from exc
        if duration <= chunk_seconds:
            return self.align(audio_path, text, language)
        if overlap_seconds <= 0 or overlap_seconds >= chunk_seconds:
            raise ModelAdapterError("Forced-alignment overlap must be smaller than the chunk size")

        words = text.split()
        if not words:
            raise ModelAdapterError("Forced alignment text is empty")
        step = chunk_seconds - overlap_seconds
        aligned: list[ASRToken] = []
        chunk_start = 0.0
        with tempfile.TemporaryDirectory(prefix="qwen-align-") as temp_dir:
            chunk_index = 0
            while chunk_start < duration:
                chunk_end = min(duration, chunk_start + chunk_seconds)
                text_start, text_end = _chunk_text_range(
                    words,
                    chunk_start,
                    chunk_end,
                    duration,
                    overlap_seconds,
                    anchor_tokens,
                    chunk_index,
                )
                chunk_text = " ".join(words[text_start:text_end])
                if chunk_text:
                    try:
                        data, _ = sf.read(
                            audio_path,
                            start=int(chunk_start * sample_rate),
                            stop=int(chunk_end * sample_rate),
                            always_2d=False,
                        )
                        chunk_path = os.path.join(temp_dir, f"chunk-{chunk_index}.wav")
                        sf.write(chunk_path, data, sample_rate)
                        chunk_tokens = self._align_once(chunk_path, chunk_text, language)
                        local_indices = _matched_lyric_indices(
                            words[text_start:text_end],
                            chunk_tokens,
                        )
                        sequence_indices = [
                            text_start + local_indices.get(index)
                            if index in local_indices
                            else None
                            for index in range(len(chunk_tokens))
                        ]
                        aligned.extend(
                            replace_token_times(
                                chunk_tokens,
                                chunk_start,
                                sequence_indices=sequence_indices,
                                chunk_index=chunk_index,
                            )
                        )
                    except ModelAdapterError:
                        raise
                    except Exception as exc:
                        raise ModelAdapterError(f"Qwen forced alignment failed for chunk {chunk_index}") from exc
                chunk_index += 1
                if chunk_end >= duration:
                    break
                chunk_start += step

        deduped: list[ASRToken] = []
        for token in sorted(aligned, key=lambda item: (item.start, item.end)):
            same_overlap_token = (
                deduped
                and token.sequence_index is not None
                and deduped[-1].sequence_index == token.sequence_index
                and token.chunk_index != deduped[-1].chunk_index
                and abs(token.start - deduped[-1].start) < overlap_seconds
            )
            if same_overlap_token:
                if token.confidence > deduped[-1].confidence:
                    deduped[-1] = token
                continue
            deduped.append(token)
        if not deduped:
            raise ModelAdapterError("Qwen forced aligner returned no chunk timestamps")
        return deduped


class WhisperXCTCAdapter:
    """Lazy wrapper for WhisperX's language-specific CTC aligner."""

    def __init__(self, device: str | None = None, model_name: str | None = None) -> None:
        self.device = _resolve_whisperx_device(device)
        self.model_name = model_name
        self._model: Any = None
        self._metadata: dict[str, Any] | None = None
        self._language = ""

    def _load(self, language: str) -> tuple[Any, dict[str, Any]]:
        if self._model is not None and self._metadata is not None and self._language == language:
            return self._model, self._metadata
        try:
            import whisperx
        except ImportError as exc:
            raise ModelAdapterError("whisperx is required for CTC forced-alignment candidates") from exc
        try:
            model, metadata = whisperx.load_align_model(
                language_code=language,
                device=self.device,
                model_name=self.model_name,
            )
        except Exception as exc:
            raise ModelAdapterError(f"Unable to load WhisperX {language} alignment model") from exc
        self._model = model
        self._metadata = metadata
        self._language = language
        return model, metadata

    def align(
        self,
        audio_path: str,
        text: str,
        language: str,
        *,
        anchor_tokens: Sequence[ASRToken] = (),
        sequence_offset: int = 0,
    ) -> list[ASRToken]:
        try:
            import soundfile as sf
            import whisperx

            info = sf.info(audio_path)
            duration = float(info.duration)
            model, metadata = self._load(language)
            segments = _anchored_segments(text, anchor_tokens, duration)
            result = whisperx.align(
                segments,
                model,
                metadata,
                audio_path,
                self.device,
            )
        except ModelAdapterError:
            raise
        except Exception as exc:
            raise ModelAdapterError(f"WhisperX forced alignment failed for {audio_path}") from exc

        result_segments = result.get("segments", []) if isinstance(result, dict) else _value(result, "segments", [])
        tokens: list[ASRToken] = []
        for segment in result_segments or []:
            words = segment.get("words", []) if isinstance(segment, dict) else _value(segment, "words", [])
            segment_start = _value(segment, "start")
            segment_end = _value(segment, "end")
            matching_segment = min(
                segments,
                key=lambda expected: abs(
                    float(expected["start"])
                    - float(segment_start if segment_start is not None else expected["start"])
                ),
                default=None,
            )
            expected_words = (
                str(matching_segment.get("text", "")).split()
                if matching_segment is not None
                else []
            )
            local_words = [
                ASRToken(
                    text=str(_value(item, "word", _value(item, "text", ""))).strip(),
                    start=float(_value(item, "start", 0.0) or 0.0),
                    end=float(_value(item, "end", 0.0) or 0.0),
                )
                for item in words or []
                if _value(item, "word", _value(item, "text", ""))
            ]
            local_indices = _matched_lyric_indices(expected_words, local_words)
            local_sequence_start = int(matching_segment.get("sequence_start", 0)) if matching_segment else 0
            for item in words or []:
                word = str(_value(item, "word", _value(item, "text", ""))).strip()
                start = _value(item, "start")
                end = _value(item, "end")
                times = _valid_times(start, end)
                if word and times is not None:
                    item_index = next(
                        (
                            index
                            for index, local_word in enumerate(local_words)
                            if local_word.text == word
                            and abs(local_word.start - times[0]) < 0.001
                            and abs(local_word.end - times[1]) < 0.001
                        ),
                        None,
                    )
                    tokens.append(
                        ASRToken(
                            text=word,
                            start=times[0],
                            end=times[1],
                            confidence=_confidence(_value(item, "score", 1.0)),
                            model="whisperx-ctc",
                            audio_source="vocal_stem",
                            evidence_type="ctc",
                            sequence_index=(
                                sequence_offset + local_sequence_start + local_indices[item_index]
                                if item_index is not None and item_index in local_indices
                                else None
                            ),
                        )
                    )
        if not tokens:
            raise ModelAdapterError("WhisperX returned no word timestamps")
        return tokens

    def align_chunked(
        self,
        audio_path: str,
        text: str,
        language: str,
        *,
        chunk_seconds: float = 240.0,
        overlap_seconds: float = 15.0,
        anchor_tokens: Sequence[ASRToken] = (),
    ) -> list[ASRToken]:
        """Bound CTC alignment memory and remove overlap duplicates."""

        try:
            import soundfile as sf

            info = sf.info(audio_path)
            duration = float(info.duration)
            sample_rate = int(info.samplerate)
        except Exception as exc:
            raise ModelAdapterError(f"Unable to inspect audio for chunked CTC alignment: {audio_path}") from exc
        if duration <= chunk_seconds:
            return self.align(audio_path, text, language, anchor_tokens=anchor_tokens)
        if overlap_seconds <= 0 or overlap_seconds >= chunk_seconds:
            raise ModelAdapterError("CTC alignment overlap must be smaller than the chunk size")
        words = text.split()
        if not words:
            raise ModelAdapterError("CTC alignment text is empty")

        aligned: list[ASRToken] = []
        step = chunk_seconds - overlap_seconds
        chunk_start = 0.0
        with tempfile.TemporaryDirectory(prefix="whisperx-align-") as temp_dir:
            chunk_index = 0
            while chunk_start < duration:
                chunk_end = min(duration, chunk_start + chunk_seconds)
                text_start, text_end = _chunk_text_range(
                    words,
                    chunk_start,
                    chunk_end,
                    duration,
                    overlap_seconds,
                    anchor_tokens,
                    chunk_index,
                )
                chunk_text = " ".join(words[text_start:text_end])
                if chunk_text:
                    try:
                        data, _ = sf.read(
                            audio_path,
                            start=int(chunk_start * sample_rate),
                            stop=int(chunk_end * sample_rate),
                            always_2d=False,
                        )
                        chunk_path = os.path.join(temp_dir, f"chunk-{chunk_index}.wav")
                        sf.write(chunk_path, data, sample_rate)
                        aligned.extend(
                            replace_token_times(
                                self.align(
                                    chunk_path,
                                    chunk_text,
                                    language,
                                    sequence_offset=text_start,
                                ),
                                chunk_start,
                                chunk_index=chunk_index,
                            )
                        )
                    except ModelAdapterError:
                        raise
                    except Exception as exc:
                        raise ModelAdapterError(f"WhisperX alignment failed for chunk {chunk_index}") from exc
                chunk_index += 1
                if chunk_end >= duration:
                    break
                chunk_start += step

        deduped: list[ASRToken] = []
        for token in sorted(aligned, key=lambda item: (item.start, item.end)):
            same_overlap_token = (
                deduped
                and token.sequence_index is not None
                and deduped[-1].sequence_index == token.sequence_index
                and token.chunk_index != deduped[-1].chunk_index
                and abs(token.start - deduped[-1].start) < overlap_seconds
            )
            if same_overlap_token:
                if token.confidence > deduped[-1].confidence:
                    deduped[-1] = token
                continue
            deduped.append(token)
        if not deduped:
            raise ModelAdapterError("WhisperX returned no chunk timestamps")
        return deduped


class ForceAlignAdapter:
    """Optional local Wav2Vec2 forced alignment for English lyrics.

    ``forcealign`` exposes a convenient wrapper, but its public ``inference``
    method also invokes a phonemizer that is not needed for word boundaries
    and is frequently unavailable in a worker environment. We use the
    wrapper's trellis/backtracking primitives directly and keep the model
    lazy so the normal WhisperX path is unchanged.
    """

    def __init__(self) -> None:
        self._aligner: Any = None
        self._audio_path = ""

    @staticmethod
    def _transcript_tokens(aligner: Any, text: str) -> tuple[str, list[int]]:
        words = [
            "".join(character for character in word.upper() if "A" <= character <= "Z")
            for word in text.split()
        ]
        words = [word for word in words if word]
        if not words:
            raise ModelAdapterError("ForceAlign text contains no English words")
        transcript = "|".join(words) + "|"
        try:
            tokens = [aligner.dictionary[character] for character in transcript]
        except KeyError as exc:
            raise ModelAdapterError(f"ForceAlign cannot encode transcript character {exc.args[0]!r}") from exc
        return transcript, tokens

    def _load(self, audio_path: str, text: str) -> Any:
        if self._aligner is not None and self._audio_path == os.path.abspath(audio_path):
            return self._aligner
        try:
            from forcealign import ForceAlign
        except ImportError as exc:
            raise ModelAdapterError(
                "forcealign is required for ALIGNMENT_WAV2VEC_BACKEND=forcealign"
            ) from exc
        try:
            self._aligner = ForceAlign(audio_path, text)
        except Exception as exc:
            raise ModelAdapterError(f"Unable to load ForceAlign for {audio_path}") from exc
        self._audio_path = os.path.abspath(audio_path)
        return self._aligner

    def align(
        self,
        audio_path: str,
        text: str,
        language: str,
        *,
        anchor_tokens: Sequence[ASRToken] = (),
        activity_spans: Sequence[tuple[float, float]] = (),
    ) -> list[ASRToken]:
        if language != "en":
            raise ModelAdapterError("ForceAlign is an English-only candidate in version 1")
        with _trim_leading_false_activity(audio_path, activity_spans, anchor_tokens) as (
            aligned_audio_path,
            timeline_offset,
        ):
            aligner = self._load(aligned_audio_path, text)
            try:
                transcript, tokens = self._transcript_tokens(aligner, text)
                # The object caches the audio emission; changing only the
                # transcript lets repeated jobs reuse it for the same audio path.
                aligner.transcript = transcript
                aligner.tokens = tokens
                path = aligner.backtrack(aligner.get_trellis())
                segments = aligner.merge_words(aligner.merge_repeats(path))
                frame_count = int(aligner.emission.size(0))
                sample_count = int(aligner.waveform.size(1))
                sample_rate = float(aligner.bundle.sample_rate)
            except ModelAdapterError:
                raise
            except Exception as exc:
                raise ModelAdapterError(f"ForceAlign failed for {audio_path}") from exc
            if frame_count <= 0 or sample_count <= 0 or sample_rate <= 0:
                raise ModelAdapterError("ForceAlign returned an invalid audio/emission shape")

            samples_per_frame = sample_count / frame_count
            result: list[ASRToken] = []
            for segment in segments or []:
                text_value = str(getattr(segment, "label", "")).strip()
                times = _valid_times(
                    timeline_offset + float(getattr(segment, "start")) * samples_per_frame / sample_rate,
                    timeline_offset + float(getattr(segment, "end")) * samples_per_frame / sample_rate,
                )
                if not text_value or times is None or times[1] <= times[0]:
                    continue
                result.append(
                    ASRToken(
                        text=text_value,
                        start=times[0],
                        end=times[1],
                        confidence=_confidence(getattr(segment, "score", 1.0)),
                        model="forcealign-wav2vec2",
                        audio_source="vocal_stem",
                        evidence_type="forced",
                        sequence_index=len(result),
                    )
                )
            if not result:
                raise ModelAdapterError("ForceAlign returned no word timestamps")
            # Trellis backtracking can assign a long leading silence to the first
            # repeated lyric token (for example ``OOH`` in a song with a 15-second
            # intro), then align the second copy to the real vocal onset.  Keep
            # the lexical sequence monotonic by dropping only this pathological
            # synthetic span; ordinary opening words are untouched.
            if len(result) >= 2:
                first, second = result[0], result[1]
                try:
                    max_leading_span = float(
                        os.environ.get("ALIGNMENT_FORCEALIGN_MAX_LEADING_SPAN", "4.0")
                    )
                    minimum_following_gap = float(
                        os.environ.get("ALIGNMENT_FORCEALIGN_MIN_FOLLOWING_GAP", "1.0")
                    )
                except ValueError:
                    max_leading_span = 4.0
                    minimum_following_gap = 1.0
                if (
                    math.isfinite(max_leading_span)
                    and math.isfinite(minimum_following_gap)
                    and first.start <= timeline_offset + 0.05
                    and first.end - first.start > max(0.5, max_leading_span)
                    and second.start - first.end >= max(0.25, minimum_following_gap)
                ):
                    kept = result[1:]
                    result = [
                        replace(
                            token,
                            # The real first vocal token now supplies lyric word
                            # zero; retain the original indices for the remaining
                            # transcript so the repeated second word stays an
                            # intentional gap for interpolation.
                            sequence_index=0 if index == 0 else token.sequence_index,
                        )
                        for index, token in enumerate(kept)
                    ]
            # If the first indexed forced token absorbs opening silence, a
            # matching early ASR word provides a safe onset correction.  Keep
            # this limited to the first few seconds and lexical agreement so
            # it cannot move a genuine instrumental intro.
            if result and anchor_tokens and result[0].start <= 0.05:
                first_anchor = min(
                    (
                        token
                        for token in anchor_tokens
                        if _valid_times(token.start, token.end) is not None
                    ),
                    key=lambda token: float(token.start),
                    default=None,
                )
                first_text = transcript.split("|", 1)[0].casefold()
                if (
                    first_anchor is not None
                    and float(first_anchor.start) <= 4.0
                    and normalize_lyric_token(first_anchor.text) == normalize_lyric_token(first_text)
                    and abs(float(first_anchor.start) - float(result[0].end)) <= 1.5
                ):
                    corrected_start = max(0.0, float(first_anchor.start) - 0.12)
                    result[0] = replace(
                        result[0],
                        start=min(corrected_start, float(result[0].end) - 0.02),
                    )
            return result

    def align_chunked(
        self,
        audio_path: str,
        text: str,
        language: str,
        *,
        anchor_tokens: Sequence[ASRToken] = (),
        chunk_seconds: float = 60.0,
        overlap_seconds: float = 8.0,
    ) -> list[ASRToken]:
        """Align long recordings in small, lexically anchored windows.

        Wav2Vec trellis backtracking is very good locally, but a missed
        syllable in a four-minute recording can move the trellis onto a later
        repetition and leave a ten-second clock error.  Windows are selected
        from the independent ASR clock and overlap just enough to keep a word
        at a boundary recoverable.  The chart is never consulted here.

        This is intentionally a separate path instead of silently changing
        ``align``: short songs retain the stable full-track behavior and a
        caller can fall back to it if a window produces no usable evidence.
        """

        if language != "en":
            raise ModelAdapterError("ForceAlign is an English-only candidate in version 1")
        try:
            import soundfile as sf

            info = sf.info(audio_path)
            duration = float(info.duration)
            sample_rate = int(info.samplerate)
        except Exception as exc:
            raise ModelAdapterError(f"Unable to inspect audio for chunked alignment: {audio_path}") from exc
        if duration <= 0.0:
            raise ModelAdapterError("ForceAlign audio has no duration")
        if not math.isfinite(chunk_seconds) or chunk_seconds <= 0.0:
            raise ModelAdapterError("ForceAlign chunk size must be positive")
        if not math.isfinite(overlap_seconds) or overlap_seconds <= 0.0 or overlap_seconds >= chunk_seconds:
            raise ModelAdapterError("ForceAlign overlap must be smaller than the chunk size")
        words = text.split()
        if not words:
            raise ModelAdapterError("Force alignment text is empty")

        # A sparse ASR transcript is not a useful clock for choosing windows;
        # in that case the full-track path is safer than inventing lexical
        # boundaries for instrumental sections.
        anchor_to_lyric = _matched_lyric_indices(words, anchor_tokens)
        if len(anchor_to_lyric) < max(4, int(len(words) * 0.12)):
            return self.align(audio_path, text, language, anchor_tokens=anchor_tokens)

        mapped_anchors = [
            (token_index, lyric_index, anchor_tokens[token_index])
            for token_index, lyric_index in anchor_to_lyric.items()
            if 0 <= token_index < len(anchor_tokens)
        ]
        mapped_anchors.sort(key=lambda item: (float(item[2].start), item[1]))
        step = chunk_seconds - overlap_seconds
        aligned: list[ASRToken] = []
        chunk_start = 0.0
        chunk_index = 0

        def text_range(window_start: float, window_end: float) -> tuple[int, int] | None:
            # Use the core window to choose transcript bounds.  Applying the
            # full audio overlap here makes the first window include words
            # from the next chorus, which lets a trellis consume them before
            # the next window gets a chance to align that occurrence.
            anchor_margin = min(1.5, overlap_seconds)
            nearby = [
                lyric_index
                for _anchor_index, lyric_index, anchor in mapped_anchors
                if window_start - anchor_margin <= float(anchor.start) < window_end + anchor_margin
            ]
            if not nearby:
                # An instrumental gap can sit between two lexical anchors. A
                # window touching that gap should still include the words on
                # either side so the next window does not start mid-phrase.
                preceding = [item[1] for item in mapped_anchors if float(item[2].end) <= window_start]
                following = [item[1] for item in mapped_anchors if float(item[2].start) >= window_end]
                if not preceding and not following:
                    return None
                left = max(preceding[-1] if preceding else 0, 0)
                right = min(following[0] + 1 if following else len(words), len(words))
                if right - left <= 1:
                    return None
                return left, right
            # Include every lyric token between the anchors.  Restricting a
            # sparse window to ±1 words is the source of many long-track
            # omissions; contiguous text preserves the local trellis clock.
            left = max(0, min(nearby) - 1)
            right = min(len(words), max(nearby) + 2)
            return left, max(left + 1, right)

        with tempfile.TemporaryDirectory(prefix="forcealign-chunk-") as temp_dir:
            while chunk_start < duration:
                chunk_end = min(duration, chunk_start + chunk_seconds)
                selected = text_range(chunk_start, chunk_end)
                if selected is not None:
                    text_start, text_end = selected
                    chunk_text = " ".join(words[text_start:text_end])
                    try:
                        data, _ = sf.read(
                            audio_path,
                            start=int(round(chunk_start * sample_rate)),
                            stop=int(round(chunk_end * sample_rate)),
                            always_2d=False,
                        )
                        chunk_path = os.path.join(temp_dir, f"chunk-{chunk_index}.wav")
                        sf.write(chunk_path, data, sample_rate)
                        # Pass only anchors that fall in this window, shifted
                        # to its local clock.  This lets the first-token
                        # leading-silence guard operate inside a chunk while
                        # avoiding a second application of the global offset.
                        local_anchors = [
                            replace(
                                anchor,
                                start=max(0.0, float(anchor.start) - chunk_start),
                                end=max(0.0, float(anchor.end) - chunk_start),
                            )
                            for _anchor_index, _lyric_index, anchor in mapped_anchors
                            if text_start <= _lyric_index < text_end
                            and chunk_start - overlap_seconds <= float(anchor.start) < chunk_end + overlap_seconds
                        ]
                        local_tokens = self.align(
                            chunk_path,
                            chunk_text,
                            language,
                            anchor_tokens=local_anchors,
                        )
                        if local_tokens and local_anchors and local_tokens[0].start <= 0.05:
                            local_first_anchor = min(
                                local_anchors,
                                key=lambda token: float(token.start),
                            )
                            first_word = normalize_lyric_token(chunk_text.split()[0])
                            if (
                                normalize_lyric_token(local_first_anchor.text) == first_word
                                and float(local_first_anchor.start) >= 0.25
                            ):
                                corrected_start = max(0.0, float(local_first_anchor.start) - 0.12)
                                leading_span = float(local_tokens[0].end - local_tokens[0].start)
                                if (
                                    corrected_start - float(local_tokens[0].start) > 1.5
                                    and leading_span <= 4.0
                                ):
                                    # The local trellis consumed a short
                                    # silent prefix before the first anchor;
                                    # shift the whole local island, not just
                                    # its first word, so it cannot remain a
                                    # fixed number of seconds early.
                                    shift = corrected_start - float(local_tokens[0].start)
                                    local_tokens = [
                                        replace(
                                            token,
                                            start=float(token.start) + shift,
                                            end=float(token.end) + shift,
                                        )
                                        for token in local_tokens
                                    ]
                                else:
                                    local_tokens[0] = replace(
                                        local_tokens[0],
                                        start=min(corrected_start, float(local_tokens[0].end) - 0.02),
                                    )
                        local_indices = _matched_lyric_indices(
                            words[text_start:text_end], local_tokens
                        )
                        # If the local trellis starts before a sparse ASR
                        # anchor, align the whole island to the shared lexical
                        # clock.  Comparing several matched pairs avoids
                        # shifting a window because of one bad ASR word.
                        anchor_local_indices = _matched_lyric_indices(
                            words[text_start:text_end], local_anchors
                        )
                        local_positions_by_lyric = {
                            lyric_index: local_index
                            for local_index, lyric_index in local_indices.items()
                        }
                        clock_deltas = [
                            float(local_anchors[anchor_index].start)
                            - float(local_tokens[local_index].start)
                            for anchor_index, lyric_index in anchor_local_indices.items()
                            if (local_index := local_positions_by_lyric.get(lyric_index)) is not None
                            and 0 <= local_index < len(local_tokens)
                        ]
                        if (
                            len(clock_deltas) >= 2
                            and local_tokens
                            and local_tokens[0].end - local_tokens[0].start <= 4.0
                        ):
                            median_delta = sorted(clock_deltas)[len(clock_deltas) // 2]
                            stable_deltas = [
                                delta
                                for delta in clock_deltas
                                if abs(delta - median_delta) <= 0.75
                            ]
                            if abs(median_delta) > 1.5 and len(stable_deltas) >= 2:
                                median_delta = sorted(stable_deltas)[len(stable_deltas) // 2]
                                local_tokens = [
                                    replace(
                                        token,
                                        start=float(token.start) + median_delta,
                                        end=float(token.end) + median_delta,
                                    )
                                    for token in local_tokens
                                ]
                        sequence_indices = [
                            text_start + local_indices[index]
                            if index in local_indices
                            else None
                            for index in range(len(local_tokens))
                        ]
                        aligned.extend(
                            replace_token_times(
                                local_tokens,
                                chunk_start,
                                sequence_indices=sequence_indices,
                                chunk_index=chunk_index,
                            )
                        )
                    except ModelAdapterError:
                        # A single instrumental or malformed window should not
                        # erase otherwise valid evidence from the song.
                        pass
                    except Exception as exc:
                        raise ModelAdapterError(
                            f"ForceAlign failed for chunk {chunk_index}"
                        ) from exc
                chunk_index += 1
                if chunk_end >= duration:
                    break
                chunk_start += step

        if not aligned:
            raise ModelAdapterError("ForceAlign returned no chunk timestamps")
        # Keep the highest-confidence copy in an overlap.  Sequence indices
        # are the canonical lyric positions, so this also removes duplicate
        # trellis outputs when a word straddles a window boundary.
        deduped_by_sequence: dict[int, ASRToken] = {}
        unindexed: list[ASRToken] = []
        for token in sorted(aligned, key=lambda item: (item.start, item.end)):
            if token.sequence_index is None:
                unindexed.append(token)
                continue
            previous = deduped_by_sequence.get(token.sequence_index)
            if previous is None or token.confidence > previous.confidence:
                deduped_by_sequence[token.sequence_index] = token
        # Keep only tokens whose local lexical match identified a canonical
        # position.  Unindexed words are often omissions/substitutions at a
        # window edge; feeding them back through a full-track lexical match
        # can attach them to an earlier repeated chorus and create a crossing
        # trusted span.  The independent ASR/CTC streams still cover those
        # words as ordinary fallback evidence.
        result = sorted(
            deduped_by_sequence.values(),
            key=lambda item: (item.start, item.end),
        )
        # A local trellis can omit a word near a repeated chorus, making the
        # lexical matcher attach the next window to an earlier occurrence.
        # Never expose those backwards sequence indices as trusted canonical
        # spans; retain the word as ordinary lexical evidence so the worker's
        # global matcher can place it without creating a crossing timeline.
        monotonic_result: list[ASRToken] = []
        last_sequence = -1
        for token in result:
            if token.sequence_index is not None:
                if token.sequence_index <= last_sequence:
                    # This is an overlap-window token attached to an earlier
                    # lexical occurrence.  It has no safe canonical index;
                    # omit it rather than reintroducing it as an unindexed
                    # forced candidate.
                    continue
                else:
                    last_sequence = token.sequence_index
            monotonic_result.append(token)
        result = monotonic_result
        if not result:
            raise ModelAdapterError("ForceAlign returned no usable chunk timestamps")
        return result


_qwen_asr: Qwen3ASRAdapter | None = None
_qwen_forced: Qwen3ForcedAlignerAdapter | None = None
_whisperx_ctc: WhisperXCTCAdapter | None = None
_forcealign: ForceAlignAdapter | None = None
_MODEL_INFERENCE_SEMAPHORE = threading.BoundedSemaphore(1)


def qwen_asr_enabled() -> bool:
    return os.environ.get("ALIGNMENT_ASR_BACKEND", "legacy").casefold() == "qwen"


def transcribe_with_qwen(audio_path: str, language: str) -> list[ASRToken]:
    global _qwen_asr
    if _qwen_asr is None:
        _qwen_asr = Qwen3ASRAdapter(
            model_name=os.environ.get("QWEN_ASR_MODEL", "Qwen/Qwen3-ASR-1.7B"),
            forced_aligner_name=os.environ.get("QWEN_FORCED_ALIGNER_MODEL", "Qwen/Qwen3-ForcedAligner-0.6B"),
            device=os.environ.get("QWEN_DEVICE", "cuda:0"),
        )
    return _qwen_asr.transcribe(audio_path, language)


def align_with_qwen_forced(
    audio_path: str,
    text: str,
    language: str,
    *,
    anchor_tokens: Sequence[ASRToken] = (),
) -> list[ASRToken]:
    global _qwen_forced
    if _qwen_forced is None:
        _qwen_forced = Qwen3ForcedAlignerAdapter(
            model_name=os.environ.get("QWEN_FORCED_ALIGNER_MODEL", "Qwen/Qwen3-ForcedAligner-0.6B"),
            device=os.environ.get("QWEN_DEVICE", "cuda:0"),
        )
    # Qwen ASR can expose the forced-aligner it loaded for timestamped
    # transcription. Reuse that instance instead of allocating a second GPU
    # copy; standalone forced alignment still lazily loads its own model.
    if _qwen_forced._model is None and _qwen_asr is not None and _qwen_asr._model is not None:
        shared_forced = getattr(_qwen_asr._model, "forced_aligner", None)
        if shared_forced is not None:
            _qwen_forced._model = shared_forced
    return _qwen_forced.align_chunked(audio_path, text, language, anchor_tokens=anchor_tokens)


def replace_token_times(
    tokens: list[ASRToken],
    offset: float,
    *,
    sequence_start: int | None = None,
    sequence_indices: Sequence[int | None] | None = None,
    chunk_index: int | None = None,
) -> list[ASRToken]:
    return [
        ASRToken(
            text=token.text,
            start=token.start + offset,
            end=token.end + offset,
            confidence=token.confidence,
            model=token.model,
            audio_source=token.audio_source,
            evidence_type=token.evidence_type,
            sequence_index=(
                sequence_indices[index]
                if sequence_indices is not None and index < len(sequence_indices)
                else (sequence_start + index) if sequence_start is not None else token.sequence_index
            ),
            chunk_index=chunk_index if chunk_index is not None else token.chunk_index,
            midi_note=token.midi_note,
        )
        for index, token in enumerate(tokens)
    ]


def _lyric_tokens_for_words(words: Sequence[str]) -> list[LyricToken]:
    """Build lightweight canonical tokens for chunk-to-anchor matching."""

    return [
        LyricToken(
            id=f"chunk-token-{index}",
            lyric_revision_id="chunk",
            display_text=word,
            normalized_text=normalize_lyric_token(word),
            line_index=0,
            token_index=index,
            source_index=index,
        )
        for index, word in enumerate(words)
    ]


def _matched_lyric_indices(
    words: Sequence[str],
    anchor_tokens: Sequence[ASRToken],
) -> dict[int, int]:
    """Map ASR positions to lyric positions without relying on token counts.

    ASR inserts, deletions, punctuation, and repeated chorus words make a
    positional ratio unsafe. A monotonic lexical match keeps overlapping
    model windows attached to the same source words.
    """

    if not words or not anchor_tokens:
        return {}
    match = global_monotonic_match(_lyric_tokens_for_words(words), anchor_tokens)
    prepared_to_original = match.prepared_to_original_indices or tuple(range(len(anchor_tokens)))
    return {
        prepared_to_original[item.asr_index]: item.lyric_index
        for item in match.matches
        if item.matched and item.asr_index is not None
        and item.asr_index < len(prepared_to_original)
    }


def _anchor_window(
    anchor_tokens: Sequence[ASRToken],
    duration: float,
    *,
    padding_seconds: float = 0.75,
) -> tuple[float, float]:
    """Bound full-track CTC alignment to the observed vocal region."""

    valid_anchors = [
        token
        for token in anchor_tokens
        if _valid_times(token.start, token.end) is not None
    ]
    if not valid_anchors or duration <= 0.0:
        return 0.0, duration
    first_start = min(token.start for token in valid_anchors)
    last_end = max(token.end for token in valid_anchors)
    start = max(0.0, first_start - padding_seconds)
    end = min(duration, max(start, last_end + padding_seconds))
    return start, end


def whisperx_enabled() -> bool:
    configured = os.environ.get("ALIGNMENT_CTC_BACKEND", "whisperx").strip().casefold()
    return configured not in {"off", "disabled", "none"}


def forcealign_enabled() -> bool:
    """Return whether optional local Wav2Vec2 evidence is requested."""

    configured = os.environ.get("ALIGNMENT_WAV2VEC_BACKEND", "off").strip().casefold()
    return configured in {"forcealign", "wav2vec2", "wav2vec2-forcealign", "on", "enabled"}


def sofa_aligner_enabled() -> bool:
    """Return whether the singing-trained SOFA aligner is the primary lane.

    SOFA is deliberately opt-out: when its local checkpoint and source tree
    are installed it becomes the production timing source.  ``off`` keeps a
    RAM-constrained or legacy deployment on the previous ASR/CTC path, while
    ``on``/``always`` gives a useful configuration error if the assets are
    missing instead of silently selecting another model.
    """

    configured = os.environ.get("ALIGNMENT_SOFA_BACKEND", "auto").strip().casefold()
    if configured in {"0", "false", "off", "no", "disabled", "none"}:
        return False
    if configured in {"1", "true", "on", "yes", "enabled", "always", "sofa"}:
        return True
    code = Path(os.environ.get("SOFA_CODE_DIR", Path(__file__).resolve().parents[1] / "optional_aligners" / "sofa-code"))
    checkpoint = Path(os.environ.get("SOFA_CHECKPOINT", Path(__file__).resolve().parents[1] / "optional_aligners" / "sofa" / "ckpt" / "combined.ckpt"))
    return code.is_dir() and checkpoint.is_file()


def _sofa_timed_tokens(
    tokens: Sequence[LyricToken],
    anchors: Sequence[ASRToken],
    duration: float,
) -> list[Any]:
    """Give SOFA a stable clock even when the ASR anchor is sparse.

    SOFA aligns short windows more reliably than a full song, so each window
    needs approximate source spans.  Keep trusted lexical anchor spans where
    available and interpolate only the gaps; this is an input clock, never
    the final timing exported by the worker.
    """

    from alternative_pipelines import TimedToken

    safe_duration = max(0.1, float(duration))
    known: dict[int, tuple[float, float]] = {}
    if anchors:
        match = global_monotonic_match(tokens, anchors)
        prepared = match.prepared_asr_tokens or tuple(anchors)
        for item in match.matches:
            if not item.matched or item.asr_index is None or item.asr_index >= len(prepared):
                continue
            anchor = prepared[item.asr_index]
            try:
                start, end = float(anchor.start), float(anchor.end)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(start) or not math.isfinite(end) or end <= start:
                continue
            known[item.lyric_index] = (
                max(0.0, min(safe_duration, start)),
                max(0.02, min(safe_duration, end)),
            )

    spans: list[tuple[float, float]] = [(0.0, safe_duration) for _ in tokens]
    if not known:
        for index in range(len(tokens)):
            spans[index] = (
                safe_duration * index / max(1, len(tokens)),
                safe_duration * (index + 1) / max(1, len(tokens)),
            )
    else:
        known_indices = sorted(known)
        for index in known_indices:
            spans[index] = known[index]
        # Before the first and after the last anchor, distribute the missing
        # words over the available interval instead of collapsing them onto a
        # single boundary.
        first = known_indices[0]
        if first:
            boundary = max(0.0, known[first][0])
            for index in range(first):
                spans[index] = (
                    boundary * index / first,
                    boundary * (index + 1) / first,
                )
        for left, right in zip(known_indices, known_indices[1:]):
            if right <= left + 1:
                continue
            start = known[left][1]
            end = known[right][0]
            if end <= start:
                end = start + 0.02 * (right - left)
            for offset, index in enumerate(range(left + 1, right), start=1):
                spans[index] = (
                    start + (end - start) * (offset - 1) / (right - left),
                    start + (end - start) * offset / (right - left),
                )
        last = known_indices[-1]
        if last < len(tokens) - 1:
            start = known[last][1]
            end = max(start + 0.02, safe_duration)
            remaining = len(tokens) - 1 - last
            for offset, index in enumerate(range(last + 1, len(tokens)), start=1):
                spans[index] = (
                    start + (end - start) * (offset - 1) / remaining,
                    start + (end - start) * offset / remaining,
                )

    result: list[Any] = []
    previous_end = 0.0
    for token, (start, end) in zip(tokens, spans):
        start = max(previous_end, min(safe_duration, float(start)))
        end = max(start + 0.02, min(safe_duration, float(end)))
        if end <= start:
            end = min(safe_duration, start + 0.02)
        result.append(TimedToken(token.display_text, start, end))
        previous_end = end
    return result


def align_with_sofa(
    audio_path: str,
    tokens: Sequence[LyricToken],
    language: str,
    *,
    anchor_tokens: Sequence[ASRToken] = (),
) -> list[ASRToken]:
    """Run SOFA and return canonical-indexed forced word evidence."""

    if not tokens:
        return []
    if language.casefold().startswith("pl") and os.environ.get("SOFA_ALLOW_NON_ENGLISH", "0").strip().casefold() not in {"1", "true", "yes", "on"}:
        raise ModelAdapterError("SOFA combined checkpoint currently supports English lyrics only")
    try:
        import soundfile as sf

        duration = float(sf.info(audio_path).duration)
    except Exception as exc:
        raise ModelAdapterError(f"Unable to read audio duration for SOFA: {exc}") from exc
    from alternative_pipelines import align_sofa

    timed_tokens = _sofa_timed_tokens(tokens, anchor_tokens, duration)
    with tempfile.TemporaryDirectory(prefix="sofa-production-") as output_dir:
        aligned_words = align_sofa(timed_tokens, Path(audio_path), Path(output_dir))
    if not aligned_words:
        raise ModelAdapterError("SOFA returned no word intervals")

    sofa_tokens = [
        ASRToken(
            text=word.text,
            start=float(word.start),
            end=float(word.end),
            confidence=max(0.5, min(1.0, float(word.confidence))),
            model="sofa-combined",
            audio_source="vocal_stem",
            evidence_type="forced",
        )
        for word in aligned_words
        if math.isfinite(float(word.start))
        and math.isfinite(float(word.end))
        and float(word.end) > float(word.start)
    ]
    if not sofa_tokens:
        raise ModelAdapterError("SOFA returned no usable word intervals")
    lexical = global_monotonic_match(tokens, sofa_tokens)
    prepared = lexical.prepared_asr_tokens or tuple(sofa_tokens)
    result: list[ASRToken] = []
    for item in lexical.matches:
        if not item.matched or item.asr_index is None or item.asr_index >= len(prepared):
            continue
        result.append(
            replace(
                prepared[item.asr_index],
                sequence_index=item.lyric_index,
                evidence_type="forced",
                model="sofa-combined",
                audio_source="vocal_stem",
            )
        )
    if not result:
        raise ModelAdapterError("SOFA output could not be matched to canonical lyrics")
    return result


def basic_pitch_enabled() -> bool:
    """Return whether optional Basic Pitch boundary evidence is requested."""

    configured = os.environ.get("ALIGNMENT_MELODY_BACKEND", "off").strip().casefold()
    return configured in {"basic_pitch", "basic-pitch", "on", "enabled"}


def melody_blend_enabled() -> bool:
    """Return whether the supplemental vocal/mix melody lane is enabled."""

    configured = os.environ.get("ALIGNMENT_MELODY_BLEND_ENABLED", "1").strip().casefold()
    return configured not in {"0", "false", "off", "no", "disabled"}


def _melody_blend_alpha() -> float:
    """Return the accompaniment ratio used for the supplemental lane."""

    try:
        alpha = float(os.environ.get("ALIGNMENT_MELODY_BLEND_MIX", "0.6"))
    except ValueError:
        alpha = 0.6
    if not math.isfinite(alpha):
        alpha = 0.6
    return max(0.0, min(1.0, alpha))


def write_melody_blend_audio(
    original_audio_path: str,
    vocal_audio_path: str,
    output_path: str,
    *,
    accompaniment_ratio: float | None = None,
) -> None:
    """Write a normalized vocal/accompaniment blend for melody detection.

    The blend is an evidence-only intermediate. It is never exposed as a
    user-facing audio asset or used as a replacement for either input.
    """

    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        raise ModelAdapterError("soundfile and numpy are required for melody blending") from exc

    try:
        original, original_rate = sf.read(original_audio_path, always_2d=False)
        vocals, vocal_rate = sf.read(vocal_audio_path, always_2d=False)
    except Exception as exc:
        raise ModelAdapterError(f"Unable to read audio for melody blending: {exc}") from exc

    def channels(data: Any) -> Any:
        array = np.asarray(data, dtype=np.float32)
        if array.ndim == 1:
            return array[:, None]
        if array.ndim != 2 or array.shape[1] <= 0:
            raise ModelAdapterError("Melody blend audio must be mono or stereo")
        return array

    original_array = channels(original)
    vocal_array = channels(vocals)
    if original_array.shape[1] != vocal_array.shape[1]:
        if original_array.shape[1] == 1:
            original_array = np.broadcast_to(
                original_array,
                (original_array.shape[0], vocal_array.shape[1]),
            )
        elif vocal_array.shape[1] == 1:
            vocal_array = np.broadcast_to(
                vocal_array,
                (vocal_array.shape[0], original_array.shape[1]),
            )
        else:
            original_array = original_array.mean(axis=1, keepdims=True)
            vocal_array = vocal_array.mean(axis=1, keepdims=True)
    if original_rate != vocal_rate:
        try:
            from scipy.signal import resample_poly

            greatest_common_divisor = math.gcd(int(original_rate), int(vocal_rate))
            original_array = resample_poly(
                original_array,
                int(vocal_rate) // greatest_common_divisor,
                int(original_rate) // greatest_common_divisor,
                axis=0,
            ).astype(np.float32)
        except Exception as exc:
            raise ModelAdapterError("Unable to resample audio for melody blending") from exc

    sample_count = min(len(original_array), len(vocal_array))
    if sample_count <= 0:
        raise ModelAdapterError("Melody blend audio is empty")
    ratio = _melody_blend_alpha() if accompaniment_ratio is None else float(accompaniment_ratio)
    if not math.isfinite(ratio):
        raise ModelAdapterError("Melody blend ratio must be finite")
    ratio = max(0.0, min(1.0, ratio))
    blended = vocal_array[:sample_count] + ratio * (
        original_array[:sample_count] - vocal_array[:sample_count]
    )
    peak = float(np.max(np.abs(blended))) if blended.size else 0.0
    if not math.isfinite(peak):
        raise ModelAdapterError("Melody blend audio contains non-finite samples")
    blended = blended * (0.98 / max(1.0, peak))
    output_audio = blended[:, 0] if blended.shape[1] == 1 else blended
    try:
        sf.write(output_path, output_audio, int(vocal_rate))
    except Exception as exc:
        raise ModelAdapterError(f"Unable to write melody blend audio: {exc}") from exc


def detect_basic_pitch_boundaries(audio_path: str) -> list[ASRToken]:
    """Detect melodic note spans as boundary candidates, never as lyric words.

    Basic Pitch is intentionally an optional adapter. Its note events are
    useful for held vowels and sung word ends, but they carry no lexical
    identity; the deterministic worker pairs them with lexical anchors later.
    The original mix is used because accompaniment often preserves the melody
    onset more clearly than an aggressively separated vocal stem.
    """

    try:
        import scipy.signal
        from scipy.signal.windows import gaussian

        # Basic Pitch versions before the current scipy API still import the
        # old location. Keep this local so the optional dependency does not
        # change the rest of the application's scipy behavior.
        if not hasattr(scipy.signal, "gaussian"):
            scipy.signal.gaussian = gaussian
        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch.inference import predict
    except Exception as exc:
        raise ModelAdapterError("basic-pitch is required for melody boundary candidates") from exc

    model_path = Path(str(ICASSP_2022_MODEL_PATH))
    if model_path.suffix == "":
        model_path = model_path.with_suffix(".onnx")
    try:
        _model_output, _midi, note_events = predict(
            audio_path,
            str(model_path),
            onset_threshold=float(os.environ.get("BASIC_PITCH_ONSET_THRESHOLD", "0.4")),
            frame_threshold=float(os.environ.get("BASIC_PITCH_FRAME_THRESHOLD", "0.3")),
            minimum_note_length=int(os.environ.get("BASIC_PITCH_MIN_NOTE_MS", "50")),
            minimum_frequency=float(os.environ.get("BASIC_PITCH_MIN_FREQUENCY", "60")),
            maximum_frequency=float(os.environ.get("BASIC_PITCH_MAX_FREQUENCY", "1200")),
        )
    except Exception as exc:
        raise ModelAdapterError(f"Basic Pitch failed for {audio_path}") from exc

    boundaries: list[ASRToken] = []
    for event in note_events or []:
        try:
            start, end, midi_note, confidence = event[:4]
            times = _valid_times(start, end)
            confidence_value = _confidence(confidence)
        except (TypeError, ValueError):
            continue
        if times is None or times[1] <= times[0]:
            continue
        boundaries.append(
            ASRToken(
                text="",
                start=times[0],
                end=times[1],
                confidence=confidence_value,
                model="basic-pitch",
                audio_source="original_mix",
                evidence_type="melody",
                midi_note=int(midi_note),
            )
        )
    if not boundaries:
        raise ModelAdapterError("Basic Pitch returned no note boundaries")
    return boundaries


def align_with_whisperx(
    audio_path: str,
    text: str,
    language: str,
    *,
    anchor_tokens: Sequence[ASRToken] = (),
) -> list[ASRToken]:
    global _whisperx_ctc
    if _whisperx_ctc is None:
        _whisperx_ctc = WhisperXCTCAdapter(
            device=os.environ.get("WHISPERX_DEVICE") or None,
            model_name=os.environ.get("WHISPERX_MODEL") or None,
        )
    return _whisperx_ctc.align_chunked(audio_path, text, language, anchor_tokens=anchor_tokens)


def align_with_forcealign(
    audio_path: str,
    text: str,
    language: str,
    *,
    anchor_tokens: Sequence[ASRToken] = (),
    activity_spans: Sequence[tuple[float, float]] = (),
) -> list[ASRToken]:
    global _forcealign
    if _forcealign is None:
        _forcealign = ForceAlignAdapter()
    # Long recordings are susceptible to trellis clock jumps after a missed
    # syllable.  Use bounded windows only when an independent ASR transcript
    # is dense enough to choose lexical ranges; short/sparse jobs keep the
    # proven full-track path.  ``on`` is useful for diagnostics, while
    # ``auto`` is the production-safe default.
    mode = os.environ.get("ALIGNMENT_FORCEALIGN_CHUNKED", "off").strip().casefold()
    use_chunked = mode in {"1", "true", "yes", "on", "enabled"}
    if mode == "auto" and anchor_tokens:
        try:
            import soundfile as sf

            duration = float(sf.info(audio_path).duration)
        except Exception:
            duration = 0.0
        lyric_words = text.split()
        mapped_count = len(_matched_lyric_indices(lyric_words, anchor_tokens))
        use_chunked = (
            duration >= float(os.environ.get("ALIGNMENT_FORCEALIGN_CHUNK_MIN_DURATION_S", "240"))
            and mapped_count >= max(20, int(len(lyric_words) * 0.35))
        )
    if use_chunked:
        try:
            chunk_seconds = float(os.environ.get("ALIGNMENT_FORCEALIGN_CHUNK_SECONDS", "60"))
            overlap_seconds = float(os.environ.get("ALIGNMENT_FORCEALIGN_CHUNK_OVERLAP_S", "8"))
        except ValueError:
            chunk_seconds, overlap_seconds = 60.0, 8.0
        return _forcealign.align_chunked(
            audio_path,
            text,
            language,
            anchor_tokens=anchor_tokens,
            chunk_seconds=chunk_seconds,
            overlap_seconds=overlap_seconds,
        )
    return _forcealign.align(
        audio_path,
        text,
        language,
        anchor_tokens=anchor_tokens,
        activity_spans=activity_spans,
    )


def run_serialized_model_call(function: Any, *args: Any, **kwargs: Any) -> Any:
    """Serialize GPU model calls across jobs in one API/worker process."""

    with _MODEL_INFERENCE_SEMAPHORE:
        return function(*args, **kwargs)


def _chunk_text_range(
    words: Sequence[str],
    chunk_start: float,
    chunk_end: float,
    duration: float,
    overlap_seconds: float,
    anchor_tokens: Sequence[ASRToken],
    chunk_index: int,
) -> tuple[int, int]:
    """Choose text from anchors, avoiding forced alignment over instrumental gaps."""

    if anchor_tokens:
        anchor_to_lyric = _matched_lyric_indices(words, anchor_tokens)
        matching = [
            lyric_index
            for anchor_index, token in enumerate(anchor_tokens)
            if chunk_start - overlap_seconds <= token.start < chunk_end + overlap_seconds
            and (lyric_index := anchor_to_lyric.get(anchor_index)) is not None
        ]
        if matching:
            start = max(0, min(matching) - 1)
            end = min(len(words), max(matching) + 2)
            return start, max(start, end)
        # A chunk with no global anchor is likely instrumental. Keep it empty
        # instead of forcing the remaining lyric text into silence.
        if chunk_index > 0:
            return 0, 0

    start = max(0, int(math.floor(len(words) * chunk_start / duration)) - (1 if chunk_index else 0))
    end = min(len(words), int(math.ceil(len(words) * chunk_end / duration)) + 1)
    return start, max(start, end)
