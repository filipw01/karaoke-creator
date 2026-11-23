"""Canonical lyric tokenization, matching, fallback, and legacy projection.

This module deliberately contains no model loading. Model adapters produce
``ASRToken`` or ``AlignmentCandidate`` values and this deterministic layer
decides what can honestly become a canonical word alignment.
"""

from __future__ import annotations

import difflib
import math
import os
import re
from bisect import bisect_left
from statistics import median
import unicodedata
from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Sequence

from models import (
    AlignmentEvidence,
    AlignmentStatus,
    LyricToken,
    Word,
    WordAlignment,
)


_NON_WHITESPACE = re.compile(r"\S+")
_MIN_FUZZY_SIMILARITY = 0.72
_BOUNDARY_EPSILON = 0.02
_UNSET = object()
_ANCHOR_CANDIDATE_TYPES = frozenset({"anchor", "asr", "transcript"})


def _enum_value(value: object) -> str:
    return str(getattr(value, "value", value))


def normalize_language(language: str | None) -> str:
    """Return the supported two-letter language code.

    The user-facing product currently supports exactly English and Polish.
    Common Polish labels from existing metadata are accepted for migration.
    """

    if language is None:
        code = "en"
    elif not isinstance(language, str):
        raise ValueError("Word alignment language must be a string")
    else:
        code = language.strip().casefold() or "en"
    if code in {"polish", "polski", "po", "pl-pl", "pl_pl"}:
        return "pl"
    if code in {"english", "en-us", "en_us", "en-gb", "en_gb"}:
        return "en"
    if code.startswith("pl"):
        return "pl"
    if code.startswith("en"):
        return "en"
    raise ValueError("Word alignment supports English (en) or Polish (pl)")


def normalize_lyric_token(text: str, language: str | None = None) -> str:
    """Create a conservative match key without transliterating source text."""

    # Validate the selected language even though the Unicode operation is
    # shared. This prevents accidental mixed-language jobs at the API seam.
    normalize_language(language)
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(
        character
        for character in normalized
        if character.isalnum() or unicodedata.category(character).startswith("M")
    )


def tokenize_lyrics(
    raw_text: str,
    lyric_revision_id: str,
    language: str | None = None,
) -> list[LyricToken]:
    """Tokenize source lyrics while preserving exact text and original lines."""

    normalized_language = normalize_language(language)
    tokens: list[LyricToken] = []
    source_index = 0
    for line_index, line in enumerate(raw_text.splitlines()):
        for token_index, match in enumerate(_NON_WHITESPACE.finditer(line)):
            display_text = match.group(0)
            normalized_text = normalize_lyric_token(display_text, normalized_language)
            # Punctuation is source formatting, not an alignable word. Keep
            # source_index gaps so IDs remain tied to the original text
            # positions while preventing zero-time punctuation from entering
            # the canonical alignment sequence.
            if not normalized_text:
                source_index += 1
                continue
            tokens.append(
                LyricToken(
                    id=f"{lyric_revision_id}:token:{source_index}",
                    lyric_revision_id=lyric_revision_id,
                    display_text=display_text,
                    normalized_text=normalized_text,
                    line_index=line_index,
                    token_index=token_index,
                    source_index=source_index,
                )
            )
            source_index += 1
    return tokens


@dataclass(frozen=True)
class ASRToken:
    text: str
    start: float
    end: float
    confidence: float = 1.0
    model: str = "qwen3-asr-1.7b"
    audio_source: str = "vocal_stem"
    evidence_type: str = "anchor"
    sequence_index: int | None = None
    chunk_index: int | None = None
    # Optional note metadata is retained for non-lexical melody evidence.
    # ASR/CTC adapters leave it unset.
    midi_note: int | None = None

    @property
    def normalized_text(self) -> str:
        return normalize_lyric_token(self.text)


def asr_tokens_from_words(
    words: Iterable[Word],
    model: str = "qwen3-asr-1.7b",
    audio_source: str = "vocal_stem",
    evidence_type: str = "anchor",
) -> list[ASRToken]:
    """Adapt the legacy model output without making it the source of truth."""

    return [
        ASRToken(
            text=word.text,
            start=float(word.start),
            end=float(word.end),
            confidence=float(word.confidence),
            model=model,
            audio_source=audio_source,
            evidence_type=evidence_type,
        )
        for word in words
        if _valid_span(float(word.start), float(word.end))
        and normalize_lyric_token(word.text)
    ]


def _merge_hyphenated_asr_tokens(
    lyric_tokens: Sequence[LyricToken],
    asr_tokens: Sequence[ASRToken],
) -> tuple[list[ASRToken], list[int]]:
    """Join ASR pieces that form one hyphenated source word.

    ASR engines commonly emit ``well`` and ``being`` for a source token such
    as ``well-being``. Treating those as two independent substitutions makes
    the global matcher delete the source word and shifts every following
    boundary. The merged token keeps the first start and last end, so all
    downstream timing consumers still receive one honest span.

    This is deliberately limited to source words containing an internal
    hyphen and to two/three adjacent lexical pieces. It cannot turn a normal
    phrase into a compound merely because the concatenated spelling happens
    to be similar.
    """

    if not lyric_tokens or not asr_tokens:
        return list(asr_tokens), list(range(len(asr_tokens)))
    hyphenated_targets = {
        token.normalized_text
        for token in lyric_tokens
        if "-" in token.display_text and token.normalized_text
    }
    if not hyphenated_targets:
        return list(asr_tokens), list(range(len(asr_tokens)))

    merged: list[ASRToken] = []
    prepared_to_original: list[int] = []
    index = 0
    while index < len(asr_tokens):
        candidate: tuple[int, str] | None = None
        for part_count in (3, 2):
            if index + part_count > len(asr_tokens):
                continue
            parts = asr_tokens[index : index + part_count]
            if any(
                not _valid_span(float(part.start), float(part.end))
                or not part.normalized_text
                for part in parts
            ):
                continue
            # A long pause is a word boundary even when the transcript has
            # omitted the hyphen. Keep the allowance below a normal breath.
            if any(
                float(right.start) - float(left.end) > 0.35
                for left, right in zip(parts, parts[1:])
            ):
                continue
            combined = "".join(part.normalized_text for part in parts)
            if combined in hyphenated_targets:
                candidate = (part_count, combined)
                break
        if candidate is None:
            merged.append(asr_tokens[index])
            prepared_to_original.append(index)
            index += 1
            continue

        part_count, _combined = candidate
        parts = asr_tokens[index : index + part_count]
        first, last = parts[0], parts[-1]
        merged.append(
            ASRToken(
                text=" ".join(part.text for part in parts),
                start=float(first.start),
                end=float(last.end),
                confidence=min(float(part.confidence) for part in parts),
                model=first.model,
                audio_source=first.audio_source,
                evidence_type=first.evidence_type,
                sequence_index=first.sequence_index,
                chunk_index=first.chunk_index,
                midi_note=first.midi_note,
            )
        )
        prepared_to_original.append(index)
        index += part_count
    return merged, prepared_to_original


def activity_spans_from_signal(
    activity_times: Sequence[float],
    activity: Sequence[float | bool],
    *,
    threshold: float = 0.005,
    clear_gap_s: float = 0.18,
) -> list[tuple[float, float]]:
    """Convert frame activity into bounded spans for honest fallback timing."""

    all_times = [float(time) for time in activity_times if math.isfinite(float(time))]
    finite_times = [
        float(time)
        for time, value in zip(activity_times, activity)
        if math.isfinite(float(time)) and float(value) >= threshold
    ]
    if not finite_times:
        return []
    positive_steps = [
        right - left
        for left, right in zip(all_times, all_times[1:])
        if right > left
    ]
    frame_step = min(max(median(positive_steps), 0.01), 0.2) if positive_steps else 0.05
    spans: list[tuple[float, float]] = []
    start = previous = finite_times[0]
    for current in finite_times[1:]:
        if current - previous > clear_gap_s:
            spans.append((start, previous + frame_step))
            start = current
        previous = current
    spans.append((start, previous + frame_step))
    return [(start, end) for start, end in spans if end > start]


def token_similarity(left: str, right: str) -> float:
    """Return deterministic lexical similarity in the [0, 1] range."""

    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    return difflib.SequenceMatcher(None, left, right, autojunk=False).ratio()


@dataclass(frozen=True)
class TokenMatch:
    lyric_index: int
    asr_index: int | None
    similarity: float
    exact: bool
    reason: str

    @property
    def matched(self) -> bool:
        return self.asr_index is not None and self.similarity >= _MIN_FUZZY_SIMILARITY


@dataclass
class GlobalAlignment:
    matches: list[TokenMatch]
    inserted_asr_indices: list[int]
    score: float
    # Prepared tokens are exposed for consumers that need the complete timing
    # span of a merged hyphenated ASR word. Ordinary callers can continue to
    # use ``matches`` exactly as before.
    prepared_asr_tokens: tuple[ASRToken, ...] = ()
    prepared_to_original_indices: tuple[int, ...] = ()


def global_monotonic_match(
    lyric_tokens: Sequence[LyricToken],
    asr_tokens: Sequence[ASRToken],
    *,
    insertion_penalty: float = -1.0,
    deletion_penalty: float = -1.0,
) -> GlobalAlignment:
    """Globally align canonical and ASR tokens in monotonic source order.

    Weak substitutions are deliberately scored below a deletion plus
    insertion. They remain unresolved instead of assigning a nearby ASR word
    to an unrelated source token.
    """

    # Normalize structural source conventions before dynamic programming. The
    # returned index still addresses the prepared monotonic ASR sequence and
    # therefore points at the complete (first-start/last-end) span.
    prepared_asr_tokens, prepared_to_original_indices = _merge_hyphenated_asr_tokens(
        lyric_tokens, asr_tokens
    )
    n, m = len(lyric_tokens), len(prepared_asr_tokens)
    scores = [[0.0] * (m + 1) for _ in range(n + 1)]
    operations: list[list[str]] = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        scores[i][0] = scores[i - 1][0] + deletion_penalty
        operations[i][0] = "delete"
    for j in range(1, m + 1):
        scores[0][j] = scores[0][j - 1] + insertion_penalty
        operations[0][j] = "insert"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            similarity = token_similarity(
                lyric_tokens[i - 1].normalized_text,
                prepared_asr_tokens[j - 1].normalized_text,
            )
            if similarity == 1.0:
                substitution = 3.0
            elif similarity >= _MIN_FUZZY_SIMILARITY:
                substitution = 1.25 + similarity
            else:
                substitution = -3.0

            choices = [
                (scores[i - 1][j - 1] + substitution, "match"),
                (scores[i - 1][j] + deletion_penalty, "delete"),
                (scores[i][j - 1] + insertion_penalty, "insert"),
            ]
            # Prefer a lexical match, then deletion, then insertion on ties.
            priority = {"match": 2, "delete": 1, "insert": 0}
            scores[i][j], operations[i][j] = max(
                choices,
                key=lambda choice: (choice[0], priority[choice[1]]),
            )

    matches: list[TokenMatch] = []
    inserted: list[int] = []
    i, j = n, m
    while i or j:
        operation = operations[i][j]
        if operation == "match":
            lyric = lyric_tokens[i - 1]
            asr = prepared_asr_tokens[j - 1]
            similarity = token_similarity(lyric.normalized_text, asr.normalized_text)
            matches.append(
                TokenMatch(
                    lyric_index=i - 1,
                    asr_index=j - 1 if similarity >= _MIN_FUZZY_SIMILARITY else None,
                    similarity=similarity,
                    exact=similarity == 1.0,
                    reason="exact lexical match"
                    if similarity == 1.0
                    else "fuzzy lexical match"
                    if similarity >= _MIN_FUZZY_SIMILARITY
                    else "weak substitution left unresolved",
                )
            )
            i -= 1
            j -= 1
        elif operation == "delete":
            matches.append(
                TokenMatch(
                    lyric_index=i - 1,
                    asr_index=None,
                    similarity=0.0,
                    exact=False,
                    reason="canonical word absent from ASR transcript",
                )
            )
            i -= 1
        elif operation == "insert":
            inserted.append(prepared_to_original_indices[j - 1])
            j -= 1
        else:
            # The zero row/column has no operation in the origin cell.
            if i:
                matches.append(
                    TokenMatch(i - 1, None, 0.0, False, "canonical word unresolved")
                )
                i -= 1
            elif j:
                inserted.append(prepared_to_original_indices[j - 1])
                j -= 1

    matches.reverse()
    inserted.reverse()
    return GlobalAlignment(
        matches=matches,
        inserted_asr_indices=inserted,
        score=scores[n][m],
        prepared_asr_tokens=tuple(prepared_asr_tokens),
        prepared_to_original_indices=tuple(prepared_to_original_indices),
    )


def match_lyric_tokens(
    lyric_tokens: Sequence[LyricToken],
    asr_tokens: Sequence[ASRToken],
) -> dict[int, int]:
    """Convenience mapping for high-agreement canonical/ASR pairs only."""

    result = global_monotonic_match(lyric_tokens, asr_tokens)
    prepared_to_original = result.prepared_to_original_indices or tuple(
        range(len(asr_tokens))
    )
    return {
        match.lyric_index: prepared_to_original[match.asr_index]
        for match in result.matches
        if match.matched
        and match.asr_index is not None
        and match.asr_index < len(prepared_to_original)
    }


@dataclass(frozen=True)
class AlignmentCandidate:
    token_id: str
    start: float
    end: float
    model: str
    audio_source: str
    text_score: float = 1.0
    model_confidence: float = 1.0
    anchor_score: float | None = None
    source_agreement: float | None = None
    quality: float = 1.0
    reasons: tuple[str, ...] = ()
    candidate_type: str = "forced"
    activity_score: float | None = None
    temporal_distance: float | None = None
    temporal_start_offset: float | None = None
    temporal_end_offset: float | None = None
    acoustic_continuity: float | None = None
    # Confidence of a vocal melody event that continues through this
    # candidate's end.  This is intentionally separate from the candidate's
    # own confidence: a short span can look locally plausible while ending in
    # the middle of a clearly sustained vocal note.
    melody_end_inside_confidence: float | None = None


@dataclass(frozen=True)
class ScoredCandidate:
    candidate: AlignmentCandidate
    score: float
    reasons: tuple[str, ...]


def score_alignment_candidate(
    candidate: AlignmentCandidate,
    *,
    previous_end: float | None = None,
) -> ScoredCandidate:
    """Score a candidate with persisted, human-readable reasons."""

    reasons = list(candidate.reasons)
    if not _valid_span(candidate.start, candidate.end):
        reasons.append("candidate has invalid finite non-negative boundaries")
        return ScoredCandidate(candidate=candidate, score=0.0, reasons=tuple(dict.fromkeys(reasons)))
    if candidate.candidate_type.casefold() in _ANCHOR_CANDIDATE_TYPES:
        reasons.append("ASR timestamp is an anchor only; forced timing is required")
        return ScoredCandidate(candidate=candidate, score=0.0, reasons=tuple(dict.fromkeys(reasons)))

    text_score = _clamp_confidence(candidate.text_score)
    model_confidence = _clamp_confidence(candidate.model_confidence)
    # ``None`` means no temporal anchor was available, not evidence of a
    # contradiction. Explicit zero is reserved for a measured incompatible
    # span and is rejected below.
    anchor_score = _clamp_confidence(candidate.anchor_score) if candidate.anchor_score is not None else 0.5
    source_agreement = _clamp_confidence(candidate.source_agreement) if candidate.source_agreement is not None else 0.0
    quality = _clamp_confidence(candidate.quality)
    activity_score = (
        _clamp_confidence(candidate.activity_score)
        if candidate.activity_score is not None
        else 0.5
    )
    score = (
        0.30 * text_score
        + 0.20 * model_confidence
        + 0.15 * anchor_score
        + 0.15 * source_agreement
        + 0.10 * quality
        + 0.10 * activity_score
    )
    candidate_type = candidate.candidate_type.casefold()
    if candidate_type in {"chart", "ultrastar", "timing_reference"}:
        if text_score < 0.99:
            reasons.append("timing-reference text does not exactly match the lyric token")
            return ScoredCandidate(candidate=candidate, score=0.0, reasons=tuple(dict.fromkeys(reasons)))
        reasons.extend((
            "explicit UltraStar timing reference",
            "chart timing is authoritative for this run",
        ))
        if previous_end is not None and candidate.start < previous_end - _BOUNDARY_EPSILON:
            reasons.append("non-monotonic timing-reference candidate rejected")
            return ScoredCandidate(candidate=candidate, score=0.0, reasons=tuple(dict.fromkeys(reasons)))
        return ScoredCandidate(candidate=candidate, score=1.0, reasons=tuple(dict.fromkeys(reasons)))
    if candidate_type in {"melody", "pitch", "basic_pitch"}:
        if text_score < 0.99 or model_confidence < 0.1:
            reasons.append("melody candidate lacks a valid lexical/confidence floor")
            return ScoredCandidate(candidate=candidate, score=0.0, reasons=tuple(dict.fromkeys(reasons)))
        if candidate.end - candidate.start < 0.12:
            reasons.append("melody candidate is an implausibly short boundary pair")
            return ScoredCandidate(candidate=candidate, score=0.0, reasons=tuple(dict.fromkeys(reasons)))
        score += 0.20
        reasons.append("melodic boundary evidence")
        if anchor_score <= 0.0:
            # A lexical anchor can point at the preceding repetition. A
            # melody boundary that lands on a fresh vocal onset is still
            # credible in that case; activity is the independent evidence
            # that lets the candidate escape the stale ASR clock.
            activity_rescue = (
                0.35 * activity_score
                if candidate.activity_score is not None
                else 0.0
            )
            score *= 0.65 + activity_rescue
            reasons.append("melodic boundary is outside the lexical anchor neighborhood")
    independently_supported = False
    if candidate_type in {"forced", "ctc", "alignment", "model"}:
        # CTC word scores are posterior confidence, not lexical confidence.
        # Singing, held vowels, and backing vocals routinely produce low CTC
        # posteriors even when the forced path is text-correct.  Keep a small
        # evidence floor, while retaining the stricter threshold for adapters
        # whose scores are intended to be calibrated model confidence.
        indexed_forced = any(
            "indexed forced timing" in reason.casefold()
            for reason in candidate.reasons
        )
        # The indexed sequence is an independent lexical constraint.  A
        # low posterior on a short function word ("the", "a", etc.) should
        # not erase its otherwise precise forced timestamp; keep the normal
        # floor for unindexed forced spans and a modest floor for indexed
        # ones so zero-confidence evidence is still rejected.
        minimum_model_confidence = (
            0.05
            if candidate_type == "ctc"
            else 0.20
            if indexed_forced
            else 0.4
        )
        independently_supported = (
            candidate_type == "ctc"
            and text_score >= 0.95
            and max(anchor_score, source_agreement) >= 0.8
        )
        if (
            model_confidence < minimum_model_confidence
            and not independently_supported
        ) or text_score < _MIN_FUZZY_SIMILARITY:
            score = 0.0
            reasons.append("forced/CTC candidate is below minimum evidence thresholds")
        else:
            # A CTC/forced candidate is useful evidence, but a constant bonus
            # made every exact CTC token saturate at 1.0 and overpower a
            # better melodic boundary. Keep the bonus calibrated to the
            # model confidence so the selector can combine complementary
            # sources instead of treating one adapter as an oracle.
            evidence_bonus = 0.08 + 0.12 * model_confidence
            score += evidence_bonus
            reasons.append(f"forced/CTC timing evidence (+{evidence_bonus:.3f})")
        if anchor_score <= 0.0 and not independently_supported:
            score = 0.0
            reasons.append("forced/CTC span is temporally incompatible with its ASR anchor")
        elif anchor_score < 0.25:
            score *= anchor_score / 0.25
            reasons.append("forced/CTC span has weak temporal agreement with its ASR anchor")
    reasons.append("source text agrees with candidate" if text_score >= 0.95 else "partial source-text agreement")
    if anchor_score >= 0.8:
        reasons.append("agrees with a global ASR anchor")
    if source_agreement >= 0.8:
        reasons.append("mix and vocal evidence agree")
    if candidate.activity_score is not None:
        reasons.append("candidate overlaps detected vocal activity" if activity_score >= 0.5 else "candidate falls outside detected vocal activity")
        if activity_score <= 0.0:
            score = 0.0
            reasons.append("candidate has no supporting vocal activity")
    if model_confidence <= 0.0 and not independently_supported:
        score = 0.0
        reasons.append("candidate has no model confidence")
    if text_score <= 0.0:
        score = 0.0
        reasons.append("candidate has no lexical agreement")
    if previous_end is not None:
        if candidate.start < previous_end - _BOUNDARY_EPSILON:
            score = 0.0
            reasons.append("non-monotonic candidate rejected")
        else:
            gap = candidate.start - previous_end
            if gap > 0.75:
                # A large unexplained jump is possible, but should not beat a
                # nearby candidate on a small lexical-score advantage. Long
                # gaps are normal between lyric phrases, so this is a gentle
                # tie-breaker rather than a hard rejection of the rest of a
                # song after its first instrumental break.
                gap_penalty = min(0.12, (gap - 0.75) * 0.008)
                score -= gap_penalty
                reasons.append(f"temporal gap penalty {gap_penalty:.3f}")
    return ScoredCandidate(
        candidate=candidate,
        score=max(0.0, min(_candidate_score_upper_bound(), score)),
        reasons=tuple(dict.fromkeys(reasons)),
    )


def _candidate_score_upper_bound() -> float:
    """Return the score ceiling used by the optional dense-lattice decoder."""

    enabled = os.environ.get("ALIGNMENT_ALLOW_SCORE_ABOVE_ONE", "0").strip().casefold()
    return 1.5 if enabled in {"1", "true", "yes", "on"} else 1.0


def _valid_span(start: float | None, end: float | None) -> bool:
    if start is None or end is None:
        return False
    try:
        start_value, end_value = float(start), float(end)
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(start_value)
        and math.isfinite(end_value)
        and start_value >= 0.0
        and end_value >= start_value
    )


def _alignment_has_span(alignment: WordAlignment) -> bool:
    return _valid_span(alignment.start, alignment.end)


def validate_alignment_sequence(alignments: Sequence[WordAlignment]) -> None:
    """Raise when canonical spans are malformed or collide in source order."""

    previous_end: float | None = None
    for alignment in alignments:
        if alignment.start is None and alignment.end is None:
            continue
        if not _valid_span(alignment.start, alignment.end):
            raise ValueError(f"Invalid canonical span for token {alignment.token_id}")
        assert alignment.start is not None and alignment.end is not None
        if previous_end is not None and alignment.start < previous_end - _BOUNDARY_EPSILON:
            raise ValueError(f"Overlapping canonical span for token {alignment.token_id}")
        previous_end = max(previous_end or alignment.end, alignment.end)


def _sanitize_estimated_sequence(alignments: Sequence[WordAlignment]) -> list[WordAlignment]:
    """Keep canonical spans source-ordered, trimming an isolated trusted conflict."""

    result: list[WordAlignment] = []
    previous_end: float | None = None
    for index, alignment in enumerate(alignments):
        if alignment.start is None and alignment.end is None:
            result.append(alignment)
            continue
        if not _valid_span(alignment.start, alignment.end):
            if _enum_value(alignment.status) == AlignmentStatus.ALIGNED.value or alignment.start_locked or alignment.end_locked:
                raise ValueError(f"Invalid trusted canonical span for token {alignment.token_id}")
            result.append(
                replace(
                    alignment,
                    start=None,
                    end=None,
                    status=AlignmentStatus.UNMATCHED,
                    confidence=0.0,
                    evidence=replace(alignment.evidence, reasons=[*alignment.evidence.reasons, "invalid estimated span discarded"]),
                )
            )
            continue
        assert alignment.start is not None and alignment.end is not None
        if previous_end is not None and alignment.start < previous_end - _BOUNDARY_EPSILON:
            if _enum_value(alignment.status) == AlignmentStatus.ALIGNED.value or alignment.start_locked or alignment.end_locked:
                next_start = next(
                    (
                        float(candidate.start)
                        for candidate in alignments[index + 1 :]
                        if _valid_span(candidate.start, candidate.end)
                        and candidate.start is not None
                    ),
                    math.inf,
                )
                trimmed_start = previous_end
                trimmed_end = min(alignment.end, next_start)
                if trimmed_end <= trimmed_start + _BOUNDARY_EPSILON:
                    result.append(
                        replace(
                            alignment,
                            start=None,
                            end=None,
                            status=AlignmentStatus.UNMATCHED,
                            confidence=0.0,
                            start_locked=False,
                            end_locked=False,
                            evidence=replace(
                                alignment.evidence,
                                reasons=[
                                    *alignment.evidence.reasons,
                                    "overlapping trusted span discarded because no non-overlapping duration remained",
                                ],
                            ),
                        )
                    )
                    continue
                alignment = replace(
                    alignment,
                    start=trimmed_start,
                    end=trimmed_end,
                    status=AlignmentStatus.ESTIMATED,
                    confidence=min(alignment.confidence, 0.35),
                    start_locked=False,
                    end_locked=alignment.end_locked and trimmed_end == alignment.end,
                    evidence=replace(
                        alignment.evidence,
                        reasons=[
                            *alignment.evidence.reasons,
                            "overlapping trusted span trimmed to fit neighboring canonical spans",
                        ],
                    ),
                )
                result.append(alignment)
                previous_end = max(previous_end, trimmed_end)
                continue
            result.append(
                replace(
                    alignment,
                    start=None,
                    end=None,
                    status=AlignmentStatus.UNMATCHED,
                    confidence=0.0,
                    evidence=replace(alignment.evidence, reasons=[*alignment.evidence.reasons, "overlapping estimated span discarded"]),
                )
            )
            continue
        previous_end = max(previous_end or alignment.end, alignment.end)
        result.append(alignment)
    return result


def _candidate_duration_prior(
    candidates: Sequence[AlignmentCandidate],
) -> float:
    """Estimate a word-duration prior from independent timing candidates."""

    durations = sorted(
        candidate.end - candidate.start
        for candidate in candidates
        if candidate.candidate_type.casefold() in {"ctc", "forced", "alignment", "model"}
        and _valid_span(candidate.start, candidate.end)
        and 0.02 <= candidate.end - candidate.start <= 3.0
    )
    if not durations:
        durations = sorted(
            candidate.end - candidate.start
            for candidate in candidates
            if _valid_span(candidate.start, candidate.end)
            and 0.02 <= candidate.end - candidate.start <= 3.0
        )
    return median(durations) if durations else 0.45


def _candidate_duration_penalty(
    candidate: AlignmentCandidate,
    duration_prior: float,
    *,
    melody_allowed_duration: float = 0.55,
) -> float:
    """Penalize spans that can consume several neighboring lyric words."""

    duration = candidate.end - candidate.start
    if candidate.candidate_type.casefold() in {"melody", "pitch", "basic_pitch"}:
        # Basic Pitch emits many overlapping note hypotheses.  A long
        # hypothesis is often a merged pair of words, even when its note
        # confidence is high.  Keep genuinely sustained words possible, but
        # make the selector prefer the shorter boundary pair when evidence is
        # otherwise comparable.
        allowed = max(0.08, float(melody_allowed_duration))
        if duration <= allowed:
            return 0.0
        penalty = min(0.8, (duration - allowed) * 0.4)
        # A long melody span is not necessarily a merged pair: held words
        # can be much longer than the lexical clock. Vocal activity support
        # is the independent signal that distinguishes those sustains from a
        # detector span crossing an instrumental gap. Keep a residual prior
        # so unsupported long hypotheses still lose to compact boundaries.
        if candidate.activity_score is not None:
            activity_support = _clamp_confidence(candidate.activity_score)
            penalty *= 1.0 - activity_support
        return penalty
    if candidate.candidate_type.casefold() == "ctc":
        # CTC frequently merges a held vowel with the following consonant or
        # even two adjacent words. Basic Pitch has the more useful boundary
        # evidence for those spans, so make an overlong CTC result expensive
        # enough to avoid swallowing its successor.
        allowed = 0.7
        if duration <= allowed:
            return 0.0
        return min(0.9, (duration - allowed) * 0.8)
    if (
        candidate.candidate_type.casefold() in {"forced", "alignment"}
        and any("indexed forced timing" in reason for reason in candidate.reasons)
    ):
        # A sequence-constrained Wav2Vec path can legitimately span a held
        # syllable for several seconds.  Keep that evidence viable while
        # still rejecting pathological end-of-track spans that swallow an
        # entire phrase.
        allowed = 4.0
        if duration <= allowed:
            return 0.0
        return min(0.75, (duration - allowed) * 0.35)
    allowed = max(0.8, duration_prior * 3.0)
    if duration <= allowed:
        return 0.0
    return min(0.75, (duration - allowed) * 0.35)


def _same_line_gap_penalty(
    tokens: Sequence[LyricToken],
    current_index: int,
    previous_index: int | None,
    previous_end: float | None,
    candidate_start: float,
    activity_spans: Sequence[tuple[float, float]],
) -> float:
    """Discourage repeated-section jumps while allowing real phrase gaps."""

    if previous_index is None or previous_end is None:
        return 0.0
    gap = candidate_start - previous_end
    if gap <= 0.45 or tokens[previous_index].line_index != tokens[current_index].line_index:
        return 0.0
    if activity_spans:
        # A missed lyric token can leave the lexical clock on the tail of the
        # previous vocal island. If this candidate begins on a later island
        # after a real silence, that gap is evidence for the candidate rather
        # than evidence against it.
        for span_start, span_end in activity_spans:
            if (
                span_start - previous_end >= 0.18
                and span_start - 0.16 <= candidate_start <= span_start + 0.16
            ):
                return 0.0
        has_activity = any(
            min(candidate_start, span_end) > max(previous_end, span_start)
            for span_start, span_end in activity_spans
            if span_end > span_start
        )
        if not has_activity:
            return 0.0
        # Within one continuous vocal island, a large hand-off gap usually
        # means the previous word was cut short. Silent phrase gaps were
        # handled above and remain unconstrained.
        if gap > 0.12:
            return min(0.8, (gap - 0.12) * 0.80)
    return min(1.5, (gap - 0.45) * 0.25)


def _timing_grid_point(value: float, step: float, offset: float) -> float:
    """Return the nearest point on an audio-derived timing grid."""

    return max(0.0, offset + round((value - offset) / step) * step)


def _timing_grid_fit(
    candidate: AlignmentCandidate,
    timing_grid_step: float | None,
    timing_grid_offset: float | None,
) -> float:
    """Return how closely both boundaries fit the optional beat grid."""

    if timing_grid_step is None or timing_grid_offset is None or not _valid_span(candidate.start, candidate.end):
        return 0.0
    tolerance = max(0.04, timing_grid_step * 0.42)
    start_distance = abs(candidate.start - _timing_grid_point(candidate.start, timing_grid_step, timing_grid_offset))
    end_distance = abs(candidate.end - _timing_grid_point(candidate.end, timing_grid_step, timing_grid_offset))
    return max(0.0, 1.0 - max(start_distance, end_distance) / tolerance)


def _grid_snapped_candidate(
    candidate: AlignmentCandidate,
    timing_grid_step: float | None,
    timing_grid_offset: float | None,
) -> AlignmentCandidate | None:
    """Create a nearby grid-snapped melody candidate when both ends agree."""

    if (
        timing_grid_step is None
        or timing_grid_offset is None
        or candidate.candidate_type.casefold() not in {"melody", "pitch", "basic_pitch"}
        or not _valid_span(candidate.start, candidate.end)
    ):
        return None
    snapped_start = _timing_grid_point(candidate.start, timing_grid_step, timing_grid_offset)
    snapped_end = _timing_grid_point(candidate.end, timing_grid_step, timing_grid_offset)
    tolerance = max(0.04, timing_grid_step * 0.42)
    if (
        abs(snapped_start - candidate.start) > tolerance
        or abs(snapped_end - candidate.end) > tolerance
        or snapped_end - snapped_start < 0.04
    ):
        return None
    return replace(
        candidate,
        start=snapped_start,
        end=snapped_end,
        reasons=(*candidate.reasons, "nearby audio timing-grid boundaries"),
    )


def select_best_candidates(
    tokens: Sequence[LyricToken],
    candidates_by_token: Mapping[str, Sequence[AlignmentCandidate]],
    *,
    activity_spans: Sequence[tuple[float, float]] = (),
    accepted_threshold: float = 0.55,
    timing_grid_step: float | None = None,
    timing_grid_offset: float | None = None,
    activity_times: Sequence[float] = (),
    activity_values: Sequence[float | bool] = (),
    max_local_candidates: int = 15,
    skip_penalty: float = 0.08,
    grid_gap_weight: float = 0.0,
    minimum_interword_gap: float = 0.0,
    melody_allowed_duration: float = 0.55,
) -> dict[str, ScoredCandidate | None]:
    """Select a single monotonic candidate per token, preventing collisions."""



    # The dynamic program retains the best path ending at each candidate (or
    # at a skipped token). Its state also remembers the previous source token
    # so a large jump within one lyric line cannot silently switch to a later
    # repetition just because that repetition has a slightly higher score.
    # A narrow beam is intentional here. The local lattice is ranked for
    # lexical-clock proximity and boundary direction; duration is applied
    # once during path scoring so long, sustained-word alternatives are not
    # discarded before the adjacent-word context can evaluate them. Keeping
    # hundreds of near-duplicate states lets a late repeated chorus overwhelm
    # a coherent earlier path and makes alignment unnecessarily expensive on
    # CPU workers.
    try:
        configured_path_states = int(
            os.environ.get("ALIGNMENT_MAX_PATH_STATES", "80")
        )
    except ValueError:
        configured_path_states = 80
    # Keep the default inexpensive for interactive jobs, while allowing
    # dense melody lattices to retain enough competing occurrences during
    # offline tuning or on workers with more CPU. The cap prevents a noisy
    # detector from turning the song-level DP into an unbounded search.
    max_path_states = max(
        64,
        min(512, configured_path_states, max(1, len(candidates_by_token))),
    )
    activity_profile = _prepare_activity_profile(activity_times, activity_values)
    duration_priors = {
        token.id: _candidate_duration_prior(candidates_by_token.get(token.id, ()))
        for token in tokens
    }
    paths: dict[
        int | None,
        tuple[
            float,
            float | None,
            float | None,
            tuple[float, float] | None,
            int | None,
            list[ScoredCandidate | None],
        ],
    ] = {
        None: (0.0, None, None, None, None, [])
    }
    previous_retained_candidates: list[AlignmentCandidate] = []
    # Some adapters deserialize ``reasons`` as a list even though the
    # candidate dataclass is frozen, so candidates are not guaranteed to be
    # hashable. Keep references alive and use object identity for caches.
    candidate_references: dict[int, AlignmentCandidate] = {}
    local_priority_cache: dict[int, float] = {}
    scored_option_cache: dict[tuple[int, float | None], ScoredCandidate] = {}
    activity_gap_cache: dict[int, float] = {}
    duration_penalty_cache: dict[tuple[int, str], float] = {}
    fresh_onset_penalty_cache: dict[int, float] = {}
    continuation_reward_cache: dict[int, float] = {}
    try:
        temporal_offset_transition_weight = float(
            os.environ.get("ALIGNMENT_TEMPORAL_OFFSET_WEIGHT", "0.35")
        )
    except ValueError:
        temporal_offset_transition_weight = 0.35
    if not math.isfinite(temporal_offset_transition_weight) or temporal_offset_transition_weight < 0.0:
        temporal_offset_transition_weight = 0.35
    temporal_offset_transition_weight = min(8.0, temporal_offset_transition_weight)
    try:
        temporal_distance_weight = float(
            os.environ.get("ALIGNMENT_TEMPORAL_DISTANCE_WEIGHT", "0.20")
        )
    except ValueError:
        temporal_distance_weight = 0.20
    if not math.isfinite(temporal_distance_weight) or temporal_distance_weight < 0.0:
        temporal_distance_weight = 0.20
    temporal_distance_weight = min(2.0, temporal_distance_weight)
    temporal_start_weight: float | None
    temporal_end_weight: float | None
    try:
        configured_start_weight = os.environ.get("ALIGNMENT_TEMPORAL_START_WEIGHT")
        temporal_start_weight = (
            float(configured_start_weight)
            if configured_start_weight is not None
            else None
        )
    except ValueError:
        temporal_start_weight = None
    try:
        configured_end_weight = os.environ.get("ALIGNMENT_TEMPORAL_END_WEIGHT")
        temporal_end_weight = (
            float(configured_end_weight)
            if configured_end_weight is not None
            else None
        )
    except ValueError:
        temporal_end_weight = None
    if (
        temporal_start_weight is None
        or temporal_end_weight is None
        or not math.isfinite(temporal_start_weight)
        or not math.isfinite(temporal_end_weight)
        or temporal_start_weight < 0.0
        or temporal_end_weight < 0.0
    ):
        temporal_start_weight = None
        temporal_end_weight = None
    else:
        temporal_start_weight = min(2.0, temporal_start_weight)
        temporal_end_weight = min(2.0, temporal_end_weight)

    def temporal_boundary_penalty(candidate: AlignmentCandidate) -> float:
        """Penalize lexical-clock drift, with optional start/end asymmetry."""

        if (
            temporal_start_weight is not None
            and temporal_end_weight is not None
            and candidate.temporal_start_offset is not None
            and candidate.temporal_end_offset is not None
        ):
            return min(
                0.90,
                abs(float(candidate.temporal_start_offset)) * temporal_start_weight
                + abs(float(candidate.temporal_end_offset)) * temporal_end_weight,
            )
        if candidate.temporal_distance is None:
            return 0.0
        return min(
            0.90,
            float(candidate.temporal_distance) * temporal_distance_weight,
        )
    try:
        timing_grid_fit_weight = float(
            os.environ.get("ALIGNMENT_TIMING_GRID_FIT_WEIGHT", "0.0")
        )
    except ValueError:
        timing_grid_fit_weight = 0.0
    if not math.isfinite(timing_grid_fit_weight) or timing_grid_fit_weight < 0.0:
        timing_grid_fit_weight = 0.0
    timing_grid_fit_weight = min(2.0, timing_grid_fit_weight)
    try:
        same_line_gap_weight = float(
            os.environ.get("ALIGNMENT_SAME_LINE_GAP_WEIGHT", "1.0")
        )
    except ValueError:
        same_line_gap_weight = 1.0
    if not math.isfinite(same_line_gap_weight) or same_line_gap_weight < 0.0:
        same_line_gap_weight = 1.0
    same_line_gap_weight = min(4.0, same_line_gap_weight)
    try:
        fresh_onset_penalty_weight = float(
            os.environ.get("ALIGNMENT_FRESH_ONSET_PENALTY_WEIGHT", "1.0")
        )
    except ValueError:
        fresh_onset_penalty_weight = 1.0
    if not math.isfinite(fresh_onset_penalty_weight) or fresh_onset_penalty_weight < 0.0:
        fresh_onset_penalty_weight = 1.0
    fresh_onset_penalty_weight = min(4.0, fresh_onset_penalty_weight)
    try:
        activity_onset_weight = float(
            os.environ.get("ALIGNMENT_ACTIVITY_ONSET_WEIGHT", "0.65")
        )
    except ValueError:
        activity_onset_weight = 0.65
    if not math.isfinite(activity_onset_weight) or activity_onset_weight < 0.0:
        activity_onset_weight = 0.65
    activity_onset_weight = min(1.0, activity_onset_weight)
    try:
        acoustic_continuity_weight = float(
            os.environ.get("ALIGNMENT_ACOUSTIC_CONTINUITY_WEIGHT", "0.0")
        )
    except ValueError:
        acoustic_continuity_weight = 0.0
    if not math.isfinite(acoustic_continuity_weight) or acoustic_continuity_weight < 0.0:
        acoustic_continuity_weight = 0.0
    acoustic_continuity_weight = min(2.0, acoustic_continuity_weight)
    try:
        continuation_reward_weight = float(
            os.environ.get("ALIGNMENT_CONTINUATION_REWARD_WEIGHT", "0.0")
        )
    except ValueError:
        continuation_reward_weight = 0.0
    if not math.isfinite(continuation_reward_weight) or continuation_reward_weight < 0.0:
        continuation_reward_weight = 0.0
    continuation_reward_weight = min(2.0, continuation_reward_weight)
    try:
        successor_gap_weight = float(
            os.environ.get("ALIGNMENT_SUCCESSOR_GAP_WEIGHT", "0.0")
        )
    except ValueError:
        successor_gap_weight = 0.0
    if not math.isfinite(successor_gap_weight) or successor_gap_weight < 0.0:
        successor_gap_weight = 0.0
    successor_gap_weight = min(4.0, successor_gap_weight)
    try:
        handoff_gap_reward_weight = float(
            os.environ.get("ALIGNMENT_HANDOFF_GAP_REWARD_WEIGHT", "0.0")
        )
    except ValueError:
        handoff_gap_reward_weight = 0.0
    if not math.isfinite(handoff_gap_reward_weight) or handoff_gap_reward_weight < 0.0:
        handoff_gap_reward_weight = 0.0
    handoff_gap_reward_weight = min(2.0, handoff_gap_reward_weight)
    try:
        handoff_gap_target = float(
            os.environ.get("ALIGNMENT_HANDOFF_GAP_TARGET", "0.06")
        )
    except ValueError:
        handoff_gap_target = 0.06
    if not math.isfinite(handoff_gap_target) or handoff_gap_target < 0.0:
        handoff_gap_target = 0.06
    handoff_gap_target = min(0.25, handoff_gap_target)
    try:
        handoff_gap_scale = float(
            os.environ.get("ALIGNMENT_HANDOFF_GAP_SCALE", "0.08")
        )
    except ValueError:
        handoff_gap_scale = 0.08
    if not math.isfinite(handoff_gap_scale) or handoff_gap_scale <= 0.0:
        handoff_gap_scale = 0.08
    handoff_gap_scale = min(0.5, handoff_gap_scale)
    try:
        active_duration_reward_weight = float(
            os.environ.get("ALIGNMENT_ACTIVE_DURATION_REWARD_WEIGHT", "0.0")
        )
    except ValueError:
        active_duration_reward_weight = 0.0
    if not math.isfinite(active_duration_reward_weight) or active_duration_reward_weight < 0.0:
        active_duration_reward_weight = 0.0
    active_duration_reward_weight = min(2.0, active_duration_reward_weight)
    try:
        melody_late_end_relief = float(
            os.environ.get("ALIGNMENT_MELODY_LATE_END_RELIEF", "0.0")
        )
    except ValueError:
        melody_late_end_relief = 0.0
    if not math.isfinite(melody_late_end_relief) or melody_late_end_relief < 0.0:
        melody_late_end_relief = 0.0
    melody_late_end_relief = min(1.0, melody_late_end_relief)
    try:
        melody_end_continuation_penalty_weight = float(
            os.environ.get("ALIGNMENT_MELODY_END_CONTINUATION_PENALTY_WEIGHT", "0.0")
        )
    except ValueError:
        melody_end_continuation_penalty_weight = 0.0
    if (
        not math.isfinite(melody_end_continuation_penalty_weight)
        or melody_end_continuation_penalty_weight < 0.0
    ):
        melody_end_continuation_penalty_weight = 0.0
    melody_end_continuation_penalty_weight = min(
        2.0, melody_end_continuation_penalty_weight
    )
    preserve_endpoint_variants = (
        os.environ.get("ALIGNMENT_PRESERVE_ENDPOINT_VARIANTS", "0")
        .strip()
        .casefold()
        in {"1", "true", "yes", "on"}
    )
    try:
        melody_overlap_tolerance = float(
            os.environ.get("ALIGNMENT_MELODY_OVERLAP_TOLERANCE", "0.0")
        )
    except ValueError:
        melody_overlap_tolerance = 0.0
    if not math.isfinite(melody_overlap_tolerance) or melody_overlap_tolerance < 0.0:
        melody_overlap_tolerance = 0.0
    melody_overlap_tolerance = min(0.25, melody_overlap_tolerance)

    def allows_melody_overlap(
        candidate: AlignmentCandidate,
        previous_end: float | None,
    ) -> bool:
        """Allow a small detector-tail overlap for a melody onset candidate."""

        return (
            melody_overlap_tolerance > 0.0
            and previous_end is not None
            and candidate.candidate_type.casefold()
            in {"melody", "pitch", "basic_pitch"}
            and 0.0 < previous_end - candidate.start
            <= melody_overlap_tolerance
        )

    def continuation_reward_for(
        token_index: int,
        candidate: AlignmentCandidate,
    ) -> float:
        """Reward a long, active span that hands off to the next vocal word."""

        if (
            candidate.candidate_type.casefold()
            not in {"melody", "pitch", "basic_pitch"}
            or not _valid_span(candidate.start, candidate.end)
            or candidate.end - candidate.start < 0.65
            or candidate.activity_score is None
            or _clamp_confidence(candidate.activity_score) < 0.70
            or token_index + 1 >= len(tokens)
        ):
            return 0.0
        next_candidates = candidates_by_token.get(tokens[token_index + 1].id, ())
        candidate_source = candidate.audio_source.casefold()
        require_same_source = (
            os.environ.get("ALIGNMENT_CONTINUATION_REQUIRE_SOURCE", "1")
            .strip()
            .casefold()
            in {"1", "true", "yes", "on"}
        )
        reward = 0.0
        for next_candidate in next_candidates:
            if (
                next_candidate.candidate_type.casefold()
                not in {"melody", "pitch", "basic_pitch"}
                or (
                    require_same_source
                    and next_candidate.audio_source.casefold() != candidate_source
                )
                or not _valid_span(next_candidate.start, next_candidate.end)
            ):
                continue
            gap = float(next_candidate.start) - float(candidate.end)
            if 0.0 <= gap <= 0.12:
                reward = max(reward, 1.0 - gap / 0.12)
        return reward

    successor_gap_cache: dict[int, float] = {}

    def successor_gap_penalty_for(
        token_index: int,
        candidate: AlignmentCandidate,
    ) -> float:
        """Penalize an unexplained gap before the next lexical onset.

        The normal DP constrains the next word's start against the current
        word's end only after that successor is selected.  Held sung words
        therefore lose locally: a short harmonic span can end early without
        violating monotonicity, leaving the following word to absorb the
        missing tail.  This look-ahead is deliberately soft and only applies
        inside one lyric line where a nearby successor candidate exists.
        """

        if (
            successor_gap_weight <= 0.0
            or token_index + 1 >= len(tokens)
            or tokens[token_index + 1].line_index != tokens[token_index].line_index
            or not _valid_span(candidate.start, candidate.end)
        ):
            return 0.0
        cached = successor_gap_cache.get(id(candidate))
        if cached is not None:
            return cached
        candidate_source = candidate.audio_source.casefold()
        next_candidates = [
            next_candidate
            for next_candidate in candidates_by_token.get(tokens[token_index + 1].id, ())
            if _valid_span(next_candidate.start, next_candidate.end)
            and next_candidate.start >= candidate.end - _BOUNDARY_EPSILON
            and (
                next_candidate.audio_source.casefold() == candidate_source
                or next_candidate.candidate_type.casefold()
                not in {"melody", "pitch", "basic_pitch"}
            )
        ]
        if not next_candidates:
            successor_gap_cache[id(candidate)] = 0.0
            return 0.0
        nearest = min(
            next_candidates,
            key=lambda next_candidate: (
                max(0.0, next_candidate.start - candidate.end),
                math.inf
                if next_candidate.temporal_distance is None
                else next_candidate.temporal_distance,
                -score_alignment_candidate(next_candidate).score,
            ),
        )
        gap = max(0.0, float(nearest.start) - float(candidate.end))
        # Phrase gaps are common. Once the gap is clearly larger than a
        # normal word handoff, stop pulling a held word toward a later lyric
        # occurrence. The residual penalty still distinguishes a short tail
        # from a nearby onset without rewriting phrase timing.
        penalty = min(
            0.75,
            max(0.0, gap - 0.10) * successor_gap_weight,
        ) if gap <= 0.90 else 0.0
        successor_gap_cache[id(candidate)] = penalty
        return penalty

    for token_index, token in enumerate(tokens):
        candidates = []
        for candidate in candidates_by_token.get(token.id, ()):
            if (
                candidate.activity_score is None
                and activity_spans
                and candidate.candidate_type.casefold() in {"melody", "pitch", "basic_pitch", "forced", "ctc"}
            ):
                activity_score = _candidate_activity_score(candidate, activity_spans)
                if (
                    activity_profile is not None
                    and candidate.candidate_type.casefold() in {"melody", "pitch", "basic_pitch"}
                ):
                    onset_score = _candidate_activity_onset_score(candidate, activity_profile)
                    activity_score = (
                        (1.0 - activity_onset_weight) * activity_score
                        + activity_onset_weight * onset_score
                    )
                candidate = replace(candidate, activity_score=activity_score)
            candidates.append(candidate)
            candidate_references[id(candidate)] = candidate
        candidates = [candidate for candidate in candidates if _valid_span(candidate.start, candidate.end)]
        # Whisper-style word ends are often early on held vowels.  The
        # melody lattice exposes a stronger negative signal for one common
        # failure mode: a short candidate whose end cuts through a
        # high-confidence vocal note.  Remove only those decoys, and only
        # when another candidate remains for the token.  The fallback is
        # important for noisy recordings where the detector may cover every
        # available boundary.
        try:
            melody_end_inside_threshold = float(
                os.environ.get("ALIGNMENT_MELODY_END_INSIDE_THRESHOLD", "0.65")
            )
        except ValueError:
            melody_end_inside_threshold = 0.65
        if (
            not math.isfinite(melody_end_inside_threshold)
            or melody_end_inside_threshold < 0.0
        ):
            melody_end_inside_threshold = 0.65
        melody_end_inside_threshold = min(1.0, melody_end_inside_threshold)
        try:
            melody_end_inside_max_duration = float(
                os.environ.get("ALIGNMENT_MELODY_END_INSIDE_MAX_DURATION", "1.0")
            )
        except ValueError:
            melody_end_inside_max_duration = 1.0
        if (
            not math.isfinite(melody_end_inside_max_duration)
            or melody_end_inside_max_duration <= 0.0
        ):
            melody_end_inside_max_duration = 1.0
        try:
            melody_end_inside_gate = os.environ.get(
                "ALIGNMENT_MELODY_END_INSIDE_GATE", "1"
            ).strip().casefold() in {"1", "true", "yes", "on"}
        except AttributeError:
            melody_end_inside_gate = True
        if melody_end_inside_gate and len(candidates) > 1:
            filtered_candidates = [
                candidate
                for candidate in candidates
                if not (
                    candidate.candidate_type.casefold()
                    in {"melody", "pitch", "basic_pitch"}
                    and candidate.melody_end_inside_confidence is not None
                    and _clamp_confidence(candidate.melody_end_inside_confidence)
                    >= melody_end_inside_threshold
                    and candidate.end - candidate.start
                    <= melody_end_inside_max_duration
                )
            ]
            if filtered_candidates:
                candidates = filtered_candidates
        if os.environ.get("ALIGNMENT_DEBUG_TOKEN") == str(token_index):
            print(
                "ALIGNMENT_DEBUG_COUNT",
                token_index,
                token.id,
                len(candidates),
                sum(
                    abs(candidate.start - 35.2) < 0.02 and candidate.end > 36.4
                    for candidate in candidates
                ),
                flush=True,
            )
        def local_priority(candidate: AlignmentCandidate) -> float:
            """Rank a bounded lattice before the global path search."""

            cached = local_priority_cache.get(id(candidate))
            if cached is not None:
                return cached
            priority = score_alignment_candidate(candidate).score
            if candidate.temporal_distance is not None:
                priority -= temporal_boundary_penalty(candidate)
            if candidate.temporal_start_offset is not None:
                priority -= _candidate_early_start_penalty(candidate)
            if candidate.temporal_end_offset is not None:
                late_end_penalty = min(
                    0.25,
                    max(0.0, float(candidate.temporal_end_offset) - 0.05) * 0.20,
                )
                if (
                    melody_late_end_relief > 0.0
                    and candidate.candidate_type.casefold()
                    in {"melody", "pitch", "basic_pitch"}
                    and candidate.end - candidate.start >= 0.65
                    and (
                        candidate.melody_end_inside_confidence is None
                        or _clamp_confidence(candidate.melody_end_inside_confidence)
                        < melody_end_inside_threshold
                    )
                ):
                    late_end_penalty *= 1.0 - melody_late_end_relief
                priority -= late_end_penalty
            activity_gap_penalty = activity_gap_cache.get(id(candidate))
            if activity_gap_penalty is None:
                activity_gap_penalty = _candidate_internal_activity_gap_penalty(candidate, activity_spans)
                activity_gap_cache[id(candidate)] = activity_gap_penalty
            if activity_gap_penalty:
                priority -= activity_gap_penalty
            if (
                melody_end_continuation_penalty_weight > 0.0
                and candidate.candidate_type.casefold()
                in {"melody", "pitch", "basic_pitch"}
                and candidate.melody_end_inside_confidence is not None
            ):
                priority -= melody_end_continuation_penalty_weight * _clamp_confidence(
                    candidate.melody_end_inside_confidence
                )
            if candidate.candidate_type.casefold() == "ctc":
                priority += 0.15
            if acoustic_continuity_weight > 0.0 and candidate.acoustic_continuity is not None:
                priority += acoustic_continuity_weight * (
                    _clamp_confidence(candidate.acoustic_continuity) - 0.5
                )
            if timing_grid_fit_weight > 0.0:
                grid_fit = _timing_grid_fit(
                    candidate,
                    timing_grid_step,
                    timing_grid_offset,
                )
                if grid_fit:
                    priority += timing_grid_fit_weight * grid_fit
            local_priority_cache[id(candidate)] = priority
            return priority

        candidates.sort(
            key=lambda candidate: (-local_priority(candidate), candidate.start, candidate.end, candidate.model)
        )
        if os.environ.get("ALIGNMENT_DEBUG_CANDIDATES") == str(token_index):
            print(
                "ALIGNMENT_DEBUG_TOP",
                token_index,
                [
                    {
                        "start": round(candidate.start, 4),
                        "end": round(candidate.end, 4),
                        "duration": round(candidate.end - candidate.start, 4),
                        "score": round(score_alignment_candidate(candidate).score, 4),
                        "priority": round(local_priority(candidate), 4),
                        "distance": round(candidate.temporal_distance or 0.0, 4),
                        "end_offset": round(candidate.temporal_end_offset or 0.0, 4),
                        "inside": round(candidate.melody_end_inside_confidence or 0.0, 4),
                        "activity": round(candidate.activity_score or 0.0, 4),
                        "source": candidate.audio_source,
                    }
                    for candidate in candidates[:40]
                ],
                flush=True,
            )
        # A dense pitch lattice can contain hundreds of near-equivalent
        # combinations for one token. Retaining the strongest local options
        # preserves recovery choices while keeping the song-level DP bounded.
        # Keep a broad enough local lattice for repeated phrases. The lexical
        # clock can be several hundred milliseconds early, so the musically
        # correct boundary is often not among the first few score ties. The
        # song-level beam below still bounds the global search.
        retained_candidates = candidates[:max(1, int(max_local_candidates))]
        if previous_retained_candidates and retained_candidates:
            # Preserve a few boundary-linked options so a good predecessor is
            # not forced to hand off to a candidate that starts too early. A
            # melody word often ends at the same detector boundary where the
            # next word begins; ranking each token independently can discard
            # that hand-off before the global DP sees it.
            linked_candidates: list[AlignmentCandidate] = []
            for previous_candidate in previous_retained_candidates:
                if not _valid_span(previous_candidate.start, previous_candidate.end):
                    continue
                assert previous_candidate.end is not None
                nearest = min(
                    candidates,
                    key=lambda candidate: (
                        abs(candidate.start - previous_candidate.end),
                        -local_priority(candidate),
                        candidate.end,
                    ),
                    default=None,
                )
                if (
                    nearest is not None
                    and abs(nearest.start - previous_candidate.end) <= 0.06
                    and nearest not in retained_candidates
                ):
                    linked_candidates.append(nearest)
            for linked_candidate in linked_candidates[:4]:
                if linked_candidate not in retained_candidates:
                    retained_candidates.append(linked_candidate)
        # Preserve a few short boundary pairs as escape hatches for a
        # melody detector span that accidentally merges adjacent lyric
        # words. The normal local ranking intentionally prefers the lexical
        # clock, so without these alternatives a later token cannot make the
        # global path reject an overlong predecessor.
        for short_candidate in sorted(
            candidates,
            key=lambda candidate: (candidate.end - candidate.start, candidate.start, candidate.end),
        )[:3]:
            if short_candidate not in retained_candidates:
                retained_candidates.append(short_candidate)
        # Keep a separate, small lexical-clock lane.  Basic Pitch confidence
        # is not calibrated against the musical boundary: the correct tail
        # of a held word can have a lower note confidence than a short
        # harmonic fragment.  The normal score lane therefore needs a few
        # closest temporal alternatives so the sequence decoder can evaluate
        # them with neighboring words.  This is deliberately bounded; the
        # environment knob is useful for tuning model deployments without
        # turning the default interactive search into a dense cross-product.
        try:
            temporal_candidate_count = int(
                os.environ.get("ALIGNMENT_RETAIN_TEMPORAL_CANDIDATES", "0")
            )
        except ValueError:
            temporal_candidate_count = 0
        temporal_candidate_count = max(0, min(16, temporal_candidate_count))
        for temporal_candidate in sorted(
            candidates,
            key=lambda candidate: (
                math.inf
                if candidate.temporal_distance is None
                else candidate.temporal_distance,
                candidate.start,
                candidate.end,
            ),
        )[:temporal_candidate_count]:
            if temporal_candidate not in retained_candidates:
                retained_candidates.append(temporal_candidate)
        # Keep a separate endpoint lane for each of the strongest onset
        # buckets.  Melody candidates are a Cartesian product of note
        # boundaries: the locally best pair often has the right onset but
        # ends at the first detector fragment of a held word.  Retaining a
        # few longer endpoint variants *within that same onset bucket* gives
        # the sequence decoder a chance to recover the tail without adding
        # arbitrary later occurrences to the path.  It is opt-in while the
        # lane is calibrated because the right count depends on detector
        # density and CPU budget.
        try:
            endpoint_variant_count = int(
                os.environ.get("ALIGNMENT_RETAIN_ENDPOINT_VARIANTS", "0")
            )
        except ValueError:
            endpoint_variant_count = 0
        endpoint_variant_count = max(0, min(8, endpoint_variant_count))
        if endpoint_variant_count > 0:
            onset_buckets: dict[tuple[str, int], list[AlignmentCandidate]] = {}
            for candidate in candidates:
                if candidate.candidate_type.casefold() not in {
                    "melody", "pitch", "basic_pitch"
                }:
                    continue
                onset_buckets.setdefault(
                    (
                        candidate.audio_source.casefold(),
                        round(candidate.start / 0.08),
                    ),
                    [],
                ).append(candidate)
            onset_groups = sorted(
                onset_buckets.values(),
                key=lambda group: (
                    -max(local_priority(candidate) for candidate in group),
                    min(candidate.start for candidate in group),
                ),
            )
            try:
                endpoint_onset_group_count = int(
                    os.environ.get("ALIGNMENT_ENDPOINT_ONSET_GROUPS", "8")
                )
            except ValueError:
                endpoint_onset_group_count = 8
            endpoint_onset_group_count = max(1, min(32, endpoint_onset_group_count))
            for group in onset_groups[:endpoint_onset_group_count]:
                # The first option preserves the normal local ranking. The
                # remaining options are the longest distinct endpoints in
                # this onset lane, bounded to avoid retaining merged phrases.
                group_candidates = sorted(
                    group,
                    key=lambda candidate: (
                        -local_priority(candidate),
                        candidate.start,
                        candidate.end,
                    ),
                )
                for group_candidate in group_candidates[:1]:
                    if group_candidate not in retained_candidates:
                        retained_candidates.append(group_candidate)
                long_candidates = sorted(
                    group,
                    key=lambda candidate: (
                        -(candidate.end - candidate.start),
                        -local_priority(candidate),
                        candidate.end,
                    ),
                )
                for long_candidate in long_candidates[:endpoint_variant_count]:
                    if long_candidate not in retained_candidates:
                        retained_candidates.append(long_candidate)
        # Keep a separate sustain lane for a long candidate that is actually
        # covered by the vocal detector.  A held word is often farther from
        # the ASR end clock than a short harmonic fragment, so it can lose
        # every local tie before the successor's onset is available.  Do not
        # retain arbitrary long spans: they commonly cross several words or
        # instrumental gaps.  The lane is bounded and opt-in while its
        # thresholds are tuned against recordings with different mixing and
        # latency characteristics.
        try:
            active_long_candidate_count = int(
                os.environ.get("ALIGNMENT_RETAIN_ACTIVE_LONG_CANDIDATES", "0")
            )
        except ValueError:
            active_long_candidate_count = 0
        active_long_candidate_count = max(0, min(16, active_long_candidate_count))
        if active_long_candidate_count > 0:
            try:
                active_long_min_duration = float(
                    os.environ.get("ALIGNMENT_ACTIVE_LONG_MIN_DURATION", "0.65")
                )
            except ValueError:
                active_long_min_duration = 0.65
            try:
                active_long_max_duration = float(
                    os.environ.get("ALIGNMENT_ACTIVE_LONG_MAX_DURATION", "1.8")
                )
            except ValueError:
                active_long_max_duration = 1.8
            try:
                active_long_min_activity = float(
                    os.environ.get("ALIGNMENT_ACTIVE_LONG_MIN_ACTIVITY", "0.70")
                )
            except ValueError:
                active_long_min_activity = 0.70
            try:
                active_long_max_distance = float(
                    os.environ.get("ALIGNMENT_ACTIVE_LONG_MAX_DISTANCE", "0.55")
                )
            except ValueError:
                active_long_max_distance = 0.55
            if (
                not math.isfinite(active_long_min_duration)
                or not math.isfinite(active_long_max_duration)
                or not math.isfinite(active_long_min_activity)
                or not math.isfinite(active_long_max_distance)
                or active_long_min_duration < 0.0
                or active_long_max_duration < active_long_min_duration
                or not 0.0 <= active_long_min_activity <= 1.0
                or active_long_max_distance < 0.0
            ):
                active_long_min_duration = 0.65
                active_long_max_duration = 1.8
                active_long_min_activity = 0.70
                active_long_max_distance = 0.55
            active_long_candidates = [
                candidate
                for candidate in candidates
                if candidate.candidate_type.casefold()
                in {"melody", "pitch", "basic_pitch"}
                and _valid_span(candidate.start, candidate.end)
                and active_long_min_duration
                <= candidate.end - candidate.start
                <= active_long_max_duration
                and candidate.activity_score is not None
                and _clamp_confidence(candidate.activity_score) >= active_long_min_activity
                and (
                    candidate.temporal_distance is None
                    or candidate.temporal_distance <= active_long_max_distance
                )
            ]
            for active_long_candidate in sorted(
                active_long_candidates,
                key=lambda candidate: (
                    math.inf
                    if candidate.temporal_distance is None
                    else candidate.temporal_distance,
                    -(candidate.activity_score or 0.0),
                    -float(candidate.end - candidate.start),
                    candidate.start,
                ),
            )[:active_long_candidate_count]:
                if active_long_candidate not in retained_candidates:
                    retained_candidates.append(active_long_candidate)
        # Keep a few candidates whose end is a plausible handoff into the
        # next lyric token.  The local lexical clock tends to prefer the
        # first part of a held syllable, while the next token's onset is often
        # the only evidence that the current word must continue.  Without
        # this look-ahead, a valid long span can be discarded before the
        # monotonic decoder has a chance to evaluate the pair.  Compare
        # against the raw neighboring lattice, but cap the additions so a
        # dense detector cannot make the local search unbounded.
        if token_index + 1 < len(tokens):
            next_token = tokens[token_index + 1]
            try:
                melody_handoff_max_distance = float(
                    os.environ.get("ALIGNMENT_MELODY_HANDOFF_MAX_DISTANCE", "0.12")
                )
            except ValueError:
                melody_handoff_max_distance = 0.12
            if (
                not math.isfinite(melody_handoff_max_distance)
                or melody_handoff_max_distance < 0.0
            ):
                melody_handoff_max_distance = 0.12
            melody_handoff_max_distance = min(0.5, melody_handoff_max_distance)
            next_starts = sorted(
                {
                    float(next_candidate.start)
                    for next_candidate in candidates_by_token.get(next_token.id, ())
                    if _valid_span(next_candidate.start, next_candidate.end)
                }
            )
            if next_starts:
                def nearest_start(target: float) -> float:
                    """Return the closest value from an already sorted list."""

                    insertion = bisect_left(next_starts, target)
                    if insertion <= 0:
                        return next_starts[0]
                    if insertion >= len(next_starts):
                        return next_starts[-1]
                    before = next_starts[insertion - 1]
                    after = next_starts[insertion]
                    return before if target - before < after - target else after

                handoff_candidates = []
                for candidate in candidates:
                    # A candidate start from the next token that falls
                    # inside this span is not a handoff; it is usually a
                    # nearby boundary hypothesis generated from the same
                    # detector lane. Only compare with non-overlapping next
                    # starts so a long held-word candidate can be retained
                    # for the actual successor.
                    following_index = bisect_left(
                        next_starts,
                        candidate.end - _BOUNDARY_EPSILON,
                    )
                    if following_index >= len(next_starts):
                        continue
                    nearest_next_start = next_starts[following_index]
                    handoff_distance = abs(nearest_next_start - candidate.end)
                    if handoff_distance <= melody_handoff_max_distance:
                        handoff_candidates.append(
                            (
                                handoff_distance,
                                -local_priority(candidate),
                                candidate,
                            )
                        )
                handoff_candidates.sort(key=lambda item: (item[0], item[1]))
                if os.environ.get("ALIGNMENT_DEBUG_TOKEN") == str(token_index):
                    print(
                        "ALIGNMENT_DEBUG_HANDOFF",
                        token_index,
                        len(handoff_candidates),
                        [
                            (
                                round(item[0], 3),
                                round(item[2].start, 3),
                                round(item[2].end, 3),
                                round(item[2].end - item[2].start, 3),
                                round(-item[1], 3),
                            )
                            for item in sorted(
                                handoff_candidates,
                                key=lambda item: (
                                    -(item[2].end - item[2].start),
                                    item[0],
                                ),
                            )[:12]
                        ],
                        flush=True,
                    )
                    for target in next_starts:
                        if abs(target - 36.4784) < 0.01:
                            group = [
                                candidate
                                for candidate in candidates
                                if abs(candidate.end - target) <= 0.01
                            ]
                            print(
                                "ALIGNMENT_DEBUG_GROUP",
                                target,
                                len(group),
                                [
                                    (
                                        round(candidate.start, 3),
                                        round(candidate.end - candidate.start, 3),
                                        round(local_priority(candidate), 3),
                                    )
                                    for candidate in sorted(
                                        group,
                                        key=lambda candidate: (
                                            -(candidate.end - candidate.start),
                                            -local_priority(candidate),
                                        ),
                                    )[:20]
                                ],
                                flush=True,
                            )
                for _distance, _priority, handoff_candidate in handoff_candidates[:6]:
                    if handoff_candidate not in retained_candidates:
                        retained_candidates.append(handoff_candidate)
                try:
                    per_target_handoff_count = int(
                        os.environ.get("ALIGNMENT_PER_TARGET_HANDOFF", "1")
                    )
                except ValueError:
                    per_target_handoff_count = 0
                if per_target_handoff_count > 0:
                    handoff_by_target: dict[float, list[AlignmentCandidate]] = {}
                    for handoff_candidate in (
                        item[2]
                        for item in handoff_candidates
                    ):
                        nearest_target = nearest_start(handoff_candidate.end)
                        handoff_by_target.setdefault(
                            round(nearest_target, 2),
                            [],
                        ).append(handoff_candidate)
                    target_groups = sorted(
                        handoff_by_target.values(),
                        key=lambda group: (
                            -max(local_priority(candidate) for candidate in group),
                            min(
                                abs(candidate.end - nearest_start(candidate.end))
                                for candidate in group
                            ),
                        ),
                    )
                    try:
                        handoff_target_group_limit = int(
                            os.environ.get("ALIGNMENT_HANDOFF_TARGET_GROUP_LIMIT", "16")
                        )
                    except ValueError:
                        handoff_target_group_limit = 16
                    handoff_target_group_limit = max(
                        0,
                        min(128, handoff_target_group_limit),
                    )
                    for group in target_groups[
                        :min(handoff_target_group_limit, per_target_handoff_count)
                    ]:
                        group_ranked = sorted(
                            group,
                            key=lambda candidate: (
                                -local_priority(candidate),
                                candidate.end - candidate.start,
                            ),
                        )
                        for group_candidate in group_ranked[:2]:
                            if group_candidate not in retained_candidates:
                                retained_candidates.append(group_candidate)
                        duration_candidates = [
                            candidate
                            for candidate in group
                            if 0.65 <= candidate.end - candidate.start <= 1.8
                        ]
                        if duration_candidates:
                            duration_candidate = max(
                                duration_candidates,
                                key=lambda candidate: (
                                    local_priority(candidate),
                                    -abs((candidate.end - candidate.start) - 1.0),
                                ),
                            )
                            if duration_candidate not in retained_candidates:
                                retained_candidates.append(duration_candidate)
        # Preserve a small number of long spans as well.  Basic Pitch often
        # represents a held syllable with a family of short overlapping note
        # hypotheses; retaining only the shortest escape hatches makes the
        # global decoder unable to recover the actual musical tail.  Keep
        # this bounded because long harmonic spans are common decoys too.
        try:
            long_candidate_count = int(
                os.environ.get("ALIGNMENT_MELODY_LONG_CANDIDATES", "0")
            )
        except ValueError:
            long_candidate_count = 0
        for long_candidate in sorted(
            candidates,
            key=lambda candidate: (-(candidate.end - candidate.start), candidate.start, candidate.end),
        )[:max(0, min(8, long_candidate_count))]:
            if long_candidate not in retained_candidates:
                retained_candidates.append(long_candidate)
        candidates = retained_candidates
        if os.environ.get("ALIGNMENT_DEBUG_TOKEN") == str(token_index):
            print(
                "ALIGNMENT_DEBUG_RETAINED",
                token_index,
                len(candidates),
                sum(
                    abs(candidate.start - 35.2) < 0.02
                    and candidate.end > 36.4
                    for candidate in candidates
                ),
                flush=True,
            )
            if os.environ.get("ALIGNMENT_DEBUG_RETAINED") == "detail":
                print(
                    "ALIGNMENT_DEBUG_RETAINED_DETAIL",
                    [
                        (
                            round(candidate.start, 4),
                            round(candidate.end, 4),
                            round(candidate.end - candidate.start, 4),
                            round(local_priority(candidate), 4),
                            round(candidate.melody_end_inside_confidence or 0.0, 4),
                            round(candidate.activity_score or 0.0, 4),
                        )
                        for candidate in candidates
                    ],
                    flush=True,
                )
        previous_retained_candidates = retained_candidates
        next_paths: dict[
            int | None,
            tuple[
                float,
                float | None,
                float | None,
                tuple[float, float] | None,
                int | None,
                list[ScoredCandidate | None],
            ],
        ] = {}
        for previous_key, (
            path_score,
            previous_end,
            previous_start,
            previous_offsets,
            previous_index,
            path,
        ) in paths.items():
            skip_score = path_score - max(0.0, float(skip_penalty))
            # Skipping a token retains the previous boundary state. A single
            # None state would collapse paths with different end times.
            existing = next_paths.get(previous_key)
            skip_path = (
                skip_score,
                previous_end,
                previous_start,
                previous_offsets,
                previous_index,
                [*path, None],
            )
            if existing is None or skip_score > existing[0]:
                next_paths[previous_key] = skip_path
            for candidate in candidates:
                if (
                    os.environ.get("ALIGNMENT_DEBUG_TOKEN") == str(token_index)
                    and abs(candidate.start - 35.2) < 0.02
                    and candidate.end > 36.4
                ):
                    print(
                        "ALIGNMENT_DEBUG_SEEN",
                        token_index,
                        round(candidate.start, 4),
                        round(candidate.end, 4),
                        flush=True,
                    )
                if (
                    previous_end is not None
                    and candidate.start < previous_end - _BOUNDARY_EPSILON
                    and not allows_melody_overlap(candidate, previous_end)
                ):
                    continue
                if (
                    previous_end is not None
                    and float(minimum_interword_gap) > 0.0
                    and candidate.start < previous_end + float(minimum_interword_gap)
                    and not allows_melody_overlap(candidate, previous_end)
                ):
                    continue
                option_key = (id(candidate), previous_end)
                option = scored_option_cache.get(option_key)
                if option is None:
                    option = score_alignment_candidate(candidate, previous_end=previous_end)
                    scored_option_cache[option_key] = option
                # Do not let a candidate below the local evidence threshold
                # block a stronger later candidate.
                if option.score < accepted_threshold:
                    continue
                candidate_offsets = (
                    (candidate.temporal_start_offset, candidate.temporal_end_offset)
                    if candidate.temporal_start_offset is not None
                    and candidate.temporal_end_offset is not None
                    else None
                )
                penalties: list[str] = []
                transition_penalty = _same_line_gap_penalty(
                    tokens,
                    token_index,
                    previous_index,
                    previous_end,
                    candidate.start,
                    activity_spans,
                )
                transition_penalty *= same_line_gap_weight
                if candidate.candidate_type.casefold() not in {"melody", "pitch", "basic_pitch"}:
                    # Forced/CTC spans already carry a lexical timing clock;
                    # a phrase-sized gap is not evidence that their model
                    # result is wrong when the candidate itself is fully
                    # supported by a detected vocal island.
                    candidate_is_active = any(
                        previous_end is not None
                        and float(span_start) <= previous_end
                        and candidate.end <= float(span_end)
                        for span_start, span_end in activity_spans
                    )
                    if candidate_is_active:
                        transition_penalty = 0.0
                if transition_penalty:
                    penalties.append(f"same-line gap penalty {transition_penalty:.3f}")
                duration_key = (id(candidate), token.id)
                duration_penalty = duration_penalty_cache.get(duration_key)
                if duration_penalty is None:
                    duration_penalty = _candidate_duration_penalty(
                        candidate,
                        duration_priors[token.id],
                        melody_allowed_duration=melody_allowed_duration,
                    )
                    duration_penalty_cache[duration_key] = duration_penalty
                if duration_penalty:
                    penalties.append(f"duration prior penalty {duration_penalty:.3f}")
                fresh_onset_duration_penalty = fresh_onset_penalty_cache.get(id(candidate))
                if fresh_onset_duration_penalty is None:
                    fresh_onset_duration_penalty = _fresh_onset_duration_penalty(
                        candidate,
                        activity_spans,
                    )
                    # A long candidate that begins at a vocal-island onset is
                    # usually a merged detector span, but an exact melody
                    # handoff into the next lyric token is independent proof
                    # that this is a legitimate held word. Do not penalize
                    # that case before the DP can evaluate the pair.
                    if fresh_onset_duration_penalty:
                        handoff_support = continuation_reward_for(
                            token_index,
                            candidate,
                        )
                        if handoff_support:
                            fresh_onset_duration_penalty *= max(
                                0.0,
                                1.0 - _clamp_confidence(handoff_support),
                            )
                    fresh_onset_duration_penalty *= fresh_onset_penalty_weight
                    fresh_onset_penalty_cache[id(candidate)] = fresh_onset_duration_penalty
                if fresh_onset_duration_penalty:
                    transition_penalty += fresh_onset_duration_penalty
                    penalties.append(
                        f"fresh-onset merged-span penalty {fresh_onset_duration_penalty:.3f}"
                    )
                if candidate.temporal_distance is not None:
                    temporal_penalty = temporal_boundary_penalty(candidate)
                    transition_penalty += temporal_penalty
                    if temporal_penalty:
                        penalties.append(f"boundary distance penalty {temporal_penalty:.3f}")
                if candidate.temporal_start_offset is not None:
                    early_start_penalty = _candidate_early_start_penalty(candidate)
                    transition_penalty += early_start_penalty
                    if early_start_penalty:
                        penalties.append(f"early boundary penalty {early_start_penalty:.3f}")
                if candidate.temporal_end_offset is not None:
                    late_end_penalty = min(
                        0.25,
                        max(0.0, float(candidate.temporal_end_offset) - 0.05) * 0.20,
                    )
                    # A late melody end is normally a weak lexical-clock
                    # match, but it is exactly what a held word needs when
                    # the candidate is long and its end is not cutting
                    # through a strong vocal event.  Apply this relief only
                    # to sustained melody candidates; short candidates keep
                    # the normal clock penalty and cannot exploit it to jump
                    # to a later occurrence.
                    if (
                        melody_late_end_relief > 0.0
                        and candidate.candidate_type.casefold()
                        in {"melody", "pitch", "basic_pitch"}
                        and candidate.end - candidate.start >= 0.65
                        and (
                            candidate.melody_end_inside_confidence is None
                            or _clamp_confidence(candidate.melody_end_inside_confidence)
                            < melody_end_inside_threshold
                        )
                    ):
                        late_end_penalty *= 1.0 - melody_late_end_relief
                    if (
                        candidate.candidate_type.casefold() in {"melody", "pitch", "basic_pitch"}
                        and candidate.activity_score is not None
                    ):
                        late_end_penalty *= 1.0 - _clamp_confidence(candidate.activity_score)
                    transition_penalty += late_end_penalty
                    if late_end_penalty:
                        penalties.append(f"late boundary penalty {late_end_penalty:.3f}")
                if (
                    previous_end is not None
                    and timing_grid_step is not None
                    and timing_grid_step > 0.0
                    and candidate.start >= previous_end - _BOUNDARY_EPSILON
                    and grid_gap_weight > 0.0
                ):
                    gap = max(0.0, candidate.start - previous_end)
                    # Adjacent sung words commonly leave one small beat-grid
                    # subdivision between their canonical spans.  Use this as
                    # a soft prior only for short gaps; phrase-sized silences
                    # must remain unconstrained.
                    if gap <= max(0.45, timing_grid_step * 6.0):
                        gap_distance = abs(gap - timing_grid_step) / timing_grid_step
                        grid_gap_penalty = min(
                            0.35,
                            max(0.0, float(grid_gap_weight)) * gap_distance,
                        )
                        transition_penalty += grid_gap_penalty
                        if grid_gap_penalty:
                            penalties.append(
                                f"timing-grid inter-word gap penalty {grid_gap_penalty:.3f}"
                            )
                activity_gap_penalty = activity_gap_cache.get(id(candidate))
                if activity_gap_penalty is None:
                    activity_gap_penalty = _candidate_internal_activity_gap_penalty(candidate, activity_spans)
                    activity_gap_cache[id(candidate)] = activity_gap_penalty
                if activity_gap_penalty:
                    transition_penalty += activity_gap_penalty
                    penalties.append(f"internal vocal gap penalty {activity_gap_penalty:.3f}")
                if (
                    melody_end_continuation_penalty_weight > 0.0
                    and candidate.candidate_type.casefold()
                    in {"melody", "pitch", "basic_pitch"}
                    and candidate.melody_end_inside_confidence is not None
                ):
                    continuation_penalty = (
                        melody_end_continuation_penalty_weight
                        * _clamp_confidence(candidate.melody_end_inside_confidence)
                    )
                    transition_penalty += continuation_penalty
                    penalties.append(
                        f"melody continuation-end penalty {continuation_penalty:.3f}"
                    )
                if (
                    candidate.candidate_type.casefold() in {"melody", "pitch", "basic_pitch"}
                    and candidate.activity_score is not None
                    and candidate.end - candidate.start > 0.65
                    and candidate.activity_score < 0.45
                ):
                    unsupported_duration_penalty = min(
                        0.30,
                        (0.45 - _clamp_confidence(candidate.activity_score)) * 0.80,
                    )
                    transition_penalty += unsupported_duration_penalty
                    penalties.append(
                        f"unsupported long melody penalty {unsupported_duration_penalty:.3f}"
                    )
                continuation_bonus = 0.0
                if continuation_reward_weight > 0.0:
                    continuation_reward = continuation_reward_cache.get(id(candidate))
                    if continuation_reward is None:
                        continuation_reward = continuation_reward_for(token_index, candidate)
                        continuation_reward_cache[id(candidate)] = continuation_reward
                    if continuation_reward:
                        continuation_bonus = continuation_reward_weight * continuation_reward
                        penalties.append(
                            f"adjacent melody continuation reward {continuation_bonus:.3f}"
                        )
                active_duration_bonus = 0.0
                if (
                    active_duration_reward_weight > 0.0
                    and candidate.candidate_type.casefold()
                    in {"melody", "pitch", "basic_pitch"}
                    and candidate.activity_score is not None
                    and candidate.end - candidate.start > 0.65
                    and _clamp_confidence(candidate.activity_score) >= 0.70
                ):
                    active_duration_bonus = min(
                        0.50,
                        active_duration_reward_weight
                        * (candidate.end - candidate.start - 0.65)
                        * _clamp_confidence(candidate.activity_score),
                    )
                    if active_duration_bonus:
                        penalties.append(
                            f"active melody duration reward {active_duration_bonus:.3f}"
                        )
                successor_gap_penalty = successor_gap_penalty_for(
                    token_index,
                    candidate,
                )
                if successor_gap_penalty:
                    transition_penalty += successor_gap_penalty
                    penalties.append(
                        f"successor onset gap penalty {successor_gap_penalty:.3f}"
                    )
                handoff_gap_bonus = 0.0
                if (
                    handoff_gap_reward_weight > 0.0
                    and previous_end is not None
                    and previous_index is not None
                    and tokens[previous_index].line_index == token.line_index
                ):
                    handoff_gap = candidate.start - previous_end
                    if -melody_overlap_tolerance <= handoff_gap <= 0.35:
                        handoff_gap_bonus = handoff_gap_reward_weight * math.exp(
                            -((handoff_gap - handoff_gap_target) / handoff_gap_scale) ** 2
                        )
                        if handoff_gap_bonus:
                            penalties.append(
                                f"acoustic handoff-gap reward {handoff_gap_bonus:.3f}"
                            )
                if previous_offsets is not None and candidate_offsets is not None:
                    offset_change = max(
                        abs(candidate_offsets[0] - previous_offsets[0]),
                        abs(candidate_offsets[1] - previous_offsets[1]),
                    )
                    if offset_change > 0.12:
                        offset_penalty = min(
                            0.60,
                            (offset_change - 0.12) * temporal_offset_transition_weight,
                        )
                        transition_penalty += offset_penalty
                        penalties.append(f"temporal offset transition penalty {offset_penalty:.3f}")
                evidence_preference = (
                    0.15
                    if candidate.candidate_type.casefold() == "ctc"
                    else 0.0
                )
                adjusted_score = max(
                    0.0,
                    min(
                        _candidate_score_upper_bound(),
                        option.score
                        + evidence_preference
                        + (
                            timing_grid_fit_weight
                            * _timing_grid_fit(
                                candidate,
                                timing_grid_step,
                                timing_grid_offset,
                            )
                        )
                        + (
                            acoustic_continuity_weight
                            * (
                                _clamp_confidence(candidate.acoustic_continuity) - 0.5
                            )
                            if candidate.acoustic_continuity is not None
                            else 0.0
                        )
                        + continuation_bonus
                        + active_duration_bonus
                        + handoff_gap_bonus
                        - transition_penalty
                        - duration_penalty,
                    ),
                )
                adjusted_option = (
                    ScoredCandidate(
                        candidate=option.candidate,
                        score=adjusted_score,
                        reasons=tuple([*option.reasons, *penalties]),
                    )
                    if penalties
                    else option
                )
                candidate_path = (
                    path_score + adjusted_option.score,
                    candidate.end,
                    candidate.start,
                    candidate_offsets or previous_offsets,
                    token_index,
                    [*path, adjusted_option],
                )
                if (
                    os.environ.get("ALIGNMENT_DEBUG_TOKEN") == str(token_index)
                    and abs(candidate.start - 35.2) < 0.02
                    and abs(candidate.end - 36.478) < 0.02
                ):
                    print(
                        "ALIGNMENT_DEBUG",
                        token_index,
                        round(previous_end or -1.0, 3),
                        round(path_score, 3),
                        round(option.score, 3),
                        round(adjusted_score, 3),
                        round(candidate_path[0], 3),
                        flush=True,
                    )
                # Future transitions only depend on the previous end and
                # offset state. Collapse sub-frame endpoint variants so one
                # pitch hypothesis cannot consume the entire beam.
                key = (
                    (round(candidate.end, 2), round(candidate.start, 2))
                    if preserve_endpoint_variants
                    else round(candidate.end, 2)
                )
                existing = next_paths.get(key)
                if existing is None or candidate_path[0] > existing[0]:
                    next_paths[key] = candidate_path
        if len(next_paths) > max_path_states:
            next_paths = dict(
                sorted(
                    next_paths.items(),
                    key=lambda item: (
                        -item[1][0],
                        math.inf if item[1][1] is None else item[1][1],
                        -1 if item[0] is None else item[0],
                    ),
                )[:max_path_states]
            )
        paths = next_paths

    best = max(
        paths.values(),
        key=lambda state: state[0],
        default=(0.0, None, None, None, None, []),
    )
    result: dict[str, ScoredCandidate | None] = {}
    for token, selection in zip(tokens, best[5]):
        result[token.id] = selection if selection is not None and selection.score >= accepted_threshold else None
    return result


def _candidate_activity_score(
    candidate: AlignmentCandidate,
    activity_spans: Sequence[tuple[float, float]],
) -> float:
    """Return the fraction of a candidate span supported by vocal activity."""

    if not _valid_span(candidate.start, candidate.end):
        return 0.0
    assert candidate.start is not None and candidate.end is not None
    duration = max(candidate.end - candidate.start, _BOUNDARY_EPSILON)
    overlap = sum(
        max(0.0, min(candidate.end, span_end) - max(candidate.start, span_start))
        for span_start, span_end in activity_spans
        if math.isfinite(float(span_start))
        and math.isfinite(float(span_end))
        and span_end > span_start
    )
    if overlap <= 0.0:
        # A forced word can begin with an unvoiced consonant immediately
        # before the first voiced frame.  Keep that narrow leading edge
        # instead of rejecting the whole candidate when its end is just
        # outside the activity island; a distant instrumental transient does
        # not receive this rescue.
        nearest_edge = min(
            (
                min(
                    abs(float(span_start) - candidate.end),
                    abs(candidate.start - float(span_end)),
                )
                for span_start, span_end in activity_spans
                if math.isfinite(float(span_start))
                and math.isfinite(float(span_end))
                and span_end > span_start
            ),
            default=math.inf,
        )
        if nearest_edge <= 0.25:
            return _clamp_confidence(0.50 * (1.0 - nearest_edge / 0.25))
    return _clamp_confidence(overlap / duration)


def _candidate_internal_activity_gap_penalty(
    candidate: AlignmentCandidate,
    activity_spans: Sequence[tuple[float, float]],
) -> float:
    """Penalize a candidate that spans a substantial silent vocal gap."""

    if not _valid_span(candidate.start, candidate.end) or not activity_spans:
        return 0.0
    assert candidate.start is not None and candidate.end is not None
    overlapping = [
        (max(candidate.start, float(span_start)), min(candidate.end, float(span_end)))
        for span_start, span_end in activity_spans
        if math.isfinite(float(span_start))
        and math.isfinite(float(span_end))
        and span_end > span_start
        and min(candidate.end, float(span_end)) > max(candidate.start, float(span_start))
    ]
    if len(overlapping) < 2:
        return 0.0
    overlapping.sort()
    largest_gap = max(
        right_start - left_end
        for (_left_start, left_end), (right_start, _right_end)
        in zip(overlapping, overlapping[1:])
    )
    if largest_gap <= 0.12:
        return 0.0
    return min(0.90, (largest_gap - 0.12) * 2.00)


def _fresh_onset_duration_penalty(
    candidate: AlignmentCandidate,
    activity_spans: Sequence[tuple[float, float]],
) -> float:
    """Discourage a long melody span that starts exactly at a new island."""

    if (
        candidate.candidate_type.casefold() not in {"melody", "pitch", "basic_pitch"}
        or not _valid_span(candidate.start, candidate.end)
        or candidate.start is None
        or candidate.end is None
    ):
        return 0.0
    duration = candidate.end - candidate.start
    if duration <= 0.35:
        return 0.0
    if any(
        span_end > span_start
        and abs(candidate.start - float(span_start)) <= 0.12
        for span_start, span_end in activity_spans
    ):
        return min(0.24, (duration - 0.35) * 0.80)
    return 0.0


def _candidate_early_start_penalty(candidate: AlignmentCandidate) -> float:
    """Prefer melody boundaries just after, rather than before, ASR onset."""

    if candidate.temporal_start_offset is None:
        return 0.0
    try:
        penalty_weight = float(
            os.environ.get("ALIGNMENT_EARLY_START_PENALTY_WEIGHT", "0.30")
        )
    except ValueError:
        penalty_weight = 0.30
    if not math.isfinite(penalty_weight) or penalty_weight < 0.0:
        penalty_weight = 0.30
    penalty_weight = min(2.0, penalty_weight)
    return min(
        min(1.0, penalty_weight),
        max(0.0, -float(candidate.temporal_start_offset)) * penalty_weight,
    )


def _prepare_activity_profile(
    activity_times: Sequence[float],
    activity_values: Sequence[float | bool],
) -> tuple[list[float], list[float], float] | None:
    """Build a prefix-sum activity profile for cheap boundary scoring."""

    samples = sorted(
        (
            float(time),
            max(0.0, float(value)),
        )
        for time, value in zip(activity_times, activity_values)
        if math.isfinite(float(time)) and math.isfinite(float(value))
    )
    if len(samples) < 2:
        return None
    times = [sample[0] for sample in samples]
    values = [sample[1] for sample in samples]
    positive = sorted(value for value in values if value > 0.0)
    if not positive:
        return None
    scale = max(0.02, positive[int(0.75 * (len(positive) - 1))])
    prefix = [0.0]
    for value in values:
        prefix.append(prefix[-1] + value)
    return times, prefix, scale


def _activity_mean(
    times: Sequence[float],
    prefix: Sequence[float],
    start: float,
    end: float,
) -> float:
    left = bisect_left(times, start)
    right = bisect_left(times, end)
    if right <= left:
        return 0.0
    return (prefix[right] - prefix[left]) / (right - left)


def _candidate_activity_onset_score(
    candidate: AlignmentCandidate,
    profile: tuple[list[float], list[float], float],
) -> float:
    """Score a candidate's rising vocal activity at its proposed start."""

    times, prefix, scale = profile
    before = _activity_mean(times, prefix, candidate.start - 0.16, candidate.start)
    after = _activity_mean(times, prefix, candidate.start, candidate.start + 0.16)
    return _clamp_confidence((after - before) / scale)


def _clamp_confidence(value: float) -> float:
    numeric = float(value)
    return 0.0 if not math.isfinite(numeric) else max(0.0, min(1.0, numeric))


def _fallback_span(
    index: int,
    tokens: Sequence[LyricToken],
    alignments: Sequence[WordAlignment],
    activity_spans: Sequence[tuple[float, float]],
) -> tuple[float, float] | None:
    def known(position: int) -> bool:
        alignment = alignments[position]
        # An ASR timestamp is not an ``ALIGNED`` canonical result, but it is
        # still a real lexical clock.  Use only that explicit clock as an
        # interpolation boundary.  Synthetic fallback spans must not become
        # new anchors on the next loop iteration, or the first unresolved
        # token would consume the whole gap and all following tokens would be
        # left unmatched.
        if not _alignment_has_span(alignment):
            return False
        if _enum_value(alignment.status) == AlignmentStatus.ALIGNED.value:
            return True
        return any(
            "raw ASR lexical timestamp retained" in reason
            for reason in alignment.evidence.reasons
        )

    def allocate_from_activity(
        spans: Sequence[tuple[float, float]],
        ordinal: int,
        count: int,
        *,
        lower: float = 0.0,
        upper: float = math.inf,
    ) -> tuple[float, float] | None:
        intervals = [
            (max(float(start), lower), min(float(end), upper))
            for start, end in spans
            if float(end) > float(start) and min(float(end), upper) > max(float(start), lower)
        ]
        total_duration = sum(end - start for start, end in intervals)
        if total_duration <= 0.0 or count <= 0:
            return None
        target_start = total_duration * ordinal / count
        target_end = total_duration * (ordinal + 1) / count
        cursor = 0.0
        for start, end in intervals:
            duration = end - start
            segment_start = max(target_start, cursor)
            segment_end = min(target_end, cursor + duration)
            if segment_end > segment_start:
                allocated_start = start + segment_start - cursor
                allocated_end = start + segment_end - cursor
                # A long activity island is not enough evidence to place one
                # unresolved lyric word across an instrumental or noisy gap.
                # Keep the token unmatched rather than exporting a fabricated
                # multi-second span; model-backed sustained words are handled
                # separately and are not subject to this fallback cap.
                if allocated_end - allocated_start > 2.0:
                    return None
                return allocated_start, allocated_end
            cursor += duration
        return None

    previous_index = next(
        (position for position in range(index - 1, -1, -1) if known(position)),
        None,
    )
    following_index = next(
        (position for position in range(index + 1, len(alignments)) if known(position)),
        None,
    )
    previous = alignments[previous_index] if previous_index is not None else None
    following = alignments[following_index] if following_index is not None else None
    if previous and following and previous_index is not None and following_index is not None:
        assert previous.end is not None and following.start is not None
        missing_count = following_index - previous_index - 1
        gap = following.start - previous.end
        # Activity in a long phrase gap is commonly accompaniment bleed or a
        # backing vocal. Do not turn that whole gap into synthetic word
        # timings; only interpolate when the trusted evidence is nearby.
        if gap <= _BOUNDARY_EPSILON or gap > 4.0 or not activity_spans:
            return None
        unresolved = [
            position
            for position in range(previous_index + 1, following_index)
            if not known(position)
        ]
        ordinal = unresolved.index(index) if index in unresolved else 0
        return allocate_from_activity(
            activity_spans,
            ordinal,
            missing_count,
            lower=float(previous.end) + _BOUNDARY_EPSILON,
            upper=float(following.start) - _BOUNDARY_EPSILON,
        )
    if previous:
        assert previous.end is not None
        unresolved = [position for position in range(previous_index + 1, len(alignments)) if not known(position)]  # type: ignore[operator]
        ordinal = unresolved.index(index) if index in unresolved else 0
        return allocate_from_activity(activity_spans, ordinal, len(unresolved), lower=float(previous.end) + 0.02)
    if following:
        assert following.start is not None
        unresolved = [position for position in range(0, following_index) if not known(position)]  # type: ignore[operator]
        ordinal = unresolved.index(index) if index in unresolved else 0
        return allocate_from_activity(activity_spans, ordinal, len(unresolved), upper=float(following.start) - 0.02)
    if activity_spans:
        unresolved = [position for position, alignment in enumerate(alignments) if not known(position)]
        ordinal = unresolved.index(index) if index in unresolved else 0
        return allocate_from_activity(activity_spans, ordinal, len(unresolved))
    return None


def fallback_word_alignments(
    tokens: Sequence[LyricToken],
    alignments: Sequence[WordAlignment],
    *,
    activity_spans: Sequence[tuple[float, float]] = (),
) -> list[WordAlignment]:
    """Fill unresolved tokens only with explicitly estimated spans."""

    result = list(alignments)
    for index, (token, alignment) in enumerate(zip(tokens, result)):
        if _enum_value(alignment.status) == AlignmentStatus.ALIGNED.value or alignment.start is not None:
            continue
        if not token.normalized_text:
            continue
        span = _fallback_span(index, tokens, result, activity_spans)
        if span is None:
            continue
        start, end = span
        result[index] = replace(
            alignment,
            start=start,
            end=end,
            status=AlignmentStatus.ESTIMATED,
            confidence=min(alignment.confidence, 0.35),
            evidence=replace(
                alignment.evidence,
                reasons=[*alignment.evidence.reasons, "estimated between nearby evidence"],
            ),
        )
    return _sanitize_estimated_sequence(result)


def split_repeated_estimated_alignments(
    tokens: Sequence[LyricToken],
    alignments: Sequence[WordAlignment],
    activity_times: Sequence[float],
    activity_values: Sequence[float | bool],
    *,
    minimum_source_duration_s: float = 0.80,
    minimum_gap_s: float = 0.12,
    boundary_margin_s: float = 0.02,
) -> list[WordAlignment]:
    """Split one long forced token when two identical lyric words were sung.

    Full-track forced alignment occasionally collapses adjacent repeated
    words (for example ``ooh ooh``) into one lexical token.  The ordinary
    fallback then allocates the missing word after the forced span, which
    shifts both boundaries.  A pair of strong vocal islands inside the
    forced span is a conservative split cue; no lexical or pitch guess is
    made when that cue is absent.
    """

    if (
        len(tokens) != len(alignments)
        or len(activity_times) == 0
        or len(activity_values) == 0
    ):
        return list(alignments)
    if minimum_source_duration_s < 0.0 or minimum_gap_s < 0.0 or boundary_margin_s < 0.0:
        raise ValueError("repeated-word split thresholds must be non-negative")
    positive_values = [
        max(0.0, float(value))
        for value in activity_values
        if math.isfinite(float(value)) and float(value) > 0.0
    ]
    if not positive_values:
        return list(alignments)
    strong_spans = activity_spans_from_signal(
        activity_times,
        activity_values,
        threshold=median(positive_values),
        clear_gap_s=max(0.08, minimum_gap_s * 0.75),
    )
    if len(strong_spans) < 2:
        return list(alignments)

    result = list(alignments)
    for index in range(len(tokens) - 1):
        left = result[index]
        right = result[index + 1]
        if tokens[index].normalized_text != tokens[index + 1].normalized_text:
            continue
        if (
            not _valid_span(left.start, left.end)
            or left.start is None
            or left.end is None
            or left.end - left.start < minimum_source_duration_s
            or _enum_value(right.status) != AlignmentStatus.ESTIMATED.value
            or not _valid_span(right.start, right.end)
            or right.start is None
            or right.end is None
            or left.evidence.selected_model == ""
            or "estimated between nearby evidence" not in right.evidence.reasons
        ):
            continue
        inner = [
            span
            for span in strong_spans
            if span[1] > left.start + boundary_margin_s
            and span[0] < left.end - boundary_margin_s
        ]
        split_pair = max(
            (
                (next_start - current_end, current_end)
                for current, following in zip(inner, inner[1:])
                for current_end, next_start in [(current[1], following[0])]
                if next_start - current_end >= minimum_gap_s
            ),
            default=None,
        )
        if split_pair is None:
            continue
        _gap, split = split_pair
        split = min(left.end - boundary_margin_s, max(left.start + boundary_margin_s, split - boundary_margin_s))
        if split <= left.start + boundary_margin_s or split >= left.end - boundary_margin_s:
            continue
        # Preserve the original forced span as the total repeated phrase.
        # If the following trusted word starts soon after it, leave a small
        # consonant gap rather than overlapping the next word.
        repeated_end = float(left.end)
        following = result[index + 2] if index + 2 < len(result) else None
        if following is not None and following.start is not None and following.start > split:
            following_gap = max(
                0.45,
                min(0.90, (float(left.end) - float(left.start)) * 0.28),
            )
            repeated_end = min(
                max(repeated_end, float(following.start) - following_gap),
                float(following.start) - boundary_margin_s,
            )
        if repeated_end <= split + boundary_margin_s:
            continue
        left_evidence = replace(
            left.evidence,
            reasons=[*left.evidence.reasons, "split repeated forced token at a vocal handoff"],
        )
        right_evidence = replace(
            right.evidence,
            selected_model=left.evidence.selected_model,
            selected_audio=left.evidence.selected_audio,
            reasons=[*right.evidence.reasons, "recovered repeated word from a split forced token"],
        )
        result[index] = replace(left, end=split, evidence=left_evidence)
        result[index + 1] = replace(
            right,
            start=split,
            end=repeated_end,
            status=AlignmentStatus.ALIGNED,
            confidence=max(0.55, left.confidence),
            evidence=right_evidence,
        )
    validate_alignment_sequence(result)
    return result


def repair_delayed_forced_onsets(
    tokens: Sequence[LyricToken],
    alignments: Sequence[WordAlignment],
    activity_spans: Sequence[tuple[float, float]],
    *,
    minimum_gap_s: float = 0.35,
    maximum_fragment_duration_s: float = 0.80,
) -> list[WordAlignment]:
    """Move a short forced fragment to the active phrase boundary.

    A forced pass can attach a short word to its final consonant after a
    preceding word, leaving a large timestamp gap even though separated-vocal
    activity is continuous.  Only same-line adjacent forced spans are
    repaired, and only when one activity island covers the entire handoff.
    """

    if len(tokens) != len(alignments) or not activity_spans:
        return list(alignments)
    if minimum_gap_s < 0.0 or maximum_fragment_duration_s < 0.0:
        raise ValueError("forced onset thresholds must be non-negative")
    result = list(alignments)
    for index in range(1, len(result)):
        previous = result[index - 1]
        current = result[index]
        if (
            tokens[index - 1].line_index != tokens[index].line_index
            or not _valid_span(previous.start, previous.end)
            or not _valid_span(current.start, current.end)
            or previous.end is None
            or current.start is None
            or current.end is None
            or current.start - previous.end < minimum_gap_s
            or current.end - current.start > maximum_fragment_duration_s
            or _enum_value(previous.status) != AlignmentStatus.ALIGNED.value
            or _enum_value(current.status) != AlignmentStatus.ALIGNED.value
            or not previous.evidence.selected_model
            or not current.evidence.selected_model
        ):
            continue
        if not any(
            float(span_start) <= float(previous.end) + 0.08
            and float(span_end) >= float(current.end) - 0.08
            and float(span_end) > float(previous.end)
            for span_start, span_end in activity_spans
            if span_end > span_start
        ):
            continue
        # Preserve a short consonant handoff instead of placing the fragment
        # exactly on the previous end.  The forced span is often delayed by
        # several frames, while real neighbouring words still have a small
        # (roughly 40–120 ms) transition.  A bounded fraction of the observed
        # gap recovers that handoff without eating a genuine phrase pause.
        observed_gap = float(current.start - previous.end)
        new_start = float(previous.end) + min(0.10, max(0.03, observed_gap * 0.05))
        if new_start >= current.end - 0.04:
            continue
        result[index] = replace(
            current,
            start=new_start,
            evidence=replace(
                current.evidence,
                reasons=[
                    *current.evidence.reasons,
                    "moved delayed forced onset to a continuous vocal handoff",
                ],
            ),
        )
    validate_alignment_sequence(result)
    return result


def build_word_alignments(
    tokens: Sequence[LyricToken],
    asr_tokens: Sequence[ASRToken],
    alignment_run_id: str,
    *,
    candidates: Iterable[AlignmentCandidate] = (),
    activity_spans: Sequence[tuple[float, float]] = (),
    accepted_threshold: float = 0.55,
    timing_grid_step: float | None = None,
    timing_grid_offset: float | None = None,
    activity_times: Sequence[float] = (),
    activity_values: Sequence[float | bool] = (),
    max_local_candidates: int = 15,
    skip_penalty: float = 0.08,
    grid_gap_weight: float = 0.0,
    minimum_interword_gap: float = 0.0,
    melody_allowed_duration: float = 0.55,
) -> tuple[list[WordAlignment], GlobalAlignment]:
    """Build canonical alignments from ASR anchors and optional model candidates."""

    global_alignment = global_monotonic_match(tokens, asr_tokens)
    prepared_asr_tokens = global_alignment.prepared_asr_tokens or tuple(asr_tokens)
    asr_anchor_by_lyric = {
        match.lyric_index: prepared_asr_tokens[match.asr_index]
        for match in global_alignment.matches
        if match.matched
        and match.asr_index is not None
        and match.asr_index < len(prepared_asr_tokens)
    }
    candidate_map: dict[str, list[AlignmentCandidate]] = {}
    for candidate in candidates:
        if candidate.candidate_type.casefold() in _ANCHOR_CANDIDATE_TYPES:
            continue
        candidate_map.setdefault(candidate.token_id, []).append(candidate)

    selected = select_best_candidates(
        tokens,
        candidate_map,
        activity_spans=activity_spans,
        accepted_threshold=accepted_threshold,
        timing_grid_step=timing_grid_step,
        timing_grid_offset=timing_grid_offset,
        activity_times=activity_times,
        activity_values=activity_values,
        max_local_candidates=max_local_candidates,
        skip_penalty=skip_penalty,
        grid_gap_weight=grid_gap_weight,
        minimum_interword_gap=minimum_interword_gap,
        melody_allowed_duration=melody_allowed_duration,
    )
    alignments: list[WordAlignment] = []
    for token_index, token in enumerate(tokens):
        choice = selected.get(token.id)
        if choice is None:
            # Preserve a lexical ASR clock as an estimated span when no CTC,
            # forced, or melodic candidate exists.  Previously this path
            # discarded every anchor and let the fallback allocator spread
            # all unresolved words over the full activity signal, which can
            # shift an entire song to an instrumental intro.
            anchor = asr_anchor_by_lyric.get(token_index)
            if anchor is not None and _valid_span(anchor.start, anchor.end):
                alignments.append(
                    WordAlignment(
                        token_id=token.id,
                        alignment_run_id=alignment_run_id,
                        start=float(anchor.start),
                        end=float(anchor.end),
                        status=AlignmentStatus.ESTIMATED,
                        confidence=min(0.35, _clamp_confidence(anchor.confidence)),
                        evidence=AlignmentEvidence(
                            selected_model=anchor.model,
                            selected_audio=anchor.audio_source,
                            reasons=[
                                "raw ASR lexical timestamp retained as an estimate",
                                "no forced or melodic boundary evidence was available",
                            ],
                        ),
                    )
                )
                continue
            alignments.append(
                WordAlignment(
                    token_id=token.id,
                    alignment_run_id=alignment_run_id,
                    start=None,
                    end=None,
                    status=AlignmentStatus.UNMATCHED,
                    confidence=0.0,
                    evidence=AlignmentEvidence(reasons=["no reliable candidate found"]),
                )
            )
            continue
        candidate = choice.candidate
        candidate = _grid_snapped_candidate(
            candidate,
            timing_grid_step,
            timing_grid_offset,
        ) or candidate
        previous_selected_index = next(
            (
                index
                for index in range(len(alignments) - 1, -1, -1)
                if _alignment_has_span(alignments[index])
            ),
            None,
        )
        previous_selected = (
            alignments[previous_selected_index]
            if previous_selected_index is not None
            else None
        )
        next_selected = next(
            (
                selected.get(next_token.id)
                for next_token in tokens[token_index + 1 :]
                if selected.get(next_token.id) is not None
            ),
            None,
        )
        if previous_selected is not None:
            if previous_selected.end is not None and candidate.start < previous_selected.end - _BOUNDARY_EPSILON:
                # A lexical ASR estimate may have been retained for a token
                # that the trusted forced path can now place earlier (this is
                # common at an overlapping chunk boundary).  Remove only the
                # trailing estimated spans that collide; never rewrite a
                # trusted model span to make the sequence look monotonic.
                if (
                    candidate.candidate_type.casefold() not in {"anchor", "asr", "transcript"}
                    and previous_selected.status == AlignmentStatus.ESTIMATED
                ):
                    while alignments:
                        trailing = alignments[-1]
                        if (
                            trailing.status != AlignmentStatus.ESTIMATED
                            or trailing.start is None
                            or trailing.end is None
                            or trailing.end <= candidate.start + _BOUNDARY_EPSILON
                        ):
                            break
                        alignments[-1] = replace(
                            trailing,
                            start=None,
                            end=None,
                            status=AlignmentStatus.UNMATCHED,
                            confidence=0.0,
                            evidence=replace(
                                trailing.evidence,
                                reasons=[
                                    *trailing.evidence.reasons,
                                    "estimated ASR span discarded for trusted forced timing",
                                ],
                            ),
                    )
                    previous_selected = None
                if previous_selected is None:
                    pass
                else:
                    overlap = previous_selected.end - candidate.start
                if previous_selected is None:
                    overlap = 0.0
                previous_model = (
                    previous_selected.evidence.selected_model.casefold().replace("_", "-")
                    if previous_selected is not None
                    else ""
                )
                current_type = candidate.candidate_type.casefold()
                try:
                    overlap_tolerance = float(
                        os.environ.get("ALIGNMENT_MELODY_OVERLAP_TOLERANCE", "0.0")
                    )
                except ValueError:
                    overlap_tolerance = 0.0
                if (
                    previous_selected is not None
                    and
                    previous_selected_index is not None
                    and not previous_selected.end_locked
                    and current_type in {"melody", "pitch", "basic_pitch"}
                    and previous_model in {"basic-pitch", "basicpitch", "melody", "pitch"}
                    and math.isfinite(overlap_tolerance)
                    and 0.0 < overlap <= min(0.25, max(0.0, overlap_tolerance))
                    and previous_selected.start is not None
                ):
                    trimmed_end = max(
                        previous_selected.start + _BOUNDARY_EPSILON,
                        candidate.start - _BOUNDARY_EPSILON,
                    )
                    alignments[previous_selected_index] = replace(
                        previous_selected,
                        end=trimmed_end,
                        evidence=replace(
                            previous_selected.evidence,
                            reasons=[
                                *previous_selected.evidence.reasons,
                                "trimmed detector tail before the next melody onset",
                            ],
                        ),
                    )
                else:
                    if previous_selected is not None:
                        candidate = None
        if candidate is not None and next_selected is not None:
            if candidate.end > next_selected.candidate.start + _BOUNDARY_EPSILON:
                candidate = None
        if candidate is None:
            alignments.append(
                WordAlignment(
                    token_id=token.id,
                    alignment_run_id=alignment_run_id,
                    start=None,
                    end=None,
                    status=AlignmentStatus.UNMATCHED,
                    confidence=0.0,
                    evidence=AlignmentEvidence(
                        reasons=[
                            "selected timing candidate conflicts with neighboring source-ordered evidence",
                        ],
                    ),
                )
            )
            continue
        confidence = _clamp_confidence(min(choice.score, candidate.text_score, candidate.model_confidence))
        is_anchor_only = candidate.candidate_type.casefold() in {"anchor", "asr", "transcript"}
        if is_anchor_only:
            confidence = min(confidence, 0.35)
        evidence_reasons = list(choice.reasons)
        if is_anchor_only:
            evidence_reasons.append("ASR timestamp is an anchor, not forced-alignment evidence")
        alignments.append(
            WordAlignment(
                token_id=token.id,
                alignment_run_id=alignment_run_id,
                start=float(candidate.start),
                end=float(candidate.end),
                status=AlignmentStatus.ESTIMATED if is_anchor_only else AlignmentStatus.ALIGNED,
                confidence=confidence,
                evidence=AlignmentEvidence(
                    selected_model=candidate.model,
                    selected_audio=candidate.audio_source,
                    reasons=list(dict.fromkeys(evidence_reasons)),
                    candidate_scores=[{
                        "model": candidate.model,
                        "audio_source": candidate.audio_source,
                        "candidate_type": candidate.candidate_type,
                        "score": choice.score,
                    }],
                ),
            )
        )
    return fallback_word_alignments(tokens, alignments, activity_spans=activity_spans), global_alignment


def trim_selected_melody_gaps(
    alignments: Sequence[WordAlignment],
    melody_outputs: Mapping[str, Sequence[ASRToken]],
    *,
    gap_threshold_s: float = 0.15,
    extension_s: float = 0.08,
) -> list[WordAlignment]:
    """Trim a melody-selected word at a substantial detector silence.

    Melody detectors return note events, not word boundaries. A boundary pair
    selected from that lattice can therefore span the silence between two
    adjacent words. Keep the selected onset and trim only the end when the
    detector has a clear internal gap. This is intentionally post-selection:
    the full candidate lattice remains available to the global path, while a
    selected span cannot claim an acoustically empty interval.
    """

    if gap_threshold_s <= 0.0 or extension_s < 0.0:
        raise ValueError("melody gap thresholds must be non-negative")

    result: list[WordAlignment] = []
    for alignment in alignments:
        model = alignment.evidence.selected_model.casefold().replace("_", "-")
        if (
            model not in {"basic-pitch", "basicpitch", "melody", "pitch"}
            or alignment.end_locked
            or not _valid_span(alignment.start, alignment.end)
        ):
            result.append(alignment)
            continue
        assert alignment.start is not None and alignment.end is not None
        source = alignment.evidence.selected_audio.casefold()
        if source not in {"vocal_stem", "original_mix"}:
            result.append(alignment)
            continue

        trimmed_end = alignment.end
        # The vocal stem is a useful boundary check for a mix-selected span:
        # accompaniment can keep the original-mix detector active through a
        # real vocal word boundary. Do not use the noisier mix to trim a
        # vocal-stem selection in the opposite direction.
        gap_sources = [source]
        if source == "original_mix":
            gap_sources.append("vocal_stem")
        for gap_source in gap_sources:
            events = sorted(
                (
                    event
                    for key, values in melody_outputs.items()
                    if key.rsplit(":", 1)[-1].casefold() == gap_source
                    for event in values
                    if event.evidence_type.casefold() in {"melody", "pitch", "basic_pitch"}
                    and _valid_span(event.start, event.end)
                    and event.end > alignment.start
                    and event.start < alignment.end
                ),
                key=lambda event: (event.start, event.end),
            )
            for left, right in zip(events, events[1:]):
                gap = right.start - left.end
                if gap < gap_threshold_s:
                    continue
                cut = left.end + extension_s
                if (
                    cut <= alignment.start + _BOUNDARY_EPSILON
                    or cut - alignment.start < 0.12
                    or cut >= trimmed_end
                ):
                    continue
                trimmed_end = cut
                break
        if trimmed_end == alignment.end:
            result.append(alignment)
            continue
        result.append(
            replace(
                alignment,
                end=trimmed_end,
                evidence=replace(
                    alignment.evidence,
                    reasons=[
                        *alignment.evidence.reasons,
                        "trimmed at a substantial melody-detector gap",
                    ],
                ),
            )
        )
    validate_alignment_sequence(result)
    return result


def trim_alignment_tails_to_activity(
    alignments: Sequence[WordAlignment],
    activity_spans: Sequence[tuple[float, float]],
    *,
    gap_threshold_s: float = 0.65,
    extension_s: float = 0.04,
    minimum_tail_s: float = 0.18,
) -> list[WordAlignment]:
    """Trim a model tail that runs past an isolated vocal-activity island.

    Forced alignment can keep a held word alive through several seconds of
    silence, especially when the next lyric phrase is far away.  Activity is
    only used as a conservative end bound here: if the selected span overlaps
    one activity island and ends before the next island begins, trim the tail
    to that island plus a small frame allowance.  Continuous activity is left
    untouched so backing vocals and legato phrases are not shortened.
    """

    if gap_threshold_s < 0.0 or extension_s < 0.0 or minimum_tail_s < 0.0:
        raise ValueError("activity tail thresholds must be non-negative")
    spans = sorted(
        (float(start), float(end))
        for start, end in activity_spans
        if _valid_span(start, end)
    )
    if not spans:
        return list(alignments)

    result: list[WordAlignment] = []
    for alignment in alignments:
        if (
            alignment.end_locked
            or not _valid_span(alignment.start, alignment.end)
            or alignment.start is None
            or alignment.end is None
        ):
            result.append(alignment)
            continue
        containing_index = next(
            (
                index
                for index, (span_start, span_end) in enumerate(spans)
                if span_start - _BOUNDARY_EPSILON <= alignment.start < span_end
            ),
            None,
        )
        if containing_index is None or containing_index + 1 >= len(spans):
            result.append(alignment)
            continue
        _span_start, span_end = spans[containing_index]
        next_start, _next_end = spans[containing_index + 1]
        if (
            alignment.end <= span_end + minimum_tail_s
            or next_start <= alignment.end + _BOUNDARY_EPSILON
            or next_start - span_end < gap_threshold_s
        ):
            result.append(alignment)
            continue
        trimmed_end = min(alignment.end, span_end + extension_s)
        if trimmed_end <= alignment.start + _BOUNDARY_EPSILON:
            result.append(alignment)
            continue
        result.append(
            replace(
                alignment,
                end=trimmed_end,
                evidence=replace(
                    alignment.evidence,
                    reasons=[
                        *alignment.evidence.reasons,
                        "trimmed model tail at an isolated vocal-activity boundary",
                    ],
                ),
            )
        )
    validate_alignment_sequence(result)
    return result


def snap_alignment_starts_to_activity_onsets(
    alignments: Sequence[WordAlignment],
    activity_times: Sequence[float],
    activity_values: Sequence[float | bool],
    *,
    minimum_delay_s: float = 0.10,
    maximum_shift_s: float = 1.25,
    minimum_rise: float = 0.35,
    onset_gap_s: float = 0.45,
) -> list[WordAlignment]:
    """Move an early model onset to a clear rising vocal onset.

    Word-level forced aligners can attach a repeated or held word to the
    preceding low-level vocal energy.  Use the raw separated-vocal activity
    profile only when the candidate starts in a quiet window and a strong
    rise occurs shortly after it.  Interior dynamics of an already-active
    word do not qualify, which keeps this from retiming ordinary syllables.
    """

    if (
        minimum_delay_s < 0.0
        or maximum_shift_s < minimum_delay_s
        or minimum_rise < 0.0
        or onset_gap_s < 0.0
    ):
        raise ValueError("activity onset thresholds must be non-negative")
    profile = _prepare_activity_profile(activity_times, activity_values)
    if profile is None:
        return list(alignments)
    times, prefix, scale = profile
    positive = sorted(
        max(0.0, float(value))
        for value in activity_values
        if math.isfinite(float(value)) and float(value) > 0.0
    )
    if not positive:
        return list(alignments)
    strong_threshold = median(positive)
    strong_spans = activity_spans_from_signal(
        activity_times,
        activity_values,
        threshold=strong_threshold,
    )
    result: list[WordAlignment] = []
    previous_end: float | None = None
    for alignment in alignments:
        if (
            alignment.start_locked
            or not _valid_span(alignment.start, alignment.end)
            or alignment.start is None
            or alignment.end is None
        ):
            result.append(alignment)
            if alignment.end is not None:
                previous_end = max(previous_end or alignment.end, alignment.end)
            continue
        search_start = alignment.start + minimum_delay_s
        search_end = min(alignment.end - 0.05, alignment.start + maximum_shift_s)
        if search_end <= search_start:
            result.append(alignment)
            previous_end = max(previous_end or alignment.end, alignment.end)
            continue
        best_time: float | None = None
        best_rise = 0.0
        for index, (onset, _strong_end) in enumerate(strong_spans):
            if onset < search_start or onset > search_end:
                continue
            if index > 0 and onset - strong_spans[index - 1][1] < onset_gap_s:
                continue
            before = _activity_mean(times, prefix, onset - 0.16, onset)
            after = _activity_mean(times, prefix, onset, onset + 0.16)
            rise = (after - before) / scale
            if rise >= minimum_rise and rise > best_rise:
                best_time = onset
                best_rise = rise
        if best_time is None or (
            previous_end is not None
            and best_time <= previous_end + _BOUNDARY_EPSILON
        ):
            result.append(alignment)
            previous_end = max(previous_end or alignment.end, alignment.end)
            continue
        result.append(
            replace(
                alignment,
                start=best_time,
                evidence=replace(
                    alignment.evidence,
                    reasons=[
                        *alignment.evidence.reasons,
                        f"onset refined from separated-vocal activity rise ({best_rise:.2f})",
                    ],
                ),
            )
        )
        previous_end = max(previous_end or alignment.end, alignment.end)
    validate_alignment_sequence(result)
    return result


def trim_alignment_tails_to_strong_activity(
    alignments: Sequence[WordAlignment],
    activity_times: Sequence[float],
    activity_values: Sequence[float | bool],
    *,
    minimum_gap_s: float = 0.45,
    minimum_tail_s: float = 0.18,
    extension_s: float = 0.04,
    minimum_source_duration_s: float = 2.0,
) -> list[WordAlignment]:
    """Trim a long span at a clear high-energy handoff.

    The ordinary activity threshold intentionally keeps quiet vocals for
    fallback.  Its broad islands cannot always split two words inside one
    phrase, so this second pass uses the median voiced level as a conservative
    handoff signal. It only trims when a second strong island starts inside
    the selected span after a real gap; continuous high activity is untouched.
    """

    if (
        minimum_gap_s < 0.0
        or minimum_tail_s < 0.0
        or extension_s < 0.0
        or minimum_source_duration_s < 0.0
    ):
        raise ValueError("strong activity thresholds must be non-negative")
    values = [
        max(0.0, float(value))
        for value in activity_values
        if math.isfinite(float(value)) and float(value) > 0.0
    ]
    if not values:
        return list(alignments)
    strong_spans = activity_spans_from_signal(
        activity_times,
        activity_values,
        threshold=median(values),
    )
    if len(strong_spans) < 2:
        return list(alignments)

    result: list[WordAlignment] = []
    for alignment_index, alignment in enumerate(alignments):
        if (
            alignment_index == len(alignments) - 1
            or alignment.end_locked
            or not _valid_span(alignment.start, alignment.end)
            or alignment.start is None
            or alignment.end is None
            or alignment.end - alignment.start < minimum_source_duration_s
        ):
            result.append(alignment)
            continue
        current_index = next(
            (
                index
                for index, (start, end) in enumerate(strong_spans)
                if start - 0.08 <= alignment.start < end
            ),
            None,
        )
        if current_index is None or current_index + 1 >= len(strong_spans):
            result.append(alignment)
            continue
        _current_start, current_end = strong_spans[current_index]
        next_start, _next_end = strong_spans[current_index + 1]
        if (
            alignment.end <= current_end + minimum_tail_s
            or next_start > alignment.end + _BOUNDARY_EPSILON
            or next_start - current_end < minimum_gap_s
        ):
            result.append(alignment)
            continue
        if alignment.end - alignment.start >= max(2.0, minimum_source_duration_s):
            # Long forced spans often include the next strong island's
            # onset.  For those merged phrases, use the handoff itself as
            # the boundary; short spans retain the conservative tail trim.
            trimmed_end = min(alignment.end, next_start - _BOUNDARY_EPSILON)
        else:
            trimmed_end = min(alignment.end, current_end + extension_s)
        if trimmed_end <= alignment.start + _BOUNDARY_EPSILON:
            result.append(alignment)
            continue
        result.append(
            replace(
                alignment,
                end=trimmed_end,
                evidence=replace(
                    alignment.evidence,
                    reasons=[
                        *alignment.evidence.reasons,
                        "trimmed model tail at a strong vocal handoff",
                    ],
                ),
            )
        )
    validate_alignment_sequence(result)
    return result


def trim_forced_boundary_latency(
    alignments: Sequence[WordAlignment],
    *,
    maximum_shift_s: float = 0.18,
    base_shift_s: float = 0.04,
    duration_shift: float = 0.04,
    activity_spans: Sequence[tuple[float, float]] = (),
) -> list[WordAlignment]:
    """Compensate the small trailing-frame bias of the wav2vec forced path.

    ForceAlign's word spans end on the last accepted emission frame.  On
    separated singing that frame commonly includes a short consonant/noise
    tail, with the bias growing mildly for held words.  Trim only unlocked
    ForceAlign spans and never below their own onset; CTC/melody/manual spans
    retain their native boundaries.
    """

    if maximum_shift_s < 0.0 or base_shift_s < 0.0 or duration_shift < 0.0:
        raise ValueError("forced boundary latency thresholds must be non-negative")
    result: list[WordAlignment] = []
    for alignment in alignments:
        model = alignment.evidence.selected_model.casefold().replace("_", "-")
        if (
            model not in {"forcealign-wav2vec2", "forcealign", "wav2vec2"}
            or alignment.end_locked
            or not _valid_span(alignment.start, alignment.end)
            or alignment.start is None
            or alignment.end is None
            # A repeated lyric token recovered by ``split_repeated...`` is
            # no longer the raw forced span: its end is the vocal handoff
            # reconstructed from activity and the neighbouring token.  The
            # forced model's trailing-frame correction would trim that
            # recovered boundary a second time and systematically make the
            # tail of held/split words early.
            or any(
                marker in alignment.evidence.reasons
                for marker in (
                    "recovered repeated word from a split forced token",
                    "estimated between nearby evidence",
                )
            )
        ):
            result.append(alignment)
            continue
        # Activity already bounds a span whose end is inside the vocal
        # island.  In that case the forced endpoint is a real sung handoff,
        # not the trailing frame noise this compensation targets; trimming it
        # again makes held chart notes systematically short.
        duration = float(alignment.end - alignment.start)
        if duration >= 0.35 and any(
            float(span_start) - _BOUNDARY_EPSILON <= float(alignment.end)
            <= float(span_end) + 0.04
            and float(span_end) - float(alignment.end) <= 0.06
            for span_start, span_end in activity_spans
            if _valid_span(span_start, span_end)
        ):
            result.append(alignment)
            continue
        shift = min(maximum_shift_s, base_shift_s + duration_shift * duration)
        trimmed_end = max(float(alignment.start) + _BOUNDARY_EPSILON, float(alignment.end) - shift)
        if trimmed_end >= alignment.end - _BOUNDARY_EPSILON:
            result.append(alignment)
            continue
        result.append(
            replace(
                alignment,
                end=trimmed_end,
                evidence=replace(
                    alignment.evidence,
                    reasons=[
                        *alignment.evidence.reasons,
                        f"trimmed forced trailing-frame latency ({shift:.3f}s)",
                    ],
                ),
            )
        )
    validate_alignment_sequence(result)
    return result


def repair_melody_handoffs(
    alignments: Sequence[WordAlignment],
    melody_outputs: Mapping[str, Sequence[ASRToken]],
    *,
    tokens: Sequence[LyricToken] = (),
    anchor_outputs: Mapping[str, Sequence[ASRToken]] | None = None,
    min_run_duration_s: float = 0.50,
    min_run_confidence: float = 0.40,
    max_selected_gap_s: float = 0.14,
    max_following_gap_s: float = 0.16,
    max_run_merge_gap_s: float = 0.04,
    current_tail_margin_s: float = 0.18,
    successor_tail_margin_s: float = 0.08,
    successor_offset_s: float = 0.05,
    end_extension_s: float = 0.03,
    successor_anchor_min_duration_s: float = 0.80,
    anchor_duration_factor: float = 2.0,
) -> list[WordAlignment]:
    """Repair a narrow, acoustically supported held-word handoff.

    A melody detector commonly splits one sustained vocal note into several
    same-pitch events. The ordinary candidate scorer can then end the current
    word at an internal event boundary and start the next word on the same
    note. This helper repairs only that recognizable local shape:

    * both selected words are trusted, adjacent, and unlocked;
    * their selected boundary pair lies inside one long, confident
      same-pitch vocal run;
    * a different-pitch run begins immediately after it; and
    * independent ASR anchors do not describe an unusually short successor.

    It intentionally does not stretch arbitrary words to the next activity
    frame. If the detector does not expose a complete handoff, the original
    canonical spans remain unchanged.
    """

    if not alignments:
        return []
    numeric_parameters = (
        min_run_duration_s,
        min_run_confidence,
        max_selected_gap_s,
        max_following_gap_s,
        max_run_merge_gap_s,
        current_tail_margin_s,
        successor_tail_margin_s,
        successor_offset_s,
        end_extension_s,
        successor_anchor_min_duration_s,
        anchor_duration_factor,
    )
    if any(not math.isfinite(float(value)) for value in numeric_parameters):
        raise ValueError("melody handoff thresholds must be finite")
    if (
        min_run_duration_s < 0.0
        or not 0.0 <= min_run_confidence <= 1.0
        or max_selected_gap_s < 0.0
        or max_following_gap_s < 0.0
        or max_run_merge_gap_s < 0.0
        or current_tail_margin_s < 0.0
        or successor_tail_margin_s < 0.0
        or successor_offset_s < 0.0
        or end_extension_s < 0.0
        or successor_anchor_min_duration_s < 0.0
        or anchor_duration_factor < 0.0
    ):
        raise ValueError("melody handoff thresholds must be non-negative")

    def physical_source(source: str, event: ASRToken) -> str:
        key_source = source.rsplit(":", 1)[-1].casefold()
        if key_source in {"vocal_stem", "original_mix"}:
            return key_source
        return event.audio_source.casefold() or key_source

    vocal_events = [
        event
        for source, events in melody_outputs.items()
        for event in events
        if physical_source(source, event) == "vocal_stem"
        and event.evidence_type.casefold() in {"melody", "pitch", "basic_pitch"}
        and event.midi_note is not None
        and _valid_span(event.start, event.end)
    ]
    if not vocal_events:
        return list(alignments)

    runs: list[tuple[float, float, float, int]] = []
    by_note: dict[int, list[ASRToken]] = {}
    for event in vocal_events:
        by_note.setdefault(int(event.midi_note), []).append(event)
    for note, events in by_note.items():
        events.sort(key=lambda event: (float(event.start), float(event.end)))
        first = events[0]
        run_start = float(first.start)
        run_end = float(first.end)
        run_confidence = _clamp_confidence(first.confidence)
        for event in events[1:]:
            event_start = float(event.start)
            event_end = float(event.end)
            if event_start <= run_end + max_run_merge_gap_s:
                run_end = max(run_end, event_end)
                run_confidence = max(run_confidence, _clamp_confidence(event.confidence))
                continue
            runs.append((run_start, run_end, run_confidence, note))
            run_start = event_start
            run_end = event_end
            run_confidence = _clamp_confidence(event.confidence)
        runs.append((run_start, run_end, run_confidence, note))
    runs.sort(key=lambda item: (item[0], item[1], item[3]))

    anchor_durations: dict[int, list[float]] = {}
    if tokens and anchor_outputs:
        for source in ("vocal_stem", "original_mix"):
            words = [
                event
                for key, events in anchor_outputs.items()
                if key.casefold() == source
                for event in events
                if event.evidence_type.casefold() not in {"melody", "pitch", "basic_pitch"}
                and _valid_span(event.start, event.end)
            ]
            if not words:
                continue
            matches = global_monotonic_match(tokens, words)
            prepared_words = matches.prepared_asr_tokens or tuple(words)
            for match in matches.matches:
                if (
                    match.asr_index is None
                    or not match.matched
                    or match.asr_index >= len(prepared_words)
                ):
                    continue
                duration = (
                    float(prepared_words[match.asr_index].end)
                    - float(prepared_words[match.asr_index].start)
                )
                if duration > 0.0:
                    anchor_durations.setdefault(match.lyric_index, []).append(duration)

    def is_aligned(alignment: WordAlignment) -> bool:
        return (
            _enum_value(alignment.status) == AlignmentStatus.ALIGNED.value
            and _valid_span(alignment.start, alignment.end)
        )

    result = list(alignments)
    for index, (current, successor) in enumerate(zip(alignments, alignments[1:])):
        if (
            not is_aligned(current)
            or not is_aligned(successor)
            or current.start_locked
            or current.end_locked
            or successor.start_locked
            or successor.end_locked
        ):
            continue
        assert current.start is not None and current.end is not None
        assert successor.start is not None and successor.end is not None
        selected_gap = successor.start - current.end
        if selected_gap < -_BOUNDARY_EPSILON or selected_gap > max_selected_gap_s:
            continue

        for run_index, (run_start, run_end, confidence, note) in enumerate(runs):
            if run_end - run_start < min_run_duration_s or confidence < min_run_confidence:
                continue
            durations = anchor_durations.get(index, ())
            successor_durations = anchor_durations.get(index + 1, ())
            if durations and run_end - run_start <= min(durations) * anchor_duration_factor:
                continue
            if successor_durations and min(successor_durations) < successor_anchor_min_duration_s:
                continue
            if not (
                run_start - 0.03 <= current.end < run_end - current_tail_margin_s
                and run_start - 0.03 <= successor.start < run_end - successor_tail_margin_s
            ):
                continue

            following_run = next(
                (
                    candidate
                    for candidate in runs[run_index + 1 :]
                    if candidate[3] != note
                    and candidate[0] >= run_end - 0.04
                    and candidate[0] - run_end <= max_following_gap_s
                ),
                None,
            )
            if following_run is None:
                continue
            successor_start = following_run[0] + successor_offset_s
            current_end = min(run_end + end_extension_s, successor_start - 0.03)
            if (
                current_end <= current.start + 0.02
                or current_end >= successor.end - 0.02
                or successor_start >= successor.end - 0.02
            ):
                continue
            previous = result[index - 1] if index else None
            if previous is not None and _valid_span(previous.start, previous.end):
                assert previous.end is not None
                if current.start < previous.end - _BOUNDARY_EPSILON:
                    continue
            next_after_successor = result[index + 2] if index + 2 < len(result) else None
            if next_after_successor is not None and _valid_span(next_after_successor.start, next_after_successor.end):
                assert next_after_successor.start is not None
                if successor.end > next_after_successor.start + _BOUNDARY_EPSILON:
                    continue

            result[index] = replace(
                current,
                end=current_end,
                evidence=replace(
                    current.evidence,
                    reasons=[
                        *current.evidence.reasons,
                        "extended held word to a confident melody handoff",
                    ],
                ),
            )
            result[index + 1] = replace(
                successor,
                start=successor_start,
                evidence=replace(
                    successor.evidence,
                    reasons=[
                        *successor.evidence.reasons,
                        "started successor after a confident melody handoff",
                    ],
                ),
            )
            break

    validate_alignment_sequence(result)
    return result


def repair_unmatched_alignment_gaps(
    alignments: Sequence[WordAlignment],
    candidates: Iterable[AlignmentCandidate],
    *,
    activity_spans: Sequence[tuple[float, float]] = (),
    minimum_score: float = 0.55,
) -> list[WordAlignment]:
    """Recover an unmatched token only when a candidate fits both neighbors."""

    candidate_by_token: dict[str, list[AlignmentCandidate]] = {}
    for candidate in candidates:
        if candidate.candidate_type.casefold() in _ANCHOR_CANDIDATE_TYPES:
            continue
        if not _valid_span(candidate.start, candidate.end):
            continue
        scored_candidate = candidate
        if candidate.activity_score is None and activity_spans:
            scored_candidate = replace(
                candidate,
                activity_score=_candidate_activity_score(candidate, activity_spans),
            )
        candidate_by_token.setdefault(candidate.token_id, []).append(scored_candidate)

    result = list(alignments)
    for index, alignment in enumerate(result):
        # ``fallback_word_alignments`` may have already filled an unresolved
        # token with a low-confidence interpolated span. Treat that span as a
        # placeholder, not as evidence: otherwise this recovery pass never
        # gets a chance to use a real candidate that fits both neighbors.
        synthetic_fallback = (
            _enum_value(alignment.status) == AlignmentStatus.ESTIMATED.value
            and not alignment.start_locked
            and not alignment.end_locked
            and not alignment.evidence.selected_model
            and not alignment.evidence.selected_audio
            and "estimated between nearby evidence"
            in alignment.evidence.reasons
        )
        if _alignment_has_span(alignment) and not synthetic_fallback:
            continue
        # An unresolved immediate neighbor is a hard boundary. Do not bridge
        # across it merely because a distant candidate happens to fit.
        previous = result[index - 1] if index > 0 else None
        following = result[index + 1] if index + 1 < len(result) else None
        if previous is not None and not _alignment_has_span(previous):
            continue
        if following is not None and not _alignment_has_span(following):
            continue
        previous_end = previous.end if previous is not None else None
        following_start = following.start if following is not None else None
        options: list[tuple[float, AlignmentCandidate]] = []
        for candidate in candidate_by_token.get(alignment.token_id, ()):
            assert candidate.start is not None and candidate.end is not None
            if previous_end is not None and candidate.start < previous_end - _BOUNDARY_EPSILON:
                continue
            if following_start is not None and candidate.end > following_start + _BOUNDARY_EPSILON:
                continue
            option = score_alignment_candidate(candidate, previous_end=previous_end)
            if option.score >= minimum_score:
                options.append((option.score, candidate))
        if not options:
            continue
        score, candidate = max(
            options,
            key=lambda item: (item[0], -float(item[1].end - item[1].start), -item[1].start),
        )
        candidate_type = candidate.candidate_type.casefold()
        confidence = _clamp_confidence(min(score, candidate.text_score, candidate.model_confidence))
        result[index] = replace(
            alignment,
            start=candidate.start,
            end=candidate.end,
            status=(
                AlignmentStatus.ESTIMATED
                if candidate_type in _ANCHOR_CANDIDATE_TYPES
                else AlignmentStatus.ALIGNED
            ),
            confidence=confidence,
            evidence=AlignmentEvidence(
                selected_model=candidate.model,
                selected_audio=candidate.audio_source,
                reasons=[
                    "recovered from a bounded candidate gap",
                    *candidate.reasons,
                ],
                candidate_scores=[
                    {
                        "model": candidate.model,
                        "audio_source": candidate.audio_source,
                        "candidate_type": candidate.candidate_type,
                        "score": score,
                    }
                ],
            ),
        )
    validate_alignment_sequence(result)
    return result


def expand_sustained_end(
    start: float,
    end: float,
    activity_times: Sequence[float],
    activity: Sequence[float | bool],
    *,
    next_word_start: float | None = None,
    window_end: float | None = None,
    activity_threshold: float = 0.005,
    clear_gap_s: float = 0.18,
) -> float:
    """Extend a word through contiguous vocal activity without crossing a boundary."""

    if len(activity_times) == 0 or len(activity) == 0:
        return end
    limit = min(value for value in (next_word_start, window_end) if value is not None) if any(
        value is not None for value in (next_word_start, window_end)
    ) else math.inf
    active_times = [
        float(time)
        for time, value in zip(activity_times, activity)
        if float(time) > end and float(time) < limit and float(value) >= activity_threshold
    ]
    if not active_times:
        return end
    # Only extend when activity starts near the model end. A distant island is
    # likely another word or backing vocal, not a sustain.
    if active_times[0] - end > clear_gap_s:
        return end
    extended = end
    for current, following in zip(active_times, [*active_times[1:], None]):
        extended = max(extended, min(current, limit))
        if following is None or following - current > clear_gap_s:
            break
        extended = min(following, limit)
    return max(end, min(extended, limit))


def expand_alignment_sustains(
    alignments: Sequence[WordAlignment],
    activity_times: Sequence[float],
    activity: Sequence[float | bool],
    *,
    clear_gap_s: float = 0.18,
    activity_threshold: float = 0.005,
    window_end: float | None = None,
    max_final_extension_s: float = 1.0,
    melody_outputs: Mapping[str, Sequence[ASRToken]] | None = None,
    melody_gap_threshold_s: float = 0.15,
    melody_gap_extension_s: float = 0.08,
    melody_start_gap_threshold_s: float = 0.15,
    melody_start_tolerance_s: float = 0.08,
    melody_start_offset_s: float = 0.04,
    melody_start_min_confidence: float = 0.0,
    melody_handoff_min_gap_s: float = 0.08,
    melody_handoff_max_gap_s: float = 0.25,
    melody_handoff_offset_s: float = 0.08,
) -> list[WordAlignment]:
    """Apply sustain expansion to accepted spans only.

    When melody evidence is available, a clear detector gap can cap an
    activity-based sustain. This protects a selected word end from consuming
    the next note while leaving the independent activity signal in charge of
    ordinary held-note expansion.
    """

    if (
        melody_gap_threshold_s < 0.0
        or melody_gap_extension_s < 0.0
        or melody_start_gap_threshold_s < 0.0
        or melody_start_tolerance_s < 0.0
        or melody_start_offset_s < 0.0
        or melody_handoff_min_gap_s < 0.0
        or melody_handoff_max_gap_s < melody_handoff_min_gap_s
        or melody_handoff_offset_s < 0.0
        or not 0.0 <= melody_start_min_confidence <= 1.0
    ):
        raise ValueError("melody gap and handoff thresholds must be non-negative")

    melody_events_by_source: dict[str, list[ASRToken]] = {}
    melody_start_events_by_source: dict[str, list[ASRToken]] = {}
    for source, events in (melody_outputs or {}).items():
        physical_source = source.rsplit(":", 1)[-1].casefold()
        if physical_source not in {"vocal_stem", "original_mix"}:
            continue
        valid_events = [
            event
            for event in events
            if event.evidence_type.casefold() in {"melody", "pitch", "basic_pitch"}
            and _valid_span(event.start, event.end)
        ]
        melody_events_by_source.setdefault(physical_source, []).extend(valid_events)
        melody_start_events_by_source.setdefault(physical_source, []).extend(
            event
            for event in valid_events
            if event.confidence >= melody_start_min_confidence
        )
    for events in melody_events_by_source.values():
        events.sort(key=lambda event: (event.start, event.end))
    for events in melody_start_events_by_source.values():
        events.sort(key=lambda event: (event.start, event.end))
    activity_onset_spans = (
        activity_spans_from_signal(
            activity_times,
            activity,
            threshold=activity_threshold,
        )
        if melody_events_by_source
        else []
    )

    result: list[WordAlignment] = []
    for index, alignment in enumerate(alignments):
        if (
            _enum_value(alignment.status) != AlignmentStatus.ALIGNED.value
            or alignment.start is None
            or alignment.end is None
            or alignment.end_locked
        ):
            result.append(alignment)
            continue
        immediate_next = alignments[index + 1] if index + 1 < len(alignments) else None
        # An unresolved token is a hard boundary. Looking past it lets a
        # sustained word consume time that belongs to that unresolved word.
        if immediate_next is not None and not _valid_span(immediate_next.start, immediate_next.end):
            result.append(alignment)
            continue
        next_start = immediate_next.start if immediate_next is not None and _valid_span(immediate_next.start, immediate_next.end) else None
        final_window_end = window_end
        if immediate_next is None:
            final_extension_end = alignment.end + max(0.0, max_final_extension_s)
            final_window_end = (
                final_extension_end
                if final_window_end is None
                else min(final_window_end, final_extension_end)
            )
        expanded = expand_sustained_end(
            alignment.start,
            alignment.end,
            activity_times,
            activity,
            next_word_start=next_start,
            window_end=final_window_end,
            activity_threshold=activity_threshold,
            clear_gap_s=clear_gap_s,
        )
        adjusted_start = alignment.start
        if melody_events_by_source:
            selected_source = alignment.evidence.selected_audio.casefold()
            gap_sources = [selected_source]
            if selected_source == "original_mix":
                gap_sources.append("vocal_stem")
            melody_gap_cap: float | None = None
            for source in gap_sources:
                events = melody_events_by_source.get(source, ())
                for left, right in zip(events, events[1:]):
                    if right.start - left.end < melody_gap_threshold_s:
                        continue
                    if left.end <= alignment.end:
                        continue
                    cap = left.end + melody_gap_extension_s
                    if cap <= alignment.start + _BOUNDARY_EPSILON:
                        continue
                    if next_start is not None and cap >= next_start:
                        continue
                    melody_gap_cap = cap if melody_gap_cap is None else min(melody_gap_cap, cap)
            if melody_gap_cap is not None:
                expanded = min(expanded, melody_gap_cap)
            if not alignment.start_locked:
                start_candidates: list[float] = []
                for source in gap_sources:
                    events = melody_start_events_by_source.get(source, ())
                    for left, right in zip(events, events[1:]):
                        if right.start - left.end < melody_start_gap_threshold_s:
                            continue
                        if abs(alignment.start - right.start) > melody_start_tolerance_s:
                            continue
                        candidate_start = right.start + melody_start_offset_s
                        if candidate_start >= expanded - 0.02:
                            continue
                        previous_end = result[-1].end if result else None
                        if previous_end is not None:
                            candidate_start = max(candidate_start, previous_end)
                        if candidate_start < expanded - 0.02:
                            start_candidates.append(candidate_start)
                if start_candidates:
                    adjusted_start = min(
                        start_candidates,
                        key=lambda value: abs(value - alignment.start),
                    )
            if not alignment.start_locked:
                onset_candidates = [
                    span_start + 0.04
                    for span_start, _span_end in activity_onset_spans
                    if abs(adjusted_start - span_start) <= 0.04
                    and span_start + 0.04 < expanded - 0.02
                ]
                if onset_candidates:
                    previous_end = result[-1].end if result else None
                    onset_candidates = [
                        max(candidate, previous_end or 0.0)
                        for candidate in onset_candidates
                        if max(candidate, previous_end or 0.0) < expanded - 0.02
                    ]
                    if onset_candidates:
                        adjusted_start = min(
                            onset_candidates,
                            key=lambda value: abs(value - adjusted_start),
                        )
        if (
            melody_events_by_source
            and not alignment.start_locked
            and result
            and _enum_value(result[-1].status) == AlignmentStatus.ALIGNED.value
            and _valid_span(result[-1].start, result[-1].end)
            and not result[-1].end_locked
        ):
            previous = result[-1]
            assert previous.end is not None
            gap = adjusted_start - previous.end
            if melody_handoff_min_gap_s < gap <= melody_handoff_max_gap_s:
                has_continuous_activity = any(
                    span_start <= previous.end + 0.02
                    and span_end >= adjusted_start - 0.02
                    for span_start, span_end in activity_onset_spans
                )
                if has_continuous_activity:
                    handoff_start = previous.end + melody_handoff_offset_s
                    if handoff_start < expanded - 0.02:
                        adjusted_start = min(adjusted_start, handoff_start)
        result.append(replace(alignment, start=adjusted_start, end=expanded))
    validate_alignment_sequence(result)
    return result


def merge_alignment_locks(
    previous: Sequence[WordAlignment],
    incoming: Sequence[WordAlignment],
    *,
    reset_token_ids: Iterable[str] = (),
    token_id_aliases: Mapping[str, str] | None = None,
) -> list[WordAlignment]:
    """Preserve manually locked fields during ordinary re-alignment."""

    previous_by_token = {alignment.token_id: alignment for alignment in previous}
    resets = set(reset_token_ids)
    aliases = token_id_aliases or {}
    result: list[WordAlignment] = []
    for alignment in incoming:
        old_token_id = aliases.get(alignment.token_id, alignment.token_id)
        old = previous_by_token.get(old_token_id)
        if old is None or alignment.token_id in resets or old_token_id in resets:
            result.append(alignment)
            continue
        start_locked = old.start_locked
        end_locked = old.end_locked
        preserved_companion: list[str] = []
        old_has_span = _valid_span(old.start, old.end)
        merged_start = old.start if start_locked else alignment.start
        merged_end = old.end if end_locked else alignment.end
        if start_locked and (merged_end is None or (merged_start is not None and merged_end < merged_start)) and old_has_span:
            merged_end = old.end
            preserved_companion.append("previous end retained to keep the locked start paired")
        if end_locked and (merged_start is None or (merged_start is not None and merged_end is not None and merged_end < merged_start)) and old_has_span:
            merged_start = old.start
            preserved_companion.append("previous start retained to keep the locked end paired")
        if start_locked and end_locked and old_has_span:
            merged_start, merged_end = old.start, old.end
            preserved_companion.append("previous locked span retained as a unit")
        merged = replace(
                alignment,
                start=merged_start,
                end=merged_end,
                start_locked=start_locked,
                end_locked=end_locked,
                evidence=replace(
                    alignment.evidence,
                    reasons=[
                        *alignment.evidence.reasons,
                        *preserved_companion,
                        *(["manual start lock preserved"] if start_locked else []),
                        *(["manual end lock preserved"] if end_locked else []),
                    ],
                ),
            )
        if (
            merged.start is not None
            and merged.end is not None
            and _enum_value(merged.status) == AlignmentStatus.UNMATCHED.value
        ):
            merged = replace(merged, status=AlignmentStatus.ESTIMATED, confidence=min(merged.confidence, 0.35))
        elif (merged.start is None) != (merged.end is None):
            merged = replace(
                merged,
                start=None,
                end=None,
                start_locked=False,
                end_locked=False,
                status=AlignmentStatus.UNMATCHED,
                confidence=0.0,
                evidence=replace(
                    merged.evidence,
                    reasons=[*merged.evidence.reasons, "manual lock withheld because incoming span was incomplete"],
                ),
            )
        result.append(merged)
    validate_alignment_sequence(result)
    return result


def map_token_id_aliases(
    previous_tokens: Sequence[LyricToken],
    new_tokens: Sequence[LyricToken],
) -> dict[str, str]:
    """Map unchanged source words across immutable lyric revisions.

    Matching is stable and source-ordered, so repeated words retain the
    corresponding manual lock instead of all resolving to the first repeat.
    """

    aliases: dict[str, str] = {}
    old_values = [token.normalized_text for token in previous_tokens]
    new_values = [token.normalized_text for token in new_tokens]
    matcher = difflib.SequenceMatcher(a=old_values, b=new_values, autojunk=False)
    for old_start, new_start, size in matcher.get_matching_blocks():
        for offset in range(size):
            if old_start + offset >= len(previous_tokens) or new_start + offset >= len(new_tokens):
                continue
            aliases[new_tokens[new_start + offset].id] = previous_tokens[old_start + offset].id
    return aliases


def edit_word_boundary(
    alignment: WordAlignment,
    *,
    start: float | None | object = _UNSET,
    end: float | None | object = _UNSET,
) -> WordAlignment:
    """Apply a direct boundary edit and lock every changed field."""

    next_start = alignment.start if start is _UNSET else start
    next_end = alignment.end if end is _UNSET else end
    if next_start is None or next_end is None:
        if start is not _UNSET or end is not _UNSET:
            raise ValueError("Word boundaries must be supplied as a finite pair")
        if next_start is not None or next_end is not None:
            raise ValueError("Word boundaries must be supplied as a pair")
    else:
        try:
            start_value, end_value = float(next_start), float(next_end)
        except (TypeError, ValueError) as exc:
            raise ValueError("Word boundaries must be finite numbers") from exc
        if not math.isfinite(start_value) or not math.isfinite(end_value):
            raise ValueError("Word boundaries must be finite numbers")
        if start_value < 0 or end_value < start_value:
            raise ValueError("Word end must be at or after its start")
    reasons = list(alignment.evidence.reasons)
    if start is not _UNSET:
        reasons.append("start boundary manually edited and locked")
    if end is not _UNSET:
        reasons.append("end boundary manually edited and locked")
    status = alignment.status
    if next_start is not None and next_end is not None and _enum_value(status) == AlignmentStatus.UNMATCHED.value:
        # A user-supplied span is useful export data, but it is not model
        # evidence, so keep the honest estimated label.
        status = AlignmentStatus.ESTIMATED
    return replace(
        alignment,
        start=None if next_start is None else float(next_start),
        end=None if next_end is None else float(next_end),
        status=status,
        start_locked=alignment.start_locked or start is not _UNSET,
        end_locked=alignment.end_locked or end is not _UNSET,
        evidence=replace(alignment.evidence, reasons=list(dict.fromkeys(reasons))),
    )


def reset_word_boundaries(
    alignment: WordAlignment,
    *,
    reset_start: bool = True,
    reset_end: bool = True,
) -> WordAlignment:
    """Explicitly remove named locks; the next run may change those fields."""

    return replace(
        alignment,
        start_locked=False if reset_start else alignment.start_locked,
        end_locked=False if reset_end else alignment.end_locked,
        evidence=replace(
            alignment.evidence,
            reasons=[
                *alignment.evidence.reasons,
                *(["start boundary lock reset"] if reset_start else []),
                *(["end boundary lock reset"] if reset_end else []),
            ],
        ),
    )


def project_alignments_to_words(
    tokens: Sequence[LyricToken],
    alignments: Sequence[WordAlignment],
    *,
    existing_words: Sequence[Word] = (),
) -> list[Word]:
    """Project canonical spans to the legacy ``Word`` shape for export/pitch.

    Existing pitch fields may be copied by token ID, but existing timestamps
    are intentionally ignored. Canonical alignment remains the only timing
    source.
    """

    alignments_by_token = {alignment.token_id: alignment for alignment in alignments}
    pitch_by_token: dict[str, list[Word]] = {}
    for word in existing_words:
        if word.token_id:
            pitch_by_token.setdefault(word.token_id, []).append(word)
    projected: list[Word] = []
    for token in tokens:
        alignment = alignments_by_token.get(token.id)
        if alignment is None:
            continue
        if not _valid_span(alignment.start, alignment.end):
            # An unmatched word remains visible in the canonical editor but
            # must not become a fabricated 0.0/0.0 export word.
            continue
        assert alignment.start is not None and alignment.end is not None
        start = float(alignment.start)
        end = max(start, float(alignment.end))
        pitch_words = pitch_by_token.get(token.id, [])
        if not pitch_words:
            pitch_words = [Word(text=token.display_text, start=start, end=end)]
        source_start = min(word.start for word in pitch_words)
        source_end = max(word.end for word in pitch_words)
        source_duration = source_end - source_start
        target_duration = end - start
        for index, pitch in enumerate(pitch_words):
            if source_duration > 0:
                projected_start = start + (pitch.start - source_start) * target_duration / source_duration
                projected_end = start + (pitch.end - source_start) * target_duration / source_duration
            else:
                slot = target_duration / len(pitch_words)
                projected_start = start + index * slot
                projected_end = start + (index + 1) * slot
            projected.append(
                Word(
                    text=pitch.text if pitch_words != [pitch] else token.display_text,
                    start=projected_start,
                    end=max(projected_start, min(end, projected_end)),
                    pitch_hz=pitch.pitch_hz,
                    midi_note=pitch.midi_note,
                    confidence=alignment.confidence,
                    pitch_contour=list(pitch.pitch_contour),
                    continuation_of_previous=pitch.continuation_of_previous,
                    token_id=token.id,
                    alignment_status=_enum_value(alignment.status),
                    alignment_run_id=alignment.alignment_run_id,
                )
            )
    return projected


def line_breaks_from_tokens(tokens: Sequence[LyricToken]) -> list[int]:
    """Return canonical phrase boundaries from original source line breaks."""

    return [
        index
        for index in range(1, len(tokens))
        if tokens[index].line_index != tokens[index - 1].line_index
    ]


def line_breaks_for_projected_words(
    tokens: Sequence[LyricToken],
    words: Sequence[Word],
) -> list[int]:
    """Map source line changes onto pitch-split legacy words."""

    line_by_token = {token.id: token.line_index for token in tokens}
    return [
        index
        for index in range(1, len(words))
        if line_by_token.get(words[index].token_id) != line_by_token.get(words[index - 1].token_id)
    ]
