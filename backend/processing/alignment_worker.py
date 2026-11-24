"""Alignment worker contracts, model evidence, and timing selection.

The request/response shapes are intentionally independent of a particular
transport. A deployment can send the same dictionaries to a GPU worker over
HTTP or a queue; the local adapter is useful for development and tests.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
import math
import os
import tempfile
from dataclasses import asdict, replace
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, field_validator, model_validator

from models import AlignmentStatus, LyricRevision, LyricToken, WordAlignment
from processing.alignment import (
    ASRToken,
    AlignmentCandidate,
    build_word_alignments,
    global_monotonic_match,
    normalize_language,
    normalize_lyric_token,
    score_alignment_candidate,
    token_similarity,
    repair_unmatched_alignment_gaps,
    repair_delayed_forced_onsets,
    split_repeated_estimated_alignments,
    trim_alignment_tails_to_activity,
    snap_alignment_starts_to_activity_onsets,
    trim_alignment_tails_to_strong_activity,
    trim_forced_boundary_latency,
    trim_selected_melody_gaps,
    _valid_span,
)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().casefold() not in {"0", "false", "off", "no", "disabled"}


def sofa_language_allowed(language: str | None) -> bool:
    """Return whether SOFA is supported for the requested lyric language."""

    normalized = normalize_language(language)
    return normalized == "en" or _env_flag("SOFA_ALLOW_NON_ENGLISH", False)


def resolve_alignment_backend(language: str | None, requested: str | None = "sofa") -> str:
    """Resolve the backend while keeping SOFA English-only by default."""

    backend = (requested or "sofa").strip().casefold()
    configured_primary = os.environ.get("ALIGNMENT_PRIMARY_BACKEND", "").strip().casefold()
    if backend == "sofa" and configured_primary in {"legacy", "current", "asr", "off"}:
        backend = "legacy"
    if backend == "sofa" and not sofa_language_allowed(language):
        backend = "legacy"
    return backend


class WorkerToken(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: StrictStr
    lyric_revision_id: StrictStr
    display_text: StrictStr
    normalized_text: StrictStr
    line_index: StrictInt
    token_index: StrictInt
    source_index: StrictInt = 0

    @field_validator("id", "lyric_revision_id", "display_text", "normalized_text")
    @classmethod
    def validate_nonempty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("worker token text and IDs must be non-empty")
        return value

    @field_validator("line_index", "token_index", "source_index")
    @classmethod
    def validate_indices(cls, value: int) -> int:
        if value < 0:
            raise ValueError("worker token indices must be non-negative")
        return value


class AlignmentWorkerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: StrictStr
    alignment_run_id: StrictStr
    lyric_revision_id: StrictStr
    language: StrictStr
    original_audio_ref: StrictStr
    vocal_stem_audio_ref: StrictStr
    alignment_backend: StrictStr = "sofa"
    tokens: list[WorkerToken] = Field(default_factory=list)
    activity_spans: list[tuple[float, float]] = Field(default_factory=list)
    timing_grid_step: float | None = None
    timing_grid_offset: float | None = None

    @field_validator("activity_spans")
    @classmethod
    def validate_activity_spans(cls, spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
        previous_end: float | None = None
        for start, end in spans:
            if not math.isfinite(start) or not math.isfinite(end) or start < 0.0 or end < start:
                raise ValueError("activity spans must be finite, non-negative, and ordered")
            if previous_end is not None and start < previous_end:
                raise ValueError("activity spans must be source-ordered and non-overlapping")
            previous_end = end
        return spans

    @model_validator(mode="after")
    def validate_contract(self) -> "AlignmentWorkerRequest":
        self.language = normalize_language(self.language)
        self.alignment_backend = resolve_alignment_backend(self.language, self.alignment_backend)
        if self.alignment_backend not in {"sofa", "legacy"}:
            raise ValueError("alignment_backend must be sofa or legacy")
        token_ids = [token.id for token in self.tokens]
        if len(token_ids) != len(set(token_ids)):
            raise ValueError("worker request contains duplicate token IDs")
        for token in self.tokens:
            if token.lyric_revision_id != self.lyric_revision_id:
                raise ValueError("worker token belongs to a different lyric revision")
        if (self.timing_grid_step is None) != (self.timing_grid_offset is None):
            raise ValueError("timing grid step and offset must be supplied together")
        if self.timing_grid_step is not None and (
            not math.isfinite(self.timing_grid_step)
            or self.timing_grid_step <= 0.0
            or not math.isfinite(self.timing_grid_offset or 0.0)
        ):
            raise ValueError("timing grid values must be finite and positive")
        return self

    @field_validator("job_id", "alignment_run_id", "lyric_revision_id")
    @classmethod
    def validate_ids(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("worker request IDs must be non-empty")
        return value


class WorkerWordAlignment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    token_id: StrictStr
    alignment_run_id: StrictStr
    start: float | None = None
    end: float | None = None
    status: StrictStr
    confidence: float
    evidence: dict[str, Any] = Field(default_factory=dict)
    start_locked: bool = False
    end_locked: bool = False


class AlignmentWorkerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: StrictStr
    alignment_run_id: StrictStr
    lyric_revision_id: StrictStr
    status: StrictStr
    alignments: list[WorkerWordAlignment] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    draft_text: str | None = None


class AlignmentWorkerError(ValueError):
    """Raised when a worker payload cannot be safely applied."""


def dispatch_worker_alignment(
    request: AlignmentWorkerRequest,
    worker_url: str,
    *,
    worker_token: str | None = None,
    timeout_s: float = 900.0,
) -> AlignmentWorkerResponse:
    """Dispatch through Rust and validate the returned alignment evidence."""
    from processing.rust_utility import call

    try:
        response_data = call(
            "dispatch_worker_alignment",
            request=_model_dict(request),
            url=worker_url,
            token=worker_token,
            timeout=timeout_s,
            timeout_seconds=max(1, timeout_s + 5),
        )
    except (RuntimeError, OSError, ValueError) as exc:
        raise AlignmentWorkerError(f"Remote alignment worker request failed: {exc}") from exc
    try:
        response = AlignmentWorkerResponse(**response_data)
    except Exception as exc:
        raise AlignmentWorkerError(f"Remote alignment worker returned an invalid payload: {exc}") from exc
    return validate_worker_response(response, request)


def _model_dict(model: BaseModel) -> dict[str, Any]:
    dump = getattr(model, "model_dump", None)
    return dump() if dump else model.dict()


def build_worker_request(
    *,
    job_id: str,
    revision: LyricRevision,
    alignment_run_id: str,
    original_audio_ref: str,
    vocal_stem_audio_ref: str,
    activity_spans: Sequence[tuple[float, float]] = (),
    timing_grid: tuple[float, float] | None = None,
    alignment_backend: str = "sofa",
) -> AlignmentWorkerRequest:
    return AlignmentWorkerRequest(
        job_id=job_id,
        alignment_run_id=alignment_run_id,
        lyric_revision_id=revision.id,
        language=str(getattr(revision.language, "value", revision.language)),
        original_audio_ref=original_audio_ref,
        vocal_stem_audio_ref=vocal_stem_audio_ref,
        alignment_backend=alignment_backend,
        tokens=[WorkerToken(**asdict(token)) for token in revision.tokens],
        activity_spans=list(activity_spans),
        timing_grid_step=timing_grid[0] if timing_grid is not None else None,
        timing_grid_offset=timing_grid[1] if timing_grid is not None else None,
    )


def validate_worker_response(
    response: AlignmentWorkerResponse,
    request: AlignmentWorkerRequest,
) -> AlignmentWorkerResponse:
    """Validate identity and token coverage before persistence."""

    if response.job_id != request.job_id:
        raise AlignmentWorkerError("Worker response belongs to a different job")
    if response.alignment_run_id != request.alignment_run_id:
        raise AlignmentWorkerError("Worker response belongs to a different alignment run")
    if response.lyric_revision_id != request.lyric_revision_id:
        raise AlignmentWorkerError("Worker response belongs to a different lyric revision")
    if response.status not in {"completed", "failed"}:
        raise AlignmentWorkerError("Worker response has an invalid status")

    token_ids = {token.id for token in request.tokens}
    response_ids = {alignment.token_id for alignment in response.alignments}
    unknown_ids = response_ids - token_ids
    if unknown_ids:
        raise AlignmentWorkerError("Worker response contains unknown token IDs")
    if len(response_ids) != len(response.alignments):
        raise AlignmentWorkerError("Worker response contains duplicate token IDs")
    if response.status == "completed" and response_ids != token_ids:
        raise AlignmentWorkerError("Completed worker response must cover every lyric token")
    request_order = {token.id: index for index, token in enumerate(request.tokens)}
    previous_request_index = -1
    previous_end: float | None = None
    for alignment in response.alignments:
        if alignment.alignment_run_id != request.alignment_run_id:
            raise AlignmentWorkerError("Worker alignment belongs to a different alignment run")
        if alignment.start_locked or alignment.end_locked:
            raise AlignmentWorkerError("Worker responses cannot create manual boundary locks")
        if alignment.status not in {status.value for status in AlignmentStatus}:
            raise AlignmentWorkerError(f"Invalid alignment status: {alignment.status}")
        request_index = request_order[alignment.token_id]
        if request_index <= previous_request_index:
            raise AlignmentWorkerError("Worker response is not in source-token order")
        previous_request_index = request_index
        if not math.isfinite(alignment.confidence) or not 0.0 <= alignment.confidence <= 1.0:
            raise AlignmentWorkerError("Worker response contains invalid confidence")
        if alignment.status == AlignmentStatus.UNMATCHED.value:
            if alignment.start is not None or alignment.end is not None:
                raise AlignmentWorkerError("Unmatched worker words cannot contain timing")
            continue
        if alignment.start is None or alignment.end is None:
            raise AlignmentWorkerError("Aligned worker words require paired boundaries")
        if (
            not math.isfinite(alignment.start)
            or not math.isfinite(alignment.end)
            or alignment.start < 0.0
            or alignment.end < alignment.start
        ):
            raise AlignmentWorkerError("Worker response contains invalid word boundaries")
        if previous_end is not None and alignment.start < previous_end - 0.02:
            raise AlignmentWorkerError("Worker response contains overlapping or out-of-order spans")
        previous_end = max(previous_end or alignment.end, alignment.end)
    return response


def _alignment_to_payload(alignment: WordAlignment) -> WorkerWordAlignment:
    evidence = asdict(alignment.evidence)
    return WorkerWordAlignment(
        token_id=alignment.token_id,
        alignment_run_id=alignment.alignment_run_id,
        start=alignment.start,
        end=alignment.end,
        status=str(getattr(alignment.status, "value", alignment.status)),
        confidence=alignment.confidence,
        evidence=evidence,
        start_locked=alignment.start_locked,
        end_locked=alignment.end_locked,
    )


def alignment_response_to_models(
    response: AlignmentWorkerResponse,
) -> list[WordAlignment]:
    """Convert a validated worker payload to the canonical dataclasses."""

    from models import AlignmentEvidence

    allowed_evidence = {"selected_model", "selected_audio", "reasons", "candidate_scores"}
    converted: list[WordAlignment] = []
    for item in response.alignments:
        unknown = set(item.evidence) - allowed_evidence
        if unknown:
            raise AlignmentWorkerError(f"Worker evidence contains unknown fields: {sorted(unknown)}")
        reasons = item.evidence.get("reasons", [])
        candidate_scores = item.evidence.get("candidate_scores", [])
        if not isinstance(reasons, list) or not all(isinstance(reason, str) for reason in reasons):
            raise AlignmentWorkerError("Worker evidence reasons must be an array of strings")
        if not isinstance(candidate_scores, list) or not all(isinstance(score, dict) for score in candidate_scores):
            raise AlignmentWorkerError("Worker candidate scores must be an array of objects")
        for score in candidate_scores:
            unknown_score_fields = set(score) - {"model", "audio_source", "candidate_type", "score"}
            if unknown_score_fields:
                raise AlignmentWorkerError(
                    f"Worker candidate score contains unknown fields: {sorted(unknown_score_fields)}"
                )
            if not isinstance(score.get("model", ""), str) or not isinstance(score.get("audio_source", ""), str):
                raise AlignmentWorkerError("Worker candidate score model and audio source must be strings")
            score_value = score.get("score")
            if score_value is not None:
                if isinstance(score_value, bool) or not isinstance(score_value, (int, float)):
                    raise AlignmentWorkerError("Worker candidate score must be numeric")
                if not math.isfinite(float(score_value)) or not 0.0 <= float(score_value) <= 1.0:
                    raise AlignmentWorkerError("Worker candidate score must be finite and within [0, 1]")
        converted.append(
            WordAlignment(
                token_id=item.token_id,
                alignment_run_id=item.alignment_run_id,
                start=item.start,
                end=item.end,
                status=item.status,
                confidence=item.confidence,
                evidence=AlignmentEvidence(
                    selected_model=str(item.evidence.get("selected_model", "")),
                    selected_audio=str(item.evidence.get("selected_audio", "")),
                    reasons=reasons,
                    candidate_scores=candidate_scores,
                ),
                start_locked=False,
                end_locked=False,
            )
        )
    return converted


def _candidate_sources(
    request: AlignmentWorkerRequest,
    asr_outputs: Mapping[str, Sequence[ASRToken]],
    activity_spans: Sequence[tuple[float, float]] = (),
) -> tuple[list[ASRToken], list[AlignmentCandidate], dict[str, Any]]:
    """Create candidates from each audio source while retaining source agreement."""

    # An explicitly supplied UltraStar chart is a complete timing source, not
    # another noisy model opinion. Keep it isolated so a lower-quality model
    # candidate cannot win a tie or overwrite chart timings.
    if "ultrastar_chart" in asr_outputs:
        asr_outputs = {"ultrastar_chart": asr_outputs["ultrastar_chart"]}

    def physical_source(source_key: str, token: ASRToken) -> str:
        key_source = source_key.rsplit(":", 1)[-1]
        if key_source in {"vocal_stem", "original_mix"}:
            return key_source
        return token.audio_source or key_source

    tokens = [LyricToken(**_model_dict(token)) for token in request.tokens]
    per_source_matches: dict[str, dict[int, tuple[int, float]]] = {}
    prepared_tokens_by_source: dict[str, Sequence[ASRToken]] = {}
    for source, words in asr_outputs.items():
        match = global_monotonic_match(tokens, words)
        prepared_tokens_by_source[source] = match.prepared_asr_tokens or tuple(words)
        global_matches = {
            item.lyric_index: (item.asr_index, item.similarity)
            for item in match.matches
            if item.matched and item.asr_index is not None
        }
        indexed_matches = {
            int(asr.sequence_index): (asr_index, 1.0)
            for asr_index, asr in enumerate(words)
            if asr.sequence_index is not None
            and 0 <= int(asr.sequence_index) < len(tokens)
            and token_similarity(tokens[int(asr.sequence_index)].normalized_text, asr.normalized_text) >= 0.95
        }
        # Segment-local CTC/Qwen adapters carry an explicit source-token index.
        # Prefer it over a second global match, which can jump to a repeated
        # chorus when one short window misses a word.
        if indexed_matches:
            indexed_lyric_indices = set(indexed_matches)
            indexed_asr_indices = {
                asr_index for asr_index, _similarity in indexed_matches.values()
            }
            # Segment-local forced evidence carries an explicit transcript
            # index.  Remove conflicting global matches for the same ASR
            # token or lyric position; otherwise a repeated word can create
            # two identical candidates and the DP may choose the later copy.
            per_source_matches[source] = {
                lyric_index: match
                for lyric_index, match in global_matches.items()
                if lyric_index not in indexed_lyric_indices
                and match[0] not in indexed_asr_indices
            }
            per_source_matches[source].update(indexed_matches)
        else:
            per_source_matches[source] = global_matches

    def source_token(source: str, index: int) -> ASRToken | None:
        prepared = prepared_tokens_by_source.get(source)
        if prepared is not None and 0 <= index < len(prepared):
            return prepared[index]
        original = asr_outputs.get(source, ())
        return original[index] if 0 <= index < len(original) else None

    anchor_matches_by_source: dict[str, dict[int, tuple[int, float]]] = {}
    for source in ("vocal_stem", "original_mix"):
        words = asr_outputs.get(source, ())
        if not words:
            continue
        match = global_monotonic_match(tokens, words)
        prepared_tokens_by_source[source] = match.prepared_asr_tokens or tuple(words)
        anchor_matches_by_source[source] = {
            item.lyric_index: (item.asr_index, item.similarity)
            for item in match.matches
            if item.matched and item.asr_index is not None
        }
    preferred_anchor_source = max(
        anchor_matches_by_source,
        key=lambda source: len(anchor_matches_by_source[source]),
        default="",
    )

    # A full-track forced trellis is lexically indexed, but it can spend a
    # long instrumental gap matching words that the independent ASR clock
    # recognizes only at the next real vocal occurrence.  Instead of dropping
    # every forced span in that interval, fit a monotonic piecewise clock from
    # matching ASR anchors and warp the forced boundaries between those
    # anchors.  This retains forced word-level resolution while correcting the
    # long-range drift that otherwise shifts an entire verse by seconds.
    forced_clock_warps: dict[int, tuple[float, float]] = {}
    force_source = "forcealign:vocal_stem"
    force_words = asr_outputs.get(force_source, ())
    # Chunked ForceAlign already uses the independent ASR clock to choose
    # each local transcript/audio window.  Applying the full-track drift warp
    # or occurrence-recovery heuristic a second time can move a correct local
    # span onto an older repeated chorus at a chunk boundary.
    force_is_chunked = any(token.chunk_index is not None for token in force_words)
    force_matches = per_source_matches.get(force_source, {})
    physical_words = prepared_tokens_by_source.get("vocal_stem", asr_outputs.get("vocal_stem", ()))
    physical_matches = anchor_matches_by_source.get("vocal_stem", {})
    warp_points: list[tuple[int, float, float, float, float]] = []
    if not force_is_chunked and force_words and physical_words and force_matches and physical_matches:
        for lyric_index, (force_index, _similarity) in force_matches.items():
            physical_match = physical_matches.get(lyric_index)
            if physical_match is None:
                continue
            if not (0 <= force_index < len(force_words) and 0 <= physical_match[0] < len(physical_words)):
                continue
            forced_word = force_words[force_index]
            physical_word = physical_words[physical_match[0]]
            if (
                forced_word.sequence_index is None
                or physical_word.confidence < 0.70
                or not _valid_span(forced_word.start, forced_word.end)
                or not _valid_span(physical_word.start, physical_word.end)
            ):
                continue
            warp_points.append(
                (
                    int(lyric_index),
                    float(forced_word.start),
                    float(forced_word.end),
                    float(physical_word.start),
                    float(physical_word.end),
                )
            )
    warp_points.sort(key=lambda item: item[0])
    if len(warp_points) >= 2:
        for point_index, (left_lyric, left_fs, left_fe, left_ps, left_pe) in enumerate(warp_points[:-1]):
            right_lyric, right_fs, right_fe, right_ps, right_pe = warp_points[point_index + 1]
            if right_lyric <= left_lyric or right_fs <= left_fs or right_fe <= left_fe:
                continue
            # A clock jump of less than two seconds is normal forced-vs-ASR
            # boundary bias, not long-range drift.  Leave those spans raw so
            # held-word ends are not over-corrected.
            if abs(right_fs - right_ps) < 6.0:
                continue
            physical_span = max(0.0, right_ps - left_ps)
            forced_span = max(0.0, right_fs - left_fs)
            lyric_count = right_lyric - left_lyric
            # A repeated ASR occurrence can be lexically valid at two nearby
            # lyric indices.  If its clock compresses dozens of source words
            # into a fraction of a second, it is the wrong occurrence and
            # must not become a warp anchor.
            if physical_span < max(0.35 * forced_span, 0.12 * lyric_count):
                continue
            lyric_span = float(right_lyric - left_lyric)
            force_start_span = float(right_fs - left_fs)
            force_end_span = float(right_fe - left_fe)
            for force_token in force_words:
                sequence_index = force_token.sequence_index
                if sequence_index is None or not left_lyric <= int(sequence_index) <= right_lyric:
                    continue
                alpha = (float(sequence_index) - left_lyric) / lyric_span
                # Interpolate in the observed forced clock, then map that
                # fraction onto the independent ASR occurrence clock.
                normalized_start = (float(force_token.start) - left_fs) / force_start_span
                normalized_end = (float(force_token.end) - left_fe) / force_end_span
                warped_start = left_ps + normalized_start * (right_ps - left_ps)
                warped_end = left_pe + normalized_end * (right_pe - left_pe)
                if (
                    _valid_span(warped_start, warped_end)
                    and abs(warped_start - float(force_token.start)) <= 15.0
                ):
                    forced_clock_warps[int(sequence_index)] = (warped_start, warped_end)
    if forced_clock_warps:
        # A pair of repeated-word anchors can produce an attractive-looking
        # but isolated warp at the wrong chorus occurrence.  Require a short
        # contiguous run before allowing it to alter the canonical sequence;
        # genuine long-range drift produces many adjacent indexed words.
        warped_indices = sorted(forced_clock_warps)
        keep_indices: set[int] = set()
        group: list[int] = []
        for sequence_index in warped_indices + [None]:
            if sequence_index is not None and (not group or sequence_index == group[-1] + 1):
                group.append(sequence_index)
                continue
            if len(group) >= 3:
                keep_indices.update(group)
            group = [] if sequence_index is None else [sequence_index]
        forced_clock_warps = {
            sequence_index: span
            for sequence_index, span in forced_clock_warps.items()
            if sequence_index in keep_indices
        }
    if forced_clock_warps:
        # Linear interpolation can make a long word cross the next warped
        # onset when the ASR anchor skips intervening syllables.  Keep the
        # repaired clock strictly source-ordered so canonical validation never
        # sees overlapping trusted spans.
        previous_end: float | None = None
        for sequence_index in sorted(forced_clock_warps):
            start, end = forced_clock_warps[sequence_index]
            if previous_end is not None and start < previous_end + 0.02:
                start = previous_end + 0.02
            if end <= start + 0.02:
                continue
            forced_clock_warps[sequence_index] = (start, end)
            previous_end = end

    configured_primary = getattr(request, "alignment_backend", "") or os.environ.get(
        "ALIGNMENT_PRIMARY_BACKEND", "sofa"
    )
    environment_primary = os.environ.get("ALIGNMENT_PRIMARY_BACKEND", "").strip().casefold()
    if configured_primary == "sofa" and environment_primary in {"legacy", "current", "asr", "off"}:
        configured_primary = environment_primary
    configured_primary = configured_primary.strip().casefold()
    primary_source = (
        "ultrastar_chart"
        if "ultrastar_chart" in asr_outputs
        else "sofa:vocal_stem"
        if configured_primary not in {"legacy", "current", "asr", "off"}
        and "sofa:vocal_stem" in asr_outputs
        else "vocal_stem"
        if "vocal_stem" in asr_outputs
        else next(iter(asr_outputs), "")
    )
    primary = list(asr_outputs.get(primary_source, ()))
    candidates: list[AlignmentCandidate] = []

    ctc_source = "whisperx_ctc:vocal_stem"

    try:
        ctc_physical_clock_max_delta = float(
            os.environ.get("ALIGNMENT_CTC_PHYSICAL_CLOCK_MAX_DELTA", "0.55")
        )
    except ValueError:
        ctc_physical_clock_max_delta = 0.55
    if not math.isfinite(ctc_physical_clock_max_delta) or ctc_physical_clock_max_delta < 0.0:
        ctc_physical_clock_max_delta = 0.55

    def ctc_sequence_anchor(lyric_index: int) -> ASRToken | None:
        """Return the explicitly indexed vocal CTC token for a lyric word."""

        match = per_source_matches.get(ctc_source, {}).get(lyric_index)
        if match is None or ctc_source not in asr_outputs:
            return None
        token = prepared_tokens_by_source.get(ctc_source, asr_outputs[ctc_source])[match[0]]
        return token if token.confidence >= 0.05 else None

    def ctc_sequence_is_monotonic(lyric_index: int) -> bool:
        """Reject a CTC boundary clock inside a locally reversed run."""

        current = ctc_sequence_anchor(lyric_index)
        if current is None:
            return False
        previous = ctc_sequence_anchor(lyric_index - 1)
        following = ctc_sequence_anchor(lyric_index + 1)
        if previous is not None and current.start < previous.end - 0.25:
            return False
        if following is not None and current.end > following.start + 0.25:
            return False
        return True

    def lexical_anchors(lyric_index: int) -> list[tuple[ASRToken, str]]:
        """Return bounded lexical clocks for a melody event.

        Raw ASR is the occurrence clock, while CTC is a useful local clock
        for sung onsets.  CTC can drift into a repeated chorus, so it is only
        admitted when both of its boundaries remain close to a physical ASR
        anchor.  The melody lattice still supplies the canonical boundaries.
        """

        physical_sources = list(
            dict.fromkeys(
                source
                for source in (preferred_anchor_source, "original_mix", "vocal_stem")
                if source
            )
        )
        all_physical_anchors: list[tuple[ASRToken, str]] = []
        for source in physical_sources:
            match = anchor_matches_by_source.get(source, {}).get(lyric_index)
            if match is not None:
                all_physical_anchors.append(
                    (prepared_tokens_by_source.get(source, asr_outputs[source])[match[0]], source)
                )
        ctc_match = per_source_matches.get(ctc_source, {}).get(lyric_index)
        ctc_anchor = (
            prepared_tokens_by_source.get(ctc_source, asr_outputs[ctc_source])[ctc_match[0]]
            if ctc_match is not None and ctc_source in asr_outputs
            else None
        )

        def ctc_clock_is_locally_monotonic() -> bool:
            """Accept a CTC occurrence only inside a locally ordered run.

            CTC output can jump backwards when a sung syllable is assigned to
            the wrong repeated occurrence. A short monotonic neighborhood is
            still useful for widening the melody search window, especially
            when raw ASR selected a nearby but incorrect repetition.
            """

            if ctc_anchor is None or ctc_anchor.confidence < 0.05:
                return False
            ctc_matches = per_source_matches.get(ctc_source, {})

            previous = [
                (index, match)
                for index, match in ctc_matches.items()
                if index < lyric_index
            ]
            following = [
                (index, match)
                for index, match in ctc_matches.items()
                if index > lyric_index
            ]
            if previous:
                previous_token = prepared_tokens_by_source.get(
                    ctc_source, asr_outputs[ctc_source]
                )[max(previous, key=lambda item: item[0])[1][0]]
                if ctc_anchor.start < previous_token.end - 0.25:
                    return False
            if following:
                following_token = prepared_tokens_by_source.get(
                    ctc_source, asr_outputs[ctc_source]
                )[min(following, key=lambda item: item[0])[1][0]]
                if ctc_anchor.end > following_token.start + 0.25:
                    return False
            return True

        ctc_sequence_supported = ctc_clock_is_locally_monotonic()
        if not all_physical_anchors:
            return (
                [(ctc_anchor, ctc_source)]
                if ctc_sequence_supported and ctc_anchor is not None
                else []
            )
        preferred = all_physical_anchors[0]
        anchors = [
            anchor
            for anchor in all_physical_anchors
            if abs(anchor[0].start - preferred[0].start) <= 1.5
            and abs(anchor[0].end - preferred[0].end) <= 1.5
        ]
        # ASR can hallucinate a repeated tail during an instrumental outro.
        # When the next lexical timestamp jumps several seconds beyond the
        # immediately preceding word on the same lyric line, keep the
        # candidate window attached to the preceding musical clock instead
        # of searching only around the stale late timestamp.
        continuity_anchor: tuple[ASRToken, str] | None = None
        current_source = preferred[1]
        current_anchor = preferred[0]
        if current_source in anchor_matches_by_source:
            previous_matches = [
                (index, match)
                for index, match in anchor_matches_by_source[current_source].items()
                if index < lyric_index
            ]
            if previous_matches:
                previous_index, previous_match = max(previous_matches, key=lambda item: item[0])
                previous_anchor = prepared_tokens_by_source.get(
                    current_source, asr_outputs[current_source]
                )[previous_match[0]]
                if (
                    lyric_index == len(tokens) - 1
                    and tokens[previous_index].line_index == tokens[lyric_index].line_index
                    and current_anchor.start - previous_anchor.end > 4.0
                ):
                    continuity_duration = max(
                        0.35,
                        min(1.0, current_anchor.end - current_anchor.start),
                    )
                    continuity_anchor = (
                        replace(
                            current_anchor,
                            start=previous_anchor.end,
                            end=previous_anchor.end + continuity_duration,
                        ),
                        f"continuity:{current_source}",
                    )
        if continuity_anchor is not None:
            preferred = continuity_anchor
            anchors = [continuity_anchor]
        if ctc_anchor is not None:
            nearest_physical = min(
                all_physical_anchors,
                key=lambda anchor: min(
                    abs(ctc_anchor.start - anchor[0].start),
                    abs(ctc_anchor.end - anchor[0].end),
                ),
            )
            clock_delta = min(
                abs(ctc_anchor.start - nearest_physical[0].start),
                abs(ctc_anchor.end - nearest_physical[0].end),
            )
            ctc_span_gap = max(
                nearest_physical[0].start - ctc_anchor.end,
                ctc_anchor.start - nearest_physical[0].end,
                0.0,
            )
            if (
                continuity_anchor is None
                and ctc_anchor.confidence >= 0.05
                and clock_delta <= ctc_physical_clock_max_delta
                and ctc_span_gap <= 0.25
            ):
                # A raw ASR segment can begin well before a sung onset while
                # still ending at the right occurrence. Let the CTC clock
                # select that physical occurrence instead of discarding it
                # because one raw boundary is far away.
                preferred = nearest_physical
                anchors = [
                    anchor
                    for anchor in all_physical_anchors
                    if abs(anchor[0].start - preferred[0].start) <= 1.5
                    and abs(anchor[0].end - preferred[0].end) <= 1.5
                ]
                if ctc_sequence_supported:
                    anchors.append((ctc_anchor, ctc_source))
        force_source = "forcealign:vocal_stem"
        force_match = per_source_matches.get(force_source, {}).get(lyric_index)
        if force_match is not None and force_source in asr_outputs:
            force_anchor = prepared_tokens_by_source.get(
                force_source, asr_outputs[force_source]
            )[force_match[0]]
            nearest_physical = min(
                all_physical_anchors,
                key=lambda anchor: min(
                    abs(force_anchor.start - anchor[0].start),
                    abs(force_anchor.end - anchor[0].end),
                ),
            )
            force_clock_delta = min(
                abs(force_anchor.start - nearest_physical[0].start),
                abs(force_anchor.end - nearest_physical[0].end),
            )
            if force_anchor.confidence >= 0.55 and force_clock_delta <= 1.75:
                # ForceAlign often recovers held ends better than ASR, but its
                # full-song path can drift at omissions. Use it to widen the
                # melody boundary targets only when it remains near the raw
                # physical occurrence; it is not authoritative by itself.
                anchors.append((force_anchor, force_source))
        # A second physical clock is useful only when it describes the same
        # occurrence. Do not let a repeated CTC/ASR occurrence widen the
        # melody search window.  A source can nevertheless miss or merge a
        # local word while the other physical recording has the right
        # occurrence, so an opt-in diagnostic mode can retain both clocks.
        use_all_physical_anchors = (
            os.environ.get("ALIGNMENT_USE_ALL_PHYSICAL_ANCHORS", "0")
            .strip()
            .casefold()
            in {"1", "true", "yes", "on"}
        )
        physical_anchor_pool = (
            all_physical_anchors
            if use_all_physical_anchors and continuity_anchor is None
            else anchors
        )
        return [
            anchor
            for anchor in physical_anchor_pool
            if "ctc" in anchor[1].casefold()
            or use_all_physical_anchors
            or (
                abs(anchor[0].start - preferred[0].start) <= 1.5
                and abs(anchor[0].end - preferred[0].end) <= 1.5
            )
        ]

    melody_sources = {
        source: words
        for source, words in asr_outputs.items()
        if any(token.evidence_type.casefold() == "melody" for token in words)
    }
    token_index_by_id = {token.id: index for index, token in enumerate(tokens)}
    melody_anchor_starts: dict[tuple[str, int], float] = {}
    for source, matches in anchor_matches_by_source.items():
        words = prepared_tokens_by_source.get(source, asr_outputs.get(source, ()))
        for lyric_index, (asr_index, _similarity) in matches.items():
            if 0 <= asr_index < len(words) and _valid_span(words[asr_index].start, words[asr_index].end):
                melody_anchor_starts[(source, lyric_index)] = float(words[asr_index].start)
    melody_boundaries_by_source: dict[str, list[float]] = {}
    for source, melody_events in melody_sources.items():
        if not melody_events:
            continue
        physical = physical_source(source, melody_events[0])
        melody_boundaries_by_source.setdefault(physical, []).extend(
            float(value)
            for event in melody_events
            for value in (event.start, event.end)
            if math.isfinite(float(value)) and float(value) >= 0.0
        )
    melody_boundaries_by_source = {
        physical: sorted(values)
        for physical, values in melody_boundaries_by_source.items()
    }
    # A candidate end that falls inside a strong vocal note is usually the
    # early Whisper boundary of a held word, not a real word handoff.  Keep
    # this evidence tied to the physical vocal recording; accompaniment can
    # contain overlapping notes that must not veto a vocal boundary.
    vocal_melody_events = [
        event
        for source, events in melody_sources.items()
        for event in events
        if physical_source(source, event).casefold() == "vocal_stem"
        and _valid_span(event.start, event.end)
    ]

    def melody_end_inside_confidence(end: float) -> float | None:
        if not vocal_melody_events or not math.isfinite(end):
            return None
        try:
            boundary_margin = float(
                os.environ.get("ALIGNMENT_MELODY_END_INSIDE_MARGIN", "0.025")
            )
        except ValueError:
            boundary_margin = 0.025
        if not math.isfinite(boundary_margin) or boundary_margin < 0.0:
            boundary_margin = 0.025
        boundary_margin = min(0.20, boundary_margin)
        crossing_confidence = [
            max(0.0, min(1.0, float(event.confidence)))
            for event in vocal_melody_events
            if float(event.start) < end - boundary_margin
            and float(event.end) > end + boundary_margin
        ]
        return max(crossing_confidence, default=None)

    def melody_source_agreement(source: str, start: float, end: float) -> float:
        """Reward a boundary pair repeated by the other physical recording."""

        other_sources = [
            values
            for physical, values in melody_boundaries_by_source.items()
            if physical != source
        ]
        if not other_sources:
            return 0.5

        def has_nearby(values: Sequence[float], target: float, tolerance: float) -> bool:
            left = bisect_left(values, target - tolerance)
            return left < len(values) and values[left] <= target + tolerance

        return 1.0 if any(
            has_nearby(values, start, 0.12)
            and has_nearby(values, end, 0.12)
            for values in other_sources
        ) else 0.35

    for source, melody_events in melody_sources.items():
        boundary_values = sorted({
            round(value, 4)
            for event in melody_events
            for value in (event.start, event.end)
            if math.isfinite(float(value)) and float(value) >= 0.0
        })
        confidence_by_boundary: dict[float, float] = {}
        for event in melody_events:
            event_confidence = max(0.0, min(1.0, float(event.confidence)))
            start_key = round(float(event.start), 4)
            end_key = round(float(event.end), 4)
            confidence_by_boundary[start_key] = max(
                confidence_by_boundary.get(start_key, 0.0), event_confidence
            )
            confidence_by_boundary[end_key] = max(
                confidence_by_boundary.get(end_key, 0.0), event_confidence
            )

        try:
            boundary_limit = int(
                os.environ.get("ALIGNMENT_MELODY_BOUNDARY_LIMIT", "48")
            )
        except ValueError:
            boundary_limit = 48
        boundary_limit = max(8, min(256, boundary_limit))

        def nearby_boundaries(target: float, radius: float) -> list[float]:
            left = bisect_left(boundary_values, target - radius)
            right = bisect_right(boundary_values, target + radius)
            values = boundary_values[left:right]
            if len(values) > boundary_limit:
                values = sorted(values, key=lambda value: abs(value - target))[:boundary_limit]
            return values

        asr_start_extension = float(os.environ.get("ALIGNMENT_MELODY_START_EXTENSION", "0.35"))
        # Whisper word ends often stop at the first consonant, but extending
        # every melody target by nearly half a second makes adjacent sung
        # words look like one sustained note. Keep a modest tail allowance;
        # callers can still override it for unusually legato material.
        asr_end_extension = float(os.environ.get("ALIGNMENT_MELODY_END_EXTENSION", "0.30"))
        try:
            end_anchor_weight = float(
                os.environ.get("ALIGNMENT_MELODY_END_ANCHOR_WEIGHT", "0.5")
            )
        except ValueError:
            end_anchor_weight = 0.5
        if not math.isfinite(end_anchor_weight) or end_anchor_weight < 0.0:
            end_anchor_weight = 1.0
        end_anchor_weight = min(1.0, end_anchor_weight)

        for lyric_index, token in enumerate(tokens):
            anchors = lexical_anchors(lyric_index)
            if not anchors:
                continue
            clock_anchors = anchors
            # ASR/CTC word ends are usually early on held notes. Keep both
            # the raw clock and a conservative musical extension in the
            # candidate neighborhood; selection remains global and can use
            # continuity to decide between repeated occurrences.
            start_targets = [
                value
                for anchor, source_name in clock_anchors
                for value in (
                    float(anchor.start)
                    + (
                        0.05
                        if "ctc" in source_name
                        else asr_start_extension
                    ),
                )
            ]
            end_targets = [
                value
                for anchor, source_name in clock_anchors
                for value in (
                    float(anchor.end)
                    + (
                        0.12
                        if "ctc" in source_name
                        else asr_end_extension
                    ),
                )
            ]
            activity_start_targets = [
                float(span_start) + offset
                for anchor, _source_name in clock_anchors
                for span_start, span_end in activity_spans
                if (
                    span_end > span_start
                    and 0.0 <= float(span_start) - float(anchor.end) <= 0.50
                )
                for offset in (0.0, 0.04)
            ]
            if activity_start_targets:
                start_targets.extend(activity_start_targets)
            # CTC often assigns a held vowel's end to the wrong early frame.
            # When the current and following CTC words form a locally ordered
            # run with a real gap, the following onset is a useful end clock
            # for the current word. Keep this narrowly scoped: ordinary CTC
            # spans and reversed runs must not widen every candidate window.
            ctc_current = ctc_sequence_anchor(lyric_index)
            ctc_following = ctc_sequence_anchor(lyric_index + 1)
            late_ctc_end_targets: list[float] = []
            physical_clock_anchors = [
                anchor
                for anchor, source_name in clock_anchors
                if "ctc" not in source_name.casefold()
            ]
            if ctc_current is not None and physical_clock_anchors:
                physical_clock = physical_clock_anchors[0]
                ctc_late_start = ctc_current.start - physical_clock.end
                ctc_activity_supported = any(
                    float(span_start) - 0.12 <= ctc_current.start <= float(span_end) + 0.12
                    for span_start, span_end in activity_spans
                    if span_end > span_start
                )
                if (
                    ctc_current.confidence >= 0.75
                    and 0.55 <= ctc_late_start <= 1.50
                    and ctc_activity_supported
                    and ctc_sequence_is_monotonic(lyric_index)
                ):
                    late_ctc_end_targets = [
                        ctc_current.end,
                        ctc_current.end + 0.12,
                    ]
                    end_targets.extend(late_ctc_end_targets)
            neighboring_end_targets: list[float] = []
            use_ctc_neighboring_end = (
                os.environ.get("ALIGNMENT_USE_CTC_NEIGHBOR_END", "0")
                .strip()
                .casefold()
                in {"1", "true", "yes", "on"}
            )
            if (
                use_ctc_neighboring_end
                and ctc_current is not None
                and ctc_following is not None
                and ctc_current.end - ctc_current.start <= 0.12
                and ctc_following.start - ctc_current.end > 0.25
                and ctc_sequence_is_monotonic(lyric_index)
                and ctc_sequence_is_monotonic(lyric_index + 1)
            ):
                neighboring_end_targets = [
                    ctc_following.start - 0.04,
                    ctc_following.start,
                    ctc_following.start + 0.04,
                ]
                end_targets.extend(neighboring_end_targets)
            ctc_end_hint_targets: list[float] = []
            use_ctc_end_hint = (
                os.environ.get("ALIGNMENT_USE_CTC_END_HINT", "0")
                .strip()
                .casefold()
                in {"1", "true", "yes", "on"}
            )
            if use_ctc_end_hint and ctc_current is not None and physical_clock_anchors:
                physical_clock = physical_clock_anchors[0]
                ctc_end_delta = float(ctc_current.end) - float(physical_clock.end)
                ctc_start_delta = float(ctc_current.start) - float(physical_clock.start)
                # CTC frequently preserves the end of a held vowel while
                # assigning its onset to a neighboring frame. Admit that
                # end-only clock only when it is still a local, ordered
                # occurrence and the melody lattice has a nearby boundary.
                # A repeated-section CTC jump is rejected by the start
                # bound and the local monotonicity check.
                has_nearby_melody_boundary = any(
                    abs(float(boundary) - float(ctc_current.end)) <= 0.50
                    for boundary in boundary_values
                )
                if (
                    ctc_current.confidence >= 0.20
                    and -0.90 <= ctc_start_delta <= 1.25
                    and 0.35 <= ctc_end_delta <= 1.75
                    and ctc_sequence_is_monotonic(lyric_index)
                    and has_nearby_melody_boundary
                ):
                    ctc_end_hint_targets = [
                        float(ctc_current.end),
                        float(ctc_current.end) + 0.08,
                        float(ctc_current.end) + 0.16,
                    ]
                    end_targets.extend(ctc_end_hint_targets)
            # Once a CTC span has passed the physical-source compatibility
            # gate in ``lexical_anchors``, it is the better local boundary
            # clock than raw ASR. Raw ASR often stops at the first consonant
            # of a held word; scoring a melody end against that early clock
            # systematically selects a short note decoy. Keep all clocks in
            # the search neighborhood, but score the boundary pair against
            # the validated CTC targets when they exist.
            ctc_anchors = [
                anchor
                for anchor, source_name in anchors
                if "ctc" in source_name.casefold()
            ]
            use_ctc_scoring_clock = (
                os.environ.get("ALIGNMENT_USE_CTC_SCORING_CLOCK", "1")
                .strip()
                .casefold()
                in {"1", "true", "yes", "on"}
            )
            use_ctc_scoring_start = (
                os.environ.get(
                    "ALIGNMENT_USE_CTC_SCORING_START",
                    "1" if use_ctc_scoring_clock else "0",
                )
                .strip()
                .casefold()
                in {"1", "true", "yes", "on"}
            )
            use_ctc_scoring_end = (
                os.environ.get(
                    "ALIGNMENT_USE_CTC_SCORING_END",
                    "1" if use_ctc_scoring_clock else "0",
                )
                .strip()
                .casefold()
                in {"1", "true", "yes", "on"}
            )
            # The first word has no preceding lexical clock. CTC alignment
            # can attach its opening silence to that token, so keep the
            # physical ASR clock as the start-of-song reference.
            if lyric_index == 0:
                ctc_anchors = []
            physical_scoring_start_targets = [
                float(anchor.start) + asr_start_extension
                for anchor, source_name in anchors
                if "ctc" not in source_name.casefold()
            ]
            physical_scoring_end_targets = [
                float(anchor.end) + asr_end_extension
                for anchor, source_name in anchors
                if "ctc" not in source_name.casefold()
            ]
            scoring_start_targets = [
                value
                for anchor in ctc_anchors
                for value in (float(anchor.start), float(anchor.start) + 0.05)
            ] if use_ctc_scoring_start else []
            scoring_start_targets = (
                scoring_start_targets
                or physical_scoring_start_targets
                or start_targets
            )
            if activity_start_targets:
                scoring_start_targets.extend(activity_start_targets)
            scoring_end_targets = [
                value
                for anchor in ctc_anchors
                for value in (float(anchor.end), float(anchor.end) + 0.12)
            ] if use_ctc_scoring_end else []
            scoring_end_targets = (
                scoring_end_targets
                or physical_scoring_end_targets
                or end_targets
            )
            if late_ctc_end_targets:
                scoring_end_targets.extend(late_ctc_end_targets)
            if activity_start_targets and ctc_anchors and use_ctc_scoring_end:
                # A fresh vocal island is evidence that the CTC occurrence is
                # attached to an earlier syllable or repeated section. Keep
                # the physical/ASR end clocks in the score as well; otherwise
                # every melody span crossing that island is penalized against
                # the stale CTC end even when its onset is musically correct.
                scoring_end_targets.extend(end_targets)
            if ctc_end_hint_targets:
                scoring_end_targets.extend(ctc_end_hint_targets)
            if neighboring_end_targets:
                # Once a large, locally ordered CTC gap identifies the next
                # onset, the current CTC end is precisely the unreliable
                # boundary we are correcting. Do not let a short decoy keep
                # a perfect score merely because it is close to that stale
                # end; score the pair against the neighboring clock.
                scoring_end_targets = neighboring_end_targets
            try:
                boundary_search_radius = float(
                    os.environ.get("ALIGNMENT_MELODY_BOUNDARY_RADIUS", "2.0")
                )
            except ValueError:
                boundary_search_radius = 2.0
            if not math.isfinite(boundary_search_radius) or boundary_search_radius <= 0.0:
                boundary_search_radius = 2.0
            boundary_search_radius = min(3.0, max(0.25, boundary_search_radius))
            start_values = sorted({
                value
                for target in start_targets
                for value in nearby_boundaries(target, boundary_search_radius)
            })
            end_values = sorted({
                value
                for target in end_targets
                for value in nearby_boundaries(target, boundary_search_radius)
            })
            for start in start_values:
                for end in end_values:
                    if end < start or end - start > 4.0:
                        continue
                    start_boundary_delta = min(
                        abs(start - target) for target in scoring_start_targets
                    )
                    end_boundary_delta = min(
                        abs(end - target) for target in scoring_end_targets
                    )
                    pair_end_anchor_weight = end_anchor_weight
                    if pair_end_anchor_weight < 1.0 and (
                        end - start < 0.65 or start_boundary_delta > 0.18
                    ):
                        pair_end_anchor_weight = 1.0
                    boundary_delta = max(
                        start_boundary_delta,
                        pair_end_anchor_weight * end_boundary_delta,
                    )
                    candidates.append(
                        AlignmentCandidate(
                            token_id=token.id,
                            start=start,
                            end=end,
                            model="basic-pitch",
                            audio_source=physical_source(source, melody_events[0]) if melody_events else source,
                            text_score=1.0,
                            # A held-note tail often has a lower detector
                            # confidence than its onset even when the
                            # boundary is musically correct. Using the
                            # minimum makes those long, useful spans lose to
                            # short harmonic decoys. The average remains
                            # conservative while allowing the two endpoints
                            # to contribute independently.
                            model_confidence=(
                                confidence_by_boundary.get(round(start, 4), 0.5)
                                + confidence_by_boundary.get(round(end, 4), 0.5)
                            )
                            / 2.0,
                            anchor_score=max(0.0, 1.0 - boundary_delta / 0.8),
                            source_agreement=melody_source_agreement(
                                physical_source(source, melody_events[0]) if melody_events else source,
                                start,
                                end,
                            ),
                            quality=1.0,
                            reasons=[
                                f"{source} melodic boundary candidate",
                                f"boundary delta from lexical anchor {boundary_delta:.3f}s",
                            ],
                            candidate_type="melody",
                            temporal_distance=boundary_delta,
                            temporal_start_offset=start - min(
                                scoring_start_targets,
                                key=lambda target: abs(start - target),
                            ),
                            temporal_end_offset=end - min(
                                scoring_end_targets,
                                key=lambda target: abs(end - target),
                            ),
                            melody_end_inside_confidence=melody_end_inside_confidence(end),
                        )
                    )

    if melody_sources:
        try:
            minimum_start_delay = float(
                os.environ.get("ALIGNMENT_MELODY_MIN_START_DELAY", "0.05")
            )
        except ValueError:
            minimum_start_delay = 0.05
        if not math.isfinite(minimum_start_delay) or minimum_start_delay < 0.0:
            minimum_start_delay = 0.05
        minimum_start_delay = min(0.75, minimum_start_delay)
        # Basic Pitch boundaries from the preceding word can otherwise win
        # when CTC has drifted to a repeated occurrence. A melody candidate
        # must not begin materially before the physical ASR occurrence for
        # the same token/source. A short post-ASR guard also removes early
        # harmonic decoys; late melody onsets remain eligible because ASR
        # timestamps often include leading consonants or breath.
        candidates = [
            candidate
            for candidate in candidates
            if not (
                candidate.candidate_type.casefold() in {"melody", "pitch", "basic_pitch"}
                and (
                    anchor_start := melody_anchor_starts.get(
                        (candidate.audio_source.casefold(), token_index_by_id.get(candidate.token_id, -1))
                    )
                ) is not None
                and candidate.start < anchor_start + minimum_start_delay
            )
        ]
        # Dense pitch extraction can produce thousands of overlapping
        # harmonic events. Quantize equivalent boundary pairs before ranking
        # them; sub-frame differences are not meaningful at karaoke display
        # resolution, but every unbounded cross-product makes the global
        # selector slower and more likely to keep a harmonic decoy.
        boundary_quantum_s = 0.08
        melody_by_token_source: dict[tuple[str, str], list[AlignmentCandidate]] = {}
        non_melody_candidates: list[AlignmentCandidate] = []
        for candidate in candidates:
            if candidate.candidate_type == "melody":
                melody_by_token_source.setdefault(
                    (candidate.token_id, candidate.audio_source),
                    [],
                ).append(candidate)
            else:
                non_melody_candidates.append(candidate)
        candidates = [*non_melody_candidates]
        for (_token_id, _audio_source), token_candidates in melody_by_token_source.items():
            grouped: dict[tuple[int, int], AlignmentCandidate] = {}
            for candidate in token_candidates:
                key = (
                    round(candidate.start / boundary_quantum_s),
                    round(candidate.end / boundary_quantum_s),
                )
                previous = grouped.get(key)
                if previous is None or score_alignment_candidate(candidate).score > score_alignment_candidate(previous).score:
                    grouped[key] = candidate
            ranked = sorted(
                grouped.values(),
                key=lambda candidate: score_alignment_candidate(candidate).score,
                reverse=True,
            )
            retained: list[AlignmentCandidate] = []
            retained_ids: set[int] = set()

            def retain(candidate: AlignmentCandidate) -> None:
                key = id(candidate)
                if key not in retained_ids:
                    retained_ids.add(key)
                    retained.append(candidate)

            # Confidence is useful for tie-breaking, but using it alone
            # discards low-confidence vocal boundaries in dense harmonic
            # regions. Keep the strongest local candidates and a few options
            # from every temporal-distance bucket so the global path can
            # recover a systematic ASR clock offset without allowing one
            # word to dominate the whole song's search. The short escape
            # hatches are selected inside each bucket: a musically correct
            # onset can be far from a stale ASR clock while still being one
            # of the shortest plausible note spans.
            distance_buckets: dict[int, list[AlignmentCandidate]] = {}
            for candidate in ranked:
                distance = candidate.temporal_distance
                if distance is None:
                    continue
                bucket = min(15, max(0, int(float(distance) / 0.1)))
                distance_buckets.setdefault(bucket, []).append(candidate)
            for bucket in sorted(distance_buckets):
                shortest = sorted(
                    distance_buckets[bucket],
                    key=lambda candidate: (
                        candidate.end - candidate.start,
                        candidate.start,
                        candidate.end,
                    ),
                )
                for candidate in shortest[:4]:
                    retain(candidate)
            for candidate in ranked:
                retain(candidate)
                if len(retained) >= 256:
                    break
            try:
                max_retained_candidates = int(
                    os.environ.get("ALIGNMENT_MELODY_MAX_CANDIDATES", "256")
                )
            except ValueError:
                max_retained_candidates = 256
            max_retained_candidates = max(32, min(512, max_retained_candidates))
            candidates.extend(retained[:max_retained_candidates])

    for source, words in asr_outputs.items():
        if source in melody_sources:
            continue
        matches = per_source_matches[source]
        prepared_words = prepared_tokens_by_source.get(source, tuple(words))
        for lyric_index, (asr_index, similarity) in matches.items():
            if not 0 <= asr_index < len(prepared_words):
                continue
            asr = prepared_words[asr_index]
            candidate_source = physical_source(source, asr)
            candidate_start = float(asr.start)
            candidate_end = float(asr.end)
            warped_forced = False
            if (
                source == force_source
                and asr.sequence_index is not None
                and int(asr.sequence_index) in forced_clock_warps
            ):
                candidate_start, candidate_end = forced_clock_warps[int(asr.sequence_index)]
                warped_forced = True
            # A vocal CTC token is itself a bounded forced-timing result. A
            # raw ASR segment may start a syllable or two earlier, so using
            # that raw span as a hard compatibility gate would discard the
            # most useful CTC correction. The original-mix CTC stream remains
            # gated by the physical ASR clock because it is more vulnerable
            # to accompaniment repetitions.
            ctc_clock_supported = any(
                min(
                    abs(asr.start - (physical_anchor := source_token(physical, physical_match[0])).start),
                    abs(asr.end - physical_anchor.end),
                ) <= 1.0
                and max(
                    physical_anchor.start - asr.end,
                    asr.start - physical_anchor.end,
                    0.0,
                ) <= 0.25
                for physical, physical_matches in anchor_matches_by_source.items()
                if (physical_match := physical_matches.get(lyric_index)) is not None
                and physical in asr_outputs
            )
            ctc_self_anchor = (
                source == "whisperx_ctc:vocal_stem"
                and asr.evidence_type.casefold() == "ctc"
                and ctc_clock_supported
            )
            anchor_match = (
                None
                if ctc_self_anchor
                else anchor_matches_by_source.get(candidate_source, {}).get(lyric_index)
            )
            anchor = asr if ctc_self_anchor else (
                source_token(candidate_source, anchor_match[0])
                if anchor_match is not None and candidate_source in asr_outputs
                else None
            )
            indexed_forced_evidence = (
                asr.evidence_type.casefold() in {"forced", "alignment"}
                and asr.sequence_index is not None
            )
            forced_occurrence_conflict = False
            if indexed_forced_evidence and anchor is not None:
                # Sequence indexing proves which lyric token the forced
                # trellis was asked to label, not which *occurrence* it
                # actually found.  On long songs the trellis can spend an
                # instrumental gap on the transcript and emerge many seconds
                # late.  A high-confidence lexical ASR clock is independent
                # occurrence evidence; reject only an unmistakable large
                # disagreement and leave nearby held-word boundary differences
                # to the forced candidate.
                forced_clock_delta = max(
                    abs(float(asr.start) - float(anchor.start)),
                    abs(float(asr.end) - float(anchor.end)),
                )
                forced_duration = float(asr.end - asr.start)
                previous_forced = next(
                    (
                        token
                        for token in reversed(force_words[:asr_index])
                        if token.sequence_index is not None
                    ),
                    None,
                )
                next_forced = next(
                    (
                        token
                        for token in force_words[asr_index + 1 :]
                        if token.sequence_index is not None
                    ),
                    None,
                )
                occurrence_supported = (
                    asr.sequence_index is not None
                    and int(asr.sequence_index) in forced_clock_warps
                ) or (
                    previous_forced is None
                    or float(anchor.start) >= float(previous_forced.start) - 0.25
                ) and (
                    next_forced is None
                    or float(anchor.start) <= float(next_forced.start) + 0.25
                )
                if (
                    anchor.confidence >= 0.70
                    and forced_clock_delta > 6.0
                    and forced_duration <= 2.5
                    and occurrence_supported
                ):
                    forced_occurrence_conflict = True
                    temporal_anchor_score = 0.0
                    temporal_reason = (
                        "indexed forced timing conflicts with same-source ASR occurrence "
                        f"by {forced_clock_delta:.3f}s"
                    )
                else:
                    temporal_anchor_score = 1.0
                    temporal_reason = "indexed forced timing is independent of ASR clock"
            elif indexed_forced_evidence:
                # A forced aligner is already constrained by the supplied
                # lyric sequence.  Its indexed span is independent evidence;
                # the raw ASR clock is only an occurrence hint and must not
                # veto a valid forced result when a held word has a longer
                # end or ASR selected a neighboring repetition.
                temporal_anchor_score = 1.0
                temporal_reason = "indexed forced timing is independent of ASR clock"
            elif anchor is None:
                temporal_anchor_score = 0.5
                temporal_reason = "no same-source ASR anchor available"
            elif ctc_self_anchor:
                temporal_anchor_score = 1.0
                temporal_reason = "vocal CTC candidate carries its own lexical clock"
            else:
                boundary_distance = max(
                    abs(candidate_start - anchor.start),
                    abs(candidate_end - anchor.end),
                )
                span_gap = max(
                    anchor.start - asr.end,
                    asr.start - anchor.end,
                    0.0,
                )
                temporal_anchor_score = max(0.0, 1.0 - boundary_distance / 0.8)
                if span_gap > 0.55:
                    temporal_anchor_score = 0.0
                temporal_reason = (
                    f"same-source anchor boundary delta {boundary_distance:.3f}s"
                )
            agrees_with_other_source = any(
                lyric_index in other_matches
                and (other_anchor := source_token(other, other_matches[lyric_index][0])) is not None
                and physical_source(other, other_anchor) != candidate_source
                and abs(asr.start - other_anchor.start) < 0.35
                and abs(asr.end - other_anchor.end) < 0.5
                for other, other_matches in per_source_matches.items()
                if other != source
            )
            token = tokens[lyric_index]
            if forced_occurrence_conflict and anchor is not None:
                # Keep the lexical ASR clock as an explicit selectable
                # occurrence candidate.  ``anchor`` entries are intentionally
                # excluded from the normal candidate lattice, but once the
                # indexed trellis demonstrably chose another repetition this
                # clock is the safer timing source for the token.
                candidates.append(
                    AlignmentCandidate(
                        token_id=token.id,
                        start=float(anchor.start),
                        end=float(anchor.end),
                        model=anchor.model,
                        audio_source=candidate_source,
                        text_score=max(similarity, 0.95),
                        model_confidence=max(float(anchor.confidence), 0.35),
                        anchor_score=1.0,
                        source_agreement=0.8,
                        reasons=[
                            f"{source} occurrence clock",
                            "recovered same-source ASR occurrence after forced drift",
                        ],
                        candidate_type="occurrence",
                    )
                )
            candidates.append(
                AlignmentCandidate(
                    token_id=token.id,
                    start=candidate_start,
                    end=candidate_end,
                    model=asr.model,
                    audio_source=candidate_source,
                    text_score=similarity,
                    model_confidence=asr.confidence,
                    anchor_score=temporal_anchor_score,
                    source_agreement=1.0 if agrees_with_other_source else 0.5,
                    reasons=[
                        f"{source} ASR candidate",
                        "exact lexical match" if similarity == 1.0 else "fuzzy lexical match",
                        *(
                            ["piecewise forced clock corrected by ASR anchors"]
                            if warped_forced
                            else []
                        ),
                        temporal_reason,
                    ],
                    candidate_type=asr.evidence_type,
                )
            )
    # If the vocal ASR has no lexical anchor for the opening section, a
    # full-track CTC pass can confidently place those words on accompaniment
    # transients.  Keep a bounded look-back so genuine words before the first
    # recognized anchor remain alignable, but discard candidates that are
    # implausibly far into an instrumental intro.  This is deliberately based
    # on the vocal-stem clock, not the original mix, which is more likely to
    # hallucinate speech over drums and harmony.
    vocal_anchor_starts = [
        float(token.start)
        for token in asr_outputs.get("vocal_stem", ())
        if token.evidence_type.casefold() not in {"melody", "pitch", "basic_pitch"}
        and _valid_span(token.start, token.end)
    ]
    if vocal_anchor_starts:
        try:
            max_leading_gap = float(
                os.environ.get("ALIGNMENT_CTC_MAX_LEADING_GAP", "18.0")
            )
        except ValueError:
            max_leading_gap = 18.0
        if not math.isfinite(max_leading_gap) or max_leading_gap < 0.0:
            max_leading_gap = 18.0
        max_leading_gap = min(60.0, max_leading_gap)
        earliest_allowed_ctc = max(0.0, min(vocal_anchor_starts) - max_leading_gap)
        candidates = [
            candidate
            for candidate in candidates
            if candidate.candidate_type.casefold() != "ctc"
            or candidate.start >= earliest_allowed_ctc
        ]
    # CTC timestamps are useful as a local clock for melodic boundary search,
    # but they are not reliable enough to compete with the melody lattice as
    # final spans on the same token. Keep them only for tokens without a
    # melody candidate; this avoids a noisy CTC stream overriding an
    # acoustically supported boundary pair.
    if melody_sources:
        melody_token_ids = {
            candidate.token_id
            for candidate in candidates
            if candidate.candidate_type.casefold() == "melody"
        }
        keep_ctc_with_melody = (
            os.environ.get("ALIGNMENT_KEEP_CTC_WITH_MELODY", "0")
            .strip()
            .casefold()
            in {"1", "true", "yes", "on"}
        )
        if not keep_ctc_with_melody:
            candidates = [
                candidate
                for candidate in candidates
                if candidate.candidate_type.casefold() != "ctc"
                or candidate.token_id not in melody_token_ids
            ]
        preferred_melody_source = os.environ.get("ALIGNMENT_MELODY_SOURCE", "").strip().casefold()
        if preferred_melody_source in {"vocal_stem", "original_mix"}:
            candidates = [
                candidate
                for candidate in candidates
                if candidate.candidate_type != "melody"
                or candidate.audio_source.casefold() == preferred_melody_source
            ]
    diagnostics = {
        "asr_sources": {
            source: {
                "word_count": len(words),
                "matched_token_count": len(per_source_matches.get(source, {})),
            }
            for source, words in asr_outputs.items()
        },
        "primary_audio_source": primary_source,
    }
    return primary, candidates, diagnostics


def run_local_alignment(
    request: AlignmentWorkerRequest,
    asr_outputs: Mapping[str, Sequence[ASRToken]],
    *,
    activity_spans: Sequence[tuple[float, float]] = (),
    activity_times: Sequence[float] = (),
    activity_values: Sequence[float | bool] = (),
) -> AlignmentWorkerResponse:
    """Run deterministic selection over model evidence in a local process."""

    if not any(asr_outputs.values()):
        raise AlignmentWorkerError("No usable model evidence was provided to the alignment worker")
    tokens = [LyricToken(**_model_dict(token)) for token in request.tokens]
    effective_activity_spans = list(activity_spans or request.activity_spans)
    # If separation misses an opening vocal, a confident ASR clock can restore
    # only that first contiguous island. Keep the fallback bounded and enable
    # it only when ASR leads the detector by a clear margin.
    if effective_activity_spans:
        first_activity_start = min(start for start, _end in effective_activity_spans)
        anchors = sorted(
            (
                token
                for token in asr_outputs.get("vocal_stem", ())
                if token.evidence_type.casefold() not in {"melody", "pitch", "basic_pitch"}
                and token.confidence >= 0.35
                and _valid_span(token.start, token.end)
            ),
            key=lambda token: token.start,
        )
        if anchors and first_activity_start - anchors[0].start >= 2.0:
            island_start = max(0.0, float(anchors[0].start) - 0.12)
            island_end = float(anchors[0].end)
            for anchor in anchors[1:]:
                if float(anchor.start) - island_end > 1.5:
                    break
                island_end = max(island_end, float(anchor.end))
            effective_activity_spans = sorted(
                [*effective_activity_spans, (island_start, island_end)]
            )
            merged_spans: list[tuple[float, float]] = []
            for start, end in effective_activity_spans:
                if merged_spans and start <= merged_spans[-1][1] + 0.02:
                    merged_spans[-1] = (merged_spans[-1][0], max(merged_spans[-1][1], end))
                else:
                    merged_spans.append((start, end))
            effective_activity_spans = merged_spans
    primary, candidates, diagnostics = _candidate_sources(
        request,
        asr_outputs,
        activity_spans=effective_activity_spans,
    )
    alignments, global_result = build_word_alignments(
        tokens,
        primary,
        request.alignment_run_id,
        candidates=candidates,
        activity_spans=effective_activity_spans,
        timing_grid_step=request.timing_grid_step,
        timing_grid_offset=request.timing_grid_offset,
        activity_times=activity_times,
        activity_values=activity_values,
    )
    alignments = split_repeated_estimated_alignments(
        tokens,
        alignments,
        activity_times,
        activity_values,
    )
    alignments = repair_delayed_forced_onsets(
        tokens,
        alignments,
        effective_activity_spans,
    )
    alignments = trim_selected_melody_gaps(alignments, asr_outputs)
    alignments = trim_alignment_tails_to_activity(
        alignments,
        effective_activity_spans,
    )
    alignments = trim_alignment_tails_to_strong_activity(
        alignments,
        activity_times,
        activity_values,
    )
    alignments = snap_alignment_starts_to_activity_onsets(
        alignments,
        activity_times,
        activity_values,
    )
    alignments = trim_alignment_tails_to_strong_activity(
        alignments,
        activity_times,
        activity_values,
    )
    alignments = repair_unmatched_alignment_gaps(
        alignments,
        candidates,
        activity_spans=effective_activity_spans,
    )
    # Candidate recovery can replace a useful fallback clock with a delayed
    # forced fragment; reapply the continuous-activity onset repair after
    # that recovery pass as well.
    alignments = repair_delayed_forced_onsets(
        tokens,
        alignments,
        effective_activity_spans,
    )
    # Gap repair can restore a bounded model candidate after the first
    # activity pass. Apply the conservative tail bound once more so a
    # recovered long span cannot consume an isolated silent interval.
    alignments = trim_alignment_tails_to_activity(
        alignments,
        effective_activity_spans,
    )
    alignments = trim_alignment_tails_to_strong_activity(
        alignments,
        activity_times,
        activity_values,
    )
    alignments = trim_forced_boundary_latency(
        alignments,
        activity_spans=effective_activity_spans,
    )
    diagnostics.update(
        {
            "strategy": (
                "sofa-primary-global-match-and-candidate-score"
                if diagnostics.get("primary_audio_source") == "sofa:vocal_stem"
                else "legacy-global-match-and-candidate-score"
            ),
            "global_score": global_result.score,
            "candidate_diagnostics": [
                {
                    **asdict(candidate),
                    "score": score_alignment_candidate(candidate).score,
                }
                for candidate in candidates
            ],
            "unmatched_token_ids": [
                token.id
                for token, alignment in zip(tokens, alignments)
                if alignment.status == AlignmentStatus.UNMATCHED
            ],
        }
    )
    response = AlignmentWorkerResponse(
        job_id=request.job_id,
        alignment_run_id=request.alignment_run_id,
        lyric_revision_id=request.lyric_revision_id,
        status="completed",
        alignments=[_alignment_to_payload(alignment) for alignment in alignments],
        diagnostics=diagnostics,
    )
    return validate_worker_response(response, request)


def collect_model_evidence(
    request: AlignmentWorkerRequest,
    *,
    fallback_words: Sequence[ASRToken] = (),
) -> tuple[dict[str, list[ASRToken]], dict[str, str]]:
    """Run configured model adapters for both physical audio sources."""

    from processing.model_adapters import (
        align_with_forcealign,
        align_with_qwen_forced,
        align_with_sofa,
        align_with_whisperx,
        basic_pitch_enabled,
        forcealign_enabled,
        detect_basic_pitch_boundaries,
        melody_blend_enabled,
        qwen_asr_enabled,
        run_serialized_model_call,
        sofa_aligner_enabled,
        transcribe_with_qwen,
        whisperx_enabled,
        write_melody_blend_audio,
    )

    from processing.lyrics import LyricsParams, extract_lyrics

    def env_bool(name: str, default: bool) -> bool:
        value = os.environ.get(name)
        if value is None:
            return default
        return value.strip().casefold() not in {"0", "false", "off", "no", "disabled"}

    def env_float(name: str, default: float) -> float:
        try:
            value = float(os.environ.get(name, str(default)))
        except (TypeError, ValueError):
            return default
        return value if math.isfinite(value) else default

    def env_int(name: str, default: int) -> int:
        try:
            value = int(os.environ.get(name, str(default)))
        except (TypeError, ValueError):
            return default
        return value if value >= 0 else default

    evidence: dict[str, list[ASRToken]] = {}
    errors: dict[str, str] = {}
    # Timing references are deliberately not read here. They belong to the
    # private benchmark harness, where they are injected explicitly as
    # evidence. Production alignment must always derive timing from audio.
    source_paths = (
        ("vocal_stem", request.vocal_stem_audio_ref),
        ("original_mix", request.original_audio_ref),
    )
    ctc_failed = False
    for source, path in source_paths:
        if not path or not os.path.exists(path):
            errors[source] = "audio reference is missing on the worker host"
            continue
        try:
            if qwen_asr_enabled():
                tokens = run_serialized_model_call(transcribe_with_qwen, path, request.language)
            else:
                params = LyricsParams(
                    language=request.language,
                    audio_source="vocals" if source == "vocal_stem" else "original",
                    # Whisper's default no-VAD path is prone to inventing the
                    # first lyric phrase over an instrumental intro.  Keep
                    # this production-safe and configurable for environments
                    # with an unusually noisy separator.
                    vad_filter=env_bool("ALIGNMENT_ASR_VAD_ENABLED", True),
                    vad_threshold=max(0.05, min(0.95, env_float("ALIGNMENT_ASR_VAD_THRESHOLD", 0.35))),
                    vad_min_silence_ms=env_int("ALIGNMENT_ASR_VAD_MIN_SILENCE_MS", 250),
                    speech_pad_ms=env_int("ALIGNMENT_ASR_SPEECH_PAD_MS", 180),
                    no_speech_threshold=max(0.0, min(1.0, env_float("ALIGNMENT_ASR_NO_SPEECH_THRESHOLD", 0.5))),
                    condition_on_previous_text=env_bool(
                        "ALIGNMENT_ASR_CONDITION_ON_PREVIOUS_TEXT", False
                    ),
                    min_word_confidence=max(
                        0.0,
                        min(1.0, env_float("ALIGNMENT_ASR_MIN_WORD_CONFIDENCE", 0.15)),
                    ),
                )
                words = run_serialized_model_call(extract_lyrics, path, request.language, params)
                tokens = [
                    ASRToken(
                        text=word.text,
                        start=float(word.start),
                        end=float(word.end),
                        confidence=float(word.confidence),
                        model="faster-whisper-compat",
                        audio_source=source,
                        evidence_type="anchor",
                    )
                    for word in words
                ]
            evidence[source] = [replace(token, audio_source=source) for token in tokens]
        except Exception as exc:
            errors[source] = str(exc)

    if basic_pitch_enabled():
        for source, path in source_paths:
            if not path or not os.path.exists(path):
                continue
            try:
                melody = run_serialized_model_call(
                    detect_basic_pitch_boundaries,
                    path,
                )
                evidence[f"basic_pitch:{source}"] = [
                    replace(token, audio_source=source, evidence_type="melody")
                    for token in melody
                ]
            except Exception as exc:
                errors[f"basic_pitch:{source}"] = str(exc)
        original_path = request.original_audio_ref
        vocal_path = request.vocal_stem_audio_ref
        if (
            melody_blend_enabled()
            and original_path
            and vocal_path
            and os.path.exists(original_path)
            and os.path.exists(vocal_path)
            and os.path.abspath(original_path) != os.path.abspath(vocal_path)
        ):
            try:
                with tempfile.TemporaryDirectory(prefix="melody-blend-") as temp_dir:
                    blend_path = os.path.join(temp_dir, "vocal-mix-blend.wav")
                    write_melody_blend_audio(original_path, vocal_path, blend_path)
                    blended_melody = run_serialized_model_call(
                        detect_basic_pitch_boundaries,
                        blend_path,
                    )
                evidence.setdefault("basic_pitch:vocal_stem", []).extend(
                    replace(
                        token,
                        model="basic-pitch-blend",
                        audio_source="vocal_stem",
                        evidence_type="melody",
                    )
                    for token in blended_melody
                )
            except Exception as exc:
                errors["basic_pitch:blend"] = str(exc)

    if not any(evidence.values()) and fallback_words:
        evidence["vocal_stem"] = [replace(token, audio_source="vocal_stem") for token in fallback_words]

    anchor_pool: list[tuple[str, list[ASRToken], int]] = []
    canonical_tokens = [LyricToken(**_model_dict(token)) for token in request.tokens]
    for source in ("vocal_stem", "original_mix"):
        source_tokens = evidence.get(source, [])
        if not source_tokens or not canonical_tokens:
            continue
        match = global_monotonic_match(canonical_tokens, source_tokens)
        anchor_pool.append((source, source_tokens, sum(item.matched for item in match.matches)))
    best_anchor_source, best_anchors, best_anchor_count = max(
        anchor_pool,
        key=lambda item: item[2],
        default=("", [], 0),
    )

    # SOFA is the production timing source.  It consumes the independent ASR
    # clock only to choose bounded windows; every returned interval is still
    # matched back to the canonical lyric sequence before entering the worker
    # candidate lattice.  Invalid/empty test fixtures are skipped so the
    # legacy fallback remains deterministic in lightweight environments.
    sofa_path = request.vocal_stem_audio_ref
    sofa_language_enabled = (
        request.language.casefold().startswith("en")
        or env_bool("SOFA_ALLOW_NON_ENGLISH", False)
    )
    environment_legacy = os.environ.get("ALIGNMENT_PRIMARY_BACKEND", "").strip().casefold() in {
        "legacy", "current", "asr", "off"
    }
    if (
        request.alignment_backend != "legacy"
        and not environment_legacy
        and sofa_aligner_enabled()
        and sofa_language_enabled
        and sofa_path
        and os.path.exists(sofa_path)
        and request.tokens
    ):
        try:
            import soundfile as sf

            valid_sofa_audio = float(sf.info(sofa_path).duration) > 0.1
        except Exception:
            valid_sofa_audio = False
        if valid_sofa_audio:
            try:
                sofa_tokens = run_serialized_model_call(
                    align_with_sofa,
                    sofa_path,
                    canonical_tokens,
                    request.language,
                    anchor_tokens=best_anchors,
                )
                if sofa_tokens:
                    evidence["sofa:vocal_stem"] = [
                        replace(
                            token,
                            audio_source="vocal_stem",
                            evidence_type="forced",
                            model="sofa-combined",
                        )
                        for token in sofa_tokens
                    ]
            except Exception as exc:
                errors["sofa:vocal_stem"] = str(exc)

    if forcealign_enabled() and request.language == "en":
        path = request.vocal_stem_audio_ref
        if path and os.path.exists(path) and request.tokens:
            try:
                forced = run_serialized_model_call(
                    align_with_forcealign,
                    path,
                    " ".join(token.display_text for token in request.tokens),
                    request.language,
                    anchor_tokens=best_anchors,
                    activity_spans=request.activity_spans,
                )
                evidence["forcealign:vocal_stem"] = [
                    replace(token, audio_source="vocal_stem", evidence_type="forced")
                    for token in forced
                ]
            except Exception as exc:
                errors["forcealign:vocal_stem"] = str(exc)

    for source, path in source_paths:
        if not path or not os.path.exists(path):
            continue
        if not request.tokens:
            continue
        anchors = evidence.get(source, ())
        source_anchor_count = next(
            (count for anchor_source, _, count in anchor_pool if anchor_source == source),
            0,
        )
        if best_anchor_count and source_anchor_count < best_anchor_count:
            # A vocal stem can be cleaner but still produce a sparser ASR
            # transcript. Use the densest transcript to bound CTC/forced
            # alignment windows; it is an anchor only, never canonical timing.
            anchors = best_anchors
        if qwen_asr_enabled() and request.language == "en":
            try:
                forced = run_serialized_model_call(
                    align_with_qwen_forced,
                    path,
                    " ".join(token.display_text for token in request.tokens),
                    request.language,
                    anchor_tokens=anchors,
                )
                evidence[f"forced_alignment:{source}"] = [
                    replace(token, audio_source=source, evidence_type="forced") for token in forced
                ]
            except Exception as exc:
                errors[f"forced_alignment:{source}"] = str(exc)
        if whisperx_enabled():
            try:
                ctc = run_serialized_model_call(
                    align_with_whisperx,
                    path,
                    " ".join(token.display_text for token in request.tokens),
                    request.language,
                    anchor_tokens=anchors,
                )
                evidence[f"whisperx_ctc:{source}"] = [
                    replace(token, audio_source=source, evidence_type="ctc") for token in ctc
                ]
            except Exception as exc:
                errors[f"whisperx_ctc:{source}"] = str(exc)
                ctc_failed = True
    # WhisperX is an optional dependency and commonly fails to import on CPU
    # installations with a mismatched torchvision build.  If it failed and
    # the ASR transcript is sparse, use the installed English ForceAlign
    # backend as a bounded fallback rather than fabricating every missing
    # word from activity spans.  The opt-out keeps RAM-constrained workers in
    # the previous lightweight mode.
    if (
        ctc_failed
        and env_bool("ALIGNMENT_AUTO_FORCEALIGN", True)
        and request.language == "en"
        and best_anchor_count < max(4, int(len(request.tokens) * 0.75))
        and not evidence.get("forcealign:vocal_stem")
    ):
        path = request.vocal_stem_audio_ref
        if path and os.path.exists(path) and request.tokens:
            try:
                forced = run_serialized_model_call(
                    align_with_forcealign,
                    path,
                    " ".join(token.display_text for token in request.tokens),
                    request.language,
                    anchor_tokens=best_anchors,
                    activity_spans=request.activity_spans,
                )
                evidence["forcealign:vocal_stem"] = [
                    replace(token, audio_source="vocal_stem", evidence_type="forced")
                    for token in forced
                ]
            except Exception as exc:
                errors["forcealign:vocal_stem"] = str(exc)
    return evidence, errors


def run_worker_alignment(request: AlignmentWorkerRequest) -> AlignmentWorkerResponse:
    """Execute the same deterministic worker contract on a GPU host."""

    if not request.tokens:
        raise AlignmentWorkerError("Forced alignment requires lyric tokens")
    evidence, errors = collect_model_evidence(request)
    if not any(evidence.values()):
        response = AlignmentWorkerResponse(
            job_id=request.job_id,
            alignment_run_id=request.alignment_run_id,
            lyric_revision_id=request.lyric_revision_id,
            status="failed",
            diagnostics={"source_errors": errors, "reason": "no usable model evidence"},
        )
        return validate_worker_response(response, request)
    response = run_local_alignment(request, evidence, activity_spans=request.activity_spans)
    response.diagnostics["source_errors"] = errors
    return response
