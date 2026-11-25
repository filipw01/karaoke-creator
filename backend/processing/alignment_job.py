"""Model evidence and canonical timing for the Rust-managed workflow."""

from __future__ import annotations

from dataclasses import asdict

from models import AlignmentEvidence, LyricToken, Word, WordAlignment
from processing.alignment import (
    ASRToken,
    asr_tokens_from_words,
    expand_alignment_sustains,
    map_token_id_aliases,
    merge_alignment_locks,
    repair_melody_handoffs,
    reset_word_boundaries,
)
from processing.alignment_worker import (
    AlignmentWorkerRequest,
    AlignmentWorkerResponse,
    alignment_response_to_models,
    collect_model_evidence,
    run_local_alignment,
    validate_worker_response,
)


def managed_alignment(payload: dict) -> dict:
    request = AlignmentWorkerRequest(**payload["request"])
    tokens = [LyricToken(**token.model_dump()) for token in request.tokens]
    remote = payload.get("remote_response")
    evidence = None
    source_errors = {}
    response = None
    source_name = "remote" if remote is not None else "rust_native" if payload.get("rust_model_evidence") else "local"
    try:
        if remote is not None:
            response = validate_worker_response(AlignmentWorkerResponse(**remote), request)
            source_errors = response.diagnostics.get("source_errors", {})
            if response.status != "completed":
                raise RuntimeError(
                    f"Remote alignment worker reported failure: {response.diagnostics.get('reason', 'unknown error')}"
                )
        else:
            if "rust_model_evidence" in payload:
                model = payload.get("model_backend", "forcealign")
                native_evidence = [ASRToken(**item) for item in payload["rust_model_evidence"]]
                evidence = {f"{model}:vocal_stem": native_evidence} if native_evidence else {}
            else:
                fallback_words = [Word(**word) for word in payload.get("fallback_words", [])]
                fallback = asr_tokens_from_words(
                    fallback_words,
                    model="legacy-alignment-fallback",
                    audio_source="vocal_stem",
                    evidence_type="anchor",
                )
                evidence, source_errors = collect_model_evidence(request, fallback_words=fallback)
            if not any(evidence.values()):
                raise RuntimeError(f"No usable model evidence remained: {source_errors or 'all sources were empty'}")
            response = run_local_alignment(
                request, evidence, activity_spans=request.activity_spans,
                activity_times=payload["times"], activity_values=payload["activity"],
            )
        source = alignment_response_to_models(response)
        alignments = expand_alignment_sustains(
            source,
            payload["times"],
            payload["activity"],
            clear_gap_s=0.12,
            activity_threshold=payload["activity_threshold"],
            window_end=payload["audio_duration"] or None,
            melody_outputs=evidence,
            melody_start_gap_threshold_s=0.08,
            melody_start_tolerance_s=0.08,
            melody_start_offset_s=0.08,
            melody_start_min_confidence=0.55,
            melody_handoff_min_gap_s=0.08,
            melody_handoff_max_gap_s=0.25,
            melody_handoff_offset_s=0.08,
        )
        if evidence is not None:
            alignments = repair_melody_handoffs(alignments, evidence, tokens=tokens, anchor_outputs=evidence)

        previous_tokens = [LyricToken(**token) for token in payload.get("previous_tokens", [])]
        previous = [
            WordAlignment(**{**alignment, "evidence": AlignmentEvidence(**alignment.get("evidence", {}))})
            for alignment in payload.get("previous_alignments", [])
        ]
        reset_ids = set(payload.get("reset_token_ids", []))
        aliases = map_token_id_aliases(previous_tokens, tokens)
        old_reset_ids = {aliases.get(token_id, token_id) for token_id in reset_ids}
        previous = [
            reset_word_boundaries(alignment)
            if alignment.token_id in reset_ids or alignment.token_id in old_reset_ids
            else alignment
            for alignment in previous
        ]
        alignments = merge_alignment_locks(previous, alignments, reset_token_ids=reset_ids, token_id_aliases=aliases)
    except Exception as exc:
        return {
            "error": str(exc),
            "diagnostics": {
                **(response.diagnostics if response is not None else {}),
                "error": str(exc),
                "worker": source_name,
                "source_errors": source_errors,
            },
        }
    return {
        "alignment_backend": payload.get("model_backend", request.alignment_backend),
        "alignments": [asdict(alignment) for alignment in alignments],
        "pitch_source_alignments": [asdict(alignment) for alignment in source],
        "diagnostics": {
            **response.diagnostics,
            "worker": source_name,
            "source_errors": source_errors,
            "audio_references": {
                "original_mix": bool(request.original_audio_ref),
                "vocal_stem": bool(request.vocal_stem_audio_ref),
            },
        },
    }
