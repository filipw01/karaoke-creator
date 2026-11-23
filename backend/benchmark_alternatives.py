"""Compare the current pipeline and SOFA on the cached benchmark suite.

The original private reference TXT/MP3 pairs are not present in the current
checkout.  The suite therefore uses the cached ``generated-suite-charts`` as a
proxy reference and labels the report accordingly.  This is useful for a
repeatable relative comparison and deliberately does not claim ground-truth
accuracy.  Put the original pairs back in ``~/Downloads`` and use
``benchmark_suite.py`` for the authoritative score.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import pickle
from pathlib import Path
import re
import sys
from typing import Iterable

import benchmark_alignment as benchmark
from models import AlignmentEvidence, AlignmentRun, LyricRevision, Word, WordAlignment
from processing.alignment_worker import AlignmentWorkerResponse, WorkerWordAlignment
from processing.pitch import PitchDetection, detect_pitch
from processing.ultrastar import generate_ultrastar_txt, words_to_phrases

from alternative_pipelines import TimedToken, align_pipeline


SONGS = (
    ("dancing queen", "dancingqueen"),
    ("someone like you", "someonelikeyou"),
    ("billie jean", "billiejean"),
    ("bring me to life", "bringmetolife"),
    ("we are the champions", "wearethechampions"),
    ("bajm - ta sama chwila", "bajmtasamachwila"),
)
ROOT = Path(__file__).resolve().parent
PRIVATE = ROOT / "benchmarks" / "private"
STEMS = PRIVATE / "benchmark-suite-vocals"
TARGET_CHARTS = PRIVATE / "generated-suite-charts"
CURRENT_CHARTS = PRIVATE / "generated-model-only-charts"
ARTIFACTS = PRIVATE / "alternative-pipelines"
PITCH_CACHE = PRIVATE / "pitch-cache"


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _timed_words(reference: benchmark.UltraStarReference) -> list[TimedToken]:
    return [TimedToken(word.text, word.start, word.end) for word in reference.words]


def _reference_request(reference: benchmark.UltraStarReference, audio: Path, vocal: Path):
    return benchmark._build_request(reference, audio, vocal)


def _response_for_words(request, words: list[Word], pipeline: str) -> AlignmentWorkerResponse:
    alignments = []
    for index, token in enumerate(request.tokens):
        if index < len(words):
            word = words[index]
            alignments.append(WorkerWordAlignment(
                token_id=token.id,
                alignment_run_id=request.alignment_run_id,
                start=word.start,
                end=word.end,
                status="aligned",
                confidence=word.confidence or 0.5,
                evidence={"selected_model": pipeline, "selected_audio": "vocal_stem"},
            ))
    return AlignmentWorkerResponse(
        job_id=request.job_id,
        alignment_run_id=request.alignment_run_id,
        lyric_revision_id=request.lyric_revision_id,
        status="completed",
        alignments=alignments,
        diagnostics={"pipeline": pipeline},
    )


def _lexical_score(expected: Iterable[str], predicted: Iterable[str]) -> tuple[int, int]:
    expected_values = [_norm(value.replace("~", "")) for value in expected]
    predicted_values = [_norm(value.replace("~", "")) for value in predicted]
    scored = len(expected_values)
    correct = sum(
        left == right
        for left, right in zip(expected_values, predicted_values)
    )
    return correct, scored


def _chart_pitch_words(path: Path, reference: benchmark.UltraStarReference, request) -> list[Word]:
    _headers, notes = benchmark.parse_ultrastar_pitch_notes(path)
    by_word = benchmark._pitch_notes_by_word(reference, notes, request.language)
    words: list[Word] = []
    for token, expected, expected_notes in zip(request.tokens, reference.words, by_word):
        midi = 0
        if expected_notes:
            midi = max(expected_notes, key=lambda note: note.duration_beats).pitch + 60
        words.append(Word(
            text=expected.text,
            start=expected.start,
            end=expected.end,
            midi_note=midi,
            pitch_hz=440.0 if midi else 0.0,
            token_id=token.id,
            alignment_status="aligned",
        ))
    return words


def _map_tokens(words: list[Word], request) -> list[Word]:
    mapped: list[Word] = []
    for index, word in enumerate(words[: len(request.tokens)]):
        mapped.append(replace(word, token_id=request.tokens[index].id, alignment_run_id=request.alignment_run_id))
    return mapped


def _detect_pitch_cached(key: str, vocal_path: Path, audio_path: Path) -> PitchDetection:
    """Reuse the expensive production pitch contour across pipeline scores."""

    PITCH_CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = PITCH_CACHE / f"{key}.pickle"
    try:
        with cache_path.open("rb") as handle:
            cached = pickle.load(handle)
        if isinstance(cached, PitchDetection):
            return cached
    except (OSError, EOFError, pickle.PickleError, ValueError, TypeError):
        pass
    detection = detect_pitch(str(vocal_path), str(audio_path))
    temporary = cache_path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(detection, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(cache_path)
    return detection


def _build_candidate_chart(path: Path, reference: benchmark.UltraStarReference, words: list[Word]) -> Path:
    headers = reference.headers
    chart_bpm = float(headers.get("BPM", "0").replace(",", "."))
    metadata = {
        "title": headers.get("TITLE", reference.path.stem),
        "artist": headers.get("ARTIST", ""),
        "language": headers.get("LANGUAGE", "English"),
        "bpm": chart_bpm / 4.0,
        "gap_ms": words[0].start * 1000.0 if words else 0.0,
    }
    from models import SongMetadata

    metadata_model = SongMetadata(**metadata)
    phrases, gap_ms = words_to_phrases(words, metadata_model.bpm)
    metadata_model.gap_ms = gap_ms
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(generate_ultrastar_txt(
        metadata=metadata_model,
        phrases=phrases,
        audio_filename=f"{_norm(metadata_model.title)}.wav",
    ), encoding="utf-8")
    return path


def score_song(title: str, key: str, pipeline: str, *, keep_windows: bool = True) -> dict[str, object]:
    target_path = TARGET_CHARTS / f"{key}.generated.txt"
    current_path = CURRENT_CHARTS / f"{key}.generated.txt"
    vocal_path = STEMS / key / "vocals.wav"
    audio_path = STEMS / key / "audio.wav"
    for path in (target_path, vocal_path, audio_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    reference = benchmark.parse_ultrastar_reference(target_path)
    request = _reference_request(reference, audio_path, vocal_path)
    artifact_dir = ARTIFACTS / pipeline / key
    if not keep_windows and artifact_dir.exists():
        for file in artifact_dir.glob("*.wav"):
            file.unlink()
    if pipeline == "current":
        current_reference = benchmark.parse_ultrastar_reference(current_path)
        words = [Word(text=word.text, start=word.start, end=word.end, confidence=1.0) for word in current_reference.words]
        generated_path = current_path
    else:
        words = align_pipeline(pipeline, _timed_words(reference), vocal_path, artifact_dir)
        generated_path = _build_candidate_chart(artifact_dir / f"{key}.{pipeline}.txt", reference, _map_tokens(words, request))
    words = _map_tokens(words, request)
    response = _response_for_words(request, words, pipeline)
    lexical_correct, lexical_scored = _lexical_score(
        (word.text for word in reference.words),
        (word.text for word in words),
    )
    pitch_notes = benchmark.parse_ultrastar_pitch_notes(target_path)[1]
    pitch_words: list[Word]
    if pipeline == "current":
        pitch_words = _map_tokens(_chart_pitch_words(current_path, current_reference, request), request)
    else:
        detection = _detect_pitch_cached(key, vocal_path, audio_path)
        pitch_words = benchmark._pitch_words_for_response(
            request, response, detection.times, detection.frequencies,
            detection.confidences, detection.energies, detection.candidate_midis,
            detection.candidate_scores, detection.onset_scores,
        )
    timing = benchmark.compare_reference(
        reference, response, request, timing_tolerance_beats=1.0, worst_count=0,
    )
    pitch = benchmark.compare_pitch_reference(
        reference, pitch_notes, response, request, pitch_words, worst_count=0,
    )
    return {
        "song": title,
        "key": key,
        "pipeline": pipeline,
        "proxy_reference": str(target_path),
        "generated_chart": str(generated_path),
        "word_count": len(reference.words),
        "words": {"correct": lexical_correct, "scored": lexical_scored, "accuracy": lexical_correct / lexical_scored if lexical_scored else 0.0},
        "timing": {"within_one_beat": timing["within_tolerance"], "coverage": timing["coverage"], "start_mae_ms": timing["start_mae_ms"], "end_mae_ms": timing["end_mae_ms"]},
        "pitch": {
            "word_correct": pitch["pitch_correct_word_count"],
            "word_scored": pitch["pitch_scorable_word_count"],
            "word_accuracy": pitch["pitch_accuracy"],
            "note_correct": pitch["pitch_note_correct_count"],
            "note_scored": pitch["pitch_note_scored_count"],
            "note_accuracy": pitch["pitch_note_accuracy"],
            "split_correct": pitch["pitch_split_word_correct_count"],
            "split_scored": pitch["pitch_split_word_count"],
            "split_accuracy": pitch["pitch_split_word_accuracy"],
        },
        "aligned_words": [word.__dict__ for word in words],
    }


def _aggregate(results: Iterable[dict[str, object]]) -> dict[str, dict[str, object]]:
    """Return weighted comparison totals grouped by pipeline."""

    grouped: dict[str, list[dict[str, object]]] = {}
    for result in results:
        grouped.setdefault(str(result["pipeline"]), []).append(result)
    summary: dict[str, dict[str, object]] = {}
    for pipeline, rows in grouped.items():
        def ratio(section: str, correct: str, scored: str) -> float:
            numerator = sum(int(dict(row[section])[correct]) for row in rows)
            denominator = sum(int(dict(row[section])[scored]) for row in rows)
            return numerator / denominator if denominator else 0.0

        summary[pipeline] = {
            "songs": len(rows),
            "word_accuracy": ratio("words", "correct", "scored"),
            "pitch_word_accuracy": ratio("pitch", "word_correct", "word_scored"),
            "pitch_note_accuracy": ratio("pitch", "note_correct", "note_scored"),
            "pitch_split_accuracy": ratio("pitch", "split_correct", "split_scored"),
            "timing_within_one_beat_macro": sum(
                float(dict(row["timing"])["within_one_beat"])
                for row in rows
            ) / len(rows),
        }
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", choices=("current", "sofa", "all"), default="all")
    parser.add_argument("--song", action="append", dest="songs")
    parser.add_argument("--json-output", type=Path, default=PRIVATE / "alternative-pipelines-score.json")
    parser.add_argument("--keep-windows", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    requested = {_norm(song) for song in args.songs} if args.songs else None
    pipelines = ("current", "sofa") if args.pipeline == "all" else (args.pipeline,)
    results: list[dict[str, object]] = []
    for title, key in SONGS:
        if requested and _norm(title) not in requested and key not in requested:
            continue
        for pipeline in pipelines:
            print(f"[{pipeline}] {title}", file=sys.stderr, flush=True)
            results.append(score_song(title, key, pipeline, keep_windows=args.keep_windows))
    report = {
        "reference_mode": "cached generated-suite chart (proxy; original private pairs unavailable)",
        "summary": _aggregate(results),
        "songs": results,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
