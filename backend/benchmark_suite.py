"""Run the local alignment/pitch benchmark across a song suite.

The default suite uses each song's UltraStar chart as the timing reference so
pitch comparisons remain independent of a missing ASR model cache. The
``--model-only`` mode removes that shortcut: word timings must come from the
same alignment evidence as production, while pitch still runs on the
separated vocal stem.

By default the five benchmark pairs are discovered recursively below
``~/Downloads``.  Use ``--downloads`` to point at another directory.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
from typing import Iterable

import benchmark_alignment as benchmark
from models import SongMetadata
from processing.audio import extract_audio
from processing.alignment import activity_spans_from_signal
from processing.pitch import (
    alignment_activity_threshold,
    detect_pitch,
    voiced_activity_signal,
)
from processing.ultrastar import generate_ultrastar_txt, words_to_phrases
from processing.vocals import separate_vocals


SONG_TITLES = (
    "dancing queen",
    "someone like you",
    "billie jean",
    "bring me to life",
    "we are the champions",
    "bajm - ta sama chwila",
)


def _name_key(value: str) -> str:
    """Normalize a filename/title for matching independent of punctuation."""

    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _song_key(value: str) -> str:
    return _name_key(value)


def _candidate_score(path: Path, title_key: str, suffix: str) -> tuple[int, int, str]:
    stem_key = _name_key(path.stem)
    exact = int(stem_key == title_key)
    contains = int(title_key in stem_key)
    # Prefer an exact stem, then an artist-prefixed filename, then a shorter
    # path.  ``[MULTI]`` charts are a fallback when no plain chart exists.
    return (exact, contains, -len(str(path)))


def discover_song_pairs(downloads: Path, titles: Iterable[str] = SONG_TITLES) -> list[tuple[str, Path, Path]]:
    """Find one MP3 and matching TXT chart for every requested song title."""

    root = downloads.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Downloads directory does not exist: {root}")
    audio_files = [path for path in root.rglob("*.mp3") if path.is_file()]
    chart_files = [
        path
        for path in root.rglob("*.txt")
        if path.is_file() and not path.name.casefold().endswith(".torrent")
    ]
    pairs: list[tuple[str, Path, Path]] = []
    for title in titles:
        title_key = _song_key(title)
        audio_candidates = [
            path for path in audio_files if title_key in _name_key(path.stem)
        ]
        if not audio_candidates:
            raise FileNotFoundError(f"No MP3 found for {title!r} below {root}")
        audio = max(
            audio_candidates,
            key=lambda path: _candidate_score(path, title_key, ".mp3"),
        )
        chart_candidates = [
            path
            for path in chart_files
            if path.parent == audio.parent and title_key in _name_key(path.stem)
        ]
        if not chart_candidates:
            chart_candidates = [
                path for path in chart_files if title_key in _name_key(path.stem)
            ]
        if not chart_candidates:
            raise FileNotFoundError(f"No UltraStar TXT found for {title!r} below {root}")
        chart = max(
            chart_candidates,
            key=lambda path: _candidate_score(path, title_key, ".txt"),
        )
        pairs.append((title, audio, chart))
    return pairs


def _run_song(
    title: str,
    audio: Path,
    pipeline_audio: Path,
    chart: Path,
    vocal_audio: Path,
    *,
    pitch_octave_offset: int,
    tolerance_ms: float,
    timing_tolerance_beats: float,
    include_diff: bool = False,
    generated_output_dir: Path | None = None,
    model_only: bool = False,
    evidence_cache: Path | None = None,
) -> dict[str, object]:
    reference = benchmark.parse_ultrastar_reference(chart)
    pitch_headers, pitch_notes = benchmark.parse_ultrastar_pitch_notes(chart)
    del pitch_headers
    request = benchmark._build_request(reference, pipeline_audio, vocal_audio)

    # Compute the separated-vocal activity clock before model evidence.  The
    # worker uses it to avoid forcing the first lyric phrase onto a long
    # instrumental/piano intro; production requests already carry this field.
    detection = detect_pitch(
        str(vocal_audio),
        str(pipeline_audio),
    )
    times, frequencies, confidences, energies = detection
    activity_values = voiced_activity_signal(frequencies, energies)
    activity_spans = activity_spans_from_signal(
        times,
        activity_values,
        threshold=alignment_activity_threshold(activity_values),
    )
    request = request.model_copy(update={"activity_spans": activity_spans})

    # The normal suite uses explicit chart timing to isolate pitch tuning. In
    # model-only mode, leave evidence unset so ``_run_once`` collects the same
    # ASR/CTC/model evidence used by the application and timing is genuinely
    # measured rather than copied from the reference chart.
    if model_only:
        if evidence_cache is not None and evidence_cache.is_file():
            evidence, source_errors = benchmark._read_evidence_cache(evidence_cache)
        else:
            evidence, source_errors = benchmark.collect_model_evidence(request)
            if evidence_cache is not None:
                benchmark._write_evidence_cache(evidence_cache, evidence, source_errors)
    else:
        evidence = {
            "ultrastar_chart": benchmark._timing_reference_evidence(request, chart),
        }
        source_errors = {}
    # The same activity clock is reused by canonical selection and pitch
    # assignment; no second detector pass is needed.
    try:
        import soundfile as sf

        activity_window_end = float(sf.info(str(vocal_audio)).duration)
    except Exception:
        activity_window_end = None

    response, canonical_response = benchmark._run_once(
        request,
        evidence,
        source_errors,
        activity_spans,
        times,
        activity_values,
        activity_window_end,
        True,
    )
    # Production samples pitch from the canonical alignment spans. Activity
    # expansion is retained for export/sustain display, but using those longer
    # spans here would pull the next word's onset into the current note.
    pitch_response = canonical_response or response
    pitch_words = benchmark._pitch_words_for_response(
        request,
        pitch_response,
        times,
        frequencies,
        confidences,
        energies,
        getattr(detection, "candidate_midis", None),
        getattr(detection, "candidate_scores", None),
        getattr(detection, "onset_scores", None),
    )
    timing_response = canonical_response or response
    timing = benchmark.compare_reference(
        reference,
        timing_response,
        request,
        tolerance_ms=tolerance_ms,
        timing_tolerance_beats=timing_tolerance_beats,
        worst_count=20,
    )
    pitch = benchmark.compare_pitch_reference(
        reference,
        pitch_notes,
        pitch_response,
        request,
        pitch_words,
        worst_count=0,
        pitch_octave_offset=pitch_octave_offset,
    )
    result = {
        "song": title,
        "audio": str(audio),
        "pipeline_audio": str(pipeline_audio),
        "reference": str(chart),
        "vocal_stem": str(vocal_audio),
        "word_count": len(reference.words),
        "timing": {
            "within_tolerance": timing["within_tolerance"],
            "timing_tolerance_beats": timing["timing_tolerance_beats"],
            "timing_tolerance_ms": timing["timing_tolerance_ms"],
            "coverage": timing["coverage"],
            "start_mae_ms": timing["start_mae_ms"],
            "end_mae_ms": timing["end_mae_ms"],
            "worst": timing["worst"],
        },
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
            "global_note_correct": pitch["pitch_global_note_correct_count"],
            "global_note_scored": pitch["pitch_global_note_scored_count"],
            "global_note_accuracy": pitch["pitch_global_note_accuracy"],
            "global_note_timing": pitch["pitch_global_note_timing_within_tolerance"],
            "global_note_timing_tolerance_beats": pitch[
                "pitch_global_note_timing_tolerance_beats"
            ],
            "coverage": pitch["pitch_coverage"],
            "best_octave_offset": pitch["pitch_best_octave_offset"],
            "best_octave_correct": pitch["pitch_best_octave_correct_word_count"],
            "best_octave_accuracy": pitch["pitch_best_octave_accuracy"],
        },
    }
    if include_diff:
        result["note_diff"] = benchmark._build_pitch_note_diff(
            pitch_notes,
            pitch_words,
            chart_headers=reference.headers,
            pitch_octave_offset=pitch_octave_offset,
        )
    if generated_output_dir is not None:
        headers = reference.headers
        chart_bpm = float(headers.get("BPM", "0").replace(",", "."))
        if chart_bpm <= 0.0:
            raise ValueError(f"Invalid chart BPM for {title!r}")
        real_bpm = chart_bpm / 4.0
        phrases, gap_ms = words_to_phrases(pitch_words, real_bpm)
        metadata = SongMetadata(
            title=title,
            artist=headers.get("ARTIST", ""),
            language=headers.get("LANGUAGE", ""),
            bpm=real_bpm,
            gap_ms=gap_ms,
        )
        generated_output_dir.mkdir(parents=True, exist_ok=True)
        generated_path = generated_output_dir / f"{_song_key(title)}.generated.txt"
        generated_path.write_text(
            generate_ultrastar_txt(
                metadata=metadata,
                phrases=phrases,
                audio_filename=f"{_song_key(title)}.mp3",
            ),
            encoding="utf-8",
        )
        result["generated_chart"] = str(generated_path)
    return result


def _format_note_line(note: dict[str, object] | None) -> str:
    if note is None:
        return "—"
    return (
        f"{note.get('note_type', ':')} {int(note['start_beat'])} "
        f"{int(note['duration_beats'])} {int(note['pitch'])} {note['text']}"
    )


def _write_note_diff_report(path: Path, results: list[dict[str, object]]) -> None:
    """Write only non-perfect generated-vs-reference note lines as Markdown."""

    lines = [
        "# UltraStar benchmark note diff",
        "",
        "Generated lines are reconstructed from the production pitch segments",
        "and rounded to the integer UltraStar beat grid, matching export.",
        "Timing is exact only when start and duration deltas are both zero.",
        "The benchmark onset tolerance is ±1 chart beat.",
        "",
    ]
    for result in results:
        diff = result.get("note_diff")
        if not isinstance(diff, dict):
            continue
        rows = diff.get("rows", [])
        if not isinstance(rows, list):
            rows = []
        imperfect = [
            row
            for row in rows
            if isinstance(row, dict)
            and not (row.get("timing_exact") and row.get("pitch_correct"))
        ]
        lines.extend(
            [
                f"## {result['song']}",
                "",
                f"Expected notes: {diff.get('expected_count', 0)}; "
                f"generated segments: {diff.get('predicted_count', 0)}; "
                f"matched: {diff.get('matched_count', 0)}; "
                f"extras: {diff.get('extra_count', 0)}.",
                "",
                "| # | Reference | Generated | Δ start (beats) | Δ duration (beats) | Δ pitch (semitones) | Result |",
                "|---:|---|---|---:|---:|---:|---|",
            ]
        )
        for index, row in enumerate(imperfect, 1):
            expected = row.get("expected") if isinstance(row, dict) else None
            predicted = row.get("predicted") if isinstance(row, dict) else None
            if not isinstance(expected, dict):
                continue
            result_tags: list[str] = []
            if row.get("status") == "missing":
                result_tags.append("missing")
            if row.get("start_delta_beats") not in (None, 0) or row.get("duration_delta_beats") not in (None, 0):
                result_tags.append("timing")
            if not row.get("timing_within_tolerance", False) and row.get("status") != "missing":
                result_tags.append("outside ±1 beat")
            if not row.get("pitch_correct", False):
                result_tags.append("pitch")
            lines.append(
                f"| {index} | `{_format_note_line(expected)}` | "
                f"`{_format_note_line(predicted)}` | "
                f"{row.get('start_delta_beats', '—')} | "
                f"{row.get('duration_delta_beats', '—')} | "
                f"{row.get('pitch_delta', '—')} | {', '.join(result_tags)} |"
            )
        extras = diff.get("extras", [])
        if isinstance(extras, list) and extras:
            lines.extend(["", "Extra generated segments:", ""])
            for extra in extras:
                if isinstance(extra, dict):
                    lines.append(f"- `{_format_note_line(extra)}`")
        if not imperfect and not extras:
            lines.append("No differences.")
        lines.append("")
    path.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    path.expanduser().resolve().write_text("\n".join(lines), encoding="utf-8")


def _aggregate(results: list[dict[str, object]]) -> dict[str, object]:
    timing = [result["timing"] for result in results]
    pitch = [result["pitch"] for result in results]

    def total(key: str) -> tuple[int, int]:
        return (
            sum(int(item[f"{key}_correct"]) for item in pitch),
            sum(int(item[f"{key}_scored"]) for item in pitch),
        )

    word_correct, word_scored = total("word")
    note_correct, note_scored = total("note")
    split_correct, split_scored = total("split")
    global_note_correct, global_note_scored = total("global_note")
    return {
        "songs": len(results),
        "timing_within_tolerance_macro": sum(
            float(item["within_tolerance"]) for item in timing
        )
        / len(timing),
        "timing_coverage_macro": sum(float(item["coverage"]) for item in timing)
        / len(timing),
        "pitch_word_correct": word_correct,
        "pitch_word_scored": word_scored,
        "pitch_word_accuracy": word_correct / word_scored if word_scored else 0.0,
        "pitch_word_best_octave_correct": sum(
            int(item["best_octave_correct"]) for item in pitch
        ),
        "pitch_word_best_octave_accuracy": sum(
            int(item["best_octave_correct"]) for item in pitch
        )
        / word_scored
        if word_scored
        else 0.0,
        "pitch_note_correct": note_correct,
        "pitch_note_scored": note_scored,
        "pitch_note_accuracy": note_correct / note_scored if note_scored else 0.0,
        "pitch_split_correct": split_correct,
        "pitch_split_scored": split_scored,
        "pitch_split_accuracy": split_correct / split_scored if split_scored else 0.0,
        "pitch_global_note_correct": global_note_correct,
        "pitch_global_note_scored": global_note_scored,
        "pitch_global_note_accuracy": (
            global_note_correct / global_note_scored if global_note_scored else 0.0
        ),
        "pitch_global_note_timing_macro": sum(
            float(item["global_note_timing"]) for item in pitch
        ) / len(pitch) if pitch else 0.0,
        "best_octave_accuracy_macro": sum(
            float(item["best_octave_accuracy"]) for item in pitch
        )
        / len(pitch),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--downloads",
        type=Path,
        default=Path.home() / "Downloads",
        help="directory containing the song MP3/TXT pairs",
    )
    parser.add_argument(
        "--song",
        action="append",
        dest="songs",
        help="run only this title (repeatable; defaults to the full six-song suite)",
    )
    parser.add_argument(
        "--stem-cache",
        type=Path,
        default=Path("benchmarks/private/benchmark-suite-vocals"),
        help="directory for cached Demucs vocal/instrumental stems",
    )
    parser.add_argument(
        "--no-separate-vocals",
        action="store_true",
        help="use the original MP3 as the pitch source (diagnostic fallback)",
    )
    parser.add_argument(
        "--model-only",
        action="store_true",
        help="do not use the UltraStar chart for timing; score model alignment end-to-end",
    )
    parser.add_argument("--tolerance-ms", type=float, default=100.0)
    parser.add_argument(
        "--timing-tolerance-beats",
        type=float,
        default=1.0,
        help="word timing tolerance in UltraStar chart beats (default: ±1)",
    )
    # Keep the production detector's absolute MIDI output unshifted by
    # default.  The report still exposes a per-song best-octave diagnostic
    # for charts whose authoring octave is inconsistent with the recording.
    parser.add_argument("--pitch-octave-offset", type=int, default=0)
    parser.add_argument(
        "--diff-output",
        type=Path,
        help="write a Markdown reference-vs-generated note diff",
    )
    parser.add_argument(
        "--generated-output-dir",
        type=Path,
        help="write generated UltraStar TXT charts for direct comparison",
    )
    parser.add_argument(
        "--evidence-cache-dir",
        type=Path,
        help=(
            "read/write one model-evidence JSON cache per song in model-only mode; "
            "keeps later selector iterations from rerunning the heavy models"
        ),
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.tolerance_ms < 0:
        raise SystemExit("--tolerance-ms must be non-negative")
    if not math.isfinite(args.timing_tolerance_beats) or args.timing_tolerance_beats < 0:
        raise SystemExit("--timing-tolerance-beats must be finite and non-negative")
    requested = tuple(args.songs) if args.songs else SONG_TITLES
    try:
        pairs = discover_song_pairs(args.downloads, requested)
        results = []
        for title, audio, chart in pairs:
            stem_dir = args.stem_cache.expanduser().resolve() / _song_key(title)
            stem_dir.mkdir(parents=True, exist_ok=True)
            pipeline_audio = stem_dir / "audio.wav"
            if not pipeline_audio.is_file():
                print(f"[{title}] extracting 44.1 kHz stereo audio...", file=sys.stderr)
                pipeline_audio = Path(extract_audio(str(audio), str(stem_dir)))
            if args.no_separate_vocals:
                vocal_audio = pipeline_audio
            else:
                vocals_path = stem_dir / "vocals.wav"
                instrumental_path = stem_dir / "instrumental.wav"
                if not vocals_path.is_file() or not instrumental_path.is_file():
                    print(f"[{title}] separating vocals with Demucs...", file=sys.stderr)
                    stem_dir.mkdir(parents=True, exist_ok=True)
                    separated_vocals, _separated_instrumental = separate_vocals(
                        str(pipeline_audio),
                        str(stem_dir),
                    )
                    vocals_path = Path(separated_vocals)
                else:
                    print(f"[{title}] using cached vocal stem", file=sys.stderr)
                vocal_audio = vocals_path
            results.append(
                _run_song(
                    title,
                    audio,
                    pipeline_audio,
                    chart,
                    vocal_audio,
                    pitch_octave_offset=args.pitch_octave_offset,
                    tolerance_ms=args.tolerance_ms,
                    timing_tolerance_beats=args.timing_tolerance_beats,
                    include_diff=args.diff_output is not None,
                    generated_output_dir=args.generated_output_dir,
                    model_only=args.model_only,
                    evidence_cache=(
                        args.evidence_cache_dir.expanduser().resolve()
                        / f"{_song_key(title)}-model-evidence-current.json"
                        if args.model_only and args.evidence_cache_dir is not None
                        else None
                    ),
                )
            )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"benchmark suite failed: {exc}", file=sys.stderr)
        return 1
    if args.diff_output is not None:
        _write_note_diff_report(args.diff_output, results)
        for result in results:
            result.pop("note_diff", None)
    report = {"summary": _aggregate(results), "songs": results}
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    summary = report["summary"]
    print(f"Benchmark suite: {summary['songs']} songs")
    print(
        f"Timing ({'model-only' if args.model_only else 'chart-backed'}, "
        f"±{args.timing_tolerance_beats:g} chart beat): "
        f"{float(summary['timing_within_tolerance_macro']):.1%} within tolerance, "
        f"{float(summary['timing_coverage_macro']):.1%} coverage"
    )
    print(
        "Pitch totals: "
        f"words {summary['pitch_word_correct']}/{summary['pitch_word_scored']} "
        f"({float(summary['pitch_word_accuracy']):.1%}), "
        f"best-octave words {summary['pitch_word_best_octave_correct']}/"
        f"{summary['pitch_word_scored']} "
        f"({float(summary['pitch_word_best_octave_accuracy']):.1%}), "
        f"notes {summary['pitch_note_correct']}/{summary['pitch_note_scored']} "
        f"({float(summary['pitch_note_accuracy']):.1%}), "
        f"split {summary['pitch_split_correct']}/{summary['pitch_split_scored']} "
        f"({float(summary['pitch_split_accuracy']):.1%}), "
        f"global notes {summary['pitch_global_note_correct']}/"
        f"{summary['pitch_global_note_scored']} "
        f"({float(summary['pitch_global_note_accuracy']):.1%}), "
        f"global timing {float(summary['pitch_global_note_timing_macro']):.1%}"
    )
    print("\nIndividual songs:")
    for result in results:
        timing = result["timing"]
        pitch = result["pitch"]
        print(
            f"- {result['song']}: "
            f"timing {float(timing['within_tolerance']):.1%}; "
            f"word pitch {int(pitch['word_correct'])}/{int(pitch['word_scored'])} "
            f"({float(pitch['word_accuracy']):.1%}); "
            f"best octave {int(pitch['best_octave_correct'])}/"
            f"{int(pitch['word_scored'])} ({float(pitch['best_octave_accuracy']):.1%}, "
            f"{int(pitch['best_octave_offset']):+d} st); "
            f"notes {int(pitch['note_correct'])}/{int(pitch['note_scored'])} "
            f"({float(pitch['note_accuracy']):.1%}); "
            f"split {int(pitch['split_correct'])}/{int(pitch['split_scored'])} "
            f"({float(pitch['split_accuracy']):.1%}); "
            f"global notes {int(pitch['global_note_correct'])}/"
            f"{int(pitch['global_note_scored'])} "
            f"({float(pitch['global_note_accuracy']):.1%}), "
            f"timing {float(pitch['global_note_timing']):.1%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
