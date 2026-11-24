"""Generate UltraStar Deluxe TXT format from processed song data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import re

from models import Word, UltraStarNote, Phrase, SongMetadata
from processing.alignment import normalize_lyric_token
from processing.rust_utility import call as rust_utility


# Minimum pause between words (seconds) to trigger a line break
PHRASE_GAP_THRESHOLD = 0.8

_NOTE_TYPES = frozenset({":", "*", "F", "R", "G"})
_NOTE_PATTERN = re.compile(r"^(\S+)\s+(\d+)\s+(\d+)\s+(-?\d+)(.*)$")


def _read_ultrastar_lines(path: Path) -> list[str]:
    """Decode existing UltraStar files through the shared codec."""
    return rust_utility("read_external_chart", path=str(path)).splitlines()


@dataclass(frozen=True)
class TimedLyricWord:
    """A word span read from an UltraStar timing chart."""

    text: str
    start: float
    end: float


def parse_ultrastar_word_timings(
    path: str | Path,
    *,
    voice: int = 1,
) -> list[TimedLyricWord]:
    """Read source-order word spans from an UltraStar TXT file.

    UltraStar stores one or more note fragments per word. A leading space in
    the note text starts a new word; continuation fragments extend the current
    word. This parser intentionally returns only timing/text data so it can be
    used as an explicit, user-provided timing source without coupling the
    alignment worker to the export model.
    """

    chart_path = Path(path).expanduser().resolve()
    if not chart_path.is_file():
        raise FileNotFoundError(f"UltraStar timing file does not exist: {chart_path}")
    lines = _read_ultrastar_lines(chart_path)
    headers: dict[str, str] = {}
    for line in lines:
        if line.startswith("#"):
            key, separator, value = line[1:].partition(":")
            if separator:
                headers[key.strip().upper()] = value.strip()

    try:
        bpm = float(headers.get("BPM", "0").replace(",", "."))
        gap_seconds = float(headers.get("GAP", "0").replace(",", ".")) / 1000.0
    except ValueError as exc:
        raise ValueError(f"Invalid UltraStar BPM/GAP in {chart_path}") from exc
    if not math.isfinite(bpm) or bpm <= 0.0 or not math.isfinite(gap_seconds):
        raise ValueError(f"UltraStar chart requires a finite positive BPM: {chart_path}")
    seconds_per_beat = 60.0 / (bpm * 4.0)
    relative = headers.get("RELATIVE", "").strip().casefold() in {"1", "true", "yes", "on"}

    words: list[TimedLyricWord] = []
    current_parts: list[str] = []
    current_start: float | None = None
    current_end: float | None = None
    previous_start_beat = 0
    active_voice = 1

    def flush_word() -> None:
        nonlocal current_parts, current_start, current_end
        text = "".join(current_parts).strip()
        # Continuation markers and punctuation-only chart fragments (for
        # example a standalone ``~``) are formatting, not lyric words. Keep
        # them attached when they follow a lexical fragment, but never expose
        # an empty token to alignment/export callers.
        if (
            text
            and normalize_lyric_token(text)
            and current_start is not None
            and current_end is not None
        ):
            words.append(TimedLyricWord(text=text, start=current_start, end=current_end))
        current_parts = []
        current_start = None
        current_end = None

    for line_number, line in enumerate(lines, 1):
        if not line.strip() or line.startswith("#"):
            continue
        if line.strip() == "E":
            flush_word()
            break
        if line.startswith("-"):
            flush_word()
            continue
        if line.startswith("P") and line[1:].strip().isdigit():
            flush_word()
            active_voice = int(line[1:].strip())
            continue
        if active_voice != voice:
            continue
        match = _NOTE_PATTERN.match(line)
        if match is None or match.group(1) not in _NOTE_TYPES:
            continue
        try:
            start_beat = int(match.group(2))
            duration_beats = int(match.group(3))
        except ValueError as exc:
            raise ValueError(f"Invalid UltraStar note at {chart_path}:{line_number}") from exc
        if start_beat < 0 or duration_beats < 0:
            raise ValueError(f"Negative UltraStar timing at {chart_path}:{line_number}")
        if relative:
            start_beat = previous_start_beat + start_beat
        previous_start_beat = start_beat

        raw_text = match.group(5)
        if raw_text and raw_text[0].isspace():
            raw_text = raw_text[1:]
        text = raw_text.strip()
        if not text:
            continue
        starts_new_word = (
            not current_parts
            or raw_text[:1].isspace()
            # An opening parenthesis without a leading separator is an
            # optional/ad-lib word boundary. A closing ``)`` belongs to the
            # parenthesized word already being assembled.
            or (text.startswith("(") and bool(current_parts))
        )
        if starts_new_word:
            flush_word()
        current_parts.append(text)
        note_start = gap_seconds + start_beat * seconds_per_beat
        note_end = gap_seconds + (start_beat + duration_beats) * seconds_per_beat
        current_start = note_start if current_start is None else current_start
        current_end = note_end

    flush_word()
    if not words:
        raise ValueError(f"UltraStar chart contains no timed words: {chart_path}")
    return words


def words_to_phrases(
    words: list[Word],
    real_bpm: float,
    gap_threshold: float = PHRASE_GAP_THRESHOLD,
    line_breaks: list[int] | None = None,
) -> tuple[list[Phrase], float]:
    native = rust_utility(
        "phrases",
        words=[
            {
                "text": word.text, "start": word.start, "end": word.end, "midi_note": word.midi_note,
                "continuation_of_previous": word.continuation_of_previous,
                "alignment_status": word.alignment_status,
            }
            for word in words
        ],
        bpm=real_bpm, gap_threshold=gap_threshold, line_breaks=line_breaks or [],
    )
    return [
        Phrase(notes=[UltraStarNote(**note) for note in phrase["notes"]], end_beat=phrase["end_beat"])
        for phrase in native["phrases"]
    ], native["gap_ms"]


def generate_ultrastar_txt(
    metadata: SongMetadata,
    phrases: list[Phrase],
    audio_filename: str,
    vocals_filename: str = "",
    instrumental_filename: str = "",
    video_filename: str = "",
) -> str:
    return rust_utility(
        "chart", metadata=asdict(metadata), phrases=[asdict(phrase) for phrase in phrases],
        assets={"audio_file": audio_filename, "vocals_file": vocals_filename,
                "instrumental_file": instrumental_filename, "video_file": video_filename},
    )
