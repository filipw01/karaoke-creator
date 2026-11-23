"""Generate UltraStar Deluxe TXT format from processed song data."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from pathlib import Path
import re

from models import Word, UltraStarNote, Phrase, SongMetadata
from processing.alignment import normalize_lyric_token


log = logging.getLogger("ultrastar")

# Minimum pause between words (seconds) to trigger a line break
PHRASE_GAP_THRESHOLD = 0.8

_NOTE_TYPES = frozenset({":", "*", "F", "R", "G"})
_NOTE_PATTERN = re.compile(r"^(\S+)\s+(\d+)\s+(\d+)\s+(-?\d+)(.*)$")


def _read_ultrastar_lines(path: Path) -> list[str]:
    """Read UTF-8 and legacy Latin-1 UltraStar files."""

    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding).splitlines()
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", raw, 0, len(raw), "unsupported UltraStar encoding")


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


def time_to_beat(time_s: float, gap_s: float, ustar_bpm: float) -> int:
    """Convert a timestamp (seconds) to UltraStar beat number.

    ``ustar_bpm`` is the chart's quarter-note BPM. UltraStar note units are
    quarter-beats, so each note beat is ``60 / (ustar_bpm * 4)`` seconds.
    """
    elapsed = time_s - gap_s
    if elapsed < 0:
        elapsed = 0
    return round(elapsed * ustar_bpm * 4.0 / 60.0)


def beat_to_time(beat: int, gap_s: float, ustar_bpm: float) -> float:
    """Convert UltraStar beat back to seconds (for verification)."""
    return gap_s + beat * 60.0 / (ustar_bpm * 4.0)


def words_to_phrases(
    words: list[Word],
    real_bpm: float,
    gap_threshold: float = PHRASE_GAP_THRESHOLD,
    line_breaks: list[int] | None = None,
) -> tuple[list[Phrase], float]:
    """Convert word list to UltraStar phrases with notes.

    Args:
        line_breaks: Optional list of word indices where a forced phrase break should occur
            (from lyrics line boundaries). These take priority over gap-based breaks.

    Returns (phrases, gap_ms) where gap_ms is the GAP value in milliseconds.
    """
    if not words:
        return [], 0.0

    # A canonical unmatched token has no defensible timestamp. Keep it in
    # the alignment/editor layer, but omit it from the legacy export rather
    # than turning the null span into a fake note at time zero.
    source_breaks = set(line_breaks) if line_breaks else set()
    if any(getattr(word, "alignment_status", "aligned") in {"unmatched", "AlignmentStatus.UNMATCHED"} for word in words):
        filtered_words: list[Word] = []
        filtered_breaks: list[int] = []
        pending_break = False
        for source_index, word in enumerate(words):
            if source_index in source_breaks:
                pending_break = True
            status = getattr(word, "alignment_status", "aligned")
            if status in {"unmatched", "AlignmentStatus.UNMATCHED"}:
                continue
            if pending_break and filtered_words:
                filtered_breaks.append(len(filtered_words))
            pending_break = False
            filtered_words.append(word)
        words = filtered_words
        source_breaks = set(filtered_breaks)
    if not words:
        return [], 0.0
    break_set = source_breaks

    gap_s = words[0].start
    gap_ms = gap_s * 1000.0
    # The exported #BPM is four times the real musical BPM, and each note
    # unit is a quarter of that chart beat.  Therefore note positions use
    # real_bpm * 16 as their seconds-to-beat conversion factor.
    ustar_bpm = real_bpm * 4
    note_grid_bpm = ustar_bpm * 4

    log.info("GAP=%.0fms (first word at %.2fs), UltraStar BPM=%.2f (real BPM=%.1f)",
             gap_ms, gap_s, ustar_bpm, real_bpm)
    log.info("Note beat duration = %.1fms", 60000.0 / note_grid_bpm)
    if break_set:
        log.info("Forced line breaks at word indices: %s", sorted(break_set))

    phrases: list[Phrase] = []
    current_notes: list[UltraStarNote] = []

    for i, word in enumerate(words):
        # Forced line break before this word
        if i in break_set and current_notes:
            last_note = current_notes[-1]
            phrase_end_beat = last_note.start_beat + last_note.duration_beats + 1
            phrases.append(Phrase(notes=list(current_notes), end_beat=phrase_end_beat))
            current_notes = []

        start_beat = time_to_beat(word.start, gap_s, ustar_bpm)
        end_beat = time_to_beat(word.end, gap_s, ustar_bpm)
        duration = max(1, end_beat - start_beat)
        pitch = word.midi_note - 60

        if i < 5:
            reconstructed_s = beat_to_time(start_beat, gap_s, ustar_bpm)
            log.info("  Word '%s': %.2fs -> beat %d -> reconstructed=%.2fs (drift=%.0fms)",
                     word.text, word.start, start_beat, reconstructed_s,
                     (reconstructed_s - word.start) * 1000)

        no_leading_space = getattr(word, "continuation_of_previous", False)
        text = word.text if (no_leading_space or not current_notes) else f" {word.text}"

        note = UltraStarNote(
            note_type=":",
            start_beat=start_beat,
            duration_beats=duration,
            pitch=pitch,
            text=text,
        )
        current_notes.append(note)

        # Gap-based break (only if no forced breaks are in use for this region)
        is_last = i == len(words) - 1
        if not is_last and (i + 1) not in break_set:
            gap_to_next = words[i + 1].start - word.end
            if gap_to_next >= gap_threshold:
                phrase_end_beat = end_beat + 1
                phrases.append(Phrase(notes=list(current_notes), end_beat=phrase_end_beat))
                current_notes = []

    if current_notes:
        last_note = current_notes[-1]
        phrase_end_beat = last_note.start_beat + last_note.duration_beats + 1
        phrases.append(Phrase(notes=list(current_notes), end_beat=phrase_end_beat))

    if phrases:
        all_notes = [n for p in phrases for n in p.notes]
        last_note = all_notes[-1]
        last_time = beat_to_time(last_note.start_beat + last_note.duration_beats, gap_s, ustar_bpm)
        log.info("Generated %d phrases, %d notes", len(phrases), len(all_notes))
        log.info("Time range: %.2fs - %.2fs (last word original: %.2fs)",
                 gap_s, last_time, words[-1].end)

    return phrases, gap_ms


def generate_ultrastar_txt(
    metadata: SongMetadata,
    phrases: list[Phrase],
    audio_filename: str,
    vocals_filename: str = "",
    instrumental_filename: str = "",
    video_filename: str = "",
) -> str:
    """Generate the full UltraStar TXT file content."""
    lines: list[str] = []

    lines.append(f"#TITLE:{metadata.title}")
    lines.append(f"#ARTIST:{metadata.artist}")
    lines.append(f"#MP3:{audio_filename}")
    if video_filename:
        lines.append(f"#VIDEO:{video_filename}")
    if vocals_filename:
        lines.append(f"#VOCALS:{vocals_filename}")
    if instrumental_filename:
        lines.append(f"#INSTRUMENTAL:{instrumental_filename}")
    # UltraStar's chart beat grid is four times the real musical BPM.  The
    # application stores ``SongMetadata.bpm`` as the real BPM (the same value
    # used by ``words_to_phrases``), so export the chart-grid BPM here.
    lines.append(f"#BPM:{metadata.bpm * 4.0:.2f}")
    lines.append(f"#GAP:{metadata.gap_ms:.0f}")
    if metadata.language:
        lines.append(f"#LANGUAGE:{metadata.language}")
    if metadata.genre:
        lines.append(f"#GENRE:{metadata.genre}")
    if metadata.year:
        lines.append(f"#YEAR:{metadata.year}")
    lines.append("#CREATOR:Karaoke Creator")
    lines.append("")

    for i, phrase in enumerate(phrases):
        for note in phrase.notes:
            lines.append(
                f"{note.note_type} {note.start_beat} "
                f"{note.duration_beats} {note.pitch} {note.text}"
            )
        if i < len(phrases) - 1:
            lines.append(f"- {phrase.end_beat}")

    lines.append("E")

    return "\n".join(lines) + "\n"
