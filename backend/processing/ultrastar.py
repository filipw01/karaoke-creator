"""Generate UltraStar Deluxe TXT format from processed song data."""

from __future__ import annotations

import logging

from models import Word, UltraStarNote, Phrase, SongMetadata


log = logging.getLogger("ultrastar")

# Minimum pause between words (seconds) to trigger a line break
PHRASE_GAP_THRESHOLD = 0.8


def time_to_beat(time_s: float, gap_s: float, ustar_bpm: float) -> int:
    """Convert a timestamp (seconds) to UltraStar beat number.

    In UltraStar, BPM = real_bpm * 4, and each beat unit = 60/BPM seconds.
    So: beat = (time - gap) * BPM / 60
    """
    elapsed = time_s - gap_s
    if elapsed < 0:
        elapsed = 0
    return round(elapsed * ustar_bpm / 60.0)


def beat_to_time(beat: int, gap_s: float, ustar_bpm: float) -> float:
    """Convert UltraStar beat back to seconds (for verification)."""
    return gap_s + beat * 60.0 / ustar_bpm


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

    break_set = set(line_breaks) if line_breaks else set()

    gap_s = words[0].start
    gap_ms = gap_s * 1000.0
    ustar_bpm = real_bpm * 4

    log.info("GAP=%.0fms (first word at %.2fs), UltraStar BPM=%.2f (real BPM=%.1f)",
             gap_ms, gap_s, ustar_bpm, real_bpm)
    log.info("Beat duration = %.1fms", 60000.0 / ustar_bpm)
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

        text = f" {word.text}" if current_notes else word.text

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
    lines.append(f"#BPM:{metadata.bpm:.2f}")
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
