"""Read and update existing UltraStar charts without changing other voices."""

from __future__ import annotations

import math
import re
import statistics
from pathlib import Path

from models import Phrase, UltraStarNote

_NOTE_TYPES = frozenset({":", "*", "F", "R", "G"})
_NOTE_PATTERN = re.compile(r"^(\S+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)(.*)$")
_VOICE_PATTERN = re.compile(r"^P(\d+)\s*$")


def decode_chart_text(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1250", "iso-8859-2", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", raw, 0, len(raw), "unsupported UltraStar encoding")


def read_chart_text(path: Path) -> str:
    return decode_chart_text(path.read_bytes())


def _chart_headers(lines: list[str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for line in lines:
        if not line.startswith("#"):
            continue
        key, separator, value = line[1:].partition(":")
        if separator:
            headers[key.strip().upper()] = value.strip()
    return headers


def _number(headers: dict[str, str], key: str, default: float) -> float:
    try:
        value = float(headers.get(key, str(default)).replace(",", "."))
    except ValueError as exc:
        raise ValueError(f"Invalid #{key} value") from exc
    if not math.isfinite(value):
        raise ValueError(f"Invalid #{key} value")
    return value


def parse_chart(content: str, fallback_title: str) -> dict[str, object]:
    lines = content.splitlines()
    headers = _chart_headers(lines)
    chart_bpm = _number(headers, "BPM", 0)
    gap_ms = _number(headers, "GAP", 0)
    if chart_bpm <= 0:
        raise ValueError("Chart must have a positive #BPM value")

    relative = headers.get("RELATIVE", "").strip().casefold() in {"1", "true", "yes", "on"}
    active_voice = 1
    previous_start_beat = 0
    current_notes: list[UltraStarNote] = []
    phrases: list[Phrase] = []
    raw_pitches: list[int] = []

    def finish_phrase(end_beat: int | None = None) -> None:
        nonlocal current_notes
        if not current_notes:
            return
        last_note = current_notes[-1]
        phrase_end = max(last_note.start_beat + last_note.duration_beats, end_beat or 0)
        phrases.append(Phrase(notes=current_notes, end_beat=phrase_end))
        current_notes = []

    for line_number, line in enumerate(lines, start=1):
        if not line.strip() or line.startswith("#"):
            continue
        voice_match = _VOICE_PATTERN.fullmatch(line.strip())
        if voice_match:
            finish_phrase()
            active_voice = int(voice_match.group(1))
            if active_voice == 1:
                previous_start_beat = 0
            continue
        if line.strip() == "E":
            finish_phrase()
            break
        if line.startswith("-"):
            marker = re.match(r"^-\s*(-?\d+)", line)
            end_beat = int(marker.group(1)) if marker else None
            if active_voice == 1:
                finish_phrase(end_beat)
            continue
        if active_voice != 1:
            continue

        match = _NOTE_PATTERN.match(line)
        if match is None or match.group(1) not in _NOTE_TYPES:
            continue
        start_beat, duration_beats, pitch = (int(match.group(i)) for i in (2, 3, 4))
        if start_beat < 0 or duration_beats < 0:
            raise ValueError(f"Invalid note timing on line {line_number}")
        if relative:
            start_beat += previous_start_beat
        previous_start_beat = start_beat
        raw_pitches.append(pitch)
        current_notes.append(
            UltraStarNote(
                note_type=match.group(1),
                start_beat=start_beat,
                duration_beats=duration_beats,
                pitch=pitch,
                text=match.group(5),
            )
        )

    finish_phrase()
    if not phrases:
        raise ValueError("Chart contains no editable notes in voice 1")

    pitch_offset = 60 if statistics.median(raw_pitches) >= 50 else 0
    if pitch_offset:
        for phrase in phrases:
            for note in phrase.notes:
                note.pitch -= pitch_offset

    audio_file = headers.get("MP3") or headers.get("AUDIO") or headers.get("OGG") or ""
    return {
        "metadata": {
            "title": headers.get("TITLE") or fallback_title,
            "artist": headers.get("ARTIST", ""),
            "language": headers.get("LANGUAGE", ""),
            "genre": headers.get("GENRE", ""),
            "year": headers.get("YEAR", ""),
            "bpm": chart_bpm / 4,
            "gap_ms": gap_ms,
        },
        "phrases": phrases,
        "audio_file": audio_file,
        "pitch_offset": pitch_offset,
    }


def update_chart(content: str, phrases: list[Phrase], pitch_offset: int) -> str:
    lines = content.splitlines()
    headers = _chart_headers(lines)
    relative = headers.get("RELATIVE", "").strip().casefold() in {"1", "true", "yes", "on"}
    generated: list[str] = []
    previous_start_beat = 0
    for phrase_index, phrase in enumerate(phrases):
        for note in phrase.notes:
            if note.note_type not in _NOTE_TYPES:
                raise ValueError(f"Unsupported note type: {note.note_type}")
            if "\n" in note.text or "\r" in note.text:
                raise ValueError("Note text cannot contain line breaks")
            start_beat = note.start_beat
            if relative:
                start_beat -= previous_start_beat
                if start_beat < 0:
                    raise ValueError("Relative charts require notes to remain in chronological order")
            previous_start_beat = note.start_beat
            lyric_text = note.text if note.text[:1].isspace() else f" {note.text}"
            generated.append(
                f"{note.note_type} {start_beat} {note.duration_beats} {note.pitch + pitch_offset}{lyric_text}"
            )
        if phrase_index < len(phrases) - 1:
            generated.append(f"- {phrase.end_beat}")

    output: list[str] = []
    active_voice = 1
    inserted_notes = False
    found_voice_one_notes = False
    for line in lines:
        voice_match = _VOICE_PATTERN.fullmatch(line.strip())
        if voice_match:
            active_voice = int(voice_match.group(1))
            output.append(line)
            continue
        if active_voice == 1 and line.startswith("-"):
            continue
        match = _NOTE_PATTERN.match(line)
        if match and match.group(1) in _NOTE_TYPES:
            if active_voice == 1:
                found_voice_one_notes = True
                if not inserted_notes:
                    output.extend(generated)
                    inserted_notes = True
                continue
        if line.strip() == "E" and active_voice == 1 and not inserted_notes:
            output.extend(generated)
            inserted_notes = True
        output.append(line)

    if not found_voice_one_notes:
        raise ValueError("Chart contains no editable notes in voice 1")
    if not inserted_notes:
        output.extend(generated)
    return "\n".join(output) + "\n"
