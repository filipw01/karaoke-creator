from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

backend = Path(__file__).resolve().parent.parent
if str(backend) not in sys.path:
    sys.path.insert(0, str(backend))

import main
from processing.external_chart import parse_chart, update_chart


def test_parse_chart_converts_timing_and_absolute_pitch():
    chart = """#TITLE:Song
#ARTIST:Artist
#BPM:480
#GAP:250
: 0 4 60 Hello
: 4 4 62 world
- 9
E
"""

    parsed = parse_chart(chart, "fallback")
    phrase = parsed["phrases"][0]

    assert parsed["metadata"] == {
        "title": "Song",
        "artist": "Artist",
        "language": "",
        "genre": "",
        "year": "",
        "bpm": 120,
        "gap_ms": 250,
    }
    assert [(note.start_beat, note.pitch, note.text) for note in phrase.notes] == [
        (0, 0, " Hello"),
        (4, 2, " world"),
    ]


def test_update_chart_preserves_headers_and_other_voices():
    chart = """#TITLE:Song
#BPM:480
#GAP:250
P1
: 0 4 60 Hello
- 5
P2
: 0 4 55 Other voice
E
"""
    parsed = parse_chart(chart, "Song")
    phrases = parsed["phrases"]
    phrases[0].notes[0].pitch = 3

    updated = update_chart(chart, phrases, parsed["pitch_offset"])

    assert "#TITLE:Song" in updated
    assert "P1\n: 0 4 63 Hello\nP2" in updated
    assert ": 0 4 55 Other voice" in updated
    assert parse_chart(updated, "Song")["phrases"][0].notes[0].pitch == 3


def test_external_song_can_be_loaded_and_saved_in_place(tmp_path: Path, monkeypatch):
    song_directory = tmp_path / "Artist" / "Song"
    song_directory.mkdir(parents=True)
    chart_path = song_directory / "Song.txt"
    chart_path.write_text(
        "#TITLE:Song\n#ARTIST:Artist\n#BPM:480\n#GAP:0\n#MP3:Song.mp3\n"
        ": 0 4 0 Hello\nE\n",
        encoding="utf-8",
    )
    (song_directory / "Song.mp3").write_bytes(b"audio")
    monkeypatch.setattr(main, "_working_directory", lambda: str(tmp_path))

    loaded = asyncio.run(main.get_external_song("Artist/Song/Song.txt"))
    assert loaded["audio_available"] is True
    loaded["phrases"][0]["notes"][0]["pitch"] = 2

    saved = asyncio.run(main.save_external_song(
        "Artist/Song/Song.txt",
        {"phrases": loaded["phrases"]},
    ))

    assert ": 0 4 2 Hello" in saved["content"]
    assert chart_path.read_text(encoding="utf-8").endswith(": 0 4 2 Hello\nE\n")


def test_relative_chart_rejects_reordered_notes():
    chart = """#TITLE:Song
#BPM:480
#RELATIVE:1
: 4 4 0 first
: 4 4 1 second
E
"""
    phrases = parse_chart(chart, "Song")["phrases"]
    phrases[0].notes[1].start_beat = 2

    with pytest.raises(ValueError, match="chronological order"):
        update_chart(chart, phrases, 0)
