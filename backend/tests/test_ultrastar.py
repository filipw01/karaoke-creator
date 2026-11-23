"""Tests for UltraStar export conventions."""

from __future__ import annotations

import sys
from pathlib import Path

backend = Path(__file__).resolve().parent.parent
if str(backend) not in sys.path:
    sys.path.insert(0, str(backend))

from models import Phrase, SongMetadata, UltraStarNote
from processing.ultrastar import generate_ultrastar_txt


def test_export_writes_chart_grid_bpm_not_real_bpm():
    text = generate_ultrastar_txt(
        SongMetadata(title="Test", artist="Artist", bpm=120.0),
        [
            Phrase(
                notes=[UltraStarNote(":", 0, 4, 0, "hello")],
                end_beat=5,
            )
        ],
        audio_filename="test.mp3",
    )

    assert "#BPM:480.00" in text
