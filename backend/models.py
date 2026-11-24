from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
from typing import Any


class LyricLanguage(str, Enum):
    ENGLISH = "en"
    POLISH = "pl"


class LyricSource(str, Enum):
    ONLINE = "online"
    ASR_DRAFT = "asr_draft"
    MANUAL = "manual"


class AlignmentStatus(str, Enum):
    ALIGNED = "aligned"
    ESTIMATED = "estimated"
    UNMATCHED = "unmatched"


class AlignmentRunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Word:
    text: str
    start: float  # seconds
    end: float  # seconds
    pitch_hz: float = 0.0  # median frequency in Hz
    midi_note: int = 0  # MIDI note number
    confidence: float = 0.0
    pitch_contour: list[tuple[float, float]] = field(default_factory=list)  # (time_s, hz) for editor
    pitch_candidates: list[dict[str, Any]] = field(default_factory=list)
    continuation_of_previous: bool = False  # no leading space when same word as previous note
    token_id: str = ""
    alignment_status: str = AlignmentStatus.ALIGNED.value
    alignment_run_id: str = ""


@dataclass
class LyricToken:
    """A stable source token. ``display_text`` is never replaced by ASR text."""

    id: str
    lyric_revision_id: str
    display_text: str
    normalized_text: str
    line_index: int
    token_index: int
    source_index: int = 0


@dataclass
class LyricRevision:
    """Immutable lyric source metadata plus its source-order tokenization."""

    id: str
    song_id: str
    language: LyricLanguage | str
    source: LyricSource | str
    raw_text: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    parent_revision_id: str | None = None
    tokens: list[LyricToken] = field(default_factory=list)


@dataclass
class AlignmentEvidence:
    selected_model: str = ""
    selected_audio: str = ""
    reasons: list[str] = field(default_factory=list)
    candidate_scores: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class WordAlignment:
    """Canonical timing for one lyric token, independent of pitch/note data."""

    token_id: str
    alignment_run_id: str
    start: float | None
    end: float | None
    status: AlignmentStatus | str
    confidence: float
    evidence: AlignmentEvidence = field(default_factory=AlignmentEvidence)
    start_locked: bool = False
    end_locked: bool = False


@dataclass
class AlignmentRun:
    id: str
    lyric_revision_id: str
    status: AlignmentRunStatus | str
    selected_strategy: str
    diagnostics: Any = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class UltraStarNote:
    note_type: str  # ":" normal, "*" golden, "F" freestyle
    start_beat: int
    duration_beats: int
    pitch: int  # MIDI - 60 (so C4 = 0)
    text: str


@dataclass
class Phrase:
    notes: list[UltraStarNote] = field(default_factory=list)
    end_beat: int = 0


@dataclass
class SongMetadata:
    title: str = ""
    artist: str = ""
    language: str = ""
    genre: str = ""
    year: str = ""
    bpm: float = 0.0
    gap_ms: float = 0.0
