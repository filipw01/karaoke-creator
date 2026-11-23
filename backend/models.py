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


class JobStatus(str, Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    EXTRACTING_AUDIO = "extracting_audio"
    SEPARATING_VOCALS = "separating_vocals"
    DETECTING_BPM = "detecting_bpm"
    SEARCHING_LYRICS = "searching_lyrics"
    ALIGNING_WORDS = "aligning_words"
    AWAITING_LYRICS = "awaiting_lyrics"
    EXTRACTING_LYRICS = "extracting_lyrics"
    DETECTING_PITCH = "detecting_pitch"
    GENERATING = "generating"
    DONE = "done"
    ERROR = "error"


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


@dataclass
class AlignmentVariant:
    """Precomputed result for one alignment mode."""
    words: list[Word] = field(default_factory=list)
    phrases: list[Phrase] = field(default_factory=list)
    ultrastar_txt: str = ""
    gap_ms: float = 0.0
    # Pitch segmentation is an export/note layer. It must never replace the
    # canonical one-word-per-token alignment above.
    pitch_words: list[Word] = field(default_factory=list)
    word_alignments: list[WordAlignment] = field(default_factory=list)
    alignment_run: AlignmentRun | None = None


@dataclass
class JobResult:
    metadata: SongMetadata = field(default_factory=SongMetadata)
    words: list[Word] = field(default_factory=list)
    phrases: list[Phrase] = field(default_factory=list)
    ultrastar_txt: str = ""
    audio_file: str = ""
    vocals_file: str = ""
    instrumental_file: str = ""
    video_file: str = ""
    # All alignment variants for instant switch; key = alignment_mode
    alignment_results: dict[str, AlignmentVariant] = field(default_factory=dict)
    current_alignment_mode: str = "canonical"
    lyric_revision: LyricRevision | None = None
    word_alignments: list[WordAlignment] = field(default_factory=list)
    alignment_run: AlignmentRun | None = None
    # Background SOFA-vs-legacy comparison state and agreement metrics.
    alignment_comparison: dict[str, Any] = field(default_factory=dict)
    pitch_words: list[Word] = field(default_factory=list)
    # Manual note edits are keyed by phrase/note position and reapplied when
    # canonical timings are refreshed after a boundary edit or re-alignment.
    note_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    audio_duration: float = 0.0


@dataclass
class Job:
    id: str
    status: JobStatus = JobStatus.PENDING
    progress_message: str = ""
    progress_percent: float = 0.0
    queued_at: str = ""
    library_path: str = ""
    library_root: str = ""
    error: str = ""
    failed_phase: str = ""
    original_filename: str = ""
    metadata: SongMetadata = field(default_factory=SongMetadata)
    result: JobResult | None = None
    # internal paths
    upload_path: str = ""
    source_url: str = ""
    work_dir: str = ""
    audio_path: str = ""
    vocals_path: str = ""
    instrumental_path: str = ""
    has_video: bool = False
    real_bpm: float = 0.0
    found_lyrics: str = ""
    confirmed_lyrics: str = ""
    lyric_revisions: list[LyricRevision] = field(default_factory=list)
    selected_lyric_revision_id: str = ""
    alignment_runs: list[AlignmentRun] = field(default_factory=list)
    owner_id: str = "local"
