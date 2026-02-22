from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class JobStatus(str, Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    EXTRACTING_AUDIO = "extracting_audio"
    SEPARATING_VOCALS = "separating_vocals"
    DETECTING_BPM = "detecting_bpm"
    SEARCHING_LYRICS = "searching_lyrics"
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
    current_alignment_mode: str = "forcealign"


@dataclass
class Job:
    id: str
    status: JobStatus = JobStatus.PENDING
    progress_message: str = ""
    progress_percent: float = 0.0
    error: str = ""
    original_filename: str = ""
    metadata: SongMetadata = field(default_factory=SongMetadata)
    result: JobResult | None = None
    # internal paths
    upload_path: str = ""
    work_dir: str = ""
    audio_path: str = ""
    vocals_path: str = ""
    instrumental_path: str = ""
    has_video: bool = False
    real_bpm: float = 0.0
    found_lyrics: str = ""
    confirmed_lyrics: str = ""
