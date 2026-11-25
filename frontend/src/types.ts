export interface SongMetadata {
  title: string
  artist: string
  language: string
  genre: string
  year: string
  bpm: number
  gap_ms: number
}

export type AudioTrack =
  | "original"
  | "vocals"
  | "instrumental"
  | "instrumental_with_playback"
  | "vocals_with_playback"

export interface Word {
  text: string
  start: number
  end: number
  pitch_hz: number
  midi_note: number
  confidence: number
  /** (time_s, hz) pairs for editor pitch visualization */
  pitch_contour?: Array<[number, number]>
  /** Candidate MIDI pitches and normalized support for this lyric segment */
  pitch_candidates?: Array<{
    midi: number
    score: number
    support: number
  }>
  /** No leading space when same word as previous note (segment) */
  continuation_of_previous?: boolean
  token_id?: string
  alignment_status?: AlignmentStatus
  alignment_run_id?: string
}

export type LyricLanguage = "en" | "pl"
export type LyricSource = "online" | "asr_draft" | "manual"
export type AlignmentStatus = "aligned" | "estimated" | "unmatched"

export interface LyricToken {
  id: string
  lyric_revision_id: string
  display_text: string
  normalized_text: string
  line_index: number
  token_index: number
  source_index: number
}

export interface LyricRevision {
  id: string
  song_id: string
  language: LyricLanguage
  source: LyricSource
  raw_text: string
  created_at: string
  parent_revision_id?: string | null
  tokens: Array<LyricToken>
}

export interface AlignmentEvidence {
  selected_model?: string
  selected_audio?: string
  reasons: Array<string>
  candidate_scores?: Array<Record<string, unknown>>
}

export interface WordAlignment {
  token_id: string
  alignment_run_id: string
  start: number | null
  end: number | null
  status: AlignmentStatus
  confidence: number
  evidence: AlignmentEvidence
  start_locked: boolean
  end_locked: boolean
}

export interface AlignmentRun {
  id: string
  lyric_revision_id: string
  status: "running" | "completed" | "failed"
  selected_strategy: string
  diagnostics: Record<string, unknown>
  created_at: string
}

export interface UltraStarNote {
  note_type: string
  start_beat: number
  duration_beats: number
  pitch: number
  text: string
}

export interface Phrase {
  notes: Array<UltraStarNote>
  end_beat: number
}

export interface AlignmentVariant {
  words: Array<Word>
  pitch_words?: Array<Word>
  phrases: Array<Phrase>
  ultrastar_txt: string
  /** Gap in ms used to build this variant's phrases; use for playback when this variant is selected */
  gap_ms?: number
  word_alignments?: Array<WordAlignment>
  alignment_run?: AlignmentRun | null
}

export interface AlignmentComparison {
  status: "idle" | "running" | "completed" | "failed" | "not_supported"
  forcealign_available?: boolean
  mms_available?: boolean
  compared_word_count?: number
  timing_agreement?: number
  pitch_agreement?: number
  pitch_exact_fraction?: number
  timing_mean_abs_error_ms?: number
  one_chart_beat_ms?: number
  confidence?: number
  preferred_mode?: string
  reason?: string
  error?: string
  updated_at?: string
}

export interface JobResult {
  metadata: SongMetadata
  words: Array<Word>
  /** Pitch/note projection; canonical words remain one item per lyric token. */
  pitch_words?: Array<Word>
  phrases: Array<Phrase>
  ultrastar_txt: string
  audio_file: string
  vocals_file: string
  instrumental_file: string
  video_file: string
  /** Precomputed alignments for instant switch; key = alignment mode */
  alignment_results?: Record<string, AlignmentVariant>
  current_alignment_mode?: string
  lyric_revision?: LyricRevision | null
  lyric_revisions?: Array<LyricRevision>
  selected_lyric_revision_id?: string
  word_alignments?: Array<WordAlignment>
  alignment_run?: AlignmentRun | null
  alignment_comparison?: AlignmentComparison
  audio_duration?: number
  export_ready?: boolean
  pending_revision_id?: string | null
  note_overrides?: Record<string, Record<string, unknown>>
}

export interface ProgressEvent {
  status: string
  message: string
  error?: string
}

export type AppStep = "upload" | "download" | "prepare" | "lyrics" | "transcribe" | "generate" | "editor" | "export"
