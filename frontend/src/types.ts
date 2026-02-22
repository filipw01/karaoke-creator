export interface SongMetadata {
  title: string
  artist: string
  language: string
  genre: string
  year: string
  bpm: number
  gap_ms: number
}

export interface Word {
  text: string
  start: number
  end: number
  pitch_hz: number
  midi_note: number
  confidence: number
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

export interface JobResult {
  metadata: SongMetadata
  words: Array<Word>
  phrases: Array<Phrase>
  ultrastar_txt: string
  audio_file: string
  vocals_file: string
  instrumental_file: string
  video_file: string
}

export interface LyricsParams {
  model_size: string
  language: string
  vad_filter: boolean
  vad_threshold: number
  vad_min_silence_ms: number
  speech_pad_ms: number
  no_speech_threshold: number
  beam_size: number
  initial_prompt: string
  min_word_confidence: number
  audio_source: string
  normalize: boolean
  alignment_mode: "qwen3" | "proportional"
  lyrics_start_offset: number
}

export interface ProgressEvent {
  status: string
  message: string
  error?: string
}

export type AppStep = "upload" | "processing" | "lyrics" | "transcribing" | "editor" | "export"
