import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react"
import type { JobResult, LyricsParams, UltraStarNote } from "../types"
import { regenerateLyrics, updateNotes } from "../api"
import { AudioControls } from "./AudioControls"
import { EditorHeader } from "./EditorHeader"
import { EditorHint } from "./EditorHint"
import { KaraokeWords } from "./KaraokeWords"
import { LyricsPreviewSection } from "./LyricsPreviewSection"
import { LyricsSettingsPanel } from "./LyricsSettingsPanel"
import { NoteEditorPopover } from "./NoteEditorPopover"
import { RawUltrastarSection } from "./RawUltrastarSection"
import { TimelineScrubber } from "./TimelineScrubber"

interface EditorProps {
  jobId: string
  result: JobResult
  onResultChange: (result: JobResult) => void
  /** Called when user clicks Export; pass current alignment mode so download ZIP uses that variant's TXT */
  onExport: (alignmentMode?: string) => void
}

const DEFAULT_LYRICS_PARAMS: LyricsParams = {
  model_size: "large-v3",
  language: "",
  vad_filter: false,
  vad_threshold: 0.25,
  vad_min_silence_ms: 200,
  speech_pad_ms: 250,
  no_speech_threshold: 0.4,
  beam_size: 5,
  initial_prompt: "These are song lyrics with singing.",
  min_word_confidence: 0.15,
  audio_source: "original",
  normalize: true,
  alignment_mode: "forcealign",
}

const PIXELS_PER_BEAT = 8
const ROW_HEIGHT = 14
const PITCH_RANGE = 36 // 3 octaves display range

export const Editor = ({ jobId, result, onResultChange, onExport }: EditorProps) => {
  const [selectedNote, setSelectedNote] = useState<{ phraseIdx: number; noteIdx: number } | null>(null)
  const [audioTrack, setAudioTrack] = useState<"original" | "vocals" | "instrumental">("vocals")
  const [isPlaying, setIsPlaying] = useState(false)
  const [saving, setSaving] = useState(false)
  const [regenerating, setRegenerating] = useState(false)
  const [lyricsParams, setLyricsParams] = useState<LyricsParams>({ ...DEFAULT_LYRICS_PARAMS })
  const [showSettings, setShowSettings] = useState(false)
  const [audioDuration, setAudioDuration] = useState(0)
  const [regenerateError, setRegenerateError] = useState<string | null>(null)
  const [isDraggingTimeline, setIsDraggingTimeline] = useState(false)
  const [dragTime, setDragTime] = useState<number | null>(null)
  const [, setPlaybackTick] = useState(0)
  const playbackTimeRef = useRef(0)
  const timelineRef = useRef<HTMLDivElement>(null)
  const rafRef = useRef<number | null>(null)
  const playheadRafRef = useRef<number | null>(null)
  const audioRef = useRef<HTMLAudioElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [editorAnchor, setEditorAnchor] = useState<{
    left: number
    top: number
    width: number
    height: number
  } | null>(null)
  const [viewAlignmentMode, setViewAlignmentMode] = useState<string>(
    result.current_alignment_mode ?? "forcealign"
  )

  const displayResult: JobResult = (() => {
    const ar = result.alignment_results
    if (ar && viewAlignmentMode && ar[viewAlignmentMode]) {
      const v = ar[viewAlignmentMode]
      return {
        ...result,
        words: v.words,
        phrases: v.phrases,
        ultrastar_txt: v.ultrastar_txt,
      }
    }
    return result
  })()

  const lastSavedPhrasesRef = useRef<typeof displayResult.phrases>(displayResult.phrases)
  const isDirty = displayResult.phrases !== lastSavedPhrasesRef.current

  useEffect(() => {
    setViewAlignmentMode(result.current_alignment_mode ?? "forcealign")
  }, [result.current_alignment_mode])

  useEffect(() => {
    lastSavedPhrasesRef.current = displayResult.phrases
  }, [viewAlignmentMode, displayResult.phrases])

  // Compute pitch bounds for the display
  const allNotes = displayResult.phrases.flatMap((p) => p.notes)
  const minPitch = allNotes.length > 0 ? Math.min(...allNotes.map((n) => n.pitch)) - 6 : -12
  const maxPitch = minPitch + PITCH_RANGE
  const totalBeats = allNotes.length > 0
    ? Math.max(...allNotes.map((n) => n.start_beat + n.duration_beats)) + 20
    : 100

  const variant = result.alignment_results?.[viewAlignmentMode]
  const gap_ms_effective = variant?.gap_ms ?? result.metadata.gap_ms
  const gap_s = gap_ms_effective / 1000
  const ustar_bpm = displayResult.metadata.bpm * 4
  const timeToBeat = (t: number): number => (t - gap_s) * (ustar_bpm / 60)
  const beatToTime = useCallback(
    (beat: number): number => (beat * 60) / ustar_bpm + gap_s,
    [ustar_bpm, gap_s]
  )
  // Single value for UI: ref updated every frame when playing; sync from currentTime when paused/seek
  const displayTime = dragTime !== null ? dragTime : playbackTimeRef.current
  const currentBeat = timeToBeat(displayTime)

  // Find current phrase: the one containing currentBeat, or between verses the last one we left
  const getPhraseEndBeat = (phrase: (typeof displayResult.phrases)[number]): number => {
    const lastNote = phrase.notes[phrase.notes.length - 1]
    return phrase.end_beat > 0
      ? phrase.end_beat
      : (lastNote?.start_beat ?? 0) + (lastNote?.duration_beats ?? 0)
  }
  const getPhraseStartBeat = (phrase: (typeof displayResult.phrases)[number]): number =>
    phrase.notes[0]?.start_beat ?? 0

  // As soon as we hit the verse-end mark, show the next verse (and its timestamp)
  const displayedPhrase = (() => {
    const inPhrase = displayResult.phrases.find((phrase) => {
      const phraseStartBeat = getPhraseStartBeat(phrase)
      const phraseEndBeat = getPhraseEndBeat(phrase)
      return currentBeat >= phraseStartBeat && currentBeat <= phraseEndBeat
    })
    if (inPhrase) return inPhrase
    const nextPhrase = displayResult.phrases.find((p) => getPhraseStartBeat(p) > currentBeat) ?? null
    if (nextPhrase) return nextPhrase
    let lastPassed: (typeof displayResult.phrases)[number] | null = null
    let lastEnd = -1
    for (let i = 0; i < displayResult.phrases.length; i++) {
      const end = getPhraseEndBeat(displayResult.phrases[i])
      if (end <= currentBeat && end > lastEnd) {
        lastEnd = end
        lastPassed = displayResult.phrases[i]
      }
    }
    return lastPassed ?? displayResult.phrases[0] ?? null
  })()

  // Extract words from displayed phrase (next verse as soon as verse-end mark is hit)
  const phraseText = displayedPhrase
    ? displayedPhrase.notes.map((n) => n.text.trim()).join("").trim()
    : ""

  // Match phrase words to timestamped words by finding words that overlap with phrase time range
  const phraseWords = displayedPhrase && displayResult.words.length > 0 && phraseText
    ? (() => {
        const phraseStartTime = beatToTime(getPhraseStartBeat(displayedPhrase))
        const lastNote = displayedPhrase.notes[displayedPhrase.notes.length - 1]
        const phraseEndTime = beatToTime(
          displayedPhrase.end_beat > 0
            ? displayedPhrase.end_beat
            : (lastNote?.start_beat ?? 0) + (lastNote?.duration_beats ?? 0)
        )
        const wordsInRange = displayResult.words
          .filter((w) => w.start >= phraseStartTime - 0.2 && w.end <= phraseEndTime + 0.2)
          .sort((a, b) => a.start - b.start)
        return wordsInRange
      })()
    : []

  const displayedVerseStartTime = displayedPhrase
    ? beatToTime(getPhraseStartBeat(displayedPhrase))
    : 0

  // Find currently active word (the one being sung right now, or between words: last sung)
  const activeWordIndexRaw = phraseWords.findIndex((w) => displayTime >= w.start && displayTime <= w.end)
  const activeWordIndex =
    activeWordIndexRaw >= 0
      ? activeWordIndexRaw
      : (() => {
          // Between words: highlight the last word we've passed
          let lastPassed = -1
          for (let i = 0; i < phraseWords.length; i++) {
            if (displayTime >= phraseWords[i].end) lastPassed = i
          }
          return lastPassed
        })()

  const totalDuration = audioDuration > 0
    ? audioDuration
    : displayResult.words.length > 0
      ? Math.max(...displayResult.words.map((w) => w.end)) + 1
      : 60

  const updateTimeFromMouse = useCallback(
    (clientX: number, isDragging: boolean) => {
      const timeline = timelineRef.current
      if (!timeline) return
      const rect = timeline.getBoundingClientRect()
      const x = clientX - rect.left
      const clickedBeat = Math.max(0, Math.min(totalBeats, x / PIXELS_PER_BEAT))
      const maxTime = audioDuration > 0 ? audioDuration : beatToTime(totalBeats)
      const clickedTime = Math.max(0, Math.min(maxTime, beatToTime(clickedBeat)))
      const audio = audioRef.current
      if (audio) {
        audio.currentTime = clickedTime
        playbackTimeRef.current = clickedTime
        if (isDragging) {
          setDragTime(clickedTime)
        } else {
          setPlaybackTick((t) => t + 1)
          setDragTime(null)
        }
      }
    },
    [totalBeats, audioDuration, beatToTime]
  )

  const handleTimelineClick = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (!isDraggingTimeline) {
      // Only handle click if we weren't dragging
      updateTimeFromMouse(e.clientX, false)
    }
  }, [updateTimeFromMouse, isDraggingTimeline])

  const handleTimelineMouseDown = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (e.button !== 0) return // Only left mouse button
    setIsDraggingTimeline(true)
    setDragTime(null)
    updateTimeFromMouse(e.clientX, true)
  }, [updateTimeFromMouse])

  const handleTimelineMouseUp = useCallback(() => {
    if (dragTime !== null) {
      playbackTimeRef.current = dragTime
      setPlaybackTick((t) => t + 1)
      setDragTime(null)
    }
    setIsDraggingTimeline(false)
  }, [dragTime])

  const handleTimelineMouseLeave = useCallback(() => {
    if (dragTime !== null) {
      playbackTimeRef.current = dragTime
      setPlaybackTick((t) => t + 1)
      setDragTime(null)
    }
    setIsDraggingTimeline(false)
  }, [dragTime])

  // Global mouseup handler to stop dragging if mouse released outside timeline
  useEffect(() => {
    const onMouseUp = () => {
      setIsDraggingTimeline(false)
    }
    if (isDraggingTimeline) {
      window.addEventListener("mouseup", onMouseUp)
      return () => window.removeEventListener("mouseup", onMouseUp)
    }
  }, [isDraggingTimeline])

  // Global mousemove handler for dragging - throttled with requestAnimationFrame
  useEffect(() => {
    const onMouseMove = (e: MouseEvent) => {
      if (isDraggingTimeline) {
        // Cancel any pending RAF
        if (rafRef.current !== null) {
          cancelAnimationFrame(rafRef.current)
        }
        // Schedule update on next frame for smooth 60fps updates
        rafRef.current = requestAnimationFrame(() => {
          updateTimeFromMouse(e.clientX, true)
          rafRef.current = null
        })
      }
    }
    if (isDraggingTimeline) {
      window.addEventListener("mousemove", onMouseMove, { passive: true })
      return () => {
        window.removeEventListener("mousemove", onMouseMove)
        if (rafRef.current !== null) {
          cancelAnimationFrame(rafRef.current)
          rafRef.current = null
        }
      }
    }
  }, [isDraggingTimeline, updateTimeFromMouse])

  // One value per frame when playing: ref + tick so lyrics and scrubber stay in sync
  useEffect(() => {
    if (!isPlaying || isDraggingTimeline) return
    const audioEl = audioRef.current
    const tick = (): void => {
      const a = audioRef.current
      if (a && !a.paused) {
        playbackTimeRef.current = a.currentTime
        setPlaybackTick((t) => t + 1)
      }
      playheadRafRef.current = requestAnimationFrame(tick)
    }
    playheadRafRef.current = requestAnimationFrame(tick)
    return () => {
      if (playheadRafRef.current !== null) cancelAnimationFrame(playheadRafRef.current)
      playheadRafRef.current = null
      if (audioEl) {
        playbackTimeRef.current = audioEl.currentTime
        setPlaybackTick((t) => t + 1)
      }
    }
  }, [isPlaying, isDraggingTimeline])

  // Keyboard: Space = play/pause, Escape = deselect note
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement
      const inInput = target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.tagName === "SELECT"
      if (e.key === " ") {
        if (!inInput) {
          e.preventDefault()
          const audio = audioRef.current
          if (!audio) return
          if (audio.paused) {
            audio.play().then(() => setIsPlaying(true))
          } else {
            audio.pause()
            if (playheadRafRef.current !== null) {
              cancelAnimationFrame(playheadRafRef.current)
              playheadRafRef.current = null
            }
            playbackTimeRef.current = audio.currentTime
            setPlaybackTick((t) => t + 1)
            setIsPlaying(false)
          }
        }
      } else if (e.key === "Escape") {
        setSelectedNote(null)
      }
    }
    window.addEventListener("keydown", onKeyDown)
    return () => window.removeEventListener("keydown", onKeyDown)
  }, [])

  // Draw the piano roll
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const width = totalBeats * PIXELS_PER_BEAT
    const height = PITCH_RANGE * ROW_HEIGHT
    canvas.width = width
    canvas.height = height

    // Background
    ctx.fillStyle = "#18181b"
    ctx.fillRect(0, 0, width, height)

    // Grid lines for pitches
    for (let p = minPitch; p <= maxPitch; p++) {
      const y = (maxPitch - p) * ROW_HEIGHT
      const midi = p + 60
      const isBlackKey = [1, 3, 6, 8, 10].includes(midi % 12)

      ctx.fillStyle = isBlackKey ? "#1c1c22" : "#1f1f26"
      ctx.fillRect(0, y, width, ROW_HEIGHT)

      ctx.strokeStyle = "#27272a"
      ctx.lineWidth = 0.5
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(width, y)
      ctx.stroke()

      // C note labels
      if (midi % 12 === 0) {
        ctx.fillStyle = "#52525b"
        ctx.font = "10px monospace"
        ctx.fillText(`C${Math.floor(midi / 12) - 1}`, 2, y + ROW_HEIGHT - 2)
      }
    }

    // Beat grid lines
    for (let b = 0; b < totalBeats; b++) {
      const x = b * PIXELS_PER_BEAT
      ctx.strokeStyle = b % 16 === 0 ? "#3f3f46" : b % 4 === 0 ? "#2a2a30" : "#222228"
      ctx.lineWidth = b % 16 === 0 ? 1 : 0.5
      ctx.beginPath()
      ctx.moveTo(x, 0)
      ctx.lineTo(x, height)
      ctx.stroke()
    }

    // Draw notes
    displayResult.phrases.forEach((phrase, pIdx) => {
      phrase.notes.forEach((note, nIdx) => {
        const x = note.start_beat * PIXELS_PER_BEAT
        const y = (maxPitch - note.pitch) * ROW_HEIGHT
        const w = note.duration_beats * PIXELS_PER_BEAT
        const h = ROW_HEIGHT - 1

        const isSelected = selectedNote?.phraseIdx === pIdx && selectedNote?.noteIdx === nIdx

        // Note color by type
        const colors: Record<string, string> = {
          ":": isSelected ? "#a78bfa" : "#7c3aed",
          "*": isSelected ? "#fbbf24" : "#d97706",
          "F": isSelected ? "#6ee7b7" : "#059669",
        }
        ctx.fillStyle = colors[note.note_type] ?? colors[":"]
        ctx.beginPath()
        ctx.roundRect(x, y, Math.max(w, 3), h, 2)
        ctx.fill()

        // Note text
        if (w > 20) {
          ctx.fillStyle = "#fff"
          ctx.font = "10px sans-serif"
          ctx.fillText(note.text.trim(), x + 3, y + h - 2, w - 6)
        }
      })

      // Phrase separator
      if (phrase.end_beat > 0) {
        const x = phrase.end_beat * PIXELS_PER_BEAT
        ctx.strokeStyle = "#ef444466"
        ctx.lineWidth = 1
        ctx.setLineDash([4, 4])
        ctx.beginPath()
        ctx.moveTo(x, 0)
        ctx.lineTo(x, height)
        ctx.stroke()
        ctx.setLineDash([])
      }
    })

    // Playhead
    if (displayTime >= 0 && currentBeat >= 0 && currentBeat <= totalBeats) {
      const playheadX = currentBeat * PIXELS_PER_BEAT
      ctx.strokeStyle = "#f472b6"
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(playheadX, 0)
      ctx.lineTo(playheadX, height)
      ctx.stroke()
    }
  }, [displayResult.phrases, selectedNote, minPitch, maxPitch, totalBeats, displayTime, currentBeat])

  // Handle clicking on canvas to select notes
  const handleCanvasClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const scaleX = canvas.width / rect.width
    const scaleY = canvas.height / rect.height
    const x = (e.clientX - rect.left) * scaleX
    const y = (e.clientY - rect.top) * scaleY

    const clickedBeat = x / PIXELS_PER_BEAT
    const clickedPitch = maxPitch - Math.floor(y / ROW_HEIGHT)

    for (let pIdx = 0; pIdx < displayResult.phrases.length; pIdx++) {
      const phrase = displayResult.phrases[pIdx]
      for (let nIdx = 0; nIdx < phrase.notes.length; nIdx++) {
        const note = phrase.notes[nIdx]
        if (
          clickedBeat >= note.start_beat &&
          clickedBeat <= note.start_beat + note.duration_beats &&
          clickedPitch === note.pitch
        ) {
          setSelectedNote({ phraseIdx: pIdx, noteIdx: nIdx })
          return
        }
      }
    }
    setSelectedNote(null)
  }, [displayResult.phrases, maxPitch])

  const updateEditorAnchor = useCallback((): void => {
    if (!selectedNote || !canvasRef.current) {
      setEditorAnchor(null)
      return
    }
    const canvas = canvasRef.current
    const rect = canvas.getBoundingClientRect()
    const note = displayResult.phrases[selectedNote.phraseIdx]?.notes[selectedNote.noteIdx]
    if (!note) {
      setEditorAnchor(null)
      return
    }
    const x = note.start_beat * PIXELS_PER_BEAT
    const y = (maxPitch - note.pitch) * ROW_HEIGHT
    const w = note.duration_beats * PIXELS_PER_BEAT
    const h = ROW_HEIGHT - 1
    const scaleX = rect.width / canvas.width
    const scaleY = rect.height / canvas.height
    setEditorAnchor({
      left: rect.left + x * scaleX,
      top: rect.top + y * scaleY,
      width: w * scaleX,
      height: h * scaleY,
    })
  }, [selectedNote, displayResult.phrases, maxPitch])

  useLayoutEffect(() => {
    updateEditorAnchor()
  }, [updateEditorAnchor])

  useEffect(() => {
    if (!selectedNote) return
    const container = containerRef.current
    const onScroll = (): void => updateEditorAnchor()
    container?.addEventListener("scroll", onScroll)
    window.addEventListener("resize", onScroll)
    return () => {
      container?.removeEventListener("scroll", onScroll)
      window.removeEventListener("resize", onScroll)
    }
  }, [selectedNote, updateEditorAnchor])

  const updateNote = useCallback((field: keyof UltraStarNote, value: string | number) => {
    if (!selectedNote) return
    const newPhrases = structuredClone(displayResult.phrases)
    const note = newPhrases[selectedNote.phraseIdx].notes[selectedNote.noteIdx]
    ;(note as Record<keyof UltraStarNote, string | number>)[field] = value
    onResultChange({ ...result, phrases: newPhrases })
  }, [selectedNote, result, onResultChange, displayResult.phrases])

  const handleSave = useCallback(async () => {
    setSaving(true)
    try {
      await updateNotes(jobId, displayResult.phrases)
      lastSavedPhrasesRef.current = displayResult.phrases
    } finally {
      setSaving(false)
    }
  }, [jobId, displayResult.phrases])

  const togglePlayback = useCallback(() => {
    const audio = audioRef.current
    if (!audio) return
    if (isPlaying) {
      audio.pause()
      if (playheadRafRef.current !== null) {
        cancelAnimationFrame(playheadRafRef.current)
        playheadRafRef.current = null
      }
      playbackTimeRef.current = audio.currentTime
      setPlaybackTick((t) => t + 1)
    }
    setIsPlaying(!isPlaying)
  }, [isPlaying])

  const handleRegenerate = useCallback(async () => {
    setRegenerating(true)
    setRegenerateError(null)
    try {
      const newResult = await regenerateLyrics(jobId, lyricsParams)
      onResultChange(newResult)
      lastSavedPhrasesRef.current = newResult.phrases
      setSelectedNote(null)
    } catch (err) {
      setRegenerateError(err instanceof Error ? err.message : "Regeneration failed")
    } finally {
      setRegenerating(false)
    }
  }, [jobId, lyricsParams, onResultChange])

  const updateParam = useCallback(<TKey extends keyof LyricsParams>(key: TKey, value: LyricsParams[TKey]) => {
    setLyricsParams((prev) => ({ ...prev, [key]: value }))
  }, [])

  const sel = selectedNote
    ? displayResult.phrases[selectedNote.phraseIdx]?.notes[selectedNote.noteIdx]
    : null

  const handleSeekKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLDivElement>) => {
      const audio = audioRef.current
      if (!audio) return
      const step = e.shiftKey ? 10 : 5
      if (e.key === "ArrowLeft") {
        e.preventDefault()
        audio.currentTime = Math.max(0, audio.currentTime - step)
        playbackTimeRef.current = audio.currentTime
        setPlaybackTick((t) => t + 1)
      } else if (e.key === "ArrowRight") {
        e.preventDefault()
        audio.currentTime = Math.min(audio.duration ?? totalDuration, audio.currentTime + step)
        playbackTimeRef.current = audio.currentTime
        setPlaybackTick((t) => t + 1)
      }
    },
    [totalDuration]
  )

  const handleAudioTrackChange = useCallback((track: "original" | "vocals" | "instrumental") => {
    setAudioTrack(track)
    setIsPlaying(false)
    playbackTimeRef.current = 0
    setPlaybackTick((t) => t + 1)
    setAudioDuration(0)
  }, [])

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      <EditorHeader
        result={result}
        allNotesCount={allNotes.length}
        phrasesCount={displayResult.phrases.length}
        showSettings={showSettings}
        onToggleSettings={() => setShowSettings((s) => !s)}
        saving={saving}
        isDirty={isDirty}
        onSave={handleSave}
        onExport={() => onExport(viewAlignmentMode)}
      />

      {result.alignment_results && Object.keys(result.alignment_results).length > 1 && (
        <div className="flex items-center gap-3 flex-wrap">
          <span className="text-sm text-zinc-400">View alignment:</span>
          <select
            value={viewAlignmentMode}
            onChange={(e) => setViewAlignmentMode(e.target.value)}
            className="bg-zinc-800 border border-zinc-600 rounded-lg px-3 py-1.5 text-white text-sm
                       focus:outline-none focus:border-violet-500"
          >
            {Object.keys(result.alignment_results).map((mode) => (
              <option key={mode} value={mode}>
                {mode}
              </option>
            ))}
          </select>
          <span className="text-xs text-zinc-500">
            Switch instantly between alignments generated for this song.
          </span>
        </div>
      )}

      {showSettings && (
        <LyricsSettingsPanel
          params={lyricsParams}
          onParamChange={updateParam}
          onRegenerate={handleRegenerate}
          regenerating={regenerating}
          regenerateError={regenerateError}
        />
      )}

      <AudioControls
        jobId={jobId}
        audioTrack={audioTrack}
        isPlaying={isPlaying}
        onTogglePlayback={togglePlayback}
        onAudioTrackChange={handleAudioTrackChange}
        onEnded={() => {
          if (playheadRafRef.current !== null) {
            cancelAnimationFrame(playheadRafRef.current)
            playheadRafRef.current = null
          }
          const t = audioRef.current?.currentTime ?? 0
          playbackTimeRef.current = t
          setPlaybackTick((tick) => tick + 1)
          setIsPlaying(false)
        }}
        onTimeUpdate={() => {
          if (!isPlaying && !isDraggingTimeline) {
            const t = audioRef.current?.currentTime ?? 0
            playbackTimeRef.current = t
            setPlaybackTick((tick) => tick + 1)
          }
        }}
        onLoadedMetadata={() => setAudioDuration(audioRef.current?.duration ?? 0)}
        audioRef={audioRef}
      />

      <KaraokeWords
        phraseWords={phraseWords}
        activeWordIndex={activeWordIndex}
        displayTime={displayTime}
        displayedVerseStartTime={displayedVerseStartTime}
        isPlaying={isPlaying}
      />

      {/* Timeline + piano roll in one horizontal scroll container */}
      <div
        ref={containerRef}
        className="overflow-x-auto overflow-y-hidden rounded-xl border border-zinc-700/50"
        style={{ maxHeight: `${PITCH_RANGE * ROW_HEIGHT + 20 + 48}px` }}
      >
        <div style={{ width: totalBeats * PIXELS_PER_BEAT }}>
          <TimelineScrubber
            totalBeats={totalBeats}
            totalDuration={totalDuration}
            currentBeat={currentBeat}
            displayTime={displayTime}
            beatToTime={beatToTime}
            timelineRef={timelineRef}
            onTimelineClick={handleTimelineClick}
            onMouseDown={handleTimelineMouseDown}
            onMouseUp={handleTimelineMouseUp}
            onMouseLeave={handleTimelineMouseLeave}
            onSeekKeyDown={handleSeekKeyDown}
          />
          <section
            className="overflow-y-auto overflow-x-hidden bg-zinc-900"
            style={{ maxHeight: `${PITCH_RANGE * ROW_HEIGHT + 20}px` }}
            aria-label="Piano roll"
          >
            <canvas
              ref={canvasRef}
              onClick={handleCanvasClick}
              onContextMenu={(e) => e.preventDefault()}
              className="cursor-crosshair"
              style={{ imageRendering: "pixelated" }}
              aria-label="Note grid: click a note to select and edit"
            />
          </section>
        </div>
      </div>

      {sel && selectedNote && editorAnchor !== null && (
        <NoteEditorPopover
          note={sel}
          phraseIdx={selectedNote.phraseIdx}
          noteIdx={selectedNote.noteIdx}
          anchor={editorAnchor}
          onClose={() => setSelectedNote(null)}
          onUpdate={updateNote}
        />
      )}

      {!selectedNote && <EditorHint />}

      <LyricsPreviewSection phrases={displayResult.phrases} />

      <RawUltrastarSection ultrastarTxt={displayResult.ultrastar_txt} />
    </div>
  )
}
