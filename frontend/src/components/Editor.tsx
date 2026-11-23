import { createEffect as createPostRenderEffect, createMemo, createSignal, onCleanup } from "solid-js"
import { createEffect } from "../solidCompat"
import type { JobResult, UltraStarNote } from "../types"
import { useInstrumentalPiano } from "../useInstrumentalPiano"
import { switchAlignmentMode, updateNotes } from "../api"
import { playPianoNote, playTestPiano, type PianoResumeRef } from "../useInstrumentalPiano"
import type { AudioTrack } from "./AudioControls"
import { AudioControls } from "./AudioControls"
import { EditorHeader } from "./EditorHeader"
import { EditorHint } from "./EditorHint"
import { LyricsPreviewSection } from "./LyricsPreviewSection"
import { NoteEditorPopover } from "./NoteEditorPopover"
import { RawUltrastarSection } from "./RawUltrastarSection"
import { TimelineScrubber } from "./TimelineScrubber"
import { WordAlignmentEditor } from "./WordAlignmentEditor"

interface EditorProps {
  jobId: string
  result: JobResult
  onResultChange: (result: JobResult) => void
  onExport: () => void
  saveNotes?: (phrases: JobResult["phrases"]) => Promise<string>
  audioSourceUrl?: string | null
  audioSourceUrls?: Partial<Record<AudioTrack, string>>
  availableAudioTracks?: AudioTrack[]
  confirmSaveMessage?: string
  showAdvancedActions?: boolean
}

// Keep zoom bounded so long songs remain usable without making beat snapping
// impossible to see or allocating an unreasonably large canvas.
const MIN_PIXELS_PER_BEAT = 2
const MAX_PIXELS_PER_BEAT = 16
const ZOOM_WHEEL_SENSITIVITY = 0.0025
const CANVAS_TILE_WIDTH = 8192
const MAX_EDIT_HISTORY = 100
const DEFAULT_PIXELS_PER_BEAT = 4
const ROW_HEIGHT = 28
const MIN_PLAYBACK_RATE = 0.5
const MAX_PLAYBACK_RATE = 2
const PLAYBACK_RATE_STEP = 0.1

type NoteDragMode = "move" | "start" | "end"
type CanvasCursor = "default" | "grab" | "grabbing" | "ew-resize" | "crosshair"
type NoteSelection = { phraseIdx: number; noteIdx: number }
type NoteDragOrigin = NoteSelection & {
  startBeat: number
  duration: number
  pitch: number
}
type NoteDragState = {
  active: NoteSelection
  notes: NoteDragOrigin[]
  mode: NoteDragMode
  startX: number
  startY: number
  playbackHeadBeat: number | null
}
type SelectionRect = {
  startX: number
  startY: number
  currentX: number
  currentY: number
  additive: boolean
}
type PianoRollRenderState = {
  root: HTMLDivElement | null
  isZooming: boolean
  pitchRange: number
  pixelsPerBeat: number
  desiredCanvasWidth: number
  minPitch: number
  maxPitch: number
  totalBeats: number
  phrases: JobResult["phrases"]
  gapBeats: number
  selectedNotes: NoteSelection[]
}

const noteSelectionKey = ({ phraseIdx, noteIdx }: NoteSelection): string => `${phraseIdx}:${noteIdx}`

export const Editor = (props: EditorProps) => {
  const jobId = () => props.jobId
  const result = () => props.result
  const onResultChange = props.onResultChange
  const onExport = props.onExport
  const showAdvancedActions = props.showAdvancedActions ?? true
  const [selectedNote, setSelectedNote] = createSignal<{ phraseIdx: number; noteIdx: number } | null>(null)
  const [selectedNotes, setSelectedNotes] = createSignal<NoteSelection[]>([])
  const [canvasCursor, setCanvasCursor] = createSignal<CanvasCursor>("default")
  const [selectionRect, setSelectionRect] = createSignal<SelectionRect | null>(null)
  const [audioTrack, setAudioTrack] = createSignal<AudioTrack>(
    props.audioSourceUrl != null || props.audioSourceUrls ? "original" : "vocals",
  )
  const [isPlaying, setIsPlaying] = createSignal(false)
  const [saving, setSaving] = createSignal(false)
  const [saveError, setSaveError] = createSignal("")
  const [switchingMode, setSwitchingMode] = createSignal<string | null>(null)
  const [alignmentDirty, setAlignmentDirty] = createSignal(false)
  const [exportWarning, setExportWarning] = createSignal(false)
  const [audioDuration, setAudioDuration] = createSignal(0)
  const [isDraggingTimeline, setIsDraggingTimeline] = createSignal(false)
  const [dragTime, setDragTime] = createSignal<number | null>(null)
  const [snapToPrevious, setSnapToPrevious] = createSignal(true)
  const [snapToPlaybackHead, setSnapToPlaybackHead] = createSignal(false)
  const [pixelsPerBeat, setPixelsPerBeat] = createSignal(DEFAULT_PIXELS_PER_BEAT)
  const [isZooming, setIsZooming] = createSignal(false)
  const [playbackRate, setPlaybackRate] = createSignal(1)
  const [playbackTick, setPlaybackTick] = createSignal(0)
  const [isExpanded, setIsExpanded] = createSignal(false)
  const playbackTimeRef = { current: 0 }
  const pendingSeekTimeRef: { current: number | null } = { current: null }
  const timelineRef: { current: HTMLDivElement | null } = { current: null }
  const suppressTimelineClickRef = { current: false }
  const wheelZoomTimeoutRef: { current: number | null } = { current: null }
  const wheelZoomAnchorRef: { current: { beat: number; offsetPx: number } | null } = { current: null }
  const suspendPlayheadFollowRef = { current: false }
  const playheadBeatAtZoomRef = { current: 0 }
  const undoHistoryRef: { current: Array<JobResult["phrases"]> } = { current: [] }
  const redoHistoryRef: { current: Array<JobResult["phrases"]> } = { current: [] }
  const historyJobIdRef = { current: props.jobId }
  const dragHistoryRecordedRef = { current: false }
  const rafRef: { current: number | null } = { current: null }
  const playheadRafRef: { current: number | null } = { current: null }
  const audioRef: { current: HTMLAudioElement | null } = { current: null }
  const trackChangeTimeRef: { current: number | null } = { current: null }
  const trackChangeWasPlayingRef = { current: false }
  const pianoResumeRef: { current: PianoResumeRef | null } = { current: null }
  const canvasRef: { current: HTMLDivElement | null } = { current: null }
  const [pianoRollElement, setPianoRollElement] = createSignal<HTMLDivElement | null>(null)
  const noteDragRef: { current: NoteDragState | null } = { current: null }
  const noteDragMovedRef = { current: false }
  const selectionDragRef: { current: SelectionRect | null } = { current: null }
  const containerRef: { current: HTMLDivElement | null } = { current: null }
  const [timelineContainer, setTimelineContainer] = createSignal<HTMLDivElement | null>(null)
  const [lyricsAlignmentOpen, setLyricsAlignmentOpen] = createSignal(false)
  const [editorAnchor, setEditorAnchor] = createSignal<{
    left: number
    top: number
    width: number
    height: number
  } | null>(null)
  const displayResult = () => result()

  const lastSavedPhrasesRef: { current: JobResult["phrases"] } = { current: displayResult().phrases }
  const [savedPhrasesTick, setSavedPhrasesTick] = createSignal(0)
  const isDirty = createMemo(() => {
    savedPhrasesTick()
    return displayResult().phrases !== lastSavedPhrasesRef.current
  })

  // Compute pitch bounds for the display
  const allNotes = createMemo(() => displayResult().phrases.flatMap((p) => p.notes))
  const noteMinPitch = createMemo(() => allNotes().length > 0 ? Math.min(...allNotes().map((n) => n.pitch)) : 0)
  const noteMaxPitch = createMemo(() => allNotes().length > 0 ? Math.max(...allNotes().map((n) => n.pitch)) : 0)
  const minPitch = createMemo(() => noteMinPitch() - 1)
  const maxPitch = createMemo(() => noteMaxPitch() + 1)
  const pitchRange = createMemo(() => maxPitch() - minPitch() + 1)

  const variant = createMemo(() => result().alignment_results?.[result().current_alignment_mode ?? "canonical"])
  const gap_ms_effective = createMemo(() => variant()?.gap_ms ?? result().metadata.gap_ms)
  const gap_s = createMemo(() => gap_ms_effective() / 1000)
  const ustar_bpm = createMemo(() => displayResult().metadata.bpm * 4)
  // UltraStar note beats begin after GAP; timeline beats begin at audio 0:00.
  const secondsPerBeat = createMemo(() => 60 / (ustar_bpm() * 4))
  const gapBeats = createMemo(() => gap_s() / secondsPerBeat())
  const audioTimeToBeat = (time: number): number => time / secondsPerBeat()
  const audioBeatToTime = (beat: number): number => beat * secondsPerBeat()
  const chartBeatToTime = (beat: number): number => gap_s() + beat * secondsPerBeat()

  // Keep the seekable strip as long as the audio, including trailing silence.
  // Previously it stopped at the last note, so clicking near the end of a
  // song could seek to an earlier position than the cursor suggested.
  const chartEndBeat = createMemo(() => allNotes().length > 0
    ? Math.max(...allNotes().map((n) => n.start_beat + n.duration_beats)) + 20
    : 100)
  const audioEndBeat = createMemo(() => audioDuration() > 0 ? Math.max(0, audioTimeToBeat(audioDuration())) : 0)
  const totalBeats = createMemo(() => Math.max(
    chartEndBeat() + gapBeats(),
    Math.ceil(audioEndBeat()),
  ))
  const desiredCanvasWidth = createMemo(() => Math.max(1, Math.ceil(totalBeats() * pixelsPerBeat())))
  const canvasTileCount = createMemo(() => Math.ceil(desiredCanvasWidth() / CANVAS_TILE_WIDTH))

  useInstrumentalPiano({
    audioRef,
    audioTrack,
    isPlaying,
    phrases: () => displayResult().phrases,
    gap_s,
    ustar_bpm,
    playbackRate,
    pianoResumeRef,
  })

  const playbackPosition = (): number => {
    if (pendingSeekTimeRef.current !== null) return pendingSeekTimeRef.current
    const audio = audioRef.current
    return audio?.readyState && Number.isFinite(audio.currentTime)
      ? audio.currentTime
      : playbackTimeRef.current
  }

  createEffect(() => {
    const audio = audioRef.current
    if (audio) audio.playbackRate = playbackRate()
  }, () => undefined)

  // Single value for UI: ref updated every frame when playing; sync from currentTime when paused/seek
  const displayTime = createMemo(() => {
    playbackTick()
    const draggingTime = dragTime()
    return draggingTime !== null ? draggingTime : playbackPosition()
  })
  const currentBeat = createMemo(() => audioTimeToBeat(displayTime()))

  // Keep the editor viewport following the playhead without changing audio
  // position. When the line leaves either side, jump the shared timeline and
  // piano-roll scroller so the line starts the next visible section.
  createEffect(() => {
    const container = containerRef.current
    if (
      !container
      || isDraggingTimeline()
      || pendingSeekTimeRef.current !== null
      || !Number.isFinite(currentBeat())
    ) return
    if (wheelZoomAnchorRef.current) return
    if (suspendPlayheadFollowRef.current) {
      if (Math.abs(currentBeat() - playheadBeatAtZoomRef.current) < 0.01) return
      suspendPlayheadFollowRef.current = false
    }
    const visibleBeat = Math.max(0, Math.min(totalBeats(), currentBeat()))
    const playheadX = visibleBeat * pixelsPerBeat()
    const playheadWidth = 2
    const viewportStart = container.scrollLeft
    const viewportEnd = viewportStart + container.clientWidth
    if (playheadX >= viewportStart && playheadX + playheadWidth <= viewportEnd) return
    const maxScroll = Math.max(0, container.scrollWidth - container.clientWidth)
    const nextScrollLeft = Math.min(maxScroll, Math.max(0, playheadX))
    if (Math.abs(nextScrollLeft - container.scrollLeft) > 1) container.scrollLeft = nextScrollLeft
  }, () => undefined)

  const handleTimelineWheel = (event: WheelEvent) => {
    if (!event.altKey && !event.ctrlKey) return
    const delta = Math.abs(event.deltaY) >= Math.abs(event.deltaX) ? event.deltaY : event.deltaX
    if (delta === 0) return
    event.preventDefault()
    event.stopPropagation()
    setIsZooming(true)
    if (wheelZoomTimeoutRef.current !== null) window.clearTimeout(wheelZoomTimeoutRef.current)
    wheelZoomTimeoutRef.current = window.setTimeout(() => {
      wheelZoomTimeoutRef.current = null
      setIsZooming(false)
    }, 120)
    const container = containerRef.current
    const currentScale = pixelsPerBeat()
    const nextScale = Math.min(
      MAX_PIXELS_PER_BEAT,
      Math.max(MIN_PIXELS_PER_BEAT, currentScale * Math.exp(-delta * ZOOM_WHEEL_SENSITIVITY)),
    )
    if (nextScale === currentScale) return
    suspendPlayheadFollowRef.current = true
    playheadBeatAtZoomRef.current = currentBeat()
    if (container) {
      const rect = container.getBoundingClientRect()
      const offsetPx = event.clientX - rect.left
      const previousAnchor = wheelZoomAnchorRef.current
      wheelZoomAnchorRef.current = {
        beat: previousAnchor === null
          ? (container.scrollLeft + offsetPx) / currentScale
          : previousAnchor.beat + (offsetPx - previousAnchor.offsetPx) / currentScale,
        offsetPx,
      }
    }
    setPixelsPerBeat(nextScale)
  }

  createEffect(() => {
    const container = timelineContainer()
    if (!container) return
    container.addEventListener("wheel", handleTimelineWheel, { passive: false, capture: true })
    onCleanup(() => container.removeEventListener("wheel", handleTimelineWheel, true))
  }, () => undefined)

  createEffect(() => {
    onCleanup(() => {
      if (wheelZoomTimeoutRef.current !== null) window.clearTimeout(wheelZoomTimeoutRef.current)
    })
  }, () => undefined)

  createPostRenderEffect(
    () => [pixelsPerBeat(), containerRef.current] as const,
    ([scale, container]) => {
      const anchor = wheelZoomAnchorRef.current
      if (!container || !anchor) return
      const maxScroll = Math.max(0, container.scrollWidth - container.clientWidth)
      playheadBeatAtZoomRef.current = currentBeat()
      suspendPlayheadFollowRef.current = true
      container.scrollLeft = Math.min(
        maxScroll,
        Math.max(0, anchor.beat * scale - anchor.offsetPx),
      )
      wheelZoomAnchorRef.current = null
    },
  )

  const recordEdit = (previous: JobResult["phrases"]) => {
    undoHistoryRef.current.push(structuredClone(previous))
    if (undoHistoryRef.current.length > MAX_EDIT_HISTORY) undoHistoryRef.current.shift()
    redoHistoryRef.current = []
  }

  const applyEditHistory = (direction: "undo" | "redo"): void => {
    const source = direction === "undo" ? undoHistoryRef.current : redoHistoryRef.current
    const destination = direction === "undo" ? redoHistoryRef.current : undoHistoryRef.current
    const previous = source.pop()
    if (!previous) return
    destination.push(structuredClone(displayResult().phrases))
    if (destination.length > MAX_EDIT_HISTORY) destination.shift()
    onResultChange({ ...result(), phrases: previous })
  }

  createEffect(() => {
    const currentJobId = jobId()
    if (historyJobIdRef.current === currentJobId) return
    undoHistoryRef.current = []
    redoHistoryRef.current = []
    lastSavedPhrasesRef.current = displayResult().phrases
    setSavedPhrasesTick((tick) => tick + 1)
    historyJobIdRef.current = currentJobId
  }, () => undefined)

  const totalDuration = createMemo(() => (audioDuration() > 0
    ? audioDuration()
    : displayResult().words.length > 0
      ? Math.max(...displayResult().words.map((w) => w.end)) + 1
      : 60))

  const seekByBeat = (direction: -1 | 1): boolean => {
    const audio = audioRef.current
    if (!audio) return false
    const currentTime = playbackPosition()
    const targetBeat = Math.max(
      0,
      Math.min(totalBeats(), audioTimeToBeat(currentTime) + direction),
    )
    const maxTime = audioDuration() > 0 ? audioDuration() : audioBeatToTime(totalBeats())
    const targetTime = Math.max(0, Math.min(maxTime, audioBeatToTime(targetBeat)))
    seekAudioTo(targetTime)
    return true
  }

  const seekToSegmentBoundary = (direction: -1 | 1): boolean => {
    const audio = audioRef.current
    if (!audio) return false
    const currentTime = playbackPosition()
    const currentBeat = audioTimeToBeat(currentTime)
    const boundaries = displayResult().phrases.flatMap((phrase) =>
      phrase.notes.flatMap((note) => [
        note.start_beat + gapBeats(),
        note.start_beat + note.duration_beats + gapBeats(),
      ]),
    ).sort((left, right) => left - right)
    const epsilon = 0.01
    const targetBeat = direction < 0
      ? [...boundaries].reverse().find((beat) => beat < currentBeat - epsilon)
      : boundaries.find((beat) => beat > currentBeat + epsilon)
    if (targetBeat === undefined) return false
    const maxTime = audioDuration() > 0 ? audioDuration() : audioBeatToTime(totalBeats())
    const targetTime = Math.max(0, Math.min(maxTime, audioBeatToTime(targetBeat)))
    seekAudioTo(targetTime)
    return true
  }

  const changePlaybackRate = (delta: number): boolean => {
    const nextRate = Math.min(
      MAX_PLAYBACK_RATE,
      Math.max(MIN_PLAYBACK_RATE, Number((playbackRate() + delta).toFixed(2))),
    )
    if (nextRate === playbackRate()) return false
    setPlaybackRate(nextRate)
    const audio = audioRef.current
    if (audio) audio.playbackRate = nextRate
    return true
  }

  const nudgeSelectedNotes = (key: string): boolean => {
    const primarySelection = selectedNote()
    const selection = selectedNotes().length > 0
      ? selectedNotes()
      : primarySelection
        ? [primarySelection]
        : []
    if (selection.length === 0) return false
    const newPhrases = structuredClone(displayResult().phrases)
    let changed = false
    for (const { phraseIdx, noteIdx } of selection) {
      const note = newPhrases[phraseIdx]?.notes[noteIdx]
      if (!note) continue
      if (key === "ArrowLeft") {
        const next = Math.max(0, note.start_beat - 1)
        changed ||= next !== note.start_beat
        note.start_beat = next
      } else if (key === "ArrowRight") {
        note.start_beat += 1
        changed = true
      } else if (key === "ArrowUp") {
        const next = Math.min(48, note.pitch + 1)
        changed ||= next !== note.pitch
        note.pitch = next
      } else if (key === "ArrowDown") {
        const next = Math.max(-24, note.pitch - 1)
        changed ||= next !== note.pitch
        note.pitch = next
      }
      else return false
    }
    if (!changed) return false
    recordEdit(displayResult().phrases)
    onResultChange({ ...result(), phrases: newPhrases })
    return true
  }

  const updateTimeFromMouse = (clientX: number, isDragging: boolean) => {
    const timeline = timelineRef.current
    if (!timeline) return
    const rect = timeline.getBoundingClientRect()
    const x = clientX - rect.left
    const clickedBeat = Math.max(0, Math.min(totalBeats(), x / pixelsPerBeat()))
    const maxTime = audioDuration() > 0 ? audioDuration() : audioBeatToTime(totalBeats())
    const clickedTime = Math.max(0, Math.min(maxTime, audioBeatToTime(clickedBeat)))
    if (isDragging) {
      setDragTime(clickedTime)
    } else {
      seekAudioTo(clickedTime)
    }
    }

  const seekAudioTo = (time: number): void => {
    const audio = audioRef.current
    if (!audio || !Number.isFinite(time)) return
    const maxTime = Number.isFinite(audio.duration) && audio.duration > 0
      ? audio.duration
      : Number.POSITIVE_INFINITY
    const targetTime = Math.min(maxTime, Math.max(0, time))
    if (
      !audio.seeking
      && Number.isFinite(audio.currentTime)
      && Math.abs(audio.currentTime - targetTime) < 0.001
    ) {
      pendingSeekTimeRef.current = null
      playbackTimeRef.current = playbackPosition()
    } else {
      pendingSeekTimeRef.current = targetTime
      audio.currentTime = targetTime
    }
    setPlaybackTick((tick) => tick + 1)
    setDragTime(null)
  }

  const finishTimelineSeek = (suppressClick: boolean): void => {
    const draggingTime = dragTime()
    if (draggingTime !== null) {
      seekAudioTo(draggingTime)
      suppressTimelineClickRef.current = suppressClick
    }
    setIsDraggingTimeline(false)
  }

  const handleTimelineClick = (e: MouseEvent & { currentTarget: HTMLDivElement }) => {
    if (suppressTimelineClickRef.current) {
      suppressTimelineClickRef.current = false
      return
    }
    if (!isDraggingTimeline()) {
      updateTimeFromMouse(e.clientX, false)
    }
  }

  const handleTimelineMouseDown = (e: MouseEvent & { currentTarget: HTMLDivElement }) => {
    if (e.button !== 0) return // Only left mouse button
    setIsDraggingTimeline(true)
    setDragTime(null)
    updateTimeFromMouse(e.clientX, true)
  }

  const handleTimelineMouseUp = () => {
    finishTimelineSeek(true)
  }

  // Global mouseup handler to stop dragging if mouse released outside timeline
  createEffect(() => {
    const onMouseUp = () => finishTimelineSeek(false)
    if (isDraggingTimeline()) {
      window.addEventListener("mouseup", onMouseUp)
      onCleanup(() => window.removeEventListener("mouseup", onMouseUp))
    }
  }, () => undefined)

  // Global mousemove handler for dragging - throttled with requestAnimationFrame
  createEffect(() => {
    const onMouseMove = (e: MouseEvent) => {
      if (isDraggingTimeline()) {
        // Cancel any pending RAF
        if (rafRef.current !== null) {
          cancelAnimationFrame(rafRef.current)
        }
        // Schedule update on next frame for smooth 60fps updates
        rafRef.current = requestAnimationFrame(() => {
          if (isDraggingTimeline()) updateTimeFromMouse(e.clientX, true)
          rafRef.current = null
        })
      }
    }
    if (isDraggingTimeline()) {
      window.addEventListener("mousemove", onMouseMove, { passive: true })
      onCleanup(() => {
        window.removeEventListener("mousemove", onMouseMove)
        if (rafRef.current !== null) {
          cancelAnimationFrame(rafRef.current)
          rafRef.current = null
        }
      })
    }
  }, () => undefined)

  // One value per frame when playing: ref + tick so lyrics and scrubber stay in sync
  createEffect(() => {
    if (!isPlaying() || isDraggingTimeline()) return
    const audioEl = audioRef.current
    const tick = (): void => {
      const a = audioRef.current
      if (a && !a.paused && !a.seeking && pendingSeekTimeRef.current === null) {
        playbackTimeRef.current = a.currentTime
        setPlaybackTick((t) => t + 1)
      }
      playheadRafRef.current = requestAnimationFrame(tick)
    }
    playheadRafRef.current = requestAnimationFrame(tick)
    onCleanup(() => {
      if (playheadRafRef.current !== null) cancelAnimationFrame(playheadRafRef.current)
      playheadRafRef.current = null
      if (audioEl && !audioEl.seeking && pendingSeekTimeRef.current === null) {
        playbackTimeRef.current = audioEl.currentTime
        setPlaybackTick((t) => t + 1)
      }
    })
  }, () => undefined)

  // Keyboard: Space = play/pause, Escape = deselect notes
  createEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement
      const inInput = target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.tagName === "SELECT" || target.isContentEditable
      if (e.defaultPrevented) return
      const isUndoShortcut = (e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "z"
      const isRedoShortcut = (e.metaKey || e.ctrlKey) && (e.key.toLowerCase() === "y" || (e.key.toLowerCase() === "z" && e.shiftKey))
      if (!inInput && isUndoShortcut) {
        e.preventDefault()
        applyEditHistory(e.shiftKey ? "redo" : "undo")
      } else if (!inInput && isRedoShortcut) {
        e.preventDefault()
        applyEditHistory("redo")
      } else if (e.key === " ") {
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
            playbackTimeRef.current = playbackPosition()
            setPlaybackTick((t) => t + 1)
            setIsPlaying(false)
          }
        }
      } else if (!inInput && (e.key === "[" || e.key === "]")) {
        if (changePlaybackRate(e.key === "]" ? PLAYBACK_RATE_STEP : -PLAYBACK_RATE_STEP)) e.preventDefault()
      } else if (e.key === "Escape") {
        if (isExpanded()) {
          setIsExpanded(false)
          e.preventDefault()
        } else {
          setSelectedNote(null)
          setSelectedNotes([])
        }
      } else if (!inInput && (e.metaKey || e.ctrlKey) && !e.altKey && ["ArrowUp", "ArrowDown"].includes(e.key)) {
        if (nudgeSelectedNotes(e.key)) e.preventDefault()
      } else if (!inInput && !e.altKey && !e.ctrlKey && !e.metaKey && ["ArrowUp", "ArrowDown"].includes(e.key)) {
        if (seekToSegmentBoundary(e.key === "ArrowUp" ? -1 : 1)) e.preventDefault()
      } else if (!inInput && !e.altKey && !e.ctrlKey && !e.metaKey && e.shiftKey && (selectedNotes().length > 0 || selectedNote()) && (e.key === "ArrowLeft" || e.key === "ArrowRight")) {
        if (nudgeSelectedNotes(e.key)) e.preventDefault()
      } else if (!inInput && !e.altKey && !e.ctrlKey && !e.metaKey && (e.key === "ArrowLeft" || e.key === "ArrowRight")) {
        if (seekByBeat(e.key === "ArrowLeft" ? -1 : 1)) e.preventDefault()
      }
    }
    window.addEventListener("keydown", onKeyDown)
    onCleanup(() => window.removeEventListener("keydown", onKeyDown))
  }, () => undefined)

  // Draw the piano roll
  const drawPianoRoll = ({
    root,
    isZooming,
    pitchRange,
    pixelsPerBeat,
    desiredCanvasWidth,
    minPitch,
    maxPitch,
    totalBeats,
    phrases,
    gapBeats,
    selectedNotes,
  }: PianoRollRenderState) => {
    if (!root || isZooming) return
    const canvases = Array.from(root.querySelectorAll("canvas"))
    if (canvases.length === 0) return

    const height = pitchRange * ROW_HEIGHT
    const dpr = 1
    const pixelsPerBeatValue = pixelsPerBeat
    const colors: Record<string, string> = {
      ":": "#7c3aed",
      "*": "#d97706",
      "F": "#059669",
    }

    canvases.forEach((canvas, tileIndex) => {
      const tileOffsetX = tileIndex * CANVAS_TILE_WIDTH
      const width = Math.min(CANVAS_TILE_WIDTH, desiredCanvasWidth - tileOffsetX)
      const ctx = canvas.getContext("2d")
      if (!ctx || width <= 0) return
      canvas.style.width = `${width}px`
      canvas.style.height = `${height}px`
      canvas.width = Math.round(width * dpr)
      canvas.height = Math.round(height * dpr)
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

      ctx.fillStyle = "#18181b"
      ctx.fillRect(0, 0, width, height)

      for (let pitch = minPitch; pitch <= maxPitch; pitch++) {
        const y = (maxPitch - pitch) * ROW_HEIGHT
        const midi = pitch + 60
        const isBlackKey = [1, 3, 6, 8, 10].includes(midi % 12)
        ctx.fillStyle = isBlackKey ? "#191923" : "#22222c"
        ctx.fillRect(0, y, width, ROW_HEIGHT)
        ctx.strokeStyle = midi % 12 === 0 ? "#52525b" : "#34343f"
        ctx.lineWidth = midi % 12 === 0 ? 1 : 0.75
        ctx.beginPath()
        ctx.moveTo(0, y)
        ctx.lineTo(width, y)
        ctx.stroke()
      }

      const firstBeat = Math.max(0, Math.ceil(tileOffsetX / pixelsPerBeatValue))
      const lastBeat = Math.min(totalBeats, Math.ceil((tileOffsetX + width) / pixelsPerBeatValue))
      for (let beat = firstBeat; beat <= lastBeat; beat++) {
        const x = beat * pixelsPerBeatValue - tileOffsetX
        ctx.strokeStyle = beat % 16 === 0 ? "#71717a" : beat % 4 === 0 ? "#484852" : "#30303a"
        ctx.lineWidth = beat % 16 === 0 ? 1.5 : beat % 4 === 0 ? 1 : 0.75
        ctx.beginPath()
        ctx.moveTo(x, 0)
        ctx.lineTo(x, height)
        ctx.stroke()
      }

      phrases.forEach((phrase, phraseIdx) => {
        phrase.notes.forEach((note, noteIdx) => {
          const x = (note.start_beat + gapBeats) * pixelsPerBeatValue - tileOffsetX
          const y = (maxPitch - note.pitch) * ROW_HEIGHT
          const noteWidth = note.duration_beats * pixelsPerBeatValue
          if (x + noteWidth < 0 || x > width) return
          const isSelected = selectedNotes.some((selection) => selection.phraseIdx === phraseIdx && selection.noteIdx === noteIdx)
          ctx.fillStyle = isSelected
            ? note.note_type === "*" ? "#fbbf24" : note.note_type === "F" ? "#6ee7b7" : "#a78bfa"
            : colors[note.note_type] ?? colors[":"]
          ctx.beginPath()
          ctx.roundRect(x, y, Math.max(noteWidth, 3), ROW_HEIGHT - 1, 2)
          ctx.fill()
          ctx.strokeStyle = isSelected ? "#e4e4e7" : "#c4b5fd"
          ctx.lineWidth = 0.75
          ctx.stroke()
        })

        if (phrase.end_beat > 0) {
          const x = (phrase.end_beat + gapBeats) * pixelsPerBeatValue - tileOffsetX
          if (x < 0 || x > width) return
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
    })
  }

  createPostRenderEffect(
    (): PianoRollRenderState => ({
      root: pianoRollElement(),
      isZooming: isZooming(),
      pitchRange: pitchRange(),
      pixelsPerBeat: pixelsPerBeat(),
      desiredCanvasWidth: desiredCanvasWidth(),
      minPitch: minPitch(),
      maxPitch: maxPitch(),
      totalBeats: totalBeats(),
      phrases: displayResult().phrases,
      gapBeats: gapBeats(),
      selectedNotes: selectedNotes(),
    }),
    drawPianoRoll,
  )

  const findNoteAtPoint = (clientX: number, clientY: number): { phraseIdx: number; noteIdx: number } | null => {
    const root = canvasRef.current
    if (!root) return null
    const rect = root.getBoundingClientRect()
    const y = clientY - rect.top
    const clickedBeat = (clientX - rect.left) / pixelsPerBeat() - gapBeats()
    const clickedPitch = maxPitch() - Math.floor(y / ROW_HEIGHT)

    for (let pIdx = 0; pIdx < displayResult().phrases.length; pIdx++) {
      const phrase = displayResult().phrases[pIdx]
      for (let nIdx = 0; nIdx < phrase.notes.length; nIdx++) {
        const note = phrase.notes[nIdx]
        if (
          clickedBeat >= note.start_beat &&
          clickedBeat <= note.start_beat + note.duration_beats &&
          clickedPitch === note.pitch
        ) {
          return { phraseIdx: pIdx, noteIdx: nIdx }
        }
      }
    }
    return null
  }

  const cursorForPoint = (clientX: number, clientY: number): CanvasCursor => {
    const hit = findNoteAtPoint(clientX, clientY)
    if (!hit) return "default"
    const note = displayResult().phrases[hit.phraseIdx]?.notes[hit.noteIdx]
    const root = canvasRef.current
    if (!note || !root) return "default"
    const rect = root.getBoundingClientRect()
    const pointerBeat = (clientX - rect.left) / pixelsPerBeat() - gapBeats()
    const offset = pointerBeat - note.start_beat
    const edgeThreshold = 6 / pixelsPerBeat()
    return offset <= edgeThreshold || offset >= note.duration_beats - edgeThreshold
      ? "ew-resize"
      : "grab"
  }

  const updateSelection = (next: NoteSelection[], primary: NoteSelection | null = null): void => {
    const unique = [...new Map(next.map((selection) => [noteSelectionKey(selection), selection])).values()]
    const active = primary && unique.some((selection) => noteSelectionKey(selection) === noteSelectionKey(primary))
      ? primary
      : unique[unique.length - 1] ?? null
    setSelectedNotes(unique)
    setSelectedNote(active)
  }

  const notesInSelection = (rect: SelectionRect): NoteSelection[] => {
    const root = canvasRef.current
    if (!root || totalBeats() <= 0) return []
    const left = Math.min(rect.startX, rect.currentX)
    const right = Math.max(rect.startX, rect.currentX)
    const top = Math.min(rect.startY, rect.currentY)
    const bottom = Math.max(rect.startY, rect.currentY)
    const selected: NoteSelection[] = []
    displayResult().phrases.forEach((phrase, phraseIdx) => {
      phrase.notes.forEach((note, noteIdx) => {
        const noteLeft = (note.start_beat + gapBeats()) * pixelsPerBeat()
        const noteRight = (note.start_beat + note.duration_beats + gapBeats()) * pixelsPerBeat()
        const noteTop = (maxPitch() - note.pitch) * ROW_HEIGHT
        const noteBottom = noteTop + ROW_HEIGHT - 1
        if (noteLeft >= left && noteRight <= right && noteTop >= top && noteBottom <= bottom) {
          selected.push({ phraseIdx, noteIdx })
        }
      })
    })
    return selected
  }

  const handleCanvasPointerDown = (event: PointerEvent & { currentTarget: HTMLCanvasElement }) => {
    if (event.button !== 0) return
    const hit = findNoteAtPoint(event.clientX, event.clientY)
    const root = canvasRef.current
    if (!root) return

    if (event.shiftKey && hit) {
      const isSelected = selectedNotes().some((selection) => noteSelectionKey(selection) === noteSelectionKey(hit))
      const nextSelection = isSelected
        ? selectedNotes().filter((selection) => noteSelectionKey(selection) !== noteSelectionKey(hit))
        : [...selectedNotes(), hit]
      updateSelection(nextSelection, isSelected && noteSelectionKey(selectedNote() ?? hit) === noteSelectionKey(hit)
        ? null
        : isSelected
          ? selectedNote()
          : hit)
      setCanvasCursor(cursorForPoint(event.clientX, event.clientY))
      event.preventDefault()
      return
    }

    if (!hit) {
      const rect = root.getBoundingClientRect()
      const point = { x: event.clientX - rect.left, y: event.clientY - rect.top }
      const marquee: SelectionRect = {
        startX: point.x,
        startY: point.y,
        currentX: point.x,
        currentY: point.y,
        additive: event.shiftKey,
      }
      selectionDragRef.current = marquee
      setSelectionRect(marquee)
      if (!event.shiftKey) updateSelection([])
      setCanvasCursor("crosshair")
      event.currentTarget.setPointerCapture(event.pointerId)
      event.preventDefault()
      return
    }
    const note = displayResult().phrases[hit.phraseIdx]?.notes[hit.noteIdx]
    if (!note) return
    const rect = root.getBoundingClientRect()
    const pointerBeat = (event.clientX - rect.left) / pixelsPerBeat() - gapBeats()
    const offset = pointerBeat - note.start_beat
    const edgeThreshold = 6 / pixelsPerBeat()
    const mode: NoteDragMode = offset <= edgeThreshold ? "start" : offset >= note.duration_beats - edgeThreshold ? "end" : "move"
    const isSelected = selectedNotes().some((selection) => noteSelectionKey(selection) === noteSelectionKey(hit))
    const dragSelection = isSelected ? selectedNotes() : [hit]
    updateSelection(dragSelection, hit)
    const livePlaybackBeat = audioRef.current
      ? audioTimeToBeat(playbackPosition()) - gapBeats()
      : currentBeat() - gapBeats()
    noteDragRef.current = {
      active: hit,
      notes: dragSelection.flatMap((selection) => {
        const selected = displayResult().phrases[selection.phraseIdx]?.notes[selection.noteIdx]
        return selected
          ? [{
              ...selection,
              startBeat: selected.start_beat,
              duration: selected.duration_beats,
              pitch: selected.pitch,
            }]
          : []
      }),
      mode,
      startX: event.clientX,
      startY: event.clientY,
      playbackHeadBeat: snapToPlaybackHead() && livePlaybackBeat >= 0 && livePlaybackBeat <= totalBeats()
        ? Math.round(livePlaybackBeat)
        : null,
    }
    noteDragMovedRef.current = false
    dragHistoryRecordedRef.current = false
    setSelectedNote(hit)
    setCanvasCursor("grabbing")
    event.currentTarget.setPointerCapture(event.pointerId)
    event.preventDefault()
  }

  const handleCanvasPointerMove = (event: PointerEvent & { currentTarget: HTMLCanvasElement }) => {
    const selectionDrag = selectionDragRef.current
    if (selectionDrag) {
      const root = canvasRef.current
      if (!root) return
      const rect = root.getBoundingClientRect()
      const nextRect = {
        ...selectionDrag,
        currentX: event.clientX - rect.left,
        currentY: event.clientY - rect.top,
      }
      selectionDragRef.current = nextRect
      setSelectionRect(nextRect)
      setCanvasCursor("crosshair")
      return
    }

    const drag = noteDragRef.current
    if (!drag) {
      setCanvasCursor(cursorForPoint(event.clientX, event.clientY))
      return
    }
    setCanvasCursor("grabbing")
    const root = canvasRef.current
    if (!root) return
    const rect = root.getBoundingClientRect()
    const scaleY = (pitchRange() * ROW_HEIGHT) / rect.height
    const beatDelta = Math.round((event.clientX - drag.startX) / pixelsPerBeat())
    const pitchDelta = -Math.round(((event.clientY - drag.startY) * scaleY) / ROW_HEIGHT)
    const selectedKeys = new Set(drag.notes.map(noteSelectionKey))
    const activeOrigin = drag.notes.find((origin) => noteSelectionKey(origin) === noteSelectionKey(drag.active))
    if (!activeOrigin) return
    const livePlaybackHeadBeat = snapToPlaybackHead() && audioRef.current
      ? Math.round((playbackPosition() - gap_s()) / secondsPerBeat())
      : null
    const playbackHeadBeat = drag.playbackHeadBeat ?? livePlaybackHeadBeat
    let previousEnd = 0
    outer: for (let phraseIdx = 0; phraseIdx < displayResult().phrases.length; phraseIdx += 1) {
      const phrase = displayResult().phrases[phraseIdx]
      for (let noteIdx = 0; noteIdx < phrase.notes.length; noteIdx += 1) {
        if (phraseIdx === drag.active.phraseIdx && noteIdx === drag.active.noteIdx) break outer
        const previous = phrase.notes[noteIdx]
        if (!selectedKeys.has(noteSelectionKey({ phraseIdx, noteIdx }))) {
          previousEnd = Math.max(previousEnd, previous.start_beat + previous.duration_beats)
        }
      }
    }
    const snapCandidate = (candidate: number, previousTarget: number): number => {
      const targets: number[] = []
      if (snapToPrevious() && previousTarget > 0 && Math.abs(candidate - previousTarget) <= 1.5) {
        targets.push(previousTarget)
      }
      if (playbackHeadBeat !== null && Math.abs(candidate - playbackHeadBeat) <= 1.5) {
        targets.push(playbackHeadBeat)
      }
      return targets.reduce((closest, target) => (
        Math.abs(target - candidate) < Math.abs(closest - candidate) ? target : closest
      ), targets[0] ?? candidate)
    }
    let moveBeatDelta = beatDelta
    if (drag.mode === "move") {
      const anchor = drag.notes.reduce((earliest, origin) => (
        origin.startBeat < earliest.startBeat ? origin : earliest
      ), drag.notes[0])
      let groupPreviousEnd = 0
      displayResult().phrases.forEach((phrase, phraseIdx) => {
        phrase.notes.forEach((note, noteIdx) => {
          if (selectedKeys.has(noteSelectionKey({ phraseIdx, noteIdx }))) return
          const end = note.start_beat + note.duration_beats
          if (end <= anchor.startBeat) groupPreviousEnd = Math.max(groupPreviousEnd, end)
        })
      })
      moveBeatDelta = snapCandidate(anchor.startBeat + beatDelta, groupPreviousEnd) - anchor.startBeat
      const earliestStart = Math.min(...drag.notes.map((origin) => origin.startBeat))
      moveBeatDelta = Math.max(moveBeatDelta, -earliestStart)
    }

    const newPhrases = structuredClone(displayResult().phrases)
    let changed = false
    for (const origin of drag.notes) {
      const note = newPhrases[origin.phraseIdx]?.notes[origin.noteIdx]
      if (!note) continue
      if (drag.mode === "move") {
        note.start_beat = Math.max(0, origin.startBeat + moveBeatDelta)
        note.pitch = Math.max(-24, Math.min(48, origin.pitch + pitchDelta))
        changed ||= note.start_beat !== origin.startBeat || note.pitch !== origin.pitch
      } else if (noteSelectionKey(origin) === noteSelectionKey(drag.active)) {
        if (drag.mode === "start") {
          const nextStart = Math.max(0, Math.min(origin.startBeat + origin.duration - 1, snapCandidate(origin.startBeat + beatDelta, previousEnd)))
          note.start_beat = nextStart
          note.duration_beats = origin.duration - (nextStart - origin.startBeat)
        } else {
          const nextEnd = Math.max(origin.startBeat + 1, snapCandidate(origin.startBeat + origin.duration + beatDelta, 0))
          note.duration_beats = nextEnd - origin.startBeat
        }
        note.pitch = Math.max(-24, Math.min(48, origin.pitch + pitchDelta))
        changed ||= note.start_beat !== origin.startBeat || note.duration_beats !== origin.duration || note.pitch !== origin.pitch
      }
    }
    if (!changed) return
    noteDragMovedRef.current = true
    if (!dragHistoryRecordedRef.current) {
      recordEdit(displayResult().phrases)
      dragHistoryRecordedRef.current = true
    }
    onResultChange({ ...result(), phrases: newPhrases })
  }

  const handleCanvasPointerUp = (event: PointerEvent & { currentTarget: HTMLCanvasElement }) => {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId)

    const selectionDrag = selectionDragRef.current
    if (selectionDrag) {
      const root = canvasRef.current
      const rect = root?.getBoundingClientRect()
      const finalRect = rect
        ? {
            ...selectionDrag,
            currentX: event.clientX - rect.left,
            currentY: event.clientY - rect.top,
          }
        : selectionDrag
      const moved = Math.abs(finalRect.currentX - finalRect.startX) >= 3 || Math.abs(finalRect.currentY - finalRect.startY) >= 3
      if (moved) {
        const inside = notesInSelection(finalRect)
        updateSelection(finalRect.additive ? [...selectedNotes(), ...inside] : inside, inside[inside.length - 1] ?? (finalRect.additive ? selectedNote() : null))
      } else if (!finalRect.additive) {
        updateSelection([])
      }
      selectionDragRef.current = null
      setSelectionRect(null)
    }

    noteDragRef.current = null
    dragHistoryRecordedRef.current = false
    setCanvasCursor(cursorForPoint(event.clientX, event.clientY))
    window.setTimeout(() => { noteDragMovedRef.current = false }, 0)
  }

  const updateEditorAnchor = (
    activeNote: NoteSelection | null,
    note: UltraStarNote | undefined,
    gapBeatsValue: number,
    pixelsPerBeatValue: number,
    maxPitchValue: number,
  ): void => {
    if (!activeNote || !canvasRef.current) {
      setEditorAnchor(null)
      return
    }
    const root = canvasRef.current
    const rect = root.getBoundingClientRect()
    if (!note) {
      setEditorAnchor(null)
      return
    }
    const x = (note.start_beat + gapBeatsValue) * pixelsPerBeatValue
    const y = (maxPitchValue - note.pitch) * ROW_HEIGHT
    const w = note.duration_beats * pixelsPerBeatValue
    const h = ROW_HEIGHT - 1
    setEditorAnchor({
      left: rect.left + x,
      top: rect.top + y,
      width: w,
      height: h,
    })
  }

  createPostRenderEffect(
    () => {
      const activeNote = selectedNote()
      const note = activeNote
        ? displayResult().phrases[activeNote.phraseIdx]?.notes[activeNote.noteIdx]
        : undefined
      return {
        activeNote,
        note,
        gapBeats: gapBeats(),
        pixelsPerBeat: pixelsPerBeat(),
        maxPitch: maxPitch(),
      }
    },
    ({ activeNote, note, gapBeats: gapBeatsValue, pixelsPerBeat: pixelsPerBeatValue, maxPitch: maxPitchValue }) => {
      updateEditorAnchor(activeNote, note, gapBeatsValue, pixelsPerBeatValue, maxPitchValue)
    },
  )

  createEffect(() => {
    if (!selectedNote()) return
    const container = containerRef.current
    const onScroll = (): void => {
      const activeNote = selectedNote()
      updateEditorAnchor(
        activeNote,
        activeNote ? displayResult().phrases[activeNote.phraseIdx]?.notes[activeNote.noteIdx] : undefined,
        gapBeats(),
        pixelsPerBeat(),
        maxPitch(),
      )
    }
    container?.addEventListener("scroll", onScroll)
    window.addEventListener("resize", onScroll)
    onCleanup(() => {
      container?.removeEventListener("scroll", onScroll)
      window.removeEventListener("resize", onScroll)
    })
  }, () => undefined)

  const updateNote = (field: keyof UltraStarNote, value: string | number) => {
    const activeNote = selectedNote()
    if (!activeNote) return
    const newPhrases = structuredClone(displayResult().phrases)
    const note = newPhrases[activeNote.phraseIdx].notes[activeNote.noteIdx]
    if (note[field] === value) return
    ;(note as Record<keyof UltraStarNote, string | number>)[field] = value
    recordEdit(displayResult().phrases)
    onResultChange({ ...result(), phrases: newPhrases })
  }

  const handleSave = async () => {
    if (!isDirty() || saving()) return
    if (props.confirmSaveMessage && !window.confirm(props.confirmSaveMessage)) return
    setSaving(true)
    setSaveError("")
    try {
      const updatedChart = props.saveNotes
        ? await props.saveNotes(displayResult().phrases)
        : (await updateNotes(jobId(), displayResult().phrases), undefined)
      if (updatedChart !== undefined) {
        onResultChange({ ...result(), ultrastar_txt: updatedChart })
      }
      lastSavedPhrasesRef.current = displayResult().phrases
      setSavedPhrasesTick((tick) => tick + 1)
    } catch (error) {
      setSaveError(error instanceof Error ? error.message : "Failed to save changes")
    } finally {
      setSaving(false)
    }
  }

  const togglePlayback = () => {
    const audio = audioRef.current
    if (!audio) return
    if (isPlaying()) {
      audio.pause()
      if (playheadRafRef.current !== null) {
        cancelAnimationFrame(playheadRafRef.current)
        playheadRafRef.current = null
      }
      playbackTimeRef.current = playbackPosition()
      setPlaybackTick((t) => t + 1)
      setIsPlaying(false)
    } else {
      if (audioTrack() === "instrumental_with_playback" || audioTrack() === "vocals_with_playback") {
        pianoResumeRef.current?.resume()
      }
      audio.play().then(() => setIsPlaying(true))
    }
  }

  const sel = createMemo(() => {
    const activeNote = selectedNote()
    return selectedNotes().length === 1 && activeNote
      ? displayResult().phrases[activeNote.phraseIdx]?.notes[activeNote.noteIdx]
      : null
  })

  const selectedPitchEvidence = createMemo(() => {
    const activeNote = selectedNote()
    if (!activeNote) return undefined
    const note = displayResult().phrases[activeNote.phraseIdx]?.notes[activeNote.noteIdx]
    if (!note) return undefined
    const pitchWords = variant()?.pitch_words ?? displayResult().pitch_words ?? displayResult().words
    const start = chartBeatToTime(note.start_beat)
    const end = chartBeatToTime(note.start_beat + note.duration_beats)
    const exactText = note.text.trim().toLowerCase()
    return pitchWords.find((word) =>
      word.text.trim().toLowerCase() === exactText
      && word.start < end
      && word.end > start
    ) ?? pitchWords.find((word) =>
      word.start < end
      && word.end > start
    )
  })

  const handleSeekKeyDown = (e: KeyboardEvent & { currentTarget: HTMLDivElement }) => {
    if (e.altKey || e.metaKey || e.ctrlKey) return
    if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
      e.preventDefault()
      if (e.shiftKey && (selectedNote() || selectedNotes().length > 0)) {
        nudgeSelectedNotes(e.key)
      } else {
        seekByBeat(e.key === "ArrowLeft" ? -1 : 1)
      }
    } else if (e.key === "ArrowUp" || e.key === "ArrowDown") {
      e.preventDefault()
      seekToSegmentBoundary(e.key === "ArrowUp" ? -1 : 1)
    }
  }

  const handleAudioTrackChange = (track: AudioTrack) => {
    if (track === audioTrack()) return
    const audio = audioRef.current
    trackChangeTimeRef.current = audio
      ? playbackPosition()
      : playbackTimeRef.current
    trackChangeWasPlayingRef.current = Boolean(audio && !audio.paused && isPlaying())
    if (trackChangeTimeRef.current !== null) {
      playbackTimeRef.current = trackChangeTimeRef.current
      pendingSeekTimeRef.current = null
    }
    setAudioTrack(track)
    setPlaybackTick((t) => t + 1)
  }

  const guardedExport = () => {
    if (alignmentDirty() || result().export_ready === false) {
      setExportWarning(true)
      return
    }
    setExportWarning(false)
    onExport()
  }

  const handleModeChange = async (mode: "sofa" | "legacy" | "canonical") => {
    if (mode === (result().current_alignment_mode ?? "canonical")) return
    setSwitchingMode(mode)
    try {
      const next = await switchAlignmentMode(jobId(), mode)
      onResultChange(next)
      setAlignmentDirty(false)
    } catch (error) {
      console.error("Failed to switch alignment mode", error)
    } finally {
      setSwitchingMode(null)
    }
  }

  return (
    <div
      class={`${isExpanded() ? "fixed inset-0 z-[80] h-dvh w-screen overflow-y-auto bg-zinc-950 p-4" : "max-w-7xl mx-auto"} space-y-6`}
    >
      <EditorHeader
        result={result()}
        allNotesCount={allNotes().length}
        phrasesCount={displayResult().phrases.length}
        saving={saving()}
        isDirty={isDirty()}
        onSave={handleSave}
        onExport={guardedExport}
        onOpenLyricsAlignment={() => setLyricsAlignmentOpen(true)}
        onModeChange={handleModeChange}
        switchingMode={switchingMode()}
        showLyricsAlignment={showAdvancedActions}
        showExport={showAdvancedActions}
      />

      {saveError() && (
        <p class="text-sm text-red-300 bg-red-500/10 border border-red-500/30 rounded-lg px-3 py-2" role="alert">
          {saveError()}
        </p>
      )}
      {exportWarning() && (
        <p class="text-sm text-amber-300 bg-amber-500/10 border border-amber-500/30 rounded-lg px-3 py-2" role="alert">
          {alignmentDirty()
            ? "Save boundary or lyric revision changes before exporting."
            : "Re-align the selected lyric revision before exporting."}
        </p>
      )}
      {showAdvancedActions && (
        <WordAlignmentEditor
          jobId={jobId()}
          result={result()}
          onResultChange={onResultChange}
          onDirtyChange={setAlignmentDirty}
          open={lyricsAlignmentOpen()}
          onClose={() => setLyricsAlignmentOpen(false)}
        />
      )}

      {(props.audioSourceUrl !== null || props.audioSourceUrls !== undefined) && (
      <AudioControls
        jobId={jobId()}
        audioTrack={audioTrack()}
        isPlaying={isPlaying()}
        onTogglePlayback={togglePlayback}
        onAudioTrackChange={handleAudioTrackChange}
        audioSourceUrl={props.audioSourceUrl ?? undefined}
        audioSourceUrls={props.audioSourceUrls}
        availableTracks={props.availableAudioTracks}
        showTrackSelection={props.audioSourceUrl === undefined || props.audioSourceUrls !== undefined}
        onTestSound={() => {
          const ctx = pianoResumeRef.current?.getContext()
          if (ctx) {
            ctx.resume().then(() => {
              const now = ctx.currentTime
              ;[60, 64, 67, 64].forEach((midi, i) => {
                playPianoNote(ctx, ctx.destination, midi, 0.28, 0.7, now + i * 0.28)
              })
            }).catch(() => {})
          } else {
            playTestPiano()
          }
        }}
        onEnded={() => {
          if (playheadRafRef.current !== null) {
            cancelAnimationFrame(playheadRafRef.current)
            playheadRafRef.current = null
          }
          const t = audioRef.current?.currentTime ?? 0
          pendingSeekTimeRef.current = null
          playbackTimeRef.current = t
          setPlaybackTick((tick) => tick + 1)
          setIsPlaying(false)
        }}
        onTimeUpdate={() => {
          const audio = audioRef.current
          if (
            !isPlaying()
            && !isDraggingTimeline()
            && pendingSeekTimeRef.current === null
            && audio
            && !audio.seeking
          ) {
            const t = audio.currentTime
            playbackTimeRef.current = t
            setPlaybackTick((tick) => tick + 1)
          }
        }}
        onSeeked={() => {
          const audio = audioRef.current
          if (!audio || !Number.isFinite(audio.currentTime)) return
          const pendingTime = pendingSeekTimeRef.current
          if (pendingTime !== null && audio.seeking) return
          if (pendingTime !== null && Math.abs(audio.currentTime - pendingTime) > 0.05) {
            audio.currentTime = pendingTime
            return
          }
          pendingSeekTimeRef.current = null
          playbackTimeRef.current = audio.currentTime
          setPlaybackTick((tick) => tick + 1)
        }}
        onLoadedMetadata={() => setAudioDuration(audioRef.current?.duration ?? 0)}
        audioRef={audioRef}
        preserveTime={trackChangeTimeRef.current}
        resumeAfterLoad={trackChangeWasPlayingRef.current}
      />
      )}

      {/* Timeline + piano roll in one horizontal scroll container */}
      <div class="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-zinc-700/50 bg-zinc-900/40 px-3 py-2 text-xs text-zinc-400">
        <div class="flex flex-wrap items-center gap-x-3 gap-y-1">
          <span>Unified pitch and lyric-note editor</span>
          <span class="text-zinc-500" aria-label="Timeline zoom level">
            Zoom {pixelsPerBeat().toFixed(1)} px/beat · Option/Ctrl+scroll
          </span>
          <span class="text-zinc-500" aria-label="Playback speed">
            Speed {playbackRate().toFixed(1)}× · [ / ]
          </span>
        </div>
        <div class="flex flex-wrap items-center gap-3">
          <label class="inline-flex items-center gap-2 cursor-pointer select-none text-zinc-300">
            <input
              type="checkbox"
              checked={snapToPrevious()}
              onChange={(event) => setSnapToPrevious(event.target.checked)}
              class="accent-violet-500"
              aria-label="Snap timing to previous note"
            />
            Snap timing to previous note
          </label>
          <label class="inline-flex items-center gap-2 cursor-pointer select-none text-zinc-300">
            <input
              type="checkbox"
              checked={snapToPlaybackHead()}
              onChange={(event) => setSnapToPlaybackHead(event.target.checked)}
              class="accent-violet-500"
              aria-label="Snap timing to playback head"
            />
            Snap timing to playback head
          </label>
        </div>
        <button
          type="button"
          onClick={() => setIsExpanded((expanded) => !expanded)}
          class="rounded-md border border-zinc-700 px-3 py-1.5 text-zinc-300 hover:bg-zinc-800 hover:text-white"
          aria-label={isExpanded() ? "Restore editor size" : "Expand editor"}
        >
          {isExpanded() ? "Restore size" : "Expand editor"}
        </button>
      </div>
      <div
        class="flex max-w-full overflow-hidden rounded-xl border border-zinc-700/50"
        style={{ "max-height": `${pitchRange() * ROW_HEIGHT + 48}px` }}
      >
        <aside
          class="z-10 w-14 shrink-0 border-r border-zinc-700 bg-zinc-900"
          aria-label="Pitch labels"
        >
          <div class="h-12 border-b border-zinc-700 bg-zinc-800/80" aria-hidden="true" />
          <div style={{ height: `${pitchRange() * ROW_HEIGHT}px` }}>
            {Array.from({ length: pitchRange() }, (_, offset) => {
              const pitch = maxPitch() - offset
              const midi = pitch + 60
              const noteNames = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"]
              const label = `${noteNames[((midi % 12) + 12) % 12]}${Math.floor(midi / 12) - 1}`
              return (
                <div
                  class={`flex h-[28px] items-center justify-end border-b px-1.5 font-mono text-[9px] leading-none ${
                    midi % 12 === 0 ? "border-zinc-500 bg-zinc-800 text-zinc-100" : "border-zinc-800 text-zinc-500"
                  }`}
                >
                  {label}
                </div>
              )
            })}
          </div>
        </aside>
        <div
          ref={(container) => {
            containerRef.current = container
            setTimelineContainer(container)
          }}
          class="min-w-0 flex-1 overflow-x-auto overflow-y-hidden"
          style={{ "max-height": `${pitchRange() * ROW_HEIGHT + 48}px` }}
        >
        <div style={{ width: `${totalBeats() * pixelsPerBeat()}px` }}>
          <TimelineScrubber
            totalBeats={totalBeats()}
            totalDuration={totalDuration()}
            currentBeat={currentBeat()}
            displayTime={displayTime()}
            beatToTime={audioBeatToTime}
            pixelsPerBeat={pixelsPerBeat()}
            timelineRef={timelineRef}
            onTimelineClick={handleTimelineClick}
            onMouseDown={handleTimelineMouseDown}
            onMouseUp={handleTimelineMouseUp}
            onSeekKeyDown={handleSeekKeyDown}
          />
          <div
            ref={(root) => {
              canvasRef.current = root
              setPianoRollElement(root)
            }}
            class="relative overflow-y-auto overflow-x-hidden bg-zinc-900"
            style={{ "max-height": `${pitchRange() * ROW_HEIGHT + 20}px` }}
            aria-label="Piano roll"
          >
            <div class="flex" style={{ width: `${desiredCanvasWidth()}px`, height: `${pitchRange() * ROW_HEIGHT}px` }}>
              {Array.from({ length: canvasTileCount() }, (_, tileIndex) => (
                <canvas
                  onPointerDown={handleCanvasPointerDown}
                  onPointerMove={handleCanvasPointerMove}
                  onPointerLeave={() => {
                    if (!noteDragRef.current && !selectionDragRef.current) setCanvasCursor("default")
                  }}
                  onPointerUp={handleCanvasPointerUp}
                  onPointerCancel={handleCanvasPointerUp}
                  onContextMenu={(e) => e.preventDefault()}
                  class="block shrink-0 select-none"
                  style={{
                    width: `${Math.min(CANVAS_TILE_WIDTH, desiredCanvasWidth() - tileIndex * CANVAS_TILE_WIDTH)}px`,
                    height: `${pitchRange() * ROW_HEIGHT}px`,
                    cursor: canvasCursor(),
                  }}
                  aria-label={tileIndex === 0 ? "Note grid: click a note to select and edit" : undefined}
                />
              ))}
            </div>
            {displayTime() >= 0 && currentBeat() >= 0 && currentBeat() <= totalBeats() && (
              <div
                class="pointer-events-none absolute top-0 z-10 w-0.5 bg-pink-500 transition-none"
                style={{
                  left: `${currentBeat() * pixelsPerBeat()}px`,
                  height: `${pitchRange() * ROW_HEIGHT}px`,
                }}
                aria-hidden="true"
              />
            )}
            {selectionRect() && (
              <div
                class="pointer-events-none absolute z-20 border border-sky-300/90 bg-sky-400/15"
                style={{
                  left: `${Math.min(selectionRect()!.startX, selectionRect()!.currentX)}px`,
                  top: `${Math.min(selectionRect()!.startY, selectionRect()!.currentY)}px`,
                  width: `${Math.abs(selectionRect()!.currentX - selectionRect()!.startX)}px`,
                  height: `${Math.abs(selectionRect()!.currentY - selectionRect()!.startY)}px`,
                }}
                aria-hidden="true"
              />
            )}
            {!isZooming() && (
              <div
                class="pointer-events-none absolute left-0 top-0"
                style={{ width: `${desiredCanvasWidth()}px`, height: `${pitchRange() * ROW_HEIGHT}px` }}
                aria-hidden="true"
              >
                {displayResult().phrases.flatMap((phrase) => phrase.notes.map((note) => {
                  const width = note.duration_beats * pixelsPerBeat()
                  if (width <= 20 || !note.text.trim()) return null
                  return (
                    <span
                      class="absolute overflow-hidden truncate px-[3px] text-[10px] leading-[13px] text-white"
                      style={{
                        left: `${(note.start_beat + gapBeats()) * pixelsPerBeat()}px`,
                        top: `${(maxPitch() - note.pitch) * ROW_HEIGHT}px`,
                        width: `${Math.max(width, 3)}px`,
                        height: `${ROW_HEIGHT - 1}px`,
                      }}
                    >
                      {note.text.trim()}
                    </span>
                  )
                }))}
              </div>
            )}
          </div>
        </div>
        </div>
      </div>

      {sel() && selectedNote() && editorAnchor() !== null && (
        <NoteEditorPopover
          note={sel()!}
          phraseIdx={selectedNote()!.phraseIdx}
          noteIdx={selectedNote()!.noteIdx}
          anchor={editorAnchor()!}
          pitchEvidence={selectedPitchEvidence()}
          onClose={() => setSelectedNote(null)}
          onUpdate={updateNote}
        />
      )}

      {!selectedNote() && <EditorHint />}

      <LyricsPreviewSection phrases={displayResult().phrases} />

      <RawUltrastarSection ultrastarTxt={displayResult().ultrastar_txt} />
    </div>
  )
}
