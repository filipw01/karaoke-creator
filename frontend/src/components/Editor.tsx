import { createEffect, createMemo, createSignal, onCleanup, untrack } from "solid-js"
import type { JobResult, Phrase, UltraStarNote } from "../types"
import { switchAlignmentMode, updateNotes } from "../api"
import { isPianoTrack, playPianoNote, playTestPiano, useInstrumentalPiano, type PianoResumeRef } from "../useInstrumentalPiano"
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
const PIANO_ROLL_OVERSCAN = 384
const PIANO_ROLL_GRID = [
  { step: 1, skipEvery: 4, color: "#30303a", width: 0.75 },
  { step: 4, skipEvery: 16, color: "#484852", width: 1 },
  { step: 16, skipEvery: 0, color: "#71717a", width: 1.5 },
]
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
  notes: NoteDragOrigin[]
  mode: NoteDragMode
  startX: number
  startY: number
  anchorBeat: number
  previousEnd: number
  playbackHeadBeat: number | null
  historyRecorded: boolean
}
type PointerPosition = { clientX: number; clientY: number }
type SelectionRect = {
  startX: number
  startY: number
  currentX: number
  currentY: number
  additive: boolean
}
type PianoRollRenderState = {
  canvas: HTMLCanvasElement | null
  viewportWidth: number
  canvasLeft: number
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

const editNotes = <Selection extends NoteSelection>(
  phrases: Phrase[],
  selection: Selection[],
  edit: (note: UltraStarNote, selection: Selection) => UltraStarNote,
): Phrase[] => {
  let next: Phrase[] | undefined
  for (const selected of selection) {
    const { phraseIdx, noteIdx } = selected
    const phrase = phrases[phraseIdx]
    const note = phrase?.notes[noteIdx]
    if (!note) continue
    const edited = edit(note, selected)
    if (edited === note) continue
    next ??= phrases.slice()
    if (next[phraseIdx] === phrase) next[phraseIdx] = { ...phrase, notes: phrase.notes.slice() }
    next[phraseIdx].notes[noteIdx] = edited
  }
  return next ?? phrases
}

const MIN_NOTE_WIDTH = 3
const phraseStats = new WeakMap<Phrase, { minPitch: number; maxPitch: number; start: number; end: number }>()
const getPhraseStats = (phrase: Phrase) => {
  let stats = phraseStats.get(phrase)
  if (stats) return stats
  stats = { minPitch: Infinity, maxPitch: -Infinity, start: Infinity, end: -Infinity }
  for (const note of phrase.notes) {
    stats.minPitch = Math.min(stats.minPitch, note.pitch)
    stats.maxPitch = Math.max(stats.maxPitch, note.pitch)
    stats.start = Math.min(stats.start, note.start_beat)
    stats.end = Math.max(stats.end, note.start_beat + note.duration_beats)
  }
  phraseStats.set(phrase, stats)
  return stats
}

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
  const [playRequested, setPlayRequested] = createSignal(false)
  const [saving, setSaving] = createSignal(false)
  const [saveError, setSaveError] = createSignal("")
  const [switchingMode, setSwitchingMode] = createSignal<string | null>(null)
  const [alignmentDirty, setAlignmentDirty] = createSignal(false)
  const [exportWarning, setExportWarning] = createSignal(false)
  const [loadedAudioDuration, setLoadedAudioDuration] = createSignal(0)
  const audioDuration = createMemo(() => loadedAudioDuration() || result().audio_duration || 0)
  const [isDraggingTimeline, setIsDraggingTimeline] = createSignal(false)
  const [dragTime, setDragTime] = createSignal<number | null>(null)
  const [snapToPrevious, setSnapToPrevious] = createSignal(true)
  const [snapToPlaybackHead, setSnapToPlaybackHead] = createSignal(false)
  const [pixelsPerBeat, setPixelsPerBeat] = createSignal(DEFAULT_PIXELS_PER_BEAT)
  const [playbackRate, setPlaybackRate] = createSignal(1)
  const [mediaTime, setMediaTime] = createSignal(0)
  const [isExpanded, setIsExpanded] = createSignal(false)
  const pendingSeekTimeRef: { current: number | null } = { current: null }
  const timelineRef: { current: HTMLDivElement | null } = { current: null }
  const suppressTimelineClickRef = { current: false }
  let lastWheelZoom = -Infinity
  let wheelZoomFrame: number | null = null
  let wheelZoomScale: number | null = null
  let wheelZoomAnchor: { beat: number; offsetPx: number } | null = null
  const suspendPlayheadFollowRef = { current: false }
  const playheadBeatAtZoomRef = { current: 0 }
  const undoHistoryRef: { current: Array<JobResult["phrases"]> } = { current: [] }
  const redoHistoryRef: { current: Array<JobResult["phrases"]> } = { current: [] }
  const rafRef: { current: number | null } = { current: null }
  const audioRef: { current: HTMLAudioElement | null } = { current: null }
  let playRequestId = 0
  let audioJobId: string | null = null
  const pianoResumeRef: { current: PianoResumeRef | null } = { current: null }
  const canvasRef: { current: HTMLDivElement | null } = { current: null }
  const [pianoRollCanvas, setPianoRollCanvas] = createSignal<HTMLCanvasElement | null>(null)
  const [viewportWidth, setViewportWidth] = createSignal(0)
  const [canvasLeft, setCanvasLeft] = createSignal(0)
  let viewportScrollLeft = 0
  const noteDragRef: { current: NoteDragState | null } = { current: null }
  let pendingNoteDrag: PointerPosition | null = null
  let noteDragFrame: number | null = null
  let disposed = false
  const selectionDragRef: { current: SelectionRect | null } = { current: null }
  const containerRef: { current: HTMLDivElement | null } = { current: null }
  const [timelineContainer, setTimelineContainer] = createSignal<HTMLDivElement | null>(null)
  const [lyricsAlignmentOpen, setLyricsAlignmentOpen] = createSignal(false)
  let editorRoot: HTMLDivElement | undefined
  const [pianoRollOrigin, setPianoRollOrigin] = createSignal<{ left: number; top: number } | null>(null, {
    equals: (previous, next) => previous?.left === next?.left && previous?.top === next?.top,
  })
  const phrases = createMemo(() => result().phrases)
  const [savedPhrases, setSavedPhrases] = createSignal(untrack(phrases))
  const isDirty = createMemo(() => phrases() !== savedPhrases())

  const chartStats = createMemo(() => {
    let count = 0
    let minPitch = Infinity
    let maxPitch = -Infinity
    let end = -Infinity
    for (const phrase of phrases()) {
      const stats = getPhraseStats(phrase)
      count += phrase.notes.length
      minPitch = Math.min(minPitch, stats.minPitch)
      maxPitch = Math.max(maxPitch, stats.maxPitch)
      end = Math.max(end, stats.end)
    }
    return count > 0 ? { count, minPitch, maxPitch, end: end + 20 } : { count: 0, minPitch: 0, maxPitch: 0, end: 100 }
  })
  const minPitch = createMemo(() => chartStats().minPitch - 1)
  const maxPitch = createMemo(() => chartStats().maxPitch + 1)
  const pitchRange = createMemo(() => maxPitch() - minPitch() + 1)

  const variant = createMemo(() => result().alignment_results?.[result().current_alignment_mode ?? "canonical"])
  const gap_ms_effective = createMemo(() => variant()?.gap_ms ?? result().metadata.gap_ms)
  const gap_s = createMemo(() => gap_ms_effective() / 1000)
  const ustar_bpm = createMemo(() => result().metadata.bpm * 4)
  // UltraStar note beats begin after GAP; timeline beats begin at audio 0:00.
  const secondsPerBeat = createMemo(() => 60 / (ustar_bpm() * 4))
  const gapBeats = createMemo(() => gap_s() / secondsPerBeat())
  const audioTimeToBeat = (time: number): number => time / secondsPerBeat()
  const audioBeatToTime = (beat: number): number => beat * secondsPerBeat()
  const chartBeatToTime = (beat: number): number => gap_s() + beat * secondsPerBeat()

  // Keep the seekable strip as long as the audio, including trailing silence.
  // Previously it stopped at the last note, so clicking near the end of a
  // song could seek to an earlier position than the cursor suggested.
  const chartEndBeat = createMemo(() => chartStats().end)
  const audioEndBeat = createMemo(() => audioDuration() > 0 ? Math.max(0, audioTimeToBeat(audioDuration())) : 0)
  const totalBeats = createMemo(() => Math.max(
    chartEndBeat() + gapBeats(),
    Math.ceil(audioEndBeat()),
  ))
  const desiredCanvasWidth = createMemo(() => Math.max(1, Math.ceil(totalBeats() * pixelsPerBeat())))

  useInstrumentalPiano({
    audioRef,
    audioTrack,
    isPlaying,
    phrases,
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
      : mediaTime()
  }

  const syncPlaybackPosition = () => setMediaTime(playbackPosition())
  const displayTime = createMemo(() => dragTime() ?? mediaTime())
  const currentBeat = createMemo(() => audioTimeToBeat(displayTime()))

  const playheadX = createMemo(() => currentBeat() * pixelsPerBeat())
  const playheadVisible = createMemo(() => currentBeat() >= 0 && currentBeat() <= totalBeats())
  const updateViewport = (container: HTMLDivElement) => {
    viewportScrollLeft = container.scrollLeft
    updateCanvasLeft(viewportScrollLeft)
  }
  const updateCanvasLeft = (scrollLeft: number) => {
    const page = Math.floor(scrollLeft / PIANO_ROLL_OVERSCAN)
    setCanvasLeft(Math.max(0, (page - 1) * PIANO_ROLL_OVERSCAN))
  }
  const scrollLeftForZoom = (anchor: { beat: number; offsetPx: number }, scale: number) => {
    const maxScroll = Math.max(0, Math.ceil(totalBeats() * scale) - viewportWidth())
    return Math.min(maxScroll, Math.max(0, anchor.beat * scale - anchor.offsetPx))
  }

  // Follow the playhead one viewport at a time. Scroll and resize handlers
  // cache the viewport geometry so the playback loop can avoid layout reads.
  createEffect(
    () => ({
      container: timelineContainer(),
      beat: currentBeat(),
      x: playheadX(),
      width: viewportWidth(),
      contentWidth: desiredCanvasWidth(),
      dragging: isDraggingTimeline(),
    }),
    ({ container, beat, x, width, contentWidth, dragging }) => {
      if (
        !container || !width || dragging || wheelZoomAnchor || performance.now() - lastWheelZoom < 120
        || pendingSeekTimeRef.current !== null || !Number.isFinite(x)
      ) return
      if (suspendPlayheadFollowRef.current) {
        if (Math.abs(beat - playheadBeatAtZoomRef.current) < 0.01) return
        suspendPlayheadFollowRef.current = false
      }
      if (x >= viewportScrollLeft && x + 2 <= viewportScrollLeft + width) return
      const nextScrollLeft = Math.min(Math.max(0, contentWidth - width), Math.max(0, x))
      if (Math.abs(nextScrollLeft - viewportScrollLeft) <= 1) return
      container.scrollLeft = nextScrollLeft
      updateViewport(container)
    },
  )

  const handleTimelineWheel = (event: WheelEvent) => {
    if (!event.altKey && !event.ctrlKey) return
    const container = containerRef.current
    if (!container) return
    const delta = Math.abs(event.deltaY) >= Math.abs(event.deltaX) ? event.deltaY : event.deltaX
    if (delta === 0) return
    event.preventDefault()
    event.stopPropagation()
    const scale = pixelsPerBeat()
    const nextScale = Math.min(
      MAX_PIXELS_PER_BEAT,
      Math.max(MIN_PIXELS_PER_BEAT, (wheelZoomScale ?? scale) * Math.exp(-delta * ZOOM_WHEEL_SENSITIVITY)),
    )
    if (nextScale === (wheelZoomScale ?? scale)) return
    const offsetPx = event.clientX - container.getBoundingClientRect().left
    lastWheelZoom = performance.now()
    wheelZoomAnchor = { beat: (container.scrollLeft + offsetPx) / scale, offsetPx }
    wheelZoomScale = nextScale
    if (wheelZoomFrame !== null) return
    wheelZoomFrame = requestAnimationFrame(() => {
      wheelZoomFrame = null
      const next = wheelZoomScale
      wheelZoomScale = null
      const anchor = wheelZoomAnchor
      if (!anchor || next === null || next === pixelsPerBeat()) {
        wheelZoomAnchor = null
        return
      }
      updateCanvasLeft(Math.round(scrollLeftForZoom(anchor, next)))
      setPixelsPerBeat(next)
    })
  }

  createEffect(
    timelineContainer,
    (container) => {
      if (!container) return
      const onScroll = () => updateViewport(container)
      const observer = new ResizeObserver(() => {
        setViewportWidth(container.clientWidth)
        onScroll()
      })
      observer.observe(container)
      setViewportWidth(container.clientWidth)
      onScroll()
      container.addEventListener("scroll", onScroll, { passive: true })
      container.addEventListener("wheel", handleTimelineWheel, { passive: false, capture: true })
      return () => {
        observer.disconnect()
        container.removeEventListener("scroll", onScroll)
        container.removeEventListener("wheel", handleTimelineWheel, true)
        if (wheelZoomFrame !== null) cancelAnimationFrame(wheelZoomFrame)
      }
    },
  )

  createEffect(
    () => [pixelsPerBeat(), timelineContainer()] as const,
    ([scale, container]) => {
      const anchor = wheelZoomAnchor
      if (!container || !anchor) return
      playheadBeatAtZoomRef.current = currentBeat()
      suspendPlayheadFollowRef.current = true
      container.scrollLeft = scrollLeftForZoom(anchor, scale)
      updateViewport(container)
      wheelZoomAnchor = null
    },
  )

  const recordEdit = (previous: JobResult["phrases"]) => {
    undoHistoryRef.current.push(previous)
    if (undoHistoryRef.current.length > MAX_EDIT_HISTORY) undoHistoryRef.current.shift()
    redoHistoryRef.current = []
  }

  const applyEditHistory = (direction: "undo" | "redo"): void => {
    const source = direction === "undo" ? undoHistoryRef.current : redoHistoryRef.current
    const destination = direction === "undo" ? redoHistoryRef.current : undoHistoryRef.current
    const previous = source.pop()
    if (!previous) return
    destination.push(phrases())
    if (destination.length > MAX_EDIT_HISTORY) destination.shift()
    onResultChange({ ...result(), phrases: previous })
  }

  const cancelNoteDragFrame = () => {
    if (noteDragFrame !== null) cancelAnimationFrame(noteDragFrame)
    noteDragFrame = null
    pendingNoteDrag = null
  }

  onCleanup(() => {
    disposed = true
    cancelNoteDragFrame()
    audioRef.current?.pause()
  })

  createEffect(jobId, (current, previous) => {
    if (previous === undefined || previous === current) return
    cancelNoteDragFrame()
    noteDragRef.current = null
    undoHistoryRef.current = []
    redoHistoryRef.current = []
    setSavedPhrases(phrases())
    setSaving(false)
    setSaveError("")
  })

  const totalDuration = createMemo(() => (audioDuration() > 0
    ? audioDuration()
    : result().words.length > 0
      ? Math.max(...result().words.map((w) => w.end)) + 1
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
    const boundaries = phrases().flatMap((phrase) =>
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
    if (audio) {
      audio.defaultPlaybackRate = nextRate
      audio.playbackRate = nextRate
    }
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
    const previous = phrases()
    const next = editNotes(previous, selection, (note) => {
      let { start_beat, pitch } = note
      if (key === "ArrowLeft") start_beat = Math.max(0, start_beat - 1)
      else if (key === "ArrowRight") start_beat += 1
      else if (key === "ArrowUp") pitch = Math.min(48, pitch + 1)
      else if (key === "ArrowDown") pitch = Math.max(-24, pitch - 1)
      return start_beat === note.start_beat && pitch === note.pitch ? note : { ...note, start_beat, pitch }
    })
    if (next === previous) return false
    recordEdit(previous)
    onResultChange({ ...result(), phrases: next })
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
    if (audio.readyState < HTMLMediaElement.HAVE_METADATA) {
      pendingSeekTimeRef.current = targetTime
    } else if (!audio.seeking && Math.abs(audio.currentTime - targetTime) < 0.001) {
      pendingSeekTimeRef.current = null
    } else {
      pendingSeekTimeRef.current = targetTime
      audio.currentTime = targetTime
    }
    syncPlaybackPosition()
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

  createEffect(isDraggingTimeline, (dragging) => {
    if (!dragging) return
    const onMouseUp = () => finishTimelineSeek(false)
    const onMouseMove = (event: MouseEvent) => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current)
      rafRef.current = requestAnimationFrame(() => {
        if (isDraggingTimeline()) updateTimeFromMouse(event.clientX, true)
        rafRef.current = null
      })
    }
    window.addEventListener("mouseup", onMouseUp)
    window.addEventListener("mousemove", onMouseMove, { passive: true })
    return () => {
      window.removeEventListener("mouseup", onMouseUp)
      window.removeEventListener("mousemove", onMouseMove)
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current)
      rafRef.current = null
    }
  })

  createEffect(
    () => isPlaying() && !isDraggingTimeline(),
    (running) => {
      if (!running) return
      const audio = audioRef.current
      if (!audio) return
      const tick = () => {
        if (!audio.paused && !audio.seeking && pendingSeekTimeRef.current === null) {
          setMediaTime(audio.currentTime)
        }
        frame = requestAnimationFrame(tick)
      }
      let frame = requestAnimationFrame(tick)
      return () => cancelAnimationFrame(frame)
    },
  )

  // Keyboard: Space = play/pause, Escape = deselect notes
  createEffect(() => true, () => {
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
          if (!e.repeat) togglePlayback()
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
    return () => window.removeEventListener("keydown", onKeyDown)
  })

  // Draw only the viewport and a small margin. The same geometry is used
  // during a zoom gesture and after it ends.
  const drawPianoRoll = ({
    canvas,
    viewportWidth,
    canvasLeft,
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
    if (!canvas || !viewportWidth) return
    const width = Math.max(1, Math.min(viewportWidth + 2 * PIANO_ROLL_OVERSCAN, desiredCanvasWidth - canvasLeft))
    const height = pitchRange * ROW_HEIGHT
    const ctx = canvas.getContext("2d")
    if (!ctx) return
    canvas.style.left = `${canvasLeft}px`
    if (canvas.width !== width) canvas.width = width
    if (canvas.height !== height) canvas.height = height
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

    const firstBeat = Math.max(0, Math.ceil(canvasLeft / pixelsPerBeat))
    const lastBeat = Math.min(totalBeats, Math.ceil((canvasLeft + width) / pixelsPerBeat))
    for (const { step, skipEvery, color, width } of PIANO_ROLL_GRID) {
      ctx.strokeStyle = color
      ctx.lineWidth = width
      ctx.beginPath()
      for (let beat = Math.ceil(firstBeat / step) * step; beat <= lastBeat; beat += step) {
        if (skipEvery && beat % skipEvery === 0) continue
        const x = beat * pixelsPerBeat - canvasLeft
        ctx.moveTo(x, 0)
        ctx.lineTo(x, height)
      }
      ctx.stroke()
    }

    const colors: Record<string, string> = { ":": "#7c3aed", "*": "#d97706", "F": "#059669" }
    const selection = new Set(selectedNotes.map(noteSelectionKey))
    const labels: { text: string; x: number; y: number; width: number }[] = []
    phrases.forEach((phrase, phraseIdx) => {
      const bounds = getPhraseStats(phrase)
      const startX = (bounds.start + gapBeats) * pixelsPerBeat - canvasLeft
      const endX = (bounds.end + gapBeats) * pixelsPerBeat - canvasLeft
      if (startX <= width && endX + MIN_NOTE_WIDTH >= 0) {
        phrase.notes.forEach((note, noteIdx) => {
          const x = (note.start_beat + gapBeats) * pixelsPerBeat - canvasLeft
          const noteWidth = Math.max(note.duration_beats * pixelsPerBeat, MIN_NOTE_WIDTH)
          if (x + noteWidth < 0 || x > width) return
          const y = (maxPitch - note.pitch) * ROW_HEIGHT
          const isSelected = selection.has(noteSelectionKey({ phraseIdx, noteIdx }))
          ctx.fillStyle = isSelected
            ? note.note_type === "*" ? "#fbbf24" : note.note_type === "F" ? "#6ee7b7" : "#a78bfa"
            : colors[note.note_type] ?? colors[":"]
          ctx.beginPath()
          ctx.roundRect(x, y, noteWidth, ROW_HEIGHT - 1, 2)
          ctx.fill()
          ctx.strokeStyle = isSelected ? "#e4e4e7" : "#c4b5fd"
          ctx.lineWidth = 0.75
          ctx.stroke()
          if (noteWidth > 20 && note.text.trim()) labels.push({ text: note.text.trim(), x, y, width: noteWidth })
        })
      }

      if (phrase.end_beat > 0) {
        const x = (phrase.end_beat + gapBeats) * pixelsPerBeat - canvasLeft
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

    ctx.font = "10px system-ui"
    ctx.fillStyle = "#fff"
    for (const { text, x, y, width } of labels) {
      const available = width - 6
      let label = text
      if (ctx.measureText(label).width > available) {
        let end = text.length
        do {
          label = `${text.slice(0, --end)}…`
        } while (end > 0 && ctx.measureText(label).width > available)
      }
      ctx.fillText(label, x + 3, y + 10)
    }
  }

  createEffect(
    (): PianoRollRenderState => ({
      canvas: pianoRollCanvas(),
      viewportWidth: viewportWidth(),
      canvasLeft: canvasLeft(),
      pitchRange: pitchRange(),
      pixelsPerBeat: pixelsPerBeat(),
      desiredCanvasWidth: desiredCanvasWidth(),
      minPitch: minPitch(),
      maxPitch: maxPitch(),
      totalBeats: totalBeats(),
      phrases: phrases(),
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

    for (let pIdx = 0; pIdx < phrases().length; pIdx++) {
      const phrase = phrases()[pIdx]
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
    const note = phrases()[hit.phraseIdx]?.notes[hit.noteIdx]
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
    phrases().forEach((phrase, phraseIdx) => {
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
    const note = phrases()[hit.phraseIdx]?.notes[hit.noteIdx]
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
    const chart = phrases()
    const notes = (mode === "move" ? dragSelection : [hit]).flatMap((selection) => {
      const selected = chart[selection.phraseIdx]?.notes[selection.noteIdx]
      return selected ? [{
        ...selection,
        startBeat: selected.start_beat,
        duration: selected.duration_beats,
        pitch: selected.pitch,
      }] : []
    })
    const anchorBeat = notes.reduce((earliest, origin) => Math.min(earliest, origin.startBeat), note.start_beat)
    const selectedKeys = new Set(dragSelection.map(noteSelectionKey))
    let previousEnd = 0
    if (mode !== "end") {
      beforeNote: for (const [phraseIdx, phrase] of chart.entries()) {
        for (const [noteIdx, previous] of phrase.notes.entries()) {
          if (mode === "start" && phraseIdx === hit.phraseIdx && noteIdx === hit.noteIdx) break beforeNote
          const end = previous.start_beat + previous.duration_beats
          if ((mode === "start" || end <= anchorBeat) && !selectedKeys.has(noteSelectionKey({ phraseIdx, noteIdx }))) {
            previousEnd = Math.max(previousEnd, end)
          }
        }
      }
    }
    noteDragRef.current = {
      notes,
      mode,
      startX: event.clientX,
      startY: event.clientY,
      anchorBeat,
      previousEnd,
      playbackHeadBeat: snapToPlaybackHead() && livePlaybackBeat >= 0 && livePlaybackBeat <= totalBeats()
        ? Math.round(livePlaybackBeat)
        : null,
      historyRecorded: false,
    }
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

    if (!noteDragRef.current) {
      setCanvasCursor(cursorForPoint(event.clientX, event.clientY))
      return
    }
    pendingNoteDrag = { clientX: event.clientX, clientY: event.clientY }
    if (noteDragFrame !== null) return
    noteDragFrame = requestAnimationFrame(() => {
      const point = pendingNoteDrag
      noteDragFrame = null
      pendingNoteDrag = null
      if (point) applyNoteDrag(point)
    })
  }

  const applyNoteDrag = ({ clientX, clientY }: PointerPosition) => {
    const drag = noteDragRef.current
    const root = canvasRef.current
    if (!drag || !root) return
    const rect = root.getBoundingClientRect()
    const scaleY = (pitchRange() * ROW_HEIGHT) / rect.height
    const beatDelta = Math.round((clientX - drag.startX) / pixelsPerBeat())
    const pitchDelta = -Math.round(((clientY - drag.startY) * scaleY) / ROW_HEIGHT)
    const livePlaybackHeadBeat = snapToPlaybackHead() && audioRef.current
      ? Math.round((playbackPosition() - gap_s()) / secondsPerBeat())
      : null
    const playbackHeadBeat = drag.playbackHeadBeat ?? livePlaybackHeadBeat
    const snapCandidate = (candidate: number, previousTarget: number): number => {
      const previousDistance = snapToPrevious() && previousTarget > 0
        ? Math.abs(candidate - previousTarget)
        : Infinity
      const playheadDistance = playbackHeadBeat === null ? Infinity : Math.abs(candidate - playbackHeadBeat)
      if (playbackHeadBeat !== null && playheadDistance <= 1.5 && playheadDistance < previousDistance) return playbackHeadBeat
      return previousDistance <= 1.5 ? previousTarget : candidate
    }
    const groupStart = drag.mode === "move"
      ? Math.max(0, snapCandidate(drag.anchorBeat + beatDelta, drag.previousEnd))
      : drag.anchorBeat
    const previous = phrases()
    const next = editNotes(previous, drag.notes, (note, origin) => {
      let start_beat = origin.startBeat
      let duration_beats = origin.duration
      if (drag.mode === "move") {
        start_beat += groupStart - drag.anchorBeat
      } else if (drag.mode === "start") {
        start_beat = Math.max(0, Math.min(origin.startBeat + origin.duration - 1, snapCandidate(origin.startBeat + beatDelta, drag.previousEnd)))
        duration_beats -= start_beat - origin.startBeat
      } else {
        const end = Math.max(origin.startBeat + 1, snapCandidate(origin.startBeat + origin.duration + beatDelta, 0))
        duration_beats = end - origin.startBeat
      }
      const pitch = Math.max(-24, Math.min(48, origin.pitch + pitchDelta))
      return start_beat === note.start_beat && duration_beats === note.duration_beats && pitch === note.pitch
        ? note
        : { ...note, start_beat, duration_beats, pitch }
    })
    if (next === previous) return
    if (!drag.historyRecorded) {
      recordEdit(previous)
      drag.historyRecorded = true
    }
    onResultChange({ ...result(), phrases: next })
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

    const drag = noteDragRef.current
    const point = pendingNoteDrag
    cancelNoteDragFrame()
    if (drag && event.type === "pointerup" && (point || drag.historyRecorded
      || event.clientX !== drag.startX || event.clientY !== drag.startY)) {
      applyNoteDrag(event)
    } else if (point) {
      applyNoteDrag(point)
    }
    noteDragRef.current = null
    setCanvasCursor(cursorForPoint(event.clientX, event.clientY))
  }

  const sel = createMemo(() => {
    const activeNote = selectedNote()
    return selectedNotes().length === 1 && activeNote
      ? phrases()[activeNote.phraseIdx]?.notes[activeNote.noteIdx]
      : null
  })

  const editorAnchor = createMemo(() => {
    const note = sel()
    const origin = pianoRollOrigin()
    if (!note || !origin) return null
    return {
      left: origin.left + (note.start_beat + gapBeats()) * pixelsPerBeat(),
      top: origin.top + (maxPitch() - note.pitch) * ROW_HEIGHT,
      width: note.duration_beats * pixelsPerBeat(),
    }
  }, {
    equals: (previous, next) => previous?.left === next?.left && previous?.top === next?.top && previous?.width === next?.width,
  })
  const hasSelectedNote = createMemo(() => Boolean(sel()))

  createEffect(
    () => [hasSelectedNote(), isExpanded()] as const,
    ([selected]) => {
      const root = canvasRef.current
      if (!selected || !root) return

      const measure = () => {
        const { left, top } = root.getBoundingClientRect()
        setPianoRollOrigin({ left, top })
      }

      let frame: number | null = null
      const queueMeasure = (event?: Event) => {
        if (event?.target instanceof Element && !event.target.contains(root)) return
        if (frame !== null) return
        frame = requestAnimationFrame(() => {
          frame = null
          measure()
        })
      }
      const observer = new ResizeObserver(() => queueMeasure())
      observer.observe(root)
      if (editorRoot) observer.observe(editorRoot)
      measure()
      window.addEventListener("scroll", queueMeasure, { capture: true, passive: true })
      window.addEventListener("resize", queueMeasure, { passive: true })
      return () => {
        window.removeEventListener("scroll", queueMeasure, true)
        window.removeEventListener("resize", queueMeasure)
        observer.disconnect()
        if (frame !== null) cancelAnimationFrame(frame)
      }
    },
  )

  const updateNote = (field: keyof UltraStarNote, value: string | number) => {
    const activeNote = selectedNote()
    if (!activeNote) return
    const previous = phrases()
    const next = editNotes(previous, [activeNote], (note) => note[field] === value ? note : { ...note, [field]: value })
    if (next === previous) return
    recordEdit(previous)
    onResultChange({ ...result(), phrases: next })
  }

  const handleSave = async () => {
    if (!isDirty() || saving()) return
    if (props.confirmSaveMessage && !window.confirm(props.confirmSaveMessage)) return
    const id = jobId()
    const submitted = phrases()
    const isCurrent = () => !disposed && jobId() === id
    setSaving(true)
    setSaveError("")
    try {
      const updatedChart = props.saveNotes
        ? await props.saveNotes(submitted)
        : (await updateNotes(id, submitted), undefined)
      if (!isCurrent()) return
      if (updatedChart !== undefined) {
        onResultChange({ ...result(), ultrastar_txt: updatedChart })
      }
      setSavedPhrases(submitted)
    } catch (error) {
      if (isCurrent()) setSaveError(error instanceof Error ? error.message : "Failed to save changes")
    } finally {
      if (isCurrent()) setSaving(false)
    }
  }

  const startPlayback = (audio: HTMLAudioElement) => {
    if (isPianoTrack(audioTrack())) pianoResumeRef.current?.resume()
    const requestId = ++playRequestId
    setPlayRequested(true)
    void audio.play().catch(() => {
      if (requestId === playRequestId && audio.paused) setPlayRequested(false)
    })
  }

  const togglePlayback = () => {
    const audio = audioRef.current
    if (!audio) return
    if (playRequested() || !audio.paused) {
      ++playRequestId
      audio.pause()
      setPlayRequested(false)
    } else {
      startPlayback(audio)
    }
  }

  const loadAudioSource = (audio: HTMLAudioElement, source: string, previousSource: string | undefined) => {
    const sameSong = audioJobId === jobId()
    const shouldResume = sameSong && !audio.paused
    const time = sameSong && previousSource ? playbackPosition() : 0
    if (!sameSong) setLoadedAudioDuration(0)
    audioJobId = jobId()
    ++playRequestId
    audio.pause()
    setIsPlaying(false)
    setPlayRequested(false)
    pendingSeekTimeRef.current = time > 0 ? time : null
    setMediaTime(time)
    audio.defaultPlaybackRate = playbackRate()
    audio.playbackRate = playbackRate()
    if (!source) {
      audio.removeAttribute("src")
      audio.load()
      return
    }
    audio.src = source
    if (shouldResume) startPlayback(audio)
  }

  const selectedPitchEvidence = createMemo(() => {
    const activeNote = selectedNote()
    if (!activeNote) return undefined
    const note = phrases()[activeNote.phraseIdx]?.notes[activeNote.noteIdx]
    if (!note) return undefined
    const pitchWords = variant()?.pitch_words ?? result().pitch_words ?? result().words
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
    if (isPianoTrack(track) && audioRef.current && !audioRef.current.paused) {
      pianoResumeRef.current?.resume()
    }
    setAudioTrack(track)
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
    const id = jobId()
    setSwitchingMode(mode)
    try {
      const next = await switchAlignmentMode(id, mode)
      if (!disposed && jobId() === id) onResultChange(next)
    } catch (error) {
      console.error("Failed to switch alignment mode", error)
    } finally {
      if (!disposed && jobId() === id) setSwitchingMode(null)
    }
  }

  return (
    <div
      ref={(element) => { editorRoot = element }}
      class={`${isExpanded() ? "fixed inset-0 z-[80] h-dvh w-screen overflow-y-auto bg-zinc-950 p-4" : "max-w-7xl mx-auto"} space-y-6`}
    >
      <EditorHeader
        result={result()}
        allNotesCount={chartStats().count}
        phrasesCount={phrases().length}
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
        isPlaying={playRequested()}
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
        onSourceChange={loadAudioSource}
        onPlaying={() => {
          if (audioRef.current?.paused !== false) return
          setPlayRequested(true)
          syncPlaybackPosition()
          setIsPlaying(true)
        }}
        onWaiting={() => {
          syncPlaybackPosition()
          setIsPlaying(false)
        }}
        onPause={() => {
          if (audioRef.current?.paused !== true) return
          setPlayRequested(false)
          setIsPlaying(false)
          syncPlaybackPosition()
        }}
        onEnded={() => {
          pendingSeekTimeRef.current = null
          syncPlaybackPosition()
          setPlayRequested(false)
          setIsPlaying(false)
        }}
        onTimeUpdate={() => {
          const audio = audioRef.current
          if (audio?.paused && !audio.seeking && pendingSeekTimeRef.current === null) syncPlaybackPosition()
        }}
        onSeeked={() => {
          const audio = audioRef.current
          if (!audio || audio.seeking || !Number.isFinite(audio.currentTime)) return
          const pendingTime = pendingSeekTimeRef.current
          if (
            pendingTime !== null
            && (audio.currentTime < pendingTime - 0.05 || (audio.paused && audio.currentTime > pendingTime + 0.05))
          ) {
            audio.currentTime = pendingTime
            return
          }
          pendingSeekTimeRef.current = null
          syncPlaybackPosition()
        }}
        onLoadedMetadata={() => {
          const audio = audioRef.current
          if (!audio) return
          setLoadedAudioDuration(Number.isFinite(audio.duration) ? audio.duration : 0)
          if (pendingSeekTimeRef.current !== null) seekAudioTo(pendingSeekTimeRef.current)
        }}
        audioRef={audioRef}
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
        <div style={{ width: `${desiredCanvasWidth()}px` }}>
          <TimelineScrubber
            totalBeats={totalBeats()}
            totalDuration={totalDuration()}
            playheadX={playheadX()}
            playheadVisible={playheadVisible()}
            displayTime={displayTime()}
            beatToTime={audioBeatToTime}
            pixelsPerBeat={pixelsPerBeat()}
            visibleLeft={canvasLeft()}
            visibleWidth={viewportWidth() + 2 * PIANO_ROLL_OVERSCAN}
            timelineRef={timelineRef}
            onTimelineClick={handleTimelineClick}
            onMouseDown={handleTimelineMouseDown}
            onMouseUp={handleTimelineMouseUp}
            onSeekKeyDown={handleSeekKeyDown}
          />
          <div
            ref={(root) => {
              canvasRef.current = root
            }}
            class="relative overflow-y-auto overflow-x-hidden bg-zinc-900"
            style={{ "max-height": `${pitchRange() * ROW_HEIGHT + 20}px` }}
            aria-label="Piano roll"
          >
            <div style={{ height: `${pitchRange() * ROW_HEIGHT}px` }}>
              <canvas
                ref={setPianoRollCanvas}
                onPointerDown={handleCanvasPointerDown}
                onPointerMove={handleCanvasPointerMove}
                onPointerLeave={() => {
                  if (!noteDragRef.current && !selectionDragRef.current) setCanvasCursor("default")
                }}
                onPointerUp={handleCanvasPointerUp}
                onPointerCancel={handleCanvasPointerUp}
                onContextMenu={(e) => e.preventDefault()}
                class="absolute top-0 block select-none"
                style={{ cursor: canvasCursor() }}
                aria-label="Note grid: click a note to select and edit"
              />
            </div>
            <div
              class="pointer-events-none absolute left-0 top-0 z-10 w-0.5 bg-pink-500"
              style={{
                transform: `translateX(${playheadX()}px)`,
                visibility: playheadVisible() ? "visible" : "hidden",
                height: `${pitchRange() * ROW_HEIGHT}px`,
              }}
              aria-hidden="true"
            />
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

      <LyricsPreviewSection phrases={phrases()} />

      <RawUltrastarSection ultrastarTxt={result().ultrastar_txt} />
    </div>
  )
}
