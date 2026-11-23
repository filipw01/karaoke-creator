import { createMemo, onCleanup, type Accessor } from "solid-js"
import { createEffect } from "./solidCompat"
import type { AudioTrack, Phrase } from "./types"

const midiToFreq = (midi: number): number => 440 * 2 ** ((midi - 69) / 12)
const PIANO_SCHEDULING_LOOKAHEAD_SEC = 0.08
const PIANO_LATE_NOTE_WINDOW_SEC = 0.1

/** Play a single piano-like note via Web Audio API. Optional startTime (in context time) to schedule ahead. */
export const playPianoNote = (
  ctx: AudioContext,
  destination: AudioDestinationNode,
  midiPitch: number,
  durationSec: number,
  volume = 0.65,
  startTime?: number
): void => {
  const now = startTime ?? ctx.currentTime
  const osc = ctx.createOscillator()
  const gain = ctx.createGain()
  osc.type = "triangle"
  osc.frequency.setValueAtTime(midiToFreq(midiPitch), now)
  gain.gain.setValueAtTime(0, now)
  gain.gain.linearRampToValueAtTime(volume, now + 0.02)
  const releaseStart = now + Math.max(0.02, durationSec - 0.06)
  gain.gain.setValueAtTime(volume, releaseStart)
  gain.gain.linearRampToValueAtTime(0, now + durationSec)
  osc.connect(gain)
  gain.connect(destination)
  osc.start(now)
  osc.stop(now + durationSec)
}

/** Create an AudioContext, resume it, and play a short test phrase. Call from a user gesture (e.g. button click). */
export const playTestPiano = (): void => {
  const ctx = new AudioContext()
  ctx.resume().then(() => {
    const now = ctx.currentTime
    const notes = [60, 64, 67, 64]
    notes.forEach((midi, i) => {
      playPianoNote(ctx, ctx.destination, midi, 0.28, 0.7, now + i * 0.28)
    })
  }).catch(() => {})
}

export interface PianoResumeRef {
  resume: () => void
  /** Use this context for test sound when on piano/instrumental to verify playback context works. */
  getContext: () => AudioContext | null
}

interface UseInstrumentalPianoArgs {
  audioRef: { current: HTMLAudioElement | null }
  audioTrack: Accessor<AudioTrack>
  isPlaying: Accessor<boolean>
  phrases: Accessor<Array<Phrase>>
  gap_s: Accessor<number>
  ustar_bpm: Accessor<number>
  playbackRate: Accessor<number>
  /** Set by hook so Editor can call resume() on play click (same user gesture). */
  pianoResumeRef: { current: PianoResumeRef | null }
}

/**
 * Playback modes layer the chart's piano notes over the selected vocal or
 * instrumental stem while keeping the media element as the timing source.
 */
export const useInstrumentalPiano = ({
  audioRef,
  audioTrack,
  isPlaying,
  phrases,
  gap_s,
  ustar_bpm,
  playbackRate,
  pianoResumeRef,
}: UseInstrumentalPianoArgs): void => {
  let ctx: AudioContext | null = null
  const played = new Set<string>()
  let lastTime = 0
  let phrasesRef = phrases()
  let timingRef = { gap_s: gap_s(), ustar_bpm: ustar_bpm(), playbackRate: playbackRate() }

  createEffect(() => {
    phrasesRef = phrases()
    timingRef = { gap_s: gap_s(), ustar_bpm: ustar_bpm(), playbackRate: playbackRate() }
  }, () => undefined)

  const isPianoMode = createMemo(() =>
    audioTrack() === "instrumental_with_playback" || audioTrack() === "vocals_with_playback"
  )

  // Single effect: when piano/instrumental and playing, ensure context exists and run the note loop.
  // Doing both in one effect avoids the loop being torn down by the other effect's deps and guarantees we have ctx.
  createEffect(() => {
    if (!isPianoMode() || !isPlaying()) return
    const audio = audioRef.current
    if (!audio) return

    let localCtx = ctx
    if (!localCtx) {
      localCtx = new AudioContext()
      ctx = localCtx
      const source = localCtx.createMediaElementSource(audio)
      source.connect(localCtx.destination)
      pianoResumeRef.current = {
        resume: () => { ctx?.resume().catch(() => {}) },
        getContext: () => ctx,
      }
    }
    if (!localCtx) return
    localCtx.resume().catch(() => {})

    played.clear()
    lastTime = 0

    // UltraStar note beats are quarter-beat subdivisions of the chart BPM.
    // This must match Editor.tsx and backend/processing/ultrastar.py.
    const beatToTime = (beat: number): number =>
      (beat * 60) / (timingRef.ustar_bpm * 4) + timingRef.gap_s

    const triggerNotesForTime = (t: number): void => {
      if (t < lastTime - 0.03) played.clear()
      lastTime = t
      const phraseList = phrasesRef
      const { ustar_bpm: ubpm } = timingRef
      for (let pIdx = 0; pIdx < phraseList.length; pIdx++) {
        const phrase = phraseList[pIdx]
        for (let nIdx = 0; nIdx < phrase.notes.length; nIdx++) {
          const note = phrase.notes[nIdx]
          const startTime = beatToTime(note.start_beat)
          const rate = Math.max(0.25, timingRef.playbackRate)
          const durationSec = (note.duration_beats * 60) / (ubpm * 4 * rate)
          const key = `${pIdx}-${nIdx}`
          if (
            startTime >= t - PIANO_LATE_NOTE_WINDOW_SEC &&
            startTime <= t + PIANO_SCHEDULING_LOOKAHEAD_SEC &&
            !played.has(key)
          ) {
            played.add(key)
            const midiNote = note.pitch + 60
            const startAt = localCtx.currentTime + Math.max(0, (startTime - t) / rate)
            playPianoNote(localCtx, localCtx.destination, midiNote, durationSec, 0.65, startAt)
          }
        }
      }
    }

    let rafId = 0
    const tick = (): void => {
      // The media element is the source of truth for both the timeline and
      // the audio stem. Using performance.now() here accumulated drift and
      // made piano notes separate from a seek or a paused/resumed element.
      const t = audio.currentTime
      if (audio.seeking) {
        played.clear()
        lastTime = t
      } else {
        triggerNotesForTime(t)
      }
      rafId = requestAnimationFrame(tick)
    }
    rafId = requestAnimationFrame(tick)

    onCleanup(() => {
      cancelAnimationFrame(rafId)
    })
  }, () => undefined)
}
