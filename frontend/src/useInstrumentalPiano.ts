import { createEffect, createMemo, onCleanup, type Accessor } from "solid-js"
import type { AudioTrack, Phrase } from "./types"

const midiToFreq = (midi: number): number => 440 * 2 ** ((midi - 69) / 12)
const PIANO_SCHEDULING_LOOKAHEAD_SEC = 0.08
const PIANO_LATE_NOTE_WINDOW_SEC = 0.1

export const isPianoTrack = (track: AudioTrack): boolean =>
  track === "instrumental_with_playback" || track === "vocals_with_playback"

/** Play a single piano-like note via Web Audio API. Optional startTime (in context time) to schedule ahead. */
export const playPianoNote = (
  ctx: AudioContext,
  destination: AudioNode,
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
  pianoResumeRef.current = {
    resume: () => {
      ctx ??= new AudioContext()
      void ctx.resume().catch(() => {})
    },
    getContext: () => ctx,
  }
  onCleanup(() => {
    if (ctx) void ctx.close().catch(() => {})
    pianoResumeRef.current = null
  })

  const scheduledNotes = createMemo(() => {
    const secondsPerBeat = 60 / (ustar_bpm() * 4)
    const gap = gap_s()
    return phrases().flatMap((phrase) => phrase.notes.map((note) => ({
      start: gap + note.start_beat * secondsPerBeat,
      duration: note.duration_beats * secondsPerBeat,
      midi: note.pitch + 60,
    }))).sort((left, right) => left.start - right.start)
  })

  createEffect(
    () => isPianoTrack(audioTrack()) && isPlaying()
      ? { notes: scheduledNotes(), rate: playbackRate() }
      : null,
    (schedule) => {
      const audio = audioRef.current
      const context = ctx
      if (!schedule || !audio || !context) return
      const { notes, rate } = schedule
      let nextNote = 0
      let output: GainNode | null = null

      const disconnect = () => {
        output?.disconnect()
        output = null
      }

      const tick = () => {
        if (!audio.paused && !audio.seeking && context.state === "running") {
          const time = audio.currentTime
          if (!output) {
            output = context.createGain()
            output.connect(context.destination)
            // Jump to the current time when playback starts or the user seeks.
            let low = 0
            let high = notes.length
            while (low < high) {
              const middle = Math.floor((low + high) / 2)
              if (notes[middle].start < time - PIANO_LATE_NOTE_WINDOW_SEC) low = middle + 1
              else high = middle
            }
            nextNote = low
          }

          while (nextNote < notes.length && notes[nextNote].start <= time + PIANO_SCHEDULING_LOOKAHEAD_SEC * rate) {
            const note = notes[nextNote++]
            if (note.start < time - PIANO_LATE_NOTE_WINDOW_SEC || note.duration <= 0) continue
            const startAt = context.currentTime + Math.max(0, (note.start - time) / rate)
            playPianoNote(context, output, note.midi, note.duration / rate, 0.65, startAt)
          }
        }
        frame = requestAnimationFrame(tick)
      }

      audio.addEventListener("seeking", disconnect)
      let frame = requestAnimationFrame(tick)
      return () => {
        cancelAnimationFrame(frame)
        audio.removeEventListener("seeking", disconnect)
        disconnect()
      }
    },
  )
}
