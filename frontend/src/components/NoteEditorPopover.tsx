import { For, Show } from "solid-js"
import type { UltraStarNote, Word } from "../types"

const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"] as const

const pitchToName = (pitch: number): string => {
  const midi = pitch + 60
  const octave = Math.floor(midi / 12) - 1
  const note = NOTE_NAMES[midi % 12]
  return `${note}${octave}`
}

interface EditorAnchor {
  left: number
  top: number
  width: number
  height: number
}

interface NoteEditorPopoverProps {
  note: UltraStarNote
  phraseIdx: number
  noteIdx: number
  anchor: EditorAnchor
  pitchEvidence?: Word
  onClose: () => void
  onUpdate: (field: keyof UltraStarNote, value: string | number) => void
}

export const NoteEditorPopover = (props: NoteEditorPopoverProps) => {
  const candidates = () => props.pitchEvidence?.pitch_candidates ?? []
  const contour = () => props.pitchEvidence?.pitch_contour ?? []
  const detectedMidi = () => props.pitchEvidence?.midi_note ?? props.note.pitch + 60
  const confidence = () => props.pitchEvidence?.confidence ?? 0
  const confidencePercent = () => `${Math.round(Math.max(0, confidence()) * 100)}%`
  const isOverride = () => props.note.pitch + 60 !== detectedMidi()

  const midiToName = (midi: number): string => {
    const pitchClass = ((midi % 12) + 12) % 12
    const octave = Math.floor(midi / 12) - 1
    return `${NOTE_NAMES[pitchClass]}${octave}`
  }

  const contourPoints = () => {
    const points = contour()
    if (points.length < 2) return ""
    const frequencies = points
      .map(([, hz]) => hz)
      .filter((hz) => Number.isFinite(hz) && hz > 0)
    if (frequencies.length < 2) return ""
    const lowest = Math.log2(Math.min(...frequencies))
    const highest = Math.log2(Math.max(...frequencies))
    const range = Math.max(highest - lowest, 0.08)
    return points
      .map(([, hz], index) => {
        if (!Number.isFinite(hz) || hz <= 0) return null
        const x = points.length === 1 ? 0 : (index / (points.length - 1)) * 160
        const y = 36 - ((Math.log2(hz) - lowest) / range) * 28
        return `${x.toFixed(1)},${y.toFixed(1)}`
      })
      .filter((point): point is string => point !== null)
      .join(" ")
  }

  return (
    <div
      class="fixed z-50 w-[min(92vw,480px)] rounded-xl border border-zinc-600 bg-zinc-800 p-4 shadow-xl"
      style={{
        left: `${props.anchor.left + props.anchor.width / 2}px`,
        top: `${props.anchor.top - 8}px`,
        transform: "translate(-50%, -100%)",
      }}
      role="dialog"
      aria-label="Edit note"
    >
      <div class="mb-3 flex items-center justify-between">
        <h3 class="text-white font-medium">
          Edit Note — Phrase {props.phraseIdx + 1}, Note {props.noteIdx + 1}
        </h3>
        <button
          type="button"
          onClick={props.onClose}
          class="rounded p-1 text-zinc-400 hover:bg-zinc-700 hover:text-white"
          aria-label="Close"
        >
          ×
        </button>
      </div>
      <div class="grid grid-cols-2 gap-3 sm:grid-cols-5">
        <div>
          <label class="mb-1 block text-xs text-zinc-400">Text</label>
          <input
            type="text"
            value={props.note.text}
            onInput={(e) => props.onUpdate("text", e.currentTarget.value)}
            class="w-full rounded border border-zinc-600 bg-zinc-900 px-2 py-1.5 text-sm text-white focus:border-violet-500 focus:outline-none"
          />
        </div>
        <div>
          <label class="mb-1 block text-xs text-zinc-400">Type</label>
          <select
            value={props.note.note_type}
            onChange={(e) => props.onUpdate("note_type", e.currentTarget.value)}
            class="w-full rounded border border-zinc-600 bg-zinc-900 px-2 py-1.5 text-sm text-white focus:border-violet-500 focus:outline-none"
          >
            <option value=":">Normal (:)</option>
            <option value="*">Golden (*)</option>
            <option value="F">Freestyle (F)</option>
          </select>
        </div>
        <div>
          <label class="mb-1 block text-xs text-zinc-400">Start Beat</label>
          <input
            type="number"
            value={props.note.start_beat}
            onInput={(e) => props.onUpdate("start_beat", Number.parseInt(e.currentTarget.value) || 0)}
            class="w-full rounded border border-zinc-600 bg-zinc-900 px-2 py-1.5 text-sm text-white focus:border-violet-500 focus:outline-none"
          />
        </div>
        <div>
          <label class="mb-1 block text-xs text-zinc-400">Duration</label>
          <input
            type="number"
            value={props.note.duration_beats}
            min={1}
            onInput={(e) =>
              props.onUpdate("duration_beats", Math.max(1, Number.parseInt(e.currentTarget.value) || 1))
            }
            class="w-full rounded border border-zinc-600 bg-zinc-900 px-2 py-1.5 text-sm text-white focus:border-violet-500 focus:outline-none"
          />
        </div>
        <div>
          <label class="mb-1 block text-xs text-zinc-400">
            Pitch ({pitchToName(props.note.pitch)})
          </label>
          <input
            type="number"
            value={props.note.pitch}
            onInput={(e) => props.onUpdate("pitch", Number.parseInt(e.currentTarget.value) || 0)}
            class="w-full rounded border border-zinc-600 bg-zinc-900 px-2 py-1.5 text-sm text-white focus:border-violet-500 focus:outline-none"
          />
        </div>
      </div>
      <div class="mt-4 border-t border-zinc-700 pt-3">
        <div class="mb-2 flex items-center justify-between">
          <div>
            <div class="text-xs font-medium text-zinc-200">Vocal pitch evidence</div>
            <div class="text-[11px] text-zinc-400">
              Detected {midiToName(detectedMidi())} · confidence {confidencePercent()}
            </div>
          </div>
          <Show when={isOverride()}>
            <button
              type="button"
              onClick={() => props.onUpdate("pitch", detectedMidi() - 60)}
              class="rounded border border-zinc-600 px-2 py-1 text-xs text-zinc-300 hover:bg-zinc-700"
            >
              Reset
            </button>
          </Show>
        </div>

        <div class="grid grid-cols-[160px_1fr] gap-3">
          <div>
            <div class="mb-1 text-[11px] text-zinc-400">Timing-local histogram</div>
            <div class="flex h-16 items-end gap-1 border-b border-zinc-700 px-1">
              <For each={candidates().slice(0, 8)}>
                {(candidate) => (
                  <button
                    type="button"
                    title={`${midiToName(candidate.midi)} · ${Math.round(candidate.score * 100)}% support`}
                    onClick={() => props.onUpdate("pitch", candidate.midi - 60)}
                    class="flex min-w-3 flex-1 flex-col items-center justify-end"
                    aria-label={`Set pitch to ${midiToName(candidate.midi)}`}
                  >
                    <span
                      class="w-full rounded-t bg-violet-500/70 hover:bg-violet-400"
                      style={{ height: `${Math.max(8, candidate.score * 52)}px` }}
                    />
                    <span class="mt-1 text-[9px] text-zinc-400">
                      {NOTE_NAMES[((candidate.midi % 12) + 12) % 12]}
                    </span>
                  </button>
                )}
              </For>
            </div>
          </div>
          <div>
            <div class="mb-1 text-[11px] text-zinc-400">Detected contour</div>
            <svg viewBox="0 0 160 40" class="h-16 w-full rounded bg-zinc-900" aria-label="Pitch contour">
              <polyline
                points={contourPoints()}
                fill="none"
                stroke="#a78bfa"
                stroke-width="2"
                stroke-linejoin="round"
              />
            </svg>
          </div>
        </div>

        <div class="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-3">
          <For each={candidates().slice(0, 6)}>
            {(candidate, index) => (
              <button
                type="button"
                onClick={() => props.onUpdate("pitch", candidate.midi - 60)}
                class="flex items-center justify-between rounded border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-xs text-zinc-200 hover:border-violet-500"
              >
                <span>
                  {index() + 1}. {midiToName(candidate.midi)}
                </span>
                <span class="text-zinc-400">{Math.round(candidate.score * 100)}%</span>
              </button>
            )}
          </For>
        </div>
      </div>
    </div>
  )
}
