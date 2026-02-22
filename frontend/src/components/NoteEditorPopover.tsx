import { memo } from "react"
import type { UltraStarNote } from "../types"

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
  onClose: () => void
  onUpdate: (field: keyof UltraStarNote, value: string | number) => void
}

export const NoteEditorPopover = memo(function NoteEditorPopover({
  note,
  phraseIdx,
  noteIdx,
  anchor,
  onClose,
  onUpdate,
}: NoteEditorPopoverProps) {
  return (
    <div
      className="fixed z-50 w-[min(90vw,420px)] rounded-xl border border-zinc-600 bg-zinc-800 p-4 shadow-xl"
      style={{
        left: anchor.left + anchor.width / 2,
        top: anchor.top - 8,
        transform: "translate(-50%, -100%)",
      }}
      role="dialog"
      aria-label="Edit note"
    >
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-white font-medium">
          Edit Note — Phrase {phraseIdx + 1}, Note {noteIdx + 1}
        </h3>
        <button
          type="button"
          onClick={onClose}
          className="rounded p-1 text-zinc-400 hover:bg-zinc-700 hover:text-white"
          aria-label="Close"
        >
          ×
        </button>
      </div>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
        <div>
          <label className="mb-1 block text-xs text-zinc-400">Text</label>
          <input
            type="text"
            value={note.text}
            onChange={(e) => onUpdate("text", e.target.value)}
            className="w-full rounded border border-zinc-600 bg-zinc-900 px-2 py-1.5 text-sm text-white focus:border-violet-500 focus:outline-none"
          />
        </div>
        <div>
          <label className="mb-1 block text-xs text-zinc-400">Type</label>
          <select
            value={note.note_type}
            onChange={(e) => onUpdate("note_type", e.target.value)}
            className="w-full rounded border border-zinc-600 bg-zinc-900 px-2 py-1.5 text-sm text-white focus:border-violet-500 focus:outline-none"
          >
            <option value=":">Normal (:)</option>
            <option value="*">Golden (*)</option>
            <option value="F">Freestyle (F)</option>
          </select>
        </div>
        <div>
          <label className="mb-1 block text-xs text-zinc-400">Start Beat</label>
          <input
            type="number"
            value={note.start_beat}
            onChange={(e) => onUpdate("start_beat", Number.parseInt(e.target.value) ?? 0)}
            className="w-full rounded border border-zinc-600 bg-zinc-900 px-2 py-1.5 text-sm text-white focus:border-violet-500 focus:outline-none"
          />
        </div>
        <div>
          <label className="mb-1 block text-xs text-zinc-400">Duration</label>
          <input
            type="number"
            value={note.duration_beats}
            min={1}
            onChange={(e) =>
              onUpdate("duration_beats", Math.max(1, Number.parseInt(e.target.value) ?? 1))
            }
            className="w-full rounded border border-zinc-600 bg-zinc-900 px-2 py-1.5 text-sm text-white focus:border-violet-500 focus:outline-none"
          />
        </div>
        <div>
          <label className="mb-1 block text-xs text-zinc-400">
            Pitch ({pitchToName(note.pitch)})
          </label>
          <input
            type="number"
            value={note.pitch}
            onChange={(e) => onUpdate("pitch", Number.parseInt(e.target.value) ?? 0)}
            className="w-full rounded border border-zinc-600 bg-zinc-900 px-2 py-1.5 text-sm text-white focus:border-violet-500 focus:outline-none"
          />
        </div>
      </div>
    </div>
  )
})
