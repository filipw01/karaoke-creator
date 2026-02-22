import { memo } from "react"
import type { JobResult } from "../types"

interface EditorHeaderProps {
  result: JobResult
  allNotesCount: number
  phrasesCount: number
  showSettings: boolean
  onToggleSettings: () => void
  saving: boolean
  isDirty: boolean
  onSave: () => void
  onExport: () => void
}

export const EditorHeader = memo(function EditorHeader({
  result,
  allNotesCount,
  phrasesCount,
  showSettings,
  onToggleSettings,
  saving,
  isDirty,
  onSave,
  onExport,
}: EditorHeaderProps) {
  return (
    <div className="flex items-center justify-between">
      <div>
        <h2 className="text-2xl font-bold text-white">
          {result.metadata.artist} — {result.metadata.title}
        </h2>
        <p className="text-zinc-400 text-sm">
          BPM: {result.metadata.bpm.toFixed(1)} | GAP: {result.metadata.gap_ms.toFixed(0)}ms
          | {allNotesCount} notes in {phrasesCount} phrases
        </p>
      </div>
      <div className="flex gap-2">
        <button
          type="button"
          onClick={onToggleSettings}
          className={`px-4 py-2 rounded-lg transition-colors cursor-pointer ${
            showSettings ? "bg-fuchsia-600 text-white" : "bg-zinc-700 hover:bg-zinc-600 text-white"
          }`}
        >
          Lyrics Settings
        </button>
        <button
          type="button"
          onClick={onSave}
          disabled={saving}
          aria-busy={saving}
          aria-label={saving ? "Saving changes" : isDirty ? "Save unsaved changes" : "Save changes"}
          className="bg-zinc-700 hover:bg-zinc-600 text-white px-4 py-2 rounded-lg
                     transition-colors disabled:opacity-50 cursor-pointer"
        >
          {saving ? "Saving..." : isDirty ? "Save changes (unsaved)" : "Save Changes"}
        </button>
        <button
          type="button"
          onClick={onExport}
          className="bg-violet-600 hover:bg-violet-500 text-white px-4 py-2 rounded-lg
                     transition-colors cursor-pointer"
        >
          Export
        </button>
      </div>
    </div>
  )
})
