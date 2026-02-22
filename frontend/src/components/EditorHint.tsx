import { memo } from "react"

export const EditorHint = memo(function EditorHint() {
  return (
    <div className="bg-zinc-800/40 rounded-xl border border-dashed border-zinc-700/50 px-4 py-6 text-center">
      <p className="text-sm text-zinc-500">
        Click a note in the piano roll to select and edit it. Use{" "}
        <kbd className="rounded bg-zinc-700 px-1.5 py-0.5 font-mono text-xs text-zinc-400">
          Space
        </kbd>{" "}
        to play/pause and{" "}
        <kbd className="rounded bg-zinc-700 px-1.5 py-0.5 font-mono text-xs text-zinc-400">
          Esc
        </kbd>{" "}
        to deselect.
      </p>
    </div>
  )
})
