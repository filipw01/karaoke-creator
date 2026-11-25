import { createMemo, For } from "solid-js"
import type { JobResult } from "../types"

interface EditorHeaderProps {
  result: JobResult
  allNotesCount: number
  phrasesCount: number
  saving: boolean
  isDirty: boolean
  onSave: () => void
  onExport: () => void
  onOpenLyricsAlignment: () => void
  onModeChange: (mode: "forcealign" | "mms" | "canonical") => void
  switchingMode: string | null
  showLyricsAlignment?: boolean
  showExport?: boolean
}

export const EditorHeader = (props: EditorHeaderProps) => {
  const comparison = createMemo(() => props.result.alignment_comparison)
  const alignments = createMemo(() => props.result.alignment_results)
  const modes = createMemo(() => (["forcealign", "mms"] as const).filter((mode) => alignments()?.[mode]))
  const currentMode = createMemo(() => props.result.current_alignment_mode ?? "canonical")
  const confidence = createMemo(() => comparison()?.confidence)
  return (
    <div class="flex items-start justify-between gap-4">
      <div>
        <h2 class="text-2xl font-bold text-white">
          {props.result.metadata.artist} — {props.result.metadata.title}
        </h2>
        <p class="text-zinc-400 text-sm">
          BPM: {props.result.metadata.bpm.toFixed(1)} | GAP: {props.result.metadata.gap_ms.toFixed(0)}ms
          | {props.allNotesCount} notes in {props.phrasesCount} phrases
        </p>
        {modes().length > 0 && (
          <div class="mt-3 flex items-center gap-2 flex-wrap">
            <span class="text-xs text-zinc-500">Alignment:</span>
            <For each={modes()}>
              {(mode) => (
                <button
                  type="button"
                  onClick={() => props.onModeChange(mode)}
                  disabled={props.switchingMode !== null}
                  class={`px-2.5 py-1 rounded-md text-xs transition-colors ${
                    currentMode() === mode
                      ? "bg-violet-600 text-white"
                      : "bg-zinc-800 text-zinc-300 hover:bg-zinc-700"
                  } disabled:opacity-50`}
                >
                  {props.switchingMode === mode ? "Switching…" : mode === "forcealign" ? "ForceAlign" : "MMS_FA"}
                </button>
              )}
            </For>
            {comparison()?.status === "running" && (
              <span class="text-xs text-amber-300">MMS_FA processing in background…</span>
            )}
            {comparison()?.status === "completed" && typeof confidence() === "number" && (
              <span class="text-xs text-zinc-400" title="Agreement between ForceAlign and MMS_FA timing and pitch results">
                Agreement {Math.round(confidence()! * 100)}%
                {typeof comparison()!.timing_agreement === "number" && typeof comparison()!.pitch_agreement === "number"
                  ? ` · timing ${Math.round(comparison()!.timing_agreement! * 100)}% · pitch ${Math.round(comparison()!.pitch_agreement! * 100)}%`
                  : ""}
              </span>
            )}
          </div>
        )}
      </div>
      <div class="flex gap-2">
        {props.showLyricsAlignment !== false && (
          <button
            type="button"
            onClick={props.onOpenLyricsAlignment}
            class="bg-zinc-700 hover:bg-zinc-600 text-white px-4 py-2 rounded-lg transition-colors"
          >
            Lyrics &amp; alignment
          </button>
        )}
        <button
          type="button"
          onClick={props.onSave}
          disabled={props.saving}
          aria-busy={props.saving ? "true" : "false"}
          aria-label={props.saving ? "Saving changes" : props.isDirty ? "Save unsaved changes" : "Save changes"}
          class="bg-zinc-700 hover:bg-zinc-600 text-white px-4 py-2 rounded-lg
                     transition-colors disabled:opacity-50 cursor-pointer"
        >
          {props.saving ? "Saving..." : props.isDirty ? "Save changes (unsaved)" : "Save Changes"}
        </button>
        {props.showExport !== false && (
          <button
            type="button"
            onClick={props.onExport}
            class="bg-violet-600 hover:bg-violet-500 text-white px-4 py-2 rounded-lg
                       transition-colors cursor-pointer"
          >
            Export
          </button>
        )}
      </div>
    </div>
  )
}
