export const EditorHint = () => {
  return (
    <div class="bg-zinc-800/40 rounded-xl border border-dashed border-zinc-700/50 px-4 py-6 text-center">
      <p class="text-sm text-zinc-500">
        Drag a labeled pitch bar to change its timing, duration, and pitch. Hold Shift to select multiple bars, or drag across empty canvas space to marquee-select them. Selected bars move and nudge together. Enable playback-head snapping to align a dragged bar to the cursor. Left/Right seeks by one chart step; Up/Down jumps between note-segment boundaries. Use Shift+Left/Right to nudge selected bars in time, and Cmd+Up/Down to raise or lower selected pitches by one semitone. Option+scroll zooms around the pointer. Use [ and ] to change playback speed. Use{" "}
        <kbd class="rounded bg-zinc-700 px-1.5 py-0.5 font-mono text-xs text-zinc-400">
          Space
        </kbd>{" "}
        to play/pause. Use{" "}
        <kbd class="rounded bg-zinc-700 px-1.5 py-0.5 font-mono text-xs text-zinc-400">
          Cmd+Z
        </kbd>{" "}
        to undo and{" "}
        <kbd class="rounded bg-zinc-700 px-1.5 py-0.5 font-mono text-xs text-zinc-400">
          Cmd+Shift+Z
        </kbd>{" "}
        to redo, and{" "}
        <kbd class="rounded bg-zinc-700 px-1.5 py-0.5 font-mono text-xs text-zinc-400">
          Esc
        </kbd>{" "}
        to deselect.
      </p>
    </div>
  )
}
