import type { JobResult } from "../types"

interface LyricsPreviewSectionProps {
  phrases: JobResult["phrases"]
}

export const LyricsPreviewSection = (props: LyricsPreviewSectionProps) => {
  return (
    <section class="bg-zinc-800/60 border border-zinc-700/50 rounded-xl p-4" aria-label="Lyrics preview">
      <h3 class="text-white font-medium mb-3">Lyrics Preview</h3>
      <div class="text-zinc-300 leading-relaxed space-y-1 font-mono text-sm">
        {props.phrases.map((phrase) => (
          <p>
            {phrase.notes.map((n) => n.text).join("")}
          </p>
        ))}
      </div>
    </section>
  )
}
