import { memo } from "react"
import type { JobResult } from "../types"

interface LyricsPreviewSectionProps {
  phrases: JobResult["phrases"]
}

export const LyricsPreviewSection = memo(function LyricsPreviewSection({
  phrases,
}: LyricsPreviewSectionProps) {
  return (
    <section className="bg-zinc-800/60 border border-zinc-700/50 rounded-xl p-4" aria-label="Lyrics preview">
      <h3 className="text-white font-medium mb-3">Lyrics Preview</h3>
      <div className="text-zinc-300 leading-relaxed space-y-1 font-mono text-sm">
        {phrases.map((phrase, i) => (
          <p key={i}>
            {phrase.notes.map((n) => n.text).join("")}
          </p>
        ))}
      </div>
    </section>
  )
})
