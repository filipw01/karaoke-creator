import { Show } from "solid-js"
import type { Word } from "../types"

interface KaraokeWordsProps {
  phraseWords: Array<Word>
  activeWordIndex: number
  displayTime: number
  displayedVerseStartTime: number
  isPlaying: boolean
}

const formatVerseTimestamp = (s: number): string => {
  const m = Math.floor(s / 60)
  const sec = Math.floor(s % 60)
  return `${m}:${sec.toString().padStart(2, "0")}`
}

export const KaraokeWords = (props: KaraokeWordsProps) => {
  return (
    <Show
      when={props.phraseWords.length > 0}
      fallback={
        <div class="min-h-32 flex flex-col items-center justify-center bg-zinc-800/40 border border-zinc-700/50 rounded-xl px-8 py-6">
          <span class="text-sm text-zinc-500">
            {props.isPlaying ? "—" : "Play to see lyrics"}
          </span>
        </div>
      }
    >
      <div class="min-h-32 flex flex-col items-center justify-center bg-zinc-800/40 border border-zinc-700/50 rounded-xl px-8 py-6">
        <div class="text-sm text-zinc-500 tabular-nums mb-2" aria-label="Verse start time">
          {formatVerseTimestamp(props.displayedVerseStartTime)}
        </div>
        <div class="flex flex-wrap items-center justify-center gap-x-3 gap-y-2 max-w-4xl">
          {props.phraseWords.map((word, idx) => {
            const isActive = idx === props.activeWordIndex
            const isPast = idx < props.activeWordIndex
            const alignmentStatus = word.alignment_status ?? "aligned"
            const duration = word.end - word.start
            const fillProgress =
              isActive && duration > 0
                ? Math.min(1, Math.max(0, (props.displayTime - word.start) / duration))
                : 1
            const fillPct = `${fillProgress * 100}%`
            const wordClasses = `text-3xl font-bold ${
              alignmentStatus === "estimated"
                ? "decoration-amber-400 decoration-dashed underline"
                : alignmentStatus === "unmatched"
                  ? "decoration-red-400 decoration-dashed underline"
                  : ""
            }`
            if (isActive) {
              return (
                <span
                  class={`relative inline-block scale-105 transition-transform drop-shadow-[0_0_8px_rgba(139,92,246,0.6)] ${wordClasses}`}
                >
                  <span class="text-zinc-300">{word.text}</span>
                  <span
                    class="absolute left-0 top-0 overflow-hidden text-violet-400 transition-none"
                    style={{ width: fillPct }}
                    aria-hidden="true"
                  >
                    {word.text}
                  </span>
                </span>
              )
            }
            return (
              <span
                class={`${wordClasses} ${
                  isPast ? "text-violet-400" : "text-zinc-300"
                }`}
                title={alignmentStatus === "aligned" ? undefined : alignmentStatus}
              >
                {word.text}
              </span>
            )
          })}
        </div>
      </div>
    </Show>
  )
}
