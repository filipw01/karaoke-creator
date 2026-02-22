import { memo } from "react"
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

export const KaraokeWords = memo(function KaraokeWords({
  phraseWords,
  activeWordIndex,
  displayTime,
  displayedVerseStartTime,
  isPlaying,
}: KaraokeWordsProps) {
  if (phraseWords.length === 0) {
    return (
      <div className="min-h-32 flex flex-col items-center justify-center bg-zinc-800/40 border border-zinc-700/50 rounded-xl px-8 py-6">
        <span className="text-sm text-zinc-500">
          {isPlaying ? "—" : "Play to see lyrics"}
        </span>
      </div>
    )
  }
  return (
    <div className="min-h-32 flex flex-col items-center justify-center bg-zinc-800/40 border border-zinc-700/50 rounded-xl px-8 py-6">
      <div className="text-sm text-zinc-500 tabular-nums mb-2" aria-label="Verse start time">
        {formatVerseTimestamp(displayedVerseStartTime)}
      </div>
      <div className="flex flex-wrap items-center justify-center gap-x-3 gap-y-2 max-w-4xl">
        {phraseWords.map((word, idx) => {
          const isActive = idx === activeWordIndex
          const isPast = idx < activeWordIndex
          const duration = word.end - word.start
          const fillProgress =
            isActive && duration > 0
              ? Math.min(1, Math.max(0, (displayTime - word.start) / duration))
              : 1
          const fillPct = `${fillProgress * 100}%`
          const wordClasses = "text-3xl font-bold"
          if (isActive) {
            return (
              <span
                key={`${word.start}-${idx}`}
                className={`relative inline-block scale-105 transition-transform drop-shadow-[0_0_8px_rgba(139,92,246,0.6)] ${wordClasses}`}
              >
                <span className="text-zinc-300">{word.text}</span>
                <span
                  className="absolute  left-0 top-0 overflow-hidden text-violet-400 transition-none"
                  style={{ width: fillPct }}
                  aria-hidden
                >
                  {word.text}
                </span>
              </span>
            )
          }
          return (
            <span
              key={`${word.start}-${idx}`}
              className={`${wordClasses} ${
                isPast ? "text-violet-400" : "text-zinc-300"
              }`}
            >
              {word.text}
            </span>
          )
        })}
      </div>
    </div>
  )
})
