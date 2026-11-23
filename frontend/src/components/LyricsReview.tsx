import { createSignal } from "solid-js"
import { confirmLyrics, searchLyricsOnline } from "../api"
import type { LyricsSearchResult } from "../api"

interface LyricsReviewProps {
  jobId: string
  foundLyrics: string
  artist: string
  title: string
  onConfirmed: () => void
  onError: (error: string) => void
}

const formatDuration = (seconds: number): string => {
  const minutes = Math.floor(seconds / 60)
  const remainder = Math.floor(seconds % 60)
  return `${minutes}:${remainder.toString().padStart(2, "0")}`
}

export const LyricsReview = ({
  jobId,
  foundLyrics,
  artist,
  title,
  onConfirmed,
  onError,
}: LyricsReviewProps) => {
  const [lyrics, setLyrics] = createSignal(foundLyrics)
  const [submitting, setSubmitting] = createSignal(false)
  const [searchQuery, setSearchQuery] = createSignal(`${artist} ${title}`.trim())
  const [searching, setSearching] = createSignal(false)
  const [searchResults, setSearchResults] = createSignal<Array<LyricsSearchResult>>([])
  const [hasSearched, setHasSearched] = createSignal(false)

  const handleConfirm = async () => {
    setSubmitting(true)
    try {
      await confirmLyrics(jobId, lyrics())
      onConfirmed()
    } catch (err) {
      onError(err instanceof Error ? err.message : "Failed to confirm lyrics")
      setSubmitting(false)
    }
  }

  const handleSkip = async () => {
    setSubmitting(true)
    try {
      await confirmLyrics(jobId, "")
      onConfirmed()
    } catch (err) {
      onError(err instanceof Error ? err.message : "Failed to start transcription")
      setSubmitting(false)
    }
  }

  const handleSearch = async () => {
    if (!searchQuery().trim()) return
    setSearching(true)
    setHasSearched(true)
    try {
      const results = await searchLyricsOnline(searchQuery().trim())
      setSearchResults(results)
    } catch {
      setSearchResults([])
    } finally {
      setSearching(false)
    }
  }

  const handlePickResult = (result: LyricsSearchResult) => {
    setLyrics(result.lyrics)
    setSearchResults([])
    setHasSearched(false)
  }

  return (
    <div class="max-w-3xl mx-auto space-y-6">
      <div class="text-center">
        <div class="w-16 h-16 mx-auto mb-4 bg-violet-500/15 rounded-full flex items-center justify-center">
          <svg class="w-8 h-8 text-violet-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width={2}>
            <path stroke-linecap="round" stroke-linejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
        </div>
        <h2 class="text-2xl font-bold text-white mb-1">Review Lyrics</h2>
        <p class="text-zinc-400">
          {artist} — {title}
        </p>
      </div>

      <div class="bg-zinc-800/60 border border-zinc-700/50 rounded-xl p-5 space-y-4">
        {foundLyrics ? (
          <div class="bg-emerald-500/10 border border-emerald-500/20 rounded-lg px-4 py-3">
            <p class="text-emerald-400 text-sm font-medium">
              Lyrics found online! Review and edit if needed.
            </p>
            <p class="text-zinc-500 text-xs mt-1">
              Providing accurate lyrics dramatically improves transcription timing.
            </p>
          </div>
        ) : (
          <div class="bg-amber-500/10 border border-amber-500/20 rounded-lg px-4 py-3">
            <p class="text-amber-400 text-sm font-medium">
              No lyrics found automatically.
            </p>
            <p class="text-zinc-500 text-xs mt-1">
              Search below, paste lyrics manually, or skip to let the AI transcribe from audio only.
            </p>
          </div>
        )}

        <div class="bg-zinc-900/60 border border-zinc-700/40 rounded-lg p-4 space-y-3">
          <p class="text-sm text-zinc-300 font-medium">Search for lyrics</p>
          <div class="flex gap-2">
            <input
              type="text"
              value={searchQuery()}
              onInput={(event) => setSearchQuery(event.currentTarget.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") void handleSearch()
              }}
              placeholder="Search by artist, song title, or lyrics..."
              class="flex-1 bg-zinc-800 border border-zinc-600 rounded-lg px-3 py-2 text-white text-sm
                         focus:outline-none focus:border-violet-500"
            />
            <button
              type="button"
              onClick={() => void handleSearch()}
              disabled={searching() || !searchQuery().trim()}
              class="bg-violet-600 hover:bg-violet-500 text-white px-4 py-2 rounded-lg text-sm
                         font-medium transition-colors disabled:opacity-50 cursor-pointer shrink-0"
            >
              {searching() ? "Searching..." : "Search"}
            </button>
          </div>

          {searchResults().length > 0 && (
            <div class="space-y-2 max-h-64 overflow-y-auto">
              {searchResults().map((result) => (
                <button
                  type="button"
                  onClick={() => handlePickResult(result)}
                  class="w-full text-left bg-zinc-800 hover:bg-zinc-700/80 border border-zinc-700/50
                             hover:border-violet-500/40 rounded-lg px-4 py-3 transition-colors cursor-pointer group"
                >
                  <div class="flex items-baseline justify-between gap-2">
                    <div class="min-w-0">
                      <span class="text-white text-sm font-medium truncate block">
                        {result.track_name}
                      </span>
                      <span class="text-zinc-400 text-xs">
                        {result.artist_name}
                        {result.album_name && ` — ${result.album_name}`}
                      </span>
                    </div>
                    <div class="flex items-center gap-2 shrink-0">
                      {result.duration > 0 && (
                        <span class="text-zinc-500 text-xs">{formatDuration(result.duration)}</span>
                      )}
                      <span class="text-zinc-600 text-xs">{result.source}</span>
                      <span class="text-violet-400 text-xs opacity-0 group-hover:opacity-100 transition-opacity">
                        Use
                      </span>
                    </div>
                  </div>
                  <p class="text-zinc-500 text-xs mt-1 line-clamp-2">
                    {result.lyrics.slice(0, 120)}...
                  </p>
                </button>
              ))}
            </div>
          )}

          {hasSearched() && !searching() && searchResults().length === 0 && (
            <p class="text-zinc-500 text-sm text-center py-2">
              No results found. Try a different search term.
            </p>
          )}
        </div>

        <div>
          <label class="block text-sm text-zinc-300 font-medium mb-2">
            Song Lyrics
          </label>
          <textarea
            value={lyrics()}
            onInput={(event) => setLyrics(event.currentTarget.value)}
            placeholder={"Paste or type lyrics here...\n\nThe AI will use these as a guide for more accurate word timestamps."}
            rows={16}
            class="w-full bg-zinc-900 border border-zinc-600 rounded-lg px-4 py-3 text-white text-sm
                       leading-relaxed focus:outline-none focus:border-violet-500 resize-y font-mono"
          />
        </div>
      </div>

      <div class="flex gap-3 justify-center">
        <button
          type="button"
          onClick={() => void handleSkip()}
          disabled={submitting()}
          class="bg-zinc-700 hover:bg-zinc-600 text-white px-6 py-3 rounded-xl
                     transition-colors disabled:opacity-50 cursor-pointer"
        >
          Skip (auto-detect)
        </button>
        <button
          type="button"
          onClick={() => void handleConfirm()}
          disabled={submitting() || !lyrics().trim()}
          class="bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-500 hover:to-fuchsia-500
                     text-white font-medium px-8 py-3 rounded-xl transition-all disabled:opacity-50 cursor-pointer"
        >
        </button>
      </div>
    </div>
  )
}
