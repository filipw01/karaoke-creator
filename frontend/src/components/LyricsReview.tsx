import { useState } from "react"
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
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, "0")}`
}

export const LyricsReview = ({
  jobId,
  foundLyrics,
  artist,
  title,
  onConfirmed,
  onError,
}: LyricsReviewProps) => {
  const [lyrics, setLyrics] = useState(foundLyrics)
  const [submitting, setSubmitting] = useState(false)

  // Manual search state
  const [searchQuery, setSearchQuery] = useState(`${artist} ${title}`.trim())
  const [searching, setSearching] = useState(false)
  const [searchResults, setSearchResults] = useState<Array<LyricsSearchResult>>([])
  const [hasSearched, setHasSearched] = useState(false)

  const handleConfirm = async () => {
    setSubmitting(true)
    try {
      await confirmLyrics(jobId, lyrics)
      onConfirmed()
    } catch (err) {
      onError(err instanceof Error ? err.message : "Failed to confirm lyrics")
    } finally {
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
    } finally {
      setSubmitting(false)
    }
  }

  const handleSearch = async () => {
    if (!searchQuery.trim()) return
    setSearching(true)
    setHasSearched(true)
    try {
      const results = await searchLyricsOnline(searchQuery.trim())
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
    <div className="max-w-3xl mx-auto space-y-6">
      <div className="text-center">
        <div className="w-16 h-16 mx-auto mb-4 bg-violet-500/15 rounded-full flex items-center justify-center">
          <svg className="w-8 h-8 text-violet-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
        </div>
        <h2 className="text-2xl font-bold text-white mb-1">Review Lyrics</h2>
        <p className="text-zinc-400">
          {artist} — {title}
        </p>
      </div>

      <div className="bg-zinc-800/60 border border-zinc-700/50 rounded-xl p-5 space-y-4">
        {foundLyrics ? (
          <div className="bg-emerald-500/10 border border-emerald-500/20 rounded-lg px-4 py-3">
            <p className="text-emerald-400 text-sm font-medium">
              Lyrics found online! Review and edit if needed.
            </p>
            <p className="text-zinc-500 text-xs mt-1">
              Providing accurate lyrics dramatically improves transcription timing.
            </p>
          </div>
        ) : (
          <div className="bg-amber-500/10 border border-amber-500/20 rounded-lg px-4 py-3">
            <p className="text-amber-400 text-sm font-medium">
              No lyrics found automatically.
            </p>
            <p className="text-zinc-500 text-xs mt-1">
              Search below, paste lyrics manually, or skip to let the AI transcribe from audio only.
            </p>
          </div>
        )}

        {/* Search */}
        <div className="bg-zinc-900/60 border border-zinc-700/40 rounded-lg p-4 space-y-3">
          <p className="text-sm text-zinc-300 font-medium">Search for lyrics</p>
          <div className="flex gap-2">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") handleSearch() }}
              placeholder="Search by artist, song title, or lyrics..."
              className="flex-1 bg-zinc-800 border border-zinc-600 rounded-lg px-3 py-2 text-white text-sm
                         focus:outline-none focus:border-violet-500"
            />
            <button
              onClick={handleSearch}
              disabled={searching || !searchQuery.trim()}
              className="bg-violet-600 hover:bg-violet-500 text-white px-4 py-2 rounded-lg text-sm
                         font-medium transition-colors disabled:opacity-50 cursor-pointer shrink-0"
            >
              {searching ? "Searching..." : "Search"}
            </button>
          </div>

          {/* Search results */}
          {searchResults.length > 0 && (
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {searchResults.map((r, i) => (
                <button
                  key={`${r.artist_name}-${r.track_name}-${i}`}
                  onClick={() => handlePickResult(r)}
                  className="w-full text-left bg-zinc-800 hover:bg-zinc-700/80 border border-zinc-700/50
                             hover:border-violet-500/40 rounded-lg px-4 py-3 transition-colors cursor-pointer group"
                >
                  <div className="flex items-baseline justify-between gap-2">
                    <div className="min-w-0">
                      <span className="text-white text-sm font-medium truncate block">
                        {r.track_name}
                      </span>
                      <span className="text-zinc-400 text-xs">
                        {r.artist_name}
                        {r.album_name && ` — ${r.album_name}`}
                      </span>
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                      {r.duration > 0 && (
                        <span className="text-zinc-500 text-xs">{formatDuration(r.duration)}</span>
                      )}
                      <span className="text-zinc-600 text-xs">{r.source}</span>
                      <span className="text-violet-400 text-xs opacity-0 group-hover:opacity-100 transition-opacity">
                        Use
                      </span>
                    </div>
                  </div>
                  <p className="text-zinc-500 text-xs mt-1 line-clamp-2">
                    {r.lyrics.slice(0, 120)}...
                  </p>
                </button>
              ))}
            </div>
          )}

          {hasSearched && !searching && searchResults.length === 0 && (
            <p className="text-zinc-500 text-sm text-center py-2">
              No results found. Try a different search term.
            </p>
          )}
        </div>

        <div>
          <label className="block text-sm text-zinc-300 font-medium mb-2">
            Song Lyrics
          </label>
          <textarea
            value={lyrics}
            onChange={(e) => setLyrics(e.target.value)}
            placeholder={"Paste or type lyrics here...\n\nThe AI will use these as a guide for more accurate word timestamps."}
            rows={16}
            className="w-full bg-zinc-900 border border-zinc-600 rounded-lg px-4 py-3 text-white text-sm
                       leading-relaxed focus:outline-none focus:border-violet-500 resize-y font-mono"
          />
        </div>
      </div>

      <div className="flex gap-3 justify-center">
        <button
          onClick={handleSkip}
          disabled={submitting}
          className="bg-zinc-700 hover:bg-zinc-600 text-white px-6 py-3 rounded-xl
                     transition-colors disabled:opacity-50 cursor-pointer"
        >
          Skip (auto-detect)
        </button>
        <button
          onClick={handleConfirm}
          disabled={submitting || !lyrics.trim()}
          className="bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-500 hover:to-fuchsia-500
                     text-white font-medium px-8 py-3 rounded-xl transition-all disabled:opacity-50 cursor-pointer"
        >
          {submitting ? "Starting..." : "Use These Lyrics & Continue"}
        </button>
      </div>
    </div>
  )
}
