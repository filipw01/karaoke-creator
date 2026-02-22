import type { JobResult, LyricsParams } from "./types"

const BASE = "/api"

export const uploadFile = async (
  file: File,
  metadata: { title: string; artist: string; language: string; genre: string; year: string },
): Promise<string> => {
  const form = new FormData()
  form.append("file", file)
  form.append("title", metadata.title)
  form.append("artist", metadata.artist)
  form.append("language", metadata.language)
  form.append("genre", metadata.genre)
  form.append("year", metadata.year)

  const res = await fetch(`${BASE}/upload`, { method: "POST", body: form })
  if (!res.ok) throw new Error("Upload failed")
  const data = await res.json()
  return data.job_id as string
}

export const startProcessing = async (jobId: string): Promise<void> => {
  const res = await fetch(`${BASE}/process/${jobId}/start`, { method: "POST" })
  if (!res.ok) throw new Error("Failed to start processing")
}

export const getResult = async (jobId: string): Promise<JobResult> => {
  const res = await fetch(`${BASE}/jobs/${jobId}/result`)
  if (!res.ok) throw new Error("Failed to get result")
  return res.json()
}

export const updateNotes = async (
  jobId: string,
  phrases: JobResult["phrases"],
): Promise<void> => {
  const res = await fetch(`${BASE}/jobs/${jobId}/notes`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ phrases }),
  })
  if (!res.ok) throw new Error("Failed to update notes")
}

export const getAudioUrl = (jobId: string, track: "original" | "vocals" | "instrumental"): string =>
  `${BASE}/jobs/${jobId}/audio/${track}`

export const getDownloadUrl = (jobId: string, alignmentMode?: string): string => {
  const base = `${BASE}/jobs/${jobId}/download`
  if (alignmentMode) return `${base}?alignment_mode=${encodeURIComponent(alignmentMode)}`
  return base
}

export const regenerateLyrics = async (jobId: string, params: Partial<LyricsParams>): Promise<JobResult> => {
  const res = await fetch(`${BASE}/jobs/${jobId}/regenerate-lyrics`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  })
  if (!res.ok) throw new Error("Failed to regenerate lyrics")
  return res.json()
}

export interface LyricsSearchResult {
  track_name: string
  artist_name: string
  album_name: string
  duration: number
  lyrics: string
  source: string
}

export const searchLyricsOnline = async (query: string): Promise<Array<LyricsSearchResult>> => {
  if (!query.trim()) return []
  const res = await fetch(`${BASE}/lyrics/search?${new URLSearchParams({ q: query.trim() })}`)
  if (!res.ok) throw new Error("Search failed")
  const data = await res.json()
  return data.results
}

export interface ParsedFilename {
  title: string
  artist: string
  language: string
  year: string
  genre: string
}

export const parseFilename = async (filename: string): Promise<ParsedFilename> => {
  const res = await fetch(`${BASE}/parse-filename`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename }),
  })
  if (!res.ok) throw new Error("Failed to parse filename")
  return res.json()
}

export const confirmLyrics = async (jobId: string, lyrics: string): Promise<void> => {
  const res = await fetch(`${BASE}/jobs/${jobId}/confirm-lyrics`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ lyrics }),
  })
  if (!res.ok) throw new Error("Failed to confirm lyrics")
}

export const getStatus = async (
  jobId: string,
): Promise<{ status: string; message: string; error?: string; found_lyrics?: string; progress_percent?: number }> => {
  const res = await fetch(`${BASE}/jobs/${jobId}/status`)
  if (!res.ok) throw new Error("Failed to get status")
  return res.json()
}

export interface YouTubeSearchResult {
  id: string
  title: string
  channel: string
  duration: number | null
  thumbnail: string | null
  url: string
}

export const searchYouTube = async (query: string): Promise<Array<YouTubeSearchResult>> => {
  if (!query.trim()) return []
  const res = await fetch(`${BASE}/youtube/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query: query.trim() }),
  })
  if (!res.ok) throw new Error("YouTube search failed")
  const data = await res.json()
  return data.results
}

export const downloadYouTube = async (
  url: string,
  metadata: { title: string; artist: string; language: string; genre: string; year: string },
): Promise<string> => {
  const res = await fetch(`${BASE}/youtube/download`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, ...metadata }),
  })
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: "Download failed" }))
    throw new Error(error.detail ?? "Download failed")
  }
  const data = await res.json()
  return data.job_id as string
}
