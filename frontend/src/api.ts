import { convertFileSrc, invoke, isTauri } from "@tauri-apps/api/core"
import type {
  AlignmentRun,
  AudioTrack,
  JobResult,
  LyricLanguage,
  LyricRevision,
  SongMetadata,
  WordAlignment,
} from "./types"

const BASE = "/api"
const API_TOKEN = (import.meta.env.VITE_KARAOKE_API_TOKEN as string | undefined)?.trim()

const apiHeaders = (json = false): Record<string, string> => ({
  ...(json ? { "Content-Type": "application/json" } : {}),
  ...(API_TOKEN ? { Authorization: `Bearer ${API_TOKEN}` } : {}),
})
const authQuery = API_TOKEN ? `?access_token=${encodeURIComponent(API_TOKEN)}` : ""

const invokeDesktop = async <T>(command: string, args?: Record<string, unknown>): Promise<T> => {
  try {
    return await invoke<T>(command, args)
  } catch (error) {
    throw error instanceof Error ? error : new Error(String(error))
  }
}

type ImportMetadata = { title: string; artist: string; language: string; genre: string; year: string }

export interface LocalMedia {
  path: string
  name: string
  size: number
}

export const pickLocalMedia = () => invokeDesktop<LocalMedia | null>("pick_local_media")
export const inspectLocalMedia = (path: string) => invokeDesktop<LocalMedia>("inspect_local_media", { path })
export const prepareLocalMedia = (path: string) => invokeDesktop<string>("prepare_local_media", { path })

export const registerLocalMedia = (importId: string, metadata: ImportMetadata) =>
  invokeDesktop<string>("register_native_media", { importId, metadata })

const handOffLocalMedia = async (importId: string, metadata: ImportMetadata): Promise<void> => {
  const res = await fetch(`${BASE}/desktop/import`, {
    method: "POST",
    headers: apiHeaders(true),
    body: JSON.stringify({ import_id: importId, ...metadata }),
  })
  if (!res.ok) {
    const error = await res.json().catch(() => ({}))
    throw new Error(typeof error.detail === "string" ? error.detail : "Failed to register prepared media")
  }
  await invokeDesktop("retire_native_job", { jobId: importId })
}

export const uploadFile = async (file: File, metadata: ImportMetadata): Promise<string> => {
  const form = new FormData()
  form.append("file", file)
  form.append("title", metadata.title)
  form.append("artist", metadata.artist)
  form.append("language", metadata.language)
  form.append("genre", metadata.genre)
  form.append("year", metadata.year)

  const res = await fetch(`${BASE}/upload`, { method: "POST", headers: apiHeaders(), body: form })
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: "Upload failed" }))
    throw new Error(error.detail ?? "Upload failed")
  }
  const data = await res.json()
  return data.job_id as string
}

export const startProcessing = async (jobId: string): Promise<void> => {
  if (isTauri() && await invokeDesktop<boolean>("start_native_job", { jobId, retry: false })) return
  const res = await fetch(`${BASE}/process/${jobId}/start`, { method: "POST", headers: apiHeaders() })
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: "Failed to start processing" }))
    throw new Error(error.detail ?? "Failed to start processing")
  }
}

export const getResult = async (jobId: string): Promise<JobResult> => {
  const res = await fetch(`${BASE}/jobs/${jobId}/result`, { headers: apiHeaders() })
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: "Failed to get result" }))
    throw new Error(error.detail ?? "Failed to get result")
  }
  return res.json()
}

export const getAlignment = async (jobId: string): Promise<{
  job_id: string
  status: string
  lyric_revision: LyricRevision
  word_alignments: Array<WordAlignment>
  alignment_run: AlignmentRun | null
  last_completed_alignment_run?: AlignmentRun | null
  latest_alignment_run?: AlignmentRun | null
}> => {
  const res = await fetch(`${BASE}/jobs/${jobId}/alignment`, { headers: apiHeaders() })
  if (!res.ok) throw new Error("Failed to get word alignment")
  return res.json()
}

export const updateNotes = async (
  jobId: string,
  phrases: JobResult["phrases"],
): Promise<void> => {
  const res = await fetch(`${BASE}/jobs/${jobId}/notes`, {
    method: "PUT",
    headers: apiHeaders(true),
    body: JSON.stringify({ phrases }),
  })
  if (!res.ok) throw new Error("Failed to update notes")
}

export const getAudioUrl = (
  jobId: string,
  track: AudioTrack
): string => {
  const sourceTrack = track.endsWith("_with_playback") ? track.replace("_with_playback", "") : track
  return `${BASE}/jobs/${jobId}/audio/${sourceTrack}${authQuery}`
}

export const getDownloadUrl = (jobId: string): string =>
  `${BASE}/jobs/${jobId}/download${authQuery}`

export const saveManagedZip = async (jobId: string): Promise<void> => {
  const chart = await getResult(jobId)
  if (!chart.export_ready) throw new Error("The selected lyric revision has no completed alignment to export")
  await invokeDesktop("save_managed_zip", { jobId, chart })
}

export interface LyricsSearchResult {
  track_name: string
  artist_name: string
  album_name: string
  duration: number
  lyrics: string
  source: string
}

export const confirmLyrics = async (jobId: string, lyrics: string): Promise<void> => {
  if (isTauri()) {
    const native = await invokeDesktop<{ metadata: ImportMetadata } | null>("native_job_status", { jobId })
    if (native) await handOffLocalMedia(jobId, native.metadata)
  }
  const res = await fetch(`${BASE}/jobs/${jobId}/confirm-lyrics`, {
    method: "POST",
    headers: apiHeaders(true),
    body: JSON.stringify({ lyrics }),
  })
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: "Failed to confirm lyrics" }))
    throw new Error(error.detail ?? "Failed to confirm lyrics")
  }
}

export const searchLyricsOnline = async (query: string): Promise<Array<LyricsSearchResult>> => {
  if (!query.trim()) return []
  if (isTauri()) return invokeDesktop("search_local_lyrics", { query })
  const res = await fetch(`${BASE}/lyrics/search?${new URLSearchParams({ q: query.trim() })}`, {
    headers: apiHeaders(),
  })
  if (!res.ok) throw new Error("Search failed")
  const data = await res.json()
  return data.results as Array<LyricsSearchResult>
}

export const saveLyricRevision = async (
  jobId: string,
  rawText: string,
  language: LyricLanguage,
): Promise<LyricRevision> => {
  const res = await fetch(`${BASE}/jobs/${jobId}/lyric-revisions`, {
    method: "POST",
    headers: apiHeaders(true),
    body: JSON.stringify({ raw_text: rawText, language }),
  })
  if (!res.ok) throw new Error("Failed to save lyric revision")
  const data = await res.json()
  return data.revision as LyricRevision
}

export const startAlignmentRun = async (
  jobId: string,
  lyricRevisionId: string,
  resetTokenIds: Array<string> = [],
  alignmentBackend: "forcealign" | "mms" = "forcealign",
): Promise<{ alignment_run_id: string; lyric_revision_id: string; status: string }> => {
  const res = await fetch(`${BASE}/jobs/${jobId}/alignment-runs`, {
    method: "POST",
    headers: apiHeaders(true),
    body: JSON.stringify({
      lyric_revision_id: lyricRevisionId,
      reset_token_ids: resetTokenIds,
      alignment_backend: alignmentBackend,
    }),
  })
  if (!res.ok) throw new Error("Failed to start alignment")
  return res.json()
}

export const startAlignmentComparison = async (
  jobId: string,
  lyricRevisionId?: string,
): Promise<{ status: string; alignment_comparison?: JobResult["alignment_comparison"] }> => {
  const res = await fetch(`${BASE}/jobs/${jobId}/alignment-comparison`, {
    method: "POST",
    headers: apiHeaders(true),
    body: JSON.stringify(lyricRevisionId ? { lyric_revision_id: lyricRevisionId } : {}),
  })
  if (!res.ok) throw new Error("Failed to start alignment comparison")
  return res.json()
}

export const switchAlignmentMode = async (
  jobId: string,
  mode: "forcealign" | "mms" | "canonical",
): Promise<JobResult> => {
  const res = await fetch(`${BASE}/jobs/${jobId}/alignment-mode`, {
    method: "PATCH",
    headers: apiHeaders(true),
    body: JSON.stringify({ mode }),
  })
  if (!res.ok) throw new Error("Failed to switch alignment version")
  return res.json()
}

export const updateWordAlignment = async (
  jobId: string,
  tokenId: string,
  boundaries: { start?: number; end?: number },
): Promise<WordAlignment> => {
  const res = await fetch(`${BASE}/jobs/${jobId}/word-alignments/${encodeURIComponent(tokenId)}`, {
    method: "PATCH",
    headers: apiHeaders(true),
    body: JSON.stringify(boundaries),
  })
  if (!res.ok) throw new Error("Failed to save word boundary")
  const data = await res.json()
  return data.alignment as WordAlignment
}

export const resetWordAlignment = async (
  jobId: string,
  tokenId: string,
  boundaries: Array<"start" | "end"> = ["start", "end"],
): Promise<WordAlignment> => {
  const res = await fetch(`${BASE}/jobs/${jobId}/word-alignments/${encodeURIComponent(tokenId)}/reset`, {
    method: "POST",
    headers: apiHeaders(true),
    body: JSON.stringify({ boundaries }),
  })
  if (!res.ok) throw new Error("Failed to reset word boundary")
  const data = await res.json()
  return data.alignment as WordAlignment
}

export interface ParsedFilename {
  title: string
  artist: string
  language: string
  year: string
  genre: string
}

export const parseFilename = async (filename: string): Promise<ParsedFilename> => {
  if (isTauri()) return invokeDesktop("parse_local_filename", { filename })
  const res = await fetch(`${BASE}/parse-filename`, {
    method: "POST",
    headers: apiHeaders(true),
    body: JSON.stringify({ filename }),
  })
  if (!res.ok) throw new Error("Failed to parse filename")
  return res.json()
}

export interface SongListItem {
  id: string
  kind: "managed" | "external"
  title: string
  artist: string
  filename: string
  folder?: string
  path?: string
  status: string
  message: string
  progress_percent: number
  error?: string
  failed_phase?: string
  queue_position?: number
}

interface SongLibrary {
  working_directory: string | null
  songs: Array<SongListItem>
}

interface ManagedLibrary extends SongLibrary {
  published_paths: string[]
}

let sharedDevelopmentLibrary = false

const listHttpSongs = async (): Promise<SongLibrary> => {
  const res = await fetch(`${BASE}/library`, { headers: apiHeaders() })
  if (!res.ok) throw new Error("Failed to list songs")
  return res.json()
}

const listManagedSongs = async (): Promise<ManagedLibrary> => {
  const res = await fetch(`${BASE}/library?managed_only=true`, {
    headers: apiHeaders(),
    signal: AbortSignal.timeout(1500),
  })
  if (!res.ok) throw new Error("Failed to list managed songs")
  const library: ManagedLibrary = await res.json()
  return { ...library, songs: library.songs.filter((song) => song.kind === "managed") }
}

export const listSongs = async (): Promise<SongLibrary> => {
  if (!isTauri()) return listHttpSongs()
  const [nativeLibrary, managed, nativeJobs] = await Promise.all([
    invokeDesktop<SongLibrary>("list_native_songs"),
    import.meta.env.DEV ? listManagedSongs().catch(() => undefined) : undefined,
    invokeDesktop<SongListItem[]>("list_native_jobs"),
  ])
  let library = nativeLibrary
  if (!library.working_directory && managed?.working_directory) {
    try {
      await invokeDesktop("set_native_library_directory", { directory: managed.working_directory })
      library = await invokeDesktop("list_native_songs")
    } catch {
      // A legacy selection may point to a directory that no longer exists.
    }
  }
  sharedDevelopmentLibrary = !!library.working_directory && library.working_directory === managed?.working_directory
  const published = new Set(sharedDevelopmentLibrary ? managed?.published_paths : [])
  const serverJobs = new Set(managed?.songs.map((song) => song.id))
  return {
    working_directory: library.working_directory,
    songs: [
      ...nativeJobs.filter((song) => !serverJobs.has(song.id)),
      ...(managed?.songs ?? []),
      ...library.songs.filter((song) => !published.has(song.path ?? "")),
    ],
  }
}

export const pickWorkingDirectory = (): Promise<string | null> =>
  invokeDesktop("pick_native_library_directory")

export const setWorkingDirectory = async (directory: string): Promise<string> => {
  if (isTauri()) {
    if (import.meta.env.DEV) {
      try {
        directory = await setHttpWorkingDirectory(directory, AbortSignal.timeout(1500))
      } catch {
        // Rust can save the selection without the development API.
      }
    }
    return invokeDesktop("set_native_library_directory", { directory })
  }
  return setHttpWorkingDirectory(directory)
}

const setHttpWorkingDirectory = async (directory: string, signal?: AbortSignal): Promise<string> => {
  const res = await fetch(`${BASE}/library/working-directory`, {
    method: "PUT",
    headers: apiHeaders(true),
    body: JSON.stringify({ directory }),
    signal,
  })
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: "Unable to set working directory" }))
    throw new Error(error.detail ?? "Unable to set working directory")
  }
  const data = await res.json()
  return data.directory as string
}

export interface ExternalSong {
  path: string
  filename: string
  content: string
  metadata: SongMetadata
  phrases: JobResult["phrases"]
  audio_available: boolean
  playback_url?: string
  separation_available: boolean
  stems_available: boolean
  separation_status: ExternalSongSeparationStatus["status"]
  separation_error?: string
}

export interface ExternalSongSeparationStatus {
  status: "idle" | "separating" | "done" | "error"
  error?: string
}

export const getExternalSong = async (path: string): Promise<ExternalSong> => {
  if (isTauri()) {
    type NativeSong = Pick<ExternalSong, "path" | "filename" | "content" | "metadata" | "phrases" | "audio_available"> & {
      playback_path: string | null
    }
    const [song, separation] = await Promise.all([
      invokeDesktop<NativeSong>("open_external_chart", { path }),
      sharedDevelopmentLibrary
        ? getExternalSongSeparationStatus(path, AbortSignal.timeout(1500)).catch(() => undefined)
        : undefined,
    ])
    const { playback_path: playbackPath, ...chart } = song
    return {
      ...chart,
      playback_url: playbackPath ? convertFileSrc(playbackPath) : undefined,
      separation_available: separation !== undefined,
      stems_available: separation?.status === "done",
      separation_status: separation?.status ?? "idle",
      separation_error: separation?.error,
    }
  }
  const res = await fetch(`${BASE}/library/external-song?${new URLSearchParams({ path })}`, {
    headers: apiHeaders(),
  })
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: "Failed to open song file" }))
    throw new Error(error.detail ?? "Failed to open song file")
  }
  const song: Omit<ExternalSong, "separation_available"> = await res.json()
  return { ...song, separation_available: true }
}

export const saveExternalSong = async (
  path: string,
  phrases: JobResult["phrases"],
): Promise<string> => {
  if (isTauri()) return invokeDesktop("save_external_chart", { path, phrases })
  const res = await fetch(`${BASE}/library/external-song?${new URLSearchParams({ path })}`, {
    method: "PUT",
    headers: apiHeaders(true),
    body: JSON.stringify({ phrases }),
  })
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: "Failed to save song chart" }))
    throw new Error(error.detail ?? "Failed to save song chart")
  }
  const data: { content: string } = await res.json()
  return data.content
}

export const startExternalSongSeparation = async (path: string): Promise<ExternalSongSeparationStatus> => {
  const res = await fetch(`${BASE}/library/external-song/split?${new URLSearchParams({ path })}`, {
    method: "POST",
    headers: apiHeaders(),
  })
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: "Failed to split song audio" }))
    throw new Error(error.detail ?? "Failed to split song audio")
  }
  return res.json()
}

export const getExternalSongSeparationStatus = async (
  path: string,
  signal?: AbortSignal,
): Promise<ExternalSongSeparationStatus> => {
  const res = await fetch(`${BASE}/library/external-song/split-status?${new URLSearchParams({ path })}`, {
    headers: apiHeaders(),
    signal,
  })
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: "Failed to get separation status" }))
    throw new Error(error.detail ?? "Failed to get separation status")
  }
  return res.json()
}

export const getExternalSongAudioUrl = (
  path: string,
  track: Extract<AudioTrack, "original" | "vocals" | "instrumental"> = "original",
): string => {
  const query = new URLSearchParams({ path, track })
  if (API_TOKEN) query.set("access_token", API_TOKEN)
  return `${BASE}/library/external-song/audio?${query}`
}

export const getExternalSongDownloadUrl = (path: string): string => {
  const query = new URLSearchParams({ path })
  if (API_TOKEN) query.set("access_token", API_TOKEN)
  return `${BASE}/library/external-song/download?${query}`
}

export const exportExternalSong = (path: string): Promise<void> =>
  invokeDesktop("export_external_chart", { path })

export const removeSong = async (song: SongListItem): Promise<void> => {
  if (song.kind === "external" && isTauri()) {
    return invokeDesktop("remove_external_chart", { path: song.path })
  }
  if (song.kind === "managed" && isTauri() &&
    await invokeDesktop<boolean>("remove_native_job", { jobId: song.id })) return
  const query = song.kind === "managed"
    ? new URLSearchParams({ job_id: song.id })
    : new URLSearchParams({ path: song.path ?? "" })
  const res = await fetch(`${BASE}/library/song?${query}`, {
    method: "DELETE",
    headers: apiHeaders(),
  })
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: "Failed to remove song" }))
    throw new Error(error.detail ?? "Failed to remove song")
  }
}

export const getStatus = async (
  jobId: string,
): Promise<{
  status: string
  message: string
  metadata?: { title: string; artist: string }
  error?: string
  found_lyrics?: string
  progress_percent?: number
  lyric_revision_id?: string
  failed_phase?: string
  alignment_comparison_status?: string | null
  alignment_run?: AlignmentRun
}> => {
  if (isTauri()) {
    const native = await invokeDesktop<{ status: string; message: string; found_lyrics?: string } | null>(
      "native_job_status", { jobId },
    )
    if (native) return native
  }
  const res = await fetch(`${BASE}/jobs/${jobId}/status`, { headers: apiHeaders() })
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: "Failed to get job status" }))
    throw new Error(error.detail ?? "Failed to get job status")
  }
  return res.json()
}

export const retryJob = async (jobId: string): Promise<void> => {
  if (isTauri() && await invokeDesktop<boolean>("start_native_job", { jobId, retry: true })) return
  const res = await fetch(`${BASE}/jobs/${jobId}/retry`, { method: "POST", headers: apiHeaders() })
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: "Failed to retry processing" }))
    throw new Error(error.detail ?? "Failed to retry processing")
  }
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
  if (isTauri()) return invokeDesktop("search_local_youtube", { query })
  const res = await fetch(`${BASE}/youtube/search`, {
    method: "POST",
    headers: apiHeaders(true),
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
  if (isTauri()) return invokeDesktop("register_native_download", { url, metadata })
  const res = await fetch(`${BASE}/youtube/download`, {
    method: "POST",
    headers: apiHeaders(true),
    body: JSON.stringify({ url, ...metadata }),
  })
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: "Download failed" }))
    throw new Error(error.detail ?? "Download failed")
  }
  const data = await res.json()
  return data.job_id as string
}
