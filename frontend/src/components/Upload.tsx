import { createSignal, onCleanup } from "solid-js"
import { createEffect } from "../solidCompat"
import { uploadFile, parseFilename, searchYouTube, downloadYouTube, getStatus, type YouTubeSearchResult } from "../api"

interface UploadProps {
  onUploaded: (jobId: string, meta?: { artist: string; title: string }) => void
}

const ACCEPTED_TYPES = [
  "audio/mpeg", "audio/wav", "audio/ogg", "audio/flac", "audio/mp4",
  "video/mp4", "video/webm", "video/x-matroska", "video/avi",
]

export const Upload = ({ onUploaded }: UploadProps) => {
  const [file, setFile] = createSignal<File | null>(null)
  const [title, setTitle] = createSignal("")
  const [artist, setArtist] = createSignal("")
  const [language, setLanguage] = createSignal<"en" | "pl">("en")
  const [uploading, setUploading] = createSignal(false)
  const [parsing, setParsing] = createSignal(false)
  const [dragOver, setDragOver] = createSignal(false)
  const [error, setError] = createSignal("")
  const [useYouTube, setUseYouTube] = createSignal(false)
  const [youtubeQuery, setYoutubeQuery] = createSignal("")
  const [youtubeSearching, setYoutubeSearching] = createSignal(false)
  const [youtubeResults, setYoutubeResults] = createSignal<Array<YouTubeSearchResult>>([])
  const [selectedVideo, setSelectedVideo] = createSignal<YouTubeSearchResult | null>(null)
  const [previewVideoId, setPreviewVideoId] = createSignal<string | null>(null)
  const [downloading, setDownloading] = createSignal(false)
  const [downloadProgress, setDownloadProgress] = createSignal("")
  const [downloadPercent, setDownloadPercent] = createSignal(0)
  const [downloadJobId, setDownloadJobId] = createSignal<string | null>(null)
  let inputRef: HTMLInputElement | undefined

  const handleFile = (f: File) => {
    setFile(f)
    setError("")
    setParsing(true)
    parseFilename(f.name)
      .then((meta) => {
        if (meta.title) setTitle(meta.title)
        if (meta.artist) setArtist(meta.artist)
        if (meta.language) setLanguage(meta.language.toLowerCase().startsWith("pl") || meta.language.toLowerCase().startsWith("pol") ? "pl" : "en")
      })
      .catch(() => {
        // Fallback: simple split on " - "
        const name = f.name.replace(/\.[^.]+$/, "")
        const parts = name.split(" - ")
        if (parts.length >= 2) {
          setArtist(parts[0].trim())
          setTitle(parts.slice(1).join(" - ").trim())
        } else {
          setTitle(name)
        }
      })
      .finally(() => setParsing(false))
  }

  const handleDrop = (e: DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const f = e.dataTransfer?.files[0]
    if (f) handleFile(f)
  }

  const handleYouTubeSearch = async () => {
    if (!youtubeQuery().trim()) return
    setYoutubeSearching(true)
    setError("")
    try {
      const results = await searchYouTube(youtubeQuery())
      setYoutubeResults(results)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed")
    } finally {
      setYoutubeSearching(false)
    }
  }

  const handleSelectVideo = (video: YouTubeSearchResult) => {
    setSelectedVideo(video)
    const cleanTitle = (value: string) => {
      let cleaned = value
      cleaned = cleaned.replace(/\s*\(.*?(?:official\s+(?:music\s+)?(?:video|audio)|official\s+mv|lyrics?|lyric\s+video|visualizer|audio|remaster(?:ed)?|hq|hd|4k|8k)[^)]*\)/gi, "")
      cleaned = cleaned.replace(/\s*\[.*?(?:official\s+(?:music\s+)?(?:video|audio)|official\s+mv|lyrics?|lyric\s+video|visualizer|audio|remaster(?:ed)?|hq|hd|4k|8k).*?\]/gi, "")
      cleaned = cleaned.replace(/\s+(?:official\s+(?:music\s+)?(?:video|audio)|official\s+mv|lyrics?|lyric\s+video|visualizer|audio|remaster(?:ed)?|hq|hd|4k|8k)\s*$/gi, "")
      cleaned = cleaned.replace(/\s+[|].*$/, "")
      cleaned = cleaned.replace(/\s*[-–—]\s*(?:topic|vevo)\s*$/gi, "")
      cleaned = cleaned.replace(/\s+/g, " ")
      return cleaned.replace(/^[\s._-]+|[\s._-]+$/g, "")
    }
    const parts = video.title.split(" - ")
    if (parts.length >= 2) {
      setArtist(parts[0].trim())
      setTitle(cleanTitle(parts.slice(1).join(" - ").trim()))
    } else {
      setTitle(cleanTitle(video.title))
      setArtist(video.channel.replace(/\s*-\s*Topic$/i, "").trim())
    }
  }

  // Poll download progress
  createEffect(() => {
    const jobId = downloadJobId()
    if (!jobId || !downloading()) return

    let cancelled = false
    let timerId: number

    const pollProgress = async () => {
      if (cancelled) return
      try {
        const data = await getStatus(jobId)
        if (cancelled) return
        
        setDownloadProgress(data.message)
        if (data.progress_percent !== undefined) {
          setDownloadPercent(data.progress_percent)
        }
        
        if (data.status === "pending" || data.status === "extracting_audio") {
          // Download complete, move to processing
          setDownloading(false)
          setDownloadProgress("")
          setDownloadPercent(0)
          setDownloadJobId(null)
          onUploaded(jobId, { artist: artist(), title: title() })
          return
        }
        if (data.status === "error") {
          setError(data.error ?? data.message ?? "Download failed")
          setDownloading(false)
          setDownloadProgress("")
          setDownloadPercent(0)
          setDownloadJobId(null)
          return
        }
      } catch {
        // Network error, continue polling
      }
      if (!cancelled) {
        timerId = window.setTimeout(pollProgress, 500)
      }
    }

    timerId = window.setTimeout(pollProgress, 500)
    onCleanup(() => {
      cancelled = true
      if (timerId) window.clearTimeout(timerId)
    })
  }, () => undefined)

  const handleSubmit = async (e: SubmitEvent) => {
    e.preventDefault()
    
    if (useYouTube()) {
      if (!selectedVideo()) {
        setError("Please select a video")
        return
      }
      
      setDownloading(true)
      setDownloadProgress("Starting download...")
      setDownloadPercent(0)
      setError("")
      
      try {
        const jobId = await downloadYouTube(selectedVideo()!.url, { title: title(), artist: artist(), language: language(), genre: "", year: "" })
        setDownloadJobId(jobId)
        // Progress polling will handle completion.
      } catch (err) {
        setError(err instanceof Error ? err.message : "Download failed")
        setDownloading(false)
        setDownloadProgress("")
        setDownloadPercent(0)
        setDownloadJobId(null)
      }
    } else {
      if (!file()) return

      setUploading(true)
      setError("")

      try {
        const jobId = await uploadFile(file()!, { title: title(), artist: artist(), language: language(), genre: "", year: "" })
        onUploaded(jobId, { artist: artist(), title: title() })
      } catch (err) {
        setError(err instanceof Error ? err.message : "Upload failed")
        setUploading(false)
      }
    }
  }

  const formatDuration = (seconds: number | null): string => {
    if (!seconds) return ""
    const m = Math.floor(seconds / 60)
    const s = seconds % 60
    return `${m}:${s.toString().padStart(2, "0")}`
  }

  return (
    <div class="max-w-2xl mx-auto">
      <div class="text-center mb-8">
        <h1 class="text-4xl font-bold text-white mb-2">Karaoke Creator</h1>
        <p class="text-zinc-400 text-lg">
          Create UltraStar Deluxe songs from any audio or video file
        </p>
      </div>

      <form onSubmit={handleSubmit} class="space-y-6">
        {/* Upload method toggle */}
        <div class="flex gap-2 border-b border-zinc-700 pb-4">
          <button
            type="button"
            onClick={() => {
              setUseYouTube(false)
              setSelectedVideo(null)
              setYoutubeResults([])
            }}
            class={`px-4 py-2 rounded-lg transition-colors ${
              !useYouTube()
                ? "bg-violet-600 text-white"
                : "bg-zinc-800 text-zinc-400 hover:bg-zinc-700"
            }`}
          >
            Upload File
          </button>
          <button
            type="button"
            onClick={() => {
              setUseYouTube(true)
              setFile(null)
            }}
            class={`px-4 py-2 rounded-lg transition-colors ${
              useYouTube()
                ? "bg-violet-600 text-white"
                : "bg-zinc-800 text-zinc-400 hover:bg-zinc-700"
            }`}
          >
            YouTube
          </button>
        </div>

        {useYouTube() ? (
          <>
            {/* YouTube search */}
            <div class="bg-zinc-800/60 border border-zinc-700/50 rounded-xl p-5 space-y-4">
              <div class="flex gap-2">
                <input
                  type="text"
                  value={youtubeQuery()}
                  onInput={(e) => setYoutubeQuery(e.currentTarget.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault()
                      handleYouTubeSearch()
                    }
                  }}
                  placeholder="Search YouTube for a video..."
                  class="flex-1 bg-zinc-900 border border-zinc-600 rounded-lg px-3 py-2 text-white
                             placeholder-zinc-500 focus:outline-none focus:border-violet-500"
                />
                <button
                  type="button"
                  onClick={handleYouTubeSearch}
                  disabled={youtubeSearching() || !youtubeQuery().trim()}
                  class="bg-violet-600 hover:bg-violet-500 disabled:bg-zinc-700 disabled:text-zinc-500
                             text-white px-6 py-2 rounded-lg transition-colors shrink-0"
                >
                  {youtubeSearching() ? "Searching..." : "Search"}
                </button>
              </div>

              {/* Search results */}
              {youtubeResults().length > 0 && (
                <div class="space-y-3 max-h-96 overflow-y-auto">
                  {youtubeResults().map((video) => (
                    <div
                      class={`w-full bg-zinc-900 hover:bg-zinc-800 border rounded-lg overflow-hidden transition-colors ${
                        selectedVideo()?.id === video.id
                          ? "border-violet-500 bg-violet-500/10"
                          : "border-zinc-700 hover:border-zinc-600"
                      }`}
                    >
                      <div class="flex gap-3 p-3">
                        {/* Thumbnail with play button */}
                        <div class="relative shrink-0 group">
                          {video.thumbnail ? (
                            <>
                              <img
                                src={video.thumbnail}
                                alt={video.title}
                                class="w-32 h-20 object-cover rounded"
                              />
                              <button
                                type="button"
                                onClick={(e) => {
                                  e.stopPropagation()
                                  setPreviewVideoId(previewVideoId() === video.id ? null : video.id)
                                }}
                                class="absolute inset-0 flex items-center justify-center bg-black/40 hover:bg-black/60 transition-colors rounded"
                                aria-label="Preview video"
                              >
                                <svg class="w-8 h-8 text-white" fill="currentColor" viewBox="0 0 24 24">
                                  <path d="M8 5v14l11-7z" />
                                </svg>
                              </button>
                            </>
                          ) : (
                            <div class="w-32 h-20 bg-zinc-800 rounded flex items-center justify-center">
                              <svg class="w-8 h-8 text-zinc-600" fill="currentColor" viewBox="0 0 24 24">
                                <path d="M8 5v14l11-7z" />
                              </svg>
                            </div>
                          )}
                        </div>
                        
                        {/* Video info */}
                        <div class="flex-1 min-w-0">
                          <button
                            type="button"
                            onClick={() => handleSelectVideo(video)}
                            class="w-full text-left"
                          >
                            <div class="text-white font-medium text-sm truncate">{video.title}</div>
                            <div class="text-zinc-400 text-xs mt-1">{video.channel}</div>
                            {video.duration && (
                              <div class="text-zinc-500 text-xs mt-1">{formatDuration(video.duration)}</div>
                            )}
                          </button>
                        </div>
                      </div>
                      
                      {/* Preview player */}
                      {previewVideoId() === video.id && (
                        <div class="border-t border-zinc-700 p-3 bg-zinc-950">
                          <div class="relative w-full" style={{ "padding-bottom": "56.25%" }}>
                            <iframe
                              src={`https://www.youtube.com/embed/${video.id}?autoplay=1&rel=0`}
                              class="absolute top-0 left-0 w-full h-full rounded"
                              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                              allowfullscreen
                              title={video.title}
                            />
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </>
        ) : (
          <>
            {/* Drop zone */}
            <div
              onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              onClick={() => inputRef?.click()}
              class={`
                border-2 border-dashed rounded-xl p-12 text-center cursor-pointer
                transition-all duration-200
                ${dragOver()
                  ? "border-violet-400 bg-violet-500/10"
                  : file()
                    ? "border-emerald-500/50 bg-emerald-500/5"
                    : "border-zinc-600 hover:border-zinc-500 bg-zinc-800/50"
                }
              `}
            >
              <input
                ref={(el) => { inputRef = el }}
                type="file"
                accept={ACCEPTED_TYPES.join(",")}
                class="hidden"
                data-testid="file-input"
                onChange={(e) => {
                  const f = e.currentTarget.files?.[0]
                  if (f) handleFile(f)
                }}
              />
              {file() ? (
                <div>
                  <div class="text-emerald-400 text-lg font-medium mb-1">{file()!.name}</div>
                  <div class="text-zinc-500 text-sm">
                    {(file()!.size / 1024 / 1024).toFixed(1)} MB — click or drop to replace
                  </div>
                </div>
              ) : (
                <div>
                  <div class="text-zinc-300 text-lg mb-1">Drop audio or video file here</div>
                  <div class="text-zinc-500 text-sm">or click to browse — MP3, WAV, MP4, MKV, etc.</div>
                </div>
              )}
            </div>
          </>
        )}

        {/* Metadata fields */}
        <div class="grid grid-cols-2 gap-4">
          <div class="col-span-2 sm:col-span-1">
            <label class="block text-sm text-zinc-400 mb-1">Title *</label>
            <input
              type="text"
              value={title()}
              onInput={(e) => setTitle(e.currentTarget.value)}
              required
              class="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-white
                         placeholder-zinc-500 focus:outline-none focus:border-violet-500 transition-colors"
              placeholder="Song title"
            />
          </div>
          <div class="col-span-2 sm:col-span-1">
            <label class="block text-sm text-zinc-400 mb-1">Artist</label>
            <input
              type="text"
              value={artist()}
              onInput={(e) => setArtist(e.currentTarget.value)}
              class="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-white
                         placeholder-zinc-500 focus:outline-none focus:border-violet-500 transition-colors"
              placeholder="Artist name"
            />
          </div>
          <div>
            <label class="block text-sm text-zinc-400 mb-1">Language</label>
            <select
              value={language()}
              onChange={(e) => setLanguage(e.currentTarget.value === "pl" ? "pl" : "en")}
              class="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-white
                         focus:outline-none focus:border-violet-500 transition-colors"
            >
              <option value="en">English</option>
              <option value="pl">Polski</option>
            </select>
          </div>
        </div>

        {parsing() && (
          <div class="bg-violet-500/10 border border-violet-500/30 rounded-lg p-3 text-violet-400 text-sm flex items-center gap-2">
            <svg class="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="3" stroke-dasharray="31.4 31.4" />
            </svg>
            Detecting metadata from filename...
          </div>
        )}

        {error() && (
          <div class="bg-red-500/10 border border-red-500/30 rounded-lg p-3 text-red-400 text-sm">
            {error()}
          </div>
        )}

        {downloading() && downloadProgress() && (
          <div class="bg-violet-500/10 border border-violet-500/30 rounded-lg p-4 space-y-2">
            <div class="flex items-center justify-between text-sm">
              <span class="text-violet-400">{downloadProgress()}</span>
              <span class="text-violet-300 font-medium">{downloadPercent().toFixed(0)}%</span>
            </div>
            <div class="w-full bg-zinc-700 rounded-full h-2 overflow-hidden">
              <div
                class="bg-gradient-to-r from-violet-500 to-fuchsia-500 h-full transition-all duration-300 ease-out"
                style={{ width: `${downloadPercent()}%` }}
              />
            </div>
          </div>
        )}

        <button
          type="submit"
          disabled={
            (useYouTube() ? !selectedVideo() : !file()) ||
            uploading() ||
            downloading() ||
            parsing() ||
            youtubeSearching()
          }
          class="w-full bg-violet-600 hover:bg-violet-500 disabled:bg-zinc-700 disabled:text-zinc-500
                     text-white font-medium py-3 px-6 rounded-xl transition-colors duration-200
                     cursor-pointer disabled:cursor-not-allowed"
        >
          {downloading()
            ? "Downloading..."
            : uploading()
              ? "Uploading..."
              : useYouTube()
                ? "Download & Create Karaoke"
                : "Upload & Create Karaoke"}
        </button>
      </form>
    </div>
  )
}
