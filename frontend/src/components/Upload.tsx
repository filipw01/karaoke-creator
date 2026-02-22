import { useCallback, useEffect, useRef, useState } from "react"
import { uploadFile, parseFilename, searchYouTube, downloadYouTube, getStatus, type YouTubeSearchResult } from "../api"

interface UploadProps {
  onUploaded: (jobId: string, meta?: { artist: string; title: string }) => void
}

const ACCEPTED_TYPES = [
  "audio/mpeg", "audio/wav", "audio/ogg", "audio/flac", "audio/mp4",
  "video/mp4", "video/webm", "video/x-matroska", "video/avi",
]

export const Upload = ({ onUploaded }: UploadProps) => {
  const [file, setFile] = useState<File | null>(null)
  const [title, setTitle] = useState("")
  const [artist, setArtist] = useState("")
  const [language, setLanguage] = useState("")
  const [genre, setGenre] = useState("")
  const [year, setYear] = useState("")
  const [uploading, setUploading] = useState(false)
  const [parsing, setParsing] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const [error, setError] = useState("")
  const [useYouTube, setUseYouTube] = useState(false)
  const [youtubeQuery, setYoutubeQuery] = useState("")
  const [youtubeSearching, setYoutubeSearching] = useState(false)
  const [youtubeResults, setYoutubeResults] = useState<Array<YouTubeSearchResult>>([])
  const [selectedVideo, setSelectedVideo] = useState<YouTubeSearchResult | null>(null)
  const [previewVideoId, setPreviewVideoId] = useState<string | null>(null)
  const [downloading, setDownloading] = useState(false)
  const [downloadProgress, setDownloadProgress] = useState("")
  const [downloadPercent, setDownloadPercent] = useState(0)
  const [downloadJobId, setDownloadJobId] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFile = useCallback((f: File) => {
    setFile(f)
    setError("")
    setParsing(true)
    parseFilename(f.name)
      .then((meta) => {
        if (meta.title) setTitle(meta.title)
        if (meta.artist) setArtist(meta.artist)
        if (meta.language) setLanguage(meta.language)
        if (meta.genre) setGenre(meta.genre)
        if (meta.year) setYear(meta.year)
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
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const f = e.dataTransfer.files[0]
    if (f) handleFile(f)
  }, [handleFile])

  const handleYouTubeSearch = useCallback(async () => {
    if (!youtubeQuery.trim()) return
    setYoutubeSearching(true)
    setError("")
    try {
      const results = await searchYouTube(youtubeQuery)
      setYoutubeResults(results)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed")
    } finally {
      setYoutubeSearching(false)
    }
  }, [youtubeQuery])

  const handleSelectVideo = useCallback((video: YouTubeSearchResult) => {
    setSelectedVideo(video)
    // Try to extract artist/title from video title
    const parts = video.title.split(" - ")
    if (parts.length >= 2) {
      setArtist(parts[0].trim())
      setTitle(parts.slice(1).join(" - ").trim())
    } else {
      setTitle(video.title)
      setArtist(video.channel)
    }
  }, [])

  // Poll download progress
  useEffect(() => {
    if (!downloadJobId || !downloading) return

    let cancelled = false
    let timerId: number

    const pollProgress = async () => {
      if (cancelled) return
      try {
        const data = await getStatus(downloadJobId)
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
          onUploaded(downloadJobId, { artist, title })
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
    return () => {
      cancelled = true
      if (timerId) window.clearTimeout(timerId)
    }
  }, [downloadJobId, downloading, artist, title, onUploaded])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (useYouTube) {
      if (!selectedVideo) {
        setError("Please select a video")
        return
      }
      
      setDownloading(true)
      setDownloadProgress("Starting download...")
      setDownloadPercent(0)
      setError("")
      
      try {
        const jobId = await downloadYouTube(selectedVideo.url, { title, artist, language, genre, year })
        setDownloadJobId(jobId)
        // Progress polling will handle completion via useEffect
      } catch (err) {
        setError(err instanceof Error ? err.message : "Download failed")
        setDownloading(false)
        setDownloadProgress("")
        setDownloadPercent(0)
        setDownloadJobId(null)
      }
    } else {
      if (!file) return

      setUploading(true)
      setError("")

      try {
        const jobId = await uploadFile(file, { title, artist, language, genre, year })
        onUploaded(jobId, { artist, title })
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
    <div className="max-w-2xl mx-auto">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold text-white mb-2">Karaoke Creator</h1>
        <p className="text-zinc-400 text-lg">
          Create UltraStar Deluxe songs from any audio or video file
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Upload method toggle */}
        <div className="flex gap-2 border-b border-zinc-700 pb-4">
          <button
            type="button"
            onClick={() => {
              setUseYouTube(false)
              setSelectedVideo(null)
              setYoutubeResults([])
            }}
            className={`px-4 py-2 rounded-lg transition-colors ${
              !useYouTube
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
            className={`px-4 py-2 rounded-lg transition-colors ${
              useYouTube
                ? "bg-violet-600 text-white"
                : "bg-zinc-800 text-zinc-400 hover:bg-zinc-700"
            }`}
          >
            YouTube
          </button>
        </div>

        {useYouTube ? (
          <>
            {/* YouTube search */}
            <div className="bg-zinc-800/60 border border-zinc-700/50 rounded-xl p-5 space-y-4">
              <div className="flex gap-2">
                <input
                  type="text"
                  value={youtubeQuery}
                  onChange={(e) => setYoutubeQuery(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault()
                      handleYouTubeSearch()
                    }
                  }}
                  placeholder="Search YouTube for a video..."
                  className="flex-1 bg-zinc-900 border border-zinc-600 rounded-lg px-3 py-2 text-white
                             placeholder-zinc-500 focus:outline-none focus:border-violet-500"
                />
                <button
                  type="button"
                  onClick={handleYouTubeSearch}
                  disabled={youtubeSearching || !youtubeQuery.trim()}
                  className="bg-violet-600 hover:bg-violet-500 disabled:bg-zinc-700 disabled:text-zinc-500
                             text-white px-6 py-2 rounded-lg transition-colors shrink-0"
                >
                  {youtubeSearching ? "Searching..." : "Search"}
                </button>
              </div>

              {/* Search results */}
              {youtubeResults.length > 0 && (
                <div className="space-y-3 max-h-96 overflow-y-auto">
                  {youtubeResults.map((video) => (
                    <div
                      key={video.id}
                      className={`w-full bg-zinc-900 hover:bg-zinc-800 border rounded-lg overflow-hidden transition-colors ${
                        selectedVideo?.id === video.id
                          ? "border-violet-500 bg-violet-500/10"
                          : "border-zinc-700 hover:border-zinc-600"
                      }`}
                    >
                      <div className="flex gap-3 p-3">
                        {/* Thumbnail with play button */}
                        <div className="relative shrink-0 group">
                          {video.thumbnail ? (
                            <>
                              <img
                                src={video.thumbnail}
                                alt={video.title}
                                className="w-32 h-20 object-cover rounded"
                              />
                              <button
                                type="button"
                                onClick={(e) => {
                                  e.stopPropagation()
                                  setPreviewVideoId(previewVideoId === video.id ? null : video.id)
                                }}
                                className="absolute inset-0 flex items-center justify-center bg-black/40 hover:bg-black/60 transition-colors rounded"
                                aria-label="Preview video"
                              >
                                <svg className="w-8 h-8 text-white" fill="currentColor" viewBox="0 0 24 24">
                                  <path d="M8 5v14l11-7z" />
                                </svg>
                              </button>
                            </>
                          ) : (
                            <div className="w-32 h-20 bg-zinc-800 rounded flex items-center justify-center">
                              <svg className="w-8 h-8 text-zinc-600" fill="currentColor" viewBox="0 0 24 24">
                                <path d="M8 5v14l11-7z" />
                              </svg>
                            </div>
                          )}
                        </div>
                        
                        {/* Video info */}
                        <div className="flex-1 min-w-0">
                          <button
                            type="button"
                            onClick={() => handleSelectVideo(video)}
                            className="w-full text-left"
                          >
                            <div className="text-white font-medium text-sm truncate">{video.title}</div>
                            <div className="text-zinc-400 text-xs mt-1">{video.channel}</div>
                            {video.duration && (
                              <div className="text-zinc-500 text-xs mt-1">{formatDuration(video.duration)}</div>
                            )}
                          </button>
                        </div>
                      </div>
                      
                      {/* Preview player */}
                      {previewVideoId === video.id && (
                        <div className="border-t border-zinc-700 p-3 bg-zinc-950">
                          <div className="relative w-full" style={{ paddingBottom: "56.25%" }}>
                            <iframe
                              src={`https://www.youtube.com/embed/${video.id}?autoplay=1&rel=0`}
                              className="absolute top-0 left-0 w-full h-full rounded"
                              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                              allowFullScreen
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
              onClick={() => inputRef.current?.click()}
              className={`
                border-2 border-dashed rounded-xl p-12 text-center cursor-pointer
                transition-all duration-200
                ${dragOver
                  ? "border-violet-400 bg-violet-500/10"
                  : file
                    ? "border-emerald-500/50 bg-emerald-500/5"
                    : "border-zinc-600 hover:border-zinc-500 bg-zinc-800/50"
                }
              `}
            >
              <input
                ref={inputRef}
                type="file"
                accept={ACCEPTED_TYPES.join(",")}
                className="hidden"
                data-testid="file-input"
                onChange={(e) => {
                  const f = e.target.files?.[0]
                  if (f) handleFile(f)
                }}
              />
              {file ? (
                <div>
                  <div className="text-emerald-400 text-lg font-medium mb-1">{file.name}</div>
                  <div className="text-zinc-500 text-sm">
                    {(file.size / 1024 / 1024).toFixed(1)} MB — click or drop to replace
                  </div>
                </div>
              ) : (
                <div>
                  <div className="text-zinc-300 text-lg mb-1">Drop audio or video file here</div>
                  <div className="text-zinc-500 text-sm">or click to browse — MP3, WAV, MP4, MKV, etc.</div>
                </div>
              )}
            </div>
          </>
        )}

        {/* Metadata fields */}
        <div className="grid grid-cols-2 gap-4">
          <div className="col-span-2 sm:col-span-1">
            <label className="block text-sm text-zinc-400 mb-1">Title *</label>
            <input
              type="text"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              required
              className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-white
                         placeholder-zinc-500 focus:outline-none focus:border-violet-500 transition-colors"
              placeholder="Song title"
            />
          </div>
          <div className="col-span-2 sm:col-span-1">
            <label className="block text-sm text-zinc-400 mb-1">Artist</label>
            <input
              type="text"
              value={artist}
              onChange={(e) => setArtist(e.target.value)}
              className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-white
                         placeholder-zinc-500 focus:outline-none focus:border-violet-500 transition-colors"
              placeholder="Artist name"
            />
          </div>
          <div>
            <label className="block text-sm text-zinc-400 mb-1">Language</label>
            <input
              type="text"
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-white
                         placeholder-zinc-500 focus:outline-none focus:border-violet-500 transition-colors"
              placeholder="English"
            />
          </div>
          <div>
            <label className="block text-sm text-zinc-400 mb-1">Genre</label>
            <input
              type="text"
              value={genre}
              onChange={(e) => setGenre(e.target.value)}
              className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-white
                         placeholder-zinc-500 focus:outline-none focus:border-violet-500 transition-colors"
              placeholder="Pop"
            />
          </div>
          <div>
            <label className="block text-sm text-zinc-400 mb-1">Year</label>
            <input
              type="text"
              value={year}
              onChange={(e) => setYear(e.target.value)}
              className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-white
                         placeholder-zinc-500 focus:outline-none focus:border-violet-500 transition-colors"
              placeholder="2024"
            />
          </div>
        </div>

        {parsing && (
          <div className="bg-violet-500/10 border border-violet-500/30 rounded-lg p-3 text-violet-400 text-sm flex items-center gap-2">
            <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeDasharray="31.4 31.4" />
            </svg>
            Detecting metadata from filename...
          </div>
        )}

        {error && (
          <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3 text-red-400 text-sm">
            {error}
          </div>
        )}

        {downloading && downloadProgress && (
          <div className="bg-violet-500/10 border border-violet-500/30 rounded-lg p-4 space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span className="text-violet-400">{downloadProgress}</span>
              <span className="text-violet-300 font-medium">{downloadPercent.toFixed(0)}%</span>
            </div>
            <div className="w-full bg-zinc-700 rounded-full h-2 overflow-hidden">
              <div
                className="bg-gradient-to-r from-violet-500 to-fuchsia-500 h-full transition-all duration-300 ease-out"
                style={{ width: `${downloadPercent}%` }}
              />
            </div>
          </div>
        )}

        <button
          type="submit"
          disabled={
            (useYouTube ? !selectedVideo : !file) ||
            uploading ||
            downloading ||
            parsing ||
            youtubeSearching
          }
          className="w-full bg-violet-600 hover:bg-violet-500 disabled:bg-zinc-700 disabled:text-zinc-500
                     text-white font-medium py-3 px-6 rounded-xl transition-colors duration-200
                     cursor-pointer disabled:cursor-not-allowed"
        >
          {downloading
            ? "Downloading..."
            : uploading
              ? "Uploading..."
              : useYouTube
                ? "Download & Create Karaoke"
                : "Upload & Create Karaoke"}
        </button>
      </form>
    </div>
  )
}
