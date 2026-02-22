import { useCallback, useState } from "react"
import { Routes, Route, useNavigate, useLocation } from "react-router-dom"
import type { AppStep } from "./types"
import { Upload } from "./components/Upload"
import { Processing } from "./components/Processing"
import { LyricsReview } from "./components/LyricsReview"
import { EditorPage } from "./pages/EditorPage"
import { ExportPage } from "./pages/ExportPage"

const STEP_LABELS: Array<{ key: AppStep; label: string }> = [
  { key: "upload", label: "Upload" },
  { key: "processing", label: "Processing" },
  { key: "lyrics", label: "Lyrics" },
  { key: "transcribing", label: "Transcribing" },
  { key: "editor", label: "Editor" },
  { key: "export", label: "Export" },
]

const UploadFlow = () => {
  const navigate = useNavigate()
  const [step, setStep] = useState<AppStep>("upload")
  const [jobId, setJobId] = useState("")
  const [error, setError] = useState("")
  const [foundLyrics, setFoundLyrics] = useState("")
  const [songMeta, setSongMeta] = useState({ artist: "", title: "" })

  const handleUploaded = useCallback((id: string, meta?: { artist: string; title: string }) => {
    setJobId(id)
    setStep("processing")
    setError("")
    if (meta) setSongMeta(meta)
  }, [])

  const handleAwaitingLyrics = useCallback((lyrics: string) => {
    setFoundLyrics(lyrics)
    setStep("lyrics")
  }, [])

  const handleLyricsConfirmed = useCallback(() => {
    setStep("transcribing")
  }, [])

  const handleDone = useCallback(async () => {
    navigate(`/editor/${jobId}`)
  }, [jobId, navigate])

  const handleError = useCallback((err: string) => {
    setError(err)
    setStep("upload")
  }, [])

  return (
    <>
      {error && step === "upload" && (
        <div className="max-w-2xl mx-auto mb-6 bg-red-500/10 border border-red-500/30 rounded-lg p-4 text-red-400">
          <p className="font-medium">Processing failed</p>
          <p className="text-sm mt-1">{error}</p>
        </div>
      )}

      {step === "upload" && (
        <Upload onUploaded={handleUploaded} />
      )}

      {step === "processing" && (
        <Processing
          jobId={jobId}
          onAwaitingLyrics={handleAwaitingLyrics}
          onDone={handleDone}
          onError={handleError}
        />
      )}

      {step === "lyrics" && (
        <LyricsReview
          jobId={jobId}
          foundLyrics={foundLyrics}
          artist={songMeta.artist}
          title={songMeta.title}
          onConfirmed={handleLyricsConfirmed}
          onError={handleError}
        />
      )}

      {step === "transcribing" && (
        <Processing
          jobId={jobId}
          onAwaitingLyrics={handleAwaitingLyrics}
          onDone={handleDone}
          onError={handleError}
          resumePolling
        />
      )}
    </>
  )
}

const App = () => {
  const location = useLocation()
  const navigate = useNavigate()
  
  // Determine current step from pathname
  const getCurrentStep = (): AppStep => {
    if (location.pathname.startsWith("/editor/")) return "editor"
    if (location.pathname.startsWith("/export/")) return "export"
    return "upload"
  }
  
  const currentStep = getCurrentStep()

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      {/* Header */}
      <header className="border-b border-zinc-800/60 bg-zinc-900/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
          <div
            className="flex items-center gap-2 cursor-pointer"
            onClick={() => navigate("/")}
          >
            <svg className="w-6 h-6 text-violet-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
            </svg>
            <span className="font-semibold text-white">Karaoke Creator</span>
          </div>

          {/* Step indicator */}
          <div className="flex items-center gap-1 text-xs">
            {STEP_LABELS.map((s, i) => (
              <div key={s.key} className="flex items-center">
                {i > 0 && <div className="w-4 h-px bg-zinc-700 mx-1" />}
                <span className={`px-2 py-0.5 rounded ${
                  s.key === currentStep ? "bg-violet-600/30 text-violet-300" : "text-zinc-600"
                }`}>
                  {s.label}
                </span>
              </div>
            ))}
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="px-4 py-10">
        <Routes>
          <Route path="/" element={<UploadFlow />} />
          <Route path="/editor/:jobId" element={<EditorPage />} />
          <Route path="/export/:jobId" element={<ExportPage />} />
        </Routes>
      </main>
    </div>
  )
}

export default App
