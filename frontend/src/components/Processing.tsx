import { useEffect, useRef, useState } from "react"
import { getStatus, startProcessing } from "../api"

interface ProcessingProps {
  jobId: string
  /** Phase 1 done — awaiting lyrics review */
  onAwaitingLyrics: (foundLyrics: string) => void
  /** Full pipeline done (Phase 2 complete) */
  onDone: () => void
  onError: (error: string) => void
  /** If true, skip calling start (Phase 2 was triggered externally) */
  resumePolling?: boolean
}

const STEPS = [
  { key: "extracting_audio", label: "Extracting audio" },
  { key: "separating_vocals", label: "Separating vocals" },
  { key: "detecting_bpm", label: "Detecting BPM" },
  { key: "searching_lyrics", label: "Searching for lyrics online" },
  { key: "extracting_lyrics", label: "Transcribing lyrics" },
  { key: "detecting_pitch", label: "Analyzing pitch" },
  { key: "generating", label: "Generating UltraStar file" },
] as const

const POLL_INTERVAL = 2000

export const Processing = ({ jobId, onAwaitingLyrics, onDone, onError, resumePolling }: ProcessingProps) => {
  const [currentStatus, setCurrentStatus] = useState("pending")
  const [message, setMessage] = useState("Starting...")
  const startedRef = useRef(false)

  useEffect(() => {
    let cancelled = false
    let timerId: number

    const kickOff = async () => {
      // Only call start once across StrictMode double-mount (skip if resuming)
      if (!startedRef.current && !resumePolling) {
        startedRef.current = true
        try {
          await startProcessing(jobId)
        } catch (err) {
          if (!cancelled) onError(err instanceof Error ? err.message : "Failed to start")
          return
        }
      }
      startedRef.current = true

      const poll = async () => {
        if (cancelled) return
        try {
          const data = await getStatus(jobId)
          if (cancelled) return
          setCurrentStatus(data.status)
          setMessage(data.message)

          if (data.status === "awaiting_lyrics") {
            onAwaitingLyrics(data.found_lyrics ?? "")
            return
          }
          if (data.status === "done") {
            onDone()
            return
          }
          if (data.status === "error") {
            onError(data.error ?? data.message)
            return
          }
        } catch {
          // Network blip — retry on next poll
        }
        if (!cancelled) {
          timerId = window.setTimeout(poll, POLL_INTERVAL)
        }
      }

      timerId = window.setTimeout(poll, 500)
    }

    kickOff()

    return () => {
      cancelled = true
      clearTimeout(timerId)
    }
  }, [jobId, onAwaitingLyrics, onDone, onError, resumePolling])

  const currentStepIndex = STEPS.findIndex((s) => s.key === currentStatus)

  return (
    <div className="max-w-xl mx-auto">
      <div className="text-center mb-10">
        <h2 className="text-3xl font-bold text-white mb-2">Processing Your Song</h2>
        <p className="text-zinc-400">{message}</p>
      </div>

      <div className="space-y-3">
        {STEPS.map((step, i) => {
          const isActive = step.key === currentStatus
          const isDone = currentStepIndex > i || currentStatus === "done"
          const isPending = currentStepIndex < i && currentStatus !== "done"

          return (
            <div
              key={step.key}
              className={`
                flex items-center gap-3 p-4 rounded-xl transition-all duration-300
                ${isActive ? "bg-violet-500/15 border border-violet-500/30" : ""}
                ${isDone ? "bg-emerald-500/10 border border-emerald-500/20" : ""}
                ${isPending ? "bg-zinc-800/50 border border-zinc-700/30 opacity-50" : ""}
              `}
            >
              <div className="flex-shrink-0 w-8 h-8 flex items-center justify-center">
                {isDone && (
                  <svg className="w-6 h-6 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                  </svg>
                )}
                {isActive && (
                  <div className="w-5 h-5 border-2 border-violet-400 border-t-transparent rounded-full animate-spin" />
                )}
                {isPending && (
                  <div className="w-3 h-3 rounded-full bg-zinc-600" />
                )}
              </div>

              <span className={`font-medium ${
                isActive ? "text-violet-300" :
                isDone ? "text-emerald-400" :
                "text-zinc-500"
              }`}>
                {step.label}
              </span>
            </div>
          )
        })}
      </div>

      <div className="mt-8">
        <div className="h-2 bg-zinc-800 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-violet-500 to-fuchsia-500 rounded-full transition-all duration-500"
            style={{
              width: currentStatus === "done"
                ? "100%"
                : `${Math.max(5, ((currentStepIndex + 1) / STEPS.length) * 100)}%`
            }}
          />
        </div>
      </div>
    </div>
  )
}
