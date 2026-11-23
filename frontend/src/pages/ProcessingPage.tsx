import { createSignal, onCleanup, Show } from "solid-js"
import { useNavigate, useParams } from "@solidjs/router"
import { getResult, getStatus, retryJob, startProcessing } from "../api"
import { createEffect } from "../solidCompat"

export const ProcessingPage = () => {
  const params = useParams<{ jobId: string }>()
  const navigate = useNavigate()
  const [title, setTitle] = createSignal("Processing song")
  const [artist, setArtist] = createSignal("")
  const [status, setStatus] = createSignal("pending")
  const [message, setMessage] = createSignal("Starting processing...")
  const [progress, setProgress] = createSignal(0)
  const [error, setError] = createSignal("")
  const [retrying, setRetrying] = createSignal(false)
  const [retryAttempt, setRetryAttempt] = createSignal(0)

  createEffect(() => {
    retryAttempt()
    const id = params.jobId
    let cancelled = false
    let timer: number
    let startAttempted = false

    const poll = async () => {
      if (cancelled) return
      try {
        const data = await getStatus(id)
        if (cancelled) return
        setStatus(data.status)
        setMessage(data.message || "Processing...")
        setProgress(data.progress_percent ?? 0)

        if (data.status === "pending" && !startAttempted) {
          startAttempted = true
          try {
            await startProcessing(id)
          } catch (err) {
            const message = err instanceof Error ? err.message : "Failed to start processing"
            if (!message.toLowerCase().includes("already started")) {
              setError(message)
              return
            }
          }
        }
        if (data.status === "awaiting_lyrics") {
          navigate(`/lyrics/${id}`, { replace: true })
          return
        }
        if (data.status === "done") {
          navigate(`/editor/${id}`, { replace: true })
          return
        }
        if (data.status === "error") {
          setError(data.error || data.message || "Processing failed")
          return
        }
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : "Failed to read processing status")
      }
      if (!cancelled) timer = window.setTimeout(() => void poll(), 1500)
    }

    void getResult(id).then((result) => {
      if (cancelled) return
      setTitle(result.metadata.title || "Processing song")
      setArtist(result.metadata.artist)
    }).catch(() => {})

    void poll()
    onCleanup(() => {
      cancelled = true
      window.clearTimeout(timer)
    })
  })

  const showProgress = () => ["downloading", "extracting_audio", "separating_vocals", "detecting_bpm", "searching_lyrics", "extracting_lyrics", "detecting_pitch", "aligning_words", "generating"].includes(status())
  const retry = async () => {
    setRetrying(true)
    try {
      await retryJob(params.jobId)
      setError("")
      setStatus("pending")
      setMessage("Retrying from the last successful step...")
      setRetryAttempt(retryAttempt() + 1)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to retry processing")
    } finally {
      setRetrying(false)
    }
  }

  return (
    <div class="mx-auto max-w-xl py-16 text-center">
      <Show
        when={!error()}
        fallback={
          <>
            <div class="mx-auto mb-5 flex h-14 w-14 items-center justify-center rounded-full bg-red-500/10 text-2xl text-red-400">!</div>
            <h1 class="text-2xl font-bold text-white">{title()}</h1>
            <Show when={artist()}><p class="mt-1 text-zinc-500">{artist()}</p></Show>
            <p class="mt-5 rounded-lg border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-300">{error()}</p>
            <button
              type="button"
              onClick={() => void retry()}
              disabled={retrying()}
              class="mt-5 rounded-lg bg-violet-600 px-4 py-2 text-sm text-white hover:bg-violet-500 disabled:opacity-50"
            >
              {retrying() ? "Retrying..." : "Try again from last successful step"}
            </button>
            <button
              type="button"
              onClick={() => navigate("/")}
              class="mt-5 rounded-lg bg-zinc-800 px-4 py-2 text-sm text-zinc-200 hover:bg-zinc-700"
            >
              Back to upload
            </button>
          </>
        }
      >
        <div class="mx-auto mb-6 h-12 w-12 animate-spin rounded-full border-4 border-violet-500/25 border-t-violet-400" />
        <h1 class="text-2xl font-bold text-white">{title()}</h1>
        <Show when={artist()}><p class="mt-1 text-zinc-500">{artist()}</p></Show>
        <p class="mt-5 text-sm text-violet-300">{message()}</p>
        <Show when={showProgress() && progress() > 0}>
          <div class="mx-auto mt-5 h-2 max-w-md overflow-hidden rounded-full bg-zinc-800">
            <div class="h-full bg-violet-500 transition-all" style={{ width: `${progress()}%` }} />
          </div>
        </Show>
      </Show>
    </div>
  )
}
