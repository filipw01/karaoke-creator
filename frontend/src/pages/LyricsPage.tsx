import { createEffect, createSignal, Show } from "solid-js"
import { useNavigate, useParams } from "@solidjs/router"
import { confirmLyrics, getStatus } from "../api"

export const LyricsPage = () => {
  const params = useParams<{ jobId: string }>()
  const navigate = useNavigate()
  const [lyrics, setLyrics] = createSignal("")
  const [artist, setArtist] = createSignal("")
  const [title, setTitle] = createSignal("")
  const [loading, setLoading] = createSignal(true)
  const [submitting, setSubmitting] = createSignal(false)
  const [error, setError] = createSignal("")

  createEffect(() => params.jobId, (id) => {
    setLoading(true)
    setError("")
    setSubmitting(false)
    let cancelled = false

    const load = async () => {
      try {
        const status = await getStatus(id)
        if (cancelled) return
        setLyrics(status.found_lyrics ?? "")
        setArtist(status.metadata?.artist ?? "")
        setTitle(status.metadata?.title ?? "")
        setLoading(false)
        if (status.status !== "awaiting_lyrics") {
          if (status.status === "done") navigate(`/editor/${id}`, { replace: true })
          else if (status.status === "error") setError(status.error ?? "Processing failed")
          else navigate(`/processing/${id}`, { replace: true })
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load lyrics")
          setLoading(false)
        }
      }
    }

    void load()
    const timer = window.setInterval(() => {
      void getStatus(id).then((status) => {
        if (cancelled) return
        if (status.status === "awaiting_lyrics") setLoading(false)
        if (status.status === "done") navigate(`/editor/${id}`, { replace: true })
        if (status.status === "error") setError(status.error ?? "Processing failed")
        if (status.status !== "awaiting_lyrics" && status.status !== "done" && status.status !== "error") {
          navigate(`/processing/${id}`, { replace: true })
        }
      }).catch(() => {})
    }, 2000)

    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  })

  const approve = async (skip = false) => {
    const id = params.jobId
    setSubmitting(true)
    setError("")
    try {
      await confirmLyrics(id, skip ? "" : lyrics())
      if (params.jobId === id) navigate(`/processing/${id}`, { replace: true })
    } catch (err) {
      if (params.jobId !== id) return
      setError(err instanceof Error ? err.message : "Failed to continue processing")
      setSubmitting(false)
    }
  }

  return (
    <div class="max-w-3xl mx-auto">
      <Show when={!loading()} fallback={<div class="text-zinc-400">Loading lyrics...</div>}>
        <Show when={!submitting()} fallback={
          <div class="py-16 text-center">
            <div class="mx-auto mb-5 h-10 w-10 animate-spin rounded-full border-4 border-violet-500/25 border-t-violet-400" />
            <p class="text-violet-300">Continuing processing in the background…</p>
          </div>
        }>
          <div class="mb-6">
            <h1 class="text-2xl font-bold text-white">Review lyrics</h1>
            <p class="text-zinc-400">{artist()} — {title()}</p>
          </div>
          <textarea
            value={lyrics()}
            onInput={(event) => setLyrics(event.currentTarget.value)}
            class="w-full h-96 rounded-xl bg-zinc-900 border border-zinc-700 p-4 text-zinc-100"
          />
          <Show when={error()}>
            <p class="mt-3 text-red-400 text-sm">{error()}</p>
          </Show>
          <div class="mt-4 flex gap-3">
            <button
              type="button"
              onClick={() => void approve()}
              disabled={submitting()}
              class="bg-violet-600 hover:bg-violet-500 disabled:opacity-50 text-white px-5 py-2 rounded-lg"
            >
              Approve & continue
            </button>
            <button
              type="button"
              onClick={() => void approve(true)}
              disabled={submitting()}
              class="bg-zinc-800 hover:bg-zinc-700 disabled:opacity-50 text-zinc-300 px-5 py-2 rounded-lg"
            >
              Skip to AI transcription
            </button>
          </div>
        </Show>
      </Show>
    </div>
  )
}
