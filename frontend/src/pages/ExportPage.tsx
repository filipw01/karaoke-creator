import { createEffect, createSignal, Show } from "solid-js"
import { useParams, useNavigate } from "@solidjs/router"
import type { JobResult } from "../types"
import { getResult } from "../api"
import { Export } from "../components/Export"

export const ExportPage = () => {
  const params = useParams<{ jobId: string }>()
  const navigate = useNavigate()
  const [loaded, setLoaded] = createSignal<{ jobId: string; result: JobResult } | null>(null)
  const [loading, setLoading] = createSignal(true)
  const [error, setError] = createSignal<string | null>(null)

  createEffect(() => params.jobId, (jobId) => {
    if (!jobId) {
      setError("No job ID provided")
      setLoading(false)
      return
    }

    setLoading(true)
    setError(null)
    let cancelled = false

    const loadResult = async () => {
      try {
        const data = await getResult(jobId)
        if (cancelled) return
        if (data.export_ready === false) {
          throw new Error("The selected lyric revision has not been aligned yet. Return to the editor and re-align it first.")
        }
        setLoaded({ jobId, result: data })
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : "Failed to load job")
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    void loadResult()
    return () => { cancelled = true }
  })

  return (
    <Show when={!loading() && (loaded()?.jobId === params.jobId || error())} fallback={
      <div class="max-w-7xl mx-auto px-4 py-10">
        <div class="text-center text-zinc-400">Loading export page...</div>
      </div>
    }>
    <Show when={!error() && loaded()} fallback={
      <div class="max-w-7xl mx-auto px-4 py-10">
        <div class="max-w-2xl mx-auto bg-red-500/10 border border-red-500/30 rounded-lg p-4 text-red-400">
          <p class="font-medium">Failed to load export page</p>
          <p class="text-sm mt-1">{error() ?? "Job not found"}</p>
          <button
            onClick={() => navigate("/")}
            class="mt-4 bg-zinc-700 hover:bg-zinc-600 text-white px-4 py-2 rounded-lg transition-colors"
          >
            Go to Upload
          </button>
        </div>
      </div>
    }>
    <Export
      jobId={loaded()!.jobId}
      result={loaded()!.result}
      onBack={() => navigate(`/editor/${loaded()!.jobId}`)}
      onStartOver={() => navigate("/")}
    />
    </Show>
    </Show>
  )
}
