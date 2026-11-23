import { createSignal, onCleanup, Show } from "solid-js"
import { createEffect } from "../solidCompat"
import { useParams, useNavigate } from "@solidjs/router"
import type { JobResult } from "../types"
import { getResult, startAlignmentComparison } from "../api"
import { Editor } from "../components/Editor"

export const EditorPage = () => {
  const params = useParams<{ jobId: string }>()
  const navigate = useNavigate()
  const [result, setResult] = createSignal<JobResult | null>(null)
  const [loading, setLoading] = createSignal(true)
  const [error, setError] = createSignal<string | null>(null)
  let comparisonStarted: string | null = null

  createEffect(() => {
    const id = params.jobId
    if (!id) {
      setError("No job ID provided")
      setLoading(false)
      return
    }

    let cancelled = false
    let pollTimer: number | null = null

    const loadResult = async () => {
      try {
        const data = await getResult(id)
        if (cancelled) return
        setResult(data)
        const strategy = data.alignment_run?.selected_strategy ?? ""
        const isEnglishSofa = data.lyric_revision?.language === "en" && strategy.includes("sofa")
        const comparison = data.alignment_comparison
        if (
          isEnglishSofa &&
          comparison?.status !== "completed" &&
          comparison?.status !== "running" &&
          comparison?.status !== "not_supported" &&
          !data.alignment_results?.legacy &&
          comparisonStarted !== id
        ) {
          comparisonStarted = id
          try {
            const started = await startAlignmentComparison(id, data.selected_lyric_revision_id)
            if (cancelled) return
            if (started.status === "running") {
              pollTimer = window.setInterval(async () => {
                try {
                  const latest = await getResult(id)
                  if (cancelled) return
                  setResult(latest)
                  const status = latest.alignment_comparison?.status
                  if (status && status !== "running") {
                    if (pollTimer !== null) window.clearInterval(pollTimer)
                    pollTimer = null
                  }
                } catch {
                  // Keep the SOFA result usable if the optional comparison poll fails.
                }
              }, 2000)
            }
          } catch {
            // Comparison is optional; the immediately available SOFA result remains usable.
          }
        }
      } catch (err) {
        if (cancelled) return
        setError(err instanceof Error ? err.message : "Failed to load job")
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    loadResult()
    onCleanup(() => {
      cancelled = true
      if (pollTimer !== null) window.clearInterval(pollTimer)
    })
  }, () => undefined)

  return (
    <Show when={!loading()} fallback={
      <div class="max-w-7xl mx-auto px-4 py-10">
        <div class="text-center text-zinc-400">Loading editor...</div>
      </div>
    }>
    <Show when={!error() && result()} fallback={
      <div class="max-w-7xl mx-auto px-4 py-10">
        <div class="max-w-2xl mx-auto bg-red-500/10 border border-red-500/30 rounded-lg p-4 text-red-400">
          <p class="font-medium">Failed to load editor</p>
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
    <Editor
      jobId={params.jobId}
      result={result()!}
      onResultChange={setResult}
      onExport={() => navigate(`/export/${params.jobId}`)}
    />
    </Show>
    </Show>
  )
}
