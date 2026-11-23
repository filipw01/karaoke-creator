import { createEffect, createSignal, Show } from "solid-js"
import { useParams, useNavigate } from "@solidjs/router"
import type { JobResult } from "../types"
import { getResult, getStatus, startAlignmentComparison } from "../api"
import { Editor } from "../components/Editor"

export const EditorPage = () => {
  const params = useParams<{ jobId: string }>()
  const navigate = useNavigate()
  const [result, setResult] = createSignal<JobResult | null>(null)
  const [loading, setLoading] = createSignal(true)
  const [loadedJobId, setLoadedJobId] = createSignal<string | null>(null)
  const [error, setError] = createSignal<string | null>(null)
  let comparisonStarted: string | null = null

  createEffect(() => params.jobId, (id) => {
    if (!id) {
      setError("No job ID provided")
      setLoading(false)
      return
    }

    setLoading(true)
    setError(null)
    let cancelled = false
    let pollTimer: number | null = null

    const loadResult = async () => {
      try {
        const data = await getResult(id)
        if (cancelled) return
        setResult(data)
        setLoadedJobId(id)
        const strategy = data.alignment_run?.selected_strategy ?? ""
        const isEnglishSofa = data.lyric_revision?.language === "en" && strategy.includes("sofa")
        const comparison = data.alignment_comparison
        const pollComparison = () => {
          pollTimer = window.setTimeout(async () => {
            try {
              const { alignment_comparison_status: status } = await getStatus(id)
              if (cancelled) return
              if (status === "running") {
                pollComparison()
                return
              }
              const latest = await getResult(id)
              if (cancelled) return
              if (!latest.alignment_comparison?.status || latest.alignment_comparison.status === "running") {
                pollComparison()
                return
              }
              setResult((current) => current ? {
                ...current,
                alignment_comparison: latest.alignment_comparison,
                alignment_results: { ...current.alignment_results, ...latest.alignment_results },
              } : current)
            } catch {
              if (!cancelled) pollComparison()
            }
          }, 2000)
        }

        if (comparison?.status === "running") {
          pollComparison()
        } else if (
          isEnglishSofa &&
          comparison?.status !== "completed" &&
          comparison?.status !== "not_supported" &&
          !data.alignment_results?.legacy &&
          comparisonStarted !== id
        ) {
          comparisonStarted = id
          try {
            const started = await startAlignmentComparison(id, data.selected_lyric_revision_id)
            if (!cancelled && started.status === "running") pollComparison()
          } catch {
            // The comparison is optional and the editor can use the initial result.
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
    return () => {
      cancelled = true
      if (pollTimer !== null) window.clearTimeout(pollTimer)
    }
  })

  return (
    <Show when={!loading() && (loadedJobId() === params.jobId || error())} fallback={
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
      onResultChange={(next) => {
        if (loadedJobId() === params.jobId) setResult(next)
      }}
      onExport={() => navigate(`/export/${params.jobId}`)}
    />
    </Show>
    </Show>
  )
}
