import { useEffect, useState } from "react"
import { useParams, useNavigate } from "react-router-dom"
import type { JobResult } from "../types"
import { getResult } from "../api"
import { Editor } from "../components/Editor"

export const EditorPage = () => {
  const { jobId } = useParams<{ jobId: string }>()
  const navigate = useNavigate()
  const [result, setResult] = useState<JobResult | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!jobId) {
      setError("No job ID provided")
      setLoading(false)
      return
    }

    const loadResult = async () => {
      try {
        const data = await getResult(jobId)
        setResult(data)
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load job")
      } finally {
        setLoading(false)
      }
    }

    loadResult()
  }, [jobId])

  if (loading) {
    return (
      <div className="max-w-7xl mx-auto px-4 py-10">
        <div className="text-center text-zinc-400">Loading editor...</div>
      </div>
    )
  }

  if (error || !result) {
    return (
      <div className="max-w-7xl mx-auto px-4 py-10">
        <div className="max-w-2xl mx-auto bg-red-500/10 border border-red-500/30 rounded-lg p-4 text-red-400">
          <p className="font-medium">Failed to load editor</p>
          <p className="text-sm mt-1">{error ?? "Job not found"}</p>
          <button
            onClick={() => navigate("/")}
            className="mt-4 bg-zinc-700 hover:bg-zinc-600 text-white px-4 py-2 rounded-lg transition-colors"
          >
            Go to Upload
          </button>
        </div>
      </div>
    )
  }

  return (
    <Editor
      jobId={jobId!}
      result={result}
      onResultChange={setResult}
      onExport={() => navigate(`/export/${jobId}`)}
    />
  )
}
