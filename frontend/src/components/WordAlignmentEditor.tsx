import { createMemo, createSignal, onCleanup, Show } from "solid-js"
import { createEffect } from "../solidCompat"
import { Portal } from "@solidjs/web"
import {
  getResult,
  getStatus,
  resetWordAlignment,
  saveLyricRevision,
  startAlignmentRun,
} from "../api"
import type { AlignmentStatus, JobResult, LyricLanguage } from "../types"

interface WordAlignmentEditorProps {
  jobId: string
  result: JobResult
  onResultChange: (result: JobResult) => void
  onDirtyChange?: (dirty: boolean) => void
  open: boolean
  onClose: () => void
}

const SOURCE_LABELS: Record<string, string> = {
  online: "Online lyrics",
  asr_draft: "ASR lyric draft",
  manual: "Manual lyric revision",
}

const STATUS_LABELS: Record<AlignmentStatus, string> = {
  aligned: "Aligned",
  estimated: "Estimated",
  unmatched: "Unmatched",
}

const STATUS_CLASSES: Record<AlignmentStatus, string> = {
  aligned: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
  estimated: "bg-amber-500/15 text-amber-300 border-amber-500/30",
  unmatched: "bg-red-500/15 text-red-300 border-red-500/30",
}

const sleep = (ms: number): Promise<void> => new Promise((resolve) => window.setTimeout(resolve, ms))
type AlignmentBackend = "sofa" | "legacy"

export const WordAlignmentEditor = (props: WordAlignmentEditorProps) => {
  const jobId = () => props.jobId
  const result = () => props.result
  const onResultChange = props.onResultChange
  const onDirtyChange = props.onDirtyChange
  const revision = createMemo(() => result().lyric_revision ?? null)
  const draftRevision = createMemo(() => result().lyric_revisions?.find((item) => item.id === result().selected_lyric_revision_id) ?? revision())
  const tokens = createMemo(() => draftRevision()?.tokens ?? [])
  const alignments = createMemo(() => {
    if (!draftRevision() || draftRevision()!.id === revision()?.id) return result().word_alignments ?? []
    return tokens().map((token) => ({
      token_id: token.id,
      alignment_run_id: "",
      start: null,
      end: null,
      status: "unmatched" as const,
      confidence: 0,
      evidence: { selected_model: "", selected_audio: "", reasons: ["This lyric revision has not been aligned yet"], candidate_scores: [] },
      start_locked: false,
      end_locked: false,
    }))
  })
  const [lyricsDraft, setLyricsDraft] = createSignal(draftRevision()?.raw_text ?? "")
  const [savedLyricsDraft, setSavedLyricsDraft] = createSignal(draftRevision()?.raw_text ?? "")
  const [draftRevisionId, setDraftRevisionId] = createSignal(draftRevision()?.id ?? "")
  const [busyToken, setBusyToken] = createSignal<string | null>(null)
  const [savingLyrics, setSavingLyrics] = createSignal(false)
  const [realigning, setRealigning] = createSignal(false)
  const [alignmentBackend, setAlignmentBackend] = createSignal<AlignmentBackend>(
    result().alignment_run?.selected_strategy?.startsWith("legacy") ? "legacy" : "sofa",
  )
  const [error, setError] = createSignal<string | null>(null)

  createEffect(() => {
    setLyricsDraft(draftRevision()?.raw_text ?? "")
    setSavedLyricsDraft(draftRevision()?.raw_text ?? "")
    setDraftRevisionId(draftRevision()?.id ?? "")
  }, () => undefined)

  const alignmentByToken = createMemo(() => new Map(alignments().map((alignment) => [alignment.token_id, alignment])))
  const flaggedTokens = createMemo(() => tokens().filter((token) => alignmentByToken().get(token.id)?.status !== "aligned"))
  const lockedAlignments = createMemo(() => alignments().filter((alignment) => alignment.start_locked || alignment.end_locked))
  const lyricsDirty = createMemo(() => lyricsDraft() !== savedLyricsDraft())
  const alignmentIsCurrent = createMemo(() => draftRevision()?.id === revision()?.id)

  createEffect(() => {
    onDirtyChange?.(lyricsDirty())
  }, () => undefined)

  createEffect(() => {
    const strategy = result().alignment_run?.selected_strategy ?? ""
    if (strategy.startsWith("legacy")) setAlignmentBackend("legacy")
    else if (strategy.startsWith("sofa")) setAlignmentBackend("sofa")
  }, () => undefined)

  const refreshResult = async () => {
    const next = await getResult(jobId())
    onResultChange(next)
    return next
  }

  const resetAllLocks = async () => {
    if (lockedAlignments().length === 0) return
    setBusyToken("all")
    setError(null)
    try {
      await Promise.all(lockedAlignments().map((alignment) => resetWordAlignment(
        jobId(),
        alignment.token_id,
        [
          ...(alignment.start_locked ? ["start" as const] : []),
          ...(alignment.end_locked ? ["end" as const] : []),
        ],
      )))
      await refreshResult()
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to reset manual locks")
    } finally {
      setBusyToken(null)
    }
  }

  const saveRevision = async () => {
    if (!lyricsDraft().trim()) {
      setError("Lyrics cannot be empty.")
      return
    }
    setSavingLyrics(true)
    setError(null)
    try {
      const language: LyricLanguage = revision()?.language === "pl" ? "pl" : "en"
      const saved = await saveLyricRevision(jobId(), lyricsDraft(), language)
      setDraftRevisionId(saved.id)
      setLyricsDraft(saved.raw_text)
      setSavedLyricsDraft(saved.raw_text)
      const next = await getResult(jobId())
      onResultChange(next)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save lyric revision")
    } finally {
      setSavingLyrics(false)
    }
  }

  const realign = async () => {
    if (!draftRevisionId()) return
    if (lyricsDirty()) {
      setError("Save the lyric revision before re-aligning it.")
      return
    }
    setRealigning(true)
    setError(null)
    try {
      const started = await startAlignmentRun(jobId(), draftRevisionId(), [], alignmentBackend())
      let transientFailures = 0
      for (let attempt = 0; attempt < 900; attempt += 1) {
        await sleep(1000)
        try {
          const status = await getStatus(jobId())
          transientFailures = 0
          const matchingRun = status.alignment_run?.id === started.alignment_run_id
          if (matchingRun && status.status === "done") {
            const next = await getResult(jobId())
            if (next.alignment_run?.id !== started.alignment_run_id) {
              throw new Error("Alignment completed but the returned result belongs to another run")
            }
            onResultChange(next)
            break
          }
          if (matchingRun && status.status === "error") {
            throw new Error(status.error ?? status.message ?? "Alignment failed")
          }
        } catch (err) {
          if (err instanceof Error && (err.message.includes("completed but") || err.message.includes("Alignment failed"))) throw err
          transientFailures += 1
          if (transientFailures >= 5) throw err
        }
        if (attempt === 899) throw new Error("Alignment timed out")
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to re-align lyrics")
    } finally {
      setRealigning(false)
    }
  }

  createEffect(() => {
    if (!props.open) return
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") props.onClose()
    }
    document.addEventListener("keydown", closeOnEscape)
    onCleanup(() => document.removeEventListener("keydown", closeOnEscape))
  }, () => undefined)

  return (
    <Show when={props.open}>
      <Portal mount={document.body}>
        <div
          class="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4"
          onClick={(event) => {
            if (event.target === event.currentTarget) props.onClose()
          }}
        >
    <section
      role="dialog"
      aria-modal="true"
      aria-labelledby="word-alignment-title"
      class="max-h-[90vh] w-full max-w-4xl overflow-y-auto rounded-xl border border-zinc-700 bg-zinc-900 p-5 shadow-2xl"
    >
      <div class="mb-5 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 id="word-alignment-title" class="text-lg font-semibold text-white">Lyrics &amp; alignment</h3>
          <p class="text-sm text-zinc-400 mt-1">
            {draftRevision() ? SOURCE_LABELS[draftRevision()!.source] ?? draftRevision()!.source : "No lyric revision"}
            {draftRevision() ? ` · ${draftRevision()!.language.toUpperCase()}` : ""}
          </p>
        </div>
        <button type="button" onClick={props.onClose} aria-label="Close lyrics and alignment" class="px-3 py-2 rounded-lg bg-zinc-700 hover:bg-zinc-600 text-white text-sm">
          Close
        </button>
      </div>

      <div class="mb-5 flex flex-wrap items-center justify-between gap-3 rounded-lg border border-zinc-700/50 bg-zinc-950/50 p-3">
        <label class="flex items-center gap-2 text-sm text-zinc-300">
          <input
            type="checkbox"
            checked={alignmentBackend() === "legacy"}
            onChange={(event) => setAlignmentBackend(event.target.checked ? "legacy" : "sofa")}
            disabled={realigning()}
            class="accent-violet-500"
          />
          <span>
            Use legacy non-SOFA alignment
            <span class="ml-1 text-zinc-500">(ASR/CTC fallback)</span>
          </span>
        </label>
        <button
          type="button"
          onClick={realign}
          disabled={realigning() || !draftRevisionId() || lyricsDirty()}
          class="rounded-lg bg-violet-600 px-4 py-2 text-sm text-white hover:bg-violet-500 disabled:opacity-50"
        >
          {realigning() ? "Re-aligning..." : "Re-align revision"}
        </button>
      </div>

      <div class="grid gap-2 sm:grid-cols-3 text-xs text-zinc-400">
        {(["aligned", "estimated", "unmatched"] as const).map((status) => (
          <div class={`rounded-lg border px-3 py-2 ${STATUS_CLASSES[status]}`}>
            {STATUS_LABELS[status]}: {alignments().filter((alignment) => alignment.status === status).length}
          </div>
        ))}
      </div>

      {error() && <p class="text-sm text-red-300 bg-red-500/10 border border-red-500/30 rounded-lg px-3 py-2" role="alert">{error()}</p>}

      <div class="bg-zinc-900/60 border border-zinc-700/40 rounded-lg p-4 space-y-3">
        <div class="flex items-center justify-between gap-3">
          <label for="canonical-lyrics" class="text-sm font-medium text-zinc-200">Source lyrics</label>
          <button type="button" onClick={saveRevision} disabled={savingLyrics() || !lyricsDirty()} class="px-3 py-1.5 rounded-lg bg-zinc-700 hover:bg-zinc-600 text-white text-xs disabled:opacity-50">
            {savingLyrics() ? "Saving..." : "Save new revision"}
          </button>
        </div>
        <textarea
          id="canonical-lyrics"
          value={lyricsDraft()}
          onInput={(event) => setLyricsDraft(event.currentTarget.value)}
          rows={5}
          class="w-full bg-zinc-950 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-white font-mono leading-relaxed focus:outline-none focus:border-violet-500"
        />
        {lyricsDirty() && <p class="text-xs text-amber-300">Save this revision before re-aligning.</p>}
        {!lyricsDirty() && draftRevisionId() !== revision()?.id && <p class="text-xs text-amber-300">Saved as a new revision. Re-align explicitly to apply it.</p>}
        {!lyricsDirty() && !alignmentIsCurrent() && <p class="text-xs text-zinc-400">This saved revision must be aligned before its timing can be used.</p>}
      </div>

      <div class="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-zinc-700/50 bg-zinc-900/40 px-3 py-2 text-xs text-zinc-400">
        <span>{flaggedTokens().length} words need alignment review{lockedAlignments().length > 0 ? ` · ${lockedAlignments().length} manually locked` : ""}</span>
        {lockedAlignments().length > 0 && (
          <button
            type="button"
            onClick={resetAllLocks}
            disabled={busyToken() !== null}
            class="rounded bg-zinc-700 px-2 py-1 text-zinc-200 hover:bg-zinc-600 disabled:opacity-40"
          >
            {busyToken() === "all" ? "Resetting..." : "Reset manual locks"}
          </button>
        )}
      </div>
    </section>
        </div>
      </Portal>
    </Show>
  )
}
