import { createSignal, For, Show } from "solid-js"
import type { JSX } from "@solidjs/web"
import { createRouter, defineRoutes, hashHistory, useLocation, useNavigate } from "@solidjs/router"
import { isTauri } from "@tauri-apps/api/core"
import type { AppStep, AudioTrack, JobResult } from "./types"
import {
  exportExternalSong,
  getExternalSong,
  getExternalSongAudioUrl,
  getExternalSongDownloadUrl,
  getExternalSongSeparationStatus,
  pickWorkingDirectory,
  removeSong,
  saveExternalSong,
  startExternalSongSeparation,
  type ExternalSongSeparationStatus,
  type SongListItem,
} from "./api"
import { Upload } from "./components/Upload"
import { Editor } from "./components/Editor"
import { EditorPage } from "./pages/EditorPage"
import { ExportPage } from "./pages/ExportPage"
import { LyricsPage } from "./pages/LyricsPage"
import { ProcessingPage } from "./pages/ProcessingPage"
import { enqueueSong, refreshSongs, setLibraryDirectory, songQueue, startExistingSong, startSongQueue } from "./songQueue"

const WORKFLOW_STEPS: Array<{ key: AppStep; label: string }> = [
  { key: "upload", label: "Upload" },
  { key: "download", label: "Download" },
  { key: "prepare", label: "Prepare audio" },
  { key: "lyrics", label: "Review lyrics" },
  { key: "transcribe", label: "Transcribe & align" },
  { key: "generate", label: "Generate" },
  { key: "editor", label: "Editor" },
  { key: "export", label: "Export" },
]
const STEP_PATHS: Record<AppStep, string> = {
  upload: "/",
  download: "/processing/",
  prepare: "/processing/",
  lyrics: "/lyrics/",
  transcribe: "/processing/",
  generate: "/processing/",
  editor: "/editor/",
  export: "/export/",
}
const ALL_AUDIO_TRACKS: AudioTrack[] = [
  "original",
  "vocals",
  "instrumental",
  "instrumental_with_playback",
  "vocals_with_playback",
]
const managedJobsAvailable = () => !isTauri() || import.meta.env.DEV

const UploadFlow = () => {
  const navigate = useNavigate()
  const [error, setError] = createSignal("")

  const handleUploaded = (id: string) => {
    setError("")
    void enqueueSong(id)
    navigate(`/processing/${id}`)
  }

  return (
    <Show when={managedJobsAvailable()} fallback={
      <p class="mx-auto max-w-2xl text-sm text-zinc-400">
        {songQueue.songs().length ? "Select a chart to open it." : "No UltraStar charts found in this folder."}
      </p>
    }>
      <Show when={error()}>
        <div class="max-w-2xl mx-auto mb-6 bg-red-500/10 border border-red-500/30 rounded-lg p-4 text-red-400">
          <p class="font-medium">Processing failed</p>
          <p class="text-sm mt-1">{error()}</p>
        </div>
      </Show>
      <Upload onUploaded={handleUploaded} />
    </Show>
  )
}

const routes = defineRoutes([
  { path: "/", component: UploadFlow },
  { path: "/lyrics/:jobId", component: LyricsPage },
  { path: "/processing/:jobId", component: ProcessingPage },
  { path: "/editor/:jobId", component: EditorPage },
  { path: "/export/:jobId", component: ExportPage },
])

const Router = createRouter({ routes, history: hashHistory() })

interface ExternalChartEditor {
  filename: string
  path: string
  result: JobResult
  audioAvailable: boolean
  playbackUrl?: string
  separationAvailable: boolean
  stemsAvailable: boolean
  separationStatus: ExternalSongSeparationStatus["status"]
  separationError: string
}

const AppShell = (props: { children: JSX.Element }) => {
  const location = useLocation()
  const navigate = useNavigate()
  const [libraryError, setLibraryError] = createSignal("")
  const [directoryInput, setDirectoryInput] = createSignal("")
  const [pendingRemoval, setPendingRemoval] = createSignal<SongListItem | null>(null)
  const [removingSong, setRemovingSong] = createSignal(false)
  const [externalChart, setExternalChart] = createSignal<ExternalChartEditor | null>(null)
  const [externalSongLoading, setExternalSongLoading] = createSignal(false)
  let externalSongRequest = 0
  startSongQueue()
  
  const activeJobId = () => location.pathname.split("/")[2] ?? ""
  const activeSong = () => songQueue.songs().find(
    (song) => song.kind === "managed" && song.id === activeJobId(),
  )
  const activeWorkflowStep = (): AppStep => {
    if (!activeJobId()) return "upload"
    if (location.pathname.startsWith("/editor/")) return "editor"
    if (location.pathname.startsWith("/export/")) return "export"
    const song = activeSong()
    const status = song?.status ?? ""
    if (status === "downloading" || (status === "error" && song?.failed_phase === "download")) return "download"
    if (status === "awaiting_lyrics") return "lyrics"
    if (["extracting_lyrics", "detecting_pitch", "aligning_words"].includes(status)) return "transcribe"
    if (status === "generating") return "generate"
    if (status === "done") return "editor"
    if (status === "error" && song?.failed_phase === "phase2") return "transcribe"
    return "prepare"
  }

  const configureDirectory = async (directory: string) => {
    externalSongRequest += 1
    setExternalSongLoading(false)
    setExternalChart(null)
    setLibraryError("")
    try {
      await setLibraryDirectory(directory)
    } catch (error) {
      setLibraryError(error instanceof Error ? error.message : "Unable to set working directory")
    }
  }

  const chooseDirectory = async () => {
    let directory: string | null = null
    try {
      directory = isTauri()
        ? await pickWorkingDirectory()
        : window.prompt("Enter the full path to your songs folder")?.trim() || null
    } catch {
      directory = window.prompt("Enter the full path to your songs folder")?.trim() || null
    }
    if (directory) await configureDirectory(directory)
  }

  const openSong = async (song: SongListItem) => {
    if (song.kind === "external") {
      const requestId = ++externalSongRequest
      setExternalChart(null)
      setExternalSongLoading(true)
      setLibraryError("")
      try {
        const externalSong = await getExternalSong(song.path ?? "")
        if (requestId !== externalSongRequest) return
        setExternalChart({
          filename: externalSong.filename,
          path: externalSong.path,
          audioAvailable: externalSong.audio_available,
          playbackUrl: externalSong.playback_url,
          separationAvailable: externalSong.separation_available,
          stemsAvailable: externalSong.stems_available,
          separationStatus: externalSong.separation_status,
          separationError: externalSong.separation_error ?? "",
          result: {
            metadata: externalSong.metadata,
            words: [],
            phrases: externalSong.phrases,
            ultrastar_txt: externalSong.content,
            audio_file: "",
            vocals_file: "",
            instrumental_file: "",
            video_file: "",
            export_ready: false,
          },
        })
      } catch (error) {
        if (requestId === externalSongRequest) {
          setLibraryError(error instanceof Error ? error.message : "Unable to open song file")
        }
      } finally {
        if (requestId === externalSongRequest) setExternalSongLoading(false)
      }
      return
    }
    externalSongRequest += 1
    setExternalSongLoading(false)
    setExternalChart(null)
    const { id, status } = song
    if (status === "awaiting_lyrics") {
      navigate(`/lyrics/${id}`)
      return
    }
    if (status === "done") {
      navigate(`/editor/${id}`)
      return
    }
    if (status === "pending") {
      await startExistingSong(id)
    }
    navigate(`/processing/${id}`)
  }

  const separateExistingSong = async (path: string) => {
    const current = externalChart()
    if (
      !current
      || current.path !== path
      || !current.audioAvailable
      || !current.separationAvailable
      || current.stemsAvailable
      || current.separationStatus === "separating"
    ) return

    setExternalChart({ ...current, separationStatus: "separating", separationError: "" })
    try {
      let status = await startExternalSongSeparation(path)
      while (status.status === "separating") {
        if (externalChart()?.path !== path) return
        await new Promise<void>((resolve) => window.setTimeout(resolve, 1200))
        if (externalChart()?.path !== path) return
        status = await getExternalSongSeparationStatus(path)
      }
      setExternalChart((latest) => latest?.path === path
        ? {
          ...latest,
          stemsAvailable: status.status === "done",
          separationStatus: status.status,
          separationError: status.error ?? "",
        }
        : latest)
    } catch (error) {
      setExternalChart((latest) => latest?.path === path
        ? {
          ...latest,
          separationStatus: "error",
          separationError: error instanceof Error ? error.message : "Failed to split song audio",
        }
        : latest)
    }
  }

  const isWorking = (status: string) => ![
    "pending",
    "done",
    "error",
    "awaiting_lyrics",
  ].includes(status)
  const isSelectedSong = (song: SongListItem) => song.kind === "managed"
    ? activeJobId() === song.id
    : externalChart()?.path === song.path
  const exportChart = async (path: string) => {
    setLibraryError("")
    try {
      await exportExternalSong(path)
    } catch (error) {
      setLibraryError(error instanceof Error ? error.message : "Failed to save a copy of the chart")
    }
  }
  const confirmRemoval = async () => {
    const song = pendingRemoval()
    if (!song || removingSong()) return
    setRemovingSong(true)
    setLibraryError("")
    try {
      await removeSong(song)
      setPendingRemoval(null)
      if (song.kind === "external" && externalChart()?.path === song.path) setExternalChart(null)
      await refreshSongs()
      if (song.kind === "managed" && activeJobId() === song.id) navigate("/")
    } catch (error) {
      setLibraryError(error instanceof Error ? error.message : "Failed to remove song files")
    } finally {
      setRemovingSong(false)
    }
  }

  return (
    <Show
      when={!songQueue.loading() && songQueue.workingDirectory()}
      fallback={
        <main class="flex min-h-screen items-center justify-center bg-zinc-950 px-4 text-zinc-100">
          <section class="w-full max-w-lg rounded-2xl border border-zinc-800 bg-zinc-900/80 p-8 shadow-xl">
            <h1 class="mb-4 text-xl font-semibold">Choose your songs folder</h1>
            <p class="mb-6 text-sm leading-6 text-zinc-400">
              Select the folder where you keep your UltraStar songs. Existing charts will appear alongside songs you create here.
            </p>
            <Show when={songQueue.loading()}>
              <p class="mb-4 text-sm text-zinc-500">Checking library settings…</p>
            </Show>
            <Show when={libraryError()}>
              <p class="mb-4 rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-300">{libraryError()}</p>
            </Show>
            <button type="button" onClick={() => void chooseDirectory()} class="w-full rounded-lg bg-violet-600 px-4 py-3 font-medium text-white hover:bg-violet-500">
              Select directory…
            </button>
            <div class="mt-4 flex gap-2">
              <input
                value={directoryInput()}
                onInput={(event) => setDirectoryInput(event.currentTarget.value)}
                placeholder="/path/to/songs"
                aria-label="Working directory path"
                class="min-w-0 flex-1 rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-white placeholder:text-zinc-600"
              />
              <button
                type="button"
                onClick={() => void configureDirectory(directoryInput().trim())}
                disabled={!directoryInput().trim()}
                class="rounded-lg border border-zinc-700 px-4 py-2 text-sm text-zinc-200 hover:bg-zinc-800 disabled:opacity-40"
              >
                Use path
              </button>
            </div>
          </section>
        </main>
      }
    >
    <div class="flex h-dvh flex-col overflow-hidden bg-zinc-950 text-zinc-100">
      {/* Header */}
      <header class="shrink-0 border-b border-zinc-800/60 bg-zinc-900/50 backdrop-blur-sm">
        <div class="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
          <div
            class="flex items-center gap-2 cursor-pointer"
            onClick={() => {
              externalSongRequest += 1
              setExternalSongLoading(false)
              setExternalChart(null)
              navigate("/")
            }}
          >
            <svg class="w-6 h-6 text-violet-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width={2}>
              <path stroke-linecap="round" stroke-linejoin="round" d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
            </svg>
            <span class="font-semibold text-white">Karaoke Creator</span>
          </div>

        </div>
      </header>

      <div class="flex min-h-0 flex-1 overflow-hidden">
        <aside class="h-full w-72 shrink-0 overflow-y-auto overscroll-contain border-r border-zinc-800 bg-zinc-950/70 px-3 py-4">
          <div class="mb-4 rounded-lg border border-zinc-800 bg-zinc-900/60 p-3">
            <div class="mb-2 text-[10px] font-semibold uppercase tracking-wide text-zinc-500">Working directory</div>
            <div class="mb-2 truncate text-xs text-zinc-300" title={songQueue.workingDirectory() ?? ""}>{songQueue.workingDirectory()}</div>
            <button type="button" onClick={() => void chooseDirectory()} class="text-xs text-violet-400 hover:text-violet-300">Change directory</button>
          </div>
          <Show when={libraryError()}>
            <p class="mb-3 rounded-lg border border-red-500/30 bg-red-500/10 p-2 text-xs text-red-300">{libraryError()}</p>
          </Show>
          <div class="mb-3 flex items-center justify-between px-2">
            <h2 class="text-xs font-semibold uppercase tracking-wide text-zinc-500">Songs</h2>
            <Show when={managedJobsAvailable()}>
              <button
                type="button"
                onClick={() => {
                  externalSongRequest += 1
                  setExternalSongLoading(false)
                  setExternalChart(null)
                  navigate("/")
                }}
                class="text-xs text-violet-400 hover:text-violet-300"
              >
                + Add
              </button>
            </Show>
          </div>
          <Show when={!songQueue.loading()} fallback={<div class="px-2 text-sm text-zinc-500">Loading songs...</div>}>
            <div class="space-y-1">
              <For each={songQueue.songs()}>
                {(song) => (
                  <div class={`rounded-lg border transition-colors ${isSelectedSong(song) ? "border-violet-500/40 bg-violet-500/10" : "border-transparent hover:bg-zinc-800/70"}`}>
                    <div class="flex items-start">
                      <button
                        type="button"
                        onClick={() => void openSong(song)}
                        aria-current={isSelectedSong(song) ? "page" : undefined}
                        class={`min-w-0 flex-1 px-3 py-2.5 text-left ${isSelectedSong(song) ? "text-white" : ""}`}
                        title={song.kind === "external" ? `Open ${song.filename}` : song.message}
                      >
                        <div class="flex items-center gap-2">
                          <Show when={song.kind === "managed" && (isWorking(song.status) || song.status === "pending")}>
                            <div class="h-4 w-4 shrink-0 rounded-full border-2 border-violet-400 border-t-transparent animate-spin" />
                          </Show>
                          <Show when={song.kind === "managed" && song.status === "awaiting_lyrics"}>
                            <div class="h-2 w-2 shrink-0 rounded-full bg-amber-400" />
                          </Show>
                          <Show when={song.kind === "managed" && song.status === "done"}>
                            <div class="h-2 w-2 shrink-0 rounded-full bg-emerald-400" />
                          </Show>
                          <Show when={song.kind === "managed" && song.status === "error"}>
                            <div class="h-2 w-2 shrink-0 rounded-full bg-red-400" />
                          </Show>
                          <Show when={song.kind === "external"}>
                            <div class="h-2 w-2 shrink-0 rounded-full bg-sky-400" />
                          </Show>
                          <div class="min-w-0">
                            <div class="truncate text-sm font-medium text-zinc-200">{song.title}</div>
                            <div class="truncate text-xs text-zinc-500">{song.artist || song.filename}</div>
                            <Show when={song.kind === "managed"}>
                              <div class={`mt-1 truncate text-[11px] ${song.status === "error" ? "text-red-400" : song.status === "awaiting_lyrics" ? "text-amber-400" : "text-zinc-400"}`}>
                                {song.queue_position ? `Queue #${song.queue_position} · ` : ""}{song.message || song.status}
                              </div>
                            </Show>
                            <Show when={song.folder}>
                              <div class="truncate text-[10px] text-zinc-600">{song.folder}</div>
                            </Show>
                          </div>
                        </div>
                        <Show when={song.kind === "managed" && isWorking(song.status)}>
                          <div class="mt-2 h-1 overflow-hidden rounded-full bg-zinc-800">
                            <div
                              class="h-full bg-violet-500 transition-all"
                              style={{ width: `${Math.max(8, song.progress_percent || 0)}%` }}
                            />
                          </div>
                        </Show>
                      </button>
                      <button
                        type="button"
                        aria-label={`Remove ${song.title}`}
                        title={song.kind === "managed" && isWorking(song.status) ? "Cannot remove while processing" : "Remove song"}
                        disabled={song.kind === "managed" && isWorking(song.status)}
                        onClick={() => setPendingRemoval(song)}
                        class="shrink-0 px-2 py-2.5 text-xs text-zinc-600 hover:text-red-400 disabled:cursor-not-allowed disabled:opacity-30"
                      >
                        Remove
                      </button>
                    </div>
                  </div>
                )}
              </For>
              <Show when={songQueue.songs().length === 0}>
                <div class="px-2 py-6 text-sm text-zinc-600">No songs yet</div>
              </Show>
            </div>
          </Show>
        </aside>
        <main class="min-h-0 min-w-0 flex-1 overflow-y-auto overscroll-contain px-4 py-10">
          <Show when={externalSongLoading()}>
            <div class="mx-auto max-w-7xl text-center text-sm text-zinc-400">Loading existing song editor…</div>
          </Show>
          <Show when={!externalSongLoading()}>
            <Show when={externalChart()} fallback={props.children}>
              {(chart) => (
                <div class="mx-auto max-w-7xl space-y-4">
                  <div class="flex flex-wrap items-center justify-between gap-3">
                    <div class="flex items-center gap-3">
                      <button
                        type="button"
                        onClick={() => setExternalChart(null)}
                        class="rounded-lg border border-zinc-700 px-3 py-2 text-sm text-zinc-300 hover:bg-zinc-800"
                      >
                        Back to songs
                      </button>
                      <div class="min-w-0">
                        <div class="truncate text-sm font-medium text-white">{chart().result.metadata.artist} — {chart().result.metadata.title}</div>
                        <div class="text-xs text-zinc-500">Changes overwrite the existing chart after confirmation.</div>
                      </div>
                    </div>
                    <div class="flex flex-wrap items-center gap-2">
                      <Show when={chart().audioAvailable} fallback={
                        <span class="text-xs text-zinc-500">Source audio not found</span>
                      }>
                        <Show when={chart().stemsAvailable} fallback={
                          <Show when={chart().separationAvailable}>
                            <button
                              type="button"
                              onClick={() => void separateExistingSong(chart().path)}
                              disabled={chart().separationStatus === "separating"}
                              class="rounded-lg bg-violet-600 px-3 py-2 text-xs font-medium text-white hover:bg-violet-500 disabled:cursor-wait disabled:opacity-60"
                            >
                              {chart().separationStatus === "separating"
                                ? "Separating vocals…"
                                : chart().separationStatus === "error"
                                  ? "Retry vocal separation"
                                  : "Split vocals & instrumental"}
                            </button>
                          </Show>
                        }>
                          <span class="text-xs text-emerald-400">Vocal and instrumental stems ready</span>
                        </Show>
                      </Show>
                      <Show when={isTauri()} fallback={
                        <a
                          href={getExternalSongDownloadUrl(chart().path)}
                          class="rounded border border-zinc-700 px-3 py-2 text-xs text-zinc-300 hover:bg-zinc-800"
                        >
                          Download chart
                        </a>
                      }>
                        <button
                          type="button"
                          onClick={() => void exportChart(chart().path)}
                          class="rounded border border-zinc-700 px-3 py-2 text-xs text-zinc-300 hover:bg-zinc-800"
                        >
                          Save chart copy
                        </button>
                      </Show>
                    </div>
                  </div>
                  <Show when={chart().separationStatus === "separating"}>
                    <p class="text-sm text-violet-300" role="status">Separating vocals from the instrumental. This may take a few minutes.</p>
                  </Show>
                  <Show when={chart().separationError}>
                    <p class="text-sm text-red-300" role="alert">{chart().separationError}</p>
                  </Show>
                  <Editor
                    jobId={`external:${chart().path}`}
                    result={chart().result}
                    onResultChange={(result) => setExternalChart((current) => current ? { ...current, result } : current)}
                    onExport={() => undefined}
                    saveNotes={(phrases) => saveExternalSong(chart().path, phrases)}
                    audioSourceUrl={chart().audioAvailable ? undefined : null}
                    audioSourceUrls={chart().audioAvailable ? {
                      original: chart().playbackUrl ?? getExternalSongAudioUrl(chart().path, "original"),
                      ...(chart().stemsAvailable ? {
                        vocals: getExternalSongAudioUrl(chart().path, "vocals"),
                        instrumental: getExternalSongAudioUrl(chart().path, "instrumental"),
                      } : {}),
                    } : undefined}
                    availableAudioTracks={chart().audioAvailable
                      ? chart().stemsAvailable ? ALL_AUDIO_TRACKS : ["original"]
                      : []}
                    confirmSaveMessage={`Overwrite the original chart "${chart().filename}" with your changes?`}
                    showAdvancedActions={false}
                  />
                </div>
              )}
            </Show>
          </Show>
          <Show when={activeJobId() && !externalChart() && !externalSongLoading()}>
            <nav aria-label="Song workflow" class="mx-auto mt-12 flex max-w-4xl items-center justify-center gap-1 border-t border-zinc-800 pt-5 text-xs">
              <For each={WORKFLOW_STEPS}>
                {(step, index) => {
                  const destination = () => `${STEP_PATHS[step.key]}${step.key === "upload" ? "" : activeJobId()}`
                  const isActive = () => step.key === activeWorkflowStep()
                  const isComplete = () => index() < WORKFLOW_STEPS.findIndex((item) => item.key === activeWorkflowStep())
                  return (
                    <div class="flex items-center">
                      <Show when={index() > 0}>
                        <div class="mx-1 h-px w-4 bg-zinc-700" />
                      </Show>
                      <button
                        type="button"
                        onClick={() => navigate(destination())}
                        aria-current={isActive() ? "step" : undefined}
                        class={`rounded px-2 py-1 ${isActive() ? "bg-violet-600/30 text-violet-300" : isComplete() ? "text-emerald-400" : "text-zinc-500 hover:text-zinc-300"}`}
                      >
                        {step.label}
                      </button>
                    </div>
                  )
                }}
              </For>
            </nav>
          </Show>
        </main>
      </div>
      <Show when={pendingRemoval()}>
        {(song) => (
          <div
            class="fixed inset-0 z-[70] flex items-center justify-center bg-black/70 p-4"
            onClick={(event) => {
              if (event.target === event.currentTarget && !removingSong()) setPendingRemoval(null)
            }}
          >
            <section
              role="alertdialog"
              aria-modal="true"
              aria-labelledby="remove-song-title"
              aria-describedby="remove-song-description"
              class="w-full max-w-md rounded-xl border border-zinc-700 bg-zinc-900 p-6 shadow-2xl"
            >
              <h2 id="remove-song-title" class="text-lg font-semibold text-white">Permanently remove song?</h2>
              <p id="remove-song-description" class="mt-2 text-sm leading-6 text-zinc-300">
                <strong>{song().title}</strong>
                {song().kind === "managed"
                  ? " and its uploaded audio, generated files, and published song folder will be deleted."
                  : song().folder
                    ? ` and the "${song().folder}" directory, including all files inside it, will be deleted.`
                    : " and its chart file will be deleted from the working directory."}
                {" "}This cannot be undone.
              </p>
              <div class="mt-6 flex justify-end gap-3">
                <button
                  type="button"
                  onClick={() => setPendingRemoval(null)}
                  disabled={removingSong()}
                  class="rounded-lg border border-zinc-700 px-4 py-2 text-sm text-zinc-300 hover:bg-zinc-800 disabled:opacity-50"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={() => void confirmRemoval()}
                  disabled={removingSong()}
                  class="rounded-lg bg-red-600 px-4 py-2 text-sm font-medium text-white hover:bg-red-500 disabled:opacity-50"
                >
                  {removingSong() ? "Removing files…" : "Remove files"}
                </button>
              </div>
            </section>
          </div>
        )}
      </Show>
    </div>
    </Show>
  )
}

const App = () => (
  <Router>
    {props => <AppShell>{props.children}</AppShell>}
  </Router>
)

export default App
