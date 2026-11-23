import { createSignal, onCleanup } from "solid-js"
import { listSongs, setWorkingDirectory, startProcessing, type SongListItem } from "./api"

const [songs, setSongs] = createSignal<Array<SongListItem>>([])
const [workingDirectory, setWorkingDirectoryState] = createSignal<string | null>(null)
const [loading, setLoading] = createSignal(true)
const [error, setError] = createSignal("")
const autoStartIds = new Set<string>()
let refreshTimer: number | undefined
let refreshing = false

export const refreshSongs = async () => {
  if (refreshing) return
  refreshing = true
  try {
    const library = await listSongs()
    setWorkingDirectoryState(library.working_directory)
    setSongs(library.songs)
    setError("")

    for (const song of library.songs) {
      if (song.status === "pending" && autoStartIds.has(song.id)) {
        autoStartIds.delete(song.id)
        try {
          await startProcessing(song.id)
        } catch (err) {
          setError(err instanceof Error ? err.message : "Failed to start processing")
        }
      }
    }
  } catch (err) {
    setError(err instanceof Error ? err.message : "Failed to list songs")
  } finally {
    setLoading(false)
    refreshing = false
  }
}

export const enqueueSong = async (jobId: string) => {
  autoStartIds.add(jobId)
  await refreshSongs()
  if (songs().some((song) => song.id === jobId && song.status === "pending")) {
    try {
      await startProcessing(jobId)
      await refreshSongs()
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start processing")
    }
  }
  autoStartIds.delete(jobId)
}

export const startExistingSong = async (jobId: string) => {
  autoStartIds.delete(jobId)
  await startProcessing(jobId)
  await refreshSongs()
}

export const setLibraryDirectory = async (directory: string) => {
  const savedDirectory = await setWorkingDirectory(directory)
  setWorkingDirectoryState(savedDirectory)
  await refreshSongs()
}

export const startSongQueue = () => {
  void refreshSongs()
  refreshTimer = window.setInterval(() => void refreshSongs(), 2000)
  onCleanup(() => {
    if (refreshTimer !== undefined) window.clearInterval(refreshTimer)
  })
}

export const songQueue = {
  songs,
  workingDirectory,
  loading,
  error,
}
