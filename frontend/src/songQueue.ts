import { createSignal, createStore, onCleanup, reconcile } from "solid-js"
import { listSongs, setWorkingDirectory, type SongListItem } from "./api"

const [songs, setSongs] = createStore<SongListItem[]>([])
const [workingDirectory, setWorkingDirectoryState] = createSignal<string | null>(null)
const [loading, setLoading] = createSignal(true)
const [error, setError] = createSignal("")
let refreshing = false

export const refreshSongs = async () => {
  if (refreshing) return
  refreshing = true
  try {
    const library = await listSongs()
    setWorkingDirectoryState(library.working_directory)
    setSongs(reconcile(library.songs, "id"))
    setError("")
  } catch (err) {
    setError(err instanceof Error ? err.message : "Failed to list songs")
  } finally {
    setLoading(false)
    refreshing = false
  }
}

export const setLibraryDirectory = async (directory: string) => {
  const savedDirectory = await setWorkingDirectory(directory)
  setWorkingDirectoryState(savedDirectory)
  await refreshSongs()
}

export const startSongQueue = () => {
  void refreshSongs()
  const refreshTimer = window.setInterval(() => void refreshSongs(), 2000)
  onCleanup(() => window.clearInterval(refreshTimer))
}

export const songQueue = {
  songs,
  workingDirectory,
  loading,
  error,
}
