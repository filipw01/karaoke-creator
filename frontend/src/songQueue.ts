import { createSignal, createStore, onCleanup, reconcile } from "solid-js"
import { listSongs, setWorkingDirectory, type SongListItem } from "./api"

const [songs, setSongs] = createStore<SongListItem[]>([])
const [workingDirectory, setWorkingDirectoryState] = createSignal<string | null>(null)
const [loading, setLoading] = createSignal(true)
const [error, setError] = createSignal("")
let refreshInFlight: Promise<void> | null = null
let directoryVersion = 0
let directoryUpdate = Promise.resolve()

export const refreshSongs = (): Promise<void> => {
  if (refreshInFlight) return refreshInFlight
  const version = directoryVersion

  const refresh = async () => {
    try {
      const library = await listSongs()
      if (version !== directoryVersion) return
      setWorkingDirectoryState(library.working_directory)
      setSongs(reconcile(library.songs, "id"))
      setError("")
    } catch (err) {
      if (version === directoryVersion) setError(err instanceof Error ? err.message : "Failed to list songs")
    } finally {
      if (version === directoryVersion) {
        setLoading(false)
        refreshInFlight = null
      }
    }
  }

  refreshInFlight = refresh()
  return refreshInFlight
}

export const setLibraryDirectory = (directory: string): Promise<void> => {
  const update = directoryUpdate.catch(() => {}).then(async () => {
    const savedDirectory = await setWorkingDirectory(directory)
    directoryVersion += 1
    refreshInFlight = null
    if (savedDirectory !== workingDirectory()) {
      setSongs(reconcile([], "id"))
      setLoading(true)
    }
    setWorkingDirectoryState(savedDirectory)
    await refreshSongs()
  })
  directoryUpdate = update
  return update
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
