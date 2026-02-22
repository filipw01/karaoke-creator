import { memo } from "react"
import { getAudioUrl } from "../api"

interface AudioControlsProps {
  jobId: string
  audioTrack: "original" | "vocals" | "instrumental"
  isPlaying: boolean
  onTogglePlayback: () => void
  onAudioTrackChange: (track: "original" | "vocals" | "instrumental") => void
  onEnded: () => void
  onTimeUpdate: () => void
  onLoadedMetadata: () => void
  audioRef: React.RefObject<HTMLAudioElement | null>
}

export const AudioControls = memo(function AudioControls({
  jobId,
  audioTrack,
  isPlaying,
  onTogglePlayback,
  onAudioTrackChange,
  onEnded,
  onTimeUpdate,
  onLoadedMetadata,
  audioRef,
}: AudioControlsProps) {
  return (
    <section
      className="flex items-center gap-4 bg-zinc-800/60 border border-zinc-700/50 rounded-xl p-4"
      aria-label="Playback controls"
    >
      <button
        type="button"
        onClick={onTogglePlayback}
        aria-label={isPlaying ? "Pause" : "Play"}
        className="w-10 h-10 flex items-center justify-center bg-violet-600 hover:bg-violet-500
                   rounded-full transition-colors shrink-0 cursor-pointer"
      >
        {isPlaying ? (
          <svg className="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 24 24">
            <rect x="6" y="4" width="4" height="16" />
            <rect x="14" y="4" width="4" height="16" />
          </svg>
        ) : (
          <svg className="w-5 h-5 text-white ml-0.5" fill="currentColor" viewBox="0 0 24 24">
            <polygon points="5,3 19,12 5,21" />
          </svg>
        )}
      </button>

      <div className="flex gap-1" role="group" aria-label="Audio track">
        {(["original", "vocals", "instrumental"] as const).map((track) => (
          <button
            key={track}
            type="button"
            aria-pressed={audioTrack === track}
            aria-label={`Use ${track} track`}
            onClick={() => onAudioTrackChange(track)}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors cursor-pointer ${
              audioTrack === track
                ? "bg-violet-600 text-white"
                : "bg-zinc-700 text-zinc-300 hover:bg-zinc-600"
            }`}
          >
            {track.charAt(0).toUpperCase() + track.slice(1)}
          </button>
        ))}
      </div>

      <audio
        ref={audioRef}
        src={getAudioUrl(jobId, audioTrack)}
        onEnded={onEnded}
        onTimeUpdate={onTimeUpdate}
        onLoadedMetadata={onLoadedMetadata}
      />
    </section>
  )
})
