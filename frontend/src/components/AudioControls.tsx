import { For, createEffect } from "solid-js"
import type { AudioTrack } from "../types"
import { getAudioUrl } from "../api"

export type { AudioTrack } from "../types"

interface AudioControlsProps {
  jobId: string
  audioTrack: AudioTrack
  isPlaying: boolean
  onTogglePlayback: () => void
  onAudioTrackChange: (track: AudioTrack) => void
  onSourceChange: (audio: HTMLAudioElement, source: string, previousSource: string | undefined) => void
  onPlaying: () => void
  onWaiting: () => void
  onPause: () => void
  onEnded: () => void
  onTimeUpdate: () => void
  onSeeked: () => void
  onLoadedMetadata: () => void
  onTestSound: () => void
  audioRef: { current: HTMLAudioElement | null }
  audioSourceUrl?: string
  audioSourceUrls?: Partial<Record<AudioTrack, string>>
  availableTracks?: AudioTrack[]
  showTrackSelection?: boolean
}

const AUDIO_TRACK_OPTIONS: Array<[AudioTrack, string]> = [
  ["original", "Original"],
  ["vocals", "Vocal"],
  ["instrumental", "Instrumental"],
  ["instrumental_with_playback", "Instrumental with Playback"],
  ["vocals_with_playback", "Vocal with Playback"],
]

export const AudioControls = (props: AudioControlsProps) => {
  const sourceUrl = (): string => {
    if (props.audioSourceUrls) {
      const sourceTrack = props.audioTrack === "instrumental_with_playback"
        ? "instrumental"
        : props.audioTrack === "vocals_with_playback"
          ? "vocals"
          : props.audioTrack
      return props.audioSourceUrls[sourceTrack] ?? ""
    }
    return props.audioSourceUrl ?? getAudioUrl(props.jobId, props.audioTrack)
  }

  createEffect(sourceUrl, (source, previousSource) => {
    const audio = props.audioRef.current
    if (audio && source !== previousSource) props.onSourceChange(audio, source, previousSource)
  })

  return (
    <section
      class="flex flex-wrap items-center gap-4 bg-zinc-800/60 border border-zinc-700/50 rounded-xl p-4"
      aria-label="Playback controls"
    >
      <button
        type="button"
        onClick={props.onTogglePlayback}
        aria-label={props.isPlaying ? "Pause" : "Play"}
        class="w-10 h-10 flex items-center justify-center bg-violet-600 hover:bg-violet-500
                   rounded-full transition-colors shrink-0 cursor-pointer"
      >
        {props.isPlaying ? (
          <svg class="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 24 24">
            <rect x="6" y="4" width="4" height="16" />
            <rect x="14" y="4" width="4" height="16" />
          </svg>
        ) : (
          <svg class="w-5 h-5 text-white ml-0.5" fill="currentColor" viewBox="0 0 24 24">
            <polygon points="5,3 19,12 5,21" />
          </svg>
        )}
      </button>

      {props.showTrackSelection !== false && (
        <div class="flex flex-wrap gap-1" role="group" aria-label="Audio track">
          <For each={AUDIO_TRACK_OPTIONS.filter(([track]) => !props.availableTracks || props.availableTracks.includes(track))}>
            {([track, label]) => (
            <button
              type="button"
              aria-pressed={props.audioTrack === track ? "true" : "false"}
              aria-label={label}
              onClick={() => props.onAudioTrackChange(track)}
              class={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors cursor-pointer ${
                props.audioTrack === track
                  ? "bg-violet-600 text-white"
                  : "bg-zinc-700 text-zinc-300 hover:bg-zinc-600"
              }`}
            >
              {label}
            </button>
            )}
          </For>
        </div>
      )}

      <button
        type="button"
        onClick={props.onTestSound}
        aria-label="Test piano sound"
        class="px-3 py-1.5 rounded-lg text-sm font-medium bg-zinc-700 text-zinc-300 hover:bg-zinc-600 transition-colors cursor-pointer"
      >
        Test sound
      </button>

      <audio
        ref={(audio) => { props.audioRef.current = audio }}
        preload="auto"
        onPlaying={props.onPlaying}
        onWaiting={props.onWaiting}
        onPause={props.onPause}
        onEnded={props.onEnded}
        onTimeUpdate={props.onTimeUpdate}
        onSeeked={props.onSeeked}
        onLoadedMetadata={props.onLoadedMetadata}
      />
    </section>
  )
}
