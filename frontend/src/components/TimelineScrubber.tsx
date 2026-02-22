import { memo } from "react"

const PIXELS_PER_BEAT = 8

const TimelineScrubberPlayhead = memo(function TimelineScrubberPlayhead({
  currentBeat,
  displayTime,
  totalBeats,
}: {
  currentBeat: number
  displayTime: number
  totalBeats: number
}) {
  if (displayTime < 0 || currentBeat < 0 || currentBeat > totalBeats) return null
  return (
    <div
      className="absolute top-0 bottom-0 w-0.5 bg-pink-500 pointer-events-none z-10 transition-none"
      style={{ left: currentBeat * PIXELS_PER_BEAT }}
    >
      <div className="absolute -top-1 left-1/2 -translate-x-1/2 w-3 h-3 bg-pink-500 rounded-full border-2 border-zinc-900" />
    </div>
  )
})

const TimelineScrubberGrid = memo(function TimelineScrubberGrid({
  totalBeats,
  beatToTime,
}: {
  totalBeats: number
  beatToTime: (beat: number) => number
}) {
  return (
    <>
      {Array.from({ length: Math.ceil(totalBeats / 4) + 1 }).map((_, i) => {
        const beat = i * 4
        const x = beat * PIXELS_PER_BEAT
        const time = beatToTime(beat)
        return (
          <div
            key={beat}
            className="absolute top-0 bottom-0 border-l border-zinc-700/50"
            style={{ left: x }}
          >
            {i % 2 === 0 && time >= 0 && (
              <span className="absolute top-1 left-1 text-[10px] text-zinc-500 tabular-nums">
                {Math.floor(time / 60)}:{(Math.floor(time % 60)).toString().padStart(2, "0")}
              </span>
            )}
          </div>
        )
      })}
    </>
  )
})

interface TimelineScrubberTrackProps {
  timelineRef: React.RefObject<HTMLDivElement | null>
  width: number
  totalDuration: number
  displayTime: number
  onTimelineClick: (e: React.MouseEvent<HTMLDivElement>) => void
  onMouseDown: (e: React.MouseEvent<HTMLDivElement>) => void
  onMouseUp: () => void
  onMouseLeave: () => void
  onSeekKeyDown: (e: React.KeyboardEvent<HTMLDivElement>) => void
  children: React.ReactNode
}

const TimelineScrubberTrack = memo(function TimelineScrubberTrack({
  timelineRef,
  width,
  totalDuration,
  displayTime,
  onTimelineClick,
  onMouseDown,
  onMouseUp,
  onMouseLeave,
  onSeekKeyDown,
  children,
}: TimelineScrubberTrackProps) {
  return (
    <div
      ref={timelineRef}
      onClick={onTimelineClick}
      onMouseDown={onMouseDown}
      onMouseUp={onMouseUp}
      onMouseLeave={onMouseLeave}
      className="relative h-12 cursor-pointer select-none"
      style={{ width }}
      role="slider"
      aria-label="Seek position"
      aria-valuemin={0}
      aria-valuemax={totalDuration}
      aria-valuenow={displayTime}
      tabIndex={0}
      onKeyDown={onSeekKeyDown}
    >
      {children}
    </div>
  )
})

interface TimelineScrubberProps {
  totalBeats: number
  totalDuration: number
  currentBeat: number
  displayTime: number
  beatToTime: (beat: number) => number
  timelineRef: React.RefObject<HTMLDivElement | null>
  onTimelineClick: (e: React.MouseEvent<HTMLDivElement>) => void
  onMouseDown: (e: React.MouseEvent<HTMLDivElement>) => void
  onMouseUp: () => void
  onMouseLeave: () => void
  onSeekKeyDown: (e: React.KeyboardEvent<HTMLDivElement>) => void
}

export const TimelineScrubber = memo(function TimelineScrubber({
  totalBeats,
  totalDuration,
  currentBeat,
  displayTime,
  beatToTime,
  timelineRef,
  onTimelineClick,
  onMouseDown,
  onMouseUp,
  onMouseLeave,
  onSeekKeyDown,
}: TimelineScrubberProps) {
  return (
    <section className="border-b border-zinc-700/50 bg-zinc-800/60" aria-label="Timeline">
      <TimelineScrubberTrack
        timelineRef={timelineRef}
        width={totalBeats * PIXELS_PER_BEAT}
        totalDuration={totalDuration}
        displayTime={displayTime}
        onTimelineClick={onTimelineClick}
        onMouseDown={onMouseDown}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseLeave}
        onSeekKeyDown={onSeekKeyDown}
      >
        <TimelineScrubberGrid totalBeats={totalBeats} beatToTime={beatToTime} />
        <TimelineScrubberPlayhead
          currentBeat={currentBeat}
          displayTime={displayTime}
          totalBeats={totalBeats}
        />
      </TimelineScrubberTrack>
    </section>
  )
})
