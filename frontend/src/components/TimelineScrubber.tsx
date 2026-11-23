import { createMemo, For } from "solid-js"

const TimelineScrubberGrid = (props: {
  totalBeats: number
  beatToTime: (beat: number) => number
  pixelsPerBeat: number
  visibleLeft: number
  visibleWidth: number
}) => {
  const labelForMeasure = (measure: number): string => {
    const seconds = Math.floor(props.beatToTime(measure * 16))
    return `${Math.floor(seconds / 60)}:${(seconds % 60).toString().padStart(2, "0")}`
  }
  const visibleMeasures = createMemo(() => {
    const measureWidth = 16 * props.pixelsPerBeat
    const start = Math.max(0, Math.floor(props.visibleLeft / measureWidth))
    const end = Math.min(
      Math.floor(props.totalBeats / 16),
      Math.ceil((props.visibleLeft + props.visibleWidth) / measureWidth),
    )
    const measures: number[] = []
    for (let measure = start; measure <= end; measure++) {
      if (measure === 0 || labelForMeasure(measure) !== labelForMeasure(measure - 1)) {
        measures.push(measure)
      }
    }
    return measures
  })
  return (
    <div
      class="pointer-events-none absolute inset-0"
      aria-hidden="true"
      style={{
        "background-image": [
          "linear-gradient(to right, #71717a 0, #71717a 1.5px, transparent 1.5px)",
          "linear-gradient(to right, #52525b 0, #52525b 1px, transparent 1px)",
          "linear-gradient(to right, #3f3f46 0, #3f3f46 1px, transparent 1px)",
        ].join(", "),
        "background-size": `${props.pixelsPerBeat * 16}px 100%, ${props.pixelsPerBeat * 4}px 100%, ${props.pixelsPerBeat}px 100%`,
        "background-repeat": "repeat",
      }}
    >
      <For each={visibleMeasures()}>
        {(measure) => (
          <span
            class="absolute top-1 left-1 text-[10px] text-zinc-500 tabular-nums"
            style={{ left: `${measure * 16 * props.pixelsPerBeat}px` }}
          >
            {labelForMeasure(measure)}
          </span>
        )}
      </For>
    </div>
  )
}

interface TimelineScrubberTrackProps {
  timelineRef: { current: HTMLDivElement | null }
  onTrackReady?: (node: HTMLDivElement | null) => void
  width: number
  totalDuration: number
  displayTime: number
  onTimelineClick: (e: MouseEvent & { currentTarget: HTMLDivElement }) => void
  onMouseDown: (e: MouseEvent & { currentTarget: HTMLDivElement }) => void
  onSeekKeyDown: (e: KeyboardEvent & { currentTarget: HTMLDivElement }) => void
  children?: import("@solidjs/web").JSX.Element
}

const TimelineScrubberTrack = (props: TimelineScrubberTrackProps) => {
  const accessibleTime = createMemo(() => Math.round(props.displayTime * 10) / 10)
  return (
    <div
      ref={(node) => {
        props.timelineRef.current = node
        props.onTrackReady?.(node)
      }}
      onClick={props.onTimelineClick}
      onMouseDown={props.onMouseDown}
      class="relative h-12 cursor-pointer select-none"
      style={{ width: `${props.width}px` }}
      role="slider"
      aria-label="Seek position"
      aria-valuemin={0}
      aria-valuemax={props.totalDuration}
      aria-valuenow={accessibleTime()}
      tabindex={0}
      onKeyDown={props.onSeekKeyDown}
    >
      {props.children}
    </div>
  )
}

interface TimelineScrubberProps {
  totalBeats: number
  totalDuration: number
  playheadX: number
  playheadVisible: boolean
  displayTime: number
  beatToTime: (beat: number) => number
  pixelsPerBeat: number
  visibleLeft: number
  visibleWidth: number
  timelineRef: { current: HTMLDivElement | null }
  onTrackReady?: (node: HTMLDivElement | null) => void
  onTimelineClick: (e: MouseEvent & { currentTarget: HTMLDivElement }) => void
  onMouseDown: (e: MouseEvent & { currentTarget: HTMLDivElement }) => void
  onSeekKeyDown: (e: KeyboardEvent & { currentTarget: HTMLDivElement }) => void
}

export const TimelineScrubber = (props: TimelineScrubberProps) => {
  return (
    <section class="border-b border-zinc-700/50 bg-zinc-800/60" aria-label="Timeline">
      <TimelineScrubberTrack
        timelineRef={props.timelineRef}
        onTrackReady={props.onTrackReady}
        width={props.totalBeats * props.pixelsPerBeat}
        totalDuration={props.totalDuration}
        displayTime={props.displayTime}
        onTimelineClick={props.onTimelineClick}
        onMouseDown={props.onMouseDown}
        onSeekKeyDown={props.onSeekKeyDown}
      >
        <TimelineScrubberGrid
          totalBeats={props.totalBeats}
          beatToTime={props.beatToTime}
          pixelsPerBeat={props.pixelsPerBeat}
          visibleLeft={props.visibleLeft}
          visibleWidth={props.visibleWidth}
        />
        <div
          class="pointer-events-none absolute bottom-0 left-0 top-0 z-10 w-0.5 bg-pink-500"
          style={{
            transform: `translateX(${props.playheadX}px)`,
            visibility: props.playheadVisible ? "visible" : "hidden",
          }}
          aria-hidden="true"
        >
          <div class="absolute -top-1 left-1/2 h-3 w-3 -translate-x-1/2 rounded-full border-2 border-zinc-900 bg-pink-500" />
        </div>
      </TimelineScrubberTrack>
    </section>
  )
}
