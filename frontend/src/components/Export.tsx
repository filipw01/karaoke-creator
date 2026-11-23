import type { JobResult } from "../types"
import { getDownloadUrl } from "../api"

interface ExportProps {
  jobId: string
  result: JobResult
  onBack: () => void
  onStartOver: () => void
}

export const Export = ({ jobId, result, onBack, onStartOver }: ExportProps) => {
  const downloadUrl = getDownloadUrl(jobId)

  return (
    <div class="max-w-xl mx-auto text-center space-y-8">
      <div>
        <div class="w-20 h-20 mx-auto mb-4 bg-emerald-500/15 rounded-full flex items-center justify-center">
          <svg class="w-10 h-10 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width={2}>
            <path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7" />
          </svg>
        </div>
        <h2 class="text-3xl font-bold text-white mb-2">Your Karaoke Song is Ready!</h2>
        <p class="text-zinc-400">
          {result.metadata.artist} — {result.metadata.title}
        </p>
      </div>

      {/* What's included */}
      <div class="bg-zinc-800/60 border border-zinc-700/50 rounded-xl p-6 text-left">
        <h3 class="text-white font-medium mb-3">Package contents:</h3>
        <ul class="space-y-2 text-sm">
          <li class="flex items-center gap-2 text-zinc-300">
            <svg class="w-4 h-4 text-violet-400 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width={2}>
              <path stroke-linecap="round" stroke-linejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <span class="font-mono">{result.metadata.artist} - {result.metadata.title}.txt</span>
            <span class="text-zinc-500 ml-auto">UltraStar TXT</span>
          </li>
          <li class="flex items-center gap-2 text-zinc-300">
            <svg class="w-4 h-4 text-fuchsia-400 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width={2}>
              <path stroke-linecap="round" stroke-linejoin="round" d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
            </svg>
            <span class="font-mono">{result.audio_file}</span>
            <span class="text-zinc-500 ml-auto">Full mix</span>
          </li>
          <li class="flex items-center gap-2 text-zinc-300">
            <svg class="w-4 h-4 text-emerald-400 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width={2}>
              <path stroke-linecap="round" stroke-linejoin="round" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
            </svg>
            <span class="font-mono">{result.vocals_file}</span>
            <span class="text-zinc-500 ml-auto">Vocals only</span>
          </li>
          <li class="flex items-center gap-2 text-zinc-300">
            <svg class="w-4 h-4 text-amber-400 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width={2}>
              <path stroke-linecap="round" stroke-linejoin="round" d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2z" />
            </svg>
            <span class="font-mono">{result.instrumental_file}</span>
            <span class="text-zinc-500 ml-auto">Instrumental</span>
          </li>
          {result.video_file && (
            <li class="flex items-center gap-2 text-zinc-300">
              <svg class="w-4 h-4 text-sky-400 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width={2}>
                <path stroke-linecap="round" stroke-linejoin="round" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
              <span class="font-mono">{result.video_file}</span>
              <span class="text-zinc-500 ml-auto">Video</span>
            </li>
          )}
        </ul>
      </div>

      {/* Song stats */}
      <div class="grid grid-cols-3 gap-3">
        <div class="bg-zinc-800/60 border border-zinc-700/50 rounded-xl p-4">
          <div class="text-2xl font-bold text-white">{result.metadata.bpm.toFixed(0)}</div>
          <div class="text-zinc-500 text-xs">BPM</div>
        </div>
        <div class="bg-zinc-800/60 border border-zinc-700/50 rounded-xl p-4">
          <div class="text-2xl font-bold text-white">{result.phrases.length}</div>
          <div class="text-zinc-500 text-xs">Phrases</div>
        </div>
        <div class="bg-zinc-800/60 border border-zinc-700/50 rounded-xl p-4">
          <div class="text-2xl font-bold text-white">
            {result.phrases.reduce((sum, p) => sum + p.notes.length, 0)}
          </div>
          <div class="text-zinc-500 text-xs">Notes</div>
        </div>
      </div>

      {/* Actions */}
      <div class="flex flex-col sm:flex-row gap-3 justify-center">
        <button
          onClick={onBack}
          class="bg-zinc-700 hover:bg-zinc-600 text-white px-6 py-3 rounded-xl
                     transition-colors cursor-pointer"
        >
          Back to Editor
        </button>
        <a
          href={downloadUrl}
          download
          class="bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-500 hover:to-fuchsia-500
                     text-white font-medium px-8 py-3 rounded-xl transition-all inline-block"
        >
          Download ZIP
        </a>
        <button
          onClick={onStartOver}
          class="bg-zinc-700 hover:bg-zinc-600 text-white px-6 py-3 rounded-xl
                     transition-colors cursor-pointer"
        >
          New Song
        </button>
      </div>

      {/* Instructions */}
      <div class="bg-zinc-800/40 border border-zinc-700/30 rounded-xl p-4 text-left text-sm text-zinc-500">
        <p class="font-medium text-zinc-400 mb-2">How to use in UltraStar Deluxe:</p>
        <ol class="list-decimal list-inside space-y-1">
          <li>Extract the ZIP into your UltraStar songs folder</li>
          <li>Each song should be in its own subfolder</li>
          <li>Launch UltraStar Deluxe and the song should appear in the song list</li>
          <li>For best results, review the timing in a song editor like Yass or Karedi</li>
        </ol>
      </div>
    </div>
  )
}
