import { memo } from "react"
import type { LyricsParams } from "../types"

interface LyricsSettingsPanelProps {
  params: LyricsParams
  onParamChange: <TKey extends keyof LyricsParams>(key: TKey, value: LyricsParams[TKey]) => void
  onRegenerate: () => void
  regenerating: boolean
  regenerateError: string | null
}

export const LyricsSettingsPanel = memo(function LyricsSettingsPanel({
  params,
  onParamChange,
  onRegenerate,
  regenerating,
  regenerateError,
}: LyricsSettingsPanelProps) {
  return (
    <section
      className="bg-zinc-800/60 border border-zinc-700/50 rounded-xl p-5 space-y-4"
      aria-label="Lyrics extraction settings"
    >
      <div className="flex items-center justify-between">
        <h3 className="text-white font-medium">Lyrics Extraction Settings</h3>
        <button
          type="button"
          onClick={onRegenerate}
          disabled={regenerating}
          aria-busy={regenerating}
          aria-label={regenerating ? "Regenerating lyrics" : "Regenerate lyrics with current settings"}
          className="bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-500 hover:to-fuchsia-500
                     text-white font-medium px-5 py-2 rounded-lg transition-all disabled:opacity-50 cursor-pointer"
        >
          {regenerating ? "Regenerating..." : "Regenerate Lyrics"}
        </button>
      </div>
      {regenerateError !== null && (
        <p className="text-sm text-red-400 bg-red-950/30 border border-red-500/30 rounded-lg px-3 py-2" role="alert">
          {regenerateError}
        </p>
      )}

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-xs text-zinc-400 mb-1">Alignment Mode</label>
          <select
            value={params.alignment_mode}
            onChange={(e) =>
              onParamChange("alignment_mode", e.target.value as LyricsParams["alignment_mode"])
            }
            className="w-full bg-zinc-900 border border-zinc-600 rounded-lg px-3 py-2 text-white text-sm
                       focus:outline-none focus:border-violet-500"
          >
            <option value="forcealign">ForceAlign (English, pip)</option>
            <option value="lyrics_aligner">lyrics-aligner (phoneme DNN, English; LYRICS_ALIGNER_ROOT)</option>
            <option value="nemo">NeMo (ASR timestamps)</option>
            <option value="whisperx">WhisperX (fixed transcript)</option>
            <option value="aeneas">aeneas (pip)</option>
            <option value="mfa">Montreal Forced Aligner (MFA)</option>
            <option value="gentle">Gentle (Docker localhost:8765)</option>
            <option value="autolyrixalign">AutoLyrixAlign (AUTOLYRIXALIGN_ROOT)</option>
            <option value="whisper">Whisper (transcription fallback)</option>
          </select>
          <p className="text-[10px] text-zinc-500 mt-1">
            Choose alignment method then click Regenerate. All aligners run; pick one to compare.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
        <div>
          <label className="block text-xs text-zinc-400 mb-1">Audio Source</label>
          <select
            value={params.audio_source}
            onChange={(e) => onParamChange("audio_source", e.target.value)}
            className="w-full bg-zinc-900 border border-zinc-600 rounded-lg px-3 py-2 text-white text-sm
                       focus:outline-none focus:border-violet-500"
          >
            <option value="vocals">Separated vocals</option>
            <option value="original">Original mix</option>
          </select>
          <p className="text-[10px] text-zinc-500 mt-1">Try &quot;Original mix&quot; if vocals produce bad results</p>
        </div>
        <div>
          <label className="block text-xs text-zinc-400 mb-1">Model Size</label>
          <select
            value={params.model_size}
            onChange={(e) => onParamChange("model_size", e.target.value)}
            className="w-full bg-zinc-900 border border-zinc-600 rounded-lg px-3 py-2 text-white text-sm
                       focus:outline-none focus:border-violet-500"
          >
            <option value="tiny">Tiny (fastest)</option>
            <option value="base">Base</option>
            <option value="small">Small</option>
            <option value="medium">Medium (default)</option>
            <option value="large-v2">Large v2</option>
            <option value="large-v3">Large v3 (best)</option>
          </select>
        </div>
        <div>
          <label className="block text-xs text-zinc-400 mb-1">Language (blank = auto)</label>
          <input
            type="text"
            value={params.language}
            onChange={(e) => onParamChange("language", e.target.value)}
            placeholder="en, ja, ko..."
            className="w-full bg-zinc-900 border border-zinc-600 rounded-lg px-3 py-2 text-white text-sm
                       focus:outline-none focus:border-violet-500"
          />
        </div>
        <div className="flex flex-col justify-end">
          <label className="flex items-center gap-2 cursor-pointer py-2">
            <input
              type="checkbox"
              checked={params.normalize}
              onChange={(e) => onParamChange("normalize", e.target.checked)}
              className="accent-violet-500"
            />
            <span className="text-sm text-zinc-300">Normalize volume</span>
          </label>
          <p className="text-[10px] text-zinc-500">EBU R128 loudness normalization</p>
        </div>
      </div>

      <div className="border-t border-zinc-700/50 pt-4">
        <div className="flex items-center gap-3 mb-3">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={params.vad_filter}
              onChange={(e) => onParamChange("vad_filter", e.target.checked)}
              className="accent-violet-500"
            />
            <span className="text-sm text-zinc-300">Voice Activity Detection (VAD)</span>
          </label>
          <span className="text-xs text-zinc-500">Filters silence/noise but may cut beginnings</span>
        </div>

        {params.vad_filter && (
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
            <div>
              <label className="block text-xs text-zinc-400 mb-1">
                VAD Threshold ({params.vad_threshold.toFixed(2)})
              </label>
              <input
                type="range"
                min={0.05}
                max={0.95}
                step={0.05}
                value={params.vad_threshold}
                onChange={(e) => onParamChange("vad_threshold", Number.parseFloat(e.target.value))}
                className="w-full accent-violet-500"
              />
              <div className="flex justify-between text-[10px] text-zinc-500">
                <span>More sensitive</span>
                <span>Less sensitive</span>
              </div>
            </div>
            <div>
              <label className="block text-xs text-zinc-400 mb-1">Min Silence (ms)</label>
              <input
                type="number"
                min={50}
                max={2000}
                step={50}
                value={params.vad_min_silence_ms}
                onChange={(e) =>
                  onParamChange("vad_min_silence_ms", Number.parseInt(e.target.value) ?? 200)
                }
                className="w-full bg-zinc-900 border border-zinc-600 rounded-lg px-3 py-2 text-white text-sm
                           focus:outline-none focus:border-violet-500"
              />
            </div>
            <div>
              <label className="block text-xs text-zinc-400 mb-1">Speech Pad (ms)</label>
              <input
                type="number"
                min={0}
                max={1000}
                step={50}
                value={params.speech_pad_ms}
                onChange={(e) =>
                  onParamChange("speech_pad_ms", Number.parseInt(e.target.value) ?? 0)
                }
                className="w-full bg-zinc-900 border border-zinc-600 rounded-lg px-3 py-2 text-white text-sm
                           focus:outline-none focus:border-violet-500"
              />
            </div>
          </div>
        )}
      </div>

      <details className="border-t border-zinc-700/50 pt-4">
        <summary className="text-sm text-zinc-400 cursor-pointer hover:text-zinc-300 transition-colors">
          Advanced parameters
        </summary>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 mt-3">
          <div>
            <label className="block text-xs text-zinc-400 mb-1">Beam Size</label>
            <input
              type="number"
              min={1}
              max={10}
              value={params.beam_size}
              onChange={(e) =>
                onParamChange("beam_size", Math.max(1, Number.parseInt(e.target.value) ?? 1))
              }
              className="w-full bg-zinc-900 border border-zinc-600 rounded-lg px-3 py-2 text-white text-sm
                         focus:outline-none focus:border-violet-500"
            />
          </div>
          <div>
            <label className="block text-xs text-zinc-400 mb-1">No Speech Threshold</label>
            <input
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={params.no_speech_threshold}
              onChange={(e) =>
                onParamChange("no_speech_threshold", Number.parseFloat(e.target.value) ?? 0)
              }
              className="w-full bg-zinc-900 border border-zinc-600 rounded-lg px-3 py-2 text-white text-sm
                         focus:outline-none focus:border-violet-500"
            />
          </div>
          <div>
            <label className="block text-xs text-zinc-400 mb-1">Min Word Confidence</label>
            <input
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={params.min_word_confidence}
              onChange={(e) =>
                onParamChange("min_word_confidence", Number.parseFloat(e.target.value) ?? 0)
              }
              className="w-full bg-zinc-900 border border-zinc-600 rounded-lg px-3 py-2 text-white text-sm
                         focus:outline-none focus:border-violet-500"
            />
          </div>
        </div>
      </details>
    </section>
  )
})
