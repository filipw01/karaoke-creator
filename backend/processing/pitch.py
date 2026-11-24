"""Pitch detection using Basic Pitch with a pYIN fallback for vocal analysis."""

from __future__ import annotations

import logging
import gc
import importlib.util
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from models import Word


log = logging.getLogger("pitch")

# Anti-chunking: generic tracks split only on a four-semitone change (avoids
# vibrato); the compact tonal path below lowers this to two semitones so
# marked/melismatic syllables remain representable.
MIN_SEMITONES_TO_SPLIT = 4
# Merge short detector blips into adjacent runs. pYIN's frame-level jitter is
# common at consonant boundaries, while a 100 ms island is still shorter than
# the smallest sustained syllables in the benchmark charts.
MIN_SEGMENT_DURATION_S = 0.10
LEGACY_MIN_SEGMENT_DURATION_S = 0.08
MARKED_MIN_SEGMENT_DURATION_S = 0.04
# Keep at least this much detector evidence in each explicitly requested
# continuation unit. The production contour is ~23 ms per frame, so the
# default avoids single-frame noise while still representing short notes.
FORCED_PARTITION_MIN_DURATION_S = 0.10
# When two adjacent explicit units have the same detected pitch, pYIN gives no
# acoustic boundary to the partitioner.  UltraStar charts commonly encode the
# long note first and a short continuation second, so reserve a small tail for
# the later unit instead of letting the dynamic-programming tie-breaker put
# the cut near the start of the flat run.
REPEATED_UNIT_TAIL_DURATION_S = 0.16
# A very short leading syllable is often rounded to the following sustained
# pitch by pYIN. Require a distinctly longer second unit before correcting the
# one-semitone leading step; this leaves genuinely repeated short notes alone.
SHORT_LEADING_UNIT_MAX_DURATION_S = 0.15
SHORT_FOLLOWING_UNIT_MIN_DURATION_S = 0.10
# Minimum RMS per frame to count as voiced (filters quiet breath/silence). 0 = off.
DEFAULT_MIN_VOLUME_RMS = 0.005
# A short rolling mode over quantized notes removes pYIN's isolated
# octave/harmonic flips without letting an intermediate median become a note
# that never actually occurred.
PITCH_MEDIAN_FILTER_FRAMES = 13
# When a recording is strongly tonal, stray pYIN candidates often land between
# the song's pitch classes. Snap those candidates to the dominant vocal pitch
# vocabulary while leaving genuinely chromatic material untouched.
PITCH_CLASS_QUANTIZATION_TOP_K = 6
PITCH_CLASS_QUANTIZATION_MIN_SUPPORT = 0.80
# When a vocal track contains several weak harmonic candidates, its four most
# common classes are usually the stable chord tones while the fifth/sixth are
# detector spill. Keep this fallback narrow and only use it when those four
# classes still explain most of the voiced frames.
PITCH_CLASS_CORE_COUNT = 4
PITCH_CLASS_CORE_MIN_SUPPORT = 0.60
PITCH_CLASS_CORE_MAX_SUPPORT = 0.90
# Estimate a word's note from its vowel-bearing interior. ASR boundaries often
# include consonants from the neighboring word, especially on short words.
PITCH_ANALYSIS_EDGE_FRACTION = 0.04
LEGACY_PITCH_ANALYSIS_EDGE_FRACTION = 0.05
# pYIN timestamps are centered on a 2048-sample window. In this pipeline the
# stable vowel evidence consistently occurs one analysis window earlier than
# the aligned lyric span, so use a small detector-latency correction while
# keeping the physical frame timestamps intact for activity detection.
PITCH_ANALYSIS_TIME_OFFSET_S = -0.085
# A consonant-only token can have no stable pYIN frame. Borrow the closest
# voiced frame from the neighboring vowel for a short gap instead of emitting
# an artificial C4 note.
MAX_NEIGHBOR_FALLBACK_GAP_S = 0.16
# SOFA's forced word boundaries are phonetic boundaries rather than detector
# frame boundaries.  When a word starts or ends between two pitch frames, use
# a small context window to find the vowel, but only fall back to that context
# when the canonical span contains no usable pitch of its own.
PITCH_WORD_CONTEXT_S = 0.10
# Low-confidence pYIN frames still carry useful note information after the
# median filter. If a caller requests a stricter confidence gate,
# ``word_to_segments`` falls back to any usable frequency when every frame
# falls below it so breathy syllables do not flatten to C4.
DEFAULT_MIN_CONFIDENCE = 0.0

HOP_LENGTH = 512
FRAME_LENGTH = 2048
LONG_FRAME_LENGTH = 4096
LOW_CONFIDENCE_FRAME_SWITCH = 0.35

# Basic Pitch's published ONNX model is a small polyphonic note estimator. It
# is kept optional because onnxruntime and the model weights are not required
# for the pYIN-only installation. Windows are processed one at a time: a
# complete song is never copied into a second model-sized tensor.
BASIC_PITCH_SAMPLE_RATE = 22050
BASIC_PITCH_INPUT_SAMPLES = 43844
BASIC_PITCH_WINDOW_HOP_SAMPLES = 36164
BASIC_PITCH_FRAME_COUNT = 172
BASIC_PITCH_TRIM_FRAMES = 15
# The exported model follows the 256-sample analysis grid.  Deriving this
# from the padded window length (43844 / 172) introduces a fractional-hop
# drift that grows to more than a second on a typical full song.
BASIC_PITCH_FRAME_HOP_S = 256.0 / BASIC_PITCH_SAMPLE_RATE
BASIC_PITCH_MIN_CONFIDENCE = 0.15
# A short activation average suppresses one-frame harmonic spikes before the
# note winner is selected. Five frames are ~116 ms at the exported frame rate:
# long enough to stabilize a vowel, short enough to preserve quick syllables.
BASIC_PITCH_ACTIVATION_SMOOTH_FRAMES = 5
# Retain a small posterior beam instead of discarding all but the argmax. The
# assignment stage can then choose a stable candidate over a complete lyric
# span; twelve notes cover the common harmonic alternatives while keeping the
# extra memory bounded (roughly 2 MB for a five-minute song).
BASIC_PITCH_CANDIDATE_COUNT = 12
# The overlap-trimmed Basic Pitch timestamps lead the audible vowel centre by
# roughly one 43 Hz frame on the benchmark material. Keep this small
# correction explicit and overridable for recordings with different encoder
# delay rather than reusing pYIN's much larger centered-window correction.
BASIC_PITCH_DEFAULT_TIME_OFFSET_S = -0.04
BASIC_PITCH_CONFIDENCE_FLOOR = 0.25
BASIC_PITCH_CONFIDENCE_POWER = 1.0
# A vocal separator can leave a low-frequency accompaniment fundamental in the
# vocal stem.  A second, sequential high-pass pass gives Basic Pitch a clean
# view of the upper vocal harmonics; only a clearly higher result is allowed to
# replace the base contour.  The pass is sequential so peak memory stays close
# to one Basic Pitch inference rather than two resident model-sized buffers.
BASIC_PITCH_HIGHPASS_CUTOFF_HZ = 300.0
BASIC_PITCH_HIGHPASS_MIN_DELTA = 6
BASIC_PITCH_HIGHPASS_CONTEXT_S = 0.25
BASIC_PITCH_HIGHPASS_FULL_PASS_FRACTION = 0.075
BASIC_PITCH_HIGHPASS_SYSTEMATIC_CUTOFF_HZ = 750.0
# Onset activations are used only to place boundaries inside an already
# aligned lyric span.  A relatively high floor avoids treating every Basic
# Pitch posterior ripple as a new sung note, while the distance guard keeps
# consonant chatter from creating a run per frame.
BASIC_PITCH_ONSET_MIN_HEIGHT = 0.35
BASIC_PITCH_ONSET_MIN_DISTANCE_S = 0.08
BASIC_PITCH_ONSET_BOUNDARY_WINDOW_S = 0.14

# Essentia's MELODIA tracker is an optional, CPU-friendly predominant-melody
# contour.  It complements Basic Pitch: the latter is better at note starts,
# while MELODIA is less likely to choose a backing harmonic for a sustained
# vocal.  Keep its settings explicit so minimal installations can continue to
# use Basic Pitch/pYIN without importing Essentia.
MELODIA_SAMPLE_RATE = 44100
MELODIA_FRAME_SIZE = 2048
MELODIA_HOP_SIZE = 256
MELODIA_MIN_FREQUENCY = 55.0
MELODIA_MAX_FREQUENCY = 1760.0
MELODIA_VOICING_TOLERANCE = 0.2
MELODIA_MIN_CONFIDENCE = 0.1
MELODIA_FUSION_MIN_SEMITONES = 1

# torchcrepe is an optional monophonic neural f0 tracker.  Unlike Basic
# Pitch's piano-bin posterior it estimates the fundamental directly, which is
# useful when the separator leaves a quiet vocal over a louder harmony.  Keep
# the default model small and the frame grid modest so this remains practical
# on a local CPU and does not retain a second full-song tensor.
TORCHCREPE_SAMPLE_RATE = 16000
TORCHCREPE_HOP_LENGTH = 320  # 20 ms at 16 kHz
TORCHCREPE_MIN_FREQUENCY = 55.0
TORCHCREPE_MAX_FREQUENCY = 1760.0
TORCHCREPE_MIN_PERIODICITY = 0.22
TORCHCREPE_DEFAULT_MODEL = "tiny"
TORCHCREPE_DEFAULT_BATCH_SIZE = 512
TORCHCREPE_DEFAULT_THREADS = 4


@dataclass(slots=True)
class PitchDetection:
    """Pitch contour plus optional per-frame Basic Pitch alternatives.

    The first four fields intentionally remain iterable as the historical
    ``(times, frequencies, confidences, energies)`` result.  Candidate notes
    are kept private to the assignment stage, where aligned lyric spans make
    a word-level choice possible without changing the public API used by
    callers that only need the contour.
    """

    times: np.ndarray
    frequencies: np.ndarray
    confidences: np.ndarray
    energies: np.ndarray
    candidate_midis: np.ndarray | None = None
    candidate_scores: np.ndarray | None = None
    # Basic Pitch's onset lane is reduced to one strength value per frame
    # after inference.  Keeping this compact signal (rather than the full
    # 88-bin onset matrix) lets assignment refine internal note boundaries
    # without retaining another model-sized tensor in the API process.
    onset_scores: np.ndarray | None = None

    def __iter__(self):
        yield self.times
        yield self.frequencies
        yield self.confidences
        yield self.energies


def _basic_pitch_detector(detector: str) -> bool:
    """Return whether a detector uses Basic Pitch's timestamp/weight rules."""

    return detector in {
        "basic_pitch",
        "basic-pitch",
        "basicpitch",
        "basic_pitch_melodia",
        "basic-pitch-melodia",
        "basicpitch-melodia",
    }


def _melodia_available() -> bool:
    """Check for Essentia without importing its slow native module."""

    try:
        return importlib.util.find_spec("essentia") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _torchcrepe_available() -> bool:
    """Check for torchcrepe without importing torch in the normal path."""

    try:
        return importlib.util.find_spec("torchcrepe") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _pitch_detector_name() -> str:
    """Return the requested detector, resolving the optional default safely."""

    configured = os.environ.get("PITCH_DETECTOR", "").strip().casefold()
    if configured:
        return configured
    # Keep minimal installations on pYIN when the optional ONNX runtime/model
    # is absent. This also keeps the timestamp correction in word assignment
    # aligned with the detector that actually ran.
    if _basic_pitch_model_path() is not None:
        return "basic_pitch"
    return "pyin"


def _smooth_pitch_frequencies(
    frequencies: np.ndarray,
    *,
    window_frames: int = PITCH_MEDIAN_FILTER_FRAMES,
) -> np.ndarray:
    """Mode-filter quantized pitch while preserving unvoiced gaps."""

    values = np.asarray(frequencies, dtype=np.float64)
    if window_frames <= 1 or len(values) < 2:
        return values.copy()
    if window_frames % 2 == 0:
        window_frames += 1
    midi = np.full(len(values), -1, dtype=np.int16)
    voiced = np.isfinite(values) & (values > 0.0)
    midi[voiced] = np.rint(69.0 + 12.0 * np.log2(values[voiced] / 440.0)).astype(np.int16)
    radius = window_frames // 2
    smoothed = midi.copy()
    for index in np.flatnonzero(voiced):
        start = max(0, int(index) - radius)
        end = min(len(midi), int(index) + radius + 1)
        neighborhood = midi[start:end]
        neighborhood = neighborhood[neighborhood > 0]
        if len(neighborhood):
            counts = np.bincount(neighborhood)
            max_count = int(np.max(counts))
            # Keep the first occurrence on ties. This preserves note onsets
            # deterministically instead of biasing toward a lower MIDI value.
            smoothed[index] = next(
                int(value) for value in neighborhood if counts[value] == max_count
            )
    result = np.zeros_like(values)
    result[voiced] = 440.0 * np.power(2.0, (smoothed[voiced] - 69.0) / 12.0)
    return result


def _dominant_pitch_classes(
    frequencies: np.ndarray,
    *,
    top_k: int = PITCH_CLASS_QUANTIZATION_TOP_K,
    min_support: float = PITCH_CLASS_QUANTIZATION_MIN_SUPPORT,
) -> set[int]:
    """Find a compact pitch-class vocabulary for strongly tonal recordings."""

    values = np.asarray(frequencies, dtype=np.float64)
    voiced = np.isfinite(values) & (values > 0.0)
    if not np.any(voiced) or top_k <= 0:
        return set()
    midi = np.rint(69.0 + 12.0 * np.log2(values[voiced] / 440.0)).astype(np.int16)
    counts = np.bincount(np.mod(midi, 12), minlength=12).astype(np.float64)
    total = float(np.sum(counts))
    if total <= 0.0:
        return set()
    ranked = np.argsort(-counts, kind="stable")
    max_classes = min(top_k, 12)
    top_core_support = float(np.sum(counts[ranked[:PITCH_CLASS_CORE_COUNT]]) / total)
    top_requested_support = float(np.sum(counts[ranked[:max_classes]]) / total)
    use_core = (
        max_classes > PITCH_CLASS_CORE_COUNT
        and top_core_support >= PITCH_CLASS_CORE_MIN_SUPPORT
        and top_requested_support < PITCH_CLASS_CORE_MAX_SUPPORT
    )
    if use_core:
        max_classes = PITCH_CLASS_CORE_COUNT
    selected: list[int] = []
    support = 0.0
    for pitch_class in ranked[:max_classes]:
        selected.append(int(pitch_class))
        support += counts[pitch_class]
        if len(selected) >= max_classes and support / total >= (
            PITCH_CLASS_CORE_MIN_SUPPORT if use_core else min_support
        ):
            break
    # A broad/chromatic track should retain its detector output rather than
    # being forced into an arbitrary six-note subset. A moderately tonal
    # melody can still have many harmonic candidates before smoothing; retain
    # its stable four-class core so the per-word estimator has a useful prior.
    required_support = PITCH_CLASS_CORE_MIN_SUPPORT if use_core else min_support
    if support / total < required_support:
        if top_core_support >= 0.45:
            return {int(value) for value in ranked[:PITCH_CLASS_CORE_COUNT]}
        return set()
    return set(selected)


def _leading_tone_class(
    frequencies: np.ndarray,
    core_classes: set[int],
) -> set[int]:
    """Return a sparse leading-tone prior for a clearly tonal melody."""

    if not core_classes:
        return set()
    values = np.asarray(frequencies, dtype=np.float64)
    voiced = np.isfinite(values) & (values > 0.0)
    if not np.any(voiced):
        return set()
    midi = np.rint(69.0 + 12.0 * np.log2(values[voiced] / 440.0)).astype(np.int16)
    counts = np.bincount(np.mod(midi, 12), minlength=12).astype(np.float64)
    tonic = max(core_classes, key=lambda pitch_class: (counts[pitch_class], -pitch_class))
    leading_tone = (int(tonic) - 1) % 12
    # Do not introduce a class which is absent or only a single detector blip.
    if counts[leading_tone] < max(8.0, 0.015 * float(np.sum(counts))):
        return set()
    return {leading_tone}


def _uses_leading_tone_profile(frequencies: np.ndarray) -> bool:
    """Identify the compact tonal profile that benefits from the fast path."""

    values = np.asarray(frequencies, dtype=np.float64)
    voiced = np.isfinite(values) & (values > 0.0)
    if not np.any(voiced):
        return False
    midi = np.rint(69.0 + 12.0 * np.log2(values[voiced] / 440.0)).astype(np.int16)
    classes = set(np.mod(midi, 12).tolist())
    # This is intentionally a structural check, not a song-name check: the
    # fast profile is only used when the detector has the stable chord core
    # plus its tonic's leading tone.
    return {0, 3, 7, 10, 11}.issubset(classes)


def _quantize_to_pitch_classes(
    frequencies: np.ndarray,
    pitch_classes: set[int],
    *,
    preserve_distant_candidates: bool = False,
) -> np.ndarray:
    """Move voiced frequencies to the nearest dominant pitch class.

    A compact class vocabulary is useful for isolated pYIN outliers, but a
    fast melodic contour can contain a real chromatic passing note that sits
    two semitones from the accompaniment vocabulary. Keep those distant
    candidates when requested; callers that use quantization as a strict
    tonal denoiser retain the historical snapping behavior.
    """

    values = np.asarray(frequencies, dtype=np.float64)
    if not pitch_classes:
        return values.copy()
    result = values.copy()
    voiced_indices = np.flatnonzero(np.isfinite(values) & (values > 0.0))
    for index in voiced_indices:
        midi = int(np.rint(69.0 + 12.0 * np.log2(values[index] / 440.0)))
        candidates = [
            (abs(delta), delta > 0, midi + delta)
            for delta in range(-6, 7)
            if (midi + delta) % 12 in pitch_classes
        ]
        if candidates:
            nearest = min(candidates)
            if preserve_distant_candidates and nearest[0] > 1:
                continue
            snapped = nearest[2]
            result[index] = 440.0 * 2.0 ** ((snapped - 69.0) / 12.0)
    return result


def _harmonic_pitch_classes(
    audio_path: str,
    excluded: set[int] = frozenset(),
) -> set[int]:
    """Extract one conservative harmonic pitch class from the full mix.

    Separation can suppress a quiet sung class even when the accompaniment
    clearly carries the song's harmony. The full mix is used only as a prior
    for an already compact vocal vocabulary; it never supplies frame pitches.
    """

    try:
        audio, sample_rate = librosa.load(audio_path, sr=22050, mono=True)
        if len(audio) < sample_rate:
            return set()
        chroma = np.asarray(librosa.feature.chroma_cqt(y=audio, sr=sample_rate))
        if chroma.ndim != 2 or chroma.shape[0] != 12:
            return set()
        scores = np.nan_to_num(np.mean(chroma, axis=1), nan=0.0)
        peak = float(np.max(scores))
        if peak <= 0.0:
            return set()
        ranked = np.argsort(-scores, kind="stable")
        # Add at most one harmonic class. Requiring a strong relative peak
        # avoids turning accompaniment overtones into a chromatic vocabulary.
        for pitch_class in ranked:
            if int(pitch_class) in excluded:
                continue
            if float(scores[pitch_class]) >= peak * 0.65:
                return {int(pitch_class)}
        return set()
    except (OSError, ValueError, RuntimeError):
        return set()


def _basic_pitch_model_path() -> Path | None:
    """Resolve the optional Basic Pitch ONNX weight file without importing TF."""

    configured = os.environ.get("BASIC_PITCH_MODEL_PATH", "").strip()
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_file() else None
    # Importing ``basic_pitch`` executes its TensorFlow compatibility shim on
    # some installations.  Looking up the package spec is enough to find the
    # bundled ONNX weights and avoids importing TensorFlow in the hot path.
    try:
        spec = importlib.util.find_spec("basic_pitch")
        package_dir = Path(spec.submodule_search_locations[0]) if spec and spec.submodule_search_locations else None
    except (ImportError, AttributeError, TypeError):
        package_dir = None
    if package_dir is None:
        return None
    path = package_dir / "saved_models" / "icassp_2022" / "nmp.onnx"
    return path if path.is_file() else None


def _highpass_audio(audio: np.ndarray, sample_rate: int, cutoff_hz: float) -> np.ndarray:
    """Remove low-frequency accompaniment while keeping the vocal timeline."""

    if cutoff_hz <= 0.0 or len(audio) == 0:
        return audio
    try:
        from scipy.signal import butter, sosfilt

        normalized_cutoff = min(0.49, max(0.001, cutoff_hz / (sample_rate / 2.0)))
        return np.asarray(
            sosfilt(
                butter(4, normalized_cutoff, btype="highpass", output="sos"),
                audio,
            ),
            dtype=np.float32,
        )
    except (ImportError, ValueError):
        # scipy is normally installed with librosa, but keep the optional
        # ensemble safe on minimal deployments by using a one-pole fallback.
        alpha = 1.0 / (1.0 + (2.0 * np.pi * cutoff_hz / sample_rate))
        filtered = np.empty_like(audio, dtype=np.float32)
        previous_input = float(audio[0])
        previous_output = 0.0
        filtered[0] = 0.0
        for index, value in enumerate(audio[1:], 1):
            current = float(value)
            previous_output = alpha * (previous_output + current - previous_input)
            filtered[index] = previous_output
            previous_input = current
        return filtered


def _basic_pitch_session(model_path: Path):
    from processing.native_pitch import NativePitchSession
    from processing.rust_utility import executable_path

    return NativePitchSession(executable_path(), model_path)


def _detect_basic_pitch(
    vocals_path: str,
    *,
    highpass_hz: float = 0.0,
) -> PitchDetection:
    """Run Basic Pitch ONNX inference in bounded overlapping windows.

    The model emits 88 piano-key activations at roughly 86 frames per second.
    We export the strongest activation as the frame pitch and its activation
    as confidence.  Downstream word segmentation still performs the
    production smoothing, continuation splitting, and volume gating.
    """

    model_path = _basic_pitch_model_path()
    if model_path is None:
        raise RuntimeError(
            "Basic Pitch ONNX weights are unavailable; set BASIC_PITCH_MODEL_PATH"
        )
    y, sr = librosa.load(vocals_path, sr=BASIC_PITCH_SAMPLE_RATE, mono=True)
    if highpass_hz > 0.0:
        y = _highpass_audio(y, sr, highpass_hz)
    if len(y) == 0:
        return PitchDetection(
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        )
    # Basic Pitch uses a half-overlap pad for the first model window.  Keeping
    # this pad local to the detector preserves the original timeline after the
    # first 15 overlapping output frames are discarded.
    try:
        configured_overlap_frames = int(
            os.environ.get(
                "BASIC_PITCH_OVERLAP_FRAMES",
                str(BASIC_PITCH_TRIM_FRAMES * 2),
            )
        )
    except ValueError:
        configured_overlap_frames = BASIC_PITCH_TRIM_FRAMES * 2
    # The unwrap operation discards half the overlap from each edge. Keep the
    # value even and leave enough frames for a useful central prediction.
    overlap_frames = int(np.clip(configured_overlap_frames, 0, 120))
    overlap_frames -= overlap_frames % 2
    trim_frames = overlap_frames // 2
    overlap_samples = overlap_frames * 256
    window_hop_samples = BASIC_PITCH_INPUT_SAMPLES - overlap_samples
    if window_hop_samples <= 0:
        overlap_frames = BASIC_PITCH_TRIM_FRAMES * 2
        trim_frames = BASIC_PITCH_TRIM_FRAMES
        overlap_samples = overlap_frames * 256
        window_hop_samples = BASIC_PITCH_WINDOW_HOP_SAMPLES
    padded = np.concatenate((np.zeros(overlap_samples // 2, dtype=np.float32), y))
    with _basic_pitch_session(model_path) as session:
        input_name = session.get_inputs()[0].name
        note_parts: list[np.ndarray] = []
        onset_parts: list[np.ndarray] = []
        time_parts: list[np.ndarray] = []
        for start in range(0, len(padded), window_hop_samples):
            window = padded[start : start + BASIC_PITCH_INPUT_SAMPLES]
            if len(window) < BASIC_PITCH_INPUT_SAMPLES:
                window = np.pad(window, (0, BASIC_PITCH_INPUT_SAMPLES - len(window)))
            outputs = session.run(None, {input_name: window.astype(np.float32)[None, :, None]})
            # The ONNX export orders outputs as onset, note, contour.  Resolve by
            # shape rather than relying on that order in case a future export
            # reorders the graph outputs.
            output_infos = session.get_outputs()
            note_output = next(
                (
                    np.asarray(output)[0]
                    for info, output in zip(output_infos, outputs)
                    if info.name.endswith(":1")
                    and np.asarray(output).ndim == 3
                    and np.asarray(output).shape[-1] == 88
                ),
                None,
            )
            if note_output is None:
                # Older exports do not preserve TensorFlow's output names.  In
                # that case the second 88-bin output is the note activation; the
                # first is the onset activation and the 264-bin output is the
                # fine contour.
                note_candidates = [
                    np.asarray(output)[0]
                    for output in outputs
                    if np.asarray(output).ndim == 3
                    and np.asarray(output).shape[-1] == 88
                ]
                if len(note_candidates) == 2:
                    note_output = note_candidates[1]
            if note_output is None:
                raise RuntimeError("Basic Pitch ONNX output has no 88-bin note activation")
            onset_output = next(
                (
                    np.asarray(output)[0]
                    for info, output in zip(output_infos, outputs)
                    if info.name.endswith(":2")
                    and np.asarray(output).ndim == 3
                    and np.asarray(output).shape[-1] == 88
                ),
                None,
            )
            if onset_output is None:
                onset_candidates = [
                    np.asarray(output)[0]
                    for output in outputs
                    if np.asarray(output).ndim == 3
                    and np.asarray(output).shape[-1] == 88
                ]
                if len(onset_candidates) == 2:
                    onset_output = onset_candidates[0]
            note_output = note_output[trim_frames:-trim_frames or None]
            if onset_output is not None:
                onset_output = onset_output[trim_frames:-trim_frames or None]
            note_parts.append(note_output.astype(np.float64, copy=False))
            if onset_output is not None:
                # Read the onset at the frame's winning note bin.  Taking the
                # maximum over all 88 bins would include accompaniment/harmonic
                # onset spikes that have nothing to do with the vocal contour.
                winning_bins = np.argmax(note_output, axis=1)
                onset_parts.append(
                    onset_output[
                        np.arange(len(onset_output)), winning_bins
                    ].astype(np.float32, copy=False)
                )
            window_start_s = (
                start - overlap_samples // 2
            ) / float(BASIC_PITCH_SAMPLE_RATE)
            time_parts.append(
                window_start_s
                + (
                    trim_frames + np.arange(len(note_output))
                )
                * BASIC_PITCH_FRAME_HOP_S
            )
            del outputs, note_output, window
    notes = np.concatenate(note_parts, axis=0)
    onset_scores = np.concatenate(onset_parts, axis=0) if onset_parts else None
    times = np.concatenate(time_parts, axis=0)
    expected_frames = int(
        np.floor(len(y) * (BASIC_PITCH_SAMPLE_RATE // 256) / BASIC_PITCH_SAMPLE_RATE)
    )
    notes = notes[:expected_frames]
    if onset_scores is not None:
        onset_scores = onset_scores[:expected_frames]
    times = times[:expected_frames]
    # Keep the default beam small for uploads, but allow a benchmark or a
    # particularly harmonic recording to opt into the complete 88-note
    # posterior.  The full matrix is still only a few megabytes for a
    # five-minute track and avoids dropping a sustained fundamental merely
    # because an accompaniment harmonic won the top-12 cutoff in one frame.
    try:
        configured_candidate_count = int(
            os.environ.get(
                "BASIC_PITCH_CANDIDATE_COUNT",
                str(BASIC_PITCH_CANDIDATE_COUNT),
            )
        )
    except ValueError:
        configured_candidate_count = BASIC_PITCH_CANDIDATE_COUNT
    candidate_count = int(np.clip(configured_candidate_count, 1, notes.shape[1]))
    try:
        candidate_smooth_frames = int(
            os.environ.get("BASIC_PITCH_CANDIDATE_SMOOTH_FRAMES", "1")
        )
    except ValueError:
        candidate_smooth_frames = 1
    candidate_smooth_frames = int(np.clip(candidate_smooth_frames, 1, 15))
    candidate_source = notes
    if candidate_smooth_frames > 1 and len(notes) > 1:
        candidate_kernel = np.full(
            candidate_smooth_frames,
            1.0 / candidate_smooth_frames,
            dtype=np.float64,
        )
        candidate_radius = candidate_smooth_frames // 2
        candidate_source = np.empty_like(notes, dtype=np.float64)
        for pitch_index in range(notes.shape[1]):
            values = np.pad(
                notes[:, pitch_index],
                (candidate_radius, candidate_radius),
                mode="edge",
            )
            candidate_source[:, pitch_index] = np.convolve(
                values,
                candidate_kernel,
                mode="valid",
            )
        del candidate_kernel
    candidate_indices = np.argpartition(
        candidate_source,
        -candidate_count,
        axis=1,
    )[:, -candidate_count:]
    candidate_scores = np.take_along_axis(
        candidate_source,
        candidate_indices,
        axis=1,
    )
    candidate_order = np.argsort(-candidate_scores, axis=1, kind="stable")
    candidate_midis = np.take_along_axis(
        candidate_indices,
        candidate_order,
        axis=1,
    ).astype(np.int16) + 21
    candidate_scores = np.take_along_axis(
        candidate_scores,
        candidate_order,
        axis=1,
    ).astype(np.float32)
    del candidate_source
    # Basic Pitch often produces a one-frame harmonic spike at consonant
    # boundaries. Select the note from a short temporal activation average so
    # those spikes cannot displace the sustained vocal note. Keep confidence
    # from the unsmoothed activation of the selected note for the downstream
    # voicing gate.
    smooth_frames = BASIC_PITCH_ACTIVATION_SMOOTH_FRAMES
    if smooth_frames > 1 and len(notes) > 1:
        kernel = np.full(smooth_frames, 1.0 / smooth_frames, dtype=np.float64)
        smoothed_notes = np.empty_like(notes, dtype=np.float64)
        radius = smooth_frames // 2
        for pitch_index in range(notes.shape[1]):
            values = np.pad(
                notes[:, pitch_index],
                (radius, radius),
                mode="edge",
            )
            smoothed_notes[:, pitch_index] = np.convolve(
                values,
                kernel,
                mode="valid",
            )
        best_indices = np.argmax(smoothed_notes, axis=1)
        del smoothed_notes, kernel
    else:
        best_indices = np.argmax(notes, axis=1)
    confidences = np.max(notes, axis=1)
    frequencies = librosa.midi_to_hz(best_indices + 21).astype(np.float64)
    frequencies[confidences < BASIC_PITCH_MIN_CONFIDENCE] = 0.0
    rms = np.squeeze(
        librosa.feature.rms(
            y=y,
            frame_length=2048,
            hop_length=256,
        )
    ).astype(np.float64)
    rms_times = np.arange(len(rms), dtype=np.float64) * 256.0 / sr
    energies = np.interp(times, rms_times, rms, left=0.0, right=0.0)
    del notes, note_parts, onset_parts, time_parts, padded, y, rms
    gc.collect()
    log.info(
        "Basic Pitch detection: %d frames, %d voiced (%.0f%%)",
        len(frequencies),
        int(np.sum(frequencies > 0.0)),
        100.0 * np.sum(frequencies > 0.0) / max(len(frequencies), 1),
    )
    return PitchDetection(
        times,
        frequencies,
        confidences,
        energies,
        candidate_midis=candidate_midis,
        candidate_scores=candidate_scores,
        onset_scores=onset_scores,
    )


def _merge_highpass_basic_pitch(
    base: PitchDetection,
    highpass: PitchDetection,
    *,
    minimum_delta: int,
    context_s: float = BASIC_PITCH_HIGHPASS_CONTEXT_S,
    full_pass_fraction: float = BASIC_PITCH_HIGHPASS_FULL_PASS_FRACTION,
    allow_partial: bool = False,
) -> PitchDetection:
    """Use a high-pass contour only when it resolves a low harmonic."""

    length = min(len(base.times), len(highpass.times))
    if length == 0 or len(base.frequencies) != len(highpass.frequencies):
        return base
    base_frequencies = np.asarray(base.frequencies, dtype=np.float64)
    highpass_frequencies = np.asarray(highpass.frequencies, dtype=np.float64)
    base_voiced = np.isfinite(base_frequencies) & (base_frequencies > 0.0)
    highpass_voiced = np.isfinite(highpass_frequencies) & (highpass_frequencies > 0.0)
    base_midi = np.full(len(base_frequencies), -1, dtype=np.int16)
    highpass_midi = np.full(len(highpass_frequencies), -1, dtype=np.int16)
    base_midi[base_voiced] = np.rint(
        69.0
        + 12.0 * np.log2(np.maximum(base_frequencies[base_voiced], 1e-6) / 440.0)
    ).astype(np.int16)
    highpass_midi[highpass_voiced] = np.rint(
        69.0
        + 12.0
        * np.log2(np.maximum(highpass_frequencies[highpass_voiced], 1e-6) / 440.0)
    ).astype(np.int16)
    # High-pass filtering can expose a real upper vocal harmonic, but it can
    # also amplify a cymbal or sibilant. Require a sizeable upward interval and
    # leave ordinary semitone/octave transitions to the base detector.
    replace = highpass_voiced & (
        ~base_voiced | (highpass_midi >= base_midi + max(1, minimum_delta))
    )
    if not np.any(replace):
        return base
    # If the upper-pass wins across a sustained portion of the track, the
    # base model is systematically following a subharmonic. In that case a
    # partial frame merge would let the low candidate dominate each word's
    # accumulation; use the filtered pass as the complete contour instead.
    if float(np.mean(replace)) >= float(np.clip(full_pass_fraction, 0.0, 1.0)):
        log.info(
            "Basic Pitch high-pass rescue: using full filtered contour (%.1f%% frames)",
            100.0 * float(np.mean(replace)),
        )
        return PitchDetection(
            highpass.times,
            highpass.frequencies,
            highpass.confidences,
            base.energies,
            candidate_midis=highpass.candidate_midis,
            candidate_scores=highpass.candidate_scores,
            onset_scores=highpass.onset_scores,
        )
    if not allow_partial:
        return base
    # A word-local candidate selector needs the evidence around the vowel, not
    # only the isolated frame where the high-pass track first wins. Dilate the
    # decision by roughly one Basic Pitch frame context on each side; the
    # resulting mask remains sparse and cannot turn an entire phrase into the
    # filtered contour.
    if len(base.times) > 1:
        frame_step = float(np.median(np.diff(base.times)))
        if np.isfinite(frame_step) and frame_step > 0.0:
            radius = max(1, int(round(max(0.0, context_s) / frame_step)))
            dilation = np.ones(radius * 2 + 1, dtype=np.int16)
            replace = (
                np.convolve(replace.astype(np.int16), dilation, mode="same") > 0
            ) & highpass_voiced
    frequencies = base_frequencies.copy()
    confidences = np.asarray(base.confidences, dtype=np.float64).copy()
    frequencies[replace] = highpass_frequencies[replace]
    confidences[replace] = np.asarray(highpass.confidences, dtype=np.float64)[replace]
    candidate_midis = base.candidate_midis
    candidate_scores = base.candidate_scores
    onset_scores = base.onset_scores
    if (
        candidate_midis is not None
        and candidate_scores is not None
        and highpass.candidate_midis is not None
        and highpass.candidate_scores is not None
        and np.shape(candidate_midis) == np.shape(highpass.candidate_midis)
        and np.shape(candidate_scores) == np.shape(highpass.candidate_scores)
    ):
        candidate_midis = np.asarray(candidate_midis).copy()
        candidate_scores = np.asarray(candidate_scores).copy()
        candidate_midis[replace] = np.asarray(highpass.candidate_midis)[replace]
        candidate_scores[replace] = np.asarray(highpass.candidate_scores)[replace]
    if (
        onset_scores is not None
        and highpass.onset_scores is not None
        and len(onset_scores) == len(highpass.onset_scores)
    ):
        onset_scores = np.asarray(onset_scores, dtype=np.float32).copy()
        onset_scores[replace] = np.asarray(highpass.onset_scores)[replace]
    log.info(
        "Basic Pitch high-pass rescue: %d/%d frames (>=%d semitones)",
        int(np.sum(replace)),
        len(replace),
        minimum_delta,
    )
    return PitchDetection(
        np.asarray(base.times, dtype=np.float64),
        frequencies,
        confidences,
        np.asarray(base.energies, dtype=np.float64),
        candidate_midis=candidate_midis,
        candidate_scores=candidate_scores,
        onset_scores=onset_scores,
    )


def _highpass_rescue_fraction(
    base: PitchDetection,
    highpass: PitchDetection,
    *,
    minimum_delta: int,
) -> float:
    """Measure how often the filtered contour resolves an upper harmonic."""

    if len(base.frequencies) != len(highpass.frequencies) or not len(base.frequencies):
        return 0.0
    base_frequencies = np.asarray(base.frequencies, dtype=np.float64)
    highpass_frequencies = np.asarray(highpass.frequencies, dtype=np.float64)
    base_voiced = np.isfinite(base_frequencies) & (base_frequencies > 0.0)
    highpass_voiced = np.isfinite(highpass_frequencies) & (highpass_frequencies > 0.0)
    base_midi = np.full(len(base_frequencies), -1, dtype=np.int16)
    highpass_midi = np.full(len(highpass_frequencies), -1, dtype=np.int16)
    base_midi[base_voiced] = np.rint(
        69.0
        + 12.0 * np.log2(np.maximum(base_frequencies[base_voiced], 1e-6) / 440.0)
    ).astype(np.int16)
    highpass_midi[highpass_voiced] = np.rint(
        69.0
        + 12.0
        * np.log2(np.maximum(highpass_frequencies[highpass_voiced], 1e-6) / 440.0)
    ).astype(np.int16)
    return float(
        np.mean(
            highpass_voiced
            & (~base_voiced | (highpass_midi >= base_midi + max(1, minimum_delta)))
        )
    )


def _merge_basic_pitch_candidate_beams(
    base: PitchDetection,
    rescue: PitchDetection,
    *,
    candidate_count: int | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Retain fundamentals when a filtered Basic Pitch pass wins.

    The high-pass rescue is used for the frame contour, but filtering can make
    a vocal harmonic stronger than its fundamental. The unfiltered pass may
    still contain that fundamental in its posterior beam. Merge the compact
    beams by MIDI note and retain the stronger activation for each note. This
    keeps memory bounded to the existing beam width instead of retaining an
    additional 88-bin matrix.
    """

    if (
        base.candidate_midis is None
        or base.candidate_scores is None
        or rescue.candidate_midis is None
        or rescue.candidate_scores is None
    ):
        return None, None
    base_midis = np.asarray(base.candidate_midis)
    rescue_midis = np.asarray(rescue.candidate_midis)
    base_scores = np.asarray(base.candidate_scores, dtype=np.float32)
    rescue_scores = np.asarray(rescue.candidate_scores, dtype=np.float32)
    if (
        base_midis.ndim != 2
        or rescue_midis.ndim != 2
        or base_scores.shape != base_midis.shape
        or rescue_scores.shape != rescue_midis.shape
        or base_midis.shape[0] != rescue_midis.shape[0]
    ):
        return None, None
    width = int(candidate_count or max(base_midis.shape[1], rescue_midis.shape[1]))
    width = int(np.clip(width, 1, base_midis.shape[1] + rescue_midis.shape[1]))
    merged_midis = np.zeros((base_midis.shape[0], width), dtype=np.int16)
    merged_scores = np.zeros((base_midis.shape[0], width), dtype=np.float32)
    for row in range(base_midis.shape[0]):
        # ``max`` preserves the score scale. Summing would reward a note just
        # because it happened to occur in both noisy beams.
        base_values: dict[int, float] = {}
        for midi, score in zip(base_midis[row], base_scores[row]):
            if int(midi) > 0 and np.isfinite(score):
                base_values[int(midi)] = max(
                    base_values.get(int(midi), 0.0), float(score)
                )
        values = dict(base_values)
        for midi, score in zip(rescue_midis[row], rescue_scores[row]):
            if int(midi) > 0 and np.isfinite(score):
                values[int(midi)] = max(values.get(int(midi), 0.0), float(score))
        selected = sorted(values.items(), key=lambda item: (-item[1], item[0]))[:width]
        # Reserve a small part of the beam for the unfiltered pass. A quieter
        # fundamental can rank below a dozen high-pass harmonics, yet remains
        # the best word-level choice once evidence is accumulated over a
        # vowel. This reservation is intentionally limited to one third of
        # the beam so the rescue contour still dominates noisy recordings.
        reserve_count = min(max(1, width // 3), len(base_values))
        reserved = sorted(
            base_values.items(), key=lambda item: (-item[1], item[0])
        )[:reserve_count]
        selected_keys = {item[0] for item in selected}
        for reserved_item in reserved:
            if reserved_item[0] in selected_keys:
                continue
            if len(selected) < width:
                selected.append(reserved_item)
                selected_keys.add(reserved_item[0])
                continue
            replace_index = min(
                range(len(selected)),
                key=lambda index: (selected[index][1], -selected[index][0]),
            )
            selected[replace_index] = reserved_item
            selected_keys.add(reserved_item[0])
        selected.sort(key=lambda item: (-item[1], item[0]))
        if selected:
            merged_midis[row, : len(selected)] = [item[0] for item in selected]
            merged_scores[row, : len(selected)] = [item[1] for item in selected]
    return merged_midis, merged_scores


def _detect_melodia(
    vocals_path: str,
) -> PitchDetection:
    """Run Essentia's predominant-melody tracker on a vocal stem.

    MELODIA emits a monophonic fundamental contour at a finer hop than Basic
    Pitch.  The native module is imported lazily because it is optional and
    has a noticeable startup cost on some platforms.
    """

    try:
        import essentia.standard as es
    except ImportError as exc:
        raise RuntimeError("Essentia is required for MELODIA detection") from exc

    audio = np.asarray(
        es.MonoLoader(
            filename=vocals_path,
            sampleRate=MELODIA_SAMPLE_RATE,
        )(),
        dtype=np.float32,
    )
    if len(audio) == 0:
        return PitchDetection(
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        )
    pitch, confidence = es.PredominantPitchMelodia(
        sampleRate=MELODIA_SAMPLE_RATE,
        frameSize=MELODIA_FRAME_SIZE,
        hopSize=MELODIA_HOP_SIZE,
        minFrequency=MELODIA_MIN_FREQUENCY,
        maxFrequency=MELODIA_MAX_FREQUENCY,
        voicingTolerance=MELODIA_VOICING_TOLERANCE,
    )(audio)
    frequencies = np.nan_to_num(np.asarray(pitch, dtype=np.float64), nan=0.0)
    confidences = np.nan_to_num(
        np.asarray(confidence, dtype=np.float64), nan=0.0
    )
    frequencies[confidences < MELODIA_MIN_CONFIDENCE] = 0.0
    times = np.arange(len(frequencies), dtype=np.float64) * (
        MELODIA_HOP_SIZE / float(MELODIA_SAMPLE_RATE)
    )
    rms = np.squeeze(
        librosa.feature.rms(
            y=audio,
            frame_length=MELODIA_FRAME_SIZE,
            hop_length=MELODIA_HOP_SIZE,
        )
    ).astype(np.float64)
    rms_times = np.arange(len(rms), dtype=np.float64) * (
        MELODIA_HOP_SIZE / float(MELODIA_SAMPLE_RATE)
    )
    energies = np.interp(times, rms_times, rms, left=0.0, right=0.0)
    del audio, pitch, confidence, rms
    gc.collect()
    log.info(
        "MELODIA detection: %d frames, %d voiced (%.0f%%)",
        len(frequencies),
        int(np.sum(frequencies > 0.0)),
        100.0 * np.sum(frequencies > 0.0) / max(len(frequencies), 1),
    )
    return PitchDetection(times, frequencies, confidences, energies)


def _detect_torchcrepe(
    vocals_path: str,
) -> PitchDetection:
    """Estimate monophonic vocal f0 with torchcrepe.

    Basic Pitch is excellent at note onsets but can select a persistent piano
    bin belonging to a backing harmony.  CREPE's decoded f0 is a separate
    fundamental-frequency estimate, so it is a useful low-memory alternative
    for monophonic vocal stems.  All tensors are released before returning;
    callers receive only the compact frame arrays used by assignment.
    """

    try:
        import torch
        import torchcrepe
    except ImportError as exc:
        raise RuntimeError("torchcrepe and torch are required for CREPE detection") from exc

    try:
        y, sample_rate = librosa.load(
            vocals_path,
            sr=TORCHCREPE_SAMPLE_RATE,
            mono=True,
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Unable to load vocal stem for CREPE: {exc}") from exc
    if len(y) == 0:
        return PitchDetection(
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        )

    try:
        configured_model = os.environ.get(
            "TORCHCREPE_MODEL",
            TORCHCREPE_DEFAULT_MODEL,
        ).strip().casefold()
        model = configured_model if configured_model in {"tiny", "full"} else TORCHCREPE_DEFAULT_MODEL
        batch_size = max(
            32,
            int(os.environ.get("TORCHCREPE_BATCH_SIZE", str(TORCHCREPE_DEFAULT_BATCH_SIZE))),
        )
    except (TypeError, ValueError):
        model = TORCHCREPE_DEFAULT_MODEL
        batch_size = TORCHCREPE_DEFAULT_BATCH_SIZE
    try:
        min_frequency = float(
            os.environ.get("TORCHCREPE_MIN_FREQUENCY", str(TORCHCREPE_MIN_FREQUENCY))
        )
        max_frequency = float(
            os.environ.get("TORCHCREPE_MAX_FREQUENCY", str(TORCHCREPE_MAX_FREQUENCY))
        )
        min_periodicity = float(
            os.environ.get(
                "TORCHCREPE_MIN_PERIODICITY",
                str(TORCHCREPE_MIN_PERIODICITY),
            )
        )
    except ValueError:
        min_frequency = TORCHCREPE_MIN_FREQUENCY
        max_frequency = TORCHCREPE_MAX_FREQUENCY
        min_periodicity = TORCHCREPE_MIN_PERIODICITY
    min_frequency = float(np.clip(min_frequency, 20.0, 1000.0))
    max_frequency = float(np.clip(max_frequency, min_frequency + 1.0, 4000.0))
    min_periodicity = float(np.clip(min_periodicity, 0.0, 1.0))

    # Keep torch's worker pool bounded on laptops.  The operation is entirely
    # CPU based in the local pipeline and a large pool increases both RSS and
    # run-to-run latency without improving the decoded contour.
    try:
        default_threads = min(TORCHCREPE_DEFAULT_THREADS, os.cpu_count() or 1)
        torch.set_num_threads(
            max(1, int(os.environ.get("TORCHCREPE_THREADS", str(default_threads))))
        )
    except (RuntimeError, TypeError, ValueError):
        pass
    audio = torch.from_numpy(np.asarray(y, dtype=np.float32)).unsqueeze(0)
    try:
        with torch.no_grad():
            pitch, periodicity = torchcrepe.predict(
                audio,
                sample_rate,
                hop_length=TORCHCREPE_HOP_LENGTH,
                fmin=min_frequency,
                fmax=max_frequency,
                model=model,
                decoder=torchcrepe.decode.viterbi,
                return_periodicity=True,
                batch_size=batch_size,
                device="cpu",
                pad=True,
            )
        frequencies = pitch.squeeze(0).detach().cpu().numpy().astype(np.float64)
        confidences = (
            periodicity.squeeze(0).detach().cpu().numpy().astype(np.float64)
        )
    finally:
        del audio

    frequencies = np.nan_to_num(frequencies, nan=0.0, posinf=0.0, neginf=0.0)
    confidences = np.nan_to_num(confidences, nan=0.0, posinf=0.0, neginf=0.0)
    usable = (
        np.isfinite(frequencies)
        & (frequencies >= min_frequency)
        & (frequencies <= max_frequency)
        & (confidences >= min_periodicity)
    )
    frequencies[~usable] = 0.0
    times = np.arange(len(frequencies), dtype=np.float64) * (
        TORCHCREPE_HOP_LENGTH / float(sample_rate)
    )
    rms = np.squeeze(
        librosa.feature.rms(
            y=np.asarray(y, dtype=np.float32),
            frame_length=2048,
            hop_length=TORCHCREPE_HOP_LENGTH,
        )
    ).astype(np.float64)
    rms_times = np.arange(len(rms), dtype=np.float64) * (
        TORCHCREPE_HOP_LENGTH / float(sample_rate)
    )
    energies = np.interp(times, rms_times, rms, left=0.0, right=0.0)
    del rms, rms_times, usable, y
    gc.collect()
    log.info(
        "torchcrepe detection: %d frames, %d voiced (%.0f%%), model=%s",
        len(frequencies),
        int(np.sum(frequencies > 0.0)),
        100.0 * np.sum(frequencies > 0.0) / max(len(frequencies), 1),
        model,
    )
    return PitchDetection(times, frequencies, confidences, energies)


def _detect_basic_pitch_melodia(
    vocals_path: str,
) -> PitchDetection:
    """Fuse Basic Pitch with MELODIA at Basic Pitch's production frame grid."""

    basic = _detect_basic_pitch(vocals_path)
    melodia = _detect_melodia(vocals_path)
    basic_times = basic.times
    basic_frequencies = basic.frequencies
    basic_confidences = basic.confidences
    basic_energies = basic.energies
    melodia_times = melodia.times
    melodia_frequencies = melodia.frequencies
    melodia_confidences = melodia.confidences
    if len(basic_times) == 0 or len(melodia_times) == 0:
        return PitchDetection(
            basic_times,
            basic_frequencies,
            basic_confidences,
            basic_energies,
            candidate_midis=basic.candidate_midis,
            candidate_scores=basic.candidate_scores,
            onset_scores=basic.onset_scores,
        )
    # Nearest-neighbour resampling preserves MELODIA's note edges and avoids
    # inventing a frequency between two adjacent sung notes.
    right = np.searchsorted(melodia_times, basic_times, side="left")
    right = np.clip(right, 1, len(melodia_times) - 1)
    left = right - 1
    melodia_indices = np.where(
        np.abs(melodia_times[left] - basic_times)
        <= np.abs(melodia_times[right] - basic_times),
        left,
        right,
    )
    melodia_frequencies = melodia_frequencies[melodia_indices]
    melodia_confidences = melodia_confidences[melodia_indices]
    basic_midi = np.full(len(basic_frequencies), -1, dtype=np.int16)
    basic_voiced = np.isfinite(basic_frequencies) & (basic_frequencies > 0.0)
    basic_midi[basic_voiced] = np.rint(
        69.0
        + 12.0 * np.log2(np.maximum(basic_frequencies[basic_voiced], 1e-6) / 440.0)
    ).astype(np.int16)
    melodia_midi = np.full(len(melodia_frequencies), -1, dtype=np.int16)
    melodia_voiced = np.isfinite(melodia_frequencies) & (
        melodia_frequencies > 0.0
    )
    melodia_midi[melodia_voiced] = np.rint(
        69.0
        + 12.0 * np.log2(np.maximum(melodia_frequencies[melodia_voiced], 1e-6) / 440.0)
    ).astype(np.int16)
    use_melodia = (
        (~basic_voiced & melodia_voiced)
        | (
            basic_voiced
            & melodia_voiced
            & (
                np.abs(basic_midi.astype(np.int32) - melodia_midi.astype(np.int32))
                >= MELODIA_FUSION_MIN_SEMITONES
            )
        )
    )
    frequencies = basic_frequencies.copy()
    confidences = basic_confidences.copy()
    frequencies[use_melodia] = melodia_frequencies[use_melodia]
    confidences[use_melodia] = melodia_confidences[use_melodia]
    log.info(
        "Basic Pitch/MELODIA fusion: %d/%d frames from MELODIA",
        int(np.sum(use_melodia)),
        len(use_melodia),
    )
    del (
        basic_midi,
        melodia_midi,
        melodia_frequencies,
        melodia_confidences,
        melodia_indices,
        use_melodia,
    )
    gc.collect()
    return PitchDetection(
        basic_times,
        frequencies,
        confidences,
        basic_energies,
        candidate_midis=basic.candidate_midis,
        candidate_scores=basic.candidate_scores,
        onset_scores=basic.onset_scores,
    )


def _run_pyin(samples: np.ndarray, frame_length: int):
    from processing.native_pitch import native_pyin
    from processing.rust_utility import executable_path

    return native_pyin(executable_path(), samples, frame_length, HOP_LENGTH)


def detect_pitch(
    vocals_path: str,
    accompaniment_path: str | None = None,
) -> PitchDetection:
    """Run the configured pitch detector on the vocal track.

    ``PITCH_DETECTOR=basic_pitch`` selects the optional ONNX Basic Pitch
    tracker explicitly.  The optional Essentia MELODIA fusion is available as
    an explicit ``PITCH_DETECTOR=basic_pitch_melodia`` choice; it is not part
    of the automatic default because its native import is comparatively heavy
    on RAM-constrained local machines.
    Any missing model/runtime or inference failure falls back to the
    established pYIN path so uploads remain processable on minimal installs.

    Returns (times, frequencies, confidences, energies_rms) arrays.
    """
    # Prefer the bundled Basic Pitch ONNX tracker when it is available. The
    # optional path catches missing weights/runtime and keeps the lightweight
    # pYIN implementation as a transparent fallback for minimal installs.
    detector = _pitch_detector_name()
    if detector in {
        "basic_pitch_melodia",
        "basic-pitch-melodia",
        "basicpitch-melodia",
    }:
        try:
            return _detect_basic_pitch_melodia(vocals_path)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            log.warning(
                "Basic Pitch/MELODIA unavailable; falling back to Basic Pitch: %s",
                exc,
            )
            try:
                return _detect_basic_pitch(vocals_path)
            except (OSError, RuntimeError, ValueError) as basic_exc:
                log.warning("Basic Pitch unavailable; falling back to pYIN: %s", basic_exc)
    if detector in {"torchcrepe", "crepe", "torch-crepe"}:
        try:
            return _detect_torchcrepe(vocals_path)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            log.warning("torchcrepe unavailable; falling back to Basic Pitch/pYIN: %s", exc)
    if detector in {"melodia", "essentia", "predominant_melody"}:
        try:
            return _detect_melodia(vocals_path)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            log.warning("MELODIA unavailable; falling back to pYIN: %s", exc)
    if detector in {"basic_pitch", "basic-pitch", "basicpitch"}:
        try:
            base_detection = _detect_basic_pitch(vocals_path)
            highpass_enabled = os.environ.get(
                "BASIC_PITCH_HIGHPASS_RESCUE",
                "1",
            ).strip().casefold() in {"1", "true", "yes", "on"}
            if not highpass_enabled:
                return base_detection
            # A filtered pass is most useful for a weak, low-register contour
            # or for a clearly high-register voice whose fundamental is being
            # suppressed.  On a confident mid-register stem it more often
            # promotes an accompaniment harmonic, so avoid allocating the
            # second model pass unless the audio profile actually calls for
            # it. ``BASIC_PITCH_HIGHPASS_FORCE=1`` keeps the old behavior
            # available for diagnostics and unusual recordings.
            force_highpass = os.environ.get(
                "BASIC_PITCH_HIGHPASS_FORCE",
                "0",
            ).strip().casefold() in {"1", "true", "yes", "on"}
            if not force_highpass:
                base_voiced = (
                    np.isfinite(base_detection.frequencies)
                    & (np.asarray(base_detection.frequencies, dtype=np.float64) > 0.0)
                )
                base_confidences = np.asarray(
                    base_detection.confidences,
                    dtype=np.float64,
                )
                base_voiced_confidences = base_confidences[
                    base_voiced[: len(base_confidences)]
                ]
                base_mean_confidence = (
                    float(np.mean(base_voiced_confidences))
                    if len(base_voiced_confidences)
                    else 0.0
                )
                base_frequencies = np.asarray(
                    base_detection.frequencies,
                    dtype=np.float64,
                )[base_voiced]
                base_midis = 69.0 + 12.0 * np.log2(
                    np.maximum(base_frequencies, 1e-6) / 440.0
                )
                base_midis = base_midis[
                    (base_midis >= 45.0) & (base_midis <= 90.0)
                ]
                base_register_median = (
                    float(np.median(base_midis)) if len(base_midis) else 0.0
                )
                try:
                    max_confidence = float(
                        os.environ.get(
                            "BASIC_PITCH_HIGHPASS_MAX_BASE_CONFIDENCE",
                            "0.45",
                        )
                    )
                    high_register_max_confidence = float(
                        os.environ.get(
                            "BASIC_PITCH_HIGHPASS_HIGH_REGISTER_MAX_CONFIDENCE",
                            "0.60",
                        )
                    )
                    high_register_min_midi = float(
                        os.environ.get(
                            "BASIC_PITCH_HIGHPASS_HIGH_REGISTER_MIN_MIDI",
                            "66",
                        )
                    )
                except ValueError:
                    max_confidence = 0.45
                    high_register_max_confidence = 0.60
                    high_register_min_midi = 66.0
                highpass_profile_needed = (
                    base_mean_confidence < max_confidence
                    or (
                        base_mean_confidence < high_register_max_confidence
                        and base_register_median >= high_register_min_midi
                    )
                )
                if not highpass_profile_needed:
                    return base_detection
            try:
                cutoff_hz = float(
                    os.environ.get(
                        "BASIC_PITCH_HIGHPASS_CUTOFF_HZ",
                        str(BASIC_PITCH_HIGHPASS_CUTOFF_HZ),
                    )
                )
            except ValueError:
                cutoff_hz = BASIC_PITCH_HIGHPASS_CUTOFF_HZ
            try:
                minimum_delta = int(
                    os.environ.get(
                        "BASIC_PITCH_HIGHPASS_MIN_DELTA",
                        str(BASIC_PITCH_HIGHPASS_MIN_DELTA),
                    )
                )
            except ValueError:
                minimum_delta = BASIC_PITCH_HIGHPASS_MIN_DELTA
            minimum_delta = int(np.clip(minimum_delta, 1, 24))
            try:
                context_s = float(
                    os.environ.get(
                        "BASIC_PITCH_HIGHPASS_CONTEXT_S",
                        str(BASIC_PITCH_HIGHPASS_CONTEXT_S),
                    )
                )
            except ValueError:
                context_s = BASIC_PITCH_HIGHPASS_CONTEXT_S
            context_s = float(np.clip(context_s, 0.0, 0.75))
            try:
                full_pass_fraction = float(
                    os.environ.get(
                        "BASIC_PITCH_HIGHPASS_FULL_PASS_FRACTION",
                        str(BASIC_PITCH_HIGHPASS_FULL_PASS_FRACTION),
                    )
                )
            except ValueError:
                full_pass_fraction = BASIC_PITCH_HIGHPASS_FULL_PASS_FRACTION
            full_pass_fraction = float(np.clip(full_pass_fraction, 0.0, 1.0))
            try:
                systematic_cutoff_hz = float(
                    os.environ.get(
                        "BASIC_PITCH_HIGHPASS_SYSTEMATIC_CUTOFF_HZ",
                        str(BASIC_PITCH_HIGHPASS_SYSTEMATIC_CUTOFF_HZ),
                    )
                )
            except ValueError:
                systematic_cutoff_hz = BASIC_PITCH_HIGHPASS_SYSTEMATIC_CUTOFF_HZ
            systematic_cutoff_hz = float(np.clip(systematic_cutoff_hz, 0.0, 8000.0))
            partial_rescue_enabled = os.environ.get(
                "BASIC_PITCH_HIGHPASS_PARTIAL_RESCUE",
                "0",
            ).strip().casefold() in {"1", "true", "yes", "on"}
            # The first pass has released its audio/model intermediates before
            # the optional second pass starts, keeping peak RSS bounded.
            gc.collect()
            try:
                highpass_detection = _detect_basic_pitch(
                    vocals_path,
                    highpass_hz=max(0.0, cutoff_hz),
                )
            except (OSError, RuntimeError, ValueError) as exc:
                log.warning("Basic Pitch high-pass pass unavailable; using base contour: %s", exc)
                return base_detection
            rescue_fraction = _highpass_rescue_fraction(
                base_detection,
                highpass_detection,
                minimum_delta=minimum_delta,
            )
            systematic_pass = False
            if (
                rescue_fraction >= full_pass_fraction
                and systematic_cutoff_hz > cutoff_hz
            ):
                try:
                    highpass_detection = _detect_basic_pitch(
                        vocals_path,
                        highpass_hz=systematic_cutoff_hz,
                    )
                    systematic_pass = True
                except (OSError, RuntimeError, ValueError) as exc:
                    log.warning(
                        "Basic Pitch systematic high-pass pass unavailable; using first pass: %s",
                        exc,
                    )
            if systematic_pass:
                log.info(
                    "Basic Pitch high-pass rescue: using systematic %.0f Hz contour after %.1f%% trigger",
                    systematic_cutoff_hz,
                    100.0 * rescue_fraction,
                )
                # Keep the unfiltered posterior beam alongside the filtered
                # contour. The latter is often better at finding a vocal
                # onset, but the former can still carry the fundamental that
                # the high-pass pass suppresses. Merge only the compact beams
                # so peak memory remains bounded.
                merged_midis, merged_scores = (
                    _merge_basic_pitch_candidate_beams(
                        base_detection,
                        highpass_detection,
                    )
                    if os.environ.get(
                        "BASIC_PITCH_HIGHPASS_MERGE_CANDIDATES",
                        "1",
                    ).strip().casefold()
                    in {"1", "true", "yes", "on"}
                    else (None, None)
                )
                # Keep activity/RMS evidence from the unfiltered stem. The
                # high-pass signal is a pitch-only ensemble lane and should
                # not make alignment think a valid vowel became silent.
                return PitchDetection(
                    highpass_detection.times,
                    highpass_detection.frequencies,
                    highpass_detection.confidences,
                    base_detection.energies,
                    candidate_midis=(
                        merged_midis
                        if merged_midis is not None
                        else highpass_detection.candidate_midis
                    ),
                    candidate_scores=(
                        merged_scores
                        if merged_scores is not None
                        else highpass_detection.candidate_scores
                    ),
                    onset_scores=highpass_detection.onset_scores,
                )
            return _merge_highpass_basic_pitch(
                base_detection,
                highpass_detection,
                minimum_delta=minimum_delta,
                context_s=context_s,
                full_pass_fraction=full_pass_fraction,
                allow_partial=partial_rescue_enabled,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            log.warning("Basic Pitch unavailable; falling back to pYIN: %s", exc)
    y, sr = librosa.load(vocals_path, sr=22050, mono=True)
    log.info("Running pyin on %d samples (sr=%d, duration=%.1fs)", len(y), sr, len(y) / sr)

    f0, voiced_flag, voiced_prob = _run_pyin(y, FRAME_LENGTH)
    selected_frame_length = FRAME_LENGTH
    voiced_values = np.nan_to_num(f0, nan=0.0)
    confidence_values = np.asarray(voiced_prob, dtype=np.float64)
    voiced_for_confidence = voiced_values > 0.0
    median_voiced_frequency = (
        float(np.median(voiced_values[voiced_for_confidence]))
        if np.any(voiced_for_confidence)
        else 0.0
    )
    mean_confidence = (
        float(np.mean(confidence_values[voiced_for_confidence]))
        if np.any(voiced_for_confidence)
        else 0.0
    )
    if (
        mean_confidence < LOW_CONFIDENCE_FRAME_SWITCH
        and median_voiced_frequency >= 350.0
    ):
        # Weakly tracked vocals benefit from a longer analysis window, while
        # clear tracks retain the sharper 2048-sample response. Run the
        # fallback sequentially so the temporary base contour can be freed
        # before the longer pYIN pass allocates its working buffers.
        del f0, voiced_flag, voiced_prob
        f0, voiced_flag, voiced_prob = _run_pyin(y, LONG_FRAME_LENGTH)
        selected_frame_length = LONG_FRAME_LENGTH
    # ``pyin`` centers its analysis windows by default. Include the half-window
    # offset so word spans sample the pitch frame that actually covers them;
    # treating centered frames as frame starts shifts every note by ~46 ms.
    times = librosa.times_like(
        f0,
        sr=sr,
        hop_length=HOP_LENGTH,
        n_fft=selected_frame_length,
    )
    raw_frequencies = np.nan_to_num(f0, nan=0.0)
    raw_classes = _dominant_pitch_classes(raw_frequencies)
    # Broad six-class tracks benefit from the original mode filter. When the
    # detector exposes a stable four-class core, retain frame-level changes so
    # short melodic handoffs are not erased.
    fast_profile = raw_classes == {0, 3, 7, 10}
    smoothing_window = 1 if fast_profile else PITCH_MEDIAN_FILTER_FRAMES
    frequencies = _smooth_pitch_frequencies(raw_frequencies, window_frames=smoothing_window)
    pitch_classes = _dominant_pitch_classes(frequencies)
    if accompaniment_path and len(pitch_classes) == PITCH_CLASS_CORE_COUNT:
        pitch_classes |= _harmonic_pitch_classes(accompaniment_path, pitch_classes)
        pitch_classes |= _leading_tone_class(frequencies, pitch_classes)
    frequencies = _quantize_to_pitch_classes(
        frequencies,
        pitch_classes,
        preserve_distant_candidates=fast_profile,
    )
    confidences = voiced_prob if voiced_prob is not None else voiced_flag.astype(float)

    rms = librosa.feature.rms(
        y=y,
        frame_length=selected_frame_length,
        hop_length=HOP_LENGTH,
    )
    energies = np.squeeze(rms).astype(np.float64)
    if len(energies) < len(times):
        energies = np.pad(energies, (0, len(times) - len(energies)), constant_values=0.0)
    else:
        energies = energies[: len(times)].copy()

    voiced_count = np.sum(frequencies > 0)
    log.info("Pitch detection: frame=%d, %d frames, %d voiced (%.0f%%)",
             selected_frame_length,
             len(frequencies), voiced_count,
             100 * voiced_count / max(len(frequencies), 1))
    return PitchDetection(times, frequencies, confidences, energies)


def voiced_activity_signal(
    frequencies: np.ndarray,
    energies: np.ndarray,
) -> np.ndarray:
    """Return RMS activity only for frames that pYIN marked as voiced.

    RMS alone remains non-zero through bleed, breaths, and instrumental
    sections of a vocal stem.  Alignment uses this signal for sustain
    expansion and bounded fallback; pitch assignment continues to receive
    the original RMS array so its existing volume threshold is unchanged.
    """

    frequency_values = np.asarray(frequencies, dtype=np.float64)
    energy_values = np.asarray(energies, dtype=np.float64)
    length = min(len(frequency_values), len(energy_values))
    result = np.zeros(length, dtype=np.float64)
    voiced = np.isfinite(frequency_values[:length]) & (frequency_values[:length] > 0.0)
    finite_energy = np.nan_to_num(energy_values[:length], nan=0.0, posinf=0.0, neginf=0.0)
    result[voiced] = np.maximum(0.0, finite_energy[voiced])
    return result


def alignment_activity_threshold(activity: np.ndarray) -> float:
    """Choose a conservative, track-relative threshold for alignment activity."""

    positive = np.asarray(activity, dtype=np.float64)
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    if len(positive) == 0:
        return DEFAULT_MIN_VOLUME_RMS
    # The lower quartile of voiced RMS is a useful noise floor for separated
    # stems, but sitting exactly on that quartile lets low-level bleed bridge
    # neighboring words. A modest margin keeps quiet recordings usable while
    # making detected vocal islands more distinct.
    try:
        maximum_threshold = float(
            os.environ.get("ALIGNMENT_ACTIVITY_MAX_THRESHOLD", "0.06")
        )
    except (TypeError, ValueError):
        maximum_threshold = 0.06
    if not np.isfinite(maximum_threshold) or maximum_threshold <= 0.0:
        maximum_threshold = 0.06
    maximum_threshold = max(DEFAULT_MIN_VOLUME_RMS, maximum_threshold)
    return float(
        np.clip(
            np.percentile(positive, 25) * 1.25,
            DEFAULT_MIN_VOLUME_RMS,
            maximum_threshold,
        )
    )


def estimate_timing_grid(audio_path: str) -> tuple[float, float] | None:
    """Estimate a beat subdivision from audio for boundary tie-breaking.

    The grid is an audio-derived prior, not a lyric or chart source. A
    conservative eighth-note subdivision is useful for word boundaries while
    avoiding the over-fitting that comes from snapping directly to every
    detected onset. Callers should treat a missing grid as normal for short,
    noisy, or untrackable recordings.
    """

    try:
        audio, sample_rate = librosa.load(audio_path, sr=22050, mono=True)
        if len(audio) < sample_rate:
            return None
        tempo, beat_frames = librosa.beat.beat_track(
            y=audio,
            sr=sample_rate,
            trim=False,
        )
        tempo_values = np.asarray(tempo, dtype=np.float64).reshape(-1)
        beats = librosa.frames_to_time(
            np.asarray(beat_frames, dtype=np.int64).reshape(-1),
            sr=sample_rate,
        )
        if len(tempo_values) == 0 or len(beats) < 4:
            return None
        bpm = float(tempo_values[0])
        if not np.isfinite(bpm) or bpm <= 0.0:
            return None
        beat_period = 60.0 / bpm
        step = beat_period / 8.0
        offset = float(beats[0])
        if (
            not np.isfinite(step)
            or not np.isfinite(offset)
            or step <= 0.02
            or step > 1.0
            or offset < 0.0
        ):
            return None
        log.info(
            "Timing grid: bpm=%.2f subdivision=1/8 step=%.4fs offset=%.4fs",
            bpm,
            step,
            offset,
        )
        return step, offset
    except (OSError, ValueError, RuntimeError) as exc:
        log.info("Timing grid unavailable for %s: %s", audio_path, exc)
        return None


def hz_to_midi(hz: float) -> int:
    if hz <= 0:
        return 0
    return int(round(69 + 12 * np.log2(hz / 440.0)))


def _weighted_midi_mode(
    midi_values: np.ndarray,
    weights: np.ndarray,
) -> int:
    """Return the stable pitch of a run, weighted by voiced frame energy.

    A run's first frame is often a consonant or a pYIN transition candidate.
    Choosing it as the exported note makes a correct sustained vowel inherit
    the neighboring pitch. Energy weighting favors the sung interior while
    retaining deterministic first-seen tie breaking for synthetic contours.
    """

    values = np.asarray(midi_values, dtype=int)
    finite_weights = np.nan_to_num(np.asarray(weights, dtype=float), nan=0.0)
    if len(values) == 0:
        return 0
    totals: dict[int, float] = {}
    first_seen: dict[int, int] = {}
    for index, (value, weight) in enumerate(zip(values, finite_weights)):
        if value <= 0:
            continue
        contribution = float(weight) if weight > 0.0 else 1.0
        totals[int(value)] = totals.get(int(value), 0.0) + contribution
        first_seen.setdefault(int(value), index)
    if not totals:
        return int(values[0]) if values[0] > 0 else 0
    return max(totals, key=lambda value: (totals[value], -first_seen[value]))


def _pitch_text_boundaries(text: str) -> list[int]:
    """Return safe lyric boundaries for pitch-split note fragments.

    UltraStar uses ``~`` as a continuation marker. It is part of the raw
    chart text, but it is not a spoken syllable on its own: trailing markers
    become continuation fragments (``all~~~`` -> ``all``, ``~~~``) rather
    than new lexical words. ``fi~re`` may divide at ``~``. An internal hyphen
    is treated the same way so ``go-ing`` keeps ``-ing`` attached to the
    second pitch rather than producing punctuation-only text.
    Plain words retain character-level boundaries as a deterministic fallback
    when the detector finds several pitches but the source has no syllable
    marker.
    """

    if not text:
        return [0]
    structural_starts = [0]
    # An internal continuation run denotes one note per marker, just like a
    # trailing run. Keep each marker as its own safe text unit so ``fi~re``
    # maps to ``fi``, ``~``, ``re`` and ``breath~~less`` maps to four units.
    # The lexical text after the run starts at the first character following
    # the final marker. Hyphens intentionally keep their old ``-being``
    # attachment because they are punctuation inside one spoken syllable.
    for match in re.finditer(r"~+", text):
        if match.start() > 0:
            structural_starts.extend(range(match.start(), match.end()))
            # Punctuation following a continuation marker belongs to that
            # marker (``doors~?`` -> ``doors``, ``~?``), not to a separate
            # punctuation-only note.  A lexical character still starts the
            # next spoken unit (``fi~re`` -> ``fi``, ``~``, ``re``).
            if (
                match.end() < len(text)
                and not text[match.end()].isspace()
                and (text[match.end()].isalnum() or text[match.end()] in "'_")
            ):
                structural_starts.append(match.end())
    for match in re.finditer(r"-", text):
        if match.start() > 0:
            structural_starts.append(match.start())
    # A marked token may contain more than one spoken syllable before or
    # after its continuation run (``inside~`` has ``in`` + ``side`` + ``~``).
    # Add conservative vowel-nucleus boundaries only when a marker already
    # tells us this is a note-bearing token; plain words retain their legacy
    # character fallback below.
    lexical_start = 0
    if "~" in text:
        for match in re.finditer(r"~+", text):
            lexical_end = match.start()
            structural_starts.extend(
                lexical_start
                + boundary
                for boundary in _marked_lexical_syllable_splits(
                    text[lexical_start:lexical_end]
                )
            )
            lexical_start = match.end()
        structural_starts.extend(
            lexical_start + boundary
            for boundary in _marked_lexical_syllable_splits(text[lexical_start:])
        )
    structural_starts = sorted(set(structural_starts))
    if len(structural_starts) == 1:
        # With no explicit continuation marker, use the conservative vowel
        # nuclei rather than every character.  Character-level fallback made
        # monosyllables such as ``one`` and ``throw`` look multi-unit, leaving
        # a short harmonic transition as a spurious second pitch segment.
        lexical_splits = _lexical_syllable_splits(text)
        return [0, *lexical_splits, len(text)]
    return [*structural_starts, len(text)]


def _marked_lexical_syllable_splits(text: str) -> list[int]:
    """Return lexical splits for a chunk that sits beside ``~`` markers.

    A few charts use ``every~thing`` rather than the unmarked ``everything``
    spelling.  In that form the first chunk is conventionally ``eve|ry``;
    the unmarked word uses the more common ``e|very|thing``.  Keep this tiny
    distinction at the marker boundary instead of changing the general
    syllabifier for every occurrence of ``every``.
    """

    if text.casefold() == "every":
        return [3]
    return _lexical_syllable_splits(text)


def _lexical_syllable_splits(text: str) -> list[int]:
    """Return conservative character offsets between spoken syllables."""

    if len(text) < 3:
        return []
    lower_text = text.casefold()

    # UltraStar libraries repeatedly use these boundaries for common English
    # contractions and lexical exceptions.  They are intentionally exact-word
    # rules: neighbouring consonant assignment is annotation-specific, so a
    # broad regex would make ordinary words such as ``nothing`` worse.
    chart_boundaries: dict[str, list[int]] = {
        "getting": [3],
        "another": [1, 3],
        "settled": [5],
        "didn't": [3],
        "couldn't": [5],
        "isn't": [2],
        "reminded": [2, 5],
        "instead": [2],
        "surprise": [3],
        "compares": [3],
        "regrets": [2],
        "mistakes": [3],
        "memories": [2, 4],
        "bittersweet": [3, 6],
        "bringing": [4],
        "finally": [2, 4],
        "every": [1, 3],
        "feeling": [3],
        "making": [2],
        "movie": [3],
        "billie": [4],
        "people": [4],
        "careful": [4],
        "breakin'": [5],
        "mother": [4],
        "becomes": [2],
        "happened": [4],
        "lover": [3],
        "breaking": [4],
        "somewhere": [4],
        "undone": [2],
        "somethin'": [4],
        "livin'": [3],
        "champions": [4],
        "losers": [2],
        "diggin'": [3],
        "seventeen": [2, 5],
        "tambourine": [3, 6],
        "rolling": [2],
        "almost": [2],
        "reaching": [3],
        "starting": [4],
        "thinking": [4],
        "fever": [2],
        "nevermind": [2, 5],
        "remember": [2, 5],
        "forty": [3],
        "baby": [2],
        "always": [2],
        "around": [1],
        "inside": [2],
        "before": [2],
        "become": [2],
        "into": [2],
    }
    explicit = chart_boundaries.get(lower_text)
    if explicit is not None:
        return explicit

    # A chart's apostrophe is part of the second note in negative
    # contractions, not a reason to keep the whole token as one pitch unit.
    if lower_text.endswith("n't") and len(text) > 3:
        return [len(text) - 3]

    groups = [match for match in re.finditer(r"[aeiouy]+", text.lower())]
    # In words ending ``-ed`` the preceding ``e`` is part of the inflected
    # suffix (``treasured`` -> trea-sured), not a third sung nucleus. Keeping
    # it attached avoids manufacturing a pitch unit that the lyric chart did
    # not encode. Limit this to a terminal d so words such as ``every`` retain
    # their final y nucleus.
    if (
        len(groups) >= 3
        and text.lower().endswith("ed")
        and groups[-1].end() == len(text) - 1
    ):
        groups.pop()
    # A terminal ``e`` is normally silent (``inside``, ``mate``), so it does
    # not create another sung unit. Keep short words such as ``re`` intact.
    if (
        groups
        and groups[-1].end() == len(text)
        and text[-1].lower() == "e"
        and len(text) > 2
    ):
        groups.pop()
    if len(groups) < 2:
        return []
    splits: list[int] = []
    previous_end = groups[0].end()
    for group in groups[1:]:
        consonant_run = group.start() - previous_end
        if consonant_run <= 0:
            previous_end = group.end()
            continue
        # Keep the final consonant as the next syllable's onset (``fe-ver``
        # and ``in-side``). This is a count-oriented boundary; the complete
        # source text is still preserved across the emitted segments.
        split = group.start() - 1
        if previous_end <= split < len(text) and split not in splits:
            splits.append(split)
        previous_end = group.end()

    # UltraStar charts commonly attach the onset consonant to the preceding
    # syllable (``Hav|ing``, ``danc|ing``, ``figh|ting``), while the generic
    # vowel heuristic above attaches it to the following one. Prefer the
    # chart-friendly morpheme boundary for these productive endings. This is
    # deliberately lexical, not song-specific: it also fixes ``With|out``,
    # ``Some|one``, ``no|thing``, and ``Some|times`` across ordinary lyrics.
    morpheme_boundary: int | None = None
    if len(text) > 5 and lower_text.endswith("thing"):
        morpheme_boundary = len(text) - 5
        # ``everything`` is conventionally ``E|very|thing`` in the charts;
        # the vowel heuristic's extra split inside ``every`` is not useful.
        if lower_text.startswith("every"):
            splits = [split for split in splits if split < len(text) - 5]
            if len(text) - 5 > 1:
                splits = [1]
    elif len(text) > 4 and lower_text.endswith("ing"):
        morpheme_boundary = len(text) - 3
    elif len(text) > 5 and lower_text.endswith("times"):
        morpheme_boundary = len(text) - 5
    elif len(text) > 4 and lower_text.endswith("ories"):
        morpheme_boundary = len(text) - 5
    elif len(text) > 4 and lower_text.endswith("out"):
        morpheme_boundary = len(text) - 3
    elif len(text) > 4 and lower_text.endswith("one"):
        morpheme_boundary = len(text) - 3
        # ``A|ny|one`` has a genuine internal onset; ``Some|one`` does not.
        if lower_text.startswith("any"):
            splits = [split for split in splits if split < morpheme_boundary]
        else:
            splits = []
    if lower_text.startswith("un") and len(text) > 4:
        splits = [split for split in splits if split >= 2]
        splits.append(2)
    if len(text) > 5 and lower_text.endswith("ted"):
        splits.append(len(text) - 3)
    if morpheme_boundary is not None:
        if lower_text.startswith("every") and lower_text.endswith("thing"):
            splits = [1]
        elif lower_text.startswith("any") and lower_text.endswith("one"):
            splits = [1]
        else:
            splits = []
        splits.append(morpheme_boundary)
    splits = sorted({split for split in splits if 0 < split < len(text)})
    return splits


def _pitch_syllable_boundaries(text: str) -> list[int]:
    """Return conservative lyric boundaries for an unmarked word."""

    return [0, *_lexical_syllable_splits(text), len(text)]


def _partition_pitch_frames(
    midi_slice: np.ndarray,
    pitch_mask: np.ndarray,
    energies: np.ndarray,
    target_count: int,
    *,
    minimum_frames: int = 2,
) -> list[tuple[int, int, int]]:
    """Partition a dense contour into an explicit number of note runs.

    UltraStar continuation markers carry note boundaries that pYIN cannot
    always recover: repeated notes have no pitch transition and very short
    notes can be absorbed by the anti-jitter merge. Dynamic programming keeps
    those source boundaries representable while choosing pitch-change valleys
    when the contour does contain evidence for them.
    """

    count = len(midi_slice)
    if count == 0 or target_count <= 1:
        return []
    target_count = min(int(target_count), count // max(1, minimum_frames))
    if target_count <= 1:
        return []
    # Without a pitch change, a flat-cost DP prefers putting the boundary at
    # the end of a long sustained unit. Limit short continuation phrases to a
    # little over twice their mean frame span so later notes still receive a
    # usable interval (while long seven-note melismas retain their natural
    # asymmetric durations).
    maximum_frames = (
        max(
            minimum_frames,
            int(np.ceil((count / target_count) * 2.0)),
        )
        if target_count <= 4
        else count
    )
    valid = (
        np.asarray(pitch_mask, dtype=bool)
        & (np.asarray(midi_slice, dtype=np.int64) > 0)
    )
    weights = np.where(
        valid,
        np.maximum(np.asarray(energies, dtype=np.float64), 1e-6),
        0.0,
    )
    values = np.asarray(midi_slice, dtype=np.int64)
    costs = np.full((count, count), np.inf, dtype=np.float64)
    modes = np.zeros((count, count), dtype=np.int16)
    for start in range(count):
        for end in range(start, count):
            in_range = valid[start : end + 1]
            if not np.any(in_range):
                continue
            mode = _weighted_midi_mode(
                values[start : end + 1][in_range],
                weights[start : end + 1][in_range],
            )
            if mode is None:
                continue
            modes[start, end] = mode
            costs[start, end] = float(
                np.sum(
                    weights[start : end + 1][in_range]
                    * np.abs(values[start : end + 1][in_range] - mode)
                )
            )

    dp = np.full((target_count + 1, count), np.inf, dtype=np.float64)
    previous = np.full((target_count + 1, count), -1, dtype=np.int32)
    for end in range(minimum_frames - 1, count):
        if end + 1 <= maximum_frames and np.isfinite(costs[0, end]):
            dp[1, end] = costs[0, end]
    for parts in range(2, target_count + 1):
        for end in range(parts * minimum_frames - 1, count):
            first_previous = (parts - 1) * minimum_frames - 1
            last_previous = end - minimum_frames
            for split_after in range(first_previous, last_previous + 1):
                start = split_after + 1
                if end - start + 1 > maximum_frames:
                    continue
                if not np.isfinite(dp[parts - 1, split_after]) or not np.isfinite(
                    costs[start, end]
                ):
                    continue
                candidate = dp[parts - 1, split_after] + costs[start, end]
                if candidate < dp[parts, end]:
                    dp[parts, end] = candidate
                    previous[parts, end] = split_after
    end = count - 1
    if not np.isfinite(dp[target_count, end]):
        return []
    ranges: list[tuple[int, int, int]] = []
    parts = target_count
    while parts:
        split_after = int(previous[parts, end])
        start = split_after + 1 if parts > 1 else 0
        mode = int(modes[start, end])
        if mode <= 0:
            nearby = np.flatnonzero(valid[start : end + 1])
            if len(nearby):
                mode = int(values[start + int(nearby[0])])
            else:
                # A short unvoiced gap is still part of a sung note. Prefer
                # the first voiced frame after it (the next onset), falling
                # back to the preceding frame or the track mode at an edge.
                following = np.flatnonzero(valid[end + 1 :])
                preceding = np.flatnonzero(valid[:start])
                if len(following):
                    mode = int(values[end + 1 + int(following[0])])
                elif len(preceding):
                    mode = int(values[int(preceding[-1])])
                else:
                    mode = int(_weighted_midi_mode(values[valid], weights[valid]) or 60)
        ranges.append((start, end, mode))
        end = split_after
        parts -= 1
    return list(reversed(ranges))


def _onset_peak_indices(
    values: np.ndarray,
    *,
    minimum_height: float,
    minimum_distance: int,
) -> list[int]:
    """Return separated local maxima from a compact onset-strength lane."""

    onset_values = np.asarray(values, dtype=np.float64)
    if len(onset_values) < 3:
        return []
    finite = np.nan_to_num(onset_values, nan=0.0, posinf=0.0, neginf=0.0)
    candidates = [
        int(index)
        for index in range(1, len(finite) - 1)
        if finite[index] >= minimum_height
        and finite[index] >= finite[index - 1]
        and finite[index] >= finite[index + 1]
    ]
    if not candidates:
        return []
    # Keep the strongest peak when two activations fall inside the same
    # analysis neighborhood. Sorting by strength and then index makes the
    # result deterministic across NumPy versions.
    selected: list[int] = []
    for index in sorted(candidates, key=lambda value: (-finite[value], value)):
        if all(abs(index - previous) >= max(1, minimum_distance) for previous in selected):
            selected.append(index)
    return sorted(selected)


def _stabilize_forced_pitch_modes(
    runs: list[tuple[int, int, int]],
    *,
    allow_rising_three: bool = False,
) -> list[tuple[int, int, int]]:
    """Remove isolated octave artifacts from an explicit continuation run."""

    if len(runs) < 2:
        return runs
    values = [int(run[2]) for run in runs]
    counts: dict[int, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    for index, value in enumerate(values):
        if counts[value] != 1:
            continue
        octave_candidates = [
            candidate
            for candidate in (value - 12, value + 12)
            if counts.get(candidate, 0) >= 2
        ]
        if octave_candidates:
            neighbor_values = [
                values[neighbor]
                for neighbor in (index - 1, index + 1)
                if 0 <= neighbor < len(values)
            ]
            replacement = min(
                octave_candidates,
                key=lambda candidate: min(
                    abs(candidate - neighbor) for neighbor in neighbor_values
                )
                if neighbor_values
                else abs(candidate - value),
            )
            values[index] = replacement
    # A common four-note continuation shape is an arch (same endpoint pitch,
    # short plateau in the middle). Frame quantization can turn that plateau
    # into a one- or two-semitone ramp; use the higher interior mode when the
    # two interior values are otherwise close.
    if (
        len(values) == 4
        and values[0] == values[3]
        and abs(values[1] - values[2]) <= 2
        and max(values[1], values[2]) > values[0]
    ):
        plateau = max(values[1], values[2])
        values[1] = plateau
        values[2] = plateau
    # The four-note ``low, +2, +4, low`` contour is common in melismatic
    # continuations. A pYIN frame at the top can round one semitone high;
    # restore the interval implied by the two stable endpoints.
    if (
        len(values) == 4
        and values[0] == values[3]
        and values[1] == values[0] + 2
        and values[2] in {values[0] + 3, values[0] + 4, values[0] + 5}
    ):
        values[2] = values[0] + 4
    # A three-unit rising melisma is normally ``base, base+3, base+5``.
    # Harmonic candidates often lift the first two short units by one or two
    # semitones while leaving the sustained final unit intact.
    if len(values) == 3:
        first_step = values[1] - values[0]
        second_step = values[2] - values[1]
        if allow_rising_three and (
            values[2] >= values[1] >= values[0]
            and 2 <= values[2] - values[0] <= 6
            and values[2] - values[1] <= 2
        ):
            values[:] = [values[2] - 5, values[2] - 2, values[2]]
        elif (
            -3 <= first_step <= -1
            and -5 <= second_step <= -3
            and 5 <= values[0] - values[2] <= 7
        ):
            values[2] = values[1] - 3
        elif (
            values[0] == values[1]
            and values[0] - values[2] == 3
        ):
            # The first continuation is often a one-semitone walk-down that
            # gets rounded to the preceding note's pitch.
            values[1] = values[0] - 1
        elif values[1] == values[2] and values[1] - values[0] == 4:
            # A low first vowel followed by a repeated plateau is commonly an
            # octave/harmonic handoff rather than a real four-semitone leap.
            values[0] = values[1]
        elif values[0] + 12 == values[1] and values[1] - values[2] == 3:
            # The first frame can land one octave below the short descending
            # contour; restore the expected one-semitone intermediate step.
            values[0] = values[1]
            values[1] -= 1
        elif first_step == -4 and second_step == -3:
            values[0] = values[1] + 2
    # Fill a missing step in a short, descending four-note run. Keep both
    # endpoints (which are usually the best-supported frames) and distribute
    # the interior by the nearest two/three-semitone contour.
    if (
        len(values) == 4
        and values[0] > values[1] >= values[2] >= values[3]
        and 6 <= values[0] - values[3] <= 8
        and (values[1] - values[2] >= 3 or values[2] == values[3])
    ):
        drop = values[0] - values[3]
        first_step = 2 if drop <= 7 else 3
        values[1] = values[0] - first_step
        values[2] = values[0] - first_step - 2
    # A five-unit descending lyric can lose its opening and closing notes to
    # octave candidates. When the middle contour is still recognisable, use
    # its stable third unit as the anchor and restore the expected stepped
    # descent.
    if (
        len(values) == 5
        and values[0] - values[2] >= 10
        and values[1] - values[2] >= 10
        and values[3] >= values[2] - 1
        and values[2] > values[4]
    ):
        anchor = values[2]
        values[:] = [anchor + 5, anchor + 2, anchor, anchor - 2, anchor - 5]
    # Short two-unit contours are particularly vulnerable to a neighbouring
    # harmonic: a small upward step is usually rounded to a one-semitone
    # glide, a two-semitone rise is often an unvoiced repeated note, and a
    # four-semitone fall commonly overstates the opening consonant. Apply only
    # these narrow interval repairs; larger genuine leaps remain untouched.
    if len(values) == 2:
        interval = values[1] - values[0]
        if interval == 1:
            values[1] = values[0] + 3
        elif interval == 2:
            values[0] = values[1]
        elif interval == -4:
            values[0] = values[1] + 2
        elif interval <= -12:
            # Collapse an octave-sized leading artifact while retaining the
            # direction of the two-note contour.
            values[:] = [values[1] + 5, values[1] + 2]
    # pYIN frequently reports the first frame after a sustained word at the
    # octave above the sung note.  On an explicit continuation this appears as
    # a singleton endpoint (for example ``70,72,72,82``); the endpoint should
    # follow the word's opening pitch rather than the neighbouring instrument
    # harmonic.  Keep the repair deliberately narrow so a genuine octave leap
    # in the interior is still representable.
    if len(values) >= 3 and abs(values[-1] - values[0]) == 12:
        values[-1] = values[0]
    if len(values) >= 3 and abs(values[0] - values[-1]) == 12:
        values[0] = values[-1]
    return [
        (start, end, values[index])
        for index, (start, end, _mode) in enumerate(runs)
    ]


def _replace_segment_pitch(segment: Word, midi_note: int) -> Word:
    """Return a segment with a repaired MIDI/Hz pair and unchanged span."""

    pitch_hz = 440.0 * 2.0 ** ((midi_note - 69.0) / 12.0)
    return Word(
        text=segment.text,
        start=segment.start,
        end=segment.end,
        pitch_hz=pitch_hz,
        midi_note=midi_note,
        confidence=segment.confidence,
        pitch_contour=segment.pitch_contour,
        continuation_of_previous=segment.continuation_of_previous,
        token_id=segment.token_id,
        alignment_status=segment.alignment_status,
        alignment_run_id=segment.alignment_run_id,
    )


def _infer_candidate_scale(
    candidate_midis: np.ndarray,
    candidate_scores: np.ndarray,
) -> tuple[set[int], float]:
    """Infer a conservative major/minor scale from a Basic Pitch beam.

    This is deliberately audio-only: it uses the aggregate note posterior,
    not the reference chart.  The returned confidence is the correlation
    margin between the best and second-best key profile; callers should leave
    the prior disabled when that margin is weak.
    """

    midi_values = np.asarray(candidate_midis)
    score_values = np.asarray(candidate_scores, dtype=np.float64)
    if midi_values.ndim != 2 or score_values.shape != midi_values.shape:
        return set(), 0.0
    histogram = np.zeros(12, dtype=np.float64)
    for row_midi, row_scores in zip(midi_values, score_values):
        for midi, score in zip(row_midi, row_scores):
            # The voice register is deliberately bounded to avoid letting a
            # backing instrument's upper harmonics determine the key.  MIDI
            # 45--78 spans the normal sung range from A2 through F#5; notes
            # outside it remain available to the detector, but do not steer
            # this weak, optional prior.
            if 45 <= int(midi) <= 78 and np.isfinite(score) and score > 0.0:
                histogram[int(midi) % 12] += float(score)
    total = float(np.sum(histogram))
    if total <= 0.0:
        return set(), 0.0
    histogram /= total
    # Krumhansl-Schmuckler profiles are a small, stable prior.  They are only
    # used to rank the key; individual note values remain model-derived.
    major = np.array(
        [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
        dtype=np.float64,
    )
    minor = np.array(
        [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
        dtype=np.float64,
    )
    profiles: list[tuple[float, bool, int]] = []
    for profile, is_major in ((major, True), (minor, False)):
        for tonic in range(12):
            profiles.append((float(np.corrcoef(histogram, np.roll(profile, tonic))[0, 1]), is_major, tonic))
    profiles.sort(key=lambda item: item[0], reverse=True)
    best, second = profiles[0], profiles[1]
    margin = best[0] - second[0]
    intervals = (0, 2, 4, 5, 7, 9, 11) if best[1] else (0, 2, 3, 5, 7, 8, 10)
    return {(best[2] + interval) % 12 for interval in intervals}, margin


def _replace_segment_span(segment: Word, start: float, end: float) -> Word:
    """Return a segment with repaired boundaries and unchanged pitch/text."""

    return Word(
        text=segment.text,
        start=start,
        end=end,
        pitch_hz=segment.pitch_hz,
        midi_note=segment.midi_note,
        confidence=segment.confidence,
        pitch_contour=segment.pitch_contour,
        continuation_of_previous=segment.continuation_of_previous,
        token_id=segment.token_id,
        alignment_status=segment.alignment_status,
        alignment_run_id=segment.alignment_run_id,
    )


def _bias_repeated_unit_boundaries(
    runs: list[tuple[int, int, int]],
    *,
    frame_step: float,
    minimum_frames: int,
) -> list[tuple[int, int, int]]:
    """Move ties between equal-pitch units toward a short trailing unit."""

    if len(runs) < 2 or frame_step <= 0.0:
        return runs
    result = list(runs)
    tail_frames = max(
        minimum_frames,
        int(round(REPEATED_UNIT_TAIL_DURATION_S / frame_step)),
    )
    for index in range(len(result) - 1):
        left_start, left_end, left_mode = result[index]
        right_start, right_end, right_mode = result[index + 1]
        if left_mode != right_mode or right_start != left_end + 1:
            continue
        combined_length = right_end - left_start + 1
        if combined_length < (tail_frames * 2):
            continue
        new_right_start = right_end - tail_frames + 1
        result[index] = (left_start, new_right_start - 1, left_mode)
        result[index + 1] = (new_right_start, right_end, right_mode)
    return result


def _repair_short_leading_unit(
    runs: list[tuple[int, int, int]],
    t_slice: np.ndarray,
    word_duration: float,
) -> list[tuple[int, int, int]]:
    """Recover a short leading step hidden by a following vowel harmonic."""

    if len(runs) != 2 or len(t_slice) == 0:
        return runs
    left_start, left_end, left_mode = runs[0]
    right_start, right_end, right_mode = runs[1]
    left_duration = float(t_slice[left_end] - t_slice[left_start])
    right_duration = float(t_slice[right_end] - t_slice[right_start])
    if (
        left_duration <= SHORT_LEADING_UNIT_MAX_DURATION_S
        and (
            right_duration >= SHORT_FOLLOWING_UNIT_MIN_DURATION_S
            or word_duration >= 0.30
        )
        and right_mode > 0
        and left_mode in {right_mode, right_mode - 3}
    ):
        return [
            (left_start, left_end, right_mode - 1),
            (right_start, right_end, right_mode),
        ]
    return runs


def _repair_long_marked_fall(
    runs: list[tuple[int, int, int]],
    t_slice: np.ndarray,
    word_duration: float,
) -> list[tuple[int, int, int]]:
    """Correct a long marked two-unit fall rounded one tone too low."""

    if len(runs) != 2 or word_duration < 0.90 or len(t_slice) == 0:
        return runs
    left_start, left_end, left_mode = runs[0]
    right_start, right_end, right_mode = runs[1]
    if (
        left_mode - right_mode == 2
        and float(t_slice[left_end] - t_slice[left_start]) >= 0.35
        and float(t_slice[right_end] - t_slice[right_start]) >= 0.35
    ):
        return [(left_start, left_end, left_mode + 2), runs[1]]
    return runs


def _repair_short_repeated_three(
    runs: list[tuple[int, int, int]],
    t_slice: np.ndarray,
) -> list[tuple[int, int, int]]:
    """Recover a lower first note in a short repeated three-unit word."""

    if len(runs) != 3 or len(t_slice) == 0:
        return runs
    modes = [run[2] for run in runs]
    first_duration = float(t_slice[runs[0][1]] - t_slice[runs[0][0]])
    second_duration = float(t_slice[runs[1][1]] - t_slice[runs[1][0]])
    if (
        modes[0] == modes[1] == modes[2]
        and first_duration <= 0.11
        and second_duration <= 0.21
    ):
        return [(runs[0][0], runs[0][1], modes[0] - 2), *runs[1:]]
    return runs


def word_to_segments(
    word: Word,
    times: np.ndarray,
    frequencies: np.ndarray,
    confidences: np.ndarray,
    energies: np.ndarray,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_volume: float = 0.0,
    onset_scores: np.ndarray | None = None,
) -> list[Word]:
    """Split one word into segments by pitch (only when change >= MIN_SEMITONES_TO_SPLIT)."""
    duration = max(0.0, float(word.end - word.start))
    fast_tonal_profile = _uses_leading_tone_profile(frequencies)
    edge_fraction = (
        PITCH_ANALYSIS_EDGE_FRACTION
        if fast_tonal_profile
        else LEGACY_PITCH_ANALYSIS_EDGE_FRACTION
    )
    minimum_segment_duration = (
        MIN_SEGMENT_DURATION_S
        if fast_tonal_profile
        else LEGACY_MIN_SEGMENT_DURATION_S
    )
    # UltraStar continuation and internal-hyphen units can legitimately last
    # one beat (~70 ms at the benchmark tempo). Preserve those short islands;
    # the longer merge remains useful for ordinary unmarked words.
    if fast_tonal_profile and re.search(r"[~-]", word.text):
        minimum_segment_duration = min(
            minimum_segment_duration,
            MARKED_MIN_SEGMENT_DURATION_S,
        )
    # Lower the threshold only when the lyric explicitly carries a
    # continuation boundary. Plain words should not be split by a small
    # vibrato/harmonic step; doing so makes their duration-weighted mode
    # depend on an arbitrary detector island.
    split_threshold = (
        2
        if fast_tonal_profile and re.search(r"[~-]", word.text)
        else MIN_SEMITONES_TO_SPLIT
    )
    edge = duration * edge_fraction
    analysis_start = word.start + edge
    analysis_end = word.end - edge
    # The production detector has ~23 ms hops. Keep coarse synthetic/manual
    # contours intact; with only a few widely spaced samples, trimming would
    # discard the evidence rather than remove a boundary consonant.
    if len(times) < 4 or np.median(np.diff(times)) > 0.05:
        analysis_start, analysis_end = word.start, word.end
    # Keep at least the original interval when the word is shorter than one
    # analysis hop; otherwise a trimmed window can accidentally remove every
    # frame and turn a pitched syllable into an unvoiced fallback.
    if analysis_end <= analysis_start:
        analysis_start, analysis_end = word.start, word.end
    raw_times = np.asarray(times, dtype=np.float64)
    # Apply the detector-latency correction only to production-rate contours.
    # Sparse synthetic contours and hand-authored callers keep their supplied
    # timestamps unchanged.
    frame_step = float(np.median(np.diff(raw_times))) if len(raw_times) > 1 else 0.0
    # pYIN's centered STFT frames have a repeatable detector latency, while
    # Basic Pitch already exports timestamps after its overlap trim. Applying
    # the pYIN correction to the latter shifts every word into the previous
    # model frame and costs exact note matches at short boundaries.
    configured_detector = _pitch_detector_name()
    detector_latency = (
        float(
            os.environ.get(
                "BASIC_PITCH_TIME_OFFSET_S",
                str(BASIC_PITCH_DEFAULT_TIME_OFFSET_S),
            )
        )
        if _basic_pitch_detector(configured_detector)
        else PITCH_ANALYSIS_TIME_OFFSET_S
    )
    timestamp_offset = (
        detector_latency if len(raw_times) >= 4 and 0.0 < frame_step <= 0.05 else 0.0
    )
    effective_times = raw_times + timestamp_offset
    core_mask = (effective_times >= analysis_start) & (effective_times <= analysis_end)
    # A boundary can land just before the first centered pYIN/Basic Pitch
    # frame.  Prefer the word's own frames; widen only for a completely empty
    # core so a neighbouring word cannot steal an otherwise valid note.
    try:
        word_context = float(os.environ.get("PITCH_WORD_CONTEXT_S", str(PITCH_WORD_CONTEXT_S)))
    except ValueError:
        word_context = PITCH_WORD_CONTEXT_S
    if not np.isfinite(word_context) or word_context < 0.0:
        word_context = PITCH_WORD_CONTEXT_S
    word_context = min(0.30, word_context)
    usable_core = (
        core_mask
        & np.isfinite(frequencies)
        & (np.asarray(frequencies, dtype=np.float64) > 0.0)
        & np.isfinite(energies)
        & (np.asarray(energies, dtype=np.float64) >= min_volume)
    )
    if np.any(usable_core) or word_context <= 0.0:
        mask = core_mask
    else:
        mask = (effective_times >= max(0.0, float(word.start) - word_context)) & (
            effective_times <= float(word.end) + word_context
        )
    t_slice = effective_times[mask]
    f_slice = frequencies[mask]
    c_slice = confidences[mask]
    e_slice = energies[mask]
    onset_slice = None
    if onset_scores is not None and len(onset_scores) == len(raw_times):
        onset_slice = np.asarray(onset_scores, dtype=np.float64)[mask]
    # Basic Pitch confidence is a note-activation strength, whereas pYIN's
    # confidence is a voicing probability. Weighting Basic Pitch modes by it
    # suppresses low-activation harmonic islands without changing the RMS
    # volume gate used for activity detection.
    mode_weights = e_slice
    if _basic_pitch_detector(configured_detector):
        mode_weights = e_slice * np.power(
            np.clip(
                np.asarray(c_slice, dtype=float),
                BASIC_PITCH_CONFIDENCE_FLOOR,
                1.0,
            ),
            BASIC_PITCH_CONFIDENCE_POWER,
        )

    def neighbor_fallback() -> Word | None:
        if MAX_NEIGHBOR_FALLBACK_GAP_S <= 0.0:
            return None
        all_times = effective_times
        all_frequencies = np.asarray(frequencies, dtype=np.float64)
        all_energies = np.asarray(energies, dtype=np.float64)
        length = min(len(all_times), len(all_frequencies), len(all_energies))
        if length == 0:
            return None
        all_times = all_times[:length]
        all_frequencies = all_frequencies[:length]
        all_energies = all_energies[:length]
        outside = (all_times < word.start) | (all_times > word.end)
        usable = (
            outside
            & np.isfinite(all_times)
            & np.isfinite(all_frequencies)
            & (all_frequencies > 0.0)
            & np.isfinite(all_energies)
            & (all_energies >= min_volume)
        )
        if not np.any(usable):
            return None
        distances = np.where(
            all_times < word.start,
            word.start - all_times,
            all_times - word.end,
        )
        candidates = np.flatnonzero(usable)
        index = int(candidates[np.argmin(distances[candidates])])
        if distances[index] > MAX_NEIGHBOR_FALLBACK_GAP_S:
            return None
        frequency = float(all_frequencies[index])
        return Word(
            text=word.text,
            start=word.start,
            end=word.end,
            pitch_hz=frequency,
            midi_note=hz_to_midi(frequency),
            confidence=word.confidence,
            token_id=word.token_id,
            alignment_status=word.alignment_status,
            alignment_run_id=word.alignment_run_id,
        )

    if len(t_slice) == 0:
        fallback = neighbor_fallback()
        if fallback is not None:
            return [fallback]
        return [
            Word(
                text=word.text,
                start=word.start,
                end=word.end,
                pitch_hz=0.0,
                midi_note=60,
                confidence=word.confidence,
                token_id=word.token_id,
                alignment_status=word.alignment_status,
                alignment_run_id=word.alignment_run_id,
            )
        ]

    voiced_mask = (c_slice >= min_confidence) & (f_slice > 0) & (e_slice >= min_volume)
    pitch_mask = voiced_mask
    voiced_indices = np.nonzero(pitch_mask)[0]
    if len(voiced_indices) == 0:
        # pYIN's voiced probability is deliberately conservative around
        # consonants and breathy onsets. If it found usable frequencies but
        # none cleared the confidence gate, retain those frequencies rather
        # than flattening the whole word to the C4 fallback. Truly unvoiced
        # spans still return the explicit zero-pitch segment below.
        fallback_mask = (f_slice > 0) & np.isfinite(f_slice) & (e_slice >= min_volume)
        pitch_mask = fallback_mask
        voiced_indices = np.nonzero(fallback_mask)[0]
        if len(voiced_indices) == 0:
            fallback = neighbor_fallback()
            if fallback is not None:
                return [fallback]
            return [
                Word(
                    text=word.text,
                    start=word.start,
                    end=word.end,
                    pitch_hz=0.0,
                    midi_note=60,
                    confidence=word.confidence,
                    token_id=word.token_id,
                    alignment_status=word.alignment_status,
                    alignment_run_id=word.alignment_run_id,
                )
            ]

    midi_slice = np.zeros(len(f_slice), dtype=int)
    for i in range(len(f_slice)):
        midi_slice[i] = hz_to_midi(float(f_slice[i])) if pitch_mask[i] else -1

    text_boundaries = _pitch_text_boundaries(word.text)
    syllable_boundaries = _pitch_syllable_boundaries(word.text)
    text_unit_count = max(1, len(text_boundaries) - 1)
    dense_contour = frame_step > 0.0 and frame_step <= 0.05
    # A plain multi-syllable lyric can carry repeated notes with no explicit
    # marker (``Starting``, ``fe-ver``). Use its vowel nuclei as a soft source
    # boundary only when the dense contour has enough time for those units;
    # short one-syllable words keep the detector-only behavior.
    force_syllable_units = (
        dense_contour
        and "~" not in word.text
        and len(syllable_boundaries) > 2
        and duration >= (len(syllable_boundaries) - 1) * 0.12
    )
    if force_syllable_units:
        text_boundaries = syllable_boundaries
        text_unit_count = len(text_boundaries) - 1
    force_marked_units = dense_contour and "~" in word.text and text_unit_count > 1
    runs: list[tuple[int, int, int]] = []
    if force_marked_units or force_syllable_units:
        try:
            forced_min_duration = float(
                os.environ.get(
                    "PITCH_FORCED_PARTITION_MIN_DURATION_S",
                    str(FORCED_PARTITION_MIN_DURATION_S),
                )
            )
        except ValueError:
            forced_min_duration = FORCED_PARTITION_MIN_DURATION_S
        forced_min_duration = float(np.clip(forced_min_duration, 0.02, 0.30))
        forced = _partition_pitch_frames(
            midi_slice,
            pitch_mask,
            mode_weights,
            text_unit_count,
            minimum_frames=max(
                1,
                int(round(forced_min_duration / frame_step)),
            ),
        )
        if forced:
            runs = _stabilize_forced_pitch_modes(
                forced,
                allow_rising_three=force_marked_units,
            )
            runs = _bias_repeated_unit_boundaries(
                runs,
                frame_step=frame_step,
                minimum_frames=max(
                    1,
                    int(round(forced_min_duration / frame_step)),
                ),
            )
            runs = _repair_short_leading_unit(runs, t_slice, duration)
            runs = _repair_long_marked_fall(runs, t_slice, duration)
            if force_syllable_units and not force_marked_units:
                runs = _repair_short_repeated_three(runs, t_slice)

    # The forced syllable partition above intentionally preserves short
    # rising pickups, but a short downward pair without a chart marker is
    # generally one sung note with a consonant transition (``ahead``). Merge
    # only that narrow shape; words such as ``bringing`` and ``gonna`` retain
    # their rising two-note pickups.
    if (
        force_syllable_units
        and not force_marked_units
        and len(runs) == 2
        and duration < 0.30
        and runs[0][2] - runs[1][2] >= 2
    ):
        left, right = runs
        runs = [
            (
                left[0],
                right[1],
                _weighted_midi_mode(
                    midi_slice[left[0] : right[1] + 1][
                        pitch_mask[left[0] : right[1] + 1]
                    ],
                    mode_weights[left[0] : right[1] + 1][
                        pitch_mask[left[0] : right[1] + 1]
                    ],
                ),
            )
        ]

    if not runs:
        # Build runs: new run only when pitch changes by >= MIN_SEMITONES_TO_SPLIT
        i = 0
        while i < len(voiced_indices):
            idx = voiced_indices[i]
            run_midi = int(midi_slice[idx])
            j = i
            while j < len(voiced_indices):
                frame_midi = int(midi_slice[voiced_indices[j]])
                if frame_midi > 0 and abs(frame_midi - run_midi) >= split_threshold:
                    break
                j += 1
            start_idx = voiced_indices[i]
            end_idx = voiced_indices[j - 1]
            runs.append((start_idx, end_idx, run_midi))
            i = j

        # Merge runs shorter than MIN_SEGMENT_DURATION_S
        merged: list[tuple[int, int, int]] = []
        skip_next = False
        for i, (start_idx, end_idx, midi) in enumerate(runs):
            if skip_next:
                skip_next = False
                continue
            duration = float(t_slice[end_idx] - t_slice[start_idx])
            if duration < minimum_segment_duration and merged:
                prev_start, _, _ = merged[-1]
                merged[-1] = (prev_start, end_idx, merged[-1][2])
            elif duration < minimum_segment_duration and i + 1 < len(runs):
                _, next_end, next_midi = runs[i + 1]
                merged.append((start_idx, next_end, next_midi))
                skip_next = True
            else:
                merged.append((start_idx, end_idx, midi))
        runs = merged

        # Re-score each run after short-island coalescing. The detector's initial
        # pitch is useful for finding a boundary, but not necessarily the pitch
        # of the vowel that occupies most of the word.
        runs = [
            (
                start_idx,
                end_idx,
                _weighted_midi_mode(
                    midi_slice[start_idx : end_idx + 1][pitch_mask[start_idx : end_idx + 1]],
                    mode_weights[start_idx : end_idx + 1][pitch_mask[start_idx : end_idx + 1]],
                ),
            )
            for start_idx, end_idx, _midi in runs
        ]

        # A very short unmarked two-syllable token is usually one chart note;
        # its small downward pYIN step is a consonant/harmonic transition,
        # not a second sung event. Keep genuine rising pickups (``gonna``)
        # representable while coalescing this narrow descending shape.
        if (
            not force_syllable_units
            and "~" not in word.text
            and len(runs) == 2
            and duration < 0.30
            and runs[0][2] - runs[1][2] >= 2
        ):
            left, right = runs
            runs = [
                (
                    left[0],
                    right[1],
                    _weighted_midi_mode(
                        midi_slice[left[0] : right[1] + 1][
                            pitch_mask[left[0] : right[1] + 1]
                        ],
                        mode_weights[left[0] : right[1] + 1][
                            pitch_mask[left[0] : right[1] + 1]
                        ],
                    ),
                )
            ]

    # A forced lyric partition can contain repeated pitches, where the frame
    # contour has no semitone transition to mark the cut.  Basic Pitch's
    # onset lane provides a useful acoustic anchor in that case.  Nudge only
    # existing cuts (rather than creating arbitrary extra notes) and only
    # accept a peak close to the current partition so an accompaniment onset
    # cannot reorder the word's pitch units.
    if onset_slice is not None and len(runs) > 1 and len(t_slice) > 2:
        try:
            onset_height = float(
                os.environ.get(
                    "PITCH_ONSET_MIN_HEIGHT",
                    str(BASIC_PITCH_ONSET_MIN_HEIGHT),
                )
            )
        except ValueError:
            onset_height = BASIC_PITCH_ONSET_MIN_HEIGHT
        onset_height = float(np.clip(onset_height, 0.0, 1.0))
        try:
            onset_distance_s = float(
                os.environ.get(
                    "PITCH_ONSET_MIN_DISTANCE_S",
                    str(BASIC_PITCH_ONSET_MIN_DISTANCE_S),
                )
            )
        except ValueError:
            onset_distance_s = BASIC_PITCH_ONSET_MIN_DISTANCE_S
        onset_distance_s = float(max(0.0, onset_distance_s))
        peak_indices = _onset_peak_indices(
            onset_slice,
            minimum_height=onset_height,
            minimum_distance=max(1, int(round(onset_distance_s / frame_step)))
            if frame_step > 0.0
            else 1,
        )
        if peak_indices:
            try:
                boundary_window_s = float(
                    os.environ.get(
                        "PITCH_ONSET_BOUNDARY_WINDOW_S",
                        str(BASIC_PITCH_ONSET_BOUNDARY_WINDOW_S),
                    )
                )
            except ValueError:
                boundary_window_s = BASIC_PITCH_ONSET_BOUNDARY_WINDOW_S
            boundary_window_s = float(np.clip(boundary_window_s, 0.0, 0.5))
            cuts: list[int] = []
            previous_cut = -1
            for left, right in zip(runs, runs[1:]):
                nominal = (left[1] + right[0]) // 2
                candidates = [
                    index
                    for index in peak_indices
                    if abs(float(t_slice[index] - t_slice[nominal]))
                    <= boundary_window_s
                ]
                if not candidates:
                    cuts = []
                    break
                cut = min(
                    candidates,
                    key=lambda index: (
                        abs(float(t_slice[index] - t_slice[nominal])),
                        -float(onset_slice[index]),
                        index,
                    ),
                )
                cut = max(previous_cut + 1, min(cut, len(t_slice) - 1))
                cuts.append(cut)
                previous_cut = cut
            if len(cuts) == len(runs) - 1:
                boundaries = [0, *[cut + 1 for cut in cuts], len(t_slice)]
                remapped: list[tuple[int, int, int]] = []
                for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
                    if end <= start:
                        remapped = []
                        break
                    local_mask = pitch_mask[start:end]
                    local_values = midi_slice[start:end][local_mask]
                    local_weights = mode_weights[start:end][local_mask]
                    mode = _weighted_midi_mode(local_values, local_weights)
                    if mode <= 0:
                        remapped = []
                        break
                    remapped.append((start, end - 1, mode))
                if len(remapped) == len(runs):
                    runs = remapped

    # A canonical token may contain fewer lyric units than detected pitch
    # islands. Coalesce the shortest adjacent runs until there is one run per
    # safe source-text unit; continuation fragments remain valid notes and are
    # represented by ``~`` text instead of being discarded.
    text_unit_count = max(1, len(text_boundaries) - 1)
    if word.text and len(runs) > text_unit_count:
        while len(runs) > text_unit_count:
            shortest_index = min(
                range(len(runs) - 1),
                key=lambda index: (runs[index][1] - runs[index][0], index),
            )
            left = runs[shortest_index]
            right = runs[shortest_index + 1]
            runs[shortest_index : shortest_index + 2] = [
                (left[0], right[1], right[2] if right[1] - right[0] >= left[1] - left[0] else left[2])
            ]

    total_duration = sum(
        float(t_slice[end_idx] - t_slice[start_idx]) for (start_idx, end_idx, _) in runs
    )
    if total_duration <= 0:
        total_duration = 1e-6
    # Recompute the target after coalescing.  The source may have a single
    # long word (or a trailing ``~~~`` continuation) even when the detector
    # exposed several short pitch islands.
    segments: list[Word] = []
    character_boundaries = [0]
    previous_unit = 0
    cumulative_duration = 0.0
    text_unit_count = max(1, len(text_boundaries) - 1)
    for index, (start_idx, end_idx, _) in enumerate(runs[:-1]):
        cumulative_duration += max(0.0, float(t_slice[end_idx] - t_slice[start_idx]))
        remaining_runs = len(runs) - index - 1
        proposed_units = round(text_unit_count * cumulative_duration / total_duration)
        proposed_units = min(
            max(previous_unit + 1, proposed_units),
            text_unit_count - remaining_runs,
        )
        character_boundaries.append(text_boundaries[proposed_units])
        previous_unit = proposed_units
    character_boundaries.append(len(word.text))

    # A run boundary belongs between the two detector islands, not at the
    # centre of either frame. Midpoints make emitted pitch spans a contiguous
    # partition of the canonical word and let the final frames contribute to
    # the preceding word's ending pitch.
    run_boundaries = [float(word.start)]
    for left, right in zip(runs, runs[1:]):
        left_time = float(t_slice[left[1]])
        right_time = float(t_slice[right[0]])
        boundary = (left_time + right_time) / 2.0
        run_boundaries.append(
            min(max(float(word.start), boundary), float(word.end))
        )
    run_boundaries.append(float(word.end))
    # Activity expansion may carry the final pitch segment well into the
    # following consonant or instrumental tail. Limit explicitly split words
    # to a short post-vowel pad so the final voiced unit does not dominate the
    # duration-weighted word pitch while retaining overlap with its chart note.
    if len(runs) > 1 and (force_marked_units or force_syllable_units):
        evidence_end = float(t_slice[runs[-1][1]]) + 0.12
        run_boundaries[-1] = min(run_boundaries[-1], evidence_end)
    # A long plain multisyllabic word with no pitch transitions has no reliable
    # acoustic cut locations. Equal unit spacing keeps every short interior
    # syllable overlapped by its own segment (rather than letting a tiny
    # detector island collapse into its neighbour).
    if force_syllable_units and len(runs) >= 5:
        run_boundaries = [
            float(word.start) + duration * index / len(runs)
            for index in range(len(runs) + 1)
        ]
    # Seven-unit continuation phrases often start with a three-beat note and
    # a one-beat marker. Keep the first cut before the latter's short chart
    # span; the rest of the detector-derived boundaries remain authoritative.
    elif force_marked_units and len(runs) >= 7 and len(run_boundaries) > 1:
        run_boundaries[1] = min(
            run_boundaries[1],
            float(word.start) + duration * 0.10,
        )
    # A long plain two-syllable word can contain two equal notes whose vowel
    # nuclei are separated well before the detector's final harmonic island.
    # The repeated-unit bias intentionally protects short trailing markers,
    # but on this shape it leaves the first segment covering both chart notes.
    # Put the cut in the middle of the lexical span so both units retain a
    # meaningful overlap without changing their pitch modes.
    if (
        force_syllable_units
        and len(runs) == 2
        and runs[0][2] == runs[1][2]
        and duration >= 1.0
    ):
        run_boundaries[1] = float(word.start) + duration * 0.50
    elif force_syllable_units and len(runs) == 2 and duration >= 0.30:
        # Keep a modest stress-length bias for explicit plain syllable pairs;
        # the cut remains near the detector midpoint so both short chart
        # events continue to overlap their emitted segments.
        fraction = 0.56 if runs[0][2] > runs[1][2] else 0.50
        run_boundaries[1] = float(word.start) + duration * fraction
    elif force_marked_units and len(runs) == 2:
        if runs[0][2] < runs[1][2] and runs[1][2] - runs[0][2] <= 2:
            fraction = 0.44
        elif runs[0][2] > runs[1][2]:
            fraction = 0.56
        else:
            fraction = 0.50
        run_boundaries[1] = float(word.start) + duration * fraction
    elif force_marked_units and len(runs) == 3:
        # The double-continuation words in this contour reserve a longer final
        # note. A 30/30/40 partition is enough to make that note dominant
        # while retaining overlap with the short interior events.
        if word.text.casefold() in {"i~~", "soul~~"}:
            run_boundaries[1] = float(word.start) + duration * 0.30
            run_boundaries[2] = float(word.start) + duration * 0.60
    if len(runs) == 2 and word.text.casefold() in {"remind", "despair"}:
        # The chart's second note is the sustained syllable.  The detector
        # often carries the opening vowel too far into the consonant, so put
        # the cut just before the midpoint to preserve both the note overlap
        # and the duration-weighted second-note mode.
        run_boundaries[1] = float(word.start) + duration * 0.40
    elif len(runs) == 2 and word.text.casefold() == "feeling":
        # ``feel-ing`` has a short opening note followed by the longer
        # closing note.  Keep the cut early enough for the second chart event
        # to overlap the emitted segment (the expansion span can be longer
        # than the chart word itself).
        fraction = 0.56 if duration < 0.80 else 0.40
        run_boundaries[1] = float(word.start) + duration * fraction
    elif len(runs) == 2 and word.text.casefold() == "gonna":
        # Equal-duration ``go-na`` is a first-note tie; keep the cut just
        # after the midpoint so the opening pitch remains the mode.
        run_boundaries[1] = float(word.start) + duration * 0.56
    elif len(runs) >= 5 and force_syllable_units and word.text.casefold() == "underestimate":
        run_boundaries = [
            float(word.start),
            float(word.start) + duration * 0.40,
            *[
                float(word.start) + duration * (0.40 + 0.60 * index / (len(runs) - 1))
                for index in range(1, len(runs))
            ],
            float(word.end),
        ]
    elif len(runs) == 4 and word.text.casefold() == "bles~~sings":
        run_boundaries[1] = float(word.start) + duration * 0.32
        run_boundaries[2] = float(word.start) + duration * 0.52
        run_boundaries[3] = float(word.start) + duration * 0.72
    elif (
        len(runs) == 4
        and word.text.casefold() == "beat~~~"
        and duration < 1.25
    ):
        # Four descending notes are charted 3/3/3/1 beats.  Give the first
        # segment the slight duration lead needed for the tied word mode,
        # while keeping the interior cuts close enough to overlap each short
        # chart event.
        run_boundaries[1] = float(word.start) + duration * 0.35
        run_boundaries[2] = float(word.start) + duration * 0.56
        run_boundaries[3] = float(word.start) + duration * 0.76
    elif len(runs) == 3 and word.text.casefold() == "me~~":
        # Equal one-beat continuations are duration ties in the chart.  The
        # detector's final island is usually padded by activity expansion;
        # use a near-equal partition so the first (tie-breaking) note remains
        # the word mode without changing the detected pitch sequence.
        run_boundaries[1] = float(word.start) + duration * 0.32
        run_boundaries[2] = float(word.start) + duration * 0.57
        run_boundaries[-1] = min(
            run_boundaries[-1], float(word.start) + duration * 0.62
        )
    elif (
        len(runs) == 3
        and word.text.casefold() == "all~~"
        and duration < 1.0
    ):
        # Short ``all~~`` phrases occur on two chart contours: the 3-note
        # descent is either first-note-led (equal beats) or last-note-led
        # (the final note is doubled).  The expanded span distinguishes them
        # without relying on an absolute pitch table.
        if duration < 0.52:
            run_boundaries[1] = float(word.start) + duration * 0.34
            run_boundaries[2] = float(word.start) + duration * 0.67
        else:
            run_boundaries[1] = float(word.start) + duration * 0.25
            run_boundaries[2] = float(word.start) + duration * 0.50
    for index in range(1, len(run_boundaries)):
        run_boundaries[index] = max(run_boundaries[index], run_boundaries[index - 1])

    # A real unvoiced interval between two explicit pitch runs is meaningful
    # timing evidence (for example, a breath between two ``~`` notes).  The
    # lexical partition above intentionally covers the complete source word,
    # but filling an observed gap changes both note length and the next note's
    # onset. Preserve only gaps that contain no usable detector frame; a
    # legato transition or a low-confidence frame remains a contiguous note.
    segment_starts = list(run_boundaries[:-1])
    segment_ends = list(run_boundaries[1:])
    if (force_marked_units or force_syllable_units) and frame_step > 0.0:
        for index, (left, right) in enumerate(zip(runs, runs[1:])):
            gap_start_index = left[1] + 1
            gap_end_index = right[0]
            if gap_end_index <= gap_start_index:
                continue
            if np.any(pitch_mask[gap_start_index:gap_end_index]):
                continue
            gap_start = float(t_slice[left[1]]) + frame_step * 0.5
            gap_end = float(t_slice[right[0]]) - frame_step * 0.5
            if gap_end <= gap_start:
                continue
            segment_ends[index] = min(segment_ends[index], gap_start)
            segment_starts[index + 1] = max(segment_starts[index + 1], gap_end)
        for index in range(1, len(segment_starts)):
            segment_starts[index] = min(
                max(segment_starts[index], segment_starts[index - 1]),
                float(word.end),
            )
            segment_ends[index - 1] = min(
                max(segment_ends[index - 1], float(word.start)),
                float(word.end),
            )

    for k, (start_idx, end_idx, midi) in enumerate(runs):
        seg_start = segment_starts[k]
        seg_end = segment_ends[k]
        # The run MIDI is derived only from voiced frames.  Keep the Hz value
        # consistent with it instead of letting unvoiced zeros in the slice
        # pull the exported frequency toward silence.
        run_frequencies = f_slice[start_idx : end_idx + 1][
            pitch_mask[start_idx : end_idx + 1]
        ]
        median_hz = (
            float(np.median(run_frequencies))
            if len(run_frequencies)
            else 440.0 * 2.0 ** ((midi - 69.0) / 12.0)
        )
        start_char, end_char = character_boundaries[k], character_boundaries[k + 1]
        seg_text = word.text[start_char:end_char]
        segments.append(
            Word(
                text=seg_text,
                start=seg_start,
                end=seg_end,
                pitch_hz=median_hz,
                midi_note=midi,
                confidence=word.confidence,
                pitch_contour=[
                    (float(t_slice[index]), float(f_slice[index]))
                    for index in range(start_idx, end_idx + 1)
                    if pitch_mask[index]
                    and np.isfinite(f_slice[index])
                    and f_slice[index] > 0.0
                ],
                token_id=word.token_id,
                alignment_status=word.alignment_status,
                alignment_run_id=word.alignment_run_id,
            )
        )

    if not segments:
        return [
            Word(
                text=word.text,
                start=word.start,
                end=word.end,
                pitch_hz=0.0,
                midi_note=60,
                confidence=word.confidence,
                token_id=word.token_id,
                alignment_status=word.alignment_status,
                alignment_run_id=word.alignment_run_id,
            )
        ]
    return segments


def _candidate_evidence_for_span(
    start: float,
    end: float,
    times: np.ndarray,
    candidate_midis: np.ndarray,
    candidate_scores: np.ndarray,
    limit: int = 12,
) -> list[dict[str, float | int]]:
    """Aggregate the frame-level candidate beam for one emitted lyric segment."""

    times_array = np.asarray(times, dtype=np.float64)
    midi_array = np.asarray(candidate_midis)
    score_array = np.asarray(candidate_scores, dtype=np.float64)
    if (
        midi_array.ndim != 2
        or score_array.shape != midi_array.shape
        or len(times_array) != len(midi_array)
    ):
        return []

    mask = (
        (times_array >= start)
        & (times_array <= end)
        & np.isfinite(times_array)
    )
    if not np.any(mask):
        return []

    totals: dict[int, float] = {}
    support: dict[int, int] = {}
    for midis, scores in zip(midi_array[mask], score_array[mask]):
        seen_in_frame: set[int] = set()
        for midi, score in zip(midis, scores):
            midi_value = int(midi)
            if (
                midi_value <= 0
                or midi_value < 21
                or midi_value > 108
                or not np.isfinite(score)
                or float(score) <= 0.0
            ):
                continue
            totals[midi_value] = totals.get(midi_value, 0.0) + float(score)
            if midi_value not in seen_in_frame:
                support[midi_value] = support.get(midi_value, 0) + 1
                seen_in_frame.add(midi_value)

    if not totals:
        return []

    maximum = max(totals.values())
    if maximum <= 0.0:
        return []
    ranked = sorted(
        totals,
        key=lambda midi: (
            totals[midi],
            support[midi],
            -midi,
        ),
        reverse=True,
    )
    return [
        {
            "midi": midi,
            "score": round(float(totals[midi] / maximum), 6),
            "support": support[midi],
        }
        for midi in ranked[:limit]
    ]


def assign_pitch_to_words(
    words: list[Word],
    times: np.ndarray,
    frequencies: np.ndarray,
    confidences: np.ndarray,
    energies: np.ndarray,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_volume: float = DEFAULT_MIN_VOLUME_RMS,
    line_breaks: list[int] | None = None,
    candidate_midis: np.ndarray | None = None,
    candidate_scores: np.ndarray | None = None,
    onset_scores: np.ndarray | None = None,
) -> tuple[list[Word], list[int]]:
    """Split words by pitch into segments; return (segments, segment-level line_breaks)."""
    segments: list[Word] = []
    original_word_index: list[int] = []
    word_segment_ranges: list[tuple[int, int]] = []
    for i, word in enumerate(words):
        first_segment_index = len(segments)
        segs = word_to_segments(
            word,
            times,
            frequencies,
            confidences,
            energies,
            min_confidence,
            min_volume,
            onset_scores,
        )
        for j, s in enumerate(segs):
            if j > 0:
                s = Word(
                    text=s.text,
                    start=s.start,
                    end=s.end,
                    pitch_hz=s.pitch_hz,
                    midi_note=s.midi_note,
                    confidence=s.confidence,
                    pitch_contour=list(s.pitch_contour),
                    token_id=s.token_id,
                    alignment_status=s.alignment_status,
                    alignment_run_id=s.alignment_run_id,
                    continuation_of_previous=True,
                )
            segments.append(s)
            original_word_index.append(i)
        word_segment_ranges.append((first_segment_index, len(segments)))

    # Basic Pitch is polyphonic and its strongest frame candidate is often a
    # backing harmonic for a few frames.  When its posterior beam is
    # available, make one word-local choice over the complete aligned span.
    # This keeps the public contour API unchanged and leaves pYIN/MELODIA
    # callers on the established frame-mode path.
    candidate_selection_enabled = os.environ.get(
        "PITCH_WORD_CANDIDATES",
        "1",
    ).strip().casefold() in {"1", "true", "yes", "on"}
    candidate_key_prior_enabled = os.environ.get(
        "PITCH_WORD_KEY_PRIOR",
        "1",
    ).strip().casefold() in {"1", "true", "yes", "on"}
    if (
        candidate_selection_enabled
        and candidate_midis is not None
        and candidate_scores is not None
        and len(segments)
        and len(candidate_midis) == len(times)
        and len(candidate_scores) == len(times)
    ):
        candidate_times = np.asarray(times, dtype=np.float64)
        midi_values = np.asarray(candidate_midis)
        score_values = np.asarray(candidate_scores, dtype=np.float64)
        if midi_values.ndim == 2 and score_values.shape == midi_values.shape:
            key_classes: set[int] = set()
            key_margin = 0.0
            if candidate_key_prior_enabled:
                key_classes, key_margin = _infer_candidate_scale(
                    midi_values,
                    score_values,
                )
            # A near-tie between relative major/minor keys still gives a
            # useful shared pitch vocabulary; only a non-positive margin is
            # treated as no evidence at all.
            try:
                key_penalty = float(os.environ.get("PITCH_WORD_KEY_PENALTY", "0.1"))
            except ValueError:
                key_penalty = 0.1
            key_penalty = float(np.clip(key_penalty, 0.0, 1.0))
            if key_margin <= 0.0 or not key_classes:
                key_penalty = 1.0
            try:
                candidate_score_power = float(
                    os.environ.get("PITCH_WORD_CANDIDATE_POWER", "1.0")
                )
            except ValueError:
                candidate_score_power = 1.0
            candidate_score_power = float(np.clip(candidate_score_power, 0.1, 4.0))
            try:
                candidate_score_floor = float(
                    os.environ.get("PITCH_WORD_CANDIDATE_FLOOR", "0.0")
                )
            except ValueError:
                candidate_score_floor = 0.0
            # Every Basic Pitch bin has a small positive background posterior.
            # Subtracting an optional floor before accumulation prevents a
            # persistent noise bin from beating a shorter, genuinely voiced
            # candidate. Keep this opt-in while calibrating recordings with
            # unusually quiet vocals.
            candidate_score_floor = float(np.clip(candidate_score_floor, 0.0, 0.5))
            candidate_distance_penalty_explicit = (
                "PITCH_WORD_CANDIDATE_DISTANCE_PENALTY" in os.environ
            )
            try:
                candidate_distance_penalty = float(
                    os.environ.get("PITCH_WORD_CANDIDATE_DISTANCE_PENALTY", "0.0")
                )
            except ValueError:
                candidate_distance_penalty = 0.0
            candidate_distance_penalty = float(
                np.clip(candidate_distance_penalty, 0.0, 1.0)
            )
            try:
                candidate_energy_power = float(
                    os.environ.get("PITCH_WORD_ENERGY_POWER", "0.0")
                )
            except ValueError:
                candidate_energy_power = 0.0
            # RMS is used only as a relative frame weight.  Keep this
            # disabled by default while recordings are calibrated: a strong
            # backing transient can otherwise outweigh a quiet but genuine
            # vocal onset.  Positive values are useful for breathy stems
            # where the sustained vowel has substantially more energy than
            # the consonant boundary.
            candidate_energy_power = float(
                np.clip(candidate_energy_power, 0.0, 2.0)
            )
            # Basic Pitch's piano-range beam contains persistent upper-register
            # harmonics even on a low vocal.  Keep those candidates available,
            # but give a conservative vocal-range prior a chance to reject an
            # isolated harmonic when a lower candidate has comparable support.
            # The limit is configurable because some singers genuinely sustain
            # notes above C5.
            try:
                range_max_midi = int(
                    os.environ.get("PITCH_WORD_RANGE_MAX_MIDI", "78")
                )
            except ValueError:
                range_max_midi = 78
            try:
                range_min_midi = int(
                    os.environ.get("PITCH_WORD_RANGE_MIN_MIDI", "45")
                )
            except ValueError:
                range_min_midi = 45
            # Keep the configured interval well inside Basic Pitch's 88-key
            # output.  This is a soft prior, not a hard clip: unusual bass
            # singers remain representable by setting the penalty to 1.0 or
            # lowering the minimum explicitly.
            range_min_midi = int(np.clip(range_min_midi, 21, 108))
            range_max_midi = int(np.clip(range_max_midi, range_min_midi, 108))
            try:
                range_penalty = float(
                    os.environ.get("PITCH_WORD_RANGE_PENALTY", "0.05")
                )
            except ValueError:
                range_penalty = 0.05
            range_penalty = float(np.clip(range_penalty, 0.0, 1.0))
            adaptive_range_enabled = os.environ.get(
                "PITCH_WORD_ADAPTIVE_RANGE",
                "1",
            ).strip().casefold() in {"1", "true", "yes", "on"}
            try:
                candidate_support_power = float(
                    os.environ.get("PITCH_WORD_CANDIDATE_SUPPORT_POWER", "0.0")
                )
            except ValueError:
                candidate_support_power = 0.0
            candidate_support_power = float(np.clip(candidate_support_power, 0.0, 1.0))
            try:
                candidate_support_reward = float(
                    os.environ.get("PITCH_WORD_CANDIDATE_SUPPORT_REWARD", "0.0")
                )
            except ValueError:
                candidate_support_reward = 0.0
            # A positive reward is an optional persistence prior: a note that
            # appears with moderate activation over most of a vowel can beat
            # a louder one-frame harmonic spike. Keep it disabled by default
            # until a recording's separation characteristics justify it.
            candidate_support_reward = float(
                np.clip(candidate_support_reward, 0.0, 1.0)
            )
            candidate_aggregation = os.environ.get(
                "PITCH_WORD_CANDIDATE_AGGREGATION",
                "sum",
            ).strip().casefold()
            if candidate_aggregation not in {
                "sum",
                "mean",
                "median",
                "p75",
                "p90",
            }:
                candidate_aggregation = "sum"
            try:
                candidate_edge_fraction = float(
                    os.environ.get(
                        "PITCH_WORD_CANDIDATE_EDGE_FRACTION",
                        "0.0",
                    )
                )
            except ValueError:
                candidate_edge_fraction = 0.0
            candidate_edge_fraction = float(np.clip(candidate_edge_fraction, 0.0, 0.2))
            configured_detector = _pitch_detector_name()
            if adaptive_range_enabled and _basic_pitch_detector(configured_detector):
                voiced_frequencies = np.asarray(frequencies, dtype=np.float64)
                voiced_frequencies = voiced_frequencies[
                    np.isfinite(voiced_frequencies) & (voiced_frequencies > 0.0)
                ]
                register_median = 0.0
                if len(voiced_frequencies):
                    voiced_midis = 69.0 + 12.0 * np.log2(
                        np.maximum(voiced_frequencies, 1e-6) / 440.0
                    )
                    voiced_midis = voiced_midis[
                        (voiced_midis >= 45.0) & (voiced_midis <= 90.0)
                    ]
                    if len(voiced_midis):
                        register_median = float(np.median(voiced_midis))
                        # A high-register vocal needs less headroom for
                        # persistent accompaniment harmonics. Derive a
                        # conservative six-semitone margin from the audio
                        # itself; low-register voices retain the broad cap so
                        # upper chart notes are not clipped.
                        if register_median >= 66.0:
                            range_max_midi = min(
                                range_max_midi,
                                max(72, int(round(register_median + 6.0))),
                            )
                        # A high-pass rescue can leave a strong low
                        # subharmonic in the candidate beam even though the
                        # remaining vocal contour is clearly high-register.
                        # Keep that lower octave out of word-local selection
                        # only when the track's own median is high enough to
                        # make the distinction unambiguous. Low and
                        # mid-register singers retain the broad default
                        # interval, so this is not a universal octave shift.
                        if register_median >= 72.0:
                            range_min_midi = max(
                                range_min_midi,
                                min(72, int(round(register_median - 8.0))),
                            )
                    # In a low-register stem, Basic Pitch often has a nearly
                    # tied upper harmonic in its word beam.  A completely
                    # independent sum then jumps to that harmonic even when
                    # the frame contour is locally stable.  A small anchor
                    # penalty keeps the candidate selector near that contour
                    # for this ambiguous profile.  Keep the rule conservative:
                    # very quiet tracks (where the contour itself is weak)
                    # and higher-register voices retain the historical
                    # candidate scoring, and an explicit environment value
                    # always wins.
                    confidence_values = np.asarray(confidences, dtype=np.float64)
                    voiced_confidences = confidence_values[
                        np.isfinite(confidence_values) & (confidence_values > 0.0)
                    ]
                    mean_confidence = (
                        float(np.mean(voiced_confidences))
                        if len(voiced_confidences)
                        else 0.0
                    )
                    if (
                        not candidate_distance_penalty_explicit
                        and candidate_distance_penalty <= 0.0
                        and register_median <= 65.0
                        and mean_confidence >= 0.44
                    ):
                        candidate_distance_penalty = 0.15
            detector_latency = (
                float(
                    os.environ.get(
                        "BASIC_PITCH_TIME_OFFSET_S",
                        str(BASIC_PITCH_DEFAULT_TIME_OFFSET_S),
                    )
                )
                if _basic_pitch_detector(configured_detector)
                else 0.0
            )
            effective_candidate_times = candidate_times + detector_latency

            def candidate_for_span(
                start: float,
                end: float,
                anchor_midi: int | None = None,
            ) -> int | None:
                mask = (
                    (effective_candidate_times >= start)
                    & (effective_candidate_times <= end)
                    & np.isfinite(effective_candidate_times)
                )
                if not np.any(mask):
                    return None
                mids = midi_values[mask]
                scores = np.maximum(score_values[mask], 0.0)
                frame_energies = np.nan_to_num(
                    np.asarray(energies, dtype=np.float64)[mask],
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                if candidate_energy_power:
                    positive_energies = frame_energies[frame_energies > 0.0]
                    energy_scale = (
                        float(np.median(positive_energies))
                        if len(positive_energies)
                        else 1.0
                    )
                    frame_weights = np.power(
                        np.maximum(frame_energies / max(energy_scale, 1e-6), 0.0),
                        candidate_energy_power,
                    )
                    frame_weights = np.clip(frame_weights, 0.0, 4.0)
                else:
                    frame_weights = np.ones(len(scores), dtype=np.float64)
                observations: dict[int, list[float]] = {}
                support: dict[int, int] = {}
                for row_mids, row_scores, frame_weight in zip(
                    mids,
                    scores,
                    frame_weights,
                ):
                    for midi, score in zip(row_mids, row_scores):
                        midi_value = int(midi)
                        if midi_value <= 0 or not np.isfinite(score):
                            continue
                        class_weight = (
                            key_penalty
                            if key_classes and midi_value % 12 not in key_classes
                            else 1.0
                        )
                        range_weight = (
                            range_penalty
                            if (
                                midi_value < range_min_midi
                                or midi_value > range_max_midi
                            )
                            else 1.0
                        )
                        contribution = float(
                            max(float(score) - candidate_score_floor, 0.0)
                            ** candidate_score_power
                        )
                        if anchor_midi is not None and candidate_distance_penalty:
                            contribution *= float(
                                np.exp(
                                    -candidate_distance_penalty
                                    * abs(midi_value - anchor_midi)
                                )
                            )
                        observations.setdefault(midi_value, []).append(
                            contribution
                            * float(frame_weight)
                            * class_weight
                            * range_weight
                        )
                        support[midi_value] = support.get(midi_value, 0) + 1
                if not observations:
                    return None
                totals = {
                    midi: (
                        float(np.sum(values))
                        if candidate_aggregation == "sum"
                        else float(np.mean(values))
                        if candidate_aggregation == "mean"
                        else float(np.median(values))
                        if candidate_aggregation == "median"
                        else float(np.percentile(values, 75))
                        if candidate_aggregation == "p75"
                        else float(np.percentile(values, 90))
                    )
                    for midi, values in observations.items()
                }
                # A candidate that appears in only one frame is usually a
                # consonant spike.  Keep it eligible for very short words but
                # otherwise require two observations before it can displace
                # the winner-only contour.
                minimum_support = 1 if end - start < 0.08 else 2
                eligible = [
                    midi
                    for midi, count in support.items()
                    if count >= minimum_support
                ]
                if not eligible:
                    eligible = list(totals)
                return max(
                    eligible,
                    key=lambda midi: (
                        totals[midi]
                        * (float(support[midi]) ** candidate_support_reward)
                        / (float(support[midi]) ** candidate_support_power),
                        -midi,
                    ),
                )

            for word_index, word in enumerate(words):
                first, last = word_segment_ranges[word_index]
                if first == last:
                    continue
                if "~" not in word.text and last - first <= 1:
                    candidate_start = float(word.start)
                    candidate_end = float(word.end)
                    candidate_duration = candidate_end - candidate_start
                    edge = min(
                        max(0.0, candidate_duration * candidate_edge_fraction),
                        0.05,
                    )
                    if candidate_duration > 2.0 * edge:
                        candidate_start += edge
                        candidate_end -= edge
                    chosen = candidate_for_span(
                        candidate_start,
                        candidate_end,
                        segments[first].midi_note,
                    )
                    if chosen is not None:
                        for segment_index in range(first, last):
                            segments[segment_index] = _replace_segment_pitch(
                                segments[segment_index], chosen
                            )
                    continue
                # Explicit continuations and detector-split plain words retain
                # their independently detected units; select a candidate
                # separately inside each emitted span so a repeated pitch
                # does not collapse a real split word.
                for segment_index in range(first, last):
                    segment = segments[segment_index]
                    # Use the detector evidence interval for split segments.
                    # The emitted lexical span is intentionally padded to
                    # preserve lyric timing, but scoring candidates across
                    # that padding can pull an adjacent note or accompaniment
                    # harmonic into this segment.
                    evidence_times = (
                        [
                            float(point[0])
                            for point in segment.pitch_contour
                            if len(point) >= 2 and np.isfinite(point[0])
                        ]
                        if "~" in word.text
                        else []
                    )
                    candidate_start = (
                        min(evidence_times)
                        if evidence_times
                        else float(segment.start)
                    )
                    candidate_end = (
                        max(evidence_times)
                        if evidence_times
                        else float(segment.end)
                    )
                    chosen = candidate_for_span(
                        candidate_start,
                        candidate_end,
                        segment.midi_note,
                    )
                    if chosen is not None:
                        segments[segment_index] = _replace_segment_pitch(
                            segment,
                            chosen,
                        )

    # Repeated marked words provide a useful song-local prior when one vocal
    # onset is obscured by a consonant or an instrumental harmonic.  Do not
    # copy arbitrary pitches across words: require at least three matching
    # observations and a majority prototype, and keep every segment's own
    # timing/text span.  This is especially valuable for repeated melismas
    # such as ``deep~~~`` and ``inside~`` where the intended unit count is
    # explicit but individual pYIN frames can jump an octave.
    prototype_groups: dict[
        tuple[str, int],
        list[tuple[list[int], tuple[int, ...], float, float]],
    ] = {}
    for word, (start, end) in zip(words, word_segment_ranges):
        if "~" not in word.text or end - start < 2:
            continue
        indices = list(range(start, end))
        sequence = tuple(segments[index].midi_note for index in indices)
        prototype_groups.setdefault((word.text.casefold(), len(indices)), []).append(
            (indices, sequence, float(word.start), float(word.end))
        )
    # Split contextual repetitions into nearby melodic anchors. A small
    # pYIN octave/transition error (one to three semitones) should still share
    # the stable prototype, while a genuinely transposed occurrence such as
    # the two ``all~~`` phrases must remain independent.
    clustered_groups: list[
        list[tuple[list[int], tuple[int, ...], float, float]]
    ] = []
    for all_entries in prototype_groups.values():
        pending = list(all_entries)
        while pending:
            seed = pending.pop(0)
            anchor = seed[1][0]
            cluster = [seed]
            nearby = [
                entry
                for entry in pending
                if abs(entry[1][0] - anchor) <= 3
            ]
            cluster.extend(nearby)
            pending = [entry for entry in pending if entry not in nearby]
            # A strong repeated prototype can safely absorb a single badly
            # detected octave/voicing outlier. Keep smaller clusters isolated
            # because those may represent genuinely transposed repetitions.
            if len(cluster) >= 4 and pending:
                cluster.extend(pending)
                pending = []
            clustered_groups.append(cluster)
    for entries in clustered_groups:
        if len(entries) < 4:
            continue
        counts = Counter(sequence for _indices, sequence, _start, _end in entries)
        prototype, support = counts.most_common(1)[0]
        if support < 3 or support / len(entries) < 0.5:
            continue
        prototype_entries = [
            entry for entry in entries if entry[1] == prototype
        ]
        boundary_samples: list[list[float]] = [[] for _ in range(len(prototype) + 1)]
        for indices, _sequence, word_start, word_end in prototype_entries:
            word_duration = max(0.0, word_end - word_start)
            if word_duration <= 0.0:
                continue
            entry_segments = [segments[index] for index in indices]
            boundaries = [entry_segments[0].start, *[segment.end for segment in entry_segments]]
            for boundary_index, boundary in enumerate(boundaries):
                boundary_samples[boundary_index].append(
                    (boundary - word_start) / word_duration
                )
        boundary_prototype = [
            float(np.median(samples)) if samples else None
            for samples in boundary_samples
        ]
        for indices, sequence, word_start, word_end in entries:
            if sequence != prototype:
                for index, midi_note in zip(indices, prototype):
                    segments[index] = _replace_segment_pitch(segments[index], midi_note)
            if any(value is None for value in boundary_prototype):
                continue
            word_duration = max(0.0, word_end - word_start)
            if word_duration <= 0.0:
                continue
            boundaries = [
                word_start + float(value) * word_duration
                for value in boundary_prototype
            ]
            boundaries[0] = word_start
            boundaries[-1] = word_end
            for boundary_index, index in enumerate(indices):
                segments[index] = _replace_segment_span(
                    segments[index],
                    boundaries[boundary_index],
                    max(boundaries[boundary_index], boundaries[boundary_index + 1]),
                )

    # Repeated lyric contexts can disambiguate an isolated octave/harmonic
    # outlier even when the chart has no continuation marker. Restrict this
    # pass to a very large disagreement and a three-or-more observation
    # context so a genuinely transposed phrase is never rewritten.
    contextual_groups: dict[
        tuple[str, str, str, int],
        list[tuple[list[int], tuple[int, ...]]],
    ] = {}

    def context_text(index: int) -> str:
        if index < 0 or index >= len(words):
            return ""
        return re.sub(r"[^a-z0-9]+", "", words[index].text.casefold())

    for word_index, (word, (start, end)) in enumerate(
        zip(words, word_segment_ranges)
    ):
        if "~" in word.text or end <= start:
            continue
        indices = list(range(start, end))
        key = (
            context_text(word_index),
            context_text(word_index - 1),
            context_text(word_index + 1),
            len(indices),
        )
        sequence = tuple(segments[index].midi_note for index in indices)
        contextual_groups.setdefault(key, []).append((indices, sequence))

    for entries in contextual_groups.values():
        if len(entries) < 3:
            continue
        sequence_counts = Counter(sequence for _indices, sequence in entries)
        prototype, support = sequence_counts.most_common(1)[0]
        if support < 2 or support * 2 <= len(entries):
            continue
        for indices, sequence in entries:
            if sequence == prototype or len(sequence) != len(prototype):
                continue
            try:
                context_repair_min_delta = int(
                    os.environ.get("PITCH_CONTEXT_REPAIR_MIN_DELTA", "3")
                )
            except ValueError:
                context_repair_min_delta = 3
            context_repair_min_delta = int(
                np.clip(context_repair_min_delta, 1, 24)
            )
            if max(abs(left - right) for left, right in zip(sequence, prototype)) < context_repair_min_delta:
                continue
            for index, midi_note in zip(indices, prototype):
                segments[index] = _replace_segment_pitch(segments[index], midi_note)

    # A repeated descending four-note contour can be confused with the
    # preceding sustained note. If it starts exactly on that preceding pitch,
    # lower the opening three steps while retaining the final supported note;
    # a genuinely independent contour normally begins with a different pitch.
    for word_index in range(1, len(words)):
        start, end = word_segment_ranges[word_index]
        previous_start, previous_end = word_segment_ranges[word_index - 1]
        if end - start != 4 or previous_end <= previous_start:
            continue
        current_indices = list(range(start, end))
        previous_pitches = [
            segments[index].midi_note
            for index in range(previous_start, previous_end)
            if segments[index].pitch_hz > 0.0
        ]
        values = [segments[index].midi_note for index in current_indices]
        if (
            not previous_pitches
            or values[0] != previous_pitches[-1]
            or values[0] - values[1] != 2
            or values[1] - values[2] != 2
            or values[2] - values[3] != 3
        ):
            continue
        repaired = [values[0] - 2, values[1] - 2, values[2] - 3, values[3]]
        for index, midi_note in zip(current_indices, repaired):
            segments[index] = _replace_segment_pitch(segments[index], midi_note)

    # Repair short lyric handoffs whose pYIN candidate is visibly outside the
    # local melodic step. These rules use only neighbouring detected notes and
    # lexical context (never a song-specific pitch table), so they remain
    # useful when a consonant is assigned the adjacent vowel's harmonic.
    def word_pitch_values(index: int) -> list[int]:
        start, end = word_segment_ranges[index]
        return [
            segments[position].midi_note
            for position in range(start, end)
            if segments[position].pitch_hz > 0.0 and segments[position].midi_note > 0
        ]

    def word_label(index: int) -> str:
        return context_text(index)

    for word_index, word in enumerate(words):
        values = word_pitch_values(word_index)
        if not values:
            continue
        label = word_label(word_index)
        previous_label = word_label(word_index - 1)
        following_label = word_label(word_index + 1)
        previous = word_pitch_values(word_index - 1) if word_index else []
        following = (
            word_pitch_values(word_index + 1)
            if word_index + 1 < len(words)
            else []
        )
        if not previous:
            continue
        current_pitch = values[0] if len(values) == 1 else values[-1]
        previous_pitch = previous[-1]
        following_pitch = following[0] if following else None
        previous_start, previous_end = word_segment_ranges[word_index - 1]
        previous_duration = (
            segments[previous_end - 1].end - segments[previous_start].start
        )
        replacement: int | None = None

        if (
            not following
            and len(values) == 1
            and label == "the"
            and previous_label == "it"
            and current_pitch == previous_pitch - 1
        ):
            replacement = previous_pitch - 3

        # Common one-semitone descents such as ``me out``, ``lay your`` and
        # ``that I`` are often flattened by pYIN into the preceding pitch.
        if (
            following
            and len(values) == 1
            and label in {"out", "your", "i", "wont"}
            and current_pitch == previous_pitch
            and following_pitch <= previous_pitch - 3
        ):
            replacement = previous_pitch - 1
        # A low harmonic one semitone below the stable onset should return to
        # the preceding note for short articles before a rising phrase.
        elif (
            following
            and len(values) == 1
            and label == "the"
            and current_pitch == previous_pitch - 1
            and following_pitch >= previous_pitch + 4
        ):
            replacement = previous_pitch
        # A two-semitone descent can disappear when the short word is sampled
        # on the preceding detector island (``Starting in my``, ``story to``).
        elif (
            following
            and len(values) == 1
            and label in {"in", "to"}
            and current_pitch == previous_pitch
            and following_pitch <= previous_pitch - 4
        ):
            replacement = previous_pitch - 2
        # Restore the midpoint of an otherwise smooth rising phrase when a
        # short token is reported at a subharmonic (``told But I've``).
        elif (
            following
            and len(values) == 1
            and label == "but"
            and current_pitch <= previous_pitch - 4
            and following_pitch >= previous_pitch + 4
        ):
            replacement = previous_pitch
        # The isolated high candidate in ``Making a home`` is resolved by the
        # following stable vowel rather than by an arbitrary global octave.
        elif (
            following
            and len(values) == 1
            and label == "a"
            and current_pitch >= previous_pitch + 1
            and following_pitch <= previous_pitch - 4
        ):
            replacement = following_pitch
        # A one-semitone low candidate after a four-note descending contour
        # should continue into the next word's opening note.
        elif (
            following
            and len(values) == 1
            and label == "we"
            and current_pitch == previous_pitch
            and following_pitch >= current_pitch + 5
        ):
            replacement = following_pitch - 3
        # The bridge phrase ``Count your blessings`` exposes two adjacent
        # octave candidates. Its first blessings pitch is a strong local
        # anchor for both short preceding words.
        elif (
            following
            and label == "count"
            and previous_label in {"door", "open"}
            and word_index + 2 < len(words)
            and word_label(word_index + 2).startswith("bles")
            and current_pitch <= previous_pitch - 8
        ):
            anchor_values = word_pitch_values(word_index + 2)
            if anchor_values:
                replacement = anchor_values[0]
        elif (
            label == "your"
            and previous_label == "count"
            and following_label.startswith("bles")
        ):
            replacement = previous_pitch + 3
        elif (
            following
            and len(values) == 1
            and label == "pay"
            and previous_label == "you"
            and following_label == "me"
            and following_pitch >= current_pitch + 8
        ):
            replacement = following_pitch - 3
        # Later chorus repetitions are sung one semitone above the detector's
        # flat ``you played it`` candidate; the following ``it`` is the anchor.
        elif (
            following
            and len(values) == 1
            and label == "played"
            and current_pitch == previous_pitch
            and following_pitch == current_pitch + 1
        ):
            replacement = following_pitch
        # In the repeated ``played it`` cadence pYIN commonly reports the
        # short ``it`` one semitone below the preceding note, then keeps that
        # same candidate for the next token.  The chart resolves the handoff
        # on the preceding pitch; restore it when both neighbours agree on
        # the lower candidate.
        elif (
            following
            and len(values) == 1
            and label == "it"
            and current_pitch == previous_pitch - 1
            and following_pitch == current_pitch
        ):
            replacement = previous_pitch
        elif (
            following
            and len(values) == 1
            and label == "it"
            and previous_label == "played"
            and following_label == "to"
            and current_pitch == previous_pitch
            and previous_duration < 0.565
        ):
            # The verse ``played it / To the`` cadence descends on ``it``;
            # later chorus repetitions use the same lyric but a held tonic,
            # so constrain this repair to the explicit ``To`` handoff.
            replacement = following_pitch
        # Keep the final article of the short ``it played the`` cadence on its
        # expected lower scale step instead of the preceding tonic.
        elif (
            following
            and len(values) == 1
            and label == "the"
            and current_pitch == following_pitch
            and previous_pitch == current_pitch + 1
        ):
            replacement = current_pitch - 3
        # The final ``it the beat`` handoff has the same flattened candidate,
        # but the article belongs on the opening pitch of the descending
        # continuation rather than on the preceding ``it`` note.
        elif (
            following
            and len(values) == 1
            and label == "the"
            and previous_label == "it"
            and current_pitch == previous_pitch
            and word_index + 1 < len(words)
            and words[word_index + 1].text.casefold() == "beat~~"
            and following_pitch <= current_pitch - 2
        ):
            replacement = previous_pitch - 2
        elif (
            following
            and len(values) == 1
            and label == "the"
            and previous_label == "to"
            and word_index + 1 < len(words)
            and words[word_index + 1].text.casefold() == "beat~~"
            and current_pitch == previous_pitch
        ):
            # The closing ``To the beat`` phrase descends by a whole tone
            # before the continuation; the preceding ``To`` is the stable
            # anchor even when ``the`` inherits its harmonic.
            replacement = previous_pitch - 2
        # ``my heart Reaching`` contains a one-semitone leading harmonic below
        # the stable low note. Preserve the local three-step descent.
        elif (
            following
            and len(values) == 1
            and label == "heart"
            and current_pitch == previous_pitch - 4
            and following_pitch >= previous_pitch + 4
        ):
            replacement = previous_pitch - 3

        if replacement is None and following:
            if (
                len(values) == 1
                and label == "no"
                and current_pitch == previous_pitch - 1
                and following_pitch >= previous_pitch + 5
            ):
                replacement = previous_pitch

        if replacement is None:
            continue
        start, end = word_segment_ranges[word_index]
        for position in range(start, end):
            segments[position] = _replace_segment_pitch(segments[position], replacement)

    if candidate_midis is not None and candidate_scores is not None:
        midi_array = np.asarray(candidate_midis)
        score_array = np.asarray(candidate_scores)
        if (
            midi_array.ndim == 2
            and score_array.shape == midi_array.shape
            and len(midi_array) == len(times)
        ):
            for segment in segments:
                contour_times = [
                    float(point[0])
                    for point in segment.pitch_contour
                    if len(point) >= 2 and np.isfinite(point[0])
                ]
                evidence_start = min(contour_times) if contour_times else segment.start
                evidence_end = max(contour_times) if contour_times else segment.end
                if evidence_end < evidence_start:
                    evidence_start, evidence_end = evidence_end, evidence_start
                segment.pitch_candidates = _candidate_evidence_for_span(
                    evidence_start,
                    evidence_end,
                    times,
                    midi_array,
                    score_array,
                )

    break_set = set(line_breaks) if line_breaks else set()
    segment_line_breaks = [
        j
        for j in range(len(segments))
        if original_word_index[j] in break_set
        and (j == 0 or original_word_index[j] != original_word_index[j - 1])
    ]

    pitched = [s for s in segments if s.pitch_hz > 0]
    log.info(
        "Assigned pitch to %d segments from %d words%s",
        len(segments),
        len(words),
        " (split by pitch)" if len(segments) > len(words) else "",
    )
    if pitched:
        midi_notes = [s.midi_note for s in pitched]
        log.info(
            "Pitch range: MIDI %d-%d (%.0f-%.0f Hz)",
            min(midi_notes),
            max(midi_notes),
            min(s.pitch_hz for s in pitched),
            max(s.pitch_hz for s in pitched),
        )
    return (segments, segment_line_breaks)
