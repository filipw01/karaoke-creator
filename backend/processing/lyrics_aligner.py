"""Lyrics alignment using schufo/lyrics-aligner (phoneme-level DNN, English).

Requires: clone https://github.com/schufo/lyrics-aligner and set LYRICS_ALIGNER_ROOT.
Run from that repo's conda env or set LYRICS_ALIGNER_PYTHON to the env's python.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile

import librosa

from models import Word

from processing.lyrics import preprocess_lyrics_for_alignment


log = logging.getLogger("lyrics")


def _get_lyrics_aligner_root() -> str | None:
    root = os.environ.get("LYRICS_ALIGNER_ROOT")
    if not root or not os.path.isdir(root):
        return None
    if not os.path.isfile(os.path.join(root, "align.py")):
        return None
    if not os.path.isfile(os.path.join(root, "model_parameters.pth")):
        return None
    if not os.path.isfile(os.path.join(root, "files", "phoneme2idx.pickle")):
        return None
    return root


def _get_lyrics_aligner_python() -> str:
    return os.environ.get("LYRICS_ALIGNER_PYTHON", "python")


def _words_to_arpabet(words: list[str]) -> tuple[list[str], list[int]]:
    """Convert words to ARPAbet phoneme sequence with '>' as space. Returns (phonemes, word_end_indices)."""
    try:
        import g2p_en
    except ImportError:
        return [], []
    g2p = g2p_en.G2p()
    phonemes: list[str] = [">"]
    word_end_indices: list[int] = []
    for w in words:
        try:
            ph = g2p(w)
        except Exception:
            ph = []
        if isinstance(ph, str):
            ph = ph.split()
        for p in (ph or []):
            p_clean = p.strip().upper()
            if p_clean:
                phonemes.append(p_clean)
        phonemes.append(">")
        word_end_indices.append(len(phonemes) - 1)
    return phonemes, word_end_indices


def extract_lyrics_lyrics_aligner(
    audio_path: str,
    lyrics_text: str,
    language: str | None = None,
) -> tuple[list[Word], list[int]] | tuple[None, str]:
    """Align lyrics using https://github.com/schufo/lyrics-aligner (English, phoneme-level).

    Returns (words, line_breaks) on success, or (None, reason) if unavailable/fails.
    """
    root = _get_lyrics_aligner_root()
    if not root:
        return None, "LYRICS_ALIGNER_ROOT not set or missing align.py / model_parameters.pth / files/phoneme2idx.pickle"

    lang = (language or "").strip().lower()[:2]
    if lang != "en":
        return None, "lyrics-aligner supports English only (lang=en)"

    lines, all_words, line_breaks_pre = preprocess_lyrics_for_alignment(lyrics_text)
    if not all_words:
        return None, "No lyrics words"

    phonemes, word_end_indices = _words_to_arpabet(all_words)
    if not phonemes or phonemes == [">"]:
        return None, "G2P produced no phonemes (install g2p_en)"

    basename = "track"
    with tempfile.TemporaryDirectory(prefix="la_audio_") as audio_dir:
        with tempfile.TemporaryDirectory(prefix="la_lyrics_") as lyrics_dir:
            ext = os.path.splitext(audio_path)[1] or ".wav"
            audio_file = os.path.join(audio_dir, basename + ext)
            shutil.copy2(audio_path, audio_file)
            lyrics_file = os.path.join(lyrics_dir, basename + ".txt")
            with open(lyrics_file, "w", encoding="utf-8") as f:
                for p in phonemes:
                    f.write(p + "\n")
            python_exe = _get_lyrics_aligner_python()
            cmd = [
                python_exe, "align.py",
                audio_dir, lyrics_dir,
                "--lyrics-format", "p",
                "--onsets", "p",
                "--dataset-name", "karaoke",
                "--vad-threshold", "0",
            ]
            out_dir = os.path.join(root, "outputs", "karaoke", "phoneme_onsets")
            try:
                subprocess.run(
                    cmd,
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=600,
                    check=True,
                )
            except subprocess.TimeoutExpired:
                return None, "lyrics-aligner timed out"
            except subprocess.CalledProcessError as e:
                log.warning("lyrics-aligner failed: %s %s", e.stderr, e.stdout)
                return None, f"lyrics-aligner failed: {e.stderr or str(e)}"
            except FileNotFoundError:
                return None, "LYRICS_ALIGNER_PYTHON not found"

            onset_file = os.path.join(out_dir, basename + ".txt")
            if not os.path.isfile(onset_file):
                return None, "lyrics-aligner produced no phoneme onsets file"

            phoneme_onsets: list[float] = []
            with open(onset_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        try:
                            phoneme_onsets.append(float(parts[1]))
                        except ValueError:
                            pass
            if len(phoneme_onsets) != len(phonemes):
                return None, f"phoneme count mismatch (got {len(phoneme_onsets)}, expected {len(phonemes)})"

            duration = librosa.get_duration(path=audio_path)
            word_starts: list[int] = [1]
            for i in range(1, len(phonemes) - 1):
                if phonemes[i] == ">":
                    word_starts.append(i + 1)
            words_out: list[Word] = []
            for i, w in enumerate(all_words):
                start_idx = word_starts[i]
                start_sec = phoneme_onsets[start_idx] if start_idx < len(phoneme_onsets) else 0.0
                end_idx = word_end_indices[i]
                if end_idx < len(phoneme_onsets):
                    end_sec = phoneme_onsets[end_idx]
                elif i == len(all_words) - 1:
                    end_sec = duration
                else:
                    end_sec = start_sec + 0.3
                if end_sec <= start_sec:
                    end_sec = start_sec + 0.2
                words_out.append(Word(text=w, start=start_sec, end=end_sec, confidence=1.0))

            line_breaks = line_breaks_pre
            log.info("lyrics-aligner produced %d words, %d line breaks", len(words_out), len(line_breaks))
            return (words_out, line_breaks)
