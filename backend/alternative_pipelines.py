"""Optional SOFA alignment adapter.

Each adapter is invoked from a short-lived subprocess by the benchmark runner.
This keeps the large checkpoints and their incompatible dependency sets out of
the normal API process and limits resident memory to one aligner at a time.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable, Sequence

from models import Word


ROOT = Path(__file__).resolve().parent
OPTIONAL_ROOT = ROOT / "optional_aligners"
SOFA_CODE = Path(os.environ.get("SOFA_CODE_DIR", OPTIONAL_ROOT / "sofa-code"))
SOFA_CKPT = Path(os.environ.get(
    "SOFA_CHECKPOINT", OPTIONAL_ROOT / "sofa" / "ckpt" / "combined.ckpt"
))
SOFA_DICTIONARY = Path(os.environ.get(
    "SOFA_DICTIONARY", OPTIONAL_ROOT / "sofa" / "english-dictionary.txt"
))


@dataclass(frozen=True)
class TimedToken:
    text: str
    start: float
    end: float


def _clean_word(text: str) -> str:
    text = text.replace("~", "")
    text = re.sub(r"[^\w'’-]+", "", text, flags=re.UNICODE)
    return text or "sil"


def _safe_span(start: float, end: float, duration: float) -> tuple[float, float]:
    start = max(0.0, min(float(start), duration))
    end = max(start + 0.02, min(float(end), duration))
    return start, end


def _crop_audio(source: Path, destination: Path, start: float, end: float, sample_rate: int) -> None:
    """Extract a bounded mono window without loading a whole song into RAM."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{max(0.0, start):.6f}", "-i", str(source),
            "-t", f"{max(0.05, end - start):.6f}", "-ac", "1",
            "-ar", str(sample_rate), str(destination),
        ],
        check=True,
    )


def _window_groups(words: Sequence[TimedToken], max_words: int = 12) -> list[tuple[int, int]]:
    return [(start, min(len(words), start + max_words)) for start in range(0, len(words), max_words)]


def _ensure_sofa_dictionary(words: Iterable[str]) -> Path:
    """Create the English dictionary consumed by SOFA's DictionaryG2P."""

    existing = SOFA_DICTIONARY.read_text(encoding="utf-8").splitlines() if SOFA_DICTIONARY.is_file() else []
    known = {line.split("\t", 1)[0] for line in existing if "\t" in line}
    from g2p_en import G2p

    mapping = {"ax": "ah", "el": "l", "em": "m", "en": "n", "eng": "ng", "dx": "d", "ix": "ih", "ux": "uw"}
    allowed = {
        "aa", "ae", "ah", "ao", "aw", "ay", "b", "ch", "d", "dh", "eh", "er", "ey", "f", "g",
        "hh", "ih", "iy", "jh", "k", "l", "m", "n", "ng", "ow", "oy", "p", "r", "s", "sh", "t",
        "th", "uh", "uw", "v", "w", "y", "z", "zh",
    }
    g2p = G2p()
    for raw_word in words:
        word = _clean_word(raw_word).lower()
        if word in known or word == "sil":
            continue
        phones = []
        for phone in g2p(word):
            phone = phone.strip()
            if not phone or phone in {"<unk>", " "}:
                continue
            base = mapping.get(re.sub(r"\d+$", "", phone).lower(), re.sub(r"\d+$", "", phone).lower())
            if base in allowed:
                phones.append(base)
        if phones:
            existing.append(f"{word}\t{' '.join(phones)}")
            known.add(word)
    SOFA_DICTIONARY.parent.mkdir(parents=True, exist_ok=True)
    SOFA_DICTIONARY.write_text("\n".join(existing) + "\n", encoding="utf-8")
    return SOFA_DICTIONARY


def _run_sofa_window(
    audio_path: Path,
    transcript: str,
    *,
    checkpoint: Path,
    dictionary: Path,
    output_dir: Path,
) -> list[tuple[str, float, float]]:
    request = output_dir / "request.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    request.write_text(json.dumps({
        "audio": str(audio_path), "transcript": transcript,
        "checkpoint": str(checkpoint), "dictionary": str(dictionary), "repo": str(SOFA_CODE),
    }), encoding="utf-8")
    script = r'''
import json, os, sys
from pathlib import Path
request = json.loads(Path(sys.argv[1]).read_text())
import torch
torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
library = torch.library.Library("torchvision", "DEF")
for op in ("nms", "qnms"):
    try:
        library.define(f"{op}(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
    except Exception:
        pass
import pandas as pd
repo = Path(request["repo"])
sys.path.insert(0, str(repo))
from modules.g2p import DictionaryG2P
from modules.g2p.base_g2p import DataFrameDataset
from modules.task.forced_alignment import LitForcedAlignmentTask
from lightning.pytorch import Trainer
audio = Path(request["audio"])
g2p = DictionaryG2P(dictionary=request["dictionary"])
ph_seq, word_seq, mapping = g2p(request["transcript"])
dataset = DataFrameDataset(pd.DataFrame([(audio, ph_seq, word_seq, mapping)], columns=["wav_path", "ph_seq", "word_seq", "ph_idx_to_word_idx"]))
model = LitForcedAlignmentTask.load_from_checkpoint(request["checkpoint"], map_location="cpu")
model.set_inference_mode("force")
trainer = Trainer(logger=False, enable_progress_bar=False, enable_model_summary=False, accelerator="cpu", devices=1)
prediction = trainer.predict(model, dataloaders=dataset, return_predictions=True)[0]
_path, _duration, _confidence, _ph, _ph_intervals, words, intervals = prediction
rows = [[str(word), float(span[0]), float(span[1])] for word, span in zip(words, intervals) if str(word) not in {"SP", "AP", "<SP>", "<AP>"}]
print(json.dumps(rows))
'''
    environment = os.environ.copy()
    environment.update({"MPLCONFIGDIR": str(output_dir / "mpl"), "PYTHONPATH": str(SOFA_CODE)})
    python = ROOT / "venv" / "bin" / "python"
    completed = subprocess.run(
        [str(python), "-u", "-c", script, str(request)],
        env=environment, cwd=str(SOFA_CODE), check=True,
        capture_output=True, text=True, timeout=1800,
    )
    for line in reversed([line.strip() for line in completed.stdout.splitlines() if line.strip()]):
        try:
            rows = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rows, list):
            return [(str(row[0]), float(row[1]), float(row[2])) for row in rows if isinstance(row, list) and len(row) == 3]
    raise RuntimeError(f"SOFA returned no word intervals: {completed.stdout[-1000:]}")


def align_sofa(words: Sequence[TimedToken], vocal_audio: Path, output_dir: Path) -> list[Word]:
    """Align source words with the singing-trained SOFA checkpoint."""

    if not SOFA_CODE.is_dir() or not SOFA_CKPT.is_file():
        raise RuntimeError("SOFA assets are missing; configure SOFA_CODE_DIR/SOFA_CHECKPOINT")
    dictionary = _ensure_sofa_dictionary(word.text for word in words)
    result: list[Word] = []
    previous_end = 0.0
    for window_index, (start_index, end_index) in enumerate(_window_groups(words, max_words=36)):
        window_words = words[start_index:end_index]
        window_start = max(0.0, window_words[0].start - 0.7)
        window_end = window_words[-1].end + 0.7
        audio_window = output_dir / f"sofa-{window_index:04d}.wav"
        _crop_audio(vocal_audio, audio_window, window_start, window_end, 44100)
        rows = _run_sofa_window(
            audio_window,
            " ".join(_clean_word(word.text).lower() for word in window_words),
            checkpoint=SOFA_CKPT, dictionary=dictionary,
            output_dir=output_dir / f"sofa-run-{window_index:04d}",
        )
        cursor = 0
        for word in window_words:
            target = _clean_word(word.text).lower()
            match = next((index for index in range(cursor, len(rows)) if rows[index][0].lower() == target), None)
            if match is None:
                continue
            _label, local_start, local_end = rows[match]
            start, end = _safe_span(local_start + window_start, local_end + window_start, window_end + 1.0)
            # SOFA can emit a near-zero interval for a repeated short word or
            # let a word consume the following repeated occurrence.  The
            # source clock is only a window hint, but it is still a useful
            # safety envelope: keep a small phonetic interval inside it and
            # enforce a monotonic handoff before pitch assignment samples the
            # vocal contour.
            source_start = max(0.0, float(word.start))
            source_end = max(source_start + 0.02, float(word.end))
            source_duration = source_end - source_start
            minimum_duration = min(0.20, max(0.04, source_duration * 0.35))
            if start < source_start - 0.45:
                start = source_start
            if end > source_end + 0.50:
                end = source_end
            if end - start < minimum_duration:
                start = max(start, source_start)
                end = min(source_end, start + minimum_duration)
                if end - start < minimum_duration:
                    start = max(source_start, source_end - minimum_duration)
                    end = source_end
            if start < previous_end:
                start = previous_end
            if end <= start:
                start = max(previous_end, source_start)
                end = max(start + 0.02, source_end)
            result.append(Word(text=word.text, start=start, end=end, confidence=0.5))
            previous_end = end
            cursor = match + 1
    return result


def align_pipeline(name: str, words: Sequence[TimedToken], vocal_audio: Path, output_dir: Path) -> list[Word]:
    normalized = name.strip().casefold()
    if normalized in {"sofa", "sofa-combined"}:
        return align_sofa(words, vocal_audio, output_dir)
    raise ValueError(f"Unknown alternative pipeline: {name}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", choices=("sofa",), required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--word-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = json.loads(args.word_json.read_text(encoding="utf-8"))
    words = [TimedToken(str(row["text"]), float(row["start"]), float(row["end"])) for row in payload]
    aligned = align_pipeline(args.pipeline, words, args.audio, args.output_dir)
    if args.json:
        print(json.dumps([word.__dict__ for word in aligned], ensure_ascii=False, indent=2))
    else:
        print(f"{args.pipeline}: {len(aligned)}/{len(words)} words")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
