#!/usr/bin/env python3
"""
Download lyrics-aligner and AutoLyrixAlign and write backend/.env.optional with their paths.
Run from repo root or backend: python backend/scripts/setup_optional_aligners.py
Requires: git, backend venv with g2p_en; pip install gdown for AutoLyrixAlign.
"""
from __future__ import annotations

import os
import pickle
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# Resolve backend dir (script may be run as backend/scripts/setup_optional_aligners.py or from backend)
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
OPTIONAL_DIR = BACKEND_DIR / "optional_aligners"
LYRICS_ALIGNER_DIR = OPTIONAL_DIR / "lyrics-aligner"
AUTOLYRIXALIGN_DIR = OPTIONAL_DIR / "AutoLyrixAlign"
# AutoLyrixAlign: pre-trained model + scripts (Google Drive, from chitralekha18/AutoLyrixAlign README)
AUTOLYRIXALIGN_GDRIVE_ID = "1zJdhPr9SzTVRXIaiZ7eF_QKN0PT2sbIW"


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd or BACKEND_DIR, check=check, text=True)


def main() -> int:
    OPTIONAL_DIR.mkdir(parents=True, exist_ok=True)

    # --- lyrics-aligner (schufo/lyrics-aligner) ---
    if not (LYRICS_ALIGNER_DIR / "align.py").exists():
        print("Cloning schufo/lyrics-aligner...")
        if LYRICS_ALIGNER_DIR.exists():
            shutil.rmtree(LYRICS_ALIGNER_DIR)
        run(["git", "clone", "--depth", "1", "https://github.com/schufo/lyrics-aligner.git", str(LYRICS_ALIGNER_DIR)])
    else:
        print("lyrics-aligner already present.")

    # align.py expects files/karaoke_word2phonemes.pickle when --dataset-name karaoke
    files_dir = LYRICS_ALIGNER_DIR / "files"
    files_dir.mkdir(exist_ok=True)
    karaoke_pickle = files_dir / "karaoke_word2phonemes.pickle"
    if not karaoke_pickle.exists():
        print("Creating karaoke_word2phonemes.pickle for lyrics-aligner...")
        try:
            import g2p_en  # noqa: F401
        except ImportError:
            print("  Install g2p_en: pip install g2p_en", file=sys.stderr)
            return 1
        g2p = g2p_en.G2p()
        # Build a small word -> phonemes dict so align.py can load it (we use --lyrics-format p, but it loads the pickle)
        common = (
            "the a an and or but in on at to for of with by from as is was are were been be have has had do does did "
            "will would could should may might must shall can need dare ought used say go get make know think take see "
            "come want look give use find tell ask work seem feel try leave call"
        ).split()
        word2phonemes: dict[str, str] = {}
        for w in common:
            try:
                ph = g2p(w)
                word2phonemes[w] = " ".join(ph) if isinstance(ph, list) else str(ph)
            except Exception:
                word2phonemes[w] = "AH0"  # fallback
        with open(karaoke_pickle, "wb") as f:
            pickle.dump(word2phonemes, f)
        print("  Created karaoke_word2phonemes.pickle with", len(word2phonemes), "words.")
    else:
        print("karaoke_word2phonemes.pickle already present.")

    # --- AutoLyrixAlign (NUS, from Google Drive) ---
    if not (AUTOLYRIXALIGN_DIR / "align.py").exists():
        zip_path = OPTIONAL_DIR / "AutoLyrixAlign.zip"
        if not zip_path.exists():
            print("Downloading AutoLyrixAlign from Google Drive...")
            try:
                import gdown
            except ImportError:
                print("  Skipping (install gdown and re-run to fetch: pip install gdown)")
                print("  Or download manually: https://drive.google.com/file/d/1zJdhPr9SzTVRXIaiZ7eF_QKN0PT2sbIW/view")
            else:
                # use_cookies=False avoids writing to ~/.cache/gdown (sandbox-friendly)
                gdown.download(
                    id=AUTOLYRIXALIGN_GDRIVE_ID,
                    output=str(zip_path),
                    quiet=False,
                    fuzzy=True,
                    use_cookies=False,
                )
        if zip_path.exists():
            print("Extracting AutoLyrixAlign...")
            if AUTOLYRIXALIGN_DIR.exists():
                shutil.rmtree(AUTOLYRIXALIGN_DIR)
            AUTOLYRIXALIGN_DIR.mkdir(parents=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                # Some zips have a single top-level folder; extract and flatten if align.py is inside
                names = zf.namelist()
                top_dirs = {n.split("/")[0] for n in names if "/" in n}
                if len(top_dirs) == 1:
                    prefix = top_dirs.pop() + "/"
                    for n in names:
                        if n.endswith("/"):
                            continue
                        target = AUTOLYRIXALIGN_DIR / n[len(prefix) :]
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(zf.read(n))
                else:
                    zf.extractall(AUTOLYRIXALIGN_DIR)
            if (AUTOLYRIXALIGN_DIR / "align.py").exists():
                print("  AutoLyrixAlign ready.")
            else:
                # list first level to see structure
                for p in AUTOLYRIXALIGN_DIR.iterdir():
                    if (p / "align.py").exists():
                        for f in p.iterdir():
                            shutil.move(str(f), str(AUTOLYRIXALIGN_DIR / f.name))
                        p.rmdir()
                        break
            zip_path.unlink(missing_ok=True)
    else:
        print("AutoLyrixAlign already present.")

    # --- .env.optional ---
    env_path = BACKEND_DIR / ".env.optional"
    lyrics_root = LYRICS_ALIGNER_DIR.resolve()
    lines = [
        "# Optional aligners (generated by scripts/setup_optional_aligners.py)",
        f"LYRICS_ALIGNER_ROOT={lyrics_root}",
        f"LYRICS_ALIGNER_PYTHON={sys.executable}",
    ]
    if (AUTOLYRIXALIGN_DIR / "align.py").exists():
        lines.append(f"AUTOLYRIXALIGN_ROOT={AUTOLYRIXALIGN_DIR.resolve()}")
    else:
        lines.append("# AUTOLYRIXALIGN_ROOT=...  # run with gdown installed to download, or set manually")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote", env_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
