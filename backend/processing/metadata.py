"""Extract song metadata (artist, title, etc.) from filename using a local LLM."""

from __future__ import annotations

import json
import logging
import re
import urllib.parse
import urllib.request

log = logging.getLogger("metadata")

_FEW_SHOT_PROMPT = """Extract title and artist from a music filename. Return JSON.

Filename: Linkin Park - Numb (Official Video).mp3
{"title": "Numb", "artist": "Linkin Park", "language": "English", "year": "", "genre": ""}

Filename: Adele - Hello (Lyrics).mp4
{"title": "Hello", "artist": "Adele", "language": "English", "year": "", "genre": ""}

Filename: BTS (방탄소년단) - Dynamite Official MV.mp4
{"title": "Dynamite", "artist": "BTS", "language": "English", "year": "", "genre": ""}

Filename: Ed Sheeran ft. Beyonce - Perfect Duet (Official Audio).flac
{"title": "Perfect Duet", "artist": "Ed Sheeran feat. Beyonce", "language": "English", "year": "", "genre": ""}

Filename: """

_MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
_MODEL_FILE = "qwen2.5-0.5b-instruct-q8_0.gguf"

_llm = None


def _get_llm():
    global _llm
    if _llm is not None:
        return _llm

    log.info("Loading local LLM (%s)...", _MODEL_FILE)
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama

    model_path = hf_hub_download(repo_id=_MODEL_REPO, filename=_MODEL_FILE)
    _llm = Llama(model_path=model_path, n_ctx=512, n_threads=4, verbose=False)
    log.info("Local LLM loaded")
    return _llm


def parse_filename(filename: str) -> dict[str, str]:
    """Use a local LLM to extract metadata from a filename. Falls back to simple heuristic."""
    name = filename.rsplit(".", 1)[0] if "." in filename else filename

    try:
        return _llm_parse(name)
    except Exception as e:
        log.warning("LLM metadata extraction failed (%s), using heuristic", e)
        return _heuristic_parse(name)


def _clean_value(val: str) -> str:
    """Strip placeholder values the model might output."""
    val = val.strip()
    if val.lower() in ("?", "none", "unknown", "n/a", "null"):
        return ""
    return val


_TITLE_NOISE_PATTERNS = (
    r"\bofficial\s+(?:music\s+)?(?:video|audio|visualizer)\b",
    r"\bofficial\s+mv\b",
    r"\bofficial\s+lyric\s+video\b",
    r"\blyric\s*(?:video)?\b",
    r"\b(?:music\s+)?visualizer\b",
    r"\b(?:hq|hd|4k|8k)\b",
    r"\bremaster(?:ed)?\b",
    r"\b(?:audio|mv)\b",
)


def clean_music_title(title: str) -> str:
    """Remove common release/presentation noise from a suggested song title."""
    cleaned = title.rsplit(".", 1)[0] if title.lower().endswith((".mp3", ".wav", ".flac", ".m4a", ".mp4", ".mkv", ".webm", ".avi")) else title
    cleaned = re.sub(r"\s*[\[(].*?(?:official\s+(?:music\s+)?(?:video|audio)|official\s+mv|lyrics?|lyric\s+video|visualizer|audio|remaster(?:ed)?|hq|hd|4k|8k)[^)\]]*[\])]", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*[-–—]\s*(?:official\s+(?:music\s+)?(?:video|audio)|official\s+mv|official\s+lyric\s+video|lyric\s*(?:video)?|visualizer|audio|remaster(?:ed)?|hq|hd|4k|8k)\b.*$", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+(?:official\s+(?:music\s+)?(?:video|audio)|official\s+mv|official\s+lyric\s+video|lyric\s*(?:video)?|visualizer|audio|remaster(?:ed)?|hq|hd|4k|8k)\b.*$", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+(?:official\s+(?:music\s+)?(?:video|audio)|official\s+mv|lyrics?|lyric\s+video|visualizer|audio|remaster(?:ed)?|hq|hd|4k|8k)\s*$", "", cleaned, flags=re.I)
    for pattern in _TITLE_NOISE_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+[|].*$", "", cleaned)
    cleaned = re.sub(r"\s*[-–—]\s*(?:topic|vevo)\s*$", "", cleaned, flags=re.I)
    cleaned = re.sub(r"[\s._]+", " ", cleaned)
    return cleaned.strip(" .-_")


def _llm_parse(name: str) -> dict[str, str]:
    """Run inference on the local LLM using few-shot prompting."""
    llm = _get_llm()
    prompt = _FEW_SHOT_PROMPT + name + "\n"

    output = llm(prompt, max_tokens=100, temperature=0, stop=["\n\n", "\nFilename:"])
    content = output["choices"][0]["text"]  # type: ignore
    log.debug("LLM raw output: %s", content)

    result = _extract_json(content)
    return {
        "title": clean_music_title(_clean_value(str(result.get("title", "")))),
        "artist": _clean_value(str(result.get("artist", ""))),
        "language": _clean_value(str(result.get("language", ""))),
        "year": _clean_value(str(result.get("year", ""))),
        "genre": _clean_value(str(result.get("genre", ""))),
    }


def _extract_json(text: str) -> dict:
    """Robustly extract a JSON object from LLM output."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No JSON found in: {text[:200]}")


def _heuristic_parse(name: str) -> dict[str, str]:
    """Simple fallback: split on ' - ' separator."""
    parts = name.split(" - ", 1)
    if len(parts) == 2:
        return {
            "title": clean_music_title(parts[1]),
            "artist": parts[0].strip(),
            "language": "",
            "year": "",
            "genre": "",
        }
    return {"title": clean_music_title(name), "artist": "", "language": "", "year": "", "genre": ""}


def lookup_release_metadata(title: str, artist: str) -> dict[str, str]:
    """Look up optional release year and genre from Apple's public catalog API."""
    if not title:
        return {"year": "", "genre": ""}

    query = " ".join(part for part in (artist, title) if part)
    params = urllib.parse.urlencode({"term": query, "entity": "song", "limit": 1})
    request = urllib.request.Request(
        f"https://itunes.apple.com/search?{params}",
        headers={"User-Agent": "Karaoke-Creator/1.0"},
    )

    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        for result in payload.get("results", []):
            release_date = str(result.get("releaseDate", ""))
            year = release_date[:4] if re.match(r"^\d{4}", release_date) else ""
            genre = str(result.get("primaryGenreName", "")).strip()
            if year or genre:
                return {"year": year, "genre": genre}
    except Exception as exc:
        log.info("Release metadata lookup failed: %s", exc)
    return {"year": "", "genre": ""}
