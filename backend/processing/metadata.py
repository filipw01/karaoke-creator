"""Extract song metadata (artist, title, etc.) from filename using a local LLM."""

from __future__ import annotations

import json
import logging
import re

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


def _llm_parse(name: str) -> dict[str, str]:
    """Run inference on the local LLM using few-shot prompting."""
    llm = _get_llm()
    prompt = _FEW_SHOT_PROMPT + name + "\n"

    output = llm(prompt, max_tokens=100, temperature=0, stop=["\n\n", "\nFilename:"])
    content = output["choices"][0]["text"]  # type: ignore
    log.debug("LLM raw output: %s", content)

    result = _extract_json(content)
    return {
        "title": _clean_value(str(result.get("title", ""))),
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
            "title": parts[1].strip(),
            "artist": parts[0].strip(),
            "language": "",
            "year": "",
            "genre": "",
        }
    return {"title": name.strip(), "artist": "", "language": "", "year": "", "genre": ""}
