"""External chart adapters for the shared Rust codec."""

from dataclasses import asdict
from pathlib import Path

from models import Phrase, UltraStarNote
from processing.rust_utility import RustUtilityUnavailable, call


def read_chart_text(path: Path) -> str:
    return call("read_external_chart", path=str(path))


def parse_chart(content: str, fallback_title: str) -> dict:
    try:
        chart = call("parse_external_chart", content=content, fallback_title=fallback_title)
    except RustUtilityUnavailable:
        raise
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc
    chart["phrases"] = [
        Phrase(notes=[UltraStarNote(**note) for note in phrase["notes"]], end_beat=phrase["end_beat"])
        for phrase in chart["phrases"]
    ]
    return chart


def update_chart(content: str, phrases: list[Phrase], pitch_offset: int) -> str:
    try:
        return call(
            "update_external_chart", content=content,
            phrases=[asdict(phrase) for phrase in phrases], pitch_offset=pitch_offset,
        )
    except RustUtilityUnavailable:
        raise
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc
