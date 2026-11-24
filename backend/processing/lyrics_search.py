"""Lyrics lookup adapters for the shared Rust catalog."""

from processing.rust_utility import call


def search_lyrics_query(q: str = "") -> list[dict]:
    return call("search_lyrics", query=q.strip()) if q.strip() else []
