"""Search for song lyrics online using free APIs."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request

log = logging.getLogger("lyrics_search")

_TIMEOUT = 10


def _fetch_json(url: str) -> dict | list | None:
    """GET a URL and parse JSON response, returning None on any failure."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "KaraokeCreator/1.0"})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        log.debug("Request to %s failed: %s", url, e)
        return None


def _clean_title(title: str) -> str:
    """Remove common noise suffixes from title that hurt search results."""
    clean = title
    for noise in ["(Official Video)", "(Official Music Video)", "(Lyric Video)",
                   "(Official Audio)", "(Audio)", "[Official Video]",
                   "(ft.", "(feat.", "ft.", "feat.", "|", "｜"]:
        idx = clean.lower().find(noise.lower())
        if idx >= 0:
            clean = clean[:idx].strip()
    return clean.strip()


def _search_lyricsovh(artist: str, title: str) -> str | None:
    """Try lyrics.ovh API (exact match)."""
    url = f"https://api.lyrics.ovh/v1/{urllib.parse.quote(artist)}/{urllib.parse.quote(title)}"
    data = _fetch_json(url)
    if isinstance(data, dict) and "lyrics" in data:
        lyrics = data["lyrics"].strip()
        if lyrics:
            log.info("Found lyrics via lyrics.ovh (%d chars)", len(lyrics))
            return lyrics
    return None


def _search_lrclib(artist: str, title: str) -> str | None:
    """Try lrclib.net get endpoint (exact match)."""
    params = urllib.parse.urlencode({"artist_name": artist, "track_name": title})
    url = f"https://lrclib.net/api/get?{params}"
    data = _fetch_json(url)
    if isinstance(data, dict):
        lyrics = (data.get("plainLyrics") or "").strip()
        if lyrics:
            log.info("Found lyrics via lrclib.net (%d chars)", len(lyrics))
            return lyrics
    return None


def search_lyrics(artist: str, title: str) -> str:
    """Auto-search for lyrics by artist and title. Returns lyrics text or empty string."""
    log.info("Searching lyrics for '%s - %s'", artist, title)

    clean_title = _clean_title(title)

    # 1. Exact-match APIs (fast, precise)
    for t in ([clean_title, title] if clean_title != title else [clean_title]):
        result = _search_lrclib(artist, t)
        if result:
            return result
        result = _search_lyricsovh(artist, t)
        if result:
            return result

    # 2. Fuzzy search via combined query
    for t in ([clean_title, title] if clean_title != title else [clean_title]):
        matches = search_lyrics_query(f"{artist} {t}")
        if matches:
            log.info("Using first fuzzy match: '%s - %s'", matches[0]["artist_name"], matches[0]["track_name"])
            return matches[0]["lyrics"]

    log.info("No lyrics found online for '%s - %s'", artist, title)
    return ""


def _search_lrclib_multi(query: str) -> list[dict]:
    """Query lrclib.net /api/search and return parsed results (includes lyrics)."""
    url = f"https://lrclib.net/api/search?{urllib.parse.urlencode({'q': query})}"
    data = _fetch_json(url)
    if not isinstance(data, list):
        return []
    results: list[dict] = []
    for item in data:
        lyrics = (item.get("plainLyrics") or "").strip()
        if not lyrics:
            continue
        results.append({
            "track_name": item.get("trackName", ""),
            "artist_name": item.get("artistName", ""),
            "album_name": item.get("albumName", ""),
            "duration": item.get("duration", 0),
            "lyrics": lyrics,
            "source": "lrclib",
        })
    return results


def _suggest_lyricsovh(query: str, max_fetch: int = 5) -> list[dict]:
    """Query lyrics.ovh /suggest and fetch lyrics for top matches."""
    url = f"https://api.lyrics.ovh/suggest/{urllib.parse.quote(query)}"
    data = _fetch_json(url)
    if not isinstance(data, dict):
        return []

    items = data.get("data", [])
    results: list[dict] = []
    for item in items[:max_fetch]:
        artist = item.get("artist", {}).get("name", "")
        title = item.get("title_short") or item.get("title", "")
        album = item.get("album", {}).get("title", "")
        duration = item.get("duration", 0)
        if not artist or not title:
            continue
        lyrics = _search_lyricsovh(artist, title)
        if not lyrics:
            continue
        results.append({
            "track_name": title,
            "artist_name": artist,
            "album_name": album,
            "duration": duration,
            "lyrics": lyrics,
            "source": "lyrics.ovh",
        })
    return results


def search_lyrics_query(q: str = "") -> list[dict]:
    """Search both lrclib.net and lyrics.ovh with a single query, returning combined deduplicated results."""
    if not q.strip():
        return []

    log.info("Lyrics search: '%s'", q)

    combined: list[dict] = []
    seen: set[str] = set()

    for r in _search_lrclib_multi(q):
        key = r["lyrics"][:200]
        if key not in seen:
            seen.add(key)
            combined.append(r)

    for r in _suggest_lyricsovh(q):
        key = r["lyrics"][:200]
        if key not in seen:
            seen.add(key)
            combined.append(r)

    log.info("Combined search returned %d results", len(combined))
    return combined[:20]
