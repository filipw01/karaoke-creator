"""Song metadata adapters for the shared Rust catalog."""

from processing.rust_utility import call


def clean_music_title(title: str) -> str:
    return call("clean_title", title=title)


def parse_filename(filename: str) -> dict[str, str]:
    return call("parse_filename", filename=filename)
