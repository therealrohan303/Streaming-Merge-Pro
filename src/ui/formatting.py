"""Shared display formatting helpers for user-facing copy."""

from __future__ import annotations

_GENRE_DISPLAY = {
    "documentation": "Documentary",
    "scifi": "Sci-Fi",
}


def genre_display(key: str | None) -> str:
    if not key:
        return ""
    return _GENRE_DISPLAY.get(str(key).lower(), str(key).title())


def genre_list_display(keys) -> str:
    if not keys:
        return ""
    names = [genre_display(k) for k in keys]
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} & {names[1]}"
    return ", ".join(names[:-1]) + f" & {names[-1]}"
