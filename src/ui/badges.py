"""Shared platform badge rendering for consistent styling across all pages."""

from src.config import PLATFORMS


def platform_badge_html(platform_key: str) -> str:
    """Render a single platform badge as inline HTML.

    Uses text_color from PLATFORMS config for correct contrast
    (e.g., Apple TV uses dark text on light silver background).
    """
    meta = PLATFORMS.get(platform_key, {})
    name = meta.get("name", platform_key.title())
    bg = meta.get("color", "#555")
    fg = meta.get("text_color", "#fff")
    return (
        f'<span style="background:{bg};color:{fg};padding:2px 8px;'
        f'border-radius:4px;font-size:0.75em;font-weight:600;'
        f'letter-spacing:0.02em;margin-right:3px;">{name}</span>'
    )


def platform_badges_html(platforms) -> str:
    """Render one or more platform badges as inline HTML.

    Accepts a single platform key (str) or a list of keys.
    """
    if isinstance(platforms, str):
        return platform_badge_html(platforms)
    if isinstance(platforms, (list, tuple)):
        return "".join(platform_badge_html(p) for p in platforms)
    return ""
