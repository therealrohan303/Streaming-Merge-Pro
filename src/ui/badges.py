"""Shared platform badge rendering and UI component helpers for all pages."""

from src.config import CARD_BG, CARD_BORDER, CARD_TEXT, CARD_TEXT_MUTED, PLATFORMS

# Default teal accent matching the merged entity color
_DEFAULT_ACCENT = "#00897B"


def section_header_html(
    title: str,
    subtitle: str = "",
    accent_color: str = _DEFAULT_ACCENT,
    font_size: str = "1.35em",
) -> str:
    """Render a styled section header with a left-border accent and muted subtitle.

    Use ``font_size="2em"`` for page-level titles; default ``1.35em`` for section headers.

    Usage::

        st.markdown(section_header_html("Overview", "Key merger metrics"), unsafe_allow_html=True)
    """
    subtitle_html = (
        f'<div style="color:{CARD_TEXT_MUTED};font-size:0.85em;margin-top:4px;'
        f'font-weight:400;">{subtitle}</div>'
        if subtitle
        else ""
    )
    return (
        f'<div style="border-left:4px solid {accent_color};padding-left:12px;margin:8px 0 4px 0;">'
        f'<div style="font-size:{font_size};font-weight:700;color:{CARD_TEXT};letter-spacing:-0.01em;">'
        f'{title}</div>'
        f'{subtitle_html}'
        f'</div>'
    )


def styled_metric_card_html(
    label: str,
    value: str,
    delta: str | None = None,
    subtitle: str | None = None,
    accent_color: str = _DEFAULT_ACCENT,
    help_text: str | None = None,
) -> str:
    """Render a styled metric card.

    Layout: colored top-border → muted label → large value → delta badge → optional subtitle.

    Usage::

        st.markdown(styled_metric_card_html("Catalog Size", "9,167", "+65% vs Netflix"), unsafe_allow_html=True)
    """
    if help_text:
        # CSS tooltip — browser title-attribute tooltips don't fire reliably inside Streamlit.
        # Inject a <style> block once per render; CSS is idempotent so repeated injection is harmless.
        help_html = (
            "<style>"
            ".sm-tip{position:relative;display:inline-block;vertical-align:middle;cursor:default;}"
            ".sm-tip::after{content:attr(data-tip);display:none;position:absolute;"
            "bottom:130%;left:50%;transform:translateX(-50%);"
            "background:#111827;color:#d1d5db;padding:5px 10px;border-radius:5px;"
            "font-size:11px;font-weight:400;font-style:normal;white-space:nowrap;"
            "z-index:9999;border:1px solid #374151;pointer-events:none;line-height:1.5;}"
            ".sm-tip:hover::after{display:block;}"
            "</style>"
            f'<span class="sm-tip" data-tip="{help_text}" '
            f'style="display:inline-block;vertical-align:middle;margin-left:4px;cursor:default;">'
            f'<svg width="13" height="13" viewBox="0 0 13 13" fill="none" xmlns="http://www.w3.org/2000/svg">'
            f'<circle cx="6.5" cy="6.5" r="6" stroke="{CARD_TEXT_MUTED}" stroke-width="1"/>'
            f'<text x="6.5" y="10" text-anchor="middle" font-size="8.5" '
            f'fill="{CARD_TEXT_MUTED}" font-family="Arial,sans-serif">?</text>'
            f'</svg>'
            f'</span>'
        )
    else:
        help_html = ""
    subtitle_html = (
        f'<div style="color:{CARD_TEXT_MUTED};font-size:0.72em;margin-top:3px;">{subtitle}</div>'
        if subtitle
        else ""
    )
    # Delta badge — green if starts with "+", red if starts with "-", gray otherwise
    delta_html = ""
    if delta:
        if delta.startswith("+"):
            d_bg, d_color = "rgba(46,204,113,0.15)", "#2ecc71"
        elif delta.startswith("-"):
            d_bg, d_color = "rgba(231,76,60,0.15)", "#e74c3c"
        else:
            d_bg, d_color = "rgba(136,136,136,0.15)", CARD_TEXT_MUTED
        delta_html = (
            f'<span style="display:inline-block;background:{d_bg};color:{d_color};'
            f'padding:2px 8px;border-radius:10px;font-size:0.75em;margin-top:5px;">'
            f'{delta}</span>'
        )

    return (
        f'<div style="background:{CARD_BG};border:1px solid {CARD_BORDER};'
        f'border-top:3px solid {accent_color};border-radius:8px;'
        f'padding:14px 16px;height:100%;">'
        f'<div style="color:{CARD_TEXT_MUTED};font-size:0.78em;font-weight:500;'
        f'text-transform:uppercase;letter-spacing:0.04em;margin-bottom:6px;">'
        f'{label}{help_html}</div>'
        f'<div style="font-size:1.6em;font-weight:700;color:{CARD_TEXT};line-height:1.1;">'
        f'{value}</div>'
        f'{delta_html}'
        f'{subtitle_html}'
        f'</div>'
    )


def styled_banner_html(
    icon: str,
    text: str,
    bg: str = "#1E3A5F",
    border_color: str = "#1D6FA4",
) -> str:
    """Render a styled info/insight banner with an icon.

    Replacement for plain ``st.info()`` calls where richer styling is needed.

    Usage::

        st.markdown(styled_banner_html("ℹ️", "The merger adds **2,921 titles**..."), unsafe_allow_html=True)
    """
    icon_html = (
        f'<span style="margin-right:8px;font-size:1.1em;">{icon}</span>'
        if icon
        else ""
    )
    return (
        f'<div style="background:{bg};border:1px solid {border_color};'
        f'border-radius:8px;padding:12px 16px;margin:8px 0;">'
        f'{icon_html}'
        f'<span style="color:{CARD_TEXT};font-size:0.92em;">{text}</span>'
        f'</div>'
    )


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


def _hex_to_rgb(hex_color: str) -> str:
    """Convert a hex color string to 'r,g,b' for use in rgba()."""
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = h[0] * 2 + h[1] * 2 + h[2] * 2
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r},{g},{b}"


def page_header_html(
    title: str,
    subtitle: str = "",
    accent_color: str = "#00B4A6",
) -> str:
    """Render a prominent page-level title header.

    Visually distinct from section_header_html — uses a gradient background,
    larger bold title (2.4em), and a bottom accent border. Use this once per page
    at the very top. Use section_header_html for within-page section dividers.

    Usage::

        st.markdown(page_header_html("Explore Catalog", "Search and discover titles"), unsafe_allow_html=True)
    """
    rgb = _hex_to_rgb(accent_color)
    subtitle_html = (
        f'<div style="font-size:0.97em;color:{CARD_TEXT_MUTED};margin-top:0.45rem;'
        f'font-weight:400;line-height:1.45;">{subtitle}</div>'
        if subtitle
        else ""
    )
    return (
        f'<div style="padding:1.6rem 2rem 1.4rem 2rem;margin-bottom:1.8rem;'
        f'background:linear-gradient(135deg,rgba({rgb},0.07) 0%,rgba(30,30,46,0) 55%);'
        f'border-bottom:2px solid {accent_color};'
        f'border-top:1px solid rgba({rgb},0.18);'
        f'border-radius:4px;">'
        f'<div style="font-size:2.4em;font-weight:800;color:#FFFFFF;'
        f'letter-spacing:-0.8px;line-height:1.05;">{title}</div>'
        f'{subtitle_html}'
        f'</div>'
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
