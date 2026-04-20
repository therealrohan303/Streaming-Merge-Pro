"""Page 4: Discovery Engine — Full recommendation toolkit.

Four distinct entry points:
  Tab 1: Title Match  — smart autocomplete + polished why-similar + inline detail panel
  Tab 2: Mood Board   — 16 mood tiles (genre+keyword+tag triple signal)
  Tab 3: Vibe Search  — NLP-powered semantic + keyword + genome hybrid
  Tab 4: History      — actionable recent searches log with Run Again + pin
"""

import numpy as np
import pandas as pd
import streamlit as st

from src.analysis.discovery import (
    MOOD_TILES,
    extract_vibe_signals,
    get_similar_with_explanation,
    mood_board_recommendations,
    vibe_search,
)
from src.analysis.scoring import compute_quality_score, format_votes
from src.config import (
    CARD_ACCENT,
    CARD_BG,
    CARD_BORDER,
    CARD_TEXT,
    CARD_TEXT_MUTED,
    PLATFORMS,
    SIMILARITY_MIN_IMDB,
)
from src.data.loaders import (
    deduplicate_titles,
    get_titles_for_view,
    load_all_platforms_credits,
    load_all_platforms_titles,
    load_enriched_titles,
    load_genome_vectors,
    load_imdb_principals,
    load_merged_titles,
    load_similarity_data,
)
from src.ui.badges import page_header_html, platform_badges_html, section_header_html, styled_banner_html
from src.ui.filters import apply_filters, render_sidebar_filters
from src.ui.session import init_session_state

st.set_page_config(page_title="Discovery Engine", page_icon="🔎", layout="wide")
init_session_state()

st.markdown(
    page_header_html(
        "Discovery Engine",
        "Your full recommendation toolkit — find your next favourite title through four distinct approaches.",
    ),
    unsafe_allow_html=True,
)

# ─── Data loading ─────────────────────────────────────────────────────────────

raw_df = get_titles_for_view(st.session_state.get("platform_view", "merged"))
filters = render_sidebar_filters(raw_df)
titles_all = get_titles_for_view(filters["platform_view"])
titles = apply_filters(titles_all, filters)
titles = deduplicate_titles(titles)
titles["quality_score"] = compute_quality_score(titles)

enriched_df   = load_enriched_titles()
enriched_df   = deduplicate_titles(enriched_df)
principals_df = load_imdb_principals()
sim_df        = load_similarity_data()
genome_vectors, genome_id_map = load_genome_vectors()

with st.sidebar:
    st.metric("Titles Available", f"{len(titles):,}")

# ─── Session state ─────────────────────────────────────────────────────────────

for _key, _default in [
    ("rec_history",       []),
    ("hist_counter",      0),
    ("sim_selected_id",   None),
    ("sim_results",       None),
    ("selected_moods",    []),
    ("discovery_detail_id", None),
    ("run_again_flag",    None),
    ("run_sim_now",       False),
    ("run_mood_now",      False),
    ("run_vibe_now",      False),
    ("mood_results",      None),
    ("vibe_results",      None),
    ("vibe_signals",      []),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default

# ─── Helper functions ──────────────────────────────────────────────────────────

def _get_poster_url(title_id: str, enr_df: pd.DataFrame):
    if enr_df is None or "poster_url" not in enr_df.columns:
        return None
    rows = enr_df[enr_df["id"] == title_id]
    if rows.empty:
        return None
    url = rows.iloc[0].get("poster_url")
    if url and str(url) not in ("nan", "None", ""):
        return str(url)
    return None


def _poster_html(url, title: str, platform_keys, width=90, height=130) -> str:
    initial = (title[0].upper() if title else "?")
    keys = platform_keys if isinstance(platform_keys, list) else [platform_keys] if platform_keys else ["netflix"]
    color = "#2a2a3e"  # neutral dark placeholder; avoids jarring brand reds (e.g. Netflix)
    placeholder = (
        f'<div style="width:{width}px;height:{height}px;background:{color};'
        f'border-radius:6px;display:flex;align-items:center;justify-content:center;'
        f'font-size:{max(width//3,14)}px;font-weight:700;color:#fff;flex-shrink:0;">'
        f'{initial}</div>'
    )
    if not url:
        return placeholder
    return (
        f'<img src="{url}" width="{width}" height="{height}" '
        f'style="object-fit:cover;border-radius:6px;flex-shrink:0;" />'
    )


def _genre_pills_html(genres, max_n=4) -> str:
    """Render genre pills with Title Case (genres are lowercase in data)."""
    if genres is None:
        return ""
    lst = list(genres) if isinstance(genres, np.ndarray) else genres
    if not isinstance(lst, (list, tuple)):
        return ""
    return " ".join(
        f'<span style="background:#2a2a3e;color:{CARD_TEXT_MUTED};padding:2px 7px;'
        f'border-radius:10px;font-size:0.72em;margin-right:3px;">{str(g).title()}</span>'
        for g in lst[:max_n]
    )


def _sim_score_badge_html(score: float) -> str:
    pct = int(score * 100)
    if score >= 0.75:
        bg, fg = "rgba(46,204,113,0.18)", "#2ecc71"
    elif score >= 0.60:
        bg, fg = "rgba(243,156,18,0.18)", "#f39c12"
    else:
        bg, fg = "rgba(136,136,136,0.18)", CARD_TEXT_MUTED
    return (
        f'<span style="background:{bg};color:{fg};padding:3px 9px;'
        f'border-radius:10px;font-size:0.75em;font-weight:600;white-space:nowrap;">'
        f'{pct}% match</span>'
    )


def _mood_score_badge_html(pct: float) -> str:
    p = int(pct * 100)
    if pct >= 0.75:
        bg, fg = "rgba(46,204,113,0.18)", "#2ecc71"
    elif pct >= 0.40:
        bg, fg = "rgba(243,156,18,0.18)", "#f39c12"
    else:
        bg, fg = "rgba(136,136,136,0.18)", CARD_TEXT_MUTED
    return (
        f'<span style="background:{bg};color:{fg};padding:3px 9px;'
        f'border-radius:10px;font-size:0.75em;font-weight:600;white-space:nowrap;">'
        f'Mood {p}%</span>'
    )


def _vibe_label_badge_html(vibe_score: float) -> str:
    if vibe_score >= 0.40:
        bg, fg, label = "rgba(46,204,113,0.18)", "#2ecc71", "Strong match"
    elif vibe_score >= 0.25:
        bg, fg, label = "rgba(243,156,18,0.18)", "#f39c12", "Good match"
    else:
        bg, fg, label = "rgba(136,136,136,0.18)", CARD_TEXT_MUTED, "Partial match"
    return (
        f'<span style="background:{bg};color:{fg};padding:3px 9px;'
        f'border-radius:10px;font-size:0.75em;font-weight:600;white-space:nowrap;">'
        f'{label}</span>'
    )


def _render_rec_card(row: dict, score_badge_html: str = "") -> None:
    """Unified poster-left result card used across all three search tabs."""
    title      = row.get("title", "Untitled")
    year       = row.get("release_year", "")
    imdb       = row.get("imdb_score")
    votes      = row.get("imdb_votes")
    genres     = row.get("genres", [])
    platforms  = row.get("platforms") or ([row["platform"]] if row.get("platform") else [])
    tid        = row.get("similar_id") or row.get("id", "")

    imdb_str  = f"{imdb:.1f}" if imdb and pd.notna(imdb) else "N/A"
    votes_str = format_votes(votes) if votes and pd.notna(votes) else ""
    year_str  = str(int(year)) if year and pd.notna(year) else ""
    votes_html = (
        f'<span style="color:{CARD_TEXT_MUTED};font-size:0.88em;margin-left:4px;">({votes_str})</span>'
        if votes_str else ""
    )

    poster_url = _get_poster_url(tid, enriched_df)
    poster_html = _poster_html(poster_url, title, platforms)
    genre_html  = _genre_pills_html(genres)
    badge_html  = platform_badges_html(platforms)

    st.markdown(
        f'<div style="display:flex;gap:12px;background:{CARD_BG};border:1px solid {CARD_BORDER};'
        f'border-top:3px solid {CARD_ACCENT};border-radius:8px;padding:12px;margin-bottom:6px;">'
        f'{poster_html}'
        f'<div style="flex:1;min-width:0;">'
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:6px;">'
        f'<div style="font-weight:700;color:{CARD_TEXT};font-size:1rem;line-height:1.3;">{title}</div>'
        f'<div style="flex-shrink:0;">{score_badge_html}</div>'
        f'</div>'
        f'<div style="color:{CARD_TEXT_MUTED};font-size:0.82em;margin:2px 0 5px;">{year_str}</div>'
        f'<div style="margin-bottom:5px;">{badge_html}</div>'
        f'<div style="color:{CARD_TEXT};font-size:0.83em;margin-bottom:5px;">'
        f'{imdb_str}{votes_html}'
        f'</div>'
        f'<div>{genre_html}</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def _meta_cell(label: str, value: str) -> str:
    return (
        f'<div style="padding:8px 0;">'
        f'<div style="color:{CARD_TEXT_MUTED};font-size:0.72em;text-transform:uppercase;'
        f'letter-spacing:0.04em;margin-bottom:3px;">{label}</div>'
        f'<div style="font-weight:600;color:{CARD_TEXT};font-size:0.95em;">{value}</div>'
        f'</div>'
    )


def _render_title_detail(title_id: str) -> None:
    """Full title detail panel — inline, rendered below the triggering card."""
    all_titles = deduplicate_titles(load_all_platforms_titles())
    rows = all_titles[all_titles["id"] == title_id]
    if rows.empty:
        st.warning("Title details not found.")
        return

    row = rows.iloc[0]
    enr_rows = enriched_df[enriched_df["id"] == title_id] if enriched_df is not None else pd.DataFrame()
    enr = enr_rows.iloc[0] if not enr_rows.empty else None

    # ── Derived values ──
    poster_url   = str(enr.get("poster_url")) if enr is not None else ""
    poster_url   = poster_url if poster_url and poster_url not in ("nan", "None", "") else None
    title        = row.get("title", "")
    year         = int(row["release_year"]) if pd.notna(row.get("release_year")) else ""
    content_type = str(row.get("type", "") or "").title()
    imdb_val     = row.get("imdb_score")
    imdb_str     = f"{imdb_val:.1f}" if imdb_val and pd.notna(imdb_val) else "N/A"
    votes        = row.get("imdb_votes")
    votes_str    = format_votes(votes) if votes and pd.notna(votes) else ""
    cert         = str(row.get("age_certification") or "").strip()
    runtime      = row.get("runtime")
    runtime_str  = f"{int(runtime)} min" if runtime and pd.notna(runtime) else ""
    platforms    = row.get("platforms") or ([row.get("platform", "")])
    genres       = row.get("genres", [])
    desc         = str(row.get("description", "") or "")
    desc         = "" if desc in ("nan", "None") else desc

    wins_str = ""
    if enr is not None:
        wins = enr.get("award_wins")
        if wins and pd.notna(wins) and wins > 0:
            wi = int(wins)
            noms = enr.get("award_noms")
            ns = f" / {int(noms)} nom" if noms and pd.notna(noms) else ""
            wins_str = f"{wi} win{'s' if wi != 1 else ''}{ns}"

    bo_str = ""
    if enr is not None:
        bo = enr.get("box_office_usd")
        if bo and pd.notna(bo) and bo > 0:
            bo_str = f"${bo/1e9:.1f}B" if bo >= 1e9 else f"${bo/1e6:.0f}M" if bo >= 1e6 else f"${bo:,.0f}"

    qs_val    = compute_quality_score(pd.DataFrame([row.to_dict()])).iloc[0]
    qs_str    = f"{qs_val:.2f}"
    qs_color  = "#2ecc71" if qs_val >= 8 else "#f39c12" if qs_val >= 7 else "#e74c3c"
    genre_lst = list(genres) if isinstance(genres, np.ndarray) else (genres if isinstance(genres, (list, tuple)) else [])

    with st.container():
        # ── Close button top-right ──
        _, close_col = st.columns([12, 1])
        with close_col:
            if st.button("✕", key=f"close_detail_{title_id}", help="Close details"):
                st.session_state.discovery_detail_id = None
                st.rerun()

        # ── Poster | Info ──
        poster_col, info_col = st.columns([1, 4])
        with poster_col:
            if poster_url:
                st.image(poster_url, width=160)
            else:
                color = PLATFORMS.get(platforms[0] if platforms else "netflix", {}).get("color", "#444")
                st.markdown(
                    f'<div style="width:160px;height:232px;background:{color};border-radius:8px;'
                    f'display:flex;align-items:center;justify-content:center;'
                    f'font-size:48px;font-weight:700;color:#fff;">'
                    f'{title[0].upper() if title else "?"}</div>',
                    unsafe_allow_html=True,
                )

        with info_col:
            # Title + year + type
            type_pill = (
                f'<span style="background:#2a2a3e;color:{CARD_TEXT_MUTED};padding:2px 8px;'
                f'border-radius:8px;font-size:0.72em;font-weight:600;margin-left:6px;">'
                f'{content_type}</span>'
                if content_type else ""
            )
            st.markdown(
                f'<div style="font-size:1.35rem;font-weight:700;color:{CARD_TEXT};'
                f'line-height:1.3;margin-bottom:2px;">{title}</div>'
                f'<div style="color:{CARD_TEXT_MUTED};font-size:0.9em;margin-bottom:8px;">'
                f'{year}{type_pill}</div>',
                unsafe_allow_html=True,
            )

            # IMDb + votes + award badge
            imdb_row = f'<span style="color:{CARD_ACCENT};font-weight:700;font-size:1.05rem;">IMDb {imdb_str}</span>'
            if votes_str:
                imdb_row += f' <span style="color:{CARD_TEXT_MUTED};font-size:0.82em;">({votes_str} votes)</span>'
            if wins_str:
                imdb_row += (
                    f'&nbsp;&nbsp;<span style="background:rgba(46,204,113,0.12);color:#2ecc71;'
                    f'border:1px solid #2ecc71;padding:2px 9px;border-radius:10px;'
                    f'font-size:0.8em;font-weight:600;">{wins_str}</span>'
                )
            st.markdown(f'<div style="margin-bottom:8px;">{imdb_row}</div>', unsafe_allow_html=True)

            # Platforms + genres
            plat_html  = platform_badges_html(platforms)
            genre_html = " ".join(
                f'<span style="background:{CARD_BORDER};color:{CARD_TEXT};padding:2px 8px;'
                f'border-radius:10px;font-size:0.78em;margin-right:2px;">{g.title()}</span>'
                for g in genre_lst
            )
            st.markdown(
                f'<div style="display:flex;flex-wrap:wrap;gap:4px;align-items:center;margin-bottom:10px;">'
                f'{plat_html} {genre_html}</div>',
                unsafe_allow_html=True,
            )

            # Metadata: only non-empty fields
            meta_items = [
                (lbl, val, col) for lbl, val, col in [
                    ("Rating",     cert,        CARD_TEXT),
                    ("Runtime",    runtime_str, CARD_TEXT),
                    ("Box Office", bo_str,      CARD_TEXT),
                    ("Quality",    qs_str,      qs_color),
                ] if val
            ]
            if meta_items:
                meta_cols = st.columns(len(meta_items))
                for col, (lbl, val, txt_col) in zip(meta_cols, meta_items):
                    with col:
                        st.markdown(
                            f'<div style="padding:6px 0;">'
                            f'<div style="color:{CARD_TEXT_MUTED};font-size:0.72em;text-transform:uppercase;'
                            f'letter-spacing:0.04em;margin-bottom:3px;">{lbl}</div>'
                            f'<div style="font-weight:600;color:{txt_col};font-size:0.95em;">{val}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

        # ── Description ──
        if desc:
            st.markdown(
                f'<div style="border-top:1px solid {CARD_BORDER};margin-top:10px;padding-top:10px;'
                f'color:{CARD_TEXT};font-size:0.9em;line-height:1.65;">{desc}</div>',
                unsafe_allow_html=True,
            )

        # ── Cast & Crew ──
        credits_df_local = load_all_platforms_credits()
        if credits_df_local is not None and not credits_df_local.empty and "title_id" in credits_df_local.columns:
            tid = row.get("id")
            title_cred = credits_df_local[credits_df_local["title_id"] == tid] if tid else pd.DataFrame()
            if not title_cred.empty:
                role_col = "role" if "role" in title_cred.columns else None
                with st.expander("Cast & Crew", expanded=False):
                    if role_col:
                        directors = title_cred[title_cred[role_col] == "DIRECTOR"]
                        actors    = title_cred[title_cred[role_col] == "ACTOR"].head(12)
                    else:
                        directors = actors = pd.DataFrame()

                    if not directors.empty:
                        st.markdown(
                            f'<div style="color:{CARD_TEXT_MUTED};font-size:0.75em;text-transform:uppercase;'
                            f'letter-spacing:0.04em;margin-bottom:4px;">'
                            f'Director{"s" if len(directors) > 1 else ""}</div>',
                            unsafe_allow_html=True,
                        )
                        for _, d in directors.iterrows():
                            st.markdown(
                                f'<div style="font-size:0.88em;color:{CARD_TEXT};margin-bottom:2px;">'
                                f'<strong>{d.get("name", "")}</strong></div>',
                                unsafe_allow_html=True,
                            )
                    if not actors.empty:
                        st.markdown(
                            f'<div style="color:{CARD_TEXT_MUTED};font-size:0.75em;text-transform:uppercase;'
                            f'letter-spacing:0.04em;margin:8px 0 4px;">Cast</div>',
                            unsafe_allow_html=True,
                        )
                        ac1, ac2 = st.columns(2)
                        for i, (_, a) in enumerate(actors.iterrows()):
                            char = str(a.get("character", "") or "")
                            char_str = (
                                f' <span style="color:{CARD_TEXT_MUTED};font-size:0.8em;">as {char}</span>'
                                if char and char not in ("nan", "") else ""
                            )
                            with (ac1 if i % 2 == 0 else ac2):
                                st.markdown(
                                    f'<div style="font-size:0.85em;color:{CARD_TEXT};padding:2px 0;">'
                                    f'<strong>{a.get("name", "")}</strong>{char_str}</div>',
                                    unsafe_allow_html=True,
                                )

        # ── Explore Catalog link ──
        st.markdown(
            f'<div style="margin-top:8px;">'
            f'<a href="/Explore_Catalog" style="color:{CARD_ACCENT};font-size:0.85em;'
            f'text-decoration:none;font-weight:600;">Open in Explore Catalog →</a></div>',
            unsafe_allow_html=True,
        )


def _render_why_similar(result_row: dict, sim_score: float, source_row: dict) -> None:
    """Polished Why Similar expander body — 4 narrative sections."""
    explanation       = result_row.get("explanation", {})
    genre_overlap     = explanation.get("genre_overlap", [])
    shared_crew       = explanation.get("shared_crew", [])
    matched_vibe_tags = explanation.get("matched_vibe_tags", [])

    bar_color = "#2ecc71" if sim_score >= 0.75 else "#f39c12" if sim_score >= 0.60 else "#888"
    note      = "Very strong" if sim_score >= 0.75 else "Strong" if sim_score >= 0.60 else "Moderate"
    bar_pct   = int(sim_score * 100)

    # Section 1: Narrative Similarity bar
    st.markdown(
        f'<div style="margin-bottom:12px;">'
        f'<div style="color:{CARD_TEXT_MUTED};font-size:0.72em;text-transform:uppercase;'
        f'letter-spacing:0.04em;margin-bottom:5px;">Narrative Similarity</div>'
        f'<div style="background:#2a2a3e;border-radius:4px;height:6px;margin-bottom:5px;">'
        f'<div style="width:{bar_pct}%;height:100%;background:{bar_color};border-radius:4px;"></div>'
        f'</div>'
        f'<div style="color:{bar_color};font-size:0.83em;font-weight:600;">'
        f'{bar_pct}% — {note} match</div></div>',
        unsafe_allow_html=True,
    )

    # Section 2: Genre Alignment
    if genre_overlap:
        pills = " ".join(
            f'<span style="background:rgba(255,215,0,0.10);border:1px solid {CARD_ACCENT};'
            f'padding:2px 8px;border-radius:10px;font-size:0.78em;margin-right:3px;">'
            f'{g.title()}</span>'
            for g in genre_overlap[:4]
        )
        extra = (
            f' <span style="color:{CARD_TEXT_MUTED};font-size:0.78em;">+{len(genre_overlap)-4} more</span>'
            if len(genre_overlap) > 4 else ""
        )
        st.markdown(
            f'<div style="margin-bottom:12px;">'
            f'<div style="color:{CARD_TEXT_MUTED};font-size:0.72em;text-transform:uppercase;'
            f'letter-spacing:0.04em;margin-bottom:5px;">Genre Alignment</div>'
            f'{pills}{extra}</div>',
            unsafe_allow_html=True,
        )

    # Section 3: Creative Connections (shared crew)
    if shared_crew:
        lines = "".join(
            f'<div style="font-size:0.85em;color:{CARD_TEXT};padding:2px 0;">'
            f'👥 <strong>{c["name"]}</strong>'
            f'<span style="color:{CARD_TEXT_MUTED};"> — {c["role"].title()}</span></div>'
            for c in shared_crew[:3]
        )
        st.markdown(
            f'<div style="margin-bottom:12px;">'
            f'<div style="color:{CARD_TEXT_MUTED};font-size:0.72em;text-transform:uppercase;'
            f'letter-spacing:0.04em;margin-bottom:5px;">Creative Connections</div>'
            f'{lines}</div>',
            unsafe_allow_html=True,
        )

    # Section 4: Shared Themes
    if matched_vibe_tags:
        pills = " ".join(
            f'<span style="background:#2a2a3e;color:{CARD_TEXT_MUTED};padding:2px 7px;'
            f'border-radius:10px;font-size:0.75em;margin-right:3px;">{t}</span>'
            for t in matched_vibe_tags[:5]
        )
        st.markdown(
            f'<div><div style="color:{CARD_TEXT_MUTED};font-size:0.72em;text-transform:uppercase;'
            f'letter-spacing:0.04em;margin-bottom:5px;">Shared Themes</div>{pills}</div>',
            unsafe_allow_html=True,
        )

    if not genre_overlap and not shared_crew and not matched_vibe_tags:
        st.markdown(
            f'<div style="color:{CARD_TEXT_MUTED};font-size:0.85em;">'
            f'Matched by description and story similarity ({bar_pct}% overlap).</div>',
            unsafe_allow_html=True,
        )


def _add_to_history(entry: dict) -> None:
    """Add a search to history. Deduplicates consecutive identical searches.
    Entry schema: type, query, count, results (list of dicts), result_poster_urls,
                  pinned (bool), params (dict).
    """
    history = st.session_state.rec_history
    if (
        history
        and history[0].get("type") == entry.get("type")
        and history[0].get("query") == entry.get("query")
    ):
        history[0]["count_runs"] = history[0].get("count_runs", 1) + 1
        history[0]["count"]              = entry["count"]
        history[0]["results"]            = entry["results"]
        history[0]["result_poster_urls"] = entry.get("result_poster_urls", [])
        history[0]["params"]             = entry.get("params", {})
    else:
        entry.setdefault("pinned", False)
        entry.setdefault("params", {})
        entry["_id"] = st.session_state.hist_counter
        st.session_state.hist_counter += 1
        history.insert(0, entry)
        pinned   = [e for e in history if e.get("pinned", False)]
        unpinned = [e for e in history if not e.get("pinned", False)]
        st.session_state.rec_history = pinned + unpinned[:10]


# ─── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab_history = st.tabs(
    ["Title Match", "Mood Board", "Vibe Search", "History"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — TITLE MATCH
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown(
        section_header_html(
            "Title Match",
            "Find titles that share the same narrative DNA as something you already love.",
        ),
        unsafe_allow_html=True,
    )

    # Handle "Run Again" from History tab
    _flag = st.session_state.get("run_again_flag")
    if _flag and _flag.get("type") == "Title Match":
        st.session_state.sim_selected_id = _flag["params"].get("selected_id")
        st.session_state.sim_results     = None
        st.session_state.run_again_flag  = None
        st.session_state.run_sim_now     = True

    # Search pool covers all 6 platforms so competitor titles appear in autocomplete
    search_pool = deduplicate_titles(load_all_platforms_titles())

    # ── Controls ──
    ctrl1, ctrl2, ctrl3 = st.columns([5, 2, 2])
    with ctrl1:
        search_query = st.text_input(
            "Search for a title",
            placeholder="Start typing a title name...",
            key="sim_search",
            label_visibility="collapsed",
        )
        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            find_clicked = st.button("Find", key="sim_find", use_container_width=True)
        with btn_col2:
            surprise_clicked = st.button("Surprise Me", key="sim_surprise", use_container_width=True)
    with ctrl2:
        scope = st.radio("Scope", ["Merged", "All Platforms"], key="sim_scope", horizontal=True)
        scope_key = "merged" if scope == "Merged" else "all_platforms"
    with ctrl3:
        n_results    = st.slider("Results", 5, 20, 10, key="sim_count")
        min_imdb_sim = st.slider(
            "Min IMDb", 0.0, 9.0, float(SIMILARITY_MIN_IMDB), 0.5, key="sim_min_imdb"
        )

    # Resolve current selection
    current_selected_id = st.session_state.get("sim_selected_id")

    # ── Surprise Me ──
    if surprise_clicked:
        candidates = titles[
            (titles["quality_score"] >= 7.0) & (titles["imdb_votes"].fillna(0) >= 5000)
        ]
        if candidates.empty:
            candidates = titles[titles["quality_score"] >= 6.0]
        if not candidates.empty:
            pick = candidates.sample(1).iloc[0]
            st.session_state.sim_selected_id = pick["id"]
            st.session_state.sim_results     = None
            st.session_state.discovery_detail_id = None
            st.session_state.run_sim_now     = True
            st.rerun()

    # ── Find button — auto-select first match across all platforms ──
    if find_clicked and search_query:
        first_match = search_pool[
            search_pool["title"].str.contains(search_query.strip(), case=False, na=False)
        ]
        if not first_match.empty:
            pick = first_match.iloc[0]
            st.session_state.sim_selected_id = pick["id"]
            st.session_state.sim_results     = None
            st.session_state.discovery_detail_id = None
            st.session_state.run_sim_now     = True
            st.rerun()
        else:
            st.warning(f'No titles found matching "{search_query}".')

    # ── Autocomplete (searches all platforms) ──
    if search_query:
        matches = search_pool[
            search_pool["title"].str.contains(search_query, case=False, na=False)
        ].head(8)
        if not matches.empty:
            for _, m_row in matches.iterrows():
                mid       = m_row["id"]
                m_title   = m_row.get("title", "")
                m_year    = int(m_row["release_year"]) if pd.notna(m_row.get("release_year")) else ""
                m_plats   = m_row.get("platforms") or ([m_row.get("platform", "")])
                m_poster  = _get_poster_url(mid, enriched_df)
                is_sel    = mid == current_selected_id

                ac1, ac2, ac3, ac4 = st.columns([1, 6, 3, 1])
                with ac1:
                    st.markdown(
                        _poster_html(m_poster, m_title, m_plats, width=40, height=56),
                        unsafe_allow_html=True,
                    )
                with ac2:
                    st.markdown(
                        f'<div style="padding:6px 0;">'
                        f'<div style="font-size:0.9em;font-weight:{"700" if is_sel else "500"};'
                        f'color:{CARD_ACCENT if is_sel else CARD_TEXT};">{m_title}</div>'
                        f'<div style="font-size:0.78em;color:{CARD_TEXT_MUTED};">{m_year}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with ac3:
                    st.markdown(
                        f'<div style="padding:6px 0;">{platform_badges_html(m_plats)}</div>',
                        unsafe_allow_html=True,
                    )
                with ac4:
                    btn_label = "✓" if is_sel else "Select"
                    if st.button(btn_label, key=f"sel_{mid}", use_container_width=True):
                        st.session_state.sim_selected_id = mid
                        st.session_state.sim_results     = None
                        st.session_state.discovery_detail_id = None
                        st.session_state.run_sim_now     = True
                        st.rerun()

    # ── Selected title info + Find Similar ──
    if current_selected_id:
        sel_rows = titles[titles["id"] == current_selected_id]
        if sel_rows.empty:
            # Fallback: title may be from a competitor platform not in current scope
            sel_rows = search_pool[search_pool["id"] == current_selected_id]
        if not sel_rows.empty:
            sel_title_row  = sel_rows.iloc[0].to_dict()
            sel_title_name = sel_title_row.get("title", "")

            run_now = st.session_state.pop("run_sim_now", False)
            if run_now:
                if scope_key == "merged":
                    sim_pool = deduplicate_titles(load_merged_titles())
                else:
                    sim_pool = deduplicate_titles(load_all_platforms_titles())

                results = get_similar_with_explanation(
                    current_selected_id, sim_pool, sim_df,
                    principals_df=principals_df, enriched_df=enriched_df,
                    top_k=n_results, min_imdb=min_imdb_sim, min_votes=1000,
                )
                st.session_state.sim_results = results

                # History entry
                hist_posters = []
                for r in results[:3]:
                    url = _get_poster_url(r.get("similar_id") or r.get("id", ""), enriched_df)
                    if url:
                        hist_posters.append(url)

                _add_to_history({
                    "type": "Title Match",
                    "query": sel_title_name,
                    "count": len(results),
                    "count_runs": 1,
                    "results": [
                        {
                            "title": r.get("title"),
                            "year": r.get("release_year"),
                            "platforms": r.get("platforms", []),
                            "imdb_score": r.get("imdb_score"),
                            "id": r.get("similar_id") or r.get("id", ""),
                        }
                        for r in results[:5]
                    ],
                    "result_poster_urls": hist_posters,
                    "pinned": False,
                    "params": {
                        "selected_id": current_selected_id,
                        "scope_key": scope_key,
                        "n_results": n_results,
                        "min_imdb": min_imdb_sim,
                    },
                })

    # ── Results ──
    results = st.session_state.get("sim_results") or []
    if results:
        n_res = len(results)
        st.markdown(
            styled_banner_html(
                "✓",
                f"Found {n_res} title{'s' if n_res != 1 else ''} similar to "
                f"<strong>{sel_title_row.get('title','')}</strong>",
                bg="rgba(46,204,113,0.1)", border_color="#2ecc71",
            ),
            unsafe_allow_html=True,
        )

        for r in results:
            sim_score = r.get("similarity_score", 0)
            rid       = r.get("similar_id") or r.get("id", "")
            is_open   = st.session_state.discovery_detail_id == rid

            card_col, btn_col = st.columns([11, 1])
            with card_col:
                _render_rec_card(r, _sim_score_badge_html(sim_score))
            with btn_col:
                if st.button(
                    "▼" if is_open else "Details",
                    key=f"detail_sim_{rid}",
                    help="View full details",
                    use_container_width=True,
                ):
                    st.session_state.discovery_detail_id = None if is_open else rid
                    st.rerun()

            with st.expander("Why similar?", expanded=False):
                _render_why_similar(r, sim_score, sel_title_row if current_selected_id else {})

            # Inline detail panel — renders immediately below this specific card
            if st.session_state.discovery_detail_id == rid:
                _render_title_detail(rid)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MOOD BOARD
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown(
        section_header_html(
            "Mood Board",
            "Pick the feeling you want and we'll find titles that match the vibe — not just the genre.",
        ),
        unsafe_allow_html=True,
    )

    # Handle "Run Again" from History tab
    _flag = st.session_state.get("run_again_flag")
    if _flag and _flag.get("type") == "Mood Board":
        st.session_state.selected_moods = _flag["params"].get("selected_moods", [])
        st.session_state.run_again_flag = None
        st.session_state.run_mood_now   = True

    selected_moods = list(st.session_state.selected_moods)

    # ── CSS injection for styled mood tiles ──
    st.markdown("""<style>
div[data-testid="stVerticalBlock"] div[data-testid="column"] > div > div > div > button {
    background: #1E1E2E !important;
    border: 1px solid #333 !important;
    border-radius: 10px !important;
    padding: 14px 10px !important;
    min-height: 80px !important;
    transition: border-color 0.15s, background 0.15s !important;
    white-space: normal !important;
    line-height: 1.4 !important;
    font-size: 0.88em !important;
}
div[data-testid="stVerticalBlock"] div[data-testid="column"] > div > div > div > button:hover {
    border-color: #FFD700 !important;
    background: rgba(255,215,0,0.05) !important;
}
</style>""", unsafe_allow_html=True)

    # ── 4×4 mood tile grid ──
    for row_i in range(4):
        cols = st.columns(4)
        for col_i in range(4):
            tile_i = row_i * 4 + col_i
            if tile_i >= len(MOOD_TILES):
                break
            tile      = MOOD_TILES[tile_i]
            is_sel    = tile["label"] in selected_moods
            lbl       = f'{"✓ " if is_sel else ""}{tile["label"]}'
            with cols[col_i]:
                if st.button(lbl, key=f"mood_{tile_i}", use_container_width=True):
                    if is_sel:
                        st.session_state.selected_moods = [
                            m for m in selected_moods if m != tile["label"]
                        ]
                    else:
                        st.session_state.selected_moods = selected_moods + [tile["label"]]
                    st.rerun()

    # Selected mood pills summary
    if selected_moods:
        pills = " ".join(
            f'<span style="background:rgba(255,215,0,0.15);border:1px solid {CARD_ACCENT};'
            f'color:{CARD_ACCENT};padding:3px 10px;border-radius:12px;font-size:0.82em;">'
            f'{MOOD_TILES[[t["label"] for t in MOOD_TILES].index(m)]["emoji"]} {m}</span>'
            for m in selected_moods if m in [t["label"] for t in MOOD_TILES]
        )
        st.markdown(f'<div style="margin:8px 0;">{pills}</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div style="color:{CARD_TEXT_MUTED};font-size:0.85em;margin:8px 0;">'
            f'No moods selected yet — pick at least one tile above.</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Controls ──
    mc1, mc2 = st.columns(2)
    with mc1:
        mood_type = st.radio(
            "Content type", ["Both", "Movie", "Show"],
            key="mood_type", horizontal=True,
        )
    with mc2:
        mood_scope = st.radio(
            "Platform scope", ["Merged", "All Platforms"],
            key="mood_scope", horizontal=True,
        )

    # Clear stale results when no moods are selected
    if not selected_moods:
        st.session_state.mood_results = None

    run_mood_now = st.session_state.pop("run_mood_now", False)
    if st.button(
        "Find My Match", key="mood_go", type="primary", disabled=not selected_moods
    ) or run_mood_now:
        if mood_scope == "Merged":
            source_m = deduplicate_titles(load_merged_titles())
        else:
            source_m = deduplicate_titles(load_all_platforms_titles())

        mood_results = mood_board_recommendations(
            source_m, selected_moods, content_type=mood_type,
            top_k=20, enriched_df=enriched_df, min_imdb=6.0, min_votes=1000,
        )

        if mood_results.empty:
            st.session_state.mood_results = None
            st.warning("No matching titles found. Try selecting different moods or lowering filters.")
        else:
            st.session_state.mood_results = mood_results
            mood_query = ", ".join(selected_moods[:3]) + ("..." if len(selected_moods) > 3 else "")

            hist_posters_m = []
            for _, mood_row in mood_results.iterrows():
                url = _get_poster_url(mood_row.get("id", ""), enriched_df)
                if url and len(hist_posters_m) < 3:
                    hist_posters_m.append(url)

            _add_to_history({
                "type": "Mood Board",
                "query": mood_query,
                "count": len(mood_results),
                "count_runs": 1,
                "results": [
                    {
                        "title": r.get("title"),
                        "year": r.get("release_year"),
                        "platforms": r.get("platforms", []),
                        "imdb_score": r.get("imdb_score"),
                        "id": r.get("id", ""),
                    }
                    for _, r in mood_results.head(5).iterrows()
                ],
                "result_poster_urls": hist_posters_m,
                "pinned": False,
                "params": {
                    "selected_moods": list(selected_moods),
                    "mood_type": mood_type,
                    "mood_scope": mood_scope,
                },
            })

    # ── Render mood results (from session state — persists across reruns) ──
    _mood_results = st.session_state.get("mood_results")
    if _mood_results is not None and not _mood_results.empty:
        _primary_mood   = _mood_results[_mood_results["mood_match_pct"] >= 0.5]
        _secondary_mood = _mood_results[_mood_results["mood_match_pct"] <  0.5]
        n_m = len(_primary_mood)
        st.markdown(
            styled_banner_html(
                "✓",
                f"Showing {n_m} strong mood matches (min IMDb 6.0, min 1,000 votes)",
                bg="rgba(46,204,113,0.1)", border_color="#2ecc71",
            ),
            unsafe_allow_html=True,
        )

        def _render_mood_card_rows(rows_df):
            for _, mood_row in rows_df.iterrows():
                row_dict = mood_row.to_dict()
                mid      = row_dict.get("id", "")
                mood_pct = row_dict.get("mood_match_pct", 0)
                is_open  = st.session_state.discovery_detail_id == mid

                card_col, btn_col = st.columns([11, 1])
                with card_col:
                    _render_rec_card(row_dict, _mood_score_badge_html(mood_pct))
                with btn_col:
                    if st.button(
                        "▼" if is_open else "Details",
                        key=f"detail_mood_{mid}",
                        help="View full details",
                        use_container_width=True,
                    ):
                        st.session_state.discovery_detail_id = None if is_open else mid
                        st.rerun()

                # Matched mood tag pills
                matched = row_dict.get("matched_moods", [])
                if matched:
                    mood_lookup = {t["label"]: t["emoji"] for t in MOOD_TILES}
                    tag_pills = " ".join(
                        f'<span style="background:#2a2a3e;color:{CARD_TEXT_MUTED};'
                        f'padding:2px 8px;border-radius:10px;font-size:0.72em;">'
                        f'{mood_lookup.get(lbl,"")}{" " if mood_lookup.get(lbl,"") else ""}{lbl}</span>'
                        for lbl in matched[:4]
                    )
                    st.markdown(
                        f'<div style="margin:2px 0 8px 0;">{tag_pills}</div>',
                        unsafe_allow_html=True,
                    )

                # Inline detail panel
                if st.session_state.discovery_detail_id == mid:
                    _render_title_detail(mid)

        _render_mood_card_rows(_primary_mood)

        if not _secondary_mood.empty:
            with st.expander(
                f"Also worth considering — {len(_secondary_mood)} looser matches (<50% mood fit)",
                expanded=False,
            ):
                _render_mood_card_rows(_secondary_mood)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — VIBE SEARCH
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown(
        section_header_html(
            "Vibe Search",
            "Describe the feeling, theme, or atmosphere you're after — in your own words.",
        ),
        unsafe_allow_html=True,
    )

    # Handle "Run Again" from History tab
    _flag = st.session_state.get("run_again_flag")
    if _flag and _flag.get("type") == "Vibe Search":
        st.session_state.vibe_query = _flag["params"].get("query", "")
        st.session_state.vibe_imdb  = _flag["params"].get("min_imdb", 6.5)
        st.session_state.vibe_year  = tuple(_flag["params"].get("year_range", (1970, 2024)))
        st.session_state.vibe_scope = _flag["params"].get("scope_key", "Merged")
        st.session_state.run_again_flag = None
        st.session_state.run_vibe_now   = True

    vibe_query = st.text_area(
        "Describe the vibe",
        placeholder="e.g. A slow-burn psychological thriller with an unreliable narrator...",
        height=100,
        key="vibe_query",
        label_visibility="collapsed",
    )

    with st.expander("Filters", expanded=True):
        vf1, vf2, vf3 = st.columns(3)
        with vf1:
            vibe_min_imdb = st.slider(
                "Min IMDb", 0.0, 9.0, 6.5, 0.5, key="vibe_imdb",
                help="Titles below this score are excluded. Default 6.5 keeps quality high.",
            )
        with vf2:
            vibe_year = st.slider("Year range", 1950, 2024, (1970, 2024), key="vibe_year")
        with vf3:
            vibe_scope = st.radio(
                "Scope", ["Merged", "All Platforms"], key="vibe_scope", horizontal=True
            )

    run_vibe_now = st.session_state.pop("run_vibe_now", False)
    vibe_active  = bool(vibe_query and vibe_query.strip())

    if st.button(
        "Search Vibes", key="vibe_go", type="primary", disabled=not vibe_active
    ) or (run_vibe_now and vibe_active):

        scope_key_v = "merged" if vibe_scope == "Merged" else "all_platforms"
        if scope_key_v == "merged":
            source_v = deduplicate_titles(load_merged_titles())
        else:
            source_v = deduplicate_titles(load_all_platforms_titles())

        results_v, detected_signals = vibe_search(
            vibe_query,
            source_v,
            genome_vectors=genome_vectors,
            genome_id_map=genome_id_map,
            enriched_df=enriched_df,
            top_k=15,
            min_imdb=vibe_min_imdb,
            year_range=vibe_year,
            min_votes=5000,
        )

        if results_v is None or (hasattr(results_v, "empty") and results_v.empty):
            st.session_state.vibe_results = None
            st.session_state.vibe_signals = []
            st.warning("No results found. Try a different query or adjusting the filters.")
        else:
            st.session_state.vibe_results = results_v
            st.session_state.vibe_signals = detected_signals

            hist_posters_v = []
            for _, vrow in results_v.iterrows():
                url = _get_poster_url(vrow.get("id", ""), enriched_df)
                if url and len(hist_posters_v) < 3:
                    hist_posters_v.append(url)

            q_trunc = vibe_query[:60] + ("..." if len(vibe_query) > 60 else "")
            _add_to_history({
                "type": "Vibe Search",
                "query": q_trunc,
                "count": len(results_v),
                "count_runs": 1,
                "results": [
                    {
                        "title": r.get("title"),
                        "year": r.get("release_year"),
                        "platforms": r.get("platforms", []),
                        "imdb_score": r.get("imdb_score"),
                        "id": r.get("id", ""),
                    }
                    for _, r in results_v.head(5).iterrows()
                ],
                "result_poster_urls": hist_posters_v,
                "pinned": False,
                "params": {
                    "query": vibe_query,
                    "scope_key": scope_key_v,
                    "min_imdb": vibe_min_imdb,
                    "year_range": list(vibe_year),
                },
            })

    # ── Render vibe results (from session state — persists across reruns) ──
    _vibe_results  = st.session_state.get("vibe_results")
    _vibe_signals  = st.session_state.get("vibe_signals", [])
    if _vibe_results is not None and not _vibe_results.empty:
        # Detected signals banner
        if _vibe_signals:
            pills = " ".join(
                f'<span style="background:#2a2a3e;color:{CARD_TEXT_MUTED};padding:3px 9px;'
                f'border-radius:10px;font-size:0.78em;margin-right:3px;">{s}</span>'
                for s in _vibe_signals
            )
            st.markdown(
                f'<div style="margin:8px 0;">'
                f'<span style="color:{CARD_TEXT_MUTED};font-size:0.78em;'
                f'text-transform:uppercase;letter-spacing:0.04em;margin-right:8px;">Themes detected</span>'
                f'{pills}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="color:{CARD_TEXT_MUTED};font-size:0.85em;margin:6px 0;">'
                f'Searching by description and tone similarity.</div>',
                unsafe_allow_html=True,
            )

        n_v = len(_vibe_results)
        st.markdown(
            styled_banner_html(
                "✓",
                f"Showing top {n_v} quality-filtered matches",
                bg="rgba(46,204,113,0.1)", border_color="#2ecc71",
            ),
            unsafe_allow_html=True,
        )

        for i_v, (_, vrow) in enumerate(_vibe_results.iterrows()):
            vrow_dict = vrow.to_dict()
            vid       = vrow_dict.get("id", "")
            vibe_sc   = vrow_dict.get("vibe_score", 0.0)
            is_open   = st.session_state.discovery_detail_id == vid

            card_col, btn_col = st.columns([11, 1])
            with card_col:
                _render_rec_card(vrow_dict, _vibe_label_badge_html(vibe_sc))
            with btn_col:
                if st.button(
                    "▼" if is_open else "Details",
                    key=f"detail_vibe_{vid}_{i_v}",
                    help="View full details",
                    use_container_width=True,
                ):
                    st.session_state.discovery_detail_id = None if is_open else vid
                    st.rerun()

            # Matched genome tags (if any)
            top_tags = vrow_dict.get("top_tags")
            if top_tags is not None and isinstance(top_tags, (list, np.ndarray)) and len(top_tags) > 0:
                tags_list = list(top_tags)[:5]
                tag_pills = " ".join(
                    f'<span style="background:#2a2a3e;color:{CARD_TEXT_MUTED};'
                    f'padding:2px 7px;border-radius:10px;font-size:0.7em;">{t}</span>'
                    for t in tags_list
                )
                st.markdown(
                    f'<div style="margin:2px 0 8px 0;">{tag_pills}</div>',
                    unsafe_allow_html=True,
                )

            # Inline detail panel
            if st.session_state.discovery_detail_id == vid:
                _render_title_detail(vid)

    # How Vibe Search works
    with st.expander("How Vibe Search works", expanded=False):
        st.markdown(
            f'<div style="background:{CARD_BG};border:1px solid {CARD_BORDER};'
            f'border-radius:8px;padding:16px;">'
            f'<div style="font-weight:600;color:{CARD_TEXT};margin-bottom:12px;">Hybrid Scoring</div>',
            unsafe_allow_html=True,
        )
        components = [
            ("Description match",   35, "#4FC3F7"),
            ("Genre + keywords",    25, "#81C784"),
            ("MovieLens genome",    15, "#FFB74D"),
            ("Bayesian quality",    15, "#CE93D8"),
            ("Awards boost",        10, "#F48FB1"),
        ]
        for label, pct, color in components:
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:7px;">'
                f'<div style="width:150px;font-size:0.82em;color:{CARD_TEXT_MUTED};">{label}</div>'
                f'<div style="flex:1;background:#2a2a3e;border-radius:4px;height:8px;">'
                f'<div style="width:{pct}%;height:100%;background:{color};border-radius:4px;"></div>'
                f'</div>'
                f'<div style="font-size:0.82em;color:{CARD_TEXT};font-weight:600;width:35px;">{pct}%</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            f'<div style="color:{CARD_TEXT_MUTED};font-size:0.78em;margin-top:8px;">'
            f'Titles without MovieLens data redistribute that 15% to description (60%) and genre (40%).</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — HISTORY
# ══════════════════════════════════════════════════════════════════════════════

_HIST_ICONS   = {"Title Match": "", "Mood Board": "", "Vibe Search": ""}
_HIST_COLORS  = {
    "Title Match": "#4FC3F7",
    "Mood Board":  "#CE93D8",
    "Vibe Search": "#81C784",
}

with tab_history:
    st.markdown(
        section_header_html(
            "Recent Searches",
            "Your discovery session log — revisit, re-run, or pin searches to keep them.",
        ),
        unsafe_allow_html=True,
    )

    history = st.session_state.rec_history

    if not history:
        st.markdown(
            f'<div style="color:{CARD_TEXT_MUTED};font-size:0.9em;margin:16px 0;">'
            f'No searches yet. Try one of the tabs above!</div>',
            unsafe_allow_html=True,
        )
    else:
        # Controls
        hctrl1, hctrl2 = st.columns([4, 2])
        with hctrl1:
            group_by = st.radio(
                "View", ["Chronological", "By Tab Type"],
                key="hist_group", horizontal=True, label_visibility="collapsed",
            )
        with hctrl2:
            if st.button("Clear All", key="hist_clear_all"):
                st.session_state.rec_history = []
                st.rerun()

        # Build display order
        if group_by == "By Tab Type":
            display_groups = {
                t: [e for e in history if e.get("type") == t]
                for t in ["Title Match", "Mood Board", "Vibe Search"]
            }
        else:
            display_groups = {"_all": history}

        for group_key, group_entries in display_groups.items():
            if not group_entries:
                continue

            if group_key != "_all":
                tab_color = _HIST_COLORS.get(group_key, "#888")
                st.markdown(
                    section_header_html(
                        group_key,
                        f'{len(group_entries)} search{"es" if len(group_entries)>1 else ""}',
                        accent_color=tab_color,
                    ),
                    unsafe_allow_html=True,
                )

            for entry in group_entries:
                eid        = entry.get("_id", id(entry))
                tab_type   = entry.get("type", "Unknown")
                query      = entry.get("query", "")
                run_count  = entry.get("count_runs", 1)
                result_cnt = entry.get("count", 0)
                posters    = entry.get("result_poster_urls", [])
                results_list = entry.get("results", [])
                is_pinned  = entry.get("pinned", False)
                tab_color  = _HIST_COLORS.get(tab_type, "#888")
                icon       = _HIST_ICONS.get(tab_type, "")

                with st.container(border=True):
                    hdr_col, pin_col, rm_col = st.columns([10, 1, 1])

                    with hdr_col:
                        run_badge = (
                            f' <span style="background:#333;color:{CARD_TEXT_MUTED};'
                            f'padding:1px 7px;border-radius:8px;font-size:0.72em;">×{run_count}</span>'
                            if run_count > 1 else ""
                        )
                        pin_star = "Pinned · " if is_pinned else ""
                        st.markdown(
                            f'<span style="color:{CARD_TEXT_MUTED};font-size:0.75em;">{pin_star}</span>'
                            f'<span style="background:{tab_color};color:#111;padding:2px 8px;'
                            f'border-radius:8px;font-size:0.75em;font-weight:700;margin-right:6px;">'
                            f'{tab_type}</span>'
                            f'<span style="color:{CARD_TEXT};font-size:0.88em;">{query}</span>'
                            f'{run_badge}',
                            unsafe_allow_html=True,
                        )

                    with pin_col:
                        pin_lbl = "Pin" if not is_pinned else "Unpin"
                        if st.button(pin_lbl, key=f"hist_pin_{eid}", help="Pin / unpin this search"):
                            for e in st.session_state.rec_history:
                                if e.get("_id") == eid:
                                    e["pinned"] = not e.get("pinned", False)
                                    break
                            st.rerun()

                    with rm_col:
                        if st.button("✕", key=f"hist_rm_{eid}", help="Remove this entry"):
                            st.session_state.rec_history = [
                                e for e in st.session_state.rec_history
                                if e.get("_id") != eid
                            ]
                            st.rerun()

                    # Thumbnail strip
                    if posters:
                        thumb_cols = st.columns(len(posters) + 1)
                        for pi, purl in enumerate(posters[:3]):
                            with thumb_cols[pi]:
                                st.markdown(
                                    f'<img src="{purl}" height="42" style="border-radius:4px;'
                                    f'object-fit:cover;">',
                                    unsafe_allow_html=True,
                                )

                    # Preview count
                    st.markdown(
                        f'<div style="color:{CARD_TEXT_MUTED};font-size:0.78em;margin:4px 0;">'
                        f'{result_cnt} result{"s" if result_cnt != 1 else ""}</div>',
                        unsafe_allow_html=True,
                    )

                    # See Results expander
                    if results_list:
                        with st.expander(f"See results ({min(len(results_list), result_cnt)})", expanded=False):
                            for res in results_list:
                                if isinstance(res, dict):
                                    t_title = res.get("title", "?")
                                    t_year  = res.get("year") or res.get("release_year")
                                    t_imdb  = res.get("imdb_score")
                                    t_plats = res.get("platforms", [])
                                    yr_str  = str(int(t_year)) if t_year and pd.notna(t_year) else ""
                                    imdb_str = f"{t_imdb:.1f}" if t_imdb and pd.notna(t_imdb) else "N/A"
                                    plat_html = platform_badges_html(t_plats)
                                    st.markdown(
                                        f'<div style="display:flex;justify-content:space-between;'
                                        f'align-items:center;padding:5px 0;'
                                        f'border-bottom:1px solid {CARD_BORDER};">'
                                        f'<span style="color:{CARD_TEXT};font-size:0.85em;font-weight:600;">'
                                        f'{t_title} '
                                        f'<span style="color:{CARD_TEXT_MUTED};font-weight:400;'
                                        f'font-size:0.88em;">({yr_str})</span></span>'
                                        f'<span style="display:flex;align-items:center;gap:6px;">'
                                        f'{plat_html}'
                                        f'<span style="color:{CARD_TEXT_MUTED};font-size:0.78em;">'
                                        f'IMDb {imdb_str}</span></span></div>',
                                        unsafe_allow_html=True,
                                    )
                                else:
                                    # Legacy string format
                                    st.markdown(
                                        f'<span style="color:{CARD_TEXT};font-size:0.85em;">{res}</span>',
                                        unsafe_allow_html=True,
                                    )

                    # Run Again button
                    if entry.get("params"):
                        if st.button(
                            "▶ Run Again",
                            key=f"hist_run_{eid}",
                            help=f"Re-run this {tab_type} search",
                        ):
                            st.session_state.run_again_flag = {
                                "type": tab_type,
                                "params": entry["params"],
                            }
                            st.rerun()

# ─── Footer ───────────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    '<div style="color:#555;font-size:0.8em;text-align:center;padding:8px 0 16px;">'
    'Hypothetical merger for academic analysis. '
    'Data is a snapshot (mid-2023). '
    'All insights are illustrative, not prescriptive. '
    'As of Feb 26, 2026, Netflix withdrew from this acquisition.'
    '</div>',
    unsafe_allow_html=True,
)
