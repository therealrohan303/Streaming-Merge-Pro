"""Page 4: Discovery Engine — Full recommendation toolkit.

Four distinct entry points:
  Tab 1: Title Match  — smart autocomplete + why-similar explainer + detail panel
  Tab 2: Mood Board   — 16 mood tiles mapped to cinematic signals
  Tab 3: Vibe Search  — NLP-powered semantic + keyword + genome hybrid
  Tab 4: History      — recent searches log
"""

import pandas as pd
import streamlit as st
from collections import defaultdict

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
from src.ui.badges import platform_badges_html, section_header_html, styled_banner_html
from src.ui.filters import apply_filters, render_sidebar_filters
from src.ui.session import init_session_state

st.set_page_config(page_title="Discovery Engine", page_icon="🔎", layout="wide")
init_session_state()

st.markdown(
    section_header_html(
        "Discovery Engine",
        "Your full recommendation toolkit — find your next favorite title through four distinct approaches.",
        font_size="2em",
    ),
    unsafe_allow_html=True,
)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
raw_df = get_titles_for_view(st.session_state.get("platform_view", "merged"))
filters = render_sidebar_filters(raw_df)
titles_all = get_titles_for_view(filters["platform_view"])
titles = apply_filters(titles_all, filters)
titles = deduplicate_titles(titles)
titles["quality_score"] = compute_quality_score(titles)

st.sidebar.metric("Titles Available", f"{len(titles):,}")

# Load enrichment data (cached)
enriched_df = load_enriched_titles()
enriched_df = deduplicate_titles(enriched_df)
principals_df = load_imdb_principals()
sim_df = load_similarity_data()
genome_vectors, genome_id_map = load_genome_vectors()

# ─── Session State ────────────────────────────────────────────────────────────
if "rec_history" not in st.session_state:
    st.session_state.rec_history = []
if "hist_counter" not in st.session_state:
    st.session_state.hist_counter = 0
if "sim_selected_id" not in st.session_state:
    st.session_state.sim_selected_id = None
if "sim_results" not in st.session_state:
    st.session_state.sim_results = None
if "selected_moods" not in st.session_state:
    st.session_state.selected_moods = []
if "discovery_detail_id" not in st.session_state:
    st.session_state.discovery_detail_id = None


# ─── Shared Helpers ───────────────────────────────────────────────────────────

def _get_poster_url(title_id: str, enr_df: pd.DataFrame):
    """Return TMDB poster URL for a title id, or None."""
    if enr_df is None or enr_df.empty:
        return None
    row = enr_df[enr_df["id"] == title_id]
    if row.empty:
        return None
    url = row.iloc[0].get("poster_url")
    if pd.notna(url) and url and str(url) != "nan":
        return str(url)
    return None


def _poster_html(url, title: str, platform_keys, width=90, height=130) -> str:
    """Return HTML for poster image or styled placeholder."""
    if isinstance(platform_keys, (list, tuple)) and platform_keys:
        color = PLATFORMS.get(platform_keys[0], {}).get("color", "#555")
    elif isinstance(platform_keys, str) and platform_keys:
        color = PLATFORMS.get(platform_keys, {}).get("color", "#555")
    else:
        color = "#555"
    initial = (title or "?")[0].upper()

    if url:
        return (
            f'<img src="{url}" '
            f'style="width:{width}px;height:{height}px;object-fit:cover;border-radius:6px;'
            f'flex-shrink:0;" '
            f'onerror="this.outerHTML=\'<div style=&quot;width:{width}px;height:{height}px;'
            f'border-radius:6px;background:{color};display:flex;align-items:center;'
            f'justify-content:center;font-size:2em;font-weight:700;color:rgba(255,255,255,0.85);'
            f'flex-shrink:0;&quot;>{initial}</div>\'" />'
        )
    return (
        f'<div style="width:{width}px;height:{height}px;border-radius:6px;background:{color};'
        f'display:flex;align-items:center;justify-content:center;font-size:2em;font-weight:700;'
        f'color:rgba(255,255,255,0.85);flex-shrink:0;">{initial}</div>'
    )


def _genre_pills_html(genres, max_n=4) -> str:
    if not isinstance(genres, (list, tuple)):
        return ""
    return " ".join(
        f'<span style="background:{CARD_BG};border:1px solid {CARD_BORDER};'
        f'padding:2px 6px;border-radius:3px;font-size:0.7rem;margin-right:3px;">{g}</span>'
        for g in genres[:max_n]
    )


def _sim_score_badge_html(score) -> str:
    if score >= 0.75:
        bg, fg = "rgba(46,204,113,0.18)", "#2ecc71"
    elif score >= 0.60:
        bg, fg = "rgba(243,156,18,0.18)", "#f39c12"
    else:
        bg, fg = "rgba(136,136,136,0.18)", "#888"
    return (
        f'<span style="background:{bg};color:{fg};padding:3px 10px;border-radius:12px;'
        f'font-size:0.8em;font-weight:700;white-space:nowrap;">{score:.0%} match</span>'
    )


def _mood_score_badge_html(pct) -> str:
    if pct >= 0.75:
        bg, fg = "rgba(46,204,113,0.18)", "#2ecc71"
    elif pct >= 0.40:
        bg, fg = "rgba(243,156,18,0.18)", "#f39c12"
    else:
        bg, fg = "rgba(136,136,136,0.18)", "#888"
    return (
        f'<span style="background:{bg};color:{fg};padding:3px 10px;border-radius:12px;'
        f'font-size:0.8em;font-weight:700;white-space:nowrap;">Mood {pct:.0%}</span>'
    )


def _vibe_label_badge_html(rank_pct) -> str:
    if rank_pct < 0.20:
        label, fg, bg = "Strong match", "#2ecc71", "rgba(46,204,113,0.18)"
    elif rank_pct < 0.60:
        label, fg, bg = "Good match", "#f39c12", "rgba(243,156,18,0.18)"
    else:
        label, fg, bg = "Partial match", "#888", "rgba(136,136,136,0.18)"
    return (
        f'<span style="background:{bg};color:{fg};padding:3px 10px;border-radius:12px;'
        f'font-size:0.8em;font-weight:700;white-space:nowrap;">{label}</span>'
    )


def _render_rec_card(row: dict, score_badge_html: str = "") -> None:
    """Render a unified recommendation card: poster left, info right."""
    title = row.get("title", "Unknown")
    year = int(row["release_year"]) if pd.notna(row.get("release_year")) else ""
    imdb = row.get("imdb_score")
    imdb_str = f"{imdb:.1f}" if pd.notna(imdb) else "N/A"
    votes_str = format_votes(row.get("imdb_votes"))
    platforms = row.get("platforms", row.get("platform", ""))
    genres = row.get("genres", [])

    tid = row.get("id") or row.get("similar_id", "")
    poster_url = _get_poster_url(tid, enriched_df)
    poster_html = _poster_html(poster_url, title, platforms, width=90, height=130)
    genre_pills = _genre_pills_html(genres)
    plat_badges = platform_badges_html(platforms)

    score_html = (
        f'<div style="float:right;margin-left:8px;">{score_badge_html}</div>'
        if score_badge_html
        else ""
    )

    st.markdown(
        f'<div style="display:flex;gap:12px;background:{CARD_BG};border:1px solid {CARD_BORDER};'
        f'border-radius:8px;padding:12px;margin-bottom:4px;align-items:flex-start;">'
        f'{poster_html}'
        f'<div style="flex:1;min-width:0;">'
        f'<div style="overflow:hidden;">'
        f'{score_html}'
        f'<span style="font-size:1rem;font-weight:700;color:{CARD_TEXT};">{title}</span>'
        f'<span style="color:{CARD_TEXT_MUTED};font-size:0.85em;margin-left:6px;">{year}</span>'
        f'</div>'
        f'<div style="margin-top:5px;">{plat_badges}</div>'
        f'<div style="margin-top:5px;font-size:0.85em;color:{CARD_TEXT};">'
        f'IMDb <strong>{imdb_str}</strong>'
        f'<span style="color:{CARD_TEXT_MUTED};margin-left:4px;">({votes_str} votes)</span>'
        f'</div>'
        f'<div style="margin-top:5px;">{genre_pills}</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─── Detail Panel Helpers ─────────────────────────────────────────────────────

def _meta_cell(label: str, value: str) -> str:
    """Render a metadata label+value cell — same pattern as Platform DNA."""
    return (
        f'<div style="padding:8px 0;">'
        f'<div style="color:{CARD_TEXT_MUTED};font-size:0.72em;text-transform:uppercase;'
        f'letter-spacing:0.04em;margin-bottom:3px;">{label}</div>'
        f'<div style="font-weight:600;color:{CARD_TEXT};font-size:0.95em;">{value}</div>'
        f'</div>'
    )


def _render_title_detail(title_id: str) -> None:
    """Full title detail panel — same format as Platform DNA / Explore Catalog."""
    all_titles = load_all_platforms_titles()
    sel_rows = all_titles[all_titles["id"] == title_id]
    if sel_rows.empty:
        # Fallback: try enriched df
        sel_rows = enriched_df[enriched_df["id"] == title_id]
    if sel_rows.empty:
        st.warning("Title details not found.")
        return

    sel_rows = deduplicate_titles(sel_rows.copy()) if "platform" in sel_rows.columns else sel_rows.copy()
    sel_rows["quality_score"] = compute_quality_score(sel_rows)
    sel = sel_rows.iloc[0]

    _enr_row = enriched_df[enriched_df["id"] == title_id] if not enriched_df.empty else pd.DataFrame()
    _has_enr = not _enr_row.empty
    _enr = _enr_row.iloc[0] if _has_enr else None

    # Close button above the container
    if st.button("✕ Close Details", key=f"close_detail_{title_id}"):
        st.session_state.discovery_detail_id = None
        st.rerun()

    with st.container(border=True):
        # Poster
        if _has_enr and "poster_url" in _enr_row.columns:
            _poster = _enr.get("poster_url")
            if _poster and str(_poster) != "nan":
                st.image(str(_poster), width=180)

        # Title + award badge
        st.subheader(sel["title"])
        if _has_enr and "award_wins" in _enr_row.columns:
            _aw = _enr.get("award_wins", 0)
            _an = _enr.get("award_noms", 0)
            if _aw and _aw > 0:
                _noms_str = f", {int(_an)} nominations" if _an and _an > 0 else ""
                st.markdown(
                    f'<span style="background:rgba(46,204,113,0.15);color:#2ecc71;'
                    f'border:1px solid #2ecc71;padding:4px 12px;border-radius:12px;'
                    f'font-size:0.83em;font-weight:600;">🏆 {int(_aw)} wins{_noms_str}</span>',
                    unsafe_allow_html=True,
                )

        # Metadata row 1
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(_meta_cell("Type", str(sel.get("type", "N/A"))), unsafe_allow_html=True)
        with m2:
            yr = int(sel["release_year"]) if pd.notna(sel.get("release_year")) else "N/A"
            st.markdown(_meta_cell("Year", str(yr)), unsafe_allow_html=True)
        with m3:
            imdb_v = f"{sel['imdb_score']:.1f}" if pd.notna(sel.get("imdb_score")) else "N/A"
            st.markdown(_meta_cell("IMDb", imdb_v), unsafe_allow_html=True)
        with m4:
            plat_list = sel.get("platforms", sel.get("platform", ""))
            plat_label = "Platforms" if isinstance(plat_list, list) and len(plat_list) > 1 else "Platform"
            st.markdown(
                f'<div style="padding:8px 0;"><div style="color:{CARD_TEXT_MUTED};font-size:0.72em;'
                f'text-transform:uppercase;letter-spacing:0.04em;margin-bottom:5px;">{plat_label}</div>'
                f'<div>{platform_badges_html(plat_list)}</div></div>',
                unsafe_allow_html=True,
            )

        # Metadata row 2
        r1, r2, r3, r4 = st.columns(4)
        cert = sel.get("age_certification", "")
        with r1:
            st.markdown(_meta_cell("Rating", cert if cert and str(cert) != "nan" else "N/A"), unsafe_allow_html=True)
        rt = sel.get("runtime")
        with r2:
            st.markdown(_meta_cell("Runtime", f"{int(rt)} min" if pd.notna(rt) else "N/A"), unsafe_allow_html=True)
        votes_v = sel.get("imdb_votes")
        with r3:
            st.markdown(_meta_cell("Votes", format_votes(votes_v) if pd.notna(votes_v) else "N/A"), unsafe_allow_html=True)
        qs = sel.get("quality_score", compute_quality_score(sel.to_frame().T).iloc[0])
        _qs_color = "#2ecc71" if qs >= 8.0 else "#f39c12" if qs >= 7.0 else "#e74c3c"
        with r4:
            st.markdown(
                f'<div style="padding:8px 0;"><div style="color:{CARD_TEXT_MUTED};font-size:0.72em;'
                f'text-transform:uppercase;letter-spacing:0.04em;margin-bottom:5px;">Quality Score</div>'
                f'<div style="display:flex;align-items:center;gap:6px;">'
                f'<div style="flex:1;background:#2a2a3e;border-radius:3px;height:5px;">'
                f'<div style="width:{min(qs*10,100):.0f}%;height:100%;background:{_qs_color};border-radius:3px;"></div>'
                f'</div><span style="font-size:0.82em;color:{_qs_color};font-weight:600;">{qs:.1f}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        # Box office (if available)
        if _has_enr and "box_office_usd" in _enr_row.columns:
            _bo = _enr.get("box_office_usd")
            if _bo and str(_bo) != "nan" and pd.notna(_bo) and _bo > 0:
                _bo_str = f"${_bo/1e9:.1f}B" if _bo >= 1e9 else f"${_bo/1e6:.0f}M" if _bo >= 1e6 else f"${_bo:,.0f}"
                st.markdown(
                    f'<div style="color:{CARD_TEXT_MUTED};font-size:0.78em;margin-top:2px;">'
                    f'Box Office: <strong style="color:{CARD_TEXT};">{_bo_str}</strong></div>',
                    unsafe_allow_html=True,
                )

        # Genre pills
        genres_val = sel.get("genres", [])
        if isinstance(genres_val, list) and genres_val:
            pills = " ".join(
                f'<span style="background:{CARD_BORDER};color:{CARD_TEXT};padding:3px 10px;'
                f'border-radius:12px;font-size:0.82em;margin-right:3px;">{g.title()}</span>'
                for g in genres_val
            )
            st.markdown(f'<div style="margin:8px 0;">{pills}</div>', unsafe_allow_html=True)

        # Description
        st.divider()
        desc = sel.get("description", "")
        if desc and pd.notna(desc):
            st.markdown(
                f'<div style="font-size:0.9em;color:{CARD_TEXT};line-height:1.6;">{desc}</div>',
                unsafe_allow_html=True,
            )

        # Cast & Crew expander
        try:
            credits_df = load_all_platforms_credits()
            if credits_df is not None and not credits_df.empty:
                with st.expander("Cast & Crew"):
                    tc = credits_df[credits_df["title_id"] == title_id]
                    dirs = tc[tc["role"] == "DIRECTOR"]
                    acts = tc[tc["role"] == "ACTOR"]
                    if not dirs.empty:
                        st.markdown(
                            f'<div style="color:{CARD_TEXT_MUTED};font-size:0.75em;text-transform:uppercase;'
                            f'letter-spacing:0.04em;margin-bottom:4px;">Director</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(f"**{', '.join(dirs['name'].drop_duplicates().tolist())}**")
                    if not acts.empty:
                        st.markdown(
                            f'<div style="color:{CARD_TEXT_MUTED};font-size:0.75em;text-transform:uppercase;'
                            f'letter-spacing:0.04em;margin:8px 0 4px;">Cast</div>',
                            unsafe_allow_html=True,
                        )
                        ac1, ac2 = st.columns(2)
                        for idx, (_, actor) in enumerate(acts.head(12).iterrows()):
                            char = actor.get("character", "")
                            char_html = (
                                f'<span style="color:{CARD_TEXT_MUTED};font-size:0.8em;"> as {char}</span>'
                                if char and str(char) not in ("nan", "") else ""
                            )
                            with (ac1 if idx % 2 == 0 else ac2):
                                st.markdown(
                                    f'<div style="font-size:0.85em;color:{CARD_TEXT};padding:2px 0;">'
                                    f'<strong>{actor["name"]}</strong>{char_html}</div>',
                                    unsafe_allow_html=True,
                                )
        except Exception:
            pass

        st.markdown(
            f'<div style="margin-top:8px;">'
            f'<a href="/01_Explore_Catalog" target="_self" style="display:inline-block;'
            f'background:{CARD_BG};border:1px solid {CARD_BORDER};color:{CARD_TEXT_MUTED};'
            f'padding:6px 14px;border-radius:6px;font-size:0.82em;text-decoration:none;">'
            f'Open in Explore Catalog →</a></div>',
            unsafe_allow_html=True,
        )


# ─── History helper ───────────────────────────────────────────────────────────

def _add_to_history(entry: dict) -> None:
    """Add entry to history with consecutive-duplicate suppression and unique _id."""
    history = st.session_state.rec_history
    if (
        history
        and history[0].get("type") == entry.get("type")
        and history[0].get("query") == entry.get("query")
    ):
        history[0]["count_runs"] = history[0].get("count_runs", 1) + 1
        history[0]["count"] = entry["count"]
        history[0]["results"] = entry["results"]
        history[0]["result_poster_urls"] = entry.get("result_poster_urls", [])
    else:
        entry["_id"] = st.session_state.hist_counter
        st.session_state.hist_counter += 1
        history.insert(0, entry)
        st.session_state.rec_history = history[:10]


# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab_history = st.tabs(
    ["Title Match", "Mood Board", "Vibe Search", "History"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 1: Title Match
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(
        section_header_html(
            "Title Match",
            "Find titles similar to one you love — powered by TF-IDF similarity across descriptions, genres, and shared crew.",
        ),
        unsafe_allow_html=True,
    )

    # Controls row
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([4, 1, 1, 1])
    with ctrl_col1:
        search_query = st.text_input(
            "Search for a title",
            key="sim_search",
            placeholder="Start typing a title name...",
            label_visibility="collapsed",
        )
    with ctrl_col2:
        surprise_clicked = st.button("🎲 Surprise Me", key="sim_surprise", use_container_width=True)
    with ctrl_col3:
        scope = st.radio("Scope", ["Merged", "All"], key="sim_scope", horizontal=True)
        scope_key = "merged" if scope == "Merged" else "all_platforms"
    with ctrl_col4:
        n_results = st.slider("Results", 5, 20, 10, key="sim_count")

    min_imdb_sim = st.slider(
        "Minimum IMDb", 0.0, 9.0, float(SIMILARITY_MIN_IMDB), 0.5, key="sim_min_imdb"
    )

    # Surprise Me: pick a random high-quality title
    if surprise_clicked:
        candidates = titles[
            (titles["quality_score"] >= 7.0) &
            (titles.get("imdb_votes", pd.Series(0, index=titles.index)).fillna(0) >= 5000)
        ]
        if candidates.empty:
            candidates = titles[titles["quality_score"] >= 6.0]
        if not candidates.empty:
            surprise_row = candidates.sample(1).iloc[0]
            st.session_state.sim_selected_id = surprise_row["id"]
            st.session_state.sim_results = None
            st.session_state.discovery_detail_id = None

    current_selected_id = st.session_state.get("sim_selected_id")

    # Autocomplete: show matching titles as clickable cards
    if search_query:
        matches = titles[
            titles["title"].str.contains(search_query, case=False, na=False)
        ].head(8)

        if matches.empty:
            st.info("No titles match your search.")
        else:
            st.markdown(
                f'<div style="color:{CARD_TEXT_MUTED};font-size:0.82em;margin-bottom:4px;">'
                f'Select a title below:</div>',
                unsafe_allow_html=True,
            )
            for _, match_row in matches.iterrows():
                mid = match_row["id"]
                is_sel = mid == current_selected_id
                poster_url = _get_poster_url(mid, enriched_df)
                plat_keys = match_row.get("platforms", match_row.get("platform", ""))
                poster_th = _poster_html(poster_url, match_row["title"], plat_keys, width=40, height=56)
                year_val = int(match_row["release_year"]) if pd.notna(match_row.get("release_year")) else ""
                border = f"2px solid {CARD_ACCENT}" if is_sel else f"1px solid {CARD_BORDER}"
                bg = "rgba(255,215,0,0.06)" if is_sel else CARD_BG

                mc1, mc2 = st.columns([8, 1])
                with mc1:
                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:10px;'
                        f'background:{bg};border:{border};border-radius:6px;'
                        f'padding:6px 10px;margin-bottom:3px;">'
                        f'{poster_th}'
                        f'<div>'
                        f'<div style="font-weight:600;color:{CARD_TEXT};font-size:0.92em;">'
                        f'{match_row["title"]}</div>'
                        f'<div style="font-size:0.78em;color:{CARD_TEXT_MUTED};margin-top:2px;">'
                        f'{year_val} · {platform_badges_html(plat_keys)}</div>'
                        f'</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with mc2:
                    btn_label = "✓" if is_sel else "Select"
                    if st.button(btn_label, key=f"sel_{mid}", use_container_width=True):
                        st.session_state.sim_selected_id = mid
                        st.session_state.sim_results = None
                        st.session_state.discovery_detail_id = None
                        st.rerun()

    elif current_selected_id and not search_query:
        sel_rows = titles[titles["id"] == current_selected_id]
        if not sel_rows.empty:
            sr = sel_rows.iloc[0]
            st.markdown(
                f'<div style="color:{CARD_TEXT_MUTED};font-size:0.82em;margin-bottom:4px;">'
                f'Finding titles similar to <strong style="color:{CARD_TEXT};">{sr["title"]}</strong>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Run search when a title is selected
    if current_selected_id:
        sel_title_row = titles[titles["id"] == current_selected_id]
        if sel_title_row.empty:
            sel_title_row = enriched_df[enriched_df["id"] == current_selected_id]
        sel_title_name = sel_title_row.iloc[0]["title"] if not sel_title_row.empty else "Unknown"

        if st.button("Find Similar", key="sim_go", type="primary"):
            with st.spinner(f"Finding titles similar to {sel_title_name}..."):
                # ── Scope fix: pass the correct pool based on radio ──
                if scope_key == "merged":
                    sim_pool = deduplicate_titles(load_merged_titles())
                else:
                    sim_pool = deduplicate_titles(load_all_platforms_titles())

                results = get_similar_with_explanation(
                    current_selected_id,
                    sim_pool,
                    sim_df,
                    principals_df=principals_df,
                    enriched_df=enriched_df,
                    top_k=n_results,
                    min_imdb=min_imdb_sim,
                    min_votes=1000,
                )
            st.session_state.sim_results = results
            st.session_state.discovery_detail_id = None

            hist_posters = []
            for r in results[:3]:
                p = _get_poster_url(r.get("id") or r.get("similar_id", ""), enriched_df)
                if p:
                    hist_posters.append(p)

            _add_to_history({
                "type": "Title Match",
                "query": sel_title_name,
                "count": len(results),
                "results": [r.get("title", "?") for r in results[:5]],
                "result_poster_urls": hist_posters,
            })

        # Render results
        results = st.session_state.get("sim_results")
        if results is not None:
            if results:
                st.markdown(
                    styled_banner_html(
                        "✓",
                        f"Found {len(results)} titles similar to **{sel_title_name}**",
                        bg="rgba(46,204,113,0.08)",
                        border_color="#2ecc71",
                    ),
                    unsafe_allow_html=True,
                )

                for r in results:
                    sim_score = r.get("similarity_score", 0)
                    rid = r.get("similar_id") or r.get("id", "")
                    is_open = st.session_state.discovery_detail_id == rid

                    card_col, btn_col = st.columns([11, 1])
                    with card_col:
                        _render_rec_card(r, _sim_score_badge_html(sim_score))
                    with btn_col:
                        if st.button(
                            "▼" if is_open else "ℹ️",
                            key=f"detail_sim_{rid}",
                            help="View full details",
                        ):
                            st.session_state.discovery_detail_id = None if is_open else rid
                            st.rerun()

                    # Why similar? expander
                    exp = r.get("explanation", {})
                    sim_pct = f"{sim_score:.0%}"
                    with st.expander("Why similar?", expanded=False):
                        rows_html = ""

                        # Narrative similarity
                        if sim_score >= 0.75:
                            narrative_note = "Very strong textual and thematic overlap"
                        elif sim_score >= 0.60:
                            narrative_note = "Strong overlap in descriptions and themes"
                        else:
                            narrative_note = "Moderate overlap in descriptions and themes"
                        rows_html += (
                            f'<div style="padding:6px 0;border-bottom:1px solid {CARD_BORDER};">'
                            f'<div style="color:{CARD_TEXT_MUTED};font-size:0.75em;text-transform:uppercase;'
                            f'letter-spacing:0.04em;">Narrative Similarity</div>'
                            f'<div style="color:{CARD_TEXT};font-size:0.88em;margin-top:3px;">'
                            f'{sim_pct} description overlap — {narrative_note}</div>'
                            f'</div>'
                        )

                        # Genre alignment
                        genre_overlap = exp.get("genre_overlap", [])
                        source_genres = sel_title_row.iloc[0].get("genres", []) if not sel_title_row.empty else []
                        target_genres = r.get("genres", [])
                        if source_genres and target_genres:
                            n_overlap = len(genre_overlap)
                            n_total = len(set(source_genres) | set(target_genres))
                            genre_list = ", ".join(genre_overlap) if genre_overlap else "—"
                            rows_html += (
                                f'<div style="padding:6px 0;border-bottom:1px solid {CARD_BORDER};">'
                                f'<div style="color:{CARD_TEXT_MUTED};font-size:0.75em;text-transform:uppercase;'
                                f'letter-spacing:0.04em;">Genre Alignment</div>'
                                f'<div style="color:{CARD_TEXT};font-size:0.88em;margin-top:3px;">'
                                f'{n_overlap} of {n_total} unique genres overlap: {genre_list}</div>'
                                f'</div>'
                            )

                        # Shared crew
                        shared_crew = exp.get("shared_crew", [])
                        if shared_crew:
                            crew_parts = []
                            for c in shared_crew[:3]:
                                role = c.get("role", "crew").replace("_", " ").title()
                                name = c.get("name", "")
                                crew_parts.append(f"{name} ({role})")
                            rows_html += (
                                f'<div style="padding:6px 0;border-bottom:1px solid {CARD_BORDER};">'
                                f'<div style="color:{CARD_TEXT_MUTED};font-size:0.75em;text-transform:uppercase;'
                                f'letter-spacing:0.04em;">Shared Crew</div>'
                                f'<div style="color:{CARD_TEXT};font-size:0.88em;margin-top:3px;">'
                                f'{" · ".join(crew_parts)}</div>'
                                f'</div>'
                            )

                        # Shared vibe tags
                        vibe_tags = exp.get("matched_vibe_tags", [])
                        if vibe_tags:
                            tag_pills = " ".join(
                                f'<span style="background:rgba(255,215,0,0.1);border:1px solid {CARD_ACCENT};'
                                f'padding:2px 8px;border-radius:10px;font-size:0.75em;margin-right:3px;">{t}</span>'
                                for t in vibe_tags[:5]
                            )
                            rows_html += (
                                f'<div style="padding:6px 0;">'
                                f'<div style="color:{CARD_TEXT_MUTED};font-size:0.75em;text-transform:uppercase;'
                                f'letter-spacing:0.04em;">Shared Vibe Tags</div>'
                                f'<div style="margin-top:4px;">{tag_pills}</div>'
                                f'</div>'
                            )

                        if not rows_html:
                            rows_html = (
                                f'<div style="color:{CARD_TEXT_MUTED};font-size:0.88em;padding:6px 0;">'
                                f'Matched by description similarity ({sim_pct} overlap).</div>'
                            )

                        st.markdown(
                            f'<div style="background:{CARD_BG};border:1px solid {CARD_BORDER};'
                            f'border-radius:6px;padding:10px 14px;">{rows_html}</div>',
                            unsafe_allow_html=True,
                        )

                # Detail panel
                if st.session_state.discovery_detail_id:
                    st.markdown("---")
                    _render_title_detail(st.session_state.discovery_detail_id)

            else:
                st.warning("No similar titles found with the current filters.")


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 2: Mood Board
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(
        section_header_html(
            "Mood Board",
            "Select feelings — we match them to cinematic signals in 25,000+ titles.",
        ),
        unsafe_allow_html=True,
    )
    st.caption("Click one or more tiles to describe what you're in the mood for right now.")

    # 4×4 grid of mood tiles
    selected_moods: list = list(st.session_state.selected_moods)
    rows_of_4 = [MOOD_TILES[i : i + 4] for i in range(0, len(MOOD_TILES), 4)]

    for tile_row in rows_of_4:
        cols = st.columns(4)
        for col, tile in zip(cols, tile_row):
            with col:
                is_sel = tile["label"] in selected_moods
                btn_label = f"{'✓ ' if is_sel else ''}{tile['emoji']} {tile['label']}"
                if st.button(btn_label, key=f"mood_{tile['label']}", use_container_width=True):
                    if tile["label"] in selected_moods:
                        selected_moods.remove(tile["label"])
                    else:
                        selected_moods.append(tile["label"])
                    st.session_state.selected_moods = selected_moods
                    st.rerun()

    # Selected moods summary
    if selected_moods:
        pills_html = " ".join(
            f'<span style="background:rgba(255,215,0,0.12);border:1px solid {CARD_ACCENT};'
            f'padding:3px 10px;border-radius:12px;font-size:0.82em;">'
            f'{next((t["emoji"] for t in MOOD_TILES if t["label"] == m), "")} {m}</span>'
            for m in selected_moods
        )
        st.markdown(
            f'<div style="margin:8px 0 4px 0;color:{CARD_TEXT_MUTED};font-size:0.8em;">Selected: {pills_html}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="color:{CARD_TEXT_MUTED};font-size:0.8em;margin:8px 0;">'
            f'No moods selected yet.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Controls
    mood_ctrl1, mood_ctrl2 = st.columns(2)
    with mood_ctrl1:
        mood_type = st.radio(
            "Content type", ["Both", "Movie", "Show"], key="mood_type", horizontal=True
        )
    with mood_ctrl2:
        mood_scope = st.radio(
            "Platform scope", ["Merged", "All Platforms"], key="mood_scope", horizontal=True
        )

    if st.button("Find My Match", key="mood_go", type="primary", disabled=not selected_moods):
        if not selected_moods:
            st.warning("Select at least one mood tile first.")
        else:
            with st.spinner("Matching your moods to the catalog..."):
                scope_key_m = "merged" if mood_scope == "Merged" else "all_platforms"
                source_m = (
                    load_all_platforms_titles() if scope_key_m == "all_platforms"
                    else deduplicate_titles(load_merged_titles())
                )
                source_m = deduplicate_titles(source_m)

                mood_results = mood_board_recommendations(
                    source_m,
                    selected_mood_labels=selected_moods,
                    content_type=mood_type,
                    top_k=20,
                    enriched_df=enriched_df,
                    min_imdb=6.0,
                    min_votes=1000,
                )

            if not mood_results.empty:
                st.markdown(
                    styled_banner_html(
                        "✓",
                        f"Found {len(mood_results)} titles matching your mood",
                        bg="rgba(46,204,113,0.08)",
                        border_color="#2ecc71",
                    ),
                    unsafe_allow_html=True,
                )
                st.caption(f"Showing top {len(mood_results)} quality-filtered matches (min IMDb 6.0, min 1,000 votes)")

                hist_posters_m = []

                for _, mood_row in mood_results.iterrows():
                    row_dict = mood_row.to_dict()
                    mood_pct = row_dict.get("mood_match_pct", 0)
                    mid = row_dict.get("id", "")
                    is_open = st.session_state.discovery_detail_id == mid

                    card_col, btn_col = st.columns([11, 1])
                    with card_col:
                        _render_rec_card(row_dict, _mood_score_badge_html(mood_pct))
                    with btn_col:
                        if st.button(
                            "▼" if is_open else "ℹ️",
                            key=f"detail_mood_{mid}",
                            help="View full details",
                        ):
                            st.session_state.discovery_detail_id = None if is_open else mid
                            st.rerun()

                    # Show matched mood tags
                    matched_m = row_dict.get("matched_moods", [])
                    if matched_m:
                        mood_tag_html = " ".join(
                            f'<span style="background:rgba(255,215,0,0.1);border:1px solid {CARD_ACCENT};'
                            f'padding:2px 8px;border-radius:10px;font-size:0.72em;margin-right:3px;">'
                            f'{next((t["emoji"] for t in MOOD_TILES if t["label"] == m), "")} {m}</span>'
                            for m in matched_m
                        )
                        st.markdown(
                            f'<div style="margin:0 0 8px 0;padding-left:4px;">{mood_tag_html}</div>',
                            unsafe_allow_html=True,
                        )

                    if len(hist_posters_m) < 3:
                        p = _get_poster_url(mid, enriched_df)
                        if p:
                            hist_posters_m.append(p)

                # Detail panel
                if st.session_state.discovery_detail_id:
                    st.markdown("---")
                    _render_title_detail(st.session_state.discovery_detail_id)

                _add_to_history({
                    "type": "Mood Board",
                    "query": ", ".join(selected_moods[:3]) + ("..." if len(selected_moods) > 3 else ""),
                    "count": len(mood_results),
                    "results": mood_results["title"].head(5).tolist(),
                    "result_poster_urls": hist_posters_m,
                })
            else:
                st.warning("No matching titles found. Try selecting different moods or expanding the scope.")


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 3: Vibe Search
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(
        section_header_html(
            "Vibe Search",
            "Describe a mood or scenario in plain language — NLP-powered semantic matching finds it.",
        ),
        unsafe_allow_html=True,
    )
    st.caption("Enter a free-form description of what you want to watch. We'll handle the rest.")

    vibe_query = st.text_area(
        "What are you in the mood for?",
        placeholder="e.g., 'A slow-burn psychological thriller with a twist ending, set in a small town'",
        height=100,
        key="vibe_query",
    )

    with st.expander("Optional filters", expanded=True):
        vibe_col1, vibe_col2, vibe_col3 = st.columns(3)
        with vibe_col1:
            vibe_min_imdb = st.slider(
                "Min IMDb", 0.0, 9.0, 6.5, 0.5, key="vibe_imdb",
                help="Titles below this threshold are excluded. Default 6.5 ensures quality results."
            )
        with vibe_col2:
            vibe_year = st.slider("Year range", 1950, 2024, (1970, 2024), key="vibe_year")
        with vibe_col3:
            vibe_scope = st.radio("Scope", ["Merged", "All Platforms"], key="vibe_scope", horizontal=True)

    if vibe_query and st.button("Search Vibes", key="vibe_go", type="primary"):
        signals = extract_vibe_signals(vibe_query)
        if signals:
            signal_pills_html = " ".join(
                f'<span style="background:rgba(255,215,0,0.1);border:1px solid {CARD_ACCENT};'
                f'padding:3px 10px;border-radius:12px;font-size:0.82em;margin:2px;'
                f'display:inline-block;">{s}</span>'
                for s in signals
            )
            st.markdown(
                f'<div style="margin:8px 0 12px 0;">'
                f'<div style="color:{CARD_TEXT};font-size:0.92em;font-weight:600;margin-bottom:4px;">'
                f'🏷️ Themes detected in your search</div>'
                f'<div style="color:{CARD_TEXT_MUTED};font-size:0.8em;margin-bottom:8px;">'
                f'These signals are driving your results.</div>'
                f'{signal_pills_html}'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("No specific themes detected — searching by description similarity.")

        with st.spinner("Searching the catalog with semantic matching..."):
            scope_key_v = "merged" if vibe_scope == "Merged" else "all_platforms"
            source_v = (
                load_all_platforms_titles() if scope_key_v == "all_platforms"
                else deduplicate_titles(load_merged_titles())
            )
            source_v = deduplicate_titles(source_v)

            results_v, detected_signals = vibe_search(
                vibe_query,
                source_v,
                genome_vectors=genome_vectors,
                genome_id_map=genome_id_map,
                enriched_df=enriched_df,
                scope=scope_key_v,
                top_k=15,
                min_imdb=vibe_min_imdb if vibe_min_imdb > 0 else None,
                year_range=vibe_year,
                min_votes=1000,
            )

        if not results_v.empty:
            st.markdown(
                styled_banner_html(
                    "✓",
                    f"Showing top {len(results_v)} quality-filtered matches",
                    bg="rgba(46,204,113,0.08)",
                    border_color="#2ecc71",
                ),
                unsafe_allow_html=True,
            )

            n_v = len(results_v)
            results_v = results_v.reset_index(drop=True)
            hist_posters_v = []

            for i_v, (_, vrow) in enumerate(results_v.iterrows()):
                rank_pct = i_v / max(n_v - 1, 1)
                vibe_badge = _vibe_label_badge_html(rank_pct)
                vid = vrow.get("id", "")
                is_open = st.session_state.discovery_detail_id == vid

                card_col, btn_col = st.columns([11, 1])
                with card_col:
                    _render_rec_card(vrow.to_dict(), vibe_badge)
                with btn_col:
                    if st.button(
                        "▼" if is_open else "ℹ️",
                        key=f"detail_vibe_{vid}_{i_v}",
                        help="View full details",
                    ):
                        st.session_state.discovery_detail_id = None if is_open else vid
                        st.rerun()

                # Show matched genome tags
                top_tags = vrow.get("top_tags")
                if isinstance(top_tags, (list,)) and detected_signals:
                    matched_tags = [
                        t for t in top_tags
                        if any(s.lower() in t.lower() for s in detected_signals)
                    ]
                    if matched_tags:
                        tag_html = " ".join(
                            f'<span style="background:rgba(46,204,113,0.1);border:1px solid #2ecc71;'
                            f'padding:1px 6px;border-radius:3px;font-size:0.72rem;">{t}</span>'
                            for t in matched_tags[:5]
                        )
                        st.markdown(
                            f'<div style="margin:0 0 8px 0;padding-left:4px;">{tag_html}</div>',
                            unsafe_allow_html=True,
                        )

                if len(hist_posters_v) < 3:
                    p = _get_poster_url(vid, enriched_df)
                    if p:
                        hist_posters_v.append(p)

            # Detail panel
            if st.session_state.discovery_detail_id:
                st.markdown("---")
                _render_title_detail(st.session_state.discovery_detail_id)

            _add_to_history({
                "type": "Vibe Search",
                "query": vibe_query[:60] + ("..." if len(vibe_query) > 60 else ""),
                "count": len(results_v),
                "results": results_v["title"].head(5).tolist(),
                "result_poster_urls": hist_posters_v,
            })
        else:
            st.warning("No titles matched your vibe. Try different keywords or relax the filters.")

    # How Vibe Search works — styled info card
    with st.expander("How Vibe Search works"):
        components = [
            ("Description embedding similarity", 35, "#4FC3F7"),
            ("Genre + keyword matching", 25, "#81C784"),
            ("MovieLens genome vector match", 15, "#FFB74D"),
            ("Bayesian quality score", 15, "#CE93D8"),
            ("Awards boost", 10, "#F48FB1"),
        ]
        rows_html = ""
        for label, pct, color in components:
            bar_width = pct * 3
            rows_html += (
                f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">'
                f'<div style="width:200px;color:{CARD_TEXT};font-size:0.85em;flex-shrink:0;">{label}</div>'
                f'<div style="flex:1;background:#2a2a3e;border-radius:4px;height:8px;overflow:hidden;">'
                f'<div style="width:{bar_width}px;max-width:100%;height:100%;background:{color};border-radius:4px;"></div>'
                f'</div>'
                f'<div style="color:{color};font-size:0.82em;font-weight:700;width:32px;text-align:right;">{pct}%</div>'
                f'</div>'
            )
        st.markdown(
            f'<div style="background:{CARD_BG};border:1px solid {CARD_BORDER};border-radius:8px;padding:16px;">'
            f'<div style="font-weight:600;color:{CARD_TEXT};margin-bottom:12px;">Hybrid Scoring Formula</div>'
            f'{rows_html}'
            f'<div style="color:{CARD_TEXT_MUTED};font-size:0.78em;margin-top:12px;">'
            f'* Titles without MovieLens genome data redistribute that 15% to description (60%) and genre (40%) matching.'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 4: History — Recent Searches log
# ═══════════════════════════════════════════════════════════════════════════════
_HIST_ICONS = {"Title Match": "🔍", "Mood Board": "🎭", "Vibe Search": "✨"}

with tab_history:
    st.markdown(
        section_header_html(
            "Recent Searches",
            "A log of your discovery sessions this session.",
        ),
        unsafe_allow_html=True,
    )

    if not st.session_state.rec_history:
        st.info("No searches yet. Try one of the tabs above!")
    else:
        # Controls row
        ctrl_h1, ctrl_h2, ctrl_h3 = st.columns([3, 3, 2])
        with ctrl_h1:
            group_by = st.radio(
                "View",
                ["Chronological", "By Tab Type"],
                horizontal=True,
                key="hist_group",
                label_visibility="collapsed",
            )
        with ctrl_h3:
            if st.button("Clear All", key="hist_clear_all"):
                st.session_state.rec_history = []
                st.rerun()

        history = list(st.session_state.rec_history)

        if group_by == "By Tab Type":
            grouped: dict = defaultdict(list)
            for e in history:
                grouped[e.get("type", "Other")].append(e)
            display_entries = []
            for tab_type in ["Title Match", "Mood Board", "Vibe Search", "Other"]:
                if tab_type in grouped:
                    display_entries.extend(grouped[tab_type])
        else:
            display_entries = history

        for entry in display_entries:
            eid = entry.get("_id", id(entry))
            tab_type = entry.get("type", "Search")
            tab_icon = _HIST_ICONS.get(tab_type, "📋")
            run_count = entry.get("count_runs", 1)

            with st.container(border=True):
                hdr_col, rm_col = st.columns([9, 1])
                with hdr_col:
                    run_badge = (
                        f'<span style="background:rgba(136,136,136,0.2);color:{CARD_TEXT_MUTED};'
                        f'padding:1px 6px;border-radius:10px;font-size:0.75em;margin-left:6px;">'
                        f'Run {run_count}×</span>'
                        if run_count > 1 else ""
                    )
                    st.markdown(
                        f'<div style="font-weight:600;color:{CARD_TEXT};font-size:0.95em;">'
                        f'{tab_icon} {tab_type}{run_badge}</div>'
                        f'<div style="color:{CARD_TEXT_MUTED};font-size:0.83em;margin-top:2px;">'
                        f'{entry.get("query", "")}</div>',
                        unsafe_allow_html=True,
                    )
                with rm_col:
                    if st.button("✕", key=f"hist_rm_{eid}", help="Remove this entry"):
                        st.session_state.rec_history = [
                            e for e in st.session_state.rec_history
                            if e.get("_id", id(e)) != eid
                        ]
                        st.rerun()

                # Thumbnail strip
                poster_urls = entry.get("result_poster_urls", [])
                if poster_urls:
                    thumb_cols = st.columns([1, 1, 1, 8])
                    for tc, url in zip(thumb_cols[:3], poster_urls[:3]):
                        with tc:
                            st.image(url, width=42)

                # Results preview
                results_preview = entry.get("results", [])
                if results_preview:
                    st.markdown(
                        f'<div style="color:{CARD_TEXT_MUTED};font-size:0.78em;margin-top:4px;">'
                        f'{entry.get("count", len(results_preview))} results including: '
                        f'{", ".join(results_preview)}</div>',
                        unsafe_allow_html=True,
                    )


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="border-top:1px solid #333;padding:16px 0;color:#666;'
    'font-size:0.8em;text-align:center;">'
    "Hypothetical merger for academic analysis. Data is a snapshot (mid-2023). "
    "All insights are illustrative, not prescriptive. "
    "Update: As of Feb 26, 2026, Netflix withdrew from this acquisition after Paramount Skydance's competing bid was deemed superior by the WBD board."
    "</div>",
    unsafe_allow_html=True,
)
