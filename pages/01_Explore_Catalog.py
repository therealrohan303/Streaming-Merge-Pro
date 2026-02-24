"""Page 1: Explore Catalog — searchable catalog with similar title recommendations."""

import streamlit as st

from src.analysis.scoring import compute_quality_score, format_votes
from src.analysis.similarity import get_similar_titles
from src.config import (
    CARD_ACCENT,
    CARD_BG,
    CARD_BORDER,
    CARD_TEXT,
    CARD_TEXT_MUTED,
    CATALOG_PAGE_SIZE,
    INITIAL_SIDEBAR_STATE,
    LAYOUT,
    PAGE_ICON,
    PAGE_TITLE,
    PLATFORMS,
    SIMILARITY_TOP_K,
)
from src.data.loaders import (
    deduplicate_titles,
    get_credits_for_view,
    get_titles_for_view,
    load_all_platforms_titles,
    load_enriched_titles,
    load_merged_titles,
    load_similarity_data,
)
from src.ui.badges import platform_badges_html, section_header_html
from src.ui.filters import apply_filters, render_sidebar_filters
from src.ui.session import DEFAULTS, init_session_state

_MERGED_COLOR = PLATFORMS["merged"]["color"]

_PLATFORM_LABELS = {
    "merged": "Merged (Netflix + Max)",
    "netflix": "Netflix Only",
    "max": "Max Only",
    "all_platforms": "All 6 Platforms",
}

# ── page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title=f"Explore Catalog | {PAGE_TITLE}",
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state=INITIAL_SIDEBAR_STATE,
)

init_session_state()

# Page-specific session state
st.session_state.setdefault("explore_selected_id", None)
st.session_state.setdefault("explore_search", "")
st.session_state.setdefault("explore_sort", "Quality Score")
st.session_state.setdefault("explore_page", 0)
st.session_state.setdefault("explore_page_size", CATALOG_PAGE_SIZE)
st.session_state.setdefault("explore_sim_scope", "merged")

# ── sidebar (reuse global filters) ──────────────────────────────────────────

raw_df = get_titles_for_view(st.session_state["platform_view"])
filters = render_sidebar_filters(raw_df)
raw_df = get_titles_for_view(filters["platform_view"])
df = apply_filters(raw_df, filters)

# ── page header ──────────────────────────────────────────────────────────────

st.markdown(
    section_header_html("Explore Catalog", "Search, browse, and discover similar titles across the catalog", font_size="2em"),
    unsafe_allow_html=True,
)

# ── active filter pills ──────────────────────────────────────────────────────

active_pills = []
if filters["platform_view"] != DEFAULTS["platform_view"]:
    active_pills.append(
        f'Platform: {_PLATFORM_LABELS.get(filters["platform_view"], filters["platform_view"])}'
    )
if set(filters["content_types"]) != set(DEFAULTS["content_types"]):
    active_pills.append(f'Type: {", ".join(filters["content_types"])}')
if filters["year_range"] != tuple(DEFAULTS["year_range"]):
    active_pills.append(f'Years: {filters["year_range"][0]}–{filters["year_range"][1]}')
if filters["min_imdb"] > 0:
    active_pills.append(f'IMDb ≥ {filters["min_imdb"]:.1f}')
if filters["selected_genres"]:
    genre_str = ", ".join(g.title() for g in filters["selected_genres"][:3])
    if len(filters["selected_genres"]) > 3:
        genre_str += f' +{len(filters["selected_genres"]) - 3} more'
    active_pills.append(f"Genres: {genre_str}")

if active_pills:
    pills_html = " ".join(
        f'<span style="background:#2a2a3e;color:{CARD_TEXT_MUTED};'
        f'padding:3px 9px;border-radius:12px;font-size:0.76em;'
        f'margin-right:3px;display:inline-block;margin-bottom:3px;">{p}</span>'
        for p in active_pills
    )
    st.markdown(
        f'<div style="margin-bottom:8px;">'
        f'<span style="color:{CARD_TEXT_MUTED};font-size:0.76em;margin-right:6px;">Active filters:</span>'
        f'{pills_html}</div>',
        unsafe_allow_html=True,
    )

# ── search + sort row ─────────────────────────────────────────────────────────

ctrl_left, ctrl_right = st.columns([3, 1])

with ctrl_left:
    search_text = st.text_input(
        "Search titles",
        value=st.session_state["explore_search"],
        placeholder="Search by title, cast, or keyword...",
        help="Searches across titles, descriptions, and cast names",
        key="_explore_search_input",
        label_visibility="collapsed",
    )
    if search_text != st.session_state["explore_search"]:
        st.session_state["explore_search"] = search_text
        st.session_state["explore_page"] = 0

with ctrl_right:
    sort_options = [
        "Quality Score",
        "IMDb Score",
        "Release Year (Newest)",
        "Release Year (Oldest)",
        "Most Voted",
        "Title A-Z",
        "Popularity",
    ]
    sort_by = st.selectbox(
        "Sort by",
        options=sort_options,
        index=sort_options.index(st.session_state["explore_sort"]),
        key="_explore_sort_input",
        label_visibility="collapsed",
    )
    st.session_state["explore_sort"] = sort_by

# ── compute quality scores & deduplicate ────────────────────────────────────

df = df.copy()
df["quality_score"] = compute_quality_score(df)
df = deduplicate_titles(df)

# ── quick suggestions ─────────────────────────────────────────────────────

if search_text.strip():
    _sug_mask = df["title"].str.contains(
        search_text.strip(), case=False, na=False, regex=False
    )
    _suggestions = df[_sug_mask].nlargest(5, "quality_score")
    if not _suggestions.empty:
        st.caption("Quick suggestions:")
        sug_cols = st.columns(min(len(_suggestions), 5))
        for i, (_, sug) in enumerate(_suggestions.iterrows()):
            year = (
                str(int(sug["release_year"]))
                if sug["release_year"] == sug["release_year"]
                else "?"
            )
            with sug_cols[i]:
                if st.button(
                    f"{sug['title']} ({year})",
                    key=f"sug_{sug['id']}",
                    use_container_width=True,
                ):
                    st.session_state["explore_selected_id"] = sug["id"]
                    st.session_state["explore_search"] = ""
                    st.rerun()

# ── apply search + sort ─────────────────────────────────────────────────────

if search_text.strip():
    query = search_text.strip()
    title_mask = df["title"].str.contains(query, case=False, na=False, regex=False)
    desc_mask = df["description"].fillna("").str.contains(
        query, case=False, na=False, regex=False
    )
    _credits = get_credits_for_view(filters["platform_view"])
    _cast_ids = set(
        _credits[
            _credits["name"].str.contains(query, case=False, na=False, regex=False)
        ]["title_id"]
    )
    cast_mask = df["id"].isin(_cast_ids)
    df = df[title_mask | desc_mask | cast_mask].reset_index(drop=True)

_SORT_MAP = {
    "Quality Score": ("quality_score", False),
    "IMDb Score": ("imdb_score", False),
    "Release Year (Newest)": ("release_year", False),
    "Release Year (Oldest)": ("release_year", True),
    "Most Voted": ("imdb_votes", False),
    "Title A-Z": ("title", True),
    "Popularity": ("tmdb_popularity", False),
}
sort_col, sort_asc = _SORT_MAP[sort_by]
df = df.sort_values(sort_col, ascending=sort_asc, na_position="last").reset_index(drop=True)

# ── pagination math ──────────────────────────────────────────────────────────

page_size = st.session_state["explore_page_size"]
total_results = len(df)
total_pages = max(1, -(-total_results // page_size))
current_page = min(max(st.session_state["explore_page"], 0), total_pages - 1)
st.session_state["explore_page"] = current_page
start_idx = current_page * page_size
end_idx = min(start_idx + page_size, total_results)
page_df = df.iloc[start_idx:end_idx]

# ── two-panel layout ────────────────────────────────────────────────────────

left_col, right_col = st.columns([2, 3], gap="large")

# ── LEFT PANEL ──────────────────────────────────────────────────────────────

with left_col:

    # ── Pagination — row 1: navigation buttons ────────────────────────────────
    range_text = (
        f"Showing {start_idx + 1}–{end_idx} of {total_results:,}"
        if total_results > 0
        else "No results"
    )
    pn1, pn2, pn_center, pn3, pn4 = st.columns([1, 1, 6, 1, 1])
    with pn1:
        if st.button("«", disabled=(current_page == 0), use_container_width=True, key="pg_first"):
            st.session_state["explore_page"] = 0
            st.rerun()
    with pn2:
        if st.button("‹", disabled=(current_page == 0), use_container_width=True, key="pg_prev"):
            st.session_state["explore_page"] = current_page - 1
            st.rerun()
    with pn_center:
        st.markdown(
            f"<div style='text-align:center;padding:6px 0;'>"
            f"<span style='font-size:0.9em;font-weight:500;color:{CARD_TEXT};'>"
            f"Page {current_page + 1} of {total_pages}</span>"
            f"&nbsp;&nbsp;<span style='font-size:0.76em;color:{CARD_TEXT_MUTED};'>{range_text}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with pn3:
        if st.button("›", disabled=(current_page >= total_pages - 1), use_container_width=True, key="pg_next"):
            st.session_state["explore_page"] = current_page + 1
            st.rerun()
    with pn4:
        if st.button("»", disabled=(current_page >= total_pages - 1), use_container_width=True, key="pg_last"):
            st.session_state["explore_page"] = total_pages - 1
            st.rerun()

    # ── Pagination — row 2: per-page selector + jump-to ──────────────────────
    pp_col, jump_col, go_col, _ = st.columns([2, 2, 1, 3])
    page_size_options = [25, 50, 100]
    with pp_col:
        new_page_size = st.selectbox(
            "Per page",
            options=page_size_options,
            index=page_size_options.index(page_size) if page_size in page_size_options else 1,
            key="_explore_page_size_input",
        )
        if new_page_size != page_size:
            st.session_state["explore_page_size"] = new_page_size
            st.session_state["explore_page"] = 0
            st.rerun()
    with jump_col:
        jump_input = st.text_input(
            "Jump to page",
            value="",
            placeholder="e.g. 5",
            key="_explore_jump_input",
        )
    with go_col:
        # Spacer to align button with input field (label takes ~22px)
        st.markdown("<div style='height:22px;'></div>", unsafe_allow_html=True)
        if st.button("Go", key="pg_go", use_container_width=True):
            if jump_input.strip():
                try:
                    target = int(jump_input.strip()) - 1
                    if 0 <= target < total_pages:
                        st.session_state["explore_page"] = target
                        st.rerun()
                except ValueError:
                    pass

    st.divider()

    # ── Results list ─────────────────────────────────────────────────────────
    if len(page_df) == 0:
        if search_text.strip():
            st.markdown(
                f"<div style='text-align:center;padding:32px 16px;'>"
                f"<div style='font-size:1.05em;color:{CARD_TEXT};margin-bottom:8px;'>"
                f"No results for '<strong>{search_text.strip()}</strong>'</div>"
                f"<div style='font-size:0.85em;color:{CARD_TEXT_MUTED};'>"
                f"Try broader keywords, check spelling, or clear sidebar filters</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("No titles match the current filters.")
    else:
        for row_idx, (_, row) in enumerate(page_df.iterrows()):
            title_id = row["id"]
            is_selected = title_id == st.session_state["explore_selected_id"]

            imdb_str = (
                f"{row['imdb_score']:.1f}"
                if row["imdb_score"] == row["imdb_score"]
                else "N/A"
            )
            qs_str = (
                f"{row['quality_score']:.1f}"
                if "quality_score" in row.index and row["quality_score"] == row["quality_score"]
                else "N/A"
            )
            year_str = (
                str(int(row["release_year"]))
                if row["release_year"] == row["release_year"]
                else "?"
            )
            votes_str = format_votes(row.get("imdb_votes"))
            plat_badge = platform_badges_html(row.get("platforms", row.get("platform", "")))

            # Genre pills — max 2
            genre_html = ""
            if isinstance(row["genres"], list) and row["genres"]:
                genre_html = " ".join(
                    f'<span style="background:#2a2a3e;color:{CARD_TEXT_MUTED};'
                    f'padding:2px 7px;border-radius:10px;font-size:0.72em;">'
                    f'{g.title()}</span>'
                    for g in row["genres"][:2]
                )

            # Selected: gold left-border; default: subtle border
            if is_selected:
                border_css = f"border-left:4px solid {CARD_ACCENT};border-radius:0 8px 8px 0;border-top:1px solid {CARD_BORDER};border-right:1px solid {CARD_BORDER};border-bottom:1px solid {CARD_BORDER};"
                bg_css = f"background:rgba(255,215,0,0.05);"
            else:
                border_css = f"border:1px solid {CARD_BORDER};border-radius:8px;"
                bg_css = f"background:{CARD_BG};"

            hover_js = (
                'onmouseover="this.style.backgroundColor=\'rgba(255,255,255,0.04)\'"'
                ' onmouseout="this.style.backgroundColor=\'\'"'
            )

            st.markdown(
                f'<div style="{bg_css}{border_css}padding:9px 12px;'
                f'margin-bottom:3px;transition:background 0.1s;" {hover_js}>'
                # Line 1: title + year
                f'<div style="font-size:0.95em;font-weight:600;color:{CARD_TEXT};">'
                f'{row["title"]} <span style="color:{CARD_TEXT_MUTED};font-weight:400;">({year_str})</span>'
                f'</div>'
                # Line 2: badge + IMDb + votes + QS
                f'<div style="display:flex;align-items:center;gap:5px;flex-wrap:wrap;margin-top:3px;font-size:0.8em;">'
                f'{plat_badge}'
                f'<span style="color:{CARD_TEXT_MUTED};">{row["type"]}</span>'
                f'<span style="color:{CARD_TEXT_MUTED};">·</span>'
                f'<span style="color:{CARD_TEXT};">IMDb {imdb_str}</span>'
                f'<span style="color:{CARD_TEXT_MUTED};">({votes_str})</span>'
                f'<span style="color:{CARD_TEXT_MUTED};">·</span>'
                f'<span style="color:{CARD_ACCENT};font-weight:600;">QS {qs_str}</span>'
                f'</div>'
                # Line 3: genre pills
                f'<div style="margin-top:4px;">{genre_html}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            _, _btn_col = st.columns([4, 1])
            with _btn_col:
                if st.button(
                    "→",
                    key=f"sel_{row_idx}_{current_page}",
                    use_container_width=True,
                    help="View details",
                ):
                    st.session_state["explore_selected_id"] = title_id
                    st.rerun()

# ── RIGHT PANEL ──────────────────────────────────────────────────────────────

with right_col:
    selected_id = st.session_state["explore_selected_id"]

    if selected_id is None:
        st.info("Select a title from the results list to see details and similar titles.")
    else:
        sel_rows = df[df["id"] == selected_id]
        if sel_rows.empty:
            sel_rows = raw_df[raw_df["id"] == selected_id]
            if not sel_rows.empty:
                sel_rows = deduplicate_titles(sel_rows.copy())
                sel_rows["quality_score"] = compute_quality_score(sel_rows)
        # Fallback: title from a competitor platform (e.g. selected from similar titles)
        if sel_rows.empty:
            _all_titles = load_all_platforms_titles()
            sel_rows = _all_titles[_all_titles["id"] == selected_id]
            if not sel_rows.empty:
                sel_rows = deduplicate_titles(sel_rows.copy())
                sel_rows["quality_score"] = compute_quality_score(sel_rows)
        if sel_rows.empty:
            st.warning("Selected title not found. Try changing filters or switching to 'All 6 Platforms' view.")
        else:
            sel = sel_rows.iloc[0]

            _enriched = load_enriched_titles()
            _enr_row = _enriched[_enriched["id"] == selected_id]
            _has_enr = not _enr_row.empty
            _enr = _enr_row.iloc[0] if _has_enr else None

            with st.container(border=True):
                # ── Poster ──
                _poster = None
                if _has_enr and "poster_url" in _enr_row.columns:
                    _poster = _enr.get("poster_url")
                if _poster and str(_poster) != "nan":
                    st.image(_poster, width=200)

                # ── Title + award badge ──
                st.subheader(sel["title"])

                if _has_enr and "award_wins" in _enr_row.columns:
                    _aw = _enr.get("award_wins", 0)
                    _an = _enr.get("award_noms", 0)
                    if _aw and _aw > 0:
                        _noms_str = f", {int(_an)} nominations" if _an and _an > 0 else ""
                        st.markdown(
                            f'<span style="background:rgba(46,204,113,0.15);color:#2ecc71;'
                            f'border:1px solid #2ecc71;padding:4px 12px;border-radius:12px;'
                            f'font-size:0.83em;font-weight:600;">'
                            f'🏆 {int(_aw)} wins{_noms_str}</span>',
                            unsafe_allow_html=True,
                        )

                # ── Metadata grid — row 1: Type | Year | IMDb | Platform ──
                m1, m2, m3, m4 = st.columns(4)

                def _meta_cell(label: str, value: str) -> str:
                    return (
                        f'<div style="padding:8px 0;">'
                        f'<div style="color:{CARD_TEXT_MUTED};font-size:0.72em;text-transform:uppercase;'
                        f'letter-spacing:0.04em;margin-bottom:3px;">{label}</div>'
                        f'<div style="font-weight:600;color:{CARD_TEXT};font-size:0.95em;">{value}</div>'
                        f'</div>'
                    )

                with m1:
                    st.markdown(_meta_cell("Type", sel["type"]), unsafe_allow_html=True)
                with m2:
                    year_val = int(sel["release_year"]) if sel["release_year"] == sel["release_year"] else "N/A"
                    st.markdown(_meta_cell("Year", str(year_val)), unsafe_allow_html=True)
                with m3:
                    imdb_val = f"{sel['imdb_score']:.1f}" if sel["imdb_score"] == sel["imdb_score"] else "N/A"
                    st.markdown(_meta_cell("IMDb", imdb_val), unsafe_allow_html=True)
                with m4:
                    plat_list = sel.get("platforms", sel.get("platform", ""))
                    plat_label = "Platforms" if isinstance(plat_list, list) and len(plat_list) > 1 else "Platform"
                    st.markdown(
                        f'<div style="padding:8px 0;">'
                        f'<div style="color:{CARD_TEXT_MUTED};font-size:0.72em;text-transform:uppercase;'
                        f'letter-spacing:0.04em;margin-bottom:5px;">{plat_label}</div>'
                        f'<div>{platform_badges_html(plat_list)}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                # ── Metadata grid — row 2: Rating | Runtime | Votes | Quality Score ──
                r1, r2, r3, r4 = st.columns(4)

                cert_val = sel.get("age_certification", "")
                cert_str = cert_val if cert_val and str(cert_val) != "nan" else "N/A"
                with r1:
                    st.markdown(_meta_cell("Rating", cert_str), unsafe_allow_html=True)

                rt_val = sel.get("runtime")
                rt_str = f"{int(rt_val)} min" if rt_val and rt_val == rt_val else "N/A"
                with r2:
                    st.markdown(_meta_cell("Runtime", rt_str), unsafe_allow_html=True)

                votes_val = sel.get("imdb_votes")
                votes_display = format_votes(votes_val) if votes_val and votes_val == votes_val else "N/A"
                with r3:
                    st.markdown(_meta_cell("Votes", votes_display), unsafe_allow_html=True)

                qs = (
                    sel["quality_score"]
                    if "quality_score" in sel.index
                    else compute_quality_score(sel.to_frame().T).iloc[0]
                )
                _qs_color = "#2ecc71" if qs >= 8.0 else "#f39c12" if qs >= 7.0 else "#e74c3c"
                _qs_pct = min(qs * 10, 100)
                with r4:
                    st.markdown(
                        f'<div style="padding:8px 0;">'
                        f'<div style="color:{CARD_TEXT_MUTED};font-size:0.72em;text-transform:uppercase;'
                        f'letter-spacing:0.04em;margin-bottom:5px;">Quality Score</div>'
                        f'<div style="display:flex;align-items:center;gap:6px;">'
                        f'<div style="flex:1;background:#2a2a3e;border-radius:3px;height:5px;">'
                        f'<div style="width:{_qs_pct:.0f}%;height:100%;background:{_qs_color};border-radius:3px;"></div>'
                        f'</div>'
                        f'<span style="font-size:0.82em;color:{_qs_color};font-weight:600;">{qs:.1f}</span>'
                        f'</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                # ── Box office (enrichment) if available ──
                if _has_enr and "box_office_usd" in _enr_row.columns:
                    _bo = _enr.get("box_office_usd")
                    if _bo and str(_bo) != "nan" and _bo > 0:
                        if _bo >= 1e9:
                            _bo_str = f"${_bo/1e9:.1f}B"
                        elif _bo >= 1e6:
                            _bo_str = f"${_bo/1e6:.0f}M"
                        else:
                            _bo_str = f"${_bo:,.0f}"
                        st.markdown(
                            f'<div style="color:{CARD_TEXT_MUTED};font-size:0.78em;margin-top:2px;">'
                            f'Box Office (Wikidata): <strong style="color:{CARD_TEXT};">{_bo_str}</strong>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                # ── Genre pills ──
                if isinstance(sel["genres"], list) and sel["genres"]:
                    genre_pills = " ".join(
                        f'<span style="background:{CARD_BORDER};color:{CARD_TEXT};'
                        f'padding:3px 10px;border-radius:12px;font-size:0.82em;margin-right:3px;">'
                        f'{g.title()}</span>'
                        for g in sel["genres"]
                    )
                    st.markdown(f'<div style="margin:8px 0;">{genre_pills}</div>', unsafe_allow_html=True)

                # ── Description with max-height + read-more expander ──
                st.divider()
                desc = sel.get("description", "")
                if desc and desc == desc and len(str(desc)) > 0:
                    desc_str = str(desc)
                    st.markdown(
                        f'<div style="font-size:0.9em;color:{CARD_TEXT};line-height:1.6;">{desc_str}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption("No description available.")

                # ── Cast & Crew expander ──
                with st.expander("Cast & Crew"):
                    credits_df = get_credits_for_view(filters["platform_view"])
                    title_credits = credits_df[credits_df["title_id"] == selected_id]
                    # Fallback to all-platforms credits for competitor titles
                    if title_credits.empty:
                        from src.data.loaders import load_all_platforms_credits
                        _all_credits = load_all_platforms_credits()
                        title_credits = _all_credits[_all_credits["title_id"] == selected_id]

                    directors = title_credits[title_credits["role"] == "DIRECTOR"]
                    actors = title_credits[title_credits["role"] == "ACTOR"]

                    if not directors.empty:
                        dir_names = directors["name"].drop_duplicates().tolist()
                        dir_label = "Directors" if len(dir_names) > 1 else "Director"
                        st.markdown(
                            f'<div style="color:{CARD_TEXT_MUTED};font-size:0.75em;text-transform:uppercase;'
                            f'letter-spacing:0.04em;margin-bottom:4px;">{dir_label}</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(f"**{', '.join(dir_names)}**")

                    if not actors.empty:
                        st.markdown(
                            f'<div style="color:{CARD_TEXT_MUTED};font-size:0.75em;text-transform:uppercase;'
                            f'letter-spacing:0.04em;margin:8px 0 4px;">'
                            f'Cast</div>',
                            unsafe_allow_html=True,
                        )
                        actor_list = actors.head(16).reset_index(drop=True)
                        # 2-column grid
                        n_actors = len(actor_list)
                        ac_col1, ac_col2 = st.columns(2)
                        for idx, (_, actor) in enumerate(actor_list.iterrows()):
                            char = actor.get("character", "")
                            name = actor["name"]
                            char_html = (
                                f'<span style="color:{CARD_TEXT_MUTED};font-size:0.8em;"> as {char}</span>'
                                if char and str(char) not in ("nan", "")
                                else ""
                            )
                            entry_html = (
                                f'<div style="font-size:0.85em;color:{CARD_TEXT};padding:2px 0;">'
                                f'<strong>{name}</strong>{char_html}</div>'
                            )
                            with (ac_col1 if idx % 2 == 0 else ac_col2):
                                st.markdown(entry_html, unsafe_allow_html=True)
                        if len(actors) > 16:
                            st.caption(f"+{len(actors) - 16} more cast members")

                    if directors.empty and actors.empty:
                        st.caption("No cast & crew information available for this title.")

                    st.page_link(
                        "pages/07_Cast_Crew_Network.py",
                        label="View full network →",
                    )

            # ── Similar Titles ──
            st.divider()
            st.markdown(
                section_header_html("Similar Titles", "Multi-signal recommendations based on description, genre, type, and quality"),
                unsafe_allow_html=True,
            )

            scope_options = {
                "Merged catalog (Netflix + Max)": "merged",
                "All platforms": "all_platforms",
            }
            scope_label = st.radio(
                "Recommendation scope",
                options=list(scope_options.keys()),
                index=0 if st.session_state["explore_sim_scope"] == "merged" else 1,
                horizontal=True,
                key="_sim_scope_radio",
            )
            scope = scope_options[scope_label]
            st.session_state["explore_sim_scope"] = scope

            sim_df = load_similarity_data()
            scope_titles = load_merged_titles() if scope == "merged" else load_all_platforms_titles()

            similar = get_similar_titles(
                title_id=selected_id,
                similarity_df=sim_df,
                titles_df=scope_titles,
                top_k=SIMILARITY_TOP_K,
            )

            if similar.empty:
                st.caption("No similar titles found for this selection.")
            else:
                similar = similar.copy()
                similar["quality_score"] = compute_quality_score(similar)

                # 2-column grid
                sim_list = [similar.iloc[i] for i in range(len(similar))]
                for pair_start in range(0, len(sim_list), 2):
                    pair = sim_list[pair_start: pair_start + 2]
                    scols = st.columns(2)
                    for col, sim_row in zip(scols, pair):
                        sim_id = sim_row["similar_id"]
                        sim_score = sim_row["similarity_score"]
                        sim_imdb = sim_row["imdb_score"]
                        sim_imdb_str = f"{sim_imdb:.1f}" if sim_imdb == sim_imdb else "N/A"
                        sim_plat_badge = platform_badges_html(
                            sim_row.get("platforms", sim_row.get("platform", ""))
                        )
                        year_str = (
                            str(int(sim_row["release_year"]))
                            if sim_row["release_year"] == sim_row["release_year"]
                            else "?"
                        )
                        # Colored similarity badge
                        if sim_score >= 0.75:
                            sim_bg, sim_fg = "rgba(46,204,113,0.18)", "#2ecc71"
                        elif sim_score >= 0.60:
                            sim_bg, sim_fg = "rgba(243,156,18,0.18)", "#f39c12"
                        else:
                            sim_bg, sim_fg = "rgba(136,136,136,0.18)", CARD_TEXT_MUTED

                        with col:
                            st.markdown(
                                f'<div style="background:{CARD_BG};border-radius:8px;'
                                f'padding:9px 11px;margin-bottom:4px;border:1px solid {CARD_BORDER};">'
                                f'<div style="font-size:0.87em;font-weight:600;color:{CARD_TEXT};">'
                                f'{sim_row["title"]} '
                                f'<span style="color:{CARD_TEXT_MUTED};font-weight:400;">({year_str})</span>'
                                f'</div>'
                                f'<div style="display:flex;align-items:center;gap:5px;margin-top:4px;flex-wrap:wrap;">'
                                f'{sim_plat_badge}'
                                f'<span style="color:{CARD_TEXT_MUTED};font-size:0.75em;">IMDb {sim_imdb_str}</span>'
                                f'<span style="background:{sim_bg};color:{sim_fg};padding:2px 7px;'
                                f'border-radius:10px;font-size:0.72em;font-weight:600;">'
                                f'{sim_score:.0%} match</span>'
                                f'</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                            if st.button(
                                "Open",
                                key=f"sim_{sim_row.name}",
                                use_container_width=True,
                            ):
                                st.session_state["explore_selected_id"] = sim_id
                                st.rerun()

                # Discovery Engine CTA
                st.markdown(
                    f'<div style="margin-top:8px;">'
                    f'<a href="/04_Discovery_Engine" target="_self" style="display:inline-block;'
                    f'background:{CARD_BG};border:1px solid {_MERGED_COLOR};color:{_MERGED_COLOR};'
                    f'padding:7px 16px;border-radius:6px;font-size:0.85em;font-weight:600;'
                    f'text-decoration:none;">✨ Deeper recommendations → Discovery Engine</a>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                f'<div style="border-top:1px solid {CARD_BORDER};margin-top:20px;padding-top:4px;"></div>',
                unsafe_allow_html=True,
            )
            with st.expander("How Similar Titles Work"):
                st.markdown(
                    """Recommendations combine **four similarity signals** into a weighted score:

| Signal | Weight | Method |
|--------|--------|--------|
| Description | 30% | TF-IDF cosine similarity on plot descriptions |
| Genre | 30% | Cosine similarity on genre vectors |
| Type | 15% | Same content type bonus (Show/Movie) |
| Quality | 25% | IMDb score proximity |

- Only titles with IMDb ≥ 6.0 are recommended.
- The top 10 most similar titles are shown, filtered to the selected scope.
- Scores are precomputed offline for sub-200ms response times."""
                )

# ── footer ────────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    '<div style="color:#555;font-size:0.8em;text-align:center;padding:8px 0 16px;">'
    'Hypothetical merger for academic analysis. '
    'Data is a snapshot (mid-2023). '
    'All insights are illustrative, not prescriptive.'
    '</div>',
    unsafe_allow_html=True,
)
