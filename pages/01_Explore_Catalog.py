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
    get_credits_for_view,
    get_titles_for_view,
    load_all_platforms_titles,
    load_merged_titles,
    load_similarity_data,
)
from src.ui.filters import apply_filters, render_sidebar_filters
from src.ui.session import DEFAULTS, init_session_state


def _platform_badge(key: str) -> str:
    """Return an HTML badge for a platform."""
    meta = PLATFORMS.get(key, {})
    name = meta.get("name", key.title())
    bg = meta.get("color", "#555")
    fg = meta.get("text_color", "#FFF")
    return (
        f'<span style="background:{bg};color:{fg};padding:2px 8px;'
        f'border-radius:4px;font-size:0.75em;font-weight:600;'
        f'letter-spacing:0.02em;">{name}</span>'
    )


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

# Page-specific session state (not in global DEFAULTS — unaffected by Reset filters)
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

st.title("Explore Catalog")
st.caption("Search, browse, and discover similar titles across the catalog")

# ── active filter indicator ──────────────────────────────────────────────────

active_pills = []
if filters["platform_view"] != DEFAULTS["platform_view"]:
    active_pills.append(
        f'Platform: {_PLATFORM_LABELS.get(filters["platform_view"], filters["platform_view"])}'
    )
if set(filters["content_types"]) != set(DEFAULTS["content_types"]):
    active_pills.append(f'Type: {", ".join(filters["content_types"])}')
if filters["year_range"] != tuple(DEFAULTS["year_range"]):
    active_pills.append(
        f'Years: {filters["year_range"][0]}\u2013{filters["year_range"][1]}'
    )
if filters["min_imdb"] > 0:
    active_pills.append(f'IMDb \u2265 {filters["min_imdb"]:.1f}')
if filters["selected_genres"]:
    genre_str = ", ".join(g.title() for g in filters["selected_genres"][:5])
    if len(filters["selected_genres"]) > 5:
        genre_str += f' +{len(filters["selected_genres"]) - 5} more'
    active_pills.append(f"Genres: {genre_str}")

if active_pills:
    pills_html = " ".join(
        f'<span style="background:{CARD_BORDER};color:{CARD_TEXT};'
        f'padding:4px 12px;border-radius:16px;font-size:0.82em;'
        f'margin-right:4px;display:inline-block;margin-bottom:4px;">{p}</span>'
        for p in active_pills
    )
    st.markdown(
        f'<div style="margin-bottom:12px;">'
        f'<span style="color:{CARD_TEXT_MUTED};font-size:0.82em;margin-right:8px;">'
        f'Active filters:</span>{pills_html}</div>',
        unsafe_allow_html=True,
    )

# ── page-specific controls (search + sort) ───────────────────────────────────

ctrl_left, ctrl_right = st.columns([3, 1])

with ctrl_left:
    search_text = st.text_input(
        "Search titles",
        value=st.session_state["explore_search"],
        placeholder="Search by title, cast, or keyword...",
        help="Searches across titles, descriptions, and cast names",
        key="_explore_search_input",
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
    )
    st.session_state["explore_sort"] = sort_by

# ── compute quality scores ──────────────────────────────────────────────────

df = df.copy()
df["quality_score"] = compute_quality_score(df)

# ── quick suggestions (title matches before full search is applied) ─────────

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

# Multi-field search (title + description + cast names)
if search_text.strip():
    query = search_text.strip()
    title_mask = df["title"].str.contains(query, case=False, na=False, regex=False)
    desc_mask = df["description"].fillna("").str.contains(
        query, case=False, na=False, regex=False
    )
    # Cast name search
    _credits = get_credits_for_view(filters["platform_view"])
    _cast_ids = set(
        _credits[
            _credits["name"].str.contains(query, case=False, na=False, regex=False)
        ]["title_id"]
    )
    cast_mask = df["id"].isin(_cast_ids)

    df = df[title_mask | desc_mask | cast_mask].reset_index(drop=True)

# Sort
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
df = df.sort_values(sort_col, ascending=sort_asc, na_position="last").reset_index(
    drop=True
)

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

# ── LEFT PANEL: paginated results list ──────────────────────────────────────

with left_col:
    # Items per page + result range
    pps_col, range_col = st.columns([1, 2])
    with pps_col:
        page_size_options = [25, 50, 100]
        new_page_size = st.selectbox(
            "Per page",
            options=page_size_options,
            index=page_size_options.index(page_size),
            key="_explore_page_size_input",
        )
        if new_page_size != page_size:
            st.session_state["explore_page_size"] = new_page_size
            st.session_state["explore_page"] = 0
            st.rerun()
    with range_col:
        if total_results > 0:
            range_text = (
                f"Showing {start_idx + 1}\u2013{end_idx} of {total_results:,}"
            )
        else:
            range_text = "No results"
        st.markdown(
            f"<div style='padding:28px 0 0;color:{CARD_TEXT_MUTED};font-size:0.85em;'>"
            f"{range_text}</div>",
            unsafe_allow_html=True,
        )

    # Navigation: First / Prev / Page X of Y / Next / Last
    nav_1, nav_2, nav_3, nav_4, nav_5 = st.columns([1, 1, 2, 1, 1])
    with nav_1:
        if st.button(
            "\u23ee First",
            disabled=(current_page == 0),
            use_container_width=True,
        ):
            st.session_state["explore_page"] = 0
            st.rerun()
    with nav_2:
        if st.button(
            "\u25c4 Prev",
            disabled=(current_page == 0),
            use_container_width=True,
        ):
            st.session_state["explore_page"] = current_page - 1
            st.rerun()
    with nav_3:
        st.markdown(
            f"<div style='text-align:center;padding:8px;color:{CARD_TEXT_MUTED};'>"
            f"Page {current_page + 1} of {total_pages}</div>",
            unsafe_allow_html=True,
        )
    with nav_4:
        if st.button(
            "Next \u25ba",
            disabled=(current_page >= total_pages - 1),
            use_container_width=True,
        ):
            st.session_state["explore_page"] = current_page + 1
            st.rerun()
    with nav_5:
        if st.button(
            "Last \u23ed",
            disabled=(current_page >= total_pages - 1),
            use_container_width=True,
        ):
            st.session_state["explore_page"] = total_pages - 1
            st.rerun()

    # Jump to page
    jump_col, go_col = st.columns([3, 1])
    with jump_col:
        jump_val = st.number_input(
            "Go to page",
            min_value=1,
            max_value=total_pages,
            value=current_page + 1,
            step=1,
            key="_explore_jump_page",
        )
    with go_col:
        st.markdown(
            "<div style='padding:28px 0 0;'></div>", unsafe_allow_html=True
        )
        if st.button("Go", key="_explore_jump_go", use_container_width=True):
            target = int(jump_val) - 1
            if target != current_page:
                st.session_state["explore_page"] = target
                st.rerun()

    st.divider()

    # Results list
    if len(page_df) == 0:
        if search_text.strip():
            st.markdown(
                f"<div style='text-align:center;padding:32px 16px;'>"
                f"<div style='font-size:1.1em;color:{CARD_TEXT};margin-bottom:8px;'>"
                f"No titles found matching '<strong>{search_text.strip()}</strong>'</div>"
                f"<div style='font-size:0.88em;color:{CARD_TEXT_MUTED};'>"
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
                if "quality_score" in row.index
                and row["quality_score"] == row["quality_score"]
                else "N/A"
            )
            genres_str = ""
            if isinstance(row["genres"], list):
                genres_str = ", ".join(g.title() for g in row["genres"][:3])

            year_str = (
                str(int(row["release_year"]))
                if row["release_year"] == row["release_year"]
                else "?"
            )
            plat_badge = _platform_badge(row["platform"])

            # Selected state: gold border + subtle glow; default: hover effects
            if is_selected:
                border_css = f"2px solid {CARD_ACCENT}"
                shadow_css = f"0 0 8px {CARD_ACCENT}44"
                hover_js = ""
            else:
                border_css = f"1px solid {CARD_BORDER}"
                shadow_css = "none"
                hover_js = (
                    'onmouseover="this.style.transform=\'translateY(-1px)\';'
                    "this.style.boxShadow='0 2px 8px rgba(0,0,0,0.25)'\""
                    ' onmouseout="this.style.transform=\'none\';'
                    "this.style.boxShadow='none'\""
                )

            st.markdown(
                f"""<div style="background:{CARD_BG};border-radius:8px;padding:10px 12px;
                margin-bottom:4px;border:{border_css};box-shadow:{shadow_css};
                transition:transform 0.15s ease,box-shadow 0.15s ease;"
                {hover_js}>
                <div style="display:flex;align-items:center;gap:6px;font-size:0.9em;font-weight:600;color:{CARD_TEXT};">
                {row['title']} <span style="color:{CARD_TEXT_MUTED};font-weight:400;">({year_str})</span>
                {plat_badge}</div>
                <div style="font-size:0.78em;color:{CARD_TEXT_MUTED};margin-top:3px;">
                {row['type']} | IMDb {imdb_str} | QS <span style="color:{CARD_ACCENT};font-weight:600;">{qs_str}</span>/10 | {genres_str}</div>
                </div>""",
                unsafe_allow_html=True,
            )
            if st.button(
                "View",
                key=f"sel_{row_idx}_{current_page}",
                use_container_width=True,
            ):
                st.session_state["explore_selected_id"] = title_id
                st.rerun()

# ── RIGHT PANEL: detail view + similar titles ───────────────────────────────

with right_col:
    selected_id = st.session_state["explore_selected_id"]

    if selected_id is None:
        st.info(
            "Select a title from the results list to see details and similar titles."
        )
    else:
        # Look up the selected title
        sel_rows = df[df["id"] == selected_id]
        if sel_rows.empty:
            sel_rows = raw_df[raw_df["id"] == selected_id]
            if not sel_rows.empty:
                sel_rows = sel_rows.copy()
                sel_rows["quality_score"] = compute_quality_score(sel_rows)
        if sel_rows.empty:
            st.warning(
                "Selected title not found in current view. Try changing filters."
            )
        else:
            sel = sel_rows.iloc[0]

            with st.container(border=True):
                # ── Detail card ──
                st.subheader(sel["title"])

                # Metadata row
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Type", sel["type"])
                with m2:
                    year_val = (
                        int(sel["release_year"])
                        if sel["release_year"] == sel["release_year"]
                        else "N/A"
                    )
                    st.metric("Year", year_val)
                with m3:
                    imdb_val = (
                        f"{sel['imdb_score']:.1f}"
                        if sel["imdb_score"] == sel["imdb_score"]
                        else "N/A"
                    )
                    st.metric("IMDb", imdb_val)
                with m4:
                    st.markdown(
                        f"<div style='font-size:0.82em;color:{CARD_TEXT_MUTED};margin-bottom:4px;'>Platform</div>"
                        f"<div>{_platform_badge(sel['platform'])}</div>",
                        unsafe_allow_html=True,
                    )

                # Genre pills
                if isinstance(sel["genres"], list) and sel["genres"]:
                    genre_pills = " ".join(
                        f'<span style="background:{CARD_BORDER};color:{CARD_TEXT};'
                        f'padding:3px 10px;border-radius:12px;font-size:0.82em;'
                        f'margin-right:4px;">{g.title()}</span>'
                        for g in sel["genres"]
                    )
                    st.markdown(genre_pills, unsafe_allow_html=True)

                # Additional metadata
                captions = []
                if (
                    "age_certification" in sel.index
                    and sel["age_certification"]
                    and sel["age_certification"] == sel["age_certification"]
                ):
                    captions.append(f"Rating: {sel['age_certification']}")
                if "runtime" in sel.index and sel["runtime"] == sel["runtime"]:
                    captions.append(f"Runtime: {int(sel['runtime'])} min")
                if "imdb_votes" in sel.index and sel["imdb_votes"] == sel["imdb_votes"]:
                    captions.append(f"Votes: {format_votes(sel['imdb_votes'])}")
                if "tmdb_popularity" in sel.index and sel["tmdb_popularity"] == sel["tmdb_popularity"]:
                    pop = sel["tmdb_popularity"]
                    if pop >= 1_000:
                        pop_str = f"{pop / 1_000:.1f}K"
                    else:
                        pop_str = f"{pop:.0f}"
                    captions.append(f"Popularity: {pop_str}")
                if captions:
                    st.caption(" | ".join(captions))

                # Quality score
                qs = (
                    sel["quality_score"]
                    if "quality_score" in sel.index
                    else compute_quality_score(sel.to_frame().T).iloc[0]
                )
                st.markdown(
                    f"**Quality Score:** <span style='color:{CARD_ACCENT};font-weight:700;'>"
                    f"{qs:.1f}</span>/10",
                    unsafe_allow_html=True,
                )

                # Description
                st.divider()
                desc = sel.get("description", "")
                if desc and desc == desc:
                    st.markdown(desc)
                else:
                    st.caption("No description available.")

                # ── Cast & Crew ──
                with st.expander("Cast & Crew"):
                    credits_df = get_credits_for_view(filters["platform_view"])
                    title_credits = credits_df[
                        credits_df["title_id"] == selected_id
                    ]

                    directors = title_credits[
                        title_credits["role"] == "DIRECTOR"
                    ]
                    actors = title_credits[title_credits["role"] == "ACTOR"]

                    if not directors.empty:
                        dir_names = (
                            directors["name"].drop_duplicates().tolist()
                        )
                        st.markdown(
                            f"**Director{'s' if len(dir_names) > 1 else ''}:** "
                            f"{', '.join(dir_names)}"
                        )

                    if not actors.empty:
                        st.markdown("**Cast**")
                        for _, actor in actors.head(15).iterrows():
                            char = actor["character"]
                            if char and str(char) not in ("nan", ""):
                                st.markdown(
                                    f"<div style='font-size:0.88em;color:{CARD_TEXT};padding:2px 0;'>"
                                    f"<strong>{actor['name']}</strong>"
                                    f" <span style='color:{CARD_TEXT_MUTED};'>as {char}</span></div>",
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.markdown(
                                    f"<div style='font-size:0.88em;color:{CARD_TEXT};padding:2px 0;'>"
                                    f"<strong>{actor['name']}</strong></div>",
                                    unsafe_allow_html=True,
                                )
                        if len(actors) > 15:
                            st.caption(
                                f"+{len(actors) - 15} more cast members"
                            )

                    if directors.empty and actors.empty:
                        st.caption(
                            "No cast & crew information available for this title."
                        )

            # ── Similar Titles ──
            st.divider()
            st.subheader("Similar Titles")

            scope_options = {
                "Merged catalog (Netflix + Max)": "merged",
                "All platforms": "all_platforms",
            }
            scope_label = st.radio(
                "Recommendation scope",
                options=list(scope_options.keys()),
                index=0
                if st.session_state["explore_sim_scope"] == "merged"
                else 1,
                horizontal=True,
                key="_sim_scope_radio",
            )
            scope = scope_options[scope_label]
            st.session_state["explore_sim_scope"] = scope

            # Load similarity data (cached)
            sim_df = load_similarity_data()

            # Choose titles pool based on scope
            if scope == "merged":
                scope_titles = load_merged_titles()
            else:
                scope_titles = load_all_platforms_titles()

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

                for sim_idx, (_, sim_row) in enumerate(similar.iterrows()):
                    sim_id = sim_row["similar_id"]
                    sim_score = sim_row["similarity_score"]
                    sim_imdb = sim_row["imdb_score"]
                    sim_imdb_str = (
                        f"{sim_imdb:.1f}" if sim_imdb == sim_imdb else "N/A"
                    )
                    sim_plat_badge = _platform_badge(sim_row["platform"])

                    year_str = (
                        str(int(sim_row["release_year"]))
                        if sim_row["release_year"] == sim_row["release_year"]
                        else "?"
                    )

                    st.markdown(
                        f"""<div style="background:{CARD_BG};border-radius:8px;padding:10px 12px;
                        margin-bottom:4px;border:1px solid {CARD_BORDER};
                        transition:transform 0.15s ease,box-shadow 0.15s ease;"
                        onmouseover="this.style.transform='translateY(-1px)';this.style.boxShadow='0 2px 8px rgba(0,0,0,0.2)'"
                        onmouseout="this.style.transform='none';this.style.boxShadow='none'">
                        <div style="display:flex;align-items:center;gap:6px;font-size:0.9em;font-weight:600;color:{CARD_TEXT};">
                        {sim_row['title']} <span style="color:{CARD_TEXT_MUTED};font-weight:400;">({year_str})</span>
                        {sim_plat_badge}</div>
                        <div style="font-size:0.78em;color:{CARD_TEXT_MUTED};margin-top:3px;">
                        IMDb {sim_imdb_str} |
                        Similarity: <span style="color:{CARD_ACCENT};font-weight:600;">{sim_score:.0%}</span></div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                    if st.button(
                        "View",
                        key=f"sim_{sim_idx}",
                        use_container_width=True,
                    ):
                        st.session_state["explore_selected_id"] = sim_id
                        st.rerun()

            with st.expander("How Similar Titles Work"):
                st.markdown(
                    """Recommendations combine **four similarity signals** into a weighted score:

| Signal | Weight | Method |
|--------|--------|--------|
| Description | 30% | TF-IDF cosine similarity on plot descriptions |
| Genre | 30% | Cosine similarity on genre vectors |
| Type | 15% | Same content type bonus (Show/Movie) |
| Quality | 25% | IMDb score proximity |

- Only titles with IMDb >= 6.0 are recommended.
- The top 10 most similar titles are shown, filtered to the selected scope.
- Scores are precomputed offline for sub-200ms response times."""
                )

# ── footer ────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "Hypothetical merger for academic analysis. "
    "Data is a snapshot (mid-2023). "
    "All insights are illustrative, not prescriptive."
)
