"""Home page — executive overview dashboard for Netflix + Max merger analysis."""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analysis.scoring import compute_quality_score, format_votes
from src.config import (
    CARD_ACCENT,
    CARD_BG,
    CARD_BORDER,
    CARD_TEXT,
    CARD_TEXT_MUTED,
    CHART_HEIGHT,
    DECADE_LABELS,
    INITIAL_SIDEBAR_STATE,
    LAYOUT,
    PAGE_ICON,
    PAGE_TITLE,
    PLATFORMS,
    PLOTLY_TEMPLATE,
)
from src.data.loaders import (
    deduplicate_titles,
    get_credits_for_view,
    get_titles_for_view,
    load_enriched_titles,
    load_merged_credits,
    load_merged_titles,
)
from src.ui.badges import (
    page_header_html,
    platform_badges_html,
    section_header_html,
    styled_banner_html,
    styled_metric_card_html,
)
from src.ui.filters import apply_filters, render_sidebar_filters
from src.ui.session import init_session_state

_MERGED_COLOR = PLATFORMS["merged"]["color"]
_NETFLIX_COLOR = PLATFORMS["netflix"]["color"]
_MAX_COLOR = PLATFORMS["max"]["color"]

# Genre key corrections for display
_GENRE_DISPLAY = {"Documentation": "Documentary", "Scifi": "Sci-Fi"}


def _platform_name(key: str) -> str:
    """Convert platform key to display name: 'netflix' → 'Netflix'."""
    return PLATFORMS.get(key, {}).get("name", key.title())


_COUNTRY_NAMES = {
    "US": "United States", "GB": "United Kingdom", "IN": "India",
    "FR": "France", "JP": "Japan", "CA": "Canada", "ES": "Spain",
    "KR": "South Korea", "DE": "Germany", "MX": "Mexico",
    "IT": "Italy", "BR": "Brazil", "CN": "China", "AU": "Australia",
    "NG": "Nigeria", "TR": "Turkey", "AR": "Argentina", "PH": "Philippines",
    "EG": "Egypt", "PL": "Poland", "TH": "Thailand", "SE": "Sweden",
    "DK": "Denmark", "NO": "Norway", "BE": "Belgium", "NL": "Netherlands",
    "ZA": "South Africa", "CO": "Colombia", "ID": "Indonesia", "RU": "Russia",
}

# ── page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state=INITIAL_SIDEBAR_STATE,
)

init_session_state()
st.session_state.setdefault("home_selected_title_id", None)

# ── data loading ─────────────────────────────────────────────────────────────

raw_df = get_titles_for_view(st.session_state["platform_view"])
filters = render_sidebar_filters(raw_df)
raw_df = get_titles_for_view(filters["platform_view"])
df = apply_filters(raw_df, filters)
credits_df = get_credits_for_view(filters["platform_view"])

# ── title ────────────────────────────────────────────────────────────────────

st.markdown(
    page_header_html(
        "Netflix + Max Merger Analysis",
        "Hypothetical merger analysis across streaming platforms",
    ),
    unsafe_allow_html=True,
)

# ── precompute baselines for deltas ─────────────────────────────────────────

merged_all = load_merged_titles()
merged_credits_all = load_merged_credits()
netflix_df = merged_all[merged_all["platform"] == "netflix"]
max_df = merged_all[merged_all["platform"] == "max"]

_nf_catalog = netflix_df["id"].nunique()
_nf_avg_imdb = netflix_df["imdb_score"].mean()
_nf_credits = merged_credits_all
_nf_people = _nf_credits[_nf_credits["platform"] == "netflix"]["person_id"].nunique()
_nf_genres = len(set(g for gs in netflix_df["genres"].dropna() if isinstance(gs, list) for g in gs))

# ── section 1: hero metrics ─────────────────────────────────────────────────

st.markdown(
    section_header_html("Overview", "Key metrics showing the combined strength of the Netflix + Max merger"),
    unsafe_allow_html=True,
)

catalog_size = df["id"].nunique()
avg_imdb = df["imdb_score"].mean()
people_count = credits_df["person_id"].nunique()
genre_count = len(set(g for gs in df["genres"].dropna() if isinstance(gs, list) for g in gs))


def _pct_delta(current, baseline):
    if baseline == 0:
        return None
    pct = ((current - baseline) / baseline) * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.0f}% vs Netflix"


def _abs_delta(current, baseline):
    diff = current - baseline
    return f"{diff:+.2f} vs Netflix"


enriched = load_enriched_titles()
_award_coverage = enriched["award_wins"].notna().mean() if "award_wins" in enriched.columns else 0
_has_awards = _award_coverage >= 0.20

hero_count = 5 if _has_awards else 4
hero_cols = st.columns(hero_count)

imdb_str = f"{avg_imdb:.2f}" if avg_imdb == avg_imdb else "N/A"

with hero_cols[0]:
    st.markdown(
        styled_metric_card_html(
            "Catalog Size",
            f"{catalog_size:,}",
            delta=_pct_delta(catalog_size, _nf_catalog),
            help_text="Total unique titles in the current view. Delta shows gain vs Netflix alone.",
        ),
        unsafe_allow_html=True,
    )
with hero_cols[1]:
    st.markdown(
        styled_metric_card_html(
            "Avg IMDb Score",
            imdb_str,
            delta=_abs_delta(avg_imdb, _nf_avg_imdb) if avg_imdb == avg_imdb else None,
            help_text="Mean IMDb rating across all titles (excluding unrated). Delta shows shift vs Netflix alone.",
        ),
        unsafe_allow_html=True,
    )
with hero_cols[2]:
    st.markdown(
        styled_metric_card_html(
            "Cast & Crew",
            f"{people_count:,}",
            delta=_pct_delta(people_count, _nf_people),
            help_text="Unique actors, directors, and crew across all credits. Delta shows gain vs Netflix alone.",
        ),
        unsafe_allow_html=True,
    )
with hero_cols[3]:
    st.markdown(
        styled_metric_card_html(
            "Genre Coverage",
            f"{genre_count:,}",
            delta=_pct_delta(genre_count, _nf_genres),
            help_text="Number of distinct genres represented. Delta shows expansion vs Netflix alone.",
        ),
        unsafe_allow_html=True,
    )

if _has_awards:
    with hero_cols[4]:
        _award_titles = int((enriched["award_wins"].fillna(0) > 0).sum())
        st.markdown(
            styled_metric_card_html(
                "Award-Winning Titles",
                f"{_award_titles:,}",
                subtitle=f"(Wikidata · {_award_coverage:.0%} coverage)",
                help_text=f"Titles with at least one award win tracked by Wikidata ({_award_coverage:.0%} coverage).",
            ),
            unsafe_allow_html=True,
        )

# Insight banner
_added_titles = catalog_size - _nf_catalog
_pct_increase = ((_added_titles / _nf_catalog) * 100) if _nf_catalog > 0 else 0
_insight_parts = [
    f"The merger adds <strong>{_added_titles:,} titles</strong> to Netflix's catalog, "
    f"a <strong>{_pct_increase:.0f}% increase</strong>",
]
if avg_imdb == avg_imdb and _nf_avg_imdb == _nf_avg_imdb:
    _direction = "improving" if avg_imdb >= _nf_avg_imdb else "shifting"
    _insight_parts.append(
        f" while {_direction} average quality from "
        f"<strong>{_nf_avg_imdb:.2f}</strong> to <strong>{avg_imdb:.2f}</strong>."
    )
else:
    _insight_parts.append(".")

st.markdown(
    styled_banner_html("ℹ️", "".join(_insight_parts)),
    unsafe_allow_html=True,
)

# ── section 2: merger impact ────────────────────────────────────────────────

st.divider()
st.markdown(
    section_header_html(
        "Merger Impact",
        "How combining Netflix and Max catalogs reshapes volume, quality, and genre diversity",
    ),
    unsafe_allow_html=True,
)

mi1, mi2, mi3 = st.columns(3)

# Volume boost
with mi1:
    st.markdown(
        f'<div style="font-weight:600;color:{CARD_TEXT};margin-bottom:2px;">Catalog Size: Netflix vs Max vs Merged</div>',
        unsafe_allow_html=True,
    )
    _n_nf = netflix_df["id"].nunique()
    _n_max = max_df["id"].nunique()
    _n_merged = merged_all["id"].nunique()
    vol_data = {
        "Platform": ["Netflix", "Max", "Netflix + Max"],
        "Titles": [_n_nf, _n_max, _n_merged],
    }
    fig_vol = go.Figure(go.Bar(
        x=vol_data["Platform"],
        y=vol_data["Titles"],
        marker_color=[_NETFLIX_COLOR, _MAX_COLOR, _MERGED_COLOR],
        text=vol_data["Titles"],
        texttemplate="%{text:,}",
        textposition="outside",
    ))
    fig_vol.update_layout(
        template=PLOTLY_TEMPLATE,
        height=CHART_HEIGHT,
        showlegend=False,
        xaxis_title="",
        yaxis_title="Titles",
        yaxis=dict(range=[0, max(vol_data["Titles"]) * 1.18]),
    )
    st.plotly_chart(fig_vol, use_container_width=True)
    st.caption(
        f"The merged catalog gains {_n_merged - _n_nf:,} net titles over Netflix alone "
        f"(+{(_n_merged - _n_nf) / _n_nf * 100:.0f}%), with overlap removing duplicates."
    )

# Quality shift
with mi2:
    st.markdown(
        f'<div style="font-weight:600;color:{CARD_TEXT};margin-bottom:2px;">IMDb Score Distribution: Netflix vs Merged</div>',
        unsafe_allow_html=True,
    )
    netflix_scores = netflix_df["imdb_score"].dropna()
    merged_scores = merged_all["imdb_score"].dropna()
    fig_qual = go.Figure()
    fig_qual.add_trace(go.Histogram(
        x=netflix_scores,
        name="Netflix",
        marker_color=_NETFLIX_COLOR,
        opacity=0.55,
        nbinsx=20,
        histnorm="probability density",
    ))
    fig_qual.add_trace(go.Histogram(
        x=merged_scores,
        name="Netflix + Max",
        marker_color=_MERGED_COLOR,
        opacity=0.55,
        nbinsx=20,
        histnorm="probability density",
    ))
    fig_qual.update_layout(
        barmode="overlay",
        template=PLOTLY_TEMPLATE,
        height=CHART_HEIGHT,
        xaxis_title="IMDb Score",
        yaxis_title="Density",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    st.plotly_chart(fig_qual, use_container_width=True)
    _max_only_scores = max_df["imdb_score"].dropna()
    _max_mean = _max_only_scores.mean()
    _dir = "higher" if _max_mean >= _nf_avg_imdb else "lower"
    st.caption(
        f"Max's avg IMDb ({_max_mean:.2f}) is {_dir} than Netflix's ({_nf_avg_imdb:.2f}), "
        f"shifting the merged distribution's shape."
    )

# Genre expansion
with mi3:
    st.markdown(
        f'<div style="font-weight:600;color:{CARD_TEXT};margin-bottom:2px;">Top Genres: Netflix vs Merged Entity</div>',
        unsafe_allow_html=True,
    )

    def _top_genres(frame, n=10):
        from collections import Counter
        counter = Counter()
        for genres in frame["genres"].dropna():
            if isinstance(genres, list):
                counter.update(genres)
        return counter.most_common(n)

    netflix_genres = dict(_top_genres(netflix_df))
    merged_genres = dict(_top_genres(merged_all))
    all_top_raw = sorted(set(list(netflix_genres.keys()) + list(merged_genres.keys())))[:12]
    # Fix display labels
    all_top_display = [_GENRE_DISPLAY.get(g.title(), g.title()) for g in all_top_raw]

    fig_genre = go.Figure()
    fig_genre.add_trace(go.Bar(
        x=all_top_display,
        y=[netflix_genres.get(g, 0) for g in all_top_raw],
        name="Netflix",
        marker_color=_NETFLIX_COLOR,
    ))
    fig_genre.add_trace(go.Bar(
        x=all_top_display,
        y=[merged_genres.get(g, 0) for g in all_top_raw],
        name="Netflix + Max",
        marker_color=_MERGED_COLOR,
    ))
    fig_genre.update_layout(
        barmode="group",
        template=PLOTLY_TEMPLATE,
        height=CHART_HEIGHT,
        xaxis_title="",
        yaxis_title="Titles",
        xaxis_tickangle=-45,
        xaxis_automargin=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    st.plotly_chart(fig_genre, use_container_width=True)
    # Find genre where Max adds the most
    _top_gain_genre = max(all_top_raw, key=lambda g: merged_genres.get(g, 0) - netflix_genres.get(g, 0))
    _gain_disp = _GENRE_DISPLAY.get(_top_gain_genre.title(), _top_gain_genre.title())
    st.caption(
        f"Max's largest genre contribution is {_gain_disp} "
        f"(+{merged_genres.get(_top_gain_genre,0) - netflix_genres.get(_top_gain_genre,0):,} titles)."
    )

with st.expander("About These Comparisons"):
    st.markdown(
        """These charts compare **unfiltered** Netflix and Max catalogs to show the
raw merger impact, regardless of any sidebar filters applied above.

- **Catalog Size** counts unique titles on each platform. The merged count
  may be less than Netflix + Max combined because some titles appear on both.
- **IMDb Distribution** overlays normalized score distributions (probability density)
  so shape differences are visible even when catalog sizes differ.
- **Genre Comparison** shows the top genres by title count. Grouped bars
  let you spot where Max fills Netflix's gaps (and vice versa)."""
    )

# ── section 3: top titles ───────────────────────────────────────────────────

st.divider()
st.markdown(
    section_header_html(
        "Top Titles",
        "Ranked by a Bayesian quality score that balances IMDb ratings with vote credibility",
    ),
    unsafe_allow_html=True,
)

df["quality_score"] = compute_quality_score(df)

tab_movies, tab_shows = st.tabs(["Top Movies", "Top Shows"])


def _qs_bar_html(qs: float) -> str:
    """Render a small colored progress bar for the quality score."""
    qs_disp = round(qs, 1)  # Use rounded value for both display and color threshold
    color = "#2ecc71" if qs_disp >= 8.0 else "#f39c12" if qs_disp >= 7.0 else "#e74c3c"
    pct = min(qs * 10, 100)
    return (
        f'<div style="display:flex;align-items:center;gap:6px;margin-top:6px;">'
        f'<div style="flex:1;background:#2a2a3e;border-radius:4px;height:5px;overflow:hidden;">'
        f'<div style="width:{pct:.0f}%;height:100%;background:{color};border-radius:4px;"></div>'
        f'</div>'
        f'<span style="font-size:0.72em;color:{color};font-weight:600;white-space:nowrap;">{qs_disp:.1f}/10</span>'
        f'</div>'
    )


def _render_home_detail_panel(title_id: str):
    """Render an in-page detail panel for a selected top title."""
    # Find row in merged catalog
    sel_rows = df[df["id"] == title_id]
    if sel_rows.empty:
        sel_rows = merged_all[merged_all["id"] == title_id]
        if not sel_rows.empty:
            sel_rows = sel_rows.drop_duplicates("id")
            sel_rows = sel_rows.copy()
            sel_rows["quality_score"] = compute_quality_score(sel_rows)
    if sel_rows.empty:
        return

    sel = sel_rows.iloc[0]
    _enr_row = enriched[enriched["id"] == title_id] if not enriched.empty else None
    _has_enr = _enr_row is not None and not _enr_row.empty
    _enr = _enr_row.iloc[0] if _has_enr else None

    # Styled container
    st.markdown(
        f'<div style="background:{CARD_BG};border:1px solid {CARD_BORDER};'
        f'border-top:3px solid {_MERGED_COLOR};border-radius:10px;padding:16px 20px;margin:12px 0;">',
        unsafe_allow_html=True,
    )

    # Header row: title + close button
    hdr, close_col = st.columns([5, 1])
    with hdr:
        st.subheader(sel["title"])
    with close_col:
        if st.button("✕ Close", key="home_detail_close", use_container_width=True):
            st.session_state["home_selected_title_id"] = None
            st.rerun()

    # Award badge
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

    def _mc(label: str, value: str) -> str:
        return (
            f'<div style="padding:8px 0;">'
            f'<div style="color:{CARD_TEXT_MUTED};font-size:0.72em;text-transform:uppercase;'
            f'letter-spacing:0.04em;margin-bottom:3px;">{label}</div>'
            f'<div style="font-weight:600;color:{CARD_TEXT};font-size:0.92em;">{value}</div>'
            f'</div>'
        )

    # Poster + metadata side by side
    _poster = None
    if _has_enr and "poster_url" in _enr_row.columns:
        _poster = _enr.get("poster_url")

    if _poster and str(_poster) != "nan":
        poster_col, meta_col = st.columns([1, 4])
        with poster_col:
            st.image(str(_poster), width=130)
        with meta_col:
            _render_detail_meta(sel, _enr, _has_enr, _enr_row, _mc)
    else:
        _render_detail_meta(sel, _enr, _has_enr, _enr_row, _mc)

    # Genre pills
    if isinstance(sel.get("genres"), list) and sel["genres"]:
        genre_pills = " ".join(
            f'<span style="background:{CARD_BORDER};color:{CARD_TEXT};'
            f'padding:3px 10px;border-radius:12px;font-size:0.82em;margin-right:3px;">'
            f'{g.title()}</span>'
            for g in sel["genres"]
        )
        st.markdown(f'<div style="margin:8px 0 4px;">{genre_pills}</div>', unsafe_allow_html=True)

    # Description
    desc = sel.get("description", "")
    if desc and str(desc) not in ("", "nan"):
        st.divider()
        desc_str = str(desc)
        st.markdown(
            f'<div style="font-size:0.9em;color:{CARD_TEXT};line-height:1.6;">{desc_str}</div>',
            unsafe_allow_html=True,
        )

    # Cast & Crew
    with st.expander("Cast & Crew"):
        title_credits = merged_credits_all[merged_credits_all["title_id"] == title_id]
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
                f'letter-spacing:0.04em;margin:8px 0 4px;">Cast</div>',
                unsafe_allow_html=True,
            )
            actor_list = actors.head(12).reset_index(drop=True)
            ac1, ac2 = st.columns(2)
            for idx, (_, actor) in enumerate(actor_list.iterrows()):
                char = actor.get("character", "")
                name = actor["name"]
                char_html = (
                    f'<span style="color:{CARD_TEXT_MUTED};font-size:0.8em;"> as {char}</span>'
                    if char and str(char) not in ("nan", "")
                    else ""
                )
                with (ac1 if idx % 2 == 0 else ac2):
                    st.markdown(
                        f'<div style="font-size:0.85em;color:{CARD_TEXT};padding:2px 0;">'
                        f'<strong>{name}</strong>{char_html}</div>',
                        unsafe_allow_html=True,
                    )
            if len(actors) > 12:
                st.caption(f"+{len(actors) - 12} more cast members")

        if directors.empty and actors.empty:
            st.caption("No cast & crew information available for this title.")

    st.markdown("</div>", unsafe_allow_html=True)


def _render_detail_meta(sel, _enr, _has_enr, _enr_row, _mc):
    """Render the 2-row metadata grid for the home detail panel."""
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(_mc("Type", sel.get("type", "N/A")), unsafe_allow_html=True)
    with m2:
        year = sel.get("release_year")
        st.markdown(_mc("Year", str(int(year)) if year == year else "N/A"), unsafe_allow_html=True)
    with m3:
        imdb = sel.get("imdb_score")
        imdb_str = f"{imdb:.1f}" if imdb == imdb else "N/A"
        st.markdown(_mc("IMDb", imdb_str), unsafe_allow_html=True)
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

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        cert = sel.get("age_certification", "")
        cert_str = cert if cert and str(cert) != "nan" else "N/A"
        st.markdown(_mc("Rating", cert_str), unsafe_allow_html=True)
    with r2:
        rt = sel.get("runtime")
        rt_str = f"{int(rt)} min" if rt and rt == rt else "N/A"
        st.markdown(_mc("Runtime", rt_str), unsafe_allow_html=True)
    with r3:
        votes = sel.get("imdb_votes")
        votes_display = format_votes(votes) if votes and votes == votes else "N/A"
        st.markdown(_mc("Votes", votes_display), unsafe_allow_html=True)
    with r4:
        qs = (
            sel.get("quality_score")
            if "quality_score" in sel.index and sel["quality_score"] == sel["quality_score"]
            else compute_quality_score(sel.to_frame().T).iloc[0]
        )
        _qs_color = "#2ecc71" if round(qs, 1) >= 8.0 else "#f39c12" if round(qs, 1) >= 7.0 else "#e74c3c"
        _qs_pct = min(qs * 10, 100)
        st.markdown(
            f'<div style="padding:8px 0;">'
            f'<div style="color:{CARD_TEXT_MUTED};font-size:0.72em;text-transform:uppercase;'
            f'letter-spacing:0.04em;margin-bottom:5px;">Quality Score</div>'
            f'<div style="display:flex;align-items:center;gap:6px;">'
            f'<div style="flex:1;background:#2a2a3e;border-radius:3px;height:5px;">'
            f'<div style="width:{_qs_pct:.0f}%;height:100%;background:{_qs_color};border-radius:3px;"></div>'
            f'</div>'
            f'<span style="font-size:0.82em;color:{_qs_color};font-weight:600;">{round(qs, 1):.1f}</span>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def _render_title_cards(subset, tab_prefix: str):
    """Render a 5-column grid of title cards for the top 10 titles."""
    deduped = deduplicate_titles(subset)
    top = deduped.nlargest(10, "quality_score")
    if len(top) == 0:
        st.info("No titles match the current filters.")
        return

    # Merge poster URLs from enriched data
    poster_map = {}
    if "poster_url" not in top.columns and not enriched.empty and "poster_url" in enriched.columns:
        poster_map = enriched.dropna(subset=["poster_url"]).drop_duplicates("id").set_index("id")["poster_url"].to_dict()

    rows = [top.iloc[i: i + 5] for i in range(0, len(top), 5)]
    card_idx = 0
    for row_chunk in rows:
        cols = st.columns(5)
        for col, (_, title) in zip(cols, row_chunk.iterrows()):
            plat_list = title.get("platforms", title.get("platform", ""))
            plat_badges = platform_badges_html(plat_list)
            # Determine primary platform color for placeholder
            _first_plat = plat_list[0] if isinstance(plat_list, list) and plat_list else plat_list
            _plat_color = PLATFORMS.get(_first_plat, {}).get("color", _MERGED_COLOR)
            imdb = title["imdb_score"]
            imdb_str = f"{imdb:.1f}" if imdb == imdb else "N/A"
            votes_str = format_votes(title.get("imdb_votes"))
            qs = title["quality_score"]

            # Poster
            poster_url = title.get("poster_url") or poster_map.get(title["id"])
            if poster_url and str(poster_url) != "nan":
                poster_html = (
                    f'<img src="{poster_url}" style="width:100%;border-radius:6px;'
                    f'margin-bottom:8px;max-height:170px;object-fit:cover;" '
                    f'onerror="this.style.display=\'none\'" />'
                )
            else:
                _initial = title["title"][0].upper() if title["title"] else "?"
                poster_html = (
                    f'<div style="width:100%;height:120px;border-radius:6px;margin-bottom:8px;'
                    f'background:{_plat_color};display:flex;align-items:center;'
                    f'justify-content:center;font-size:2.5em;font-weight:700;color:rgba(255,255,255,0.85);">'
                    f'{_initial}</div>'
                )

            year_str = str(int(title["release_year"])) if title["release_year"] == title["release_year"] else "?"

            _is_selected = st.session_state.get("home_selected_title_id") == title["id"]
            _card_border = f"border:2px solid {CARD_ACCENT};" if _is_selected else f"border:1px solid {CARD_BORDER};"
            with col:
                st.markdown(
                    f'<div style="background:{CARD_BG};border-radius:10px;padding:12px 10px;'
                    f'margin-bottom:4px;{_card_border}min-height:380px;'
                    f'display:flex;flex-direction:column;">'
                    f'{poster_html}'
                    f'<div style="font-size:0.9em;font-weight:700;line-height:1.3;margin-bottom:6px;flex:1;">'
                    f'{title["title"]}'
                    f'<span style="color:{CARD_TEXT_MUTED};font-weight:400;"> ({year_str})</span></div>'
                    f'<div style="margin-bottom:6px;">{plat_badges}'
                    f'<span style="background:{CARD_BORDER};color:#ccc;padding:2px 6px;border-radius:4px;'
                    f'font-size:0.72em;margin-left:4px;">{title["type"]}</span></div>'
                    f'<div style="font-size:0.82em;color:{CARD_TEXT};margin-bottom:2px;">'
                    f'IMDb <strong>{imdb_str}</strong> '
                    f'<span style="color:{CARD_TEXT_MUTED};">({votes_str} votes)</span></div>'
                    f'{_qs_bar_html(qs)}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                # Toggle in-page detail panel
                _btn_label = "✕ Close" if _is_selected else "Details"
                if st.button(_btn_label, key=f"home_view_{tab_prefix}_{card_idx}", use_container_width=True):
                    if _is_selected:
                        st.session_state["home_selected_title_id"] = None
                    else:
                        st.session_state["home_selected_title_id"] = title["id"]
                    st.rerun()
                card_idx += 1


with tab_movies:
    _render_title_cards(df[df["type"] == "Movie"], tab_prefix="movie")

with tab_shows:
    _render_title_cards(df[df["type"] == "Show"], tab_prefix="show")

# In-page detail panel — shown below tabs when a title is selected
_home_sel_id = st.session_state.get("home_selected_title_id")
if _home_sel_id:
    _render_home_detail_panel(_home_sel_id)

with st.expander("How Rankings Work"):
    st.markdown(
        """Rankings use a **Bayesian weighted quality score** instead of raw IMDb ratings.

**Why?** A title with a 9.5 rating from 10 votes isn't necessarily better than one
with 8.9 from 100K votes. Raw scores from small samples are unreliable.

**How it works:**
- **Bayesian IMDb** (85% weight): Adjusts raw scores based on vote count credibility.
  Titles with fewer votes are pulled toward the global average (6.5), while heavily-voted
  titles keep their earned score.
- **Normalized Popularity** (15% weight): TMDB popularity scaled to 0-10, rewarding
  titles with broad audience reach.

*Formula: (votes / (votes + 10K)) * IMDb + (10K / (votes + 10K)) * 6.5*"""
    )

# ── section 4: content timeline ─────────────────────────────────────────────

st.divider()
st.markdown(
    section_header_html(
        "Content Timeline",
        "When was the content produced? Distribution of titles across decades",
    ),
    unsafe_allow_html=True,
)

tab_timeline_all, tab_timeline_movies, tab_timeline_shows = st.tabs(["All", "Movies", "Shows"])


def _render_timeline(subset_df):
    """Render a stacked area chart for title count by decade and platform."""
    if "decade" not in subset_df.columns:
        st.info("Decade data not available.")
        return

    tl = (
        subset_df[subset_df["decade"].notna()]
        .groupby(["decade", "platform"], observed=True)
        .size()
        .reset_index(name="count")
    )
    if len(tl) == 0:
        st.info("No timeline data for the current filters.")
        return

    # Ensure all decade labels appear (reindex to avoid missing decades)
    all_decades_platforms = (
        tl.set_index(["decade", "platform"])
        .reindex(
            [(d, p) for d in DECADE_LABELS for p in tl["platform"].unique()],
            fill_value=0,
        )
        .reset_index()
    )
    all_decades_platforms["Platform"] = all_decades_platforms["platform"].map(_platform_name)
    platform_display_colors = {_platform_name(k): v["color"] for k, v in PLATFORMS.items()}

    fig_tl = px.area(
        all_decades_platforms,
        x="decade",
        y="count",
        color="Platform",
        color_discrete_map=platform_display_colors,
        template=PLOTLY_TEMPLATE,
        height=CHART_HEIGHT,
        labels={"decade": "Decade", "count": "Titles"},
    )
    fig_tl.update_traces(opacity=0.75)
    fig_tl.update_layout(
        xaxis=dict(categoryorder="array", categoryarray=DECADE_LABELS),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_tl, use_container_width=True)

    # Auto-generated insight: which decade had the most titles?
    decade_totals = tl.groupby("decade")["count"].sum()
    if len(decade_totals) > 0:
        peak_decade = decade_totals.idxmax()
        peak_count = int(decade_totals.max())
        total_count = int(decade_totals.sum())
        peak_pct = peak_count / total_count * 100 if total_count > 0 else 0
        st.caption(
            f"{peak_decade} was the most active decade, contributing "
            f"{peak_pct:.0f}% of the catalog ({peak_count:,} titles)."
        )


with tab_timeline_all:
    _render_timeline(df)

with tab_timeline_movies:
    _render_timeline(df[df["type"] == "Movie"])

with tab_timeline_shows:
    _render_timeline(df[df["type"] == "Show"])

# ── section 5: global reach ────────────────────────────────────────────────

st.divider()
st.markdown(
    section_header_html(
        "Global Reach",
        "Where content comes from — production countries across the catalog",
    ),
    unsafe_allow_html=True,
)

_geo_col = "production_countries"
if _geo_col in df.columns:
    _geo_rows = df[df[_geo_col].apply(lambda x: isinstance(x, list) and len(x) > 0)]
    _all_countries = _geo_rows.explode(_geo_col)[_geo_col]
    _country_counts = _all_countries.value_counts()

    # Three metric cards ABOVE the chart
    _unique_countries = int(_country_counts.index.nunique())
    _has_countries = len(_geo_rows)
    _intl_count = _geo_rows[_geo_col].apply(lambda x: "US" not in x).sum()
    _intl_pct = (_intl_count / _has_countries * 100) if _has_countries > 0 else 0.0

    _top_intl_code = (
        _country_counts[~_country_counts.index.isin(["US"])].index[0]
        if len(_country_counts[~_country_counts.index.isin(["US"])]) > 0
        else None
    )
    _top_intl_name = _COUNTRY_NAMES.get(_top_intl_code, _top_intl_code) if _top_intl_code else "N/A"

    geo_m1, geo_m2, geo_m3 = st.columns(3)
    with geo_m1:
        st.markdown(
            styled_metric_card_html(
                "Countries Represented",
                f"{_unique_countries:,}",
                help_text="Number of distinct production countries in the current view.",
            ),
            unsafe_allow_html=True,
        )
    with geo_m2:
        st.markdown(
            styled_metric_card_html(
                "International Content",
                f"{_intl_pct:.0f}%",
                help_text="Percentage of titles produced outside the United States.",
            ),
            unsafe_allow_html=True,
        )
    with geo_m3:
        st.markdown(
            styled_metric_card_html(
                "Top International Market",
                _top_intl_name,
                help_text="Non-US country with the most titles in the current catalog view.",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)

    _top10 = _country_counts.head(10).reset_index()
    _top10.columns = ["code", "Titles"]
    _top10["Country"] = _top10["code"].map(lambda c: _COUNTRY_NAMES.get(c, c))
    _top10 = _top10.sort_values("Titles", ascending=True)
    _max_titles = int(_top10["Titles"].max())

    fig_geo = go.Figure(go.Bar(
        x=_top10["Titles"],
        y=_top10["Country"],
        orientation="h",
        marker_color=_MERGED_COLOR,
        text=_top10["Titles"],
        texttemplate="%{text:,}",
        textposition="outside",
    ))
    fig_geo.update_layout(
        template=PLOTLY_TEMPLATE,
        height=CHART_HEIGHT,
        showlegend=False,
        yaxis_title="",
        xaxis=dict(range=[0, _max_titles * 1.18]),
        margin=dict(l=20, r=60, t=20, b=20),
    )
    st.plotly_chart(fig_geo, use_container_width=True)
else:
    st.info("Production country data not available for this view.")

# ── section 6: navigation cards ─────────────────────────────────────────────

st.divider()
st.markdown(section_header_html("Explore"), unsafe_allow_html=True)

_NAV_PAGES = [
    ("01_Explore_Catalog", "Explore Catalog", "Search, browse, and discover similar titles", "🔍"),
    ("02_Platform_Comparisons", "Platform Comparisons", "Benchmark the merged entity against competitors", "📊"),
    ("03_Platform_DNA", "Platform DNA", "Understand each platform's unique content identity", "🧬"),
    ("04_Discovery_Engine", "Discovery Engine", "Get personalized recommendations", "✨"),
    ("05_Strategic_Insights", "Strategic Insights", "Merger overlap, gap analysis, and market simulation", "🎯"),
    ("06_Interactive_Lab", "Interactive Lab", "Build your service and predict title success", "🧪"),
    ("07_Cast_Crew_Network", "Cast & Crew Network", "Explore collaboration networks and influence scores", "🕸️"),
]

nav_cols = st.columns(3)
for i, (slug, name, desc, icon) in enumerate(_NAV_PAGES):
    with nav_cols[i % 3]:
        st.markdown(
            f'<div style="background:{CARD_BG};border:1px solid {CARD_BORDER};'
            f'border-left:4px solid {_MERGED_COLOR};border-radius:6px;'
            f'padding:12px 16px;margin-bottom:8px;">'
            f'<div style="display:flex;align-items:center;justify-content:space-between;">'
            f'<span style="font-size:1.1em;">{icon}</span>'
            f'<span style="color:{CARD_TEXT_MUTED};font-size:0.9em;">→</span>'
            f'</div>'
            f'<div style="font-weight:700;font-size:0.95em;color:{CARD_TEXT};margin:4px 0 2px;">{name}</div>'
            f'<div style="font-size:0.8em;color:{CARD_TEXT_MUTED};">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.page_link(f"pages/{slug}.py", label=f"Open {name} →")

# ── section 7: footer ───────────────────────────────────────────────────────

st.divider()
st.markdown(
    '<div style="color:#555;font-size:0.8em;text-align:center;padding:8px 0 16px;">'
    'Hypothetical merger for academic analysis. '
    'Data is a snapshot (mid-2023). '
    'All insights are illustrative, not prescriptive. '
    'Update: As of Feb 26, 2026, Netflix withdrew from this acquisition after Paramount Skydance\'s competing bid was deemed superior by the WBD board.'
    '</div>',
    unsafe_allow_html=True,
)
