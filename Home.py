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
    INITIAL_SIDEBAR_STATE,
    LAYOUT,
    PAGE_ICON,
    PAGE_TITLE,
    PLATFORMS,
    PLOTLY_TEMPLATE,
)
from src.data.loaders import (
    get_credits_for_view,
    get_titles_for_view,
    load_merged_credits,
    load_merged_titles,
)
from src.ui.filters import apply_filters, render_quick_stats, render_sidebar_filters
from src.ui.session import init_session_state


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

# ── data loading ─────────────────────────────────────────────────────────────

# Reserve a placeholder at the top of the sidebar for quick stats
_stats_placeholder = st.sidebar.container()

# Load data for current platform view
raw_df = get_titles_for_view(st.session_state["platform_view"])
filters = render_sidebar_filters(raw_df)

# Re-load after platform view may have changed via sidebar
raw_df = get_titles_for_view(filters["platform_view"])
df = apply_filters(raw_df, filters)

credits_df = get_credits_for_view(filters["platform_view"])

# Fill the placeholder with quick stats (appears above filters)
with _stats_placeholder:
    render_quick_stats(df, total_count=len(raw_df), total_avg_imdb=raw_df["imdb_score"].mean())

# ── title ────────────────────────────────────────────────────────────────────

st.title("Netflix + Max Merger Analysis")
st.caption("Hypothetical merger analysis across streaming platforms")

# ── precompute baselines for deltas ─────────────────────────────────────────

merged_all = load_merged_titles()
merged_credits_all = load_merged_credits()
netflix_df = merged_all[merged_all["platform"] == "netflix"]
max_df = merged_all[merged_all["platform"] == "max"]

# Netflix-only baselines (for delta comparison)
_nf_catalog = netflix_df["id"].nunique()
_nf_avg_imdb = netflix_df["imdb_score"].mean()
_nf_credits = merged_credits_all
_nf_people = _nf_credits[_nf_credits["platform"] == "netflix"]["person_id"].nunique()
_nf_genres = len(set(g for gs in netflix_df["genres"].dropna() if isinstance(gs, list) for g in gs))

# ── section 1: hero metrics ─────────────────────────────────────────────────

st.header("Overview")
st.caption("Key metrics showing the combined strength of the Netflix + Max merger")

catalog_size = df["id"].nunique()
avg_imdb = df["imdb_score"].mean()
people_count = credits_df["person_id"].nunique()
genre_count = len(set(g for gs in df["genres"].dropna() if isinstance(gs, list) for g in gs))

# Compute deltas vs Netflix-only
def _pct_delta(current, baseline):
    if baseline == 0:
        return None
    return f"{((current - baseline) / baseline) * 100:+.0f}% vs Netflix"

def _abs_delta(current, baseline):
    diff = current - baseline
    return f"{diff:+.2f} vs Netflix"

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Catalog Size",
        f"{catalog_size:,}",
        delta=_pct_delta(catalog_size, _nf_catalog),
        help="Total unique titles in the current view. Delta shows gain vs Netflix alone.",
    )
with c2:
    imdb_str = f"{avg_imdb:.2f}" if avg_imdb == avg_imdb else "N/A"
    st.metric(
        "Avg IMDb Score",
        imdb_str,
        delta=_abs_delta(avg_imdb, _nf_avg_imdb) if avg_imdb == avg_imdb else None,
        help="Mean IMDb rating across all titles (excluding unrated). Delta shows shift vs Netflix alone.",
    )
with c3:
    st.metric(
        "Cast & Crew",
        f"{people_count:,}",
        delta=_pct_delta(people_count, _nf_people),
        help="Unique actors, directors, and crew across all credits. Delta shows gain vs Netflix alone.",
    )
with c4:
    st.metric(
        "Genre Coverage",
        f"{genre_count:,}",
        delta=_pct_delta(genre_count, _nf_genres),
        help="Number of distinct genres represented. Delta shows expansion vs Netflix alone.",
    )

# Insight callout — computed from actual data
_added_titles = catalog_size - _nf_catalog
_pct_increase = ((_added_titles / _nf_catalog) * 100) if _nf_catalog > 0 else 0
_insight_parts = [
    f"The merger adds **{_added_titles:,} titles** to Netflix's catalog, "
    f"a **{_pct_increase:.0f}% increase**",
]
if avg_imdb == avg_imdb and _nf_avg_imdb == _nf_avg_imdb:
    _imdb_direction = "improving" if avg_imdb >= _nf_avg_imdb else "shifting"
    _insight_parts.append(
        f" while {_imdb_direction} average quality from "
        f"**{_nf_avg_imdb:.2f}** to **{avg_imdb:.2f}**."
    )
else:
    _insight_parts.append(".")
st.info("".join(_insight_parts))

# ── section 2: merger impact ────────────────────────────────────────────────

st.markdown("---")
st.header("Merger Impact")
st.caption("How combining Netflix and Max catalogs reshapes volume, quality, and genre diversity")

mi1, mi2, mi3 = st.columns(3)

# Volume boost
with mi1:
    st.subheader("Catalog Size: Netflix vs Max vs Merged")
    vol_data = {
        "Platform": ["Netflix", "Max", "Merged"],
        "Titles": [
            netflix_df["id"].nunique(),
            max_df["id"].nunique(),
            merged_all["id"].nunique(),
        ],
    }
    fig_vol = px.bar(
        vol_data,
        x="Platform",
        y="Titles",
        color="Platform",
        color_discrete_map={
            "Netflix": PLATFORMS["netflix"]["color"],
            "Max": PLATFORMS["max"]["color"],
            "Merged": PLATFORMS["merged"]["color"],
        },
        template=PLOTLY_TEMPLATE,
        height=CHART_HEIGHT,
    )
    fig_vol.update_layout(showlegend=False)
    st.plotly_chart(fig_vol, use_container_width=True)

# Quality shift
with mi2:
    st.subheader("IMDb Score Distribution: Netflix vs Merged")
    netflix_scores = netflix_df["imdb_score"].dropna()
    merged_scores = merged_all["imdb_score"].dropna()
    fig_qual = go.Figure()
    fig_qual.add_trace(go.Histogram(
        x=netflix_scores, name="Netflix",
        marker_color=PLATFORMS["netflix"]["color"], opacity=0.7,
        nbinsx=20,
    ))
    fig_qual.add_trace(go.Histogram(
        x=merged_scores, name="Merged",
        marker_color=PLATFORMS["merged"]["color"], opacity=0.7,
        nbinsx=20,
    ))
    fig_qual.update_layout(
        barmode="overlay", template=PLOTLY_TEMPLATE,
        height=CHART_HEIGHT, xaxis_title="IMDb Score", yaxis_title="Count",
    )
    st.plotly_chart(fig_qual, use_container_width=True)

# Genre expansion
with mi3:
    st.subheader("Top Genres: Netflix vs Merged Entity")

    def _top_genres(frame, n=10):
        from collections import Counter
        counter = Counter()
        for genres in frame["genres"].dropna():
            if isinstance(genres, list):
                counter.update(genres)
        return counter.most_common(n)

    netflix_genres = dict(_top_genres(netflix_df))
    merged_genres = dict(_top_genres(merged_all))
    all_top = sorted(set(list(netflix_genres.keys()) + list(merged_genres.keys())))[:12]
    all_top_display = [g.title() for g in all_top]

    fig_genre = go.Figure()
    fig_genre.add_trace(go.Bar(
        x=all_top_display, y=[netflix_genres.get(g, 0) for g in all_top],
        name="Netflix", marker_color=PLATFORMS["netflix"]["color"],
    ))
    fig_genre.add_trace(go.Bar(
        x=all_top_display, y=[merged_genres.get(g, 0) for g in all_top],
        name="Merged", marker_color=PLATFORMS["merged"]["color"],
    ))
    fig_genre.update_layout(
        barmode="group", template=PLOTLY_TEMPLATE,
        height=CHART_HEIGHT, xaxis_title="Genre", yaxis_title="Titles",
    )
    st.plotly_chart(fig_genre, use_container_width=True)

with st.expander("About These Comparisons"):
    st.markdown(
        """These charts compare **unfiltered** Netflix and Max catalogs to show the
raw merger impact, regardless of any sidebar filters applied above.

- **Catalog Size** counts unique titles on each platform. The merged count
  may be less than Netflix + Max combined because some titles appear on both.
- **IMDb Distribution** overlays score histograms so you can see how the
  quality profile shifts after merging.
- **Genre Comparison** shows the top genres by title count. Grouped bars
  let you spot where Max fills Netflix's gaps (and vice versa)."""
    )

# ── section 3: top titles ───────────────────────────────────────────────────

st.markdown("---")
st.header("Top Titles")
st.caption("Ranked by a Bayesian quality score that balances IMDb ratings with vote credibility")

# Compute quality score for filtered data
df["quality_score"] = compute_quality_score(df)

tab_movies, tab_shows = st.tabs(["Top Movies", "Top Shows"])


def _render_title_cards(subset):
    """Render a 4-column grid of title cards for the top 20 titles."""
    top = subset.nlargest(20, "quality_score")
    if len(top) == 0:
        st.info("No titles match the current filters.")
        return
    rows = [top.iloc[i : i + 4] for i in range(0, len(top), 4)]
    for row_chunk in rows:
        cols = st.columns(4)
        for col, (_, title) in zip(cols, row_chunk.iterrows()):
            plat_key = title["platform"]
            plat_info = PLATFORMS.get(plat_key, {})
            plat_name = plat_info.get("name", plat_key.title())
            plat_color = plat_info.get("color", "#555")
            imdb = title["imdb_score"]
            imdb_str = f"{imdb:.1f}" if imdb == imdb else "N/A"
            votes_str = format_votes(title.get("imdb_votes"))
            qs = title["quality_score"]

            with col:
                st.markdown(
                    f"""<div style="background:{CARD_BG};border-radius:10px;padding:14px 12px;
                    margin-bottom:8px;border:1px solid {CARD_BORDER};
                    transition:transform 0.15s ease,box-shadow 0.15s ease;"
                    onmouseover="this.style.transform='translateY(-2px)';this.style.boxShadow='0 4px 12px rgba(0,0,0,0.3)'"
                    onmouseout="this.style.transform='none';this.style.boxShadow='none'">
                    <div style="font-size:0.95em;font-weight:700;line-height:1.3;margin-bottom:6px;">
                    {title['title']}<span style="color:{CARD_TEXT_MUTED};font-weight:400;"> ({int(title['release_year'])})</span></div>
                    <span style="background:{plat_color};color:#fff;padding:2px 8px;border-radius:4px;
                    font-size:0.75em;font-weight:600;">{plat_name}</span>
                    <span style="background:{CARD_BORDER};color:#ccc;padding:2px 8px;border-radius:4px;
                    font-size:0.75em;margin-left:4px;">{title['type']}</span>
                    <div style="margin-top:8px;font-size:0.85em;color:{CARD_TEXT};">
                    IMDb {imdb_str} <span style="color:{CARD_TEXT_MUTED};">({votes_str} votes)</span></div>
                    <div style="margin-top:4px;font-size:0.85em;">
                    Quality Score: <strong style="color:{CARD_ACCENT};">{qs:.1f}</strong>/10</div>
                    </div>""",
                    unsafe_allow_html=True,
                )


with tab_movies:
    _render_title_cards(df[df["type"] == "Movie"])

with tab_shows:
    _render_title_cards(df[df["type"] == "Show"])

with st.expander("How Rankings Work"):
    st.markdown(
        """Rankings use a **Bayesian weighted quality score** instead of raw IMDb ratings.

**Why?** A title with a 9.5 rating from 10 votes isn't necessarily better than one
with 8.9 from 100K votes. Raw scores from small samples are unreliable.

**How it works:**
- **Bayesian IMDb** (70% weight): Adjusts raw scores based on vote count credibility.
  Titles with fewer votes are pulled toward the global average (6.5), while heavily-voted
  titles keep their earned score.
- **Normalized Popularity** (30% weight): TMDB popularity scaled to 0-10, rewarding
  titles with broad audience reach.

*Formula: (votes / (votes + 10K)) * IMDb + (10K / (votes + 10K)) * 6.5*"""
    )

# ── section 4: content timeline ─────────────────────────────────────────────

st.markdown("---")
st.header("Content Timeline")
st.caption("When was the content produced? Distribution of titles across decades")

timeline = (
    df[df["decade"].notna()]
    .groupby(["decade", "platform"], observed=True)
    .size()
    .reset_index(name="count")
)
if len(timeline) > 0:
    timeline["Platform"] = timeline["platform"].map(_platform_name)
    platform_display_colors = {_platform_name(k): v["color"] for k, v in PLATFORMS.items()}
    fig_timeline = px.area(
        timeline,
        x="decade",
        y="count",
        color="Platform",
        color_discrete_map=platform_display_colors,
        template=PLOTLY_TEMPLATE,
        height=CHART_HEIGHT,
        labels={"decade": "Decade", "count": "Titles"},
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

# ── section 5: global reach ────────────────────────────────────────────────

st.markdown("---")
st.header("Global Reach")
st.caption("Where content comes from \u2014 production countries across the catalog")

# Explode production_countries to count titles per country
_geo_col = "production_countries"
if _geo_col in df.columns:
    _geo_rows = df[df[_geo_col].apply(lambda x: isinstance(x, list) and len(x) > 0)]
    _all_countries = _geo_rows.explode(_geo_col)[_geo_col]
    _country_counts = _all_countries.value_counts()

    geo_left, geo_right = st.columns([2, 1])

    with geo_left:
        _top10 = _country_counts.head(10).reset_index()
        _top10.columns = ["code", "Titles"]
        _top10["Country"] = _top10["code"].map(lambda c: _COUNTRY_NAMES.get(c, c))
        # Sort ascending so largest bar is at top in horizontal layout
        _top10 = _top10.sort_values("Titles", ascending=True)
        fig_geo = px.bar(
            _top10,
            x="Titles",
            y="Country",
            orientation="h",
            template=PLOTLY_TEMPLATE,
            height=CHART_HEIGHT,
        )
        fig_geo.update_traces(marker_color=PLATFORMS.get("merged", {}).get("color", "#8B5CF6"))
        fig_geo.update_layout(showlegend=False, yaxis_title="")
        st.plotly_chart(fig_geo, use_container_width=True)

    with geo_right:
        _unique_countries = _country_counts.index.nunique()
        st.metric("Countries Represented", f"{_unique_countries:,}")

        _has_countries = len(_geo_rows)
        _intl_count = _geo_rows[_geo_col].apply(lambda x: "US" not in x).sum()
        _intl_pct = (_intl_count / _has_countries * 100) if _has_countries > 0 else 0
        st.metric("International Content", f"{_intl_pct:.0f}%",
                  help="Percentage of titles produced outside the United States")
else:
    st.info("Production country data not available for this view.")

# ── section 6: navigation cards ─────────────────────────────────────────────

st.markdown("---")
st.header("Explore")

nav_pages = [
    ("01_Explore_Catalog", "Explore Catalog", "Search and discover titles with similar-title recommendations"),
    ("02_Platform_Comparisons", "Platform Comparisons", "Compare the merged entity against competitors"),
    ("03_Platform_DNA", "Platform DNA", "Understand each platform's unique content identity"),
    ("04_Discovery_Engine", "Discovery Engine", "Get personalized recommendations"),
    ("05_Strategic_Insights", "Strategic Insights", "Merger overlap and gap analysis"),
    ("06_Interactive_Lab", "Interactive Lab", "Build your own streaming service and predict ratings"),
    ("07_Cast_Crew_Network", "Cast & Crew Network", "Explore collaboration networks"),
]

cols = st.columns(3)
for i, (slug, name, desc) in enumerate(nav_pages):
    with cols[i % 3]:
        st.page_link(f"pages/{slug}.py", label=f"**{name}**")
        st.caption(desc)

# ── section 7: footer ───────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "Hypothetical merger for academic analysis. "
    "Data is a snapshot (mid-2023). "
    "All insights are illustrative, not prescriptive."
)
