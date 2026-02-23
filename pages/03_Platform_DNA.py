"""Page 3: Platform DNA — what makes each streaming platform feel different."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analysis.platform_dna import (
    compute_cluster_summaries,
    compute_landscape_clusters,
    compute_landscape_insights,
    compute_platform_comparison_data,
    compute_swipe_results,
    curate_quiz_titles,
)
from src.config import (
    ALL_PLATFORMS,
    CARD_ACCENT,
    CARD_BG,
    CARD_BORDER,
    CARD_TEXT,
    CARD_TEXT_MUTED,
    CHART_HEIGHT,
    DNA_UMAP_SAMPLE_SIZE,
    INITIAL_SIDEBAR_STATE,
    LAYOUT,
    PAGE_ICON,
    PAGE_TITLE,
    PLATFORMS,
    PLOTLY_TEMPLATE,
)
from src.data.loaders import load_all_platforms_titles, load_enriched_titles, load_umap_coords
from src.ui.filters import apply_filters, render_sidebar_filters
from src.ui.session import init_session_state

# -- page config ---------------------------------------------------------------

st.set_page_config(
    page_title=f"Platform DNA | {PAGE_TITLE}",
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state=INITIAL_SIDEBAR_STATE,
)

# CSS fix for text cutoff in side-by-side comparison columns
st.markdown(
    """
<style>
    [data-testid="column"] {
        overflow-wrap: break-word;
        word-wrap: break-word;
        word-break: break-word;
    }
    [data-testid="column"] .stMarkdown div {
        overflow-wrap: break-word;
        word-wrap: break-word;
    }
    [data-testid="stMetricLabel"] {
        overflow: visible !important;
        white-space: normal !important;
    }
    [data-testid="stMetricValue"] {
        overflow: visible !important;
        white-space: normal !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

init_session_state()

# Page-specific session state
st.session_state.setdefault("dna_platform", "merged")
st.session_state.setdefault("dna_compare_platform", None)
st.session_state.setdefault("dna_normalized", False)


# -- sidebar -------------------------------------------------------------------

raw_df = load_all_platforms_titles()
filters = render_sidebar_filters(raw_df)
df = apply_filters(raw_df, filters)


# -- page header ---------------------------------------------------------------

st.title("Platform DNA")
st.caption(
    "What makes each platform feel different — identity fingerprints, "
    "content landscape patterns, and your personal platform match"
)

# Standalone methodology expander (separate from content)
with st.expander("How Platform DNA Works"):
    st.markdown(
        "**Identity Radar** maps 6 normalized dimensions (0-100) for each "
        "platform, comparing it against all other platforms: Freshness "
        "(median release year), Quality (avg IMDb), Breadth (catalog size), "
        "Global Reach (international content %), Genre Diversity (unique genres), "
        "and Series Focus (show %). This creates a unique fingerprint for "
        "each platform's content strategy.\n\n"
        "**Content Landscape** uses UMAP (Uniform Manifold Approximation "
        "and Projection) to project titles from a high-dimensional feature "
        "space (TF-IDF descriptions + genre vectors) into 2D. The density "
        "contour shows where each platform's content concentrates. KMeans "
        "clustering (k=8) identifies content neighborhoods with auto-generated "
        "names. UMAP coordinates are precomputed offline.\n\n"
        "**Platform Matcher** scores your stated preferences against each "
        "platform's real data profile. Genre matching uses weighted overlap, "
        "while 6 slider dimensions measure alignment. The combined score "
        "gives your match percentage."
    )


# ==============================================================================
# SECTION 1: PLATFORM IDENTITY PROFILE
# ==============================================================================

st.markdown("---")
st.subheader("Platform Identity Profile")
st.caption(
    "A radar fingerprint of each platform's content strategy — "
    "six dimensions that define what makes it unique."
)

# Controls row: platform selectors + normalized toggle
ctrl_left, ctrl_mid, ctrl_right = st.columns([2, 2, 1])

platform_options = {PLATFORMS[k]["name"]: k for k in ["merged"] + ALL_PLATFORMS}

with ctrl_left:
    primary_name = st.selectbox(
        "Select platform",
        options=list(platform_options.keys()),
        index=list(platform_options.values()).index(
            st.session_state.get("dna_platform", "merged")
        ),
        key="dna_primary_select",
    )
    primary_key = platform_options[primary_name]
    st.session_state["dna_platform"] = primary_key

with ctrl_mid:
    compare_names = ["None"] + [
        name for name, key in platform_options.items() if key != primary_key
    ]
    current_compare = st.session_state.get("dna_compare_platform")
    current_compare_name = "None"
    if current_compare:
        for name, key in platform_options.items():
            if key == current_compare and key != primary_key:
                current_compare_name = name
                break
    compare_name = st.selectbox(
        "Compare with (optional)",
        options=compare_names,
        index=(
            compare_names.index(current_compare_name)
            if current_compare_name in compare_names
            else 0
        ),
        key="dna_compare_select",
    )
    compare_key = (
        platform_options.get(compare_name) if compare_name != "None" else None
    )
    st.session_state["dna_compare_platform"] = compare_key

with ctrl_right:
    normalized = st.toggle(
        "Normalized",
        value=st.session_state["dna_normalized"],
        help="Show percentages instead of raw counts",
    )
    st.session_state["dna_normalized"] = normalized

# Build profile data
platforms_to_profile = [primary_key]
if compare_key:
    platforms_to_profile.append(compare_key)

profiles = compute_platform_comparison_data(df, platforms_to_profile)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a hex color like '#E50914' to 'rgba(229,9,20,0.12)'."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _render_radar(platform_keys: list[str], profile_data: dict):
    """Render a radar chart for one or two platforms."""
    fig = go.Figure()
    for key in platform_keys:
        radar = profile_data[key]["radar"]
        dimensions = list(radar.keys())
        values = list(radar.values())
        # Close the polygon
        dimensions_closed = dimensions + [dimensions[0]]
        values_closed = values + [values[0]]
        p_color = PLATFORMS.get(key, {}).get("color", CARD_ACCENT)
        fig.add_trace(
            go.Scatterpolar(
                r=values_closed,
                theta=dimensions_closed,
                fill="toself",
                fillcolor=_hex_to_rgba(p_color, 0.12),
                line=dict(color=p_color, width=2.5),
                name=profile_data[key]["display_name"],
                hovertemplate="%{theta}: %{r:.0f}/100<extra>%{fullData.name}</extra>",
            )
        )
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=True,
                tickfont=dict(size=9, color=CARD_TEXT_MUTED),
                gridcolor="rgba(255,255,255,0.08)",
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color=CARD_TEXT),
                gridcolor="rgba(255,255,255,0.08)",
            ),
        ),
        template=PLOTLY_TEMPLATE,
        height=380,
        margin=dict(t=40, b=40, l=60, r=60),
        showlegend=len(platform_keys) > 1,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5,
            font=dict(size=12),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


_RADAR_DIMENSION_TABLE = (
    "| Dimension | What it measures | Scale |\n"
    "|-----------|-----------------|-------|\n"
    "| **Freshness** | How recent the catalog skews (median release year) | 0 = oldest catalog, 100 = newest |\n"
    "| **Quality** | Average IMDb score across all rated titles | 0 = lowest avg, 100 = highest avg |\n"
    "| **Breadth** | Total catalog size relative to other platforms | 0 = smallest, 100 = largest |\n"
    "| **Global Reach** | % of content produced outside the US | 0 = most domestic, 100 = most international |\n"
    "| **Genre Diversity** | Number of unique genres represented | 0 = least diverse, 100 = most diverse |\n"
    "| **Series Focus** | % of catalog that is TV shows (vs movies) | 0 = movie-focused, 100 = series-focused |\n\n"
    "All dimensions are normalized 0-100 relative to the range across all 6 platforms."
)


def _render_metrics_row(platform_key: str, data: dict):
    """Render compact quality metrics for a platform."""
    quality = data["quality_profile"]
    p_color = PLATFORMS.get(platform_key, {}).get("color", CARD_ACCENT)

    st.markdown(
        f'<div style="background:{CARD_BG};border-left:4px solid {p_color};'
        f'border-radius:8px;padding:12px 16px;margin-bottom:8px;">'
        f'<span style="font-size:1.15em;font-weight:700;color:{CARD_TEXT};">'
        f'{data["display_name"]}</span>'
        f'<span style="color:{CARD_TEXT_MUTED};font-size:0.88em;margin-left:12px;">'
        f'{data["title_count"]:,} titles</span></div>',
        unsafe_allow_html=True,
    )

    q1, q2, q3 = st.columns(3)
    with q1:
        avg_label = (
            f"{quality['avg_imdb']:.2f}" if quality["avg_imdb"] else "N/A"
        )
        st.metric(
            "Avg IMDb", avg_label,
            help="Average IMDb rating across all rated titles on this platform",
        )
    with q2:
        excellent_pct = quality["tier_pcts"].get("Excellent", 0)
        good_pct = quality["tier_pcts"].get("Good", 0)
        if normalized:
            st.metric(
                "Premium Content",
                f"{excellent_pct + good_pct:.1f}%",
                help="Excellent (8+) + Good (7-8) combined",
            )
        else:
            premium_count = quality["tier_counts"].get("Excellent", 0) + quality["tier_counts"].get("Good", 0)
            st.metric(
                "Premium Titles",
                f"{premium_count:,}",
                help="Excellent (8+) + Good (7-8) combined",
            )
    with q3:
        st.metric(
            "Rated Titles",
            f"{quality['total_rated']:,} / {quality['total_titles']:,}",
            help="How many titles have an IMDb rating vs total catalog size",
        )


def _render_traits(traits: list[dict]):
    """Render defining traits as narrative cards with colored left-border."""
    if not traits:
        return
    st.markdown("**Defining Traits**")
    for trait in traits:
        direction = trait["direction"]
        border_color = (
            "#2ecc71"
            if direction == "high"
            else "#e74c3c" if direction == "low" else CARD_TEXT_MUTED
        )
        st.markdown(
            f'<div style="background:{CARD_BG};border-left:4px solid {border_color};'
            f'border-radius:6px;padding:10px 14px;margin-bottom:6px;">'
            f'<span style="color:{CARD_ACCENT};font-weight:700;">'
            f'{trait["label"]}</span>'
            f'<span style="color:{CARD_TEXT_MUTED};"> — </span>'
            f'<span style="color:{CARD_TEXT};font-size:0.9em;">'
            f'{trait["detail"]}</span></div>',
            unsafe_allow_html=True,
        )


# Render identity profile
if primary_key not in profiles:
    st.info("No data available for this platform with the current filters.")
elif compare_key and compare_key in profiles:
    # Overlaid radar chart for comparison
    _render_radar([primary_key, compare_key], profiles)
    with st.expander("What do the radar dimensions mean?"):
        st.markdown(_RADAR_DIMENSION_TABLE)
    # Side-by-side metrics and traits
    left_col, right_col = st.columns(2)
    with left_col:
        _render_metrics_row(primary_key, profiles[primary_key])
        _render_traits(profiles[primary_key]["traits"])
    with right_col:
        _render_metrics_row(compare_key, profiles[compare_key])
        _render_traits(profiles[compare_key]["traits"])
else:
    # Single platform: radar + metrics + traits
    _render_radar([primary_key], profiles)
    with st.expander("What do the radar dimensions mean?"):
        st.markdown(_RADAR_DIMENSION_TABLE)
    _render_metrics_row(primary_key, profiles[primary_key])
    _render_traits(profiles[primary_key]["traits"])


# ==============================================================================
# SECTION 2: CONTENT LANDSCAPE
# ==============================================================================

st.markdown("---")
st.subheader("Content Landscape")
st.caption(
    "A density map of streaming content — where each platform's titles "
    "cluster by thematic and genre similarity."
)

try:
    umap_coords = load_umap_coords()
except FileNotFoundError:
    st.warning(
        "UMAP coordinates not found. Run `scripts/04_compute_umap.py` first."
    )
    umap_coords = None

if umap_coords is not None:
    with st.spinner("Mapping the content landscape..."):
        landscape = compute_landscape_clusters(umap_coords, df)
        # Drop rows where the merge left platform as NaN
        landscape = landscape.dropna(subset=["platform"])

    if landscape.empty:
        st.info("No titles match current filters for landscape visualization.")
    else:
        # Compute cluster summaries (needed for annotations and explorer)
        cluster_summaries = compute_cluster_summaries(landscape)

        if not cluster_summaries:
            st.info("Not enough data to identify content neighborhoods.")
        else:
            # Multi-insight callout — auto-generated from landscape analysis
            insights = compute_landscape_insights(landscape, cluster_summaries)
            if insights:
                insight_items = ""
                for ins in insights:
                    insight_items += (
                        f'<div style="margin-bottom:6px;padding-left:8px;'
                        f'border-left:2px solid {CARD_ACCENT};">'
                        f'{ins["text"]}</div>'
                    )
                st.markdown(
                    f'<div style="background:{CARD_BG};border:1px solid {CARD_BORDER};'
                    f'border-radius:8px;padding:14px 18px;margin-bottom:12px;'
                    f'font-size:0.88em;color:{CARD_TEXT};line-height:1.7;">'
                    f'<strong style="color:{CARD_ACCENT};font-size:1.05em;">'
                    f'Landscape Insights</strong>'
                    f'<div style="margin-top:8px;">{insight_items}</div></div>',
                    unsafe_allow_html=True,
                )

            # Controls row: platform highlight + title search
            ctrl_l, ctrl_m = st.columns([3, 2])
            available_platforms = sorted(landscape["platform"].unique().tolist())
            neighborhood_names = {cid: s["name"] for cid, s in cluster_summaries.items()}

            with ctrl_l:
                highlight_platforms = st.multiselect(
                    "Highlight platforms (overlay individual titles)",
                    options=[PLATFORMS.get(p, {}).get("name", p) for p in available_platforms],
                    default=[],
                    help="Select platforms to overlay sampled title points on the density map",
                )
            highlight_keys = [
                k for k in available_platforms
                if PLATFORMS.get(k, {}).get("name", k) in highlight_platforms
            ]

            with ctrl_m:
                title_search = st.text_input(
                    "Search for a title on the map",
                    placeholder="e.g. Breaking Bad",
                    help="Find where a title sits in the content landscape",
                )

            # Read neighborhood selection from session state (selectbox rendered below map)
            selected_neighborhood = st.session_state.get("dna_sel_neighborhood", "None")

            # Find selected cluster ID
            sel_cid = None
            if selected_neighborhood != "None":
                for cid, hood_name in neighborhood_names.items():
                    if hood_name == selected_neighborhood:
                        sel_cid = cid
                        break

            # Build content landscape figure — archetype-colored scatter
            fig_landscape = go.Figure()

            # Archetype color palette (distinct, readable on dark theme)
            _ARCH_COLORS = {
                0: "#FF6B6B",  # Reality — coral
                1: "#4ECDC4",  # Documentary — teal
                2: "#9B59B6",  # Horror — purple
                3: "#F39C12",  # Animation — orange
                4: "#3498DB",  # Sci-Fi — blue
                5: "#E74C3C",  # Crime — red
                6: "#2ECC71",  # Action — green
                7: "#F1C40F",  # Comedy — yellow
                8: "#E91E63",  # Romance — pink
                9: "#95A5A6",  # Drama — silver
            }

            # Base layer: sampled scatter per archetype
            _max_per_arch = 800
            for cid, summary in cluster_summaries.items():
                arch_data = landscape[landscape["cluster"] == cid]
                if len(arch_data) > _max_per_arch:
                    arch_data = arch_data.sample(n=_max_per_arch, random_state=42)
                a_color = _ARCH_COLORS.get(cid, "#888")

                fig_landscape.add_trace(
                    go.Scatter(
                        x=arch_data["umap_x"],
                        y=arch_data["umap_y"],
                        mode="markers",
                        name=summary["name"],
                        marker=dict(size=4, color=a_color, opacity=0.45),
                        text=arch_data["title"],
                        hovertemplate=(
                            "<b>%{text}</b><br>"
                            + summary["name"]
                            + "<br>IMDb: %{customdata[0]:.1f}"
                            + "<br>Year: %{customdata[1]}<extra></extra>"
                        ),
                        customdata=arch_data[["imdb_score", "release_year"]].fillna(0).values,
                    )
                )

            # Overlay highlighted platforms with larger markers
            for plat_key in highlight_keys:
                plat_data = landscape[landscape["platform"] == plat_key]
                if len(plat_data) > DNA_UMAP_SAMPLE_SIZE:
                    plat_data = plat_data.sample(
                        n=DNA_UMAP_SAMPLE_SIZE, random_state=42
                    )
                p_color = PLATFORMS.get(plat_key, {}).get("color", "#888")
                p_name = PLATFORMS.get(plat_key, {}).get("name", plat_key)

                fig_landscape.add_trace(
                    go.Scatter(
                        x=plat_data["umap_x"],
                        y=plat_data["umap_y"],
                        mode="markers",
                        name=f"{p_name} titles",
                        marker=dict(
                            size=6,
                            color=p_color,
                            opacity=0.8,
                            line=dict(width=0.5, color="#fff"),
                        ),
                        text=plat_data["title"],
                        hovertemplate=(
                            "<b>%{text}</b><br>"
                            f"{p_name}<br>"
                            "IMDb: %{customdata[0]:.2f}<br>"
                            "Year: %{customdata[1]}<extra></extra>"
                        ),
                        customdata=plat_data[["imdb_score", "release_year"]].fillna(0).values,
                    )
                )

            # Neighborhood highlight on map
            if sel_cid is not None:
                hood_data = landscape[landscape["cluster"] == sel_cid]
                fig_landscape.add_trace(
                    go.Scatter(
                        x=hood_data["umap_x"],
                        y=hood_data["umap_y"],
                        mode="markers",
                        name=f"{selected_neighborhood}",
                        marker=dict(
                            size=7,
                            color="rgba(255,215,0,0.5)",
                            line=dict(width=1.5, color="#FFD700"),
                        ),
                        text=hood_data["title"] if "title" in hood_data.columns else None,
                        hovertemplate=(
                            "<b>%{text}</b><br>"
                            "IMDb: %{customdata[0]:.2f}<br>"
                            "Year: %{customdata[1]}<extra></extra>"
                        ) if "title" in hood_data.columns else None,
                        customdata=hood_data[["imdb_score", "release_year"]].fillna(0).values if {"imdb_score", "release_year"}.issubset(hood_data.columns) else None,
                    )
                )

            # Title search highlight
            search_matches = pd.DataFrame()
            if title_search and len(title_search) >= 2:
                search_matches = landscape[
                    landscape["title"].str.contains(title_search, case=False, na=False)
                ]
                if not search_matches.empty:
                    fig_landscape.add_trace(
                        go.Scatter(
                            x=search_matches["umap_x"],
                            y=search_matches["umap_y"],
                            mode="markers+text",
                            name="Search results",
                            marker=dict(
                                size=14,
                                symbol="star",
                                color="#FFD700",
                                line=dict(width=1.5, color="#fff"),
                            ),
                            text=search_matches["title"],
                            textposition="top center",
                            textfont=dict(size=10, color="#FFD700"),
                            hovertemplate=(
                                "<b>%{text}</b><br>"
                                "IMDb: %{customdata[0]:.2f}<br>"
                                "Year: %{customdata[1]}<extra></extra>"
                            ),
                            customdata=search_matches[["imdb_score", "release_year"]].fillna(0).values,
                        )
                    )

            fig_landscape.update_layout(
                xaxis=dict(
                    showticklabels=False, showgrid=False, zeroline=False, title=""
                ),
                yaxis=dict(
                    showticklabels=False, showgrid=False, zeroline=False, title=""
                ),
                template=PLOTLY_TEMPLATE,
                height=CHART_HEIGHT + 200,
                margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(
                    title="Content Categories",
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=10),
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )

            st.plotly_chart(fig_landscape, use_container_width=True)

            st.caption(
                "Dot positions reflect content similarity (based on descriptions). "
                "Colors show genre-based content categories. "
                "Titles in the same category may appear in different map regions "
                "due to diverse descriptions."
            )

            # Neighborhood selector below the map
            st.selectbox(
                "Explore a neighborhood",
                options=["None"] + list(neighborhood_names.values()),
                index=(
                    (["None"] + list(neighborhood_names.values())).index(selected_neighborhood)
                    if selected_neighborhood in (["None"] + list(neighborhood_names.values()))
                    else 0
                ),
                key="dna_sel_neighborhood",
                help="Select a neighborhood to see details and highlight it on the map",
            )

            # Search results info below the map
            if title_search and len(title_search) >= 2:
                if search_matches.empty:
                    st.caption(f'No titles matching "{title_search}" found on the map.')
                else:
                    for _, row in search_matches.head(5).iterrows():
                        cluster_id = row.get("cluster")
                        neighborhood = cluster_summaries.get(cluster_id, {}).get("name", "Unknown")
                        p_name = PLATFORMS.get(row.get("platform", ""), {}).get("name", "")
                        imdb = f"{row['imdb_score']:.2f}" if pd.notna(row.get("imdb_score")) else "N/A"
                        st.caption(
                            f"**{row['title']}** — {p_name} · IMDb {imdb} "
                            f"· Neighborhood: {neighborhood}"
                        )

            # ---- Neighborhood explorer (expanded detail panel) ----
            if sel_cid is not None:
                sel_summary = cluster_summaries[sel_cid]
                sel_data = landscape[landscape["cluster"] == sel_cid]

                # Neighborhood header with description
                st.markdown("---")
                st.markdown(
                    f'<div style="background:{CARD_BG};border:1px solid {CARD_BORDER};'
                    f'border-radius:10px;padding:16px 20px;margin-bottom:12px;">'
                    f'<div style="font-size:1.2em;font-weight:700;color:{CARD_ACCENT};'
                    f'margin-bottom:6px;">{sel_summary["name"]}</div>'
                    f'<div style="color:{CARD_TEXT_MUTED};font-size:0.88em;">'
                    f'A {sel_summary["size"]:,}-title neighborhood centered on '
                    f'{", ".join(sel_summary["top_genres"][:3])}.</div></div>',
                    unsafe_allow_html=True,
                )

                # Genre pills
                genre_pills_html = ""
                for g in sel_summary["top_genres"]:
                    genre_pills_html += (
                        f'<span style="display:inline-block;background:rgba(255,215,0,0.12);'
                        f"color:{CARD_ACCENT};border-radius:10px;padding:3px 10px;"
                        f'font-size:0.82em;margin-right:5px;margin-bottom:4px;">'
                        f"{g}</span>"
                    )
                st.markdown(genre_pills_html, unsafe_allow_html=True)

                # Dominant franchise from enrichment data (if ≥3 titles in cluster)
                _enr_dna = load_enriched_titles()
                if not _enr_dna.empty and "collection_name" in _enr_dna.columns:
                    _cluster_ids = set(sel_data["id"])
                    _cluster_enr = _enr_dna[
                        _enr_dna["id"].isin(_cluster_ids) &
                        _enr_dna["collection_name"].notna()
                    ]
                    if not _cluster_enr.empty:
                        _franchise_counts = _cluster_enr["collection_name"].value_counts()
                        _top_franchise = _franchise_counts.index[0]
                        _top_count = _franchise_counts.iloc[0]
                        if _top_count >= 3:
                            st.markdown(
                                f'<div style="color:{CARD_TEXT_MUTED};font-size:0.85em;margin:4px 0;">'
                                f'Dominant Franchise: <span style="color:{CARD_ACCENT};font-weight:600;">'
                                f'{_top_franchise}</span> ({_top_count} titles)</div>',
                                unsafe_allow_html=True,
                            )

                # Awards-based trait from enrichment data
                if not _enr_dna.empty and "award_wins" in _enr_dna.columns:
                    _cluster_enr_awards = _enr_dna[_enr_dna["id"].isin(_cluster_ids)]
                    _total_wins = _cluster_enr_awards["award_wins"].fillna(0).sum()
                    _n_titles = len(_cluster_enr_awards)
                    _wins_per_1k = (_total_wins / _n_titles * 1000) if _n_titles > 0 else 0
                    _award_coverage = _cluster_enr_awards["award_wins"].notna().mean()
                    if _award_coverage >= 0.15 and _total_wins > 5:
                        st.markdown(
                            f'<div style="color:{CARD_TEXT_MUTED};font-size:0.85em;margin:4px 0;">'
                            f'Awards Magnet: <span style="color:#FFD700;font-weight:600;">'
                            f'{int(_total_wins)} wins</span>, {_wins_per_1k:.0f} per 1,000 titles</div>',
                            unsafe_allow_html=True,
                        )

                # Leaders line
                pl = sel_summary.get("platform_leader")
                ql = sel_summary.get("quality_leader")
                leader_parts = []
                if pl:
                    pl_color = PLATFORMS.get(pl, {}).get("color", CARD_TEXT_MUTED)
                    pl_name = PLATFORMS.get(pl, {}).get("name", pl)
                    leader_parts.append(
                        f'Volume leader: <span style="color:{pl_color};'
                        f'font-weight:600;">{pl_name}</span>'
                    )
                if ql and ql != pl:
                    ql_color = PLATFORMS.get(ql, {}).get("color", CARD_TEXT_MUTED)
                    ql_name = PLATFORMS.get(ql, {}).get("name", ql)
                    leader_parts.append(
                        f'Quality leader: <span style="color:{ql_color};'
                        f'font-weight:600;">{ql_name}</span>'
                    )
                if leader_parts:
                    st.markdown(
                        f'<div style="color:{CARD_TEXT_MUTED};font-size:0.85em;'
                        f'margin:4px 0 8px;">'
                        + " · ".join(leader_parts) + "</div>",
                        unsafe_allow_html=True,
                    )

                # Quality metrics row
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric(
                        "Titles", f"{sel_summary['size']:,}",
                        help="Total titles in this content neighborhood",
                    )
                with m2:
                    avg_label = f"{sel_summary['avg_imdb']:.2f}" if sel_summary["avg_imdb"] else "N/A"
                    st.metric(
                        "Avg IMDb", avg_label,
                        help="Average IMDb rating of rated titles in this neighborhood",
                    )
                with m3:
                    rated_in_cluster = sel_data.dropna(subset=["imdb_score"])
                    premium = rated_in_cluster[rated_in_cluster["imdb_score"] >= 7.0]
                    prem_pct = len(premium) / len(rated_in_cluster) * 100 if len(rated_in_cluster) > 0 else 0
                    st.metric(
                        "Premium (7.0+)", f"{prem_pct:.1f}%",
                        help="Percentage of rated titles scoring 7.0 or higher on IMDb",
                    )
                with m4:
                    med_year = int(sel_data["release_year"].median()) if sel_data["release_year"].notna().any() else "N/A"
                    st.metric(
                        "Median Year", med_year,
                        help="The middle release year — half the titles are older, half newer",
                    )

                # Platform breakdown bar
                if sel_summary["platform_mix"]:
                    plat_bar_data = []
                    for pk in ALL_PLATFORMS:
                        count = sel_summary["platform_mix"].get(pk, 0)
                        if count > 0:
                            plat_bar_data.append({
                                "Platform": PLATFORMS.get(pk, {}).get("name", pk),
                                "Count": count,
                                "Color": PLATFORMS.get(pk, {}).get("color", "#888"),
                            })
                    if plat_bar_data:
                        bar_df = pd.DataFrame(plat_bar_data)
                        color_map = {r["Platform"]: r["Color"] for r in plat_bar_data}
                        fig_bar = px.bar(
                            bar_df, x="Count", y="Platform", orientation="h",
                            color="Platform", color_discrete_map=color_map,
                            template=PLOTLY_TEMPLATE, height=200,
                        )
                        fig_bar.update_layout(
                            showlegend=False,
                            margin=dict(l=10, r=10, t=10, b=10),
                            yaxis=dict(categoryorder="total ascending"),
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)

                # Top 10 titles
                st.markdown("**Top Titles**")
                rated_cluster = sel_data.dropna(subset=["imdb_score"]).copy()
                if len(rated_cluster) > 0:
                    from src.analysis.scoring import compute_quality_score as _qs
                    rated_cluster["quality_score"] = _qs(rated_cluster)

                    # Genre-relevance boost using cluster's actual top genres
                    _top_genres = sel_summary.get("cluster_top_genres", set())

                    def _gr(gv):
                        if not isinstance(gv, list):
                            return 0.0
                        m = sum(1 for g in gv if str(g).lower() in _top_genres)
                        return m / max(len(gv), 1)

                    rated_cluster["genre_rel"] = rated_cluster["genres"].apply(_gr)
                    rated_cluster["final_score"] = (
                        rated_cluster["quality_score"] * 0.7
                        + rated_cluster["genre_rel"] * 0.3 * 100
                    )

                    for threshold in [10_000, 1_000, 0]:
                        pool = rated_cluster[rated_cluster["imdb_votes"].fillna(0) >= threshold]
                        if len(pool) >= 10:
                            break
                    else:
                        pool = rated_cluster
                    top_titles = pool.nlargest(10, "final_score")
                    t_cols = st.columns(2)
                    for idx, (_, row) in enumerate(top_titles.iterrows()):
                        with t_cols[idx % 2]:
                            p_color = PLATFORMS.get(row.get("platform", ""), {}).get("color", CARD_BORDER)
                            p_name = PLATFORMS.get(row.get("platform", ""), {}).get("name", "")
                            imdb_str = f"{row['imdb_score']:.2f}" if pd.notna(row.get("imdb_score")) else "N/A"
                            year_str = str(int(row["release_year"])) if pd.notna(row.get("release_year")) else ""
                            # Genre pills for title
                            title_genres = ""
                            if isinstance(row.get("genres"), list):
                                for g in row["genres"][:3]:
                                    title_genres += (
                                        f'<span style="display:inline-block;'
                                        f"background:rgba(255,255,255,0.06);"
                                        f"color:{CARD_TEXT_MUTED};border-radius:8px;"
                                        f'padding:1px 6px;font-size:0.72em;'
                                        f'margin-right:3px;">{g.title()}</span>'
                                    )
                            st.markdown(
                                f'<div style="background:{CARD_BG};border-left:3px solid {p_color};'
                                f'border-radius:6px;padding:10px 14px;margin-bottom:2px;">'
                                f'<span style="color:{CARD_TEXT};font-weight:600;">{row["title"]}</span>'
                                f'<br/><span style="color:{CARD_TEXT_MUTED};font-size:0.82em;">'
                                f'{p_name} · {year_str} · IMDb {imdb_str} · {row.get("type", "")}</span>'
                                f'<br/>{title_genres}</div>',
                                unsafe_allow_html=True,
                            )
                            title_id = row.get("id")
                            if title_id is not None and st.button(
                                "View Details",
                                key=f"hub_detail_{title_id}_{idx}",
                                use_container_width=True,
                            ):
                                st.session_state["explore_selected_id"] = title_id
                                st.switch_page("pages/01_Explore_Catalog.py")
                else:
                    st.caption("No rated titles in this neighborhood.")


# ==============================================================================
# SECTION 3: WHAT PLATFORM ARE YOU? — Hybrid Quiz
# ==============================================================================

st.markdown("---")
st.subheader("What Platform Are You?")
st.caption(
    "Pick your favorite genres, swipe through title cards, "
    "and discover which streaming platform is your perfect match."
)

# Session state for quiz flow
st.session_state.setdefault("dna_quiz_phase", "A")
st.session_state.setdefault("dna_quiz_prefs", {})
st.session_state.setdefault("dna_quiz_titles", [])
st.session_state.setdefault("dna_quiz_current", 0)
st.session_state.setdefault("dna_quiz_liked", [])
st.session_state.setdefault("dna_quiz_results", None)

_quiz_phase = st.session_state["dna_quiz_phase"]


def _reset_quiz():
    st.session_state["dna_quiz_phase"] = "A"
    st.session_state["dna_quiz_prefs"] = {}
    st.session_state["dna_quiz_titles"] = []
    st.session_state["dna_quiz_current"] = 0
    st.session_state["dna_quiz_liked"] = []
    st.session_state["dna_quiz_results"] = None


# Collect genre options
_all_genre_counts: dict[str, int] = {}
for _g_list in raw_df["genres"].dropna():
    if isinstance(_g_list, list):
        for _g in _g_list:
            _all_genre_counts[_g] = _all_genre_counts.get(_g, 0) + 1
_sorted_genres = sorted(_all_genre_counts.keys(), key=lambda g: -_all_genre_counts[g])

# Genre display with icons
_GENRE_ICONS = {
    "drama": "🎭", "comedy": "😂", "thriller": "😰", "action": "💥",
    "romance": "💕", "documentation": "🎬", "crime": "🔍", "family": "👨‍👩‍👧‍👦",
    "animation": "✏️", "scifi": "🚀", "fantasy": "🐉", "horror": "👻",
    "music": "🎵", "history": "📜", "western": "🤠", "war": "⚔️",
    "sport": "⚽", "reality": "📺", "european": "🌍",
}
_GENRE_DISPLAY_NAMES = {
    "documentation": "Documentary", "scifi": "Sci-Fi",
}

# ---- PHASE A: Genre Selection + Preferences ----
if _quiz_phase == "A":
    st.markdown(
        f'<div style="color:{CARD_ACCENT};font-weight:600;font-size:1.05em;'
        f'margin-bottom:8px;">Step 1: Pick Your Favorites</div>',
        unsafe_allow_html=True,
    )

    # Genre toggle grid — use session state for selection
    st.session_state.setdefault("dna_quiz_genres", [])

    _top_genres = _sorted_genres[:12]
    _gcols = st.columns(4)
    for gi, gkey in enumerate(_top_genres):
        with _gcols[gi % 4]:
            icon = _GENRE_ICONS.get(gkey, "🎬")
            label = _GENRE_DISPLAY_NAMES.get(gkey, gkey.title())
            is_selected = gkey in st.session_state["dna_quiz_genres"]
            border_color = CARD_ACCENT if is_selected else CARD_BORDER
            bg = f"rgba(255,215,0,0.08)" if is_selected else CARD_BG
            if st.button(
                f"{icon} {label}",
                key=f"genre_toggle_{gkey}",
                use_container_width=True,
            ):
                if gkey in st.session_state["dna_quiz_genres"]:
                    st.session_state["dna_quiz_genres"].remove(gkey)
                else:
                    st.session_state["dna_quiz_genres"].append(gkey)
                st.rerun()

    sel_genres = st.session_state["dna_quiz_genres"]
    if sel_genres:
        sel_labels = [_GENRE_DISPLAY_NAMES.get(g, g.title()) for g in sel_genres]
        st.markdown(
            f'<div style="color:{CARD_TEXT_MUTED};font-size:0.85em;margin:4px 0 12px;">'
            f'Selected: <span style="color:{CARD_ACCENT};">{", ".join(sel_labels)}</span></div>',
            unsafe_allow_html=True,
        )

    # Quick preferences
    st.markdown(
        f'<div style="color:{CARD_ACCENT};font-weight:600;font-size:1.05em;'
        f'margin:16px 0 8px;">Step 2: Set Your Vibe</div>',
        unsafe_allow_html=True,
    )
    pref_c1, pref_c2, pref_c3 = st.columns(3)
    with pref_c1:
        quality_pref = st.radio(
            "What do you look for?",
            options=["No Preference", "Award Winners", "Crowd Favorites"],
            index=0,
            help="Award Winners = high IMDb ratings, Crowd Favorites = popular hits",
        )
    with pref_c2:
        type_pref = st.radio(
            "Movies or shows?",
            options=["Both", "Movies", "Shows"],
            index=0,
        )
    with pref_c3:
        vibe_pref = st.radio(
            "What's your vibe?",
            options=["Mix of Both", "Feel-Good & Light", "Dark & Intense"],
            index=0,
        )

    # Start quiz button
    if st.button(
        "Start the Quiz",
        use_container_width=True,
        disabled=len(sel_genres) == 0,
        help="Select at least one genre to begin",
    ):
        prefs = {
            "genres": sel_genres,
            "quality": quality_pref,
            "type": type_pref,
            "vibe": vibe_pref,
        }
        with st.spinner("Curating titles for you..."):
            titles = curate_quiz_titles(
                raw_df,
                selected_genres=sel_genres,
                quality_pref=quality_pref,
                type_pref=type_pref,
                vibe_pref=vibe_pref,
            )
        st.session_state["dna_quiz_prefs"] = prefs
        st.session_state["dna_quiz_titles"] = titles
        st.session_state["dna_quiz_current"] = 0
        st.session_state["dna_quiz_liked"] = []
        st.session_state["dna_quiz_phase"] = "B"
        st.rerun()

    if len(sel_genres) == 0:
        st.caption("Select at least one genre above to start the quiz.")


# ---- PHASE B: Title Swipe ----
elif _quiz_phase == "B":
    quiz_titles = st.session_state["dna_quiz_titles"]
    current_idx = st.session_state["dna_quiz_current"]

    if not quiz_titles:
        st.warning("Could not find enough titles. Try different preferences.")
        if st.button("Back to Preferences"):
            _reset_quiz()
            st.rerun()
    elif current_idx >= len(quiz_titles):
        # All titles swiped — go to results
        st.session_state["dna_quiz_phase"] = "C"
        st.rerun()
    else:
        title = quiz_titles[current_idx]
        progress = (current_idx + 1) / len(quiz_titles)
        st.progress(progress, text=f"Title {current_idx + 1} of {len(quiz_titles)}")

        # Title card
        t_platform = title.get("platform", "")
        t_color = PLATFORMS.get(t_platform, {}).get("color", CARD_BORDER)
        t_pname = PLATFORMS.get(t_platform, {}).get("name", t_platform)
        t_imdb = f"{title['imdb_score']:.1f}" if title.get("imdb_score") else "N/A"
        t_year = str(int(title["release_year"])) if title.get("release_year") else ""
        t_type = title.get("type", "")

        # Genre pills
        genre_pills = ""
        for g in (title.get("genres") or [])[:4]:
            genre_pills += (
                f'<span style="display:inline-block;background:rgba(255,255,255,0.06);'
                f"color:{CARD_TEXT_MUTED};border-radius:8px;padding:2px 8px;"
                f'font-size:0.78em;margin-right:4px;">{str(g).title()}</span>'
            )

        # Platform badges
        plat_badges = ""
        for p in (title.get("platforms") or [title.get("platform", "")]):
            pc = PLATFORMS.get(p, {}).get("color", CARD_BORDER)
            pn = PLATFORMS.get(p, {}).get("name", p)
            plat_badges += (
                f'<span style="display:inline-block;background:{pc};color:#fff;'
                f'border-radius:4px;padding:1px 8px;font-size:0.75em;'
                f'margin-right:4px;font-weight:600;">{pn}</span>'
            )

        desc = title.get("description", "")

        st.markdown(
            f'<div style="background:{CARD_BG};border:2px solid {t_color};'
            f'border-radius:12px;padding:24px;max-width:600px;margin:0 auto 16px;">'
            f'<div style="font-size:1.3em;font-weight:700;color:{CARD_TEXT};'
            f'margin-bottom:6px;">{title["title"]}</div>'
            f'<div style="margin-bottom:8px;">{plat_badges}</div>'
            f'<div style="color:{CARD_TEXT_MUTED};font-size:0.88em;margin-bottom:8px;">'
            f'{t_year} · {t_type} · IMDb {t_imdb}</div>'
            f'<div style="margin-bottom:10px;">{genre_pills}</div>'
            f'<div style="color:{CARD_TEXT};font-size:0.92em;line-height:1.6;">'
            f'{desc}</div></div>',
            unsafe_allow_html=True,
        )

        # Swipe buttons
        btn_l, btn_r = st.columns(2)
        with btn_l:
            if st.button(
                "Not For Me",
                key=f"quiz_pass_{current_idx}",
                use_container_width=True,
            ):
                st.session_state["dna_quiz_current"] = current_idx + 1
                st.rerun()
        with btn_r:
            if st.button(
                "I'd Watch This",
                key=f"quiz_like_{current_idx}",
                use_container_width=True,
                type="primary",
            ):
                st.session_state["dna_quiz_liked"].append(title["id"])
                st.session_state["dna_quiz_current"] = current_idx + 1
                st.rerun()

        # Skip to results button
        if current_idx >= 3:
            if st.button("Skip to results", key="quiz_skip"):
                st.session_state["dna_quiz_phase"] = "C"
                st.rerun()


# ---- PHASE C: Results ----
elif _quiz_phase == "C":
    liked_ids = st.session_state["dna_quiz_liked"]
    quiz_titles = st.session_state["dna_quiz_titles"]

    if not liked_ids:
        st.info("You didn't like any titles. Try again with different preferences!")
        if st.button("Start Over"):
            _reset_quiz()
            st.rerun()
    else:
        # Compute results if not cached
        if st.session_state["dna_quiz_results"] is None:
            with st.spinner("Analyzing your taste..."):
                st.session_state["dna_quiz_results"] = compute_swipe_results(
                    liked_ids, quiz_titles, raw_df
                )

        results = st.session_state["dna_quiz_results"]
        rankings = results.get("rankings", [])
        personality = results.get("personality", "")
        recommendations = results.get("recommendations", [])

        if rankings:
            # Hero card — best match
            best = rankings[0]
            best_color = PLATFORMS.get(best["platform"], {}).get("color", CARD_ACCENT)

            st.markdown(
                f'<div style="background:{CARD_BG};border:2px solid {best_color};'
                f'border-radius:12px;padding:24px;text-align:center;margin:16px 0;">'
                f'<div style="font-size:0.9em;color:{CARD_TEXT_MUTED};">'
                f"You're a</div>"
                f'<div style="font-size:2em;font-weight:700;color:{best_color};'
                f'margin:4px 0;">{best["display_name"]} Person</div>'
                f'<div style="font-size:1.4em;color:{CARD_ACCENT};margin-bottom:8px;">'
                f'{best["match_pct"]:.1f}% Match</div>'
                f'<div style="font-size:0.9em;color:{CARD_TEXT};line-height:1.6;'
                f'max-width:500px;margin:0 auto;">{best["explanation"]}</div></div>',
                unsafe_allow_html=True,
            )

            # Personality summary
            if personality:
                st.markdown(
                    f'<div style="background:{CARD_BG};border:1px solid {CARD_BORDER};'
                    f'border-radius:8px;padding:12px 16px;margin-bottom:12px;'
                    f'font-size:0.92em;color:{CARD_TEXT};line-height:1.6;">'
                    f'<strong style="color:{CARD_ACCENT};">Your Viewing DNA:</strong> '
                    f'{personality}</div>',
                    unsafe_allow_html=True,
                )

            # Liked count
            st.caption(
                f"You liked {len(liked_ids)} of {len(quiz_titles)} titles "
                f"({len(liked_ids)/max(len(quiz_titles),1)*100:.0f}% swipe rate)."
            )

            # Recommendations from best platform
            if recommendations:
                st.markdown(
                    f'<div style="color:{CARD_ACCENT};font-weight:600;font-size:1.05em;'
                    f'margin:16px 0 8px;">Titles You\'ll Love on {best["display_name"]}</div>',
                    unsafe_allow_html=True,
                )
                rec_cols = st.columns(min(len(recommendations), 5))
                for ri, rec in enumerate(recommendations[:5]):
                    with rec_cols[ri % len(rec_cols)]:
                        r_imdb = f"{rec['imdb_score']:.1f}" if rec.get("imdb_score") else "N/A"
                        r_year = str(int(rec["release_year"])) if rec.get("release_year") else ""
                        r_genres = ""
                        if isinstance(rec.get("genres"), list):
                            for rg in rec["genres"][:2]:
                                r_genres += (
                                    f'<span style="display:inline-block;'
                                    f"background:rgba(255,255,255,0.06);"
                                    f"color:{CARD_TEXT_MUTED};border-radius:8px;"
                                    f'padding:1px 6px;font-size:0.72em;'
                                    f'margin-right:3px;">{str(rg).title()}</span>'
                                )
                        st.markdown(
                            f'<div style="background:{CARD_BG};border-left:3px solid {best_color};'
                            f'border-radius:6px;padding:10px 12px;margin-bottom:6px;">'
                            f'<span style="color:{CARD_TEXT};font-weight:600;font-size:0.9em;">'
                            f'{rec["title"]}</span>'
                            f'<br/><span style="color:{CARD_TEXT_MUTED};font-size:0.78em;">'
                            f'{r_year} · IMDb {r_imdb}</span>'
                            f'<br/>{r_genres}</div>',
                            unsafe_allow_html=True,
                        )

            # Match bar chart
            match_df = pd.DataFrame(rankings)
            match_colors = {
                PLATFORMS.get(k, {}).get("name", k): PLATFORMS.get(k, {}).get("color", "#888")
                for k in ALL_PLATFORMS
            }
            fig_match = px.bar(
                match_df,
                x="display_name",
                y="match_pct",
                color="display_name",
                color_discrete_map=match_colors,
                template=PLOTLY_TEMPLATE,
                height=350,
                labels={"match_pct": "Match %", "display_name": ""},
            )
            fig_match.update_layout(
                showlegend=False,
                yaxis_range=[0, 100],
                margin=dict(t=20, b=10, l=10, r=10),
            )
            fig_match.update_traces(
                texttemplate="%{y:.1f}%",
                textposition="outside",
                textfont_size=13,
            )
            st.plotly_chart(fig_match, use_container_width=True)

            # Runner-up cards
            for result in rankings[1:]:
                p_color = PLATFORMS.get(result["platform"], {}).get("color", CARD_BORDER)
                st.markdown(
                    f'<div style="background:{CARD_BG};border-left:3px solid {p_color};'
                    f'border-radius:6px;padding:10px 14px;margin-bottom:6px;">'
                    f'<span style="color:{CARD_TEXT};font-weight:600;">'
                    f'{result["display_name"]}</span>'
                    f' <span style="color:{CARD_ACCENT};font-weight:600;">'
                    f'{result["match_pct"]:.1f}%</span>'
                    f' &mdash; <span style="color:{CARD_TEXT_MUTED};font-size:0.88em;">'
                    f'{result["explanation"]}</span></div>',
                    unsafe_allow_html=True,
                )

        # Retake quiz button
        if st.button("Take Quiz Again", use_container_width=True):
            _reset_quiz()
            st.rerun()


# -- footer --------------------------------------------------------------------

st.markdown("---")
st.caption(
    "Hypothetical merger for academic analysis. "
    "Data is a snapshot (mid-2023). "
    "All insights are illustrative, not prescriptive."
)
