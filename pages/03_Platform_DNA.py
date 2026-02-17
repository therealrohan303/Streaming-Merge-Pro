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
    compute_user_match_scores,
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
from src.data.loaders import load_all_platforms_titles, load_umap_coords
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

            # Controls row: platform highlight + title search + neighborhood
            ctrl_l, ctrl_m, ctrl_r = st.columns([3, 2, 2])
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

            with ctrl_r:
                selected_neighborhood = st.selectbox(
                    "Explore a neighborhood",
                    options=["None"] + list(neighborhood_names.values()),
                    index=0,
                    help="Select a neighborhood to see details and highlight it on the map",
                )

            # Find selected cluster ID
            sel_cid = None
            if selected_neighborhood != "None":
                for cid, hood_name in neighborhood_names.items():
                    if hood_name == selected_neighborhood:
                        sel_cid = cid
                        break

            # Build density contour figure
            fig_landscape = go.Figure()

            # Per-platform density contours
            for plat_key in available_platforms:
                plat_data = landscape[landscape["platform"] == plat_key]
                if len(plat_data) < 10:
                    continue
                p_color = PLATFORMS.get(plat_key, {}).get("color", "#888")
                p_name = PLATFORMS.get(plat_key, {}).get("name", plat_key)

                fig_landscape.add_trace(
                    go.Histogram2dContour(
                        x=plat_data["umap_x"],
                        y=plat_data["umap_y"],
                        name=p_name,
                        colorscale=[
                            [0, "rgba(0,0,0,0)"],
                            [0.3, _hex_to_rgba(p_color, 0.2)],
                            [1, _hex_to_rgba(p_color, 0.67)],
                        ],
                        contours=dict(coloring="fill", showlabels=False),
                        ncontours=20,
                        showscale=False,
                        opacity=0.45,
                        hoverinfo="skip",
                        line=dict(width=0.5, color=_hex_to_rgba(p_color, 0.4)),
                    )
                )

            # Overlay sampled points for highlighted platforms
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
                            size=5,
                            color=p_color,
                            opacity=0.7,
                            line=dict(width=0.5, color="#fff"),
                        ),
                        text=plat_data["title"],
                        hovertemplate=(
                            "<b>%{text}</b><br>"
                            f"{p_name}<br>"
                            "IMDb: %{customdata[0]:.2f}<br>"
                            "Year: %{customdata[1]}<extra></extra>"
                        ),
                        customdata=plat_data[["imdb_score", "release_year"]].values,
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
                            size=6,
                            color="rgba(255,215,0,0.4)",
                            line=dict(width=1, color="#FFD700"),
                        ),
                        text=hood_data["title"] if "title" in hood_data.columns else None,
                        hovertemplate=(
                            "<b>%{text}</b><br>"
                            "IMDb: %{customdata[0]:.2f}<br>"
                            "Year: %{customdata[1]}<extra></extra>"
                        ) if "title" in hood_data.columns else None,
                        customdata=hood_data[["imdb_score", "release_year"]].values if {"imdb_score", "release_year"}.issubset(hood_data.columns) else None,
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
                            customdata=search_matches[["imdb_score", "release_year"]].values,
                        )
                    )

            # Cluster center annotations
            for cid, summary in cluster_summaries.items():
                center_x = landscape[landscape["cluster"] == cid]["umap_x"].mean()
                center_y = landscape[landscape["cluster"] == cid]["umap_y"].mean()
                fig_landscape.add_annotation(
                    x=center_x,
                    y=center_y,
                    text=summary["name"],
                    showarrow=False,
                    font=dict(size=10, color="#FAFAFA", family="Arial"),
                    bgcolor="rgba(30,30,46,0.75)",
                    borderpad=3,
                    bordercolor="rgba(255,255,255,0.15)",
                    borderwidth=1,
                )

            fig_landscape.update_layout(
                xaxis=dict(
                    showticklabels=False, showgrid=False, zeroline=False, title=""
                ),
                yaxis=dict(
                    showticklabels=False, showgrid=False, zeroline=False, title=""
                ),
                template=PLOTLY_TEMPLATE,
                height=CHART_HEIGHT + 180,
                margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(
                    title="",
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=11),
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )

            st.plotly_chart(fig_landscape, use_container_width=True)

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
                    for threshold in [10_000, 1_000, 0]:
                        pool = rated_cluster[rated_cluster["imdb_votes"].fillna(0) >= threshold]
                        if len(pool) >= 10:
                            break
                    else:
                        pool = rated_cluster
                    top_titles = pool.nlargest(10, "quality_score")
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
                                f'border-radius:6px;padding:10px 14px;margin-bottom:6px;">'
                                f'<span style="color:{CARD_TEXT};font-weight:600;">{row["title"]}</span>'
                                f'<br/><span style="color:{CARD_TEXT_MUTED};font-size:0.82em;">'
                                f'{p_name} · {year_str} · IMDb {imdb_str} · {row.get("type", "")}</span>'
                                f'<br/>{title_genres}</div>',
                                unsafe_allow_html=True,
                            )
                else:
                    st.caption("No rated titles in this neighborhood.")


# ==============================================================================
# SECTION 3: WHAT PLATFORM ARE YOU?
# ==============================================================================

st.markdown("---")
st.subheader("What Platform Are You?")
st.caption(
    "Tell us your preferences and we'll find the streaming platform "
    "that best matches your taste."
)

# Collect all genres for the multiselect
_all_genre_counts: dict[str, int] = {}
for genres in raw_df["genres"].dropna():
    if isinstance(genres, list):
        for g in genres:
            _all_genre_counts[g] = _all_genre_counts.get(g, 0) + 1
_sorted_genres = sorted(_all_genre_counts.keys(), key=lambda g: -_all_genre_counts[g])
_genre_display = [g.title() for g in _sorted_genres]
_display_to_raw = {g.title(): g for g in _sorted_genres}

with st.form("platform_matcher_form"):
    selected_genre_labels = st.multiselect(
        "Your favorite genres",
        options=_genre_display,
        default=[],
        help="Select the genres you enjoy most",
    )

    sl1, sl2 = st.columns(2)
    with sl1:
        recency = st.slider(
            "Classics vs New Releases",
            min_value=0,
            max_value=100,
            value=50,
            help="0 = Classic films and shows, 100 = Latest releases only",
        )
        popularity = st.slider(
            "Hidden Gems vs Blockbusters",
            min_value=0,
            max_value=100,
            value=50,
            help="0 = Hidden gems and indie content, 100 = Mainstream hits",
        )
        runtime_pref = st.slider(
            "Shorter vs Longer Content",
            min_value=0,
            max_value=100,
            value=50,
            help="0 = Quick watches (under 90 min), 100 = Longer epics",
        )

    with sl2:
        maturity = st.slider(
            "Family-Friendly vs Mature",
            min_value=0,
            max_value=100,
            value=50,
            help="0 = Family-friendly (G, PG), 100 = Mature (R, TV-MA)",
        )
        content_type = st.slider(
            "Movies vs Shows",
            min_value=0,
            max_value=100,
            value=50,
            help="0 = Primarily series, 100 = Primarily movies",
        )
        international = st.slider(
            "Domestic vs International",
            min_value=0,
            max_value=100,
            value=50,
            help="0 = US/domestic content, 100 = International content",
        )

    submitted = st.form_submit_button(
        "Find My Platform", use_container_width=True
    )

if submitted:
    if not selected_genre_labels:
        st.warning("Please select at least one genre to find your match.")
    else:
        user_prefs = {
            "genres": [_display_to_raw[g] for g in selected_genre_labels],
            "recency": recency,
            "popularity": popularity,
            "runtime": runtime_pref,
            "maturity": maturity,
            "content_type": content_type,
            "international": international,
        }

        with st.spinner("Finding your perfect platform match..."):
            results = compute_user_match_scores(user_prefs, raw_df)

        if not results:
            st.info("Unable to compute match scores. Try adjusting your filters.")
        else:
            # Best match hero card
            best = results[0]
            best_color = PLATFORMS[best["platform"]]["color"]

            st.markdown(
                f'<div style="background:{CARD_BG};border:2px solid {best_color};'
                f'border-radius:12px;padding:24px;text-align:center;margin:16px 0;">'
                f'<div style="font-size:0.9em;color:{CARD_TEXT_MUTED};">'
                f"Your best match is</div>"
                f'<div style="font-size:2em;font-weight:700;color:{best_color};'
                f'margin:4px 0;">{best["display_name"]}</div>'
                f'<div style="font-size:1.4em;color:{CARD_ACCENT};margin-bottom:8px;">'
                f'{best["match_pct"]:.1f}% Match</div>'
                f'<div style="font-size:0.9em;color:{CARD_TEXT};line-height:1.6;'
                f'max-width:500px;margin:0 auto;">{best["explanation"]}</div></div>',
                unsafe_allow_html=True,
            )

            # Your Viewing DNA personality card
            dna_traits = []
            if recency > 70:
                dna_traits.append("new release hunter")
            elif recency < 30:
                dna_traits.append("classics lover")
            if popularity > 70:
                dna_traits.append("mainstream fan")
            elif popularity < 30:
                dna_traits.append("hidden gem seeker")
            if maturity > 70:
                dna_traits.append("mature content viewer")
            elif maturity < 30:
                dna_traits.append("family-friendly watcher")
            if international > 70:
                dna_traits.append("global cinema explorer")
            elif international < 30:
                dna_traits.append("domestic content fan")
            if content_type > 70:
                dna_traits.append("movie buff")
            elif content_type < 30:
                dna_traits.append("series binger")
            genre_str = ", ".join(selected_genre_labels[:4])
            if dna_traits:
                persona = ", ".join(dna_traits[:3])
                st.markdown(
                    f'<div style="background:{CARD_BG};border:1px solid {CARD_BORDER};'
                    f'border-radius:8px;padding:12px 16px;margin-bottom:12px;'
                    f'font-size:0.88em;color:{CARD_TEXT};line-height:1.6;">'
                    f'<strong style="color:{CARD_ACCENT};">Your Viewing DNA:</strong> '
                    f"You're a {persona} who loves {genre_str}.</div>",
                    unsafe_allow_html=True,
                )

            # Match bar chart for all platforms
            match_df = pd.DataFrame(results)
            match_colors = {
                PLATFORMS[k]["name"]: PLATFORMS[k]["color"] for k in ALL_PLATFORMS
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

            # Per-platform explanation cards
            for result in results[1:]:
                p_color = PLATFORMS[result["platform"]]["color"]
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


# -- footer --------------------------------------------------------------------

st.markdown("---")
st.caption(
    "Hypothetical merger for academic analysis. "
    "Data is a snapshot (mid-2023). "
    "All insights are illustrative, not prescriptive."
)
