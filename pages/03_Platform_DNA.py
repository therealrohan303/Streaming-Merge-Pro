"""Page 3: Platform DNA — identity fingerprints, content landscape, and your personal platform match."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analysis.platform_dna import (
    compute_cluster_summaries,
    compute_enriched_platform_stats,
    compute_landscape_clusters,
    compute_landscape_insights_v2,
    compute_neighborhood_top_titles,
    compute_platform_comparison_data,
    compute_swipe_results_v2,
    curate_quiz_titles,
)
from src.analysis.scoring import compute_quality_score, format_votes
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
from src.data.loaders import (
    load_all_platforms_credits,
    load_all_platforms_titles,
    load_enriched_titles,
    load_umap_coords,
)
from src.ui.badges import page_header_html, platform_badges_html, section_header_html, styled_metric_card_html
from src.ui.filters import apply_filters, render_sidebar_filters
from src.ui.session import init_session_state

# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title=f"Platform DNA | {PAGE_TITLE}",
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state=INITIAL_SIDEBAR_STATE,
)

st.markdown(
    """
<style>
    [data-testid="column"] { overflow-wrap: break-word; word-wrap: break-word; word-break: break-word; }
    [data-testid="column"] .stMarkdown div { overflow-wrap: break-word; word-wrap: break-word; }
    [data-testid="stMetricLabel"] { overflow: visible !important; white-space: normal !important; }
    div[data-testid="stSlider"] > label { font-size: 0.88em !important; }
</style>
""",
    unsafe_allow_html=True,
)

init_session_state()

# Page-specific session state
st.session_state.setdefault("dna_platform", "merged")
st.session_state.setdefault("dna_compare_platform", None)
st.session_state.setdefault("dna_normalized", False)
st.session_state.setdefault("dna_sel_neighborhood", "None")
st.session_state.setdefault("dna_neighborhood_title_id", None)

# ── data loading ─────────────────────────────────────────────────────────────

raw_df = load_all_platforms_titles()
enriched_df = load_enriched_titles()
filters = render_sidebar_filters(raw_df)
df = apply_filters(raw_df, filters)

# ── page header ──────────────────────────────────────────────────────────────

st.markdown(
    page_header_html(
        "Platform DNA",
        "What makes each platform unique — identity fingerprints, content landscape patterns, and your personal streaming match",
    ),
    unsafe_allow_html=True,
)

with st.expander("How Platform DNA Works"):
    st.markdown(
        "**Identity Radar** maps 6 normalized dimensions (0–100) for each platform, comparing it against "
        "all others: Freshness (median release year), Quality (avg IMDb), Breadth (catalog size), "
        "Global Reach (international content %), Genre Diversity (unique genres), and Series Focus (show %). "
        "Enrichment data from Wikidata (award wins, production budgets) and TMDB (franchises, posters) "
        "powers the **Defining Traits** section.\n\n"
        "**Content Landscape** uses precomputed UMAP coordinates to project titles into 2D based on "
        "description similarity. Genre-priority rules assign each title to one of 10 content archetypes. "
        "Insights are generated from actual cluster ownership data — not generic platitudes.\n\n"
        "**Platform Matcher Quiz** collects genre preferences, 6 preference sliders, and a title swipe "
        "session, then combines all signals (genre cosine similarity, platform affinity, quality alignment, "
        "slider dimension matching) to rank all 6 platforms against your taste profile."
    )


# ── helpers shared across sections ───────────────────────────────────────────

def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _meta_cell(label: str, value: str) -> str:
    return (
        f'<div style="padding:8px 0;">'
        f'<div style="color:{CARD_TEXT_MUTED};font-size:0.72em;text-transform:uppercase;'
        f'letter-spacing:0.04em;margin-bottom:3px;">{label}</div>'
        f'<div style="font-weight:600;color:{CARD_TEXT};font-size:0.95em;">{value}</div>'
        f'</div>'
    )


def _render_inline_title_detail(title_id, all_df, enriched_df, credits_df=None):
    """Render a full title detail panel inline — same format as Explore Catalog."""
    sel_rows = all_df[all_df["id"] == title_id]
    if sel_rows.empty:
        st.warning("Title not found.")
        return
    sel_rows = sel_rows.copy()
    sel_rows["quality_score"] = compute_quality_score(sel_rows)
    sel = sel_rows.iloc[0]

    _enr_row = enriched_df[enriched_df["id"] == title_id] if enriched_df is not None and not enriched_df.empty else pd.DataFrame()
    _has_enr = not _enr_row.empty
    _enr = _enr_row.iloc[0] if _has_enr else None

    with st.container(border=True):
        # Poster
        if _has_enr and "poster_url" in _enr_row.columns:
            _poster = _enr.get("poster_url")
            if _poster and str(_poster) != "nan":
                st.image(_poster, width=180)

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
            st.markdown(_meta_cell("Type", sel["type"]), unsafe_allow_html=True)
        with m2:
            yr = int(sel["release_year"]) if sel["release_year"] == sel["release_year"] else "N/A"
            st.markdown(_meta_cell("Year", str(yr)), unsafe_allow_html=True)
        with m3:
            imdb_v = f"{sel['imdb_score']:.1f}" if sel["imdb_score"] == sel["imdb_score"] else "N/A"
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
            st.markdown(_meta_cell("Runtime", f"{int(rt)} min" if rt and rt == rt else "N/A"), unsafe_allow_html=True)
        votes_v = sel.get("imdb_votes")
        with r3:
            st.markdown(_meta_cell("Votes", format_votes(votes_v) if votes_v and votes_v == votes_v else "N/A"), unsafe_allow_html=True)
        qs = sel["quality_score"] if "quality_score" in sel.index else compute_quality_score(sel.to_frame().T).iloc[0]
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

        # Box office
        if _has_enr and "box_office_usd" in _enr_row.columns:
            _bo = _enr.get("box_office_usd")
            if _bo and str(_bo) != "nan" and _bo > 0:
                _bo_str = f"${_bo/1e9:.1f}B" if _bo >= 1e9 else f"${_bo/1e6:.0f}M" if _bo >= 1e6 else f"${_bo:,.0f}"
                st.markdown(
                    f'<div style="color:{CARD_TEXT_MUTED};font-size:0.78em;margin-top:2px;">'
                    f'Box Office: <strong style="color:{CARD_TEXT};">{_bo_str}</strong></div>',
                    unsafe_allow_html=True,
                )

        # Genre pills
        if isinstance(sel["genres"], list) and sel["genres"]:
            pills = " ".join(
                f'<span style="background:{CARD_BORDER};color:{CARD_TEXT};padding:3px 10px;'
                f'border-radius:12px;font-size:0.82em;margin-right:3px;">{g.title()}</span>'
                for g in sel["genres"]
            )
            st.markdown(f'<div style="margin:8px 0;">{pills}</div>', unsafe_allow_html=True)

        # Description
        st.divider()
        desc = sel.get("description", "")
        if desc and desc == desc:
            st.markdown(
                f'<div style="font-size:0.9em;color:{CARD_TEXT};line-height:1.6;">{desc}</div>',
                unsafe_allow_html=True,
            )

        # Credits expander
        if credits_df is not None:
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

        st.markdown(
            f'<div style="margin-top:8px;">'
            f'<a href="/01_Explore_Catalog" target="_self" style="display:inline-block;'
            f'background:{CARD_BG};border:1px solid {CARD_BORDER};color:{CARD_TEXT_MUTED};'
            f'padding:6px 14px;border-radius:6px;font-size:0.82em;text-decoration:none;">'
            f'Open in Explore Catalog →</a></div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: PLATFORM IDENTITY PROFILE
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.markdown(
    section_header_html(
        "Platform Identity Profile",
        "Six-dimension radar fingerprint — what makes each platform strategically unique",
    ),
    unsafe_allow_html=True,
)

ctrl_left, ctrl_mid, ctrl_right = st.columns([2, 2, 1])
platform_options = {PLATFORMS[k]["name"]: k for k in ["merged"] + ALL_PLATFORMS}

with ctrl_left:
    primary_name = st.selectbox(
        "Select platform",
        options=list(platform_options.keys()),
        index=list(platform_options.values()).index(st.session_state.get("dna_platform", "merged")),
        key="dna_primary_select",
    )
    primary_key = platform_options[primary_name]
    st.session_state["dna_platform"] = primary_key

with ctrl_mid:
    compare_names = ["None"] + [n for n, k in platform_options.items() if k != primary_key]
    current_compare = st.session_state.get("dna_compare_platform")
    current_compare_name = "None"
    if current_compare:
        for n, k in platform_options.items():
            if k == current_compare and k != primary_key:
                current_compare_name = n
                break
    compare_name = st.selectbox(
        "Compare with (optional)",
        options=compare_names,
        index=compare_names.index(current_compare_name) if current_compare_name in compare_names else 0,
        key="dna_compare_select",
    )
    compare_key = platform_options.get(compare_name) if compare_name != "None" else None
    st.session_state["dna_compare_platform"] = compare_key

with ctrl_right:
    normalized = st.toggle(
        "Normalized",
        value=st.session_state["dna_normalized"],
        help="Show percentages instead of raw counts",
    )
    st.session_state["dna_normalized"] = normalized

platforms_to_profile = [primary_key] + ([compare_key] if compare_key else [])
profiles = compute_platform_comparison_data(df, platforms_to_profile, enriched_df=enriched_df)
enriched_stats = compute_enriched_platform_stats(df, enriched_df)


def _render_radar(platform_keys: list[str], profile_data: dict):
    fig = go.Figure()
    for key in platform_keys:
        radar = profile_data[key]["radar"]
        dims = list(radar.keys())
        vals = list(radar.values())
        dims_c = dims + [dims[0]]
        vals_c = vals + [vals[0]]
        p_color = PLATFORMS.get(key, {}).get("color", CARD_ACCENT)
        fig.add_trace(go.Scatterpolar(
            r=vals_c, theta=dims_c, fill="toself",
            fillcolor=_hex_to_rgba(p_color, 0.12),
            line=dict(color=p_color, width=2.5),
            name=profile_data[key]["display_name"],
            hovertemplate="%{theta}: %{r:.0f}/100<extra>%{fullData.name}</extra>",
        ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=True,
                            tickfont=dict(size=9, color=CARD_TEXT_MUTED),
                            gridcolor="rgba(255,255,255,0.08)"),
            angularaxis=dict(tickfont=dict(size=12, color=CARD_TEXT),
                             gridcolor="rgba(255,255,255,0.08)"),
        ),
        template=PLOTLY_TEMPLATE, height=380,
        margin=dict(t=40, b=40, l=60, r=60),
        showlegend=len(platform_keys) > 1,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5, font=dict(size=12)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_metrics_row(platform_key: str, data: dict):
    quality = data["quality_profile"]
    p_color = PLATFORMS.get(platform_key, {}).get("color", CARD_ACCENT)
    st.markdown(
        f'<div style="background:{CARD_BG};border-left:4px solid {p_color};border-radius:8px;'
        f'padding:12px 16px;margin-bottom:8px;">'
        f'<span style="font-size:1.15em;font-weight:700;color:{CARD_TEXT};">{data["display_name"]}</span>'
        f'<span style="color:{CARD_TEXT_MUTED};font-size:0.88em;margin-left:12px;">'
        f'{data["title_count"]:,} titles</span></div>',
        unsafe_allow_html=True,
    )
    q1, q2, q3 = st.columns(3)
    excellent_pct = quality["tier_pcts"].get("Excellent", 0)
    good_pct = quality["tier_pcts"].get("Good", 0)
    with q1:
        avg_label = f"{quality['avg_imdb']:.2f}" if quality["avg_imdb"] else "N/A"
        st.markdown(styled_metric_card_html("Avg IMDb", avg_label, accent_color=p_color,
                    help_text="Average IMDb rating across all rated titles"), unsafe_allow_html=True)
    with q2:
        if normalized:
            st.markdown(styled_metric_card_html("Premium Content", f"{excellent_pct + good_pct:.1f}%",
                        accent_color=p_color, help_text="Excellent (8+) + Good (7-8) combined"),
                        unsafe_allow_html=True)
        else:
            premium_count = quality["tier_counts"].get("Excellent", 0) + quality["tier_counts"].get("Good", 0)
            st.markdown(styled_metric_card_html("Premium Titles", f"{premium_count:,}", accent_color=p_color,
                        help_text="Excellent (8+) + Good (7-8) combined"), unsafe_allow_html=True)
    with q3:
        st.markdown(styled_metric_card_html("Rated Titles",
                    f"{quality['total_rated']:,} / {quality['total_titles']:,}",
                    accent_color=p_color, help_text="Titles with an IMDb rating vs total catalog size"),
                    unsafe_allow_html=True)

    # Enrichment metrics row (awards, franchises, era, international)
    _est = enriched_stats.get(platform_key, {})
    has_any_enr = (
        _est.get("award_wins", 0) > 0 or
        _est.get("franchise_count", 0) > 0
    )
    if has_any_enr:
        e1, e2, e3, e4 = st.columns(4)
        with e1:
            aw = _est.get("award_wins", 0)
            an = _est.get("award_noms", 0)
            aw_label = f"{aw:,}" if aw > 0 else "N/A"
            aw_sub = f"{an:,} nominations" if an > 0 else ""
            st.markdown(styled_metric_card_html("Award Wins", aw_label, accent_color=p_color,
                        subtitle=aw_sub, help_text="Total award wins from Wikidata enrichment"),
                        unsafe_allow_html=True)
        with e2:
            fc = _est.get("franchise_count", 0)
            st.markdown(styled_metric_card_html("Franchises", f"{fc:,}" if fc > 0 else "N/A",
                        accent_color=p_color, help_text="Distinct TMDB collection/franchise names"),
                        unsafe_allow_html=True)
        with e3:
            p2015 = _est.get("post_2015_pct", 0)
            st.markdown(styled_metric_card_html("Post-2015", f"{p2015:.0f}%", accent_color=p_color,
                        help_text="Percentage of catalog released in 2015 or later"),
                        unsafe_allow_html=True)
        with e4:
            intl = _est.get("intl_pct", 0)
            st.markdown(styled_metric_card_html("International", f"{intl:.0f}%", accent_color=p_color,
                        help_text="Percentage of titles produced outside the US"),
                        unsafe_allow_html=True)


def _render_traits(traits: list[dict], p_color: str = CARD_ACCENT):
    if not traits:
        return
    st.markdown(
        f'<div style="color:{CARD_TEXT_MUTED};font-size:0.72em;text-transform:uppercase;'
        f'letter-spacing:0.04em;margin:12px 0 8px;">Defining Traits</div>',
        unsafe_allow_html=True,
    )
    for trait in traits:
        direction = trait.get("direction", "neutral")
        icon = trait.get("icon", "")
        border_color = (
            "#2ecc71" if direction == "high" else "#e74c3c" if direction == "low" else CARD_TEXT_MUTED
        )
        st.markdown(
            f'<div style="background:{CARD_BG};border-left:4px solid {border_color};'
            f'border-radius:6px;padding:10px 14px;margin-bottom:6px;display:flex;gap:10px;">'
            f'<span style="font-size:1.2em;line-height:1.4;">{icon}</span>'
            f'<div><span style="color:{p_color};font-weight:700;">{trait["label"]}</span>'
            f'<span style="color:{CARD_TEXT_MUTED};"> — </span>'
            f'<span style="color:{CARD_TEXT};font-size:0.9em;">{trait["detail"]}</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )


_RADAR_DIM_TABLE = (
    "| Dimension | What it measures | Scale |\n"
    "|-----------|-----------------|-------|\n"
    "| **Freshness** | How recent the catalog skews (median release year) | 0 = oldest, 100 = newest |\n"
    "| **Quality** | Average IMDb score across rated titles | 0 = lowest avg, 100 = highest avg |\n"
    "| **Breadth** | Total catalog size relative to other platforms | 0 = smallest, 100 = largest |\n"
    "| **Global Reach** | % of content produced outside the US | 0 = most domestic, 100 = most international |\n"
    "| **Genre Diversity** | Number of unique genres represented | 0 = least diverse, 100 = most diverse |\n"
    "| **Series Focus** | % of catalog that is TV shows (vs movies) | 0 = movie-focused, 100 = series-focused |\n\n"
    "All dimensions are normalized 0–100 relative to the range across all 6 platforms."
)

if primary_key not in profiles:
    st.info("No data available for this platform with the current filters.")
elif compare_key and compare_key in profiles:
    _render_radar([primary_key, compare_key], profiles)
    with st.expander("What do the radar dimensions mean?"):
        st.markdown(_RADAR_DIM_TABLE)
    l_col, r_col = st.columns(2)
    with l_col:
        _render_metrics_row(primary_key, profiles[primary_key])
        _render_traits(profiles[primary_key]["traits"], PLATFORMS.get(primary_key, {}).get("color", CARD_ACCENT))
    with r_col:
        _render_metrics_row(compare_key, profiles[compare_key])
        _render_traits(profiles[compare_key]["traits"], PLATFORMS.get(compare_key, {}).get("color", CARD_ACCENT))
else:
    _render_radar([primary_key], profiles)
    with st.expander("What do the radar dimensions mean?"):
        st.markdown(_RADAR_DIM_TABLE)
    _render_metrics_row(primary_key, profiles[primary_key])
    _render_traits(profiles[primary_key]["traits"], PLATFORMS.get(primary_key, {}).get("color", CARD_ACCENT))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: CONTENT LANDSCAPE
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.markdown(
    section_header_html(
        "Content Landscape",
        "A 2D map of all streaming titles — see where each platform's content clusters by genre and theme similarity",
    ),
    unsafe_allow_html=True,
)

try:
    umap_coords = load_umap_coords()
except FileNotFoundError:
    umap_coords = None
    st.warning("UMAP coordinates not found. Run `scripts/04_compute_umap.py` first.")

if umap_coords is not None:
    with st.spinner("Mapping the content landscape…"):
        landscape = compute_landscape_clusters(umap_coords, df)
        landscape = landscape.dropna(subset=["platform"])

    if landscape.empty:
        st.info("No titles match current filters for landscape visualization.")
    else:
        cluster_summaries = compute_cluster_summaries(landscape)

        if not cluster_summaries:
            st.info("Not enough data to identify content neighborhoods.")
        else:
            # ── Landscape insights ────────────────────────────────────────────
            insights = compute_landscape_insights_v2(landscape, cluster_summaries)
            if insights:
                _INSIGHT_ICONS = {"crown": "👑", "overlap": "🔀", "focus": "🎯", "diverge": "↔️"}
                items_html = "".join(
                    f'<div style="margin-bottom:8px;padding:8px 12px;'
                    f'background:rgba(255,255,255,0.03);border-radius:6px;'
                    f'border-left:3px solid {CARD_ACCENT};">'
                    f'<span style="font-size:1.1em;margin-right:8px;">'
                    f'{_INSIGHT_ICONS.get(ins.get("icon",""), "•")}</span>'
                    f'<span style="color:{CARD_TEXT};font-size:0.9em;line-height:1.6;">'
                    f'{ins["text"]}</span></div>'
                    for ins in insights
                )
                st.markdown(
                    f'<div style="background:{CARD_BG};border:1px solid {CARD_BORDER};'
                    f'border-radius:10px;padding:14px 18px;margin-bottom:14px;">'
                    f'<div style="color:{CARD_ACCENT};font-weight:700;font-size:1em;margin-bottom:8px;">'
                    f'📊 Landscape Insights</div>'
                    f'{items_html}</div>',
                    unsafe_allow_html=True,
                )

            # ── Controls ─────────────────────────────────────────────────────
            available_platforms = sorted(landscape["platform"].unique().tolist())
            neighborhood_names = {cid: s["name"] for cid, s in cluster_summaries.items()}

            st.markdown(
                '<p style="color:#888;font-size:0.83em;margin-bottom:4px;">'
                'Each dot is a title, positioned by content similarity (UMAP). '
                'Labeled zones group titles by theme — use the controls below to explore.</p>',
                unsafe_allow_html=True,
            )

            ctrl_l, ctrl_m = st.columns([3, 2])
            with ctrl_l:
                highlight_platforms = st.multiselect(
                    "Highlight platforms on map",
                    options=[PLATFORMS.get(p, {}).get("name", p) for p in available_platforms],
                    default=[],
                    help="Selected platforms show full-opacity colored markers; others dim to 10%",
                )
            highlight_keys = [
                k for k in available_platforms
                if PLATFORMS.get(k, {}).get("name", k) in highlight_platforms
            ]
            with ctrl_m:
                title_search = st.text_input(
                    "Find a title on the map",
                    placeholder="e.g. Breaking Bad",
                    help="Highlights matching titles on the map with a star marker",
                )

            # ── UMAP figure ──────────────────────────────────────────────────
            _ARCH_COLORS = {
                0: "#FF6B6B", 1: "#4ECDC4", 2: "#9B59B6", 3: "#F39C12",
                4: "#3498DB", 5: "#E74C3C", 6: "#2ECC71", 7: "#F1C40F",
                8: "#E91E63", 9: "#95A5A6",
            }

            # Determine opacity based on highlight selection
            base_opacity = 0.12 if highlight_keys else 0.40
            fig_landscape = go.Figure()

            # Base archetype scatter (max 200/archetype to reduce density)
            _MAX_PER_ARCH = 200
            for cid, summary in cluster_summaries.items():
                arch_data = landscape[landscape["cluster"] == cid]
                if len(arch_data) > _MAX_PER_ARCH:
                    arch_data = arch_data.sample(n=_MAX_PER_ARCH, random_state=42)
                a_color = _ARCH_COLORS.get(cid, "#888")

                # Build hover customdata: imdb, year, platform_name
                _plat_names = arch_data["platform"].map(
                    lambda p: PLATFORMS.get(p, {}).get("name", p)
                )
                _custom = arch_data[["imdb_score", "release_year"]].fillna(0).copy()
                _custom["plat"] = _plat_names.values

                fig_landscape.add_trace(go.Scatter(
                    x=arch_data["umap_x"], y=arch_data["umap_y"],
                    mode="markers",
                    name=summary["name"],
                    marker=dict(size=4, color=a_color, opacity=base_opacity),
                    text=arch_data["title"],
                    customdata=_custom[["imdb_score", "release_year", "plat"]].values,
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "%{customdata[2]}<br>"
                        f"<em>{summary['name']}</em><br>"
                        "IMDb: %{customdata[0]:.1f} · %{customdata[1]:.0f}"
                        "<extra></extra>"
                    ),
                ))

            # Platform overlay (full opacity for selected, very dim for unselected)
            if highlight_keys:
                # Show unselected at very low opacity
                for plat_key in available_platforms:
                    if plat_key in highlight_keys:
                        continue
                    plat_data = landscape[landscape["platform"] == plat_key]
                    if len(plat_data) > 100:
                        plat_data = plat_data.sample(n=100, random_state=42)
                    p_color = PLATFORMS.get(plat_key, {}).get("color", "#888")
                    fig_landscape.add_trace(go.Scatter(
                        x=plat_data["umap_x"], y=plat_data["umap_y"],
                        mode="markers", showlegend=False,
                        marker=dict(size=4, color=p_color, opacity=0.08),
                        hoverinfo="skip",
                    ))
                # Show selected at full opacity
                for plat_key in highlight_keys:
                    plat_data = landscape[landscape["platform"] == plat_key]
                    if len(plat_data) > DNA_UMAP_SAMPLE_SIZE:
                        plat_data = plat_data.sample(n=DNA_UMAP_SAMPLE_SIZE, random_state=42)
                    p_color = PLATFORMS.get(plat_key, {}).get("color", "#888")
                    p_name = PLATFORMS.get(plat_key, {}).get("name", plat_key)
                    _pnames = [p_name] * len(plat_data)
                    _pdata = plat_data[["imdb_score", "release_year"]].fillna(0).copy()
                    _pdata["pn"] = _pnames
                    fig_landscape.add_trace(go.Scatter(
                        x=plat_data["umap_x"], y=plat_data["umap_y"],
                        mode="markers",
                        name=f"● {p_name}",
                        marker=dict(size=7, color=p_color, opacity=0.85,
                                    line=dict(width=0.5, color="#fff")),
                        text=plat_data["title"],
                        customdata=_pdata[["imdb_score", "release_year", "pn"]].values,
                        hovertemplate=(
                            "<b>%{text}</b><br>%{customdata[2]}<br>"
                            "IMDb: %{customdata[0]:.1f} · %{customdata[1]:.0f}<extra></extra>"
                        ),
                    ))

            # Selected neighborhood highlight
            sel_neighborhood = st.session_state.get("dna_sel_neighborhood", "None")
            sel_cid = None
            if sel_neighborhood != "None":
                for cid, hname in neighborhood_names.items():
                    if hname == sel_neighborhood:
                        sel_cid = cid
                        break
            if sel_cid is not None:
                hood_data = landscape[landscape["cluster"] == sel_cid]
                fig_landscape.add_trace(go.Scatter(
                    x=hood_data["umap_x"], y=hood_data["umap_y"],
                    mode="markers",
                    name=f"📍 {sel_neighborhood}",
                    marker=dict(size=7, color="rgba(255,215,0,0.5)",
                                line=dict(width=1.5, color="#FFD700")),
                    text=hood_data.get("title", pd.Series()),
                    hovertemplate="<b>%{text}</b><br>" + sel_neighborhood + "<extra></extra>",
                ))

            # Title search highlight
            search_matches = pd.DataFrame()
            if title_search and len(title_search) >= 2:
                search_matches = landscape[
                    landscape["title"].str.contains(title_search, case=False, na=False)
                ]
                if not search_matches.empty:
                    fig_landscape.add_trace(go.Scatter(
                        x=search_matches["umap_x"], y=search_matches["umap_y"],
                        mode="markers+text",
                        name="🔍 Search results",
                        marker=dict(size=14, symbol="star", color="#FFD700",
                                    line=dict(width=1.5, color="#fff")),
                        text=search_matches["title"],
                        textposition="top center",
                        textfont=dict(size=10, color="#FFD700"),
                        hovertemplate=(
                            "<b>%{text}</b><br>"
                            "IMDb: %{customdata[0]:.1f} · %{customdata[1]:.0f}<extra></extra>"
                        ),
                        customdata=search_matches[["imdb_score", "release_year"]].fillna(0).values,
                    ))

            # Cluster centroid labels — turns a dot splatter into a readable map
            for _cid, _summary in cluster_summaries.items():
                _arch_data = landscape[landscape["cluster"] == _cid]
                if len(_arch_data) >= 5:
                    _cx = float(_arch_data["umap_x"].mean())
                    _cy = float(_arch_data["umap_y"].mean())
                    _a_color = _ARCH_COLORS.get(_cid, "#888")
                    fig_landscape.add_annotation(
                        x=_cx, y=_cy,
                        text=f"<b>{_summary['name']}</b>",
                        showarrow=False,
                        font=dict(size=9, color="#ffffff"),
                        bgcolor=_hex_to_rgba(_a_color, 0.82),
                        bordercolor=_a_color,
                        borderwidth=1,
                        borderpad=4,
                        opacity=0.95,
                    )

            fig_landscape.update_layout(
                xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, title=""),
                yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, title=""),
                template=PLOTLY_TEMPLATE, height=CHART_HEIGHT + 200,
                margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(title="Theme Zones", orientation="h",
                            yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                            font=dict(size=10)),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_landscape, use_container_width=True)

            # Title search results
            if title_search and len(title_search) >= 2:
                if search_matches.empty:
                    st.caption(f'No titles matching "{title_search}" found on the map.')
                else:
                    for _, sr in search_matches.head(5).iterrows():
                        cid_r = sr.get("cluster")
                        hood_r = cluster_summaries.get(cid_r, {}).get("name", "Unknown")
                        p_n = PLATFORMS.get(sr.get("platform", ""), {}).get("name", "")
                        imdb_r = f"{sr['imdb_score']:.1f}" if pd.notna(sr.get("imdb_score")) else "N/A"
                        st.caption(
                            f"**{sr['title']}** — {p_n} · IMDb {imdb_r} · {hood_r}"
                        )

            # ── Neighborhood selector ─────────────────────────────────────────
            st.selectbox(
                "Explore a neighborhood",
                options=["None"] + list(neighborhood_names.values()),
                index=(
                    (["None"] + list(neighborhood_names.values())).index(sel_neighborhood)
                    if sel_neighborhood in (["None"] + list(neighborhood_names.values())) else 0
                ),
                key="dna_sel_neighborhood",
                help="Select a neighborhood to explore its top titles and platform breakdown",
            )

            # ── Neighborhood explorer ─────────────────────────────────────────
            if sel_cid is not None:
                sel_summary = cluster_summaries[sel_cid]
                sel_data = landscape[landscape["cluster"] == sel_cid]

                st.divider()
                p_color_hood = CARD_ACCENT

                # Header
                st.markdown(
                    f'<div style="background:{CARD_BG};border:1px solid {CARD_BORDER};'
                    f'border-radius:10px;padding:16px 20px;margin-bottom:12px;">'
                    f'<div style="font-size:1.25em;font-weight:700;color:{CARD_ACCENT};'
                    f'margin-bottom:6px;">📍 {sel_summary["name"]}</div>'
                    f'<div style="color:{CARD_TEXT_MUTED};font-size:0.88em;">'
                    f'{sel_summary["size"]:,} titles centered on '
                    f'{", ".join(sel_summary["top_genres"][:3])}</div></div>',
                    unsafe_allow_html=True,
                )

                # Genre pills
                gp_html = "".join(
                    f'<span style="display:inline-block;background:rgba(255,215,0,0.10);'
                    f'color:{CARD_ACCENT};border-radius:10px;padding:3px 10px;'
                    f'font-size:0.82em;margin-right:5px;margin-bottom:4px;">{g}</span>'
                    for g in sel_summary["top_genres"]
                )
                st.markdown(gp_html, unsafe_allow_html=True)

                # Enrichment info for neighborhood
                _cluster_ids = set(sel_data["id"])
                if not enriched_df.empty:
                    _cluster_enr = enriched_df[enriched_df["id"].isin(_cluster_ids)]
                    _enr_lines = []

                    # Franchise: require ≥5 titles AND not adult-content names
                    _BAD_FRANCHISE_KW = {
                        "adult", "hooker", "erotic", "xxx", "porn", "sex tape", "playboy",
                        "nude", "naked", "hustler",
                    }
                    if "collection_name" in _cluster_enr.columns:
                        _fc = _cluster_enr["collection_name"].dropna().value_counts()
                        _bad_mask = [
                            any(_kw in str(_n).lower() for _kw in _BAD_FRANCHISE_KW)
                            for _n in _fc.index
                        ]
                        import pandas as _pd
                        _fc_valid = _fc[
                            (_fc >= 5) & (~_pd.array(_bad_mask, dtype=bool))
                        ]
                        if len(_fc_valid) > 0:
                            _enr_lines.append(
                                f'<div style="color:{CARD_TEXT_MUTED};font-size:0.82em;margin-bottom:3px;">'
                                f'Top franchise: <span style="color:{CARD_ACCENT};font-weight:600;">'
                                f'{_fc_valid.index[0]}</span>'
                                f' <span style="color:{CARD_TEXT_MUTED};">({_fc_valid.iloc[0]} titles in this neighborhood)</span>'
                                f'</div>'
                            )

                    # Awards: clearly labeled as neighborhood total
                    if "award_wins" in _cluster_enr.columns:
                        _aw_cov = _cluster_enr["award_wins"].notna().mean()
                        if _aw_cov >= 0.10:
                            _aw_tot = int(_cluster_enr["award_wins"].fillna(0).sum())
                            if _aw_tot > 5:
                                _enr_lines.append(
                                    f'<div style="color:{CARD_TEXT_MUTED};font-size:0.82em;margin-bottom:3px;">'
                                    f'🏆 <span style="color:#FFD700;font-weight:600;">'
                                    f'{_aw_tot:,} total award wins</span>'
                                    f' <span style="color:{CARD_TEXT_MUTED};">across all titles in this neighborhood</span>'
                                    f'</div>'
                                )

                    if _enr_lines:
                        st.markdown(
                            f'<div style="margin:4px 0 8px;">{"".join(_enr_lines)}</div>',
                            unsafe_allow_html=True,
                        )

                # Leaders line
                pl = sel_summary.get("platform_leader")
                ql = sel_summary.get("quality_leader")
                lp = []
                if pl:
                    pc = PLATFORMS.get(pl, {}).get("color", CARD_TEXT_MUTED)
                    pn = PLATFORMS.get(pl, {}).get("name", pl)
                    lp.append(f'Volume: <span style="color:{pc};font-weight:600;">{pn}</span>')
                if ql and ql != pl:
                    qc = PLATFORMS.get(ql, {}).get("color", CARD_TEXT_MUTED)
                    qn = PLATFORMS.get(ql, {}).get("name", ql)
                    lp.append(f'Quality: <span style="color:{qc};font-weight:600;">{qn}</span>')
                if lp:
                    st.markdown(
                        f'<div style="color:{CARD_TEXT_MUTED};font-size:0.85em;margin:4px 0 8px;">'
                        + " · ".join(lp) + "</div>",
                        unsafe_allow_html=True,
                    )

                # Quality metrics row
                m1, m2, m3, m4 = st.columns(4)
                rated_c = sel_data.dropna(subset=["imdb_score"])
                prem_pct = (rated_c["imdb_score"] >= 7.0).mean() * 100 if len(rated_c) > 0 else 0
                med_yr = int(sel_data["release_year"].median()) if sel_data["release_year"].notna().any() else "N/A"
                with m1:
                    st.markdown(styled_metric_card_html("Titles", f"{sel_summary['size']:,}",
                                help_text="Total titles in this neighborhood"), unsafe_allow_html=True)
                with m2:
                    avg_label = f"{sel_summary['avg_imdb']:.1f}" if sel_summary["avg_imdb"] else "N/A"
                    st.markdown(styled_metric_card_html("Avg IMDb", avg_label,
                                help_text="Average IMDb rating in this neighborhood"), unsafe_allow_html=True)
                with m3:
                    st.markdown(styled_metric_card_html("Premium (7.0+)", f"{prem_pct:.0f}%",
                                help_text="% of rated titles scoring 7.0+"), unsafe_allow_html=True)
                with m4:
                    st.markdown(styled_metric_card_html("Median Year", str(med_yr),
                                help_text="Median release year in this neighborhood"), unsafe_allow_html=True)

                # Platform breakdown bar
                if sel_summary["platform_mix"]:
                    bar_data = [
                        {"Platform": PLATFORMS.get(pk, {}).get("name", pk), "Count": cnt,
                         "Color": PLATFORMS.get(pk, {}).get("color", "#888")}
                        for pk in ALL_PLATFORMS
                        if (cnt := sel_summary["platform_mix"].get(pk, 0)) > 0
                    ]
                    if bar_data:
                        bar_df = pd.DataFrame(bar_data)
                        fig_bar = px.bar(bar_df, x="Count", y="Platform", orientation="h",
                                         color="Platform",
                                         color_discrete_map={r["Platform"]: r["Color"] for r in bar_data},
                                         template=PLOTLY_TEMPLATE, height=200)
                        fig_bar.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10),
                                              yaxis=dict(categoryorder="total ascending"))
                        st.plotly_chart(fig_bar, use_container_width=True)

                # ── Top 5 titles in neighborhood ──────────────────────────────
                st.markdown(
                    f'<div style="color:{CARD_TEXT_MUTED};font-size:0.72em;text-transform:uppercase;'
                    f'letter-spacing:0.04em;margin:12px 0 8px;">Top 5 Titles in this Neighborhood</div>',
                    unsafe_allow_html=True,
                )
                top_titles = compute_neighborhood_top_titles(landscape, enriched_df, sel_cid, top_n=5)

                if not top_titles:
                    st.caption("No rated titles in this neighborhood.")
                else:
                    t_cols = st.columns(5)
                    for ti, ttitle in enumerate(top_titles[:5]):
                        with t_cols[ti]:
                            t_color = PLATFORMS.get(ttitle["platform"], {}).get("color", CARD_BORDER)
                            t_pname = PLATFORMS.get(ttitle["platform"], {}).get("name", "")
                            t_imdb = f"{ttitle['imdb_score']:.1f}" if ttitle.get("imdb_score") else "N/A"
                            t_yr = str(int(ttitle["release_year"])) if ttitle.get("release_year") else ""

                            # Award badge (kept outside card body so heights are predictable)
                            aw_badge = ""
                            if ttitle.get("award_wins") and ttitle["award_wins"] > 0:
                                aw_badge = (
                                    f'<span style="background:rgba(46,204,113,0.15);color:#2ecc71;'
                                    f'border-radius:4px;padding:1px 5px;font-size:0.68em;'
                                    f'font-weight:600;margin-right:3px;">🏆 {int(ttitle["award_wins"])}W</span>'
                                )

                            # Genre pills (max 2)
                            genres_html = ""
                            if isinstance(ttitle.get("genres"), list):
                                for g in ttitle["genres"][:2]:
                                    genres_html += (
                                        f'<span style="display:inline-block;'
                                        f'background:rgba(255,255,255,0.06);'
                                        f'color:{CARD_TEXT_MUTED};border-radius:6px;'
                                        f'padding:1px 5px;font-size:0.65em;margin-right:2px;">'
                                        f'{g.title()}</span>'
                                    )

                            # Full card: poster + body in one container, fixed total height
                            _poster_src = ttitle.get("poster_url")
                            _poster_block = (
                                f'<img src="{_poster_src}" style="width:100%;height:150px;'
                                f'object-fit:cover;border-radius:6px 6px 0 0;display:block;" '
                                f'onerror="this.style.display=\'none\'" />'
                                if _poster_src else
                                f'<div style="width:100%;height:150px;border-radius:6px 6px 0 0;'
                                f'background:{t_color}22;border-bottom:2px solid {t_color};'
                                f'display:flex;align-items:center;justify-content:center;'
                                f'font-size:2em;">🎬</div>'
                            )
                            st.markdown(
                                f'<div style="background:{CARD_BG};border:1px solid {CARD_BORDER};'
                                f'border-top:3px solid {t_color};border-radius:8px;'
                                f'overflow:hidden;min-height:320px;display:flex;flex-direction:column;">'
                                f'{_poster_block}'
                                f'<div style="padding:8px 10px;flex:1;display:flex;flex-direction:column;">'
                                f'<div style="color:{CARD_TEXT};font-weight:600;font-size:0.83em;'
                                f'line-height:1.3;margin-bottom:6px;flex:1;'
                                f'overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;'
                                f'-webkit-box-orient:vertical;">{ttitle["title"]}</div>'
                                f'<div style="margin-top:auto;">'
                                f'<div style="margin-bottom:3px;">{aw_badge}</div>'
                                f'<div style="color:{CARD_TEXT_MUTED};font-size:0.72em;">'
                                f'<span style="color:{t_color};font-weight:600;">{t_pname}</span>'
                                f' · {t_yr} · ⭐ {t_imdb}</div>'
                                f'<div style="margin-top:3px;">{genres_html}</div>'
                                f'</div></div></div>',
                                unsafe_allow_html=True,
                            )
                            _is_active = st.session_state.get("dna_neighborhood_title_id") == ttitle.get("id")
                            if ttitle.get("id") and st.button(
                                "✕ Close" if _is_active else "Details",
                                key=f"hood_detail_{ttitle['id']}_{ti}",
                                use_container_width=True,
                                type="secondary",
                            ):
                                if _is_active:
                                    st.session_state["dna_neighborhood_title_id"] = None
                                else:
                                    st.session_state["dna_neighborhood_title_id"] = ttitle["id"]
                                st.rerun()

                # ── Title detail panel for selected neighborhood title ──────────
                _detail_id = st.session_state.get("dna_neighborhood_title_id")
                if _detail_id is not None:
                    # Check it's still in this neighborhood's titles
                    if _detail_id in {t["id"] for t in top_titles}:
                        st.markdown(
                            f'<div style="color:{CARD_TEXT_MUTED};font-size:0.72em;'
                            f'text-transform:uppercase;letter-spacing:0.04em;margin:12px 0 6px;">'
                            f'Title Details</div>',
                            unsafe_allow_html=True,
                        )
                        try:
                            _credits_df = load_all_platforms_credits()
                        except Exception:
                            _credits_df = None
                        _render_inline_title_detail(_detail_id, df, enriched_df, _credits_df)
                    else:
                        st.session_state["dna_neighborhood_title_id"] = None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: WHAT PLATFORM ARE YOU?
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.markdown(
    section_header_html(
        "What Platform Are You?",
        "Pick genres, set your vibe, swipe through titles — discover your streaming DNA",
    ),
    unsafe_allow_html=True,
)

# Session state for quiz
st.session_state.setdefault("dna_quiz_phase", "A")
st.session_state.setdefault("dna_quiz_prefs", {})
st.session_state.setdefault("dna_quiz_titles", [])
st.session_state.setdefault("dna_quiz_current", 0)
st.session_state.setdefault("dna_quiz_liked", [])
st.session_state.setdefault("dna_quiz_results", None)
st.session_state.setdefault("dna_quiz_slider_prefs", {})
st.session_state.setdefault("dna_quiz_genres", [])
st.session_state.setdefault("dna_quiz_rec_detail_id", None)

_quiz_phase = st.session_state["dna_quiz_phase"]


def _reset_quiz():
    for k in ("dna_quiz_phase", "dna_quiz_prefs", "dna_quiz_titles", "dna_quiz_current",
              "dna_quiz_liked", "dna_quiz_results", "dna_quiz_slider_prefs",
              "dna_quiz_rec_detail_id", "_dna_type_pref"):
        st.session_state.pop(k, None)
    st.session_state["dna_quiz_genres"] = []
    st.session_state.setdefault("dna_quiz_phase", "A")


def _render_step_chips(phase: str):
    """Render progressive step chips — only show steps that have been reached."""
    _chip_done = (
        f'padding:5px 14px;background:rgba(46,204,113,0.15);color:#2ecc71;'
        f'border:1px solid #2ecc71;border-radius:20px;font-size:0.82em;'
    )
    _chip_active = (
        f'padding:5px 14px;background:rgba(255,215,0,0.15);color:{CARD_ACCENT};'
        f'border:1px solid {CARD_ACCENT};border-radius:20px;font-size:0.82em;font-weight:700;'
    )
    chips = []
    if phase == "A":
        chips.append(f'<div style="{_chip_active}">Step 1: Pick Your Genres</div>')
    elif phase == "A2":
        chips.append(f'<div style="{_chip_done}">✓ Genres Selected</div>')
        chips.append(f'<div style="{_chip_active}">Step 2: Set Your Vibe</div>')
    elif phase == "B":
        chips.append(f'<div style="{_chip_done}">✓ Genres</div>')
        chips.append(f'<div style="{_chip_done}">✓ Vibe</div>')
        chips.append(f'<div style="{_chip_active}">Step 3: Swipe Titles</div>')
    elif phase == "C":
        chips.append(f'<div style="{_chip_done}">✓ Genres</div>')
        chips.append(f'<div style="{_chip_done}">✓ Vibe</div>')
        chips.append(f'<div style="{_chip_done}">✓ Swiped</div>')
        chips.append(f'<div style="{_chip_active}">Step 4: Your Results</div>')
    st.markdown(
        f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:18px;">'
        + "".join(chips) + "</div>",
        unsafe_allow_html=True,
    )


# Build genre options
_all_genre_counts: dict[str, int] = {}
for _g_list in raw_df["genres"].dropna():
    if isinstance(_g_list, list):
        for _g in _g_list:
            _all_genre_counts[_g] = _all_genre_counts.get(_g, 0) + 1
_sorted_genres = sorted(_all_genre_counts.keys(), key=lambda g: -_all_genre_counts[g])

_GENRE_ICONS = {
    "drama": "🎭", "comedy": "😄", "thriller": "😰", "action": "💥",
    "romance": "💕", "documentation": "🎥", "crime": "🔍", "family": "👨‍👩‍👧",
    "animation": "✏️", "scifi": "🚀", "fantasy": "🐉", "horror": "👻",
    "music": "🎵", "history": "📜", "western": "🤠", "war": "⚔️",
    "sport": "⚽", "reality": "📺", "european": "🌍",
}
_GENRE_DISPLAY_NAMES = {"documentation": "Documentary", "scifi": "Sci-Fi"}


# ─────────────────────────────────────────────────────────────────────────────
# PHASE A: Genre Selection only
# ─────────────────────────────────────────────────────────────────────────────
if _quiz_phase == "A":
    _render_step_chips("A")

    st.markdown(
        f'<div style="color:{CARD_TEXT};font-weight:700;font-size:1.1em;margin-bottom:6px;">'
        f'Which genres do you love?</div>'
        f'<div style="color:{CARD_TEXT_MUTED};font-size:0.88em;margin-bottom:14px;">'
        f'Pick at least 2 — we\'ll use these to find titles and match your taste to a platform</div>',
        unsafe_allow_html=True,
    )

    sel_genres = st.session_state["dna_quiz_genres"]
    _top_genres_quiz = _sorted_genres[:12]
    _gcols = st.columns(4)
    for gi, gkey in enumerate(_top_genres_quiz):
        with _gcols[gi % 4]:
            icon = _GENRE_ICONS.get(gkey, "🎬")
            label = _GENRE_DISPLAY_NAMES.get(gkey, gkey.title())
            is_sel = gkey in sel_genres
            if st.button(
                f"{icon} {label}" + (" ✓" if is_sel else ""),
                key=f"gtoggle_{gkey}",
                use_container_width=True,
                type="primary" if is_sel else "secondary",
            ):
                if gkey in st.session_state["dna_quiz_genres"]:
                    st.session_state["dna_quiz_genres"].remove(gkey)
                else:
                    st.session_state["dna_quiz_genres"].append(gkey)
                st.rerun()

    sel_genres = st.session_state["dna_quiz_genres"]
    n_sel = len(sel_genres)
    _badge_color = CARD_ACCENT if n_sel >= 2 else CARD_TEXT_MUTED
    st.markdown(
        f'<div style="margin:10px 0 18px;">'
        f'<span style="background:rgba(255,215,0,0.12);color:{_badge_color};padding:4px 12px;'
        f'border-radius:12px;font-size:0.85em;font-weight:600;">{n_sel} selected</span>'
        + (f'<span style="color:{CARD_TEXT_MUTED};font-size:0.82em;margin-left:10px;">'
           f'— select at least 2 to continue</span>'
           if n_sel < 2 else
           f'<span style="color:#2ecc71;font-size:0.82em;margin-left:10px;">Good choices ✓</span>') +
        '</div>',
        unsafe_allow_html=True,
    )

    _can_next = n_sel >= 2
    if st.button(
        "Next: Set Your Vibe →",
        use_container_width=True,
        disabled=not _can_next,
        type="primary",
        help="Select at least 2 genres to continue",
    ):
        st.session_state["dna_quiz_phase"] = "A2"
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PHASE A2: Vibe Sliders + Type Preference
# ─────────────────────────────────────────────────────────────────────────────
elif _quiz_phase == "A2":
    _render_step_chips("A2")

    _saved_genres = st.session_state.get("dna_quiz_genres", [])
    _genre_display = ", ".join(
        _GENRE_DISPLAY_NAMES.get(g, g.title()) for g in _saved_genres[:4]
    )
    st.markdown(
        f'<div style="background:rgba(46,204,113,0.08);border-left:3px solid #2ecc71;'
        f'border-radius:0 6px 6px 0;padding:8px 14px;margin-bottom:16px;">'
        f'<span style="color:#2ecc71;font-size:0.82em;font-weight:600;">Genres locked in: </span>'
        f'<span style="color:{CARD_TEXT};font-size:0.85em;">{_genre_display}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div style="color:{CARD_TEXT};font-weight:700;font-size:1.1em;margin-bottom:4px;">'
        f'Now tune your viewing vibe</div>'
        f'<div style="color:{CARD_TEXT_MUTED};font-size:0.88em;margin-bottom:16px;">'
        f'These sliders help us understand your ideal streaming experience</div>',
        unsafe_allow_html=True,
    )

    sl_col1, sl_col2 = st.columns(2)
    with sl_col1:
        era_val = st.slider(
            "Content Era", 0, 100, 60, format="%d",
            help="How recent do you want the content?",
        )
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;font-size:0.72em;'
            f'color:{CARD_TEXT_MUTED};margin-top:-8px;margin-bottom:14px;">'
            f'<span>All-time classics</span><span>Latest releases</span></div>',
            unsafe_allow_html=True,
        )

        gems_val = st.slider(
            "Popularity", 0, 100, 60, format="%d",
            help="Hidden undiscovered gems or mainstream crowd-pleasers?",
        )
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;font-size:0.72em;'
            f'color:{CARD_TEXT_MUTED};margin-top:-8px;margin-bottom:14px;">'
            f'<span>Hidden gems</span><span>Mainstream hits</span></div>',
            unsafe_allow_html=True,
        )

        runtime_val = st.slider(
            "Runtime", 0, 100, 50, format="%d",
            help="Do you prefer shorter episodes/films or longer, epic content?",
        )
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;font-size:0.72em;'
            f'color:{CARD_TEXT_MUTED};margin-top:-8px;margin-bottom:14px;">'
            f'<span>Quick watches</span><span>Long, immersive</span></div>',
            unsafe_allow_html=True,
        )

    with sl_col2:
        tone_val = st.slider(
            "Tone", 0, 100, 40, format="%d",
            help="Light and uplifting, or dark and intense?",
        )
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;font-size:0.72em;'
            f'color:{CARD_TEXT_MUTED};margin-top:-8px;margin-bottom:14px;">'
            f'<span>Feel-good & fun</span><span>Dark & gritty</span></div>',
            unsafe_allow_html=True,
        )

        intl_val = st.slider(
            "Origin", 0, 100, 40, format="%d",
            help="Prefer Hollywood productions or international world cinema?",
        )
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;font-size:0.72em;'
            f'color:{CARD_TEXT_MUTED};margin-top:-8px;margin-bottom:14px;">'
            f'<span>Hollywood focus</span><span>World cinema</span></div>',
            unsafe_allow_html=True,
        )

        awards_val = st.slider(
            "Critical vs Popular", 0, 100, 50, format="%d",
            help="Emmy/Oscar award winners, or box-office entertainment?",
        )
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;font-size:0.72em;'
            f'color:{CARD_TEXT_MUTED};margin-top:-8px;margin-bottom:14px;">'
            f'<span>Award winners</span><span>Pure entertainment</span></div>',
            unsafe_allow_html=True,
        )

    slider_prefs_captured = {
        "recency": era_val,
        "popularity": gems_val,
        "runtime": runtime_val,
        "maturity": tone_val,
        "international": intl_val,
        "awards": awards_val,
    }

    st.markdown(
        f'<div style="color:{CARD_TEXT};font-weight:700;font-size:1.0em;margin:12px 0 6px;">'
        f'Prefer movies, shows, or both?</div>',
        unsafe_allow_html=True,
    )
    tp_c1, tp_c2, tp_c3 = st.columns(3)
    type_pref = st.session_state.get("_dna_type_pref", "Both")
    with tp_c1:
        if st.button("🎥 Movies", use_container_width=True,
                     type="primary" if type_pref == "Movies" else "secondary"):
            st.session_state["_dna_type_pref"] = "Movies"
            st.rerun()
    with tp_c2:
        if st.button("📺 Shows", use_container_width=True,
                     type="primary" if type_pref == "Shows" else "secondary"):
            st.session_state["_dna_type_pref"] = "Shows"
            st.rerun()
    with tp_c3:
        if st.button("✨ Both", use_container_width=True,
                     type="primary" if type_pref == "Both" else "secondary"):
            st.session_state["_dna_type_pref"] = "Both"
            st.rerun()
    type_pref = st.session_state.get("_dna_type_pref", "Both")

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    quality_pref = "Award Winners" if awards_val <= 30 else ("Crowd Favorites" if gems_val >= 70 else "No Preference")
    vibe_pref = "Dark & Intense" if tone_val >= 70 else ("Feel-Good & Light" if tone_val <= 30 else "Mix of Both")

    _a2_cols = st.columns([3, 1])
    with _a2_cols[0]:
        if st.button("Start Swiping →", use_container_width=True, type="primary"):
            sel_genres = st.session_state.get("dna_quiz_genres", [])
            with st.spinner("Picking titles for you…"):
                titles = curate_quiz_titles(
                    raw_df,
                    selected_genres=sel_genres,
                    quality_pref=quality_pref,
                    type_pref=type_pref,
                    vibe_pref=vibe_pref,
                    enriched_df=enriched_df,
                )
            st.session_state["dna_quiz_prefs"] = {
                "genres": sel_genres, "quality": quality_pref,
                "type": type_pref, "vibe": vibe_pref,
            }
            st.session_state["dna_quiz_slider_prefs"] = slider_prefs_captured
            st.session_state["dna_quiz_titles"] = titles
            st.session_state["dna_quiz_current"] = 0
            st.session_state["dna_quiz_liked"] = []
            st.session_state["dna_quiz_phase"] = "B"
            st.rerun()
    with _a2_cols[1]:
        if st.button("← Back", use_container_width=True):
            st.session_state["dna_quiz_phase"] = "A"
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PHASE B: Title Swipe
# ─────────────────────────────────────────────────────────────────────────────
elif _quiz_phase == "B":
    quiz_titles = st.session_state["dna_quiz_titles"]
    current_idx = st.session_state["dna_quiz_current"]

    _render_step_chips("B")

    if not quiz_titles:
        st.warning("Could not find enough titles. Try different preferences.")
        if st.button("← Back to Preferences"):
            _reset_quiz()
            st.rerun()
    elif current_idx >= len(quiz_titles):
        st.session_state["dna_quiz_phase"] = "C"
        st.rerun()
    else:
        title = quiz_titles[current_idx]
        progress_val = (current_idx + 1) / len(quiz_titles)
        liked_so_far = len(st.session_state["dna_quiz_liked"])

        st.progress(progress_val, text=f"Title {current_idx + 1} of {len(quiz_titles)}  ·  {liked_so_far} liked so far")

        # Layout: poster left + card right
        poster_col, card_col = st.columns([1, 3], gap="large")

        with poster_col:
            if title.get("poster_url"):
                st.image(title["poster_url"], use_container_width=True)
            else:
                # Placeholder
                t_platform = title.get("platform", "")
                pc = PLATFORMS.get(t_platform, {}).get("color", CARD_BORDER)
                st.markdown(
                    f'<div style="background:{CARD_BG};border:2px solid {pc};border-radius:8px;'
                    f'height:200px;display:flex;align-items:center;justify-content:center;'
                    f'color:{CARD_TEXT_MUTED};font-size:2em;">🎬</div>',
                    unsafe_allow_html=True,
                )

        with card_col:
            t_platform = title.get("platform", "")
            t_color = PLATFORMS.get(t_platform, {}).get("color", CARD_BORDER)
            t_imdb = f"{title['imdb_score']:.1f}" if title.get("imdb_score") else "N/A"
            t_year = str(int(title["release_year"])) if title.get("release_year") else ""
            t_type = title.get("type", "")

            # Award badge
            aw_badge = ""
            if title.get("award_wins") and title["award_wins"] > 0:
                aw_badge = (
                    f'<span style="background:rgba(46,204,113,0.15);color:#2ecc71;'
                    f'border:1px solid #2ecc71;padding:2px 8px;border-radius:10px;'
                    f'font-size:0.78em;font-weight:600;margin-right:6px;">'
                    f'🏆 {int(title["award_wins"])} wins</span>'
                )

            # Platform badges
            plat_badges = "".join(
                f'<span style="display:inline-block;background:{PLATFORMS.get(p,{}).get("color",CARD_BORDER)};'
                f'color:#fff;border-radius:4px;padding:1px 8px;font-size:0.75em;'
                f'margin-right:4px;font-weight:600;">'
                f'{PLATFORMS.get(p,{}).get("name",p)}</span>'
                for p in (title.get("platforms") or [title.get("platform", "")])
            )

            # Genre pills
            genre_pills = "".join(
                f'<span style="display:inline-block;background:rgba(255,255,255,0.06);'
                f'color:{CARD_TEXT_MUTED};border-radius:8px;padding:2px 8px;'
                f'font-size:0.78em;margin-right:4px;">{str(g).title()}</span>'
                for g in (title.get("genres") or [])[:4]
            )

            st.markdown(
                f'<div style="background:{CARD_BG};border:2px solid {t_color};'
                f'border-radius:12px;padding:20px 24px;">'
                f'<div style="font-size:1.4em;font-weight:700;color:{CARD_TEXT};margin-bottom:6px;">'
                f'{title["title"]}</div>'
                f'<div style="margin-bottom:8px;">{aw_badge}{plat_badges}</div>'
                f'<div style="color:{CARD_TEXT_MUTED};font-size:0.88em;margin-bottom:8px;">'
                f'{t_year} · {t_type} · ⭐ {t_imdb}</div>'
                f'<div style="margin-bottom:12px;">{genre_pills}</div>'
                f'<div style="color:{CARD_TEXT};font-size:0.92em;line-height:1.6;">'
                f'{title.get("description","")}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Swipe buttons
            b1, b2 = st.columns(2)
            with b1:
                if st.button("👎  Not for me", key=f"quiz_pass_{current_idx}",
                             use_container_width=True):
                    st.session_state["dna_quiz_current"] = current_idx + 1
                    st.rerun()
            with b2:
                if st.button("❤️  I'd watch this!", key=f"quiz_like_{current_idx}",
                             use_container_width=True, type="primary"):
                    st.session_state["dna_quiz_liked"].append(title["id"])
                    st.session_state["dna_quiz_current"] = current_idx + 1
                    st.rerun()

        if current_idx >= 3:
            if st.button("Skip to results →", key="quiz_skip"):
                st.session_state["dna_quiz_phase"] = "C"
                st.rerun()

        if st.button("← Start Over", key="quiz_restart_b"):
            _reset_quiz()
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PHASE C: Results
# ─────────────────────────────────────────────────────────────────────────────
elif _quiz_phase == "C":
    liked_ids = st.session_state["dna_quiz_liked"]
    quiz_titles = st.session_state["dna_quiz_titles"]
    slider_prefs_saved = st.session_state.get("dna_quiz_slider_prefs", {})

    _render_step_chips("C")

    if not liked_ids:
        st.info("You didn't like any titles. Try again with different preferences!")
        if st.button("Start Over"):
            _reset_quiz()
            st.rerun()
    else:
        # Compute results (cached in session state)
        if st.session_state["dna_quiz_results"] is None:
            with st.spinner("Analyzing your taste…"):
                st.session_state["dna_quiz_results"] = compute_swipe_results_v2(
                    liked_ids=liked_ids,
                    all_titles=quiz_titles,
                    all_df=raw_df,
                    slider_prefs=slider_prefs_saved,
                    enriched_df=enriched_df,
                    enriched_stats=enriched_stats,
                    user_selected_genres=st.session_state.get("dna_quiz_genres", []),
                )

        results = st.session_state["dna_quiz_results"]
        rankings = results.get("rankings", [])
        personality = results.get("personality", "")
        recommendations = results.get("recommendations", [])
        why_match = results.get("why_match", [])

        if not rankings:
            st.warning("Could not compute rankings. Try different preferences.")
        else:
            best = rankings[0]
            best_color = PLATFORMS.get(best["platform"], {}).get("color", CARD_ACCENT)

            # ── Hero match card ───────────────────────────────────────────────
            hero_cols = st.columns([3, 1])
            with hero_cols[0]:
                st.markdown(
                    f'<div style="background:{CARD_BG};border:2px solid {best_color};'
                    f'border-radius:14px;padding:24px 28px;margin:8px 0;">'
                    f'<div style="font-size:0.88em;color:{CARD_TEXT_MUTED};margin-bottom:4px;">Your match is</div>'
                    f'<div style="font-size:2.2em;font-weight:800;color:{best_color};margin:2px 0;">'
                    f'{best["display_name"]}</div>'
                    f'<div style="font-size:1.6em;color:{CARD_ACCENT};font-weight:700;margin-bottom:12px;">'
                    f'{best["match_pct"]:.1f}% match</div>'
                    f'<div style="font-size:0.9em;color:{CARD_TEXT};line-height:1.7;">'
                    f'{best["explanation"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with hero_cols[1]:
                if personality:
                    st.markdown(
                        f'<div style="background:{CARD_BG};border:1px solid {CARD_BORDER};'
                        f'border-radius:10px;padding:14px 16px;height:100%;">'
                        f'<div style="color:{CARD_ACCENT};font-weight:700;font-size:0.85em;'
                        f'margin-bottom:6px;">Your Viewing DNA</div>'
                        f'<div style="color:{CARD_TEXT};font-size:0.88em;line-height:1.6;">'
                        f'{personality}</div>'
                        f'<div style="margin-top:8px;color:{CARD_TEXT_MUTED};font-size:0.78em;">'
                        f'{len(liked_ids)} of {len(quiz_titles)} titles liked '
                        f'({len(liked_ids)/max(len(quiz_titles),1)*100:.0f}%)</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # ── Why This Match ────────────────────────────────────────────────
            if why_match:
                st.markdown(
                    f'<div style="color:{CARD_TEXT_MUTED};font-size:0.72em;text-transform:uppercase;'
                    f'letter-spacing:0.04em;margin:16px 0 8px;">Why This Match</div>',
                    unsafe_allow_html=True,
                )
                bullets_html = "".join(
                    f'<div style="display:flex;gap:10px;align-items:flex-start;margin-bottom:6px;">'
                    f'<span style="color:{best_color};font-weight:700;font-size:1.1em;">✓</span>'
                    f'<span style="color:{CARD_TEXT};font-size:0.9em;">{b}</span></div>'
                    for b in why_match
                )
                st.markdown(
                    f'<div style="background:{CARD_BG};border-left:4px solid {best_color};'
                    f'border-radius:0 8px 8px 0;padding:14px 18px;margin-bottom:16px;">'
                    f'<div style="font-size:0.85em;color:{CARD_TEXT_MUTED};margin-bottom:8px;">'
                    f'You\'re a <strong style="color:{best_color};">{best["display_name"]}</strong> '
                    f'match because you prefer:</div>'
                    f'{bullets_html}</div>',
                    unsafe_allow_html=True,
                )

            # ── How You Match Every Platform — ranked scorecard ───────────────
            st.markdown(
                f'<div style="color:{CARD_TEXT_MUTED};font-size:0.72em;text-transform:uppercase;'
                f'letter-spacing:0.04em;margin:16px 0 10px;">How You Match Every Platform</div>',
                unsafe_allow_html=True,
            )
            _rank_badges = {
                1: ("🥇", "#FFD700", "#1a1a2e"),
                2: ("🥈", "#C0C0C0", "#1a1a2e"),
                3: ("🥉", "#CD7F32", "#1a1a2e"),
            }
            _scorecard_rows = ""
            for _ri, _result in enumerate(rankings):
                _rank = _ri + 1
                _pc = PLATFORMS.get(_result["platform"], {}).get("color", CARD_BORDER)
                _pct = _result["match_pct"]
                _badge_icon, _badge_bg, _badge_fg = _rank_badges.get(_rank, ("", _pc + "33", CARD_TEXT_MUTED))
                _is_top = _rank == 1
                _row_bg = f"rgba(255,255,255,0.05)" if _is_top else "rgba(255,255,255,0.02)"
                _border = f"2px solid {_pc}44" if _is_top else f"1px solid {CARD_BORDER}"
                _scorecard_rows += (
                    f'<div style="display:flex;align-items:center;gap:12px;padding:10px 14px;'
                    f'margin-bottom:6px;border-radius:8px;background:{_row_bg};border:{_border};">'
                    f'<div style="font-size:1.3em;min-width:28px;text-align:center;">{_badge_icon or f"#{_rank}"}</div>'
                    f'<div style="min-width:110px;color:{_pc};font-weight:{"800" if _is_top else "600"};'
                    f'font-size:{"1.0em" if _is_top else "0.92em"};">{_result["display_name"]}</div>'
                    f'<div style="flex:1;height:10px;background:rgba(255,255,255,0.08);'
                    f'border-radius:5px;overflow:hidden;">'
                    f'<div style="width:{_pct}%;height:100%;background:{_pc};border-radius:5px;'
                    f'transition:width 0.3s;"></div></div>'
                    f'<div style="color:{_pc};font-weight:800;font-size:{"1.15em" if _is_top else "0.95em"};'
                    f'min-width:52px;text-align:right;">{_pct:.0f}%</div>'
                    f'</div>'
                )
            st.markdown(
                f'<div style="background:{CARD_BG};border:1px solid {CARD_BORDER};'
                f'border-radius:12px;padding:14px 16px;">{_scorecard_rows}</div>',
                unsafe_allow_html=True,
            )

            # ── Platform Breakdown — natural language per-platform explanation ─
            st.markdown(
                f'<div style="color:{CARD_TEXT_MUTED};font-size:0.72em;text-transform:uppercase;'
                f'letter-spacing:0.04em;margin:14px 0 8px;">Platform Breakdown</div>',
                unsafe_allow_html=True,
            )
            _plat_expls = results.get("platform_explanations", {})
            for _result in rankings:
                _pk2 = _result["platform"]
                _pc2 = PLATFORMS.get(_pk2, {}).get("color", CARD_BORDER)
                _expl_text = _plat_expls.get(_pk2, "")
                _is_best2 = _result["platform"] == best["platform"]
                _row_bg2 = "rgba(255,255,255,0.04)" if _is_best2 else "rgba(255,255,255,0.015)"
                _border2 = f"1px solid {_pc2}44" if _is_best2 else f"1px solid {CARD_BORDER}22"
                st.markdown(
                    f'<div style="padding:10px 14px;margin-bottom:6px;border-radius:8px;'
                    f'background:{_row_bg2};border:{_border2};">'
                    f'<div style="display:flex;align-items:center;justify-content:space-between;'
                    f'margin-bottom:4px;">'
                    f'<span style="color:{_pc2};font-weight:700;font-size:0.92em;">'
                    f'{_result["display_name"]}</span>'
                    f'<span style="color:{_pc2};font-weight:800;font-size:0.92em;">'
                    f'{_result["match_pct"]:.0f}% match</span></div>'
                    f'<div style="color:{CARD_TEXT_MUTED};font-size:0.83em;line-height:1.55;">'
                    f'{_expl_text}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # ── Recommended titles ────────────────────────────────────────────
            if recommendations:
                st.markdown(
                    f'<div style="color:{CARD_TEXT_MUTED};font-size:0.72em;text-transform:uppercase;'
                    f'letter-spacing:0.04em;margin:8px 0 8px;">Recommended for You on {best["display_name"]}</div>',
                    unsafe_allow_html=True,
                )
                rec_cols = st.columns(min(len(recommendations), 5))
                for ri, rec in enumerate(recommendations[:5]):
                    with rec_cols[ri % len(rec_cols)]:
                        r_imdb = f"{rec['imdb_score']:.1f}" if rec.get("imdb_score") else "N/A"
                        r_year = str(int(rec["release_year"])) if rec.get("release_year") else ""
                        r_aw_badge = ""
                        if rec.get("award_wins") and rec["award_wins"] > 0:
                            _aw_n = int(rec["award_wins"])
                            _aw_word = "win" if _aw_n == 1 else "wins"
                            r_aw_badge = (
                                f'<span style="background:rgba(46,204,113,0.15);color:#2ecc71;'
                                f'border-radius:4px;padding:1px 5px;font-size:0.68em;'
                                f'font-weight:600;margin-right:3px;">🏆 {_aw_n} {_aw_word}</span>'
                            )
                        r_genres = "".join(
                            f'<span style="display:inline-block;background:rgba(255,255,255,0.06);'
                            f'color:{CARD_TEXT_MUTED};border-radius:6px;padding:1px 5px;'
                            f'font-size:0.65em;margin-right:2px;">{str(rg).title()}</span>'
                            for rg in (rec.get("genres") or [])[:2]
                        )
                        _rec_poster = rec.get("poster_url")
                        _rec_poster_block = (
                            f'<img src="{_rec_poster}" style="width:100%;height:150px;'
                            f'object-fit:cover;border-radius:6px 6px 0 0;display:block;" '
                            f'onerror="this.style.display=\'none\'" />'
                            if _rec_poster else
                            f'<div style="width:100%;height:150px;border-radius:6px 6px 0 0;'
                            f'background:{best_color}22;border-bottom:2px solid {best_color};'
                            f'display:flex;align-items:center;justify-content:center;font-size:2em;">🎬</div>'
                        )
                        _rec_is_active = st.session_state.get("dna_quiz_rec_detail_id") == rec.get("id")
                        st.markdown(
                            f'<div style="background:{CARD_BG};border:1px solid {CARD_BORDER};'
                            f'border-top:3px solid {best_color};border-radius:8px;'
                            f'overflow:hidden;min-height:320px;display:flex;flex-direction:column;">'
                            f'{_rec_poster_block}'
                            f'<div style="padding:8px 10px;flex:1;display:flex;flex-direction:column;">'
                            f'<div style="color:{CARD_TEXT};font-weight:600;font-size:0.83em;'
                            f'line-height:1.3;margin-bottom:6px;flex:1;'
                            f'overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;'
                            f'-webkit-box-orient:vertical;">{rec["title"]}</div>'
                            f'<div style="margin-top:auto;">'
                            f'<div style="margin-bottom:3px;">{r_aw_badge}</div>'
                            f'<div style="color:{CARD_TEXT_MUTED};font-size:0.72em;">'
                            f'{r_year} · ⭐ {r_imdb} · {rec.get("type","")}</div>'
                            f'<div style="margin-top:3px;">{r_genres}</div>'
                            f'</div></div></div>',
                            unsafe_allow_html=True,
                        )
                        if rec.get("id") and st.button(
                            "✕ Close" if _rec_is_active else "Details",
                            key=f"rec_detail_{rec['id']}_{ri}", use_container_width=True
                        ):
                            cur = st.session_state.get("dna_quiz_rec_detail_id")
                            st.session_state["dna_quiz_rec_detail_id"] = None if cur == rec["id"] else rec["id"]
                            st.rerun()

                # Show detail panel for selected recommendation
                _rec_detail_id = st.session_state.get("dna_quiz_rec_detail_id")
                if _rec_detail_id is not None:
                    st.markdown(
                        f'<div style="color:{CARD_TEXT_MUTED};font-size:0.72em;text-transform:uppercase;'
                        f'letter-spacing:0.04em;margin:12px 0 6px;">Title Details</div>',
                        unsafe_allow_html=True,
                    )
                    _render_inline_title_detail(_rec_detail_id, raw_df, enriched_df)

            # ── Compare your preferences to each platform — polished table ────
            with st.expander("Compare your preferences to each platform"):
                _cq_genres = st.session_state.get("dna_quiz_genres", [])[:3]
                # Build column headers
                _g_headers = [_GENRE_DISPLAY_NAMES.get(g, g.title()) for g in _cq_genres]
                _header_cells = "".join(
                    f'<th style="padding:8px 12px;color:{CARD_TEXT_MUTED};font-size:0.78em;'
                    f'font-weight:600;text-align:right;white-space:nowrap;">{h} in catalog</th>'
                    for h in _g_headers
                )
                _tbl_html = (
                    f'<table style="width:100%;border-collapse:collapse;font-size:0.88em;">'
                    f'<thead><tr style="border-bottom:2px solid {CARD_BORDER};">'
                    f'<th style="padding:8px 12px;color:{CARD_TEXT_MUTED};font-size:0.78em;'
                    f'font-weight:600;text-align:left;">Platform</th>'
                    f'<th style="padding:8px 12px;color:{CARD_TEXT_MUTED};font-size:0.78em;'
                    f'font-weight:600;text-align:right;">Your Match</th>'
                    f'{_header_cells}'
                    f'<th style="padding:8px 12px;color:{CARD_TEXT_MUTED};font-size:0.78em;'
                    f'font-weight:600;text-align:right;">Avg IMDb</th>'
                    f'<th style="padding:8px 12px;color:{CARD_TEXT_MUTED};font-size:0.78em;'
                    f'font-weight:600;text-align:right;">Post-2015</th>'
                    f'<th style="padding:8px 12px;color:{CARD_TEXT_MUTED};font-size:0.78em;'
                    f'font-weight:600;text-align:right;">Intl</th>'
                    f'<th style="padding:8px 12px;color:{CARD_TEXT_MUTED};font-size:0.78em;'
                    f'font-weight:600;text-align:right;">Award Wins</th>'
                    f'</tr></thead><tbody>'
                )
                for _ri2, r in enumerate(rankings):
                    pk = r["platform"]
                    _pc3 = PLATFORMS.get(pk, {}).get("color", CARD_BORDER)
                    _est_r = enriched_stats.get(pk, {})
                    plat_df_r = raw_df[raw_df["platform"] == pk]
                    _pn = max(len(plat_df_r), 1)
                    rated_r = plat_df_r.dropna(subset=["imdb_score"])
                    avg_q = rated_r["imdb_score"].mean() if len(rated_r) > 0 else 0
                    post2015 = (plat_df_r["release_year"] >= 2015).mean() * 100 if len(plat_df_r) > 0 else 0
                    valid_c = plat_df_r[plat_df_r["production_countries"].apply(
                        lambda c: isinstance(c, list) and len(c) > 0
                    )]
                    intl = (
                        valid_c["production_countries"].apply(lambda c: "US" not in c).mean() * 100
                        if len(valid_c) > 0 else 0
                    )
                    # Per-genre share cells
                    _gc3: dict[str, int] = {}
                    for _gs3 in plat_df_r["genres"].dropna():
                        if isinstance(_gs3, list):
                            for _g3 in _gs3:
                                _gc3[str(_g3).lower()] = _gc3.get(str(_g3).lower(), 0) + 1
                    _genre_cells = ""
                    for _cg in _cq_genres:
                        _cg_pct = _gc3.get(_cg.lower(), 0) / _pn * 100
                        _cg_color = "#2ecc71" if _cg_pct >= 12 else (CARD_TEXT if _cg_pct >= 6 else CARD_TEXT_MUTED)
                        _genre_cells += (
                            f'<td style="padding:8px 12px;text-align:right;'
                            f'color:{_cg_color};font-weight:{"700" if _cg_pct >= 12 else "400"};">'
                            f'{_cg_pct:.0f}%</td>'
                        )
                    # Match pct with color coding
                    _mpct = r["match_pct"]
                    _mpct_color = "#2ecc71" if _mpct >= 80 else (_pc3 if _mpct >= 65 else CARD_TEXT_MUTED)
                    _row_bg = "rgba(255,255,255,0.04)" if _ri2 == 0 else ("rgba(255,255,255,0.02)" if _ri2 % 2 == 0 else "transparent")
                    _best_badge = '<span style="font-size:0.72em;color:#FFD700;margin-left:6px;">Best match</span>' if _ri2 == 0 else ""
                    _tbl_html += (
                        f'<tr style="border-bottom:1px solid {CARD_BORDER}22;background:{_row_bg};">'
                        f'<td style="padding:8px 12px;">'
                        f'<span style="color:{_pc3};font-weight:{"700" if _ri2==0 else "600"};'
                        f'font-size:0.9em;">{r["display_name"]}</span>{_best_badge}'
                        f'</td>'
                        f'<td style="padding:8px 12px;text-align:right;color:{_mpct_color};'
                        f'font-weight:700;">{_mpct:.0f}%</td>'
                        f'{_genre_cells}'
                        f'<td style="padding:8px 12px;text-align:right;color:{CARD_TEXT};">{avg_q:.1f}</td>'
                        f'<td style="padding:8px 12px;text-align:right;color:{CARD_TEXT_MUTED};">{post2015:.0f}%</td>'
                        f'<td style="padding:8px 12px;text-align:right;color:{CARD_TEXT_MUTED};">{intl:.0f}%</td>'
                        f'<td style="padding:8px 12px;text-align:right;color:{CARD_TEXT_MUTED};">'
                        f'{_est_r.get("award_wins", 0):,}</td>'
                        f'</tr>'
                    )
                _tbl_html += "</tbody></table>"
                st.markdown(
                    f'<div style="background:{CARD_BG};border:1px solid {CARD_BORDER};'
                    f'border-radius:10px;overflow:hidden;">{_tbl_html}</div>',
                    unsafe_allow_html=True,
                )

        # Retake button
        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        if st.button("🔄  Take Quiz Again", use_container_width=True):
            _reset_quiz()
            st.rerun()


# ── footer ────────────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    '<div style="color:#555;font-size:0.8em;text-align:center;padding:8px 0 16px;">'
    'Hypothetical merger for academic analysis. Data is a snapshot (mid-2023). '
    'All insights are illustrative, not prescriptive. '
    'Update: As of Feb 26, 2026, Netflix withdrew from this acquisition after Paramount Skydance\'s competing bid was deemed superior by the WBD board.'
    '</div>',
    unsafe_allow_html=True,
)
