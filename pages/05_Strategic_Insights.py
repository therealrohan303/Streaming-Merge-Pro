"""Page 5: Strategic Insights — Merger business intelligence.

6 sections with decision traces, not just dashboards:
  1. Merger Value Dashboard
  2. Prestige Index (Wikidata)
  3. Content Overlap Analysis
  4. Gap Analysis with Decision Trace
  5. IP Synergy Map (TMDB)
  6. Market Impact Simulation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.config import (
    ALL_PLATFORMS,
    CARD_ACCENT,
    CARD_BG,
    CARD_BORDER,
    CARD_TEXT,
    CARD_TEXT_MUTED,
    COMPETITOR_PLATFORMS,
    MERGED_PLATFORMS,
    PLATFORMS,
    PLOTLY_TEMPLATE,
    PRECOMPUTED_DIR,
    WIKIDATA_MIN_COVERAGE,
    TMDB_MIN_COVERAGE,
)
from src.data.loaders import (
    load_enriched_titles,
)
from src.analysis.strategic import (
    build_merged_entity,
    compute_gap_analysis,
    compute_ip_synergy,
    compute_market_simulation,
    compute_merger_kpis,
    compute_overlap_analysis,
    compute_overlap_heatmap,
)
from src.ui.session import init_session_state

st.set_page_config(page_title="Strategic Insights", page_icon="📊", layout="wide")
init_session_state()

st.title("Strategic Insights")
st.caption("Merger business intelligence with decision traces and evidence-based recommendations.")

# ─── Data Loading ───────────────────────────────────────────────────────────
titles = load_enriched_titles()

# ─── Section 1: Merger Value Dashboard ──────────────────────────────────────
st.header("Merger Value Dashboard")
st.caption("Headline metrics quantifying the strategic value of the Netflix + Max merger.")

kpis = compute_merger_kpis(titles)

cols = st.columns(len(kpis))
for i, (key, kpi) in enumerate(kpis.items()):
    with cols[i]:
        st.metric(
            label=kpi["label"],
            value=kpi["value"],
            help=kpi["detail"],
        )

st.markdown("---")

# ─── Section 2: Prestige Index ──────────────────────────────────────────────
st.header("Prestige Index")
st.caption("Award wins per 1,000 titles by platform and genre — measures prestige content concentration.")

prestige_path = PRECOMPUTED_DIR / "strategic_analysis" / "prestige_index.parquet"
if prestige_path.exists():
    prestige = pd.read_parquet(prestige_path)

    # Check coverage
    award_coverage = titles["award_wins"].notna().mean() if "award_wins" in titles.columns else 0

    if award_coverage >= WIKIDATA_MIN_COVERAGE:
        st.info(f"Prestige Index (Wikidata, {award_coverage:.0%} coverage)")

        # Bar chart: platforms ranked by overall prestige
        platform_prestige = (
            prestige.groupby("platform")
            .agg({"prestige_per_1k": "mean", "award_wins": "sum", "title_count": "sum"})
            .reset_index()
            .sort_values("prestige_per_1k", ascending=False)
        )
        platform_prestige["name"] = platform_prestige["platform"].map(
            lambda p: PLATFORMS.get(p, {}).get("name", p)
        )

        fig_bar = px.bar(
            platform_prestige, x="name", y="prestige_per_1k",
            color="platform",
            color_discrete_map={p: PLATFORMS[p]["color"] for p in PLATFORMS},
            title="Prestige Score by Platform (Award Wins per 1,000 Titles)",
            template=PLOTLY_TEMPLATE,
        )
        fig_bar.update_layout(showlegend=False, xaxis_title="", yaxis_title="Prestige per 1K titles")
        st.plotly_chart(fig_bar, use_container_width=True)

        # Heatmap: genre × platform prestige intensity
        heatmap_data = prestige.pivot_table(
            index="genre", columns="platform", values="prestige_per_1k", fill_value=0
        )
        heatmap_data.columns = [PLATFORMS.get(p, {}).get("name", p) for p in heatmap_data.columns]

        fig_heat = px.imshow(
            heatmap_data, aspect="auto",
            color_continuous_scale="Purples",
            title="Genre × Platform Prestige Intensity",
            template=PLOTLY_TEMPLATE,
        )
        fig_heat.update_layout(height=500)
        st.plotly_chart(fig_heat, use_container_width=True)

        # ROI proxy column (box_office / budget)
        if "budget_usd" in titles.columns and "box_office_usd" in titles.columns:
            roi_data = titles[
                titles["budget_usd"].notna() & (titles["budget_usd"] > 0) &
                titles["box_office_usd"].notna() & (titles["box_office_usd"] > 0)
            ].copy()
            if len(roi_data) > 20:
                roi_data["roi_proxy"] = roi_data["box_office_usd"] / roi_data["budget_usd"]
                st.subheader("ROI Proxy")
                st.caption("Estimated ROI proxy (Wikidata, partial coverage) — box office / budget")
                platform_roi = (
                    roi_data.groupby("platform")["roi_proxy"]
                    .agg(["median", "count"])
                    .reset_index()
                )
                platform_roi["name"] = platform_roi["platform"].map(
                    lambda p: PLATFORMS.get(p, {}).get("name", p)
                )
                st.dataframe(
                    platform_roi.rename(columns={"median": "Median ROI", "count": "Titles with Data"}),
                    hide_index=True,
                )
    else:
        st.info(f"Prestige Index requires at least {WIKIDATA_MIN_COVERAGE:.0%} award data coverage. "
                f"Current coverage: {award_coverage:.0%}. "
                "Run the Wikidata enrichment pipeline to improve coverage.")
else:
    st.info("Prestige index data not found. Run `scripts/11_precompute_strategic.py` first.")

st.markdown("---")

# ─── Section 3: Content Overlap Analysis ────────────────────────────────────
st.header("Content Overlap Analysis")
st.caption("Mapping where Netflix and Max duplicate vs complement each other.")

overlap = compute_overlap_analysis(titles)
if not overlap.empty:
    heatmap_data = compute_overlap_heatmap(overlap)

    # Complementarity score: what fraction of genre-type combos have low overlap
    total_combos = len(overlap)
    complementary = len(overlap[overlap["confidence"] < 0.3])
    complementarity_pct = complementary / max(total_combos, 1)

    # Summary metrics
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Genre-Type Combinations", f"{total_combos}")
    with m2:
        high_overlap_count = len(overlap[overlap["confidence"] > 0.5])
        st.metric("High Overlap Pairs", f"{high_overlap_count}",
                  help="Genre-type combos where both platforms have similar content volume")
    with m3:
        st.metric("Complementarity Score", f"{complementarity_pct:.0%}",
                  help="Percentage of genre-type combos with low overlap — higher = more complementary")

    if not heatmap_data.empty:
        # Stacked bar showing Netflix-exclusive, overlap, and Max-exclusive per genre
        plot_data = heatmap_data.head(15).copy()
        plot_data["netflix_exclusive"] = plot_data["netflix_count"] - plot_data["overlap"]
        plot_data["max_exclusive"] = plot_data["max_count"] - plot_data["overlap"]

        fig_overlap = go.Figure()
        fig_overlap.add_trace(go.Bar(
            name="Netflix Exclusive", x=plot_data["genre"],
            y=plot_data["netflix_exclusive"],
            marker_color=PLATFORMS["netflix"]["color"],
        ))
        fig_overlap.add_trace(go.Bar(
            name="Shared", x=plot_data["genre"],
            y=plot_data["overlap"],
            marker_color="#FFD700",
        ))
        fig_overlap.add_trace(go.Bar(
            name="Max Exclusive", x=plot_data["genre"],
            y=plot_data["max_exclusive"],
            marker_color=PLATFORMS["max"]["color"],
        ))
        fig_overlap.update_layout(
            barmode="stack",
            title="Content Distribution: Netflix vs Max by Genre (Top 15)",
            xaxis_title="", yaxis_title="Title Count",
            template=PLOTLY_TEMPLATE, height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_overlap, use_container_width=True)

    # Audit table
    with st.expander("Overlap audit table"):
        audit = overlap.sort_values("confidence", ascending=False)
        st.dataframe(
            audit[["genre", "type", "netflix_count", "max_count", "overlap", "confidence"]],
            hide_index=True,
            column_config={
                "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1),
            },
        )

    # Strategic implication — data-driven
    high_overlap = overlap[overlap["confidence"] > 0.5]
    low_overlap = overlap[overlap["confidence"] < 0.2]
    top_overlapping = high_overlap.nlargest(3, "overlap")["genre"].tolist() if not high_overlap.empty else []
    top_complementary = low_overlap.nlargest(3, "netflix_count")["genre"].tolist() if not low_overlap.empty else []

    if len(high_overlap) > len(low_overlap):
        overlap_genres_str = ", ".join(top_overlapping[:3]) if top_overlapping else "multiple genres"
        st.success(
            f"High overlap detected in {len(high_overlap)} genre-type pairs "
            f"(notably {overlap_genres_str}). "
            "Strong curation opportunity to reduce redundancy and focus on quality over quantity."
        )
    else:
        comp_genres_str = ", ".join(top_complementary[:3]) if top_complementary else "multiple genres"
        st.success(
            f"The catalogs are largely complementary ({complementarity_pct:.0%} low-overlap). "
            f"Max adds depth in {comp_genres_str}, maximizing genre breadth in the merged entity."
        )

st.markdown("---")

# ─── Section 4: Gap Analysis with Decision Trace ────────────────────────────
st.header("Gap Analysis")
st.caption("Identifying content gaps with full decision traces for strategic acquisition planning.")

gap_col1, gap_col2 = st.columns(2)
with gap_col1:
    gap_perspective = st.selectbox("Perspective", ["merged"] + ALL_PLATFORMS, key="gap_persp",
                                   format_func=lambda x: PLATFORMS.get(x, {}).get("name", x) if x != "merged" else "Merged (Netflix + Max)")
with gap_col2:
    gap_competitor = st.selectbox("Compare against", ["All Competitors"] + COMPETITOR_PLATFORMS,
                                  key="gap_comp",
                                  format_func=lambda x: PLATFORMS.get(x, {}).get("name", x) if x != "All Competitors" else x)

gaps = compute_gap_analysis(
    titles,
    perspective=gap_perspective,
    competitor=gap_competitor if gap_competitor != "All Competitors" else None,
)

if not gaps.empty:
    # Summary metrics
    high_gaps = len(gaps[gaps["severity"] == "High"])
    med_gaps = len(gaps[gaps["severity"] == "Medium"])

    g1, g2, g3 = st.columns(3)
    with g1:
        st.metric("Total Gaps Found", len(gaps))
    with g2:
        st.metric("High Priority", high_gaps, help="Genres where competitors lead by 3x+ or base has <3% share")
    with g3:
        st.metric("Medium Priority", med_gaps, help="Genres where competitors lead by 1.5-3x")

    # Render gaps as cards
    for _, gap in gaps.iterrows():
        severity_icon = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(gap["severity"], "")
        severity_border = {"High": "#E74C3C", "Medium": "#F39C12", "Low": "#2ECC71"}.get(gap["severity"], CARD_BORDER)

        st.markdown(f"""
        <div style="background:{CARD_BG};border-left:4px solid {severity_border};
                    border-radius:6px;padding:12px 16px;margin-bottom:8px;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <span style="font-size:1rem;font-weight:600;color:{CARD_TEXT};">
                    {severity_icon} {gap['genre']}</span>
                <span style="color:{CARD_TEXT_MUTED};font-size:0.85em;">
                    Coverage: {gap['coverage_pct']:.1f}% | {gap['competitor_lead']}</span>
            </div>
            <div style="margin-top:6px;font-size:0.88em;color:{CARD_TEXT};">
                {gap['recommendation']}</div>
            <div style="margin-top:4px;font-size:0.78em;color:{CARD_TEXT_MUTED};">
                Quality benchmark: IMDb {gap['quality_benchmark']:.2f if pd.notna(gap['quality_benchmark']) else 'N/A'}
                | Box office tier: {gap['box_office_tier']}
                | Confidence: {gap['confidence']}</div>
        </div>
        """, unsafe_allow_html=True)

    # Precomputed targets
    targets_path = PRECOMPUTED_DIR / "strategic_analysis" / "acquisition_targets.parquet"
    if targets_path.exists():
        with st.expander("Precomputed acquisition targets (all competitors)"):
            targets = pd.read_parquet(targets_path)
            st.dataframe(targets, hide_index=True, height=300)
else:
    st.info("No significant gaps found for this perspective — the catalog covers all genres as well or better than competitors.")

st.markdown("---")

# ─── Section 5: IP Synergy Map ──────────────────────────────────────────────
st.header("IP Synergy Map")
st.caption("Franchise and collection analysis — which IP portfolios become dominant post-merger.")

synergy_data, coverage = compute_ip_synergy(titles)
if synergy_data is not None:
    st.info(f"TMDB collection coverage: {coverage:.0%}")

    tab_pre, tab_post = st.tabs(["Pre-Merger", "Post-Merger"])

    with tab_pre:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Netflix Top Franchises")
            if not synergy_data["netflix"].empty:
                st.dataframe(
                    synergy_data["netflix"][["franchise", "title_count", "avg_imdb", "quality_score"]],
                    hide_index=True,
                    column_config={
                        "avg_imdb": st.column_config.NumberColumn("Avg IMDb", format="%.1f"),
                        "quality_score": st.column_config.NumberColumn("Quality Score", format="%.1f"),
                    },
                )
            else:
                st.caption("No franchise data available for Netflix.")
        with col2:
            st.subheader("Max Top Franchises")
            if not synergy_data["max"].empty:
                st.dataframe(
                    synergy_data["max"][["franchise", "title_count", "avg_imdb", "quality_score"]],
                    hide_index=True,
                    column_config={
                        "avg_imdb": st.column_config.NumberColumn("Avg IMDb", format="%.1f"),
                        "quality_score": st.column_config.NumberColumn("Quality Score", format="%.1f"),
                    },
                )
            else:
                st.caption("No franchise data available for Max.")

    with tab_post:
        st.subheader("Merged Entity — Top 15 Franchises by Quality Score")
        if not synergy_data["merged"].empty:
            fig_franchise = px.bar(
                synergy_data["merged"].head(15),
                x="franchise", y="quality_score",
                color="avg_imdb",
                color_continuous_scale="Viridis",
                title="Top Franchises in Merged Catalog",
                template=PLOTLY_TEMPLATE,
            )
            fig_franchise.update_layout(xaxis_title="", xaxis_tickangle=-45, height=450)
            st.plotly_chart(fig_franchise, use_container_width=True)

            # Show synergy: franchises that exist on BOTH platforms
            netflix_f = set(synergy_data["netflix"]["franchise"]) if not synergy_data["netflix"].empty else set()
            max_f = set(synergy_data["max"]["franchise"]) if not synergy_data["max"].empty else set()
            shared_f = netflix_f & max_f
            if shared_f:
                st.markdown(f"**Cross-platform franchises ({len(shared_f)}):** {', '.join(sorted(shared_f)[:10])}")
        else:
            st.caption("No franchise data available for merged entity.")
else:
    # Check if TMDB cache exists to provide progress info
    tmdb_cache_path = PRECOMPUTED_DIR.parent / "enriched" / "tmdb_cache.parquet"
    if tmdb_cache_path.exists():
        try:
            cache = pd.read_parquet(tmdb_cache_path)
            total_titles = len(titles)
            cached = len(cache)
            pct = cached / max(total_titles, 1)
            st.info(
                f"TMDB enrichment in progress: {cached:,}/{total_titles:,} titles cached ({pct:.0%}). "
                "IP Synergy will be available after enrichment completes and "
                "`scripts/09_build_enriched_titles.py` is re-run."
            )
        except Exception:
            st.info("IP Synergy Map requires TMDB enrichment data. "
                    "Run `scripts/08_enrich_tmdb.py` and `scripts/09_build_enriched_titles.py` first.")
    elif coverage > 0:
        st.info(f"IP Synergy Map requires at least {TMDB_MIN_COVERAGE:.0%} TMDB collection coverage. "
                f"Current coverage: {coverage:.0%}.")
    else:
        st.info("IP Synergy Map requires TMDB enrichment data. "
                "Run `scripts/08_enrich_tmdb.py` and `scripts/09_build_enriched_titles.py` first.")

st.markdown("---")

# ─── Section 6: Market Impact Simulation ────────────────────────────────────
st.header("Market Impact Simulation")
st.caption("Simulated market based on catalog data only. Not actual financial market share.")
st.warning("This simulation uses catalog size as a proxy for market position. "
           "It does not represent actual subscriber counts or financial market share.")

sim = compute_market_simulation(titles)

# Pre-merger vs post-merger comparison
col_pre, col_post = st.columns(2)

with col_pre:
    # Pre-merger: all 6 platforms individually
    pre_data = []
    for key in ALL_PLATFORMS:
        val = sim["platforms"].get(key)
        if val:
            pre_data.append({
                "platform": val["name"],
                "size": val["catalog_size"],
                "color": PLATFORMS.get(key, {}).get("color", "#555"),
            })
    pre_df = pd.DataFrame(pre_data)

    fig_pre = px.pie(
        pre_df, names="platform", values="size",
        title="Pre-Merger: Catalog Size Share",
        color="platform",
        color_discrete_map={d["platform"]: d["color"] for d in pre_data},
        template=PLOTLY_TEMPLATE,
    )
    fig_pre.update_layout(showlegend=True)
    st.plotly_chart(fig_pre, use_container_width=True)

with col_post:
    # Post-merger: merged entity + competitors
    post_data = []
    for key, val in sim["platforms"].items():
        if key not in ("netflix", "max"):
            post_data.append({
                "platform": val["name"],
                "size": val["catalog_size"],
                "color": PLATFORMS.get(key, {}).get("color", "#555"),
            })
    post_df = pd.DataFrame(post_data)

    fig_post = px.pie(
        post_df, names="platform", values="size",
        title="Post-Merger: Catalog Size Share",
        color="platform",
        color_discrete_map={d["platform"]: d["color"] for d in post_data},
        template=PLOTLY_TEMPLATE,
    )
    fig_post.update_layout(showlegend=True)
    st.plotly_chart(fig_post, use_container_width=True)

# HHI gauge
hhi = sim["hhi"]
reg_label = sim["regulatory_label"]

# Visual HHI indicator
hhi_col1, hhi_col2 = st.columns([2, 1])
with hhi_col1:
    fig_hhi = go.Figure(go.Indicator(
        mode="gauge+number",
        value=hhi,
        title={"text": "Post-Merger HHI (Herfindahl-Hirschman Index)"},
        gauge={
            "axis": {"range": [0, 5000], "tickwidth": 1},
            "bar": {"color": CARD_ACCENT},
            "steps": [
                {"range": [0, 1500], "color": "#1a3a1a"},
                {"range": [1500, 2500], "color": "#3a3a1a"},
                {"range": [2500, 5000], "color": "#3a1a1a"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 2},
                "thickness": 0.8,
                "value": hhi,
            },
        },
    ))
    fig_hhi.update_layout(height=280, template=PLOTLY_TEMPLATE)
    st.plotly_chart(fig_hhi, use_container_width=True)

with hhi_col2:
    st.markdown(f"""
    <div style="background:{CARD_BG};border:1px solid {CARD_BORDER};border-radius:8px;
                padding:16px;margin-top:20px;">
        <div style="font-size:1.1em;font-weight:600;color:{CARD_TEXT};margin-bottom:8px;">
            {reg_label}</div>
        <div style="font-size:0.85em;color:{CARD_TEXT_MUTED};">
            <b>HHI Thresholds:</b><br>
            &lt; 1,500 = Unconcentrated<br>
            1,500 - 2,500 = Moderate<br>
            &gt; 2,500 = Highly concentrated<br><br>
            Based on catalog size proxy, not actual market share.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Hypothetical merger for academic analysis. Data is a snapshot (mid-2023). "
    "Enrichment data: IMDb datasets, Wikidata, MovieLens 20M, TMDB API. "
    "No section overclaims financial or subscriber data."
)
