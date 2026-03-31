"""Page 5: Strategic Insights — Merger business intelligence.

9 sections:
  1. Merger Value Dashboard
  2. Content Quality Ladder
  3. Prestige Index
  4. ROI Proxy
  5. Content Overlap Analysis
  6. Gap Analysis with Decision Trace
  7. Competitive Positioning
  8. Alternative Merger Scenarios
  9. Catalog Concentration Index
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
)
from src.data.loaders import load_enriched_titles
from src.analysis.strategic import (
    build_merged_entity,
    compute_acquisition_shortlist,
    compute_alternative_scenario,
    compute_best_alternative_scenario,
    compute_competitive_positioning,
    compute_gap_analysis,
    compute_market_simulation,
    compute_merger_kpis,
    compute_overlap_analysis,
    compute_overlap_heatmap,
    compute_quality_distribution,
    compute_temporal_momentum,
)
from src.ui.badges import (
    page_header_html,
    platform_badge_html,
    section_header_html,
    styled_banner_html,
    styled_metric_card_html,
)
from src.ui.session import init_session_state


# ─── Page-level helpers ────────────────────────────────────────────────────────

def _fmt_genre(g):
    if isinstance(g, str):
        return g.strip().title()
    return str(g).title()


def _confidence_badge(conf_val):
    if isinstance(conf_val, (int, float)):
        if conf_val >= 0.7:
            label, color = "High", "#2ECC71"
        elif conf_val >= 0.4:
            label, color = "Medium", "#F39C12"
        else:
            label, color = "Low", "#888"
    else:
        s = str(conf_val).lower()
        if s == "high":
            label, color = "High", "#2ECC71"
        elif s == "medium":
            label, color = "Medium", "#F39C12"
        else:
            label, color = "Low", "#888"
    return (
        f'<span style="background:rgba(0,0,0,0.3);color:{color};'
        f'padding:1px 7px;border-radius:8px;font-size:0.78em;'
        f'border:1px solid {color};font-weight:600;">{label}</span>'
    )


def _rank_badge(rank_str):
    """Convert '#1' etc. to a colored pill badge."""
    try:
        n = int(str(rank_str).replace("#", ""))
    except (ValueError, TypeError):
        return str(rank_str)
    color = {1: "#FFD700", 2: "#C0C0C0", 3: "#CD7F32"}.get(n, CARD_TEXT_MUTED)
    bg = {
        1: "rgba(255,215,0,0.15)",
        2: "rgba(192,192,192,0.10)",
        3: "rgba(205,127,50,0.10)",
    }.get(n, "rgba(136,136,136,0.08)")
    trophy = "🏆 " if n == 1 else ""
    return (
        f'<span style="background:{bg};color:{color};padding:1px 7px;'
        f'border-radius:8px;font-size:0.8em;font-weight:700;border:1px solid {color};">'
        f'{trophy}{rank_str}</span>'
    )


def _imdb_color(val_str):
    try:
        v = float(val_str)
    except (ValueError, TypeError):
        return CARD_TEXT_MUTED
    if v >= 7.0:
        return "#2ECC71"
    elif v >= 6.5:
        return "#F39C12"
    return CARD_TEXT_MUTED


def _strategic_profile(pa, pb):
    pair = frozenset([pa, pb])
    profiles = {
        frozenset(["netflix", "disney"]): "Volume + Franchise play: Netflix catalog depth plus Disney's branded IP portfolio",
        frozenset(["netflix", "prime"]): "Pure Volume play: dominant catalog scale with broad but uneven quality distribution",
        frozenset(["netflix", "paramount"]): "Volume + Sports play: breadth catalog with live sports and theatrical library",
        frozenset(["netflix", "appletv"]): "Volume + Prestige play: Netflix scale elevated by Apple's quality-first originals",
        frozenset(["max", "disney"]): "Prestige + Franchise play: premium adult drama combined with branded family IP",
        frozenset(["max", "prime"]): "Volume + Prestige play: broad catalog with Max's award-winning premium content",
        frozenset(["max", "paramount"]): "Prestige + Sports play: premium drama plus live sports differentiation",
        frozenset(["disney", "prime"]): "Family + Volume play: Disney's IP depth amplified by Prime's catalog scale",
        frozenset(["disney", "paramount"]): "Franchise + Sports play: branded IP with live sports rights portfolio",
        frozenset(["prime", "paramount"]): "Volume + Sports play: largest combined catalog with live sports differentiation",
        frozenset(["netflix", "max"]): "Volume + Prestige play: largest catalog with highest award density",
    }
    return profiles.get(pair, "Diversified streaming play across multiple content verticals")


# ─── Dimension text dicts for trade-off analysis ───────────────────────────────

_DIM_GAIN_TEXT = {
    "Scale": "Larger combined catalog — more content breadth across genres and release years",
    "Quality": "Higher average IMDb — stronger per-title quality positioning",
    "Prestige": "More award-winning titles per 1K — stronger critical perception",
    "Diversity": "Broader genre distribution — reduces content monoculture risk",
    "International": "More international content — better positioned for global subscriber growth",
    "Franchise": "Deeper franchise catalog — stronger IP-driven engagement and retention",
}
_DIM_LOSS_TEXT = {
    "Scale": "Smaller combined catalog — less content breadth",
    "Quality": "Lower average IMDb — weaker per-title quality positioning",
    "Prestige": "Fewer award-winning titles per 1K — weaker critical perception",
    "Diversity": "Narrower genre distribution — higher content concentration risk",
    "International": "Less international content — more exposure to single-market subscriber risk",
    "Franchise": "Shallower franchise depth — less IP-driven subscriber engagement",
}

# ─── Competitor-specific context ───────────────────────────────────────────────

_COMP_CONTEXT = {
    "disney": "Disney+ leads in family content and franchise IP — the merger should prioritize original family and animation co-productions to compete in this space.",
    "paramount": "Paramount+ is strong in live sports and news content — the merger should focus on documentary and premium drama to defend prestige positioning rather than competing directly in sports.",
    "prime": "Prime Video's advantage is sheer volume and international content breadth — the merged entity should focus on quality over quantity and prioritize original prestige content to differentiate.",
    "appletv": "Apple TV+ is a small but high-quality catalog — the merged entity already leads significantly in volume; the priority is maintaining quality benchmarks per-title to match Apple's prestige.",
}

_COMP_CLOSING = {
    "disney": " Marvel and Star Wars franchise depth give Disney+ an unmatched IP advantage in blockbuster sequels — catalog acquisitions alone cannot replicate this franchise ecosystem.",
    "paramount": " Paramount+'s live sports rights (NFL, UEFA) and CBS library represent assets the merged entity cannot acquire through catalog deals; original sports documentary production is the nearest lever.",
    "prime": " Amazon Studios' commitment to international production (India, UK, Japan) means Prime Video's global depth grows faster than catalog size alone — co-productions with local studios are the strategic counter.",
    "appletv": " Apple TV+'s originals-only strategy — with no catalog diluting quality — means every title is intentional; the merged entity's brand strategy should clearly segment its own prestige originals tier to compete on the same perception.",
}

# ─── App setup ─────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Strategic Insights", page_icon="📊", layout="wide")
init_session_state()

st.markdown(
    page_header_html(
        "Strategic Insights",
        "Merger business intelligence with decision traces and evidence-based recommendations.",
    ),
    unsafe_allow_html=True,
)

titles = load_enriched_titles()
_MERGED_COLOR = PLATFORMS["merged"]["color"]

# ─── Section 1: Merger Value Dashboard ────────────────────────────────────────
st.markdown(
    section_header_html(
        "Merger Value Dashboard",
        "Headline metrics quantifying the strategic value of the Netflix + Max merger.",
    ),
    unsafe_allow_html=True,
)

kpis = compute_merger_kpis(titles)

cols = st.columns(len(kpis))
for i, (key, kpi) in enumerate(kpis.items()):
    with cols[i]:
        st.markdown(
            styled_metric_card_html(
                kpi["label"], kpi["value"],
                help_text=kpi["detail"],
                accent_color=_MERGED_COLOR,
            ),
            unsafe_allow_html=True,
        )

try:
    merged_entity = build_merged_entity(titles)
    merged_size = len(merged_entity)
    comp_sizes = {p: len(titles[titles["platform"] == p]) for p in ALL_PLATFORMS if p not in MERGED_PLATFORMS}
    avg_competitor = int(np.mean(list(comp_sizes.values()))) if comp_sizes else 1
    size_ratio = merged_size / max(avg_competitor, 1)
    overlap_val = kpis.get("overlap_rate", {}).get("value", "N/A")
    prestige_kpi = kpis.get("prestige_lift", {})
    prestige_val = prestige_kpi.get("value") if prestige_kpi else None
    prestige_sentence = (
        f" A <b>{prestige_val}</b> jump in award-winning content marks a decisive prestige advantage."
        if prestige_val else ""
    )
    summary = (
        f"With <b>{merged_size:,} titles</b> — {size_ratio:.1f}x the average competitor — "
        f"the merged entity holds one of the largest catalogs in streaming. "
        f"A <b>{overlap_val}</b> content overlap between Netflix and Max signals an immediate "
        f"curation opportunity to consolidate and focus resources on underserved genres."
        f"{prestige_sentence}"
    )
    st.markdown(styled_banner_html("📊", summary), unsafe_allow_html=True)
except Exception:
    pass

st.divider()

# ─── Section 2: Content Quality Ladder ────────────────────────────────────────
st.markdown(
    section_header_html(
        "Content Quality Ladder",
        "How does the merger shift the IMDb score distribution? Ridge plot, quality thresholds, and platform momentum.",
    ),
    unsafe_allow_html=True,
)

try:
    qdist = compute_quality_distribution(titles)

    # Ridge plot
    display_order = ["merged", "netflix", "max", "disney", "prime", "paramount", "appletv"]
    valid_keys = [k for k in display_order if k in qdist and qdist[k]["scores"]]
    spacing = 1.2
    bin_edges = np.arange(1.0, 10.5, 0.25)
    midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig_ridge = go.Figure()
    for i, key in enumerate(reversed(valid_keys)):
        scores = qdist[key]["scores"]
        counts, _ = np.histogram(scores, bins=bin_edges, density=True)
        scale = 1.5 if key == "merged" else 1.0
        y_offset = i * spacing
        name = PLATFORMS.get(key, {}).get("name", key)
        color = PLATFORMS.get(key, {}).get("color", "#555")
        r_int = int(color[1:3], 16)
        g_int = int(color[3:5], 16)
        b_int = int(color[5:7], 16)
        alpha = 0.55 if key == "merged" else 0.35

        y_top = counts * scale + y_offset
        y_base = np.full(len(counts), y_offset)
        x_poly = np.concatenate([midpoints, midpoints[::-1]])
        y_poly = np.concatenate([y_top, y_base[::-1]])

        fig_ridge.add_trace(go.Scatter(
            x=x_poly, y=y_poly, fill="toself",
            fillcolor=f"rgba({r_int},{g_int},{b_int},{alpha})",
            line=dict(color=color, width=2.5 if key == "merged" else 1.5),
            name=name, mode="lines",
            hovertemplate=f"<b>{name}</b><br>IMDb: %{{x:.1f}}<extra></extra>",
        ))
        label_y = y_offset + counts[:8].mean() * scale * 0.4 + spacing * 0.1
        fig_ridge.add_annotation(
            x=1.5, y=label_y, text=f"<b>{name}</b>",
            showarrow=False, xanchor="left",
            font=dict(color=color, size=13 if key == "merged" else 11),
        )

    fig_ridge.add_vline(x=7.0, line_dash="dot", line_color="#aaa",
                        annotation_text="Good (7.0)", annotation_position="top right",
                        annotation_font_color="#aaa")
    fig_ridge.add_vline(x=8.0, line_dash="dot", line_color="#FFD700",
                        annotation_text="Excellent (8.0)", annotation_position="top right",
                        annotation_font_color="#FFD700")
    fig_ridge.update_layout(
        title="IMDb Score Distribution by Platform (Ridge Plot — Netflix+Max highlighted)",
        xaxis_title="IMDb Score", height=520,
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        template=PLOTLY_TEMPLATE, showlegend=False,
    )
    st.plotly_chart(fig_ridge, use_container_width=True)

    # Threshold table as styled HTML
    threshold_rows = []
    for key in ["merged", "netflix", "max", "disney", "prime", "paramount", "appletv"]:
        if key not in qdist:
            continue
        d = qdist[key]
        threshold_rows.append({
            "Platform": PLATFORMS.get(key, {}).get("name", key),
            "Catalog Size": f"{d['n']:,}",
            "% ≥ 7.0 (Good)": d["above_7"],
            "% ≥ 8.0 (Excellent)": d["above_8"],
            "Avg IMDb": d["avg"] if d["avg"] else 0.0,
        })

    df_thresh = pd.DataFrame(threshold_rows)
    df_thresh["Rank (7.0)"] = df_thresh["% ≥ 7.0 (Good)"].rank(ascending=False, method="min").astype(int).apply(lambda r: f"#{r}")
    df_thresh["Rank (8.0)"] = df_thresh["% ≥ 8.0 (Excellent)"].rank(ascending=False, method="min").astype(int).apply(lambda r: f"#{r}")
    df_thresh["Rank (Avg)"] = df_thresh["Avg IMDb"].rank(ascending=False, method="min").astype(int).apply(lambda r: f"#{r}")
    df_thresh["% ≥ 7.0 (Good)"] = df_thresh["% ≥ 7.0 (Good)"].apply(lambda v: f"{v:.0%}")
    df_thresh["% ≥ 8.0 (Excellent)"] = df_thresh["% ≥ 8.0 (Excellent)"].apply(lambda v: f"{v:.0%}")
    df_thresh["Avg IMDb"] = df_thresh["Avg IMDb"].apply(lambda v: f"{v:.2f}" if v else "N/A")

    header_html = (
        f'<tr style="border-bottom:2px solid {CARD_BORDER};color:{CARD_TEXT_MUTED};'
        f'font-size:0.78em;text-transform:uppercase;letter-spacing:0.04em;">'
        f'<th style="padding:6px 10px;text-align:left;">Platform</th>'
        f'<th style="padding:6px 10px;text-align:right;">Catalog Size</th>'
        f'<th style="padding:6px 10px;text-align:center;">% ≥ 7.0</th>'
        f'<th style="padding:6px 10px;text-align:center;">Rank</th>'
        f'<th style="padding:6px 10px;text-align:center;">% ≥ 8.0</th>'
        f'<th style="padding:6px 10px;text-align:center;">Rank</th>'
        f'<th style="padding:6px 10px;text-align:center;">Avg IMDb</th>'
        f'<th style="padding:6px 10px;text-align:center;">Rank</th>'
        f'</tr>'
    )
    rows_html = ""
    for i, row in df_thresh.iterrows():
        is_merged = row["Platform"] == "Netflix + Max"
        row_bg = f"rgba(0,180,166,0.05)" if is_merged else ("rgba(255,255,255,0.02)" if i % 2 == 0 else "transparent")
        border_left = f"border-left:3px solid {_MERGED_COLOR};" if is_merged else ""
        fw = "font-weight:700;" if is_merged else ""
        ic = _imdb_color(row["Avg IMDb"])
        rows_html += (
            f'<tr style="border-bottom:1px solid {CARD_BORDER};background:{row_bg};{border_left}{fw}">'
            f'<td style="padding:6px 10px;">{row["Platform"]}</td>'
            f'<td style="padding:6px 10px;text-align:right;color:{CARD_TEXT_MUTED};">{row["Catalog Size"]}</td>'
            f'<td style="padding:6px 10px;text-align:center;">{row["% ≥ 7.0 (Good)"]}</td>'
            f'<td style="padding:6px 10px;text-align:center;">{_rank_badge(row["Rank (7.0)"])}</td>'
            f'<td style="padding:6px 10px;text-align:center;">{row["% ≥ 8.0 (Excellent)"]}</td>'
            f'<td style="padding:6px 10px;text-align:center;">{_rank_badge(row["Rank (8.0)"])}</td>'
            f'<td style="padding:6px 10px;text-align:center;color:{ic};font-weight:600;">{row["Avg IMDb"]}</td>'
            f'<td style="padding:6px 10px;text-align:center;">{_rank_badge(row["Rank (Avg)"])}</td>'
            f'</tr>'
        )
    st.markdown(
        f'<table style="width:100%;border-collapse:collapse;font-size:0.84em;color:{CARD_TEXT};">'
        f'<thead>{header_html}</thead><tbody>{rows_html}</tbody></table>',
        unsafe_allow_html=True,
    )

    with st.expander("Content Quality by Era — Platform Momentum Over Decades"):
        momentum = compute_temporal_momentum(titles)
        if not momentum.empty:
            momentum["_decade_num"] = (
                momentum["decade"].astype(str).str.extract(r"(\d{4})")[0].astype(float)
            )
            sorted_decade_nums = sorted(momentum["_decade_num"].dropna().unique())
            decade_labels = [f"{int(d)}s" for d in sorted_decade_nums]

            all_momentum_keys = ALL_PLATFORMS + ["merged"]
            avail_map = {
                k: PLATFORMS.get(k, {}).get("name", k)
                for k in all_momentum_keys
                if not momentum[momentum["platform"] == k].empty
            }
            default_sel = list(avail_map.keys())
            selected_mom = st.multiselect(
                "Show platforms:", options=list(avail_map.keys()),
                default=default_sel,
                format_func=lambda x: avail_map.get(x, x),
                key="momentum_sel",
            )
            if "merged" not in selected_mom:
                selected_mom = ["merged"] + selected_mom

            fig_mom = go.Figure()
            for key in selected_mom:
                p_data = momentum[momentum["platform"] == key].copy().sort_values("_decade_num")
                if p_data.empty:
                    continue
                name = PLATFORMS.get(key, {}).get("name", key)
                color = PLATFORMS.get(key, {}).get("color", "#555")
                lw = 3 if key == "merged" else 1.5

                if key == "merged":
                    fig_mom.add_trace(go.Scatter(
                        x=p_data["decade"], y=p_data["avg_imdb"],
                        mode="lines", line=dict(color=color, width=10),
                        opacity=0.15, showlegend=False, hoverinfo="skip",
                    ))

                fig_mom.add_trace(go.Scatter(
                    x=p_data["decade"], y=p_data["avg_imdb"],
                    name=name, mode="lines+markers",
                    line=dict(color=color, width=lw),
                    marker=dict(size=8 if key == "merged" else 6),
                    customdata=p_data["title_count"].values,
                    hovertemplate=(
                        f"<b>{name}</b><br>Decade: %{{x}}<br>"
                        "Avg IMDb: %{y:.2f}<br>Titles: %{customdata}<extra></extra>"
                    ),
                ))

            fig_mom.add_hline(y=7.0, line_dash="dot", line_color="#555",
                              annotation_text="Quality threshold (7.0)")
            fig_mom.update_layout(
                title="Average IMDb Score by Release Decade",
                xaxis_title="Decade", yaxis_title="Avg IMDb",
                template=PLOTLY_TEMPLATE, height=320, yaxis=dict(range=[5.5, 9.0]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            fig_mom.update_xaxes(categoryorder="array", categoryarray=decade_labels)
            st.plotly_chart(fig_mom, use_container_width=True)
            st.caption("Only decades with ≥10 titles shown. Merged entity includes deduplicated Netflix + Max.")

except Exception as e:
    st.warning(f"Content Quality Ladder could not be computed: {e}")

st.divider()

# ─── Section 3: Prestige Index ─────────────────────────────────────────────────
st.markdown(
    section_header_html(
        "Prestige Index",
        "Award wins per 1,000 titles by platform and genre — measures prestige content concentration.",
    ),
    unsafe_allow_html=True,
)

prestige_path = PRECOMPUTED_DIR / "strategic_analysis" / "prestige_index.parquet"
if prestige_path.exists():
    prestige = pd.read_parquet(prestige_path)
    award_coverage = titles["award_wins"].notna().mean() if "award_wins" in titles.columns else 0

    if award_coverage >= WIKIDATA_MIN_COVERAGE:
        st.markdown(styled_banner_html("ℹ️", f"Prestige Index (Wikidata, {award_coverage:.0%} coverage)"), unsafe_allow_html=True)

        platform_prestige = (
            prestige.groupby("platform")
            .agg({"prestige_per_1k": "mean", "award_wins": "sum", "title_count": "sum"})
            .reset_index()
            .sort_values("prestige_per_1k", ascending=False)
        )
        platform_prestige["name"] = platform_prestige["platform"].map(
            lambda p: PLATFORMS.get(p, {}).get("name", p)
        )
        color_map = {p: PLATFORMS[p]["color"] for p in PLATFORMS}

        fig_bar = px.bar(
            platform_prestige, x="name", y="prestige_per_1k",
            color="platform", color_discrete_map=color_map,
            title="Prestige Score by Platform (Award Wins per 1,000 Titles)",
            template=PLOTLY_TEMPLATE,
        )
        fig_bar.update_layout(showlegend=False, xaxis_title="", yaxis_title="Prestige per 1K titles", xaxis_tickangle=-30)
        st.plotly_chart(fig_bar, use_container_width=True)

        heatmap_data = prestige.pivot_table(index="genre", columns="platform", values="prestige_per_1k", fill_value=0)
        heatmap_data.index = [_fmt_genre(g) for g in heatmap_data.index]
        heatmap_data.columns = [PLATFORMS.get(p, {}).get("name", p) for p in heatmap_data.columns]

        fig_heat = px.imshow(
            heatmap_data, aspect="auto",
            color_continuous_scale=[[0, "#1E1E2E"], [1, "#00B4A6"]],
            title="Genre × Platform Prestige Intensity", template=PLOTLY_TEMPLATE,
        )
        fig_heat.update_layout(height=max(500, 38 * len(heatmap_data)), coloraxis_colorbar_title="Prestige/1K", margin=dict(r=130))
        fig_heat.update_xaxes(tickangle=30)
        st.plotly_chart(fig_heat, use_container_width=True)

        try:
            top_row = platform_prestige.iloc[0]
            genre_wins = prestige.groupby("genre")["award_wins"].sum().sort_values(ascending=False)
            top_genre = _fmt_genre(genre_wins.index[0]) if len(genre_wins) > 0 else "Drama"
            top_genre_pct = genre_wins.iloc[0] / max(genre_wins.sum(), 1)
            merged_wins = prestige[prestige["platform"].isin(["merged", "netflix", "max"])]["award_wins"].sum()
            disney_wins = prestige[prestige["platform"] == "disney"]["award_wins"].sum()
            prime_wins = prestige[prestige["platform"] == "prime"]["award_wins"].sum()
            combined = disney_wins + prime_wins
            b1 = f"<b>{top_row['name']}</b> leads all platforms in prestige density — <b>{top_row['prestige_per_1k']:.1f}</b> award wins per 1,000 titles."
            b2 = f"<b>{top_genre}</b> content drives the majority of prestige concentration, accounting for <b>{top_genre_pct:.0%}</b> of total award wins."
            b3 = (
                f"Post-merger, Netflix+Max holds <b>{int(merged_wins):,}</b> total award wins — "
                f"{'more than' if merged_wins > combined else 'comparable to'} Disney+ and Prime Video combined (<b>{int(combined):,}</b>)."
            )
            bullets = f"<ul style='margin:4px 0;padding-left:18px;'><li style='margin-bottom:5px;'>{b1}</li><li style='margin-bottom:5px;'>{b2}</li><li>{b3}</li></ul>"
            st.markdown(styled_banner_html("💡", bullets), unsafe_allow_html=True)
        except Exception:
            pass
    else:
        st.info(f"Prestige Index requires ≥{WIKIDATA_MIN_COVERAGE:.0%} award data coverage. Current: {award_coverage:.0%}.")
else:
    st.info("Prestige index data not found. Run `scripts/11_precompute_strategic.py` first.")

st.divider()

# ─── Section 4: ROI Proxy ──────────────────────────────────────────────────────
st.markdown(
    section_header_html(
        "ROI Proxy",
        "Box office / production budget ratio by platform — which catalogs recover their theatrical investment?",
    ),
    unsafe_allow_html=True,
)

if "budget_usd" in titles.columns and "box_office_usd" in titles.columns:
    roi_data = titles[
        titles["budget_usd"].notna() & (titles["budget_usd"] > 0) &
        titles["box_office_usd"].notna() & (titles["box_office_usd"] > 0)
    ].copy()
    if len(roi_data) > 20:
        roi_data["roi_proxy"] = roi_data["box_office_usd"] / roi_data["budget_usd"]
        platform_roi = (
            roi_data.groupby("platform")["roi_proxy"]
            .agg(["median", "count"])
            .reset_index()
            .sort_values("median")
        )
        platform_roi["Platform"] = platform_roi["platform"].map(
            lambda p: PLATFORMS.get(p, {}).get("name", p)
        )

        # Add Netflix+Max merged bar
        merged_roi_data = roi_data[roi_data["platform"].isin(["netflix", "max"])].copy()
        blended_roi_val = merged_roi_data["roi_proxy"].median() if len(merged_roi_data) > 0 else 0.0
        merged_row = pd.DataFrame([{
            "platform": "merged", "Platform": "Netflix + Max",
            "median": blended_roi_val, "count": len(merged_roi_data),
        }])
        platform_roi_with_merged = pd.concat([platform_roi, merged_row], ignore_index=True).sort_values("median")
        bar_colors = [PLATFORMS.get(p, {}).get("color", "#555") for p in platform_roi_with_merged["platform"]]

        fig_roi = go.Figure(go.Bar(
            y=platform_roi_with_merged["Platform"],
            x=platform_roi_with_merged["median"],
            orientation="h",
            marker_color=bar_colors,
            text=[f"{v:.2f}x" for v in platform_roi_with_merged["median"]],
            textposition="outside",
            cliponaxis=False,
        ))
        fig_roi.add_vline(x=1.0, line_dash="dot", line_color="#aaa",
                          annotation_text="Break-even (1.0x)", annotation_position="top right",
                          annotation_font_color="#aaa")

        appletv_name = PLATFORMS.get("appletv", {}).get("name", "Apple TV+")
        appletv_row = platform_roi_with_merged[platform_roi_with_merged["platform"] == "appletv"]
        if not appletv_row.empty:
            fig_roi.add_annotation(
                y=appletv_name, x=appletv_row["median"].iloc[0],
                text="Streaming-first — theatrical BO not primary metric",
                showarrow=True, arrowhead=2, ax=120, ay=0,
                font=dict(color=CARD_TEXT_MUTED, size=11), arrowcolor=CARD_TEXT_MUTED,
            )

        fig_roi.update_layout(
            title="Median Box Office ROI by Platform",
            xaxis_title="Median ROI (box office / budget)", yaxis_title="",
            template=PLOTLY_TEMPLATE,
            height=max(280, 55 * len(platform_roi_with_merged)),
            margin=dict(r=160),
        )
        st.plotly_chart(fig_roi, use_container_width=True)

        total_roi_titles = len(roi_data)
        total_titles = len(titles)
        paramount_row = platform_roi_with_merged[platform_roi_with_merged["platform"] == "paramount"]
        paramount_roi_val = paramount_row["median"].iloc[0] if not paramount_row.empty else None
        paramount_sentence = (
            f" Paramount+'s {paramount_roi_val:.2f}x median reflects its heavy reliance on "
            f"theatrically-released CBS and Paramount Pictures catalog titles such as "
            f"<i>Top Gun</i> and <i>Mission: Impossible</i> — a licensing advantage the merged "
            f"entity could partially replicate through targeted theatrical catalog acquisition."
            if paramount_roi_val else ""
        )
        st.markdown(
            styled_banner_html(
                "📈",
                f"The merged Netflix+Max entity carries a blended ROI of <b>{blended_roi_val:.2f}x</b> "
                f"across {total_roi_titles:,} theatrically-released titles — competitive with Disney+ "
                f"but below Paramount+'s theatrical-driven library. The merger's ROI story lies in "
                f"volume and awards prestige rather than box office returns.{paramount_sentence}",
            ),
            unsafe_allow_html=True,
        )
        st.caption(
            f"Coverage: {total_roi_titles:,} of ~{total_titles:,} titles "
            f"({total_roi_titles/max(total_titles,1):.0%}). "
            "Excludes streaming-only originals where no theatrical release occurred."
        )

        # ── ROI Explorer ──────────────────────────────────────────────────
        with st.expander("ROI Explorer — Drill into Individual Titles"):
            roi_plat_options = platform_roi["platform"].tolist()
            roi_plat = st.selectbox(
                "Select platform", roi_plat_options,
                format_func=lambda p: PLATFORMS.get(p, {}).get("name", p),
                key="roi_explorer_plat",
            )
            plat_roi_data = roi_data[roi_data["platform"] == roi_plat].copy()
            if len(plat_roi_data) > 0:
                def _budget_tier(b):
                    if b < 10e6: return "Low (<$10M)"
                    if b < 50e6: return "Mid ($10–50M)"
                    if b < 200e6: return "High ($50–200M)"
                    return "Blockbuster (>$200M)"

                plat_roi_data["Budget Tier"] = plat_roi_data["budget_usd"].apply(_budget_tier)
                plat_roi_data["genre_display"] = plat_roi_data["genres"].apply(
                    lambda g: _fmt_genre(g[0]) if isinstance(g, list) and len(g) > 0 else "Other"
                )

                all_genres_roi = sorted(plat_roi_data["genre_display"].dropna().unique())
                selected_genres_roi = st.multiselect(
                    "Filter by genre", all_genres_roi, default=all_genres_roi, key="roi_genre_filter"
                )
                filtered_roi = plat_roi_data[plat_roi_data["genre_display"].isin(selected_genres_roi)].copy()

                search_title = st.text_input(
                    "Highlight a title", placeholder="Type title name...", key="roi_title_search"
                )

                tier_map = {"Low (<$10M)": 0, "Mid ($10–50M)": 1, "High ($50–200M)": 2, "Blockbuster (>$200M)": 3}
                filtered_roi["tier_num"] = filtered_roi["Budget Tier"].map(tier_map).fillna(0)
                np.random.seed(42)
                filtered_roi = filtered_roi.reset_index(drop=True)
                filtered_roi["x_jitter"] = filtered_roi["tier_num"] + np.random.uniform(-0.35, 0.35, size=len(filtered_roi))

                if "imdb_votes" in filtered_roi.columns:
                    filtered_roi["dot_size"] = (
                        filtered_roi["imdb_votes"].fillna(0).clip(upper=500_000) / 500_000 * 25 + 6
                    ).round(1)
                else:
                    filtered_roi["dot_size"] = 8.0

                genre_color_map = {g: px.colors.qualitative.Plotly[i % 10] for i, g in enumerate(all_genres_roi)}

                if search_title.strip():
                    mask = filtered_roi["title"].str.contains(search_title.strip(), case=False, na=False)
                    df_norm = filtered_roi[~mask]
                    df_hl = filtered_roi[mask]
                else:
                    df_norm = filtered_roi
                    df_hl = pd.DataFrame()

                hover_cols = ["title", "release_year", "imdb_score", "budget_usd", "box_office_usd", "roi_proxy"]
                missing = [c for c in hover_cols if c not in df_norm.columns]
                for mc in missing:
                    df_norm[mc] = None
                    if not df_hl.empty:
                        df_hl[mc] = None

                fig_detail = go.Figure()
                fig_detail.add_trace(go.Scatter(
                    x=df_norm["x_jitter"], y=df_norm["roi_proxy"],
                    mode="markers",
                    marker=dict(
                        size=df_norm["dot_size"],
                        color=[genre_color_map.get(g, "#555") for g in df_norm["genre_display"]],
                        opacity=0.75, line=dict(width=0),
                    ),
                    customdata=df_norm[hover_cols].values,
                    hovertemplate=(
                        "<b>%{customdata[0]}</b> (%{customdata[1]:.0f})<br>"
                        "IMDb: %{customdata[2]:.1f}<br>"
                        "Budget: $%{customdata[3]:,.0f}<br>"
                        "Box Office: $%{customdata[4]:,.0f}<br>"
                        "ROI: %{customdata[5]:.2f}x<extra></extra>"
                    ),
                    name="Titles",
                ))

                if not df_hl.empty:
                    for mc in missing:
                        df_hl[mc] = None
                    fig_detail.add_trace(go.Scatter(
                        x=df_hl["x_jitter"], y=df_hl["roi_proxy"],
                        mode="markers+text",
                        marker=dict(size=16, color="#FFD700", line=dict(color="#fff", width=2)),
                        text=df_hl["title"],
                        textposition="top center",
                        textfont=dict(color="#FFD700", size=11),
                        customdata=df_hl[hover_cols].values,
                        hovertemplate=(
                            "<b>%{customdata[0]}</b> (%{customdata[1]:.0f})<br>"
                            "IMDb: %{customdata[2]:.1f}<br>"
                            "ROI: %{customdata[5]:.2f}x<extra></extra>"
                        ),
                        name="Highlighted", showlegend=False,
                    ))

                fig_detail.add_hline(y=1.0, line_dash="dot", line_color="#aaa", annotation_text="Break-even")
                fig_detail.update_layout(
                    title=f"{PLATFORMS.get(roi_plat, {}).get('name', roi_plat)} — Individual Title ROI",
                    xaxis=dict(
                        tickvals=[0, 1, 2, 3],
                        ticktext=["Low (<$10M)", "Mid ($10–50M)", "High ($50–200M)", "Blockbuster (>$200M)"],
                        title="Budget Tier",
                    ),
                    yaxis_title="ROI (box office / budget)",
                    template=PLOTLY_TEMPLATE, height=450,
                )
                st.plotly_chart(fig_detail, use_container_width=True)
                st.caption(
                    f"Each dot is one title with Wikidata budget and box office data "
                    f"({len(filtered_roi)} titles shown). "
                    "Only titles with both fields populated are shown. Dot size = audience (IMDb votes)."
                )
            else:
                st.info("No budget/box office data available for this platform.")
    else:
        st.info("Not enough ROI data (requires budget + box office for at least 20 titles).")
else:
    st.info("ROI proxy requires `budget_usd` and `box_office_usd` columns. Run the Wikidata pipeline.")

st.divider()

# ─── Section 5: Content Overlap Analysis ──────────────────────────────────────
st.markdown(
    section_header_html(
        "Content Overlap Analysis",
        "Mapping where Netflix and Max duplicate vs complement each other — and what to cut first.",
    ),
    unsafe_allow_html=True,
)

overlap = compute_overlap_analysis(titles)
if not overlap.empty:
    heatmap_overlap = compute_overlap_heatmap(overlap)

    total_combos = len(overlap)
    complementary = len(overlap[overlap["confidence"] < 0.3])
    complementarity_pct = complementary / max(total_combos, 1)
    high_overlap_count = len(overlap[overlap["confidence"] > 0.5])

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(styled_metric_card_html("Genre-Type Combinations", f"{total_combos}", accent_color=_MERGED_COLOR), unsafe_allow_html=True)
    with m2:
        st.markdown(styled_metric_card_html("High Overlap Pairs", f"{high_overlap_count}", accent_color=_MERGED_COLOR), unsafe_allow_html=True)
    with m3:
        st.markdown(styled_metric_card_html("Complementarity Score", f"{complementarity_pct:.0%}", accent_color=_MERGED_COLOR), unsafe_allow_html=True)

    if not heatmap_overlap.empty:
        plot_data = heatmap_overlap.head(15).copy()
        plot_data["netflix_exclusive"] = (plot_data["netflix_count"] - plot_data["overlap"]).clip(lower=0)
        plot_data["max_exclusive"] = (plot_data["max_count"] - plot_data["overlap"]).clip(lower=0)
        plot_data["genre_label"] = plot_data["genre"].apply(_fmt_genre)

        fig_overlap = go.Figure()
        fig_overlap.add_trace(go.Bar(name="Netflix Exclusive", x=plot_data["genre_label"], y=plot_data["netflix_exclusive"], marker_color=PLATFORMS["netflix"]["color"]))
        fig_overlap.add_trace(go.Bar(name="Shared", x=plot_data["genre_label"], y=plot_data["overlap"], marker_color="#FFD700"))
        fig_overlap.add_trace(go.Bar(name="Max Exclusive", x=plot_data["genre_label"], y=plot_data["max_exclusive"], marker_color=PLATFORMS["max"]["color"]))
        fig_overlap.update_layout(barmode="stack", title="Content Distribution: Netflix vs Max by Genre (Top 15)", template=PLOTLY_TEMPLATE, height=420, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_overlap, use_container_width=True)

    high_overlap = overlap[overlap["confidence"] > 0.5]
    low_overlap = overlap[overlap["confidence"] < 0.2]
    top_overlapping = [_fmt_genre(g) for g in high_overlap.nlargest(3, "overlap")["genre"].tolist()] if not high_overlap.empty else []
    top_complementary = [_fmt_genre(g) for g in low_overlap.nlargest(3, "netflix_count")["genre"].tolist()] if not low_overlap.empty else []

    if len(high_overlap) > len(low_overlap):
        genres_str = ", ".join(top_overlapping[:3]) if top_overlapping else "multiple genres"
        overall_pct = len(high_overlap) / max(total_combos, 1)
        callout = (
            f"<b>High Overlap in {genres_str} — Strong Curation Signal</b><br>"
            f"<span style='font-size:0.9em;'>{overall_pct:.0%} of genre-type pairs show significant "
            f"duplication. The merged entity can consolidate to highest-rated titles in each overlapping "
            f"genre and redirect resources to underserved categories.</span>"
        )
        st.markdown(styled_banner_html("✂️", callout, bg="rgba(46,204,113,0.08)", border_color="#2ecc71"), unsafe_allow_html=True)
    else:
        comp_str = ", ".join(top_complementary[:3]) if top_complementary else "multiple genres"
        callout = (
            f"<b>Largely Complementary Catalogs — Low Redundancy Risk</b><br>"
            f"<span style='font-size:0.9em;'>{complementarity_pct:.0%} of genre-type combinations "
            f"show low overlap. Max adds depth in {comp_str}, maximizing genre breadth.</span>"
        )
        st.markdown(styled_banner_html("✅", callout, bg="rgba(46,204,113,0.08)", border_color="#2ecc71"), unsafe_allow_html=True)

    with st.expander("Curation Targets — Low-Rated Titles in High-Overlap Genres"):
        if not heatmap_overlap.empty:
            high_overlap_genres = heatmap_overlap[heatmap_overlap["overlap_pct"] > 0.3]["genre"].tolist()
            if high_overlap_genres:
                nf_max = titles[titles["platform"].isin(["netflix", "max"])].copy()
                nf_max_exp = nf_max.explode("genres")
                in_genre = nf_max_exp[nf_max_exp["genres"].isin(high_overlap_genres)].copy()
                in_genre = in_genre[in_genre["imdb_score"].notna()]
                if not in_genre.empty:
                    deduped = (
                        in_genre.groupby(["id", "title", "release_year", "platform", "imdb_score"])
                        .agg(genres_list=("genres", list)).reset_index()
                        .sort_values("imdb_score").head(20)
                    )
                    deduped["Platform"] = deduped["platform"].map(lambda p: PLATFORMS.get(p, {}).get("name", p))
                    deduped["Genres"] = deduped["genres_list"].apply(lambda gs: ", ".join(_fmt_genre(g) for g in sorted(set(gs))))
                    _RED_BADGE = '<span style="background:rgba(231,76,60,0.15);color:#e74c3c;padding:2px 8px;border-radius:8px;font-size:0.78em;font-weight:600;border:1px solid rgba(231,76,60,0.4);">Review for removal</span>'
                    rows_html = ""
                    for _, row in deduped.iterrows():
                        year_str = str(int(row["release_year"])) if pd.notna(row["release_year"]) else ""
                        imdb_str = f"{row['imdb_score']:.1f}" if pd.notna(row["imdb_score"]) else "N/A"
                        rows_html += (
                            f"<tr style='border-bottom:1px solid {CARD_BORDER};'>"
                            f"<td style='padding:5px 8px;'>{row['title']}</td>"
                            f"<td style='padding:5px 8px;color:{CARD_TEXT_MUTED};'>{year_str}</td>"
                            f"<td style='padding:5px 8px;'>{row['Platform']}</td>"
                            f"<td style='padding:5px 8px;color:{CARD_TEXT_MUTED};'>{row['Genres']}</td>"
                            f"<td style='padding:5px 8px;'>{imdb_str}</td>"
                            f"<td style='padding:5px 8px;'>{_RED_BADGE}</td></tr>"
                        )
                    st.markdown(
                        f'<table style="width:100%;border-collapse:collapse;font-size:0.84em;color:{CARD_TEXT};">'
                        f'<thead><tr style="border-bottom:2px solid {CARD_BORDER};color:{CARD_TEXT_MUTED};font-size:0.82em;text-transform:uppercase;letter-spacing:0.04em;">'
                        f'<th style="padding:5px 8px;text-align:left;">Title</th><th style="padding:5px 8px;text-align:left;">Year</th>'
                        f'<th style="padding:5px 8px;text-align:left;">Platform</th><th style="padding:5px 8px;text-align:left;">Genres</th>'
                        f'<th style="padding:5px 8px;text-align:left;">IMDb</th><th style="padding:5px 8px;text-align:left;">Recommendation</th>'
                        f'</tr></thead><tbody>{rows_html}</tbody></table>', unsafe_allow_html=True,
                    )
                    st.caption("Lowest-rated Netflix/Max titles in high-overlap genres. First candidates for post-merger catalog rationalization.")
                else:
                    st.info("No titles found in high-overlap genres with IMDb scores.")
            else:
                st.info("No high-overlap genres found above 30% overlap threshold.")

    with st.expander("Overlap Audit Table"):
        audit = overlap.sort_values("confidence", ascending=False).copy()
        audit["Genre"] = audit["genre"].apply(_fmt_genre)
        audit["Type"] = audit["type"].apply(lambda t: str(t).title() if pd.notna(t) else "")
        audit["confidence_pct"] = (audit["confidence"] * 100).round(1)
        st.dataframe(
            audit[["Genre", "Type", "netflix_count", "max_count", "overlap", "confidence_pct"]].rename(
                columns={"netflix_count": "Netflix Count", "max_count": "Max Count", "overlap": "Shared", "confidence_pct": "Overlap Confidence (%)"}
            ),
            hide_index=True,
            column_config={"Overlap Confidence (%)": st.column_config.ProgressColumn("Overlap Confidence (%)", min_value=0, max_value=100, format="%.1f%%")},
        )

st.divider()

# ─── Section 6: Gap Analysis ──────────────────────────────────────────────────
st.markdown(
    section_header_html(
        "Gap Analysis",
        "Content gaps with full decision traces — acquisition priorities and specific title recommendations.",
    ),
    unsafe_allow_html=True,
)

gap_col1, gap_col2 = st.columns(2)
with gap_col1:
    gap_perspective = st.selectbox(
        "Perspective", ["merged"] + ALL_PLATFORMS, key="gap_persp",
        format_func=lambda x: PLATFORMS.get(x, {}).get("name", x) if x != "merged" else "Merged (Netflix + Max)",
    )
with gap_col2:
    gap_competitor = st.selectbox(
        "Compare against", ["All Competitors"] + COMPETITOR_PLATFORMS, key="gap_comp",
        format_func=lambda x: PLATFORMS.get(x, {}).get("name", x) if x != "All Competitors" else x,
    )

try:
    gaps = compute_gap_analysis(
        titles, perspective=gap_perspective,
        competitor=gap_competitor if gap_competitor != "All Competitors" else None,
    )
except Exception as e:
    gaps = pd.DataFrame()
    st.warning(f"Gap analysis could not be computed: {e}")

if not gaps.empty:
    high_gaps = len(gaps[gaps["severity"] == "High"])
    med_gaps = len(gaps[gaps["severity"] == "Medium"])

    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown(styled_metric_card_html("Total Gaps Found", str(len(gaps)), accent_color=_MERGED_COLOR), unsafe_allow_html=True)
    with g2:
        st.markdown(styled_metric_card_html("High Priority", str(high_gaps), accent_color="#E74C3C"), unsafe_allow_html=True)
    with g3:
        st.markdown(styled_metric_card_html("Medium Priority", str(med_gaps), accent_color="#F39C12"), unsafe_allow_html=True)

    # Gap Priority Matrix
    try:
        scatter_rows = []
        for _, row in gaps.iterrows():
            lead_str = str(row.get("competitor_lead", ""))
            try:
                multiple = float(lead_str.split("has ")[-1].replace("x", "").strip()) if "has" in lead_str else 1.5
            except (ValueError, AttributeError):
                multiple = 1.5
            scatter_rows.append({
                "Genre": _fmt_genre(row.get("genre", "")),
                "Coverage (%)": float(row.get("coverage_pct") or 0),
                "Avg IMDb": float(row.get("quality_benchmark") or 6.5),
                "Priority": str(row.get("severity", "Low")),
                "Competitor Lead (x)": multiple,
            })
        if scatter_rows:
            scatter_df = pd.DataFrame(scatter_rows)

            # Competitor benchmark toggle (placed before fig)
            show_comp_bench = st.toggle("Show competitor benchmarks", value=False, key="gap_matrix_bench")

            fig_matrix = px.scatter(
                scatter_df, x="Coverage (%)", y="Avg IMDb",
                size="Competitor Lead (x)", color="Priority",
                color_discrete_map={"High": "#E74C3C", "Medium": "#F39C12", "Low": "#2ECC71"},
                text="Genre", title="Gap Priority Matrix — Size = Competitor Lead Multiple",
                template=PLOTLY_TEMPLATE, height=450, size_max=40,
            )
            fig_matrix.update_traces(textposition="top center", textfont_size=10)

            # Median-based crosshairs
            med_coverage = float(scatter_df["Coverage (%)"].median())
            med_imdb = float(scatter_df["Avg IMDb"].median())
            fig_matrix.add_hline(y=med_imdb, line_dash="dot", line_color="#555",
                                  annotation_text=f"Median IMDb ({med_imdb:.1f})")
            fig_matrix.add_vline(x=med_coverage, line_dash="dot", line_color="#555",
                                  annotation_text=f"Median Coverage ({med_coverage:.1f}%)")

            # Corner quadrant labels
            x_max = scatter_df["Coverage (%)"].max() * 1.1 if len(scatter_df) > 0 else 20
            y_max = scatter_df["Avg IMDb"].max() + 0.4 if len(scatter_df) > 0 else 9.0
            y_min = scatter_df["Avg IMDb"].min() - 0.4 if len(scatter_df) > 0 else 6.0
            x_min = -0.5
            for ax, ay, txt, anchor in [
                (x_min + 0.3, y_max - 0.05, "Priority Targets", "left"),
                (x_max - 0.3, y_max - 0.05, "Maintain", "right"),
                (x_min + 0.3, y_min + 0.05, "Selective", "left"),
                (x_max - 0.3, y_min + 0.05, "Curation Needed", "right"),
            ]:
                fig_matrix.add_annotation(
                    x=ax, y=ay, text=f"<i>{txt}</i>",
                    showarrow=False, font=dict(color=CARD_TEXT_MUTED, size=12), xanchor=anchor,
                )

            # Optional competitor benchmark overlay
            if show_comp_bench:
                comp_bench_rows = []
                for genre_key in scatter_df["Genre"]:
                    raw_genre = genre_key.lower()
                    comp_exp = titles[~titles["platform"].isin(["netflix", "max"])].explode("genres")
                    genre_data = comp_exp[comp_exp["genres"] == raw_genre]
                    if len(genre_data) > 0:
                        n_comp = len(titles[~titles["platform"].isin(["netflix", "max"])])
                        genre_coverage = len(genre_data) / max(n_comp, 1) * 100
                        genre_imdb = genre_data["imdb_score"].mean()
                        if pd.notna(genre_imdb):
                            comp_bench_rows.append({
                                "Coverage (%)": genre_coverage,
                                "Avg IMDb": genre_imdb,
                                "Genre": genre_key,
                            })
                if comp_bench_rows:
                    bench_df = pd.DataFrame(comp_bench_rows)
                    fig_matrix.add_trace(go.Scatter(
                        x=bench_df["Coverage (%)"], y=bench_df["Avg IMDb"],
                        mode="markers", name="Competitor Avg",
                        marker=dict(color="rgba(136,136,136,0.5)", size=10, symbol="x"),
                        text=bench_df["Genre"],
                        hovertemplate="<b>%{text}</b><br>Competitor avg<br>Coverage: %{x:.1f}%<br>IMDb: %{y:.2f}<extra></extra>",
                    ))

            fig_matrix.update_layout(xaxis_range=[x_min, x_max], yaxis_range=[y_min, y_max])
            st.plotly_chart(fig_matrix, use_container_width=True)
    except Exception:
        pass

    # Gap cards
    for _, gap in gaps.iterrows():
        try:
            severity = str(gap.get("severity", "Low"))
            genre = _fmt_genre(str(gap.get("genre", "Unknown")))
            sev_icon = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(severity, "")
            sev_border = {"High": "#E74C3C", "Medium": "#F39C12", "Low": "#2ECC71"}.get(severity, CARD_BORDER)
            sev_bg = {"High": "rgba(231,76,60,0.10)", "Medium": "rgba(243,156,18,0.10)", "Low": "rgba(46,204,113,0.06)"}.get(severity, "transparent")
            qv = gap.get("quality_benchmark")
            quality_str = f"{float(qv):.2f}" if pd.notna(qv) else "N/A"
            cv = gap.get("coverage_pct")
            coverage_str = f"{float(cv):.1f}%" if pd.notna(cv) else "N/A"
            leader_str = str(gap.get("competitor_lead") or "N/A")
            rec_str = str(gap.get("recommendation") or "N/A")
            demand_str = str(gap.get("audience_demand") or "Unknown")

            st.markdown(f"""
            <div style="background:{CARD_BG};border-left:4px solid {sev_border};border-radius:6px;padding:12px 16px;margin-bottom:8px;">
                <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">
                    <span style="font-size:1.05rem;font-weight:700;color:{CARD_TEXT};">{sev_icon} {genre}</span>
                    <div style="display:flex;align-items:center;gap:8px;">
                        <span style="background:{sev_bg};color:{sev_border};padding:2px 10px;border-radius:10px;font-size:0.78em;font-weight:600;border:1px solid {sev_border};">{severity} Priority</span>
                        <span style="color:{CARD_TEXT_MUTED};font-size:0.83em;">Coverage: <b style="color:{CARD_TEXT};">{coverage_str}</b></span>
                    </div>
                </div>
                <div style="display:flex;gap:20px;flex-wrap:wrap;font-size:0.83em;color:{CARD_TEXT_MUTED};margin-top:8px;">
                    <span>Avg IMDb: <b style="color:{CARD_TEXT};">{quality_str}</b></span>
                    <span>Leader: <b style="color:{CARD_TEXT};">{leader_str}</b></span>
                    <span>Audience Demand: <b style="color:{CARD_TEXT};">{demand_str}</b></span>
                </div>
                <div style="margin-top:8px;font-size:0.88em;color:{CARD_TEXT};border-top:1px solid {CARD_BORDER};padding-top:6px;">🎯 {rec_str}</div>
            </div>""", unsafe_allow_html=True)
        except Exception:
            continue

    # Full Acquisition Report — High/Medium only, row-by-row with View buttons + posters
    st.markdown(
        section_header_html(
            "Full Acquisition Report",
            "Real titles from competitor catalogs — High and Medium priority gaps only.",
        ),
        unsafe_allow_html=True,
    )

    try:
        shortlist = compute_acquisition_shortlist(titles, gaps)
        poster_map = {}
        if "poster_url" in titles.columns:
            poster_map = titles.dropna(subset=["poster_url"]).set_index("title")["poster_url"].to_dict()

        priority_gaps = gaps[gaps["severity"].isin(["High", "Medium"])].copy()
        priority_gaps["_sev"] = priority_gaps["severity"].map({"High": 0, "Medium": 1})
        priority_gaps = priority_gaps.sort_values(["_sev", "coverage_pct"])

        for _, gap in priority_gaps.iterrows():
            try:
                severity = str(gap.get("severity", "Medium"))
                genre = _fmt_genre(str(gap.get("genre", "Unknown")))
                genre_key = gap.get("genre")
                sev_border = {"High": "#E74C3C", "Medium": "#F39C12"}.get(severity, CARD_BORDER)

                with st.expander(f"{genre} — {severity} Priority"):
                    cv = gap.get("coverage_pct")
                    cv_str = f"{float(cv):.1f}%" if cv is not None and pd.notna(cv) else "N/A"
                    qv = gap.get("quality_benchmark")
                    leader_str = str(gap.get("competitor_lead") or "N/A")
                    demand_str = str(gap.get("audience_demand") or "Unknown")
                    rec_str = str(gap.get("recommendation") or "N/A")

                    st.markdown(
                        f'<div style="border-left:3px solid {sev_border};padding:8px 12px;background:rgba(0,0,0,0.15);border-radius:4px;margin-bottom:10px;">'
                        f'<div style="font-size:0.83em;color:{CARD_TEXT_MUTED};display:flex;gap:16px;flex-wrap:wrap;">'
                        f'<span>Coverage: <b style="color:{CARD_TEXT};">{cv_str}</b></span>'
                        f'<span>Avg IMDb: <b style="color:{CARD_TEXT};">{"%.2f" % float(qv) if pd.notna(qv) else "N/A"}</b></span>'
                        f'<span>Leader: <b style="color:{CARD_TEXT};">{leader_str}</b></span>'
                        f'<span>Demand: <b style="color:{CARD_TEXT};">{demand_str}</b></span>'
                        f'</div><div style="margin-top:6px;font-size:0.86em;color:{CARD_TEXT};">🎯 {rec_str}</div></div>',
                        unsafe_allow_html=True,
                    )

                    candidates = shortlist.get(genre_key, pd.DataFrame())
                    if not candidates.empty:
                        # Column header row
                        h_cols = st.columns([0.5, 2.8, 0.6, 1.0, 0.6, 0.7, 0.9])
                        for hc, hl in zip(h_cols, ["", "Title", "Year", "Platform", "IMDb", "Fit", "Action"]):
                            with hc:
                                st.markdown(f"<span style='color:{CARD_TEXT_MUTED};font-size:0.75em;text-transform:uppercase;letter-spacing:0.04em;'>{hl}</span>", unsafe_allow_html=True)

                        for row_idx, cand_row in candidates.iterrows():
                            c_poster, c_title, c_year, c_plat, c_imdb, c_fit, c_btn = st.columns([0.5, 2.8, 0.6, 1.0, 0.6, 0.7, 0.9])
                            with c_poster:
                                poster_url = poster_map.get(cand_row.get("title", ""))
                                _t_initial = (cand_row.get("title") or "?")[0].upper()
                                _plat_bg = PLATFORMS.get(cand_row.get("platform", ""), {}).get("color", "#2a2a3e")
                                _placeholder_html = (
                                    f'<div style="width:36px;height:50px;background:{_plat_bg};border-radius:4px;'
                                    f'display:flex;align-items:center;justify-content:center;'
                                    f'font-size:15px;font-weight:700;color:#fff;opacity:0.85;">{_t_initial}</div>'
                                )
                                if poster_url:
                                    try:
                                        st.image(poster_url, width=36)
                                    except Exception:
                                        st.markdown(_placeholder_html, unsafe_allow_html=True)
                                else:
                                    st.markdown(_placeholder_html, unsafe_allow_html=True)
                            with c_title:
                                st.markdown(f"**{cand_row.get('title', '')}**")
                            with c_year:
                                yr = cand_row.get("release_year")
                                st.markdown(f"<span style='color:{CARD_TEXT_MUTED};font-size:0.9em;'>{int(yr) if pd.notna(yr) else ''}</span>", unsafe_allow_html=True)
                            with c_plat:
                                plat_key = cand_row.get("platform", "")
                                plat_color = PLATFORMS.get(plat_key, {}).get("color", CARD_TEXT_MUTED)
                                plat_name = PLATFORMS.get(plat_key, {}).get("name", plat_key)
                                st.markdown(f"<span style='color:{plat_color};font-size:0.85em;font-weight:600;'>{plat_name}</span>", unsafe_allow_html=True)
                            with c_imdb:
                                imdb_v = cand_row.get("imdb_score")
                                st.markdown(f"<span style='color:{CARD_ACCENT};font-weight:600;'>{imdb_v:.1f}</span>" if pd.notna(imdb_v) else "N/A", unsafe_allow_html=True)
                            with c_fit:
                                fit_v = (cand_row.get("fit_score") or 0) * 100
                                fit_color = "#2ECC71" if fit_v >= 70 else "#F39C12"
                                st.markdown(f"<span style='color:{fit_color};font-size:0.85em;font-weight:600;'>{fit_v:.0f}%</span>", unsafe_allow_html=True)
                            with c_btn:
                                title_name = cand_row.get("title", "")
                                title_id = cand_row.get("id")
                                _detail_key = f"gap_detail_{genre_key}"
                                _is_open = st.session_state.get(_detail_key) == row_idx
                                if st.button("▲ Close" if _is_open else "Details ▾", key=f"view_{genre_key}_{row_idx}"):
                                    st.session_state[_detail_key] = None if _is_open else row_idx
                                    st.rerun()

                            # ── Inline detail card ──
                            if st.session_state.get(f"gap_detail_{genre_key}") == row_idx:
                                _tid = cand_row.get("id")
                                _drows = titles[titles["id"] == _tid] if _tid else pd.DataFrame()
                                if _drows.empty:
                                    _drows = titles[titles["title"] == cand_row.get("title", "")]
                                _d = _drows.iloc[0].to_dict() if not _drows.empty else {}

                                _poster_d = poster_map.get(cand_row.get("title", ""))
                                _plat_k = cand_row.get("platform", "")
                                _plat_c = PLATFORMS.get(_plat_k, {}).get("color", CARD_TEXT_MUTED)
                                _plat_n = PLATFORMS.get(_plat_k, {}).get("name", _plat_k)
                                _plat_bg2 = PLATFORMS.get(_plat_k, {}).get("color", "#2a2a3e")
                                _init2 = (cand_row.get("title") or "?")[0].upper()
                                _placeholder2 = (
                                    f'<div style="width:80px;height:112px;background:{_plat_bg2};border-radius:6px;'
                                    f'display:flex;align-items:center;justify-content:center;'
                                    f'font-size:28px;font-weight:700;color:#fff;opacity:0.85;">{_init2}</div>'
                                )

                                _desc = str(_d.get("description") or "")
                                _genres_raw = _d.get("genres") or []
                                if isinstance(_genres_raw, str):
                                    try:
                                        import ast as _ast
                                        _genres_raw = _ast.literal_eval(_genres_raw)
                                    except Exception:
                                        _genres_raw = [g.strip() for g in _genres_raw.strip("[]").split(",") if g.strip()]
                                _genre_pills = "".join(
                                    f'<span style="background:rgba(0,180,166,0.12);color:#00B4A6;padding:2px 8px;border-radius:10px;font-size:0.75em;margin-right:4px;">{g}</span>'
                                    for g in (_genres_raw[:5] if _genres_raw else [])
                                )

                                _imdb_d = _d.get("imdb_score") or cand_row.get("imdb_score")
                                _votes_d = _d.get("imdb_votes")
                                _imdb_str = f"★ {_imdb_d:.1f}" if _imdb_d and pd.notna(_imdb_d) else ""
                                if _votes_d and pd.notna(_votes_d):
                                    _v = int(_votes_d)
                                    _votes_fmt = f"{_v/1_000_000:.1f}M" if _v >= 1_000_000 else f"{_v/1_000:.0f}K" if _v >= 1_000 else str(_v)
                                    _imdb_str += f" ({_votes_fmt} votes)"

                                _meta_parts = []
                                _yr2 = _d.get("release_year") or cand_row.get("release_year")
                                if _yr2 and pd.notna(_yr2): _meta_parts.append(str(int(_yr2)))
                                _type2 = _d.get("type", "")
                                if _type2: _meta_parts.append(_type2.replace("MOVIE", "Movie").replace("SHOW", "TV Show"))
                                _rt = _d.get("runtime")
                                if _rt and pd.notna(_rt): _meta_parts.append(f"{int(_rt)} min")
                                _cert = _d.get("age_certification")
                                if _cert and pd.notna(_cert) and str(_cert) != "nan": _meta_parts.append(str(_cert))
                                _meta_str = " · ".join(_meta_parts)

                                _aw = _d.get("award_wins")
                                _awards_html = (
                                    f'<span style="color:#FFD700;font-size:0.8em;margin-left:8px;">🏆 {int(_aw)} award{"s" if int(_aw) != 1 else ""}</span>'
                                    if _aw and pd.notna(_aw) and int(_aw) > 0 else ""
                                )
                                _coll = _d.get("collection_name")
                                _coll_html = (
                                    f'<div style="font-size:0.78em;color:{CARD_TEXT_MUTED};margin-top:4px;">Part of: <i>{_coll}</i></div>'
                                    if _coll and pd.notna(_coll) else ""
                                )
                                _fit2 = (cand_row.get("fit_score") or 0) * 100
                                _fit2_color = "#2ECC71" if _fit2 >= 70 else "#F39C12"

                                st.markdown(
                                    f'<div style="background:{CARD_BG};border:1px solid {CARD_BORDER};border-radius:8px;padding:12px 16px;margin:4px 0 10px 0;">',
                                    unsafe_allow_html=True,
                                )
                                _dc1, _dc2 = st.columns([1, 7])
                                with _dc1:
                                    if _poster_d:
                                        try:
                                            st.image(_poster_d, width=80)
                                        except Exception:
                                            st.markdown(_placeholder2, unsafe_allow_html=True)
                                    else:
                                        st.markdown(_placeholder2, unsafe_allow_html=True)
                                with _dc2:
                                    st.markdown(
                                        f'<div style="padding:4px 0 4px 8px;">'
                                        f'<div style="font-size:1.05em;font-weight:700;color:{CARD_TEXT};">'
                                        f'{cand_row.get("title", "")}{_awards_html}</div>'
                                        f'<div style="font-size:0.82em;color:{CARD_TEXT_MUTED};margin-top:2px;">{_meta_str}</div>'
                                        f'<div style="font-size:0.85em;color:{CARD_ACCENT};margin-top:3px;font-weight:600;">{_imdb_str}</div>'
                                        f'<div style="margin-top:6px;">'
                                        f'<span style="color:{_plat_c};font-size:0.78em;font-weight:700;background:rgba(0,0,0,0.25);padding:2px 8px;border-radius:8px;">{_plat_n}</span>'
                                        f'<span style="color:{_fit2_color};font-size:0.78em;font-weight:700;margin-left:8px;">Fit: {_fit2:.0f}%</span>'
                                        f'</div>'
                                        f'<div style="margin-top:6px;">{_genre_pills}</div>'
                                        + (f'<div style="font-size:0.83em;color:{CARD_TEXT_MUTED};margin-top:8px;line-height:1.5;">{_desc[:300]}{"…" if len(_desc) > 300 else ""}</div>' if _desc else "")
                                        + _coll_html
                                        + f'</div>',
                                        unsafe_allow_html=True,
                                    )
                                st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.caption("No competitor titles found for this genre.")
            except Exception:
                continue

    except Exception as e:
        st.warning(f"Full Acquisition Report could not be computed: {e}")

    try:
        st.download_button("📥 Download Gap Report", data=gaps.to_csv(index=False), file_name="gap_analysis.csv", mime="text/csv")
    except Exception:
        pass

else:
    st.info("No significant gaps found for this perspective.")

st.divider()

# ─── Section 7: Competitive Positioning ───────────────────────────────────────
st.markdown(
    section_header_html("Competitive Positioning",
                        "Head-to-head genre and quality comparison — where the merged entity leads, trails, and competes."),
    unsafe_allow_html=True,
)

comp_options = {p: PLATFORMS[p]["name"] for p in COMPETITOR_PLATFORMS}
selected_comp = st.selectbox("Compare Netflix+Max against", list(comp_options.keys()),
                              format_func=lambda x: comp_options[x], key="comp_pos_select")

try:
    pos = compute_competitive_positioning(titles, selected_comp)
    comp_name = comp_options[selected_comp]
    merger_color = _MERGED_COLOR
    comp_color = PLATFORMS.get(selected_comp, {}).get("color", "#555")

    col_ours, col_theirs, col_battle = st.columns(3)

    with col_ours:
        st.markdown(f'<div style="font-weight:700;color:{merger_color};margin-bottom:8px;font-size:0.95em;">✅ Merger Leads</div>', unsafe_allow_html=True)
        if pos["our_leads"]:
            items = pos["our_leads"][:6]
            multiples = [float(str(i["lead"]).replace("x", "")) for i in reversed(items)]
            labels = [_fmt_genre(i["genre"]) + (f" ({i['avg_imdb']:.1f})" if i.get("avg_imdb") and pd.notna(i.get("avg_imdb")) else "") for i in reversed(items)]
            fig_leads = go.Figure(go.Bar(y=labels, x=multiples, orientation="h", marker_color=merger_color, text=[i["lead"] for i in reversed(items)], textposition="outside", cliponaxis=False))
            fig_leads.update_layout(height=max(140, 38 * len(items) + 60), template=PLOTLY_TEMPLATE, margin=dict(l=0, r=60, t=10, b=10), xaxis_title="Lead multiple (x)", showlegend=False)
            st.plotly_chart(fig_leads, use_container_width=True)
        else:
            st.caption("No clear leads identified.")

    with col_theirs:
        st.markdown(f'<div style="font-weight:700;color:{comp_color};margin-bottom:8px;font-size:0.95em;">⚠️ {comp_name} Leads</div>', unsafe_allow_html=True)
        if pos["their_leads"]:
            items = pos["their_leads"][:6]
            multiples = [float(str(i["lead"]).replace("x", "")) for i in reversed(items)]
            labels = [_fmt_genre(i["genre"]) + (f" ({i['avg_imdb']:.1f})" if i.get("avg_imdb") and pd.notna(i.get("avg_imdb")) else "") for i in reversed(items)]
            fig_theirs = go.Figure(go.Bar(y=labels, x=multiples, orientation="h", marker_color=comp_color, text=[i["lead"] for i in reversed(items)], textposition="outside", cliponaxis=False))
            fig_theirs.update_layout(height=max(140, 38 * len(items) + 60), template=PLOTLY_TEMPLATE, margin=dict(l=0, r=60, t=10, b=10), xaxis_title="Lead multiple (x)", showlegend=False)
            st.plotly_chart(fig_theirs, use_container_width=True)
        else:
            st.caption("No clear leads identified.")

    with col_battle:
        st.markdown(f'<div style="font-weight:700;color:{CARD_ACCENT};margin-bottom:8px;font-size:0.95em;">⚔️ Battleground</div>', unsafe_allow_html=True)
        for bg in pos["battlegrounds"]:
            genre_label = _fmt_genre(bg["genre"] if isinstance(bg, dict) else bg)
            closeness = bg.get("closeness", 0.5) if isinstance(bg, dict) else 0.5
            pct = int(closeness * 100)
            st.markdown(
                f'<div style="display:inline-block;background:rgba(255,215,0,0.08);border:1px solid rgba(255,215,0,0.3);border-radius:8px;padding:6px 10px;margin:3px 2px;min-width:110px;">'
                f'<div style="font-size:0.85em;color:{CARD_ACCENT};font-weight:600;margin-bottom:4px;">{genre_label}</div>'
                f'<div style="background:{CARD_BORDER};border-radius:3px;height:4px;">'
                f'<div style="background:{CARD_ACCENT};width:{pct}%;height:4px;border-radius:3px;"></div></div>'
                f'<div style="font-size:0.72em;color:{CARD_TEXT_MUTED};margin-top:2px;">{pct}% merger share</div></div>',
                unsafe_allow_html=True,
            )

    battleground_top = [_fmt_genre(bg["genre"] if isinstance(bg, dict) else bg) for bg in pos["battlegrounds"][:2]] if pos["battlegrounds"] else []
    their_top = [_fmt_genre(item["genre"]) for item in pos["their_leads"][:2]] if pos["their_leads"] else []
    invest_genres = battleground_top if battleground_top else their_top
    genres_display = " and ".join(invest_genres) if invest_genres else "key battleground genres"
    merged_imdb = pos.get("merged_avg_imdb", 0)
    comp_imdb = pos.get("comp_avg_imdb", 0)
    quality_note = (
        f"The merger's quality edge ({merged_imdb:.1f} vs {comp_imdb:.1f} avg IMDb) provides a strong foundation."
        if merged_imdb > comp_imdb
        else f"Improving content quality in {genres_display} should be the primary focus."
    )
    comp_context = _COMP_CONTEXT.get(selected_comp, "")
    comp_closing = _COMP_CLOSING.get(selected_comp, "")
    rec_text = (
        f"To close the gap with <b>{comp_name}</b>, the merged entity should prioritize "
        f"targeted acquisition and original production in <b>{genres_display}</b>. "
        f"{quality_note} {comp_context}{comp_closing}"
    )
    st.markdown(styled_banner_html("🎯", rec_text), unsafe_allow_html=True)

except Exception as e:
    st.warning(f"Competitive positioning could not be computed: {e}")

st.divider()

# ─── Section 8: Alternative Merger Scenarios ──────────────────────────────────
st.markdown(
    section_header_html(
        "Alternative Merger Scenarios",
        "Compare any two-platform merger scenario against the Netflix+Max baseline.",
    ),
    unsafe_allow_html=True,
)

if "best_scenario_pair" not in st.session_state:
    try:
        st.session_state["best_scenario_pair"] = compute_best_alternative_scenario(titles)
    except Exception:
        st.session_state["best_scenario_pair"] = None

best_pair = st.session_state.get("best_scenario_pair")
best_a, best_b = best_pair if best_pair else ("disney", "prime")

scen_c1, scen_c2 = st.columns(2)
with scen_c1:
    platform_a = st.selectbox("Platform A", ALL_PLATFORMS,
                               index=ALL_PLATFORMS.index("netflix") if "netflix" in ALL_PLATFORMS else 0,
                               key="scen_a", format_func=lambda x: PLATFORMS[x]["name"])
with scen_c2:
    platform_b = st.selectbox("Platform B", ALL_PLATFORMS,
                               index=ALL_PLATFORMS.index("disney") if "disney" in ALL_PLATFORMS else 1,
                               key="scen_b", format_func=lambda x: PLATFORMS[x]["name"])

if platform_a == platform_b:
    st.warning("Select two different platforms to compare scenarios.")
else:
    try:
        baseline_kpis = compute_alternative_scenario(titles, "netflix", "max")
        scenario_kpis = compute_alternative_scenario(titles, platform_a, platform_b)
        best_kpis = compute_alternative_scenario(titles, best_a, best_b)

        name_a = PLATFORMS[platform_a]["name"]
        name_b = PLATFORMS[platform_b]["name"]
        best_name_a = PLATFORMS[best_a]["name"]
        best_name_b = PLATFORMS[best_b]["name"]
        color_a = PLATFORMS.get(platform_a, {}).get("color", "#555")

        col_base, col_user, col_best = st.columns(3)

        def _render_scenario_col(kpis_dict, col_title, col_subtitle, accent):
            st.markdown(
                f'<div style="font-weight:700;color:{accent};font-size:0.95em;">{col_title}</div>'
                f'<div style="font-size:0.78em;color:{CARD_TEXT_MUTED};margin-bottom:8px;">{col_subtitle}</div>',
                unsafe_allow_html=True,
            )
            for k, kpi in kpis_dict.items():
                if k.startswith("_"):
                    continue
                st.markdown(
                    styled_metric_card_html(kpi["label"], kpi["value"], help_text=kpi.get("detail"), accent_color=accent),
                    unsafe_allow_html=True,
                )

        with col_base:
            _render_scenario_col(baseline_kpis, "Netflix + Max", "Current Proposal", _MERGED_COLOR)
            st.markdown(
                f'<div style="font-size:0.82em;color:{CARD_TEXT_MUTED};margin-top:8px;border-top:1px solid {CARD_BORDER};padding-top:6px;">'
                f'<b>Strategic Profile:</b> {_strategic_profile("netflix", "max")}</div>',
                unsafe_allow_html=True,
            )

        with col_user:
            _render_scenario_col(scenario_kpis, f"{name_a} + {name_b}", "Your Scenario", color_a)
            st.markdown(
                f'<div style="font-size:0.82em;color:{CARD_TEXT_MUTED};margin-top:8px;border-top:1px solid {CARD_BORDER};padding-top:6px;">'
                f'<b>Strategic Profile:</b> {_strategic_profile(platform_a, platform_b)}</div>',
                unsafe_allow_html=True,
            )

        with col_best:
            _render_scenario_col(
                best_kpis,
                f'{best_name_a} + {best_name_b} <span style="font-size:0.7em;background:rgba(255,215,0,0.15);color:{CARD_ACCENT};padding:1px 6px;border-radius:8px;margin-left:6px;border:1px solid rgba(255,215,0,0.3);">auto-computed</span>',
                "System Recommendation",
                CARD_ACCENT,
            )
            st.markdown(
                f'<div style="font-size:0.82em;color:{CARD_TEXT_MUTED};margin-top:8px;border-top:1px solid {CARD_BORDER};padding-top:6px;">'
                f'<b>Strategic Profile:</b> {_strategic_profile(best_a, best_b)}</div>',
                unsafe_allow_html=True,
            )

        # Radar chart
        try:
            def _build_raw(df, pa, pb):
                return pd.concat([df[df["platform"] == pa], df[df["platform"] == pb]]).drop_duplicates(subset="id", keep="first")

            merged_raw = build_merged_entity(titles)
            user_raw = _build_raw(titles, platform_a, platform_b)
            best_raw = _build_raw(titles, best_a, best_b)

            def _radar_raw_vals(df_c):
                n = len(df_c)
                avg_imdb = float(df_c["imdb_score"].mean() or 0)
                award_wins = float(df_c["award_wins"].fillna(0).sum()) if "award_wins" in df_c.columns else 0.0
                prestige = award_wins / max(n, 1) * 1000
                genres_vc = df_c.explode("genres")["genres"].value_counts()
                tot = genres_vc.sum()
                probs = genres_vc / max(tot, 1)
                entropy = float(-np.sum(probs * np.log2(probs + 1e-9)))
                intl = float((df_c["language"].fillna("en") != "en").sum() / max(n, 1)) if "language" in df_c.columns else 0.0
                franchise = float(df_c["collection_name"].notna().sum() / max(n, 1)) if "collection_name" in df_c.columns else 0.0
                return [float(n), avg_imdb, prestige, entropy, intl, franchise]

            vals_base = _radar_raw_vals(merged_raw)
            vals_user = _radar_raw_vals(user_raw)
            vals_best_r = _radar_raw_vals(best_raw)

            all_v = list(zip(vals_base, vals_user, vals_best_r))
            norm_max = [max(a, b, c) or 1.0 for a, b, c in all_v]
            vals_base_n = [v / m for v, m in zip(vals_base, norm_max)]
            vals_user_n = [v / m for v, m in zip(vals_user, norm_max)]
            vals_best_n = [v / m for v, m in zip(vals_best_r, norm_max)]

            categories = ["Scale", "Quality", "Prestige", "Diversity", "International", "Franchise"]

            fig_radar = go.Figure()
            for vals_n, rname, rcolor, lw, fill_opacity in [
                (vals_base_n, "Netflix + Max", _MERGED_COLOR, 3, 0.15),
                (vals_user_n, f"{name_a} + {name_b}", color_a, 2, 0.08),
                (vals_best_n, f"{best_name_a} + {best_name_b}", CARD_ACCENT, 2, 0.08),
            ]:
                r_hex = rcolor.lstrip("#")
                r_int, g_int, b_int = int(r_hex[0:2], 16), int(r_hex[2:4], 16), int(r_hex[4:6], 16)
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals_n + [vals_n[0]], theta=categories + [categories[0]],
                    fill="toself",
                    fillcolor=f"rgba({r_int},{g_int},{b_int},{fill_opacity})",
                    line=dict(color=rcolor, width=lw), name=rname,
                ))

            fig_radar.update_layout(
                template=PLOTLY_TEMPLATE, height=450,
                polar=dict(radialaxis=dict(range=[0, 1], showticklabels=False, gridcolor="#333"), angularaxis=dict(gridcolor="#333")),
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
                title="Scenario Comparison Radar",
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            # Trade-off analysis
            try:
                gain_dims = sorted([(d, uv - bv) for d, uv, bv in zip(categories, vals_user_n, vals_base_n) if uv - bv > 0.05], key=lambda x: -x[1])
                loss_dims = sorted([(d, bv - uv) for d, uv, bv in zip(categories, vals_user_n, vals_base_n) if bv - uv > 0.05], key=lambda x: -x[1])

                # Build concrete metric context strings for each dimension
                _dim_raw_user  = dict(zip(categories, vals_user))
                _dim_raw_base  = dict(zip(categories, vals_base))
                _dim_concrete = {
                    "Scale":         f"{int(_dim_raw_user['Scale']):,} combined titles vs {int(_dim_raw_base['Scale']):,} in the Netflix+Max baseline",
                    "Quality":       f"avg IMDb {_dim_raw_user['Quality']:.2f} vs {_dim_raw_base['Quality']:.2f} for Netflix+Max",
                    "Prestige":      f"{_dim_raw_user['Prestige']:.1f} award-wins per 1K titles vs {_dim_raw_base['Prestige']:.1f} for Netflix+Max",
                    "Diversity":     f"Shannon entropy {_dim_raw_user['Diversity']:.2f} vs {_dim_raw_base['Diversity']:.2f} — {'more' if _dim_raw_user['Diversity'] > _dim_raw_base['Diversity'] else 'less'} evenly spread across genres",
                    "International": f"{_dim_raw_user['International']:.0%} non-English-market titles vs {_dim_raw_base['International']:.0%} for Netflix+Max",
                    "Franchise":     f"{_dim_raw_user['Franchise']:.0%} of titles in franchises vs {_dim_raw_base['Franchise']:.0%} for Netflix+Max",
                }

                gains_html = "".join(
                    f'<li style="margin-bottom:8px;">'
                    f'<div style="display:flex;align-items:flex-start;gap:8px;">'
                    f'<span style="background:rgba(46,204,113,0.18);color:#2ECC71;padding:2px 7px;border-radius:4px;'
                    f'font-size:0.76em;font-weight:700;white-space:nowrap;margin-top:1px;">+{delta:.2f}</span>'
                    f'<div><span style="font-weight:600;">{d}</span>'
                    f'<div style="color:{CARD_TEXT_MUTED};font-size:0.84em;margin-top:2px;">{_dim_concrete.get(d, _DIM_GAIN_TEXT.get(d, ""))}</div>'
                    f'</div></div></li>'
                    for d, delta in gain_dims[:3]
                ) or f'<li style="color:#888;font-style:italic;">No significant advantages over the Netflix+Max baseline on any of the six dimensions.</li>'

                losses_html = "".join(
                    f'<li style="margin-bottom:8px;">'
                    f'<div style="display:flex;align-items:flex-start;gap:8px;">'
                    f'<span style="background:rgba(231,76,60,0.18);color:#E74C3C;padding:2px 7px;border-radius:4px;'
                    f'font-size:0.76em;font-weight:700;white-space:nowrap;margin-top:1px;">−{delta:.2f}</span>'
                    f'<div><span style="font-weight:600;">{d}</span>'
                    f'<div style="color:{CARD_TEXT_MUTED};font-size:0.84em;margin-top:2px;">{_dim_concrete.get(d, _DIM_LOSS_TEXT.get(d, ""))}</div>'
                    f'</div></div></li>'
                    for d, delta in loss_dims[:3]
                ) or f'<li style="color:#888;font-style:italic;">No significant disadvantages vs the Netflix+Max baseline on any of the six dimensions.</li>'

                # Decision Guidance
                user_total = sum(vals_user_n)
                base_total = sum(vals_base_n)
                best_total = sum(vals_best_n)
                top_gain_dim = gain_dims[0][0] if gain_dims else None
                top_loss_dim = loss_dims[0][0] if loss_dims else None
                user_size   = int(_dim_raw_user["Scale"])
                base_size   = int(_dim_raw_base["Scale"])
                user_imdb   = _dim_raw_user["Quality"]
                base_imdb   = _dim_raw_base["Quality"]
                size_delta_pct = (user_size - base_size) / max(base_size, 1) * 100

                if user_total >= base_total * 0.97:
                    verdict = (
                        f"<b>{name_a} + {name_b}</b> is a near-equivalent alternative to Netflix+Max, scoring within 3% "
                        f"of the baseline across all six dimensions. The combined catalog of <b>{user_size:,} titles</b> "
                        f"({'larger' if user_size > base_size else 'smaller'} by {abs(size_delta_pct):.0f}%) "
                        f"carries an avg IMDb of <b>{user_imdb:.2f}</b> vs {base_imdb:.2f} for Netflix+Max. "
                        f"Either merger would represent a defensible strategic outcome — the choice depends on "
                        f"partner availability and licensing terms."
                    )
                elif top_gain_dim and top_loss_dim:
                    verdict = (
                        f"<b>{name_a} + {name_b}</b> trades <b>{top_loss_dim}</b> for <b>{top_gain_dim}</b> relative to Netflix+Max. "
                        f"The combined catalog of <b>{user_size:,} titles</b> ({size_delta_pct:+.0f}% vs baseline) "
                        f"carries an avg IMDb of <b>{user_imdb:.2f}</b>. "
                        f"This is the right choice if <b>{top_gain_dim}</b> is your primary strategic lever — "
                        f"for example, {'content breadth and subscriber acquisition' if top_gain_dim == 'Scale' else 'critical reputation and awards-season visibility' if top_gain_dim == 'Prestige' else 'genre balance and churn reduction' if top_gain_dim == 'Diversity' else 'global expansion and international subscriber growth' if top_gain_dim == 'International' else 'IP-driven engagement and merchandising revenue' if top_gain_dim == 'Franchise' else 'per-title quality and premium brand perception'}. "
                        + (f"For a more balanced profile, the system-recommended <b>{best_name_a} + {best_name_b}</b> merger scores higher overall." if best_total > user_total else "")
                    )
                elif top_gain_dim:
                    verdict = (
                        f"<b>{name_a} + {name_b}</b> outperforms the Netflix+Max baseline on <b>{top_gain_dim}</b> "
                        f"with no material trade-offs elsewhere. The <b>{user_size:,}</b>-title catalog "
                        f"(avg IMDb <b>{user_imdb:.2f}</b>) represents a strong alternative. "
                        + (f"The system-recommended <b>{best_name_a} + {best_name_b}</b> pair scores even higher overall." if best_total > user_total else "")
                    )
                else:
                    verdict = (
                        f"The Netflix+Max baseline outperforms <b>{name_a} + {name_b}</b> across all six dimensions. "
                        f"At <b>{user_size:,} titles</b> (avg IMDb {user_imdb:.2f}), this scenario is strategically weaker "
                        f"than the proposed merger. "
                        + (f"If you are looking for a stronger alternative, consider the system-recommended <b>{best_name_a} + {best_name_b}</b> merger." if best_total > user_total else "")
                    )

                best_use_profile = _strategic_profile(platform_a, platform_b)

                st.markdown(f"""
                <div style="background:{CARD_BG};border:1px solid {CARD_BORDER};border-radius:8px;padding:18px 22px;margin-top:16px;">
                    <div style="font-weight:700;color:{CARD_TEXT};margin-bottom:14px;font-size:0.97em;letter-spacing:0.01em;">
                        Strategic Trade-off Analysis
                        <span style="font-weight:400;color:{CARD_TEXT_MUTED};font-size:0.82em;margin-left:8px;">{name_a} + {name_b} vs Netflix+Max baseline</span>
                    </div>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:16px;">
                        <div>
                            <div style="font-size:0.78em;color:#2ECC71;font-weight:700;margin-bottom:10px;text-transform:uppercase;letter-spacing:0.07em;">↑ Competitive Advantages</div>
                            <ul style="margin:0;padding-left:0;list-style:none;color:{CARD_TEXT};font-size:0.88em;line-height:1.55;">{gains_html}</ul>
                        </div>
                        <div>
                            <div style="font-size:0.78em;color:#E74C3C;font-weight:700;margin-bottom:10px;text-transform:uppercase;letter-spacing:0.07em;">↓ Strategic Trade-offs</div>
                            <ul style="margin:0;padding-left:0;list-style:none;color:{CARD_TEXT};font-size:0.88em;line-height:1.55;">{losses_html}</ul>
                        </div>
                    </div>
                    <div style="border-top:1px solid {CARD_BORDER};padding-top:14px;">
                        <div style="font-size:0.78em;color:{CARD_TEXT_MUTED};font-weight:700;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:8px;">Strategic Assessment</div>
                        <div style="font-size:0.88em;color:{CARD_TEXT};line-height:1.6;">{verdict}</div>
                        <div style="font-size:0.82em;color:{CARD_TEXT_MUTED};margin-top:10px;padding-top:8px;border-top:1px solid rgba(255,255,255,0.05);">
                            <span style="color:{CARD_TEXT};font-weight:600;">Positioning identity:</span> {best_use_profile}
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)
            except Exception:
                pass

            # Dimension Breakdown table
            with st.expander("Dimension Breakdown — Exact Values Behind the Radar"):
                dim_df = pd.DataFrame({
                    "Dimension": categories,
                    "Netflix + Max": [f"{v:.3f}" for v in vals_base_n],
                    f"{name_a} + {name_b}": [f"{v:.3f}" for v in vals_user_n],
                    f"{best_name_a} + {best_name_b}": [f"{v:.3f}" for v in vals_best_n],
                })
                st.dataframe(dim_df, hide_index=True)
                st.caption("Values normalized 0–1 within this comparison set. 1.0 = highest across all three scenarios.")

        except Exception:
            sc_size_str = scenario_kpis.get("catalog_size", {}).get("value", "N/A")
            sc_imdb_val = scenario_kpis.get("avg_imdb", {}).get("value", "N/A")
            takeaway = (
                f"The <b>{name_a} + {name_b}</b> scenario yields <b>{sc_size_str} titles</b> "
                f"with an avg IMDb of <b>{sc_imdb_val}</b>. "
                f"Strategic profile: {_strategic_profile(platform_a, platform_b)}."
            )
            st.markdown(styled_banner_html("🔍", takeaway), unsafe_allow_html=True)

    except Exception as e:
        st.warning(f"Alternative scenario could not be computed: {e}")

st.divider()

# ─── Section 9: Catalog Concentration Index ───────────────────────────────────
st.markdown(
    section_header_html(
        "Catalog Concentration Index",
        "How does the Netflix+Max merger shift market concentration? HHI analysis before and after.",
    ),
    unsafe_allow_html=True,
)

try:
    sim = compute_market_simulation(titles)

    # Pre-merger HHI: all 6 individual platforms
    pre_sizes = {k: len(titles[titles["platform"] == k]) for k in ALL_PLATFORMS}
    total_pre = sum(pre_sizes.values())
    hhi_pre = round(sum((v / max(total_pre, 1) * 100) ** 2 for v in pre_sizes.values()))
    hhi_post = sim["hhi"]
    delta_hhi = hhi_post - hhi_pre

    hhi_col1, hhi_col2, hhi_col3 = st.columns(3)
    with hhi_col1:
        st.markdown(
            styled_metric_card_html("Pre-Merger HHI", f"{hhi_pre:,}",
                                    subtitle="6 separate platforms",
                                    help_text="Herfindahl-Hirschman Index based on catalog sizes before the merger",
                                    accent_color=_MERGED_COLOR),
            unsafe_allow_html=True,
        )
    with hhi_col2:
        hhi_color = "#E74C3C" if hhi_post > 2500 else "#F39C12" if hhi_post > 1500 else "#2ECC71"
        st.markdown(
            styled_metric_card_html("Post-Merger HHI", f"{hhi_post:,}",
                                    subtitle=sim["regulatory_label"],
                                    accent_color=hhi_color,
                                    help_text="HHI after Netflix+Max catalog consolidation"),
            unsafe_allow_html=True,
        )
    with hhi_col3:
        delta_color = "#E74C3C" if delta_hhi > 200 else "#F39C12"
        st.markdown(
            styled_metric_card_html("Concentration Increase", f"+{delta_hhi:,} pts",
                                    subtitle="Antitrust threshold: +200 pts",
                                    accent_color=delta_color,
                                    help_text="Increases above 200 pts when post-merger HHI > 1,500 trigger DOJ/FTC review"),
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div style="font-size:0.88em;color:{CARD_TEXT_MUTED};margin-top:12px;line-height:1.8;">'
        f'The HHI measures catalog concentration — higher values mean fewer players control the content landscape. '
        f'The pre-merger HHI of <b style="color:{CARD_TEXT};">{hhi_pre:,}</b> rises to '
        f'<b style="color:{hhi_color};">{hhi_post:,}</b> post-merger ({sim["regulatory_label"]}). '
        f'The <b>+{delta_hhi:,} point increase</b> would, in a real acquisition scenario, require mandatory '
        f'regulatory review under DOJ/FTC merger guidelines (threshold: +200 pts when post-merger HHI &gt; 1,500).<br>'
        f'<b>Thresholds:</b> &lt;1,500 Unconcentrated · 1,500–2,500 Moderately Concentrated · &gt;2,500 Highly Concentrated<br>'
        f'<i>Note: Catalog size is used as a proxy for market share. Prime Video\'s catalog breadth reflects '
        f'broad licensing, not subscriber engagement. HHI based on catalog counts only.</i>'
        f'</div>',
        unsafe_allow_html=True,
    )

except Exception as e:
    st.warning(f"Catalog Concentration Index could not be computed: {e}")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="border-top:1px solid #333;padding:16px 0;color:#666;'
    'font-size:0.8em;text-align:center;">'
    'Hypothetical merger for academic analysis only. Data is a snapshot (mid-2023). '
    'Enrichment data from Wikidata (72% award coverage), TMDB, IMDb public datasets. '
    'Netflix withdrew from this acquisition (Feb 26, 2026). '
    'All insights are illustrative, not prescriptive.'
    '</div>',
    unsafe_allow_html=True,
)
