"""Page 2: Platform Comparisons — merged Netflix+Max vs streaming competitors."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from src.analysis.comparisons import (
    build_comparison_df,
    compute_age_profile,
    compute_era_focus,
    compute_genre_drilldown,
    compute_genre_heatmap,
    compute_geographic_diversity,
    compute_market_positioning,
    compute_quality_tiers,
    compute_quick_comparison,
    compute_strategic_insights,
    compute_volume_stats,
    compute_volume_summary,
)
from src.analysis.scoring import compute_quality_score, format_votes
from src.config import (
    CARD_ACCENT,
    CARD_BG,
    CARD_BORDER,
    CARD_TEXT,
    CARD_TEXT_MUTED,
    CHART_HEIGHT,
    COMPARISON_MAX_COMPETITORS,
    COMPARISON_TOP_GENRES,
    COMPETITOR_PLATFORMS,
    INITIAL_SIDEBAR_STATE,
    LAYOUT,
    PAGE_ICON,
    PAGE_TITLE,
    PLATFORMS,
    PLOTLY_TEMPLATE,
    QUALITY_TIERS,
)
from src.config import WIKIDATA_COMPARISON_MIN_COVERAGE
from src.data.loaders import load_all_platforms_credits, load_all_platforms_titles, load_enriched_titles
from src.ui.filters import apply_filters, render_sidebar_filters
from src.ui.session import init_session_state

# ── page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title=f"Platform Comparisons | {PAGE_TITLE}",
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state=INITIAL_SIDEBAR_STATE,
)

init_session_state()

# Page-specific session state
st.session_state.setdefault("comp_competitors", ["disney", "prime"])
st.session_state.setdefault("comp_normalized", False)
st.session_state.setdefault("comp_selected_genre", None)
st.session_state.setdefault("comp_expanded_title", None)



# ── helpers ──────────────────────────────────────────────────────────────────

def _render_expanded_card(title_row, title_id: str, platform_name: str = ""):
    """Render an expanded detail card with metadata and cast/crew (full-width)."""
    # Header bar with title and platform
    st.markdown(
        f'<div style="background:{CARD_BG};border:2px solid {CARD_ACCENT};'
        f'border-radius:10px;padding:14px 18px;margin:8px 0 4px 0;">'
        f'<span style="font-size:1.1em;font-weight:700;color:{CARD_TEXT};">'
        f'{title_row.get("title", "Unknown")}</span>'
        f'<span style="color:{CARD_TEXT_MUTED};font-size:0.88em;"> — {platform_name}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Metadata row — 6 columns for full-width
    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
    with mc1:
        st.metric("Type", title_row.get("type", "N/A"))
    with mc2:
        year = title_row.get("release_year")
        st.metric("Year", int(year) if year == year else "N/A")
    with mc3:
        rt = title_row.get("runtime")
        rt_str = f"{int(rt)} min" if rt and rt == rt else "N/A"
        st.metric("Runtime", rt_str)
    with mc4:
        cert = title_row.get("age_certification")
        st.metric("Rating", cert if cert and str(cert) != "nan" else "N/A")
    with mc5:
        imdb = title_row.get("imdb_score")
        votes = title_row.get("imdb_votes")
        imdb_str = f"{imdb:.1f}" if imdb and imdb == imdb else "N/A"
        votes_str = format_votes(votes) if votes else ""
        label = f"{imdb_str} ({votes_str})" if votes_str else imdb_str
        st.metric("IMDb", label)
    with mc6:
        qs = None
        if "imdb_score" in title_row.index and "imdb_votes" in title_row.index:
            _tmp = pd.DataFrame([title_row])
            if "tmdb_popularity" not in _tmp.columns:
                _tmp["tmdb_popularity"] = 0
            qs = compute_quality_score(_tmp).iloc[0]
        if qs is not None:
            st.metric("Quality Score", f"{qs:.1f}/10")

    # Genre pills
    genres = title_row.get("genres")
    if isinstance(genres, list) and genres:
        pills = " ".join(
            f'<span style="background:{CARD_BORDER};color:{CARD_TEXT};'
            f'padding:3px 10px;border-radius:12px;font-size:0.82em;">'
            f'{g.title()}</span>'
            for g in genres
        )
        st.markdown(pills, unsafe_allow_html=True)

    # Description + Cast side by side
    desc_col, cast_col = st.columns([3, 2])

    with desc_col:
        desc = title_row.get("description")
        if desc and str(desc) != "nan":
            st.markdown("**Description**")
            st.markdown(
                f'<div style="font-size:0.88em;color:{CARD_TEXT};line-height:1.6;">'
                f'{desc}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("No description available.")

    # Top Franchises from TMDB collection data
    _enr_detail = load_enriched_titles()
    if not _enr_detail.empty and "collection_name" in _enr_detail.columns:
        _enr_match = _enr_detail[_enr_detail["id"] == title_id]
        if not _enr_match.empty:
            _coll = _enr_match.iloc[0].get("collection_name")
            if _coll and str(_coll) != "nan":
                st.markdown(f"**Franchise:** {_coll}")

    with cast_col:
        credits_df = load_all_platforms_credits()
        title_credits = credits_df[credits_df["title_id"] == title_id]

        if not title_credits.empty:
            directors = title_credits[title_credits["role"] == "DIRECTOR"]
            actors = title_credits[title_credits["role"] == "ACTOR"]

            if not directors.empty:
                dir_names = directors["name"].drop_duplicates().tolist()
                label = "Directors" if len(dir_names) > 1 else "Director"
                st.markdown(f"**{label}:** {', '.join(dir_names)}")

            if not actors.empty:
                st.markdown("**Cast**")
                cast_lines = []
                for _, actor in actors.head(8).iterrows():
                    char = actor.get("character", "")
                    name = actor["name"]
                    if char and str(char) not in ("nan", ""):
                        cast_lines.append(
                            f'<span style="color:{CARD_TEXT};">'
                            f'<strong>{name}</strong></span> '
                            f'<span style="color:{CARD_TEXT_MUTED};">as {char}</span>'
                        )
                    else:
                        cast_lines.append(
                            f'<span style="color:{CARD_TEXT};"><strong>{name}</strong></span>'
                        )
                st.markdown("<br>".join(cast_lines), unsafe_allow_html=True)
                if len(actors) > 8:
                    st.caption(f"+{len(actors) - 8} more")
        else:
            st.caption("No cast/crew data available.")

    st.markdown("---")


def _build_insight_prose(comp_name: str, data: dict, merged_name: str) -> list[str]:
    """Generate 2-3 formal, business-centric strategic insights for a competitor."""
    s = data["summary"]
    insights = []

    # 1. Scale & Quality Assessment
    vol_diff = s["comp_vol"] - s["merged_vol"]
    vol_pct = round(abs(vol_diff) / max(s["merged_vol"], 1) * 100)
    qual_diff = s["merged_avg_imdb"] - s["comp_avg_imdb"]

    if vol_diff > 0 and qual_diff > 0.15:
        insights.append(
            f"{comp_name} operates at greater scale ({s['comp_vol']:,} vs "
            f"{s['merged_vol']:,} titles, +{vol_pct}%), but the merged entity "
            f"holds a quality advantage (avg IMDb {s['merged_avg_imdb']:.2f} vs "
            f"{s['comp_avg_imdb']:.2f}). This positions the merged entity as the "
            f"stronger option for quality-conscious subscribers."
        )
    elif vol_diff > 0 and qual_diff < -0.15:
        insights.append(
            f"{comp_name} leads in both catalog size ({s['comp_vol']:,} titles, "
            f"+{vol_pct}%) and average quality (IMDb {s['comp_avg_imdb']:.2f} vs "
            f"{s['merged_avg_imdb']:.2f}), representing a significant competitive "
            f"challenge on multiple fronts."
        )
    elif vol_diff < 0 and qual_diff > 0.15:
        insights.append(
            f"The merged entity surpasses {comp_name} in both catalog size "
            f"({s['merged_vol']:,} vs {s['comp_vol']:,} titles) and content quality "
            f"(avg IMDb {s['merged_avg_imdb']:.2f} vs {s['comp_avg_imdb']:.2f}). "
            f"The merger strengthens competitive positioning on both dimensions."
        )
    elif vol_diff < 0 and qual_diff < -0.15:
        insights.append(
            f"While the merged entity leads in catalog volume ({s['merged_vol']:,} "
            f"vs {s['comp_vol']:,} titles), {comp_name} maintains a quality edge "
            f"(IMDb {s['comp_avg_imdb']:.2f} vs {s['merged_avg_imdb']:.2f}), "
            f"suggesting a more selective curation approach."
        )
    else:
        insights.append(
            f"The merged entity and {comp_name} operate at comparable scale and "
            f"quality ({s['merged_vol']:,} vs {s['comp_vol']:,} titles; IMDb "
            f"{s['merged_avg_imdb']:.2f} vs {s['comp_avg_imdb']:.2f}). Genre "
            f"strategy and content differentiation are the primary competitive levers."
        )

    # 2. Genre Differentiation
    their_genres = [
        x["category"] for x in data["their_strengths"]
        if "tier" not in x["category"].lower()
    ][:3]
    our_genres = [
        x["category"] for x in data["our_strengths"]
        if "tier" not in x["category"].lower()
    ][:3]

    if their_genres and our_genres:
        insights.append(
            f"Content differentiation is clear: {comp_name} leads in "
            f"{', '.join(their_genres)}, while the merged entity holds stronger "
            f"positions in {', '.join(our_genres)}. These distinct genre profiles "
            f"suggest naturally segmented audience targets."
        )
    elif their_genres:
        insights.append(
            f"{comp_name} holds genre-level advantages in "
            f"{', '.join(their_genres)} — areas where the merged entity may "
            f"benefit from targeted content investment."
        )
    elif our_genres:
        insights.append(
            f"The merged entity leads across major genres, particularly in "
            f"{', '.join(our_genres)}. {comp_name} does not present a decisive "
            f"genre-level challenge."
        )

    # 3. Contested Territory
    if data["battlegrounds"]:
        bg = ", ".join(data["battlegrounds"][:3])
        insights.append(
            f"Closely contested categories ({bg}) are where subscriber "
            f"acquisition will be most competitive. Exclusive content and "
            f"recommendation quality in these genres may tip the balance."
        )

    return insights


# ── sidebar (global filters apply to all-platforms data) ─────────────────────

raw_df = load_all_platforms_titles()
filters = render_sidebar_filters(raw_df)
df = apply_filters(raw_df, filters)

# ── page header ──────────────────────────────────────────────────────────────

st.title("Platform Comparisons")
st.caption("Compare the merged Netflix + Max catalog against streaming competitors")

# ── controls ─────────────────────────────────────────────────────────────────

ctrl_left, ctrl_right = st.columns([3, 1])

with ctrl_left:
    competitor_options = {PLATFORMS[k]["name"]: k for k in COMPETITOR_PLATFORMS}
    default_names = [
        PLATFORMS[k]["name"]
        for k in st.session_state["comp_competitors"]
        if k in competitor_options.values()
    ]
    selected_names = st.multiselect(
        "Compare against (up to 3 competitors)",
        options=list(competitor_options.keys()),
        default=default_names,
        max_selections=COMPARISON_MAX_COMPETITORS,
        help="The merged Netflix + Max entity is always included as the baseline",
    )
    selected_keys = [competitor_options[n] for n in selected_names]
    st.session_state["comp_competitors"] = selected_keys

with ctrl_right:
    normalized = st.toggle(
        "Normalized mode",
        value=st.session_state["comp_normalized"],
        help="Show percentages instead of raw counts",
    )
    st.session_state["comp_normalized"] = normalized

# ── Competitor validation ────────────────────────────────────────────────────

if not selected_keys:
    st.warning("At least one competitor is required. Defaulting to Prime Video.")
    st.session_state["comp_competitors"] = ["prime"]
    selected_keys = ["prime"]

# Build comparison dataset
comp_df = build_comparison_df(df, selected_keys)

# Build color map for charts
color_map = {PLATFORMS["merged"]["name"]: PLATFORMS["merged"]["color"]}
for k in selected_keys:
    color_map[PLATFORMS[k]["name"]] = PLATFORMS[k]["color"]

# Platform display order
display_order = [PLATFORMS["merged"]["name"]] + [
    PLATFORMS[k]["name"] for k in sorted(selected_keys)
]


# ── Quick Comparison ─────────────────────────────────────────────────────────

quick = compute_quick_comparison(comp_df)
merged_name = PLATFORMS["merged"]["name"]

vol_label = quick["volume_leader"]
qual_label = quick["quality_leader"]
genre_leads = quick["merged_genre_leads"]
total_genres = quick["total_genres"]

st.info(
    f"📊 **Volume Leader:** {vol_label} ({quick['volume_count']:,} titles)  \n"
    f"⭐ **Quality Leader:** {qual_label} (Avg IMDb {quick['quality_avg']:.2f})  \n"
    f"🏆 **Genre Dominance:** {merged_name} leads in "
    f"{genre_leads}/{total_genres} top genres"
)

with st.expander("About These Comparisons"):
    st.markdown(
        f"**Merged entity:** Netflix and Max catalogs combined — titles on both "
        f"platforms counted once.\n\n"
        f"**Normalized mode** converts raw counts to % of each platform's catalog, "
        f"enabling fair comparison across platforms of different sizes. Toggle it "
        f"above to switch every chart and table.\n\n"
        f"**Quality score** = 85% Bayesian-adjusted IMDb + 15% normalized TMDB "
        f"popularity. This prevents obscure titles with few votes from ranking "
        f"artificially high.\n\n"
        f"**Genre counting:** Multi-genre titles (e.g., Drama + Comedy) count "
        f"toward each genre they belong to."
    )


# ── Section 1: Catalog Volume ────────────────────────────────────────────────

st.markdown("---")
st.subheader("Catalog Volume")
st.caption("Total library size per platform, split by movies and shows — a baseline measure of competitive scale.")

volume_stats = compute_volume_stats(comp_df, normalize=normalized)
volume_summary = compute_volume_summary(comp_df)

vol_chart, vol_table = st.columns([3, 2])

with vol_chart:
    if normalized:
        fig_vol = px.bar(
            volume_stats,
            x="platform_display",
            y="pct",
            color="type",
            barmode="stack",
            template=PLOTLY_TEMPLATE,
            height=CHART_HEIGHT,
            labels={"pct": "% of Platform Catalog", "platform_display": "Platform", "type": "Type"},
            category_orders={"platform_display": display_order},
        )
    else:
        fig_vol = px.bar(
            volume_stats,
            x="platform_display",
            y="count",
            color="type",
            barmode="group",
            template=PLOTLY_TEMPLATE,
            height=CHART_HEIGHT,
            labels={"count": "Title Count", "platform_display": "Platform", "type": "Type"},
            category_orders={"platform_display": display_order},
        )
    fig_vol.update_layout(legend_title_text="", xaxis_title="")
    st.plotly_chart(fig_vol, use_container_width=True)

with vol_table:
    st.markdown("**Summary**")
    display_summary = volume_summary[
        ["platform_display", "total_titles", "movies", "shows", "avg_imdb"]
    ].rename(
        columns={
            "platform_display": "Platform",
            "total_titles": "Total",
            "movies": "Movies",
            "shows": "Shows",
            "avg_imdb": "Avg IMDb",
        }
    )

    # Add Prestige Score column if award data has sufficient coverage
    _enr_for_prestige = load_enriched_titles()
    _prestige_coverage = _enr_for_prestige["award_wins"].notna().mean() if "award_wins" in _enr_for_prestige.columns else 0
    if _prestige_coverage >= WIKIDATA_COMPARISON_MIN_COVERAGE and "award_wins" in _enr_for_prestige.columns:
        prestige_scores = []
        for _, row in display_summary.iterrows():
            plat_name = row["Platform"]
            # Find platform key from display name
            plat_key = None
            for k, v in PLATFORMS.items():
                if v.get("name") == plat_name:
                    plat_key = k
                    break
            if plat_key and plat_key != "merged":
                plat_data = _enr_for_prestige[_enr_for_prestige["platform"] == plat_key]
            elif plat_key == "merged":
                from src.config import MERGED_PLATFORMS
                plat_data = _enr_for_prestige[_enr_for_prestige["platform"].isin(MERGED_PLATFORMS)]
            else:
                plat_data = pd.DataFrame()
            if not plat_data.empty:
                n_titles = len(plat_data)
                award_wins = plat_data["award_wins"].fillna(0).sum()
                prestige_scores.append(round(award_wins / n_titles * 1000, 1))
            else:
                prestige_scores.append(None)
        display_summary["Prestige"] = prestige_scores
    else:
        display_summary["Prestige"] = "—"

    st.dataframe(display_summary, use_container_width=True, hide_index=True)
    if _prestige_coverage < WIKIDATA_COMPARISON_MIN_COVERAGE:
        st.caption(f"Prestige Score unavailable (award coverage: {_prestige_coverage:.0%}, need ≥{WIKIDATA_COMPARISON_MIN_COVERAGE:.0%})")


# ── Section 2: Content Quality ───────────────────────────────────────────────

st.markdown("---")
st.subheader("Content Quality")
st.caption("IMDb score distributions reveal how each platform balances volume with quality. Tier breakdown shows the proportion of premium vs filler content.")

quality_df = comp_df.dropna(subset=["imdb_score"])

q_chart, q_table = st.columns([3, 2])

with q_chart:
    if normalized:
        fig_q = px.violin(
            quality_df,
            x="platform_display",
            y="imdb_score",
            color="platform_display",
            color_discrete_map=color_map,
            box=True,
            template=PLOTLY_TEMPLATE,
            height=CHART_HEIGHT,
            labels={"imdb_score": "IMDb Score", "platform_display": "Platform"},
            category_orders={"platform_display": display_order},
        )
    else:
        fig_q = px.box(
            quality_df,
            x="platform_display",
            y="imdb_score",
            color="platform_display",
            color_discrete_map=color_map,
            template=PLOTLY_TEMPLATE,
            height=CHART_HEIGHT,
            labels={"imdb_score": "IMDb Score", "platform_display": "Platform"},
            category_orders={"platform_display": display_order},
        )
    fig_q.update_layout(showlegend=False, xaxis_title="")
    st.plotly_chart(fig_q, use_container_width=True)

with q_table:
    with st.expander("ℹ️ Tier Definitions"):
        st.markdown(
            "- **Excellent:** IMDb 8.0 – 10.0\n"
            "- **Good:** 7.0 – 7.9\n"
            "- **Average:** 6.0 – 6.9\n"
            "- **Below Average:** 5.0 – 5.9\n"
            "- **Poor:** Below 5.0\n"
            "- **Unrated:** No IMDb score available"
        )
    st.markdown("**Quality Tier Breakdown**")
    tier_df = compute_quality_tiers(comp_df, QUALITY_TIERS, normalize=normalized)
    # Reorder columns to match display_order
    ordered_cols = [c for c in display_order if c in tier_df.columns]
    tier_df = tier_df[ordered_cols]
    tier_df.index.name = "Tier"
    if normalized:
        st.dataframe(
            tier_df.style.format("{:.1f}%"),
            use_container_width=True,
        )
    else:
        st.dataframe(tier_df, use_container_width=True)


# ── Section 3: Market Positioning ────────────────────────────────────────────

st.markdown("---")
st.subheader("Market Positioning")
st.caption("Each platform's trade-off between catalog breadth, content quality, and audience reach — where they sit in the competitive landscape.")

positioning = compute_market_positioning(comp_df)

fig_pos = px.scatter(
    positioning,
    x="total_titles",
    y="avg_imdb",
    size="total_popularity",
    color="platform_display",
    color_discrete_map=color_map,
    text="platform_display",
    template=PLOTLY_TEMPLATE,
    height=CHART_HEIGHT + 150,
    labels={
        "total_titles": "Catalog Size (titles)",
        "avg_imdb": "Average IMDb Score",
        "total_popularity": "Total TMDB Popularity",
        "platform_display": "Platform",
    },
    size_max=70,
)
fig_pos.update_traces(
    textposition="top center",
    textfont=dict(size=13, family="Arial"),
    marker=dict(line=dict(width=1.5, color="rgba(255,255,255,0.4)")),
)

# Axis ranges — extra top padding so "top center" text labels aren't clipped
x_min, x_max = positioning["total_titles"].min(), positioning["total_titles"].max()
y_min, y_max = positioning["avg_imdb"].min(), positioning["avg_imdb"].max()
x_pad = (x_max - x_min) * 0.22 or 200
y_pad_bottom = (y_max - y_min) * 0.20 or 0.25
y_pad_top = (y_max - y_min) * 0.35 or 0.45  # extra room for text above highest bubble
x_lo, x_hi = x_min - x_pad, x_max + x_pad
y_lo, y_hi = y_min - y_pad_bottom, y_max + y_pad_top

# Quadrant dividers at mean values
mean_x = positioning["total_titles"].mean()
mean_y = positioning["avg_imdb"].mean()

# Subtle quadrant background shading
_q_colors = [
    (x_lo, mean_x, mean_y, y_hi, "rgba(46,204,113,0.04)"),   # top-left: boutique+quality
    (mean_x, x_hi, mean_y, y_hi, "rgba(52,152,219,0.06)"),   # top-right: large+quality
    (x_lo, mean_x, y_lo, mean_y, "rgba(231,76,60,0.04)"),     # bottom-left: small+lower
    (mean_x, x_hi, y_lo, mean_y, "rgba(241,196,15,0.04)"),    # bottom-right: large+lower
]
for x0, x1, y0, y1, fill in _q_colors:
    fig_pos.add_shape(
        type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
        fillcolor=fill, line_width=0, layer="below",
    )

fig_pos.add_hline(y=mean_y, line_dash="dot", line_color="rgba(255,255,255,0.25)")
fig_pos.add_vline(x=mean_x, line_dash="dot", line_color="rgba(255,255,255,0.25)")

# Quadrant labels — anchored to paper corners
_qlabels = [
    (0.98, 0.97, "right", "top", "Large & High Quality"),
    (0.02, 0.97, "left", "top", "Boutique & High Quality"),
    (0.98, 0.03, "right", "bottom", "Large & Lower Quality"),
    (0.02, 0.03, "left", "bottom", "Small & Lower Quality"),
]
for qx, qy, xa, ya, qlabel in _qlabels:
    fig_pos.add_annotation(
        xref="paper", yref="paper", x=qx, y=qy, text=qlabel,
        showarrow=False, xanchor=xa, yanchor=ya,
        font=dict(size=11, color="rgba(255,255,255,0.45)"),
    )

fig_pos.update_layout(
    showlegend=False,
    xaxis=dict(range=[x_lo, x_hi], title_font=dict(size=13)),
    yaxis=dict(range=[y_lo, y_hi], title_font=dict(size=13)),
)
st.plotly_chart(fig_pos, use_container_width=True)
st.caption("Bubble size reflects total TMDB popularity. Dotted lines mark cross-platform averages.")


# ── Section 4: Content Profile ───────────────────────────────────────────────

st.markdown("---")
st.subheader("Content Profile")
st.caption("Audience targeting signals: age-rating mix, share of international productions, and how recent each catalog skews.")

cp1, cp2, cp3 = st.columns(3)

with cp1:
    st.markdown("**Family-Friendly vs Mature Content**")
    age_data = compute_age_profile(comp_df)
    # Distinctive colors for each certification so none look alike
    _cert_colors = {
        "TV-MA": "#e63946",   # strong red
        "R": "#7e0c0c",       # magenta
        "NC-17": "#7209b7",   # deep purple
        "TV-14": "#f77f00",   # orange
        "PG-13": "#fcbf49",   # gold
        "TV-PG": "#2a9d8f",   # teal
        "PG": "#2305cb",      # green
        "TV-G": "#EA0AB9",    # slate blue
        "G": "#4cc9f0",       # sky blue
        "TV-Y7": "#90be6d",   # lime
        "TV-Y": "#b5e48c",    # light green
        "Other": "#6c757d",   # gray
        "Unknown": "#AEDA0E", # dark gray
    }
    fig_age = px.bar(
        age_data,
        x="platform_display",
        y="pct",
        color="certification",
        color_discrete_map=_cert_colors,
        barmode="stack",
        template=PLOTLY_TEMPLATE,
        height=CHART_HEIGHT,
        labels={"pct": "% of Catalog", "platform_display": "Platform", "certification": "Rating"},
        category_orders={"platform_display": display_order},
    )
    fig_age.update_layout(
        legend_title_text="",
        xaxis_title="",
        legend=dict(font=dict(size=10)),
    )
    st.plotly_chart(fig_age, use_container_width=True)

with cp2:
    st.markdown("**International Content**")
    geo_data = compute_geographic_diversity(comp_df)
    # Ensure merged entity appears first
    _geo_merged = geo_data[geo_data["platform_display"] == merged_name]
    _geo_others = geo_data[geo_data["platform_display"] != merged_name]
    geo_data_ordered = pd.concat([_geo_merged, _geo_others], ignore_index=True)
    merged_intl = _geo_merged["international_pct"]
    merged_intl_val = float(merged_intl.iloc[0]) if not merged_intl.empty else None
    for _, row in geo_data_ordered.iterrows():
        delta = None
        if merged_intl_val is not None and row["platform_display"] != merged_name:
            delta = round(row["international_pct"] - merged_intl_val, 1)
            delta = f"{delta:+.1f}pp"
        st.metric(
            row["platform_display"],
            f"{row['international_pct']:.1f}% International",
            delta=delta,
        )

with cp3:
    st.markdown("**Content Recency**")
    era_data = compute_era_focus(comp_df)
    # Ensure merged entity appears first
    _era_merged = era_data[era_data["platform_display"] == merged_name]
    _era_others = era_data[era_data["platform_display"] != merged_name]
    era_data_ordered = pd.concat([_era_merged, _era_others], ignore_index=True)
    merged_year = _era_merged["median_year"]
    merged_year_val = int(merged_year.iloc[0]) if not merged_year.empty else None
    for _, row in era_data_ordered.iterrows():
        delta = None
        if merged_year_val is not None and row["platform_display"] != merged_name:
            delta = int(row["median_year"] - merged_year_val)
            delta = f"{delta:+d} yrs"
        st.metric(
            row["platform_display"],
            f"Median Year: {int(row['median_year'])}",
            delta=delta,
        )


# ── Section 5: Genre Landscape ───────────────────────────────────────────────

st.markdown("---")
st.subheader("Genre Landscape")
st.caption("Side-by-side genre presence across platforms — the competitive map of content strategy. Rows sorted by overall popularity.")

heatmap_data = compute_genre_heatmap(
    comp_df, top_n=COMPARISON_TOP_GENRES, normalize=normalized
)

if not heatmap_data.empty:
    z_values = heatmap_data.values

    # Dark colorscale — all cells dark enough for white text
    _HEATMAP_SCALE = [
        [0.0, "#0B0B1A"],   # near-black
        [0.25, "#1B1464"],   # deep indigo
        [0.5, "#3B1C71"],    # dark purple
        [0.75, "#5E2C8C"],   # rich violet
        [1.0, "#7B3FA0"],    # medium violet
    ]

    fig_heatmap = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=list(heatmap_data.columns),
            y=list(heatmap_data.index),
            colorscale=_HEATMAP_SCALE,
            colorbar_title="% of Catalog" if normalized else "Title Count",
            hoverongaps=False,
            xgap=3,
            ygap=3,
            hovertemplate="<b>%{y}</b><br>%{x}: %{z}<extra></extra>",
            showscale=True,
        )
    )

    # Add per-cell annotations — white text is always readable on this scale
    genres_list = list(heatmap_data.index)
    platforms_list = list(heatmap_data.columns)
    for row_idx, genre in enumerate(genres_list):
        row_values = z_values[row_idx]
        max_val = np.nanmax(row_values)
        for col_idx, platform in enumerate(platforms_list):
            val = z_values[row_idx][col_idx]
            label = f"{val:.1f}%" if normalized else str(int(val))
            if val == max_val and max_val > 0:
                label += " ★"
            fig_heatmap.add_annotation(
                x=platform, y=genre, text=label,
                showarrow=False,
                font=dict(size=12, color="#ffffff", family="Arial"),
            )

    fig_heatmap.update_layout(
        template=PLOTLY_TEMPLATE,
        height=max(500, len(heatmap_data) * 40),
        xaxis=dict(side="top", tickfont=dict(size=13)),
        yaxis=dict(tickfont=dict(size=12), autorange="reversed"),
        margin=dict(l=130, t=40, r=10, b=10),
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
    st.caption("★ = genre leader per row")

    # ── Genre Deep Dive ──────────────────────────────────────────────────────

    st.markdown("---")
    st.subheader("Genre Deep Dive")
    st.caption("Drill into any genre to compare platform depth, average quality, and the highest-rated titles.")

    # Genre options from heatmap (already display-fixed), map back to raw keys
    _DISPLAY_TO_RAW = {"Documentary": "documentation", "Sci-Fi": "scifi"}
    genre_options = list(heatmap_data.index)
    genre_display_to_raw = {
        g: _DISPLAY_TO_RAW.get(g, g.lower()) for g in genre_options
    }

    # Resolve default
    current_genre = st.session_state.get("comp_selected_genre")
    raw_to_display = {v: k for k, v in genre_display_to_raw.items()}
    current_display = raw_to_display.get(current_genre)
    if current_display and current_display in genre_options:
        default_idx = genre_options.index(current_display)
    else:
        default_idx = 0

    selected_genre_display = st.selectbox(
        "Select a genre to explore",
        options=genre_options,
        index=default_idx,
    )
    selected_genre_raw = genre_display_to_raw[selected_genre_display]
    st.session_state["comp_selected_genre"] = selected_genre_raw

    drilldown = compute_genre_drilldown(comp_df, selected_genre_raw)
    dd_stats = drilldown["stats"]
    top_titles = drilldown["top_titles"]

    # Track expanded card data to render full-width below columns
    _expanded_card_data = None

    # Render one column per platform
    n_platforms = len(dd_stats)
    if n_platforms > 0:
        cols = st.columns(n_platforms)
        for i, (_, row) in enumerate(dd_stats.iterrows()):
            pname = row["platform_display"]
            with cols[i]:
                if normalized:
                    st.metric(f"{pname}", f"{row['pct_of_catalog']}% of catalog")
                    st.caption(f"{int(row['count']):,} titles")
                else:
                    st.metric(f"{pname}", f"{int(row['count']):,} titles")
                    st.caption(f"{row['pct_of_catalog']}% of catalog")
                st.metric("Avg IMDb", f"{row['avg_imdb']:.2f}")

                if pname in top_titles and not top_titles[pname].empty:
                    st.markdown("**Top Titles**")
                    for t_idx, (_, t) in enumerate(top_titles[pname].iterrows()):
                        votes_str = format_votes(t.get("imdb_votes"))
                        title_id = t.get("id", "")
                        card_key = f"dd_{pname}_{t_idx}"
                        expanded_key = st.session_state.get("comp_expanded_title")
                        is_expanded = expanded_key == (pname, title_id)

                        # Card with hover animation
                        border_color = CARD_ACCENT if is_expanded else CARD_BORDER
                        border_width = "2px" if is_expanded else "1px"
                        st.markdown(
                            f'<div style="background:{CARD_BG};border:{border_width} solid '
                            f'{border_color};'
                            f'border-radius:8px;padding:8px 10px;margin-bottom:2px;'
                            f'transition:transform 0.15s ease,box-shadow 0.15s ease;cursor:pointer;"'
                            f' onmouseover="this.style.transform=\'translateY(-2px)\';'
                            f'this.style.boxShadow=\'0 4px 12px rgba(0,0,0,0.3)\'"'
                            f' onmouseout="this.style.transform=\'none\';'
                            f'this.style.boxShadow=\'none\'">'
                            f'<span style="color:{CARD_TEXT};font-size:0.85em;font-weight:600;">'
                            f'{t["title"]}</span><br/>'
                            f'<span style="color:{CARD_TEXT_MUTED};font-size:0.75em;">'
                            f'{t["type"]} &middot; {int(t["release_year"])} &middot; '
                            f'IMDb {t["imdb_score"]:.1f} &middot; {votes_str} votes'
                            f'</span></div>',
                            unsafe_allow_html=True,
                        )

                        # Toggle button
                        btn_label = "Close" if is_expanded else "Details"
                        if st.button(btn_label, key=card_key, use_container_width=True):
                            if is_expanded:
                                st.session_state["comp_expanded_title"] = None
                            else:
                                st.session_state["comp_expanded_title"] = (pname, title_id)
                            st.rerun()

                        # Store expanded card data for full-width rendering below
                        if is_expanded and title_id:
                            _expanded_card_data = (t, title_id, pname)
                else:
                    st.caption("No rated titles in this genre")

    # Render expanded card full-width below the columns
    if _expanded_card_data is not None:
        _exp_row, _exp_id, _exp_platform = _expanded_card_data
        _render_expanded_card(_exp_row, _exp_id, _exp_platform)

    # ── Strategic Insights ────────────────────────────────────────────────────

    st.markdown("---")
    st.subheader("Strategic Insights")
    st.caption("High-level competitive assessment per competitor — what the data implies for the merged entity's market position.")

    # Use mode-appropriate data for insights
    _insights_heatmap = compute_genre_heatmap(
        comp_df, top_n=COMPARISON_TOP_GENRES, normalize=normalized
    )
    _insights_tiers = compute_quality_tiers(comp_df, QUALITY_TIERS, normalize=normalized)
    _insights_tier_ordered = _insights_tiers[
        [c for c in display_order if c in _insights_tiers.columns]
    ]
    insights = compute_strategic_insights(
        comp_df, _insights_heatmap, _insights_tier_ordered
    )

    if insights:
        insight_cols = st.columns(len(insights))
        for col_idx, (comp_name, data) in enumerate(insights.items()):
            with insight_cols[col_idx]:
                # Determine platform color for header accent
                _p_key = [k for k, v in PLATFORMS.items() if v["name"] == comp_name]
                _p_color = PLATFORMS[_p_key[0]]["color"] if _p_key else CARD_BORDER

                # Generate prose insights
                prose = _build_insight_prose(comp_name, data, merged_name)

                # Render as a styled card with platform accent
                prose_html = "".join(
                    f'<p style="margin:0 0 10px 0;line-height:1.65;">{p}</p>'
                    for p in prose
                )
                st.markdown(
                    f'<div style="background:{CARD_BG};border-left:4px solid {_p_color};'
                    f'border-radius:6px;padding:14px 16px;">'
                    f'<div style="font-size:1.05em;font-weight:700;color:{CARD_TEXT};'
                    f'margin-bottom:10px;">vs {comp_name}</div>'
                    f'<div style="font-size:0.85em;color:{CARD_TEXT};">'
                    f'{prose_html}</div></div>',
                    unsafe_allow_html=True,
                )
    else:
        st.info("Select competitors to see strategic insights.")

else:
    st.info("No genre data available with current filters.")


# ── footer ────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "Hypothetical merger for academic analysis. "
    "Data is a snapshot (mid-2023). "
    "All insights are illustrative, not prescriptive."
)


