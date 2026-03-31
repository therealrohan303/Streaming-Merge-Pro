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
from src.ui.badges import page_header_html, platform_badges_html, section_header_html
from src.ui.filters import apply_filters, render_sidebar_filters
from src.ui.session import init_session_state

_MERGED_COLOR = PLATFORMS["merged"]["color"]

# ── page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title=f"Platform Comparisons | {PAGE_TITLE}",
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state=INITIAL_SIDEBAR_STATE,
)

init_session_state()

st.session_state.setdefault("comp_competitors", ["disney", "prime"])
st.session_state.setdefault("comp_normalized", False)
st.session_state.setdefault("comp_selected_genre", None)
st.session_state.setdefault("comp_expanded_title", None)


# ── helpers ──────────────────────────────────────────────────────────────────

def _render_expanded_card(title_row, title_id: str, platform_name: str = ""):
    """Render an expanded detail card matching the Explore Catalog detail panel style."""
    _enr_detail = load_enriched_titles()
    _enr_row = _enr_detail[_enr_detail["id"] == title_id] if not _enr_detail.empty else pd.DataFrame()
    _has_enr = not _enr_row.empty
    _enr = _enr_row.iloc[0] if _has_enr else None

    def _mc(label: str, value: str) -> str:
        return (
            f'<div style="padding:8px 0;">'
            f'<div style="color:{CARD_TEXT_MUTED};font-size:0.72em;text-transform:uppercase;'
            f'letter-spacing:0.04em;margin-bottom:3px;">{label}</div>'
            f'<div style="font-weight:600;color:{CARD_TEXT};font-size:0.92em;">{value}</div>'
            f'</div>'
        )

    with st.container(border=True):
        # Poster + title row
        _poster = None
        if _has_enr and "poster_url" in _enr_row.columns:
            _poster = _enr.get("poster_url")

        if _poster and str(_poster) != "nan":
            _pcol, _hcol = st.columns([1, 5])
            with _pcol:
                st.image(str(_poster), width=120)
            with _hcol:
                st.subheader(title_row.get("title", "Unknown"))
                if platform_name:
                    _p_key = next((k for k, v in PLATFORMS.items() if v["name"] == platform_name), None)
                    if _p_key:
                        st.markdown(platform_badges_html(_p_key), unsafe_allow_html=True)
        else:
            st.subheader(title_row.get("title", "Unknown"))
            if platform_name:
                _p_key = next((k for k, v in PLATFORMS.items() if v["name"] == platform_name), None)
                if _p_key:
                    st.markdown(platform_badges_html(_p_key), unsafe_allow_html=True)

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

        # Metadata row 1: Type | Year | IMDb | Rating
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(_mc("Type", title_row.get("type", "N/A")), unsafe_allow_html=True)
        with m2:
            year = title_row.get("release_year")
            st.markdown(_mc("Year", str(int(year)) if year and year == year else "N/A"), unsafe_allow_html=True)
        with m3:
            imdb = title_row.get("imdb_score")
            imdb_str = f"{imdb:.1f}" if imdb and imdb == imdb else "N/A"
            st.markdown(_mc("IMDb", imdb_str), unsafe_allow_html=True)
        with m4:
            cert = title_row.get("age_certification")
            cert_str = cert if cert and str(cert) != "nan" else "N/A"
            st.markdown(_mc("Rating", cert_str), unsafe_allow_html=True)

        # Metadata row 2: Runtime | Votes | Quality Score | Box Office
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            rt = title_row.get("runtime")
            rt_str = f"{int(rt)} min" if rt and rt == rt else "N/A"
            st.markdown(_mc("Runtime", rt_str), unsafe_allow_html=True)
        with r2:
            votes = title_row.get("imdb_votes")
            votes_display = format_votes(votes) if votes and votes == votes else "N/A"
            st.markdown(_mc("Votes", votes_display), unsafe_allow_html=True)
        with r3:
            _tmp = pd.DataFrame([title_row])
            if "tmdb_popularity" not in _tmp.columns:
                _tmp["tmdb_popularity"] = 0
            qs = compute_quality_score(_tmp).iloc[0]
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
        with r4:
            if _has_enr and "box_office_usd" in _enr_row.columns:
                _bo = _enr.get("box_office_usd")
                if _bo and str(_bo) != "nan" and _bo > 0:
                    _bo_str = f"${_bo/1e9:.1f}B" if _bo >= 1e9 else f"${_bo/1e6:.0f}M" if _bo >= 1e6 else f"${_bo:,.0f}"
                    st.markdown(_mc("Box Office", _bo_str), unsafe_allow_html=True)
                else:
                    st.markdown(_mc("Box Office", "N/A"), unsafe_allow_html=True)
            else:
                st.markdown(_mc("Box Office", "—"), unsafe_allow_html=True)

        # Genre pills
        genres = title_row.get("genres")
        if isinstance(genres, list) and genres:
            pills = " ".join(
                f'<span style="background:{CARD_BORDER};color:{CARD_TEXT};'
                f'padding:3px 10px;border-radius:12px;font-size:0.82em;margin-right:3px;">'
                f'{g.title()}</span>'
                for g in genres
            )
            st.markdown(f'<div style="margin:8px 0;">{pills}</div>', unsafe_allow_html=True)

        # Franchise (if enriched)
        if _has_enr and "collection_name" in _enr_row.columns:
            _coll = _enr.get("collection_name")
            if _coll and str(_coll) != "nan":
                st.caption(f"Franchise: {_coll}")

        # Description + Cast side by side
        st.divider()
        desc_col, cast_col = st.columns([3, 2])

        with desc_col:
            desc = title_row.get("description")
            if desc and str(desc) != "nan":
                desc_str = str(desc)
                st.markdown(
                    f'<div style="font-size:0.9em;color:{CARD_TEXT};line-height:1.6;">{desc_str}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.caption("No description available.")

        with cast_col:
            credits_df = load_all_platforms_credits()
            title_credits = credits_df[credits_df["title_id"] == title_id]
            if not title_credits.empty:
                with st.expander("Cast & Crew", expanded=True):
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
                        actor_list = actors.head(8).reset_index(drop=True)
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
                        if len(actors) > 8:
                            st.caption(f"+{len(actors) - 8} more")
            else:
                st.caption("No cast/crew data available.")

    st.markdown("<div style='margin-top:4px;'></div>", unsafe_allow_html=True)


def _build_insight_prose(comp_name: str, data: dict, merged_name: str) -> list[str]:
    """Generate 2–4 data-driven strategic insights for a competitor.

    Each insight is unique to the actual competitive data for this pair — no boilerplate.
    """
    s = data["summary"]
    insights = []

    comp_vol = s["comp_vol"]
    merged_vol = s["merged_vol"]
    comp_imdb = s["comp_avg_imdb"]
    merged_imdb = s["merged_avg_imdb"]
    vol_diff = comp_vol - merged_vol
    qual_diff = merged_imdb - comp_imdb

    # 1. Scale / Quality framing — specific numbers, no generic fallback
    vol_pct = round(abs(vol_diff) / max(merged_vol, 1) * 100)
    if vol_diff > 0 and qual_diff > 0.15:
        insights.append(
            f"{comp_name} operates at larger scale ({comp_vol:,} vs {merged_vol:,} titles, "
            f"+{vol_pct}% more), but the merged catalog holds a meaningful quality edge "
            f"(avg IMDb {merged_imdb:.2f} vs {comp_imdb:.2f}). "
            f"Volume disadvantage is offset by a higher-rated catalog."
        )
    elif vol_diff > 0 and qual_diff < -0.15:
        insights.append(
            f"{comp_name} leads on both catalog depth ({comp_vol:,} vs {merged_vol:,} titles) "
            f"and average quality (IMDb {comp_imdb:.2f} vs {merged_imdb:.2f}). "
            f"This is a dual-front competitive challenge requiring both acquisition and quality improvement."
        )
    elif vol_diff < 0 and qual_diff > 0.15:
        insights.append(
            f"The merged catalog dominates {comp_name} on catalog size "
            f"({merged_vol:,} vs {comp_vol:,} titles, +{vol_pct}% larger) "
            f"and quality (IMDb {merged_imdb:.2f} vs {comp_imdb:.2f}). "
            f"The merger strengthens a pre-existing advantage on both dimensions."
        )
    elif vol_diff < 0 and qual_diff < -0.15:
        insights.append(
            f"The merged catalog leads in catalog volume ({merged_vol:,} vs {comp_vol:,} titles), "
            f"but {comp_name} maintains a quality premium (IMDb {comp_imdb:.2f} vs {merged_imdb:.2f}). "
            f"This suggests {comp_name} pursues a selective, prestige-focused curation strategy."
        )
    else:
        # Near-parity — highlight the most distinctive difference available
        closer_metric = "quality" if abs(qual_diff) < 0.05 else "volume"
        insights.append(
            f"The merged catalog and {comp_name} are closely matched in {closer_metric} "
            f"({merged_vol:,} vs {comp_vol:,} titles; IMDb {merged_imdb:.2f} vs {comp_imdb:.2f}). "
            f"The competitive battle will be fought in genre execution and original content."
        )

    # 2. Genre differentiation — contextual, not a template
    their_genres = [
        x["category"] for x in data["their_strengths"]
        if "tier" not in x["category"].lower()
    ][:3]
    our_genres = [
        x["category"] for x in data["our_strengths"]
        if "tier" not in x["category"].lower()
    ][:3]

    if their_genres and our_genres:
        # Check if their strengths overlap with our top genres (competitive threat vs niche)
        overlap = set(their_genres) & set(our_genres)
        if overlap:
            insights.append(
                f"{comp_name} challenges directly in {', '.join(our_genres)} — "
                f"genres where the merged catalog also has depth. "
                f"The merged catalog counters with stronger positions in {', '.join(our_genres[:2])}."
            )
        else:
            insights.append(
                f"Genre footprints are largely complementary: {comp_name} concentrates in "
                f"{', '.join(their_genres)}, while the merged catalog leads in {', '.join(our_genres)}. "
                f"Audience overlap is likely limited to general drama and action."
            )
    elif their_genres:
        # Determine if these are high-volume or niche genres
        insights.append(
            f"{comp_name} holds targeted advantages in {', '.join(their_genres)} — "
            f"areas where the merged catalog has an identified content gap worth addressing "
            f"through licensing or original production."
        )
    elif our_genres:
        insights.append(
            f"The merged catalog holds dominant positions across major genres "
            f"(including {', '.join(our_genres[:2])}). "
            f"{comp_name} does not present a concentrated genre-level challenge."
        )

    # 3. Battlegrounds — action-oriented, ranked by stakes
    if data["battlegrounds"]:
        bgs = data["battlegrounds"][:3]
        # Frame based on number of battleground genres
        if len(bgs) == 1:
            insights.append(
                f"{bgs[0]} is the highest-stakes contested territory between the merged catalog "
                f"and {comp_name}. Original content and recommendation quality in this genre "
                f"will be the primary subscriber acquisition lever."
            )
        else:
            insights.append(
                f"Parity exists across {', '.join(bgs)} — where neither side has a decisive edge. "
                f"These genres represent the front lines where exclusive originals "
                f"will have the highest competitive impact."
            )

    # 4. Platform-specific insight based on identity signals
    comp_key = None
    for k, v in PLATFORMS.items():
        if v["name"] == comp_name:
            comp_key = k
            break

    if comp_key == "appletv":
        title_ratio = round(merged_vol / max(comp_vol, 1))
        insights.append(
            f"Apple TV+ operates with {title_ratio}× fewer titles than the merged catalog — "
            f"a deliberate prestige-over-volume strategy. Its IMDb {comp_imdb:.2f} average "
            f"{'exceeds' if comp_imdb > merged_imdb else 'trails'} the merged catalog's "
            f"{merged_imdb:.2f}, "
            f"{'suggesting quality-per-title is a genuine differentiator' if comp_imdb > merged_imdb else 'despite this, quality output per title has not systematically outperformed'}."
        )
    elif comp_key == "disney":
        insights.append(
            f"Disney+ derives competitive strength from franchise concentration "
            f"(Marvel, Star Wars, Pixar) that is not fully captured in raw genre counts. "
            f"The merger's scale advantage ({merged_vol:,} vs {comp_vol:,}) is less relevant "
            f"against a catalog driven by IP loyalty rather than breadth."
        )
    elif comp_key == "prime":
        insights.append(
            f"Prime Video's catalog ({comp_vol:,} titles) reflects an aggregator model — "
            f"breadth across genres rather than depth in any single area. "
            f"The merged catalog's tighter, quality-focused catalog (IMDb {merged_imdb:.2f}) "
            f"offers a differentiated value proposition for subscribers who prioritize curation."
        )
    elif comp_key == "paramount":
        if merged_vol > comp_vol * 1.5:
            insights.append(
                f"Paramount+'s catalog of {comp_vol:,} titles is significantly smaller than the "
                f"merged catalog's {merged_vol:,}, limiting its breadth. "
                f"Its competitive position relies on sports rights and CBS library content "
                f"that lies outside the scope of this title-level analysis."
            )

    return insights


# ── sidebar ───────────────────────────────────────────────────────────────────

raw_df = load_all_platforms_titles()
filters = render_sidebar_filters(raw_df)
df = apply_filters(raw_df, filters)

# ── page header ──────────────────────────────────────────────────────────────

st.markdown(
    page_header_html(
        "Platform Comparisons",
        "Benchmark the merged Netflix + Max catalog against streaming competitors",
    ),
    unsafe_allow_html=True,
)

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

if not selected_keys:
    st.warning("At least one competitor is required. Defaulting to Prime Video.")
    st.session_state["comp_competitors"] = ["prime"]
    selected_keys = ["prime"]

comp_df = build_comparison_df(df, selected_keys)
color_map = {PLATFORMS["merged"]["name"]: PLATFORMS["merged"]["color"]}
for k in selected_keys:
    color_map[PLATFORMS[k]["name"]] = PLATFORMS[k]["color"]
merged_name = PLATFORMS["merged"]["name"]
display_order = [merged_name] + [PLATFORMS[k]["name"] for k in sorted(selected_keys)]

# ── Quick Comparison — styled 3-column cards ──────────────────────────────────

quick = compute_quick_comparison(comp_df)
vol_label = quick["volume_leader"]
qual_label = quick["quality_leader"]
genre_leads = quick["merged_genre_leads"]
total_genres_n = quick["total_genres"]

# Identify the platform color for each card's accent
def _platform_color(name: str) -> str:
    for k, v in PLATFORMS.items():
        if v["name"] == name:
            return v["color"]
    return CARD_BORDER

qc1, qc2, qc3 = st.columns(3)

with qc1:
    _vol_color = _platform_color(vol_label)
    st.markdown(
        f'<div style="background:{CARD_BG};border:1px solid {CARD_BORDER};'
        f'border-left:4px solid {_vol_color};border-radius:6px;padding:12px 16px;">'
        f'<div style="font-size:0.75em;color:{CARD_TEXT_MUTED};text-transform:uppercase;'
        f'letter-spacing:0.04em;margin-bottom:6px;">Volume Leader</div>'
        f'<div style="font-weight:700;color:{CARD_TEXT};font-size:1.05em;">{vol_label}</div>'
        f'<div style="color:{CARD_TEXT_MUTED};font-size:0.82em;margin-top:2px;">'
        f'{quick["volume_count"]:,} titles</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with qc2:
    _qual_color = _platform_color(qual_label)
    st.markdown(
        f'<div style="background:{CARD_BG};border:1px solid {CARD_BORDER};'
        f'border-left:4px solid {_qual_color};border-radius:6px;padding:12px 16px;">'
        f'<div style="font-size:0.75em;color:{CARD_TEXT_MUTED};text-transform:uppercase;'
        f'letter-spacing:0.04em;margin-bottom:6px;">Quality Leader</div>'
        f'<div style="font-weight:700;color:{CARD_TEXT};font-size:1.05em;">{qual_label}</div>'
        f'<div style="color:{CARD_TEXT_MUTED};font-size:0.82em;margin-top:2px;">'
        f'Avg IMDb {quick["quality_avg"]:.2f}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with qc3:
    genre_sub = f"Leads {genre_leads}/{total_genres_n} top genres by volume"
    st.markdown(
        f'<div style="background:{CARD_BG};border:1px solid {CARD_BORDER};'
        f'border-left:4px solid {_MERGED_COLOR};border-radius:6px;padding:12px 16px;">'
        f'<div style="font-size:0.75em;color:{CARD_TEXT_MUTED};text-transform:uppercase;'
        f'letter-spacing:0.04em;margin-bottom:6px;">Genre Leader</div>'
        f'<div style="font-weight:700;color:{CARD_TEXT};font-size:1.05em;">{merged_name}</div>'
        f'<div style="color:{CARD_TEXT_MUTED};font-size:0.82em;margin-top:2px;">{genre_sub}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
with st.expander("About These Comparisons"):
    st.markdown(
        f"**Merged catalog:** Netflix and Max catalogs combined — titles on both "
        f"platforms counted once.\n\n"
        f"**Normalized mode** converts raw counts to % of each platform's catalog, "
        f"enabling fair comparison across platforms of different sizes.\n\n"
        f"**Quality score** = 85% Bayesian-adjusted IMDb + 15% normalized TMDB "
        f"popularity.\n\n"
        f"**Genre counting:** Multi-genre titles count toward each genre they belong to."
    )

# ── Section 1: Catalog Volume ────────────────────────────────────────────────

st.divider()
st.markdown(
    section_header_html(
        "Catalog Volume",
        "Total library size per platform, split by movies and shows — a baseline measure of competitive scale.",
    ),
    unsafe_allow_html=True,
)

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
            labels={"pct": "", "platform_display": "Platform", "type": "Type"},
            category_orders={"platform_display": display_order},
            text="pct",
        )
        fig_vol.update_traces(texttemplate="%{text:.1f}%", textposition="inside")
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

    # Prestige Score column
    _enr_for_prestige = load_enriched_titles()
    _prestige_coverage = (
        _enr_for_prestige["award_wins"].notna().mean()
        if "award_wins" in _enr_for_prestige.columns
        else 0
    )
    if _prestige_coverage >= WIKIDATA_COMPARISON_MIN_COVERAGE and "award_wins" in _enr_for_prestige.columns:
        prestige_scores = []
        for _, row in display_summary.iterrows():
            plat_name = row["Platform"]
            plat_key = next((k for k, v in PLATFORMS.items() if v.get("name") == plat_name), None)
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
        display_summary = display_summary.copy()
        display_summary["Prestige Score"] = prestige_scores
    else:
        display_summary = display_summary.copy()
        display_summary["Prestige Score"] = "—"

    # Pre-format numeric columns before styling to prevent raw float display
    display_summary = display_summary.copy()
    for _int_col in ["Total", "Movies", "Shows"]:
        if _int_col in display_summary.columns:
            display_summary[_int_col] = display_summary[_int_col].apply(
                lambda x: f"{int(x):,}" if isinstance(x, (int, float)) and x == x else "—"
            )
    if "Avg IMDb" in display_summary.columns:
        display_summary["Avg IMDb"] = display_summary["Avg IMDb"].apply(
            lambda x: f"{x:.2f}" if isinstance(x, (int, float)) and x == x else "—"
        )
    if "Prestige Score" in display_summary.columns:
        display_summary["Prestige Score"] = display_summary["Prestige Score"].apply(
            lambda x: f"{x:.1f}" if isinstance(x, (int, float)) and x == x else str(x)
        )

    # Style merged row with teal background
    def _highlight_merged(row):
        if row["Platform"] == merged_name:
            return [f"background-color:rgba(0,137,123,0.12);font-weight:600;"] * len(row)
        return [""] * len(row)

    st.dataframe(
        display_summary.style.apply(_highlight_merged, axis=1),
        use_container_width=True,
        hide_index=True,
    )
    if _prestige_coverage >= WIKIDATA_COMPARISON_MIN_COVERAGE:
        st.caption("ℹ️ Prestige Score = award wins per 1,000 titles (Wikidata)")
    else:
        st.caption(
            f"Prestige Score unavailable (Wikidata coverage: {_prestige_coverage:.0%}, need ≥{WIKIDATA_COMPARISON_MIN_COVERAGE:.0%})"
        )


# ── Section 2: Content Quality ───────────────────────────────────────────────

st.divider()
st.markdown(
    section_header_html(
        "Content Quality",
        "IMDb score distributions reveal how each platform balances volume with quality.",
    ),
    unsafe_allow_html=True,
)

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
    fig_q.update_layout(
        showlegend=False,
        xaxis_title="",
        yaxis=dict(range=[0, 10], title="IMDb Score"),
    )
    # Reference lines
    fig_q.add_hline(
        y=7.0,
        line_dash="dot",
        line_color="rgba(255,255,255,0.28)",
        annotation_text="Good (7.0)",
        annotation_position="right",
        annotation_font=dict(size=10, color="rgba(255,255,255,0.45)"),
    )
    fig_q.add_hline(
        y=8.0,
        line_dash="dot",
        line_color="rgba(255,255,255,0.38)",
        annotation_text="Excellent (8.0)",
        annotation_position="right",
        annotation_font=dict(size=10, color="rgba(255,255,255,0.55)"),
    )
    st.plotly_chart(fig_q, use_container_width=True)

with q_table:
    with st.expander("Tier Definitions"):
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
    ordered_cols = [c for c in display_order if c in tier_df.columns]
    tier_df = tier_df[ordered_cols]
    tier_df.index.name = "Tier"
    if normalized:
        styled_tier = tier_df.style.background_gradient(
            cmap="Blues", axis=None
        ).format("{:.1f}%")
    else:
        styled_tier = tier_df.style.background_gradient(
            cmap="Blues", axis=None
        ).format("{:,.0f}")
    st.dataframe(styled_tier, use_container_width=True)


# ── Section 3: Market Positioning ────────────────────────────────────────────

st.divider()
st.markdown(
    section_header_html(
        "Market Positioning",
        "Each platform's trade-off between catalog breadth, content quality, and audience reach.",
    ),
    unsafe_allow_html=True,
)

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
    cliponaxis=False,
)

x_min, x_max = positioning["total_titles"].min(), positioning["total_titles"].max()
y_min, y_max = positioning["avg_imdb"].min(), positioning["avg_imdb"].max()
x_pad_left = (x_max - x_min) * 0.15 or 150
x_pad_right = (x_max - x_min) * 0.32 or 300  # extra right padding to prevent clipping
y_pad_bottom = (y_max - y_min) * 0.20 or 0.25
y_pad_top = (y_max - y_min) * 0.38 or 0.48
x_lo, x_hi = x_min - x_pad_left, x_max + x_pad_right
y_lo, y_hi = y_min - y_pad_bottom, y_max + y_pad_top

mean_x = positioning["total_titles"].mean()
mean_y = positioning["avg_imdb"].mean()

_q_colors = [
    (x_lo, mean_x, mean_y, y_hi, "rgba(46,204,113,0.04)"),
    (mean_x, x_hi, mean_y, y_hi, "rgba(52,152,219,0.06)"),
    (x_lo, mean_x, y_lo, mean_y, "rgba(231,76,60,0.04)"),
    (mean_x, x_hi, y_lo, mean_y, "rgba(241,196,15,0.04)"),
]
for x0, x1, y0, y1, fill in _q_colors:
    fig_pos.add_shape(
        type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
        fillcolor=fill, line_width=0, layer="below",
    )

fig_pos.add_hline(y=mean_y, line_dash="dot", line_color="rgba(255,255,255,0.25)")
fig_pos.add_vline(x=mean_x, line_dash="dot", line_color="rgba(255,255,255,0.25)")

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
        font=dict(size=11, color="rgba(255,255,255,0.40)"),
    )

fig_pos.update_layout(
    showlegend=False,
    xaxis=dict(range=[x_lo, x_hi], title_font=dict(size=13)),
    yaxis=dict(range=[y_lo, y_hi], title_font=dict(size=13)),
)
st.plotly_chart(fig_pos, use_container_width=True)
st.caption("Bubble size reflects total TMDB popularity. Dotted lines mark cross-platform averages.")


# ── Section 4: Content Profile ───────────────────────────────────────────────

st.divider()
st.markdown(
    section_header_html(
        "Content Profile",
        "Audience targeting signals: age-rating mix, international share, and catalog recency.",
    ),
    unsafe_allow_html=True,
)

cp1, cp2, cp3 = st.columns(3)

with cp1:
    st.markdown(f'<div style="font-weight:600;color:{CARD_TEXT};margin-bottom:6px;">Family-Friendly vs Mature Content</div>', unsafe_allow_html=True)
    age_data = compute_age_profile(comp_df)
    # Semantic color palette: mature=red, teen=amber, family=green, unknown=gray
    _cert_colors = {
        "TV-MA": "#C62828",
        "R": "#EF5350",
        "NC-17": "#B71C1C",
        "TV-14": "#EF6C00",
        "PG-13": "#FFA726",
        "TV-PG": "#66BB6A",
        "PG": "#43A047",
        "TV-G": "#2E7D32",
        "G": "#1B5E20",
        "TV-Y7": "#81C784",
        "TV-Y": "#A5D6A7",
        "NR": "#616161",
        "Unknown": "#424242",
        "Other": "#757575",
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
        legend=dict(font=dict(size=9), orientation="v"),
    )
    st.plotly_chart(fig_age, use_container_width=True)

with cp2:
    st.markdown(f'<div style="font-weight:600;color:{CARD_TEXT};margin-bottom:6px;">International Content</div>', unsafe_allow_html=True)
    geo_data = compute_geographic_diversity(comp_df)
    _geo_merged = geo_data[geo_data["platform_display"] == merged_name]
    _geo_others = geo_data[geo_data["platform_display"] != merged_name]
    geo_data_ordered = pd.concat([_geo_merged, _geo_others], ignore_index=True)
    merged_intl_val = float(_geo_merged["international_pct"].iloc[0]) if not _geo_merged.empty else None

    for _, row in geo_data_ordered.iterrows():
        pname = row["platform_display"]
        intl_pct = row["international_pct"]
        is_merged = pname == merged_name
        delta_str = ""
        delta_color = CARD_TEXT_MUTED
        strategic_note = ""

        if not is_merged and merged_intl_val is not None:
            delta = round(intl_pct - merged_intl_val, 1)
            sign = "+" if delta >= 0 else ""
            delta_str = f"{sign}{delta:.1f}pp vs merged"
            delta_color = "#2ecc71" if delta > 5 else "#e74c3c" if delta < -5 else CARD_TEXT_MUTED
            strategic_note = (
                "More international-focused catalog"
                if delta > 5
                else "More US-centric catalog"
                if delta < -5
                else "Similar international footprint"
            )

        _p_color = _platform_color(pname)
        st.markdown(
            f'<div style="background:{CARD_BG};border:1px solid {CARD_BORDER};'
            f'border-left:3px solid {_p_color};border-radius:5px;'
            f'padding:8px 12px;margin-bottom:6px;">'
            f'<div style="font-size:0.75em;color:{CARD_TEXT_MUTED};margin-bottom:2px;">{pname}</div>'
            f'<div style="font-weight:600;color:{CARD_TEXT};">{intl_pct:.1f}% International</div>'
            + (f'<div style="color:{delta_color};font-size:0.78em;">{delta_str}</div>' if delta_str else "")
            + (f'<div style="color:{CARD_TEXT_MUTED};font-size:0.72em;margin-top:2px;">{strategic_note}</div>' if strategic_note else "")
            + f'</div>',
            unsafe_allow_html=True,
        )

with cp3:
    st.markdown(f'<div style="font-weight:600;color:{CARD_TEXT};margin-bottom:6px;">Content Recency</div>', unsafe_allow_html=True)
    era_data = compute_era_focus(comp_df)
    _era_merged = era_data[era_data["platform_display"] == merged_name]
    _era_others = era_data[era_data["platform_display"] != merged_name]
    era_data_ordered = pd.concat([_era_merged, _era_others], ignore_index=True)
    merged_year_val = int(_era_merged["median_year"].iloc[0]) if not _era_merged.empty else None

    for _, row in era_data_ordered.iterrows():
        pname = row["platform_display"]
        med_year = int(row["median_year"])
        is_merged = pname == merged_name
        delta_str = ""
        delta_color = CARD_TEXT_MUTED
        strategic_note = ""

        if not is_merged and merged_year_val is not None:
            delta = med_year - merged_year_val
            sign = "+" if delta >= 0 else ""
            delta_str = f"{sign}{delta} yrs vs merged"
            delta_color = "#2ecc71" if delta > 2 else "#e74c3c" if delta < -2 else CARD_TEXT_MUTED
            strategic_note = (
                "Skews toward newer content"
                if delta > 2
                else "Deeper library catalog"
                if delta < -2
                else "Similar recency profile"
            )

        _p_color = _platform_color(pname)
        st.markdown(
            f'<div style="background:{CARD_BG};border:1px solid {CARD_BORDER};'
            f'border-left:3px solid {_p_color};border-radius:5px;'
            f'padding:8px 12px;margin-bottom:6px;">'
            f'<div style="font-size:0.75em;color:{CARD_TEXT_MUTED};margin-bottom:2px;">{pname}</div>'
            f'<div style="font-weight:600;color:{CARD_TEXT};">Median Year: {med_year}</div>'
            + (f'<div style="color:{delta_color};font-size:0.78em;">{delta_str}</div>' if delta_str else "")
            + (f'<div style="color:{CARD_TEXT_MUTED};font-size:0.72em;margin-top:2px;">{strategic_note}</div>' if strategic_note else "")
            + f'</div>',
            unsafe_allow_html=True,
        )


# ── Section 5: Genre Landscape ───────────────────────────────────────────────

st.divider()
st.markdown(
    section_header_html(
        "Genre Landscape",
        "Side-by-side genre presence across platforms — the competitive map of content strategy.",
    ),
    unsafe_allow_html=True,
)

heatmap_data = compute_genre_heatmap(comp_df, top_n=COMPARISON_TOP_GENRES, normalize=normalized)

if not heatmap_data.empty:
    z_values = heatmap_data.values.astype(float)

    _HEATMAP_SCALE = [
        [0.0, "#0B0B1A"],
        [0.25, "#1B1464"],
        [0.5, "#3B1C71"],
        [0.75, "#5E2C8C"],
        [1.0, "#7B3FA0"],
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

    # Alternating row subtle shading
    genres_list = list(heatmap_data.index)
    platforms_list = list(heatmap_data.columns)
    for i in range(0, len(genres_list), 2):
        fig_heatmap.add_shape(
            type="rect",
            x0=-0.5, x1=len(platforms_list) - 0.5,
            y0=i - 0.5, y1=i + 0.5,
            fillcolor="rgba(255,255,255,0.03)",
            line_width=0,
            layer="above",
        )

    # Cell annotations — gold bold for row leader
    for row_idx, genre in enumerate(genres_list):
        row_values = z_values[row_idx]
        max_val = np.nanmax(row_values) if len(row_values) > 0 else 0
        for col_idx, platform in enumerate(platforms_list):
            val = z_values[row_idx][col_idx]
            label = f"{val:.1f}%" if normalized else str(int(val))
            is_leader = (val == max_val and max_val > 0)
            if is_leader:
                label += " ★"
            fig_heatmap.add_annotation(
                x=platform, y=genre, text=label,
                showarrow=False,
                font=dict(
                    size=11,
                    color="#FFD700" if is_leader else "#ffffff",
                    family="Arial",
                ),
            )

    fig_heatmap.update_layout(
        template=PLOTLY_TEMPLATE,
        height=max(500, len(heatmap_data) * 40),
        xaxis=dict(side="top", tickfont=dict(size=13)),
        yaxis=dict(tickfont=dict(size=12), autorange="reversed"),
        margin=dict(l=130, t=40, r=10, b=10),
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
    st.caption("★ = genre leader per row (gold)")

    # ── Genre Deep Dive ──────────────────────────────────────────────────────

    st.divider()
    st.markdown(
        section_header_html(
            "Genre Deep Dive",
            "Drill into any genre to compare platform depth, average quality, and the highest-rated titles.",
        ),
        unsafe_allow_html=True,
    )

    _DISPLAY_TO_RAW = {"Documentary": "documentation", "Sci-Fi": "scifi"}
    genre_options = list(heatmap_data.index)
    genre_display_to_raw = {g: _DISPLAY_TO_RAW.get(g, g.lower()) for g in genre_options}

    current_genre = st.session_state.get("comp_selected_genre")
    raw_to_display = {v: k for k, v in genre_display_to_raw.items()}
    current_display = raw_to_display.get(current_genre)
    default_idx = genre_options.index(current_display) if current_display in genre_options else 0

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

    _expanded_card_data = None

    n_platforms = len(dd_stats)
    if n_platforms > 0:
        # Find quality leader
        best_imdb_platform = dd_stats.loc[dd_stats["avg_imdb"].idxmax(), "platform_display"]

        cols = st.columns(n_platforms)
        for i, (_, row) in enumerate(dd_stats.iterrows()):
            pname = row["platform_display"]
            _p_col = _platform_color(pname)
            with cols[i]:
                # Platform header with color accent
                st.markdown(
                    f'<div style="border-bottom:2px solid {_p_col};padding-bottom:4px;'
                    f'margin-bottom:8px;font-weight:600;color:{CARD_TEXT};">{pname}</div>',
                    unsafe_allow_html=True,
                )
                if normalized:
                    st.metric(f"Share", f"{row['pct_of_catalog']}% of catalog")
                    st.caption(f"{int(row['count']):,} titles")
                else:
                    st.metric(f"Titles", f"{int(row['count']):,}")
                    st.caption(f"{row['pct_of_catalog']}% of catalog")
                st.metric("Avg IMDb", f"{row['avg_imdb']:.2f}")

                if pname in top_titles and not top_titles[pname].empty:
                    st.markdown(f'<div style="font-size:0.8em;font-weight:600;color:{CARD_TEXT_MUTED};margin:8px 0 4px;">Top Titles</div>', unsafe_allow_html=True)
                    for t_idx, (_, t) in enumerate(top_titles[pname].iterrows()):
                        title_id = t.get("id", "")
                        card_key = f"dd_{pname}_{t_idx}"
                        expanded_key = st.session_state.get("comp_expanded_title")
                        is_expanded = expanded_key == (pname, title_id)

                        border_color = CARD_ACCENT if is_expanded else CARD_BORDER
                        border_width = "2px" if is_expanded else "1px"

                        _imdb_score = t.get("imdb_score")
                        _imdb_str = f"{_imdb_score:.1f}" if _imdb_score and _imdb_score == _imdb_score else "N/A"
                        _votes_str = format_votes(t.get("imdb_votes"))
                        _year_str = str(int(t["release_year"])) if t.get("release_year") == t.get("release_year") else "?"
                        _type_str = t.get("type", "")
                        _plat_key = next((k for k, v in PLATFORMS.items() if v["name"] == pname), None)
                        _plat_badge = platform_badges_html(_plat_key) if _plat_key else ""

                        st.markdown(
                            f'<div style="background:{CARD_BG};border:{border_width} solid '
                            f'{border_color};border-radius:8px;padding:9px 12px;margin-bottom:3px;">'
                            f'<div style="font-size:0.88em;font-weight:600;color:{CARD_TEXT};">'
                            f'{t["title"]} <span style="color:{CARD_TEXT_MUTED};font-weight:400;">({_year_str})</span></div>'
                            f'<div style="display:flex;align-items:center;gap:4px;flex-wrap:wrap;'
                            f'margin-top:3px;font-size:0.76em;">'
                            f'{_plat_badge}'
                            f'<span style="color:{CARD_TEXT_MUTED};">{_type_str}</span>'
                            f'<span style="color:{CARD_TEXT_MUTED};">·</span>'
                            f'<span style="color:{CARD_TEXT};">IMDb <strong>{_imdb_str}</strong></span>'
                            f'<span style="color:{CARD_TEXT_MUTED};">({_votes_str})</span>'
                            f'</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                        btn_label = "Close" if is_expanded else "Details"
                        if st.button(btn_label, key=card_key, use_container_width=True):
                            if is_expanded:
                                st.session_state["comp_expanded_title"] = None
                            else:
                                st.session_state["comp_expanded_title"] = (pname, title_id)
                            st.rerun()

                        if is_expanded and title_id:
                            _expanded_card_data = (t, title_id, pname)
                else:
                    st.caption("No rated titles in this genre")

    if _expanded_card_data is not None:
        _exp_row, _exp_id, _exp_platform = _expanded_card_data
        _render_expanded_card(_exp_row, _exp_id, _exp_platform)

    # Auto-generated genre strategic insight
    if len(dd_stats) >= 2:
        _best = dd_stats.loc[dd_stats["avg_imdb"].idxmax()]
        _worst = dd_stats.loc[dd_stats["avg_imdb"].idxmin()]
        _vol_leader = dd_stats.loc[dd_stats["count"].idxmax()]
        _vol_laggard = dd_stats.loc[dd_stats["count"].idxmin()]
        _q_gap = _best["avg_imdb"] - _worst["avg_imdb"]
        _vol_ratio = _vol_leader["count"] / max(_vol_laggard["count"], 1)

        if _best["platform_display"] != _vol_leader["platform_display"] and _q_gap > 0.2:
            _insight_txt = (
                f"In **{selected_genre_display}**, {_best['platform_display']} achieves higher quality "
                f"(IMDb {_best['avg_imdb']:.2f} vs {_worst['avg_imdb']:.2f}) with "
                f"{'fewer' if _best['count'] < _vol_leader['count'] else 'more'} titles than "
                f"{_vol_leader['platform_display']} ({_best['count']:,} vs {_vol_leader['count']:,}) "
                f"— a {'prestige-over-volume' if _best['count'] < _vol_leader['count'] else 'scale-and-quality'} strategy."
            )
        elif _q_gap < 0.1:
            _insight_txt = (
                f"Quality is roughly even across platforms in **{selected_genre_display}** "
                f"(range: {_worst['avg_imdb']:.2f}–{_best['avg_imdb']:.2f} IMDb). "
                f"Volume differences ({_vol_laggard['count']:,}–{_vol_leader['count']:,} titles) are the main competitive differentiator."
            )
        else:
            _insight_txt = (
                f"In **{selected_genre_display}**, {_best['platform_display']} leads on quality "
                f"(IMDb {_best['avg_imdb']:.2f} vs {_worst['avg_imdb']:.2f}). "
                f"{_vol_leader['platform_display']} leads on volume ({_vol_leader['count']:,} titles)."
            )
        st.info(_insight_txt)

    # ── Strategic Insights ────────────────────────────────────────────────────

    st.divider()
    st.markdown(
        section_header_html(
            "Strategic Insights",
            "Data-driven competitive assessment per competitor — what the numbers imply for the merged catalog's market position.",
        ),
        unsafe_allow_html=True,
    )

    _insights_heatmap = compute_genre_heatmap(comp_df, top_n=COMPARISON_TOP_GENRES, normalize=normalized)
    _insights_tiers = compute_quality_tiers(comp_df, QUALITY_TIERS, normalize=normalized)
    _insights_tier_ordered = _insights_tiers[
        [c for c in display_order if c in _insights_tiers.columns]
    ]
    insights = compute_strategic_insights(comp_df, _insights_heatmap, _insights_tier_ordered)

    if insights:
        insight_cols = st.columns(len(insights))
        for col_idx, (comp_name, data) in enumerate(insights.items()):
            with insight_cols[col_idx]:
                _p_key = next((k for k, v in PLATFORMS.items() if v["name"] == comp_name), None)
                _p_color = PLATFORMS[_p_key]["color"] if _p_key else CARD_BORDER

                prose = _build_insight_prose(comp_name, data, merged_name)
                prose_html = "".join(
                    f'<p style="margin:0 0 10px 0;line-height:1.65;font-size:0.85em;">{p}</p>'
                    for p in prose
                )
                st.markdown(
                    f'<div style="background:{CARD_BG};border-left:4px solid {_p_color};'
                    f'border-radius:6px;padding:14px 16px;">'
                    f'<div style="font-size:1.0em;font-weight:700;color:{CARD_TEXT};'
                    f'margin-bottom:10px;">vs {comp_name}</div>'
                    f'<div style="color:{CARD_TEXT};">{prose_html}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.info("Select competitors to see strategic insights.")

else:
    st.info("No genre data available with current filters.")


# ── footer ────────────────────────────────────────────────────────────────

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
