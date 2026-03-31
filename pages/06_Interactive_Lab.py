"""Page 6: Interactive Lab — High-engagement, playful but data-driven features.

Feature 1: Build Your Streaming Service (budget drafting game)
Feature 2: Hypothetical Title Predictor (greenlight model)
Feature 3: Insight Generator
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.config import (
    ALL_PLATFORMS,
    CARD_ACCENT,
    CARD_BG,
    CARD_BORDER,
    CARD_TEXT,
    CARD_TEXT_MUTED,
    PLATFORMS,
    PLOTLY_TEMPLATE,
)
from src.data.loaders import (
    deduplicate_titles,
    load_all_platforms_titles,
    load_enriched_titles,
    load_greenlight_model,
    load_imdb_principals,
    load_merged_titles,
)
from src.analysis.scoring import compute_quality_score
from src.analysis.lab import (
    compute_service_stats,
    compute_title_value,
    compare_services,
    generate_insights,
    get_random_insight,
    get_talent_suggestions,
    predict_title,
)
from src.analysis.strategic import build_merged_entity
from src.ui.badges import page_header_html, section_header_html, styled_banner_html, styled_metric_card_html
from src.ui.session import init_session_state

st.set_page_config(page_title="Interactive Lab", page_icon="📊", layout="wide")
init_session_state()

st.markdown(
    page_header_html(
        "Interactive Lab",
        "Playful, data-driven features — build a service, predict a title's success, or discover surprising insights.",
    ),
    unsafe_allow_html=True,
)

# ─── Data ───────────────────────────────────────────────────────────────────
enriched = load_enriched_titles()
enriched = deduplicate_titles(enriched)
enriched["quality_score"] = compute_quality_score(enriched)

# ─── Feature Tabs ───────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "Build Your Streaming Service",
    "Title Predictor",
    "Insight Generator",
])

# ─── Feature 1: Build Your Streaming Service ────────────────────────────────
with tab1:
    st.markdown(
        section_header_html("Build Your Streaming Service", "Draft titles within a budget and compare your service against Netflix + Max."),
        unsafe_allow_html=True,
    )

    BUDGET = 5000  # Total budget in "millions"

    if "drafted" not in st.session_state:
        st.session_state.drafted = []

    # Compute values for all titles
    enriched_with_value = enriched.copy()
    has_box_office = "box_office_usd" in enriched_with_value.columns
    enriched_with_value["value"] = enriched_with_value.apply(
        lambda row: compute_title_value(row, use_box_office=has_box_office), axis=1
    )

    # Controls
    col_ctrl1, col_ctrl2 = st.columns([2, 1])
    with col_ctrl1:
        search = st.text_input("Search titles to draft", placeholder="Search by title...",
                               key="draft_search")
    with col_ctrl2:
        sort_by = st.selectbox("Sort by", ["Quality Score", "IMDb", "Value (cheapest)"],
                               key="draft_sort")

    # Show available titles
    available = enriched_with_value[
        ~enriched_with_value["id"].isin([d["id"] for d in st.session_state.drafted])
    ]
    if search:
        available = available[available["title"].str.contains(search, case=False, na=False)]

    if sort_by == "Quality Score":
        available = available.sort_values("quality_score", ascending=False)
    elif sort_by == "IMDb":
        available = available.sort_values("imdb_score", ascending=False)
    else:
        available = available.sort_values("value", ascending=True)

    # Current stats
    stats = compute_service_stats(st.session_state.drafted)
    remaining = BUDGET - stats["spend"]

    st.markdown(f"""
    <div style="background:{CARD_BG};border:1px solid {CARD_BORDER};border-top:3px solid {CARD_ACCENT};
                border-radius:8px;padding:12px 16px;margin-bottom:16px;">
        <div style="display:flex;justify-content:space-between;">
            <span style="color:{CARD_TEXT};">Budget: <b style="color:{CARD_ACCENT};">${remaining:,.0f}M</b> remaining of ${BUDGET:,}M</span>
            <span style="color:{CARD_TEXT};">Titles: <b>{stats['count']}</b> | Avg IMDb: <b>{stats['avg_imdb']:.2f}</b> | Diversity: <b>{stats['diversity']:.2f}</b></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Title list
    for _, row in available.head(20).iterrows():
        col_title, col_info, col_btn = st.columns([3, 2, 1])
        with col_title:
            imdb_str = f"{row['imdb_score']:.1f}" if pd.notna(row.get("imdb_score")) else "N/A"
            year_str = int(row["release_year"]) if pd.notna(row.get("release_year")) else ""
            poster_note = ""
            if has_box_office and pd.notna(row.get("box_office_usd")):
                bo = row["box_office_usd"]
                if bo > 1e9:
                    poster_note = f" | Box Office: ${bo/1e9:.1f}B"
                elif bo > 1e6:
                    poster_note = f" | Box Office: ${bo/1e6:.0f}M"
            st.markdown(f"**{row['title']}** ({year_str}) — IMDb {imdb_str}{poster_note}")
        with col_info:
            st.caption(f"Cost: ${row['value']:.0f}M | {row['type']}")
        with col_btn:
            if row["value"] <= remaining:
                if st.button("Draft", key=f"draft_{row['id']}"):
                    st.session_state.drafted.append(row.to_dict())
                    st.rerun()
            else:
                st.caption("Over budget")

    # Drafted titles
    if st.session_state.drafted:
        st.divider()
        st.markdown(section_header_html("Your Drafted Catalog"), unsafe_allow_html=True)
        for i, d in enumerate(st.session_state.drafted):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{d['title']}** — ${d['value']:.0f}M")
            with col2:
                if st.button("Remove", key=f"undraft_{i}"):
                    st.session_state.drafted.pop(i)
                    st.rerun()

        # Compare vs Netflix+Max
        st.markdown(section_header_html("Compare vs Netflix + Max"), unsafe_allow_html=True)
        merged = load_merged_titles()
        comparison = compare_services(stats, merged)

        col_you, col_them = st.columns(2)
        with col_you:
            st.markdown(styled_metric_card_html("Your Service", f"{comparison['drafted']['count']} titles"), unsafe_allow_html=True)
            st.markdown(styled_metric_card_html("Avg IMDb", f"{comparison['drafted']['avg_imdb']:.2f}"), unsafe_allow_html=True)
            st.markdown(styled_metric_card_html("Genre Diversity", f"{comparison['drafted']['diversity']:.2f}"), unsafe_allow_html=True)
        with col_them:
            st.markdown(styled_metric_card_html("Netflix + Max", f"{comparison['merged']['count']:,} titles"), unsafe_allow_html=True)
            st.markdown(styled_metric_card_html("Avg IMDb", f"{comparison['merged']['avg_imdb']:.2f}"), unsafe_allow_html=True)
            st.markdown(styled_metric_card_html("Genre Diversity", f"{comparison['merged']['diversity']:.2f}"), unsafe_allow_html=True)

        # Genre distribution donut
        if stats["genres"]:
            genre_df = pd.DataFrame([
                {"genre": g, "count": c} for g, c in stats["genres"].items()
            ]).sort_values("count", ascending=False)
            fig_donut = px.pie(genre_df, names="genre", values="count",
                              title="Your Service: Genre Distribution",
                              hole=0.4, template=PLOTLY_TEMPLATE)
            st.plotly_chart(fig_donut, use_container_width=True)

        # Export
        if st.button("Export Catalog as CSV"):
            csv_cols = ["title", "type", "release_year", "imdb_score", "value"]
            draft_df = pd.DataFrame(st.session_state.drafted)
            available_cols = [c for c in csv_cols if c in draft_df.columns]
            csv = draft_df[available_cols].to_csv(index=False)
            st.download_button("Download CSV", csv, "my_streaming_service.csv", "text/csv")

        if st.button("Clear All"):
            st.session_state.drafted = []
            st.rerun()

# ─── Feature 2: Hypothetical Title Predictor ────────────────────────────────
with tab2:
    st.markdown(
        section_header_html("Hypothetical Title Predictor", "Predict how a hypothetical title would perform using our trained Greenlight model."),
        unsafe_allow_html=True,
    )

    col_input1, col_input2 = st.columns(2)

    with col_input1:
        pred_type = st.radio("Type", ["Movie", "Show"], key="pred_type", horizontal=True)
        pred_description = st.text_area("Description", key="pred_desc",
                                        placeholder="A brief synopsis of the title...")

        all_genres = sorted(set(
            g for genres in enriched["genres"].dropna()
            if isinstance(genres, (list,))
            for g in genres
        ))
        pred_genres = st.multiselect("Genres", all_genres, key="pred_genres")
        pred_runtime = st.slider("Runtime (minutes)", 20, 300,
                                 90 if pred_type == "Movie" else 45,
                                 key="pred_runtime")

    with col_input2:
        pred_year = st.slider("Release Year", 1990, 2025, 2024, key="pred_year")
        pred_country = st.selectbox("Primary Market",
                                    ["US", "GB", "KR", "JP", "FR", "DE", "IN", "Other"],
                                    key="pred_country")
        pred_cert = st.selectbox("Certification",
                                 ["TV-MA", "R", "PG-13", "TV-14", "PG", "G", "TV-Y"],
                                 key="pred_cert")
        pred_budget = st.selectbox("Budget Tier",
                                   ["Unknown", "Low", "Mid", "High", "Blockbuster"],
                                   key="pred_budget")

    if st.button("Predict Performance", key="pred_go", type="primary"):
        model = load_greenlight_model(pred_type.lower())

        if model is None:
            st.error("Greenlight model not found. Run `scripts/10_train_predictor.py` first.")
        else:
            country_tier = 3 if pred_country == "US" else (2 if pred_country in ["GB", "KR", "JP", "FR", "DE", "IN"] else 1)
            budget_tier_map = {"Unknown": 0, "Low": 1, "Mid": 2, "High": 3, "Blockbuster": 4}

            features = {
                "genres": pred_genres,
                "runtime": pred_runtime,
                "release_year": pred_year,
                "country_tier": country_tier,
                "has_franchise": 0,
                "budget_tier": budget_tier_map.get(pred_budget, 0),
                "award_genre_avg": 0,
            }

            result = predict_title(model, features, all_genres)

            if result:
                col_pred1, col_pred2, col_pred3 = st.columns(3)
                with col_pred1:
                    st.metric("Predicted IMDb",
                              f"{result['prediction']:.1f} ± {result['uncertainty']:.1f}")
                with col_pred2:
                    tier_colors = {
                        "Blockbuster": "🟢", "Strong": "🔵",
                        "Moderate": "🟡", "High Risk": "🔴"
                    }
                    st.metric("Success Tier",
                              f"{tier_colors.get(result['tier'], '')} {result['tier']}")
                with col_pred3:
                    if result.get("cv_rmse"):
                        st.metric("Model Accuracy (RMSE)",
                                  f"{result['cv_rmse']:.3f}")

                # Feature importances (mandatory)
                st.markdown(section_header_html("Feature Importances"), unsafe_allow_html=True)
                imp_df = pd.DataFrame({
                    "Feature": result["importances"].index,
                    "Importance": result["importances"].values,
                })
                fig_imp = px.bar(
                    imp_df.head(15), x="Importance", y="Feature",
                    orientation="h", template=PLOTLY_TEMPLATE,
                    title="What Drives the Prediction",
                )
                fig_imp.update_layout(yaxis=dict(autorange="reversed"), height=400)
                st.plotly_chart(fig_imp, use_container_width=True)

                # Model card (mandatory)
                with st.expander("Model Card", expanded=True):
                    global_mean = result.get('global_mean')
                    global_mean_str = f"{global_mean:.2f}" if global_mean is not None else "N/A"
                    st.markdown(f"""
                    **Model type:** GradientBoostingRegressor (scikit-learn)

                    **Training data:** {pred_type}s from enriched catalog

                    **Cross-validation:** 5-fold

                    **CV RMSE:** {result.get('cv_rmse', 'N/A')}

                    **Baseline RMSE:** {result.get('baseline_rmse', 'N/A')} (predicting global mean of {global_mean_str})

                    **Top 5 features:** {', '.join(result['importances'].head(5).index.tolist())}

                    **Limitations:** Trained on catalog metadata only. Does not account for
                    marketing, star power beyond genre averages, or cultural timing.
                    Predictions should be treated as directional, not precise.
                    """)

                # Talent suggestions
                st.markdown(section_header_html("Talent Suggestions"), unsafe_allow_html=True)
                principals = load_imdb_principals()
                all_titles = load_all_platforms_titles()

                if not principals.empty and pred_genres:
                    for role, role_name in [("director", "Directors"), ("writer", "Writers"), ("actor", "Actors")]:
                        max_k = 5 if role != "actor" else 10
                        suggestions = get_talent_suggestions(
                            principals, all_titles, pred_genres, role=role, top_k=max_k
                        )
                        if not suggestions.empty:
                            st.markdown(f"**Top {role_name} for {', '.join(pred_genres)}:**")
                            for _, s in suggestions.iterrows():
                                titles_str = ", ".join(s["titles"][:3])
                                st.markdown(
                                    f"- **{s['name']}** — Avg IMDb: {s['avg_imdb']:.1f}, "
                                    f"{s['title_count']} titles ({titles_str})"
                                )
                else:
                    st.info("Select genres to see talent suggestions.")

# ─── Feature 3: Insight Generator ───────────────────────────────────────────
with tab3:
    st.markdown(
        section_header_html("Insight Generator", "Discover surprising, data-backed insights from the catalog — exploratory fun facts mode."),
        unsafe_allow_html=True,
    )

    col_scope, col_btn = st.columns([2, 1])
    with col_scope:
        insight_scope = st.selectbox(
            "Scope",
            ["all_platforms"] + ALL_PLATFORMS + ["merged"],
            format_func=lambda x: PLATFORMS.get(x, {}).get("name", x) if x != "all_platforms" else "All Platforms",
            key="insight_scope",
        )
    with col_btn:
        st.markdown("")
        generate_btn = st.button("Generate Insights", key="insight_go", type="primary")
        surprise_btn = st.button("Surprise Me!", key="insight_surprise")

    if generate_btn:
        insights = generate_insights(
            load_all_platforms_titles(), scope=insight_scope,
            enriched_df=enriched if "award_wins" in enriched.columns else None,
        )
        if insights:
            for insight in insights:
                st.markdown(f"""
                <div style="background:{CARD_BG};border-left:3px solid {CARD_ACCENT};
                            padding:10px 16px;margin-bottom:8px;border-radius:4px;">
                    <span style="color:{CARD_TEXT};">{insight}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No insights generated for this scope.")

    if surprise_btn:
        insight = get_random_insight(
            load_all_platforms_titles(),
            enriched_df=enriched if "award_wins" in enriched.columns else None,
        )
        st.markdown(f"""
        <div style="background:{CARD_BG};border:2px solid {CARD_ACCENT};
                    padding:16px;border-radius:8px;text-align:center;">
            <span style="color:{CARD_TEXT};font-size:1.1rem;">{insight}</span>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ─── Section: Franchise Footprint ─────────────────────────────────────────────
st.markdown(
    section_header_html(
        "Franchise Footprint",
        "Top content franchises in the merged Netflix+Max catalog — which IP portfolios become dominant post-merger.",
    ),
    unsafe_allow_html=True,
)

if "collection_name" in enriched.columns and enriched["collection_name"].notna().sum() > 0:
    try:
        # Build merged entity from enriched (enriched already has platform column pre-deduplicate)
        merged_ff = enriched[enriched["platform"].isin(["netflix", "max"])].copy()
        merged_ff = merged_ff.drop_duplicates(subset="id", keep="first")

        franchises = (
            merged_ff[merged_ff["collection_name"].notna()]
            .groupby("collection_name")
            .agg(
                title_count=("title", "count"),
                avg_imdb=("imdb_score", "mean"),
                latest_year=("release_year", "max"),
            )
            .reset_index()
        )
        franchises = (
            franchises[franchises["title_count"] >= 2]
            .sort_values("title_count", ascending=False)
            .head(20)
        )

        if not franchises.empty:
            fig_franchise = px.bar(
                franchises.sort_values("title_count"),
                x="title_count",
                y="collection_name",
                orientation="h",
                color="avg_imdb",
                color_continuous_scale=[[0, "#555"], [1, "#00B4A6"]],
                template=PLOTLY_TEMPLATE,
                height=max(400, 28 * len(franchises)),
                title="Merged Catalog: Top 20 Franchises by Catalog Depth",
            )
            fig_franchise.update_layout(
                coloraxis_colorbar_title="Avg IMDb",
                margin=dict(r=130),
                xaxis_title="Titles in Catalog",
                yaxis_title="",
            )
            st.plotly_chart(fig_franchise, use_container_width=True)

            top_f = franchises.iloc[0]
            avg_str = f"{top_f['avg_imdb']:.1f}" if pd.notna(top_f["avg_imdb"]) else "N/A"
            st.markdown(
                styled_banner_html(
                    "🎬",
                    f"The merged entity's deepest franchise is <b>{top_f['collection_name']}</b> "
                    f"with <b>{int(top_f['title_count'])}</b> titles and an average IMDb of <b>{avg_str}</b>.",
                ),
                unsafe_allow_html=True,
            )
            coverage_pct = enriched["collection_name"].notna().mean()
            st.caption(
                f"Based on TMDB collection data ({coverage_pct:.0%} catalog coverage). "
                "Franchises with fewer than 2 titles excluded."
            )
        else:
            st.info("No franchises with 2+ titles found in the merged catalog.")
    except Exception as e:
        st.warning(f"Franchise Footprint could not be computed: {e}")
else:
    st.info(
        "Franchise data requires TMDB enrichment (`collection_name` field). "
        "Run `scripts/08_enrich_tmdb.py` and `scripts/09_build_enriched_titles.py` to populate."
    )

# ─── Footer ─────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="border-top:1px solid #333;padding:16px 0;color:#666;'
    'font-size:0.8em;text-align:center;">'
    'Hypothetical merger for academic analysis. Data is a snapshot (mid-2023). '
    'All insights are illustrative, not prescriptive. '
    'Update: As of Feb 26, 2026, Netflix withdrew from this acquisition after Paramount Skydance\'s competing bid was deemed superior by the WBD board.'
    '</div>',
    unsafe_allow_html=True,
)
