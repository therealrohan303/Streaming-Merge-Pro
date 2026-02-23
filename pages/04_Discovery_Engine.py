"""Page 4: Discovery Engine — Full recommendation toolkit.

Three distinct entry points:
  Tab 1: Similar to a Title (with why-similar explainer)
  Tab 2: Preference-Based Recommendations
  Tab 3: Vibe Search (NLP-powered semantic + keyword + genome hybrid)

Plus session-based recommendation history.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from src.config import (
    CARD_ACCENT,
    CARD_BG,
    CARD_BORDER,
    CARD_TEXT,
    CARD_TEXT_MUTED,
    SIMILARITY_MIN_IMDB,
)
from src.data.loaders import (
    deduplicate_titles,
    get_titles_for_view,
    load_all_platforms_titles,
    load_enriched_titles,
    load_genome_vectors,
    load_imdb_principals,
    load_similarity_data,
)
from src.analysis.scoring import compute_quality_score
from src.analysis.discovery import (
    extract_vibe_signals,
    get_similar_with_explanation,
    preference_based_recommendations,
    vibe_search,
)
from src.ui.badges import platform_badges_html
from src.ui.filters import render_sidebar_filters, apply_filters
from src.ui.session import init_session_state

st.set_page_config(page_title="Discovery Engine", page_icon="📊", layout="wide")
init_session_state()

st.title("Discovery Engine")
st.caption("Your full recommendation toolkit — find your next favorite title through three distinct approaches.")

# ─── Sidebar ────────────────────────────────────────────────────────────────
raw_df = get_titles_for_view(st.session_state.get("platform_view", "merged"))
filters = render_sidebar_filters(raw_df)
titles_all = get_titles_for_view(filters["platform_view"])
titles = apply_filters(titles_all, filters)
titles = deduplicate_titles(titles)
titles["quality_score"] = compute_quality_score(titles)

st.sidebar.metric("Titles Available", f"{len(titles):,}")

# Load enrichment data
enriched_df = load_enriched_titles()
enriched_df = deduplicate_titles(enriched_df)
principals_df = load_imdb_principals()
sim_df = load_similarity_data()
genome_vectors, genome_id_map = load_genome_vectors()

# ─── Session History ────────────────────────────────────────────────────────
if "rec_history" not in st.session_state:
    st.session_state.rec_history = []


def _render_rec_card(row, idx, show_explanation=False):
    """Render a recommendation card."""
    title = row.get("title", "Unknown")
    year = int(row.get("release_year", 0)) if pd.notna(row.get("release_year")) else ""
    imdb = row.get("imdb_score")
    imdb_str = f"{imdb:.1f}" if pd.notna(imdb) else "N/A"
    content_type = row.get("type", "")
    platforms = row.get("platforms", row.get("platform", ""))
    score_key = None
    score_val = None

    for key in ["similarity_score", "fit_score", "vibe_score"]:
        if key in row and pd.notna(row.get(key)):
            score_key = key.replace("_", " ").title()
            score_val = f"{row[key]:.0%}" if row[key] <= 1 else f"{row[key]:.1f}"
            break

    genres = row.get("genres", [])
    genre_pills = ""
    if isinstance(genres, (list,)):
        genre_pills = " ".join(
            f'<span style="background:#2a2a3e;padding:2px 6px;border-radius:3px;'
            f'font-size:0.7rem;margin-right:3px;">{g}</span>'
            for g in genres[:4]
        )

    st.markdown(f"""
    <div style="background:{CARD_BG};border:1px solid {CARD_BORDER};border-radius:8px;
                padding:12px 16px;margin-bottom:8px;">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <div>
                <span style="font-size:1rem;font-weight:600;color:{CARD_TEXT};">{title}</span>
                <span style="color:{CARD_TEXT_MUTED};margin-left:8px;">{year} · {content_type}</span>
            </div>
            <div>
                {platform_badges_html(platforms)}
            </div>
        </div>
        <div style="margin-top:6px;display:flex;align-items:center;gap:12px;">
            <span style="color:{CARD_ACCENT};font-weight:600;">IMDb {imdb_str}</span>
            {f'<span style="color:{CARD_TEXT_MUTED};">{score_key}: {score_val}</span>' if score_key else ""}
        </div>
        <div style="margin-top:6px;">{genre_pills}</div>
    </div>
    """, unsafe_allow_html=True)

    # Why-similar explanation
    if show_explanation and "explanation" in row:
        exp = row["explanation"]
        if exp:
            with st.expander("Why similar?", expanded=False):
                if "genre_overlap" in exp:
                    st.write(f"**Genre overlap:** {', '.join(exp['genre_overlap'])}")
                if "matched_vibe_tags" in exp:
                    st.write(f"**Matched vibe tags:** {', '.join(exp['matched_vibe_tags'])}")
                if "shared_crew" in exp:
                    crew_strs = [f"{c['name']} ({c['role']})" for c in exp["shared_crew"]]
                    st.write(f"**Shared crew:** {', '.join(crew_strs)}")


# ─── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab_history = st.tabs([
    "Similar to a Title", "Preference-Based", "Vibe Search", "History"
])

# ─── Tab 1: Similar to a Title ──────────────────────────────────────────────
with tab1:
    st.subheader("Find titles similar to one you love")
    st.caption("Enhanced similarity with genre overlap, vibe tag matching, and shared crew explanations.")

    col_search, col_scope, col_count = st.columns([3, 1, 1])
    with col_search:
        search_query = st.text_input("Search for a title", key="sim_search",
                                     placeholder="Start typing a title name...")
    with col_scope:
        scope = st.radio("Scope", ["Merged", "All Platforms"], key="sim_scope", horizontal=True)
        scope_key = "merged" if scope == "Merged" else "all_platforms"
    with col_count:
        n_results = st.slider("Results", 5, 20, 10, key="sim_count")

    min_imdb = st.slider("Minimum IMDb", 0.0, 9.0, SIMILARITY_MIN_IMDB, 0.5, key="sim_min_imdb")

    # Find matching titles
    if search_query:
        matches = titles[titles["title"].str.contains(search_query, case=False, na=False)].head(10)
        if not matches.empty:
            selected_title = st.selectbox(
                "Select a title",
                options=matches["id"].tolist(),
                format_func=lambda x: matches[matches["id"] == x].iloc[0]["title"],
                key="sim_select",
            )

            if st.button("Find Similar", key="sim_go", type="primary"):
                with st.spinner("Finding similar titles..."):
                    all_titles = load_all_platforms_titles()
                    results = get_similar_with_explanation(
                        selected_title, all_titles, sim_df,
                        principals_df=principals_df,
                        enriched_df=enriched_df,
                        scope=scope_key, top_k=n_results, min_imdb=min_imdb,
                    )

                if results:
                    st.success(f"Found {len(results)} similar titles")
                    for i, r in enumerate(results):
                        _render_rec_card(r, i, show_explanation=True)

                    # Add to history
                    source_title = matches[matches["id"] == selected_title].iloc[0]["title"]
                    st.session_state.rec_history.insert(0, {
                        "type": "Similar to Title",
                        "query": source_title,
                        "count": len(results),
                        "results": [r.get("title", "?") for r in results[:5]],
                    })
                    st.session_state.rec_history = st.session_state.rec_history[:10]
                else:
                    st.warning("No similar titles found with the current filters.")
        else:
            st.info("No titles match your search.")

# ─── Tab 2: Preference-Based ────────────────────────────────────────────────
with tab2:
    st.subheader("Recommendations based on your preferences")
    st.caption("Tell us what you like, and we'll find the best matches in the catalog.")

    col_left, col_right = st.columns(2)

    with col_left:
        # Get available genres
        all_genres_list = sorted(set(
            g for genres in titles["genres"].dropna()
            if isinstance(genres, (list,))
            for g in genres
        ))
        pref_genres = st.multiselect("Favorite genres (1-5)", all_genres_list,
                                     max_selections=5, key="pref_genres")
        pref_min_imdb = st.slider("Minimum IMDb", 0.0, 9.0, 6.0, 0.5, key="pref_imdb")
        pref_type = st.radio("Content type", ["Both", "Movie", "Show"], key="pref_type", horizontal=True)

    with col_right:
        pref_year = st.slider("Release year range", 1950, 2024, (1990, 2024), key="pref_year")
        pref_popularity = st.slider(
            "Discovery style", 0.0, 1.0, 0.5, 0.1,
            help="0 = Hidden gems, 1 = Popular blockbusters",
            key="pref_pop",
        )
        if pref_type == "Movie":
            pref_runtime = st.slider("Runtime (minutes)", 60, 240, (80, 180), key="pref_runtime")
        else:
            pref_runtime = None

        pref_scope = st.radio("Platform scope", ["Merged", "All Platforms"],
                              key="pref_scope", horizontal=True)

    if st.button("Get Recommendations", key="pref_go", type="primary"):
        with st.spinner("Finding your perfect matches..."):
            scope_key = "merged" if pref_scope == "Merged" else "all_platforms"
            source = load_all_platforms_titles() if scope_key == "all_platforms" else titles_all
            source = deduplicate_titles(source)

            results = preference_based_recommendations(
                source,
                genres=pref_genres or None,
                min_imdb=pref_min_imdb,
                content_type=pref_type,
                min_runtime=pref_runtime[0] if pref_runtime else None,
                max_runtime=pref_runtime[1] if pref_runtime else None,
                year_range=pref_year,
                popularity_weight=pref_popularity,
                scope=scope_key,
                top_k=20,
            )

        if not results.empty:
            st.success(f"Found {len(results)} recommendations")
            for i, (_, row) in enumerate(results.iterrows()):
                _render_rec_card(row.to_dict(), i)
                if "why_match" in row and pd.notna(row["why_match"]):
                    st.caption(f"  {row['why_match']}")

            # Add to history
            st.session_state.rec_history.insert(0, {
                "type": "Preference-Based",
                "query": f"Genres: {', '.join(pref_genres)}" if pref_genres else "All genres",
                "count": len(results),
                "results": results["title"].head(5).tolist(),
            })
            st.session_state.rec_history = st.session_state.rec_history[:10]
        else:
            st.warning("No titles match your preferences. Try relaxing some filters.")

# ─── Tab 3: Vibe Search ─────────────────────────────────────────────────────
with tab3:
    st.subheader("Vibe Search")
    st.caption("Describe what you're in the mood for, and we'll find it using NLP-powered semantic matching.")

    vibe_query = st.text_area(
        "What are you in the mood for?",
        placeholder="e.g., 'A slow-burn psychological thriller with a twist ending, set in a small town'",
        height=100,
        key="vibe_query",
    )

    with st.expander("Optional filters", expanded=False):
        vibe_col1, vibe_col2, vibe_col3 = st.columns(3)
        with vibe_col1:
            vibe_min_imdb = st.slider("Min IMDb", 0.0, 9.0, 0.0, 0.5, key="vibe_imdb")
        with vibe_col2:
            vibe_year = st.slider("Year range", 1950, 2024, (1970, 2024), key="vibe_year")
        with vibe_col3:
            vibe_scope = st.radio("Scope", ["Merged", "All Platforms"],
                                  key="vibe_scope", horizontal=True)

    if vibe_query and st.button("Search Vibes", key="vibe_go", type="primary"):
        # Show detected signals BEFORE results
        signals = extract_vibe_signals(vibe_query)
        if signals:
            signal_pills = " ".join(
                f'<span style="background:#2a2a3e;border:1px solid {CARD_ACCENT};'
                f'padding:3px 10px;border-radius:12px;font-size:0.8rem;margin:2px;">{s}</span>'
                for s in signals
            )
            st.markdown(f"**Detected themes:** {signal_pills}", unsafe_allow_html=True)
            st.markdown("")
        else:
            st.info("No specific themes detected — searching by description similarity.")

        with st.spinner("Searching the catalog with semantic matching..."):
            scope_key = "merged" if vibe_scope == "Merged" else "all_platforms"
            source = load_all_platforms_titles() if scope_key == "all_platforms" else titles_all
            source = deduplicate_titles(source)

            results, detected_signals = vibe_search(
                vibe_query, source,
                genome_vectors=genome_vectors,
                genome_id_map=genome_id_map,
                enriched_df=enriched_df,
                scope=scope_key,
                top_k=15,
                min_imdb=vibe_min_imdb if vibe_min_imdb > 0 else None,
                year_range=vibe_year,
            )

        if not results.empty:
            st.success(f"Found {len(results)} matching titles")

            # Highlight matched tags for each result
            for i, (_, row) in enumerate(results.iterrows()):
                _render_rec_card(row.to_dict(), i)

                # Show matched tags if available
                top_tags = row.get("top_tags")
                if isinstance(top_tags, (list,)) and signals:
                    matched = [t for t in top_tags if any(s.lower() in t.lower() for s in signals)]
                    if matched:
                        tag_html = " ".join(
                            f'<span style="background:#1a3a1a;border:1px solid #4a4;'
                            f'padding:1px 6px;border-radius:3px;font-size:0.7rem;">{t}</span>'
                            for t in matched[:5]
                        )
                        st.markdown(f"  Matched tags: {tag_html}", unsafe_allow_html=True)

            # Add to history
            st.session_state.rec_history.insert(0, {
                "type": "Vibe Search",
                "query": vibe_query[:50] + ("..." if len(vibe_query) > 50 else ""),
                "count": len(results),
                "results": results["title"].head(5).tolist(),
            })
            st.session_state.rec_history = st.session_state.rec_history[:10]
        else:
            st.warning("No titles matched your vibe. Try different keywords or relax the filters.")

    # How it works
    with st.expander("How Vibe Search works"):
        st.markdown("""
        **Hybrid scoring formula:**
        - 35% — Description embedding similarity (semantic matching)
        - 25% — Genre cosine similarity
        - 15% — MovieLens genome vector match (movies with coverage only)
        - 15% — Bayesian quality score
        - 10% — Awards boost (normalized award wins)

        For titles without MovieLens genome data, the genome weight is redistributed
        to description (60%) and genre (40%) matching.

        **Signal detection** extracts themes from your query (genres, moods, atmospheres)
        and matches them against MovieLens vibe tags and TMDB keywords.
        """)

# ─── Tab 4: History ─────────────────────────────────────────────────────────
with tab_history:
    st.subheader("Recommendation History")
    st.caption("Your last 10 recommendation sessions.")

    if not st.session_state.rec_history:
        st.info("No recommendations yet. Try one of the tabs above!")
    else:
        for i, entry in enumerate(st.session_state.rec_history):
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{entry['type']}** — {entry['query']}")
                    st.caption(f"{entry['count']} results: {', '.join(entry['results'])}")
                with col2:
                    # User marking buttons
                    cols = st.columns(3)
                    if cols[0].button("Interested", key=f"hist_int_{i}"):
                        entry["mark"] = "interested"
                    if cols[1].button("Watched", key=f"hist_watch_{i}"):
                        entry["mark"] = "watched"
                    if cols[2].button("Not for me", key=f"hist_skip_{i}"):
                        entry["mark"] = "skip"

                    mark = entry.get("mark")
                    if mark:
                        st.caption(f"Marked: {mark}")
                st.markdown("---")

        if st.button("Clear History"):
            st.session_state.rec_history = []
            st.rerun()

# ─── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Hypothetical merger for academic analysis. Data is a snapshot (mid-2023). "
    "Enrichment data: IMDb datasets, Wikidata, MovieLens 20M, TMDB API. "
    "Enrichment field coverage varies by title — see data_confidence."
)
