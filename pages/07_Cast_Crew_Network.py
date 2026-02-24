"""Page 7: Cast & Crew Network — Explore the people behind the content.

Sections:
  1. Person Search & Profile
  2. Community Detection (Louvain)
  3. Influence Scoring
  4. Rankings (Directors / Actors / Writers)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import (
    CARD_ACCENT,
    CARD_BG,
    CARD_BORDER,
    CARD_TEXT,
    CARD_TEXT_MUTED,
    PLATFORMS,
    PLOTLY_TEMPLATE,
    ENRICHED_DIR,
)
from src.data.loaders import (
    load_all_platforms_credits,
    load_all_platforms_titles,
    load_enriched_titles,
    load_imdb_principals,
    load_network_edges,
    load_person_stats,
)
from src.analysis.network import (
    get_community_details,
    get_cross_platform_bridges,
    get_person_profile,
    get_rankings,
    search_person,
)
from src.ui.badges import section_header_html, styled_metric_card_html, styled_banner_html
from src.ui.session import init_session_state

st.set_page_config(page_title="Cast & Crew Network", page_icon="📊", layout="wide")
init_session_state()

st.markdown(
    section_header_html(
        "Cast & Crew Network",
        "Explore collaboration networks, rankings, and influence across the streaming landscape.",
        font_size="2em",
    ),
    unsafe_allow_html=True,
)

# ─── Data Loading ───────────────────────────────────────────────────────────
person_stats = load_person_stats()
edges_df = load_network_edges()
principals = load_imdb_principals()
credits = load_all_platforms_credits()
# Normalize person_id to string (credits has int, person_stats has str from script 12)
if not credits.empty and "person_id" in credits.columns:
    credits["person_id"] = credits["person_id"].astype(str)
titles = load_all_platforms_titles()
enriched = load_enriched_titles()

has_network = not person_stats.empty and not edges_df.empty
has_principals = not principals.empty and len(principals) > 1000

# ─── Section 1: Person Search & Profile ─────────────────────────────────────
st.markdown(
    section_header_html("Person Search & Profile", "Search for any actor, director, writer, producer, composer, or cinematographer."),
    unsafe_allow_html=True,
)

col_search, col_role, col_min = st.columns([3, 1, 1])
with col_search:
    query = st.text_input("Search by name", placeholder="e.g., Christopher Nolan",
                          key="person_search")
with col_role:
    roles = ["All", "ACTOR", "DIRECTOR", "WRITER", "PRODUCER", "COMPOSER", "CINEMATOGRAPHER"]
    role_filter = st.selectbox("Role", roles, key="person_role")
with col_min:
    min_titles = st.number_input("Min titles", 1, 100, 3, key="person_min_titles")

if query and has_network:
    results = search_person(person_stats, query, role_filter, min_titles)
    if not results.empty:
        # Show search results
        selected_person = st.selectbox(
            f"Found {len(results)} matches",
            options=results["person_id"].tolist(),
            format_func=lambda pid: f"{results[results['person_id'] == pid].iloc[0]['name']} "
                                    f"({results[results['person_id'] == pid].iloc[0]['primary_role']}, "
                                    f"{results[results['person_id'] == pid].iloc[0]['title_count']} titles)",
            key="person_select",
        )

        # Profile
        profile = get_person_profile(
            selected_person, person_stats, edges_df,
            credits, titles, principals if has_principals else None,
        )

        if profile:
            # Key stats
            st.markdown(section_header_html(profile["name"]), unsafe_allow_html=True)
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.markdown(styled_metric_card_html("Avg IMDb", f"{profile.get('avg_imdb', 'N/A')}"), unsafe_allow_html=True)
            with col2:
                st.markdown(styled_metric_card_html("Titles", profile.get("title_count", 0)), unsafe_allow_html=True)
            with col3:
                career = ""
                if profile.get("career_start") and profile.get("career_end"):
                    career = f"{profile['career_start']}-{profile['career_end']}"
                st.markdown(styled_metric_card_html("Career Span", career or "N/A"), unsafe_allow_html=True)
            with col4:
                st.markdown(styled_metric_card_html("Top Genre", profile.get("top_genre", "N/A")), unsafe_allow_html=True)
            with col5:
                st.markdown(styled_metric_card_html("Role", profile.get("primary_role", "N/A")), unsafe_allow_html=True)

            # Awards context
            award_titles = profile.get("award_titles", [])
            if award_titles:
                st.markdown(
                    styled_banner_html("🏆", f"Award-winning work: {len(award_titles)} titles with awards",
                                       bg="rgba(255,215,0,0.1)", border_color="#FFD700"),
                    unsafe_allow_html=True,
                )

            # Top collaborators
            collabs = profile.get("top_collaborators", [])
            if collabs:
                st.markdown(section_header_html("Top Collaborators"), unsafe_allow_html=True)
                for c in collabs[:5]:
                    st.markdown(f"- **{c.get('name', 'Unknown')}** ({c.get('role', '')}) — "
                               f"{c.get('weight', 0)} shared titles")

            # Filmography
            filmography = profile.get("filmography", pd.DataFrame())
            if not filmography.empty:
                st.markdown(section_header_html("Filmography"), unsafe_allow_html=True)
                display_cols = ["title", "release_year", "imdb_score", "type", "platform"]
                if "role" in filmography.columns:
                    display_cols.insert(3, "role")
                available_cols = [c for c in display_cols if c in filmography.columns]
                st.dataframe(
                    filmography[available_cols].head(25),
                    hide_index=True,
                    column_config={
                        "imdb_score": st.column_config.NumberColumn("IMDb", format="%.1f"),
                    },
                )

            # Career trend line
            career_data = profile.get("career_trend", [])
            if len(career_data) >= 3:
                career_df = pd.DataFrame(career_data)
                fig_career = px.scatter(
                    career_df, x="release_year", y="imdb_score",
                    hover_data=["title"],
                    title=f"Career Trend: IMDb Over Time",
                    template=PLOTLY_TEMPLATE,
                    trendline="lowess",
                )
                fig_career.update_layout(height=350, xaxis_title="Year", yaxis_title="IMDb Score")
                st.plotly_chart(fig_career, use_container_width=True)
    else:
        st.info("No matches found. Try a different name or relax the filters.")
elif query and not has_network:
    st.warning(f"Network data: person_stats={'found (' + str(len(person_stats)) + ' rows)' if not person_stats.empty else 'MISSING'}, "
               f"edges={'found (' + str(len(edges_df)) + ' rows)' if not edges_df.empty else 'MISSING'}. "
               "Run `scripts/12_precompute_network.py` to generate.")

st.divider()

# ─── Section 2: Community Detection ─────────────────────────────────────────
st.markdown(
    section_header_html("Creative Circles", "Communities detected via Louvain algorithm on the collaboration graph."),
    unsafe_allow_html=True,
)

if has_network and has_principals:
    # Get unique communities
    community_ids = person_stats["community_id"].value_counts()
    top_communities = community_ids.head(8)

    st.info(f"Detected {len(community_ids)} creative circles across {len(person_stats):,} people")

    for comm_id in top_communities.index:
        details = get_community_details(person_stats, comm_id, titles)
        if details:
            with st.expander(
                f"{details['dominant_genre']} Circle — "
                f"{details['member_count']} members, avg IMDb {details['avg_imdb']:.2f}",
                expanded=False,
            ):
                # Top members
                st.markdown("**Key Members:**")
                for m in details["top_members"][:5]:
                    st.markdown(f"- {m['name']} ({m['primary_role']}) — "
                               f"influence: {m['influence_score']:.6f}, {m['title_count']} titles")

                # Platform breakdown
                if details["platform_breakdown"]:
                    plat_data = [{"platform": PLATFORMS.get(p, {}).get("name", p), "count": c}
                                for p, c in details["platform_breakdown"].items()]
                    plat_df = pd.DataFrame(plat_data).sort_values("count", ascending=False)
                    fig = px.bar(plat_df, x="platform", y="count",
                                title="Platform Distribution", template=PLOTLY_TEMPLATE,
                                height=250)
                    fig.update_layout(xaxis_title="", yaxis_title="People")
                    st.plotly_chart(fig, use_container_width=True)

    # Cross-platform bridges
    st.markdown(
        section_header_html("Cross-Platform Bridges",
                            "Talent connecting Netflix-heavy and Max-heavy creative clusters — "
                            "the connective tissue of the merged entity."),
        unsafe_allow_html=True,
    )

    bridges = get_cross_platform_bridges(person_stats, edges_df)
    if not bridges.empty:
        st.dataframe(
            bridges[["name", "primary_role", "title_count", "avg_imdb",
                     "influence_score", "bridge_count"]].head(15),
            hide_index=True,
            column_config={
                "avg_imdb": st.column_config.NumberColumn("Avg IMDb", format="%.2f"),
                "influence_score": st.column_config.NumberColumn("Influence", format="%.6f"),
                "bridge_count": "Bridge Connections",
            },
        )
    else:
        st.info("No cross-platform bridges detected with current thresholds.")

elif not has_principals:
    st.info("Community detection requires IMDb principals enrichment data with >1,000 rows. "
            "Run `scripts/05_enrich_imdb.py` and `scripts/12_precompute_network.py` first.")
else:
    st.info("Network data required. Run `scripts/12_precompute_network.py` to generate collaboration graph.")

st.divider()

# ─── Section 3: Influence Scoring ───────────────────────────────────────────
st.markdown(
    section_header_html("Influence Scoring",
                        "Influence = PageRank × Avg IMDb × (1 + normalized award wins) — "
                        "identifies the most central and high-quality talent."),
    unsafe_allow_html=True,
)

if has_network:
    top_influential = person_stats.nlargest(50, "influence_score")

    with st.expander("Methodology"):
        st.markdown("""
        **Influence Score** combines three signals:
        - **PageRank**: Measures centrality in the collaboration graph — people who work
          with many other well-connected people score higher
        - **Average IMDb**: Quality of the person's work
        - **Award bonus**: (1 + normalized award wins) gives a boost to award-winning talent

        Note: The most central talent are not always the most famous — they bridge
        the most creative communities.
        """)

    st.dataframe(
        top_influential[["name", "primary_role", "title_count", "avg_imdb",
                         "pagerank", "influence_score", "top_genre",
                         "award_title_count"]].head(50),
        hide_index=True,
        column_config={
            "avg_imdb": st.column_config.NumberColumn("Avg IMDb", format="%.2f"),
            "pagerank": st.column_config.NumberColumn("PageRank", format="%.6f"),
            "influence_score": st.column_config.NumberColumn("Influence", format="%.6f"),
            "award_title_count": "Award Titles",
        },
        height=500,
    )
else:
    st.info("Network data required. Run `scripts/12_precompute_network.py` to generate collaboration graph.")

st.divider()

# ─── Section 4: Rankings ────────────────────────────────────────────────────
st.markdown(section_header_html("Rankings"), unsafe_allow_html=True)

if has_network:
    tab_dir, tab_act, tab_wri = st.tabs(["Directors", "Actors", "Writers"])

    for tab, role_key, role_name in [
        (tab_dir, "DIRECTOR", "Directors"),
        (tab_act, "ACTOR", "Actors"),
        (tab_wri, "WRITER", "Writers"),
    ]:
        with tab:
            sort_col = st.selectbox(
                f"Rank {role_name} by",
                ["influence_score", "most_titles", "highest_avg_imdb"],
                format_func=lambda x: x.replace("_", " ").title(),
                key=f"rank_sort_{role_key}",
            )

            ranked = get_rankings(person_stats, role_filter=role_key, sort_by=sort_col, top_k=50)

            if not ranked.empty:
                st.dataframe(
                    ranked[["name", "title_count", "avg_imdb", "influence_score",
                            "top_genre", "career_start", "career_end"]].reset_index(drop=True),
                    hide_index=True,
                    column_config={
                        "avg_imdb": st.column_config.NumberColumn("Avg IMDb", format="%.2f"),
                        "influence_score": st.column_config.NumberColumn("Influence", format="%.6f"),
                    },
                    height=400,
                )

                # Compare 2-3 people
                compare_options = ranked["person_id"].head(20).tolist()
                compare_names = {pid: ranked[ranked["person_id"] == pid].iloc[0]["name"]
                                for pid in compare_options}
                compare_selected = st.multiselect(
                    "Compare (select 2-3)",
                    compare_options,
                    max_selections=3,
                    format_func=lambda x: compare_names.get(x, str(x)),
                    key=f"compare_{role_key}",
                )

                if len(compare_selected) >= 2:
                    compare_data = ranked[ranked["person_id"].isin(compare_selected)]
                    fig = go.Figure()
                    categories = ["Title Count", "Avg IMDb", "Career Span"]

                    for _, row in compare_data.iterrows():
                        span = (row.get("career_end", 2000) or 2000) - (row.get("career_start", 2000) or 2000)
                        fig.add_trace(go.Scatterpolar(
                            r=[row["title_count"] / max(compare_data["title_count"].max(), 1) * 100,
                               (row.get("avg_imdb", 5) or 5) / 10 * 100,
                               min(span / 50 * 100, 100)],
                            theta=categories,
                            fill="toself",
                            name=row["name"],
                        ))
                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                        template=PLOTLY_TEMPLATE,
                        title="Side-by-Side Comparison",
                        height=400,
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No {role_name.lower()} found in the network.")
else:
    st.info("Network data required. Run `scripts/12_precompute_network.py` to generate collaboration graph.")

# ─── Footer ─────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="border-top:1px solid #333;padding:16px 0;color:#666;'
    'font-size:0.8em;text-align:center;">'
    'Hypothetical merger for academic analysis. Data is a snapshot (mid-2023). '
    'All insights are illustrative, not prescriptive.'
    '</div>',
    unsafe_allow_html=True,
)
