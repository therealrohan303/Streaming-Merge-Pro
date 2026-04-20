"""Page 6: Interactive Lab — Greenlight Studio, Franchise Explorer, Draft Room."""

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from src.config import (
    ALL_PLATFORMS,
    CARD_ACCENT,
    CARD_BG,
    CARD_BORDER,
    CARD_TEXT,
    CARD_TEXT_MUTED,
    MERGED_PLATFORMS,
    PLATFORMS,
    PLOTLY_TEMPLATE,
    PRECOMPUTED_DIR,
)
from src.data.loaders import (
    deduplicate_titles,
    load_enriched_titles,
    load_genome_vectors,
    load_greenlight_model,
    load_imdb_principals,
    load_person_stats,
    load_tfidf_vectorizer,
)
from src.analysis.discovery import vibe_search
from src.analysis.scoring import bayesian_imdb, compute_quality_score, format_votes
from src.analysis.lab import predict_title
from src.ui.badges import (
    page_header_html,
    platform_badges_html,
    section_header_html,
)
from src.ui.session import init_session_state

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Interactive Lab", page_icon="🔬", layout="wide")
init_session_state()

# ─── Session state (namespaced per tab) ──────────────────────────────────────
for _key, _default in [
    # Greenlight Studio
    ("gs_result", None),
    # Franchise Explorer
    ("fe_selected", None),
    # Draft Room
    ("draft_active", False),
    ("draft_round", 0),
    ("draft_user_picks", []),
    ("draft_ai_picks", []),
    ("draft_available_pool", None),
    ("draft_settings", {}),
    ("draft_complete", False),
    ("draft_gs_genre", None),  # Genre Specialist dominant genre
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default

# ─── Shared data (loaded once) ───────────────────────────────────────────────
@st.cache_resource
def _load_catalog_tfidf_matrix():
    """Returns (vectorizer, sparse_matrix, ids, enriched_df) cached once per session."""
    vec = load_tfidf_vectorizer()
    enriched = load_enriched_titles()
    enriched = deduplicate_titles(enriched)
    descs = enriched["description"].fillna("").tolist()
    mat = vec.transform(descs)
    return vec, mat, enriched["id"].tolist(), enriched


@st.cache_data
def _get_enriched():
    df = load_enriched_titles()
    df = deduplicate_titles(df)
    df["quality_score"] = compute_quality_score(df)
    return df


@st.cache_data
def _compute_franchise_table(_enriched_hash, enriched_df):
    """Precompute per-franchise stats. Use _enriched_hash as cache key."""
    if "collection_name" not in enriched_df.columns:
        return pd.DataFrame()
    has_cols = enriched_df.copy()
    has_cols = has_cols[has_cols["collection_name"].notna()].copy()
    if has_cols.empty:
        return pd.DataFrame()

    rows = []
    for name, grp in has_cols.groupby("collection_name"):
        imdb_vals = pd.to_numeric(grp["imdb_score"], errors="coerce").dropna()
        votes_vals = pd.to_numeric(grp.get("imdb_votes", pd.Series(dtype=float)), errors="coerce").fillna(0)
        # Bayesian weighted avg
        if len(imdb_vals) > 0:
            m = 10_000
            c = 6.5
            weighted_imdb = ((votes_vals * imdb_vals).sum() + m * c) / (votes_vals.sum() + m) if votes_vals.sum() > 0 else imdb_vals.mean()
        else:
            weighted_imdb = float("nan")

        award_col = grp.get("award_wins", pd.Series(dtype=float))
        total_awards = pd.to_numeric(award_col, errors="coerce").fillna(0).sum()

        # Platform distribution
        plat_dist = {}
        if "platforms" in grp.columns:
            for plats in grp["platforms"].dropna():
                if isinstance(plats, (list, np.ndarray)):
                    for p in plats:
                        plat_dist[p] = plat_dist.get(p, 0) + 1
        elif "platform" in grp.columns:
            for p in grp["platform"].dropna():
                plat_dist[p] = plat_dist.get(p, 0) + 1

        latest_year = pd.to_numeric(grp["release_year"], errors="coerce").max()

        rows.append({
            "collection_name": name,
            "title_count": len(grp),
            "avg_imdb": round(weighted_imdb, 2) if not math.isnan(weighted_imdb) else None,
            "total_awards": int(total_awards),
            "platform_dist": plat_dist,
            "latest_year": int(latest_year) if pd.notna(latest_year) else None,
        })

    result = pd.DataFrame(rows)
    return result[result["title_count"] >= 2].reset_index(drop=True)


@st.cache_data
def _get_acquisition_targets():
    path = PRECOMPUTED_DIR / "strategic_analysis" / "acquisition_targets.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_data
def _box_office_stats(_enriched):
    """Per-genre median box-office/budget ratio. Returns (genre_medians, global_median, counts)."""
    df = _enriched[
        _enriched.get("box_office_usd", pd.Series(dtype=float)).notna()
        & _enriched.get("budget_usd", pd.Series(dtype=float)).notna()
        & (_enriched.get("budget_usd", pd.Series(0)) > 0)
    ].copy()
    if df.empty:
        return {}, 1.8, {}
    df["_ratio"] = df["box_office_usd"] / df["budget_usd"]
    per_genre = {}
    for _, r in df.iterrows():
        gl = r.get("genres")
        if isinstance(gl, (list, np.ndarray)):
            for g in gl:
                per_genre.setdefault(g, []).append(r["_ratio"])
    medians = {g: float(np.median(v)) for g, v in per_genre.items() if len(v) >= 5}
    counts = {g: len(v) for g, v in per_genre.items()}
    return medians, float(df["_ratio"].median()), counts


@st.cache_data
def _platform_profiles(_enriched):
    """Per-platform median IMDb + movie ratio, used by Platform Fit."""
    out = {}
    for plat in ALL_PLATFORMS:
        sub = _enriched[_enriched["platform"] == plat] if "platform" in _enriched.columns else _enriched.iloc[0:0]
        imdb_series = pd.to_numeric(sub["imdb_score"], errors="coerce").dropna() if len(sub) else pd.Series(dtype=float)
        out[plat] = {
            "median_imdb": float(imdb_series.median()) if not imdb_series.empty else 6.5,
            "movie_ratio": float((sub["type"] == "Movie").mean()) if len(sub) else 0.5,
            "count": len(sub),
        }
    return out


# ─── Card-render helpers (mirrored from Discovery Engine) ────────────────────
def _gs_sim_badge_html(score: float) -> str:
    pct = int(round(max(0.0, min(1.0, score)) * 100))
    if score >= 0.75:
        bg, fg = "rgba(46,204,113,0.18)", "#2ecc71"
    elif score >= 0.60:
        bg, fg = "rgba(243,156,18,0.18)", "#f39c12"
    else:
        bg, fg = "rgba(136,136,136,0.18)", CARD_TEXT_MUTED
    return (
        f'<span style="background:{bg};color:{fg};padding:3px 9px;'
        f'border-radius:10px;font-size:0.72em;font-weight:600;white-space:nowrap;">'
        f'{pct}% match</span>'
    )


def _gs_poster_html(url, title, platform_keys, width=90, height=130) -> str:
    initial = (title[0].upper() if title else "?")
    placeholder = (
        f'<div style="width:{width}px;height:{height}px;background:#2a2a3e;'
        f'border-radius:6px;display:flex;align-items:center;justify-content:center;'
        f'font-size:{max(width//3,14)}px;font-weight:700;color:#fff;flex-shrink:0;margin:0 auto;">'
        f'{initial}</div>'
    )
    if not url or str(url) in ("nan", "None", ""):
        return placeholder
    return (
        f'<img src="{url}" width="{width}" height="{height}" '
        f'style="object-fit:cover;border-radius:6px;flex-shrink:0;display:block;margin:0 auto;" />'
    )


def _gs_genre_pills_html(genres, max_n=3) -> str:
    if genres is None:
        return ""
    lst = list(genres) if isinstance(genres, np.ndarray) else genres
    if not isinstance(lst, (list, tuple)):
        return ""
    return " ".join(
        f'<span style="background:#2a2a3e;color:{CARD_TEXT_MUTED};padding:1px 6px;'
        f'border-radius:8px;font-size:0.65em;margin-right:2px;">{str(g).title()}</span>'
        for g in lst[:max_n]
    )


def _gs_render_similar_card(row, score: float) -> None:
    """Compact vertical card for Most Similar Titles grid."""
    title = row.get("title", "Untitled")
    year = row.get("release_year")
    imdb = row.get("imdb_score")
    votes = row.get("imdb_votes")
    genres = row.get("genres", [])
    platforms = row.get("platforms") or ([row["platform"]] if row.get("platform") else [])
    poster_url = row.get("poster_url")

    imdb_str = f"{imdb:.1f}" if pd.notna(imdb) else "N/A"
    votes_str = format_votes(votes) if pd.notna(votes) and votes else ""
    year_str = str(int(year)) if pd.notna(year) else ""
    badge_html = platform_badges_html(platforms)
    genre_html = _gs_genre_pills_html(genres, max_n=2)
    sim_badge = _gs_sim_badge_html(score)
    poster_html = _gs_poster_html(poster_url, title, platforms, width=110, height=160)
    votes_html = (
        f' <span style="color:{CARD_TEXT_MUTED};">({votes_str})</span>' if votes_str else ""
    )

    st.markdown(
        f'<div style="background:{CARD_BG};border:1px solid {CARD_BORDER};'
        f'border-top:3px solid {CARD_ACCENT};border-radius:8px;padding:8px;'
        f'height:100%;display:flex;flex-direction:column;gap:6px;">'
        f'{poster_html}'
        f'<div style="font-weight:700;color:{CARD_TEXT};font-size:0.82em;line-height:1.2;'
        f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;" title="{title}">{title}</div>'
        f'<div style="color:{CARD_TEXT_MUTED};font-size:0.72em;">{year_str}</div>'
        f'<div>{badge_html}</div>'
        f'<div style="color:{CARD_TEXT};font-size:0.75em;">IMDb {imdb_str}{votes_html}</div>'
        f'<div>{genre_html}</div>'
        f'<div style="margin-top:auto;">{sim_badge}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


ALL_GENRES_LIST = [
    "action", "animation", "comedy", "crime", "documentation",
    "drama", "european", "family", "fantasy", "history",
    "horror", "music", "reality", "romance", "scifi",
    "sport", "thriller", "war", "western",
]

# ─── Page header ─────────────────────────────────────────────────────────────
st.markdown(
    page_header_html(
        "Interactive Lab",
        "Pitch a concept, explore franchises, or compete in the streaming draft.",
    ),
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["Greenlight Studio", "Franchise Explorer", "Draft Room"])

# =============================================================================
# TAB 1: GREENLIGHT STUDIO
# =============================================================================
with tab1:
    st.markdown(
        section_header_html(
            "Greenlight Studio",
            "Describe a concept and get a full analysis — predicted score, platform fit, talent, and catalog gaps.",
        ),
        unsafe_allow_html=True,
    )

    # ── Input form ────────────────────────────────────────────────────────────
    desc = st.text_area(
        "Concept Description",
        placeholder=(
            "Describe your concept in plain English — e.g. 'A slow-burn psychological thriller "
            "set in 1970s rural Japan, following a detective investigating a series of ritualistic "
            "disappearances that seem connected to local folklore.'"
        ),
        height=120,
        key="gs_desc",
    )

    content_type = st.radio(
        "Content Type",
        ["Movie", "Show"],
        horizontal=True,
        key="gs_type",
    )

    col_l, col_r = st.columns(2)

    if content_type == "Movie":
        with col_l:
            genres = st.multiselect(
                "Genres (up to 4)",
                ALL_GENRES_LIST,
                max_selections=4,
                key="gs_genres",
            )
            st.markdown(f"**Runtime** — {st.session_state.get('gs_runtime', 100)} min")
            runtime = st.slider("Runtime (min)", 60, 240, 100, key="gs_runtime", label_visibility="collapsed")
            st.markdown(f"**Release Year** — {st.session_state.get('gs_year', 2025)}")
            rel_year = st.slider("Release Year", 1970, 2030, 2025, key="gs_year", label_visibility="collapsed")
            country = st.selectbox(
                "Production Country",
                ["US/UK", "Europe", "Asia Pacific", "Other"],
                key="gs_country",
            )
        with col_r:
            age_cert = st.radio(
                "Age Certification",
                ["G/PG", "PG-13", "R/NC-17"],
                horizontal=True,
                key="gs_cert_movie",
            )
            budget_m = st.slider(
                "Budget (USD millions)",
                min_value=1, max_value=500, value=25, step=1,
                format="$%dM",
                key="gs_budget_m",
            )
            st.caption(
                "e.g. $8M indie · $45M mid-budget · $120M studio tent-pole · $300M franchise blockbuster"
            )
            has_franchise = st.toggle("Part of an existing franchise?", key="gs_franchise")

        # Map inputs to model features
        country_map = {"US/UK": 2, "Europe": 1, "Asia Pacific": 1, "Other": 0}
        cert_map_m = {"G/PG": 0, "PG-13": 1, "R/NC-17": 2}
        if   budget_m < 10:  budget_tier_int = 1
        elif budget_m < 50:  budget_tier_int = 2
        elif budget_m < 200: budget_tier_int = 3
        else:                budget_tier_int = 4
        features_dict = {
            "genres": genres,
            "runtime": runtime,
            "release_year": rel_year,
            "production_country_tier": country_map.get(country, 0),
            "age_cert_tier": cert_map_m.get(age_cert, 2),
            "budget_tier": budget_tier_int,
            "has_franchise": int(has_franchise),
            "award_genre_avg": 0,  # computed below after enriched load
            "decade": (rel_year // 10) * 10,
        }

    else:  # Show
        with col_l:
            genres = st.multiselect(
                "Genres (up to 4)",
                ALL_GENRES_LIST,
                max_selections=4,
                key="gs_genres",
            )
            st.markdown(f"**Episode Runtime** — {st.session_state.get('gs_ep_runtime', 45)} min")
            runtime = st.slider("Episode Runtime (min)", 20, 90, 45, key="gs_ep_runtime", label_visibility="collapsed")
            st.markdown(f"**Planned Seasons** — {st.session_state.get('gs_seasons', 1)}")
            num_seasons = st.slider("Planned Seasons", 1, 8, 1, key="gs_seasons", label_visibility="collapsed")
            st.markdown(f"**Release Year** — {st.session_state.get('gs_year', 2025)}")
            rel_year = st.slider("Release Year", 1970, 2030, 2025, key="gs_year", label_visibility="collapsed")
        with col_r:
            age_cert = st.radio(
                "Age Certification",
                ["TV-G/TV-PG", "TV-14", "TV-MA"],
                horizontal=True,
                key="gs_cert_show",
            )
            country = st.selectbox(
                "Production Country",
                ["US/UK", "Europe", "Asia Pacific", "Other"],
                key="gs_country",
            )

        country_map = {"US/UK": 2, "Europe": 1, "Asia Pacific": 1, "Other": 0}
        cert_map_s = {"TV-G/TV-PG": 0, "TV-14": 1, "TV-MA": 2}
        features_dict = {
            "genres": genres,
            "runtime": runtime,
            "release_year": rel_year,
            "production_country_tier": country_map.get(country, 0),
            "age_cert_tier": cert_map_s.get(age_cert, 2),
            "num_seasons": num_seasons,
            "award_genre_avg": 0,
            "decade": (rel_year // 10) * 10,
        }

    consult_btn = st.button("Consult the Studio", type="primary", use_container_width=True)

    # ── Output ────────────────────────────────────────────────────────────────
    if consult_btn:
        with st.spinner("Analyzing your concept…"):
            enriched = _get_enriched()
            model_key = "movie" if content_type == "Movie" else "show"
            model = load_greenlight_model(model_key)

            # Compute award_genre_avg at runtime
            if "award_wins" in enriched.columns and genres:
                primary_genre = genres[0]
                genre_mask = enriched["genres"].apply(
                    lambda g: isinstance(g, (list, np.ndarray)) and primary_genre in g
                )
                genre_awards = enriched.loc[genre_mask, "award_wins"].dropna()
                features_dict["award_genre_avg"] = float(genre_awards.mean()) if len(genre_awards) > 0 else 0.0

            if model is None:
                st.error("Model not found. Run `scripts/10_train_greenlight_predictor.py` first.")
            else:
                result = predict_title(model, features_dict, ALL_GENRES_LIST)
                st.session_state.gs_result = result

    # Render cards if result exists
    if st.session_state.gs_result:
        res = st.session_state.gs_result
        enriched = _get_enriched()
        pred = res["prediction"]
        genres_sel = st.session_state.get("gs_genres", [])
        desc_text = st.session_state.get("gs_desc", "").strip()
        budget_m = st.session_state.get("gs_budget_m", 25)

        # Quality band
        if   pred < 5.5:  band_label, band_color = "Likely to underperform", "#e74c3c"
        elif pred < 6.5:  band_label, band_color = "Moderate — competitive but undistinguished", "#f39c12"
        elif pred < 7.5:  band_label, band_color = "Strong — platform-worthy", "#2ecc71"
        else:             band_label, band_color = "Exceptional — likely award/prestige tier", "#FFD700"

        cv_rmse_val = res.get("cv_rmse")
        cv_str_for_footnote = f"{cv_rmse_val:.3f}" if isinstance(cv_rmse_val, (int, float)) else "N/A"
        gauge_pct = max(0, min(100, (pred - 1) / 9 * 100))

        # ── Row 1: Predicted IMDb Score | Box Office Projection (movies) ─────
        col_r1a, col_r1b = st.columns(2)

        with col_r1a:
            st.markdown(
                section_header_html(
                    "Predicted IMDb Score",
                    "Based on comparable titles in the catalog.",
                ),
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div style="background:{CARD_BG};border:1px solid {CARD_BORDER};
                            border-top:3px solid {band_color};border-radius:8px;padding:16px;">
                  <div style="display:flex;align-items:baseline;gap:12px;margin-bottom:8px;">
                    <span style="font-size:2.8em;font-weight:800;color:{band_color};">{pred:.1f}</span>
                    <span style="color:{CARD_TEXT_MUTED};font-size:0.95em;">/ 10 IMDb</span>
                  </div>
                  <div style="background:{band_color}22;color:{band_color};padding:5px 12px;
                              border-radius:10px;font-size:0.82em;border:1px solid {band_color};
                              font-weight:600;display:inline-block;margin-bottom:10px;">
                    {band_label}
                  </div>
                  <div style="position:relative;height:18px;border-radius:9px;overflow:hidden;
                              background:linear-gradient(90deg,#e74c3c 0%,#e74c3c 22%,
                                #f39c12 22%,#f39c12 33%,#f1c40f 33%,#f1c40f 44%,
                                #2ecc71 44%,#2ecc71 78%,#FFD700 78%,#FFD700 100%);">
                    <div style="position:absolute;top:1px;left:calc({gauge_pct:.1f}% - 5px);
                                width:10px;height:16px;background:#fff;border-radius:3px;
                                box-shadow:0 0 4px rgba(0,0,0,0.6);"></div>
                  </div>
                  <div style="display:flex;justify-content:space-between;
                              font-size:0.72em;color:{CARD_TEXT_MUTED};margin-top:3px;">
                    <span>1 — Poor</span><span>6.5</span><span>8 — Great</span><span>10</span>
                  </div>
                  <div style="color:{CARD_TEXT_MUTED};font-size:0.72em;margin-top:8px;">
                    GradientBoostingRegressor · 5-fold CV RMSE {cv_str_for_footnote}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_r1b:
            if content_type == "Movie":
                st.markdown(
                    section_header_html(
                        "Box Office Projection",
                        "Heuristic estimate from historic budget → box office ratios.",
                    ),
                    unsafe_allow_html=True,
                )
                genre_medians, global_median, genre_counts = _box_office_stats(enriched)
                if genres_sel:
                    ratios = [genre_medians.get(g, global_median) for g in genres_sel]
                    genre_mult = float(np.mean(ratios)) if ratios else global_median
                    n_comparables = sum(genre_counts.get(g, 0) for g in genres_sel)
                else:
                    genre_mult = global_median
                    n_comparables = sum(genre_counts.values())

                if   pred >= 7.5: quality_mult = 1.40
                elif pred >= 7.0: quality_mult = 1.15
                elif pred <= 5.5: quality_mult = 0.70
                else:             quality_mult = 1.00

                cert_current = st.session_state.get("gs_cert_movie", "PG-13")
                if   cert_current == "G/PG":     cert_mult = 1.10
                elif cert_current == "R/NC-17":  cert_mult = 0.75
                else:                            cert_mult = 1.00

                projected = budget_m * 1e6 * genre_mult * quality_mult * cert_mult
                low = projected * 0.6
                high = projected * 1.4

                def _fmt_money(v):
                    if v >= 1e9:
                        return f"${v/1e9:.2f}B"
                    if v >= 1e6:
                        return f"${v/1e6:.0f}M"
                    return f"${v:,.0f}"

                st.markdown(
                    f"""
                    <div style="background:{CARD_BG};border:1px solid {CARD_BORDER};
                                border-top:3px solid {CARD_ACCENT};border-radius:8px;padding:16px;">
                      <div style="color:{CARD_TEXT_MUTED};font-size:0.78em;">Projected worldwide gross</div>
                      <div style="font-size:2.6em;font-weight:800;color:{CARD_ACCENT};line-height:1.1;">
                        {_fmt_money(projected)}
                      </div>
                      <div style="color:{CARD_TEXT};font-size:0.85em;margin-top:4px;">
                        Range: <b>{_fmt_money(low)}</b> – <b>{_fmt_money(high)}</b>
                      </div>
                      <div style="color:{CARD_TEXT_MUTED};font-size:0.75em;margin-top:6px;">
                        Budget ${budget_m}M × genre ratio {genre_mult:.1f}× × quality {quality_mult:.2f}× × cert {cert_mult:.2f}×
                      </div>
                      <div style="color:{CARD_TEXT_MUTED};font-size:0.72em;margin-top:4px;">
                        Based on {n_comparables} comparable movies with budget+box-office data.
                      </div>
                      <div style="color:{CARD_TEXT_MUTED};font-size:0.7em;margin-top:8px;font-style:italic;">
                        Heuristic estimate — not a full financial model.
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    section_header_html(
                        "Show Economics",
                        "Box office projection is movie-only. Series ROI is driven by episode count and subscriber retention.",
                    ),
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div style="background:{CARD_BG};border:1px solid {CARD_BORDER};'
                    f'border-radius:8px;padding:16px;color:{CARD_TEXT_MUTED};font-size:0.88em;">'
                    f'No theatrical revenue model applies to streaming shows. Compare against the '
                    f'Similar Titles panel below for benchmark performance.'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # ── Row 2: Most Similar Titles (full width) ──────────────────────────
        st.markdown(
            section_header_html(
                "Most Similar Titles",
                "Titles closest to your concept by vibe, genre, and quality.",
            ),
            unsafe_allow_html=True,
        )

        if not desc_text:
            st.markdown(
                f'<div style="background:{CARD_BG};border:1px solid {CARD_BORDER};'
                f'border-radius:8px;padding:20px;text-align:center;color:{CARD_TEXT_MUTED};'
                f'font-size:0.9em;">Enter a concept description to see similar titles.</div>',
                unsafe_allow_html=True,
            )
        else:
            try:
                gv, gmap = load_genome_vectors()
                results_v, _signals = vibe_search(
                    desc_text, enriched,
                    genome_vectors=gv, genome_id_map=gmap, enriched_df=enriched,
                    top_k=20, min_imdb=None, year_range=None, min_votes=0,
                )
                if results_v is None or results_v.empty:
                    st.info("No similar titles found for this concept.")
                else:
                    rated = results_v[results_v["imdb_score"].notna()].head(5)
                    if len(rated) < 5:
                        unrated = results_v[results_v["imdb_score"].isna()].head(5 - len(rated))
                        rated = pd.concat([rated, unrated], ignore_index=False)
                    top_rows = rated.head(5)

                    grid_cols = st.columns(5)
                    for gi, (_, trow) in enumerate(top_rows.iterrows()):
                        badge_score = min(1.0, float(trow.get("vibe_score", 0)) / 0.40)
                        with grid_cols[gi]:
                            _gs_render_similar_card(trow.to_dict(), badge_score)
            except Exception as e:
                st.info(f"Similarity search unavailable: {e}")

        # ── Row 3: Platform Fit | Catalog Gap Signal ─────────────────────────
        col_r3a, col_r3b = st.columns(2)

        with col_r3a:
            st.markdown(
                section_header_html(
                    "Platform Fit",
                    "How well does this concept align with each platform's catalog?",
                ),
                unsafe_allow_html=True,
            )
            profiles = _platform_profiles(enriched)

            if not genres_sel:
                st.info("Select at least one genre to compute Platform Fit.")
            else:
                fit_scores = {}
                min_overlap = 2 if len(genres_sel) >= 2 else 1
                for plat in ALL_PLATFORMS:
                    sub = enriched[enriched["platform"] == plat] if "platform" in enriched.columns else enriched.iloc[0:0]
                    if len(sub) == 0:
                        fit_scores[plat] = 0
                        continue

                    g_component = sub["genres"].apply(
                        lambda gl: isinstance(gl, (list, np.ndarray))
                        and len(set(gl) & set(genres_sel)) >= min_overlap
                    ).mean() * 100

                    delta = abs(pred - profiles[plat]["median_imdb"])
                    q_component = max(0.0, 1 - max(0.0, delta - 0.5) / 1.5) * 100

                    mr = profiles[plat]["movie_ratio"]
                    if content_type == "Movie":
                        t_component = min(mr / 0.7, 1.0) * 100
                    else:
                        t_component = min((1 - mr) / 0.7, 1.0) * 100

                    fit = round(0.40 * g_component + 0.30 * q_component + 0.30 * t_component)
                    fit_scores[plat] = max(0, min(100, fit))

                sorted_plats = sorted(fit_scores.keys(), key=lambda p: fit_scores[p], reverse=True)
                plat_names = [PLATFORMS.get(p, {}).get("name", p) for p in sorted_plats]
                plat_colors = [PLATFORMS.get(p, {}).get("color", "#555") for p in sorted_plats]
                plat_vals = [fit_scores[p] for p in sorted_plats]

                # Top bar gets a star annotation
                y_labels = list(plat_names)
                if y_labels:
                    y_labels[0] = f"⭐ {y_labels[0]}"

                fig_fit = go.Figure(go.Bar(
                    x=plat_vals,
                    y=y_labels,
                    orientation="h",
                    marker_color=plat_colors,
                    text=[f"{v}" for v in plat_vals],
                    textposition="outside",
                    cliponaxis=False,
                ))
                fig_fit.update_layout(
                    template=PLOTLY_TEMPLATE,
                    height=260,
                    margin=dict(l=0, r=30, t=10, b=10),
                    xaxis=dict(title="Fit Score (0–100)", range=[0, 110]),
                    yaxis=dict(title="", autorange="reversed"),
                )
                st.plotly_chart(fig_fit, use_container_width=True)

                top_plat_name = PLATFORMS.get(sorted_plats[0], {}).get("name", sorted_plats[0])
                st.markdown(
                    f'<div style="color:{CARD_TEXT};font-size:0.88em;padding:6px 2px;">'
                    f'Based on catalog composition, <b style="color:{CARD_ACCENT};">{top_plat_name}</b> '
                    f'is the strongest strategic home for this concept.'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        with col_r3b:
            st.markdown(
                section_header_html(
                    "Catalog Gap Signal",
                    "How saturated is the merged catalog in your selected genres?",
                ),
                unsafe_allow_html=True,
            )
            if not genres_sel:
                st.info("Select genres to see catalog gap analysis.")
            else:
                merged_df = enriched[enriched["platform"].isin(MERGED_PLATFORMS)] if "platform" in enriched.columns else enriched.iloc[0:0]
                merged_total = max(len(merged_df), 1)
                for genre in genres_sel:
                    share = merged_df["genres"].apply(
                        lambda g: isinstance(g, (list, np.ndarray)) and genre in g
                    ).mean() if len(merged_df) else 0.0
                    share_pct = share * 100

                    if share < 0.15:
                        label, color, bg = "High Priority Gap", "#2ecc71", "rgba(46,204,113,0.10)"
                        detail = "This concept addresses a real acquisition need in the merged catalog."
                    elif share < 0.30:
                        label, color, bg = "Moderate Gap", "#f39c12", "rgba(243,156,18,0.10)"
                        detail = "The genre has room for differentiation."
                    else:
                        label, color, bg = "Saturated Category", "#e67e22", "rgba(230,126,34,0.10)"
                        detail = "Well-represented — your concept will need strong differentiation to stand out."

                    st.markdown(
                        f'<div style="background:{bg};border:1px solid {color};'
                        f'border-left:4px solid {color};border-radius:6px;'
                        f'padding:10px 14px;margin-bottom:8px;">'
                        f'<div style="color:{color};font-weight:700;font-size:0.95em;">'
                        f'{label} — {genre.title()}</div>'
                        f'<div style="color:{CARD_TEXT};font-size:0.85em;margin-top:2px;">'
                        f'{genre.title()} represents <b>{share_pct:.1f}%</b> of the merged catalog. {detail}'
                        f'</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        # ── Row 4: Directors | Cast ──────────────────────────────────────────
        st.markdown(
            section_header_html(
                "Talent Recommendations",
                "Directors and cast whose catalog work most closely fits your concept.",
            ),
            unsafe_allow_html=True,
        )

        principals = load_imdb_principals()
        if principals.empty:
            st.info("Principals data unavailable — run `scripts/06_fetch_imdb_principals.py`.")
        elif not genres_sel:
            st.info("Select genres to see talent recommendations.")
        else:
            try:
                vec_gs, mat_gs, id_list_gs, enriched_with_desc = _load_catalog_tfidf_matrix()

                def _score_persons(role_filter, min_titles, min_imdb_floor):
                    pr = principals[principals["category"].str.lower().isin(role_filter)]
                    if pr.empty:
                        return []
                    needed_cols = [c for c in
                                   ["imdb_id", "id", "genres", "imdb_score", "platform", "poster_url", "title"]
                                   if c in enriched_with_desc.columns]
                    m = pr.merge(enriched_with_desc[needed_cols], on="imdb_id", how="inner")
                    if m.empty:
                        return []
                    m = m[m["genres"].apply(
                        lambda g: isinstance(g, (list, np.ndarray)) and bool(set(g) & set(genres_sel))
                    )]
                    if m.empty:
                        return []

                    if desc_text:
                        q_vec = vec_gs.transform([desc_text])
                        id_to_row = {tid: i for i, tid in enumerate(id_list_gs)}
                        m = m.assign(_row=m["id"].map(id_to_row))
                        m = m[m["_row"].notna()].copy()
                        if not m.empty:
                            sims = cosine_similarity(q_vec, mat_gs[m["_row"].astype(int).values]).flatten()
                            m["_sim"] = sims
                        else:
                            return []
                    else:
                        m["_sim"] = 0.0

                    grouped = m.groupby("person_id").agg(
                        name=("name", "first"),
                        avg_imdb=("imdb_score", "mean"),
                        title_count=("title", "count"),
                        vibe=("_sim", "mean"),
                        titles=("title", lambda s: list(dict.fromkeys(s))[:2]),
                        platforms=("platform", lambda s: list(dict.fromkeys(s))[:3]),
                    ).reset_index()

                    grouped = grouped[
                        (grouped["title_count"] >= min_titles)
                        & (grouped["avg_imdb"] >= min_imdb_floor)
                    ]
                    if grouped.empty:
                        return []

                    grouped["_qnorm"] = ((grouped["avg_imdb"] - 5.0) / 5.0).clip(0, 1)
                    grouped["rank_score"] = 0.6 * grouped["vibe"] + 0.4 * grouped["_qnorm"]
                    grouped = grouped.sort_values("rank_score", ascending=False)

                    seen, out = set(), []
                    for _, r in grouped.iterrows():
                        pid = r["person_id"]
                        if pid in seen:
                            continue
                        seen.add(pid)
                        out.append(r.to_dict())
                        if len(out) >= 5:
                            break
                    return out

                directors = _score_persons({"director"}, min_titles=2, min_imdb_floor=6.5)
                cast = _score_persons({"actor", "actress"}, min_titles=2, min_imdb_floor=6.0)

                def _render_person_card(p):
                    name = p.get("name", "?")
                    avg_i = p.get("avg_imdb", 0)
                    vibe = p.get("vibe", 0)
                    titles_list = p.get("titles", []) or []
                    plats_list = p.get("platforms", []) or []
                    imdb_c = "#2ecc71" if avg_i >= 7.5 else (CARD_ACCENT if avg_i >= 7.0 else CARD_TEXT_MUTED)
                    title_pills = " ".join(
                        f'<span style="background:#2a2a3e;color:{CARD_TEXT_MUTED};'
                        f'padding:1px 6px;border-radius:8px;font-size:0.7em;margin-right:3px;">{t}</span>'
                        for t in titles_list if t
                    )
                    plat_badges = platform_badges_html(plats_list) if plats_list else ""
                    vibe_badge = ""
                    if desc_text and vibe > 0:
                        vp = int(round(min(1.0, vibe / 0.40) * 100))
                        vibe_badge = (
                            f'<span style="background:rgba(0,180,166,0.15);color:{CARD_ACCENT};'
                            f'padding:1px 6px;border-radius:8px;font-size:0.72em;'
                            f'border:1px solid {CARD_ACCENT};margin-left:6px;">Match {vp}%</span>'
                        )
                    st.markdown(
                        f'<div style="background:{CARD_BG};border:1px solid {CARD_BORDER};'
                        f'border-radius:6px;padding:10px 12px;margin-bottom:6px;">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center;gap:6px;">'
                        f'<div><span style="font-weight:700;color:{CARD_TEXT};font-size:0.92em;">{name}</span>'
                        f'<span style="margin-left:6px;background:{imdb_c}22;color:{imdb_c};'
                        f'padding:1px 7px;border-radius:8px;font-size:0.72em;'
                        f'border:1px solid {imdb_c};">IMDb {avg_i:.1f}</span>'
                        f'{vibe_badge}</div>'
                        f'</div>'
                        f'<div style="margin-top:4px;">{title_pills}</div>'
                        f'<div style="margin-top:4px;">{plat_badges}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                tcol_d, tcol_a = st.columns(2)
                with tcol_d:
                    st.markdown(
                        f'<div style="color:{CARD_TEXT_MUTED};font-size:0.78em;font-weight:600;'
                        f'text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Directors</div>',
                        unsafe_allow_html=True,
                    )
                    if not directors:
                        st.caption("No directors matching this concept — recommendations are catalog-derived.")
                    else:
                        for p in directors:
                            _render_person_card(p)
                with tcol_a:
                    st.markdown(
                        f'<div style="color:{CARD_TEXT_MUTED};font-size:0.78em;font-weight:600;'
                        f'text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Cast</div>',
                        unsafe_allow_html=True,
                    )
                    if not cast:
                        st.caption("No cast matching this concept — recommendations are catalog-derived.")
                    else:
                        for p in cast:
                            _render_person_card(p)

            except Exception as e:
                st.info(f"Talent data unavailable: {e}")

        # ── Row 5: Model Card expander ───────────────────────────────────────
        with st.expander("Model Card", expanded=False):
            model = load_greenlight_model("movie" if content_type == "Movie" else "show")
            if model:
                cv_r = getattr(model, "cv_rmse_", None)
                base_r = getattr(model, "baseline_rmse_", None)
                t_size = getattr(model, "training_size_", "N/A")
                g_mean = getattr(model, "global_mean_", None)
                f_names = getattr(model, "feature_names_", [])

                cv_str = f"{cv_r:.3f}" if cv_r is not None else "N/A"
                base_str = f"{base_r:.3f}" if base_r is not None else "N/A"
                impr_str = f"{(1 - cv_r/base_r)*100:.1f}%" if (cv_r and base_r and base_r > 0) else "N/A"

                def _clean(f):
                    if f.startswith("genre_"):
                        return f.replace("genre_", "").title() + " (genre)"
                    return f.replace("_", " ").title()

                importances = pd.Series(model.feature_importances_, index=f_names)
                top5 = importances.nlargest(5)
                top5_display = [f"**{_clean(f)}** ({v:.3f})" for f, v in top5.items()]

                st.markdown(f"""
**Model type:** GradientBoostingRegressor (scikit-learn)

**Training data:** {t_size:,} {content_type.lower()}s with IMDb score and votes > 1,000

**Cross-validation:** 5-fold

**CV RMSE:** {cv_str}

**Baseline RMSE:** {base_str} (predicting global mean {f"{g_mean:.2f}" if g_mean else "N/A"})

**Improvement:** {impr_str} over naive mean prediction

**Top 5 features:** {", ".join(top5_display)}

**Limitations:** Trained on catalog metadata only. Does not account for marketing,
star power, cultural timing, or release strategy.
Predictions are directional estimates, not precise forecasts.

**Data snapshot:** June 2025 catalog (pre-merger). Hypothetical merger for academic analysis.
""")


# =============================================================================
# TAB 2: FRANCHISE EXPLORER
# =============================================================================
with tab2:
    st.markdown(
        section_header_html(
            "Franchise Explorer",
            "Browse and analyze content franchises across all platforms — depth, quality, ownership, and gaps.",
        ),
        unsafe_allow_html=True,
    )

    enriched_fe = _get_enriched()

    if "collection_name" not in enriched_fe.columns or enriched_fe["collection_name"].notna().sum() < 2:
        st.info(
            "Franchise data requires TMDB enrichment (`collection_name` field). "
            "Run `scripts/08_enrich_tmdb.py` and `scripts/09_build_enriched_titles.py` first."
        )
    else:
        # Compute franchise table (cached)
        franchise_table = _compute_franchise_table(len(enriched_fe), enriched_fe)

        if franchise_table.empty:
            st.info("No franchises with 2+ titles found.")
        else:
            # ── Controls ──────────────────────────────────────────────────────
            col_s, col_sort, col_min = st.columns([3, 3, 1])
            with col_s:
                fe_search = st.text_input("Search franchises...", key="fe_search_input", placeholder="Search franchises...")
            with col_sort:
                fe_sort = st.radio(
                    "Sort By",
                    ["Catalog Depth", "Avg IMDb", "Prestige", "Recency"],
                    horizontal=True,
                    key="fe_sort_radio",
                )
            with col_min:
                fe_min = st.slider("Min titles", 2, 10, 2, key="fe_min_slider")

            # Filter + sort
            display_df = franchise_table[franchise_table["title_count"] >= fe_min].copy()
            if fe_search:
                display_df = display_df[
                    display_df["collection_name"].str.contains(fe_search, case=False, na=False)
                ]

            if fe_sort == "Catalog Depth":
                display_df = display_df.sort_values("title_count", ascending=False)
            elif fe_sort == "Avg IMDb":
                display_df = display_df.sort_values("avg_imdb", ascending=False)
            elif fe_sort == "Prestige":
                display_df = display_df.sort_values("total_awards", ascending=False)
            elif fe_sort == "Recency":
                display_df = display_df.sort_values("latest_year", ascending=False)

            # ── Franchise Grid ────────────────────────────────────────────────
            def _platform_bar_html(plat_dist):
                if not plat_dist:
                    return ""
                total = sum(plat_dist.values()) or 1
                segs = []
                for p in ALL_PLATFORMS:
                    cnt = plat_dist.get(p, 0)
                    if cnt == 0:
                        continue
                    pct = cnt / total * 100
                    color = PLATFORMS.get(p, {}).get("color", "#555")
                    segs.append(f'<div style="background:{color};width:{pct:.1f}%;height:100%;display:inline-block;" title="{PLATFORMS.get(p,{}).get("name",p)}: {cnt}"></div>')
                return (
                    f'<div style="width:100%;height:6px;border-radius:3px;overflow:hidden;'
                    f'background:#333;display:flex;margin:4px 0;">'
                    + "".join(segs) + "</div>"
                )

            rows_of_3 = [display_df.iloc[i:i+3] for i in range(0, len(display_df), 3)]
            for row_df in rows_of_3:
                cols = st.columns(3)
                for col, (_, frow) in zip(cols, row_df.iterrows()):
                    fname = frow["collection_name"]
                    tc = frow["title_count"]
                    ai = frow["avg_imdb"]
                    ly = frow.get("latest_year")
                    pd_dist = frow.get("platform_dist", {})
                    awards = frow.get("total_awards", 0)

                    if pd.notna(ai) and ai >= 7.5:
                        imdb_color = "#2ecc71"
                    elif pd.notna(ai) and ai >= 6.5:
                        imdb_color = CARD_ACCENT
                    else:
                        imdb_color = CARD_TEXT_MUTED

                    imdb_str = f"{ai:.1f}" if pd.notna(ai) else "N/A"
                    bar_html = _platform_bar_html(pd_dist)

                    with col:
                        st.markdown(
                            f'<div style="background:{CARD_BG};border:1px solid {CARD_BORDER};'
                            f'border-radius:8px;padding:12px;margin-bottom:8px;">'
                            f'<div style="font-weight:700;color:{CARD_TEXT};font-size:0.92em;'
                            f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;" title="{fname}">{fname}</div>'
                            f'<div style="display:flex;align-items:center;gap:8px;margin-top:4px;">'
                            f'<span style="color:{CARD_TEXT_MUTED};font-size:0.82em;">📽 {tc} titles</span>'
                            f'<span style="background:{imdb_color}22;color:{imdb_color};padding:1px 7px;'
                            f'border-radius:8px;font-size:0.78em;border:1px solid {imdb_color};">IMDb {imdb_str}</span>'
                            + (f'<span style="color:{CARD_TEXT_MUTED};font-size:0.78em;">{int(awards)} awards</span>' if awards > 0 else "")
                            + f'</div>'
                            f'{bar_html}'
                            f'<div style="color:{CARD_TEXT_MUTED};font-size:0.75em;margin-top:2px;">Latest: {ly or "N/A"}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        if st.button("View Details", key=f"fe_btn_{fname[:30]}", use_container_width=True):
                            st.session_state.fe_selected = fname
                            st.rerun()

            # ── Detail Panel ──────────────────────────────────────────────────
            if st.session_state.fe_selected:
                sel_name = st.session_state.fe_selected
                st.divider()
                col_title, col_close = st.columns([5, 1])
                with col_title:
                    st.markdown(
                        section_header_html(f"Franchise: {sel_name}"),
                        unsafe_allow_html=True,
                    )
                with col_close:
                    if st.button("✕ Close", key="fe_close"):
                        st.session_state.fe_selected = None
                        st.rerun()

                # Get all titles in this franchise
                sel_titles = enriched_fe[
                    enriched_fe["collection_name"] == sel_name
                ].sort_values("release_year").copy()

                if sel_titles.empty:
                    st.info("No titles found for this franchise.")
                else:
                    # Section 1: Horizontal poster strip
                    st.markdown(f'<div style="color:{CARD_TEXT_MUTED};font-size:0.8em;font-weight:600;text-transform:uppercase;margin-bottom:8px;">Titles in Franchise</div>', unsafe_allow_html=True)
                    strip_items = []
                    for _, tr in sel_titles.iterrows():
                        poster_url = tr.get("poster_url", "")
                        plat = tr.get("platform", tr.get("platforms", ""))
                        if isinstance(plat, (list, np.ndarray)):
                            plat = plat[0] if len(plat) > 0 else ""
                        plat_color = PLATFORMS.get(str(plat), {}).get("color", "#555")
                        initial = str(tr.get("title", "?"))[0].upper()
                        year_s = str(int(tr["release_year"])) if pd.notna(tr.get("release_year")) else ""
                        imdb_s = f"IMDb {tr['imdb_score']:.1f}" if pd.notna(tr.get("imdb_score")) else ""
                        if pd.notna(poster_url) and str(poster_url).startswith("http"):
                            img_html = f'<img src="{poster_url}" style="width:120px;height:170px;object-fit:cover;border-radius:4px;">'
                        else:
                            img_html = (
                                f'<div style="width:120px;height:170px;background:{plat_color};'
                                f'border-radius:4px;display:flex;align-items:center;justify-content:center;'
                                f'font-size:2em;font-weight:700;color:#fff;">{initial}</div>'
                            )
                        badge_html = platform_badges_html(str(plat)) if plat else ""
                        strip_items.append(
                            f'<div style="flex:0 0 auto;text-align:center;width:130px;">'
                            f'{img_html}'
                            f'<div style="font-weight:600;color:{CARD_TEXT};font-size:0.78em;'
                            f'margin-top:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">'
                            f'{tr.get("title","?")} </div>'
                            f'<div style="color:{CARD_TEXT_MUTED};font-size:0.72em;">{year_s}</div>'
                            f'<div style="font-size:0.72em;color:{CARD_ACCENT};">{imdb_s}</div>'
                            f'{badge_html}'
                            f'</div>'
                        )
                    st.markdown(
                        f'<div style="display:flex;gap:12px;overflow-x:auto;padding:8px 0;'
                        f'scrollbar-width:thin;">{"".join(strip_items)}</div>',
                        unsafe_allow_html=True,
                    )

                    # Section 2: Quality timeline
                    timeline_data = sel_titles[sel_titles["imdb_score"].notna() & sel_titles["release_year"].notna()].copy()
                    if not timeline_data.empty:
                        st.markdown(f'<div style="color:{CARD_TEXT_MUTED};font-size:0.8em;font-weight:600;text-transform:uppercase;margin:12px 0 4px 0;">Quality Over Time</div>', unsafe_allow_html=True)
                        votes_raw = pd.to_numeric(timeline_data.get("imdb_votes", pd.Series(dtype=float)), errors="coerce").fillna(1000)
                        max_v = votes_raw.max() or 1
                        sizes = (votes_raw / max_v * 20 + 6).tolist()

                        fig_tl = go.Figure()
                        fig_tl.add_trace(go.Scatter(
                            x=timeline_data["release_year"].tolist(),
                            y=timeline_data["imdb_score"].tolist(),
                            mode="lines+markers",
                            marker=dict(size=sizes, color=CARD_ACCENT),
                            line=dict(color=CARD_ACCENT, width=2),
                            text=timeline_data["title"].tolist(),
                            hovertemplate="%{text}<br>Year: %{x}<br>IMDb: %{y}<extra></extra>",
                        ))
                        # Award-winning stars
                        if "award_wins" in timeline_data.columns:
                            award_titles = timeline_data[timeline_data["award_wins"].fillna(0) > 0]
                            if not award_titles.empty:
                                fig_tl.add_trace(go.Scatter(
                                    x=award_titles["release_year"].tolist(),
                                    y=award_titles["imdb_score"].tolist(),
                                    mode="markers",
                                    marker=dict(symbol="star", size=16, color="#FFD700", line=dict(color="#fff", width=1)),
                                    name="Award winner",
                                    hovertemplate="%{text}<extra>Award winner</extra>",
                                    text=award_titles["title"].tolist(),
                                ))
                        fig_tl.add_hline(y=7.0, line_dash="dash", line_color=CARD_TEXT_MUTED,
                                         annotation_text="Quality threshold (7.0)", annotation_position="bottom right")
                        fig_tl.update_layout(
                            template=PLOTLY_TEMPLATE, height=280,
                            margin=dict(l=0, r=0, t=10, b=10),
                            showlegend=False,
                            xaxis_title="Year", yaxis_title="IMDb Score",
                        )
                        st.plotly_chart(fig_tl, use_container_width=True)

                    # Section 3: Platform ownership
                    st.markdown(f'<div style="color:{CARD_TEXT_MUTED};font-size:0.8em;font-weight:600;text-transform:uppercase;margin:8px 0 4px 0;">Platform Ownership</div>', unsafe_allow_html=True)
                    plat_dist_raw = {}
                    for _, tr in sel_titles.iterrows():
                        plat = tr.get("platform", tr.get("platforms", ""))
                        if isinstance(plat, (list, np.ndarray)):
                            for p in plat:
                                plat_dist_raw[str(p)] = plat_dist_raw.get(str(p), 0) + 1
                        elif plat:
                            plat_dist_raw[str(plat)] = plat_dist_raw.get(str(plat), 0) + 1

                    total_f = sum(plat_dist_raw.values()) or 1
                    own_rows = []
                    for p, cnt in sorted(plat_dist_raw.items(), key=lambda x: -x[1]):
                        pct = cnt / total_f * 100
                        color = PLATFORMS.get(p, {}).get("color", "#555")
                        name = PLATFORMS.get(p, {}).get("name", p)
                        own_rows.append(
                            f'<div style="display:flex;align-items:center;gap:8px;padding:3px 0;">'
                            f'<span style="background:{color};color:#fff;padding:1px 7px;border-radius:4px;font-size:0.78em;">{name}</span>'
                            f'<span style="color:{CARD_TEXT};font-size:0.85em;">{cnt} titles ({pct:.0f}%)</span>'
                            f'</div>'
                        )
                    st.markdown("".join(own_rows), unsafe_allow_html=True)

                    # Strategic insight
                    if plat_dist_raw:
                        top_plat = max(plat_dist_raw, key=plat_dist_raw.get)
                        top_cnt = plat_dist_raw[top_plat]
                        merged_cnt = sum(plat_dist_raw.get(p, 0) for p in MERGED_PLATFORMS)
                        top_name = PLATFORMS.get(top_plat, {}).get("name", top_plat)
                        if top_plat not in MERGED_PLATFORMS and top_cnt / total_f > 0.6:
                            insight_text = f"{top_name} dominates this franchise — {top_cnt} of {total_f} titles."
                        elif merged_cnt >= top_cnt:
                            insight_text = f"Netflix+Max leads with {merged_cnt} of {total_f} titles."
                        else:
                            gap = top_cnt - merged_cnt
                            insight_text = f"{top_name} leads with {top_cnt} of {total_f} titles — acquiring the remaining {gap} would consolidate this IP."
                        st.markdown(
                            f'<div style="background:rgba(0,180,166,0.1);border:1px solid {CARD_ACCENT};'
                            f'border-radius:6px;padding:8px 12px;margin-top:8px;">'
                            f''
                            f'<span style="color:{CARD_TEXT};font-size:0.88em;">{insight_text}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                    # Section 4: Franchise Health Score
                    avg_i = pd.to_numeric(sel_titles["imdb_score"], errors="coerce").mean()
                    latest_y = pd.to_numeric(sel_titles["release_year"], errors="coerce").max()
                    awards_total = pd.to_numeric(sel_titles.get("award_wins", pd.Series(dtype=float)), errors="coerce").fillna(0).sum() if "award_wins" in sel_titles.columns else 0

                    # Normalize: IMDb 0–10 → 0–1; recency 1970–2030 → 0–1; prestige per title 0–1
                    imdb_norm = (float(avg_i) - 1) / 9 if pd.notna(avg_i) else 0.5
                    recency_norm = (float(latest_y) - 1970) / 60 if pd.notna(latest_y) else 0.5
                    prestige_norm = min(1.0, float(awards_total) / (len(sel_titles) * 5 + 1))

                    health = round((imdb_norm * 0.4 + recency_norm * 0.3 + prestige_norm * 0.3) * 100)
                    if health < 40:
                        h_label, h_color = "Declining", "#e74c3c"
                    elif health < 65:
                        h_label, h_color = "Stable", "#f39c12"
                    elif health < 80:
                        h_label, h_color = "Thriving", "#2ecc71"
                    else:
                        h_label, h_color = "Legendary", "#FFD700"

                    st.markdown(
                        f'<div style="background:{CARD_BG};border:1px solid {CARD_BORDER};'
                        f'border-top:3px solid {h_color};border-radius:8px;padding:12px;margin-top:12px;'
                        f'display:inline-block;">'
                        f'<div style="color:{CARD_TEXT_MUTED};font-size:0.78em;text-transform:uppercase;letter-spacing:1px;">Franchise Health Score</div>'
                        f'<div style="font-size:2.4em;font-weight:800;color:{h_color};">{health}</div>'
                        f'<div style="color:{h_color};font-size:0.85em;font-weight:600;">{h_label}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # ── Franchise Gaps Section ────────────────────────────────────────
            st.divider()
            st.markdown(
                section_header_html(
                    "Franchise Gaps",
                    "Franchises where a competitor holds significantly more titles than the merged entity.",
                ),
                unsafe_allow_html=True,
            )

            gap_rows = []
            for _, frow in franchise_table.iterrows():
                pd_dist = frow.get("platform_dist", {})
                if not pd_dist:
                    continue
                merged_cnt = sum(pd_dist.get(p, 0) for p in MERGED_PLATFORMS)
                # Find top non-merged platform
                competitor_counts = {p: c for p, c in pd_dist.items() if p not in MERGED_PLATFORMS}
                if not competitor_counts:
                    continue
                top_comp = max(competitor_counts, key=competitor_counts.get)
                top_comp_cnt = competitor_counts[top_comp]
                gap_mag = top_comp_cnt - merged_cnt
                if gap_mag >= 3:
                    gap_rows.append({
                        "franchise": frow["collection_name"],
                        "competitor": top_comp,
                        "comp_count": top_comp_cnt,
                        "merged_count": merged_cnt,
                        "gap": gap_mag,
                    })

            gap_rows = sorted(gap_rows, key=lambda x: -x["gap"])[:5]
            if not gap_rows:
                st.info("No significant franchise gaps found (no competitor leads by 3+ titles in any franchise).")
            else:
                for gr in gap_rows:
                    comp_name = PLATFORMS.get(gr["competitor"], {}).get("name", gr["competitor"])
                    comp_color = PLATFORMS.get(gr["competitor"], {}).get("color", "#e74c3c")
                    st.markdown(
                        f'<div style="background:{CARD_BG};border:1px solid {CARD_BORDER};'
                        f'border-left:4px solid {comp_color};border-radius:8px;'
                        f'padding:12px 16px;margin-bottom:8px;display:flex;justify-content:space-between;align-items:center;">'
                        f'<div>'
                        f'<div style="font-weight:700;color:{CARD_TEXT};">{gr["franchise"]}</div>'
                        f'<div style="color:{CARD_TEXT_MUTED};font-size:0.82em;">'
                        f'{comp_name}: {gr["comp_count"]} titles | Netflix+Max: {gr["merged_count"]} titles</div>'
                        f'</div>'
                        f'<span style="background:{comp_color}22;color:{comp_color};padding:3px 10px;'
                        f'border-radius:8px;border:1px solid {comp_color};font-size:0.82em;font-weight:600;'
                        f'white-space:nowrap;">Competitor +{gr["gap"]}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )


# =============================================================================
# TAB 3: DRAFT ROOM
# =============================================================================
with tab3:
    st.markdown(
        section_header_html(
            "Draft Room",
            "Draft titles against an AI opponent — build competing streaming services and compare them.",
        ),
        unsafe_allow_html=True,
    )

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _genre_entropy(picks):
        genre_counts = {}
        for p in picks:
            g = p.get("genres", [])
            if isinstance(g, (list, np.ndarray)):
                for gg in g:
                    genre_counts[gg] = genre_counts.get(gg, 0) + 1
        total = sum(genre_counts.values()) or 1
        probs = [c / total for c in genre_counts.values()]
        return -sum(pp * math.log2(pp + 1e-9) for pp in probs)

    def _service_metrics(picks):
        if not picks:
            return {"avg_imdb": 0, "pct_above_7": 0, "prestige": 0, "diversity": 0, "freshness": 0, "total": 0}
        df = pd.DataFrame(picks)
        imdb = pd.to_numeric(df.get("imdb_score", pd.Series(dtype=float)), errors="coerce").dropna()
        avg_imdb = float(imdb.mean()) if len(imdb) > 0 else 0
        pct_above_7 = float((imdb >= 7.0).mean() * 100) if len(imdb) > 0 else 0
        award_wins = pd.to_numeric(df.get("award_wins_val", df.get("award_wins", pd.Series(0, index=df.index))), errors="coerce").fillna(0)
        prestige = float((award_wins > 0).mean() * 100) if len(award_wins) > 0 else 0
        diversity = _genre_entropy(picks)
        ry = pd.to_numeric(df.get("release_year", pd.Series(dtype=float)), errors="coerce")
        freshness = float((ry >= 2015).mean() * 100) if len(ry) > 0 else 0
        return {
            "total": len(picks),
            "avg_imdb": avg_imdb,
            "pct_above_7": pct_above_7,
            "prestige": prestige,
            "diversity": diversity,
            "freshness": freshness,
        }

    def _ai_pick(strategy, pool_df, gs_genre):
        """Pick the best title from pool_df per strategy. Returns a dict."""
        if pool_df.empty:
            return None
        if strategy == "Quality Maximizer":
            row = pool_df.sort_values("draft_value", ascending=False).iloc[0]
        elif strategy == "Genre Specialist":
            if gs_genre:
                def _has_gs(gl):
                    return isinstance(gl, (list, np.ndarray)) and gs_genre in gl
                genre_pool = pool_df[pool_df["genres"].apply(_has_gs)]
                if not genre_pool.empty:
                    row = genre_pool.sort_values("draft_value", ascending=False).iloc[0]
                else:
                    row = pool_df.sort_values("draft_value", ascending=False).iloc[0]
            else:
                row = pool_df.sort_values("draft_value", ascending=False).iloc[0]
        elif strategy == "Prestige Chaser":
            row = pool_df.sort_values(["award_wins_val", "draft_value"], ascending=[False, False]).iloc[0]
        elif strategy == "Volume Hoarder":
            pop_col = "tmdb_popularity" if "tmdb_popularity" in pool_df.columns else "draft_value"
            row = pool_df.sort_values(pop_col, ascending=False).iloc[0]
        else:
            row = pool_df.sort_values("draft_value", ascending=False).iloc[0]
        return row.to_dict()

    def _pick_card_html(pick_dict):
        title = pick_dict.get("title", "?")
        imdb = pick_dict.get("imdb_score", "")
        imdb_str = f"IMDb {float(imdb):.1f}" if pd.notna(imdb) and imdb != "" else ""
        return f"**{title}** {imdb_str}"

    # ── Setup Panel ───────────────────────────────────────────────────────────
    if not st.session_state.draft_active and not st.session_state.draft_complete:
        st.markdown(
            f'<div style="background:{CARD_BG};border:1px solid {CARD_BORDER};'
            f'border-radius:8px;padding:20px;max-width:700px;margin:0 auto;">',
            unsafe_allow_html=True,
        )
        st.markdown(f'<div style="color:{CARD_TEXT_MUTED};font-size:0.85em;text-align:center;margin-bottom:16px;">Configure your draft before starting</div>', unsafe_allow_html=True)

        strategy = st.radio(
            "AI Strategy",
            [
                "Quality Maximizer — always picks highest-rated available title",
                "Genre Specialist — locks onto 2-3 genres and drafts deeply",
                "Prestige Chaser — prioritizes award-winning titles",
                "Volume Hoarder — drafts high-popularity titles regardless of ratings",
            ],
            key="draft_strategy_choice",
        )
        # Extract clean strategy name
        strategy_name = strategy.split(" — ")[0]

        rounds = st.radio("Draft Rounds", [5, 8, 12], horizontal=True, key="draft_rounds_choice")

        pool_filter = st.radio(
            "Title Pool",
            ["All Platforms", "Merged Only (Netflix + Max)"] + [f"Genre: {g.title()}" for g in ALL_GENRES_LIST[:6]],
            horizontal=False,
            key="draft_pool_filter",
        )

        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("Start Draft", type="primary", use_container_width=True):
            # Build pool
            pool = _get_enriched().copy()
            pool = pool[pd.to_numeric(pool.get("imdb_votes", pd.Series(0, index=pool.index)), errors="coerce").fillna(0) > 5000].copy()

            if pool_filter == "Merged Only (Netflix + Max)":
                pool = pool[pool["platform"].isin(MERGED_PLATFORMS)] if "platform" in pool.columns else pool
            elif pool_filter.startswith("Genre: "):
                gf = pool_filter.replace("Genre: ", "").lower()
                pool = pool[pool["genres"].apply(lambda g: isinstance(g, (list, np.ndarray)) and gf in g)]

            pool["award_wins_val"] = pd.to_numeric(pool.get("award_wins", pd.Series(0, index=pool.index)), errors="coerce").fillna(0)
            max_aw = pool["award_wins_val"].max() or 1
            pool["bayesian_score"] = bayesian_imdb(
                pd.to_numeric(pool["imdb_score"], errors="coerce"),
                pd.to_numeric(pool.get("imdb_votes", pd.Series(10000, index=pool.index)), errors="coerce"),
            )
            pool["draft_value"] = (
                pool["bayesian_score"].fillna(6.5) * 0.5
                + np.log1p(pd.to_numeric(pool.get("imdb_votes", pd.Series(10000, index=pool.index)), errors="coerce").fillna(10000)) * 0.3
                + (pool["award_wins_val"] / max_aw) * 0.2
            )

            # Determine Genre Specialist genre
            gs_genre = None
            if "Genre Specialist" in strategy_name:
                genre_imdb = {}
                for g in ALL_GENRES_LIST:
                    mask = pool["genres"].apply(lambda gl: isinstance(gl, (list, np.ndarray)) and g in gl)
                    g_imdb = pd.to_numeric(pool.loc[mask, "bayesian_score"], errors="coerce").mean()
                    if pd.notna(g_imdb):
                        genre_imdb[g] = g_imdb
                if genre_imdb:
                    gs_genre = max(genre_imdb, key=genre_imdb.get)

            st.session_state.draft_active = True
            st.session_state.draft_complete = False
            st.session_state.draft_round = 0
            st.session_state.draft_user_picks = []
            st.session_state.draft_ai_picks = []
            st.session_state.draft_available_pool = pool.reset_index(drop=True)
            st.session_state.draft_settings = {
                "strategy": strategy_name,
                "rounds": rounds,
                "pool_filter": pool_filter,
            }
            st.session_state.draft_gs_genre = gs_genre
            st.rerun()

    # ── Active Draft ──────────────────────────────────────────────────────────
    elif st.session_state.draft_active and not st.session_state.draft_complete:
        settings = st.session_state.draft_settings
        rounds = settings["rounds"]
        strategy_name = settings["strategy"]
        current_round = st.session_state.draft_round

        pool = st.session_state.draft_available_pool.copy()
        # Remove already-picked ids
        picked_ids = set(
            [p.get("id") for p in st.session_state.draft_user_picks]
            + [p.get("id") for p in st.session_state.draft_ai_picks]
        )
        available = pool[~pool["id"].isin(picked_ids)].copy() if "id" in pool.columns else pool.copy()

        # Status bar
        st.markdown(
            f'<div style="background:{CARD_BG};border:1px solid {CARD_BORDER};'
            f'border-radius:6px;padding:8px 16px;margin-bottom:12px;'
            f'display:flex;justify-content:space-between;align-items:center;">'
            f'<span style="color:{CARD_TEXT};">Round <b style="color:{CARD_ACCENT};">{current_round + 1}</b> of {rounds}</span>'
            f'<span style="color:{CARD_TEXT_MUTED};font-size:0.85em;">Strategy: <b>{strategy_name}</b></span>'
            f'<span style="color:{CARD_TEXT};">Your picks: <b>{len(st.session_state.draft_user_picks)}</b> | '
            f'AI picks: <b>{len(st.session_state.draft_ai_picks)}</b></span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        col_pool, col_score = st.columns([3, 2])

        with col_pool:
            st.markdown(section_header_html("Available Titles"), unsafe_allow_html=True)
            pool_search = st.text_input("Search pool...", key="draft_pool_search")
            display_pool = available.copy()
            if pool_search:
                display_pool = display_pool[display_pool["title"].str.contains(pool_search, case=False, na=False)]
            display_pool = display_pool.sort_values("draft_value", ascending=False).head(30)

            for idx, row in display_pool.iterrows():
                col_info, col_btn = st.columns([4, 1])
                with col_info:
                    imdb_s = f"IMDb {row['imdb_score']:.1f}" if pd.notna(row.get("imdb_score")) else ""
                    year_s = str(int(row["release_year"])) if pd.notna(row.get("release_year")) else ""
                    plat = str(row.get("platform", ""))
                    badge = platform_badges_html(plat) if plat else ""
                    poster_url = row.get("poster_url", "")
                    plat_color = PLATFORMS.get(plat, {}).get("color", "#555")
                    initial = str(row.get("title", "?"))[0].upper()
                    if pd.notna(poster_url) and str(poster_url).startswith("http"):
                        thumb = f'<img src="{poster_url}" style="width:50px;height:72px;object-fit:cover;border-radius:3px;vertical-align:middle;">'
                    else:
                        thumb = (
                            f'<div style="display:inline-block;width:50px;height:72px;background:{plat_color};'
                            f'border-radius:3px;text-align:center;line-height:72px;font-weight:700;'
                            f'color:#fff;font-size:1.2em;vertical-align:middle;">{initial}</div>'
                        )
                    st.markdown(
                        f'<div style="display:flex;gap:8px;align-items:center;padding:4px 0;">'
                        f'{thumb}'
                        f'<div>'
                        f'<div style="font-weight:600;color:{CARD_TEXT};font-size:0.88em;">'
                        f'{row.get("title","?")} ({year_s})</div>'
                        f'<div style="font-size:0.78em;">{badge} {imdb_s}</div>'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )
                with col_btn:
                    if st.button("Draft", key=f"draft_pick_{row.get('id', idx)}"):
                        pick = row.to_dict()
                        st.session_state.draft_user_picks.append(pick)

                        # AI picks immediately
                        remaining = available[available["id"] != pick.get("id")] if "id" in available.columns else available
                        ai_pick = _ai_pick(strategy_name, remaining, st.session_state.draft_gs_genre)
                        if ai_pick:
                            st.session_state.draft_ai_picks.append(ai_pick)

                        st.session_state.draft_round += 1

                        if st.session_state.draft_round >= rounds:
                            st.session_state.draft_active = False
                            st.session_state.draft_complete = True

                        st.rerun()

        with col_score:
            st.markdown(section_header_html("Live Scoreboard"), unsafe_allow_html=True)
            user_stats = _service_metrics(st.session_state.draft_user_picks)
            ai_stats = _service_metrics(st.session_state.draft_ai_picks)

            score_col_u, score_col_a = st.columns(2)
            with score_col_u:
                st.markdown(f'<div style="color:{CARD_ACCENT};font-weight:700;text-align:center;">Your Service</div>', unsafe_allow_html=True)
                st.metric("Avg IMDb", f"{user_stats['avg_imdb']:.2f}")
                st.metric("Diversity", f"{user_stats['diversity']:.2f}")
                st.metric("Prestige", f"{user_stats['prestige']:.0f}%")
            with score_col_a:
                st.markdown(f'<div style="color:{CARD_TEXT_MUTED};font-weight:700;text-align:center;">AI Service</div>', unsafe_allow_html=True)
                st.metric("Avg IMDb", f"{ai_stats['avg_imdb']:.2f}")
                st.metric("Diversity", f"{ai_stats['diversity']:.2f}")
                st.metric("Prestige", f"{ai_stats['prestige']:.0f}%")

            with st.expander("View AI's Picks"):
                for p in st.session_state.draft_ai_picks:
                    st.markdown(_pick_card_html(p))

    # ── Results Screen ────────────────────────────────────────────────────────
    elif st.session_state.draft_complete:
        user_picks = st.session_state.draft_user_picks
        ai_picks = st.session_state.draft_ai_picks
        user_stats = _service_metrics(user_picks)
        ai_stats = _service_metrics(ai_picks)

        # ── Banner ────────────────────────────────────────────────────────────
        metrics_to_compare = ["avg_imdb", "pct_above_7", "prestige", "diversity", "freshness"]
        user_wins = sum(1 for m in metrics_to_compare if user_stats[m] > ai_stats[m])
        ai_wins = sum(1 for m in metrics_to_compare if ai_stats[m] > user_stats[m])
        total_m = len(metrics_to_compare)

        if user_wins > ai_wins:
            banner_color, banner_bg, result_text = CARD_ACCENT, "rgba(255,215,0,0.1)", f"You win! {user_wins}–{ai_wins} across metrics 🎉"
        elif ai_wins > user_wins:
            banner_color, banner_bg, result_text = "#aaa", "rgba(100,100,100,0.1)", f"AI wins! {ai_wins}–{user_wins} across metrics 🤖"
        else:
            banner_color, banner_bg, result_text = "#00B4A6", "rgba(0,180,166,0.1)", f"Tie! {user_wins}–{ai_wins} 🤝"

        st.markdown(
            f'<div style="background:{banner_bg};border:2px solid {banner_color};'
            f'border-radius:12px;padding:20px;text-align:center;margin-bottom:20px;">'
            f'<div style="font-size:1.8em;font-weight:800;color:{banner_color};">{result_text}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Section 1: 6-metric table ─────────────────────────────────────────
        st.markdown(section_header_html("Head-to-Head Metrics"), unsafe_allow_html=True)
        metric_labels = {
            "total": ("Total Titles", "{:.0f}"),
            "avg_imdb": ("Avg IMDb", "{:.2f}"),
            "pct_above_7": ("Titles above 7.0", "{:.1f}%"),
            "prestige": ("Prestige Score", "{:.1f}%"),
            "diversity": ("Genre Diversity", "{:.2f}"),
            "freshness": ("Catalog Freshness (post-2015)", "{:.1f}%"),
        }
        teal = "#00B4A6"
        for mk, (mlabel, mfmt) in metric_labels.items():
            u_val = user_stats[mk]
            a_val = ai_stats[mk]
            u_wins_this = u_val > a_val
            a_wins_this = a_val > u_val
            u_color = teal if u_wins_this else CARD_TEXT
            a_color = teal if a_wins_this else CARD_TEXT
            u_check = " ✓" if u_wins_this else ""
            a_check = " ✓" if a_wins_this else ""
            st.markdown(
                f'<div style="display:flex;align-items:center;padding:6px 0;border-bottom:1px solid {CARD_BORDER};">'
                f'<span style="flex:1;text-align:right;color:{u_color};font-weight:{"700" if u_wins_this else "400"};">'
                f'{mfmt.format(u_val)}{u_check}</span>'
                f'<span style="flex:1.5;text-align:center;color:{CARD_TEXT_MUTED};font-size:0.85em;padding:0 12px;">{mlabel}</span>'
                f'<span style="flex:1;text-align:left;color:{a_color};font-weight:{"700" if a_wins_this else "400"};">'
                f'{mfmt.format(a_val)}{a_check}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Section 2: Radar chart ────────────────────────────────────────────
        st.markdown(section_header_html("Radar Comparison"), unsafe_allow_html=True)
        radar_metrics = ["avg_imdb", "pct_above_7", "prestige", "diversity", "freshness"]
        radar_labels = ["Avg IMDb", "% above 7.0", "Prestige", "Genre Diversity", "Freshness"]

        # Normalize 0–100
        def _norm(val, metric):
            ranges = {
                "avg_imdb": (1, 10),
                "pct_above_7": (0, 100),
                "prestige": (0, 100),
                "diversity": (0, 5),
                "freshness": (0, 100),
            }
            lo, hi = ranges.get(metric, (0, 100))
            return max(0, min(100, (val - lo) / (hi - lo) * 100))

        u_radar = [_norm(user_stats[m], m) for m in radar_metrics]
        a_radar = [_norm(ai_stats[m], m) for m in radar_metrics]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=u_radar + [u_radar[0]],
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            name="Your Service",
            line_color=CARD_ACCENT,
            fillcolor=f"rgba(255,215,0,0.2)",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=a_radar + [a_radar[0]],
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            name="AI Service",
            line_color=teal,
            fillcolor=f"rgba(0,180,166,0.2)",
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            template=PLOTLY_TEMPLATE,
            height=350,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # ── Section 3: Top 5 highlights ───────────────────────────────────────
        st.markdown(section_header_html("Highlights"), unsafe_allow_html=True)
        col_uh, col_ah = st.columns(2)

        def _top5_cards(picks, col, label):
            with col:
                st.markdown(f'<div style="color:{CARD_TEXT};font-weight:700;margin-bottom:8px;">{label}</div>', unsafe_allow_html=True)
                if not picks:
                    st.caption("No picks yet.")
                    return
                df_p = pd.DataFrame(picks)
                df_p["imdb_f"] = pd.to_numeric(df_p.get("imdb_score", pd.Series(0, index=df_p.index)), errors="coerce").fillna(0)
                top5 = df_p.nlargest(5, "imdb_f")
                for _, p in top5.iterrows():
                    poster_url = p.get("poster_url", "")
                    plat = str(p.get("platform", ""))
                    plat_color = PLATFORMS.get(plat, {}).get("color", "#555")
                    initial = str(p.get("title", "?"))[0].upper()
                    year_s = str(int(p["release_year"])) if pd.notna(p.get("release_year")) else ""
                    imdb_s = f"IMDb {p['imdb_f']:.1f}" if p["imdb_f"] > 0 else ""
                    badge = platform_badges_html(plat) if plat else ""
                    if pd.notna(poster_url) and str(poster_url).startswith("http"):
                        poster_h = f'<img src="{poster_url}" style="width:60px;height:88px;object-fit:cover;border-radius:4px;">'
                    else:
                        poster_h = (
                            f'<div style="width:60px;height:88px;background:{plat_color};border-radius:4px;'
                            f'display:flex;align-items:center;justify-content:center;'
                            f'font-size:1.4em;font-weight:700;color:#fff;">{initial}</div>'
                        )
                    st.markdown(
                        f'<div style="display:flex;gap:8px;align-items:flex-start;'
                        f'background:{CARD_BG};border:1px solid {CARD_BORDER};'
                        f'border-radius:6px;padding:8px;margin-bottom:6px;">'
                        f'{poster_h}'
                        f'<div><div style="font-weight:600;color:{CARD_TEXT};font-size:0.88em;">'
                        f'{p.get("title","?")} ({year_s})</div>'
                        f'<div style="margin-top:2px;">{badge}</div>'
                        f'<div style="color:{CARD_ACCENT};font-size:0.85em;">{imdb_s}</div>'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )

        _top5_cards(user_picks, col_uh, "Your Highlights")
        _top5_cards(ai_picks, col_ah, "AI's Highlights")

        # ── Action buttons ────────────────────────────────────────────────────
        col_again, col_export = st.columns(2)
        with col_again:
            if st.button("🔄 Draft Again", use_container_width=True):
                for k in ["draft_active", "draft_complete", "draft_round",
                          "draft_user_picks", "draft_ai_picks", "draft_available_pool",
                          "draft_settings", "draft_gs_genre"]:
                    default = False if k in ("draft_active", "draft_complete") else (
                        0 if k == "draft_round" else ([] if k in ("draft_user_picks", "draft_ai_picks") else
                        ({} if k == "draft_settings" else None))
                    )
                    st.session_state[k] = default
                st.rerun()

        with col_export:
            if user_picks:
                export_cols = ["title", "release_year", "platform", "imdb_score", "genres"]
                df_export = pd.DataFrame(user_picks)
                avail_cols = [c for c in export_cols if c in df_export.columns]
                csv_str = df_export[avail_cols].to_csv(index=False)
                st.download_button(
                    "📥 Export My Service",
                    csv_str,
                    "my_drafted_service.csv",
                    "text/csv",
                    use_container_width=True,
                )

# =============================================================================
# FOOTER
# =============================================================================
st.divider()
st.markdown(
    '<div style="color:#555;font-size:0.8em;text-align:center;padding:8px 0 16px;">'
    'Hypothetical merger for academic analysis. '
    'Data is a snapshot (mid-2023). '
    'All insights are illustrative, not prescriptive. '
    'As of Feb 26, 2026, Netflix withdrew from this acquisition.'
    '</div>',
    unsafe_allow_html=True,
)
