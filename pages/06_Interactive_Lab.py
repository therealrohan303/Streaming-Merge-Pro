"""Page 6: Interactive Lab — Greenlight Studio, Franchise Explorer, Draft Room."""

import hashlib
import math
from datetime import date

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
    load_all_platforms_credits,
    load_enriched_titles,
    load_genome_vectors,
    load_greenlight_model,
    load_imdb_principals,
    load_person_stats,
    load_prestige_index,
    load_tfidf_vectorizer,
)
from src.analysis.discovery import vibe_search
from src.analysis.scoring import bayesian_imdb, compute_quality_score, format_votes
from src.analysis.greenlight_model import GreenlightStackedModel  # noqa: F401 — registers class for joblib
from src.analysis.lab import (
    GREENLIGHT_TONE_LABELS,
    _build_person_award_map,
    _build_person_keyword_map,
    _build_person_top_title_map,
    _compute_box_office_lookup,
    greenlight_box_office,
    greenlight_platform_fit,
    greenlight_similar_titles,
    greenlight_talent_picks,
    predict_title,
)
from src.ui.badges import (
    page_header_html,
    platform_badges_html,
    section_header_html,
)
from src.ui.formatting import genre_display, genre_list_display
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
    ("fe_title_selected", None),
    ("fe_page_size", 15),
    ("fe_filter_sig", None),
    ("fe_show_all_gaps", False),
    # Draft Room
    ("draft_active", False),
    ("draft_round", 0),
    ("draft_user_picks", []),
    ("draft_ai_picks", []),
    ("draft_available_pool", None),
    ("draft_settings", {}),
    ("draft_complete", False),
    ("draft_ai_prepicked_round", -1),  # round index in which AI has pre-picked (when AI goes first)
    ("draft_win_celebrated", False),
    ("draft_gs_genre", None),  # Genre Specialist dominant genre
    # CinemaGuess
    ("cg", None),
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


@st.cache_resource
def _get_box_office_lookup(enriched_df):
    return _compute_box_office_lookup(enriched_df)


@st.cache_resource
def _get_person_top_title_map():
    return _build_person_top_title_map(
        load_imdb_principals(), load_enriched_titles(), load_all_platforms_credits(),
    )


@st.cache_resource
def _get_person_keyword_map():
    return _build_person_keyword_map(
        load_imdb_principals(), load_enriched_titles(), load_all_platforms_credits(),
    )


@st.cache_resource
def _get_person_award_map():
    return _build_person_award_map(
        load_imdb_principals(), load_enriched_titles(), load_all_platforms_credits(),
    )


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
        f'border-radius:8px;font-size:0.65em;margin-right:2px;">{genre_display(g)}</span>'
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

# ─── CinemaGuess constants & loader ─────────────────────────────────────────
_GAME_CATALOG_PATH = PRECOMPUTED_DIR / "game_catalog.parquet"
_CG_MAX_GUESSES = 6
_CG_BLUR_SCHEDULE = [18, 14, 10, 7, 4, 2, 0]

_CG_CLUE_DEFS = [
    (
        "Genre(s)",
        lambda r: (
            ", ".join(g.capitalize() for g in list(r["genres"])[:3])
            if r.get("genres") is not None and len(r["genres"]) > 0
            else "Unknown"
        ),
    ),
    (
        "Decade",
        lambda r: (
            f"{int(r['release_year'] // 10) * 10}s"
            if pd.notna(r.get("release_year"))
            else "Unknown"
        ),
    ),
    ("Type", lambda r: r.get("type", "Unknown")),
    (
        "Runtime / Seasons",
        lambda r: (
            f"{int(r['runtime'])} min"
            if r.get("type") == "Movie" and pd.notna(r.get("runtime"))
            else f"{int(r['seasons'])} season(s)"
            if pd.notna(r.get("seasons"))
            else "Unknown"
        ),
    ),
    (
        "First letter",
        lambda r: (
            "The " + r["title"][4].upper()
            if r.get("title", "").startswith("The ")
            else r.get("title", "?")[0].upper()
        ),
    ),
]


@st.cache_data
def _load_game_catalog():
    if not _GAME_CATALOG_PATH.exists():
        return None
    df = pd.read_parquet(_GAME_CATALOG_PATH)
    has_backdrop = df["backdrop_url"].notna()
    return df[has_backdrop].reset_index(drop=True)


def _cg_filtered_pool(df: pd.DataFrame, answer: dict, clues_revealed: int) -> pd.DataFrame:
    """Return subset of df that matches all revealed clues for the answer."""
    mask = pd.Series(True, index=df.index)

    # Clue 0 — genres (revealed after guess 1)
    if clues_revealed >= 1:
        _raw = answer.get("genres")
        ans_genres = set(_raw) if _raw is not None and len(_raw) > 0 else set()
        if ans_genres:
            def _has_genre_overlap(row_genres):
                if row_genres is None:
                    return False
                try:
                    return bool(ans_genres & set(row_genres))
                except TypeError:
                    return False
            mask &= df["genres"].apply(_has_genre_overlap)

    # Clue 1 — decade (revealed after guess 2)
    if clues_revealed >= 2 and pd.notna(answer.get("release_year")):
        decade = int(answer["release_year"] // 10) * 10
        mask &= (
            df["release_year"]
            .apply(lambda y: int(y // 10) * 10 if pd.notna(y) else None)
            == decade
        )

    # Clue 2 — type (revealed after guess 3)
    if clues_revealed >= 3 and answer.get("type"):
        mask &= df["type"] == answer["type"]

    # Clue 3 — runtime / seasons (revealed after guess 4)
    if clues_revealed >= 4:
        if answer.get("type") == "Movie" and pd.notna(answer.get("runtime")):
            rt = int(answer["runtime"])
            mask &= df["runtime"].apply(
                lambda v: abs(int(v) - rt) <= 30 if pd.notna(v) else False
            )
        elif pd.notna(answer.get("seasons")):
            mask &= df["seasons"].apply(
                lambda v: abs(int(v) - int(answer["seasons"])) <= 1 if pd.notna(v) else False
            )

    # Clue 4 — first letter (revealed after guess 5)
    if clues_revealed >= 5 and answer.get("title"):
        t = answer["title"]
        first_letter = (t[4].upper() if t.startswith("The ") else t[0].upper())
        def _matches_letter(title):
            if not isinstance(title, str):
                return False
            t2 = title.strip()
            ch = t2[4].upper() if t2.startswith("The ") else t2[0].upper()
            return ch == first_letter
        mask &= df["title"].apply(_matches_letter)

    # Always keep the answer itself in the pool
    mask |= (df["title"].str.strip().str.lower() == answer.get("title", "").strip().lower())

    return df[mask]


# ─── Page header ─────────────────────────────────────────────────────────────
st.markdown(
    page_header_html(
        "Interactive Lab",
        "Pitch a concept, explore franchises, or compete in the streaming draft.",
    ),
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs(["Greenlight Studio", "Franchise Explorer", "Draft Room", "CinemaGuess"])

# =============================================================================
# TAB 1: GREENLIGHT STUDIO
# =============================================================================
with tab1:
    st.markdown(
        section_header_html(
            "Greenlight Studio",
            "Pitch a concept — get a grounded IMDb forecast, box-office projection, "
            "comparable titles, platform fit, and talent shortlist.",
        ),
        unsafe_allow_html=True,
    )

    # ── Input form ────────────────────────────────────────────────────────────
    _gs_tones = list(GREENLIGHT_TONE_LABELS.keys())
    _tone_label_to_key = {v: k for k, v in GREENLIGHT_TONE_LABELS.items()}

    desc = st.text_area(
        "Concept Description",
        placeholder=(
            "Describe the concept in 2–3 sentences. The more specific — setting, "
            "tone, stakes, main character — the more the forecast responds to it.\n"
            "e.g. 'A disgraced Tokyo detective is dragged back to a mountain village "
            "to investigate a chain of ritualistic disappearances tied to local folklore.'"
        ),
        height=110,
        key="gs_desc",
    )

    _type_col, _audience_col, _origin_col = st.columns([1, 1.2, 1.2])
    with _type_col:
        content_type = st.radio(
            "Content Type", ["Movie", "Show"], horizontal=True, key="gs_type",
        )
    with _audience_col:
        audience = st.radio(
            "Audience",
            ["Family", "Teen", "Mature"],
            horizontal=True,
            key="gs_audience",
            help="Family maps to G/PG/TV-G · Teen to PG-13/TV-14 · Mature to R/TV-MA.",
        )
    with _origin_col:
        origin = st.radio(
            "Production Origin",
            ["US-led", "International co-pro"],
            horizontal=True,
            key="gs_origin",
        )

    _genre_col, _tone_col = st.columns(2)
    with _genre_col:
        _genre_display_to_key = {genre_display(g): g for g in ALL_GENRES_LIST}
        genre_labels = st.multiselect(
            "Genres (up to 3)",
            list(_genre_display_to_key.keys()),
            max_selections=3,
            key="gs_genre_labels",
        )
        genres = [_genre_display_to_key[g] for g in genre_labels]
    with _tone_col:
        tone_labels = st.multiselect(
            "Tone (up to 3)",
            [GREENLIGHT_TONE_LABELS[k] for k in _gs_tones],
            max_selections=3,
            key="gs_tone_labels",
            help="Shapes the vibe used by similar-titles, box-office, and the predictor.",
        )
    tone_selection = [_tone_label_to_key[t] for t in tone_labels]

    if content_type == "Movie":
        _rt_col, _budget_col, _ip_col = st.columns([1, 1.5, 1.2])
        with _rt_col:
            runtime = st.slider(
                "Runtime (min)", 70, 210, 110, step=5, key="gs_runtime",
            )
        with _budget_col:
            budget_m = st.slider(
                "Budget (USD millions)",
                min_value=2, max_value=400, value=45, step=1,
                key="gs_budget_m",
                help="Move the slider to the concept's real budget — downstream projections respond continuously, not in fixed brackets.",
            )
        with _ip_col:
            ip_status = st.radio(
                "IP Status",
                ["Original", "Franchise entry", "Adaptation"],
                horizontal=True,
                key="gs_ip",
            )

        budget_usd = float(budget_m) * 1_000_000
        if budget_m < 20:
            budget_tier_int, tier_label = 1, "Indie"
        elif budget_m < 80:
            budget_tier_int, tier_label = 2, "Mid-budget"
        elif budget_m < 200:
            budget_tier_int, tier_label = 3, "Studio"
        else:
            budget_tier_int, tier_label = 4, "Blockbuster"
        budget_label = f"${budget_m}M · {tier_label}"
        with _budget_col:
            st.markdown(
                f'<div style="text-align:right;color:{CARD_TEXT_MUTED};font-size:0.78em;'
                f'margin-top:-8px;">${budget_m}M · <b style="color:{CARD_TEXT};">{tier_label}</b></div>',
                unsafe_allow_html=True,
            )
        num_seasons_val = None
    else:
        _rt_col, _seasons_col = st.columns(2)
        with _rt_col:
            runtime = st.slider(
                "Episode Runtime (min)", 20, 90, 45, step=5, key="gs_ep_runtime",
            )
        with _seasons_col:
            num_seasons_val = st.slider(
                "Planned Seasons", 1, 8, 1, key="gs_seasons",
            )
        budget_tier_int, budget_usd, ip_status, budget_label = 0, None, None, None

    _cert_map = {"Family": 1, "Teen": 2, "Mature": 3}
    _origin_map = {"US-led": 1, "International co-pro": 2}
    _ip_map = {"Original": 0, "Franchise entry": 1, "Adaptation": 2}

    features_dict: dict = {
        "genres": genres,
        "runtime": runtime,
        "release_year": 2026,
        "decade": 2020,
        "production_country_tier": _origin_map[origin],
        "age_cert_tier": _cert_map[audience],
    }
    if content_type == "Movie":
        features_dict.update({
            "budget_tier":    budget_tier_int,
            "budget_usd":     budget_usd,
            "franchise_type": _ip_map[ip_status],
        })
    else:
        features_dict["num_seasons"] = num_seasons_val

    consult_btn = st.button("Consult the Studio", type="primary", use_container_width=True)

    # ── Output ────────────────────────────────────────────────────────────────
    if consult_btn:
        if not genres:
            st.warning("Select at least one genre to run the forecast.")
        else:
            with st.spinner("Analyzing your concept…"):
                enriched = _get_enriched()
                model = load_greenlight_model("movie" if content_type == "Movie" else "show")
                if model is None:
                    st.error("Greenlight model not found. Run `python scripts/11_train_greenlight_models.py`.")
                else:
                    result = predict_title(
                        model, features_dict, ALL_GENRES_LIST,
                        description=desc, tone_selection=tone_selection,
                    )
                    st.session_state.gs_result = {
                        "pred": result,
                        "features": features_dict,
                        "desc": desc,
                        "tone_selection": tone_selection,
                        "content_type": content_type,
                        "budget_label": budget_label,
                    }

    if st.session_state.gs_result:
        res          = st.session_state.gs_result
        features     = res["features"]
        desc_text    = (res["desc"] or "").strip()
        tone_sel     = res["tone_selection"]
        content_type = res["content_type"]
        budget_label = res["budget_label"]
        pred_obj     = res["pred"]
        pred         = pred_obj["prediction"]
        peer         = pred_obj["peer"] or {}
        genres_sel   = features.get("genres") or []
        enriched     = _get_enriched()

        _TIER_STYLE = {
            "top_10":           {"label": "Top 10% of recent {g}", "color": "#FFD700"},
            "upper_quartile":   {"label": "Upper quartile of {g}", "color": "#2ecc71"},
            "above_median":     {"label": "Above {g} median",       "color": "#5dcb8f"},
            "below_median":     {"label": "Below {g} median",       "color": "#f39c12"},
            "bottom_quartile":  {"label": "Bottom quartile — high risk", "color": "#e74c3c"},
        }
        tier_info = _TIER_STYLE.get(pred_obj["tier"], _TIER_STYLE["above_median"])
        band_color = tier_info["color"]
        _peer_genre_display = genre_display(pred_obj["primary_genre"]) or "peers"
        band_label = tier_info["label"].format(g=_peer_genre_display)

        cv_rmse_val = pred_obj.get("cv_rmse") or 0.45
        err = float(cv_rmse_val)
        ci_low  = max(1.0, pred - err)
        ci_high = min(10.0, pred + err)

        # ── Row 1: Predicted IMDb Score | Box Office Projection ──────────────
        col_r1a, col_r1b = st.columns(2)

        with col_r1a:
            st.markdown(
                section_header_html(
                    "Predicted IMDb Score",
                    "Peer-relative grade against real catalog titles in the same genre.",
                ),
                unsafe_allow_html=True,
            )
            p50 = peer.get("p50", 6.5)
            p75 = peer.get("p75", 7.2)
            p90 = peer.get("p90", 7.7)
            gauge_pct  = max(0.0, min(100.0, (pred - 1) / 9 * 100))
            low_pct    = max(0.0, min(100.0, (ci_low  - 1) / 9 * 100))
            high_pct   = max(0.0, min(100.0, (ci_high - 1) / 9 * 100))
            median_pct = max(0.0, min(100.0, (p50 - 1) / 9 * 100))
            peer_n     = peer.get("n", 0)

            p25 = peer.get("p25", 5.5)
            p25_pct = max(0.0, min(100.0, (p25 - 1) / 9 * 100))
            p90_pct = max(0.0, min(100.0, (p90 - 1) / 9 * 100))
            st.markdown(
                f"""
                <div style="background:{CARD_BG};border:1px solid {CARD_BORDER};
                            border-top:3px solid {band_color};border-radius:8px;padding:18px;">
                  <div style="display:flex;align-items:baseline;gap:10px;">
                    <span style="font-size:2.8em;font-weight:800;color:{band_color};line-height:1;">{pred:.1f}</span>
                    <span style="color:{CARD_TEXT_MUTED};font-size:0.85em;">/ 10 IMDb</span>
                  </div>
                  <div style="color:{CARD_TEXT_MUTED};font-size:0.78em;margin-top:2px;margin-bottom:10px;">
                    Could land anywhere between <b style="color:{CARD_TEXT};">{ci_low:.1f}</b> and
                    <b style="color:{CARD_TEXT};">{ci_high:.1f}</b> once audiences weigh in.
                  </div>
                  <div style="background:{band_color}22;color:{band_color};padding:4px 11px;
                              border-radius:10px;font-size:0.82em;border:1px solid {band_color};
                              font-weight:600;display:inline-block;margin-bottom:14px;">
                    {band_label}
                  </div>
                  <div style="position:relative;height:10px;border-radius:5px;
                              background:linear-gradient(90deg,#5a2020,#5a4020 35%,#305a30 65%,#5a5020);
                              overflow:visible;margin-top:4px;">
                    <div style="position:absolute;top:0;left:{low_pct:.1f}%;height:10px;
                                width:{max(0.5, high_pct-low_pct):.1f}%;
                                background:{band_color}66;"></div>
                    <div style="position:absolute;top:-2px;left:calc({median_pct:.1f}% - 1px);
                                width:2px;height:14px;background:#bbb;"></div>
                    <div style="position:absolute;top:-4px;left:calc({gauge_pct:.1f}% - 7px);
                                width:14px;height:18px;background:#fff;border-radius:3px;
                                box-shadow:0 0 6px rgba(0,0,0,0.7);"></div>
                  </div>
                  <div style="position:relative;height:28px;margin-top:2px;
                              font-size:0.66em;color:{CARD_TEXT_MUTED};">
                    <span style="position:absolute;left:{p25_pct:.1f}%;transform:translateX(-50%);text-align:center;">
                      <span style="display:block;font-weight:600;color:#999;">Weakest peers</span>
                      <span style="display:block;">{p25:.1f}</span>
                    </span>
                    <span style="position:absolute;left:{median_pct:.1f}%;transform:translateX(-50%);text-align:center;">
                      <span style="display:block;font-weight:600;color:#ccc;">Typical</span>
                      <span style="display:block;">{p50:.1f}</span>
                    </span>
                    <span style="position:absolute;left:{p90_pct:.1f}%;transform:translateX(-50%);text-align:center;">
                      <span style="display:block;font-weight:600;color:#999;">Hit tier</span>
                      <span style="display:block;">{p90:.1f}</span>
                    </span>
                  </div>
                  <div style="color:{CARD_TEXT_MUTED};font-size:0.76em;margin-top:14px;
                              border-top:1px solid {CARD_BORDER};padding-top:8px;">
                    Benchmarked against <b style="color:{CARD_TEXT};">{peer_n:,}</b>
                    {_peer_genre_display} titles that cleared 500 IMDb votes.
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
                        "ROI grounded in comparable catalog releases.",
                    ),
                    unsafe_allow_html=True,
                )
                bo_lookup = _get_box_office_lookup(enriched)
                bo = greenlight_box_office(features, pred, content_type, bo_lookup, peer) if bo_lookup else None

                def _fmt_money(v):
                    if v >= 1e9:
                        return f"${v/1e9:.2f}B"
                    if v >= 1e6:
                        return f"${v/1e6:.0f}M"
                    return f"${v:,.0f}"

                if bo is None:
                    st.markdown(
                        f'<div style="background:{CARD_BG};border:1px solid {CARD_BORDER};'
                        f'border-radius:8px;padding:16px;color:{CARD_TEXT_MUTED};">'
                        f'Too few comparable movies in this (genre, budget, rating) cell to produce '
                        f'a grounded projection.</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    bo_genre_display = genre_display(bo.get("primary_genre")) or "catalog"
                    comp_n = int(bo.get("comp_count", 0))
                    comp_lo = bo.get("comp_budget_lo_m")
                    comp_hi = bo.get("comp_budget_hi_m")
                    relaxation = bo.get("relaxation", "tight")
                    ft = int(bo.get("franchise_type", 0) or 0)
                    ft_label = {1: "Franchise entry", 2: "Adaptation"}.get(ft, "Original")

                    rows_html = [
                        f"""<div style='display:flex;justify-content:space-between;padding:3px 0;'>
                              <span style='color:{CARD_TEXT_MUTED};'>Budget</span>
                              <span style='color:{CARD_TEXT};font-weight:600;'>{_fmt_money(bo['budget_used'])}</span>
                            </div>""",
                        f"""<div style='display:flex;justify-content:space-between;padding:3px 0;'>
                              <span style='color:{CARD_TEXT_MUTED};'>Catalog ROI ({bo_genre_display})</span>
                              <span style='color:{CARD_TEXT};font-weight:600;'>{bo['roi_med']:.2f}×</span>
                            </div>""",
                    ]
                    q_mult = float(bo.get("quality_mult", 1.0) or 1.0)
                    if abs(q_mult - 1.0) >= 0.01:
                        q_pct = (q_mult - 1.0) * 100
                        q_sign = "+" if q_pct >= 0 else ""
                        q_color = "#6abf6a" if q_pct >= 0 else "#d47070"
                        rows_html.append(
                            f"""<div style='display:flex;justify-content:space-between;padding:3px 0;'>
                                  <span style='color:{CARD_TEXT_MUTED};'>Quality adjustment</span>
                                  <span style='color:{q_color};font-weight:600;'>{q_sign}{q_pct:.0f}%</span>
                                </div>"""
                        )
                    f_mult = float(bo.get("franchise_mult", 1.0) or 1.0)
                    if ft != 0 and abs(f_mult - 1.0) >= 0.01:
                        f_pct = (f_mult - 1.0) * 100
                        f_sign = "+" if f_pct >= 0 else ""
                        f_color = "#6abf6a" if f_pct >= 0 else "#d47070"
                        f_row_label = "Franchise entry lift" if ft == 1 else "Adaptation lift"
                        rows_html.append(
                            f"""<div style='display:flex;justify-content:space-between;padding:3px 0;'>
                                  <span style='color:{CARD_TEXT_MUTED};'>{f_row_label}</span>
                                  <span style='color:{f_color};font-weight:600;'>{f_sign}{f_pct:.0f}%</span>
                                </div>"""
                        )

                    if comp_lo is not None and comp_hi is not None and comp_n > 0:
                        footnote = (
                            f"Based on <b style='color:{CARD_TEXT};'>{comp_n}</b> "
                            f"{bo_genre_display} releases between "
                            f"<b style='color:{CARD_TEXT};'>${comp_lo:.0f}M–${comp_hi:.0f}M</b>."
                        )
                    else:
                        footnote = f"Based on {comp_n} comparable releases."

                    caveat_map = {
                        "genre_only": f"Cert-specific cell was small — weighted across all {bo_genre_display} releases.",
                        "cert_only": f"{bo_genre_display} cell too sparse — used weighted average of comparable-rated releases.",
                        "global": "Cell too sparse — used global catalog comps.",
                    }
                    caveat_html = ""
                    if relaxation in caveat_map:
                        caveat_html = (
                            f"<div style='color:{CARD_TEXT_MUTED};font-size:0.72em;"
                            f"font-style:italic;margin-top:6px;'>{caveat_map[relaxation]}</div>"
                        )

                    ip_chip = (
                        f"<div style='display:inline-block;background:{CARD_ACCENT}22;color:{CARD_ACCENT};"
                        f"padding:2px 9px;border-radius:10px;font-size:0.72em;font-weight:600;"
                        f"border:1px solid {CARD_ACCENT}55;margin-top:4px;'>IP · {ft_label}</div>"
                    )

                    st.markdown(
                        f"""
                        <div style="background:{CARD_BG};border:1px solid {CARD_BORDER};
                                    border-top:3px solid {CARD_ACCENT};border-radius:8px;padding:16px;">
                          <div style="color:{CARD_TEXT_MUTED};font-size:0.72em;letter-spacing:0.5px;
                                      text-transform:uppercase;font-weight:600;">
                            Projected worldwide gross
                          </div>
                          <div style="font-size:2.6em;font-weight:800;color:{CARD_ACCENT};line-height:1.05;
                                      margin-top:2px;">
                            {_fmt_money(bo['projected'])}
                          </div>
                          <div style="color:{CARD_TEXT_MUTED};font-size:0.82em;margin-top:2px;">
                            Most comparable releases land between
                            <b style="color:{CARD_TEXT};">{_fmt_money(bo['low'])}</b> and
                            <b style="color:{CARD_TEXT};">{_fmt_money(bo['high'])}</b>.
                          </div>
                          {ip_chip}
                          <div style="border-top:1px solid {CARD_BORDER};margin-top:12px;padding-top:10px;">
                            <div style="color:{CARD_TEXT_MUTED};font-size:0.72em;letter-spacing:0.5px;
                                        text-transform:uppercase;font-weight:600;margin-bottom:6px;">
                              What drives this projection
                            </div>
                            <div style="font-size:0.84em;">
                              {''.join(rows_html)}
                            </div>
                          </div>
                          <div style="color:{CARD_TEXT_MUTED};font-size:0.74em;margin-top:10px;
                                      border-top:1px solid {CARD_BORDER};padding-top:8px;">
                            {footnote}
                          </div>
                          {caveat_html}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    section_header_html(
                        "Series Economics",
                        "Theatrical projection does not apply to episodic content.",
                    ),
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div style="background:{CARD_BG};border:1px solid {CARD_BORDER};'
                    f'border-radius:8px;padding:16px;color:{CARD_TEXT_MUTED};font-size:0.88em;">'
                    f'Streaming shows monetize via subscription retention, not box office. '
                    f'See Similar Titles and Platform Fit for benchmark performance.'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # ── Row 2: Most Similar Titles ───────────────────────────────────────
        st.markdown(
            section_header_html(
                "Most Similar Titles",
                "Ranked by description, genre, quality, and tone alignment.",
            ),
            unsafe_allow_html=True,
        )
        if not desc_text:
            st.markdown(
                f'<div style="background:{CARD_BG};border:1px solid {CARD_BORDER};'
                f'border-radius:8px;padding:18px;text-align:center;color:{CARD_TEXT_MUTED};'
                f'font-size:0.9em;">Add a concept description to see comparable titles.</div>',
                unsafe_allow_html=True,
            )
        else:
            try:
                vec_gs, mat_gs, id_list_gs, enriched_full = _load_catalog_tfidf_matrix()
                sim_df = greenlight_similar_titles(
                    desc_text, features, tone_sel, enriched_full,
                    vec_gs, mat_gs, id_list_gs, content_type, top_k=6,
                )
                if sim_df is None or sim_df.empty:
                    st.info("No comparable titles matched the filters. Try broadening genre or runtime.")
                else:
                    grid = st.columns(min(6, len(sim_df)))
                    for i, (_, srow) in enumerate(sim_df.iterrows()):
                        # Explainer pill — pick the strongest contribution
                        tf = float(srow.get("_tfidf", 0))
                        gf = float(srow.get("_genre_frac", 0))
                        tn = float(srow.get("_tone", 0))
                        fb = float(srow.get("_fboost", 0))
                        if fb >= 0.04 and tf < 0.12 and tn < 0.5:
                            why = "Franchise comp"
                        elif tf >= 0.12 and tn >= 0.5:
                            why = "Theme + tone"
                        elif tf >= 0.12:
                            why = "Theme match"
                        elif tn >= 0.5:
                            why = "Tone match"
                        elif gf >= 0.66:
                            why = "Genre match"
                        else:
                            why = "Catalog comp"
                        score_badge = min(1.0, 0.4 + float(srow.get("_score", 0.3)) * 1.1)
                        with grid[i]:
                            _gs_render_similar_card(srow.to_dict(), score_badge)
                            st.markdown(
                                f'<div style="text-align:center;margin-top:-4px;'
                                f'color:{CARD_TEXT_MUTED};font-size:0.68em;">{why}</div>',
                                unsafe_allow_html=True,
                            )
            except Exception as e:
                st.info(f"Similarity search unavailable: {e}")

        # ── Row 3: Platform Fit (full width) ─────────────────────────────────
        st.markdown(
            section_header_html(
                "Platform Fit",
                "Each platform scored on prestige, quality, type mix, and budget DNA for your selected genres.",
            ),
            unsafe_allow_html=True,
        )
        try:
            prestige = load_prestige_index()
        except Exception as e:
            prestige = pd.DataFrame()
            st.info(f"Prestige index unavailable: {e}")

        if prestige.empty:
            st.info("Platform prestige index missing — run `scripts/11_precompute_strategic.py`.")
        elif not genres_sel:
            st.info("Select at least one genre to compute Platform Fit.")
        else:
            fit_rows = greenlight_platform_fit(pred, features, content_type, prestige, enriched)
            if not fit_rows:
                st.info("Could not compute platform fit for this configuration.")
            else:
                plat_names  = [PLATFORMS.get(r["platform"], {}).get("name", r["platform"]) for r in fit_rows]
                plat_colors = [PLATFORMS.get(r["platform"], {}).get("color", "#555") for r in fit_rows]
                plat_vals   = [r["fit"] for r in fit_rows]

                line_widths = [3 if i == 0 else 0 for i in range(len(fit_rows))]
                line_colors = ["#FFFFFF" if i == 0 else plat_colors[i] for i in range(len(fit_rows))]
                bar_text = [
                    f"{v}  ·  BEST FIT" if i == 0 else f"{v}"
                    for i, v in enumerate(plat_vals)
                ]

                fig_fit = go.Figure(go.Bar(
                    x=plat_vals,
                    y=plat_names,
                    orientation="h",
                    marker=dict(
                        color=plat_colors,
                        line=dict(color=line_colors, width=line_widths),
                    ),
                    text=bar_text,
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate="<b>%{y}</b><br>Fit score: %{x}/100<extra></extra>",
                ))
                fig_fit.update_layout(
                    template=PLOTLY_TEMPLATE,
                    height=260,
                    margin=dict(l=0, r=60, t=6, b=6),
                    xaxis=dict(title="Fit Score (0–100)", range=[0, 115]),
                    yaxis=dict(title="", autorange="reversed"),
                )
                st.plotly_chart(fig_fit, use_container_width=True)

                top = fit_rows[0]
                top_name = PLATFORMS.get(top["platform"], {}).get("name", top["platform"])
                sel_genres_disp = genre_list_display(top.get("selected_genres") or genres_sel)
                p_mult = top.get("prestige_multiple")
                p_mean = float(top.get("platform_mean_imdb") or 0)
                ind_mean = float(top.get("industry_mean_imdb") or 0)

                if p_mult and p_mult >= 1.15:
                    rationale_lead = (
                        f'<b style="color:{CARD_ACCENT};">{top_name}</b> is the strongest fit for your '
                        f'<b>{sel_genres_disp}</b> concept — its catalog indexes '
                        f'<b>{p_mult:.1f}×</b> the industry prestige rate in those genres.'
                    )
                else:
                    rationale_lead = (
                        f'<b style="color:{CARD_ACCENT};">{top_name}</b> is the strongest fit for your '
                        f'<b>{sel_genres_disp}</b> concept based on its overall catalog DNA.'
                    )

                imdb_trailer = ""
                if p_mean and ind_mean and (p_mean - ind_mean) >= 0.3:
                    trailer_genre = genre_display(top.get("top_genre")) or sel_genres_disp
                    imdb_trailer = (
                        f' Its {trailer_genre} titles average IMDb <b>{p_mean:.1f}</b> — above the '
                        f'<b>{ind_mean:.1f}</b> industry norm.'
                    )

                st.markdown(
                    f'<div style="color:{CARD_TEXT};font-size:0.9em;padding:6px 2px;line-height:1.5;">'
                    f'{rationale_lead}{imdb_trailer}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Per-platform component breakdown
                with st.expander("Score breakdown", expanded=False):
                    comp_rows = []
                    for r in fit_rows:
                        comp_rows.append({
                            "Platform":        PLATFORMS.get(r["platform"], {}).get("name", r["platform"]),
                            "Fit score":       r["fit"],
                            "Prestige match":  f"{r['prestige_alignment']:.2f}",
                            "Quality fit":     f"{r['quality_alignment']:.2f}",
                            "Type mix":        f"{r['type_fit']:.2f}",
                            "Budget fit":      f"{r['budget_fit']:.2f}" if r["budget_fit"] is not None else "—",
                            "Platform avg IMDb": f"{r['platform_mean_imdb']:.2f}",
                        })
                    st.dataframe(pd.DataFrame(comp_rows), hide_index=True, use_container_width=True)

        # ── Row 4: Talent Recommendations ────────────────────────────────────
        st.markdown(
            section_header_html(
                "Talent Recommendations",
                "Directors and cast whose catalog strengths fit your genres and quality target.",
            ),
            unsafe_allow_html=True,
        )
        person_stats = load_person_stats()
        if person_stats.empty:
            st.info("Talent data unavailable — run `scripts/12_precompute_network.py`.")
        elif not genres_sel:
            st.info("Select genres to see talent recommendations.")
        else:
            top_title_map = _get_person_top_title_map()
            keyword_map = _get_person_keyword_map()
            award_map = _get_person_award_map()

            directors = greenlight_talent_picks(
                person_stats, genres_sel, role="director",
                min_title_count=5, min_avg_imdb=6.8, top_k=5,
                tone_selection=tone_sel,
                top_title_map=top_title_map,
                keyword_map=keyword_map,
                award_map=award_map,
            )
            cast = greenlight_talent_picks(
                person_stats, genres_sel, role="actor",
                min_title_count=6, min_avg_imdb=6.5, top_k=5,
                tone_selection=tone_sel,
                top_title_map=top_title_map,
                keyword_map=keyword_map,
                award_map=award_map,
            )

            def _talent_pill(text: str, color: str, *, filled: bool = True) -> str:
                """One-line utility for a standardized rounded pill."""
                bg = f"{color}22" if filled else "transparent"
                return (
                    f'<span style="display:inline-flex;align-items:center;gap:4px;'
                    f'background:{bg};color:{color};padding:3px 9px;border-radius:999px;'
                    f'font-size:0.72em;font-weight:600;border:1px solid {color}55;'
                    f'margin:0 4px 4px 0;white-space:nowrap;">{text}</span>'
                )

            def _render_person_card_v2(p: dict) -> None:
                name    = p["name"]
                avg_i   = float(p.get("avg_imdb") or 0)
                titlen  = int(p.get("title_count") or 0)
                awards  = int(p.get("award_titles") or 0)
                top_g   = p.get("top_genre")
                prom    = float(p.get("prominence") or 0)

                imdb_c = "#2ecc71" if avg_i >= 7.5 else (CARD_ACCENT if avg_i >= 7.0 else "#b8b8b8")
                prom_pct = int(round(prom * 100))
                prom_label = (
                    f"Top {100 - prom_pct}% in field" if prom_pct >= 70
                    else ("Established voice" if prom_pct >= 40 else "Emerging")
                )

                # Metric pill row (always three pills, same order, same widths)
                pills = [
                    _talent_pill(f"IMDb avg {avg_i:.1f}", imdb_c),
                    _talent_pill(
                        f"🏆 {awards} award title" + ("s" if awards != 1 else ""),
                        "#FFD700" if awards > 0 else "#6c6c6c",
                    ),
                    _talent_pill(prom_label, "#8BB6FF"),
                ]
                pill_row = (
                    f'<div style="display:flex;flex-wrap:wrap;margin-top:8px;">'
                    f'{"".join(pills)}</div>'
                )

                genre_disp = genre_display(top_g) if top_g else "Multi-genre"
                subtitle = f'{genre_disp} veteran · {titlen} catalog titles'

                # "Known for" block — always rendered so every card has the same height
                top_t = p.get("top_title") or {}
                if top_t.get("title"):
                    year_part = f" ({int(top_t['year'])})" if top_t.get("year") else ""
                    imdb_part = f"IMDb {float(top_t['imdb']):.1f}" if top_t.get("imdb") else ""
                    votes_part = (
                        f" · {format_votes(top_t['votes'])} votes"
                        if top_t.get("votes") else ""
                    )
                    known_body = (
                        f'<div style="color:{CARD_TEXT};font-size:0.84em;font-weight:600;'
                        f'line-height:1.35;margin-top:3px;">'
                        f'<i>{top_t["title"]}</i>{year_part}'
                        f'</div>'
                        f'<div style="color:{CARD_TEXT_MUTED};font-size:0.74em;margin-top:2px;">'
                        f'{imdb_part}{votes_part}'
                        f'</div>'
                    )
                else:
                    known_body = (
                        f'<div style="color:{CARD_TEXT_MUTED};font-size:0.8em;'
                        f'font-style:italic;margin-top:3px;">Deep catalog, no single standout</div>'
                    )

                known_block = (
                    f'<div style="margin-top:10px;padding-top:8px;'
                    f'border-top:1px dashed {CARD_BORDER};">'
                    f'<div style="color:{CARD_TEXT_MUTED};font-size:0.66em;font-weight:700;'
                    f'text-transform:uppercase;letter-spacing:0.8px;">Known for</div>'
                    f'{known_body}</div>'
                )

                # "Shared tones" block — always rendered for consistent card height
                matched = p.get("matched_tones") or []
                if matched:
                    tone_pills = "".join(
                        _talent_pill(str(t).title(), "#9ad")
                        for t in matched[:3]
                    )
                    tone_body = f'<div style="margin-top:4px;">{tone_pills}</div>'
                else:
                    tone_body = (
                        f'<div style="color:{CARD_TEXT_MUTED};font-size:0.75em;'
                        f'font-style:italic;margin-top:3px;">Broad tonal range — no single-lane bias.</div>'
                    )

                tone_block = (
                    f'<div style="margin-top:10px;">'
                    f'<div style="color:{CARD_TEXT_MUTED};font-size:0.66em;font-weight:700;'
                    f'text-transform:uppercase;letter-spacing:0.8px;">Shared tones</div>'
                    f'{tone_body}</div>'
                )

                st.markdown(
                    f'<div style="background:{CARD_BG};border:1px solid {CARD_BORDER};'
                    f'border-radius:10px;padding:14px 16px 10px;margin-bottom:10px;'
                    f'min-height:300px;display:flex;flex-direction:column;">'
                    f'<div style="font-weight:700;color:{CARD_TEXT};font-size:1.02em;line-height:1.2;">{name}</div>'
                    f'<div style="color:{CARD_TEXT_MUTED};font-size:0.76em;margin-top:2px;">{subtitle}</div>'
                    f'{pill_row}{known_block}{tone_block}'
                    f'<div style="flex:1;"></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                btn_key = f'gs_talent_net_{p["role"]}_{p["person_id"]}'
                if st.button(
                    "Explore collaboration network →",
                    key=btn_key,
                    use_container_width=True,
                    type="secondary",
                ):
                    st.session_state["net_seed"] = p["person_id"]
                    st.switch_page("pages/07_Cast_Crew_Network.py")

            tcol_d, tcol_a = st.columns(2)
            with tcol_d:
                st.markdown(
                    f'<div style="color:{CARD_TEXT_MUTED};font-size:0.78em;font-weight:600;'
                    f'text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Directors</div>',
                    unsafe_allow_html=True,
                )
                if not directors:
                    st.caption("No directors in this genre cross-section above the quality floor.")
                else:
                    for p in directors:
                        _render_person_card_v2(p)
            with tcol_a:
                st.markdown(
                    f'<div style="color:{CARD_TEXT_MUTED};font-size:0.78em;font-weight:600;'
                    f'text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Cast</div>',
                    unsafe_allow_html=True,
                )
                if not cast:
                    st.caption("No cast in this genre cross-section above the quality floor.")
                else:
                    for p in cast:
                        _render_person_card_v2(p)

        # ── Row 5: Model Card expander ───────────────────────────────────────
        with st.expander("Model Card", expanded=False):
            model = load_greenlight_model("movie" if content_type == "Movie" else "show")
            if model:
                cv_r   = getattr(model, "cv_rmse_", None)
                base_r = getattr(model, "baseline_rmse_", None)
                t_size = getattr(model, "training_size_", "N/A")
                g_mean = getattr(model, "global_mean_", None)
                f_names = getattr(model, "feature_names_", [])
                n_svd   = len(getattr(model, "svd_cols_", []) or [])
                tones_n = len(getattr(model, "tone_keys_", []) or [])

                cv_str     = f"{cv_r:.3f}"   if cv_r   is not None else "N/A"
                base_str   = f"{base_r:.3f}" if base_r is not None else "N/A"
                impr_str   = f"{(1 - cv_r/base_r)*100:.1f}%" if (cv_r and base_r and base_r > 0) else "N/A"
                t_size_str = f"{t_size:,}" if isinstance(t_size, (int, float)) else str(t_size)

                gbm = getattr(model, "gbm_", None)
                if gbm is not None and hasattr(gbm, "feature_importances_"):
                    imps = pd.Series(gbm.feature_importances_, index=f_names)
                    top5 = imps.sort_values(ascending=False).head(5)
                    def _clean(f):
                        if f.startswith("genre_"):
                            return f"{genre_display(f.replace('genre_', ''))} (genre)"
                        if f.startswith("tone_"):    return f.replace("tone_", "").replace("_", " ").title() + " (tone)"
                        if f.startswith("gpair_"):
                            parts = f.replace("gpair_", "").split("_x_")
                            if len(parts) == 2:
                                return f"{genre_display(parts[0])} × {genre_display(parts[1])} (pair)"
                            return f.replace("gpair_", "").replace("_x_", "×").title() + " (pair)"
                        if f.startswith("svd_"):     return f"Concept text dim {f.split('_')[1]}"
                        if f.startswith("ftype_"):   return f.replace("ftype_", "").title() + " (IP)"
                        return f.replace("_", " ").title()
                    top5_display = ", ".join([f"**{_clean(f)}** ({v:.3f})" for f, v in top5.items()])
                else:
                    top5_display = "(HistGradientBoosting does not expose importances in this sklearn build)"

                st.markdown(f"""
**Model type:** Stacked HistGradientBoostingRegressor (structured + SVD) blended 0.8/0.2 with a Ridge on the SVD subspace. Target: Bayesian-shrunk IMDb score.

**Features:** {len(f_names)} total — {len(f_names) - n_svd} structured (genres, pairwise interactions, tones, runtime², cert×budget, franchise one-hots) + {n_svd} SVD components from the concept description TF-IDF + {tones_n} tone one-hots.

**Training data:** {t_size_str} {content_type.lower()}s with IMDb score and ≥ 500 votes.

**Cross-validation:** 5-fold

**CV RMSE:** {cv_str}   **Baseline RMSE:** {base_str} (naive global mean {f"{g_mean:.2f}" if g_mean else "N/A"})

**Improvement:** {impr_str} over naive mean prediction

**Top 5 features:** {top5_display}

**Franchise labeling rule:** `collection_name` populated → Franchise entry; `tmdb_keywords` containing phrases like “based on novel/comic/true story/video game/play” → Adaptation; else → Original.

**Limitations:** Catalog metadata only — does not account for marketing spend, release window, or star power. Predictions are directional estimates, not forecasts.
""")
            else:
                st.info("Greenlight model not loaded.")





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

            # Reset pagination whenever any filter changes
            _filter_sig = (fe_search or "", fe_sort, int(fe_min))
            if st.session_state.fe_filter_sig != _filter_sig:
                st.session_state.fe_filter_sig = _filter_sig
                st.session_state.fe_page_size = 15

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

            total_franchises = len(display_df)
            page_size = int(st.session_state.fe_page_size)
            visible_df = display_df.iloc[:page_size]

            rows_of_3 = [visible_df.iloc[i:i+3] for i in range(0, len(visible_df), 3)]
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
                            if st.session_state.fe_selected != fname:
                                st.session_state.fe_title_selected = None
                            st.session_state.fe_selected = fname
                            st.rerun()

            # ── Pagination footer ─────────────────────────────────────────────
            shown = min(page_size, total_franchises)
            st.markdown(
                f'<div style="text-align:center;color:{CARD_TEXT_MUTED};font-size:0.82em;'
                f'margin:6px 0 4px 0;">Showing {shown} of {total_franchises} franchises</div>',
                unsafe_allow_html=True,
            )
            if total_franchises > page_size:
                _, _lm_col, _ = st.columns([2, 1, 2])
                with _lm_col:
                    remaining = total_franchises - page_size
                    inc = min(15, remaining)
                    if st.button(f"Load {inc} more", key="fe_load_more", use_container_width=True):
                        st.session_state.fe_page_size = page_size + inc
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
                    # Section 1: Clickable poster strip — columns so buttons work
                    st.markdown(
                        section_header_html(
                            "Titles in Franchise",
                            "Click a title to see its full details below.",
                        ),
                        unsafe_allow_html=True,
                    )
                    CHUNK = 6
                    sel_list = list(sel_titles.iterrows())
                    for chunk_start in range(0, len(sel_list), CHUNK):
                        chunk = sel_list[chunk_start:chunk_start + CHUNK]
                        strip_cols = st.columns(CHUNK)
                        for sc, (_, tr) in zip(strip_cols, chunk):
                            poster_url = tr.get("poster_url", "")
                            plat = tr.get("platform", tr.get("platforms", ""))
                            if isinstance(plat, (list, np.ndarray)):
                                plat_first = plat[0] if len(plat) > 0 else ""
                            else:
                                plat_first = plat
                            plat_color = PLATFORMS.get(str(plat_first), {}).get("color", "#555")
                            initial = str(tr.get("title", "?"))[0].upper()
                            year_s = str(int(tr["release_year"])) if pd.notna(tr.get("release_year")) else ""
                            imdb_s = f"IMDb {tr['imdb_score']:.1f}" if pd.notna(tr.get("imdb_score")) else ""
                            title_id = tr.get("id", "")
                            is_sel = (st.session_state.fe_title_selected == title_id)
                            border_style = f"2px solid {CARD_ACCENT}" if is_sel else "2px solid transparent"
                            if pd.notna(poster_url) and str(poster_url).startswith("http"):
                                img_html = (
                                    f'<img src="{poster_url}" '
                                    f'style="width:100%;aspect-ratio:2/3;object-fit:cover;'
                                    f'border-radius:4px;border:{border_style};box-sizing:border-box;">'
                                )
                            else:
                                img_html = (
                                    f'<div style="width:100%;aspect-ratio:2/3;background:{plat_color};'
                                    f'border-radius:4px;display:flex;align-items:center;justify-content:center;'
                                    f'font-size:2em;font-weight:700;color:#fff;border:{border_style};'
                                    f'box-sizing:border-box;">{initial}</div>'
                                )
                            badge_html = platform_badges_html(plat) if plat is not None else ""
                            with sc:
                                st.markdown(
                                    f'<div style="text-align:center;">'
                                    f'{img_html}'
                                    f'<div style="font-weight:600;color:{CARD_TEXT};font-size:0.78em;'
                                    f'margin-top:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;" '
                                    f'title="{tr.get("title","?")}">'
                                    f'{tr.get("title","?")}</div>'
                                    f'<div style="color:{CARD_TEXT_MUTED};font-size:0.72em;">{year_s}</div>'
                                    f'<div style="font-size:0.72em;color:{CARD_ACCENT};">{imdb_s}</div>'
                                    f'<div style="margin:2px 0;">{badge_html}</div>'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )
                                btn_label = "Hide ✕" if is_sel else "View ▸"
                                if st.button(btn_label, key=f"fe_title_btn_{title_id}", use_container_width=True):
                                    st.session_state.fe_title_selected = None if is_sel else title_id
                                    st.rerun()

                    # ── Inline Title Card (replicates Explore Catalog pattern) ──
                    if st.session_state.fe_title_selected:
                        _sel_id = st.session_state.fe_title_selected
                        _sel_rows = sel_titles[sel_titles["id"] == _sel_id]
                        if not _sel_rows.empty:
                            _sel = _sel_rows.iloc[0]
                            _enr = _sel  # enriched view already IS the enriched table
                            st.markdown("")
                            _tc_head, _tc_close = st.columns([6, 1])
                            with _tc_head:
                                st.markdown(
                                    section_header_html(
                                        "Title Details",
                                        f"{_sel.get('title', '')}",
                                    ),
                                    unsafe_allow_html=True,
                                )
                            with _tc_close:
                                if st.button("✕ Close", key="fe_title_close"):
                                    st.session_state.fe_title_selected = None
                                    st.rerun()

                            with st.container(border=True):
                                _poster = _enr.get("poster_url")
                                if _poster and str(_poster) != "nan" and str(_poster).startswith("http"):
                                    st.image(_poster, width=200)

                                st.subheader(_sel.get("title", ""))

                                _aw = _enr.get("award_wins", 0)
                                _an = _enr.get("award_noms", 0)
                                try:
                                    _aw = float(_aw) if pd.notna(_aw) else 0
                                    _an = float(_an) if pd.notna(_an) else 0
                                except Exception:
                                    _aw, _an = 0, 0
                                if _aw and _aw > 0:
                                    _noms_str = f", {int(_an)} nominations" if _an and _an > 0 else ""
                                    st.markdown(
                                        f'<span style="background:rgba(46,204,113,0.15);color:#2ecc71;'
                                        f'border:1px solid #2ecc71;padding:4px 12px;border-radius:12px;'
                                        f'font-size:0.83em;font-weight:600;">'
                                        f'{int(_aw)} wins{_noms_str}</span>',
                                        unsafe_allow_html=True,
                                    )

                                def _meta_cell(label: str, value: str) -> str:
                                    return (
                                        f'<div style="padding:8px 0;">'
                                        f'<div style="color:{CARD_TEXT_MUTED};font-size:0.72em;text-transform:uppercase;'
                                        f'letter-spacing:0.04em;margin-bottom:3px;">{label}</div>'
                                        f'<div style="font-weight:600;color:{CARD_TEXT};font-size:0.95em;">{value}</div>'
                                        f'</div>'
                                    )

                                m1, m2, m3, m4 = st.columns(4)
                                with m1:
                                    st.markdown(_meta_cell("Type", str(_sel.get("type", "N/A"))), unsafe_allow_html=True)
                                with m2:
                                    _yr = _sel.get("release_year")
                                    _yr_s = str(int(_yr)) if pd.notna(_yr) else "N/A"
                                    st.markdown(_meta_cell("Year", _yr_s), unsafe_allow_html=True)
                                with m3:
                                    _ims = _sel.get("imdb_score")
                                    _ims_s = f"{_ims:.1f}" if pd.notna(_ims) else "N/A"
                                    st.markdown(_meta_cell("IMDb", _ims_s), unsafe_allow_html=True)
                                with m4:
                                    _plat_list = _sel.get("platforms", _sel.get("platform", ""))
                                    _plat_label = "Platforms" if isinstance(_plat_list, (list, np.ndarray)) and len(_plat_list) > 1 else "Platform"
                                    st.markdown(
                                        f'<div style="padding:8px 0;">'
                                        f'<div style="color:{CARD_TEXT_MUTED};font-size:0.72em;text-transform:uppercase;'
                                        f'letter-spacing:0.04em;margin-bottom:5px;">{_plat_label}</div>'
                                        f'<div>{platform_badges_html(_plat_list)}</div>'
                                        f'</div>',
                                        unsafe_allow_html=True,
                                    )

                                r1, r2, r3, r4 = st.columns(4)
                                _cert = _sel.get("age_certification", "")
                                _cert_s = _cert if _cert and str(_cert) != "nan" else "N/A"
                                with r1:
                                    st.markdown(_meta_cell("Rating", _cert_s), unsafe_allow_html=True)
                                _rt = _sel.get("runtime")
                                _rt_s = f"{int(_rt)} min" if pd.notna(_rt) and _rt else "N/A"
                                with r2:
                                    st.markdown(_meta_cell("Runtime", _rt_s), unsafe_allow_html=True)
                                _vv = _sel.get("imdb_votes")
                                _vv_s = format_votes(_vv) if pd.notna(_vv) and _vv else "N/A"
                                with r3:
                                    st.markdown(_meta_cell("Votes", _vv_s), unsafe_allow_html=True)
                                try:
                                    _qs = float(compute_quality_score(_sel.to_frame().T).iloc[0])
                                except Exception:
                                    _qs = 0.0
                                _qs_color = "#2ecc71" if _qs >= 8.0 else "#f39c12" if _qs >= 7.0 else "#e74c3c"
                                _qs_pct = min(_qs * 10, 100)
                                with r4:
                                    st.markdown(
                                        f'<div style="padding:8px 0;">'
                                        f'<div style="color:{CARD_TEXT_MUTED};font-size:0.72em;text-transform:uppercase;'
                                        f'letter-spacing:0.04em;margin-bottom:5px;">Quality Score</div>'
                                        f'<div style="display:flex;align-items:center;gap:6px;">'
                                        f'<div style="flex:1;background:#2a2a3e;border-radius:3px;height:5px;">'
                                        f'<div style="width:{_qs_pct:.0f}%;height:100%;background:{_qs_color};border-radius:3px;"></div>'
                                        f'</div>'
                                        f'<span style="font-size:0.82em;color:{_qs_color};font-weight:600;">{_qs:.1f}</span>'
                                        f'</div>'
                                        f'</div>',
                                        unsafe_allow_html=True,
                                    )

                                _bo = _enr.get("box_office_usd")
                                if pd.notna(_bo) and _bo and float(_bo) > 0:
                                    _bo_f = float(_bo)
                                    if _bo_f >= 1e9:
                                        _bo_str = f"${_bo_f/1e9:.1f}B"
                                    elif _bo_f >= 1e6:
                                        _bo_str = f"${_bo_f/1e6:.0f}M"
                                    else:
                                        _bo_str = f"${_bo_f:,.0f}"
                                    st.markdown(
                                        f'<div style="color:{CARD_TEXT_MUTED};font-size:0.78em;margin-top:2px;">'
                                        f'Box Office (Wikidata): <strong style="color:{CARD_TEXT};">{_bo_str}</strong>'
                                        f'</div>',
                                        unsafe_allow_html=True,
                                    )

                                _genres = _sel.get("genres")
                                if isinstance(_genres, (list, np.ndarray)) and len(_genres) > 0:
                                    _genre_pills = " ".join(
                                        f'<span style="background:{CARD_BORDER};color:{CARD_TEXT};'
                                        f'padding:3px 10px;border-radius:12px;font-size:0.82em;margin-right:3px;">'
                                        f'{str(g).title()}</span>'
                                        for g in _genres
                                    )
                                    st.markdown(f'<div style="margin:8px 0;">{_genre_pills}</div>', unsafe_allow_html=True)

                                st.divider()
                                _desc = _sel.get("description", "")
                                if pd.notna(_desc) and str(_desc):
                                    st.markdown(
                                        f'<div style="font-size:0.9em;color:{CARD_TEXT};line-height:1.6;">{_desc}</div>',
                                        unsafe_allow_html=True,
                                    )
                                else:
                                    st.caption("No description available.")

                                with st.expander("Cast & Crew"):
                                    _all_credits = load_all_platforms_credits()
                                    _tc = _all_credits[_all_credits["title_id"] == _sel_id]
                                    _dirs = _tc[_tc["role"] == "DIRECTOR"]
                                    _acts = _tc[_tc["role"] == "ACTOR"]
                                    if not _dirs.empty:
                                        _dn = _dirs["name"].drop_duplicates().tolist()
                                        _dl = "Directors" if len(_dn) > 1 else "Director"
                                        st.markdown(
                                            f'<div style="color:{CARD_TEXT_MUTED};font-size:0.75em;text-transform:uppercase;'
                                            f'letter-spacing:0.04em;margin-bottom:4px;">{_dl}</div>',
                                            unsafe_allow_html=True,
                                        )
                                        st.markdown(f"**{', '.join(_dn)}**")
                                    if not _acts.empty:
                                        st.markdown(
                                            f'<div style="color:{CARD_TEXT_MUTED};font-size:0.75em;text-transform:uppercase;'
                                            f'letter-spacing:0.04em;margin:8px 0 4px;">Cast</div>',
                                            unsafe_allow_html=True,
                                        )
                                        _al = _acts.head(16).reset_index(drop=True)
                                        _ac1, _ac2 = st.columns(2)
                                        for _i, (_, _a) in enumerate(_al.iterrows()):
                                            _ch = _a.get("character", "")
                                            _ch_html = (
                                                f'<span style="color:{CARD_TEXT_MUTED};font-size:0.8em;"> as {_ch}</span>'
                                                if _ch and str(_ch) not in ("nan", "") else ""
                                            )
                                            _entry = (
                                                f'<div style="font-size:0.85em;color:{CARD_TEXT};padding:2px 0;">'
                                                f'<strong>{_a["name"]}</strong>{_ch_html}</div>'
                                            )
                                            with (_ac1 if _i % 2 == 0 else _ac2):
                                                st.markdown(_entry, unsafe_allow_html=True)
                                        if len(_acts) > 16:
                                            st.caption(f"+{len(_acts) - 16} more cast members")
                                    if _dirs.empty and _acts.empty:
                                        st.caption("No cast & crew information available for this title.")
                                    st.page_link(
                                        "pages/07_Cast_Crew_Network.py",
                                        label="View full network →",
                                    )

                    # Section 2: Quality timeline
                    timeline_data = sel_titles[sel_titles["imdb_score"].notna() & sel_titles["release_year"].notna()].copy()
                    timeline_data = timeline_data.sort_values("release_year")
                    if not timeline_data.empty:
                        years = pd.to_numeric(timeline_data["release_year"], errors="coerce").astype(int)
                        scores = pd.to_numeric(timeline_data["imdb_score"], errors="coerce")
                        y_min, y_max = int(years.min()), int(years.max())
                        span = max(1, y_max - y_min)

                        # Dormancy = mean gap between consecutive releases
                        sorted_years = sorted(years.unique().tolist())
                        if len(sorted_years) >= 2:
                            gaps = [sorted_years[i+1] - sorted_years[i] for i in range(len(sorted_years)-1)]
                            avg_gap = sum(gaps) / len(gaps)
                            dormancy_str = f"{avg_gap:.1f} yr avg between releases"
                        else:
                            dormancy_str = "single-year release"

                        subtitle = f"{len(timeline_data)} entries · {y_min}–{y_max} · {dormancy_str}"
                        st.markdown(
                            section_header_html("Quality Over Time", subtitle),
                            unsafe_allow_html=True,
                        )

                        votes_raw = pd.to_numeric(timeline_data.get("imdb_votes", pd.Series(dtype=float)), errors="coerce").fillna(1000)
                        max_v = votes_raw.max() or 1
                        sizes = np.clip((votes_raw / max_v * 20 + 8).to_numpy(), 8, 22).tolist()

                        votes_fmt = [format_votes(v) if pd.notna(v) else "N/A" for v in votes_raw]
                        award_col = pd.to_numeric(timeline_data.get("award_wins", pd.Series(dtype=float)), errors="coerce").fillna(0).astype(int).tolist()
                        customdata = list(zip(votes_fmt, award_col))

                        fig_tl = go.Figure()

                        # Shaded quality bands
                        fig_tl.add_hrect(y0=6.5, y1=7.5, fillcolor=CARD_TEXT_MUTED, opacity=0.06, line_width=0)
                        fig_tl.add_hrect(y0=7.5, y1=10, fillcolor="#2ecc71", opacity=0.07, line_width=0)

                        # Trajectory line
                        fig_tl.add_trace(go.Scatter(
                            x=years.tolist(),
                            y=scores.tolist(),
                            mode="lines+markers",
                            marker=dict(size=sizes, color=CARD_ACCENT, line=dict(color="#1a1a2e", width=1)),
                            line=dict(color=CARD_ACCENT, width=1.5, dash="dot"),
                            text=timeline_data["title"].tolist(),
                            customdata=customdata,
                            hovertemplate=(
                                "<b>%{text}</b><br>Year: %{x}<br>IMDb: %{y:.1f}"
                                "<br>Votes: %{customdata[0]}<br>Awards: %{customdata[1]}<extra></extra>"
                            ),
                            name="Entries",
                        ))

                        # Linear trendline (≥3 entries)
                        if len(years) >= 3:
                            coef = np.polyfit(years.values, scores.values, 1)
                            trend_y = np.polyval(coef, years.values)
                            fig_tl.add_trace(go.Scatter(
                                x=years.tolist(),
                                y=trend_y.tolist(),
                                mode="lines",
                                line=dict(color=CARD_ACCENT, width=1.5, dash="solid"),
                                opacity=0.35,
                                name="Trend",
                                hoverinfo="skip",
                                showlegend=False,
                            ))

                        # Award-winning stars
                        if "award_wins" in timeline_data.columns:
                            award_titles = timeline_data[pd.to_numeric(timeline_data["award_wins"], errors="coerce").fillna(0) > 0]
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

                        fig_tl.add_hline(
                            y=7.0, line_dash="dash", line_color=CARD_TEXT_MUTED,
                            annotation_text="Quality threshold (7.0)", annotation_position="bottom right",
                        )
                        fig_tl.update_layout(
                            template=PLOTLY_TEMPLATE, height=320,
                            margin=dict(l=10, r=10, t=20, b=10),
                            showlegend=False,
                            xaxis_title="Year", yaxis_title="IMDb Score",
                            xaxis=dict(tickmode="linear", dtick=max(1, span // 10), range=[y_min - 0.5, y_max + 0.5]),
                            yaxis=dict(range=[max(0, float(scores.min()) - 0.5), 10]),
                        )
                        st.plotly_chart(fig_tl, use_container_width=True)

                    # Section 3: Platform ownership
                    st.markdown(
                        section_header_html(
                            "Platform Ownership",
                            "Which platforms hold titles in this franchise.",
                        ),
                        unsafe_allow_html=True,
                    )
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

                    # Section 4: Cast & Crew Continuity — the "spine" of the franchise
                    st.markdown(
                        section_header_html(
                            "Cast & Crew Continuity",
                            "People who carry this franchise across multiple entries.",
                        ),
                        unsafe_allow_html=True,
                    )

                    fe_title_ids = sel_titles["id"].dropna().tolist()
                    _fe_credits_all = load_all_platforms_credits()
                    _fe_credits = _fe_credits_all[_fe_credits_all["title_id"].isin(fe_title_ids)]

                    if _fe_credits.empty:
                        st.markdown(
                            f'<div style="color:{CARD_TEXT_MUTED};font-size:0.85em;font-style:italic;">'
                            f'No cast or crew data available for this franchise.</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        _person_key = "person_id" if "person_id" in _fe_credits.columns else "name"
                        _dirs = _fe_credits[_fe_credits["role"] == "DIRECTOR"]
                        _acts = _fe_credits[_fe_credits["role"] == "ACTOR"]

                        def _top_recurring(df_role, limit):
                            if df_role.empty:
                                return []
                            grp = (
                                df_role.drop_duplicates(subset=[_person_key, "title_id"])
                                .groupby(_person_key)
                                .agg(name=("name", "first"), n=("title_id", "nunique"))
                                .reset_index()
                            )
                            grp = grp[grp["n"] >= 2].sort_values("n", ascending=False).head(limit)
                            return list(zip(grp["name"].tolist(), grp["n"].tolist()))

                        top_dirs = _top_recurring(_dirs, 6)
                        top_acts = _top_recurring(_acts, 8)

                        def _chip_row(label, items, color):
                            if not items:
                                return
                            chips = "".join(
                                f'<span style="background:{color}1f;color:{color};'
                                f'border:1px solid {color};padding:3px 10px;border-radius:12px;'
                                f'font-size:0.82em;margin:2px 4px 2px 0;display:inline-block;">'
                                f'{nm} · {n} films</span>'
                                for nm, n in items
                            )
                            st.markdown(
                                f'<div style="margin:6px 0;">'
                                f'<span style="color:{CARD_TEXT_MUTED};font-size:0.75em;text-transform:uppercase;'
                                f'letter-spacing:0.04em;margin-right:8px;">{label}</span>'
                                f'{chips}</div>',
                                unsafe_allow_html=True,
                            )

                        if not top_dirs and not top_acts:
                            st.markdown(
                                f'<div style="color:{CARD_TEXT_MUTED};font-size:0.85em;font-style:italic;">'
                                f'No recurring cast or crew across this franchise\'s entries.</div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            _chip_row("Directors", top_dirs, CARD_ACCENT)
                            _chip_row("Actors", top_acts, "#8EA9DB")
                            st.page_link(
                                "pages/07_Cast_Crew_Network.py",
                                label="Explore these people in the Cast & Crew Network →",
                            )

                    # Section 5: Franchise Health Score — with diagnostic breakdown
                    avg_i = pd.to_numeric(sel_titles["imdb_score"], errors="coerce").mean()
                    latest_y = pd.to_numeric(sel_titles["release_year"], errors="coerce").max()
                    awards_total = pd.to_numeric(sel_titles.get("award_wins", pd.Series(dtype=float)), errors="coerce").fillna(0).sum() if "award_wins" in sel_titles.columns else 0

                    imdb_norm = (float(avg_i) - 1) / 9 if pd.notna(avg_i) else 0.5
                    recency_norm = (float(latest_y) - 1970) / 60 if pd.notna(latest_y) else 0.5
                    prestige_norm = min(1.0, float(awards_total) / (len(sel_titles) * 5 + 1))

                    quality_component = round(imdb_norm * 100)
                    recency_component = round(recency_norm * 100)
                    prestige_component = round(prestige_norm * 100)
                    health = round((imdb_norm * 0.4 + recency_norm * 0.3 + prestige_norm * 0.3) * 100)

                    if health < 40:
                        h_label, h_color = "Declining", "#e74c3c"
                    elif health < 65:
                        h_label, h_color = "Stable", "#f39c12"
                    elif health < 80:
                        h_label, h_color = "Thriving", "#2ecc71"
                    else:
                        h_label, h_color = "Legendary", "#FFD700"

                    def _component_bar(label, value, weight):
                        color = "#2ecc71" if value >= 75 else "#f39c12" if value >= 50 else "#e74c3c"
                        return (
                            f'<div style="margin:6px 0;">'
                            f'<div style="display:flex;justify-content:space-between;font-size:0.8em;color:{CARD_TEXT};margin-bottom:3px;">'
                            f'<span>{label} <span style="color:{CARD_TEXT_MUTED};font-size:0.85em;">({weight}%)</span></span>'
                            f'<span style="color:{color};font-weight:600;">{value}</span>'
                            f'</div>'
                            f'<div style="background:#2a2a3e;border-radius:3px;height:5px;">'
                            f'<div style="width:{min(value,100)}%;height:100%;background:{color};border-radius:3px;"></div>'
                            f'</div>'
                            f'</div>'
                        )

                    st.markdown(
                        section_header_html("Franchise Health Score"),
                        unsafe_allow_html=True,
                    )
                    h_col1, h_col2 = st.columns([1, 2])
                    with h_col1:
                        st.markdown(
                            f'<div style="background:{CARD_BG};border:1px solid {CARD_BORDER};'
                            f'border-top:3px solid {h_color};border-radius:8px;padding:16px;text-align:center;">'
                            f'<div style="color:{CARD_TEXT_MUTED};font-size:0.72em;text-transform:uppercase;letter-spacing:1px;">Overall</div>'
                            f'<div style="font-size:2.8em;font-weight:800;color:{h_color};line-height:1;margin:4px 0;">{health}</div>'
                            f'<div style="color:{h_color};font-size:0.9em;font-weight:600;">{h_label}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    with h_col2:
                        st.markdown(
                            f'<div style="background:{CARD_BG};border:1px solid {CARD_BORDER};'
                            f'border-radius:8px;padding:12px 16px;">'
                            f'<div style="color:{CARD_TEXT_MUTED};font-size:0.75em;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Driven By</div>'
                            + _component_bar("Quality (avg IMDb)", quality_component, 40)
                            + _component_bar("Recency (latest entry)", recency_component, 30)
                            + _component_bar("Prestige (awards)", prestige_component, 30)
                            + f'</div>',
                            unsafe_allow_html=True,
                        )

            # ── Franchise Gaps Section ────────────────────────────────────────
            st.divider()
            st.markdown(
                section_header_html(
                    "Franchise Gaps — Acquisition Targets",
                    "Franchises where competitors hold more titles than Netflix+Max combined. Acquiring these titles would consolidate the IP under the merged entity.",
                ),
                unsafe_allow_html=True,
            )

            gap_rows = []
            for _, frow in franchise_table.iterrows():
                pd_dist = frow.get("platform_dist", {})
                if not pd_dist:
                    continue
                total_titles = sum(pd_dist.values())
                merged_cnt = sum(pd_dist.get(p, 0) for p in MERGED_PLATFORMS)
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
                        "total": total_titles,
                        "plat_dist": pd_dist,
                        "gap": gap_mag,
                    })

            gap_rows = sorted(gap_rows, key=lambda x: -x["gap"])
            if not gap_rows:
                st.info("No significant franchise gaps found (no competitor leads by 3+ titles in any franchise).")
            else:
                total_acquirable = sum(g["comp_count"] for g in gap_rows)
                st.markdown(
                    f'<div style="background:rgba(0,180,166,0.08);border:1px solid {CARD_ACCENT};'
                    f'border-radius:6px;padding:10px 14px;margin-bottom:10px;'
                    f'display:flex;gap:24px;align-items:center;">'
                    f'<div><span style="color:{CARD_TEXT_MUTED};font-size:0.72em;text-transform:uppercase;letter-spacing:1px;">Franchises with gap</span>'
                    f'<div style="color:{CARD_ACCENT};font-weight:700;font-size:1.3em;">{len(gap_rows)}</div></div>'
                    f'<div><span style="color:{CARD_TEXT_MUTED};font-size:0.72em;text-transform:uppercase;letter-spacing:1px;">Acquirable titles</span>'
                    f'<div style="color:{CARD_ACCENT};font-weight:700;font-size:1.3em;">{total_acquirable}</div></div>'
                    f'<div><span style="color:{CARD_TEXT_MUTED};font-size:0.72em;text-transform:uppercase;letter-spacing:1px;">Top gap</span>'
                    f'<div style="color:{CARD_ACCENT};font-weight:700;font-size:1.3em;">+{gap_rows[0]["gap"]}</div></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                default_limit = 8
                visible_gaps = gap_rows if st.session_state.fe_show_all_gaps else gap_rows[:default_limit]

                for gi, gr in enumerate(visible_gaps):
                    comp_name = PLATFORMS.get(gr["competitor"], {}).get("name", gr["competitor"])
                    comp_color = PLATFORMS.get(gr["competitor"], {}).get("color", "#e74c3c")
                    tier_label = "Acquisition target" if gr["gap"] >= 5 else "Partial gap"
                    tier_color = "#e74c3c" if gr["gap"] >= 5 else "#f39c12"
                    bar_html = _platform_bar_html(gr["plat_dist"])

                    gc_left, gc_right = st.columns([5, 1])
                    with gc_left:
                        st.markdown(
                            f'<div style="background:{CARD_BG};border:1px solid {CARD_BORDER};'
                            f'border-left:4px solid {comp_color};border-radius:8px;'
                            f'padding:12px 16px;margin-bottom:4px;">'
                            f'<div style="display:flex;justify-content:space-between;align-items:center;gap:12px;">'
                            f'<div style="font-weight:700;color:{CARD_TEXT};font-size:0.98em;">{gr["franchise"]}</div>'
                            f'<div style="display:flex;gap:6px;align-items:center;white-space:nowrap;">'
                            f'<span style="background:{comp_color}22;color:{comp_color};padding:3px 10px;'
                            f'border-radius:8px;border:1px solid {comp_color};font-size:0.82em;font-weight:700;">+{gr["gap"]} titles</span>'
                            f'<span style="background:{tier_color}22;color:{tier_color};padding:3px 10px;'
                            f'border-radius:8px;font-size:0.75em;font-weight:600;">{tier_label}</span>'
                            f'</div>'
                            f'</div>'
                            f'{bar_html}'
                            f'<div style="color:{CARD_TEXT_MUTED};font-size:0.82em;margin-top:2px;">'
                            f'<strong style="color:{comp_color};">{comp_name}</strong> holds {gr["comp_count"]} of {gr["total"]} titles · '
                            f'<strong style="color:{CARD_TEXT};">Netflix+Max</strong> holds {gr["merged_count"]}'
                            f'</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    with gc_right:
                        if st.button("View franchise →", key=f"fe_gap_view_{gi}_{gr['franchise'][:20]}", use_container_width=True):
                            if st.session_state.fe_selected != gr["franchise"]:
                                st.session_state.fe_title_selected = None
                            st.session_state.fe_selected = gr["franchise"]
                            st.rerun()

                if len(gap_rows) > default_limit:
                    _, _exp_col, _ = st.columns([2, 2, 2])
                    with _exp_col:
                        if st.session_state.fe_show_all_gaps:
                            if st.button(f"Show top {default_limit} only", key="fe_gaps_collapse", use_container_width=True):
                                st.session_state.fe_show_all_gaps = False
                                st.rerun()
                        else:
                            if st.button(f"Show all {len(gap_rows)} gaps", key="fe_gaps_expand", use_container_width=True):
                                st.session_state.fe_show_all_gaps = True
                                st.rerun()


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

    # ── Strategy definitions ──────────────────────────────────────────────────
    STRATEGIES = {
        "Blockbuster Hunter": {
            "tagline": "Chases mass-appeal hits — huge vote counts, broad audiences.",
            "focus": ["pct_above_7", "avg_votes", "freshness"],
            "icon": "🎬",
        },
        "Critic's Darling": {
            "tagline": "Chases prestige — top IMDb scores and award winners.",
            "focus": ["avg_imdb", "prestige", "pct_above_7"],
            "icon": "🏆",
        },
        "Balanced Strategist": {
            "tagline": "Picks the strongest all-round title every round — no weakness.",
            "focus": ["avg_imdb", "pct_above_7", "prestige", "diversity", "freshness"],
            "icon": "⚖️",
        },
    }

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
            return {"avg_imdb": 0, "pct_above_7": 0, "prestige": 0, "diversity": 0,
                    "freshness": 0, "total": 0, "avg_votes": 0}
        df = pd.DataFrame(picks)
        imdb = pd.to_numeric(df.get("imdb_score", pd.Series(dtype=float)), errors="coerce").dropna()
        avg_imdb = float(imdb.mean()) if len(imdb) > 0 else 0
        pct_above_7 = float((imdb >= 7.0).mean() * 100) if len(imdb) > 0 else 0
        award_wins = pd.to_numeric(df.get("award_wins_val", df.get("award_wins", pd.Series(0, index=df.index))), errors="coerce").fillna(0)
        prestige = float((award_wins > 0).mean() * 100) if len(award_wins) > 0 else 0
        diversity = _genre_entropy(picks)
        ry = pd.to_numeric(df.get("release_year", pd.Series(dtype=float)), errors="coerce")
        freshness = float((ry >= 2015).mean() * 100) if len(ry) > 0 else 0
        votes = pd.to_numeric(df.get("imdb_votes", pd.Series(dtype=float)), errors="coerce").dropna()
        avg_votes = float(votes.mean()) if len(votes) > 0 else 0
        return {
            "total": len(picks),
            "avg_imdb": avg_imdb,
            "pct_above_7": pct_above_7,
            "prestige": prestige,
            "diversity": diversity,
            "freshness": freshness,
            "avg_votes": avg_votes,
        }

    def _ai_strategy_score(pool_df, strategy):
        """Returns a per-row score series matching pool_df index, per AI strategy."""
        bay = pool_df["bayesian_score"].fillna(6.0)
        votes = pd.to_numeric(pool_df.get("imdb_votes", pd.Series(1000, index=pool_df.index)), errors="coerce").fillna(1000)
        log_votes = np.log1p(votes)
        log_votes_norm = (log_votes - log_votes.min()) / (log_votes.max() - log_votes.min() + 1e-9)
        aw = pool_df["award_wins_val"]
        aw_norm = aw / (aw.max() + 1e-9)
        bay_norm = (bay - 5.0) / 5.0  # scale ~0-1 assuming 5-10 range
        bay_norm = bay_norm.clip(0, 1)

        if strategy == "Blockbuster Hunter":
            # Heavy on vote-count/popularity, moderate quality floor
            return 0.55 * log_votes_norm + 0.30 * bay_norm + 0.15 * aw_norm
        if strategy == "Critic's Darling":
            # Heavy on quality + prestige
            return 0.55 * bay_norm + 0.35 * aw_norm + 0.10 * log_votes_norm
        # Balanced Strategist — dominant all-rounder
        return 0.40 * bay_norm + 0.30 * log_votes_norm + 0.30 * aw_norm

    def _ai_pick(strategy, pool_df, user_picks):
        """Pick the best title from pool_df per strategy. Returns a dict."""
        if pool_df.empty:
            return None
        scores = _ai_strategy_score(pool_df, strategy)
        # Small diversification nudge: if AI already has ≥3 titles in a single genre,
        # discount further titles in that genre by 10% so it doesn't mono-stack.
        ai_picks_so_far = st.session_state.draft_ai_picks
        if ai_picks_so_far:
            genre_counts = {}
            for p in ai_picks_so_far:
                for g in (p.get("genres") or []):
                    genre_counts[g] = genre_counts.get(g, 0) + 1
            saturated = {g for g, c in genre_counts.items() if c >= 3}
            if saturated:
                def _over(gl):
                    return isinstance(gl, (list, np.ndarray)) and any(g in saturated for g in gl)
                mask = pool_df["genres"].apply(_over)
                scores = scores - (mask.astype(float) * 0.1)
        idx = scores.idxmax()
        return pool_df.loc[idx].to_dict()

    def _user_picks_first(round_idx):
        """Snake draft: odd rounds (0-indexed 0,2,...) user first; even rounds (1,3,...) AI first."""
        return round_idx % 2 == 0

    # ── Setup Panel ───────────────────────────────────────────────────────────
    if not st.session_state.draft_active and not st.session_state.draft_complete:
        st.markdown(
            f'<div style="color:{CARD_TEXT_MUTED};font-size:0.9em;text-align:center;'
            f'max-width:700px;margin:0 auto 18px;">'
            f'Pick the kind of rival you want to face, then build a better streaming service across {6} to 12 rounds. '
            f'Draft order alternates — whoever picks second one round goes first the next.'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Strategy cards — click to select
        st.markdown(section_header_html("Choose Your AI Opponent"), unsafe_allow_html=True)
        strat_cols = st.columns(3)
        for i, (sname, sinfo) in enumerate(STRATEGIES.items()):
            selected = st.session_state.get("draft_strategy_choice") == sname
            border = CARD_ACCENT if selected else CARD_BORDER
            bg = "rgba(255,215,0,0.08)" if selected else CARD_BG
            with strat_cols[i]:
                st.markdown(
                    f'<div style="background:{bg};border:2px solid {border};border-radius:10px;'
                    f'padding:16px;text-align:center;min-height:140px;">'
                    f'<div style="font-size:2em;margin-bottom:4px;">{sinfo["icon"]}</div>'
                    f'<div style="color:{CARD_TEXT};font-weight:700;font-size:1em;margin-bottom:6px;">{sname}</div>'
                    f'<div style="color:{CARD_TEXT_MUTED};font-size:0.82em;line-height:1.35;">{sinfo["tagline"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                btn_label = "✓ Selected" if selected else "Select"
                if st.button(btn_label, key=f"strat_btn_{sname}", use_container_width=True,
                             type="primary" if selected else "secondary"):
                    st.session_state["draft_strategy_choice"] = sname
                    st.rerun()

        strategy_name = st.session_state.get("draft_strategy_choice") or "Balanced Strategist"

        st.markdown("<div style='margin-top:18px;'></div>", unsafe_allow_html=True)
        rounds = st.radio("Draft Rounds", [6, 8, 12], horizontal=True, key="draft_rounds_choice", index=1)

        st.markdown(
            f'<div style="color:{CARD_TEXT_MUTED};font-size:0.8em;text-align:center;margin:10px 0 6px;">'
            f'Title pool: all titles across all 6 platforms (IMDb votes ≥ 5,000).'
            f'</div>',
            unsafe_allow_html=True,
        )

        if st.button("Start Draft", type="primary", use_container_width=True):
            pool = _get_enriched().copy()
            pool = pool[pd.to_numeric(pool.get("imdb_votes", pd.Series(0, index=pool.index)), errors="coerce").fillna(0) > 5000].copy()

            pool["award_wins_val"] = pd.to_numeric(pool.get("award_wins", pd.Series(0, index=pool.index)), errors="coerce").fillna(0)
            pool["bayesian_score"] = bayesian_imdb(
                pd.to_numeric(pool["imdb_score"], errors="coerce"),
                pd.to_numeric(pool.get("imdb_votes", pd.Series(10000, index=pool.index)), errors="coerce"),
            )
            max_aw = pool["award_wins_val"].max() or 1
            pool["draft_value"] = (
                pool["bayesian_score"].fillna(6.5) * 0.5
                + np.log1p(pd.to_numeric(pool.get("imdb_votes", pd.Series(10000, index=pool.index)), errors="coerce").fillna(10000)) * 0.3
                + (pool["award_wins_val"] / max_aw) * 0.2
            )

            st.session_state.draft_active = True
            st.session_state.draft_complete = False
            st.session_state.draft_round = 0
            st.session_state.draft_user_picks = []
            st.session_state.draft_ai_picks = []
            st.session_state.draft_ai_prepicked_round = -1
            st.session_state.draft_win_celebrated = False
            st.session_state.draft_available_pool = pool.reset_index(drop=True)
            st.session_state.draft_settings = {
                "strategy": strategy_name,
                "rounds": rounds,
            }
            st.rerun()

    # ── Active Draft ──────────────────────────────────────────────────────────
    elif st.session_state.draft_active and not st.session_state.draft_complete:
        settings = st.session_state.draft_settings
        rounds = settings["rounds"]
        strategy_name = settings["strategy"]
        strat_icon = STRATEGIES.get(strategy_name, {}).get("icon", "🤖")
        current_round = st.session_state.draft_round

        pool = st.session_state.draft_available_pool.copy()
        picked_ids = set(
            [p.get("id") for p in st.session_state.draft_user_picks]
            + [p.get("id") for p in st.session_state.draft_ai_picks]
        )
        available = pool[~pool["id"].isin(picked_ids)].copy() if "id" in pool.columns else pool.copy()

        # Snake draft — AI pre-picks when it's AI's turn to go first
        user_first = _user_picks_first(current_round)
        if (not user_first) and st.session_state.draft_ai_prepicked_round != current_round:
            ai_pick = _ai_pick(strategy_name, available, st.session_state.draft_user_picks)
            if ai_pick:
                st.session_state.draft_ai_picks.append(ai_pick)
            st.session_state.draft_ai_prepicked_round = current_round
            st.rerun()

        # Turn indicator
        turn_label = "Your pick" if user_first else f"AI picked — now your pick"
        turn_color = CARD_ACCENT if user_first else "#00b4d8"
        st.markdown(
            f'<div style="background:{CARD_BG};border:1px solid {CARD_BORDER};'
            f'border-left:4px solid {turn_color};'
            f'border-radius:8px;padding:10px 18px;margin-bottom:14px;'
            f'display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap;">'
            f'<div><span style="color:{CARD_TEXT_MUTED};font-size:0.85em;">Round</span> '
            f'<b style="color:{CARD_ACCENT};font-size:1.2em;">{current_round + 1}</b>'
            f'<span style="color:{CARD_TEXT_MUTED};"> / {rounds}</span>'
            f'<span style="margin-left:14px;color:{turn_color};font-weight:700;">{turn_label}</span></div>'
            f'<div style="color:{CARD_TEXT_MUTED};font-size:0.85em;">'
            f'{strat_icon} <b style="color:{CARD_TEXT};">{strategy_name}</b></div>'
            f'<div style="color:{CARD_TEXT};font-size:0.9em;">'
            f'You: <b style="color:{CARD_ACCENT};">{len(st.session_state.draft_user_picks)}</b> &nbsp; · &nbsp; '
            f'AI: <b style="color:#00b4d8;">{len(st.session_state.draft_ai_picks)}</b></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # If AI just pre-picked, show a small notification
        if (not user_first) and st.session_state.draft_ai_picks:
            last_ai = st.session_state.draft_ai_picks[-1]
            st.markdown(
                f'<div style="background:rgba(0,180,216,0.08);border:1px solid #00b4d8;'
                f'border-radius:6px;padding:8px 14px;margin-bottom:12px;color:#9fd;">'
                f'🤖 AI just drafted <b>{last_ai.get("title","?")}</b> '
                f'({int(last_ai["release_year"]) if pd.notna(last_ai.get("release_year")) else "—"}) · '
                f'IMDb {float(last_ai["imdb_score"]):.1f}'
                f'</div>',
                unsafe_allow_html=True,
            )

        col_pool, col_score = st.columns([3, 2])

        with col_pool:
            st.markdown(section_header_html("Available Titles", "Top 30 by draft value — or search the full pool."), unsafe_allow_html=True)
            pool_search = st.text_input("Search pool...", key="draft_pool_search", label_visibility="collapsed", placeholder="Search titles…")
            display_pool = available.copy()
            if pool_search:
                display_pool = display_pool[display_pool["title"].str.contains(pool_search, case=False, na=False)]
            display_pool = display_pool.sort_values("draft_value", ascending=False).head(30)

            for idx, row in display_pool.iterrows():
                col_info, col_btn = st.columns([4, 1])
                with col_info:
                    imdb_s = f"IMDb {row['imdb_score']:.1f}" if pd.notna(row.get("imdb_score")) else ""
                    votes_s = format_votes(row.get("imdb_votes")) if pd.notna(row.get("imdb_votes")) else ""
                    year_s = str(int(row["release_year"])) if pd.notna(row.get("release_year")) else ""
                    plat = str(row.get("platform", ""))
                    badge = platform_badges_html(plat) if plat else ""
                    poster_url = row.get("poster_url", "")
                    plat_color = PLATFORMS.get(plat, {}).get("color", "#555")
                    initial = str(row.get("title", "?"))[0].upper()
                    if pd.notna(poster_url) and str(poster_url).startswith("http"):
                        thumb = f'<img src="{poster_url}" style="width:50px;height:72px;object-fit:cover;border-radius:3px;">'
                    else:
                        thumb = (
                            f'<div style="display:inline-block;width:50px;height:72px;background:{plat_color};'
                            f'border-radius:3px;text-align:center;line-height:72px;font-weight:700;'
                            f'color:#fff;font-size:1.2em;">{initial}</div>'
                        )
                    st.markdown(
                        f'<div style="display:flex;gap:10px;align-items:center;padding:4px 0;">'
                        f'{thumb}'
                        f'<div>'
                        f'<div style="font-weight:600;color:{CARD_TEXT};font-size:0.9em;">'
                        f'{row.get("title","?")} <span style="color:{CARD_TEXT_MUTED};font-weight:400;">({year_s})</span></div>'
                        f'<div style="font-size:0.78em;margin-top:2px;">{badge} '
                        f'<span style="color:{CARD_ACCENT};">{imdb_s}</span> '
                        f'<span style="color:{CARD_TEXT_MUTED};">· {votes_s}</span></div>'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )
                with col_btn:
                    if st.button("Draft", key=f"draft_pick_{row.get('id', idx)}", use_container_width=True):
                        pick = row.to_dict()
                        st.session_state.draft_user_picks.append(pick)

                        # If user was first, AI picks after. Otherwise AI already pre-picked.
                        if user_first:
                            remaining = available[available["id"] != pick.get("id")] if "id" in available.columns else available
                            ai_pick = _ai_pick(strategy_name, remaining, st.session_state.draft_user_picks)
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
                st.markdown(f'<div style="color:{CARD_ACCENT};font-weight:700;text-align:center;">You</div>', unsafe_allow_html=True)
                st.metric("Avg IMDb", f"{user_stats['avg_imdb']:.2f}")
                st.metric("Prestige", f"{user_stats['prestige']:.0f}%")
                st.metric("Diversity", f"{user_stats['diversity']:.2f}")
            with score_col_a:
                st.markdown(f'<div style="color:#00b4d8;font-weight:700;text-align:center;">AI</div>', unsafe_allow_html=True)
                st.metric("Avg IMDb", f"{ai_stats['avg_imdb']:.2f}")
                st.metric("Prestige", f"{ai_stats['prestige']:.0f}%")
                st.metric("Diversity", f"{ai_stats['diversity']:.2f}")

            with st.expander(f"AI's picks ({len(st.session_state.draft_ai_picks)})"):
                for p in st.session_state.draft_ai_picks:
                    t = p.get("title", "?")
                    yr = int(p["release_year"]) if pd.notna(p.get("release_year")) else "—"
                    im = f"{float(p['imdb_score']):.1f}" if pd.notna(p.get("imdb_score")) else "—"
                    st.markdown(
                        f'<div style="font-size:0.85em;color:{CARD_TEXT};padding:3px 0;">'
                        f'<b>{t}</b> <span style="color:{CARD_TEXT_MUTED};">({yr})</span> '
                        f'<span style="color:{CARD_ACCENT};">· {im}</span></div>',
                        unsafe_allow_html=True,
                    )

    # ── Results Screen ────────────────────────────────────────────────────────
    elif st.session_state.draft_complete:
        user_picks = st.session_state.draft_user_picks
        ai_picks = st.session_state.draft_ai_picks
        user_stats = _service_metrics(user_picks)
        ai_stats = _service_metrics(ai_picks)
        settings = st.session_state.draft_settings
        strategy_name = settings.get("strategy", "Balanced Strategist")
        strat_info = STRATEGIES.get(strategy_name, STRATEGIES["Balanced Strategist"])
        strat_icon = strat_info["icon"]
        focus_metrics = strat_info["focus"]

        # ── Strategy-aware winner logic ───────────────────────────────────────
        # Metrics compared on the final scoreboard
        all_metrics = ["avg_imdb", "pct_above_7", "prestige", "diversity", "freshness"]
        # Weight focus metrics 2x, others 1x
        weights = {m: (2.0 if m in focus_metrics else 1.0) for m in all_metrics}

        user_weighted = 0.0
        ai_weighted = 0.0
        user_metric_wins = 0
        ai_metric_wins = 0
        for m in all_metrics:
            u, a = user_stats[m], ai_stats[m]
            if u > a:
                user_weighted += weights[m]
                user_metric_wins += 1
            elif a > u:
                ai_weighted += weights[m]
                ai_metric_wins += 1

        user_won = user_weighted > ai_weighted
        ai_won = ai_weighted > user_weighted

        # ── Banner (with balloons on user win, matching Wordle pattern) ───────
        focus_labels = {
            "avg_imdb": "Avg IMDb", "pct_above_7": "% above 7.0",
            "prestige": "Prestige", "diversity": "Diversity",
            "freshness": "Freshness", "avg_votes": "Audience Reach",
        }
        focus_blurb = " · ".join(focus_labels.get(m, m) for m in focus_metrics)

        if user_won:
            if not st.session_state.get("draft_win_celebrated"):
                st.balloons()
                st.session_state.draft_win_celebrated = True
            st.markdown(
                f'<div style="background:linear-gradient(135deg,#0a2e18,#0d4022);'
                f'border:2px solid #2ecc71;border-top:4px solid #2ecc71;border-radius:14px;'
                f'padding:32px 24px;margin-bottom:20px;text-align:center;'
                f'box-shadow:0 0 40px rgba(46,204,113,0.25);">'
                f'<div style="color:#2ecc71;font-size:0.75rem;text-transform:uppercase;letter-spacing:3px;margin-bottom:10px;">Victory</div>'
                f'<div style="color:#ffffff;font-size:2rem;font-weight:900;letter-spacing:-0.5px;margin-bottom:6px;">'
                f'You beat {strat_icon} {strategy_name}</div>'
                f'<div style="color:#5dde8a;font-size:1rem;margin-bottom:18px;">'
                f'Outperformed on <b>{user_metric_wins}</b> of {len(all_metrics)} metrics · weighted score '
                f'<b>{user_weighted:.1f}</b>–<b>{ai_weighted:.1f}</b></div>'
                f'<div style="display:inline-flex;gap:24px;justify-content:center;flex-wrap:wrap;'
                f'background:rgba(46,204,113,0.08);border-radius:10px;padding:12px 24px;">'
                f'<span style="color:#aaa;font-size:0.9rem;">Focus: {focus_blurb}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
        elif ai_won:
            st.markdown(
                f'<div style="background:linear-gradient(135deg,#2d0000,#3a0a0a);'
                f'border:2px solid #e74c3c;border-top:4px solid #e74c3c;border-radius:14px;'
                f'padding:28px 24px;margin-bottom:20px;text-align:center;">'
                f'<div style="color:#e74c3c;font-size:0.75rem;text-transform:uppercase;letter-spacing:3px;margin-bottom:10px;">Defeat</div>'
                f'<div style="color:#ffffff;font-size:1.7rem;font-weight:900;margin-bottom:6px;">'
                f'{strat_icon} {strategy_name} wins</div>'
                f'<div style="color:#ff9a8a;font-size:0.95rem;">AI outperformed on '
                f'<b>{ai_metric_wins}</b> of {len(all_metrics)} metrics · weighted <b>{ai_weighted:.1f}</b>–<b>{user_weighted:.1f}</b></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="background:rgba(0,180,166,0.1);border:2px solid #00B4A6;'
                f'border-radius:14px;padding:24px;margin-bottom:20px;text-align:center;">'
                f'<div style="color:#00B4A6;font-size:0.75rem;text-transform:uppercase;letter-spacing:3px;margin-bottom:8px;">Draw</div>'
                f'<div style="color:#fff;font-size:1.6rem;font-weight:800;">Dead heat vs {strat_icon} {strategy_name}</div>'
                f'<div style="color:#aaa;margin-top:6px;">Weighted score {user_weighted:.1f}–{ai_weighted:.1f}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Strategy Scorecard — focus metrics highlighted ────────────────────
        st.markdown(
            section_header_html(
                f"{strat_icon} {strategy_name} Scorecard",
                f"Focus metrics (weighted 2×): {focus_blurb}",
            ),
            unsafe_allow_html=True,
        )

        metric_labels = {
            "avg_imdb": ("Avg IMDb", "{:.2f}"),
            "pct_above_7": ("% above 7.0", "{:.1f}%"),
            "prestige": ("Prestige (award winners)", "{:.1f}%"),
            "diversity": ("Genre Diversity", "{:.2f}"),
            "freshness": ("Freshness (post-2015)", "{:.1f}%"),
        }
        for mk, (mlabel, mfmt) in metric_labels.items():
            u_val = user_stats[mk]
            a_val = ai_stats[mk]
            u_wins_this = u_val > a_val
            a_wins_this = a_val > u_val
            is_focus = mk in focus_metrics
            u_color = "#2ecc71" if u_wins_this else CARD_TEXT
            a_color = "#00b4d8" if a_wins_this else CARD_TEXT
            u_check = " ✓" if u_wins_this else ""
            a_check = " ✓" if a_wins_this else ""
            label_prefix = "★ " if is_focus else ""
            label_color = CARD_ACCENT if is_focus else CARD_TEXT_MUTED
            row_bg = "rgba(255,215,0,0.04)" if is_focus else "transparent"
            st.markdown(
                f'<div style="display:flex;align-items:center;padding:8px 12px;background:{row_bg};'
                f'border-bottom:1px solid {CARD_BORDER};border-radius:4px;">'
                f'<span style="flex:1;text-align:right;color:{u_color};font-weight:{"700" if u_wins_this else "500"};">'
                f'{mfmt.format(u_val)}{u_check}</span>'
                f'<span style="flex:1.8;text-align:center;color:{label_color};font-size:0.88em;padding:0 12px;font-weight:{"600" if is_focus else "400"};">'
                f'{label_prefix}{mlabel}</span>'
                f'<span style="flex:1;text-align:left;color:{a_color};font-weight:{"700" if a_wins_this else "500"};">'
                f'{mfmt.format(a_val)}{a_check}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            f'<div style="display:flex;justify-content:space-around;padding:8px 12px;'
            f'color:{CARD_TEXT_MUTED};font-size:0.78em;margin-top:4px;">'
            f'<span>You</span><span>AI ({strategy_name})</span></div>',
            unsafe_allow_html=True,
        )

        # ── Radar chart ───────────────────────────────────────────────────────
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
            name=f"AI · {strategy_name}",
            line_color="#00b4d8",
            fillcolor=f"rgba(0,180,216,0.2)",
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            template=PLOTLY_TEMPLATE,
            height=360,
            margin=dict(l=40, r=40, t=30, b=30),
            legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # ── Section 3: Top 5 highlights ───────────────────────────────────────
        st.markdown(section_header_html("Top Picks Head-to-Head", "Each side's 5 highest-rated picks."), unsafe_allow_html=True)
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

        _top5_cards(user_picks, col_uh, "Your Top 5")
        _top5_cards(ai_picks, col_ah, f"{strat_icon} {strategy_name}'s Top 5")

        # ── Action buttons ────────────────────────────────────────────────────
        col_again, col_export = st.columns(2)
        with col_again:
            if st.button("🔄 Draft Again", use_container_width=True, type="primary"):
                st.session_state.draft_active = False
                st.session_state.draft_complete = False
                st.session_state.draft_round = 0
                st.session_state.draft_user_picks = []
                st.session_state.draft_ai_picks = []
                st.session_state.draft_available_pool = None
                st.session_state.draft_settings = {}
                st.session_state.draft_ai_prepicked_round = -1
                st.session_state.draft_win_celebrated = False
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
# TAB 4: CINEMA GUESS
# =============================================================================
with tab4:
    st.markdown(
        section_header_html(
            "CinemaGuess",
            "Guess the movie or show from a still frame — 6 chances.",
        ),
        unsafe_allow_html=True,
    )

    _game_df = _load_game_catalog()

    if _game_df is None or _game_df.empty:
        st.error(
            "Game catalog not found. Run `python scripts/13_enrich_tmdb_game.py` to generate it."
        )
    else:
        # ── Game state helpers ────────────────────────────────────────────────

        def _cg_pick_daily(df):
            h = int(hashlib.md5(date.today().isoformat().encode()).hexdigest(), 16)
            return df.iloc[h % len(df)].to_dict()

        def _cg_pick_random(df):
            return df.sample(1).iloc[0].to_dict()

        def _cg_init_game(mode="random"):
            answer = _cg_pick_daily(_game_df) if mode == "daily" else _cg_pick_random(_game_df)
            st.session_state.cg = {
                "answer": answer,
                "guesses": [],
                "game_over": False,
                "won": False,
                "mode": mode,
            }

        if st.session_state.cg is None:
            _cg_init_game("random")

        cg = st.session_state.cg
        answer = cg["answer"]
        guesses = cg["guesses"]
        num_guesses = len(guesses)
        clues_revealed = num_guesses

        # ── Mode controls ─────────────────────────────────────────────────────

        col_mode, col_new = st.columns([3, 1])
        with col_mode:
            mode_choice = st.radio(
                "Mode",
                ["Random", "Daily Challenge"],
                horizontal=True,
                key="cg_mode_radio",
                help="Daily Challenge: same title for everyone today.",
            )
        with col_new:
            st.markdown("&nbsp;", unsafe_allow_html=True)
            if st.button("New Game", type="primary", use_container_width=True, key="cg_new_game"):
                _cg_init_game("daily" if mode_choice == "Daily Challenge" else "random")
                st.rerun()

        # ── Visual assets ─────────────────────────────────────────────────────

        def _cg_blur_px():
            if cg["game_over"]:
                return 0
            return _CG_BLUR_SCHEDULE[min(num_guesses, len(_CG_BLUR_SCHEDULE) - 1)]

        def _cg_blurred_img_html(url, height=None):
            blur = _cg_blur_px()
            style = (
                f"filter: blur({blur}px); transition: filter 0.4s ease; "
                f"width:100%; border-radius:6px; display:block;"
            )
            if height:
                style += f" height:{height}px; object-fit:cover;"
            return f'<img src="{url}" style="{style}">'

        has_backdrop = bool(answer.get("backdrop_url"))

        if num_guesses == 0 and not cg["game_over"]:
            st.caption(
                f"Image starts blurred — unblurs a little with each guess. "
                f"Current blur: {_cg_blur_px()}px"
            )

        if has_backdrop:
            st.markdown(_cg_blurred_img_html(answer["backdrop_url"]), unsafe_allow_html=True)
        else:
            st.info("No visual assets available for this title — use the clues below.")

        # ── Clue cards ────────────────────────────────────────────────────────

        if clues_revealed > 0:
            st.markdown(
                section_header_html("Clues", "Revealed after each wrong guess."),
                unsafe_allow_html=True,
            )
            n_clues = min(clues_revealed, len(_CG_CLUE_DEFS))
            clue_cols = st.columns(n_clues)
            for i in range(n_clues):
                label, fn = _CG_CLUE_DEFS[i]
                try:
                    val = fn(answer)
                except Exception:
                    val = "Unknown"
                with clue_cols[i]:
                    st.markdown(
                        f"""
                        <div style="background:{CARD_BG};border:1px solid {CARD_BORDER};
                                    border-top:3px solid {CARD_ACCENT};border-radius:6px;
                                    padding:10px 12px;text-align:center;">
                            <div style="color:#888;font-size:0.72rem;margin-bottom:4px;">{label}</div>
                            <div style="color:{CARD_TEXT};font-weight:600;">{val}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # ── Previous guesses ──────────────────────────────────────────────────

        if guesses:
            st.markdown(section_header_html("Your Guesses"), unsafe_allow_html=True)
            for g in guesses:
                correct = g.strip().lower() == answer["title"].strip().lower()
                bg = "#1a5c38" if correct else "#5c1a1a"
                icon = "✓" if correct else "✗"
                st.markdown(
                    f'<div style="background:{bg};border-radius:4px;padding:7px 14px;'
                    f'margin-bottom:5px;color:#fff;">{icon} {g}</div>',
                    unsafe_allow_html=True,
                )

        # ── Game over banner ──────────────────────────────────────────────────

        def _cg_result_card():
            year = answer.get("release_year")
            year_str = f" ({int(year)})" if pd.notna(year) else ""
            score = answer.get("imdb_score")
            genres = answer.get("genres", [])
            genre_str = (
                ", ".join(g.capitalize() for g in list(genres)[:3])
                if genres is not None and len(genres) > 0
                else "N/A"
            )

            if cg["won"]:
                st.success(f"Correct in {num_guesses} guess(es)!  **{answer['title']}**{year_str}")
            else:
                st.error(f"The answer was  **{answer['title']}**{year_str}")

            info_cols = st.columns(4)
            with info_cols[0]:
                st.metric("IMDb", f"{score:.1f}" if pd.notna(score) else "N/A")
            with info_cols[1]:
                st.metric("Type", answer.get("type", "N/A"))
            with info_cols[2]:
                st.metric("Genre(s)", genre_str if genre_str else "N/A")
            with info_cols[3]:
                st.metric("Year", int(year) if pd.notna(year) else "N/A")

            if has_backdrop:
                st.markdown(_cg_blurred_img_html(answer["backdrop_url"]), unsafe_allow_html=True)

            if st.button("Play Again", type="primary", key="cg_play_again"):
                _cg_init_game("daily" if mode_choice == "Daily Challenge" else "random")
                st.rerun()

        if cg["game_over"]:
            _cg_result_card()

        # ── Guess input ───────────────────────────────────────────────────────

        elif not cg["game_over"]:
            st.divider()
            guesses_left = _CG_MAX_GUESSES - num_guesses
            st.markdown(
                f"**Guess {num_guesses + 1} of {_CG_MAX_GUESSES}** &nbsp;—&nbsp; "
                f"{guesses_left} guess(es) remaining",
                unsafe_allow_html=True,
            )

            _pool = _cg_filtered_pool(_game_df, answer, clues_revealed)
            all_titles = [""] + sorted(_pool["title"].dropna().unique().tolist())
            if clues_revealed > 0:
                st.caption(f"Pool narrowed to {len(all_titles) - 1} title(s) matching revealed clues.")

            guess_val = st.selectbox(
                "Select a title",
                options=all_titles,
                key=f"cg_guess_{num_guesses}",
                label_visibility="collapsed",
            )

            col_submit, col_skip = st.columns([2, 1])
            with col_submit:
                if st.button(
                    "Submit Guess",
                    disabled=not guess_val,
                    type="primary",
                    use_container_width=True,
                    key="cg_submit",
                ):
                    if len(cg["guesses"]) == num_guesses:
                        cg["guesses"].append(guess_val)
                        if guess_val.strip().lower() == answer["title"].strip().lower():
                            cg["game_over"] = True
                            cg["won"] = True
                        elif len(cg["guesses"]) >= _CG_MAX_GUESSES:
                            cg["game_over"] = True
                            cg["won"] = False
                    st.rerun()

            with col_skip:
                if st.button("Skip (count as wrong)", use_container_width=True, key="cg_skip"):
                    if len(cg["guesses"]) == num_guesses:
                        cg["guesses"].append("— skipped —")
                        if len(cg["guesses"]) >= _CG_MAX_GUESSES:
                            cg["game_over"] = True
                            cg["won"] = False
                    st.rerun()

        # ── Progress bar ──────────────────────────────────────────────────────

        st.markdown("&nbsp;", unsafe_allow_html=True)
        progress_pct = num_guesses / _CG_MAX_GUESSES
        bar_color = CARD_ACCENT if not cg["game_over"] else ("#1a5c38" if cg["won"] else "#8b0000")
        st.markdown(
            f"""
            <div style="background:#333;border-radius:4px;height:6px;margin-top:4px;">
                <div style="background:{bar_color};width:{progress_pct*100:.0f}%;height:6px;border-radius:4px;"></div>
            </div>
            <div style="color:#888;font-size:0.75rem;margin-top:4px;">
                {num_guesses} / {_CG_MAX_GUESSES} guesses used
            </div>
            """,
            unsafe_allow_html=True,
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
