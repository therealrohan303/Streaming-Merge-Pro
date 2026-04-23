"""Analysis functions for the Interactive Lab page (Page 6).

Features:
  1. Build Your Streaming Service (budget game)
  2. Greenlight Studio (prediction + similar titles + platform fit + talent)
  3. Insight Generator
"""

import random
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.analysis.scoring import bayesian_imdb, compute_quality_score
from src.config import ALL_PLATFORMS, MERGED_PLATFORMS, PLATFORMS
from src.ui.formatting import genre_display, genre_list_display

_ADAPTATION_TOKENS = (
    "based on novel", "based on book", "based on comic",
    "based on true story", "based on play", "based on video game",
    "remake", "reboot",
)


# =============================================================================
# Greenlight Studio — shared constants
# =============================================================================
# Mirror of the training script so UI callers don't need to import from scripts/.
# The fitted model stores `tone_keys_` and `genre_pairs_` which are the source
# of truth at inference time; these lists are only used for UI presentation.

GREENLIGHT_TONE_LABELS: dict[str, str] = {
    "prestige":   "Prestige drama",
    "gritty":     "Dark & gritty",
    "character":  "Character study",
    "action":     "High-octane",
    "mystery":    "Mystery-thriller",
    "feel_good":  "Feel-good",
    "romantic":   "Romantic",
    "tearjerker": "Tearjerker",
    "quirky":     "Quirky / offbeat",
    "epic":       "Epic / sweeping",
    "satirical":  "Satirical",
    "family":     "Family-friendly",
}

GREENLIGHT_TONE_VOCAB: dict[str, list[str]] = {
    "prestige":   ["award", "oscar", "acclaimed", "prestige", "critically", "masterpiece", "emmy", "biopic"],
    "gritty":     ["gritty", "dark", "brutal", "violent", "raw", "grim", "noir", "bleak", "visceral"],
    "character":  ["character", "introspective", "portrait", "study", "intimate", "internal", "meditative"],
    "action":     ["action", "explosive", "fight", "chase", "combat", "high-octane", "relentless", "shootout"],
    "mystery":    ["mystery", "thriller", "suspense", "investigation", "detective", "whodunit", "murder"],
    "feel_good":  ["feel-good", "heartwarming", "uplifting", "charming", "wholesome", "sweet", "joyful"],
    "romantic":   ["romance", "romantic", "love", "lovers", "relationship", "passion", "courtship"],
    "tearjerker": ["tragic", "heartbreak", "tearjerker", "grief", "loss", "devastating", "mourning"],
    "quirky":     ["quirky", "offbeat", "eccentric", "indie", "weird", "whimsical", "surreal", "absurd"],
    "epic":       ["epic", "sweeping", "grand", "sprawling", "historical", "vast", "saga"],
    "satirical":  ["satire", "satirical", "biting", "ironic", "political", "sharp-tongued", "dark comedy"],
    "family":     ["family", "kids", "child", "animated", "heartfelt", "cartoon", "coming-of-age"],
}

GREENLIGHT_GENRE_PAIRS: list[tuple[str, str]] = [
    ("drama", "thriller"),
    ("drama", "romance"),
    ("action", "scifi"),
    ("action", "thriller"),
    ("comedy", "romance"),
    ("crime", "drama"),
    ("horror", "thriller"),
    ("animation", "family"),
]


# =============================================================================
# Feature 1: Build Your Streaming Service
# =============================================================================

def compute_title_value(row, use_box_office=False):
    """Compute a title's 'cost' for the drafting game.

    Value formula: f(IMDb score, TMDB popularity, recency, franchise bonus)
    Optionally incorporates box office tier.
    """
    base_value = 100  # Base cost in millions

    # IMDb score factor (0.5x to 2.0x)
    imdb = row.get("imdb_score", 6.0) or 6.0
    imdb_factor = max(0.5, min(2.0, (imdb - 4.0) / 4.0))

    # Popularity factor (0.3x to 2.5x)
    pop = row.get("tmdb_popularity", 10.0) or 10.0
    pop_factor = max(0.3, min(2.5, pop / 30.0))

    # Recency factor (newer = more expensive)
    year = row.get("release_year", 2000) or 2000
    recency = max(0.5, min(1.5, (year - 1990) / 30.0))

    # Franchise bonus (20% premium)
    franchise_bonus = 1.2 if pd.notna(row.get("collection_name")) else 1.0

    value = base_value * imdb_factor * pop_factor * recency * franchise_bonus

    # Box office tier multiplier
    if use_box_office and pd.notna(row.get("box_office_usd")):
        bo = row["box_office_usd"]
        if bo > 500e6:
            value *= 1.5
        elif bo > 200e6:
            value *= 1.3
        elif bo > 50e6:
            value *= 1.1

    return round(value, 1)


def compute_service_stats(drafted_titles):
    """Compute live dashboard stats for drafted service."""
    if not drafted_titles:
        return {"count": 0, "avg_imdb": 0, "genres": {}, "diversity": 0, "spend": 0}

    df = pd.DataFrame(drafted_titles)
    genre_counts = {}
    for genres in df.get("genres", pd.Series()):
        if isinstance(genres, (list, np.ndarray)):
            for g in genres:
                genre_counts[g] = genre_counts.get(g, 0) + 1

    # Genre diversity (Shannon entropy)
    total = sum(genre_counts.values()) or 1
    probs = [c / total for c in genre_counts.values()]
    diversity = -sum(p * np.log2(p + 1e-9) for p in probs) if probs else 0

    return {
        "count": len(drafted_titles),
        "avg_imdb": df["imdb_score"].mean() if "imdb_score" in df.columns else 0,
        "genres": genre_counts,
        "diversity": round(diversity, 2),
        "spend": df["value"].sum() if "value" in df.columns else 0,
    }


def compare_services(drafted_stats, merged_df):
    """Compare drafted service vs Netflix+Max."""
    merged = merged_df.drop_duplicates(subset="id", keep="first")
    merged_stats = {
        "count": len(merged),
        "avg_imdb": merged["imdb_score"].mean(),
    }

    # Genre diversity of merged
    merged_genres = {}
    for genres in merged["genres"]:
        if isinstance(genres, (list, np.ndarray)):
            for g in genres:
                merged_genres[g] = merged_genres.get(g, 0) + 1
    total = sum(merged_genres.values()) or 1
    probs = [c / total for c in merged_genres.values()]
    merged_stats["diversity"] = round(-sum(p * np.log2(p + 1e-9) for p in probs), 2)

    return {
        "drafted": drafted_stats,
        "merged": merged_stats,
    }


# =============================================================================
# Feature 2: Greenlight Model helpers
# =============================================================================

def _tone_binary_vec(
    description: str,
    tone_selection: list[str] | None,
    tone_keys: list[str],
) -> dict[str, int]:
    """Build tone_* feature values from description text + explicit tone selection.

    A tone is activated if the user picked it OR any of its keywords appears in
    the description. This is the same rule used at training time on catalog
    descriptions + TMDB keywords.
    """
    text = (description or "").lower()
    picked = set(tone_selection or [])
    out: dict[str, int] = {}
    for key in tone_keys:
        active = key in picked
        if not active:
            for kw in GREENLIGHT_TONE_VOCAB.get(key, []):
                if kw in text:
                    active = True
                    break
        out[f"tone_{key}"] = int(active)
    return out


def _inference_features_row(model, features_dict: dict, description: str, tone_selection: list[str] | None) -> pd.DataFrame:
    """Assemble a single-row feature DataFrame matching model.feature_names_.

    Handles both the legacy GradientBoostingRegressor (flat features) and the
    new GreenlightStackedModel (structured + SVD + tones).
    """
    feature_names = list(model.feature_names_)
    row = {f: 0 for f in feature_names}

    # Genre one-hots
    genres_sel = features_dict.get("genres", []) or []
    for genre in genres_sel:
        col = f"genre_{genre}"
        if col in row:
            row[col] = 1

    # Pairwise genre interactions
    genre_set = set(genres_sel)
    for a, b in GREENLIGHT_GENRE_PAIRS:
        col = f"gpair_{a}_x_{b}"
        if col in row:
            row[col] = int((a in genre_set) and (b in genre_set))

    # Direct numeric passthroughs
    for key, val in features_dict.items():
        if key in ("genres",):
            continue
        if key in row:
            row[key] = val

    # Derived numerics
    if "runtime_sq" in row:
        runtime = features_dict.get("runtime", 0) or 0
        row["runtime_sq"] = float(runtime) ** 2
    if "cert_x_budget" in row:
        row["cert_x_budget"] = int(features_dict.get("age_cert_tier", 0) or 0) * int(features_dict.get("budget_tier", 0) or 0)
    if "budget_log" in row:
        bu = features_dict.get("budget_usd")
        if bu is None:
            bt = int(features_dict.get("budget_tier", 0) or 0)
            bu = {1: 10e6, 2: 45e6, 3: 120e6, 4: 280e6}.get(bt, 0)
        row["budget_log"] = float(np.log1p(max(bu, 0)))

    # Franchise one-hots (movies only)
    ftype = features_dict.get("franchise_type")
    if ftype is not None:
        for i, tag in enumerate(("original", "franchise", "adaptation")):
            col = f"ftype_{tag}"
            if col in row:
                row[col] = int(ftype == i)

    # Tone one-hots
    tone_keys = list(getattr(model, "tone_keys_", []) or [])
    if tone_keys:
        row.update(_tone_binary_vec(description, tone_selection, tone_keys))

    # SVD components from description
    svd_cols = list(getattr(model, "svd_cols_", []) or [])
    if svd_cols and getattr(model, "svd_", None) is not None and getattr(model, "vectorizer_", None) is not None:
        vec = model.vectorizer_.transform([description or ""])
        svd_vec = model.svd_.transform(vec)[0]
        for i, col in enumerate(svd_cols):
            if col in row:
                row[col] = float(svd_vec[i])

    return pd.DataFrame([row], columns=feature_names)


def predict_title(
    model,
    features_dict: dict,
    genre_names: list[str] | None = None,
    *,
    description: str = "",
    tone_selection: list[str] | None = None,
) -> dict | None:
    """Run the greenlight predictor on a user concept.

    Accepts both the new stacked model (description + tone + SVD) and the
    legacy GradientBoostingRegressor (structured features only).
    """
    if model is None:
        return None

    X = _inference_features_row(model, features_dict, description, tone_selection)
    prediction = float(np.asarray(model.predict(X)).ravel()[0])

    # Peer-relative tier using genre_percentiles_ if available
    percentiles = getattr(model, "genre_percentiles_", {}) or {}
    primary_genre = (features_dict.get("genres") or [None])[0]
    peer = percentiles.get(primary_genre) or percentiles.get("__global__")
    if peer:
        if prediction >= peer.get("p90", 7.5):
            tier = "top_10"
        elif prediction >= peer.get("p75", 7.0):
            tier = "upper_quartile"
        elif prediction >= peer.get("p50", 6.5):
            tier = "above_median"
        elif prediction >= peer.get("p25", 5.5):
            tier = "below_median"
        else:
            tier = "bottom_quartile"
    else:
        tier = "above_median" if prediction >= 6.5 else "below_median"

    # Feature importances (only available on the GBM half of the stack)
    gbm = getattr(model, "gbm_", None)
    if gbm is not None and hasattr(gbm, "feature_importances_"):
        imps = pd.Series(gbm.feature_importances_, index=model.feature_names_)
        importances = imps[imps > 0.005].sort_values(ascending=False)
    elif hasattr(model, "feature_importances_"):
        imps = pd.Series(model.feature_importances_, index=model.feature_names_)
        importances = imps[imps > 0.005].sort_values(ascending=False)
    else:
        importances = pd.Series(dtype=float)

    return {
        "prediction": round(prediction, 2),
        "tier": tier,
        "peer": peer,
        "primary_genre": primary_genre,
        "importances": importances,
        "cv_rmse":       getattr(model, "cv_rmse_", None),
        "baseline_rmse": getattr(model, "baseline_rmse_", None),
        "global_mean":   getattr(model, "global_mean_", None),
        "training_size": getattr(model, "training_size_", None),
    }


# =============================================================================
# Greenlight Studio — Similar Titles
# =============================================================================
def greenlight_similar_titles(
    description: str,
    features_dict: dict,
    tone_selection: list[str] | None,
    enriched: pd.DataFrame,
    vectorizer,
    tfidf_matrix,
    id_list: list,
    content_type: str,
    top_k: int = 6,
) -> pd.DataFrame:
    """Rank catalog titles as comps for the user's concept.

    Pipeline:
      1. Hard filters: `type` matches, genre overlap ≥ 1, `imdb_votes` ≥ 500,
         runtime within ±40 min (movies) / ±20 min (shows), budget tier within
         ±1 (movies, when `budget_usd` is known for the candidate).
      2. Composite score:
           0.55 * tfidf_cos + 0.25 * genre_overlap_frac
         + 0.10 * quality_norm + 0.10 * tone_match
      3. Soft boost (+0.05) when franchise_type matches user selection.
    """
    if enriched is None or enriched.empty or not description or not description.strip():
        return pd.DataFrame()

    sel_genres = set(features_dict.get("genres") or [])
    if not sel_genres:
        return pd.DataFrame()

    df = enriched.copy()

    # Hard filters
    df = df[df.get("type", pd.Series([None])) == content_type]
    votes = pd.to_numeric(df.get("imdb_votes", pd.Series(dtype=float)), errors="coerce").fillna(0)
    df = df[votes >= 500]

    runtime_req = features_dict.get("runtime")
    if runtime_req and "runtime" in df.columns:
        rt = pd.to_numeric(df["runtime"], errors="coerce")
        band = 40 if content_type == "Movie" else 20
        df = df[(rt.isna()) | ((rt >= runtime_req - band) & (rt <= runtime_req + band))]

    if content_type == "Movie" and "budget_usd" in df.columns and features_dict.get("budget_tier"):
        user_tier = int(features_dict.get("budget_tier") or 0)
        def _bt(v):
            if pd.isna(v) or v <= 0:
                return 0
            v = float(v)
            if v < 20_000_000: return 1
            if v < 80_000_000: return 2
            if v < 200_000_000: return 3
            return 4
        cand_tier = df["budget_usd"].apply(_bt)
        df = df[(cand_tier == 0) | ((cand_tier >= user_tier - 1) & (cand_tier <= user_tier + 1))]

    df = df[df["genres"].apply(lambda g: isinstance(g, (list, np.ndarray)) and bool(set(g) & sel_genres))]
    if df.empty:
        return df

    # TF-IDF similarity — align the candidate subset to the precomputed matrix
    id_to_row = {tid: i for i, tid in enumerate(id_list)}
    rows = df["id"].map(id_to_row)
    df = df[rows.notna()].copy()
    if df.empty:
        return df
    row_idx = rows.loc[df.index].astype(int).values

    q_vec = vectorizer.transform([description])
    cand_mat = tfidf_matrix[row_idx]
    tfidf_cos = cosine_similarity(q_vec, cand_mat).ravel()

    # Genre overlap fraction
    n_sel = max(1, len(sel_genres))
    genre_frac = df["genres"].apply(
        lambda g: len(set(g) & sel_genres) / n_sel if isinstance(g, (list, np.ndarray)) else 0.0
    ).astype(float).values

    # Quality norm (Bayesian IMDb, mapped to [0,1] over 5–9)
    q_raw = np.asarray(bayesian_imdb(df["imdb_score"], df["imdb_votes"]).values, dtype=float)
    quality_norm = np.clip((q_raw - 5.0) / 4.0, 0.0, 1.0)

    # Tone match — overlap between selected tone keyword clusters and
    # description + top_tags + tmdb_keywords for the candidate.
    tone_sel = list(tone_selection or [])
    if tone_sel:
        tone_kws = {k for key in tone_sel for k in GREENLIGHT_TONE_VOCAB.get(key, [])}
        def _tone_hit(row):
            bag = str(row.get("description") or "").lower()
            for col in ("top_tags", "tmdb_keywords"):
                v = row.get(col)
                if isinstance(v, (list, np.ndarray)):
                    bag += " " + " ".join(str(x).lower() for x in v)
            hits = sum(1 for k in tone_kws if k in bag)
            return min(1.0, hits / max(1, len(tone_kws)) * 3.0)  # amplify hits
        tone_match = df.apply(_tone_hit, axis=1).values
    else:
        tone_match = np.zeros(len(df))

    # Franchise soft boost
    ftype = features_dict.get("franchise_type")
    if ftype is not None and "collection_name" in df.columns:
        cand_is_franchise = df["collection_name"].notna().astype(int).values
        if ftype == 1:      # user wants franchise entry
            fboost = cand_is_franchise * 0.05
        elif ftype == 0:    # user wants original
            fboost = (1 - cand_is_franchise) * 0.05
        else:
            fboost = np.zeros(len(df))
    else:
        fboost = np.zeros(len(df))

    composite = (
        0.55 * tfidf_cos
        + 0.25 * genre_frac
        + 0.10 * quality_norm
        + 0.10 * tone_match
        + fboost
    )

    df = df.assign(
        _tfidf=tfidf_cos,
        _genre_frac=genre_frac,
        _quality=quality_norm,
        _tone=tone_match,
        _fboost=fboost,
        _score=composite,
    ).sort_values("_score", ascending=False)

    return df.head(top_k).reset_index(drop=True)


# =============================================================================
# Greenlight Studio — Platform Fit
# =============================================================================
def greenlight_platform_fit(
    prediction: float,
    features_dict: dict,
    content_type: str,
    prestige_index: pd.DataFrame,
    enriched: pd.DataFrame,
) -> list[dict]:
    """Compute a 0–100 Fit Score per platform, grounded in real catalog data.

    Components: prestige alignment (40), quality alignment (25), type fit (20),
    budget fit (15 movies / redistributed 0 for shows).
    """
    sel_genres = list(features_dict.get("genres") or [])
    if not sel_genres or prestige_index is None or prestige_index.empty:
        return []

    # Lookup: {(platform, genre) -> prestige_per_1k}
    pi = prestige_index.set_index(["platform", "genre"])
    # Max prestige per genre across all platforms (normalizer)
    genre_max = prestige_index.groupby("genre")["prestige_per_1k"].max().to_dict()

    # Per-platform genre mean IMDb + movie ratio + budget distribution
    def _platform_slice(plat: str) -> pd.DataFrame:
        if "platform" in enriched.columns:
            return enriched[enriched["platform"] == plat]
        if "platforms" in enriched.columns:
            return enriched[enriched["platforms"].apply(
                lambda ps: isinstance(ps, (list, np.ndarray)) and plat in ps
            )]
        return enriched.iloc[0:0]

    def _bt(v):
        if pd.isna(v) or v <= 0: return 0
        v = float(v)
        if v < 20_000_000: return 1
        if v < 80_000_000: return 2
        if v < 200_000_000: return 3
        return 4

    user_budget_tier = int(features_dict.get("budget_tier") or 0) if content_type == "Movie" else 0

    out: list[dict] = []
    for plat in ALL_PLATFORMS:
        sub = _platform_slice(plat)
        if len(sub) == 0:
            continue

        # Prestige alignment across selected genres
        scores = []
        for g in sel_genres:
            try:
                val = float(pi.loc[(plat, g), "prestige_per_1k"])
            except (KeyError, TypeError):
                val = 0.0
            norm = (val / genre_max[g]) if genre_max.get(g, 0) > 0 else 0.0
            scores.append(norm)
        prestige_alignment = float(np.mean(scores)) if scores else 0.0

        # Quality alignment — average per-genre mean IMDb on the platform
        genre_mask = sub["genres"].apply(
            lambda gl: isinstance(gl, (list, np.ndarray)) and bool(set(gl) & set(sel_genres))
        )
        platform_genre = sub[genre_mask]
        if len(platform_genre) >= 10:
            pg_mean = pd.to_numeric(platform_genre["imdb_score"], errors="coerce").dropna().mean()
        else:
            pg_mean = pd.to_numeric(sub["imdb_score"], errors="coerce").dropna().mean()
        if pd.isna(pg_mean):
            pg_mean = 6.5
        quality_alignment = float(np.exp(-abs(prediction - pg_mean) / 1.0))

        # Type fit
        movie_ratio = float((sub["type"] == "Movie").mean()) if "type" in sub.columns else 0.5
        if content_type == "Movie":
            type_fit = min(1.0, movie_ratio / 0.55)
        else:
            type_fit = min(1.0, (1 - movie_ratio) / 0.55)

        # Budget fit (movies only)
        if content_type == "Movie" and user_budget_tier > 0 and "budget_usd" in sub.columns:
            movies_sub = sub[sub["type"] == "Movie"]
            if len(movies_sub) >= 20:
                tiers = movies_sub["budget_usd"].apply(_bt)
                tier_counts = tiers.value_counts(normalize=True).to_dict()
                # share in tier ±1
                near = sum(tier_counts.get(t, 0) for t in (user_budget_tier - 1, user_budget_tier, user_budget_tier + 1))
                budget_fit = float(min(1.0, near / 0.5))
            else:
                budget_fit = 0.4
        else:
            budget_fit = None

        if content_type == "Movie":
            fit = 0.40 * prestige_alignment + 0.25 * quality_alignment + 0.20 * type_fit + 0.15 * (budget_fit or 0)
        else:
            fit = 0.45 * prestige_alignment + 0.30 * quality_alignment + 0.25 * type_fit

        # Reason fields — averaged across all selected genres, not just one
        multiples = []
        for g in sel_genres:
            try:
                plat_val = float(pi.loc[(plat, g), "prestige_per_1k"])
            except (KeyError, TypeError):
                plat_val = 0.0
            avg_val = float(prestige_index[prestige_index["genre"] == g]["prestige_per_1k"].mean())
            if avg_val > 0:
                multiples.append(plat_val / avg_val)
        avg_multiple = float(np.mean(multiples)) if multiples else 0.0

        top_genre_for_plat = max(
            sel_genres,
            key=lambda g: (pi.loc[(plat, g), "prestige_per_1k"] if (plat, g) in pi.index else 0.0),
        )

        # Industry-average IMDb for the selected genres (for "above the norm" copy)
        industry_mean = float(
            pd.to_numeric(
                enriched[enriched["genres"].apply(
                    lambda gl: isinstance(gl, (list, np.ndarray)) and bool(set(gl) & set(sel_genres))
                )]["imdb_score"],
                errors="coerce",
            ).dropna().mean()
        )
        if pd.isna(industry_mean):
            industry_mean = 6.5

        out.append({
            "platform": plat,
            "fit": int(round(100 * fit)),
            "prestige_alignment": round(prestige_alignment, 3),
            "quality_alignment":  round(quality_alignment, 3),
            "type_fit":           round(type_fit, 3),
            "budget_fit":         None if budget_fit is None else round(budget_fit, 3),
            "platform_mean_imdb": round(float(pg_mean), 2),
            "industry_mean_imdb": round(industry_mean, 2),
            "top_genre":          top_genre_for_plat,
            "selected_genres":    list(sel_genres),
            "prestige_multiple":  round(avg_multiple, 2) if avg_multiple else None,
        })

    out.sort(key=lambda d: d["fit"], reverse=True)
    return out


# =============================================================================
# Greenlight Studio — Talent Picks
# =============================================================================
def _build_person_top_title_map(
    principals: pd.DataFrame,
    enriched: pd.DataFrame,
    credits: pd.DataFrame | None = None,
) -> dict[str, dict]:
    """For each person_id return their strongest title by Bayesian-shrunk IMDb score.

    ``imdb_principals`` only contains writer/producer/director/composer/
    cinematographer — actors come in via ``all_platforms_credits`` (keyed on
    ``title_id``). Passing both gives us a map that covers every talent role.
    Uses the same Bayesian shrinkage as the Home page so a single high-vote
    classic ranks above noisy low-vote titles.
    """
    if enriched is None or enriched.empty:
        return {}

    from src.analysis.scoring import bayesian_imdb

    tcols = [c for c in ("id", "imdb_id", "title", "release_year", "imdb_score", "imdb_votes")
             if c in enriched.columns]
    titles = enriched[tcols].drop_duplicates(subset=[c for c in ("id",) if c in tcols]).copy()
    if {"imdb_score", "imdb_votes"}.issubset(titles.columns):
        titles["_bscore"] = bayesian_imdb(titles["imdb_score"], titles["imdb_votes"])
    else:
        titles["_bscore"] = titles.get("imdb_score", 0)

    frames: list[pd.DataFrame] = []
    if principals is not None and not principals.empty and "imdb_id" in titles.columns:
        frames.append(principals.merge(
            titles.dropna(subset=["imdb_id"]).drop_duplicates("imdb_id"),
            on="imdb_id", how="inner",
        ))
    if credits is not None and not credits.empty and "id" in titles.columns:
        cj = credits[["person_id", "title_id"]].merge(
            titles.drop_duplicates("id"), left_on="title_id", right_on="id", how="inner",
        )
        frames.append(cj)
    if not frames:
        return {}

    joined = pd.concat(frames, ignore_index=True, sort=False)
    votes = pd.to_numeric(joined.get("imdb_votes", pd.Series(dtype=float)), errors="coerce").fillna(0)
    joined = joined[votes >= 250]
    if joined.empty:
        return {}
    joined = joined.sort_values("_bscore", ascending=False, na_position="last")
    joined = joined.drop_duplicates(subset=["person_id"], keep="first")

    out: dict[str, dict] = {}
    for _, r in joined.iterrows():
        try:
            year = int(r["release_year"]) if pd.notna(r.get("release_year")) else None
        except (ValueError, TypeError):
            year = None
        out[str(r["person_id"])] = {
            "title": r.get("title"),
            "year":  year,
            "imdb":  float(r["imdb_score"]) if pd.notna(r.get("imdb_score")) else None,
            "votes": int(r["imdb_votes"]) if pd.notna(r.get("imdb_votes")) else None,
        }
    return out


def _build_person_award_map(
    principals: pd.DataFrame,
    enriched: pd.DataFrame,
    credits: pd.DataFrame | None = None,
) -> dict[str, int]:
    """For each person_id return the number of titles in their filmography with award_wins > 0.

    The precomputed ``person_stats.parquet`` has ``award_title_count`` hard-zero
    because the upstream script joined against ``all_platforms_titles`` (no
    award column). This helper derives the real count from the enriched table,
    covering both principals (writer/producer/etc.) and credits (actor/director).
    """
    if enriched is None or enriched.empty or "award_wins" not in enriched.columns:
        return {}

    has_award = (pd.to_numeric(enriched["award_wins"], errors="coerce").fillna(0) > 0)

    frames: list[pd.DataFrame] = []
    if principals is not None and not principals.empty and "imdb_id" in enriched.columns:
        t = (enriched.loc[has_award, ["id", "imdb_id"]]
                      .dropna(subset=["imdb_id"])
                      .drop_duplicates("imdb_id"))
        if not t.empty:
            m = principals[["person_id", "imdb_id"]].merge(t, on="imdb_id", how="inner")
            frames.append(m[["person_id", "id"]].rename(columns={"id": "_tid"}))
    if credits is not None and not credits.empty and "id" in enriched.columns:
        t = enriched.loc[has_award, ["id"]].drop_duplicates("id")
        if not t.empty:
            m = credits[["person_id", "title_id"]].merge(
                t, left_on="title_id", right_on="id", how="inner",
            )
            frames.append(m[["person_id", "title_id"]].rename(columns={"title_id": "_tid"}))
    if not frames:
        return {}

    j = pd.concat(frames, ignore_index=True, sort=False)
    j = j.dropna(subset=["_tid"]).drop_duplicates(subset=["person_id", "_tid"])
    out = j.groupby("person_id").size()
    return {str(k): int(v) for k, v in out.items() if v > 0}


def _build_person_keyword_map(
    principals: pd.DataFrame,
    enriched: pd.DataFrame,
    credits: pd.DataFrame | None = None,
) -> dict[str, set]:
    """For each person_id return the union of tmdb_keywords + top_tags across their top titles."""
    if enriched is None or enriched.empty:
        return {}

    has_kw = "tmdb_keywords" in enriched.columns or "top_tags" in enriched.columns
    if not has_kw:
        return {}

    kw_cols = [c for c in ("tmdb_keywords", "top_tags") if c in enriched.columns]
    base_cols = kw_cols + [c for c in ("imdb_score",) if c in enriched.columns]

    frames: list[pd.DataFrame] = []
    if principals is not None and not principals.empty and "imdb_id" in enriched.columns:
        t = enriched[["imdb_id"] + base_cols].dropna(subset=["imdb_id"]).drop_duplicates("imdb_id")
        frames.append(principals[["person_id", "imdb_id"]].merge(t, on="imdb_id", how="inner"))
    if credits is not None and not credits.empty and "id" in enriched.columns:
        t = enriched[["id"] + base_cols].drop_duplicates("id")
        frames.append(credits[["person_id", "title_id"]].merge(
            t, left_on="title_id", right_on="id", how="inner",
        ))
    if not frames:
        return {}
    joined = pd.concat(frames, ignore_index=True, sort=False)
    # Keep top ~5 titles per person by IMDb score
    if "imdb_score" in joined.columns:
        joined = joined.sort_values("imdb_score", ascending=False, na_position="last")
    joined = joined.groupby("person_id").head(5)

    def _extract(row) -> set:
        out: set = set()
        for col in ("tmdb_keywords", "top_tags"):
            v = row.get(col)
            if isinstance(v, (list, np.ndarray)):
                for k in v:
                    if k:
                        out.add(str(k).lower())
        return out

    by_person: dict[str, set] = {}
    for pid, grp in joined.groupby("person_id"):
        kw: set = set()
        for _, r in grp.iterrows():
            kw |= _extract(r)
        if kw:
            by_person[str(pid)] = kw
    return by_person


def greenlight_talent_picks(
    person_stats: pd.DataFrame,
    genres: list[str],
    role: str,
    *,
    min_title_count: int = 5,
    min_avg_imdb: float = 6.5,
    top_k: int = 5,
    tone_selection: list[str] | None = None,
    top_title_map: dict[str, dict] | None = None,
    keyword_map: dict[str, set] | None = None,
    award_map: dict[str, int] | None = None,
    min_career_end: int = 2010,
    min_career_start: int = 1960,
) -> list[dict]:
    """Return top-ranked directors or actors, grounded in prominence + relevance.

    Ranking (no merger bonus — contradicts Platform Fit when pointing away from N/M):
      0.30 * prominence (pagerank + influence_score, percentile-normed within peers)
      0.25 * quality    (avg_imdb)
      0.20 * genre_match (top_genre in selection)
      0.15 * award_norm
      0.10 * tone_relevance
    """
    if person_stats is None or person_stats.empty or not genres:
        return []

    role_u = role.upper()
    df = person_stats[person_stats["primary_role"] == role_u].copy()
    if df.empty:
        return []

    # Dedup by person_id, keeping the densest profile
    df = df.sort_values("title_count", ascending=False).drop_duplicates(subset=["person_id"], keep="first")

    df = df[df["top_genre"].isin(set(genres))]
    df = df[(df["title_count"] >= min_title_count) & (df["avg_imdb"] >= min_avg_imdb)]

    # Recency — drop people who haven't worked in the last ~15 years
    if "career_end" in df.columns and min_career_end is not None:
        df = df[pd.to_numeric(df["career_end"], errors="coerce").fillna(0) >= min_career_end]

    # Skip people whose careers started pre-1960 — streaming catalogs contain
    # long-dead filmmakers/actors whose films are re-released or credited
    # posthumously, which inflates ``career_end``. This cutoff is a cheap
    # proxy for "still likely living & actively working."
    if "career_start" in df.columns and min_career_start is not None:
        df = df[pd.to_numeric(df["career_start"], errors="coerce").fillna(9999) >= min_career_start]

    # Prominence floor — remove isolated / disconnected profiles that signal
    # "barely left a mark" (no real collaborators in the network).
    if "influence_score" in df.columns:
        df = df[pd.to_numeric(df["influence_score"], errors="coerce").fillna(0) > 0]
    if df.empty:
        return []

    # Override the hard-zero award_title_count with a derived-at-load count
    if award_map:
        df["award_title_count"] = df["person_id"].astype(str).map(award_map).fillna(0).astype(int)

    # Percentile-rank prominence signals within this filtered subset
    df["_pr_pct"]  = df["pagerank"].fillna(0).rank(pct=True)
    df["_inf_pct"] = df["influence_score"].fillna(0).rank(pct=True)
    df["_prominence"] = 0.5 * df["_pr_pct"] + 0.5 * df["_inf_pct"]

    # Tighten further — only people whose prominence is in the upper half of
    # the filtered set (keeps household names, drops trivia-level entries).
    df = df[df["_prominence"] >= 0.50]
    if df.empty:
        return []

    df["_quality_norm"] = ((df["avg_imdb"] - 5.0) / 5.0).clip(0, 1)
    df["_genre_match"]  = 1.0
    df["_award_norm"]   = (df["award_title_count"].fillna(0) / 5.0).clip(0, 1)

    # Tone relevance — share of selected tones that appear in person's keyword union
    from src.analysis.lab import GREENLIGHT_TONE_VOCAB
    tone_keys = [t for t in (tone_selection or []) if t in GREENLIGHT_TONE_VOCAB]
    if tone_keys and keyword_map:
        def _tone_rel(pid):
            pid = str(pid)
            kws = keyword_map.get(pid)
            if not kws:
                return 0.0
            hits = 0
            for t in tone_keys:
                vocab = set(GREENLIGHT_TONE_VOCAB.get(t, []))
                if kws & vocab:
                    hits += 1
            return hits / len(tone_keys)
        df["_tone_rel"] = df["person_id"].apply(_tone_rel)
    else:
        df["_tone_rel"] = 0.0

    df["_rank"] = (
        0.30 * df["_prominence"]
        + 0.25 * df["_quality_norm"]
        + 0.20 * df["_genre_match"]
        + 0.15 * df["_award_norm"]
        + 0.10 * df["_tone_rel"]
    )
    df = df.sort_values(["_rank", "pagerank"], ascending=[False, False])
    # People often have multiple person_id entries (credits vs principals
    # namespace). Keep the single highest-ranked row per name.
    df = df.drop_duplicates(subset=["name"], keep="first").head(top_k)

    records: list[dict] = []
    for _, r in df.iterrows():
        plats = r["platform_list"]
        if isinstance(plats, np.ndarray):
            plats = list(plats)
        elif not isinstance(plats, list):
            plats = []
        pid = str(r["person_id"])

        matched_tones = []
        if tone_keys and keyword_map:
            kws = keyword_map.get(pid, set())
            for t in tone_keys:
                vocab = set(GREENLIGHT_TONE_VOCAB.get(t, []))
                if kws & vocab:
                    matched_tones.append(t)

        top_title = (top_title_map or {}).get(pid)

        records.append({
            "person_id":      pid,
            "name":           r["name"],
            "role":           role_u,
            "avg_imdb":       float(r["avg_imdb"]),
            "title_count":    int(r["title_count"]),
            "top_genre":      r["top_genre"],
            "platforms":      plats,
            "award_titles":   int(r.get("award_title_count", 0) or 0),
            "prominence":     float(r["_prominence"]),
            "tone_relevance": float(r["_tone_rel"]),
            "matched_tones":  matched_tones,
            "top_title":      top_title,
        })
    return records


# =============================================================================
# Greenlight Studio — Box Office Projection (movies only)
# =============================================================================
def _compute_box_office_lookup(enriched: pd.DataFrame) -> dict:
    """Return a long-form comp table for budget-smooth ROI projection.

    Columns: ``_ratio``, ``_bt``, ``_cert``, ``_pg``, ``_log_budget``,
    ``_franchise`` (bool), ``_adaptation`` (bool), ``budget_usd``.
    """
    def _bt(v):
        if pd.isna(v) or v <= 0: return 0
        v = float(v)
        if v < 20_000_000: return 1
        if v < 80_000_000: return 2
        if v < 200_000_000: return 3
        return 4

    cert_map = {"G": 1, "PG": 1, "PG-13": 2, "R": 3, "NC-17": 3, "NR": 3}

    df = enriched.copy()
    df = df[(df.get("type") == "Movie")
            & df.get("budget_usd").notna()
            & df.get("box_office_usd").notna()]
    votes = pd.to_numeric(df.get("imdb_votes", pd.Series(dtype=float)), errors="coerce").fillna(0)
    df = df[votes >= 500]
    if df.empty:
        return {"comps": pd.DataFrame(), "training_n": 0}

    primary = df["genres"].apply(
        lambda g: g[0] if isinstance(g, (list, np.ndarray)) and len(g) > 0 else "unknown"
    )

    def _is_adaptation(kws):
        if not isinstance(kws, (list, np.ndarray)):
            return False
        joined = " ".join(str(k).lower() for k in kws)
        return any(tok in joined for tok in _ADAPTATION_TOKENS)

    adaptation = (
        df["tmdb_keywords"].apply(_is_adaptation)
        if "tmdb_keywords" in df.columns
        else pd.Series(False, index=df.index)
    )

    comps = pd.DataFrame({
        "budget_usd":  df["budget_usd"].astype(float),
        "_ratio":      df["box_office_usd"].astype(float) / df["budget_usd"].replace(0, np.nan).astype(float),
        "_bt":         df["budget_usd"].apply(_bt).astype(int),
        "_cert":       df["age_certification"].fillna("").map(cert_map).fillna(3).astype(int),
        "_pg":         primary.astype(str),
        "_log_budget": np.log(df["budget_usd"].astype(float).clip(lower=1.0)),
        "_franchise":  df["collection_name"].notna() if "collection_name" in df.columns else False,
        "_adaptation": adaptation.values,
    })
    comps = comps[comps["_ratio"].between(0.05, 50)].reset_index(drop=True)
    return {"comps": comps, "training_n": int(len(comps))}


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Weighted quantile (linear interpolation on the CDF)."""
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cum = np.cumsum(w)
    if cum[-1] <= 0:
        return float(np.median(values))
    cdf = cum / cum[-1]
    return float(np.interp(q, cdf, v))


def greenlight_box_office(
    features_dict: dict,
    prediction: float,
    content_type: str,
    lookup: dict,
    peer: dict | None,
) -> dict | None:
    """Return a grounded box office projection for movies.

    Smooth kNN over log(budget) within a matched comp set — responds
    continuously to the user's budget instead of snapping to discrete tiers.
    """
    if content_type != "Movie" or not lookup:
        return None
    comps_all: pd.DataFrame = lookup.get("comps", pd.DataFrame())
    if comps_all.empty:
        return None

    budget_usd = features_dict.get("budget_usd")
    if budget_usd is None:
        tier = int(features_dict.get("budget_tier") or 0)
        budget_usd = {1: 10e6, 2: 45e6, 3: 120e6, 4: 280e6}.get(tier, 30e6)
    budget_usd = float(budget_usd)

    cert_tier     = int(features_dict.get("age_cert_tier") or 3)
    primary_genre = (features_dict.get("genres") or ["unknown"])[0]
    ftype         = int(features_dict.get("franchise_type") or 0)

    # ── Tiered comp-set relaxation ───────────────────────────────────────────
    relax_steps = [
        ("tight",       (comps_all["_pg"] == primary_genre) & (comps_all["_cert"] == cert_tier)),
        ("genre_only",   comps_all["_pg"] == primary_genre),
        ("cert_only",    comps_all["_cert"] == cert_tier),
        ("global",       pd.Series(True, index=comps_all.index)),
    ]
    comps = pd.DataFrame()
    relaxation = "global"
    for name, mask in relax_steps:
        sub = comps_all[mask]
        if len(sub) >= 20:
            comps = sub
            relaxation = name
            break
    if comps.empty:
        comps = comps_all
        relaxation = "global"

    # ── Kernel-weighted ROI quantiles over log(budget) ───────────────────────
    log_b     = float(np.log(max(budget_usd, 1.0)))
    distances = comps["_log_budget"].to_numpy() - log_b
    weights   = np.exp(-(distances ** 2) / (2 * 0.5 ** 2))
    if weights.sum() <= 1e-6:
        weights = np.ones_like(weights)
    ratios = comps["_ratio"].to_numpy()
    roi_lo  = _weighted_quantile(ratios, weights, 0.25)
    roi_med = _weighted_quantile(ratios, weights, 0.50)
    roi_hi  = _weighted_quantile(ratios, weights, 0.75)

    # ── Quality multiplier (percentile-anchored) ─────────────────────────────
    if peer and peer.get("p25") is not None:
        p25, p50, p75, p90 = peer["p25"], peer["p50"], peer["p75"], peer["p90"]
        if prediction <= p25:   quality_mult = 0.75
        elif prediction <= p50: quality_mult = 0.90
        elif prediction <= p75: quality_mult = 1.10
        elif prediction <= p90: quality_mult = 1.25
        else:                   quality_mult = 1.40
    else:
        quality_mult = 1.0

    # ── Franchise / adaptation empirical lift ────────────────────────────────
    franchise_mult = 1.0
    franchise_n    = 0
    if ftype == 1:
        f_rows = comps[comps["_franchise"]]
        o_rows = comps[~comps["_franchise"] & ~comps["_adaptation"]]
        if len(f_rows) >= 15 and len(o_rows) >= 15:
            fm = float(np.median(f_rows["_ratio"]))
            om = float(np.median(o_rows["_ratio"]))
            if om > 0:
                franchise_mult = float(np.clip(fm / om, 0.9, 1.6))
                franchise_n    = int(len(f_rows))
        if franchise_n == 0:
            f_rows_all = comps_all[comps_all["_franchise"]]
            o_rows_all = comps_all[~comps_all["_franchise"] & ~comps_all["_adaptation"]]
            if len(f_rows_all) >= 30 and len(o_rows_all) >= 30:
                franchise_mult = float(np.clip(
                    np.median(f_rows_all["_ratio"]) / np.median(o_rows_all["_ratio"]),
                    0.9, 1.6,
                ))
                franchise_n    = int(len(f_rows_all))
    elif ftype == 2:
        a_rows = comps[comps["_adaptation"]]
        o_rows = comps[~comps["_adaptation"] & ~comps["_franchise"]]
        if len(a_rows) >= 15 and len(o_rows) >= 15:
            franchise_mult = float(np.clip(
                np.median(a_rows["_ratio"]) / np.median(o_rows["_ratio"]),
                0.9, 1.4,
            ))
            franchise_n = int(len(a_rows))
        if franchise_n == 0:
            a_rows_all = comps_all[comps_all["_adaptation"]]
            o_rows_all = comps_all[~comps_all["_adaptation"] & ~comps_all["_franchise"]]
            if len(a_rows_all) >= 30 and len(o_rows_all) >= 30:
                franchise_mult = float(np.clip(
                    np.median(a_rows_all["_ratio"]) / np.median(o_rows_all["_ratio"]),
                    0.9, 1.4,
                ))
                franchise_n = int(len(a_rows_all))

    projected = budget_usd * roi_med * quality_mult * franchise_mult
    low       = budget_usd * roi_lo  * quality_mult * franchise_mult
    high      = budget_usd * roi_hi  * quality_mult * franchise_mult

    budget_p10 = float(comps["budget_usd"].quantile(0.10))
    budget_p90 = float(comps["budget_usd"].quantile(0.90))

    return {
        "projected":         float(projected),
        "low":               float(low),
        "high":              float(high),
        "budget_used":       float(budget_usd),
        "roi_med":           float(roi_med),
        "roi_lo":            float(roi_lo),
        "roi_hi":            float(roi_hi),
        "quality_mult":      float(quality_mult),
        "franchise_mult":    float(franchise_mult),
        "franchise_n":       int(franchise_n),
        "comp_count":        int(len(comps)),
        "comp_budget_lo_m":  budget_p10 / 1e6,
        "comp_budget_hi_m":  budget_p90 / 1e6,
        "primary_genre":     primary_genre,
        "relaxation":        relaxation,
        "franchise_type":    ftype,
    }


def get_talent_suggestions(principals_df, titles_df, genres, role="director", top_k=5):
    """Get top talent for given genres and role."""
    if principals_df.empty:
        return pd.DataFrame()

    # Map imdb_id to title info (drop duplicates first to avoid ValueError on set_index)
    title_info = (
        titles_df
        .drop_duplicates(subset=["imdb_id"])
        .set_index("imdb_id")[["imdb_score", "genres", "title"]]
        .to_dict("index")
    )

    # Filter principals by role
    role_df = principals_df[principals_df["category"] == role]
    if role_df.empty:
        return pd.DataFrame()

    # Score each person
    person_stats = []
    for person_id, group in role_df.groupby("person_id"):
        scores = []
        genre_match = 0
        titles = []
        for _, row in group.iterrows():
            info = title_info.get(row["imdb_id"], {})
            if info.get("imdb_score"):
                scores.append(info["imdb_score"])
            title_genres = info.get("genres", [])
            if isinstance(title_genres, (list, np.ndarray)):
                if set(title_genres) & set(genres):
                    genre_match += 1
            if info.get("title"):
                titles.append(info["title"])

        if len(scores) < 2 or genre_match == 0:
            continue

        person_stats.append({
            "name": group.iloc[0]["name"],
            "avg_imdb": round(np.mean(scores), 2),
            "title_count": len(scores),
            "genre_match": genre_match,
            "titles": titles[:5],
        })

    if not person_stats:
        return pd.DataFrame()

    df = pd.DataFrame(person_stats)
    df["score"] = df["avg_imdb"] * np.log1p(df["genre_match"])
    return df.nlargest(top_k, "score")


# =============================================================================
# Feature 3: Insight Generator
# =============================================================================

def generate_insights(titles_df, scope="all_platforms", enriched_df=None):
    """Generate 5-8 data-backed insights for the given scope.

    Uses exploratory 'fun facts' tone, distinct from Strategic Insights.
    """
    if scope != "all_platforms":
        df = titles_df[titles_df["platform"] == scope]
        scope_name = PLATFORMS.get(scope, {}).get("name", scope)
    else:
        df = titles_df
        scope_name = "All Platforms"

    insights = []

    # 1. Top-rated hidden gem
    hidden = df[(df["imdb_score"] >= 7.5) & (df["imdb_votes"] < 10000)].sort_values("imdb_score", ascending=False)
    if len(hidden) > 0:
        gem = hidden.iloc[0]
        insights.append(
            f"Hidden gem alert: \"{gem['title']}\" ({int(gem.get('release_year', 0))}) "
            f"has a {gem['imdb_score']:.1f} IMDb rating with only {int(gem.get('imdb_votes', 0)):,} votes"
        )

    # 2. Genre dominance
    genre_counts = df.explode("genres")["genres"].value_counts()
    if len(genre_counts) > 0:
        top_genre = genre_counts.index[0]
        top_pct = genre_counts.iloc[0] / len(df) * 100
        insights.append(
            f"{scope_name}'s most dominant genre is {top_genre}, "
            f"appearing in {top_pct:.1f}% of all titles"
        )

    # 3. Quality by decade
    if "release_year" in df.columns:
        df_temp = df[df["imdb_score"].notna()].copy()
        df_temp["decade"] = (df_temp["release_year"] // 10) * 10
        decade_avg = df_temp.groupby("decade")["imdb_score"].mean()
        if len(decade_avg) > 2:
            best_decade = decade_avg.idxmax()
            insights.append(
                f"The {int(best_decade)}s produced the highest-rated content "
                f"with an average IMDb of {decade_avg[best_decade]:.2f}"
            )

    # 4. International content
    if "production_countries" in df.columns:
        intl = df["production_countries"].apply(
            lambda c: "US" not in c if isinstance(c, (list, np.ndarray)) else True
        )
        intl_pct = intl.mean() * 100
        intl_quality = df[intl]["imdb_score"].mean()
        domestic_quality = df[~intl]["imdb_score"].mean()
        if pd.notna(intl_quality) and pd.notna(domestic_quality):
            diff = intl_quality - domestic_quality
            direction = "higher" if diff > 0 else "lower"
            insights.append(
                f"International productions ({intl_pct:.0f}% of catalog) average "
                f"{abs(diff):.1f} IMDb points {direction} than US productions"
            )

    # 5. Movies vs Shows quality
    movies = df[df["type"] == "Movie"]["imdb_score"].mean()
    shows = df[df["type"] == "Show"]["imdb_score"].mean()
    if pd.notna(movies) and pd.notna(shows):
        better = "Movies" if movies > shows else "Shows"
        diff = abs(movies - shows)
        insights.append(
            f"{better} average {diff:.2f} IMDb points higher than "
            f"{'shows' if better == 'Movies' else 'movies'} on {scope_name}"
        )

    # 6. Awards insight (if enrichment available)
    if enriched_df is not None and "award_wins" in enriched_df.columns:
        if scope != "all_platforms":
            award_df = enriched_df[enriched_df["platform"] == scope]
        else:
            award_df = enriched_df

        award_titles = award_df[award_df["award_wins"].notna() & (award_df["award_wins"] > 0)]
        if len(award_titles) > 10:
            avg_award_imdb = award_titles["imdb_score"].mean()
            avg_all_imdb = award_df["imdb_score"].mean()
            insights.append(
                f"Award-winning titles average {avg_award_imdb:.2f} IMDb — "
                f"{avg_award_imdb - avg_all_imdb:.1f} points above the catalog average"
            )

    # 7. Runtime insight
    if "runtime" in df.columns:
        runtime = pd.to_numeric(df["runtime"], errors="coerce")
        movies_runtime = runtime[df["type"] == "Movie"].dropna()
        if len(movies_runtime) > 50:
            median_rt = movies_runtime.median()
            long_movies = (movies_runtime > 150).sum()
            insights.append(
                f"Median movie runtime is {int(median_rt)} minutes, "
                f"with {long_movies:,} epic-length films over 2.5 hours"
            )

    # 8. Platform-specific surprise
    if scope == "all_platforms":
        platform_avgs = df.groupby("platform")["imdb_score"].mean()
        if len(platform_avgs) > 1:
            best = platform_avgs.idxmax()
            worst = platform_avgs.idxmin()
            insights.append(
                f"{PLATFORMS.get(best, {}).get('name', best)} leads in average quality "
                f"({platform_avgs[best]:.2f} IMDb) while "
                f"{PLATFORMS.get(worst, {}).get('name', worst)} trails at {platform_avgs[worst]:.2f}"
            )

    return insights[:8]


def get_random_insight(titles_df, enriched_df=None):
    """Generate a single random 'surprise me' insight."""
    scopes = ["all_platforms"] + ALL_PLATFORMS
    scope = random.choice(scopes)
    insights = generate_insights(titles_df, scope, enriched_df)
    if insights:
        return random.choice(insights)
    return "No insights available for this scope."
