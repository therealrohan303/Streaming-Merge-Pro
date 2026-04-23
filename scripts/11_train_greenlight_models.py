"""
Script 11: Train Greenlight Studio predictors (v3 — rich feature set).

Replaces scripts/10_train_greenlight_predictor.py with a model that actually
responds to the concept description and tone selection in the UI.

What's new vs v2:
  - TruncatedSVD(n_components=40) features from the title description TF-IDF
    (so concept text drives the prediction at inference).
  - Tone-cluster binary features (12 tones; each tone is a keyword bag).
  - Pairwise genre-interaction flags for 8 common pairs.
  - `franchise_type` (original / franchise entry / adaptation) as 3 one-hots,
    with a data-driven labeling rule using collection_name + tmdb_keywords.
  - Runtime² to capture non-linear duration effects.
  - HistGradientBoostingRegressor (primary) blended 0.8 / 0.2 with a Ridge over
    just the SVD components, which sharpens the description signal.
  - Target = Bayesian-shrunk IMDb score (v / (v+m)) * R + (m / (v+m)) * C
    to stop low-vote outliers from noise-training the regressor.

Output: the same pickle paths as before so the loader is unchanged.
"""

import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import ENRICHED_DIR, MODELS_DIR
from src.analysis.scoring import bayesian_imdb
from src.analysis.greenlight_model import GreenlightStackedModel


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
MIN_VOTES = 500
MIN_SAMPLES = 200
N_SVD_COMPONENTS = 40
RIDGE_BLEND_WEIGHT = 0.20  # blend weight for Ridge-on-SVD in the final stacked prediction
RNG = 42

TONE_VOCAB: dict[str, list[str]] = {
    "prestige":        ["award", "oscar", "acclaimed", "prestige", "critically", "masterpiece", "emmy", "biopic"],
    "gritty":          ["gritty", "dark", "brutal", "violent", "raw", "grim", "noir", "bleak", "visceral"],
    "character":       ["character", "introspective", "portrait", "study", "intimate", "internal", "meditative"],
    "action":          ["action", "explosive", "fight", "chase", "combat", "high-octane", "relentless", "shootout"],
    "mystery":         ["mystery", "thriller", "suspense", "investigation", "detective", "whodunit", "murder"],
    "feel_good":       ["feel-good", "heartwarming", "uplifting", "charming", "wholesome", "sweet", "joyful"],
    "romantic":        ["romance", "romantic", "love", "lovers", "relationship", "passion", "courtship"],
    "tearjerker":      ["tragic", "heartbreak", "tearjerker", "grief", "loss", "devastating", "mourning"],
    "quirky":          ["quirky", "offbeat", "eccentric", "indie", "weird", "whimsical", "surreal", "absurd"],
    "epic":            ["epic", "sweeping", "grand", "sprawling", "historical", "vast", "saga"],
    "satirical":       ["satire", "satirical", "biting", "ironic", "political", "sharp-tongued", "dark comedy"],
    "family":          ["family", "kids", "child", "animated", "heartfelt", "cartoon", "coming-of-age"],
}
TONE_KEYS = list(TONE_VOCAB.keys())

# Pairs that are common + statistically distinct combinations
GENRE_PAIRS = [
    ("drama", "thriller"),
    ("drama", "romance"),
    ("action", "scifi"),
    ("action", "thriller"),
    ("comedy", "romance"),
    ("crime", "drama"),
    ("horror", "thriller"),
    ("animation", "family"),
]

ADAPTATION_PATTERNS = [
    "based on novel", "based on a novel", "based on book", "based on comic",
    "based on true story", "based on a true story", "remake", "reboot",
    "based on play", "based on video game", "based on short story",
    "based on manga", "based on anime",
]

EUROPE_ASIA = {
    "FR", "DE", "JP", "KR", "IT", "ES", "CA", "AU", "IN", "CN", "MX", "BR",
    "NL", "SE", "NO", "DK", "FI", "PL", "PT", "RU", "TR", "TH", "TW", "HK",
    "SG", "AR", "CL", "CO", "ZA", "NG", "EG", "PK", "BD",
}
MOVIE_CERT_TIER = {"G": 1, "PG": 1, "PG-13": 2, "R": 3, "NC-17": 3, "NR": 3}
SHOW_CERT_TIER  = {"TV-G": 1, "TV-PG": 1, "TV-Y": 1, "TV-Y7": 1, "TV-Y7-FV": 1,
                   "TV-14": 2, "TV-MA": 3, "NR": 3}


# ──────────────────────────────────────────────────────────────────────────────
# Feature helpers
# ──────────────────────────────────────────────────────────────────────────────
def _country_tier(countries) -> int:
    if not isinstance(countries, (list, np.ndarray)) or len(countries) == 0:
        return 1
    top = str(countries[0])
    if top in ("US", "GB"):
        return 1
    return 2


def _budget_tier(val) -> int:
    if pd.isna(val) or val <= 0:
        return 0
    v = float(val)
    if v < 20_000_000:   return 1
    if v < 80_000_000:   return 2
    if v < 200_000_000:  return 3
    return 4


def _franchise_type(collection_name, tmdb_keywords) -> int:
    """0=Original, 1=Franchise entry, 2=Adaptation."""
    if isinstance(collection_name, str) and collection_name.strip():
        return 1
    if isinstance(tmdb_keywords, (list, np.ndarray)):
        joined = " ".join(str(k).lower() for k in tmdb_keywords)
        if any(p in joined for p in ADAPTATION_PATTERNS):
            return 2
    return 0


def _extract_genres(df: pd.DataFrame) -> list[str]:
    s: set[str] = set()
    for g in df["genres"].dropna():
        if isinstance(g, (list, np.ndarray)):
            s.update(g)
    return sorted(s)


def _tone_binary(text: str, tag_list) -> np.ndarray:
    """Return binary vector indicating tone-cluster keyword hits."""
    combined = (text or "").lower()
    if isinstance(tag_list, (list, np.ndarray)):
        combined += " " + " ".join(str(t).lower() for t in tag_list)
    out = np.zeros(len(TONE_KEYS), dtype=np.int8)
    for i, key in enumerate(TONE_KEYS):
        for kw in TONE_VOCAB[key]:
            if kw in combined:
                out[i] = 1
                break
    return out


def _compute_tone_features(df: pd.DataFrame) -> pd.DataFrame:
    tag_col = df.get("tmdb_keywords") if "tmdb_keywords" in df.columns else pd.Series([None] * len(df), index=df.index)
    out = np.vstack([
        _tone_binary(d, t)
        for d, t in zip(df["description"].fillna("").tolist(), tag_col.tolist())
    ])
    return pd.DataFrame(out, index=df.index, columns=[f"tone_{k}" for k in TONE_KEYS])


def _compute_genre_pair_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for a, b in GENRE_PAIRS:
        col = f"gpair_{a}_x_{b}"
        rows[col] = df["genres"].apply(
            lambda gl: int(isinstance(gl, (list, np.ndarray)) and (a in gl) and (b in gl))
        )
    return pd.DataFrame(rows, index=df.index)


def _compute_svd(descriptions: list[str], vectorizer) -> tuple[TruncatedSVD, np.ndarray]:
    mat = vectorizer.transform(descriptions)
    svd = TruncatedSVD(n_components=N_SVD_COMPONENTS, random_state=RNG)
    emb = svd.fit_transform(mat)
    return svd, emb


def _genre_percentiles(df: pd.DataFrame, all_genres: list[str]) -> dict:
    """Per-genre IMDb distribution — used in UI to render peer-relative labels."""
    out: dict[str, dict] = {}
    s = pd.to_numeric(df["imdb_score"], errors="coerce")
    for g in all_genres:
        mask = df["genres"].apply(lambda gl: isinstance(gl, (list, np.ndarray)) and g in gl)
        vals = s[mask].dropna()
        if len(vals) >= 40:
            out[g] = {
                "p25": float(np.percentile(vals, 25)),
                "p50": float(np.percentile(vals, 50)),
                "p75": float(np.percentile(vals, 75)),
                "p90": float(np.percentile(vals, 90)),
                "n":   int(len(vals)),
            }
    # global fallback
    all_vals = s.dropna()
    out["__global__"] = {
        "p25": float(np.percentile(all_vals, 25)),
        "p50": float(np.percentile(all_vals, 50)),
        "p75": float(np.percentile(all_vals, 75)),
        "p90": float(np.percentile(all_vals, 90)),
        "n":   int(len(all_vals)),
    }
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Feature builders
# ──────────────────────────────────────────────────────────────────────────────
def build_structured_features(
    df: pd.DataFrame, all_genres: list[str], content_type: str,
) -> pd.DataFrame:
    feats = pd.DataFrame(index=df.index)

    # Genre one-hots
    for g in all_genres:
        feats[f"genre_{g}"] = df["genres"].apply(
            lambda gl: int(isinstance(gl, (list, np.ndarray)) and g in gl)
        )

    # Genre-pair interactions
    pair_df = _compute_genre_pair_features(df)
    feats = pd.concat([feats, pair_df], axis=1)

    # Runtime + runtime^2
    rt = pd.to_numeric(df["runtime"], errors="coerce")
    rt_fill = float(rt.median())
    rt = rt.fillna(rt_fill)
    feats["runtime"] = rt
    feats["runtime_sq"] = rt ** 2

    # Decade (int 1990 / 2000 / 2010 / 2020)
    ry = pd.to_numeric(df["release_year"], errors="coerce").fillna(2010)
    feats["decade"] = ((ry // 10) * 10).astype(int)

    # Country tier
    feats["production_country_tier"] = df["production_countries"].apply(_country_tier)

    # Cert tier (keyed by type)
    cert = df.get("age_certification", pd.Series("", index=df.index)).fillna("")
    if content_type == "Movie":
        feats["age_cert_tier"] = cert.map(MOVIE_CERT_TIER).fillna(2).astype(int)
    else:
        feats["age_cert_tier"] = cert.map(SHOW_CERT_TIER).fillna(2).astype(int)

    # Cert × budget_tier interaction (movies only — budget_tier is 0 for shows)
    if content_type == "Movie":
        bt = df["budget_usd"].apply(_budget_tier) if "budget_usd" in df.columns else pd.Series(0, index=df.index)
        feats["budget_tier"] = bt
        feats["budget_log"] = np.log1p(pd.to_numeric(df.get("budget_usd", pd.Series(0, index=df.index)), errors="coerce").fillna(0).clip(lower=0))
        feats["cert_x_budget"] = feats["age_cert_tier"] * feats["budget_tier"]

        # Franchise type (3 one-hots)
        ft = df.apply(
            lambda r: _franchise_type(r.get("collection_name"), r.get("tmdb_keywords")),
            axis=1,
        )
        feats["ftype_original"]   = (ft == 0).astype(int)
        feats["ftype_franchise"]  = (ft == 1).astype(int)
        feats["ftype_adaptation"] = (ft == 2).astype(int)
    else:
        # Shows: num_seasons as the budget-equivalent scope feature
        s = pd.to_numeric(df.get("seasons", pd.Series(1, index=df.index)), errors="coerce").fillna(1).clip(1, 20)
        feats["num_seasons"] = s

    # Tone cluster binaries
    tone_df = _compute_tone_features(df)
    feats = pd.concat([feats, tone_df], axis=1)

    return feats


# ──────────────────────────────────────────────────────────────────────────────
# Training / CV
# ──────────────────────────────────────────────────────────────────────────────
def _fit_cv(X: pd.DataFrame, y: np.ndarray, svd_cols: list[str]) -> tuple[float, float]:
    """5-fold CV RMSE for the stacked model."""
    kf = KFold(n_splits=5, shuffle=True, random_state=RNG)
    rmses = []
    for tr, te in kf.split(X):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y[tr], y[te]
        gbm = HistGradientBoostingRegressor(
            learning_rate=0.06, max_iter=400, max_depth=6,
            min_samples_leaf=20, l2_regularization=0.1, random_state=RNG,
        )
        gbm.fit(Xtr, ytr)
        pred_gbm = gbm.predict(Xte)

        if svd_cols:
            ridge = Ridge(alpha=1.0, random_state=RNG)
            ridge.fit(Xtr[svd_cols], ytr)
            pred_ridge = ridge.predict(Xte[svd_cols])
            pred = (1 - RIDGE_BLEND_WEIGHT) * pred_gbm + RIDGE_BLEND_WEIGHT * pred_ridge
        else:
            pred = pred_gbm
        rmses.append(np.sqrt(mean_squared_error(yte, pred)))
    return float(np.mean(rmses)), float(np.std(rmses))


def _fit_final(X: pd.DataFrame, y: np.ndarray, svd_cols: list[str]):
    gbm = HistGradientBoostingRegressor(
        learning_rate=0.06, max_iter=400, max_depth=6,
        min_samples_leaf=20, l2_regularization=0.1, random_state=RNG,
    )
    gbm.fit(X, y)
    ridge = None
    if svd_cols:
        ridge = Ridge(alpha=1.0, random_state=RNG)
        ridge.fit(X[svd_cols], y)
    return gbm, ridge


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def _load_catalog() -> pd.DataFrame:
    path = ENRICHED_DIR / "titles_enriched.parquet"
    if not path.exists():
        print("ERROR: titles_enriched.parquet not found. Run script 09 first.")
        sys.exit(1)
    df = pd.read_parquet(path)
    before = len(df)
    df = df[df["imdb_score"].notna()].copy()
    votes = pd.to_numeric(df.get("imdb_votes", pd.Series(dtype=float)), errors="coerce").fillna(0)
    df = df[votes >= MIN_VOTES].copy()
    print(f"Loaded {len(df):,} titles (from {before:,}) with imdb_score + votes >= {MIN_VOTES}")
    return df


def _build_target(df: pd.DataFrame) -> np.ndarray:
    """Bayesian-shrunk IMDb score as the regression target."""
    return np.asarray(bayesian_imdb(df["imdb_score"], df["imdb_votes"]).values, dtype=float)


def _train_one(df: pd.DataFrame, all_genres: list[str], content_type: str, vectorizer) -> GreenlightStackedModel | None:
    sub = df[df["type"] == content_type].reset_index(drop=True)
    if len(sub) < MIN_SAMPLES:
        print(f"\nSKIPPED {content_type} model: only {len(sub)} samples (need {MIN_SAMPLES})")
        return None

    print(f"\n{'='*66}")
    print(f"Training {content_type.upper()} model — {len(sub):,} samples")
    structured = build_structured_features(sub, all_genres, content_type)

    svd, svd_emb = _compute_svd(sub["description"].fillna("").tolist(), vectorizer)
    svd_cols = [f"svd_{i:02d}" for i in range(N_SVD_COMPONENTS)]
    svd_df = pd.DataFrame(svd_emb, columns=svd_cols, index=sub.index)

    X = pd.concat([structured, svd_df], axis=1)
    y = _build_target(sub)

    cv_rmse, cv_std = _fit_cv(X, y, svd_cols)
    baseline_rmse = float(np.sqrt(np.mean((y - y.mean()) ** 2)))
    improvement = (1 - cv_rmse / baseline_rmse) * 100
    print(f"  Features:      {X.shape[1]} ({len(svd_cols)} SVD + {X.shape[1]-len(svd_cols)} structured)")
    print(f"  CV RMSE:       {cv_rmse:.3f} ± {cv_std:.3f}")
    print(f"  Baseline RMSE: {baseline_rmse:.3f} (global mean {y.mean():.2f})")
    print(f"  Improvement:   {improvement:.1f}% over naive mean")

    gbm, ridge = _fit_final(X, y, svd_cols)

    # Feature importances
    imps = pd.Series(gbm.feature_importances_ if hasattr(gbm, "feature_importances_") else np.zeros(X.shape[1]),
                     index=X.columns)
    # HistGradientBoostingRegressor doesn't expose feature_importances_ in older sklearn,
    # so fall back to permutation-ish zero array; still fine — UI shows top structured features.
    top = imps.sort_values(ascending=False).head(10)
    if top.sum() > 0:
        print("  Top 10 features:")
        for f, v in top.items():
            print(f"    {f}: {v:.4f}")

    # Wrap into stacked model
    model = GreenlightStackedModel()
    model.gbm_ = gbm
    model.ridge_ = ridge
    model.vectorizer_ = vectorizer
    model.svd_ = svd
    model.feature_names_ = list(X.columns)
    model.svd_cols_ = svd_cols
    model.tone_keys_ = list(TONE_KEYS)
    model.genre_pairs_ = list(GENRE_PAIRS)
    model.genre_names_ = all_genres
    model.content_type_ = content_type
    model.cv_rmse_ = round(cv_rmse, 6)
    model.cv_rmse_std_ = round(cv_std, 6)
    model.baseline_rmse_ = round(baseline_rmse, 6)
    model.global_mean_ = round(float(y.mean()), 4)
    model.training_size_ = int(len(sub))
    model.genre_percentiles_ = _genre_percentiles(sub, all_genres)
    model.ridge_blend_ = RIDGE_BLEND_WEIGHT

    return model


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = _load_catalog()
    all_genres = _extract_genres(df)
    print(f"Genres ({len(all_genres)}): {all_genres}")

    # Load the catalog TF-IDF vectorizer (same one the UI uses downstream)
    vec_path = MODELS_DIR / "tfidf_vectorizer.pkl"
    if not vec_path.exists():
        print(f"ERROR: {vec_path} not found. Run script 03 first.")
        sys.exit(1)
    vectorizer = joblib.load(vec_path)
    print(f"TF-IDF vectorizer: {vectorizer}")

    # Train movie + show models
    movie_model = _train_one(df, all_genres, "Movie", vectorizer)
    if movie_model is not None:
        out = MODELS_DIR / "greenlight_movie_predictor.pkl"
        joblib.dump(movie_model, out)
        print(f"\n  Saved → {out}")

    show_model = _train_one(df, all_genres, "Show", vectorizer)
    if show_model is not None:
        out = MODELS_DIR / "greenlight_show_predictor.pkl"
        joblib.dump(show_model, out)
        print(f"\n  Saved → {out}")

    print("\nDone!")


if __name__ == "__main__":
    main()
