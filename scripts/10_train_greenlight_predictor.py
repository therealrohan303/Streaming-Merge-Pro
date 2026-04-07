"""
Script 10 (v2): Train Greenlight Predictor — separate models for movies and shows.

Changes vs 10_train_predictor.py:
  - Adds age_cert_tier (G/PG/PG-13/R/NC-17 → 0/1/2)
  - Adds decade feature (encoded as integer: 2020, 2010, …)
  - Country tier re-encoded: 0=Other, 1=Europe/Asia, 2=US/UK
  - Budget tier uses fixed thresholds ($10M / $50M / $200M)
  - Show model excludes budget_tier and has_franchise (not meaningful for episodic)
  - Show model includes num_seasons instead
  - Filter: imdb_votes > 1000 (removes low-credibility ratings)
  - GBR params: n_estimators=200, max_depth=4, learning_rate=0.05

Output:
  models/greenlight_movie_predictor.pkl
  models/greenlight_show_predictor.pkl
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import ENRICHED_DIR, MODELS_DIR

# Countries considered "Europe/Asia" for tier=1
EUROPE_ASIA = {
    "FR", "DE", "JP", "KR", "IT", "ES", "CA", "AU", "IN", "CN", "MX", "BR",
    "NL", "SE", "NO", "DK", "FI", "PL", "PT", "RU", "TR", "TH", "TW", "HK",
    "SG", "AR", "CL", "CO", "ZA", "NG", "EG", "PK", "BD",
}

# Movie age cert → tier
MOVIE_CERT_TIER = {
    "G": 0, "PG": 0,
    "PG-13": 1,
    "R": 2, "NC-17": 2, "NR": 2,
}

# Show age cert → tier
SHOW_CERT_TIER = {
    "TV-G": 0, "TV-PG": 0, "TV-Y": 0, "TV-Y7": 0,
    "TV-14": 1,
    "TV-MA": 2, "NR": 2,
}

MIN_VOTES = 1_000
MIN_SAMPLES = 200


def load_data():
    path = ENRICHED_DIR / "titles_enriched.parquet"
    if not path.exists():
        print("ERROR: titles_enriched.parquet not found. Run script 09 first.")
        sys.exit(1)
    df = pd.read_parquet(path)
    before = len(df)
    df = df[df["imdb_score"].notna()].copy()
    votes = pd.to_numeric(df.get("imdb_votes", pd.Series(dtype=float)), errors="coerce").fillna(0)
    df = df[votes > MIN_VOTES].copy()
    print(f"Loaded {len(df):,} titles (from {before:,}) with imdb_score + votes > {MIN_VOTES:,}")
    return df


def extract_genres(df):
    all_genres: set = set()
    for g in df["genres"].dropna():
        if isinstance(g, (list, np.ndarray)):
            all_genres.update(g)
    return sorted(all_genres)


def _country_tier(countries):
    if not isinstance(countries, (list, np.ndarray)) or len(countries) == 0:
        return 0
    top = str(countries[0])
    if top in ("US", "GB"):
        return 2
    if top in EUROPE_ASIA:
        return 1
    return 0


def _budget_tier(val):
    if pd.isna(val) or val <= 0:
        return 0
    if val < 10_000_000:
        return 1
    if val < 50_000_000:
        return 2
    if val < 200_000_000:
        return 3
    return 4


def _award_genre_avg(df):
    """Compute average award_wins per primary genre across the dataset."""
    if "award_wins" not in df.columns:
        return pd.Series(0, index=df.index)

    def primary_genre(g):
        if isinstance(g, (list, np.ndarray)) and len(g) > 0:
            return g[0]
        return "unknown"

    tmp = df.copy()
    tmp["_pg"] = tmp["genres"].apply(primary_genre)
    awards_valid = tmp[tmp["award_wins"].notna() & (tmp["award_wins"] > 0)]
    avg_map = (
        awards_valid.groupby("_pg")["award_wins"].mean().to_dict()
    )
    return tmp["_pg"].map(avg_map).fillna(0)


def build_movie_features(df, all_genres):
    features = pd.DataFrame(index=df.index)

    # Genre binary
    for g in all_genres:
        features[f"genre_{g}"] = df["genres"].apply(
            lambda gl: 1 if isinstance(gl, (list, np.ndarray)) and g in gl else 0
        )

    # Runtime
    rt = pd.to_numeric(df["runtime"], errors="coerce")
    features["runtime"] = rt.fillna(rt.median())

    # Release year
    ry = pd.to_numeric(df["release_year"], errors="coerce")
    features["release_year"] = ry.fillna(ry.median())

    # Country tier (0=Other, 1=Europe/Asia, 2=US/UK)
    features["production_country_tier"] = df["production_countries"].apply(_country_tier)

    # Age cert tier
    cert = df.get("age_certification", pd.Series("", index=df.index)).fillna("")
    features["age_cert_tier"] = cert.map(MOVIE_CERT_TIER).fillna(2).astype(int)

    # Budget tier (fixed thresholds)
    if "budget_usd" in df.columns:
        features["budget_tier"] = df["budget_usd"].apply(_budget_tier)
    else:
        features["budget_tier"] = 0

    # Has franchise
    if "collection_name" in df.columns:
        features["has_franchise"] = df["collection_name"].notna().astype(int)
    else:
        features["has_franchise"] = 0

    # Award genre average
    features["award_genre_avg"] = _award_genre_avg(df)

    # Decade (integer: 1990, 2000, 2010…)
    ry2 = pd.to_numeric(df["release_year"], errors="coerce").fillna(2000)
    features["decade"] = ((ry2 // 10) * 10).astype(int)

    return features


def build_show_features(df, all_genres):
    features = pd.DataFrame(index=df.index)

    # Genre binary
    for g in all_genres:
        features[f"genre_{g}"] = df["genres"].apply(
            lambda gl: 1 if isinstance(gl, (list, np.ndarray)) and g in gl else 0
        )

    # Episode runtime
    rt = pd.to_numeric(df["runtime"], errors="coerce")
    features["runtime"] = rt.fillna(rt.median())

    # Release year
    ry = pd.to_numeric(df["release_year"], errors="coerce")
    features["release_year"] = ry.fillna(ry.median())

    # Country tier
    features["production_country_tier"] = df["production_countries"].apply(_country_tier)

    # Age cert tier (show scale)
    cert = df.get("age_certification", pd.Series("", index=df.index)).fillna("")
    features["age_cert_tier"] = cert.map(SHOW_CERT_TIER).fillna(2).astype(int)

    # Number of seasons (fill missing with 1)
    if "seasons" in df.columns:
        features["num_seasons"] = pd.to_numeric(df["seasons"], errors="coerce").fillna(1).clip(1, 20)
    else:
        features["num_seasons"] = 1

    # Award genre average
    features["award_genre_avg"] = _award_genre_avg(df)

    # Decade
    ry2 = pd.to_numeric(df["release_year"], errors="coerce").fillna(2000)
    features["decade"] = ((ry2 // 10) * 10).astype(int)

    return features


def train_model(X, y, label):
    print(f"\n{'='*60}")
    print(f"Training {label} model — {len(X):,} samples, {X.shape[1]} features")

    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
    )

    # 5-fold CV
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="neg_root_mean_squared_error")
    cv_rmse = -cv_scores.mean()
    cv_rmse_std = cv_scores.std()
    baseline_rmse = float(np.sqrt(np.mean((y - y.mean()) ** 2)))

    print(f"  CV RMSE:       {cv_rmse:.3f} ± {cv_rmse_std:.3f}")
    print(f"  Baseline RMSE: {baseline_rmse:.3f} (global mean = {y.mean():.2f})")
    improvement = (1 - cv_rmse / baseline_rmse) * 100
    print(f"  Improvement:   {improvement:.1f}% over baseline")

    if cv_rmse >= baseline_rmse:
        print("  WARNING: model does NOT beat naive mean. Check training data.")

    # Fit on full dataset
    model.fit(X, y)

    # Top features
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("  Top 10 features:")
    for feat, imp in importances.head(10).items():
        print(f"    {feat}: {imp:.4f}")

    # Attach metadata
    model.cv_rmse_ = round(cv_rmse, 6)
    model.cv_rmse_std_ = round(cv_rmse_std, 6)
    model.baseline_rmse_ = round(baseline_rmse, 6)
    model.global_mean_ = round(float(y.mean()), 4)
    model.feature_names_ = list(X.columns)
    model.genre_names_ = [c.replace("genre_", "") for c in X.columns if c.startswith("genre_")]
    model.training_size_ = len(X)

    return model


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    all_genres = extract_genres(df)
    print(f"Genres ({len(all_genres)}): {all_genres}")

    # Movie model
    movies = df[df["type"] == "Movie"].copy()
    if len(movies) >= MIN_SAMPLES:
        X_m = build_movie_features(movies, all_genres)
        y_m = movies["imdb_score"].values
        model_m = train_model(X_m, y_m, "MOVIE")
        path_m = MODELS_DIR / "greenlight_movie_predictor.pkl"
        joblib.dump(model_m, path_m)
        print(f"\n  Saved → {path_m}")
    else:
        print(f"\nSKIPPED movie model: only {len(movies)} samples (need {MIN_SAMPLES})")

    # Show model
    shows = df[df["type"] == "Show"].copy()
    if len(shows) >= MIN_SAMPLES:
        X_s = build_show_features(shows, all_genres)
        y_s = shows["imdb_score"].values
        model_s = train_model(X_s, y_s, "SHOW")
        path_s = MODELS_DIR / "greenlight_show_predictor.pkl"
        joblib.dump(model_s, path_s)
        print(f"\n  Saved → {path_s}")
    else:
        print(f"\nSKIPPED show model: only {len(shows)} samples (need {MIN_SAMPLES})")

    print("\nDone!")


if __name__ == "__main__":
    main()
