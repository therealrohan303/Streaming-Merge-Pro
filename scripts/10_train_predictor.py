"""
Script 10: Train Greenlight Predictor models.

Trains separate GradientBoosting regressors for movies and shows to predict
IMDb scores based on title features.

Features:
  - genre_vector: binary vector of 19 genres
  - runtime, release_year
  - country_tier: US=3, major_markets=2, other=1
  - has_franchise: bool (from collection_name)
  - budget_tier: Low/Mid/High/Blockbuster quartiles (from Wikidata, where available)
  - award_genre_avg: avg award_wins in the title's primary genre

5-fold cross-validation. Reports RMSE and baseline RMSE (global mean).

Output:
  - models/greenlight_movie_predictor.pkl
  - models/greenlight_show_predictor.pkl
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

MAJOR_MARKETS = {"US", "GB", "FR", "DE", "JP", "KR", "IN", "CA", "AU", "IT", "ES"}
MIN_SAMPLES = 200  # Minimum samples to train a model


def load_data():
    path = ENRICHED_DIR / "titles_enriched.parquet"
    if not path.exists():
        print("ERROR: titles_enriched.parquet not found. Run script 09 first.")
        sys.exit(1)
    df = pd.read_parquet(path)
    # Only use titles with IMDb scores
    df = df[df["imdb_score"].notna()].copy()
    print(f"Loaded {len(df):,} titles with IMDb scores")
    return df


def extract_genres(df):
    """Get all unique genres across the dataset."""
    all_genres = set()
    for g_list in df["genres"].dropna():
        if isinstance(g_list, (list, np.ndarray)):
            all_genres.update(g_list)
    return sorted(all_genres)


def build_features(df, all_genres):
    """Build feature matrix from title data."""
    features = pd.DataFrame(index=df.index)

    # Genre binary vector
    for genre in all_genres:
        features[f"genre_{genre}"] = df["genres"].apply(
            lambda g: 1 if isinstance(g, (list, np.ndarray)) and genre in g else 0
        )

    # Runtime (fill missing with median)
    features["runtime"] = pd.to_numeric(df["runtime"], errors="coerce")
    features["runtime"] = features["runtime"].fillna(features["runtime"].median())

    # Release year
    features["release_year"] = df["release_year"].fillna(df["release_year"].median())

    # Country tier
    def get_country_tier(countries):
        if not isinstance(countries, (list, np.ndarray)) or len(countries) == 0:
            return 1
        top = countries[0] if isinstance(countries[0], str) else str(countries[0])
        if top == "US":
            return 3
        if top in MAJOR_MARKETS:
            return 2
        return 1

    features["country_tier"] = df["production_countries"].apply(get_country_tier)

    # Has franchise (from TMDB collection_name)
    if "collection_name" in df.columns:
        features["has_franchise"] = df["collection_name"].notna().astype(int)
    else:
        features["has_franchise"] = 0

    # Budget tier (from Wikidata)
    if "budget_usd" in df.columns:
        budget = df["budget_usd"].copy()
        budget_valid = budget[budget.notna() & (budget > 0)]
        if len(budget_valid) > 100:
            q25, q50, q75 = budget_valid.quantile([0.25, 0.5, 0.75])
            features["budget_tier"] = budget.apply(
                lambda b: 0 if pd.isna(b) or b <= 0
                else (1 if b <= q25 else (2 if b <= q50 else (3 if b <= q75 else 4)))
            )
        else:
            features["budget_tier"] = 0
    else:
        features["budget_tier"] = 0

    # Award genre average
    if "award_wins" in df.columns:
        # Compute avg award_wins per primary genre
        def primary_genre(g):
            if isinstance(g, (list, np.ndarray)) and len(g) > 0:
                return g[0]
            return "unknown"

        df_temp = df.copy()
        df_temp["_primary_genre"] = df_temp["genres"].apply(primary_genre)
        genre_award_avg = (
            df_temp[df_temp["award_wins"].notna() & (df_temp["award_wins"] > 0)]
            .groupby("_primary_genre")["award_wins"]
            .mean()
            .to_dict()
        )
        features["award_genre_avg"] = df_temp["_primary_genre"].map(genre_award_avg).fillna(0)
    else:
        features["award_genre_avg"] = 0

    return features


def train_model(X, y, model_type):
    """Train a GradientBoosting model with 5-fold CV."""
    print(f"\n{'='*60}")
    print(f"Training {model_type} model on {len(X):,} samples, {X.shape[1]} features")

    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42,
    )

    # 5-fold CV
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="neg_root_mean_squared_error")
    cv_rmse = -cv_scores.mean()
    cv_rmse_std = cv_scores.std()

    # Baseline: predict global mean
    baseline_rmse = np.sqrt(np.mean((y - y.mean()) ** 2))

    print(f"  CV RMSE:       {cv_rmse:.4f} (+/- {cv_rmse_std:.4f})")
    print(f"  Baseline RMSE: {baseline_rmse:.4f} (predict global mean {y.mean():.2f})")
    print(f"  Improvement:   {(1 - cv_rmse/baseline_rmse)*100:.1f}% over baseline")

    # Fit final model on all data
    model.fit(X, y)

    # Feature importances
    importances = pd.Series(model.feature_importances_, index=X.columns)
    top_features = importances.nlargest(10)
    print(f"\n  Top 10 features:")
    for feat, imp in top_features.items():
        print(f"    {feat}: {imp:.4f}")

    # Store metadata on the model object
    model.cv_rmse_ = cv_rmse
    model.cv_rmse_std_ = cv_rmse_std
    model.baseline_rmse_ = baseline_rmse
    model.feature_names_ = list(X.columns)
    model.genre_names_ = [c.replace("genre_", "") for c in X.columns if c.startswith("genre_")]
    model.global_mean_ = y.mean()

    return model


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    all_genres = extract_genres(df)
    print(f"Genres: {len(all_genres)}")

    for content_type, model_name, output_path in [
        ("Movie", "movie", MODELS_DIR / "greenlight_movie_predictor.pkl"),
        ("Show", "show", MODELS_DIR / "greenlight_show_predictor.pkl"),
    ]:
        subset = df[df["type"] == content_type].copy()
        if len(subset) < MIN_SAMPLES:
            print(f"\n  SKIPPED {model_name}: only {len(subset)} {content_type} titles "
                  f"(need {MIN_SAMPLES})")
            continue

        features = build_features(subset, all_genres)
        target = subset["imdb_score"].values

        model = train_model(features, target, model_name)
        joblib.dump(model, output_path)
        print(f"\n  Saved {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
