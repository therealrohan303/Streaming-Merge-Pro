"""Analysis functions for the Interactive Lab page (Page 6).

Features:
  1. Build Your Streaming Service (budget game)
  2. Hypothetical Title Predictor (greenlight model)
  3. Insight Generator
"""

import random

import numpy as np
import pandas as pd

from src.analysis.scoring import bayesian_imdb, compute_quality_score
from src.config import ALL_PLATFORMS, MERGED_PLATFORMS, PLATFORMS


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

def predict_title(model, features_dict, genre_names):
    """Run prediction with the greenlight model.

    Returns prediction, uncertainty, feature importances.
    """
    if model is None:
        return None

    # Build feature vector matching model's training features
    feature_names = model.feature_names_
    X = pd.DataFrame([{f: 0 for f in feature_names}])

    # Set genre features
    for genre in features_dict.get("genres", []):
        col = f"genre_{genre}"
        if col in X.columns:
            X[col] = 1

    # Set numeric features
    for key in ["runtime", "release_year", "country_tier", "has_franchise", "budget_tier", "award_genre_avg"]:
        if key in features_dict and key in X.columns:
            X[key] = features_dict[key]

    # Predict
    prediction = model.predict(X)[0]

    # Uncertainty via individual tree predictions (for GBR)
    if hasattr(model, "estimators_"):
        tree_preds = []
        for stage in model.estimators_:
            tree_preds.append(stage[0].predict(X)[0])
        # Cumulative predictions for uncertainty
        cum_preds = np.cumsum(tree_preds) * model.learning_rate + model.init_.constant_[0]
        # Use last 20% of trees for uncertainty estimate
        n_last = max(1, len(cum_preds) // 5)
        uncertainty = np.std(cum_preds[-n_last:])
    else:
        uncertainty = 0.3  # Default

    # Feature importances
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances[importances > 0.01].sort_values(ascending=False)

    # Success tier
    if prediction >= 8.0:
        tier = "Blockbuster"
    elif prediction >= 7.0:
        tier = "Strong"
    elif prediction >= 6.0:
        tier = "Moderate"
    else:
        tier = "High Risk"

    return {
        "prediction": round(prediction, 2),
        "uncertainty": round(uncertainty, 2),
        "tier": tier,
        "importances": importances,
        "cv_rmse": getattr(model, "cv_rmse_", None),
        "baseline_rmse": getattr(model, "baseline_rmse_", None),
        "global_mean": getattr(model, "global_mean_", None),
    }


def get_talent_suggestions(principals_df, titles_df, genres, role="director", top_k=5):
    """Get top talent for given genres and role."""
    if principals_df.empty:
        return pd.DataFrame()

    # Map imdb_id to title info
    title_info = titles_df.set_index("imdb_id")[["imdb_score", "genres", "title"]].to_dict("index")

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
