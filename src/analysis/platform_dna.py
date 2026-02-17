"""Platform DNA analysis: personality profiles, traits, and landscape clustering."""

import pandas as pd
import numpy as np

from src.analysis.scoring import compute_quality_score
from src.config import (
    ALL_PLATFORMS,
    DECADE_LABELS,
    DNA_MATCHER_GENRE_WEIGHT,
    DNA_MATCHER_SLIDER_WEIGHT,
    DNA_TOP_GENRES,
    DNA_TOP_TRAITS,
    MERGED_PLATFORMS,
    PLATFORMS,
    QUALITY_TIERS,
)


# Genre keys that need special display names (shared with comparisons module)
_GENRE_DISPLAY = {
    "Documentation": "Documentary",
    "Scifi": "Sci-Fi",
}


def _genre_label(raw: str) -> str:
    """Title-case a genre key with special-case fixes."""
    titled = raw.title()
    return _GENRE_DISPLAY.get(titled, titled)


def get_platform_titles(
    df: pd.DataFrame, platform_key: str
) -> pd.DataFrame:
    """Extract titles for a given platform key.

    For 'merged', combines netflix + max and deduplicates by id.
    """
    if platform_key == "merged":
        subset = df[df["platform"].isin(MERGED_PLATFORMS)].copy()
        subset = subset.drop_duplicates(subset="id", keep="first")
        return subset.reset_index(drop=True)
    return df[df["platform"] == platform_key].reset_index(drop=True)


def compute_genre_mix(df: pd.DataFrame, top_n: int = DNA_TOP_GENRES) -> pd.DataFrame:
    """Genre proportions for a single platform's titles.

    Returns DataFrame with columns: genre, count, pct — sorted by count descending.
    Top N genres kept; the rest grouped into 'Other'.
    """
    exploded = df[df["genres"].apply(lambda g: isinstance(g, list) and len(g) > 0)].copy()
    exploded = exploded.explode("genres")

    counts = exploded["genres"].value_counts()

    # Top N + Other
    top = counts.head(top_n)
    other_count = counts.iloc[top_n:].sum() if len(counts) > top_n else 0

    records = [
        {"genre": _genre_label(g), "count": int(c)}
        for g, c in top.items()
    ]
    if other_count > 0:
        records.append({"genre": "Other", "count": int(other_count)})

    result = pd.DataFrame(records)
    total = result["count"].sum()
    result["pct"] = (result["count"] / total * 100).round(1) if total > 0 else 0.0
    return result


def compute_era_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Decade distribution for a platform's titles.

    Returns DataFrame with columns: decade, count, pct — ordered chronologically.
    """
    if "decade" not in df.columns:
        return pd.DataFrame(columns=["decade", "count", "pct"])

    counts = df["decade"].value_counts()
    total = counts.sum()

    records = []
    for decade in DECADE_LABELS:
        c = int(counts.get(decade, 0))
        records.append({
            "decade": decade,
            "count": c,
            "pct": round(c / total * 100, 1) if total > 0 else 0.0,
        })

    return pd.DataFrame(records)


def compute_quality_profile(df: pd.DataFrame) -> dict:
    """Quality summary for a platform.

    Returns dict with:
        avg_imdb, median_imdb, tier_counts (dict[str, int]), tier_pcts (dict[str, float]),
        total_rated, total_titles
    """
    rated = df.dropna(subset=["imdb_score"])
    total = len(df)
    total_rated = len(rated)

    avg_imdb = round(float(rated["imdb_score"].mean()), 2) if total_rated > 0 else None
    median_imdb = round(float(rated["imdb_score"].median()), 2) if total_rated > 0 else None

    tier_counts = {}
    tier_pcts = {}
    for tier_name, (low, high) in QUALITY_TIERS.items():
        inclusive = "both" if high >= 10.0 else "left"
        count = int(rated["imdb_score"].between(low, high, inclusive=inclusive).sum())
        tier_counts[tier_name] = count
        tier_pcts[tier_name] = round(count / total_rated * 100, 1) if total_rated > 0 else 0.0

    return {
        "avg_imdb": avg_imdb,
        "median_imdb": median_imdb,
        "tier_counts": tier_counts,
        "tier_pcts": tier_pcts,
        "total_rated": total_rated,
        "total_titles": total,
    }


def _compute_all_platform_stats(all_df: pd.DataFrame) -> dict:
    """Compute per-platform stats for radar normalization and trait ranking.

    Returns dict mapping platform_key -> {
        median_year, avg_imdb, size, intl_pct, n_genres, show_pct,
        movie_pct, premium_count, premium_pct, intl_count
    }
    """
    per_plat = {}
    for key in ALL_PLATFORMS:
        plat = get_platform_titles(all_df, key)
        rated = plat.dropna(subset=["imdb_score"])
        valid_countries = plat[plat["production_countries"].apply(
            lambda c: isinstance(c, list) and len(c) > 0
        )]
        intl_count = 0
        intl_pct = 0.0
        if len(valid_countries) > 0:
            intl_mask = valid_countries["production_countries"].apply(lambda c: "US" not in c)
            intl_count = int(intl_mask.sum())
            intl_pct = intl_mask.mean() * 100

        genre_set = set()
        for g in plat["genres"].dropna():
            if isinstance(g, list):
                genre_set.update(g)

        premium_count = int((rated["imdb_score"] >= 7.0).sum()) if len(rated) > 0 else 0
        premium_pct = premium_count / len(rated) * 100 if len(rated) > 0 else 0

        movie_pct = (plat["type"] == "Movie").mean() * 100 if len(plat) > 0 else 50
        show_pct = 100 - movie_pct

        per_plat[key] = {
            "median_year": float(plat["release_year"].median()) if len(plat) > 0 else 2000,
            "avg_imdb": float(rated["imdb_score"].mean()) if len(rated) > 0 else 5.0,
            "size": len(plat),
            "intl_pct": intl_pct,
            "intl_count": intl_count,
            "n_genres": len(genre_set),
            "show_pct": show_pct,
            "movie_pct": movie_pct,
            "premium_count": premium_count,
            "premium_pct": premium_pct,
        }
    return per_plat


def _rank_among(value: float, all_values: list[float], higher_is_better: bool = True) -> str:
    """Return a rank label like 'highest of all 6' or '2nd highest'."""
    if higher_is_better:
        rank = sum(1 for v in all_values if v > value) + 1
    else:
        rank = sum(1 for v in all_values if v < value) + 1
    if rank == 1:
        return "highest of all 6 platforms"
    elif rank == 2:
        return "2nd highest"
    elif rank == len(all_values):
        return "lowest of all 6 platforms" if higher_is_better else "highest of all 6 platforms"
    else:
        ordinals = {3: "3rd", 4: "4th", 5: "5th", 6: "6th"}
        return f"{ordinals.get(rank, f'{rank}th')}"


def compute_platform_identity_radar(
    platform_df: pd.DataFrame, all_df: pd.DataFrame
) -> dict:
    """Compute 6 identity dimensions as 0-100 scores for radar chart.

    Dimensions:
      - Freshness: how recent the catalog skews (median release year)
      - Quality: average IMDb score mapped to 0-100
      - Breadth: catalog size relative to largest platform
      - Global Reach: % international (non-US) content
      - Genre Diversity: unique genre count relative to max possible
      - Series Focus: % of catalog that is shows (vs movies)
    """
    per_plat = _compute_all_platform_stats(all_df)

    # Get ranges for normalization
    def _norm(val, all_vals):
        lo, hi = min(all_vals), max(all_vals)
        if hi == lo:
            return 50.0
        return round(max(0, min(100, (val - lo) / (hi - lo) * 100)), 1)

    all_years = [v["median_year"] for v in per_plat.values()]
    all_imdb = [v["avg_imdb"] for v in per_plat.values()]
    all_sizes = [v["size"] for v in per_plat.values()]
    all_intl = [v["intl_pct"] for v in per_plat.values()]
    all_genres = [v["n_genres"] for v in per_plat.values()]
    all_shows = [v["show_pct"] for v in per_plat.values()]

    # Compute platform-specific values
    plat_rated = platform_df.dropna(subset=["imdb_score"])
    valid_countries = platform_df[platform_df["production_countries"].apply(
        lambda c: isinstance(c, list) and len(c) > 0
    )]
    plat_intl = (
        valid_countries["production_countries"].apply(lambda c: "US" not in c).mean() * 100
        if len(valid_countries) > 0 else 0.0
    )
    plat_genre_set = set()
    for g in platform_df["genres"].dropna():
        if isinstance(g, list):
            plat_genre_set.update(g)

    median_year = float(platform_df["release_year"].median()) if len(platform_df) > 0 else 2000
    avg_imdb = float(plat_rated["imdb_score"].mean()) if len(plat_rated) > 0 else 5.0
    show_pct = (platform_df["type"] == "Show").mean() * 100 if len(platform_df) > 0 else 50

    return {
        "Freshness": _norm(median_year, all_years),
        "Quality": _norm(avg_imdb, all_imdb),
        "Breadth": _norm(len(platform_df), all_sizes),
        "Global Reach": _norm(plat_intl, all_intl),
        "Genre Diversity": _norm(len(plat_genre_set), all_genres),
        "Series Focus": _norm(show_pct, all_shows),
    }


def compute_defining_traits(
    platform_df: pd.DataFrame,
    all_df: pd.DataFrame,
    platform_key: str,
    max_traits: int = DNA_TOP_TRAITS,
) -> list[dict]:
    """Generate 3-5 defining traits that make this platform distinctive.

    Uses narrative-style labels and detail strings with personality.
    Each trait has: label, detail, direction.
    """
    traits = []
    per_plat = _compute_all_platform_stats(all_df)
    global_df = all_df.drop_duplicates(subset="id", keep="first")
    name = PLATFORMS.get(platform_key, {}).get("name", platform_key.title())

    # Current platform stats (use pre-computed if available, else compute)
    if platform_key in per_plat:
        ps = per_plat[platform_key]
    else:
        # "merged" key — compute fresh
        rated = platform_df.dropna(subset=["imdb_score"])
        valid_c = platform_df[platform_df["production_countries"].apply(
            lambda c: isinstance(c, list) and len(c) > 0
        )]
        intl_mask = valid_c["production_countries"].apply(lambda c: "US" not in c) if len(valid_c) > 0 else pd.Series(dtype=bool)
        genre_set = set()
        for g in platform_df["genres"].dropna():
            if isinstance(g, list):
                genre_set.update(g)
        premium_n = int((rated["imdb_score"] >= 7.0).sum()) if len(rated) > 0 else 0
        ps = {
            "median_year": float(platform_df["release_year"].median()) if len(platform_df) > 0 else 2000,
            "avg_imdb": float(rated["imdb_score"].mean()) if len(rated) > 0 else 5.0,
            "size": len(platform_df),
            "intl_pct": float(intl_mask.mean() * 100) if len(intl_mask) > 0 else 0,
            "intl_count": int(intl_mask.sum()) if len(intl_mask) > 0 else 0,
            "n_genres": len(genre_set),
            "show_pct": (platform_df["type"] == "Show").mean() * 100 if len(platform_df) > 0 else 50,
            "movie_pct": (platform_df["type"] == "Movie").mean() * 100 if len(platform_df) > 0 else 50,
            "premium_count": premium_n,
            "premium_pct": premium_n / len(rated) * 100 if len(rated) > 0 else 0,
        }

    # Global averages
    avg_size = sum(v["size"] for v in per_plat.values()) / len(per_plat)
    avg_imdb = sum(v["avg_imdb"] for v in per_plat.values()) / len(per_plat)
    avg_year = sum(v["median_year"] for v in per_plat.values()) / len(per_plat)
    avg_intl = sum(v["intl_pct"] for v in per_plat.values()) / len(per_plat)
    avg_movie = sum(v["movie_pct"] for v in per_plat.values()) / len(per_plat)

    # --- Trait 1: Catalog scale ---
    ratio = ps["size"] / avg_size if avg_size > 0 else 1.0
    size_rank = _rank_among(ps["size"], [v["size"] for v in per_plat.values()])
    if ratio > 1.5:
        traits.append({
            "label": "The Content Giant",
            "detail": (
                f"With {ps['size']:,} titles, {name} has the "
                f"{size_rank} library in streaming — "
                f"{ratio:.1f}x larger than the average platform"
            ),
            "direction": "high",
        })
    elif ratio < 0.5:
        traits.append({
            "label": "Small but Mighty",
            "detail": (
                f"{name} keeps it tight with just {ps['size']:,} titles — "
                f"a curated collection {1/ratio:.1f}x smaller than average"
            ),
            "direction": "neutral",
        })

    # --- Trait 2: Quality positioning ---
    imdb_diff = ps["avg_imdb"] - avg_imdb
    quality_rank = _rank_among(ps["avg_imdb"], [v["avg_imdb"] for v in per_plat.values()])
    if imdb_diff > 0.25:
        traits.append({
            "label": "The Quality Standard",
            "detail": (
                f"{ps['premium_count']:,} titles score 7.0+ on IMDb "
                f"({ps['premium_pct']:.1f}% of the catalog) — "
                f"{name} has the {quality_rank} average rating in streaming"
            ),
            "direction": "high",
        })
    elif imdb_diff < -0.25:
        traits.append({
            "label": "Quantity Over Quality",
            "detail": (
                f"At {ps['avg_imdb']:.2f} avg IMDb, {name} prioritizes "
                f"breadth over selectivity — {abs(imdb_diff):.2f} below "
                f"the streaming average"
            ),
            "direction": "low",
        })

    # --- Trait 3: Content freshness ---
    year_diff = ps["median_year"] - avg_year
    fresh_rank = _rank_among(ps["median_year"], [v["median_year"] for v in per_plat.values()])
    if year_diff >= 3:
        recent_count = int((platform_df["release_year"] >= 2018).sum()) if len(platform_df) > 0 else 0
        recent_pct = recent_count / len(platform_df) * 100 if len(platform_df) > 0 else 0
        traits.append({
            "label": "Always Fresh",
            "detail": (
                f"{recent_pct:.1f}% of {name}'s catalog is from 2018 or later — "
                f"{int(abs(year_diff))} years newer than the industry median"
            ),
            "direction": "high",
        })
    elif year_diff <= -3:
        traits.append({
            "label": "The Timeless Library",
            "detail": (
                f"{name} reaches back to a median {int(ps['median_year'])} "
                f"release — {int(abs(year_diff))} years deeper than most platforms"
            ),
            "direction": "low",
        })

    # --- Trait 4: Signature genre ---
    plat_genres_df = platform_df[platform_df["genres"].apply(lambda g: isinstance(g, list) and len(g) > 0)]
    plat_exploded = plat_genres_df.explode("genres")
    global_genres_df = global_df[global_df["genres"].apply(lambda g: isinstance(g, list) and len(g) > 0)]
    global_exploded = global_genres_df.explode("genres")

    if len(plat_exploded) > 0 and len(global_exploded) > 0:
        plat_dist = plat_exploded["genres"].value_counts(normalize=True)
        plat_counts = plat_exploded["genres"].value_counts()
        global_dist = global_exploded["genres"].value_counts(normalize=True)

        deviations = {}
        for genre in plat_dist.index:
            plat_share = plat_dist.get(genre, 0)
            global_share = global_dist.get(genre, 0)
            if global_share > 0.01:
                deviations[genre] = plat_share - global_share

        if deviations:
            top_genre = max(deviations, key=deviations.get)
            dev = deviations[top_genre]
            if dev > 0.03:
                genre_count = int(plat_counts.get(top_genre, 0))
                plat_pct = plat_dist[top_genre] * 100
                genre_name = _genre_label(top_genre)
                traits.append({
                    "label": f"The {genre_name} Destination",
                    "detail": (
                        f"{genre_count:,} {genre_name.lower()} titles make up "
                        f"{plat_pct:.1f}% of {name}'s library — "
                        f"{dev*100:+.1f} percentage points above the industry norm"
                    ),
                    "direction": "high",
                })

    # --- Trait 5: International diversity ---
    intl_diff = ps["intl_pct"] - avg_intl
    intl_rank = _rank_among(ps["intl_pct"], [v["intl_pct"] for v in per_plat.values()])
    if intl_diff > 10:
        traits.append({
            "label": "The World's Screen",
            "detail": (
                f"{ps['intl_pct']:.1f}% of {name}'s catalog "
                f"({ps['intl_count']:,} titles) comes from outside the US — "
                f"{intl_rank} for international content"
            ),
            "direction": "high",
        })
    elif intl_diff < -10:
        traits.append({
            "label": "Made in America",
            "detail": (
                f"{100-ps['intl_pct']:.1f}% domestic content — "
                f"{name} is firmly focused on the US market"
            ),
            "direction": "low",
        })

    # --- Trait 6: Content type mix ---
    movie_diff = ps["movie_pct"] - avg_movie
    movie_count = int((platform_df["type"] == "Movie").sum()) if len(platform_df) > 0 else 0
    show_count = int((platform_df["type"] == "Show").sum()) if len(platform_df) > 0 else 0
    movie_rank = _rank_among(ps["movie_pct"], [v["movie_pct"] for v in per_plat.values()])
    if movie_diff > 10:
        traits.append({
            "label": "The Silver Screen",
            "detail": (
                f"{movie_count:,} movies make up {ps['movie_pct']:.1f}% of "
                f"{name}'s library — the {movie_rank} movie focus in streaming"
            ),
            "direction": "neutral",
        })
    elif movie_diff < -10:
        show_rank = _rank_among(ps["show_pct"], [v["show_pct"] for v in per_plat.values()])
        traits.append({
            "label": "Binge-Watch Central",
            "detail": (
                f"{show_count:,} series make up {ps['show_pct']:.1f}% of "
                f"{name}'s library — {show_rank} for TV show lovers"
            ),
            "direction": "neutral",
        })

    return traits[:max_traits]


def compute_platform_comparison_data(
    all_df: pd.DataFrame,
    platform_keys: list[str],
) -> dict:
    """Compute profile data for multiple platforms for side-by-side comparison.

    Returns dict mapping platform_key -> {
        'display_name', 'genre_mix', 'era_profile', 'quality_profile', 'traits'
    }
    """
    result = {}
    for key in platform_keys:
        plat_df = get_platform_titles(all_df, key)
        display_name = PLATFORMS.get(key, {}).get("name", key.title())
        result[key] = {
            "display_name": display_name,
            "genre_mix": compute_genre_mix(plat_df),
            "era_profile": compute_era_profile(plat_df),
            "quality_profile": compute_quality_profile(plat_df),
            "radar": compute_platform_identity_radar(plat_df, all_df),
            "traits": compute_defining_traits(plat_df, all_df, key),
            "title_count": len(plat_df),
        }
    return result


def compute_landscape_clusters(
    umap_df: pd.DataFrame,
    titles_df: pd.DataFrame,
    n_clusters: int = 8,
) -> pd.DataFrame:
    """Label UMAP points with cluster IDs using KMeans.

    Returns merged DataFrame with umap_x, umap_y, cluster, and title metadata.
    """
    from sklearn.cluster import KMeans

    coords = umap_df[["umap_x", "umap_y"]].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    umap_df = umap_df.copy()
    umap_df["cluster"] = kmeans.fit_predict(coords)

    # Merge with title metadata
    keep_cols = ["id", "title", "platform", "type", "genres", "imdb_score",
                 "imdb_votes", "tmdb_popularity", "release_year", "age_certification",
                 "runtime", "production_countries"]
    keep_cols = [c for c in keep_cols if c in titles_df.columns]
    meta = titles_df[keep_cols].drop_duplicates(subset="id", keep="first")
    merged = umap_df.merge(meta, on="id", how="left")

    return merged


_FAMILY_RATINGS = {"G", "PG", "TV-Y", "TV-Y7", "TV-G", "TV-PG"}
_MATURE_CLUSTER_RATINGS = {"R", "NC-17", "TV-MA", "TV-14"}


def generate_cluster_name(
    top_genres: list[str],
    avg_year: float | None,
    avg_imdb: float | None,
    movie_pct: float = 50.0,
    dominant_platform: str | None = None,
    used_names: set | None = None,
    cluster_df: pd.DataFrame | None = None,
) -> str:
    """Generate a unique, thematic name for a content cluster.

    Uses rule-based archetype matching on genre, quality, audience,
    and geography signals. Each name is human-readable and instantly
    descriptive of the cluster's character.
    """
    if not top_genres:
        return "Mixed Collection"

    _used = used_names or set()

    # Compute cluster signals from the DataFrame
    intl_pct = 0.0
    family_pct = 0.0
    mature_pct = 0.0
    median_votes = 0.0
    if cluster_df is not None and len(cluster_df) > 0:
        # International %
        valid_c = cluster_df[cluster_df["production_countries"].apply(
            lambda c: isinstance(c, list) and len(c) > 0
        )] if "production_countries" in cluster_df.columns else pd.DataFrame()
        if len(valid_c) > 0:
            intl_pct = valid_c["production_countries"].apply(
                lambda c: "US" not in c
            ).mean() * 100

        # Audience certification breakdown
        certs = cluster_df["age_certification"].dropna() if "age_certification" in cluster_df.columns else pd.Series(dtype=str)
        if len(certs) > 0:
            family_pct = certs.isin(_FAMILY_RATINGS).mean() * 100
            mature_pct = certs.isin(_MATURE_CLUSTER_RATINGS).mean() * 100

        # Median vote count (popularity proxy)
        if "imdb_votes" in cluster_df.columns:
            votes = cluster_df["imdb_votes"].dropna()
            median_votes = float(votes.median()) if len(votes) > 0 else 0.0

    # Lowercase genre set for matching (top_genres are already title-cased)
    g_lower = {g.lower() for g in top_genres[:3]}
    g_top2 = {g.lower() for g in top_genres[:2]}

    # ---- Ordered archetype rules (first match wins) ----
    candidate = None

    if "animation" in g_top2 or family_pct > 60:
        candidate = "Kids & Animation Hub"
    elif "documentary" in g_lower or "documentation" in g_lower:
        candidate = "Documentary & Factual Hub"
    elif ("crime" in g_top2 or "thriller" in g_top2) and mature_pct > 40:
        candidate = "Dark Thriller Hub"
    elif "drama" in g_top2 and avg_imdb is not None and avg_imdb >= 7.0:
        candidate = "Prestige Drama Hub"
    elif "comedy" in g_top2 and family_pct > 35:
        candidate = "Family Comedy Hub"
    elif "scifi" in g_lower or "fantasy" in g_lower:
        candidate = "Sci-Fi & Fantasy Hub"
    elif "horror" in g_top2:
        candidate = "Horror & Supernatural Hub"
    elif intl_pct > 55:
        candidate = "International Cinema Hub"
    elif "action" in g_top2:
        candidate = "Action & Adventure Hub"
    elif "romance" in g_top2 and "drama" in g_top2:
        candidate = "Romantic Drama Hub"
    elif "reality" in g_lower:
        candidate = "Reality & Lifestyle Hub"
    elif (avg_imdb is not None and avg_imdb >= 6.5
          and median_votes < 5000 and movie_pct > 55):
        candidate = "Indie & Art House Hub"
    elif avg_year is not None and avg_year < 2005 and movie_pct > 55:
        candidate = "Classic Cinema Hub"
    elif median_votes > 20000:
        candidate = "Mainstream Hits Hub"
    else:
        # Fallback: use top genre
        candidate = f"{top_genres[0]} & {top_genres[1]} Hub" if len(top_genres) >= 2 else f"{top_genres[0]} Hub"

    # ---- Deduplication ----
    if candidate not in _used:
        _used.add(candidate)
        return candidate

    # Append a differentiator
    differentiators = []
    if intl_pct > 40:
        differentiators.append("International")
    if movie_pct < 40:
        differentiators.append("Series")
    elif movie_pct > 65:
        differentiators.append("Cinema")
    if avg_year is not None and avg_year < 2010:
        differentiators.append("Classics")
    if avg_imdb is not None and avg_imdb >= 7.0:
        differentiators.append("Premium")
    differentiators.append(f"Region {len(_used) + 1}")

    for diff in differentiators:
        name = f"{candidate.replace(' Hub', '')} Hub ({diff})"
        if name not in _used:
            _used.add(name)
            return name

    # Absolute fallback
    fallback = f"{candidate} #{len(_used) + 1}"
    _used.add(fallback)
    return fallback


def compute_cluster_summaries(landscape_df: pd.DataFrame) -> dict:
    """Summarize each cluster: dominant genres, platform mix, sample titles.

    Returns dict mapping cluster_id -> {
        'name': str, 'top_genres': list[str], 'platform_mix': dict[str, int],
        'avg_imdb': float, 'size': int, 'sample_titles': list[str],
        'platform_leader': str | None, 'quality_leader': str | None,
    }
    """
    used_names: set[str] = set()
    summaries = {}
    for cluster_id, grp in landscape_df.groupby("cluster"):
        # Top genres
        genre_counts = {}
        for genres in grp["genres"].dropna():
            if isinstance(genres, list):
                for g in genres:
                    genre_counts[g] = genre_counts.get(g, 0) + 1
        top_genres_raw = sorted(genre_counts, key=genre_counts.get, reverse=True)[:4]
        top_genres = [_genre_label(g) for g in top_genres_raw]

        # Platform mix
        platform_mix = grp["platform"].value_counts().to_dict() if "platform" in grp.columns else {}

        # Dominant platform (for naming)
        dominant_platform = None
        if platform_mix:
            top_plat = max(platform_mix, key=platform_mix.get)
            top_plat_pct = platform_mix[top_plat] / len(grp) * 100
            if top_plat_pct > 40:
                dominant_platform = top_plat

        # Platform leader (most titles in cluster, shown if >30%)
        platform_leader = None
        if platform_mix:
            leader = max(platform_mix, key=platform_mix.get)
            if platform_mix[leader] / len(grp) * 100 > 30:
                platform_leader = leader

        # Avg IMDb
        rated = grp.dropna(subset=["imdb_score"])
        avg_imdb = round(float(rated["imdb_score"].mean()), 2) if len(rated) > 0 else None

        # Quality leader (platform with highest avg IMDb in this cluster, min 5 titles)
        quality_leader = None
        if "platform" in grp.columns and len(rated) > 0:
            plat_quality = {}
            for pk, pg in rated.groupby("platform"):
                if len(pg) >= 5:
                    plat_quality[pk] = pg["imdb_score"].mean()
            if plat_quality:
                quality_leader = max(plat_quality, key=plat_quality.get)

        # Avg year
        avg_year = float(grp["release_year"].mean()) if "release_year" in grp.columns else None

        # Content type mix
        movie_pct = (grp["type"] == "Movie").mean() * 100 if "type" in grp.columns else 50.0

        # Sample titles — ranked by quality_score with progressive vote thresholds
        if len(rated) > 0:
            scored = rated.copy()
            scored["quality_score"] = compute_quality_score(scored)
            for threshold in [10_000, 1_000, 0]:
                pool = scored[scored["imdb_votes"].fillna(0) >= threshold]
                if len(pool) >= 5:
                    break
            else:
                pool = scored
            sample = pool.nlargest(5, "quality_score")["title"].tolist()
        else:
            sample = []

        name = generate_cluster_name(
            top_genres, avg_year, avg_imdb,
            movie_pct=movie_pct,
            dominant_platform=dominant_platform,
            used_names=used_names,
            cluster_df=grp,
        )

        summaries[cluster_id] = {
            "name": name,
            "top_genres": top_genres[:3],
            "platform_mix": platform_mix,
            "avg_imdb": avg_imdb,
            "avg_year": avg_year,
            "size": len(grp),
            "sample_titles": sample,
            "platform_leader": platform_leader,
            "quality_leader": quality_leader,
        }

    return summaries


def compute_landscape_insights(
    landscape_df: pd.DataFrame,
    cluster_summaries: dict,
) -> list[dict]:
    """Generate 3-4 human-readable insights about the content landscape.

    Returns list of dicts with 'icon' (str) and 'text' (str with HTML spans
    for platform coloring).
    """
    import itertools

    insights: list[dict] = []
    if "platform" not in landscape_df.columns:
        return insights

    # Platform centroids
    plat_centroids = {}
    for pk in landscape_df["platform"].unique():
        pk_data = landscape_df[landscape_df["platform"] == pk]
        if len(pk_data) >= 10:
            plat_centroids[pk] = (pk_data["umap_x"].mean(), pk_data["umap_y"].mean())

    def _colored(pk: str) -> str:
        c = PLATFORMS.get(pk, {}).get("color", "#ddd")
        n = PLATFORMS.get(pk, {}).get("name", pk)
        return f'<span style="color:{c};font-weight:700;">{n}</span>'

    # 1. Most separated platforms
    if len(plat_centroids) >= 2:
        max_dist = 0
        most_sep = ("", "")
        min_dist = float("inf")
        most_ovl = ("", "")
        for a, b in itertools.combinations(plat_centroids, 2):
            dx = plat_centroids[a][0] - plat_centroids[b][0]
            dy = plat_centroids[a][1] - plat_centroids[b][1]
            dist = (dx**2 + dy**2) ** 0.5
            if dist > max_dist:
                max_dist = dist
                most_sep = (a, b)
            if dist < min_dist:
                min_dist = dist
                most_ovl = (a, b)

        # Find top neighborhoods for separated platforms
        sep_a_top = landscape_df[landscape_df["platform"] == most_sep[0]]["cluster"].value_counts()
        sep_b_top = landscape_df[landscape_df["platform"] == most_sep[1]]["cluster"].value_counts()
        hood_a = cluster_summaries.get(sep_a_top.index[0], {}).get("name", "") if len(sep_a_top) > 0 else ""
        hood_b = cluster_summaries.get(sep_b_top.index[0], {}).get("name", "") if len(sep_b_top) > 0 else ""

        text = f"{_colored(most_sep[0])} and {_colored(most_sep[1])} live in different worlds"
        if hood_a and hood_b:
            text += (
                f" — {PLATFORMS.get(most_sep[0], {}).get('name', most_sep[0])} "
                f"dominates <em>{hood_a}</em> while "
                f"{PLATFORMS.get(most_sep[1], {}).get('name', most_sep[1])} "
                f"owns <em>{hood_b}</em>"
            )
        insights.append({"icon": "diverge", "text": text})

        # 2. Most overlapping
        text_ovl = (
            f"{_colored(most_ovl[0])} and {_colored(most_ovl[1])} "
            f"compete on the same turf — their content catalogs cluster "
            f"in the same neighborhoods"
        )
        insights.append({"icon": "overlap", "text": text_ovl})

    # 3. Most focused vs most diffuse platform
    plat_cluster_dist = {}
    for pk in plat_centroids:
        pk_data = landscape_df[landscape_df["platform"] == pk]
        top_cluster_pct = pk_data["cluster"].value_counts(normalize=True).iloc[0] * 100 if len(pk_data) > 0 else 0
        top_cluster_id = pk_data["cluster"].value_counts().index[0] if len(pk_data) > 0 else None
        plat_cluster_dist[pk] = {"top_pct": top_cluster_pct, "top_id": top_cluster_id}

    if plat_cluster_dist:
        most_focused = max(plat_cluster_dist, key=lambda k: plat_cluster_dist[k]["top_pct"])
        most_spread = min(plat_cluster_dist, key=lambda k: plat_cluster_dist[k]["top_pct"])
        f_pct = plat_cluster_dist[most_focused]["top_pct"]
        f_hood = cluster_summaries.get(plat_cluster_dist[most_focused]["top_id"], {}).get("name", "")
        s_pct = plat_cluster_dist[most_spread]["top_pct"]

        text = (
            f"{_colored(most_focused)} is the most focused platform — "
            f"{f_pct:.0f}% of its content sits in <em>{f_hood}</em>. "
            f"{_colored(most_spread)} is the most diverse, "
            f"with no single neighborhood exceeding {s_pct:.0f}%"
        )
        insights.append({"icon": "focus", "text": text})

    # 4. Cluster ownership (any platform >50% of a cluster)
    for cid, summary in cluster_summaries.items():
        pmix = summary.get("platform_mix", {})
        total = summary.get("size", 1)
        for pk, count in pmix.items():
            pct = count / total * 100
            if pct > 50:
                text = (
                    f"{_colored(pk)} nearly owns <em>{summary['name']}</em> "
                    f"with {pct:.0f}% of all titles there"
                )
                insights.append({"icon": "crown", "text": text})
                break  # one ownership insight per cluster max

    return insights[:4]  # Cap at 4 insights


def compute_overlap_stats(landscape_df: pd.DataFrame) -> dict:
    """Compute Netflix vs Max overlap/divergence stats from UMAP clusters.

    Returns dict with:
        'shared_clusters': clusters where both platforms have >10% presence
        'netflix_dominated': clusters where Netflix > 60%
        'max_dominated': clusters where Max > 60%
        'overlap_pct': % of clusters with significant shared presence
    """
    if "platform" not in landscape_df.columns:
        return {}

    clusters = landscape_df.groupby("cluster")
    shared = []
    netflix_dom = []
    max_dom = []

    for cluster_id, grp in clusters:
        plat_counts = grp["platform"].value_counts()
        total = len(grp)
        nf_pct = plat_counts.get("netflix", 0) / total * 100
        mx_pct = plat_counts.get("max", 0) / total * 100

        if nf_pct > 10 and mx_pct > 10:
            shared.append(cluster_id)
        if nf_pct > 60:
            netflix_dom.append(cluster_id)
        if mx_pct > 60:
            max_dom.append(cluster_id)

    total_clusters = landscape_df["cluster"].nunique()
    return {
        "shared_clusters": shared,
        "netflix_dominated": netflix_dom,
        "max_dominated": max_dom,
        "overlap_pct": round(len(shared) / total_clusters * 100, 1) if total_clusters > 0 else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PLATFORM MATCHER
# ═══════════════════════════════════════════════════════════════════════════════

# Mature ratings for maturity scoring
_MATURE_RATINGS = {"R", "NC-17", "TV-MA", "TV-14"}


def compute_platform_profile_vector(
    all_df: pd.DataFrame, platform_key: str
) -> dict:
    """Compute a platform's profile as normalized 0-100 dimension values.

    Dimensions:
      - genre_dist: dict[str, float] (genre -> % of catalog, normalized)
      - recency: 0=oldest median year, 100=newest
      - popularity: 0=lowest avg IMDb, 100=highest
      - runtime: 0=shortest median runtime, 100=longest
      - maturity: 0=most family-friendly, 100=most mature
      - content_type: 0=all shows, 100=all movies
      - international: 0=all domestic, 100=all international
    """
    plat_df = get_platform_titles(all_df, platform_key)
    if len(plat_df) == 0:
        return {"genre_dist": {}, "recency": 50, "popularity": 50,
                "runtime": 50, "maturity": 50, "content_type": 50,
                "international": 50}

    # Genre distribution
    genre_counts = {}
    for genres in plat_df["genres"].dropna():
        if isinstance(genres, list):
            for g in genres:
                genre_counts[g] = genre_counts.get(g, 0) + 1
    total_genre_tags = sum(genre_counts.values()) or 1
    genre_dist = {g: c / total_genre_tags for g, c in genre_counts.items()}

    # Recency: median release year → 0-100 (1940=0, 2023=100)
    median_year = float(plat_df["release_year"].median())
    recency = max(0, min(100, (median_year - 1940) / (2023 - 1940) * 100))

    # Popularity: avg IMDb → 0-100 (4.0=0, 9.0=100)
    rated = plat_df.dropna(subset=["imdb_score"])
    avg_imdb = float(rated["imdb_score"].mean()) if len(rated) > 0 else 6.0
    popularity = max(0, min(100, (avg_imdb - 4.0) / (9.0 - 4.0) * 100))

    # Runtime: median runtime → 0-100 (30=0, 150=100)
    median_runtime = float(plat_df["runtime"].median()) if plat_df["runtime"].notna().any() else 80
    runtime = max(0, min(100, (median_runtime - 30) / (150 - 30) * 100))

    # Maturity: % of titles with mature ratings → 0-100
    cert = plat_df["age_certification"].dropna()
    if len(cert) > 0:
        maturity = cert.isin(_MATURE_RATINGS).mean() * 100
    else:
        maturity = 50.0  # neutral fallback

    # Content type: movie % → 0-100
    content_type = (plat_df["type"] == "Movie").mean() * 100

    # International: % non-US → 0-100
    valid = plat_df[plat_df["production_countries"].apply(
        lambda c: isinstance(c, list) and len(c) > 0
    )]
    if len(valid) > 0:
        international = valid["production_countries"].apply(
            lambda c: "US" not in c
        ).mean() * 100
    else:
        international = 50.0

    return {
        "genre_dist": genre_dist,
        "recency": round(recency, 1),
        "popularity": round(popularity, 1),
        "runtime": round(runtime, 1),
        "maturity": round(maturity, 1),
        "content_type": round(content_type, 1),
        "international": round(international, 1),
    }


def compute_user_match_scores(
    user_prefs: dict, all_df: pd.DataFrame
) -> list[dict]:
    """Match user preferences against all 6 platform profiles.

    Args:
        user_prefs: {
            'genres': list[str],   # selected genre keys (lowercase)
            'recency': float,      # 0-100 slider
            'popularity': float,   # 0-100 slider
            'runtime': float,      # 0-100 slider
            'maturity': float,     # 0-100 slider
            'content_type': float, # 0-100 slider
            'international': float # 0-100 slider
        }

    Returns:
        list of dicts sorted by match_pct descending:
        [{'platform': key, 'display_name': str, 'match_pct': float,
          'explanation': str}, ...]
    """
    slider_dims = ["recency", "popularity", "runtime", "maturity",
                   "content_type", "international"]
    slider_labels = {
        "recency": ("classic catalog", "fresh releases"),
        "popularity": ("hidden gems", "popular hits"),
        "runtime": ("shorter content", "longer content"),
        "maturity": ("family-friendly titles", "mature content"),
        "content_type": ("series focus", "movie focus"),
        "international": ("domestic focus", "international catalog"),
    }
    n_sliders = len(slider_dims)
    per_slider_weight = DNA_MATCHER_SLIDER_WEIGHT / n_sliders

    results = []
    for key in ALL_PLATFORMS:
        profile = compute_platform_profile_vector(all_df, key)

        # Genre similarity (cosine-like: intersection / union)
        user_genres = set(user_prefs.get("genres", []))
        plat_genres = profile["genre_dist"]
        if user_genres and plat_genres:
            # Weighted overlap: sum of platform genre shares for user's selected genres
            overlap = sum(plat_genres.get(g, 0) for g in user_genres)
            max_possible = sum(sorted(plat_genres.values(), reverse=True)[:len(user_genres)])
            genre_score = (overlap / max_possible * 100) if max_possible > 0 else 0
        else:
            genre_score = 50  # neutral if no genres selected

        # Slider dimension scores
        dim_scores = {}
        for dim in slider_dims:
            user_val = user_prefs.get(dim, 50)
            plat_val = profile.get(dim, 50)
            dim_scores[dim] = max(0, 100 - abs(user_val - plat_val))

        # Weighted total
        total = (
            DNA_MATCHER_GENRE_WEIGHT * genre_score
            + sum(per_slider_weight * dim_scores[d] for d in slider_dims)
        )

        # Generate explanation: highlight top 2 strengths and top weakness
        sorted_dims = sorted(dim_scores.items(), key=lambda x: x[1], reverse=True)
        strengths = []
        for dim, score in sorted_dims[:2]:
            if score >= 60:
                user_val = user_prefs.get(dim, 50)
                label = slider_labels[dim][1] if user_val >= 50 else slider_labels[dim][0]
                strengths.append(label)

        weakest_dim, weakest_score = sorted_dims[-1]
        weakness = None
        if weakest_score < 40:
            user_val = user_prefs.get(weakest_dim, 50)
            weakness = slider_labels[weakest_dim][1] if user_val >= 50 else slider_labels[weakest_dim][0]

        explanation_parts = []
        if strengths:
            explanation_parts.append(f"Strong match on {' and '.join(strengths)}")
        if genre_score >= 70:
            explanation_parts.append("great genre alignment")
        elif genre_score < 30:
            explanation_parts.append("limited genre overlap")
        if weakness:
            explanation_parts.append(f"less aligned on {weakness}")

        explanation = ". ".join(explanation_parts) + "." if explanation_parts else "Moderate match across dimensions."

        results.append({
            "platform": key,
            "display_name": PLATFORMS.get(key, {}).get("name", key.title()),
            "match_pct": round(total, 1),
            "explanation": explanation,
        })

    results.sort(key=lambda x: x["match_pct"], reverse=True)
    return results
