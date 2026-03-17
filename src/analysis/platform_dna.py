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
    QUIZ_MAX_PER_PLATFORM,
    QUIZ_MIN_VOTES,
    QUIZ_N_TITLES,
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
    enriched_df: pd.DataFrame | None = None,
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
            "label": "The Scale Titan",
            "detail": (
                f"With {ps['size']:,} titles, {name} fields the "
                f"{size_rank} library in all of streaming — "
                f"{ratio:.1f}× larger than the average competitor. "
                f"More content means more chances to find something you love."
            ),
            "direction": "high",
            "icon": "📚",
        })
    elif ratio < 0.5:
        traits.append({
            "label": "The Curated Collection",
            "detail": (
                f"At just {ps['size']:,} titles, {name} takes a quality-over-volume approach — "
                f"{1/ratio:.1f}× smaller than average. Every title earns its place."
            ),
            "direction": "neutral",
            "icon": "💎",
        })

    # --- Trait 2: Quality positioning ---
    imdb_diff = ps["avg_imdb"] - avg_imdb
    quality_rank = _rank_among(ps["avg_imdb"], [v["avg_imdb"] for v in per_plat.values()])
    if imdb_diff > 0.25:
        traits.append({
            "label": "The Critics' Darling",
            "detail": (
                f"{ps['premium_count']:,} titles rated 7.0+ on IMDb "
                f"({ps['premium_pct']:.1f}% of the catalog) — "
                f"{name} carries the {quality_rank} average score across all platforms, "
                f"meaning fewer disappointments per click"
            ),
            "direction": "high",
            "icon": "⭐",
        })
    elif imdb_diff < -0.25:
        traits.append({
            "label": "The Volume Play",
            "detail": (
                f"At avg IMDb {ps['avg_imdb']:.2f}, {name} favors breadth over selectivity — "
                f"{abs(imdb_diff):.2f} points below the streaming average. "
                f"Great for browsing; worth filtering by rating before you commit."
            ),
            "direction": "low",
            "icon": "📦",
        })

    # --- Trait 3: Content freshness ---
    year_diff = ps["median_year"] - avg_year
    fresh_rank = _rank_among(ps["median_year"], [v["median_year"] for v in per_plat.values()])
    if year_diff >= 3:
        recent_count = int((platform_df["release_year"] >= 2018).sum()) if len(platform_df) > 0 else 0
        recent_pct = recent_count / len(platform_df) * 100 if len(platform_df) > 0 else 0
        traits.append({
            "label": "Constantly Refreshed",
            "detail": (
                f"{recent_pct:.1f}% of {name}'s catalog dropped since 2018 — "
                f"its median release year sits {int(abs(year_diff))} years ahead of the streaming norm. "
                f"If you want what's new, you'll find it here first."
            ),
            "direction": "high",
            "icon": "🌱",
        })
    elif year_diff <= -3:
        classic_count = int((platform_df["release_year"] < 2000).sum()) if len(platform_df) > 0 else 0
        classic_pct = classic_count / len(platform_df) * 100 if len(platform_df) > 0 else 0
        traits.append({
            "label": "Built on Classics",
            "detail": (
                f"{classic_pct:.0f}% of {name}'s catalog predates 2000 — "
                f"its median release year runs {int(abs(year_diff))} years behind the industry. "
                f"A paradise for fans of timeless film and television."
            ),
            "direction": "low",
            "icon": "📖",
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
                    "label": f"{genre_name} Powerhouse",
                    "detail": (
                        f"{genre_count:,} {genre_name.lower()} titles — "
                        f"{plat_pct:.1f}% of {name}'s library vs {(plat_pct - dev*100):.1f}% across all platforms. "
                        f"If {genre_name.lower()} is your thing, {name} goes deeper than anyone else."
                    ),
                    "direction": "high",
                    "icon": "🎬",
                })

    # --- Trait 5: International diversity ---
    intl_diff = ps["intl_pct"] - avg_intl
    intl_rank = _rank_among(ps["intl_pct"], [v["intl_pct"] for v in per_plat.values()])
    if intl_diff > 10:
        traits.append({
            "label": "Globally Minded",
            "detail": (
                f"{ps['intl_pct']:.1f}% of {name}'s catalog ({ps['intl_count']:,} titles) "
                f"originates outside the US — {intl_rank} for international reach. "
                f"Foreign-language fans and world cinema enthusiasts have a natural home here."
            ),
            "direction": "high",
            "icon": "🌍",
        })
    elif intl_diff < -10:
        dom_count = len(platform_df) - ps["intl_count"]
        traits.append({
            "label": "Home-Court Focused",
            "detail": (
                f"{100-ps['intl_pct']:.1f}% of {name}'s {ps['size']:,} titles are US-produced — "
                f"a clear domestic focus that reflects its core subscriber market. "
                f"American storytelling at scale."
            ),
            "direction": "low",
            "icon": "🇺🇸",
        })

    # --- Trait 6: Content type mix ---
    movie_diff = ps["movie_pct"] - avg_movie
    movie_count = int((platform_df["type"] == "Movie").sum()) if len(platform_df) > 0 else 0
    show_count = int((platform_df["type"] == "Show").sum()) if len(platform_df) > 0 else 0
    movie_rank = _rank_among(ps["movie_pct"], [v["movie_pct"] for v in per_plat.values()])
    if movie_diff > 10:
        traits.append({
            "label": "The Cinephile's Pick",
            "detail": (
                f"{movie_count:,} movies ({ps['movie_pct']:.1f}% of the library) — "
                f"{name} leans heavily toward film over episodic TV. "
                f"If you want a movie night, this is your go-to catalog."
            ),
            "direction": "neutral",
            "icon": "🎥",
        })
    elif movie_diff < -10:
        show_rank = _rank_among(ps["show_pct"], [v["show_pct"] for v in per_plat.values()])
        traits.append({
            "label": "The Series Stronghold",
            "detail": (
                f"{show_count:,} series ({ps['show_pct']:.1f}% of the library) — "
                f"{name} is built for binge-watchers. {show_rank} for serial storytelling "
                f"across all streaming platforms."
            ),
            "direction": "neutral",
            "icon": "📺",
        })

    # --- Trait 7: Awards presence (enrichment-aware) ---
    if enriched_df is not None and not enriched_df.empty and "award_wins" in enriched_df.columns:
        _plat_ids = set(platform_df["id"])
        _plat_enr = enriched_df[enriched_df["id"].isin(_plat_ids)]
        _aw_cov = _plat_enr["award_wins"].notna().mean() if len(_plat_enr) > 0 else 0
        if _aw_cov >= 0.10:
            _total_wins = int(_plat_enr["award_wins"].fillna(0).sum())
            _total_noms = (
                int(_plat_enr["award_noms"].fillna(0).sum())
                if "award_noms" in _plat_enr.columns else 0
            )
            # Cross-platform comparison
            _all_wins = []
            for _pk in ALL_PLATFORMS:
                _pdf = get_platform_titles(all_df, _pk)
                _pids = set(_pdf["id"])
                _penr = enriched_df[enriched_df["id"].isin(_pids)]
                if len(_penr) > 0 and _penr["award_wins"].notna().mean() >= 0.08:
                    _all_wins.append(int(_penr["award_wins"].fillna(0).sum()))
            _award_rank = _rank_among(_total_wins, _all_wins) if _all_wins else "notable"
            if _total_wins > 50:
                _noms_str = f" ({_total_noms:,} nominations)" if _total_noms > 0 else ""
                traits.append({
                    "label": "The Award Magnet",
                    "detail": (
                        f"{name}'s catalog has racked up {_total_wins:,} award wins{_noms_str} — "
                        f"{_award_rank} across all streaming platforms. "
                        f"Prestige and critical acclaim are baked into the DNA here."
                    ),
                    "direction": "high",
                    "icon": "🏆",
                })
    elif "award_wins" in platform_df.columns:
        _aw_coverage = platform_df["award_wins"].notna().mean()
        if _aw_coverage >= 0.15:
            _total_wins = platform_df["award_wins"].fillna(0).sum()
            _wins_per_1k = (_total_wins / len(platform_df) * 1000) if len(platform_df) > 0 else 0
            if _total_wins > 10 and _wins_per_1k > 50:
                traits.append({
                    "label": "The Award Magnet",
                    "detail": (
                        f"{int(_total_wins)} award wins across the catalog — "
                        f"{_wins_per_1k:.0f} per 1,000 titles"
                    ),
                    "direction": "high",
                    "icon": "🏆",
                })

    # --- Trait 8: Production budget power ---
    if enriched_df is not None and not enriched_df.empty and "budget_millions" in enriched_df.columns:
        _plat_ids = set(platform_df["id"])
        _plat_enr = enriched_df[enriched_df["id"].isin(_plat_ids)]
        _vb = _plat_enr["budget_millions"].dropna()
        _vb = _vb[_vb > 0]
        if len(_vb) >= 5:
            _avg_budget = float(_vb.mean())
            _all_budgets = []
            for _pk in ALL_PLATFORMS:
                _pdf = get_platform_titles(all_df, _pk)
                _pids = set(_pdf["id"])
                _penr = enriched_df[enriched_df["id"].isin(_pids)]
                _pvb = _penr["budget_millions"].dropna() if "budget_millions" in _penr.columns else pd.Series(dtype=float)
                _pvb = _pvb[_pvb > 0]
                if len(_pvb) >= 5:
                    _all_budgets.append(float(_pvb.mean()))
            if _all_budgets:
                _global_avg = sum(_all_budgets) / len(_all_budgets)
                _budget_rank = _rank_among(_avg_budget, _all_budgets)
                if _avg_budget > _global_avg * 1.15:
                    traits.append({
                        "label": "The Premium Producer",
                        "detail": (
                            f"Average production budget of ${_avg_budget:.0f}M per title — "
                            f"{_budget_rank} for per-title investment in all of streaming. "
                            f"The production values show on screen."
                        ),
                        "direction": "high",
                        "icon": "💰",
                    })

    # --- Trait 9: Franchise universe strength ---
    if enriched_df is not None and not enriched_df.empty and "collection_name" in enriched_df.columns:
        _plat_ids = set(platform_df["id"])
        _plat_enr = enriched_df[enriched_df["id"].isin(_plat_ids)]
        _franchises = _plat_enr["collection_name"].dropna()
        _franchise_count = int(_franchises.nunique())
        _franchise_titles = int(_franchises.notna().sum())
        if _franchise_count >= 3:
            _all_fc = []
            for _pk in ALL_PLATFORMS:
                _pdf = get_platform_titles(all_df, _pk)
                _pids = set(_pdf["id"])
                _penr = enriched_df[enriched_df["id"].isin(_pids)]
                _fc = int(_penr["collection_name"].dropna().nunique()) if "collection_name" in _penr.columns else 0
                _all_fc.append(_fc)
            if _all_fc:
                _fr_rank = _rank_among(_franchise_count, _all_fc)
                _max_fc = max(_all_fc)
                if _franchise_count >= _max_fc * 0.45:
                    _top_franchise = (
                        _franchises.value_counts().index[0]
                        if len(_franchises) > 0 else ""
                    )
                    _fr_suffix = f", anchored by {_top_franchise}" if _top_franchise else ""
                    traits.append({
                        "label": "The Universe Builder",
                        "detail": (
                            f"{_franchise_count} interconnected franchises spanning {_franchise_titles:,} titles"
                            f"{_fr_suffix} — {_fr_rank} for franchise depth in streaming. "
                            f"Fans of extended storytelling have a lot to explore."
                        ),
                        "direction": "high",
                        "icon": "🌌",
                    })

    return traits[:max_traits]


def compute_platform_comparison_data(
    all_df: pd.DataFrame,
    platform_keys: list[str],
    enriched_df: pd.DataFrame | None = None,
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
            "traits": compute_defining_traits(plat_df, all_df, key, enriched_df=enriched_df),
            "title_count": len(plat_df),
        }
    return result


def compute_landscape_clusters(
    umap_df: pd.DataFrame,
    titles_df: pd.DataFrame,
    n_clusters: int = 10,
) -> pd.DataFrame:
    """Assign titles to content archetypes using genre-priority rules.

    Each title is assigned to the first matching archetype in ``_ARCHETYPES``
    based on a priority ordering from most specific genres (reality,
    documentary, horror) to most generic (drama catch-all).  UMAP coordinates
    are preserved for map visualization only.

    Returns merged DataFrame with umap_x, umap_y, cluster, and title metadata.
    """
    # 1. Merge UMAP coords with title metadata
    keep_cols = ["id", "title", "platform", "type", "genres", "imdb_score",
                 "imdb_votes", "tmdb_popularity", "release_year", "age_certification",
                 "runtime", "production_countries"]
    keep_cols = [c for c in keep_cols if c in titles_df.columns]
    meta = titles_df[keep_cols].drop_duplicates(subset="id", keep="first")
    merged = umap_df.copy().merge(meta, on="id", how="left")

    # 2. Assign each title to its best-matching archetype
    fallback_id = _ARCHETYPES[-1]["id"]

    def _assign_archetype(genres_val) -> int:
        if not isinstance(genres_val, list) or len(genres_val) == 0:
            return fallback_id
        genre_set = {str(g).lower() for g in genres_val}
        for arch in _ARCHETYPES:
            if genre_set & arch["match"] and not (genre_set & arch["exclude"]):
                return arch["id"]
        return fallback_id

    merged["cluster"] = merged["genres"].apply(_assign_archetype)
    return merged


_FAMILY_RATINGS = {"G", "PG", "TV-Y", "TV-Y7", "TV-G", "TV-PG"}
_MATURE_CLUSTER_RATINGS = {"R", "NC-17", "TV-MA", "TV-14"}

# Content archetypes for deterministic genre-priority assignment.
# Ordered from most specific genres → most generic (catch-all).
# Each title is assigned to the FIRST archetype whose genre rule matches.
_ARCHETYPES = [
    {"id": 0, "name": "Reality & Lifestyle",       "match": {"reality"},                                           "exclude": set()},
    {"id": 1, "name": "Documentaries & Factual",   "match": {"documentation"},                                     "exclude": set()},
    {"id": 2, "name": "Dark & Suspenseful",        "match": {"horror"},                                            "exclude": set()},
    {"id": 3, "name": "Animated & All-Ages",       "match": {"animation"},                                         "exclude": set()},
    {"id": 4, "name": "Sci-Fi & Fantasy",          "match": {"scifi", "fantasy"},                                  "exclude": set()},
    {"id": 5, "name": "Crime & Thriller",          "match": {"crime", "thriller"},                                 "exclude": {"horror"}},
    {"id": 6, "name": "Action-Packed & Epic",      "match": {"action", "western", "war", "sport"},                 "exclude": set()},
    {"id": 7, "name": "Comedy & Feel-Good",        "match": {"comedy"},                                            "exclude": set()},
    {"id": 8, "name": "Romance & Relationships",   "match": {"romance"},                                           "exclude": set()},
    {"id": 9, "name": "Prestige Drama",            "match": {"drama", "history", "music", "european", "family"},   "exclude": set()},
]


def generate_cluster_name(cluster_id: int, **kwargs) -> str:
    """Return the predefined archetype name for a given cluster ID."""
    for arch in _ARCHETYPES:
        if arch["id"] == cluster_id:
            return arch["name"]
    return "Mixed Collection"


def compute_cluster_summaries(landscape_df: pd.DataFrame) -> dict:
    """Summarize each cluster: dominant genres, platform mix, sample titles.

    Returns dict mapping cluster_id -> {
        'name': str, 'top_genres': list[str], 'platform_mix': dict[str, int],
        'avg_imdb': float, 'size': int, 'sample_titles': list[str],
        'platform_leader': str | None, 'quality_leader': str | None,
        'archetype_genres': set[str],
    }
    """
    summaries = {}

    # Build archetype lookup for genre-relevance scoring
    arch_lookup = {a["id"]: a for a in _ARCHETYPES}

    for cluster_id, grp in landscape_df.groupby("cluster"):
        arch = arch_lookup.get(cluster_id, _ARCHETYPES[-1])
        arch_genre_set = arch["match"]

        # Top genres — only include genres with ≥5% share of cluster
        genre_counts: dict[str, int] = {}
        titles_with_genres = 0
        for genres in grp["genres"].dropna():
            if isinstance(genres, list) and len(genres) > 0:
                titles_with_genres += 1
                for g in genres:
                    genre_counts[g] = genre_counts.get(g, 0) + 1
        min_share = max(1, titles_with_genres * 0.05)
        significant = {g: c for g, c in genre_counts.items() if c >= min_share}
        if not significant:
            significant = genre_counts
        top_genres_raw = sorted(significant, key=significant.get, reverse=True)[:4]
        top_genres = [_genre_label(g) for g in top_genres_raw]

        # Platform mix
        platform_mix = grp["platform"].value_counts().to_dict() if "platform" in grp.columns else {}

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

        # Sample titles — ranked by quality + cluster genre relevance
        # Use actual top genres in this cluster (not just archetype triggers)
        # so titles like Breaking Bad (crime, drama, thriller) get full credit
        cluster_top_set = {g.lower() for g in top_genres_raw[:4]}
        if len(rated) > 0:
            scored = rated.copy()
            scored["quality_score"] = compute_quality_score(scored)

            def _genre_relevance(genres_val):
                if not isinstance(genres_val, list):
                    return 0.0
                matches = sum(1 for g in genres_val if str(g).lower() in cluster_top_set)
                return matches / max(len(genres_val), 1)

            scored["genre_rel"] = scored["genres"].apply(_genre_relevance)
            scored["final_score"] = (
                scored["quality_score"] * 0.7
                + scored["genre_rel"] * 0.3 * 100
            )

            for threshold in [10_000, 1_000, 0]:
                pool = scored[scored["imdb_votes"].fillna(0) >= threshold]
                if len(pool) >= 5:
                    break
            else:
                pool = scored
            sample = pool.nlargest(5, "final_score")["title"].tolist()
        else:
            sample = []

        name = generate_cluster_name(cluster_id)

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
            "archetype_genres": arch_genre_set,
            "cluster_top_genres": cluster_top_set,
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
# ENRICHMENT STATS + ENHANCED LANDSCAPE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def compute_enriched_platform_stats(
    all_df: pd.DataFrame,
    enriched_df: pd.DataFrame | None = None,
) -> dict:
    """Compute enrichment-based platform statistics for trait generation and quiz matching.

    Returns dict mapping platform_key → {
        'award_wins': int, 'award_noms': int, 'wins_per_1k': float,
        'avg_budget_millions': float | None, 'franchise_count': int,
        'post_2015_pct': float, 'intl_pct': float,
    }
    """
    result = {}
    for key in ALL_PLATFORMS:
        plat_df = get_platform_titles(all_df, key)
        n = max(len(plat_df), 1)

        post_2015_pct = float((plat_df["release_year"] >= 2015).sum() / n * 100) if len(plat_df) > 0 else 0.0

        valid_c = plat_df[plat_df["production_countries"].apply(
            lambda c: isinstance(c, list) and len(c) > 0
        )]
        intl_pct = 0.0
        if len(valid_c) > 0:
            intl_pct = valid_c["production_countries"].apply(lambda c: "US" not in c).mean() * 100

        award_wins = 0
        award_noms = 0
        avg_budget: float | None = None
        franchise_count = 0

        if enriched_df is not None and not enriched_df.empty:
            plat_ids = set(plat_df["id"])
            plat_enr = enriched_df[enriched_df["id"].isin(plat_ids)]

            if "award_wins" in plat_enr.columns and plat_enr["award_wins"].notna().mean() >= 0.08:
                award_wins = int(plat_enr["award_wins"].fillna(0).sum())
            if "award_noms" in plat_enr.columns and plat_enr["award_noms"].notna().mean() >= 0.08:
                award_noms = int(plat_enr["award_noms"].fillna(0).sum())
            if "budget_millions" in plat_enr.columns:
                _vb = plat_enr["budget_millions"].dropna()
                _vb = _vb[_vb > 0]
                if len(_vb) >= 5:
                    avg_budget = round(float(_vb.mean()), 1)
            if "collection_name" in plat_enr.columns:
                franchise_count = int(plat_enr["collection_name"].dropna().nunique())

        result[key] = {
            "award_wins": award_wins,
            "award_noms": award_noms,
            "wins_per_1k": round(award_wins / n * 1000, 1),
            "avg_budget_millions": avg_budget,
            "franchise_count": franchise_count,
            "post_2015_pct": round(post_2015_pct, 1),
            "intl_pct": round(intl_pct, 1),
        }
    return result


def compute_landscape_insights_v2(
    landscape_df: pd.DataFrame,
    cluster_summaries: dict,
) -> list[dict]:
    """Generate specific, data-backed landscape insights with platform colors and numbers.

    Returns list of dicts: {'icon': str, 'text': str (HTML with colored spans)}.
    """
    insights: list[dict] = []
    if "platform" not in landscape_df.columns:
        return insights

    def _colored(pk: str) -> str:
        c = PLATFORMS.get(pk, {}).get("color", "#ddd")
        n = PLATFORMS.get(pk, {}).get("name", pk)
        return f'<span style="color:{c};font-weight:700;">{n}</span>'

    # 1. Platform dominance per cluster (top 2 strongest ownership stories)
    dominance_insights = []
    for cid, summary in cluster_summaries.items():
        pmix = summary.get("platform_mix", {})
        cluster_total = summary.get("size", 1)
        avg_imdb = summary.get("avg_imdb") or 0
        name = summary.get("name", "")
        for pk, count in sorted(pmix.items(), key=lambda x: -x[1]):
            pct = count / cluster_total * 100
            if pct >= 40 and count >= 15:
                dominance_insights.append({
                    "icon": "crown",
                    "text": (
                        f'{_colored(pk)} owns the <strong>{name}</strong> zone — '
                        f'{pct:.0f}% of all {count:,} titles in this space carry its brand '
                        f'(avg IMDb {avg_imdb:.1f})'
                    ),
                    "_sort": pct,
                })
                break
    dominance_insights.sort(key=lambda x: -x["_sort"])
    insights.extend(dominance_insights[:2])

    # 2. Netflix + Max merger overlap insight
    nf_data = landscape_df[landscape_df["platform"] == "netflix"]
    mx_data = landscape_df[landscape_df["platform"] == "max"]
    if len(nf_data) >= 10 and len(mx_data) >= 10:
        nf_clusters = nf_data["cluster"].value_counts(normalize=True)
        mx_clusters = mx_data["cluster"].value_counts(normalize=True)
        all_cids_set = set(nf_clusters.index) | set(mx_clusters.index)
        dot = sum(float(nf_clusters.get(c, 0)) * float(mx_clusters.get(c, 0)) for c in all_cids_set)
        norm_nf = float((sum(v**2 for v in nf_clusters.values)) ** 0.5)
        norm_mx = float((sum(v**2 for v in mx_clusters.values)) ** 0.5)
        sim_pct = dot / (norm_nf * norm_mx) * 100 if norm_nf and norm_mx else 0
        top_shared = max(
            all_cids_set,
            key=lambda c: min(float(nf_clusters.get(c, 0)), float(mx_clusters.get(c, 0))),
        )
        shared_name = cluster_summaries.get(top_shared, {}).get("name", "")
        if shared_name:
            insights.append({
                "icon": "overlap",
                "text": (
                    f'Post-merger {_colored("netflix")} + {_colored("max")} territory: '
                    f'both platforms converge heavily in <strong>{shared_name}</strong>, '
                    f'giving the combined entity a commanding position there '
                    f'({sim_pct:.0f}% content distribution overlap overall)'
                ),
            })

    # 3. Most focused vs most diverse platform
    plat_focus: dict[str, dict] = {}
    for pk in landscape_df["platform"].unique():
        pk_data = landscape_df[landscape_df["platform"] == pk]
        if len(pk_data) >= 20:
            top_pct = float(pk_data["cluster"].value_counts(normalize=True).iloc[0] * 100)
            top_cid = int(pk_data["cluster"].value_counts().index[0])
            plat_focus[pk] = {"top_pct": top_pct, "top_cid": top_cid}

    if len(plat_focus) >= 2:
        most_focused = max(plat_focus, key=lambda k: plat_focus[k]["top_pct"])
        f_pct = plat_focus[most_focused]["top_pct"]
        f_name = cluster_summaries.get(plat_focus[most_focused]["top_cid"], {}).get("name", "")
        most_spread = min(plat_focus, key=lambda k: plat_focus[k]["top_pct"])
        s_pct = plat_focus[most_spread]["top_pct"]
        if f_name:
            insights.append({
                "icon": "focus",
                "text": (
                    f'{_colored(most_focused)} doubles down — {f_pct:.0f}% of all its titles '
                    f'live in <strong>{f_name}</strong>, making it the most genre-focused platform. '
                    f'{_colored(most_spread)} takes the opposite approach: '
                    f'spread across every neighborhood with no single zone exceeding {s_pct:.0f}%'
                ),
            })

    return insights[:4]


def compute_neighborhood_top_titles(
    landscape_df: pd.DataFrame,
    enriched_df: pd.DataFrame | None,
    cluster_id: int,
    top_n: int = 5,
) -> list[dict]:
    """Return top N titles for a content neighborhood with enrichment metadata.

    Returns list of dicts with title metadata + poster_url, award_wins, etc.
    """
    cluster_data = landscape_df[landscape_df["cluster"] == cluster_id].copy()
    if cluster_data.empty:
        return []

    rated = cluster_data.dropna(subset=["imdb_score"]).copy()
    if rated.empty:
        return []

    rated["quality_score"] = compute_quality_score(rated)

    for threshold in [10_000, 1_000, 0]:
        pool = rated[rated["imdb_votes"].fillna(0) >= threshold]
        if len(pool) >= min(top_n, len(rated)):
            break
    else:
        pool = rated

    top = pool.nlargest(top_n, "quality_score")

    result = []
    for _, row in top.iterrows():
        tid = row.get("id")
        rec: dict = {
            "id": tid,
            "title": row.get("title", ""),
            "platform": row.get("platform", ""),
            "type": row.get("type", ""),
            "genres": row.get("genres", []),
            "imdb_score": row.get("imdb_score"),
            "imdb_votes": row.get("imdb_votes"),
            "release_year": row.get("release_year"),
            "age_certification": row.get("age_certification"),
            "runtime": row.get("runtime"),
            "description": row.get("description", "") if "description" in row.index else "",
            "quality_score": row.get("quality_score"),
            "poster_url": None,
            "award_wins": None,
            "award_noms": None,
            "box_office_usd": None,
        }
        if enriched_df is not None and tid is not None and not enriched_df.empty:
            enr_rows = enriched_df[enriched_df["id"] == tid]
            if not enr_rows.empty:
                enr = enr_rows.iloc[0]
                for field in ("poster_url", "award_wins", "award_noms", "box_office_usd"):
                    if field in enr_rows.columns:
                        val = enr.get(field)
                        rec[field] = val if val is not None and str(val) != "nan" else None
        result.append(rec)

    return result


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


# =============================================================================
# PLATFORM QUIZ — Hybrid Genre Selector + Title Swipe
# =============================================================================

_DARK_GENRES = {"horror", "thriller", "crime"}
_LIGHT_GENRES = {"horror", "thriller"}  # excluded in feel-good mode


def curate_quiz_titles(
    all_df: pd.DataFrame,
    selected_genres: list[str],
    quality_pref: str = "No Preference",
    type_pref: str = "Both",
    vibe_pref: str = "Mix of Both",
    enriched_df: pd.DataFrame | None = None,
) -> list[dict]:
    """Curate diverse title cards for the swipe phase of the quiz.

    Returns list of dicts: {id, title, platform, type, genres, imdb_score,
    release_year, description, imdb_votes}.
    """
    from src.data.loaders import deduplicate_titles

    pool = all_df.copy()

    # Build genre set for matching
    genre_set: set[str] = {g.lower() for g in selected_genres} if selected_genres else set()

    # Filter to titles with at least one selected genre
    if genre_set:
        pool = pool[pool["genres"].apply(
            lambda gs: isinstance(gs, list) and bool({str(g).lower() for g in gs} & genre_set)
        )]

    # Exclude animation when user didn't explicitly select it — prevents animated
    # kids' movies from dominating comedy/romance picks
    if "animation" not in genre_set:
        pool = pool[pool["genres"].apply(
            lambda gs: not (
                isinstance(gs, list) and len(gs) > 0
                and str(gs[0]).lower() == "animation"
            )
        )]

    # Quality filter
    if quality_pref == "Award Winners":
        pool = pool[pool["imdb_score"].fillna(0) >= 7.5]
    elif quality_pref == "Crowd Favorites":
        pool = pool[pool["imdb_votes"].fillna(0) >= 10_000]

    # Type filter
    if type_pref == "Movies":
        pool = pool[pool["type"] == "Movie"]
    elif type_pref == "Shows":
        pool = pool[pool["type"] == "Show"]

    # Vibe filter
    if vibe_pref == "Feel-Good & Light":
        pool = pool[pool["genres"].apply(
            lambda gs: isinstance(gs, list) and not bool({str(g).lower() for g in gs} & _LIGHT_GENRES)
        )]
    elif vibe_pref == "Dark & Intense":
        pool = pool[pool["genres"].apply(
            lambda gs: isinstance(gs, list) and bool({str(g).lower() for g in gs} & _DARK_GENRES)
        )]

    # Require minimum votes for recognizable titles
    pool = pool[pool["imdb_votes"].fillna(0) >= QUIZ_MIN_VOTES]

    if len(pool) < QUIZ_N_TITLES:
        # Relax vote threshold if not enough titles
        pool = all_df.copy()
        if genre_set:
            pool = pool[pool["genres"].apply(
                lambda gs: isinstance(gs, list) and bool({str(g).lower() for g in gs} & genre_set)
            )]
            # Re-apply animation exclusion even in fallback
            if "animation" not in genre_set:
                pool = pool[pool["genres"].apply(
                    lambda gs: not (
                        isinstance(gs, list) and len(gs) > 0
                        and str(gs[0]).lower() == "animation"
                    )
                )]
        pool = pool[pool["imdb_votes"].fillna(0) >= 1000]

    # Deduplicate (multi-platform titles appear once)
    pool = deduplicate_titles(pool)

    # Score by quality AND genre relevance (how many user genres does the title match?)
    pool = pool.copy()
    pool["_qs"] = compute_quality_score(pool)
    if genre_set:
        def _count_matches(gs):
            if not isinstance(gs, list):
                return 0
            return len({str(g).lower() for g in gs} & genre_set)
        pool["_genre_matches"] = pool["genres"].apply(_count_matches)
        # Boost titles that match more of the user's selected genres (up to 30% bonus)
        pool["_final_score"] = pool["_qs"] * (1 + pool["_genre_matches"].clip(0, 3) * 0.10)
    else:
        pool["_final_score"] = pool["_qs"]

    # Sample ensuring platform diversity + genre coverage
    selected: list[pd.Series] = []
    plat_counts: dict[str, int] = {}
    seen_ids: set = set()

    # Rank by final score (quality × genre relevance) for strong, relevant candidates
    ranked = pool.nlargest(min(300, len(pool)), "_final_score")

    for _, row in ranked.iterrows():
        if len(selected) >= QUIZ_N_TITLES:
            break
        rid = row.get("id")
        if rid in seen_ids:
            continue

        # Platform diversity: check across all platforms this title is on
        platforms = row.get("platforms", [row.get("platform", "")])
        if isinstance(platforms, str):
            platforms = [platforms]
        if any(plat_counts.get(p, 0) >= QUIZ_MAX_PER_PLATFORM for p in platforms):
            continue

        selected.append(row)
        seen_ids.add(rid)
        for p in platforms:
            plat_counts[p] = plat_counts.get(p, 0) + 1

    # If not enough, fill from remaining without platform constraint
    if len(selected) < QUIZ_N_TITLES:
        for _, row in ranked.iterrows():
            if len(selected) >= QUIZ_N_TITLES:
                break
            rid = row.get("id")
            if rid not in seen_ids:
                selected.append(row)
                seen_ids.add(rid)

    # Build poster map from enriched data
    poster_map: dict = {}
    award_map: dict = {}
    if enriched_df is not None and not enriched_df.empty:
        if "poster_url" in enriched_df.columns:
            poster_map = enriched_df.dropna(subset=["poster_url"]).set_index("id")["poster_url"].to_dict()
        if "award_wins" in enriched_df.columns:
            award_map = enriched_df.dropna(subset=["award_wins"]).set_index("id")["award_wins"].to_dict()

    # Build output dicts
    result = []
    for row in selected:
        platforms = row.get("platforms", [row.get("platform", "")])
        if isinstance(platforms, str):
            platforms = [platforms]
        desc = row.get("description", "")
        if isinstance(desc, str) and len(desc) > 250:
            desc = desc[:247] + "..."
        tid = row.get("id")
        raw_poster = poster_map.get(tid, "")
        result.append({
            "id": tid,
            "title": row.get("title", "Unknown"),
            "platform": platforms[0] if platforms else "",
            "platforms": platforms,
            "type": row.get("type", ""),
            "genres": row.get("genres", []),
            "imdb_score": row.get("imdb_score"),
            "imdb_votes": row.get("imdb_votes"),
            "release_year": row.get("release_year"),
            "description": desc,
            "poster_url": str(raw_poster) if raw_poster and str(raw_poster) != "nan" else None,
            "award_wins": award_map.get(tid),
        })
    return result


def compute_swipe_results(
    liked_ids: list,
    all_titles: list[dict],
    all_df: pd.DataFrame,
) -> dict:
    """Analyze liked titles to compute platform match scores and recommendations.

    Args:
        liked_ids: IDs of titles the user liked
        all_titles: full list of quiz title dicts (for reference)
        all_df: full all_platforms_titles DataFrame

    Returns:
        {
            'rankings': [{platform, display_name, match_pct, explanation}, ...],
            'personality': str,
            'recommendations': [{title, platform, imdb_score, ...}, ...],
        }
    """
    # Build liked title data
    liked_data = [t for t in all_titles if t.get("id") in set(liked_ids)]
    if not liked_data:
        return {
            "rankings": [
                {"platform": k, "display_name": PLATFORMS[k]["name"],
                 "match_pct": 50.0, "explanation": "Like some titles to get personalized results."}
                for k in ALL_PLATFORMS
            ],
            "personality": "Undecided viewer",
            "recommendations": [],
        }

    # 1. Genre affinity from liked titles
    genre_counts: dict[str, int] = {}
    for t in liked_data:
        for g in (t.get("genres") or []):
            gl = str(g).lower()
            genre_counts[gl] = genre_counts.get(gl, 0) + 1
    total_genre_mentions = sum(genre_counts.values()) or 1
    user_genre_dist = {g: c / total_genre_mentions for g, c in genre_counts.items()}
    top_user_genres = sorted(genre_counts, key=genre_counts.get, reverse=True)[:3]

    # 2. Platform affinity from liked titles
    plat_like_counts: dict[str, int] = {}
    for t in liked_data:
        for p in (t.get("platforms") or [t.get("platform", "")]):
            plat_like_counts[p] = plat_like_counts.get(p, 0) + 1

    # 3. Quality preference
    liked_scores = [t["imdb_score"] for t in liked_data if t.get("imdb_score")]
    user_avg_quality = sum(liked_scores) / len(liked_scores) if liked_scores else 6.5

    # 4. Score each platform
    rankings = []
    for key in ALL_PLATFORMS:
        plat_df = all_df[all_df["platform"] == key]
        if plat_df.empty:
            continue

        # Genre affinity: cosine-like overlap between user genre dist and platform genre dist
        plat_genre_counts: dict[str, int] = {}
        plat_n = 0
        for gs in plat_df["genres"].dropna():
            if isinstance(gs, list) and gs:
                plat_n += 1
                for g in gs:
                    plat_genre_counts[str(g).lower()] = plat_genre_counts.get(str(g).lower(), 0) + 1
        plat_genre_dist = {g: c / plat_n for g, c in plat_genre_counts.items()} if plat_n else {}

        # Dot product / (norm_user * norm_plat)
        all_genres_union = set(user_genre_dist) | set(plat_genre_dist)
        dot = sum(user_genre_dist.get(g, 0) * plat_genre_dist.get(g, 0) for g in all_genres_union)
        norm_u = sum(v**2 for v in user_genre_dist.values()) ** 0.5
        norm_p = sum(v**2 for v in plat_genre_dist.values()) ** 0.5
        genre_score = (dot / (norm_u * norm_p) * 100) if norm_u and norm_p else 50

        # Platform affinity: direct like count
        like_score = plat_like_counts.get(key, 0) / max(len(liked_data), 1) * 100

        # Quality alignment
        plat_avg = plat_df["imdb_score"].dropna().mean() if plat_df["imdb_score"].notna().any() else 6.5
        quality_score = max(0, 100 - abs(user_avg_quality - plat_avg) * 20)

        # Combined: 40% genre + 30% platform affinity + 30% quality
        total = 0.40 * genre_score + 0.30 * like_score + 0.30 * quality_score

        # Explanation
        parts = []
        if genre_score >= 70:
            parts.append("strong genre alignment")
        elif genre_score >= 50:
            parts.append("moderate genre overlap")
        else:
            parts.append("different genre focus")
        if like_score >= 40:
            parts.append("you liked their titles")
        if quality_score >= 70:
            parts.append("matches your quality taste")

        rankings.append({
            "platform": key,
            "display_name": PLATFORMS.get(key, {}).get("name", key.title()),
            "match_pct": round(total, 1),
            "explanation": ". ".join(parts).capitalize() + "." if parts else "Moderate match.",
        })

    rankings.sort(key=lambda x: x["match_pct"], reverse=True)

    # 5. Personality summary
    persona_parts = []
    if top_user_genres:
        genre_labels = [_genre_label(g) for g in top_user_genres[:3]]
        persona_parts.append(f"loves {', '.join(genre_labels)}")
    if user_avg_quality >= 7.5:
        persona_parts.append("seeks premium quality")
    elif user_avg_quality < 6.0:
        persona_parts.append("enjoys casual entertainment")
    personality = "You're a viewer who " + " and ".join(persona_parts) + "." if persona_parts else ""

    # 6. Recommendations from best-match platform
    recs = []
    if rankings:
        best_key = rankings[0]["platform"]
        best_df = all_df[all_df["platform"] == best_key].copy()

        # Filter to titles matching user's top genres
        if top_user_genres:
            user_genre_set = set(top_user_genres)
            best_df = best_df[best_df["genres"].apply(
                lambda gs: isinstance(gs, list) and bool({str(g).lower() for g in gs} & user_genre_set)
            )]

        # Exclude titles already seen in quiz
        seen_ids = {t["id"] for t in all_titles}
        best_df = best_df[~best_df["id"].isin(seen_ids)]

        # Top quality titles
        if len(best_df) > 0:
            best_df = best_df.copy()
            best_df["_qs"] = compute_quality_score(best_df)
            for threshold in [10_000, 1_000, 0]:
                _pool = best_df[best_df["imdb_votes"].fillna(0) >= threshold]
                if len(_pool) >= 5:
                    break
            else:
                _pool = best_df
            top_recs = _pool.nlargest(5, "_qs")
            for _, r in top_recs.iterrows():
                recs.append({
                    "id": r.get("id"),
                    "title": r.get("title", ""),
                    "platform": best_key,
                    "imdb_score": r.get("imdb_score"),
                    "release_year": r.get("release_year"),
                    "type": r.get("type", ""),
                    "genres": r.get("genres", []),
                })

    return {
        "rankings": rankings,
        "personality": personality,
        "recommendations": recs,
    }


# =============================================================================
# ENHANCED QUIZ RESULTS V2 — combines swipe + slider preferences
# =============================================================================

_SLIDER_DIMS = ["recency", "popularity", "runtime", "maturity", "international"]

# Genres that indicate dark/intense content (for tone slider)
_DARK_GENRE_SET: set[str] = {"crime", "thriller", "horror"}
# Genres that indicate light/feel-good content
_FEEL_GOOD_GENRE_SET: set[str] = {"comedy", "romance", "animation", "family", "music"}


def _compute_darkness_score(plat_df: pd.DataFrame) -> float:
    """Compute a 0-100 darkness score for a platform.

    Combines age-cert maturity (60%) with dark-genre % (40%) for a
    richer tone signal than cert alone.
    """
    certs = plat_df["age_certification"].dropna()
    cert_mature = float(certs.isin(_MATURE_RATINGS).mean() * 100) if len(certs) > 0 else 50.0

    n = max(len(plat_df), 1)
    dark_count = int(plat_df["genres"].apply(
        lambda gs: isinstance(gs, list) and bool({str(g).lower() for g in gs} & _DARK_GENRE_SET)
    ).sum())
    dark_pct = dark_count / n * 100

    return min(100.0, cert_mature * 0.6 + dark_pct * 0.4)


def _compute_popularity_score(plat_df: pd.DataFrame, global_med: float = 4.4) -> float:
    """Compute a 0-100 mainstream score using median tmdb_popularity.

    0 = very niche / arthouse, 100 = very mainstream.
    Calibrated: median 4.4 → ~50, top-shelf mainstream (>20) → ~90+
    """
    pop_vals = plat_df["tmdb_popularity"].dropna()
    if len(pop_vals) == 0:
        return 50.0
    med_pop = float(pop_vals.median())
    # Log-scale to compress the heavy tail
    import math
    log_med = math.log1p(med_pop)
    log_ref = math.log1p(global_med)
    log_high = math.log1p(30.0)
    return max(0.0, min(100.0, (log_med / log_high) * 100))


def compute_swipe_results_v2(
    liked_ids: list,
    all_titles: list[dict],
    all_df: pd.DataFrame,
    slider_prefs: dict | None = None,
    enriched_df: pd.DataFrame | None = None,
    enriched_stats: dict | None = None,
    user_selected_genres: list | None = None,
) -> dict:
    """Rebuilt platform matching — swipe signals + slider preferences + genre specialization.

    Platform profiles use richer real-data signals:
      - maturity: cert-based + dark genre % (not just cert)
      - popularity: median tmdb_popularity (not IMDb quality)
      - recency, runtime, international: unchanged

    Scoring weights: 30% genre spec + 25% like precision + 35% slider + 10% quality.
    Sliders now drive meaningful differentiation (e.g., feel-good vs. dark).

    Recommendations filtered by genre overlap fraction × tone fit × quality, so
    a feel-good comedy+romance profile won't surface crime dramas or R-rated films.

    Returns:
        {
            'rankings':             list[{platform, display_name, match_pct, explanation}],
            'personality':          str,
            'recommendations':      list[{...title fields + poster_url}],
            'why_match':            list[str],
            'platform_explanations': dict[str, str],  # natural-language per-platform narrative
        }
    """
    liked_data = [t for t in all_titles if t.get("id") in set(liked_ids)]
    if not liked_data:
        return {
            "rankings": [
                {
                    "platform": k,
                    "display_name": PLATFORMS[k]["name"],
                    "match_pct": 50.0,
                    "explanation": "Like some titles to get personalized results.",
                    "bullets": [],
                }
                for k in ALL_PLATFORMS
            ],
            "personality": "Undecided viewer",
            "recommendations": [],
            "why_match": [],
        }

    # ── 1. Genre affinity from liked titles ──────────────────────────────────
    genre_counts: dict[str, int] = {}
    for t in liked_data:
        for g in (t.get("genres") or []):
            gl = str(g).lower()
            genre_counts[gl] = genre_counts.get(gl, 0) + 1
    total_genre_mentions = sum(genre_counts.values()) or 1
    user_genre_dist = {g: c / total_genre_mentions for g, c in genre_counts.items()}
    top_user_genres = sorted(genre_counts, key=genre_counts.get, reverse=True)[:3]

    # ── 2. Platform affinity from liked titles ────────────────────────────────
    plat_like_counts: dict[str, int] = {}
    for t in liked_data:
        for p in (t.get("platforms") or [t.get("platform", "")]):
            plat_like_counts[p] = plat_like_counts.get(p, 0) + 1

    # ── 3. Quality preference from liked titles ───────────────────────────────
    liked_scores = [t["imdb_score"] for t in liked_data if t.get("imdb_score")]
    user_avg_quality = sum(liked_scores) / len(liked_scores) if liked_scores else 6.5

    # ── 4. Build platform profiles for slider matching ────────────────────────
    # Compute per-platform stats needed for slider dimensions.
    # Key improvements vs v1:
    #   - maturity: cert-based (60%) + dark-genre % (40%) → captures prestige dramas
    #   - popularity: median tmdb_popularity (real mainstream proxy, not IMDb quality)
    #   - recency, runtime, international: same as before
    _plat_profiles: dict[str, dict] = {}
    for key in ALL_PLATFORMS:
        plat_df = get_platform_titles(all_df, key)
        if len(plat_df) == 0:
            _plat_profiles[key] = {d: 50.0 for d in _SLIDER_DIMS + ["award_level"]}
            _plat_profiles[key]["avg_imdb"] = 6.0
            continue

        rated = plat_df.dropna(subset=["imdb_score"])
        avg_imdb = float(rated["imdb_score"].mean()) if len(rated) > 0 else 6.0

        # Recency (0=all-time classics, 100=latest releases)
        # Calibrated: 1940→0, 2023→100
        med_year = float(plat_df["release_year"].median())
        recency = max(0.0, min(100.0, (med_year - 1940) / (2023 - 1940) * 100))

        # Popularity: median tmdb_popularity (higher = more mainstream)
        # Calibrated on log-scale: 1.0→~0, 4.4 (global med)→~50, 30+→~90
        popularity = _compute_popularity_score(plat_df)

        # Runtime (median runtime → 0-100; 30 min=0, 150 min=100)
        med_rt = float(plat_df["runtime"].median()) if plat_df["runtime"].notna().any() else 80.0
        runtime = max(0.0, min(100.0, (med_rt - 30) / (150 - 30) * 100))

        # Maturity / darkness: cert-based + dark genre % for richer tone signal
        # A platform with 60% R-rated BUT low crime/thriller still scores lower than
        # a platform with 60% R-rated AND heavy crime/thriller/horror catalog
        maturity = _compute_darkness_score(plat_df)

        # International (% non-US → 0-100)
        valid_c = plat_df[plat_df["production_countries"].apply(
            lambda c: isinstance(c, list) and len(c) > 0
        )]
        international = (
            float(valid_c["production_countries"].apply(lambda c: "US" not in c).mean() * 100)
            if len(valid_c) > 0 else 50.0
        )

        # Award level (wins_per_1k normalized → 0-100)
        award_level = 0.0
        if enriched_stats and key in enriched_stats:
            wins_per_1k = enriched_stats[key].get("wins_per_1k", 0) or 0
            all_wk = [v.get("wins_per_1k", 0) or 0 for v in enriched_stats.values()]
            mx_wk = max(all_wk) if all_wk else 1
            award_level = wins_per_1k / mx_wk * 100 if mx_wk > 0 else 0.0

        _plat_profiles[key] = {
            "recency": recency,
            "popularity": popularity,
            "runtime": runtime,
            "maturity": maturity,
            "international": international,
            "award_level": award_level,
            "avg_imdb": avg_imdb,
            # Store raw values for natural-language explanations
            "_med_year": med_year,
            "_post2015_pct": float((plat_df["release_year"] >= 2015).mean() * 100),
        }

    # ── 5a. Pre-compute per-platform genre % for specialization scoring ──────
    # For each platform, compute what % of its catalog falls in each genre.
    # This raw % gives stronger differentiation than cosine similarity because
    # a platform with 20% romance vs one with 8% romance is clearly better for
    # a romance fan — even if their overall genre distributions are similar shapes.
    _genre_pct_by_plat: dict[str, dict[str, float]] = {}
    for _k in ALL_PLATFORMS:
        _kdf = get_platform_titles(all_df, _k)
        _kn = max(len(_kdf), 1)
        _gc: dict[str, int] = {}
        for _gs in _kdf["genres"].dropna():
            if isinstance(_gs, list):
                for _g in _gs:
                    _gc[str(_g).lower()] = _gc.get(str(_g).lower(), 0) + 1
        _genre_pct_by_plat[_k] = {g: c / _kn * 100 for g, c in _gc.items()}

    # Genres to score against: prefer quiz-selected genres over derived liked-title genres
    _scoring_genres: list[str] = [g.lower() for g in (user_selected_genres or [])]
    if not _scoring_genres:
        _scoring_genres = top_user_genres[:5]

    # Genre specialization score per platform: for each user genre, how does this
    # platform compare to the best platform for that genre? (0-100 per genre)
    _genre_spec_scores: dict[str, float] = {k: 0.0 for k in ALL_PLATFORMS}
    _n_sg = max(len(_scoring_genres), 1)
    for _sg in _scoring_genres:
        _sg_pcts = {k: _genre_pct_by_plat[k].get(_sg, 0.0) for k in ALL_PLATFORMS}
        _max_pct = max(_sg_pcts.values()) or 1.0
        for _k in ALL_PLATFORMS:
            _genre_spec_scores[_k] += _sg_pcts[_k] / _max_pct * 100
    for _k in ALL_PLATFORMS:
        _genre_spec_scores[_k] /= _n_sg

    # Count how many quiz titles were shown per platform (for like-precision)
    _plat_quiz_counts: dict[str, int] = {}
    for _t in all_titles:
        for _p in (_t.get("platforms") or [_t.get("platform", "")]):
            _plat_quiz_counts[_p] = _plat_quiz_counts.get(_p, 0) + 1

    # ── 5b. Score each platform ───────────────────────────────────────────────
    rankings = []
    for key in ALL_PLATFORMS:
        plat_df_key = all_df[all_df["platform"] == key]
        if plat_df_key.empty:
            continue

        profile = _plat_profiles.get(key, {})

        # 1. Genre specialization (how strongly does this platform cover user genres)
        genre_spec = _genre_spec_scores.get(key, 50.0)

        # 2. Like precision: of this platform's quiz titles, what % did user like?
        #    Much more discriminative than raw like count / total liked.
        _shown = _plat_quiz_counts.get(key, 0)
        _liked = plat_like_counts.get(key, 0)
        if _shown >= 2:
            like_precision = _liked / _shown * 100
        elif _liked > 0:
            like_precision = 55.0   # some evidence, low confidence
        else:
            like_precision = 20.0   # not shown or nothing liked from this platform

        # 3. Quality alignment (softer penalty: 15 pts per IMDb point difference)
        plat_avg_q = profile.get("avg_imdb", 6.5)
        quality_score = max(0.0, 100.0 - abs(user_avg_quality - plat_avg_q) * 15)

        # 4. Slider alignment
        slider_score = 50.0
        if slider_prefs:
            _ds = []
            for dim in _SLIDER_DIMS:
                _u = float(slider_prefs.get(dim, 50))
                _pv = float(profile.get(dim, 50))
                _ds.append(max(0.0, 100.0 - abs(_u - _pv)))
            _aw_pref = float(slider_prefs.get("awards", 50))
            _aw_eff = 100.0 - _aw_pref
            _aw_plat = float(profile.get("award_level", 50))
            _ds.append(max(0.0, 100.0 - abs(_aw_eff - _aw_plat)))
            slider_score = sum(_ds) / len(_ds)

        # Combined weights — sliders now carry more weight (0.35) because they capture
        # vibe/tone/era signals that genre % alone can't distinguish.  Genre spec
        # is reduced to 0.30 so e.g. a "feel-good" slider choice can meaningfully
        # push a dark-catalog platform (Max) below a lighter one (Netflix/Paramount).
        if slider_prefs:
            total = (
                0.30 * genre_spec
                + 0.25 * like_precision
                + 0.35 * slider_score
                + 0.10 * quality_score
            )
        else:
            total = (
                0.45 * genre_spec
                + 0.40 * like_precision
                + 0.15 * quality_score
            )

        # Build specific explanation using actual data
        _plat_name = PLATFORMS.get(key, {}).get("name", key.title())
        _sg_labels = [_genre_label(g) for g in _scoring_genres[:2]]
        _genre_share = sum(_genre_pct_by_plat[key].get(g, 0) for g in _scoring_genres[:2])
        _parts = []
        if _scoring_genres:
            if genre_spec >= 80:
                _parts.append(
                    f"{' & '.join(_sg_labels)} depth: {_genre_share:.0f}% of catalog — "
                    f"one of the strongest fits for your tastes"
                )
            elif genre_spec >= 55:
                _parts.append(
                    f"Decent {' & '.join(_sg_labels)} coverage: {_genre_share:.0f}% of catalog"
                )
            else:
                _parts.append(
                    f"Limited {' & '.join(_sg_labels)} presence: only {_genre_share:.0f}% of catalog"
                )
        if _liked >= 2:
            _parts.append(f"you liked {_liked} of {_shown} of their titles shown")
        elif _liked == 1:
            _parts.append(f"1 of their titles caught your eye")
        if slider_prefs and slider_score >= 75:
            _parts.append("vibe preferences are closely aligned")
        elif slider_prefs and slider_score < 40:
            _parts.append("your vibe preferences diverge here")

        explanation = ". ".join(_parts).capitalize() + "." if _parts else "Moderate match across factors."

        rankings.append({
            "platform": key,
            "display_name": _plat_name,
            "match_pct": round(min(total, 99.0), 1),
            "explanation": explanation,
            "bullets": [],
            "_genre_spec": round(genre_spec, 1),
            "_like_precision": round(like_precision, 1),
            "_slider_score": round(slider_score, 1),
            "_genre_share": round(_genre_share, 1),
        })

    rankings.sort(key=lambda x: x["match_pct"], reverse=True)

    # ── 5c. Spread normalization ──────────────────────────────────────────────
    # If all scores are bunched within 15 pts, force spread to [58, 95] while
    # preserving the ranking order — making the top pick clearly distinct.
    _raw_scores = [r["match_pct"] for r in rankings]
    _min_s, _max_s = min(_raw_scores), max(_raw_scores)
    if _max_s - _min_s < 18 and len(rankings) >= 2:
        _spread = max(_max_s - _min_s, 0.1)
        for r in rankings:
            r["match_pct"] = round(58 + (r["match_pct"] - _min_s) / _spread * 37, 1)

    # ── 6. Personality summary ────────────────────────────────────────────────
    # Build a conversational narrative, not a robot checklist
    _liked_pct = int(len(liked_data) / max(len(all_titles), 1) * 100)

    # Genre sentence
    _p_genres = [_genre_label(g) for g in (user_selected_genres or top_user_genres)[:2]]
    if len(_p_genres) == 2:
        _genre_sent = f"You gravitate toward {_p_genres[0]} and {_p_genres[1]}"
    elif len(_p_genres) == 1:
        _genre_sent = f"You're drawn to {_p_genres[0]} content"
    else:
        _genre_sent = "You have wide-ranging taste"

    # Quality nuance
    if user_avg_quality >= 7.5:
        _q_sent = "with a clear eye for critically acclaimed titles"
    elif user_avg_quality >= 6.8:
        _q_sent = "and you like your content to be genuinely good"
    else:
        _q_sent = "and you're open to entertainment that doesn't take itself too seriously"

    # Slider-driven flavour
    _flavour = []
    if slider_prefs:
        _era = float(slider_prefs.get("recency", 50))
        _tone = float(slider_prefs.get("maturity", 50))
        _rt = float(slider_prefs.get("runtime", 50))
        _intl = float(slider_prefs.get("international", 50))
        if _era >= 70:
            _flavour.append("staying current with fresh releases")
        elif _era <= 30:
            _flavour.append("a love of timeless classics")
        if _tone >= 70:
            _flavour.append("a taste for darker, grittier stories")
        elif _tone <= 30:
            _flavour.append("a preference for feel-good, lighter content")
        if _rt >= 70:
            _flavour.append("patience for long, immersive viewing")
        elif _rt <= 30:
            _flavour.append("a preference for quick, punchy watches")
        if _intl >= 65:
            _flavour.append("an appetite for international storytelling")

    if _flavour:
        _flavour_sent = "Your session signals: " + ", ".join(_flavour) + "."
    else:
        _flavour_sent = ""

    _swipe_note = f"You liked {len(liked_data)} of {len(all_titles)} titles shown ({_liked_pct}%)."

    personality = f"{_genre_sent}, {_q_sent}. {_flavour_sent} {_swipe_note}".strip()

    # ── 7. Why-Match bullets for best platform ────────────────────────────────
    why_match: list[str] = []
    if rankings:
        best_key = rankings[0]["platform"]
        best_display = PLATFORMS.get(best_key, {}).get("name", best_key.title())
        best_df = get_platform_titles(all_df, best_key)
        prof = _plat_profiles.get(best_key, {})

        # Genre bullet — most specific first
        if top_user_genres:
            best_genre_labels = [_genre_label(g) for g in top_user_genres[:2]]
            # Compute how much of the platform catalog matches user's top genres
            if len(best_df) > 0:
                ug_set = set(top_user_genres)
                genre_match_pct = float(best_df["genres"].apply(
                    lambda gs: isinstance(gs, list) and bool({str(g).lower() for g in gs} & ug_set)
                ).mean() * 100)
                why_match.append(
                    f"Genre match — {genre_match_pct:.0f}% of {best_display}'s catalog features "
                    f"your top genres ({', '.join(best_genre_labels)})"
                )

        # Recency bullet
        if slider_prefs:
            recency_pref = float(slider_prefs.get("recency", 50))
            post2015 = float((best_df["release_year"] >= 2015).mean() * 100) if len(best_df) > 0 else 0
            pre2000 = float((best_df["release_year"] < 2000).mean() * 100) if len(best_df) > 0 else 0
            if recency_pref >= 65:
                why_match.append(
                    f"Fresh catalog — {post2015:.0f}% of {best_display}'s titles released since 2015, "
                    f"matching your preference for new content"
                )
            elif recency_pref <= 35:
                why_match.append(
                    f"Classic roots — {pre2000:.0f}% of {best_display}'s catalog predates 2000, "
                    f"perfect for your love of timeless titles"
                )

        # Quality bullet
        avg_imdb_v = prof.get("avg_imdb", 0)
        rated_b = best_df.dropna(subset=["imdb_score"])
        prem_pct = (rated_b["imdb_score"] >= 7.0).mean() * 100 if len(rated_b) > 0 else 0
        if avg_imdb_v > 0 and user_avg_quality >= 7.0:
            why_match.append(
                f"Quality standard — avg IMDb {avg_imdb_v:.1f} with {prem_pct:.0f}% of titles "
                f"rated 7.0+, aligned with your taste for well-regarded content"
            )
        elif avg_imdb_v > 0:
            why_match.append(
                f"Strong catalog — {prem_pct:.0f}% of {best_display}'s titles score 7.0+ on IMDb"
            )

        # International bullet
        if slider_prefs:
            intl_pref = float(slider_prefs.get("international", 50))
            intl_val = float(prof.get("international", 50))
            if intl_pref >= 60 and intl_val >= 30:
                why_match.append(
                    f"Global reach — {intl_val:.0f}% international content on {best_display}, "
                    f"matching your taste for world cinema"
                )
            elif intl_pref <= 35 and intl_val <= 40:
                dom_pct = 100 - intl_val
                why_match.append(
                    f"Home-court advantage — {dom_pct:.0f}% US-produced content, "
                    f"aligned with your preference for domestic programming"
                )

        # Awards bullet
        if enriched_stats and best_key in enriched_stats:
            aw = enriched_stats[best_key].get("award_wins", 0) or 0
            noms = enriched_stats[best_key].get("award_noms", 0) or 0
            if aw > 50:
                noms_str = f" ({noms:,} nominations)" if noms > 0 else ""
                why_match.append(
                    f"Award pedigree — {aw:,} wins{noms_str} across the catalog, "
                    f"signalling consistent critical recognition"
                )

    # Cap at 4 bullets
    why_match = why_match[:4]

    # ── 8. Recommendations from best-match platform ───────────────────────────
    # Scoring: quality_score × genre_overlap_fraction × tone_fit
    # genre_overlap_fraction penalises titles that only partially match the
    # user's selected genres — e.g. a film tagged only "comedy" scores lower
    # for a "comedy + romance" user than one tagged both.
    # tone_fit sharply penalises dark/mature titles when user wants feel-good.
    recs: list[dict] = []
    if rankings:
        best_key = rankings[0]["platform"]
        best_df_r = all_df[all_df["platform"] == best_key].copy()

        _rec_genre_set = {g.lower() for g in (user_selected_genres or [])}
        if not _rec_genre_set:
            _rec_genre_set = {g.lower() for g in top_user_genres[:4]}

        # Require at least one matching genre to be in the pool at all
        if _rec_genre_set:
            best_df_r = best_df_r[best_df_r["genres"].apply(
                lambda gs: isinstance(gs, list) and bool({str(g).lower() for g in gs} & _rec_genre_set)
            )]

        seen_ids = {t["id"] for t in all_titles}
        best_df_r = best_df_r[~best_df_r["id"].isin(seen_ids)]

        # Era filter from recency slider (only hard-filter when strongly expressed)
        _rec_era = float(slider_prefs.get("recency", 50)) if slider_prefs else 50.0
        _rec_tone = float(slider_prefs.get("maturity", 50)) if slider_prefs else 50.0
        if _rec_era >= 75:
            best_df_r = best_df_r[best_df_r["release_year"].fillna(0) >= 2015]
        elif _rec_era <= 20:
            best_df_r = best_df_r[best_df_r["release_year"].fillna(9999) <= 2005]

        # Fallback if the era filter emptied the pool
        if len(best_df_r) < 5:
            best_df_r = all_df[all_df["platform"] == best_key].copy()
            if _rec_genre_set:
                best_df_r = best_df_r[best_df_r["genres"].apply(
                    lambda gs: isinstance(gs, list) and bool({str(g).lower() for g in gs} & _rec_genre_set)
                )]
            best_df_r = best_df_r[~best_df_r["id"].isin(seen_ids)]

        if len(best_df_r) > 0:
            best_df_r = best_df_r.copy()
            best_df_r["_qs"] = compute_quality_score(best_df_r)

            _n_sel = max(len(_rec_genre_set), 1)

            def _rec_final_score(row_: pd.Series) -> float:
                """Score = quality × genre_overlap^0.7 × tone_fit."""
                genres_r = row_.get("genres") or []
                if isinstance(genres_r, list):
                    matched = len({str(g).lower() for g in genres_r} & _rec_genre_set)
                else:
                    matched = 0
                # Fractional genre overlap: penalise partial matches but don't zero-out
                genre_frac = (matched / _n_sel) ** 0.7  # 0.0–1.0

                # Tone fit: if user wants feel-good (tone ≤ 35), heavily penalise
                # titles that are dark/mature (R-rated, TV-MA, crime/thriller/horror)
                tone_fit = 1.0
                if _rec_tone <= 35:
                    cert = str(row_.get("age_certification") or "")
                    is_mature_cert = cert in _MATURE_RATINGS
                    is_dark_genre = isinstance(genres_r, list) and bool(
                        {str(g).lower() for g in genres_r} & _DARK_GENRE_SET
                    )
                    if is_mature_cert or is_dark_genre:
                        tone_fit = 0.15  # strong penalty — pushes these to the bottom
                elif _rec_tone >= 70:
                    # User wants dark/intense — boost dark titles
                    cert = str(row_.get("age_certification") or "")
                    is_mature_cert = cert in _MATURE_RATINGS
                    if is_mature_cert:
                        tone_fit = 1.3

                return float(row_.get("_qs", 5.0)) * genre_frac * tone_fit

            best_df_r["_rec_score"] = best_df_r.apply(_rec_final_score, axis=1)

            for threshold in [10_000, 1_000, 0]:
                _pool = best_df_r[best_df_r["imdb_votes"].fillna(0) >= threshold]
                if len(_pool) >= 5:
                    break
            else:
                _pool = best_df_r

            top_recs = _pool.nlargest(5, "_rec_score")

            # Build poster / award maps
            _poster_map: dict = {}
            _award_map: dict = {}
            if enriched_df is not None and not enriched_df.empty:
                if "poster_url" in enriched_df.columns:
                    _poster_map = enriched_df.dropna(subset=["poster_url"]).set_index("id")["poster_url"].to_dict()
                if "award_wins" in enriched_df.columns:
                    _award_map = enriched_df.dropna(subset=["award_wins"]).set_index("id")["award_wins"].to_dict()

            for _, r in top_recs.iterrows():
                tid = r.get("id")
                raw_p = _poster_map.get(tid, "")
                recs.append({
                    "id": tid,
                    "title": r.get("title", ""),
                    "platform": best_key,
                    "imdb_score": r.get("imdb_score"),
                    "imdb_votes": r.get("imdb_votes"),
                    "release_year": r.get("release_year"),
                    "type": r.get("type", ""),
                    "genres": r.get("genres", []),
                    "description": r.get("description", ""),
                    "poster_url": str(raw_p) if raw_p and str(raw_p) != "nan" else None,
                    "award_wins": _award_map.get(tid),
                })

    # ── 9. Per-platform natural-language explanations ─────────────────────────
    # Each platform gets a two-sentence narrative driven by the actual score
    # components so the user understands *why* it ranked the way it did.
    platform_explanations: dict[str, str] = {}
    _sel_genres_display = [_genre_label(g) for g in (user_selected_genres or top_user_genres)[:2]]
    for _r in rankings:
        _pk = _r["platform"]
        _pname = PLATFORMS.get(_pk, {}).get("name", _pk.title())
        _prof = _plat_profiles.get(_pk, {})

        # Sentence 1: genre coverage
        _g_parts = []
        for _sg in (user_selected_genres or top_user_genres)[:2]:
            _sg_l = _sg.lower()
            _g_pct = _genre_pct_by_plat.get(_pk, {}).get(_sg_l, 0)
            _g_parts.append(f"{_genre_label(_sg)} makes up {_g_pct:.0f}% of the catalog")
        if _g_parts:
            _sent1 = f"{_pname}: " + " and ".join(_g_parts) + "."
        else:
            _sent1 = f"{_pname}: broad content library."

        # Sentence 2: vibe alignment — describe what aligns and what diverges
        _vibe_notes = []
        _tone_diff = 0.0  # default so it's always defined for the _sent2 guard below
        if slider_prefs:
            _u_recency = float(slider_prefs.get("recency", 50))
            _u_tone = float(slider_prefs.get("maturity", 50))
            _u_pop = float(slider_prefs.get("popularity", 50))

            _p_recency = _prof.get("recency", 50)
            _p_tone = _prof.get("maturity", 50)
            _p_pop = _prof.get("popularity", 50)
            _post2015 = _prof.get("_post2015_pct", 50)

            # Recency alignment
            if _u_recency >= 60 and _p_recency >= 70:
                _vibe_notes.append(f"{_post2015:.0f}% of titles released after 2015 suits your preference for fresh content")
            elif _u_recency >= 60 and _p_recency < 50:
                _vibe_notes.append(f"the catalog skews older ({100 - _post2015:.0f}% pre-2015), which may not match your preference for newer content")
            elif _u_recency <= 35 and _p_recency <= 40:
                _vibe_notes.append("its deeper back-catalog aligns with your love of classics")

            # Tone alignment
            _tone_diff = _p_tone - _u_tone
            if _u_tone <= 35 and _p_tone >= 55:
                _vibe_notes.append(f"it skews darker/more intense than your feel-good preference (tone score {_p_tone:.0f}/100)")
            elif _u_tone <= 35 and _p_tone <= 35:
                _vibe_notes.append("its lighter, family-friendly tone aligns well with your feel-good preference")
            elif _u_tone >= 65 and _p_tone <= 40:
                _vibe_notes.append("it leans lighter than your preference for dark, intense content")
            elif abs(_tone_diff) <= 20:
                _vibe_notes.append("the content tone closely matches your vibe preference")

            # Popularity alignment
            if _u_pop >= 65 and _p_pop >= 65:
                _vibe_notes.append("mainstream crowd-pleasers dominate the catalog — matching your taste")
            elif _u_pop <= 35 and _p_pop <= 40:
                _vibe_notes.append("its niche, less-mainstream catalog aligns with your hidden-gems preference")
            elif _u_pop >= 65 and _p_pop <= 40:
                _vibe_notes.append("the catalog skews more niche/arthouse than your mainstream preference")

        if _vibe_notes:
            _sent2 = ("However, " if _tone_diff > 20 and _u_tone <= 40 else "").capitalize() + \
                     "; ".join(_vibe_notes[:2]).capitalize() + "."
        else:
            _sent2 = f"Overall match score: {_r['match_pct']:.0f}%."

        platform_explanations[_pk] = f"{_sent1} {_sent2}".strip()

    return {
        "rankings": rankings,
        "personality": personality,
        "recommendations": recs,
        "why_match": why_match,
        "platform_explanations": platform_explanations,
    }
