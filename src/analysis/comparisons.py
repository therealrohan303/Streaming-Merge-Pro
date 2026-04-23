"""Platform comparison analysis for the merged Netflix+Max entity vs competitors."""

import pandas as pd

from src.analysis.scoring import compute_quality_score
from src.config import COMPARISON_TOP_GENRES, MERGED_PLATFORMS, PLATFORMS, QUALITY_TIERS
from src.ui.formatting import genre_display


def _fix_genre_labels(index: pd.Index) -> pd.Index:
    """Apply special-case genre display names after title-casing."""
    return index.map(lambda g: genre_display(g))


def _explode_genres(df: pd.DataFrame) -> pd.DataFrame:
    """Explode the genres list column into one row per title-genre pair."""
    exploded = df[df["genres"].apply(lambda g: isinstance(g, list) and len(g) > 0)].copy()
    return exploded.explode("genres")


def build_merged_entity(df: pd.DataFrame) -> pd.DataFrame:
    """Combine Netflix + Max rows into a single 'merged' platform entity.

    Deduplicates by id so titles on both platforms count once.
    """
    merged = df[df["platform"].isin(MERGED_PLATFORMS)].copy()
    merged["platform"] = "merged"
    merged = merged.drop_duplicates(subset="id", keep="first")
    return merged


def build_comparison_df(
    df: pd.DataFrame,
    competitor_keys: list[str],
) -> pd.DataFrame:
    """Build DataFrame with merged entity + selected competitors.

    Adds a 'platform_display' column with human-readable names.
    """
    merged = build_merged_entity(df)
    competitors = df[df["platform"].isin(competitor_keys)].copy()
    comp_df = pd.concat([merged, competitors], ignore_index=True)

    comp_df["platform_display"] = comp_df["platform"].map(
        {k: v["name"] for k, v in PLATFORMS.items()}
    )

    # Enforce ordering: merged first, then competitors alphabetically by display name
    platform_order = ["merged"] + sorted(competitor_keys)
    comp_df["platform"] = pd.Categorical(
        comp_df["platform"], categories=platform_order, ordered=True
    )
    comp_df = comp_df.sort_values("platform").reset_index(drop=True)

    return comp_df


def compute_volume_stats(
    comp_df: pd.DataFrame,
    normalize: bool = False,
) -> pd.DataFrame:
    """Title counts per platform and type, optionally as percentages."""
    grouped = (
        comp_df.groupby(["platform", "platform_display", "type"], observed=True)
        .size()
        .reset_index(name="count")
    )
    if normalize:
        totals = grouped.groupby("platform", observed=True)["count"].transform("sum")
        grouped["pct"] = (grouped["count"] / totals * 100).round(1)
    return grouped


def compute_volume_summary(comp_df: pd.DataFrame) -> pd.DataFrame:
    """One-row-per-platform summary: total titles, movies, shows, avg/median IMDb."""
    summary = (
        comp_df.groupby(["platform", "platform_display"], observed=True)
        .agg(
            total_titles=("id", "count"),
            movies=("type", lambda s: (s == "Movie").sum()),
            shows=("type", lambda s: (s == "Show").sum()),
            avg_imdb=("imdb_score", "mean"),
        )
        .reset_index()
    )
    summary["avg_imdb"] = summary["avg_imdb"].round(2)
    return summary


def compute_quality_tiers(
    comp_df: pd.DataFrame,
    tiers: dict,
    normalize: bool = False,
) -> pd.DataFrame:
    """Count/percentage of titles per quality tier per platform.

    Args:
        tiers: mapping of tier name -> (low, high) IMDb range.
        normalize: if True, values are percentages of platform total.

    Returns pivoted DataFrame: index=tier, columns=platform display names.
    """
    records = []
    for platform_display, grp in comp_df.groupby("platform_display", observed=True):
        total = len(grp)
        for tier_name, (low, high) in tiers.items():
            # Use "both" for the top tier so 10.0 scores are included
            inclusive = "both" if high >= 10.0 else "left"
            count = grp["imdb_score"].between(low, high, inclusive=inclusive).sum()
            records.append(
                {
                    "tier": tier_name,
                    "platform_display": platform_display,
                    "count": int(count),
                    "pct": round(count / total * 100, 1) if total else 0.0,
                }
            )
        # Unrated
        unrated = int(grp["imdb_score"].isna().sum())
        records.append(
            {
                "tier": "Unrated",
                "platform_display": platform_display,
                "count": unrated,
                "pct": round(unrated / total * 100, 1) if total else 0.0,
            }
        )

    result = pd.DataFrame(records)
    value_col = "pct" if normalize else "count"
    pivoted = result.pivot(
        index="tier", columns="platform_display", values=value_col
    )

    # Order tiers logically
    tier_order = list(tiers.keys()) + ["Unrated"]
    pivoted = pivoted.reindex(tier_order)
    return pivoted


def compute_genre_heatmap(
    comp_df: pd.DataFrame,
    top_n: int = 15,
    normalize: bool = False,
) -> pd.DataFrame:
    """Genre-by-platform matrix for the top N genres.

    Returns pivoted DataFrame: index=genre (title-cased), columns=platform display names.
    """
    exploded = _explode_genres(comp_df)

    # Count by genre + platform
    counts = (
        exploded.groupby(["genres", "platform_display"], observed=True)
        .size()
        .reset_index(name="count")
    )

    # Identify top N genres by overall volume
    top_genres = (
        counts.groupby("genres")["count"]
        .sum()
        .nlargest(top_n)
        .index.tolist()
    )
    counts = counts[counts["genres"].isin(top_genres)]

    # Pivot
    pivoted = counts.pivot(
        index="genres", columns="platform_display", values="count"
    ).fillna(0).astype(int)

    if normalize:
        # Normalize: count / platform total titles * 100
        platform_totals = comp_df.groupby("platform_display", observed=True).size()
        pivoted = (pivoted / platform_totals * 100).round(1)

    # Sort genres by total count descending
    pivoted["_total"] = pivoted.sum(axis=1)
    pivoted = pivoted.sort_values("_total", ascending=False).drop(columns="_total")

    # Title-case genre names for display, with special-case fixes
    pivoted.index = _fix_genre_labels(pivoted.index)

    # Ensure merged platform column is first
    merged_name = PLATFORMS["merged"]["name"]
    if merged_name in pivoted.columns:
        cols = [merged_name] + [c for c in pivoted.columns if c != merged_name]
        pivoted = pivoted[cols]

    return pivoted


def compute_genre_drilldown(
    comp_df: pd.DataFrame,
    genre: str,
    top_titles_n: int = 5,
) -> dict:
    """Drill-down data for a specific genre across platforms.

    Args:
        genre: raw genre key (lowercase).
        top_titles_n: number of top-rated titles to return per platform.

    Returns:
        dict with:
            'stats': DataFrame [platform_display, count, avg_imdb, median_imdb, pct_of_catalog]
            'top_titles': dict mapping platform_display -> DataFrame of top titles
    """
    # Filter to titles containing this genre
    mask = comp_df["genres"].apply(
        lambda g: genre in g if isinstance(g, list) else False
    )
    genre_df = comp_df[mask]

    # Per-platform totals for pct calculation
    platform_totals = comp_df.groupby("platform_display", observed=True).size()

    # Stats per platform
    stats = (
        genre_df.groupby("platform_display", observed=True)
        .agg(
            count=("id", "count"),
            avg_imdb=("imdb_score", "mean"),
        )
        .reset_index()
    )
    stats["avg_imdb"] = stats["avg_imdb"].round(2)
    stats["pct_of_catalog"] = (
        stats.apply(
            lambda r: round(r["count"] / platform_totals.get(r["platform_display"], 1) * 100, 1),
            axis=1,
        )
    )

    # Ensure merged first
    merged_name = PLATFORMS["merged"]["name"]
    if merged_name in stats["platform_display"].values:
        merged_row = stats[stats["platform_display"] == merged_name]
        others = stats[stats["platform_display"] != merged_name]
        stats = pd.concat([merged_row, others], ignore_index=True)

    # Top titles per platform — ranked by quality score with vote floor
    _VOTE_THRESHOLDS = [10_000, 1_000, 0]
    top_titles = {}
    for platform_display, grp in genre_df.groupby("platform_display", observed=True):
        rated = grp.dropna(subset=["imdb_score"]).copy()
        if rated.empty:
            top_titles[platform_display] = rated[
                ["title", "type", "release_year", "imdb_score", "imdb_votes"]
            ].head(0)
            continue

        rated["quality_score"] = compute_quality_score(rated)

        # Try progressively relaxed vote thresholds until we have enough
        for threshold in _VOTE_THRESHOLDS:
            pool = rated[rated["imdb_votes"].fillna(0) >= threshold]
            if len(pool) >= top_titles_n:
                break
        else:
            pool = rated

        _keep = [
            "id", "title", "type", "release_year", "imdb_score", "imdb_votes",
            "description", "age_certification", "runtime", "genres", "tmdb_popularity",
        ]
        _keep = [c for c in _keep if c in pool.columns]
        top = (
            pool
            .nlargest(top_titles_n, "quality_score")[_keep]
            .reset_index(drop=True)
        )
        top_titles[platform_display] = top

    return {"stats": stats, "top_titles": top_titles}


def compute_quick_comparison(
    comp_df: pd.DataFrame,
    top_n_genres: int = COMPARISON_TOP_GENRES,
) -> dict:
    """Compute a TL;DR summary comparing platforms.

    Returns dict with keys:
        volume_leader, volume_count, quality_leader, quality_avg,
        merged_genre_leads, total_genres
    """
    merged_name = PLATFORMS["merged"]["name"]

    # Volume leader
    vol = comp_df.groupby("platform_display", observed=True).size()
    volume_leader = vol.idxmax()
    volume_count = int(vol.max())

    # Quality leader (avg IMDb, excluding NaN)
    qual = comp_df.groupby("platform_display", observed=True)["imdb_score"].mean()
    quality_leader = qual.idxmax()
    quality_avg = float(qual.max())

    # Genre leadership: who leads the most genres (top N)?
    heatmap = compute_genre_heatmap(comp_df, top_n=top_n_genres, normalize=False)
    genre_leaders = heatmap.idxmax(axis=1)  # Series: genre -> platform_display
    merged_leads = int((genre_leaders == merged_name).sum())

    return {
        "volume_leader": volume_leader,
        "volume_count": volume_count,
        "quality_leader": quality_leader,
        "quality_avg": round(quality_avg, 2),
        "merged_genre_leads": merged_leads,
        "total_genres": len(heatmap),
    }


def compute_market_positioning(comp_df: pd.DataFrame) -> pd.DataFrame:
    """Per-platform bubble chart data: volume, avg IMDb, total popularity."""
    stats = (
        comp_df.groupby(["platform", "platform_display"], observed=True)
        .agg(
            total_titles=("id", "count"),
            avg_imdb=("imdb_score", "mean"),
            total_popularity=("tmdb_popularity", "sum"),
        )
        .reset_index()
    )
    stats["avg_imdb"] = stats["avg_imdb"].round(2)
    stats["total_popularity"] = stats["total_popularity"].fillna(0)
    return stats


# Canonical age certification ordering (mature → family-friendly)
_CERT_ORDER = ["TV-MA", "R", "NC-17", "TV-14", "PG-13", "TV-PG", "PG", "TV-G", "G", "TV-Y7", "TV-Y"]


def compute_age_profile(comp_df: pd.DataFrame) -> pd.DataFrame:
    """Stacked bar data: % of titles per age certification per platform."""
    df = comp_df.copy()
    df["certification"] = df["age_certification"].fillna("Unknown")
    # Coalesce rare certifications into "Other"
    known = set(_CERT_ORDER) | {"Unknown"}
    df.loc[~df["certification"].isin(known), "certification"] = "Other"

    counts = (
        df.groupby(["platform_display", "certification"], observed=True)
        .size()
        .reset_index(name="count")
    )
    totals = counts.groupby("platform_display", observed=True)["count"].transform("sum")
    counts["pct"] = (counts["count"] / totals * 100).round(1)

    # Order certifications
    cert_order = _CERT_ORDER + ["Other", "Unknown"]
    counts["certification"] = pd.Categorical(
        counts["certification"], categories=cert_order, ordered=True
    )
    return counts.sort_values(["platform_display", "certification"]).reset_index(drop=True)


def compute_geographic_diversity(comp_df: pd.DataFrame) -> pd.DataFrame:
    """% of non-US titles per platform."""
    records = []
    for platform_display, grp in comp_df.groupby("platform_display", observed=True):
        total = len(grp)
        international = grp["production_countries"].apply(
            lambda c: "US" not in c if isinstance(c, list) and len(c) > 0 else True
        ).sum()
        records.append({
            "platform_display": platform_display,
            "international_pct": round(international / total * 100, 1) if total else 0.0,
            "total_titles": total,
        })
    return pd.DataFrame(records)


def compute_era_focus(comp_df: pd.DataFrame) -> pd.DataFrame:
    """Median release year per platform."""
    stats = (
        comp_df.groupby("platform_display", observed=True)["release_year"]
        .median()
        .reset_index()
        .rename(columns={"release_year": "median_year"})
    )
    stats["median_year"] = stats["median_year"].astype(int)
    return stats


def compute_strategic_insights(
    comp_df: pd.DataFrame,
    heatmap_data: pd.DataFrame,
    tier_df: pd.DataFrame,
) -> dict:
    """Per-competitor competitive snapshot vs merged entity.

    Returns dict mapping platform_display -> {
        "summary": {"comp_vol", "merged_vol", "comp_avg_imdb", "merged_avg_imdb"},
        "their_strengths": [{"category", "theirs", "ours", "diff_pct"}, ...] (top 4),
        "our_strengths": [{"category", "theirs", "ours", "diff_pct"}, ...] (top 4),
        "battlegrounds": [genre_name, ...],
    }
    Values in their_strengths/our_strengths reflect the heatmap_data units
    (raw counts or % of catalog depending on what was passed).
    """
    merged_name = PLATFORMS["merged"]["name"]
    if merged_name not in heatmap_data.columns:
        return {}

    merged_genre_counts = heatmap_data[merged_name]

    # Volume per platform (always raw counts for summary)
    vol = comp_df.groupby("platform_display", observed=True).size()
    merged_vol = vol.get(merged_name, 0)

    # Avg IMDb per platform (for summary)
    avg_imdb = comp_df.groupby("platform_display", observed=True)["imdb_score"].mean()
    merged_avg = float(avg_imdb.get(merged_name, 0))

    results = {}
    for col in heatmap_data.columns:
        if col == merged_name:
            continue

        comp_vol = vol.get(col, 0)
        comp_avg = float(avg_imdb.get(col, 0))

        their_strengths = []
        our_strengths = []
        battlegrounds = []

        comp_genre_counts = heatmap_data[col]

        # Genre-by-genre comparison (uses heatmap values — counts or %)
        for genre in heatmap_data.index:
            m_val = merged_genre_counts.get(genre, 0)
            c_val = comp_genre_counts.get(genre, 0)
            if m_val == 0 and c_val == 0:
                continue
            bigger = max(m_val, c_val)
            diff = abs(c_val - m_val)
            if bigger > 0 and diff / bigger <= 0.10:
                battlegrounds.append(genre)
            elif c_val > m_val:
                pct = round((c_val - m_val) / m_val * 100) if m_val > 0 else 999
                their_strengths.append(
                    {"category": genre, "theirs": c_val, "ours": m_val, "diff_pct": pct}
                )
            elif m_val > c_val:
                pct = round((m_val - c_val) / c_val * 100) if c_val > 0 else 999
                our_strengths.append(
                    {"category": genre, "theirs": c_val, "ours": m_val, "diff_pct": pct}
                )

        # Quality tier comparison
        if merged_name in tier_df.columns and col in tier_df.columns:
            for tier_name in tier_df.index:
                m_tier = tier_df.loc[tier_name, merged_name]
                c_tier = tier_df.loc[tier_name, col]
                if pd.isna(m_tier) or pd.isna(c_tier):
                    continue
                if c_tier > m_tier and m_tier > 0:
                    pct = round((c_tier - m_tier) / m_tier * 100)
                    their_strengths.append(
                        {"category": f"{tier_name} tier", "theirs": c_tier, "ours": m_tier, "diff_pct": pct}
                    )
                elif m_tier > c_tier and c_tier > 0:
                    pct = round((m_tier - c_tier) / c_tier * 100)
                    our_strengths.append(
                        {"category": f"{tier_name} tier", "theirs": c_tier, "ours": m_tier, "diff_pct": pct}
                    )

        # Sort by magnitude and keep top 4
        their_strengths.sort(key=lambda x: x["diff_pct"], reverse=True)
        our_strengths.sort(key=lambda x: x["diff_pct"], reverse=True)

        results[col] = {
            "summary": {
                "comp_vol": int(comp_vol),
                "merged_vol": int(merged_vol),
                "comp_avg_imdb": round(comp_avg, 2),
                "merged_avg_imdb": round(merged_avg, 2),
            },
            "their_strengths": their_strengths[:4],
            "our_strengths": our_strengths[:4],
            "battlegrounds": battlegrounds[:4],
        }

    return results
