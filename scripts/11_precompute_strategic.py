"""
Script 11: Precompute strategic analysis artifacts.

Produces:
  - data/precomputed/strategic_analysis/prestige_index.parquet
    Award wins per 1,000 titles by platform and genre (normalized)
  - data/precomputed/strategic_analysis/acquisition_targets.parquet
    Gap recommendations with decision trace fields

Uses titles_enriched.parquet as primary source with graceful fallback.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    ALL_PLATFORMS,
    ENRICHED_DIR,
    MERGED_PLATFORMS,
    PRECOMPUTED_DIR,
    PROCESSED_DIR,
)

STRATEGIC_DIR = PRECOMPUTED_DIR / "strategic_analysis"
TOP_GENRES = 15
GAP_THRESHOLDS = {
    "genre_share": 0.05,  # < 5% of catalog
    "avg_imdb": 6.5,      # avg IMDb < 6.5
    "decade_count": 100,   # < 100 titles in a decade
}


def load_titles():
    """Load enriched or base titles."""
    enriched_path = ENRICHED_DIR / "titles_enriched.parquet"
    if enriched_path.exists():
        df = pd.read_parquet(enriched_path)
        print(f"Loaded enriched titles: {len(df):,} rows")
        return df
    base_path = PROCESSED_DIR / "all_platforms_titles.parquet"
    df = pd.read_parquet(base_path)
    print(f"Loaded base titles (enriched not found): {len(df):,} rows")
    return df


def build_merged_entity(df):
    """Create merged Netflix+Max entity, deduplicating by id."""
    merged = df[df["platform"].isin(MERGED_PLATFORMS)].copy()
    merged = merged.drop_duplicates(subset="id", keep="first")
    merged["platform"] = "merged"
    return merged


def compute_prestige_index(df):
    """Compute award wins per 1,000 titles by platform and genre."""
    if "award_wins" not in df.columns or df["award_wins"].notna().sum() == 0:
        print("  SKIPPED prestige_index: no award data available")
        return pd.DataFrame()

    # Add merged entity
    merged = build_merged_entity(df)
    all_df = pd.concat([df, merged], ignore_index=True)

    # Explode genres
    exploded = all_df.explode("genres")
    exploded = exploded[exploded["genres"].notna()]

    # Top genres
    top_genres = exploded["genres"].value_counts().nlargest(TOP_GENRES).index.tolist()
    exploded = exploded[exploded["genres"].isin(top_genres)]

    # Compute prestige per platform × genre
    rows = []
    platforms = ALL_PLATFORMS + ["merged"]
    for platform in platforms:
        pdata = exploded[exploded["platform"] == platform]
        for genre in top_genres:
            gdata = pdata[pdata["genres"] == genre]
            total = len(gdata)
            if total == 0:
                continue
            wins = gdata["award_wins"].fillna(0).sum()
            noms = gdata["award_noms"].fillna(0).sum() if "award_noms" in gdata.columns else 0
            has_awards = (gdata["award_wins"].notna() & (gdata["award_wins"] > 0)).sum()
            coverage = gdata["award_wins"].notna().sum() / total if total > 0 else 0

            # Prestige per 1,000 titles
            prestige = (wins / total) * 1000 if total > 0 else 0

            rows.append({
                "platform": platform,
                "genre": genre,
                "title_count": total,
                "award_wins": int(wins),
                "award_noms": int(noms),
                "titles_with_awards": has_awards,
                "prestige_per_1k": round(prestige, 2),
                "coverage": round(coverage, 3),
            })

    result = pd.DataFrame(rows)
    print(f"  Prestige index: {len(result):,} rows "
          f"({result['platform'].nunique()} platforms × {result['genre'].nunique()} genres)")
    return result


def compute_acquisition_targets(df):
    """Compute gap analysis with decision trace fields."""
    merged = build_merged_entity(df)

    # Explode genres for merged
    merged_exploded = merged.explode("genres")
    merged_exploded = merged_exploded[merged_exploded["genres"].notna()]

    merged_total = len(merged)
    all_genres = merged_exploded["genres"].value_counts()

    # Compute gaps for each competitor
    rows = []
    competitors = [p for p in ALL_PLATFORMS if p not in MERGED_PLATFORMS]

    for competitor in competitors:
        comp_data = df[df["platform"] == competitor]
        comp_exploded = comp_data.explode("genres")
        comp_exploded = comp_exploded[comp_exploded["genres"].notna()]
        comp_genres = comp_exploded["genres"].value_counts()

        # Genre gaps
        for genre in comp_genres.index:
            merged_count = all_genres.get(genre, 0)
            comp_count = comp_genres[genre]
            merged_share = merged_count / merged_total if merged_total > 0 else 0
            comp_share = comp_count / len(comp_data) if len(comp_data) > 0 else 0

            # Is this a gap? (competitor has > 2x our share, or we have < 5%)
            if merged_share >= comp_share:
                continue

            ratio = comp_share / merged_share if merged_share > 0 else 10.0

            # Quality benchmark
            merged_genre_titles = merged_exploded[merged_exploded["genres"] == genre]
            comp_genre_titles = comp_exploded[comp_exploded["genres"] == genre]
            merged_avg_imdb = merged_genre_titles["imdb_score"].mean()
            comp_avg_imdb = comp_genre_titles["imdb_score"].mean()

            # Severity
            if ratio >= 3.0 or merged_share < 0.03:
                severity = "High"
            elif ratio >= 1.5 or merged_share < 0.05:
                severity = "Medium"
            else:
                severity = "Low"

            # Box office tier
            box_office_tier = "Unknown"
            if "box_office_usd" in comp_genre_titles.columns:
                bo = comp_genre_titles["box_office_usd"].dropna()
                if len(bo) > 5:
                    median_bo = bo.median()
                    if median_bo > 200_000_000:
                        box_office_tier = "High"
                    elif median_bo > 50_000_000:
                        box_office_tier = "Mid"
                    else:
                        box_office_tier = "Low"

            # Data confidence
            confidence = "High"
            if "data_confidence" in comp_genre_titles.columns:
                avg_conf = comp_genre_titles["data_confidence"].mean()
                if avg_conf < 0.2:
                    confidence = "Low"
                elif avg_conf < 0.5:
                    confidence = "Medium"

            # Year range for recommendation
            recent = comp_genre_titles[comp_genre_titles["release_year"] >= 2018]
            year_range = "2018-2023" if len(recent) > 10 else "2015-2023"

            # Acquisition recommendation
            target_count = min(max(int(comp_count * 0.3), 10), 50)
            min_imdb = max(round(comp_avg_imdb - 0.5, 1), 6.5) if pd.notna(comp_avg_imdb) else 7.0
            recommendation = (
                f"Acquire {target_count}-{target_count + 20} titles, "
                f"{genre}, {min_imdb}+ IMDb, {year_range}"
            )

            rows.append({
                "gap_type": "genre_share",
                "genre": genre,
                "competitor": competitor,
                "severity": severity,
                "merged_share": round(merged_share, 4),
                "competitor_share": round(comp_share, 4),
                "competitor_lead": round(ratio, 2),
                "merged_count": merged_count,
                "competitor_count": comp_count,
                "merged_avg_imdb": round(merged_avg_imdb, 2) if pd.notna(merged_avg_imdb) else None,
                "competitor_avg_imdb": round(comp_avg_imdb, 2) if pd.notna(comp_avg_imdb) else None,
                "box_office_tier": box_office_tier,
                "confidence": confidence,
                "recommendation": recommendation,
            })

    result = pd.DataFrame(rows)
    if len(result) > 0:
        result = result.sort_values(["severity", "competitor_lead"], ascending=[True, False])
    print(f"  Acquisition targets: {len(result):,} gaps across {result['competitor'].nunique()} competitors")
    return result


def main():
    STRATEGIC_DIR.mkdir(parents=True, exist_ok=True)

    df = load_titles()

    # Prestige Index
    prestige = compute_prestige_index(df)
    if len(prestige) > 0:
        prestige_path = STRATEGIC_DIR / "prestige_index.parquet"
        prestige.to_parquet(prestige_path, index=False)
        print(f"  Saved {prestige_path}")

    # Acquisition Targets
    targets = compute_acquisition_targets(df)
    if len(targets) > 0:
        targets_path = STRATEGIC_DIR / "acquisition_targets.parquet"
        targets.to_parquet(targets_path, index=False)
        print(f"  Saved {targets_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
