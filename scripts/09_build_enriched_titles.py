"""
Script 09: Build the master enriched titles table.

Left-joins all enrichment parquets onto all_platforms_titles on imdb_id:
  - imdb_enrichment.parquet (originalTitle, isAdult)
  - wikidata_enrichment.parquet (budget_usd, box_office_usd, award_wins, award_noms)
  - movielens_genome.parquet (top_tags, ml_avg_rating, ml_rating_count)
  - tmdb_enrichment.parquet (tmdb_keywords, collection_name, production_companies, poster_url)
  - imdb_principals.parquet (aggregated into imdb_writers, imdb_producers lists)

Computes a single data_confidence float per row (fraction of enrichment fields non-null).

Output: data/enriched/titles_enriched.parquet
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import ENRICHED_DIR, PROCESSED_DIR


ENRICHMENT_FILES = {
    "imdb": "imdb_enrichment.parquet",
    "wikidata": "wikidata_enrichment.parquet",
    "movielens": "movielens_genome.parquet",
    "tmdb": "tmdb_enrichment.parquet",
}

# Columns used to compute data_confidence
ENRICHMENT_FIELDS = [
    "original_title", "is_adult",
    "budget_usd", "box_office_usd", "award_wins", "award_noms",
    "top_tags", "ml_avg_rating", "ml_rating_count",
    "tmdb_keywords", "collection_name", "production_companies", "poster_url",
    "imdb_writers", "imdb_producers",
]


def aggregate_principals():
    """Aggregate imdb_principals into per-title writer and producer lists."""
    principals_path = ENRICHED_DIR / "imdb_principals.parquet"
    if not principals_path.exists():
        print("  WARNING: imdb_principals.parquet not found, skipping writer/producer aggregation")
        return pd.DataFrame(columns=["imdb_id", "imdb_writers", "imdb_producers"])

    df = pd.read_parquet(principals_path)
    print(f"  Principals: {len(df):,} rows")

    # Aggregate writers
    writers = (
        df[df["category"] == "writer"]
        .groupby("imdb_id")["name"]
        .apply(list)
        .reset_index()
        .rename(columns={"name": "imdb_writers"})
    )

    # Aggregate producers
    producers = (
        df[df["category"] == "producer"]
        .groupby("imdb_id")["name"]
        .apply(list)
        .reset_index()
        .rename(columns={"name": "imdb_producers"})
    )

    result = writers.merge(producers, on="imdb_id", how="outer")
    print(f"  Writers for {len(writers):,} titles, producers for {len(producers):,} titles")
    return result


def compute_data_confidence(df):
    """Compute fraction of enrichment fields that are non-null per row."""
    present_fields = [f for f in ENRICHMENT_FIELDS if f in df.columns]
    if not present_fields:
        df["data_confidence"] = 0.0
        return df

    def row_confidence(row):
        non_null = 0
        for field in present_fields:
            val = row[field]
            if val is not None and not (isinstance(val, float) and pd.isna(val)):
                # For numeric fields, also check for 0 on award columns
                if field in ("award_wins", "award_noms") and val == 0:
                    continue
                non_null += 1
        return non_null / len(present_fields)

    df["data_confidence"] = df.apply(row_confidence, axis=1)
    return df


def main():
    # Load base titles
    base_path = PROCESSED_DIR / "all_platforms_titles.parquet"
    base = pd.read_parquet(base_path)
    print(f"Base titles: {len(base):,} rows, {len(base.columns)} columns")

    # Join each enrichment source
    for source_name, filename in ENRICHMENT_FILES.items():
        path = ENRICHED_DIR / filename
        if path.exists():
            enrich = pd.read_parquet(path)
            # Drop the source's own data_confidence if it has one (we compute a unified one)
            if "data_confidence" in enrich.columns:
                enrich = enrich.drop(columns=["data_confidence"])
            base = base.merge(enrich, on="imdb_id", how="left")
            print(f"  Joined {source_name}: {len(enrich):,} rows -> {len(base):,} total")
        else:
            print(f"  SKIPPED {source_name}: {path} not found")

    # Aggregate and join principals (writers + producers)
    principals = aggregate_principals()
    if len(principals) > 0:
        base = base.merge(principals, on="imdb_id", how="left")
        print(f"  Joined principals: {len(principals):,} rows")

    # Compute unified data_confidence
    base = compute_data_confidence(base)

    # Save
    output_path = ENRICHED_DIR / "titles_enriched.parquet"
    base.to_parquet(output_path, index=False)
    print(f"\nSaved {output_path}")
    print(f"  {len(base):,} rows, {len(base.columns)} columns")
    print(f"  Columns: {sorted(base.columns.tolist())}")

    # Coverage summary
    print("\nEnrichment coverage:")
    for field in ENRICHMENT_FIELDS:
        if field in base.columns:
            if base[field].dtype == "object":
                non_null = base[field].notna().sum()
            else:
                non_null = base[field].notna().sum()
            pct = non_null / len(base) * 100
            print(f"  {field}: {non_null:,} ({pct:.1f}%)")
    print(f"\n  avg data_confidence: {base['data_confidence'].mean():.3f}")
    print(f"  median data_confidence: {base['data_confidence'].median():.3f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
