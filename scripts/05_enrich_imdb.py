"""
Script 05: Enrich catalog with IMDb public datasets.

Reads IMDb TSV files (title.basics, title.principals, name.basics),
filters to our catalog's imdb_id set, and produces:
  - data/enriched/imdb_enrichment.parquet  (originalTitle, isAdult per imdb_id)
  - data/enriched/imdb_principals.parquet  (imdb_id, person_id, name, category, job)
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import ENRICHED_DIR, PROCESSED_DIR, RAW_DIR

IMDB_DIR = RAW_DIR / "imdb"
CHUNK_SIZE = 500_000
KEEP_CATEGORIES = {"director", "writer", "producer", "composer", "cinematographer"}


def load_our_imdb_ids():
    """Load the set of imdb_ids present in our catalog."""
    df = pd.read_parquet(PROCESSED_DIR / "all_platforms_titles.parquet", columns=["imdb_id"])
    ids = set(df["imdb_id"].dropna().unique())
    print(f"  Our catalog has {len(ids):,} unique imdb_ids")
    return ids


def enrich_basics(our_ids):
    """Extract originalTitle and isAdult from title.basics.tsv.gz."""
    print("\n[1/3] Processing title.basics.tsv.gz ...")
    path = IMDB_DIR / "title.basics.tsv.gz"
    rows = []
    total_read = 0
    for chunk in pd.read_csv(
        path, sep="\t", compression="gzip", chunksize=CHUNK_SIZE,
        usecols=["tconst", "originalTitle", "isAdult"],
        dtype={"tconst": str, "originalTitle": str, "isAdult": str},
        na_values="\\N",
    ):
        total_read += len(chunk)
        matched = chunk[chunk["tconst"].isin(our_ids)]
        if len(matched) > 0:
            rows.append(matched)
        if total_read % 2_000_000 == 0:
            print(f"    ... read {total_read:,} rows, matched {sum(len(r) for r in rows):,}")

    if not rows:
        print("  WARNING: No matches found in title.basics")
        return pd.DataFrame(columns=["imdb_id", "original_title", "is_adult"])

    df = pd.concat(rows, ignore_index=True)
    df = df.rename(columns={"tconst": "imdb_id", "originalTitle": "original_title", "isAdult": "is_adult"})
    df["is_adult"] = df["is_adult"].map({"0": False, "1": True}).fillna(False)
    df = df.drop_duplicates(subset="imdb_id")
    print(f"  title.basics: {len(df):,} matched titles")
    return df


def enrich_principals(our_ids):
    """Extract director/writer/producer/composer/cinematographer from title.principals.tsv.gz."""
    print("\n[2/3] Processing title.principals.tsv.gz ...")
    path = IMDB_DIR / "title.principals.tsv.gz"
    rows = []
    total_read = 0
    for chunk in pd.read_csv(
        path, sep="\t", compression="gzip", chunksize=CHUNK_SIZE,
        usecols=["tconst", "nconst", "category", "job"],
        dtype={"tconst": str, "nconst": str, "category": str, "job": str},
        na_values="\\N",
    ):
        total_read += len(chunk)
        matched = chunk[
            (chunk["tconst"].isin(our_ids)) & (chunk["category"].isin(KEEP_CATEGORIES))
        ]
        if len(matched) > 0:
            rows.append(matched)
        if total_read % 5_000_000 == 0:
            print(f"    ... read {total_read:,} rows, matched {sum(len(r) for r in rows):,}")

    if not rows:
        print("  WARNING: No matches found in title.principals")
        return pd.DataFrame(columns=["imdb_id", "person_id", "category", "job"])

    df = pd.concat(rows, ignore_index=True)
    df = df.rename(columns={"tconst": "imdb_id", "nconst": "person_id"})
    print(f"  title.principals: {len(df):,} rows across {df['imdb_id'].nunique():,} titles")
    print(f"  Category breakdown:\n{df['category'].value_counts().to_string()}")
    return df


def add_person_names(principals_df):
    """Join name.basics.tsv.gz to add person names to principals."""
    print("\n[3/3] Processing name.basics.tsv.gz ...")
    person_ids = set(principals_df["person_id"].unique())
    print(f"  Looking up {len(person_ids):,} unique person IDs")

    path = IMDB_DIR / "name.basics.tsv.gz"
    rows = []
    total_read = 0
    for chunk in pd.read_csv(
        path, sep="\t", compression="gzip", chunksize=CHUNK_SIZE,
        usecols=["nconst", "primaryName"],
        dtype={"nconst": str, "primaryName": str},
        na_values="\\N",
    ):
        total_read += len(chunk)
        matched = chunk[chunk["nconst"].isin(person_ids)]
        if len(matched) > 0:
            rows.append(matched)
        if total_read % 2_000_000 == 0:
            print(f"    ... read {total_read:,} rows, matched {sum(len(r) for r in rows):,}")

    if not rows:
        print("  WARNING: No person names found")
        principals_df["name"] = None
        return principals_df

    names_df = pd.concat(rows, ignore_index=True).drop_duplicates(subset="nconst")
    names_df = names_df.rename(columns={"nconst": "person_id", "primaryName": "name"})
    result = principals_df.merge(names_df, on="person_id", how="left")
    print(f"  Names matched: {result['name'].notna().sum():,} / {len(result):,}")
    return result


def main():
    ENRICHED_DIR.mkdir(parents=True, exist_ok=True)
    our_ids = load_our_imdb_ids()

    # 1. Title basics enrichment
    basics_df = enrich_basics(our_ids)
    basics_path = ENRICHED_DIR / "imdb_enrichment.parquet"
    basics_df.to_parquet(basics_path, index=False)
    print(f"\n  Saved {basics_path} ({len(basics_df):,} rows)")

    # 2. Principals enrichment
    principals_df = enrich_principals(our_ids)
    principals_df = add_person_names(principals_df)

    # Reorder columns
    principals_df = principals_df[["imdb_id", "person_id", "name", "category", "job"]]
    principals_path = ENRICHED_DIR / "imdb_principals.parquet"
    principals_df.to_parquet(principals_path, index=False)
    print(f"  Saved {principals_path} ({len(principals_df):,} rows)")

    print("\nDone!")


if __name__ == "__main__":
    main()
