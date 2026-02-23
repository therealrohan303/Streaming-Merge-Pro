"""
Script 07: Enrich catalog with MovieLens 20M data.

Joins via links.csv (movieId -> imdbId) to add:
  - Tag genome top 20 tags per movie
  - Dense genome vectors for similarity search
  - Average MovieLens rating and rating count

Output:
  - data/enriched/movielens_genome.parquet (imdb_id, top_tags, ml_avg_rating, ml_rating_count)
  - data/precomputed/embeddings/genome_vectors.npy (dense matrix)
  - data/precomputed/embeddings/genome_id_map.parquet (row index -> imdb_id)

Note: Movies-only coverage is expected.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import ENRICHED_DIR, PRECOMPUTED_DIR, PROCESSED_DIR, RAW_DIR

ML_DIR = RAW_DIR / "movielens"
EMBEDDINGS_DIR = PRECOMPUTED_DIR / "embeddings"
TOP_TAGS_K = 20


def load_id_mapping():
    """Load MovieLens -> IMDb ID mapping from links.csv."""
    print("[1/4] Loading ID mapping from links.csv ...")
    links = pd.read_csv(ML_DIR / "links.csv", dtype={"movieId": int, "imdbId": str, "tmdbId": str})
    # MovieLens stores imdbId as numeric (without 'tt' prefix)
    # Pad to 7 digits and add 'tt' prefix to match our format
    links["imdb_id"] = "tt" + links["imdbId"].str.zfill(7)
    links = links[["movieId", "imdb_id"]].dropna()

    # Filter to our catalog
    our_titles = pd.read_parquet(PROCESSED_DIR / "all_platforms_titles.parquet", columns=["imdb_id"])
    our_ids = set(our_titles["imdb_id"].dropna().unique())
    links = links[links["imdb_id"].isin(our_ids)]
    print(f"  {len(links):,} MovieLens movies matched to our catalog (of {len(our_ids):,} imdb_ids)")
    return links


def build_genome_data(links):
    """Build tag genome top tags and dense vectors."""
    print("\n[2/4] Loading genome tags and scores ...")

    # Load tag names
    tags_df = pd.read_csv(ML_DIR / "genome-tags.csv")
    tag_names = dict(zip(tags_df["tagId"], tags_df["tag"]))
    print(f"  {len(tag_names)} genome tags loaded")

    # Load genome scores — this is large (~309 MB)
    print("  Loading genome-scores.csv (this may take a moment) ...")
    scores = pd.read_csv(ML_DIR / "genome-scores.csv",
                         dtype={"movieId": int, "tagId": int, "relevance": float})
    print(f"  {len(scores):,} genome score rows loaded")

    # Filter to our movies
    our_movie_ids = set(links["movieId"].unique())
    scores = scores[scores["movieId"].isin(our_movie_ids)]
    print(f"  {len(scores):,} rows after filtering to our catalog ({scores['movieId'].nunique():,} movies)")

    if len(scores) == 0:
        return pd.DataFrame(columns=["imdb_id", "top_tags"]), None, None

    # Pivot to dense matrix (movies × tags)
    print("  Pivoting to dense matrix ...")
    pivot = scores.pivot(index="movieId", columns="tagId", values="relevance")
    pivot = pivot.fillna(0.0)

    # Map movieId -> imdb_id
    movie_to_imdb = dict(zip(links["movieId"], links["imdb_id"]))
    valid_movie_ids = [mid for mid in pivot.index if mid in movie_to_imdb]
    pivot = pivot.loc[valid_movie_ids]
    imdb_ids = [movie_to_imdb[mid] for mid in pivot.index]

    # Dense genome vectors
    genome_vectors = pivot.values.astype(np.float32)
    id_map = pd.DataFrame({"imdb_id": imdb_ids})
    print(f"  Genome matrix shape: {genome_vectors.shape}")

    # Top K tags per movie
    tag_id_list = pivot.columns.tolist()
    top_tags_list = []
    for row_idx in range(len(pivot)):
        row_vals = pivot.iloc[row_idx].values
        top_indices = np.argsort(row_vals)[-TOP_TAGS_K:][::-1]
        top = [tag_names.get(tag_id_list[i], f"tag_{tag_id_list[i]}") for i in top_indices
               if row_vals[i] > 0.0]
        top_tags_list.append(top[:TOP_TAGS_K])

    top_tags_df = pd.DataFrame({"imdb_id": imdb_ids, "top_tags": top_tags_list})
    return top_tags_df, genome_vectors, id_map


def compute_ratings(links):
    """Compute average rating and count from ratings.csv."""
    print("\n[3/4] Computing MovieLens ratings ...")
    our_movie_ids = set(links["movieId"].unique())

    # Stream ratings to avoid memory issues
    chunk_size = 1_000_000
    rating_sums = {}
    rating_counts = {}
    total_read = 0

    for chunk in pd.read_csv(ML_DIR / "ratings.csv", chunksize=chunk_size,
                             dtype={"movieId": int, "rating": float},
                             usecols=["movieId", "rating"]):
        total_read += len(chunk)
        matched = chunk[chunk["movieId"].isin(our_movie_ids)]
        for mid, rating in zip(matched["movieId"], matched["rating"]):
            rating_sums[mid] = rating_sums.get(mid, 0.0) + rating
            rating_counts[mid] = rating_counts.get(mid, 0) + 1
        if total_read % 5_000_000 == 0:
            print(f"    ... read {total_read:,} rows")

    movie_to_imdb = dict(zip(links["movieId"], links["imdb_id"]))
    rows = []
    for mid in rating_sums:
        if mid in movie_to_imdb:
            rows.append({
                "imdb_id": movie_to_imdb[mid],
                "ml_avg_rating": round(rating_sums[mid] / rating_counts[mid], 3),
                "ml_rating_count": rating_counts[mid],
            })

    df = pd.DataFrame(rows)
    print(f"  Ratings computed for {len(df):,} movies")
    return df


def main():
    ENRICHED_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    links = load_id_mapping()

    # Build genome data
    top_tags_df, genome_vectors, id_map = build_genome_data(links)

    # Compute ratings
    ratings_df = compute_ratings(links)

    # Merge top_tags + ratings
    print("\n[4/4] Merging and saving ...")
    if len(top_tags_df) > 0 and len(ratings_df) > 0:
        result = top_tags_df.merge(ratings_df, on="imdb_id", how="outer")
    elif len(top_tags_df) > 0:
        result = top_tags_df
    elif len(ratings_df) > 0:
        result = ratings_df
    else:
        result = pd.DataFrame(columns=["imdb_id", "top_tags", "ml_avg_rating", "ml_rating_count"])

    # Save enrichment parquet
    output_path = ENRICHED_DIR / "movielens_genome.parquet"
    result.to_parquet(output_path, index=False)
    print(f"  Saved {output_path} ({len(result):,} rows)")

    # Save genome vectors
    if genome_vectors is not None:
        vectors_path = EMBEDDINGS_DIR / "genome_vectors.npy"
        np.save(vectors_path, genome_vectors)
        print(f"  Saved {vectors_path} (shape: {genome_vectors.shape})")

        id_map_path = EMBEDDINGS_DIR / "genome_id_map.parquet"
        id_map.to_parquet(id_map_path, index=False)
        print(f"  Saved {id_map_path} ({len(id_map):,} rows)")

    # Coverage stats
    print(f"\n  top_tags coverage: {result['top_tags'].notna().sum():,} movies")
    if "ml_avg_rating" in result.columns:
        print(f"  ml_avg_rating coverage: {result['ml_avg_rating'].notna().sum():,} movies")
        print(f"  avg ml_avg_rating: {result['ml_avg_rating'].dropna().mean():.2f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
