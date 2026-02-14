"""
03_compute_similarity.py
Compute multi-signal similarity and store top-K neighbors per title.

Combines four signals:
  1. Description TF-IDF cosine similarity (30%) — thematic/plot overlap
  2. Genre cosine similarity (30%) — genre profile overlap
  3. Type match bonus (15%) — same content type preference
  4. Quality (IMDb) proximity (25%) — similar quality tier

Outputs:
  - data/precomputed/similarity/tfidf_top_k.parquet
  - models/tfidf_vectorizer.pkl
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    MODELS_DIR,
    PRECOMPUTED_DIR,
    PROCESSED_DIR,
    SIMILARITY_STORE_TOP_K,
    TFIDF_MAX_DF,
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
)

BATCH_SIZE = 2000  # Rows per batch (limits memory)

# Multi-signal weights (tuned for quality recommendations)
W_DESCRIPTION = 0.30  # TF-IDF cosine on descriptions
W_GENRE = 0.30  # Cosine similarity on genre vectors
W_TYPE = 0.15  # Same type bonus
W_QUALITY = 0.25  # IMDb proximity (groups prestige titles together)


def _build_genre_matrix(genres_list: list[list[str]]) -> tuple[np.ndarray, list[str]]:
    """Build a binary genre matrix (n_titles x n_genres) for cosine computation."""
    all_genres: set[str] = set()
    for genres in genres_list:
        if isinstance(genres, (list, np.ndarray)):
            all_genres.update(str(g).lower() for g in genres)
    genre_vocab = sorted(all_genres)
    genre_to_idx = {g: i for i, g in enumerate(genre_vocab)}

    matrix = np.zeros((len(genres_list), len(genre_vocab)), dtype=np.float32)
    for row_idx, genres in enumerate(genres_list):
        if isinstance(genres, (list, np.ndarray)):
            for g in genres:
                g_lower = str(g).lower()
                if g_lower in genre_to_idx:
                    matrix[row_idx, genre_to_idx[g_lower]] = 1.0
    return matrix, genre_vocab


def main():
    print("=== 03_compute_similarity (multi-signal) ===")

    # 1. Load and deduplicate
    all_titles = pd.read_parquet(PROCESSED_DIR / "all_platforms_titles.parquet")
    for col in ("genres", "production_countries"):
        if col in all_titles.columns:
            all_titles[col] = all_titles[col].apply(
                lambda x: list(x) if hasattr(x, "tolist") else x
            )

    deduped = all_titles.drop_duplicates(subset="id", keep="first").reset_index(
        drop=True
    )
    print(f"  Deduplicated: {len(all_titles)} rows -> {len(deduped)} unique titles")

    id_index = deduped["id"].values

    # 2. Signal A: Description TF-IDF (description only — genres handled separately)
    print("  Building TF-IDF on descriptions...")
    descriptions = deduped["description"].fillna("")
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        ngram_range=TFIDF_NGRAM_RANGE,
        stop_words="english",
    )
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    print(f"  TF-IDF matrix: {tfidf_matrix.shape}")

    # 3. Signal B: Genre binary matrix (for cosine similarity)
    print("  Building genre matrix...")
    genre_matrix, genre_vocab = _build_genre_matrix(deduped["genres"].tolist())
    print(f"  Genre matrix: {genre_matrix.shape} ({len(genre_vocab)} unique genres)")

    # Precompute genre norms for cosine similarity
    genre_norms = np.linalg.norm(genre_matrix, axis=1, keepdims=True)
    genre_norms = np.maximum(genre_norms, 1e-10)  # avoid division by zero

    # 4. Signal C: Type vector (for type matching)
    type_values = deduped["type"].fillna("").values  # "Movie" or "Show"

    # 5. Signal D: IMDb scores (for quality proximity)
    imdb_scores = deduped["imdb_score"].fillna(0.0).values.astype(np.float32)

    # 6. Compute multi-signal top-K neighbors in batches
    records = []
    n = len(deduped)
    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        batch_size = end - start

        # A: Description cosine similarity (sklearn handles sparse efficiently)
        desc_sim = cosine_similarity(tfidf_matrix[start:end], tfidf_matrix)

        # B: Genre cosine similarity (manual for dense binary vectors)
        batch_genres = genre_matrix[start:end]
        batch_norms = genre_norms[start:end]
        genre_dot = batch_genres @ genre_matrix.T
        genre_sim = genre_dot / (batch_norms * genre_norms.T)
        genre_sim = np.clip(genre_sim, 0.0, 1.0)

        # C: Type match (1.0 = same type, 0.3 = different type)
        batch_types = type_values[start:end]
        type_sim = np.where(
            batch_types[:, np.newaxis] == type_values[np.newaxis, :],
            1.0,
            0.3,
        )

        # D: Quality proximity = 1 - |imdb_a - imdb_b| / 10
        batch_imdb = imdb_scores[start:end]
        quality_sim = 1.0 - np.abs(
            batch_imdb[:, np.newaxis] - imdb_scores[np.newaxis, :]
        ) / 10.0
        quality_sim = np.clip(quality_sim, 0.0, 1.0)

        # Weighted combination
        combined = (
            W_DESCRIPTION * desc_sim
            + W_GENRE * genre_sim
            + W_TYPE * type_sim
            + W_QUALITY * quality_sim
        )

        for local_idx in range(batch_size):
            global_idx = start + local_idx
            scores = combined[local_idx]
            scores[global_idx] = 0.0  # zero out self-similarity

            top_k = min(SIMILARITY_STORE_TOP_K, n - 1)
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

            for rank, neighbor_idx in enumerate(top_indices):
                sim_score = scores[neighbor_idx]
                if sim_score > 0:
                    records.append(
                        {
                            "source_id": id_index[global_idx],
                            "similar_id": id_index[int(neighbor_idx)],
                            "score": round(float(sim_score), 4),
                            "rank": rank + 1,
                        }
                    )

        print(f"  Processed {end}/{n} titles...")

    # 7. Save similarity table
    sim_df = pd.DataFrame(records)
    out_dir = PRECOMPUTED_DIR / "similarity"
    out_dir.mkdir(parents=True, exist_ok=True)
    sim_df.to_parquet(out_dir / "tfidf_top_k.parquet", index=False)
    print(f"  Saved tfidf_top_k.parquet: {len(sim_df)} rows")

    # 8. Save vectorizer (for reuse on Page 4)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.pkl")
    print("  Saved tfidf_vectorizer.pkl")

    print("=== Done ===")


if __name__ == "__main__":
    main()
