"""
04_compute_umap.py
Compute UMAP 2D coordinates from TF-IDF description + genre features.

Outputs:
  - data/precomputed/dimensionality_reduction/umap_coords.parquet
    Columns: id, umap_x, umap_y
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    MODELS_DIR,
    PRECOMPUTED_DIR,
    PROCESSED_DIR,
    UMAP_MIN_DIST,
    UMAP_METRIC,
    UMAP_N_COMPONENTS,
    UMAP_N_NEIGHBORS,
    UMAP_RANDOM_STATE,
)


def _build_genre_matrix(genres_list: list) -> np.ndarray:
    """Build a binary genre matrix (n_titles x n_genres)."""
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
    return matrix


def main():
    print("=== 04_compute_umap ===")

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

    # 2. Load fitted TF-IDF vectorizer and transform descriptions
    print("  Loading TF-IDF vectorizer and transforming descriptions...")
    vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")
    descriptions = deduped["description"].fillna("")
    tfidf_matrix = vectorizer.transform(descriptions)
    print(f"  TF-IDF matrix: {tfidf_matrix.shape}")

    # 3. Build genre matrix
    print("  Building genre matrix...")
    genre_matrix = _build_genre_matrix(deduped["genres"].tolist())
    print(f"  Genre matrix: {genre_matrix.shape}")

    # 4. Combine features: TF-IDF (scaled) + genre vectors (weighted)
    # Weight genres higher since they're strong platform-identity signals
    genre_weight = 2.0
    combined = hstack([tfidf_matrix, csr_matrix(genre_matrix * genre_weight)])
    print(f"  Combined feature matrix: {combined.shape}")

    # 5. Run UMAP
    print("  Running UMAP (this may take a minute)...")
    import umap

    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=UMAP_N_COMPONENTS,
        metric=UMAP_METRIC,
        random_state=UMAP_RANDOM_STATE,
        verbose=True,
    )
    coords = reducer.fit_transform(combined)
    print(f"  UMAP output: {coords.shape}")

    # 6. Save coordinates
    umap_df = pd.DataFrame(
        {
            "id": deduped["id"].values,
            "umap_x": coords[:, 0].astype(np.float32),
            "umap_y": coords[:, 1].astype(np.float32),
        }
    )

    out_dir = PRECOMPUTED_DIR / "dimensionality_reduction"
    out_dir.mkdir(parents=True, exist_ok=True)
    umap_df.to_parquet(out_dir / "umap_coords.parquet", index=False)
    print(f"  Saved umap_coords.parquet: {len(umap_df)} rows")

    # 7. Save UMAP model for potential reuse
    joblib.dump(reducer, MODELS_DIR / "umap_reducer.pkl")
    print("  Saved umap_reducer.pkl")

    print("=== Done ===")


if __name__ == "__main__":
    main()
