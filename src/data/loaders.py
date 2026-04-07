"""Cached data loading functions for processed parquets."""

import numpy as np
import pandas as pd
import streamlit as st

from src.config import (
    ALL_PLATFORMS,
    ENRICHED_DIR,
    GREENLIGHT_MOVIE_MODEL,
    GREENLIGHT_SHOW_MODEL,
    MODELS_DIR,
    PRECOMPUTED_DIR,
    PROCESSED_DIR,
)


def _fix_list_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numpy arrays back to Python lists for list-type columns."""
    for col in ("genres", "production_countries"):
        if col in df.columns:
            df[col] = df[col].apply(lambda x: list(x) if hasattr(x, "tolist") else x)
    return df


_PLATFORM_ORDER = {p: i for i, p in enumerate(ALL_PLATFORMS)}


def deduplicate_titles(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate titles by id, aggregating platforms into a sorted list.

    Replaces the ``platform`` column with a ``platforms`` column (list of keys).
    Platform order follows ALL_PLATFORMS (netflix, max, disney, prime, ...).
    """
    if "platform" not in df.columns or df.empty:
        return df

    platforms_agg = (
        df.groupby("id")["platform"]
        .apply(lambda x: sorted(set(x), key=lambda p: _PLATFORM_ORDER.get(p, 99)))
        .reset_index()
        .rename(columns={"platform": "platforms"})
    )

    deduped = df.drop_duplicates(subset="id", keep="first").copy()
    deduped = deduped.drop(columns=["platform"])
    deduped = deduped.merge(platforms_agg, on="id", how="left")

    return deduped


@st.cache_resource
def load_merged_titles() -> pd.DataFrame:
    return _fix_list_cols(pd.read_parquet(PROCESSED_DIR / "merged_titles.parquet"))


@st.cache_resource
def load_merged_credits() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "merged_credits.parquet")


@st.cache_resource
def load_all_platforms_titles() -> pd.DataFrame:
    return _fix_list_cols(pd.read_parquet(PROCESSED_DIR / "all_platforms_titles.parquet"))


@st.cache_resource
def load_all_platforms_credits() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "all_platforms_credits.parquet")


@st.cache_resource
def load_similarity_data() -> pd.DataFrame:
    """Load precomputed TF-IDF similarity top-K table."""
    return pd.read_parquet(PRECOMPUTED_DIR / "similarity" / "tfidf_top_k.parquet")


@st.cache_resource
def load_umap_coords() -> pd.DataFrame:
    """Load precomputed UMAP 2D coordinates."""
    return pd.read_parquet(PRECOMPUTED_DIR / "dimensionality_reduction" / "umap_coords.parquet")


@st.cache_resource
def load_tfidf_vectorizer():
    """Load fitted TF-IDF vectorizer (for future use in Discovery Engine)."""
    import joblib

    return joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")


@st.cache_data
def load_platform_profiles() -> dict:
    """Precompute platform profile vectors for the matcher (cached)."""
    from src.analysis.platform_dna import compute_platform_profile_vector

    df = load_all_platforms_titles()
    profiles = {}
    for key in ALL_PLATFORMS:
        profiles[key] = compute_platform_profile_vector(df, key)
    return profiles


@st.cache_resource
def load_enriched_titles() -> pd.DataFrame:
    """Load enriched titles. Falls back to base all_platforms_titles if enriched file absent."""
    path = ENRICHED_DIR / "titles_enriched.parquet"
    if path.exists():
        return _fix_list_cols(pd.read_parquet(path))
    return load_all_platforms_titles()


@st.cache_data
def load_imdb_principals() -> pd.DataFrame:
    """Load IMDb principals (directors, writers, producers, composers, cinematographers)."""
    path = ENRICHED_DIR / "imdb_principals.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=["imdb_id", "person_id", "name", "category", "job"])


@st.cache_data
def load_genome_vectors():
    """Load MovieLens genome vectors and ID map.

    Returns (vectors_np_array, id_map_df) tuple, or (None, None) if absent.
    """
    vectors_path = PRECOMPUTED_DIR / "embeddings" / "genome_vectors.npy"
    id_map_path = PRECOMPUTED_DIR / "embeddings" / "genome_id_map.parquet"
    if vectors_path.exists() and id_map_path.exists():
        vectors = np.load(vectors_path)
        id_map = pd.read_parquet(id_map_path)
        return vectors, id_map
    return None, None


@st.cache_resource
def load_network_edges() -> pd.DataFrame:
    """Load precomputed collaboration network edges."""
    path = PRECOMPUTED_DIR / "network" / "edges.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=["person_a", "person_b", "weight"])


@st.cache_resource
def load_person_stats() -> pd.DataFrame:
    """Load precomputed per-person statistics."""
    path = PRECOMPUTED_DIR / "network" / "person_stats.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=[
        "person_id", "name", "primary_role", "title_count", "avg_imdb",
        "pagerank", "community_id", "influence_score", "top_genre",
        "award_title_count", "total_award_wins", "platform_list",
        "career_start", "career_end",
    ])


@st.cache_resource
def load_greenlight_model(model_type: str):
    """Load trained greenlight predictor model.

    Args:
        model_type: 'movie' or 'show'
    """
    import joblib

    path = GREENLIGHT_MOVIE_MODEL if model_type == "movie" else GREENLIGHT_SHOW_MODEL
    if path.exists():
        return joblib.load(path)
    return None


@st.cache_data
def get_titles_for_view(platform_view: str) -> pd.DataFrame:
    """Route to the correct loader based on platform view.

    platform_view: "merged", "netflix", "max", or "all_platforms"
    Returns a copy so callers can safely mutate it.
    """
    if platform_view == "all_platforms":
        return load_all_platforms_titles().copy()
    df = load_merged_titles()
    if platform_view in ("netflix", "max"):
        return df[df["platform"] == platform_view].reset_index(drop=True)
    return df.copy()


@st.cache_data
def get_credits_for_view(platform_view: str) -> pd.DataFrame:
    """Route to the correct credits loader based on platform view."""
    if platform_view == "all_platforms":
        return load_all_platforms_credits().copy()
    df = load_merged_credits()
    if platform_view in ("netflix", "max"):
        return df[df["platform"] == platform_view].reset_index(drop=True)
    return df.copy()
