"""Cached data loading functions for processed parquets."""

import pandas as pd
import streamlit as st

from src.config import ALL_PLATFORMS, MODELS_DIR, PRECOMPUTED_DIR, PROCESSED_DIR


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


@st.cache_data
def load_merged_titles() -> pd.DataFrame:
    return _fix_list_cols(pd.read_parquet(PROCESSED_DIR / "merged_titles.parquet"))


@st.cache_data
def load_merged_credits() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "merged_credits.parquet")


@st.cache_data
def load_all_platforms_titles() -> pd.DataFrame:
    return _fix_list_cols(pd.read_parquet(PROCESSED_DIR / "all_platforms_titles.parquet"))


@st.cache_data
def load_all_platforms_credits() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "all_platforms_credits.parquet")


@st.cache_data
def load_similarity_data() -> pd.DataFrame:
    """Load precomputed TF-IDF similarity top-K table."""
    return pd.read_parquet(PRECOMPUTED_DIR / "similarity" / "tfidf_top_k.parquet")


@st.cache_data
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


def get_titles_for_view(platform_view: str) -> pd.DataFrame:
    """Route to the correct loader based on platform view.

    platform_view: "merged", "netflix", "max", or "all_platforms"
    """
    if platform_view == "all_platforms":
        return load_all_platforms_titles()
    df = load_merged_titles()
    if platform_view in ("netflix", "max"):
        return df[df["platform"] == platform_view].reset_index(drop=True)
    # "merged" → return full merged dataset
    return df


def get_credits_for_view(platform_view: str) -> pd.DataFrame:
    """Route to the correct credits loader based on platform view."""
    if platform_view == "all_platforms":
        return load_all_platforms_credits()
    df = load_merged_credits()
    if platform_view in ("netflix", "max"):
        return df[df["platform"] == platform_view].reset_index(drop=True)
    return df
