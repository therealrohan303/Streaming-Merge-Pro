"""Cached data loading functions for processed parquets."""

import pandas as pd
import streamlit as st

from src.config import PROCESSED_DIR


def _fix_list_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numpy arrays back to Python lists for list-type columns."""
    for col in ("genres", "production_countries"):
        if col in df.columns:
            df[col] = df[col].apply(lambda x: list(x) if hasattr(x, "tolist") else x)
    return df


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
