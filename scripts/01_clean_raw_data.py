"""
01_clean_raw_data.py
Read 12 raw CSVs (6 platforms × titles/credits), standardize schema,
and write 12 parquet files to data/interim/.
"""

import ast
import sys
from pathlib import Path

import pandas as pd

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    ALL_PLATFORMS,
    DECADE_BINS,
    DECADE_LABELS,
    INTERIM_DIR,
    QUALITY_TIERS,
    RAW_DIR,
)

# ── helpers ──────────────────────────────────────────────────────────────────


def _parse_list_col(val):
    """Convert string repr like \"['drama', 'action']\" → Python list."""
    if pd.isna(val):
        return []
    if isinstance(val, list):
        return val
    try:
        parsed = ast.literal_eval(str(val))
        if isinstance(parsed, list):
            return parsed
        return [str(parsed)]
    except (ValueError, SyntaxError):
        return [s.strip() for s in str(val).split(",") if s.strip()]


def _standardize_type(val):
    """MOVIE → Movie, SHOW → Show."""
    if pd.isna(val):
        return val
    mapping = {"MOVIE": "Movie", "SHOW": "Show"}
    return mapping.get(str(val).upper(), str(val).title())


def _assign_quality_tier(score):
    """Map IMDb score to quality tier using config thresholds."""
    if pd.isna(score):
        return "Unknown"
    for tier, (lo, hi) in QUALITY_TIERS.items():
        if lo <= score < hi:
            return tier
    # Edge case: perfect 10.0 falls into Excellent
    if score == 10.0:
        return "Excellent"
    return "Unknown"


# ── core cleaning ────────────────────────────────────────────────────────────


def clean_titles(platform: str) -> pd.DataFrame:
    """Clean a single platform's titles CSV."""
    path = RAW_DIR / f"{platform}_titles.csv"
    df = pd.read_csv(path)

    # Add platform column
    df["platform"] = platform

    # Parse list columns
    df["genres"] = df["genres"].apply(_parse_list_col)
    if "production_countries" in df.columns:
        df["production_countries"] = df["production_countries"].apply(_parse_list_col)

    # Standardize type
    df["type"] = df["type"].apply(_standardize_type)

    # Coerce numeric columns
    numeric_cols = ["imdb_score", "imdb_votes", "tmdb_popularity", "tmdb_score",
                    "runtime", "release_year"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Compute decade
    df["decade"] = pd.cut(
        df["release_year"],
        bins=DECADE_BINS,
        labels=DECADE_LABELS,
        right=False,
    )

    # Compute quality tier
    df["quality_tier"] = df["imdb_score"].apply(_assign_quality_tier)

    # Fill missing descriptions
    df["description"] = df["description"].fillna("")

    return df


def clean_credits(platform: str) -> pd.DataFrame:
    """Clean a single platform's credits CSV."""
    path = RAW_DIR / f"{platform}_credits.csv"
    df = pd.read_csv(path)

    # Add platform column
    df["platform"] = platform

    # Rename id → title_id (per PROJECT_SPEC §2.4)
    if "id" in df.columns:
        df = df.rename(columns={"id": "title_id"})

    # Fill missing character
    df["character"] = df["character"].fillna("")

    # Coerce person_id to numeric
    if "person_id" in df.columns:
        df["person_id"] = pd.to_numeric(df["person_id"], errors="coerce")

    return df


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    for platform in ALL_PLATFORMS:
        # Titles
        titles_raw = RAW_DIR / f"{platform}_titles.csv"
        if titles_raw.exists():
            df_titles = clean_titles(platform)
            out = INTERIM_DIR / f"{platform}_titles.parquet"
            df_titles.to_parquet(out, index=False)
            print(f"  ✓ {out.name}: {len(df_titles):,} rows, "
                  f"cols={list(df_titles.columns)}")
        else:
            print(f"  ⚠ {titles_raw} not found, skipping")

        # Credits
        credits_raw = RAW_DIR / f"{platform}_credits.csv"
        if credits_raw.exists():
            df_credits = clean_credits(platform)
            out = INTERIM_DIR / f"{platform}_credits.parquet"
            df_credits.to_parquet(out, index=False)
            print(f"  ✓ {out.name}: {len(df_credits):,} rows, "
                  f"cols={list(df_credits.columns)}")
        else:
            print(f"  ⚠ {credits_raw} not found, skipping")


if __name__ == "__main__":
    print("=" * 60)
    print("Step 01: Cleaning raw data → interim parquets")
    print("=" * 60)
    main()
    print("\nDone.")
