"""
02_merge_platforms.py
Merge interim parquets into 4 processed datasets:
  - merged_titles.parquet (netflix + max)
  - merged_credits.parquet (netflix + max)
  - all_platforms_titles.parquet (all 6)
  - all_platforms_credits.parquet (all 6)
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import ALL_PLATFORMS, INTERIM_DIR, MERGED_PLATFORMS, PROCESSED_DIR


def _concat_parquets(platforms: list[str], suffix: str) -> pd.DataFrame:
    """Read and concatenate parquets for the given platforms and file suffix."""
    frames = []
    for platform in platforms:
        path = INTERIM_DIR / f"{platform}_{suffix}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
        else:
            print(f"  ⚠ Missing {path.name}, skipping")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Merged (Netflix + Max)
    merged_titles = _concat_parquets(MERGED_PLATFORMS, "titles")
    merged_credits = _concat_parquets(MERGED_PLATFORMS, "credits")

    merged_titles.to_parquet(PROCESSED_DIR / "merged_titles.parquet", index=False)
    merged_credits.to_parquet(PROCESSED_DIR / "merged_credits.parquet", index=False)
    print(f"  ✓ merged_titles.parquet: {len(merged_titles):,} rows, "
          f"platforms={merged_titles['platform'].unique().tolist()}")
    print(f"  ✓ merged_credits.parquet: {len(merged_credits):,} rows")

    # All platforms
    all_titles = _concat_parquets(ALL_PLATFORMS, "titles")
    all_credits = _concat_parquets(ALL_PLATFORMS, "credits")

    all_titles.to_parquet(PROCESSED_DIR / "all_platforms_titles.parquet", index=False)
    all_credits.to_parquet(PROCESSED_DIR / "all_platforms_credits.parquet", index=False)
    print(f"  ✓ all_platforms_titles.parquet: {len(all_titles):,} rows, "
          f"platforms={all_titles['platform'].unique().tolist()}")
    print(f"  ✓ all_platforms_credits.parquet: {len(all_credits):,} rows")


if __name__ == "__main__":
    print("=" * 60)
    print("Step 02: Merging platforms → processed parquets")
    print("=" * 60)
    main()
    print("\nDone.")
