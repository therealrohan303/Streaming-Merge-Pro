"""
Script 13: Enrich top 10% of catalog with TMDB backdrop image URL.

Fetches for each title:
  - TMDB ID (via /find/{imdb_id})
  - Backdrop image URL (from /find results backdrop_path)

Caches per-title to data/cache/tmdb_game/{imdb_id}.json for resumability.
Rate limit: 40 requests per 10 seconds (same as script 08).

Output: data/precomputed/game_catalog.parquet
Columns: imdb_id, title, type, release_year, genres, imdb_score, imdb_votes,
         quality_score, poster_url, backdrop_url, tmdb_id,
         age_certification, runtime, seasons, production_countries
"""

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import CACHE_DIR, ENRICHED_DIR, PRECOMPUTED_DIR
from src.analysis.scoring import compute_quality_score

TOP_PCT = 0.20
TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_BACKDROP_BASE = "https://image.tmdb.org/t/p/w1280"
CACHE_SUBDIR = CACHE_DIR / "tmdb_game"
RATE_WINDOW = 10.0
RATE_LIMIT = 40


def load_api_key():
    load_dotenv()
    key = os.getenv("TMDB_API_KEY")
    if not key:
        sys.exit("ERROR: TMDB_API_KEY not found in .env")
    return key


def load_top_titles():
    df = pd.read_parquet(ENRICHED_DIR / "titles_enriched.parquet")
    df["quality_score"] = compute_quality_score(df)
    df = df.dropna(subset=["imdb_id"]).drop_duplicates(subset="imdb_id")
    n = int(len(df) * TOP_PCT)
    top = df.nlargest(n, "quality_score")
    print(f"  Top {TOP_PCT:.0%} by quality score: {len(top):,} titles (quality_score >= {top['quality_score'].min():.2f})")
    return top


def _rate_limit(request_times):
    """Block until we're under the rate limit."""
    now = time.time()
    request_times[:] = [t for t in request_times if now - t < RATE_WINDOW]
    if len(request_times) >= RATE_LIMIT:
        sleep_time = RATE_WINDOW - (now - request_times[0]) + 0.1
        if sleep_time > 0:
            time.sleep(sleep_time)
        now = time.time()
        request_times[:] = [t for t in request_times if now - t < RATE_WINDOW]


def fetch_game_data(imdb_id, api_key, request_times):
    """Fetch backdrop URL for one IMDb ID."""
    result = {
        "imdb_id": imdb_id,
        "tmdb_id": None,
        "backdrop_url": None,
    }

    # Find by IMDb ID → get tmdb_id and backdrop_path (single API call)
    _rate_limit(request_times)
    resp = requests.get(
        f"{TMDB_BASE}/find/{imdb_id}",
        params={"api_key": api_key, "external_source": "imdb_id"},
        timeout=15,
    )
    request_times.append(time.time())
    resp.raise_for_status()
    find = resp.json()

    backdrop_path = None
    if find.get("movie_results"):
        item = find["movie_results"][0]
        result["tmdb_id"] = item["id"]
        backdrop_path = item.get("backdrop_path")
    elif find.get("tv_results"):
        item = find["tv_results"][0]
        result["tmdb_id"] = item["id"]
        backdrop_path = item.get("backdrop_path")

    if backdrop_path:
        result["backdrop_url"] = TMDB_BACKDROP_BASE + backdrop_path

    return result


def main():
    CACHE_SUBDIR.mkdir(parents=True, exist_ok=True)
    PRECOMPUTED_DIR.mkdir(parents=True, exist_ok=True)

    api_key = load_api_key()
    top_df = load_top_titles()

    keep_cols = [
        "imdb_id", "title", "type", "release_year", "genres", "imdb_score",
        "imdb_votes", "quality_score", "poster_url", "age_certification",
        "runtime", "seasons", "production_countries",
    ]
    meta = top_df[[c for c in keep_cols if c in top_df.columns]].set_index("imdb_id")

    request_times = []
    rows = []
    cached = fetched = errors = 0

    for idx, imdb_id in enumerate(top_df["imdb_id"].tolist()):
        cache_path = CACHE_SUBDIR / f"{imdb_id}.json"

        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    data = json.load(f)
                rows.append(data)
                cached += 1
                continue
            except json.JSONDecodeError:
                pass  # re-fetch on corrupt cache

        try:
            data = fetch_game_data(imdb_id, api_key, request_times)
            with open(cache_path, "w") as f:
                json.dump(data, f)
            rows.append(data)
            fetched += 1
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                print(f"  Rate limited at {idx}, waiting 15s ...")
                time.sleep(15)
                try:
                    data = fetch_game_data(imdb_id, api_key, request_times)
                    with open(cache_path, "w") as f:
                        json.dump(data, f)
                    rows.append(data)
                    fetched += 1
                except Exception:
                    errors += 1
                    rows.append({"imdb_id": imdb_id, "tmdb_id": None, "youtube_key": None, "backdrop_url": None})
            else:
                errors += 1
                rows.append({"imdb_id": imdb_id, "tmdb_id": None, "youtube_key": None, "backdrop_url": None})
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  ERROR on {imdb_id}: {e}")
            rows.append({"imdb_id": imdb_id, "tmdb_id": None, "youtube_key": None, "backdrop_url": None})

        if (idx + 1) % 100 == 0:
            print(f"  Progress: {idx+1}/{len(top_df)} ({cached} cached, {fetched} fetched, {errors} errors)")

    print(f"\n  Total: {len(rows):,} ({cached} cached, {fetched} fetched, {errors} errors)")

    game_df = (
        pd.DataFrame(rows)
        .set_index("imdb_id")
        .join(meta, how="left")
        .reset_index()
        .drop_duplicates(subset="imdb_id")
    )

    # Keep only titles with at least one visual asset
    has_visual = game_df["backdrop_url"].notna() | game_df["poster_url"].notna()
    game_df = game_df[has_visual].reset_index(drop=True)

    out = PRECOMPUTED_DIR / "game_catalog.parquet"
    game_df.to_parquet(out, index=False)
    print(f"  Saved {len(game_df):,} titles → {out}")
    print(f"  With backdrop image  : {game_df['backdrop_url'].notna().sum():,}")
    print(f"  With poster          : {game_df['poster_url'].notna().sum():,}")
    print("\nDone!")


if __name__ == "__main__":
    main()
