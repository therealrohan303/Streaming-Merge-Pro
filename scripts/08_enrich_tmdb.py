"""
Script 08: Enrich catalog with TMDB API data.

Uses the TMDB "Find by IMDb ID" endpoint to get TMDB IDs, then fetches
movie/TV details for keywords, collection, production companies, and poster.

Caches every API response to data/cache/tmdb/{imdb_id}.json for resumability.
Rate limit: 40 requests per 10 seconds.

Output: data/enriched/tmdb_enrichment.parquet
  Columns: imdb_id, tmdb_keywords (list), collection_name, production_companies (list), poster_url
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
from src.config import CACHE_DIR, ENRICHED_DIR, PROCESSED_DIR

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
CACHE_SUBDIR = CACHE_DIR / "tmdb"
RATE_WINDOW = 10.0  # seconds
RATE_LIMIT = 40  # requests per window


def load_api_key():
    load_dotenv()
    key = os.getenv("TMDB_API_KEY")
    if not key:
        print("ERROR: TMDB_API_KEY not found in .env file")
        sys.exit(1)
    return key


def load_our_imdb_ids():
    df = pd.read_parquet(PROCESSED_DIR / "all_platforms_titles.parquet",
                         columns=["imdb_id", "type"])
    df = df.dropna(subset=["imdb_id"])
    ids = df.drop_duplicates(subset="imdb_id")
    print(f"  {len(ids):,} unique imdb_ids to process")
    return ids


def get_cache_path(imdb_id):
    return CACHE_SUBDIR / f"{imdb_id}.json"


def fetch_tmdb_data(imdb_id, title_type, api_key):
    """Fetch TMDB data for a single IMDb ID.

    Returns dict with tmdb_keywords, collection_name, production_companies, poster_url.
    """
    result = {
        "imdb_id": imdb_id,
        "tmdb_keywords": None,
        "collection_name": None,
        "production_companies": None,
        "poster_url": None,
    }

    # Step 1: Find by IMDb ID
    find_url = f"{TMDB_BASE}/find/{imdb_id}"
    resp = requests.get(find_url, params={
        "api_key": api_key,
        "external_source": "imdb_id",
    }, timeout=15)
    resp.raise_for_status()
    find_data = resp.json()

    # Determine if movie or TV
    tmdb_id = None
    media_type = None
    if find_data.get("movie_results"):
        tmdb_id = find_data["movie_results"][0]["id"]
        media_type = "movie"
    elif find_data.get("tv_results"):
        tmdb_id = find_data["tv_results"][0]["id"]
        media_type = "tv"
    else:
        return result  # Not found on TMDB

    # Step 2: Get details
    detail_url = f"{TMDB_BASE}/{media_type}/{tmdb_id}"
    resp = requests.get(detail_url, params={
        "api_key": api_key,
        "append_to_response": "keywords",
    }, timeout=15)
    resp.raise_for_status()
    detail = resp.json()

    # Extract keywords
    kw_data = detail.get("keywords", {})
    if media_type == "movie":
        keywords = [k["name"] for k in kw_data.get("keywords", [])]
    else:
        keywords = [k["name"] for k in kw_data.get("results", [])]
    result["tmdb_keywords"] = keywords if keywords else None

    # Extract collection/franchise
    collection = detail.get("belongs_to_collection")
    if collection:
        result["collection_name"] = collection.get("name")

    # Extract production companies
    companies = detail.get("production_companies", [])
    if companies:
        result["production_companies"] = [c["name"] for c in companies]

    # Extract poster
    poster_path = detail.get("poster_path")
    if poster_path:
        result["poster_url"] = f"{TMDB_IMAGE_BASE}{poster_path}"

    return result


def main():
    CACHE_SUBDIR.mkdir(parents=True, exist_ok=True)
    ENRICHED_DIR.mkdir(parents=True, exist_ok=True)

    api_key = load_api_key()
    ids_df = load_our_imdb_ids()
    imdb_ids = ids_df["imdb_id"].tolist()
    type_map = dict(zip(ids_df["imdb_id"], ids_df["type"]))

    all_rows = []
    cached_count = 0
    fetched_count = 0
    error_count = 0
    request_times = []

    for idx, imdb_id in enumerate(imdb_ids):
        cache_path = get_cache_path(imdb_id)

        # Check cache
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    data = json.load(f)
                all_rows.append(data)
                cached_count += 1
                continue
            except json.JSONDecodeError:
                pass  # Re-fetch on corrupt cache

        # Rate limiting
        now = time.time()
        request_times = [t for t in request_times if now - t < RATE_WINDOW]
        if len(request_times) >= RATE_LIMIT:
            sleep_time = RATE_WINDOW - (now - request_times[0]) + 0.1
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Fetch
        try:
            title_type = type_map.get(imdb_id, "Movie")
            data = fetch_tmdb_data(imdb_id, title_type, api_key)
            with open(cache_path, "w") as f:
                json.dump(data, f)
            all_rows.append(data)
            fetched_count += 1
            request_times.append(time.time())
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                print(f"  Rate limited at {idx}, waiting 10s ...")
                time.sleep(10)
                try:
                    data = fetch_tmdb_data(imdb_id, type_map.get(imdb_id, "Movie"), api_key)
                    with open(cache_path, "w") as f:
                        json.dump(data, f)
                    all_rows.append(data)
                    fetched_count += 1
                    request_times.append(time.time())
                except Exception:
                    error_count += 1
            else:
                error_count += 1
        except Exception as e:
            error_count += 1
            if error_count <= 5:
                print(f"  ERROR on {imdb_id}: {e}")

        # Progress
        if (idx + 1) % 500 == 0:
            print(f"  Progress: {idx + 1}/{len(imdb_ids)} "
                  f"({cached_count} cached, {fetched_count} fetched, {error_count} errors)")

    print(f"\n  Total: {len(all_rows):,} results "
          f"({cached_count} cached, {fetched_count} fetched, {error_count} errors)")

    # Build DataFrame
    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset="imdb_id")

    output_path = ENRICHED_DIR / "tmdb_enrichment.parquet"
    df.to_parquet(output_path, index=False)
    print(f"  Saved {output_path} ({len(df):,} rows)")

    # Coverage stats
    for col in ["tmdb_keywords", "collection_name", "production_companies", "poster_url"]:
        non_null = df[col].notna().sum()
        print(f"  {col}: {non_null:,} non-null ({non_null/len(df)*100:.1f}%)")

    print("\nDone!")


if __name__ == "__main__":
    main()
