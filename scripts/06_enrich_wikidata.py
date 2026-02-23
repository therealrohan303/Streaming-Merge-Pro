"""
Script 06: Enrich catalog with Wikidata SPARQL data.

Queries Wikidata for budget, box office, award wins, and award nominations
using IMDb IDs. Caches each batch response to data/cache/wikidata/ as JSON
for resumability.

Output: data/enriched/wikidata_enrichment.parquet
  Columns: imdb_id, budget_usd, box_office_usd, award_wins, award_noms, data_confidence
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import CACHE_DIR, ENRICHED_DIR, PROCESSED_DIR

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
BATCH_SIZE = 50
RATE_LIMIT_SEC = 1.0
CACHE_SUBDIR = CACHE_DIR / "wikidata"

SPARQL_TEMPLATE = """
SELECT ?imdb_id
       (SAMPLE(?budget) AS ?budget_usd)
       (SAMPLE(?box_office) AS ?box_office_usd)
       (COUNT(DISTINCT ?award_won) AS ?award_wins)
       (COUNT(DISTINCT ?award_nom) AS ?award_noms)
WHERE {{
  VALUES ?imdb_id {{ {values} }}
  ?item wdt:P345 ?imdb_id .
  OPTIONAL {{ ?item wdt:P2130 ?budget . }}
  OPTIONAL {{ ?item wdt:P2142 ?box_office . }}
  OPTIONAL {{ ?item p:P166 ?award_stmt .
              ?award_stmt ps:P166 ?award_won . }}
  OPTIONAL {{ ?item p:P1411 ?nom_stmt .
              ?nom_stmt ps:P1411 ?award_nom . }}
}}
GROUP BY ?imdb_id
"""


def load_our_imdb_ids():
    """Load unique imdb_ids from our catalog."""
    df = pd.read_parquet(PROCESSED_DIR / "all_platforms_titles.parquet", columns=["imdb_id"])
    ids = sorted(df["imdb_id"].dropna().unique().tolist())
    print(f"  Our catalog has {len(ids):,} unique imdb_ids")
    return ids


def get_cache_path(batch_idx):
    return CACHE_SUBDIR / f"batch_{batch_idx:04d}.json"


def query_wikidata(imdb_ids):
    """Query Wikidata SPARQL for a batch of IMDb IDs."""
    values_str = " ".join(f'"{iid}"' for iid in imdb_ids)
    query = SPARQL_TEMPLATE.format(values=values_str)

    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "StreamingMergerCapstone/1.0 (academic project)",
    }

    resp = requests.get(
        WIKIDATA_ENDPOINT,
        params={"query": query},
        headers=headers,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def parse_wikidata_results(data):
    """Parse SPARQL JSON results into list of dicts."""
    rows = []
    for binding in data.get("results", {}).get("bindings", []):
        row = {"imdb_id": binding["imdb_id"]["value"]}
        for field in ["budget_usd", "box_office_usd"]:
            if field in binding:
                try:
                    row[field] = float(binding[field]["value"])
                except (ValueError, TypeError):
                    row[field] = None
            else:
                row[field] = None
        for field in ["award_wins", "award_noms"]:
            if field in binding:
                try:
                    row[field] = int(binding[field]["value"])
                except (ValueError, TypeError):
                    row[field] = 0
            else:
                row[field] = 0
        rows.append(row)
    return rows


def main():
    CACHE_SUBDIR.mkdir(parents=True, exist_ok=True)
    ENRICHED_DIR.mkdir(parents=True, exist_ok=True)

    imdb_ids = load_our_imdb_ids()
    batches = [imdb_ids[i:i + BATCH_SIZE] for i in range(0, len(imdb_ids), BATCH_SIZE)]
    print(f"  {len(batches)} batches of {BATCH_SIZE}")

    all_rows = []
    cached_count = 0
    queried_count = 0
    error_count = 0

    for batch_idx, batch in enumerate(batches):
        cache_path = get_cache_path(batch_idx)

        # Check cache
        if cache_path.exists():
            with open(cache_path) as f:
                cached_data = json.load(f)
            rows = parse_wikidata_results(cached_data)
            all_rows.extend(rows)
            cached_count += 1
            continue

        # Query Wikidata
        try:
            data = query_wikidata(batch)
            # Cache the response
            with open(cache_path, "w") as f:
                json.dump(data, f)
            rows = parse_wikidata_results(data)
            all_rows.extend(rows)
            queried_count += 1
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                print(f"  Rate limited at batch {batch_idx}, waiting 30s ...")
                time.sleep(30)
                try:
                    data = query_wikidata(batch)
                    with open(cache_path, "w") as f:
                        json.dump(data, f)
                    rows = parse_wikidata_results(data)
                    all_rows.extend(rows)
                    queried_count += 1
                except Exception:
                    error_count += 1
                    print(f"  ERROR on batch {batch_idx} (retry failed)")
            else:
                error_count += 1
                print(f"  ERROR on batch {batch_idx}: {e}")
        except Exception as e:
            error_count += 1
            print(f"  ERROR on batch {batch_idx}: {e}")

        # Progress
        if (batch_idx + 1) % 20 == 0:
            print(f"  Progress: {batch_idx + 1}/{len(batches)} batches "
                  f"({cached_count} cached, {queried_count} queried, {error_count} errors, "
                  f"{len(all_rows):,} results)")

        # Rate limit
        time.sleep(RATE_LIMIT_SEC)

    print(f"\n  Total: {len(all_rows):,} results from {len(batches)} batches "
          f"({cached_count} cached, {queried_count} queried, {error_count} errors)")

    if not all_rows:
        print("  WARNING: No results from Wikidata")
        df = pd.DataFrame(columns=["imdb_id", "budget_usd", "box_office_usd",
                                    "award_wins", "award_noms", "data_confidence"])
    else:
        df = pd.DataFrame(all_rows)
        df = df.drop_duplicates(subset="imdb_id")

        # Compute data_confidence: fraction of 4 enrichment fields that are non-null
        enrichment_cols = ["budget_usd", "box_office_usd", "award_wins", "award_noms"]
        df["data_confidence"] = df[enrichment_cols].apply(
            lambda row: sum(1 for v in row if pd.notna(v) and v != 0) / len(enrichment_cols),
            axis=1,
        )

    output_path = ENRICHED_DIR / "wikidata_enrichment.parquet"
    df.to_parquet(output_path, index=False)
    print(f"\n  Saved {output_path} ({len(df):,} rows)")

    # Coverage stats
    if len(df) > 0:
        for col in ["budget_usd", "box_office_usd", "award_wins", "award_noms"]:
            if col in ["award_wins", "award_noms"]:
                non_zero = (df[col] > 0).sum()
                print(f"  {col}: {non_zero:,} non-zero ({non_zero/len(df)*100:.1f}%)")
            else:
                non_null = df[col].notna().sum()
                print(f"  {col}: {non_null:,} non-null ({non_null/len(df)*100:.1f}%)")
        print(f"  avg data_confidence: {df['data_confidence'].mean():.3f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
