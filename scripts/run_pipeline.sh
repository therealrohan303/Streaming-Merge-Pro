#!/usr/bin/env bash
# Run the full data pipeline (01 -> 12).
# Scripts 01-04: Base data processing
# Scripts 05-09: Enrichment pipeline
# Scripts 10-12: ML training + precomputation
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Phase 1: Base data processing ==="
python3 "$SCRIPT_DIR/01_clean_raw_data.py"
python3 "$SCRIPT_DIR/02_merge_platforms.py"
python3 "$SCRIPT_DIR/03_compute_similarity.py"
python3 "$SCRIPT_DIR/04_compute_umap.py"

echo "=== Phase 2: Enrichment pipeline ==="
python3 "$SCRIPT_DIR/05_enrich_imdb.py"
python3 "$SCRIPT_DIR/06_enrich_wikidata.py"
python3 "$SCRIPT_DIR/07_enrich_movielens.py"
python3 "$SCRIPT_DIR/08_enrich_tmdb.py"
python3 "$SCRIPT_DIR/09_build_enriched_titles.py"

echo "=== Phase 3: ML training + precomputation ==="
python3 "$SCRIPT_DIR/10_train_predictor.py"
python3 "$SCRIPT_DIR/11_precompute_strategic.py"
python3 "$SCRIPT_DIR/12_precompute_network.py"

echo "=== Pipeline complete ==="
