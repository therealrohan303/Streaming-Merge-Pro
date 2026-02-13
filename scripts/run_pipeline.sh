#!/usr/bin/env bash
# Run the full data pipeline (01 then 02).
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Running data pipeline ==="
python3 "$SCRIPT_DIR/01_clean_raw_data.py"
python3 "$SCRIPT_DIR/02_merge_platforms.py"
echo "=== Pipeline complete ==="
