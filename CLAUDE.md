# CLAUDE.md: Streaming Merger Capstone (Netflix + Max)

## What this repo is
A 7-page Streamlit app that analyzes a hypothetical Netflix + Warner Bros (Max) merger using catalogs + credits from 6 platforms.

## Source of truth docs (read as needed)
- PROJECT_SPEC.md: page-by-page requirements and acceptance criteria
- PROGRESS_LOG.md: current week, what is done, what is next
- src/config.py: paths, platform keys, constants, model params

Use progressive disclosure: only open the files you need for the task at hand, do not ingest everything by default. Keep context lean. 

## Non-negotiables
- Keep `data/raw/` immutable. Do not edit raw CSVs.
- Prefer reproducing artifacts via scripts in `scripts/` over manual notebook steps.
- All platform keys must be exactly: `netflix`, `max`, `disney`, `prime`, `paramount`, `appletv`.
- Any expensive computation should be precomputed offline into `data/precomputed/` and loaded in the app.

## Repo structure (mental model)
- `scripts/`: builds data artifacts (raw -> interim -> processed -> precomputed)
- `src/`: reusable code (loading, filtering, ML, analysis, viz, UI)
- `pages/`: Streamlit pages (thin pages that call into `src/`)
- `models/`: serialized model objects only (vectorizer, umap, predictor)

## How to work in this repo (workflow)
When implementing a task:
1) State what files you will touch.
2) Make the smallest correct change.
3) Run the quickest relevant check (script or test).
4) Update PROGRESS_LOG.md with what changed and what is next.

If requirements are unclear, consult PROJECT_SPEC.md before inventing new behavior.

## Local commands (common)
- Run app: `streamlit run Home.py`
- Run tests: `pytest -q`
- Run pipeline: `bash scripts/run_pipeline.sh` (or run scripts 01 -> 09 in order)

## Streamlit performance rules
- DataFrames/arrays: cache with `st.cache_data` (in `src/data/loaders.py`)
- Models/resources: cache with `st.cache_resource` (in `src/models/loaders.py`)
- Avoid re-reading parquet/npz in page code. Page code should call loader functions.

## What not to do
- Do not add lots of style/lint rules here. Keep this file short and stable.
- Do not duplicate specs here. Point to PROJECT_SPEC.md instead.
- Do not generate new “one-off” folders unless there is a clear need.