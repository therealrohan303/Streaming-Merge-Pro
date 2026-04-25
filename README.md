# Streaming Merge Pro

An 8-page interactive Streamlit app analyzing a hypothetical **Netflix + Warner Bros. Discovery (Max) merger** using catalog and credits data from 6 streaming platforms.

Built as a data science capstone, the app provides:
- Searchable cross-platform catalog with filtering and enriched metadata
- Competitive comparisons (genre gaps, audience overlap, prestige/awards)
- AI-powered title discovery (similarity search, mood board, vibe/semantic search)
- Strategic merger insights and acquisition target recommendations
- Interactive lab (Greenlight Predictor, Head-to-Head Arena, CinemaGuess game)
- Cast & crew collaboration network explorer

> **Data files are not included in this repo** due to size. Follow the [Data Setup](#data-setup) section exactly before running anything.

---

## Prerequisites

- Python 3.9+
- ~10 GB free disk space (raw data + artifacts)
- ~4 GB free RAM (MovieLens genome matrix)
- A free [TMDB API key](https://www.themoviedb.org/settings/api)

---

## Installation

```bash
git clone https://github.com/therealrohan303/Streaming-Merge-Pro.git
cd Streaming-Merge-Pro

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Create required data directories (not tracked in git)
mkdir -p data/raw/imdb data/raw/movielens data/interim data/processed \
         data/enriched data/precomputed data/cache models
```

---

## Data Setup

Complete all 4 steps before running the pipeline.

### 1. Platform Catalogs

Search Kaggle for the **JustWatch streaming catalogs** dataset (by dgoenrique) — it contains separate titles and credits CSVs for each platform following the exact schema below. Place the **12 CSV files** in `data/raw/`:

```
netflix_titles.csv   netflix_credits.csv
max_titles.csv       max_credits.csv
disney_titles.csv    disney_credits.csv
prime_titles.csv     prime_credits.csv
paramount_titles.csv paramount_credits.csv
appletv_titles.csv   appletv_credits.csv
```

Each `*_titles.csv` must have columns: `id, title, type, description, release_year, age_certification, runtime, genres, production_countries, seasons, imdb_id, imdb_score, imdb_votes, tmdb_popularity, tmdb_score`

Each `*_credits.csv` must have columns: `person_id, id, name, character, role`

### 2. IMDb Public Datasets

Download from **https://datasets.imdbws.com/** and place in `data/raw/imdb/`. Keep as `.tsv.gz` — do not extract.

| File | Purpose |
|------|---------|
| `title.basics.tsv.gz` | Title metadata |
| `title.principals.tsv.gz` | Cast & crew per title |
| `name.basics.tsv.gz` | Person name lookup |
| `title.ratings.tsv.gz` | Ratings and vote counts |

> IMDb data is for non-commercial use only.

### 3. MovieLens 20M

Download from **https://grouplens.org/datasets/movielens/20m/**, unzip `ml-20m.zip`, and place these 4 files in `data/raw/movielens/`:

`links.csv` · `genome-tags.csv` · `genome-scores.csv` · `ratings.csv`

> Covers movies only (~12% of catalog will have genome tag data). Requires ~4 GB RAM to process.

### 4. TMDB API Key

Copy `.env.example` to `.env` and fill in your key:

```bash
cp .env.example .env
# then edit .env and set TMDB_API_KEY=your_actual_key
```

Get a free key at https://www.themoviedb.org/settings/api. Script 08 makes ~2 API calls per title (~25K total) and runs 30–60 minutes. Responses are cached so it is safe to interrupt and resume.

---

## Running the Pipeline

Verify your `data/raw/` contains all files from the 3 sources above, then run:

```bash
bash scripts/run_pipeline.sh
```

This runs all 12 scripts in order (~2–3 hours total, mostly API calls in scripts 06–08). To run individually or resume from a specific step:

```bash
python scripts/01_clean_raw_data.py         # Parse + clean 12 raw CSVs → data/interim/
python scripts/02_merge_platforms.py         # Build merged + all-platform parquets → data/processed/
python scripts/03_compute_similarity.py      # TF-IDF similarity (top 50 per title) → data/precomputed/
python scripts/04_compute_umap.py            # 2D UMAP coordinates → data/precomputed/
python scripts/05_enrich_imdb.py             # IMDb principals + metadata → data/enriched/
python scripts/06_enrich_wikidata.py         # Awards, budget, box office → data/enriched/
python scripts/07_enrich_movielens.py        # Genre genome tag vectors → data/enriched/
python scripts/08_enrich_tmdb.py             # Posters, keywords, franchises → data/enriched/
python scripts/09_build_enriched_titles.py   # Join all enrichment → data/enriched/titles_enriched.parquet
python scripts/10_train_predictor.py             # Train IMDb score predictor → models/
python scripts/11_precompute_strategic.py    # Prestige index + acquisition targets → data/precomputed/
python scripts/12_precompute_network.py      # Collaboration graph + PageRank → data/precomputed/
```

Scripts 06 and 08 cache their API responses — re-running them skips already-fetched data.

**Optional — CinemaGuess game catalog** (used by the Interactive Lab page):
```bash
python scripts/13_enrich_tmdb_game.py
```
If skipped, the CinemaGuess game shows a "catalog not found" message but the rest of the app works fine.

---

## Running the App

```bash
streamlit run Home.py
```

Opens at `http://localhost:8501`. All 8 pages are available in the sidebar.

---

## Project Structure

```
├── Home.py                        # App entry point
├── pages/                         # 7 content pages (thin UI layer)
├── src/
│   ├── config.py                  # All constants, paths, platform keys
│   ├── data/loaders.py            # All cached loaders: @st.cache_data + @st.cache_resource
│   ├── analysis/                  # ML, scoring, similarity, strategic analysis
│   └── ui/                        # Reusable UI components (badges, cards, filters)
├── scripts/                       # Pipeline scripts 01–12 (+ optional 13)
├── models/                        # Serialized ML models (generated by pipeline)
├── data/                          # All data (not in repo — see Data Setup)
│   ├── raw/                       # Immutable source files — never modified by pipeline
│   ├── interim/                   # Script 01 output
│   ├── processed/                 # Script 02 output
│   ├── enriched/                  # Scripts 05–09 output
│   ├── precomputed/               # ML artifacts and precomputed tables
│   └── cache/                     # API response cache (auto-created)
└── requirements.txt
```

Platform keys used throughout the codebase: `netflix` `max` `disney` `prime` `paramount` `appletv`

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| App page fails to load or throws `KeyError` | A pipeline artifact is missing — run `bash scripts/run_pipeline.sh` to completion |
| Script 06 (Wikidata) stalls or returns 429 | Kill and re-run — cached batches are skipped automatically |
| Script 07 (MovieLens) runs out of memory | Free at least 4 GB RAM before running |
| Script 08 (TMDB) is slow | Expected — 30–60 min for ~25K titles. Cached, so safe to interrupt and resume |
| Script 10: "not enough samples" warning | Normal if enrichment coverage is low for a content type |
| Poster images missing | Script 08 or 09 incomplete — re-run both, then restart the app |

---

## License

Educational capstone project. Platform catalog data used under academic fair use. [IMDb Non-Commercial License](https://www.imdb.com/interfaces/). MovieLens data is CC0.
