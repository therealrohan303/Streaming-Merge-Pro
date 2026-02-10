# PROJECT_SPEC.md: Netflix + Max Merger Streamlit App

## 1) Goal
Build a 7-page Streamlit app that analyzes a hypothetical Netflix + Warner Bros (Max) merger.
Primary outputs: searchable catalog, recommendation tools, competitive comparisons, merger overlap + gap insights, and cast/crew network exploration.

## 2) Datasets and artifact contracts

### 2.1 Canonical processed datasets (must exist)
Located in `data/processed/`:
- `merged_titles.parquet`: Netflix + Max combined (PRIMARY for most pages)
- `merged_credits.parquet`: Netflix + Max credits
- `all_platforms_titles.parquet`: all 6 platforms (PRIMARY for comparisons vs competitors)
- `all_platforms_credits.parquet`: all 6 platforms credits

### 2.2 Precomputed artifacts (only if needed by pages)
Located in `data/precomputed/`:
- `embeddings/`: sentence-embedding vectors + id mapping
- `similarity/`: TF-IDF matrix and/or top-k similarity table
- `dimensionality_reduction/`: UMAP 2D coordinates
- `platform_stats/`: summary tables used by Home + Comparisons
- `strategic_analysis/`: overlap + gap analysis tables
- `network/`: edges + person-level aggregations

### 2.3 Minimum required columns (titles)
Every titles table used in the app must include at least:
- `id` (stable unique id)
- `title`
- `platform` (one of the platform keys)
- `type` (Movie/Show)
- `release_year`
- `genres` (list-like or delimiter string, but consistent)
- `imdb_score` (float, can be null)
- `tmdb_popularity` (float, can be null)
- `description` (string, can be empty but not null if possible)
- Optional but recommended: `runtime`, `age_certification`, `country` or `production_countries`

### 2.4 Minimum required columns (credits)
Every credits table used in the app must include at least:
- `title_id` (joins to titles `id`)
- `person_id`
- `name`
- `role` (actor/director/etc)
- Optional: `character`, `department`, `job`

If raw data is missing fields, standardize during `scripts/01_clean_raw_data.py`.

## 3) Global UX requirements (all pages)

### 3.1 Global filters (persist via session state)
Filters shown in the sidebar on every page:
- Platform view: merged, netflix, max, all_platforms
- Type: Movies, Shows (multi-select ok)
- Release year range
- Minimum IMDb
- Genre multi-select
- Reset button

### 3.2 Quick stats panel
On every page (top or sidebar), show:
- Count of titles matching filters
- Average IMDb for filtered set (ignore nulls)

### 3.3 Performance targets
- Initial page load target: under ~2 seconds after first cache warmup
- Any heavy computation must be precomputed offline and loaded fast in-app
- Use pagination for tables and long lists

### 3.4 Transparency footer
On every page include:
- “Hypothetical merger for academic analysis.”
- “Data is a snapshot (mid-2023).”
- Short “How it works” expanders for similarity, embeddings, predictor outputs.

## 4) Page specifications (acceptance criteria)

### Page 0: Home (Home.py)
Purpose: executive overview dashboard.
Must include:
- 4 hero metrics: combined catalog size, avg IMDb, unique people count, genre count
- Merger impact section:
  - Volume boost chart: Netflix vs Max vs merged
  - Quality shift: distribution comparison (Netflix vs merged)
  - Genre expansion: top genres comparison
- Top titles module:
  - Tabs: By Rating, By Popularity
  - Clickable title card opens a detail panel or links to Explore Catalog with that title selected
- Content timeline by decade
- “Jump to page” navigation cards

Definition of done:
- Works with global filters
- No expensive computation at runtime

### Page 1: Explore Catalog (pages/01_Explore_Catalog.py)
Purpose: searchable catalog + “similar titles.”
Must include:
- Search + filter controls: autocomplete title search, type, year range, min IMDb, genre, sort
- Two-panel layout:
  - Left: paginated results list (50 per page)
  - Right: detail view (metadata + description)
- Similar titles block:
  - Uses TF-IDF and/or embeddings similarity (precomputed preferred)
  - Scope toggle: merged only vs all platforms
  - Show top 10 with similarity score
  - Clicking a recommendation loads that title in the detail view

Definition of done:
- Similar titles returns results in under 200ms after cache warmup

### Page 2: Platform Comparisons (pages/02_Platform_Comparisons.py)
Purpose: compare merged entity vs competitors.
Must include:
- Competitor selector (up to 3) plus merged baseline always included
- Absolute vs normalized mode
- Volume chart + summary table
- Quality chart (box/violin ok) plus a threshold table
- Genre heatmap:
  - Top 15 genres by overall volume
  - Drill-down: for a clicked genre show counts, avg IMDb, and top titles per platform

Definition of done:
- All comparisons work off `all_platforms_titles.parquet`
- Heatmap drill-down is fast and correct

### Page 3: Platform DNA (pages/03_Platform_DNA.py)
Purpose: platform “identity” and content landscape.
Must include:
- Platform profile cards: genre mix, era focus, quality tier, 3 to 5 traits
- UMAP 2D plot:
  - Each point is a title, colored by platform
  - Selecting a region or clicking reveals sample titles and interpretation text
- “What platform are you?” mini matcher:
  - User preferences -> platform similarity scores + short explanation

Definition of done:
- UMAP uses precomputed coordinates only (no fitting in-app)

### Page 4: Discovery Engine (pages/04_Discovery_Engine.py)
Purpose: recommendation toolkit with 3 entry points.
Must include 3 tabs:
1) Similar to a title (fast, explainable)
2) Preference-based recommender (genres, min IMDb, type, year range, scope)
3) Vibe search (text prompt using embeddings over descriptions)

Also:
- Recommendation history (last 10 runs in session state)

Definition of done:
- Each recommender explains why items appear (briefly, via tooltip/expander)

### Page 5: Strategic Insights (pages/05_Strategic_Insights.py)
Purpose: merger overlap + gap analysis and executive narrative.
Must include:
- Value dashboard: overlap rate, gap coverage, quality lift, diversity proxy metric
- Overlap analysis:
  - Identify overlaps between Netflix and Max
  - Break down overlap by genre, type, decade, certification
  - Audit table with confidence score and evidence fields
- Gap analysis tool:
  - Perspective selector: merged vs competitor
  - Output prioritized gaps with severity labels
- Strategic recommendations block:
  - 5 to 10 bullet insights that are directly supported by metrics shown

Definition of done:
- Overlap and gaps come from precomputed tables in `data/precomputed/strategic_analysis/`

### Page 6: Interactive Lab (pages/06_Interactive_Lab.py)
Purpose: high-engagement interactive features.
Must include:
- “Build your streaming service” draft mini-game:
  - Budget, pick titles, live metrics dashboard
  - Compare vs merged baseline
- Hypothetical title predictor:
  - User inputs -> predicted IMDb range + uncertainty band
  - Suggest talent options based on genre track record
- Insight generator:
  - Generates 5 to 8 data-backed insights from a chosen scope

Definition of done:
- Predictor is loaded via `models/imdb_predictor.pkl` and is cached
- Generator outputs must cite the metric used (not external facts)

### Page 7: Cast & Crew Network (pages/07_Cast_Crew_Network.py)
Purpose: explore collaboration network and people profiles.
Must include:
- Person search (autocomplete) + filters (role, min title count, scope)
- Person profile:
  - Key stats: avg IMDb, popularity proxy, career span, top genres
  - Top collaborators (clickable)
  - Filmography table (sortable + paginated)
  - Trend line of IMDb over time (by release year)
- Rankings tabs:
  - Most titles, highest avg IMDb, most popular
  - Compare 2 to 3 people side by side

Definition of done:
- Network uses `data/precomputed/network/edges.parquet` and `person_stats.parquet`

## 5) Engineering requirements

### 5.1 App code rules
- Pages should be thin. Most logic lives in `src/`.
- All file paths and constants come from `src/config.py`.
- No repeated data loading logic in pages. Use `src/data/loaders.py`.

### 5.2 Caching rules
- `st.cache_data` for data tables and computed tables
- `st.cache_resource` for models/vectorizers/UMAP objects

### 5.3 Testing (minimal)
- `tests/test_data_pipeline.py`: validates processed datasets exist and have expected columns
- `tests/test_similarity.py`: sanity checks similarity outputs
- `tests/test_analysis.py`: overlap + gap logic sanity checks

## 6) Out of scope (explicit)
- Live web scraping or real-time market share data
- Full UI snapshot tests
- Perfect causal claims about merger outcomes

## 7) Success criteria
- All 7 pages functional with consistent global filters
- Recommendations and similarity feel sensible and are explainable
- App runs locally and is deployable on Streamlit Cloud