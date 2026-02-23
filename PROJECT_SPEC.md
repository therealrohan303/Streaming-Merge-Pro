# PROJECT_SPEC.md: Netflix + Max Merger Streamlit App

## 1) Goal
Build an 8-page Streamlit app (Home + 7 content pages) that analyzes a hypothetical Netflix + Warner Bros (Max) merger.
Primary outputs: searchable catalog, hybrid recommendation tools, competitive comparisons, merger overlap + gap insights, prestige/awards analysis, and cast/crew network exploration.

---

## 2) Datasets and artifact contracts

### 2.1 Canonical processed datasets (must exist)
Located in `data/processed/`:
- `merged_titles.parquet`: Netflix + Max combined (PRIMARY for most pages)
- `merged_credits.parquet`: Netflix + Max credits
- `all_platforms_titles.parquet`: all 6 platforms (PRIMARY for comparisons vs competitors)
- `all_platforms_credits.parquet`: all 6 platforms credits

### 2.2 Enrichment layer datasets (built by scripts 05–09)
Located in `data/enriched/`:
- `imdb_enrichment.parquet`: from IMDb public TSVs — original titles, isAdult flag
- `imdb_principals.parquet`: writers, producers, composers, cinematographers (not just actors/directors)
  - Columns: imdb_id, person_id, name, category, job
- `wikidata_enrichment.parquet`: awards, box office, budget from Wikidata SPARQL
  - Columns: imdb_id, budget_usd, box_office_usd, award_wins, award_noms, data_confidence
- `movielens_genome.parquet`: MovieLens 20M tag genome vibe vectors (movies only)
  - Columns: imdb_id, top_tags (list of top 20 tags by score), ml_avg_rating, ml_rating_count
- `tmdb_enrichment.parquet`: TMDB keywords, franchise/collection, production companies, poster URLs
  - Columns: imdb_id, tmdb_keywords (list), collection_name, production_companies (list), poster_url
- `titles_enriched.parquet`: master join of all_platforms_titles + all enrichment parquets on imdb_id
  - This is the PRIMARY table for pages that need enriched data
  - All enrichment columns present; missing values expected and handled gracefully
  - Includes `data_confidence` float per enrichment source

### 2.3 Precomputed artifacts (only if needed by pages)
Located in `data/precomputed/`:
- `embeddings/`: sentence-embedding vectors (all-MiniLM-L6-v2) + id mapping
- `embeddings/genome_vectors.npy`: MovieLens genome vectors for movies with coverage
- `similarity/`: TF-IDF matrix and/or top-k similarity table (top 50 per title)
- `dimensionality_reduction/umap_coords.parquet`: UMAP 2D coordinates (precomputed offline)
- `platform_stats/`: summary tables used by Home + Comparisons
- `strategic_analysis/`: overlap + gap analysis tables
- `network/edges.parquet`: collaboration edges including writers, producers, all roles
- `network/person_stats.parquet`: per-person aggregated stats
- `strategic_analysis/prestige_index.parquet`: award wins per platform per genre, normalized per 1,000 titles
- `strategic_analysis/acquisition_targets.parquet`: precomputed gap + acquisition recommendations with decision trace fields

### 2.4 Minimum required columns (titles — base)
- `id` (stable unique id)
- `title`
- `platform` (one of the platform keys)
- `type` (Movie/Show)
- `release_year`
- `genres` (list-like, consistent)
- `imdb_score` (float, nullable)
- `imdb_votes` (float, nullable)
- `tmdb_popularity` (float, nullable)
- `description` (string)
- `runtime`, `age_certification`, `production_countries` (recommended)

### 2.5 Minimum required columns (titles — enriched)
All base columns plus (nullable, confidence-flagged):
- `award_wins`, `award_noms` (int, from Wikidata)
- `budget_usd`, `box_office_usd` (float, from Wikidata)
- `data_confidence` (float 0.0–1.0, fraction of enrichment fields present)
- `tmdb_keywords` (list)
- `collection_name` (string, nullable — franchise/collection from TMDB)
- `top_tags` (list — MovieLens genome top tags, movies only)
- `ml_avg_rating`, `ml_rating_count` (float, movies only)
- `poster_url` (string, nullable)
- `imdb_writers`, `imdb_producers` (list of names, from IMDb principals)

### 2.6 Minimum required columns (credits — enriched)
Base credits columns plus:
- `category` (actor/director/writer/producer/composer/cinematographer — from IMDb principals)
- `job` (string, nullable — specific job title)

### 2.7 Pipeline script order
```
scripts/01_clean_raw_data.py        → data/interim/
scripts/02_merge_platforms.py       → data/processed/
scripts/03_compute_similarity.py    → data/precomputed/similarity/ + models/tfidf_vectorizer.pkl
scripts/04_compute_umap.py          → data/precomputed/dimensionality_reduction/ + models/umap_reducer.pkl
scripts/05_enrich_imdb.py           → data/enriched/imdb_enrichment.parquet + imdb_principals.parquet
scripts/06_enrich_wikidata.py       → data/enriched/wikidata_enrichment.parquet
scripts/07_enrich_movielens.py      → data/enriched/movielens_genome.parquet + precomputed/embeddings/genome_vectors.npy
scripts/08_enrich_tmdb.py           → data/enriched/tmdb_enrichment.parquet
scripts/09_build_enriched_titles.py → data/enriched/titles_enriched.parquet
scripts/10_train_predictor.py       → models/greenlight_movie_predictor.pkl + models/greenlight_show_predictor.pkl
scripts/11_precompute_strategic.py  → data/precomputed/strategic_analysis/ (prestige_index, acquisition_targets)
scripts/12_precompute_network.py    → data/precomputed/network/ (edges with all roles, person_stats)
```

`scripts/run_pipeline.sh` runs all scripts in order 01 → 12.

---

## 3) Global UX requirements (all pages)

### 3.1 Global filters (persist via session state)
- Platform view: merged, netflix, max, all_platforms
- Type: Movies, Shows
- Release year range
- Minimum IMDb
- Genre multi-select
- Reset button

### 3.2 Quick stats panel
- Count of titles matching filters
- Average IMDb for filtered set

### 3.3 Performance targets
- Initial page load: under ~2 seconds after first cache warmup
- All heavy computation precomputed offline
- Pagination for tables and long lists

### 3.4 Transparency footer (every page)
- "Hypothetical merger for academic analysis."
- "Data is a snapshot (mid-2023)."
- "Enrichment data: IMDb datasets, Wikidata, MovieLens 20M, TMDB API."
- "Enrichment field coverage varies by title — see data_confidence."
- Short "How it works" expanders for similarity, embeddings, predictor outputs.
- Wherever ML or enrichment data appears: small info icon with "data source + coverage %" on hover.

---

## 4) Page specifications (acceptance criteria)

---

### Page 0: Home (Home.py)
**Purpose:** Executive overview dashboard of the merger.

Must include:
- 4 hero metrics: combined catalog size, avg IMDb (with delta vs Netflix-only), unique people count, genre count
- Optional 5th hero metric: "Award-winning titles in merged catalog" (from Wikidata, only if coverage >20%)
- Merger impact section:
  - Volume boost chart: Netflix vs Max vs merged
  - Quality shift: IMDb distribution comparison
  - Genre expansion: top genre share comparison
- Top titles module:
  - Tabs: By Rating (Bayesian quality score), By Popularity
  - Grid cards with title, year, type, IMDb score + votes, platform badge
  - Clickable → routes to Explore Catalog with that title preloaded
- Content timeline by decade (stacked area, Netflix vs Max, toggle Movies/Shows)
- Geographic footprint: choropleth + top 10 countries bar chart
- Quick navigation cards: Explore, Compare, Discovery Engine

**What is NOT here** (to avoid redundancy):
- No prestige/awards deep dives (Strategic Insights owns that)
- No genre analytics (Comparisons and DNA own that)
- No recommendations (Discovery Engine owns that)

**Definition of done:**
- Works with global filters
- No expensive computation at runtime
- Award metric only shown if enrichment coverage is sufficient (>20% of titles have award_wins)

---

### Page 1: Explore Catalog (pages/01_Explore_Catalog.py)
**Purpose:** Searchable database + quick "similar titles" teaser.

Must include:
- Search box with autocomplete (title + description + cast name search)
- Compact filters: type, year range, min IMDb (quick thresholds), genre multi-select, sort
- Active filter indicator (pill badges for non-default filters)
- Results count ("Showing X of Y titles")
- Two-panel layout:
  - Left: paginated results list (50 per page), compact cards with hover + selected state
  - Right: detail view with metadata grid, description, genre pills, quality score
- Cast and crew teaser (expander, collapsed by default): top actors + directors; link to Network page
- Similar titles ("More like this"):
  - Uses precomputed multi-signal similarity (TF-IDF desc + genre + type + IMDb proximity)
  - Scope toggle: merged only vs all platforms
  - Shows top 10 with similarity score and platform badge
  - This is the **lightweight version** — 10 quick matches, minimal controls
  - Full recommendation engine lives on Discovery Engine page
- If enrichment available: show award badge (🏆) on detail view for titles with award_wins > 0

**Definition of done:**
- Similar titles returns results under 200ms after cache warmup
- "More like this" is clearly labeled as distinct from the full Discovery Engine

---

### Page 2: Platform Comparisons (pages/02_Platform_Comparisons.py)
**Purpose:** Cross-platform benchmarking and competitive metrics.

Must include:
- Competitor selector (up to 3), merged baseline always locked
- Absolute vs normalized mode toggle
- Volume comparison: grouped bar (movies/shows split) + summary table
- Quality comparison: box/violin plot + threshold table (>7.0, >8.0)
- Genre heatmap (top 15 genres × platforms):
  - Drill-down on genre: count bar, avg IMDb bar, top 3 titles per platform (cards)
  - Leader badge per genre
- Content profile section (3 columns):
  - Age certification distribution (stacked bars)
  - International share (% non-US production)
  - Era focus (median release year + optional violin)
- Market positioning matrix: scatter (catalog size × avg IMDb, bubble = TMDB popularity), quadrant labels
- Strategic insights: per-competitor SWOT-lite prose (strengths/merged advantages/battlegrounds)
- If enrichment available: add "Prestige Score" column to summary table (award_wins per 1,000 titles, with confidence note)

**Definition of done:**
- All sections work off `all_platforms_titles.parquet` (base) or `titles_enriched.parquet` (with graceful fallback if enrichment missing)
- Prestige column only renders if wikidata coverage >15% of titles for that platform
- Normalized mode applies to all sections consistently

---

### Page 3: Platform DNA (pages/03_Platform_DNA.py)
**Purpose:** Platform identity, personality, and content landscape. Interpretive, not just metrics.

Must include:

**Section 1: Platform Identity Profile**
- Platform dropdown (7 options: merged + 6 platforms)
- Radar chart (6 axes, normalized 0–100): Freshness, Quality, Breadth, Global Reach, Genre Diversity, Series Focus
- Optional: compare 2 platforms (overlaid radar traces)
- 3–5 defining traits (narrative style, e.g., "The Content Giant: With 9,058 titles...")
- Compact metrics row: avg IMDb, rated titles %, total titles, premium %, median year
- "What do the radar dimensions mean?" expander with dimension table
- Normalized toggle matching Page 2 pattern

**Section 2: Content Landscape**
- Density contour plot (Histogram2dContour per platform) over UMAP coordinates
- Optional point overlay for selected platforms (sampled, max 500 per platform)
- Platform highlight multiselect
- Title search: highlights matching titles on map with gold markers; results shown below
- Neighborhood explorer (selectbox): genre pills, leader tags, auto-description, top 10 titles (quality-score ranked)
- Multi-insight callout: most separated platform pair, most overlapping pair (colored platform names)
- Cluster center annotations with thematic names (e.g., "Prestige Drama Hub", "Kids & Animation Hub")

**Section 3: What Platform Are You? Matcher**
- Genre multiselect (top 3 favorites)
- 6 preference sliders (2-column layout): classics↔new, hidden gems↔blockbusters, short↔long, family↔mature, single↔series, arthouse↔mainstream
- Output: best match platform + %, bar chart of all 6, per-platform explanation cards
- "Your Viewing DNA" personality card: maps slider values to narrative traits

**Definition of done:**
- UMAP uses precomputed coordinates only (no fitting in-app)
- Cluster names are thematic archetypes (not generic "Cluster 1")
- Defining traits use narrative language with concrete numbers

---

### Page 4: Discovery Engine (pages/04_Discovery_Engine.py)
**Purpose:** Full recommendation toolkit — the "power tool" version. Three distinct entry points.

Must include:

**Tab 1: Similar to a Title**
- Title search (autocomplete)
- Platform scope radio (merged only / all platforms / specific platform)
- Results count slider (5–20)
- Optional quality filter (min IMDb)
- Output: recommendation cards with similarity score, platform badge
- "Why similar?" expander per card:
  - Genre overlap breakdown
  - Key description terms matched
  - Matched vibe tags (from MovieLens top_tags, if available)
  - Same writer or director (from IMDb principals, if available)

**Tab 2: Preference-Based Recommendations**
- Genres (1–5 multi-select)
- Min IMDb slider
- Type toggle (Movies/Shows/Both)
- Runtime preference slider (movies only)
- Year range dual slider
- Popular vs hidden gems slider (weights TMDB popularity vs inverse popularity)
- Platform scope
- Output: ranked list top 20 with fit score and "why it matches" text per item

**Tab 3: Vibe Search (NLP-powered)**
- Text area input ("Describe what you're in the mood for...")
- Optional collapsed filters: min IMDb, year range, platform scope
- Semantic search via sentence-transformers (all-MiniLM-L6-v2), pre-computed description embeddings
- Keyword matching against `tmdb_keywords` and `top_tags` (MovieLens genome, where available)
- Display extracted signals to user: "Detected themes: psychological thriller, atmospheric, twist ending"
- Output: top 15 matches with relevance score and matched tags highlighted
- Hybrid scoring for movies with genome data: 0.35 description embedding + 0.25 genre + 0.15 genome vector + 0.15 Bayesian quality + 0.10 awards boost (if available)
- For titles without genome data: redistribute weights to description + genre + quality

**Session-Based Recommendation History**
- Last 10 recommendation sets in session state
- User marking: Interested / Watched / Not Interested
- Clear history button

**Definition of done:**
- Each tab clearly explains why items appear (tooltip or expander)
- Vibe search shows detected signals before results
- Hybrid scoring degrades gracefully when enrichment data is absent
- This page is the "full engine"; Explore Catalog is "quick teaser only"

---

### Page 5: Strategic Insights (pages/05_Strategic_Insights.py)
**Purpose:** Merger business intelligence. Decision-support outputs with decision traces, not just dashboards.

Must include:

**Section 1: Merger Value Dashboard**
- 4 headline metrics with decision-trace tooltips:
  - Genre gap coverage: "HBO/Max fills X% of Netflix's genre gaps"
  - Content overlap rate: "Y% duplicate content by genre+type matching"
  - Quality lift: "Avg IMDb +Z post-merger"
  - Catalog diversity: genre entropy score increase
- Optional metric (if wikidata coverage >20%): Prestige lift ("Award-winning titles +N% vs Netflix alone")

**Section 2: Prestige Index** *(new — requires Wikidata enrichment)*
- Award wins and nominations per 1,000 titles by platform and by genre
- Bar chart: platforms ranked by prestige index
- Heatmap: genre × platform prestige intensity
- Coverage note: "X% of titles have award data (Wikidata)"
- Key insight: "Post-merger prestige index vs each competitor"
- Only renders if wikidata coverage >20%; otherwise shows placeholder with coverage stats

**Section 3: Content Overlap Analysis**
- Overlap detection by genre, type, decade, certification
- Heatmap of genre overlap intensity (Netflix vs Max)
- Audit table: matched title pairs with confidence score and evidence fields
  - Columns: title, platform_a, platform_b, genre_match, type_match, year_proximity, confidence
- Strategic implication text: high overlap = curation opportunity, low overlap = complementary

**Section 4: Gap Analysis Tool with Decision Trace**
- Perspective selector (merged vs any competitor)
- Gap types: genre share <5%, avg IMDb <6.5 in a genre, decade coverage <100 titles, geographic absence
- Output: prioritized gap list (High/Medium/Low severity)
- **Decision trace per gap** (this is mandatory, not optional):
  - Coverage %: current share of this content type
  - Quality benchmark: avg IMDb for this gap category
  - Competitor lead: which platform leads and by how much (e.g., "Prime Video has 4.2×")
  - Confidence: High/Medium/Low based on data_confidence average for relevant titles
  - Acquisition recommendation: "Acquire 20–30 titles, Korean thriller, 7.0+ IMDb, 2018–2023"

**Section 5: IP Synergy Map** *(new — requires TMDB enrichment)*
- Franchise/collection concentration: top 15 franchises by total quality score
- Pre vs post-merger franchise portfolio comparison
- Which franchises become dominant in the merged entity
- Only renders if TMDB collection coverage >15%

**Section 6: Competitive Positioning**
- Select competitor
- Output: where they lead, where merged entity leads, battleground genres, white space
- Prose-format strategic recommendation (generated from data, not hardcoded)
- Portfolio priority sliders: fill genre gaps / boost quality / expand international / acquire prestige
  - Output: expected impact on key KPIs given priority allocation

**Section 7: Market Impact Simulation**
- Catalog size share chart (pie)
- Quality-weighted share chart (catalog size × avg IMDb)
- HHI calculation with regulatory context label ("Post-merger HHI: XXXX — Moderate concentration")
- Clearly labeled: "Simulated market based on catalog data only. Not actual financial market share."

**Definition of done:**
- Overlap and gap tables come from `data/precomputed/strategic_analysis/`
- Decision trace fields present on every gap output row
- Prestige Index and IP Synergy Map each show a coverage disclaimer and graceful fallback
- No section overclaims financial or subscriber data

---

### Page 6: Interactive Lab (pages/06_Interactive_Lab.py)
**Purpose:** High-engagement, playful but data-driven features.

**Feature 1: Build Your Streaming Service**
- Budget-constrained title drafting game
- Title value formula: f(IMDb score, TMDB popularity, recency, franchise bonus from collection_name)
- If budget/box office available: optional "real economics" mode where value incorporates box office tier
- Live dashboard: spend remaining, avg IMDb, genre distribution donut, diversity score
- Compare drafted service vs Netflix+Max: side-by-side table + radar chart (quality, diversity, size, value)
- Export drafted catalog as CSV

**Feature 2: Hypothetical Title Predictor (Greenlight Model)**
- Inputs: description, type (Movie/Show), genres, runtime, release year, production country, certification, budget tier
- Separate trained models for movies vs shows (`models/greenlight_movie_predictor.pkl`, `models/greenlight_show_predictor.pkl`)
- Training features include (where available): genre vector, runtime, release_year, country tier, has_franchise flag, budget_tier, award_genre_avg (avg awards in that genre from Wikidata)
- Output:
  - Predicted IMDb score range with uncertainty band ("7.4 ± 0.3")
  - Success tier badge (High risk / Moderate / Strong / Blockbuster)
  - Feature importances chart (mandatory — makes model defensible)
  - Model card expander: CV RMSE, baseline comparison (global mean), top 5 features
- Talent suggestions:
  - Top 5 directors in selected genres (avg IMDb, title count, platform presence)
  - Top 10 actors with strong track record (avg IMDb, genre versatility score)
  - Data now includes writers/producers from IMDb principals — optionally surface top writers too
- Platform fit: "This concept fits [Platform]" with reasoning from platform genre strengths

**Feature 3: Insight Generator**
- Scope dropdown: platform, genre, decade, all platforms
- "Generate Insights" button → 5–8 data-backed, specific insights
- Insights cite the exact metric: "Netflix+HBO has 4.2× more prestige TV than Paramount+ (award_wins normalized per 1,000 titles)"
- "Surprise me!" button → random insight from precomputed pool
- Insight pool generated offline from `titles_enriched.parquet` including enrichment-powered insights when data available

**Definition of done:**
- Predictor loaded via cached model files, never retrained at runtime
- Feature importances chart always shown
- Model card always shown (CV RMSE + baseline)
- Insight generator cites metric source inline

---

### Page 7: Cast and Crew Network (pages/07_Cast_Crew_Network.py)
**Purpose:** Explore the people behind the content — collaboration networks, rankings, influence.

Must include:

**Section 1: Person Search and Profile**
- Search by name (autocomplete), filters: role, min title count, platform scope
- Role filter now includes: Actor, Director, Writer, Producer, Composer, Cinematographer (from IMDb principals)
- Profile:
  - Key stats: avg IMDb, total votes proxy, career span, top genre, top role category
  - Awards context (if available): titles they worked on with award_wins > 0
  - Top collaborators: top 5, clickable, with collaboration count and shared role
  - Filmography table: sortable, paginated (25/page), columns: Title | Year | IMDb | Role | Character | Platform
  - Career trend line: IMDb scores over time (release year)

**Section 2: Community Detection** *(new — requires IMDb principals)*
- Louvain community detection on collaboration graph (actors + directors + writers as nodes, shared titles as edges)
- Communities labeled by dominant genre + top titles: "Crime Drama Circle (led by X, Y)"
- "Creative Circles" visualization: top 5–8 communities with member count, top titles, avg IMDb
- Cross-platform bridges: people whose work spans both Netflix-heavy and HBO/Max-heavy clusters — "connective tissue" talent of the merged entity
- Only renders if IMDb principals enrichment is present

**Section 3: Influence Scoring** *(new)*
- PageRank on collaboration graph
- Influence score = PageRank × avg IMDb × (1 + normalized award_wins if available)
- Rankings: top 50 most influential people in merged catalog
- Insight: "Most central talent are not always the most famous — they bridge the most creative communities"

**Section 4: Rankings**
- Tabs: Directors | Actors | Writers (new tab, requires IMDb principals)
- Ranking toggle: most titles, highest avg IMDb, most popular (TMDB), highest influence score
- Compare 2–3 people side-by-side

**Optional: Career Trajectory Change-Point Detection**
- Detect when a person's average IMDb score jumps or falls significantly across their filmography
- Simple z-score or moving average approach
- Only shown on profile if person has ≥8 titles

**Definition of done:**
- Network uses `data/precomputed/network/edges.parquet` and `person_stats.parquet`
- Edges now include all role categories (not just actor/director)
- Community detection only renders if IMDb principals enrichment present; graceful fallback to basic profiles
- Influence score shown with methodology expander

---

## 5) Engineering requirements

### 5.1 App code rules
- Pages are thin. Logic lives in `src/`.
- All paths and constants come from `src/config.py`.
- No repeated data loading in pages. Use `src/data/loaders.py`.
- Enriched data accessed via `load_enriched_titles()` — falls back to base titles if enriched file absent.

### 5.2 Caching rules
- `st.cache_data` for data tables and computed tables
- `st.cache_resource` for models, vectorizers, UMAP objects
- Enrichment loaders follow same pattern as base loaders

### 5.3 Enrichment graceful degradation rules
Every feature that depends on enrichment data must:
1. Check for column existence and non-null coverage before rendering
2. Show a `st.info()` or placeholder if coverage is insufficient (threshold per feature defined above)
3. Never crash or show misleading empty charts — always show a message explaining what data is missing and why

### 5.4 Testing (minimal)
- `tests/test_data_pipeline.py`: validates processed and enriched datasets exist with expected columns
- `tests/test_similarity.py`: sanity checks similarity outputs
- `tests/test_analysis.py`: overlap + gap logic sanity checks
- `tests/test_enrichment.py`: validates enrichment joins produce expected schema, coverage >0 for key fields

---

## 6) Out of scope (explicit)
- Live web scraping or real-time market share data
- Real financial market share or subscriber counts (use only what is in public earnings reports if manually compiled)
- Full UI snapshot tests
- Perfect causal claims about merger outcomes
- Training large deep learning models at runtime
- OMDb API (Wikidata covers the same fields more cleanly)

---

## 7) Success criteria

### Technical
- All 7 pages functional with consistent global filters
- Enrichment pipeline runs end-to-end without errors; missing data handled gracefully
- Hybrid recommender returns sensible results with vibe tag matching
- ML models have documented CV RMSE and feature importance
- Community detection and influence scoring render on Cast & Crew page
- App runs locally and deploys on Streamlit Cloud

### Product
- Clear separation of purpose per page (no feature duplication)
- Decision traces visible on all Strategic Insights gap outputs
- Prestige Index and IP Synergy Map render with appropriate coverage disclaimers
- Vibe search shows detected signals before results

### Academic
- Demonstrates real data engineering (pipeline, enrichment joins, parquet, caching, CI)
- Demonstrates ML thoughtfully (hybrid recommender, greenlight predictor with evaluation, community detection)
- Demonstrates NLP (sentence embeddings, vibe tag matching, genre vectors)
- Demonstrates network science (PageRank, Louvain community detection, cross-platform bridges)
- Demonstrates decision-support thinking (decision traces, prestige index, gap analysis with evidence)
- Transparent about all limitations, data sources, and enrichment coverage