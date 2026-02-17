# Progress Log: Netflix + Max Merger Capstone

## [2026-02-10] Week 1: Foundation

### Done
- ✅ Created lean folder structure (data/, src/, scripts/, tests/, pages/)
- ✅ Renamed and moved 12 raw CSVs to data/raw/ (standardized platform keys)
- ✅ Created documentation: CLAUDE.md, PROJECT_SPEC.md, README.md
- ✅ Created src/config.py with paths and platform metadata
- ✅ Created .streamlit/config.toml with theme
- ✅ Created .gitignore
- ✅ Made initial git commit
- ✅ Repo hygiene pass:
  - Fixed .gitignore duplicate .DS_Store entry
  - Fixed CLAUDE.md page count (7-page → 8-page)
  - Removed stale requirements.txt note from README.md
  - Pinned dependencies with ~= (compatible release) in requirements.txt
  - Added .python-version (3.9)
  - Added .editorconfig for consistent formatting
  - Added .pre-commit-config.yaml (black, isort, trailing-whitespace, end-of-file-fixer)
  - Added tests/conftest.py with shared fixtures (sample_titles_df, sample_credits_df, tmp_data_dir)
  - Added .github/workflows/ci.yml (pytest on Python 3.9 + 3.11)

- ✅ Implemented `scripts/01_clean_raw_data.py` — 12 raw CSVs → 12 interim parquets
  - Adds `platform` column, parses genres/countries to lists, standardizes type (Movie/Show)
  - Renames credits `id` → `title_id`, coerces numerics, computes decade + quality_tier
- ✅ Implemented `scripts/02_merge_platforms.py` — 4 processed parquets
  - merged_titles (9,167 rows), merged_credits (146,234 rows)
  - all_platforms_titles (25,246 rows), all_platforms_credits (370,546 rows)
- ✅ Created `scripts/run_pipeline.sh` (runs 01 → 02)
- ✅ Implemented `src/data/loaders.py` with @st.cache_data loaders + view routing
- ✅ Implemented `src/ui/session.py` (session state defaults + reset)
- ✅ Implemented `src/ui/filters.py` (sidebar filters + apply_filters + quick stats)
- ✅ Created `Home.py` — hero metrics, merger impact charts, top titles, timeline, nav cards
- ✅ Created 7 page stubs in `pages/`
- ✅ Updated `tests/conftest.py` fixtures to match pipeline output format

### Next
- [ ] Add `tests/test_data_pipeline.py` to validate processed datasets
- [ ] Implement Page 1: Explore Catalog
- [ ] Implement Page 2: Platform Comparisons

### Blockers
- None

---

## [2026-02-12] Week 2: Home Dashboard Polish

### Done
- ✅ **Phase 1 — Bayesian Quality Scoring & Top Titles Redesign**
  - Created `src/analysis/scoring.py`: bayesian_imdb(), normalize_popularity(), compute_quality_score(), format_votes()
  - Replaced naive IMDb sort with 70% Bayesian IMDb + 30% normalized TMDB popularity composite
  - Redesigned Top Titles as 4-column card grid with Movies/Shows tabs
  - Added "How Rankings Work" methodology expander
- ✅ **Phase 2 — Capitalization & Professional Formatting**
  - Added `_platform_name()` helper for consistent display names
  - Title-cased genre labels, descriptive chart subheaders, proper legend names
- ✅ **Phase 3 — Enhanced Overview Metrics**
  - Computed Netflix-only baselines for delta comparison
  - All 4 hero metrics show % or absolute deltas vs Netflix alone with help tooltips
- ✅ **Phase 4 — Quick Stats Panel**
  - Sidebar container placeholder pattern: stats render above filters
  - "Titles Shown" and "Avg IMDb Score" with deltas from unfiltered totals
- ✅ **Phase 5 — Enhanced Filter Controls** (`src/ui/filters.py`)
  - Genre multiselect: title-cased labels with counts, sorted by frequency
  - IMDb quick preset buttons (6+, 7+, 8+)
  - Help text on all 5 filter widgets
- ✅ **Phase 6 — Section Descriptions & Context**
  - Dynamic insight callout after Overview (computed merger gains)
  - "About These Comparisons" methodology expander after Merger Impact
  - Captions on all section headers
- ✅ **Phase 7 — Global Reach Geographic Section**
  - Top 10 production countries horizontal bar chart + 2 metrics (Countries Represented, International Content %)
  - ISO country code → display name mapping (30 entries, no external dependency)
  - Responds to sidebar filters like all other sections
- ✅ **Phase 8 — Final Polish Pass**
  - Fixed redundant `load_merged_credits()` call → reuses `merged_credits_all`
  - Consistent comma formatting on all metrics
  - Removed emojis from headers and card HTML for professional tone
  - Insight-driven section captions
  - `st.markdown("---")` separators between all major sections
  - Card hover effect (CSS transition + inline JS lift-and-shadow)
  - Extracted 5 card color constants to `src/config.py` (CARD_BG, CARD_BORDER, CARD_TEXT, CARD_TEXT_MUTED, CARD_ACCENT)

### Files touched
- `Home.py` — full dashboard (7 content sections + nav + footer)
- `src/analysis/scoring.py` — new: quality scoring module
- `src/ui/filters.py` — enhanced sidebar filters + quick stats
- `src/ui/session.py` — session state (unchanged, created in Week 1)
- `src/data/loaders.py` — added `_fix_list_cols` for parquet numpy→list conversion
- `src/config.py` — added card theme constants

### Next
- [ ] Add `tests/test_data_pipeline.py` to validate processed datasets
- [ ] Implement Page 1: Explore Catalog
- [ ] Implement Page 2: Platform Comparisons

### Blockers
- None

---

## [2026-02-13] Week 3: Explore Catalog

### Done
- ✅ **Precomputation — Multi-Signal Similarity**
  - Created `scripts/03_compute_similarity.py`: multi-signal similarity on 23,357 deduplicated titles
  - **4 signals**: description TF-IDF cosine (30%), genre cosine (30%), type match (15%), IMDb proximity (25%)
  - Description TF-IDF: 5000 features, (1,2) ngrams, English stop words
  - Genre cosine similarity on binary genre vectors (19 unique genres) — better than Jaccard for partial overlaps
  - Batched computation (2K rows/batch) to limit memory
  - Output: `data/precomputed/similarity/tfidf_top_k.parquet` (1,167,850 rows — top 50 per title)
  - Saved `models/tfidf_vectorizer.pkl` for reuse on Page 4 (Discovery Engine)
  - Config: `SIMILARITY_STORE_TOP_K = 50`, `SIMILARITY_MIN_SCORE = 0.4`, `SIMILARITY_MIN_IMDB = 6.0`
  - Validated: Breaking Bad → Ozark (#2), The Wire (#3); GoT → House of the Dragon (#1); The Batman → Dark Knight trilogy + Joker
- ✅ **Similarity Lookup Module** (`src/analysis/similarity.py`)
  - `get_similar_titles()`: filters by scope, min score, min IMDb (6.0+), top-K
  - Returns enriched DataFrame with title metadata + similarity_score
- ✅ **Cached Loaders** (`src/data/loaders.py`)
  - `load_similarity_data()` with `@st.cache_data`
  - `load_tfidf_vectorizer()` with `@st.cache_resource`
- ✅ **Page 1: Explore Catalog** (`pages/01_Explore_Catalog.py`)
  - Global sidebar filters (reuses filters.py pattern from Home)
  - Two-panel layout: left = paginated results, right = detail view + similar titles
  - Detail view: 4 metrics row, genre pills, rating/runtime/votes, quality score, description
  - Similar titles: scope toggle (merged vs all platforms), top 10 with similarity %, clickable View buttons
  - "How Similar Titles Work" methodology expander
  - Transparency footer
- ✅ **Explore Catalog — Enhanced Pagination**
  - Items per page selector (25, 50, 100)
  - Full navigation: ⏮ First / ◄ Prev / Page X of Y / Next ► / Last ⏭
  - Jump-to-page number input with Go button
  - Range display: "Showing 1–50 of 9,067"
- ✅ **Explore Catalog — Active Filter Indicator**
  - Filter summary bar at top of page with styled pill badges
  - Shows non-default filters: platform, content type, year range, IMDb threshold, genres
- ✅ **Explore Catalog — Search Enhancements**
  - Multi-field search: title + description + cast names (via credits join)
  - Quick suggestions: top 5 title matches shown as buttons before full search
  - Placeholder text: "Search by title, cast, or keyword..."
  - Empty state: friendly message with hints (search vs filter-no-results)
- ✅ **Explore Catalog — Sort & Display**
  - 7 sort options: Quality Score, IMDb Score, Release Year (Newest/Oldest), Most Voted, Title A-Z, Popularity
  - Result cards: hover effects, selected state (gold border + glow)
  - Detail panel in bordered container for visual separation
  - TMDB popularity formatted as K/M, consistent score formatting
- ✅ **Explore Catalog — Cast & Crew Expander**
  - Collapsible section in detail panel showing directors and up to 15 actors with character names
  - Singular/plural director label, "+X more" overflow indicator
- ✅ **Home → Explore Navigation**
  - "View Details" button on each top title card navigates to Explore Catalog detail panel
  - Uses `st.switch_page()` with pre-set `explore_selected_id` session state
- ✅ **Bug Fixes**
  - Fixed duplicate Streamlit element key errors in Home.py (`_render_title_cards` tab prefix + card index)
  - Fixed duplicate key errors in Explore page result list and similar titles (row index keying)
  - Removed "Current Selection" quick stats panel from sidebar on both pages
- ✅ **Tests** (`tests/test_similarity.py`)
  - 7 test cases: top-K limit, min score filter, scope filtering, missing ID, sort order, self-exclusion, min IMDb filter
- ✅ Updated `scripts/run_pipeline.sh` to include script 03

### Files touched
- `src/config.py` — added SIMILARITY_STORE_TOP_K=50, SIMILARITY_MIN_SCORE=0.4, SIMILARITY_MIN_IMDB=6.0, CATALOG_PAGE_SIZE
- `scripts/03_compute_similarity.py` — rewritten: multi-signal similarity (desc TF-IDF + genre cosine + type match + IMDb proximity)
- `src/analysis/similarity.py` — updated: added min_imdb filtering
- `src/data/loaders.py` — added load_similarity_data(), load_tfidf_vectorizer()
- `pages/01_Explore_Catalog.py` — full page with enhanced pagination, search, sort, cast & crew, visual polish
- `Home.py` — added View Details buttons linking to Explore page, fixed duplicate keys
- `tests/test_similarity.py` — new: similarity tests
- `scripts/run_pipeline.sh` — added script 03

### Next
- [ ] Implement Page 2: Platform Comparisons
- [ ] Add `tests/test_data_pipeline.py` to validate processed datasets

### Blockers
- None

---

## [2026-02-13] Week 4: Platform Comparisons

### Done
- ✅ **Analysis Module** (`src/analysis/comparisons.py`)
  - `build_merged_entity()`: combines Netflix + Max into single "merged" platform, deduplicates by id (9,158 titles)
  - `build_comparison_df()`: merged entity + selected competitors with display names and ordering
  - `compute_volume_stats()`: title counts per platform/type, absolute or normalized (%)
  - `compute_volume_summary()`: one-row-per-platform summary (total, movies, shows, avg/median IMDb)
  - `compute_quality_tiers()`: titles binned into Excellent/Good/Average/Below Average/Poor/Unrated, pivoted by platform
  - `compute_genre_heatmap()`: top 15 genres × platforms matrix, absolute counts or % of catalog
  - `compute_genre_drilldown()`: per-platform stats + top 5 titles for a selected genre
  - Genre display name fixes: "documentation" → "Documentary", "scifi" → "Sci-Fi"
- ✅ **Page 2: Platform Comparisons** (`pages/02_Platform_Comparisons.py`)
  - Competitor selector: multiselect up to 3 competitors, merged baseline always included
  - Absolute/normalized toggle affecting all sections
  - **Catalog Volume**: grouped bar chart (Movie/Show split) + summary table
  - **Content Quality**: box plot (absolute) / violin plot (normalized) of IMDb distributions + quality tier breakdown table
  - **Genre Landscape**: heatmap of top 15 genres × platforms with text annotations
  - **Genre Deep Dive**: selectbox drill-down showing per-platform count, avg IMDb, and top 5 titles with styled cards
  - Methodology expander explaining merged entity construction and normalization
- ✅ Added `COMPARISON_MAX_COMPETITORS = 3` and `COMPARISON_TOP_GENRES = 15` to `src/config.py`

### Files touched
- `src/analysis/comparisons.py` — new: all comparison analysis functions
- `pages/02_Platform_Comparisons.py` — replaced stub with full page
- `src/config.py` — added comparison constants

### Next
- [ ] Implement Page 3: Platform DNA
- [ ] Add `tests/test_data_pipeline.py` to validate processed datasets

### Blockers
- None

---

## [2026-02-14] Title Deduplication with Multi-Platform Badges

### Done
- ✅ **Title Deduplication** (`src/data/loaders.py`)
  - Created `deduplicate_titles(df)`: groups by `id`, aggregates `platform` column into sorted `platforms` list
  - Platform order follows `ALL_PLATFORMS` (netflix, max, disney, prime, paramount, appletv)
  - Eliminates duplicate rows for titles appearing on multiple platforms
  - Merged view: 9,167 → 9,158 rows (9 dual-platform titles like The Dark Knight, LOTR trilogy)
  - All-platforms view: 25,246 → 23,357 rows (1,870 multi-platform titles, 19 on 3+ platforms)
- ✅ **Home Page** (`Home.py`)
  - Added `_platform_badges_html()` helper for rendering multiple colored badges
  - `_render_title_cards()` now deduplicates before selecting top 20 — no more duplicate cards
  - Cards show all platform badges (e.g., Netflix + Max badges side by side)
- ✅ **Explore Catalog** (`pages/01_Explore_Catalog.py`)
  - Deduplicates after quality score computation — results list shows unique titles
  - Added `_platform_badges()` helper that handles both list and single platform
  - Results list, detail view, and similar titles all show multi-platform badges
  - Detail view label switches between "Platform" / "Platforms" based on count
- ✅ **Similarity Module** (`src/analysis/similarity.py`)
  - Uses `deduplicate_titles()` instead of `drop_duplicates` — aggregates platforms in results
  - Similar titles return `platforms` column (list) instead of `platform` (string)
- ✅ Updated `tests/test_similarity.py` scope filtering test for `platforms` column

### Files touched
- `src/data/loaders.py` — added `deduplicate_titles()`, imported `ALL_PLATFORMS`
- `Home.py` — added `_platform_badges_html()`, dedup in `_render_title_cards()`
- `pages/01_Explore_Catalog.py` — added `_platform_badges()`, dedup after quality score
- `src/analysis/similarity.py` — uses `deduplicate_titles()`, returns `platforms`
- `tests/test_similarity.py` — updated scope filtering assertion

### Next
- [ ] Implement Page 3: Platform DNA
- [ ] Add `tests/test_data_pipeline.py` to validate processed datasets

### Blockers
- None

---

## [2026-02-14] Platform Comparisons — Major Enhancements (Round 2)

### Done
- **Analysis Module** (`src/analysis/comparisons.py`)
  - `compute_market_positioning()`: per-platform bubble chart data (volume, avg IMDb, total popularity)
  - `compute_age_profile()`: stacked bar data for age certification breakdown per platform
  - `compute_geographic_diversity()`: % non-US titles per platform
  - `compute_era_focus()`: median release year per platform
  - `compute_strategic_insights()`: per-competitor SWOT-lite analysis (strengths/weaknesses/battlegrounds)
  - Expanded `compute_genre_drilldown()` to return `id`, `description`, `age_certification`, `runtime`, `genres`, `tmdb_popularity` in top titles
  - Added `QUALITY_TIERS` import for strategic insights tier comparison
- **Page 2: Platform Comparisons** (`pages/02_Platform_Comparisons.py`)
  - **Competitor Validation**: empty selection auto-defaults to Prime Video with warning
  - **Market Positioning Matrix** (new section): scatter/bubble plot — x=catalog size, y=avg IMDb, bubble size=total popularity, quadrant labels + median reference lines
  - **Content Profile** (new section, 3 columns): Age Certification stacked bar, International Content % metrics with delta vs merged, Content Recency median year metrics with delta
  - **Interactive Genre Deep Dive Cards**: hover animation (translateY + boxShadow), Details/Close toggle button, expanded card shows metadata (type, runtime, rating, quality score), genre pills, description, cast & crew (directors + top 10 actors with characters)
  - **Strategic Insights** (new section): per-competitor expanders with 3-column layout (Strengths, Merged Advantages, Battlegrounds) showing genre/tier comparisons with specific numbers
  - Updated methodology expander to document all new sections
  - Added `comp_expanded_title` session state for card expansion tracking

### Files touched
- `src/analysis/comparisons.py` — 5 new functions, expanded drilldown return columns
- `pages/02_Platform_Comparisons.py` — 3 new sections, competitor validation, interactive cards, expanded methodology

### Next
- [ ] Implement Page 3: Platform DNA
- [ ] Add `tests/test_data_pipeline.py` to validate processed datasets

### Blockers
- None

---

## [2026-02-14] Platform Comparisons — Visual & UX Polish (Round 3)

### Done
- **Analysis Module** (`src/analysis/comparisons.py`)
  - Removed redundant `median_imdb` from `compute_volume_summary()` and `compute_genre_drilldown()` — avg IMDb is sufficient, medians were adding noise
  - Rewrote `compute_strategic_insights()` to return structured, prioritized data: `summary` (volume/quality stats), `their_strengths`/`our_strengths` (top 4 sorted by magnitude), `battlegrounds` (list of genre names)
- **Page 2: Platform Comparisons** (`pages/02_Platform_Comparisons.py`)
  - **Genre Deep Dive expanded cards**: now render full-width below the columns instead of inside narrow columns — text is readable at any platform count
  - **Expanded card layout**: 6-column metadata row, description + cast side-by-side, gold accent header with title and platform name
  - **Strategic Insights redesign**: side-by-side competitor cards (not expanders), colored header with summary stats, concise natural language ("Where X leads" / "Where merged leads"), formatted values adapt to normalized mode
  - **Genre heatmap**: Plasma colorscale (better on dark theme), cell gaps for cleaner grid, improved hover template, dynamic height
  - **Normalized mode**: now applies to Genre Deep Dive stats (primary metric swaps between count and %), Strategic Insights (compares % of catalog instead of raw counts)
  - **About These Comparisons**: moved to right after Quick Comparison box (separate from Strategic Insights), condensed to 4 focused paragraphs
  - **Volume summary table**: removed Median IMDb column (redundant with Avg IMDb)

### Files touched
- `src/analysis/comparisons.py` — removed medians, rewrote strategic insights return structure
- `pages/02_Platform_Comparisons.py` — expanded card full-width, insights redesign, heatmap styling, normalized mode, methodology relocation

### Next
- [ ] Implement Page 3: Platform DNA
- [ ] Add `tests/test_data_pipeline.py` to validate processed datasets

### Blockers
- None

---

## [2026-02-14] Platform Comparisons — Final Polish (Round 4)

### Done
- **Page 2: Platform Comparisons** (`pages/02_Platform_Comparisons.py`)
  - **Section descriptions**: added contextual captions below every section header explaining what each visualization shows and why it matters (Catalog Volume, Content Quality, Market Positioning, Content Profile, Genre Landscape, Genre Deep Dive, Strategic Insights)
  - **Genre heatmap readability**: replaced single-color `texttemplate` with per-cell `add_annotation()` calls that compute adaptive text color — dark text (#1a0a2e) on bright Plasma cells (brightness > 70%), white text on dark cells
  - **Market Positioning improvements**: switched to paper-relative quadrant labels (always visible in corners), mean-based reference lines with inline value labels (more stable with 3-5 platforms), added axis padding so bubbles don't crowd edges
  - **Strategic Insights prose rewrite**: replaced data-dump rendering (raw numbers, percentages, bullet lists) with formal business-centric prose via `_build_insight_prose()` helper — generates 2-3 contextual insights per competitor covering scale/quality assessment, genre differentiation, and contested territory

### Files touched
- `pages/02_Platform_Comparisons.py` — section captions, heatmap annotations, market positioning layout, strategic insights prose

### Next
- [ ] Implement Page 3: Platform DNA
- [ ] Add `tests/test_data_pipeline.py` to validate processed datasets

### Blockers
- None

---

## [2026-02-15] Platform Comparisons — Polish (Round 5)

### Done
- **Page 2: Platform Comparisons** (`pages/02_Platform_Comparisons.py`)
  - **Content Profile ordering**: International Content and Content Recency metrics now always list the merged entity first, with competitors after
  - **Genre heatmap colorscale**: replaced Plasma (bright yellow caused readability issues) with a custom dark indigo-to-violet gradient — white text is consistently readable on all cells
  - **Quality Tier Breakdown**: capitalized index label from `tier` to `Tier`
  - **Market Positioning enhancements**: taller chart (+150px), larger bubbles (size_max=70) with white edge outlines, subtle quadrant background shading, bolder quadrant labels, asymmetric y-axis padding (35% top vs 20% bottom) so text labels above the highest bubble aren't clipped
  - **Age certification colors**: assigned distinctive custom `color_discrete_map` per rating (TV-MA red, R magenta, NC-17 purple, TV-14 orange, PG-13 gold, etc.) — no more same-shade confusion between R and Unknown
  - Moved `import pandas as pd` to top-level (removed local import from `_render_expanded_card`)

### Files touched
- `pages/02_Platform_Comparisons.py` — Content Profile ordering, heatmap colorscale, tier capitalization, market positioning layout, age cert colors

### Next
- [ ] Implement Page 3: Platform DNA
- [ ] Add `tests/test_data_pipeline.py` to validate processed datasets

### Blockers
- None

---

## [2026-02-15] Week 5: Platform DNA

### Done
- ✅ **Precomputation — UMAP Dimensionality Reduction**
  - Created `scripts/04_compute_umap.py`: projects 23,357 deduplicated titles into 2D
  - Features: TF-IDF description vectors (5,000 dims) + genre vectors (19 dims, 2x weighted)
  - Reuses fitted TF-IDF vectorizer from script 03 (no redundant fitting)
  - UMAP params: 15 neighbors, 0.1 min_dist, cosine metric, seed 42
  - Output: `data/precomputed/dimensionality_reduction/umap_coords.parquet` (23,357 rows)
  - Saved `models/umap_reducer.pkl` for potential reuse
  - Runs in ~30 seconds on Apple Silicon
- ✅ **Analysis Module** (`src/analysis/platform_dna.py`)
  - `get_platform_titles()`: extracts per-platform data; "merged" deduplicates Netflix+Max
  - `compute_genre_mix()`: top 10 genres + "Other" with counts and percentages (donut chart)
  - `compute_era_profile()`: decade distribution ordered chronologically
  - `compute_quality_profile()`: avg/median IMDb, quality tier counts/percentages, rated vs total
  - `compute_defining_traits()`: 3-5 data-driven traits comparing platform vs all-platform average
    - 6 trait categories: catalog freshness, quality positioning, content type mix, international diversity, signature genre, catalog scale
  - `compute_platform_comparison_data()`: builds full profile dict for 1-2 platforms
  - `compute_landscape_clusters()`: merges UMAP coords with title metadata, KMeans k=8 clusters
  - `compute_cluster_summaries()`: per-cluster top genres, platform mix, avg IMDb, sample titles
  - `compute_overlap_stats()`: Netflix vs Max cluster overlap/dominance metrics
- ✅ **Cached Loaders** (`src/data/loaders.py`)
  - Added `load_umap_coords()` with `@st.cache_data`
- ✅ **Page 3: Platform DNA** (`pages/03_Platform_DNA.py`)
  - Global sidebar filters (reuses filters.py pattern)
  - **Section 1: Platform Personality Profile**
    - Platform dropdown with 7 options (merged + 6 platforms)
    - Optional side-by-side comparison (2 DNA cards)
    - DNA card: styled header with platform color accent, 3 quality metrics, genre donut chart, era bar chart, quality tier horizontal bar, defining traits with direction indicators
  - **Section 2: Content Landscape**
    - UMAP scatter plot (23K points, colored by platform, hover with title/type/IMDb/year)
    - Platform legend as horizontal bar above chart
    - 8 content neighborhood cluster cards in 4x2 grid: top 3 genres, platform composition %, avg IMDb, top 4 sample titles
    - Netflix + Max overlap metrics: shared clusters, dominated clusters, overlap rate
    - Interpretive text adapts to overlap pattern (complementarity vs convergence)
  - Methodology expander explaining profiles + UMAP + clustering
  - Transparency footer
- ✅ Added `DNA_TOP_GENRES=10`, `DNA_TOP_TRAITS=5`, `DNA_UMAP_SAMPLE_SIZE=200` to `src/config.py`
- ✅ Updated `scripts/run_pipeline.sh` to include script 04

### Files touched
- `scripts/04_compute_umap.py` — new: UMAP precomputation
- `src/analysis/platform_dna.py` — new: all platform DNA analysis functions
- `src/data/loaders.py` — added `load_umap_coords()`
- `pages/03_Platform_DNA.py` — replaced stub with full page
- `src/config.py` — added DNA constants
- `scripts/run_pipeline.sh` — added script 04

### Next
- [ ] Implement Page 4: Discovery Engine
- [ ] Add `tests/test_data_pipeline.py` to validate processed datasets

### Blockers
- None

---

## [2026-02-15] Platform DNA — Major Overhaul (Round 2)

### Done
- ✅ **Apple TV Color Fix** (`src/config.py`)
  - Changed `appletv.color` from `#000000` to `#A2AAAD` (Apple Silver) — readable on dark theme
  - Changed `appletv.text_color` to `#1a1a2e` — dark text on light badge background
  - Affects all pages globally (Home, Explore Catalog, Comparisons, DNA)
- ✅ **New Analysis Functions** (`src/analysis/platform_dna.py`)
  - `compute_platform_identity_radar()`: 6 normalized (0-100) dimensions — Freshness, Quality, Breadth, Global Reach, Genre Diversity, Series Focus
  - `generate_cluster_name()`: auto-generates meaningful labels from top genres + era/quality qualifier
  - `compute_platform_profile_vector()`: 7-dimension normalized profile per platform for matcher
  - `compute_user_match_scores()`: scores user preferences against all 6 platforms — genre cosine (30%) + 6 slider dimensions (70%)
  - Modified `compute_cluster_summaries()` to include auto-generated `name` key
  - Modified `compute_platform_comparison_data()` to include radar data
- ✅ **Cached Loader** (`src/data/loaders.py`)
  - Added `load_platform_profiles()` — precomputes all 6 platform profile vectors with `@st.cache_data`
- ✅ **Page 3: Platform DNA — Full Rewrite** (`pages/03_Platform_DNA.py`)
  - **Methodology expander**: moved to standalone position at top of page (separate from content)
  - **Normalized toggle**: added in controls row (matches Page 02 pattern)
  - **CSS fix**: added overflow-wrap/word-break styles to prevent text cutoff in comparison mode
  - **Section 1: Platform Identity Profile**: replaced genre donut + era bar + quality tier bar with radar chart (Scatterpolar, 6 axes). Comparison mode overlays both platform traces on same radar. Kept compact metrics row and defining traits.
  - **Section 2: Content Landscape**: replaced 23K-point scatter with density contour plot (Histogram2dContour per platform). Added platform highlight multiselect for optional point overlay (sampled, max 500 per platform). Cluster center annotations with auto-generated names on the plot.
  - **Section 3: Content Neighborhoods**: redesigned cluster cards with named headers (gold accent), genre pills, flex layout metadata, border-top divider before sample titles, better spacing (12px margin). Normalized toggle switches platform counts ↔ percentages. Netflix+Max overlap stats kept.
  - **Section 4: What Platform Are You?** (NEW): user form with genre multiselect + 6 labeled sliders in 2 columns. Results: hero card with best match platform + %, bar chart of all 6 platforms, per-platform explanation cards.
- ✅ **Redundancy Reduction**: removed genre donut, era bar, and quality tier bar (all available on Page 02). Replaced with unique radar fingerprint.
- ✅ Added DNA config constants: `DNA_N_CLUSTERS`, `DNA_CONTOUR_NBINS`, `DNA_MATCHER_GENRE_WEIGHT`, `DNA_MATCHER_SLIDER_WEIGHT`

### Files touched
- `src/config.py` — Apple TV color, new DNA constants
- `src/analysis/platform_dna.py` — 4 new functions, 2 modified
- `src/data/loaders.py` — added `load_platform_profiles()`
- `pages/03_Platform_DNA.py` — full rewrite (4 sections, ~800 lines)

### Next
- [ ] Implement Page 4: Discovery Engine
- [ ] Add `tests/test_data_pipeline.py` to validate processed datasets

### Blockers
- None

---

## [2026-02-15] Platform DNA — Round 3 Fixes (8 Phases)

### Done
- ✅ **Phase 1: Fix Content Landscape Tooltip** (`pages/03_Platform_DNA.py`)
  - Changed `hoverinfo="name"` to `hoverinfo="skip"` on Histogram2dContour traces
  - Density contours no longer show misleading platform names on hover
- ✅ **Phase 2: Standardize Top Titles** (`src/analysis/platform_dna.py`)
  - Replaced `nlargest(5, "imdb_score")` with `compute_quality_score()` + progressive vote thresholds (10K→1K→0)
  - Matches the standard methodology used on Home, Explore, and Comparisons pages
  - Added `imdb_votes`, `tmdb_popularity`, and other metadata columns to `compute_landscape_clusters()` keep_cols
- ✅ **Phase 3: Unique Cluster Names** (`src/analysis/platform_dna.py`)
  - Redesigned `generate_cluster_name()` with expanded signature: top_genres, avg_year, avg_imdb, movie_pct, dominant_platform, used_names
  - When top 2 genres are both Drama+Comedy (common), swaps one with 3rd genre
  - Expanded qualifiers: Classics, Premium, Vault, Cinema, Series, {Platform} Zone, Hub
  - Deduplication: tries each qualifier until unique name found; falls back to era decade
  - Updated `compute_cluster_summaries()` to compute type_mix, dominant_platform, leaders, and pass used_names set
- ✅ **Phase 4: Content Landscape Interactivity** (`pages/03_Platform_DNA.py`)
  - Added title search input above the map — highlights matching titles with gold star markers
  - Search results shown below map with title, platform, IMDb, and neighborhood info
  - Added "Explore a neighborhood" selectbox below the map
  - Neighborhood expander shows: quality metrics (4-col), platform breakdown bar, top 10 titles (quality_score ranked, 2-col grid)
- ✅ **Phase 5: Radar Chart Explainer** (`pages/03_Platform_DNA.py`)
  - Added "What do the radar dimensions mean?" expander below radar chart (both single and compare modes)
  - Table explaining all 6 dimensions with scale descriptions
- ✅ **Phase 6: Distinctive Defining Traits** (`src/analysis/platform_dna.py`)
  - Extracted `_compute_all_platform_stats()` shared helper (used by radar + traits)
  - Added `_rank_among()` helper for superlative labels ("highest of all 6 platforms", "2nd highest")
  - Rewrote all 6 trait types with concrete numbers: actual counts, percentages, comparison values, and platform rankings
  - Example: "Volume Leader: 9,058 titles — 2.4x industry average (vs 3,773 avg per platform). Highest of all 6 platforms"
- ✅ **Phase 7: Removed Overlap Analysis** (`pages/03_Platform_DNA.py`)
  - Deleted "Netflix + Max: Overlap & Divergence" section (60+ lines)
  - Removed `compute_overlap_stats` import (function kept in platform_dna.py for Strategic Insights page)
- ✅ **Phase 8: Insight Callouts** (`pages/03_Platform_DNA.py`)
  - Auto-generated key insight above density map: identifies most separated and most overlapping platform pairs
  - Added per-neighborhood leaders in cluster cards: Volume leader + Quality leader (with platform colors)

### Files touched
- `src/analysis/platform_dna.py` — 3 new helpers, 3 functions rewritten, 1 function updated
- `pages/03_Platform_DNA.py` — tooltip fix, interactivity, radar explainer, overlap removal, insight callouts, leaders

### Next
- [ ] Implement Page 4: Discovery Engine
- [ ] Add `tests/test_data_pipeline.py` to validate processed datasets

### Blockers
- None

---

## [2026-02-17] Platform DNA — Round 4: Comprehensive Revamp

### Done
- ✅ **Phase 1: Thematic Hub Naming** (`src/analysis/platform_dna.py`)
  - Completely rewrote `generate_cluster_name()` with rule-based archetype matching
  - 15 thematic archetypes: "Prestige Drama Hub", "Dark Thriller Hub", "Kids & Animation Hub", etc.
  - Uses cluster signals: certification mix (family_pct, mature_pct), intl_pct, median_votes, genre composition
  - Deduplication with contextual differentiators: (International), (Series), (Classics), (Premium)
- ✅ **Phase 2: Narrative Defining Traits** (`src/analysis/platform_dna.py`)
  - Rewrote `compute_defining_traits()` with personality-driven labels and narrative details
  - Old: "Volume Leader" / "9,058 titles — 1.5x industry average"
  - New: "The Content Giant" / "With 9,058 titles, Netflix has the biggest library in streaming"
  - 11 trait types with platform names woven into natural sentences
- ✅ **Phase 3: Multi-Insight Landscape Callout** (`src/analysis/platform_dna.py`)
  - Added `compute_landscape_insights()` function returning 3-4 insights with HTML-colored platform names
  - Insight types: separation, overlap, concentration/focus, cluster ownership
  - Replaced single robotic sentence with styled multi-bullet card
- ✅ **Phase 4: Remove Redundancy + Enhance Explorer** (`pages/03_Platform_DNA.py`)
  - Deleted entire Content Neighborhoods card grid (~120 lines) — fully redundant with explorer
  - Enhanced neighborhood explorer: direct rendering (no expander), genre pills, leader tags, auto-description
  - Added UMAP highlight: selecting a neighborhood highlights its points on the density map
  - Moved neighborhood selectbox into controls row above the chart
  - Title cards now show genre tags
- ✅ **Phase 5: Tooltips Everywhere** (`pages/03_Platform_DNA.py`)
  - Added `help=` to all 6 `st.metric()` calls: Avg IMDb, Rated Titles, Titles, Premium %, Median Year
- ✅ **Phase 6: Number Formatting** (`pages/03_Platform_DNA.py`)
  - IMDb scores: `.2f` everywhere (was `.1f` in some places)
  - Percentages: `.1f%` everywhere (was `.0f%` in some places)
  - Match percentages: `.1f%` (was `.0f%`)
- ✅ **Phase 7: Loading States + Empty States** (`pages/03_Platform_DNA.py`)
  - Added `st.spinner("Mapping the content landscape...")` around UMAP computation
  - Added `st.spinner("Finding your perfect platform match...")` around matcher
  - Added empty state guards: no profiles, no cluster summaries, no matcher results
- ✅ **Phase 8: Platform Colors in Insights** (`src/analysis/platform_dna.py`)
  - `compute_landscape_insights()` wraps platform names in colored HTML spans
- ✅ **Phase 9: Your Viewing DNA** (`pages/03_Platform_DNA.py`)
  - Added personality card after matcher results summarizing user preferences in natural language
  - Maps slider values to traits: "new release hunter", "global cinema explorer", "series binger", etc.
- ✅ **Code cleanup**: deduplicated radar dimension table into `_RADAR_DIMENSION_TABLE` constant, updated trait renderer to narrative em-dash style

### Files touched
- `src/analysis/platform_dna.py` — `generate_cluster_name()` rewritten, `compute_defining_traits()` rewritten, `compute_landscape_insights()` added
- `pages/03_Platform_DNA.py` — grid deleted, explorer enhanced, traits restyled, insights multi-bullet, tooltips, formatting, spinners, empty states, viewing DNA card

### Next
- [ ] Implement Page 4: Discovery Engine
- [ ] Add `tests/test_data_pipeline.py` to validate processed datasets

### Blockers
- None

---

## [Date] Week 6-7: Discovery Engine

### Done
-

### Next
-

### Blockers
-

---

## [Date] Week 8: Strategic Insights

### Done
-

### Next
-

### Blockers
-

---

## [Date] Week 9: Interactive Lab

### Done
-

### Next
-

### Blockers
-

---

## [Date] Week 10: Cast & Crew Network

### Done
-

### Next
-

### Blockers
-

---

## [Date] Week 11: Integration & Optimization

### Done
-

### Next
-

### Blockers
-

---

## [Date] Week 12: Final Polish & Deployment

### Done
-

### Next
-

### Blockers
-
