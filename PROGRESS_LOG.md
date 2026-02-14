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

## [Date] Week 4: Platform Comparisons

### Done
-

### Next
-

### Blockers
-

---

## [Date] Week 5: Platform DNA

### Done
-

### Next
-

### Blockers
-

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
