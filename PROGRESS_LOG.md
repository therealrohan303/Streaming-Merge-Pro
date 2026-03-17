# Progress Log: Netflix + Max Merger Capstone

## [2026-02-26] Discovery Engine — round 2 refinements

### Done
- ✅ **Title detail panel** — each recommendation card in all 3 tabs now has an ℹ️ toggle button. Clicking opens a full detail panel below the card list (mirrors Platform DNA's inline detail): poster (180px), 2×4 metadata grid (Type, Year, IMDb, Platforms / Rating, Runtime, Votes, Quality Score), award badge, box office, genre pills, description, Cast & Crew expander, "Open in Explore Catalog" link. Clicking ▼ or the close button dismisses the panel. Uses `discovery_detail_id` session state.
- ✅ **Vote threshold** — added `min_votes: int = 0` parameter to `get_similar_titles()`, `get_similar_with_explanation()`, `vibe_search()`, `mood_board_recommendations()`. All page call sites pass `min_votes=1000`, filtering out titles with fewer than 1,000 IMDb votes.
- ✅ **Mood Board quality weight raised** — `mood_score` formula changed from `0.7 × mood_match + 0.3 × quality` to `0.6 × mood_match + 0.4 × quality`. Higher quality weight surfaces iconic/popular titles (e.g. romcoms like "How to Lose a Guy in 10 Days") over obscure keyword matches.
- ✅ **Scope bug fixed** — Title Match now correctly passes `load_merged_titles()` when scope = "Merged" and `load_all_platforms_titles()` when scope = "All Platforms". Previously always passed the full 25K-title pool regardless of scope.
- ✅ **History tab revamped** — removed session-level "Watched / Interested / Not for me" feedback buttons (nonsensical for a list of 15 recommendations). Replaced with a clean read-only "Recent Searches" log: per-entry ✕ remove button (keyed by unique `_id`), tab-type icon (🔍 🎭 ✨), poster thumbnail strip, group-by toggle (Chronological / By Tab Type), Clear All button.

### Files changed
- `src/analysis/similarity.py` — added `min_votes` param to `get_similar_titles()`
- `src/analysis/discovery.py` — added `min_votes` to 3 functions; raised Mood Board quality weight 0.3→0.4
- `pages/04_Discovery_Engine.py` — detail panel (`_meta_cell`, `_render_title_detail`), scope fix, vote thresholds, history redesign

### Next
- [ ] Review remaining pages for card/visual inconsistencies
- [ ] Continue with any new tasks

## [2026-02-26] Discovery Engine full revamp

### Done
- ✅ **Tab 1: Title Match** — replaced clunky two-step search (text_input → dropdown → button) with smart autocomplete: inline poster thumbnail + year + platform badge per result, click to select, search fires on button. Added "Surprise Me" button (random high-quality title, quality ≥ 7.0, votes ≥ 5K).
- ✅ **Why similar? expander rebuilt** — now shows 4 labeled rows: Narrative Similarity (%, note), Genre Alignment (n of n overlap with genre list), Shared Crew (name + role for up to 3 people), Shared Vibe Tags (pills). No longer dumps raw genre list — feels like a human editor wrote it.
- ✅ **Tab 2: Mood Board** — completely replaced the generic Preference-Based tab. 16 mood tiles in 4×4 grid (emoji + label), each mapped to MovieLens genome tags + TMDB keywords. Content type + platform scope controls. Results show Mood Match % badge and matched mood tag pills below each card. Backend: `mood_board_recommendations()` added to `src/analysis/discovery.py`; `MOOD_TILES` + `MOOD_TILE_BY_LABEL` constants defined.
- ✅ **Tab 3: Vibe Search** — fixed default min IMDb 0.0 → 6.5 (eliminates Scooby-Doo / Viking Wolf / Rip Tide from results). Optional filters now open by default. Detected themes display rebuilt as styled pills with 🏷️ header + explanatory note. Vibe score replaced with relative label: "Strong match" / "Good match" / "Partial match" (rank percentile within results). "How Vibe Search works" expander now shows a proper info card with 5-component bar chart visualization instead of raw bullet list.
- ✅ **Tab 4: History** — fixed vertically-stacked word-wrapping buttons: now 3 compact icon buttons (🔖 ✓ ✕) side by side with tooltips; mark shows as colored text after click. Added group-by toggle (Chronological / By Tab Type). Added thumbnail strip (first 3 result posters, 42px). Added consecutive-duplicate suppression (same type+query increments count_runs instead of inserting duplicate). Moved Clear History to top alongside toggle.
- ✅ **Unified poster card** across all tabs: `_render_rec_card()` renders poster-left layout (90×130px poster or platform-colored placeholder with initial), title+year, platform badges, IMDb+votes, genre pills, tab-specific score badge (similarity % colored pill / mood % / vibe label). Same layout everywhere — no tab has a different card structure.
- ✅ **History thumbnail storage** — each history entry now stores `result_poster_urls` (first 3 poster URLs) captured at search time for display in History tab.
- ✅ **History deduplication** — `_add_to_history()` helper checks if top of history has same type+query; if so, increments `count_runs` instead of inserting duplicate.

### Files changed
- `pages/04_Discovery_Engine.py` — full rewrite
- `src/analysis/discovery.py` — added `MOOD_TILES`, `MOOD_TILE_BY_LABEL`, `mood_board_recommendations()`

### Next
- [ ] Review remaining pages for card/visual inconsistencies
- [ ] Continue with any new tasks

## [2026-02-25] Platform DNA matching system rebuild (session 2)

### Done
- ✅ **Award badge fix** — "1W" → "1 win" / "3 wins" in recommendation cards
- ✅ **Matching algorithm rebuilt** (`compute_swipe_results_v2` v3):
  - `maturity` dimension now uses cert-based maturity (60%) + dark genre % (40%) — captures prestige-drama platforms like Max more accurately
  - `popularity` dimension now uses median `tmdb_popularity` (real mainstream proxy) instead of IMDb quality score
  - Slider weight increased 0.25 → 0.35 (genre spec reduced 0.40 → 0.30) — sliders now drive meaningful differentiation
  - Result: comedy+romance+feel-good profile produces ~37 point spread (was ~3 pts) with Disney/lighter platforms ranking above Max/dark platforms
- ✅ **Recommendation scoring rebuilt**: `quality × genre_overlap_fraction^0.7 × tone_fit` — genre_overlap_fraction penalises titles that only partially match selected genres; tone_fit=0.15 for dark/mature titles when feel-good mode active; Wolf of Wall Street no longer surfaces for a comedy+romance feel-good profile
- ✅ **Platform Breakdown replaced** with natural-language two-sentence explanations per platform (genre coverage sentence + vibe alignment sentence dynamically generated from score components)
- ✅ **Compare Preferences table** — fixed "Best match" badge (was rendering `&quot;` HTML entities, now uses variable interpolation)
- ✅ Simulation verified: comedy + romance + feel-good (maturity=20) → Shrek, Will & Grace, It's a Wonderful Life in top recs; no crime dramas

## [2026-02-25] Platform DNA polish pass (session 1)

### Done
- ✅ Fixed `AttributeError` on Platform DNA page (NaN in collection_name index for franchise filter)
- ✅ Fixed `ImportError` on Platform Comparisons page (matplotlib required for background_gradient — installed)
- ✅ **Title card uniformity** — both neighborhood top-5 and recommendation cards now use `object-fit:cover` fixed-height poster + `min-height:320px` flex layout; match Home page card pattern
- ✅ **Algorithm v2** (`compute_swipe_results_v2`): genre-specialization score, like-precision, spread normalization, `user_selected_genres` param
- ✅ **Personality text**, **How You Match scorecard**, initial **Platform Breakdown**, **Compare Preferences table**, **Recommendations** genre+era filter

### Next
- [ ] Review remaining pages for similar card/visual inconsistencies

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

## [2026-02-20] Enrichment Pipeline + Config

### Done
- ✅ **Config updates** (`src/config.py`)
  - Added `ENRICHED_DIR`, `CACHE_DIR`, `PRECOMPUTED_DIR`, `MODELS_DIR` paths
  - Added enrichment coverage thresholds: `WIKIDATA_MIN_COVERAGE=0.20`, `WIKIDATA_COMPARISON_MIN_COVERAGE=0.15`, `TMDB_MIN_COVERAGE=0.15`
  - Added hybrid scoring weights for Discovery Engine vibe search
  - Added model paths: `GREENLIGHT_MOVIE_MODEL`, `GREENLIGHT_SHOW_MODEL`
  - Added network params: `NETWORK_MIN_TITLES=3`, `NETWORK_MIN_EDGE_WEIGHT=2`
- ✅ **Created `.env`** with TMDB API key
- ✅ **Script 05: `scripts/05_enrich_imdb.py`**
  - Streams `title.principals.tsv.gz` in chunks, filters to our `imdb_id` set
  - Extracts writer/producer/composer/cinematographer roles from IMDb principals
  - Joins `name.basics` for person names, `title.basics` for originalTitle/isAdult
  - Output: `data/enriched/imdb_enrichment.parquet`, `data/enriched/imdb_principals.parquet`
- ✅ **Script 06: `scripts/06_enrich_wikidata.py`**
  - Batch SPARQL queries (50 IMDb IDs per request) to Wikidata
  - Pulls: budget_usd (P2130), box_office_usd (P2142), award_wins (P166 count), award_noms (P1411 count)
  - Resumable caching to `data/cache/wikidata/` as JSON
  - Output: `data/enriched/wikidata_enrichment.parquet` with `data_confidence` column
- ✅ **Script 07: `scripts/07_enrich_movielens.py`**
  - Joins MovieLens 20M via `links.csv` using imdbId
  - Top 20 genome tags per movie by relevance, dense genome vectors
  - Output: `data/enriched/movielens_genome.parquet`, `data/precomputed/embeddings/genome_vectors.npy`, `data/precomputed/embeddings/genome_id_map.parquet`
- ✅ **Script 08: `scripts/08_enrich_tmdb.py`** (still running ~48%)
  - TMDB "Find by IMDb ID" → keywords, collections, production companies, poster_url
  - Per-response caching to `data/cache/tmdb/{imdb_id}.json`
  - Rate limited at 40 req/10 sec
  - Output: `data/enriched/tmdb_enrichment.parquet`
- ✅ **Script 09: `scripts/09_build_enriched_titles.py`**
  - Left-joins all 4 enrichment parquets on `imdb_id`
  - Computes `data_confidence` = fraction of enrichment fields non-null per row
  - Adds `imdb_writers` and `imdb_producers` list columns from principals
  - Output: `data/enriched/titles_enriched.parquet`

### Files touched
- `src/config.py` — enrichment config constants
- `.env` — TMDB API key
- `scripts/05_enrich_imdb.py` — new
- `scripts/06_enrich_wikidata.py` — new
- `scripts/07_enrich_movielens.py` — new
- `scripts/08_enrich_tmdb.py` — new
- `scripts/09_build_enriched_titles.py` — new

---

## [2026-02-20] ML Training + Precomputation (Scripts 10-12)

### Done
- ✅ **Script 10: `scripts/10_train_predictor.py`**
  - GradientBoostingRegressor for movies and shows (separate models)
  - Features: genre vector, runtime, release_year, country tier, has_franchise, budget_tier, award_genre_avg
  - 5-fold CV with RMSE + baseline RMSE reporting
  - Output: `models/greenlight_movie_predictor.pkl`, `models/greenlight_show_predictor.pkl`
- ✅ **Script 11: `scripts/11_precompute_strategic.py`**
  - Prestige index: award wins per 1,000 titles by platform and genre
  - Acquisition targets: gap recommendations with decision trace fields
  - Output: `data/precomputed/strategic_analysis/prestige_index.parquet`, `acquisition_targets.parquet`
- ✅ **Script 12: `scripts/12_precompute_network.py`**
  - Collaboration edges from `imdb_principals.parquet` (all role categories) + credits
  - PageRank on collaboration graph, Louvain community detection (`python-louvain`)
  - Per-person stats: avg IMDb, title count, career span, top genre, influence score
  - Fixed mixed int/str person_id issue (credits=int, principals=string) with upfront `.astype(str)`
  - Output: `data/precomputed/network/edges.parquet`, `data/precomputed/network/person_stats.parquet`

### Files touched
- `scripts/10_train_predictor.py` — new
- `scripts/11_precompute_strategic.py` — new
- `scripts/12_precompute_network.py` — new (fixed 3x: mixed type sort, perf, arrow serialization)

---

## [2026-02-20] Analysis Modules + Loaders + Tests

### Done
- ✅ **`src/analysis/discovery.py`** — Discovery Engine analysis
  - `get_similar_with_explanation()`: why-similar explanations (genre overlap, shared crew, matched tags)
  - `preference_based_search()`: multi-criteria filtering with fit score
  - `vibe_search()`: hybrid NLP search (description embeddings + keyword matching + genome tags)
  - Session history management (last 10 recommendation sets)
- ✅ **`src/analysis/strategic.py`** — Strategic Insights analysis
  - `compute_merger_kpis()`: 4 headline KPIs with tooltips
  - `compute_prestige_index()`: award wins per 1,000 titles
  - `compute_content_overlap()`: genre/type/decade heatmap + audit table
  - `compute_gap_analysis()`: gap recommendations with decision trace
  - `compute_ip_synergy()`: top franchises pre/post-merger
  - `compute_competitive_positioning()`: per-competitor SWOT
  - `compute_market_simulation()`: catalog share, quality-weighted share, HHI
- ✅ **`src/analysis/lab.py`** — Interactive Lab analysis
  - `streaming_service_builder()`: budget game with title value and live dashboard
  - `predict_title_score()`: greenlight prediction with uncertainty + feature importances
  - `generate_insights()`: fun facts insight pool with "Surprise me!" support
- ✅ **`src/analysis/network.py`** — Cast & Crew Network analysis
  - `search_person()`: person search with role filter
  - `get_person_profile()`: career stats, top collaborators, career trend line
  - `get_community_info()`: Louvain community details
  - `get_influence_rankings()`: PageRank-based rankings with role tabs
  - `compare_persons()`: side-by-side person comparison
- ✅ **`src/data/loaders.py`** — new loaders
  - `load_enriched_titles()` — fallback to base titles if enriched not available
  - `load_genome_vectors()` — (vectors_np, id_map_df) tuple
  - `load_network_edges()` and `load_person_stats()`
  - `load_greenlight_model(type)` — 'movie' or 'show', `@st.cache_resource`
- ✅ **`tests/test_enrichment.py`** — enrichment validation
  - Validates titles_enriched.parquet columns and data_confidence range
  - Validates imdb_principals.parquet role categories
  - Validates genome vectors shape matches id_map length
- ✅ **`tests/test_analysis.py`** — analysis logic sanity checks

### Files touched
- `src/analysis/discovery.py` — new
- `src/analysis/strategic.py` — new
- `src/analysis/lab.py` — new
- `src/analysis/network.py` — new
- `src/data/loaders.py` — 5 new loader functions
- `tests/test_enrichment.py` — new
- `tests/test_analysis.py` — new

---

## [2026-02-20] Pages 4-7: Discovery Engine, Strategic Insights, Interactive Lab, Cast & Crew Network

### Done
- ✅ **Page 4: Discovery Engine** (`pages/04_Discovery_Engine.py`)
  - Tab 1: Similar to a Title — autocomplete search, platform scope, results count slider, "Why similar?" expanders
  - Tab 2: Preference-Based — genre multiselect, min IMDb, type toggle, runtime slider, year range, popular↔hidden gems
  - Tab 3: Vibe Search (NLP) — text area input, sentence-transformer embeddings, detected signals display, hybrid scoring
  - Session History: last 10 recommendation sets, user marking (Interested/Watched/Not Interested)
- ✅ **Page 5: Strategic Insights** (`pages/05_Strategic_Insights.py`)
  - 7 sections: Merger Value Dashboard, Prestige Index, Content Overlap, Gap Analysis with Decision Trace, IP Synergy Map, Competitive Positioning, Market Impact Simulation
  - Conditional sections based on Wikidata/TMDB coverage thresholds
- ✅ **Page 6: Interactive Lab** (`pages/06_Interactive_Lab.py`)
  - Build Your Streaming Service — budget game, title value, live dashboard, compare vs Netflix+Max
  - Hypothetical Title Predictor — separate movie/show models, predicted IMDb ± uncertainty, feature importances chart, model card expander, talent suggestions
  - Insight Generator — fun facts tone, scope dropdown, "Surprise me!" button
- ✅ **Page 7: Cast & Crew Network** (`pages/07_Cast_Crew_Network.py`)
  - Person Search & Profile — role filter (incl. writer/producer/composer/cinematographer), awards context, career trend line, top collaborators
  - Community Detection — Louvain creative circles, cross-platform bridges
  - Influence Scoring — PageRank × avg IMDb × (1 + normalized awards), top 50
  - Rankings — Directors/Actors/Writers tabs, multiple ranking modes, side-by-side compare

### Files touched
- `pages/04_Discovery_Engine.py` — new
- `pages/05_Strategic_Insights.py` — new
- `pages/06_Interactive_Lab.py` — new
- `pages/07_Cast_Crew_Network.py` — new

---

## [2026-02-20] Existing Page Enhancements + Pipeline Update

### Done
- ✅ **Home.py** enhancements
  - 5th hero metric: "Award-Winning Titles" (conditional on ≥20% Wikidata coverage)
  - Poster thumbnails on top title cards via enriched data `poster_url`
- ✅ **Explore Catalog** (`pages/01_Explore_Catalog.py`) enhancements
  - Poster image at top of detail panel
  - Award badge: "🏆 X wins, Y nominations" from Wikidata
  - Box office in metadata captions (Wikidata)
  - "For deeper recommendations → Discovery Engine" link after similar titles
- ✅ **Platform Comparisons** (`pages/02_Platform_Comparisons.py`) enhancements
  - "Prestige Score" column in summary table (award wins per 1,000 titles, "—" if <15% coverage)
  - "Top Franchises" row in genre drill-down expanded card (from TMDB `collection_name`)
- ✅ **Platform DNA** (`pages/03_Platform_DNA.py`) enhancements
  - Cluster cards: dominant franchise if ≥3 titles in cluster
  - Awards-based trait in neighborhood explorer ("Awards Magnet: X wins, Y per 1,000 titles")
  - Added awards-based defining trait (Trait 7) in `src/analysis/platform_dna.py`
- ✅ **Updated `scripts/run_pipeline.sh`** — all 12 scripts in 3 phases

### Files touched
- `Home.py` — award metric, poster thumbnails
- `pages/01_Explore_Catalog.py` — poster, awards, box office, discovery link
- `pages/02_Platform_Comparisons.py` — prestige score, franchises
- `pages/03_Platform_DNA.py` — franchises, awards trait
- `src/analysis/platform_dna.py` — awards-based defining trait
- `scripts/run_pipeline.sh` — scripts 05-12 added

### Blockers
- Script 08 (TMDB API) still running (~48% done, ~10K/21K titles cached)

---

## [2026-02-21] Bug Fixes, Cross-Page Standardization, and Polish

### Done

#### TMDB Enrichment Complete
- ✅ Script 08 finished: 21,407/21,408 titles (1 error)
  - `poster_url`: 91.6% coverage (19,601 titles)
  - `production_companies`: 76.6% (16,402)
  - `tmdb_keywords`: 63.8% (13,651)
  - `collection_name`: 8.4% (1,796)
- ✅ Re-ran `scripts/09_build_enriched_titles.py` — 25,246 rows, 34 columns, avg confidence 0.397
- ✅ Re-ran `scripts/11_precompute_strategic.py` — prestige index (105 rows) + 34 acquisition targets

#### Phase 1: Critical Bug Fixes
- ✅ **Pages 4 & 6**: Fixed `compute_quality_score()` misuse — was `df = compute_quality_score(df)` replacing DataFrame with Series; fixed to `df["quality_score"] = compute_quality_score(df)`
- ✅ **Discovery Engine (`src/analysis/discovery.py`)**: Fixed parameter order in `get_similar_titles()` call (was `titles_df, sim_df`, should be `sim_df, titles_df`); removed invalid `scope` kwarg
- ✅ **Discovery Engine (page)**: Fixed `render_sidebar_filters()` call — was passing `st.sidebar` as the dataframe argument; fixed to standard pattern
- ✅ **Discovery Engine (`src/analysis/discovery.py`)**: Found and fixed 2 more `= compute_quality_score(df)` misuses in `preference_based_recommendations()` and `vibe_search()` — these would have crashed preference-based and vibe search tabs
- ✅ **Cast & Crew Network loaders** (`src/data/loaders.py`): Fixed `load_network_edges()` empty schema (removed nonexistent `shared_titles` column); fixed `load_person_stats()` empty schema to include all 14 actual columns
- ✅ **Network analysis** (`src/analysis/network.py`): Fixed `filmography.get("award_wins")` misuse (`.get()` on DataFrame); replaced with proper `if "award_wins" in filmography.columns:` check
- ✅ **Cast & Crew page**: Added `credits["person_id"].astype(str)` to fix type mismatch (credits=int, person_stats=str from script 12); improved diagnostic messages for missing data
- ✅ **Home page**: Fixed raw HTML rendering in title cards (multi-line f-string template issue)

#### Phase 2: Cross-Page Standardization
- ✅ Created shared `src/ui/badges.py` with `platform_badge_html()` and `platform_badges_html()` — canonical badge rendering using `text_color` from PLATFORMS config
- ✅ Replaced 3 duplicate badge implementations:
  - `Home.py` — removed local `_platform_badges_html()`, imported shared helper
  - `pages/01_Explore_Catalog.py` — removed local `_platform_badge()` + `_platform_badges()`, imported shared helper
  - `pages/04_Discovery_Engine.py` — removed local `_platform_badge()` + `_platform_badges()`, imported shared helper
- ✅ Cleaned up unused imports in Discovery Engine (`ALL_PLATFORMS`, `PLATFORMS`, `PLOTLY_TEMPLATE`, `get_credits_for_view`)
- ✅ Verified all `compute_quality_score()` call sites across entire codebase — all 13 instances use correct `df["col"] = compute_quality_score(df)` pattern

#### Phase 3: Strategic Insights Polish
- ✅ **Removed Section 6 (Competitive Positioning)** — redundant with Page 2's Strategic Insights section (genre leads, battlegrounds, SWOT analysis already covered there)
- ✅ **Section 3 (Content Overlap)**: Added complementarity score metric, replaced grouped bar with stacked bar (Netflix Exclusive / Shared / Max Exclusive), data-driven strategic implication text citing specific genres
- ✅ **Section 4 (Gap Analysis)**: Added summary metrics (total/high/medium gaps), replaced flat table with card-style rendering using severity-colored left borders
- ✅ **Section 5 (IP Synergy)**: Better fallback showing TMDB enrichment progress (reads cache to show X/Y progress); added column formatting for dataframes; added cross-platform franchise detection
- ✅ **Section 6 (Market Simulation, was 7)**: Added pre-merger vs post-merger side-by-side pie charts; replaced plain HHI metric with gauge visualization + threshold legend card
- ✅ Removed unused imports: `compute_competitive_positioning`, `render_sidebar_filters`, `apply_filters`, `WIKIDATA_COMPARISON_MIN_COVERAGE`

#### Phase 4: Cast & Crew Network Robustness
- ✅ Added defensive `platform_list` handling in `get_community_details()` — checks column existence, handles stringified lists from parquet
- ✅ Added defensive `platform_list` handling in `get_cross_platform_bridges()` — `_parse_platform_list()` helper, early return if column missing

#### Verification
- ✅ All 21 tests pass (21/21) — including previously-failing TMDB enrichment test

### Files touched
- `src/ui/badges.py` — **new**: shared platform badge rendering
- `src/analysis/discovery.py` — param order fix, removed `scope` kwarg, 2 quality score fixes
- `src/analysis/network.py` — award_wins fix, platform_list defensive handling
- `src/analysis/strategic.py` — no changes (page-side only)
- `src/data/loaders.py` — network loader empty schema fixes
- `Home.py` — shared badge import, HTML template fix
- `pages/01_Explore_Catalog.py` — shared badge import, removed local helpers
- `pages/04_Discovery_Engine.py` — shared badge import, sidebar fix, removed local helpers, cleaned imports
- `pages/05_Strategic_Insights.py` — full rewrite: removed Section 6, improved Sections 3-5 and 7
- `pages/06_Interactive_Lab.py` — quality score fix
- `pages/07_Cast_Crew_Network.py` — person_id type fix, diagnostic messages

### Next
- [ ] Manual page-by-page testing (`streamlit run Home.py`)
- [ ] Final integration verification of all 8 pages
- [ ] Update PROGRESS_LOG to mark project as feature-complete

### Blockers
- None

---

## [2026-02-23] UI Polish: Home, Explore Catalog, Platform Comparisons

### Done

#### Shared Utilities
- ✅ **`src/config.py`**: Changed merged entity color `#7B1FA2` → `#00897B` (teal); affects all platform badges, charts, and accent colors globally
- ✅ **`src/ui/badges.py`**: Added 3 shared HTML helpers used across all 3 pages:
  - `section_header_html()` — left-border accent header with bold title + muted subtitle
  - `styled_metric_card_html()` — colored top-border card: label → value → delta badge → subtitle; supports `help_text` tooltip
  - `styled_banner_html()` — icon + text info banner replacing plain `st.info()` calls

#### Home.py
- ✅ Replaced all `st.metric()` hero cards with `styled_metric_card_html()` — teal top-border, muted label above value, delta badge below; "Award-Winning Titles" shows `subtitle="(Wikidata)"`
- ✅ Replaced `st.info()` merger insight callout with `styled_banner_html()`
- ✅ Volume bar chart: added value labels on top of each bar, explicit color map (Netflix=`#E50914`, Max=`#002BE7`, Merged=`#00897B`), legend visible
- ✅ IMDb distribution: named legend entries, distinguishable fills
- ✅ Genre chart: x-axis labels rotated 45° with `automargin`, "Documentary" label corrected
- ✅ Top titles: changed 4-column → 5-column grid, show top 10 (was 20), added colored quality score bar (green/amber/red), poster placeholder div with platform color + initial letter when no poster URL, "View Details" button right-aligned
- ✅ Global Reach: 3 metric cards (Countries, International %, Top Market) moved above chart; chart full-width with teal bars and value labels; x-axis padded for label overflow
- ✅ Navigation cards: replaced `st.page_link()` with styled HTML cards (teal left-border, icon, arrow)
- ✅ Footer: replaced `st.caption()` with styled footer div (border-top, centered muted text)
- ✅ All `st.markdown("---")` separators replaced with `st.divider()`
- ✅ All section headers use `section_header_html()` with muted subtitles

#### pages/01_Explore_Catalog.py
- ✅ Pagination: consolidated from 3 rows to a single 8-column row; arrows changed from mixed Unicode blocks (`⏮ ◄ ► ⏭`) to consistent typographic family (`« ‹ › »`)
- ✅ Compact inline filter strip below search row: Type / Year / IMDb presets / Genre / Reset
- ✅ Selected card: changed from glow shadow to `border-left:4px solid` gold indicator
- ✅ Detail panel: award badge styled as green pill, metadata in 2-row × 4-column grid (row 2: Rating / Runtime / Votes / Quality Score bar)
- ✅ Descriptions: removed "Continue reading" expander — full description always shown
- ✅ Similar titles: changed single-column list to 2-column grid; similarity badges colored by score (green ≥ 0.75, amber ≥ 0.60, gray below)
- ✅ Cast & Crew expander: directors shown first with "Director:" label; actors in 2-column grid

#### pages/02_Platform_Comparisons.py
- ✅ Quick summary bar: replaced `st.info()` with 3 styled cards (Volume Leader, Quality Leader, Genre Leader); Genre Leader card now shows merged entity name explicitly for clarity
- ✅ Summary table: pre-formatted numeric columns before Styler to prevent raw float display (e.g. `205.300000` → `205.3`)
- ✅ Violin plot: fixed `yaxis_range=[0, 10]`; added reference lines at 7.0 ("Good") and 8.0 ("Excellent")
- ✅ Market Positioning: increased right-side x-axis padding to prevent merged entity bubble clipping; added `cliponaxis=False`
- ✅ Age certification chart: semantic color palette (mature → red tones, teen → amber, family/children → green tones, unrated → gray)
- ✅ Genre heatmap: leader annotation uses `color="#FFD700"` (gold); all annotations use `font.family="Arial"` uniformly (was "Arial Bold" for leaders — not a valid Plotly font)
- ✅ Genre Deep Dive top title cards: reformatted to match Explore Catalog card style (platform badge + type + IMDb + vote count)
- ✅ Genre Deep Dive: Quality Leader badge shown on best-average-IMDb platform per selected genre
- ✅ Strategic Insights prose: removed generic boilerplate phrases; each competitor gets data-driven framing
- ✅ Descriptions in expanded cards: removed "Continue reading" expander — full text always shown

### Files touched
- `src/config.py` — merged color, teal accent
- `src/ui/badges.py` — 3 new shared helpers
- `Home.py` — all 6 sections + footer
- `pages/01_Explore_Catalog.py` — pagination, filter strip, cards, detail panel, similar titles, cast & crew
- `pages/02_Platform_Comparisons.py` — summary cards, table formatting, violin, positioning, age certs, heatmap, deep dive, insights

### Next
- [ ] Final integration testing of all 8 pages end-to-end
- [ ] Mark project as feature-complete

### Blockers
- None

---

## [2026-02-24] Info Icon Tooltip — Iterative Fix

### Done
- ✅ **`src/ui/badges.py`** — resolved `help_text` tooltip icon in `styled_metric_card_html()` through multiple iterations:
  - Removed `cursor:help` (was rendering as question-mark cursor, not a tooltip)
  - Switched from HTML `title` attribute to CSS `::after` pseudo-element tooltip using `data-tip` attribute — `title` doesn't fire reliably inside Streamlit's rendering sandbox
  - Tried `font-style:italic` letter "i" → removed (appeared as slanted stroke)
  - Tried `font-family:Georgia,serif` letter "i" → still ambiguous vs capital "I" at small sizes
  - Tried Unicode `ℹ` (U+2139) and `ⓘ` (U+24D8) — both rendered as capital "I" at the displayed size
  - **Final fix**: inline SVG (`13×13 px`) drawing a circle + `?` character — pixel-perfect, font-independent, consistent across all OS/browsers

### Files touched
- `src/ui/badges.py` — tooltip icon implementation in `styled_metric_card_html()`

### Blockers
- None

---

## [2026-02-24] Visual Standardization: Pages 03–07

### Done
Applied the same visual UI/UX patterns from pages 00–02 uniformly across pages 03–07. No new patterns introduced — only the existing shared helpers and conventions were applied.

#### Rules applied to all 5 pages
- `st.header/subheader(X)` + `st.caption(Y)` → `st.markdown(section_header_html(X, Y), unsafe_allow_html=True)`
- `st.metric(label, value)` KPI rows → `st.markdown(styled_metric_card_html(label, value), unsafe_allow_html=True)`
- Substantive `st.success(...)` → `styled_banner_html("✓", ..., green)`
- Substantive `st.info(...)` (context/insights) → `styled_banner_html("ℹ️"/"🏆", ...)`
- `st.markdown("---")` → `st.divider()`
- Footer `st.caption(...)` → styled footer div (border-top, centered muted text)
- Hardcoded `#2a2a3e` / `#1a3a1a` color values → `{CARD_BG}` from config

#### pages/03_Platform_DNA.py
- Added imports: `section_header_html`, `styled_metric_card_html` from `src.ui.badges`
- Replaced 3 `st.subheader` + `st.caption` pairs ("Platform Identity Profile", "Content Landscape", "What Platform Are You?") with `section_header_html`
- Replaced 3× `st.metric` in `_render_metrics_row()` with `styled_metric_card_html` (platform accent color)
- Replaced 4× `st.metric` in neighborhood quality row with `styled_metric_card_html`
- Replaced 5× `st.markdown("---")` with `st.divider()`
- Replaced footer

#### pages/04_Discovery_Engine.py
- Added imports: `section_header_html`, `styled_banner_html` from `src.ui.badges`
- Fixed `_render_rec_card`: hardcoded `#2a2a3e` → `{CARD_BG}` in genre pills; added `border-top:3px solid {CARD_ACCENT}`
- Fixed vibe signal pills and matched tag pills (hardcoded colors → config constants / semantic green)
- Replaced 4 `st.subheader` + `st.caption` pairs with `section_header_html`
- Replaced 3× `st.success(f"Found {n}...")` with `styled_banner_html("✓", ..., green)`
- Replaced `st.markdown("---")` with `st.divider()`
- Replaced footer

#### pages/05_Strategic_Insights.py
- Added imports: `section_header_html`, `styled_metric_card_html`, `styled_banner_html` from `src.ui.badges`
- Replaced 8 `st.header/subheader` + `st.caption` pairs with `section_header_html`
- Replaced KPI `st.metric` loop with `styled_metric_card_html` (with `help_text`)
- Replaced overlap analysis 3× `st.metric` and gap analysis 3× `st.metric` with `styled_metric_card_html`
- Replaced `st.info` (Prestige Index note) with `styled_banner_html("ℹ️", ...)`
- Replaced 2× `st.success` overlap findings with `styled_banner_html("✓", ..., green)`
- Replaced all 6 `st.markdown("---")` with `st.divider()`
- Replaced footer

#### pages/06_Interactive_Lab.py
- Added imports: `section_header_html`, `styled_metric_card_html` from `src.ui.badges`
- Replaced 7 `st.subheader` + `st.caption` pairs with `section_header_html`
- Added `border-top:3px solid {CARD_ACCENT}` to budget stats box outer div
- Replaced 6× `st.metric` (3 "Your Service" + 3 "Netflix + Max" comparison) with `styled_metric_card_html`
- Replaced 2× `st.markdown("---")` with `st.divider()`
- Replaced footer

#### pages/07_Cast_Crew_Network.py
- Added imports: `section_header_html`, `styled_metric_card_html`, `styled_banner_html` from `src.ui.badges`
- Replaced `st.header("Person Search & Profile")` + `st.caption(...)` with `section_header_html`
- Replaced `st.subheader(profile["name"])` (dynamic) with `section_header_html(profile["name"])`
- Replaced 5× `st.metric` (Avg IMDb, Titles, Career Span, Top Genre, Role) with `styled_metric_card_html`
- Replaced `st.info("Award-winning work...")` with `styled_banner_html("🏆", ..., bg="rgba(255,215,0,0.1)", border_color="#FFD700")`
- Replaced `st.subheader("Top Collaborators")` and `st.subheader("Filmography")` with `section_header_html`
- Replaced 4 `st.header/subheader` + `st.caption` pairs ("Creative Circles", "Cross-Platform Bridges", "Influence Scoring", "Rankings") with `section_header_html`
- Replaced 4× `st.markdown("---")` with `st.divider()`
- Replaced footer
- Kept all diagnostic `st.info/warning` (data missing, no matches found) as native Streamlit

### Files touched
- `pages/03_Platform_DNA.py`
- `pages/04_Discovery_Engine.py`
- `pages/05_Strategic_Insights.py`
- `pages/06_Interactive_Lab.py`
- `pages/07_Cast_Crew_Network.py`

### Next
- [ ] Final integration testing of all 8 pages end-to-end
- [ ] Mark project as feature-complete

### Blockers
- None

## [2026-02-24] Platform DNA Comprehensive Overhaul

### Done

#### src/analysis/platform_dna.py
- Updated `_ARCHETYPES` names: "Horror & Supernatural" → "Dark & Suspenseful", "Animation & Family" → "Animated & All-Ages", "Action & Epic Adventures" → "Action-Packed & Epic"
- Added `icon` field to all trait dicts in `compute_defining_traits` (🏆 📚 ⭐ 🌱 etc.)
- Updated `compute_defining_traits` to accept `enriched_df=None` param; added 3 new enrichment-based traits: "The Award Magnet" (cross-platform award comparison), "The Premium Producer" (avg budget_millions), "The Universe Builder" (TMDB franchise count)
- Updated `compute_platform_comparison_data` to accept and pass `enriched_df` through to `compute_defining_traits`
- Added `compute_enriched_platform_stats(all_df, enriched_df)` → per-platform award_wins, award_noms, wins_per_1k, avg_budget_millions, franchise_count, post_2015_pct, intl_pct
- Added `compute_landscape_insights_v2(landscape_df, cluster_summaries)` → specific data-backed insights with platform dominance %, cluster names, Netflix+Max overlap distribution similarity, focused vs diversified platforms
- Added `compute_neighborhood_top_titles(landscape_df, enriched_df, cluster_id, top_n)` → top N titles per neighborhood with poster_url, award_wins, box_office_usd from enriched data
- Updated `curate_quiz_titles` to accept `enriched_df=None` param and include `poster_url`, `award_wins` in returned dicts
- Added `compute_swipe_results_v2(liked_ids, all_titles, all_df, slider_prefs, enriched_df, enriched_stats)` → comprehensive matching: 35% genre cosine similarity + 20% platform affinity + 15% quality + 30% slider alignment (6 dims including awards); generates data-backed `why_match` bullets; includes enriched poster_url in recommendations

#### pages/03_Platform_DNA.py (complete rewrite)
- **Section 1 (Platform Identity Profile)**: radar chart + 3-metric quality row + new 4-metric enrichment row (Award Wins, Franchises, Post-2015 %, International %) + enhanced trait cards with emoji icons and enrichment data
- **Section 2 (Content Landscape)**: data-backed insights panel with specific numbers (cluster dominance %, avg IMDb, Netflix+Max overlap %); UMAP scatter with reduced density (200/archetype base); platform highlight multiselect with opacity control (selected=full, unselected=8%); richer hover tooltips (title, platform, IMDb, year); neighborhood top-5 title cards with poster images, award badges, and inline detail panel (same format as Explore Catalog)
- **Section 3 (Quiz — Phase A)**: 12-genre grid with primary/secondary button toggle + selection counter; 6 sliders (Era, Gems vs Blockbusters, Runtime, Tone, International, Awards vs Entertainment) with end-label annotations; movies/shows/both selector
- **Section 3 (Quiz — Phase B)**: poster image + card layout with award badge; ❤️/👎 two-button swipe with liked counter in progress bar
- **Section 3 (Quiz — Phase C)**: hero card (platform name + match %) + "Your Viewing DNA" panel; "Why This Match" section with specific data-backed checkmark bullets; horizontal bar chart (platform match scores); 5-title recommendation grid with poster images + inline detail panel; runner-up list with inline progress bars; "Compare to other platforms" expander with full data table

### Files touched
- `src/analysis/platform_dna.py`
- `pages/03_Platform_DNA.py`

### Next
- [ ] Final integration testing of all 8 pages end-to-end
- [ ] Mark project as feature-complete
