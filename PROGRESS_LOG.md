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

### Next
- [ ] Implement `scripts/01_clean_raw_data.py`
  - Read 12 raw CSVs
  - Standardize schema across platforms
  - Handle missing values (IMDb, release years, descriptions)
  - Output to data/interim/ (12 parquet files)
- [ ] Implement `scripts/02_merge_platforms.py`
  - Merge Netflix + Max → data/processed/merged_titles.parquet (~9K rows)
  - Merge all 6 platforms → data/processed/all_platforms_titles.parquet (~25K rows)
  - Do same for credits
- [ ] Implement `src/data/loaders.py`
  - `load_merged_titles()` with @st.cache_data
  - `load_all_platforms_titles()` with @st.cache_data
  - `load_merged_credits()` with @st.cache_data
- [ ] Implement `src/ui/filters.py` (global filter sidebar)
- [ ] Implement `src/ui/session.py` (session state management)
- [ ] Create basic `Home.py` that loads merged data and renders filters

### Blockers
- None

---

## [Date] Week 2: Home Dashboard

### Done
-

### Next
-

### Blockers
-

---

## [Date] Week 3: Explore Catalog

### Done
-

### Next
-

### Blockers
-

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
