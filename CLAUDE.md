# Claude Project Guide: Netflix + Max Merger Analysis

## Project Overview
7-page Streamlit app analyzing a hypothetical Netflix + Warner Bros (Max) merger using data from 6 streaming platforms.

## Architecture

### Data Pipeline (4 Tiers)
```
data/raw/           → Original CSVs (12 files, immutable)
data/interim/       → Cleaned parquet files (12 files)
data/processed/     → Merged datasets (4 files: merged_titles, all_platforms_titles, + credits)
data/precomputed/   → Heavy artifacts (embeddings, similarity, UMAP, stats, analysis, network)
```

### Code Organization
```
src/data/      → Loading, filtering, feature engineering
src/models/    → ALL ML/NLP (similarity, embeddings, clustering, recommendations, predictor)
src/analysis/  → Business logic (overlap detection, gap analysis, platform DNA)
src/network/   → Graph building, person profiles
src/viz/       → Charts, cards, tables
src/ui/        → Global filters, session state
```

### Platform Keys (Standardized)
Use these everywhere: `netflix`, `max`, `disney`, `prime`, `paramount`, `appletv`

## Coding Conventions

### Imports
```python
# Always import from src
from src.config import PLATFORMS, PROCESSED_DIR, PRECOMPUTED_DIR
from src.data.loaders import load_merged_titles
from src.ui.filters import render_global_filters
```

### Caching
```python
# Use @st.cache_data for DataFrames/arrays
@st.cache_data
def load_merged_titles():
    return pd.read_parquet(PROCESSED_DIR / "merged_titles.parquet")

# Use @st.cache_resource for ML models
@st.cache_resource
def load_tfidf_model():
    return joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")
```

### Global Filters (Every Page)
```python
# In every page's main function:
from src.ui.filters import render_global_filters
from src.ui.session import get_filtered_data

filters = render_global_filters()  # Sidebar
df = load_merged_titles()
filtered_df = get_filtered_data(df, filters)
```

## File Conventions
- Scripts: `01_clean_raw_data.py` (numbered, action verbs)
- Modules: `loaders.py`, `overlap_detector.py` (lowercase_with_underscores)
- Parquet files: `merged_titles.parquet`, `umap_coords.parquet` (descriptive)

## Development Flow
1. **Week 1**: Data pipeline (`scripts/01_clean_raw_data.py`, `02_merge_platforms.py`)
2. **Weeks 2-10**: Build artifacts as pages are developed (see `scripts/` for numbered pipeline)
3. **Week 11**: Integration, optimization, testing

## Key References
- **PROJECT_SPEC.md**: Detailed page-by-page requirements
- **PROGRESS_LOG.md**: Weekly progress tracking
- **BRAINSTORM.md**: Original comprehensive plan
