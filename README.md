# Netflix + Max Merger Analysis

An 8-page Streamlit application (Home + 7 content pages) analyzing a hypothetical Netflix + Warner Bros (Max) merger using data from 6 streaming platforms.

## Project Structure

```
streaming-merger/
├── Home.py                    # Entry point
├── pages/                     # 7 Streamlit pages
├── data/
│   ├── raw/                   # Original CSVs (12 files)
│   ├── interim/               # Cleaned parquet files
│   ├── processed/             # Merged datasets
│   └── precomputed/           # Heavy artifacts (embeddings, UMAP, etc.)
├── models/                    # Trained ML models
├── src/                       # Reusable Python package
├── scripts/                   # Data pipeline scripts
├── notebooks/                 # Jupyter exploration
└── tests/                     # Minimal testing
```

## Setup

### Prerequisites
- Python 3.9+
- pip or conda

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/therealrohan303/Streaming-Merge-Pro.git
cd Streaming-Merge-Pro
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate  # On Windows
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Data Pipeline

### Step 1: Clean Raw Data
```bash
python scripts/01_clean_raw_data.py
```
- Reads 12 raw CSVs from `data/raw/`
- Standardizes schema across platforms
- Handles missing values
- Outputs 12 cleaned parquet files to `data/interim/`

### Step 2: Merge Platforms
```bash
python scripts/02_merge_platforms.py
```
- Creates `data/processed/merged_titles.parquet` (Netflix + Max, ~9K rows)
- Creates `data/processed/all_platforms_titles.parquet` (all 6 platforms, ~25K rows)
- Does same for credits files

### Step 3: Run Full Pipeline (Optional)
```bash
bash scripts/run_pipeline.sh
```
- Runs all pipeline scripts in sequence
- Generates embeddings, similarity matrices, UMAP coordinates, etc.
- **Note:** Some scripts are built iteratively in Weeks 3-10

## Running the App

### Local Development
```bash
streamlit run Home.py
```
The app will open in your browser at `http://localhost:8501`

### Production Deployment
Deploy to Streamlit Cloud:
1. Push code to GitHub
2. Connect repository at [share.streamlit.io](https://share.streamlit.io)
3. Set `Home.py` as the main file

## Development Workflow

### Week 1: Foundation
- [ ] Create folder structure
- [ ] Implement data pipeline (`01_clean_raw_data.py`, `02_merge_platforms.py`)
- [ ] Build core utilities (`src/config.py`, `src/data/loaders.py`, `src/ui/filters.py`)
- [ ] Create basic `Home.py`

### Weeks 2-10: Page Development
Each week focuses on 1-2 pages:
- Week 2: Home dashboard
- Week 3: Explore Catalog
- Week 4: Platform Comparisons
- Week 5: Platform DNA
- Weeks 6-7: Discovery Engine
- Week 8: Strategic Insights
- Week 9: Interactive Lab
- Week 10: Cast & Crew Network

### Week 11: Integration & Optimization
- Consistent global filters
- Caching and performance tuning
- Bug fixes and polish

### Week 12: Deployment
- Documentation
- Deploy to Streamlit Cloud
- Presentation preparation

## Testing

Run tests with pytest:
```bash
pytest tests/
```

Tests include:
- `test_data_pipeline.py`: Data loading and merging
- `test_similarity.py`: TF-IDF and embeddings
- `test_analysis.py`: Overlap and gap detection

## Key Documentation

- **[CLAUDE.md](CLAUDE.md)**: Project rules, architecture, coding conventions
- **[PROJECT_SPEC.md](PROJECT_SPEC.md)**: Detailed page-by-page requirements
- **[BRAINSTORM.md](brainstorm.md)**: Original comprehensive plan
- **[PROGRESS_LOG.md](PROGRESS_LOG.md)**: Weekly progress tracking

## Platform Keys

Internal platform keys used throughout the project:
- `netflix`, `max`, `disney`, `prime`, `paramount`, `appletv`

## Data Sources

6 streaming platforms (mid-2023 snapshot):
- Netflix
- Max (Warner Bros Discovery)
- Disney+
- Prime Video (Amazon)
- Paramount+
- Apple TV+

**Disclaimer:** This is a hypothetical merger analysis for educational purposes. Data represents a mid-2023 snapshot.

## License

Educational project for capstone course.

## Contact

For questions or feedback, contact the project team.
