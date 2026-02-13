"""
Central configuration for Netflix + Max Merger Analysis project.

This module defines all file paths, platform metadata, and ML hyperparameters
used throughout the application.
"""

from pathlib import Path

# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
PRECOMPUTED_DIR = DATA_DIR / "precomputed"
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"

# =============================================================================
# PLATFORM METADATA
# =============================================================================

PLATFORMS = {
    "netflix": {
        "name": "Netflix",
        "color": "#E50914",
        "text_color": "#FFFFFF",
    },
    "max": {
        "name": "Max",
        "color": "#002BE7",
        "text_color": "#FFFFFF",
    },
    "merged": {
        "name": "Netflix + Max",
        "color": "#7B1FA2",  # Purple blend
        "text_color": "#FFFFFF",
    },
    "disney": {
        "name": "Disney+",
        "color": "#113CCF",
        "text_color": "#FFFFFF",
    },
    "prime": {
        "name": "Prime Video",
        "color": "#00A8E1",
        "text_color": "#FFFFFF",
    },
    "paramount": {
        "name": "Paramount+",
        "color": "#0064FF",
        "text_color": "#FFFFFF",
    },
    "appletv": {
        "name": "Apple TV+",
        "color": "#000000",
        "text_color": "#FFFFFF",
    },
}

# Platform groups for analysis
MERGED_PLATFORMS = ["netflix", "max"]
ALL_PLATFORMS = ["netflix", "max", "disney", "prime", "paramount", "appletv"]
COMPETITOR_PLATFORMS = ["disney", "prime", "paramount", "appletv"]

# =============================================================================
# ML HYPERPARAMETERS
# =============================================================================

# TF-IDF Vectorizer
TFIDF_MAX_FEATURES = 5000
TFIDF_MIN_DF = 2  # Ignore terms that appear in < 2 documents
TFIDF_MAX_DF = 0.8  # Ignore terms that appear in > 80% of documents
TFIDF_NGRAM_RANGE = (1, 2)  # Unigrams and bigrams

# UMAP (Dimensionality Reduction)
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_N_COMPONENTS = 2  # 2D visualization
UMAP_METRIC = "cosine"
UMAP_RANDOM_STATE = 42

# Similarity Search
SIMILARITY_TOP_K = 10  # Number of similar titles to recommend
SIMILARITY_MIN_SCORE = 0.1  # Minimum similarity score threshold

# Embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Sentence-BERT model
EMBEDDING_DIM = 384  # Dimension of all-MiniLM-L6-v2

# =============================================================================
# UI CONFIGURATION
# =============================================================================

# Page configuration
PAGE_TITLE = "Netflix + Max Merger Analysis"
PAGE_ICON = "📊"
LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "expanded"

# Chart theme
PLOTLY_TEMPLATE = "plotly_dark"
CHART_HEIGHT = 400

# Card styling (dark theme)
CARD_BG = "#1E1E2E"
CARD_BORDER = "#333"
CARD_TEXT = "#ddd"
CARD_TEXT_MUTED = "#888"
CARD_ACCENT = "#FFD700"

# =============================================================================
# DATA QUALITY THRESHOLDS
# =============================================================================

MIN_IMDB_VOTES = 100  # Minimum votes for title to be considered
MIN_DESCRIPTION_LENGTH = 10  # Minimum description length (characters)
MAX_MISSING_FIELDS = 3  # Maximum missing fields before dropping a title

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

# Decade bins for temporal analysis
DECADE_BINS = [1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020, 2030]
DECADE_LABELS = ["1920s", "1930s", "1940s", "1950s", "1960s", "1970s",
                 "1980s", "1990s", "2000s", "2010s", "2020s"]

# Quality tiers based on IMDb score
QUALITY_TIERS = {
    "Excellent": (8.0, 10.0),
    "Good": (7.0, 8.0),
    "Average": (6.0, 7.0),
    "Below Average": (5.0, 6.0),
    "Poor": (0.0, 5.0),
}

# =============================================================================
# LOGGING
# =============================================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
