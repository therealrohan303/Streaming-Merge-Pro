"""Tests for enrichment pipeline outputs."""

import numpy as np
import pandas as pd
import pytest

from src.config import ENRICHED_DIR, PRECOMPUTED_DIR

REQUIRED_ENRICHED_COLUMNS = [
    # Base columns
    "id", "title", "platform", "type", "release_year", "genres",
    "imdb_score", "imdb_id", "description",
    # Enrichment columns (nullable)
    "original_title", "is_adult",
    "budget_usd", "box_office_usd", "award_wins", "award_noms",
    "top_tags", "ml_avg_rating", "ml_rating_count",
    "tmdb_keywords", "collection_name", "production_companies", "poster_url",
    "imdb_writers", "imdb_producers",
    "data_confidence",
]

PRINCIPAL_CATEGORIES = {"director", "writer", "producer", "composer", "cinematographer"}


class TestTitlesEnriched:
    """Validate titles_enriched.parquet schema and data quality."""

    @pytest.fixture(autouse=True)
    def load_data(self):
        path = ENRICHED_DIR / "titles_enriched.parquet"
        if not path.exists():
            pytest.skip("titles_enriched.parquet not found")
        self.df = pd.read_parquet(path)

    def test_has_required_columns(self):
        missing = [col for col in REQUIRED_ENRICHED_COLUMNS if col not in self.df.columns]
        assert not missing, f"Missing columns: {missing}"

    def test_data_confidence_range(self):
        assert self.df["data_confidence"].between(0.0, 1.0).all(), \
            "data_confidence must be between 0.0 and 1.0"

    def test_data_confidence_not_all_zero(self):
        assert self.df["data_confidence"].sum() > 0, \
            "data_confidence should have some non-zero values"

    def test_row_count_matches_base(self):
        from src.config import PROCESSED_DIR
        base = pd.read_parquet(PROCESSED_DIR / "all_platforms_titles.parquet")
        assert len(self.df) == len(base), \
            f"Enriched ({len(self.df)}) should match base ({len(base)}) row count"

    def test_imdb_id_preserved(self):
        non_null = self.df["imdb_id"].notna().sum()
        assert non_null > 20000, f"Expected >20K imdb_ids, got {non_null}"

    def test_some_enrichment_present(self):
        """At least some enrichment data should be present."""
        enrichment_cols = ["award_wins", "budget_usd", "top_tags", "tmdb_keywords"]
        present = {col: self.df[col].notna().sum()
                   for col in enrichment_cols if col in self.df.columns}
        total_enriched = sum(present.values())
        assert total_enriched > 0, f"No enrichment data found: {present}"


class TestImdbPrincipals:
    """Validate imdb_principals.parquet."""

    @pytest.fixture(autouse=True)
    def load_data(self):
        path = ENRICHED_DIR / "imdb_principals.parquet"
        if not path.exists():
            pytest.skip("imdb_principals.parquet not found")
        self.df = pd.read_parquet(path)

    def test_has_required_columns(self):
        required = ["imdb_id", "person_id", "name", "category", "job"]
        missing = [col for col in required if col not in self.df.columns]
        assert not missing, f"Missing columns: {missing}"

    def test_has_all_five_categories(self):
        categories = set(self.df["category"].unique())
        missing = PRINCIPAL_CATEGORIES - categories
        assert not missing, f"Missing categories: {missing}"

    def test_minimum_row_count(self):
        assert len(self.df) > 1000, f"Expected >1000 principal rows, got {len(self.df)}"

    def test_names_mostly_present(self):
        name_coverage = self.df["name"].notna().mean()
        assert name_coverage > 0.9, f"Name coverage {name_coverage:.1%} is too low"


class TestGenomeVectors:
    """Validate genome vectors shape and id_map."""

    @pytest.fixture(autouse=True)
    def load_data(self):
        vectors_path = PRECOMPUTED_DIR / "embeddings" / "genome_vectors.npy"
        id_map_path = PRECOMPUTED_DIR / "embeddings" / "genome_id_map.parquet"
        if not vectors_path.exists() or not id_map_path.exists():
            pytest.skip("Genome vectors not found")
        self.vectors = np.load(vectors_path)
        self.id_map = pd.read_parquet(id_map_path)

    def test_shape_matches(self):
        assert self.vectors.shape[0] == len(self.id_map), \
            f"Vectors rows ({self.vectors.shape[0]}) != id_map rows ({len(self.id_map)})"

    def test_id_map_has_imdb_id(self):
        assert "imdb_id" in self.id_map.columns

    def test_vectors_not_all_zero(self):
        assert self.vectors.sum() > 0, "Genome vectors should not be all zeros"

    def test_vectors_reasonable_shape(self):
        assert self.vectors.shape[1] > 100, \
            f"Expected >100 tag dimensions, got {self.vectors.shape[1]}"
