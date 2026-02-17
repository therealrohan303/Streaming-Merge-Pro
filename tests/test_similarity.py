"""Sanity checks for similarity lookup logic."""

import pandas as pd
import pytest

from src.analysis.similarity import get_similar_titles


@pytest.fixture
def sample_similarity_df():
    """Small synthetic similarity table."""
    return pd.DataFrame(
        {
            "source_id": ["tt001", "tt001", "tt001", "tt002", "tt002"],
            "similar_id": ["tt002", "tt003", "tt004", "tt001", "tt003"],
            "score": [0.85, 0.60, 0.35, 0.85, 0.70],
            "rank": [1, 2, 3, 1, 2],
        }
    )


class TestGetSimilarTitles:
    def test_returns_top_k_results(self, sample_titles_df, sample_similarity_df):
        result = get_similar_titles(
            title_id="tt001",
            similarity_df=sample_similarity_df,
            titles_df=sample_titles_df,
            top_k=2,
            min_score=0.1,
            min_imdb=0.0,
        )
        assert len(result) <= 2

    def test_filters_by_min_score(self, sample_titles_df, sample_similarity_df):
        result = get_similar_titles(
            title_id="tt001",
            similarity_df=sample_similarity_df,
            titles_df=sample_titles_df,
            min_score=0.5,
            min_imdb=0.0,
        )
        assert all(result["similarity_score"] >= 0.5)

    def test_respects_scope_filtering(self, sample_titles_df, sample_similarity_df):
        netflix_only = sample_titles_df[sample_titles_df["platform"] == "netflix"]
        result = get_similar_titles(
            title_id="tt001",
            similarity_df=sample_similarity_df,
            titles_df=netflix_only,
            min_score=0.1,
            min_imdb=0.0,
        )
        if not result.empty:
            assert all(
                all(p == "netflix" for p in platforms)
                for platforms in result["platforms"]
            )

    def test_missing_id_returns_empty(self, sample_titles_df, sample_similarity_df):
        result = get_similar_titles(
            title_id="tt_nonexistent",
            similarity_df=sample_similarity_df,
            titles_df=sample_titles_df,
            min_imdb=0.0,
        )
        assert result.empty

    def test_results_sorted_by_score(self, sample_titles_df, sample_similarity_df):
        result = get_similar_titles(
            title_id="tt001",
            similarity_df=sample_similarity_df,
            titles_df=sample_titles_df,
            min_score=0.1,
            min_imdb=0.0,
        )
        if len(result) >= 2:
            scores = result["similarity_score"].tolist()
            assert scores == sorted(scores, reverse=True)

    def test_excludes_self(self, sample_titles_df, sample_similarity_df):
        result = get_similar_titles(
            title_id="tt001",
            similarity_df=sample_similarity_df,
            titles_df=sample_titles_df,
            min_score=0.1,
            min_imdb=0.0,
        )
        assert "tt001" not in result["similar_id"].values

    def test_filters_by_min_imdb(self, sample_titles_df, sample_similarity_df):
        result = get_similar_titles(
            title_id="tt001",
            similarity_df=sample_similarity_df,
            titles_df=sample_titles_df,
            min_score=0.1,
            min_imdb=7.0,
        )
        if not result.empty:
            assert all(result["imdb_score"] >= 7.0)
