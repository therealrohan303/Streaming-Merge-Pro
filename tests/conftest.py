"""Shared pytest fixtures for the Streaming Merger test suite."""

import pandas as pd
import pytest


@pytest.fixture
def sample_titles_df():
    """Small synthetic titles DataFrame for unit tests."""
    return pd.DataFrame(
        {
            "id": ["tt001", "tt002", "tt003", "tt004"],
            "title": [
                "Action Movie",
                "Comedy Show",
                "Drama Series",
                "Sci-Fi Film",
            ],
            "type": ["MOVIE", "SHOW", "SHOW", "MOVIE"],
            "release_year": [2020, 2019, 2021, 2022],
            "genres": ["action,thriller", "comedy", "drama", "scifi,action"],
            "imdb_score": [7.5, 6.8, 8.2, 7.0],
            "imdb_votes": [15000, 8000, 22000, 12000],
            "description": [
                "An action-packed adventure.",
                "A hilarious comedy series.",
                "A gripping drama about family.",
                "A journey through space and time.",
            ],
            "platform": ["netflix", "netflix", "max", "max"],
        }
    )


@pytest.fixture
def sample_credits_df():
    """Small synthetic credits DataFrame for unit tests."""
    return pd.DataFrame(
        {
            "person_id": [1, 2, 3, 1, 4],
            "id": ["tt001", "tt001", "tt002", "tt003", "tt003"],
            "name": [
                "Alice Smith",
                "Bob Jones",
                "Carol White",
                "Alice Smith",
                "Dan Brown",
            ],
            "role": ["ACTOR", "DIRECTOR", "ACTOR", "ACTOR", "DIRECTOR"],
            "character": ["Hero", None, "Lead", "Villain", None],
            "platform": ["netflix", "netflix", "netflix", "max", "max"],
        }
    )


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Temporary directory mimicking the project data layout."""
    for subdir in ("raw", "interim", "processed", "precomputed"):
        (tmp_path / subdir).mkdir()
    return tmp_path
