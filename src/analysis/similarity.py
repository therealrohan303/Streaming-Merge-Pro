"""Similarity lookup for title recommendations using precomputed multi-signal data."""

import pandas as pd

from src.config import SIMILARITY_MIN_IMDB, SIMILARITY_MIN_SCORE, SIMILARITY_TOP_K
from src.data.loaders import deduplicate_titles


def get_similar_titles(
    title_id: str,
    similarity_df: pd.DataFrame,
    titles_df: pd.DataFrame,
    top_k: int = SIMILARITY_TOP_K,
    min_score: float = SIMILARITY_MIN_SCORE,
    min_imdb: float = SIMILARITY_MIN_IMDB,
    min_votes: int = 0,
) -> pd.DataFrame:
    """Look up the top-K most similar titles for a given title ID.

    Args:
        title_id: The source title's ``id`` value.
        similarity_df: Precomputed similarity table with columns
            ``source_id``, ``similar_id``, ``score``, ``rank``.
        titles_df: Titles DataFrame that defines the result pool.
            Pass merged titles for merged scope, all-platforms for full scope.
        top_k: Maximum number of results to return.
        min_score: Minimum similarity score threshold.
        min_imdb: Minimum IMDb score for recommended titles.
        min_votes: Minimum IMDb vote count; 0 means no filter.

    Returns:
        DataFrame with title metadata plus ``similarity_score`` and
        ``platforms`` (list of platform keys),
        sorted by similarity_score descending and limited to *top_k* rows.
        Returns an empty DataFrame when no matches are found.
    """
    # Filter similarity rows for the source title
    matches = similarity_df[similarity_df["source_id"] == title_id]
    if matches.empty:
        return pd.DataFrame()

    # Apply minimum score threshold
    matches = matches[matches["score"] >= min_score]

    # Restrict to titles present in the scoped pool
    available_ids = set(titles_df["id"].unique())
    matches = matches[matches["similar_id"].isin(available_ids)]

    # Deduplicate titles, aggregating platforms into a list
    titles_deduped = deduplicate_titles(titles_df)

    # Join with title metadata
    result = matches.merge(
        titles_deduped,
        left_on="similar_id",
        right_on="id",
        how="inner",
    )

    # Filter out low-quality recommendations
    if "imdb_score" in result.columns:
        result = result[result["imdb_score"] >= min_imdb]
    if min_votes and "imdb_votes" in result.columns:
        result = result[result["imdb_votes"] >= min_votes]

    # Take top-K by score
    result = result.nlargest(top_k, "score")

    result = result.rename(columns={"score": "similarity_score"})

    keep_cols = [
        "similar_id",
        "similarity_score",
        "title",
        "type",
        "release_year",
        "genres",
        "imdb_score",
        "imdb_votes",
        "tmdb_popularity",
        "description",
        "platforms",
    ]
    keep_cols = [c for c in keep_cols if c in result.columns]
    return result[keep_cols].reset_index(drop=True)
