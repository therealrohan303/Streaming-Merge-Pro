"""Quality scoring using Bayesian averaging and popularity normalization."""

import pandas as pd


# Bayesian prior parameters
_PRIOR_VOTES = 10_000  # Weight of the prior (confidence threshold)
_PRIOR_MEAN = 6.5      # Global mean IMDb score (prior)

# Composite weights
_IMDB_WEIGHT = 0.70
_POPULARITY_WEIGHT = 0.30


def bayesian_imdb(imdb_score: pd.Series, imdb_votes: pd.Series) -> pd.Series:
    """Bayesian average that shrinks low-vote scores toward the global mean.

    Formula: (v / (v + m)) * R + (m / (v + m)) * C
      where v = votes, m = prior vote threshold, R = raw score, C = prior mean.
    """
    v = imdb_votes.fillna(0)
    r = imdb_score.fillna(_PRIOR_MEAN)
    return (v / (v + _PRIOR_VOTES)) * r + (_PRIOR_VOTES / (v + _PRIOR_VOTES)) * _PRIOR_MEAN


def normalize_popularity(tmdb_popularity: pd.Series) -> pd.Series:
    """Normalize TMDB popularity to a 0-10 scale using percentile-based capping."""
    pop = tmdb_popularity.fillna(0)
    cap = pop.quantile(0.99) if len(pop) > 0 else 1.0
    cap = max(cap, 1.0)  # avoid division by zero
    return (pop.clip(upper=cap) / cap) * 10.0


def compute_quality_score(df: pd.DataFrame) -> pd.Series:
    """Compute composite quality score: 70% Bayesian IMDb + 30% normalized popularity.

    Returns a Series aligned with df's index, values in roughly 0-10 range.
    """
    b_imdb = bayesian_imdb(df["imdb_score"], df["imdb_votes"])
    n_pop = normalize_popularity(df["tmdb_popularity"])
    return (_IMDB_WEIGHT * b_imdb + _POPULARITY_WEIGHT * n_pop).round(2)


def format_votes(votes) -> str:
    """Format vote count for display: 1234567 → '1.2M', 45000 → '45K'."""
    if pd.isna(votes) or votes == 0:
        return "N/A"
    if votes >= 1_000_000:
        return f"{votes / 1_000_000:.1f}M"
    if votes >= 1_000:
        return f"{votes / 1_000:.0f}K"
    return str(int(votes))
