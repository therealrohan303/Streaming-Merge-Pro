"""Analysis functions for the Discovery Engine page (Page 4).

Three recommendation modes:
  1. Similar to a title (enhanced similarity with why-similar explainer)
  2. Preference-based (genre, quality, type, runtime, year, popularity)
  3. Vibe search (NLP semantic + keyword matching + genome hybrid)
"""

import numpy as np
import pandas as pd

from src.analysis.scoring import bayesian_imdb, compute_quality_score
from src.config import (
    HYBRID_AWARDS_WEIGHT,
    HYBRID_DESC_WEIGHT,
    HYBRID_GENOME_WEIGHT,
    HYBRID_GENRE_WEIGHT,
    HYBRID_QUALITY_WEIGHT,
)


def get_similar_with_explanation(
    title_id, titles_df, sim_df, credits_df=None, principals_df=None,
    enriched_df=None, scope="merged", top_k=10, min_imdb=6.0,
):
    """Get similar titles with detailed 'why similar?' explanation.

    Returns list of dicts with similarity_score + explanation fields.
    """
    from src.analysis.similarity import get_similar_titles
    from src.data.loaders import deduplicate_titles

    similar = get_similar_titles(
        title_id, sim_df, titles_df,
        top_k=top_k, min_imdb=min_imdb,
    )

    if similar.empty:
        return []

    # Get source title info
    source = titles_df[titles_df["id"] == title_id]
    if source.empty:
        return similar.to_dict("records")

    source_row = source.iloc[0]
    source_genres = set(source_row.get("genres", []) or [])

    results = []
    for _, row in similar.iterrows():
        explanation = {}

        # Genre overlap
        target_genres = set(row.get("genres", []) or [])
        shared_genres = source_genres & target_genres
        if shared_genres:
            explanation["genre_overlap"] = sorted(shared_genres)

        # Vibe tags (from MovieLens top_tags)
        if enriched_df is not None and "top_tags" in enriched_df.columns:
            source_enriched = enriched_df[enriched_df["id"] == title_id]
            target_enriched = enriched_df[enriched_df["id"] == row.get("id")]
            if not source_enriched.empty and not target_enriched.empty:
                source_tags = set(source_enriched.iloc[0].get("top_tags") or [])
                target_tags = set(target_enriched.iloc[0].get("top_tags") or [])
                shared_tags = source_tags & target_tags
                if shared_tags:
                    explanation["matched_vibe_tags"] = sorted(list(shared_tags)[:5])

        # Shared crew (from IMDb principals)
        if principals_df is not None and not principals_df.empty:
            source_imdb = source_row.get("imdb_id")
            target_imdb = row.get("imdb_id")
            if source_imdb and target_imdb:
                source_crew = principals_df[principals_df["imdb_id"] == source_imdb]
                target_crew = principals_df[principals_df["imdb_id"] == target_imdb]
                shared_people = set(source_crew["person_id"]) & set(target_crew["person_id"])
                if shared_people:
                    shared_names = (
                        principals_df[principals_df["person_id"].isin(shared_people)]
                        .drop_duplicates("person_id")[["name", "category"]]
                        .head(3)
                    )
                    explanation["shared_crew"] = [
                        {"name": r["name"], "role": r["category"]}
                        for _, r in shared_names.iterrows()
                    ]

        result = row.to_dict()
        result["explanation"] = explanation
        results.append(result)

    return results


def preference_based_recommendations(
    titles_df, genres=None, min_imdb=6.0, content_type="Both",
    min_runtime=None, max_runtime=None, year_range=None,
    popularity_weight=0.5, scope="merged", top_k=20,
):
    """Generate recommendations based on user preferences.

    Args:
        popularity_weight: 0.0 = hidden gems, 1.0 = blockbusters
    """
    df = titles_df.copy()

    # Apply filters
    if content_type != "Both":
        df = df[df["type"] == content_type]

    if genres:
        df = df[df["genres"].apply(
            lambda g: bool(set(g or []) & set(genres)) if isinstance(g, (list, np.ndarray)) else False
        )]

    if min_imdb:
        df = df[df["imdb_score"] >= min_imdb]

    if min_runtime is not None and "runtime" in df.columns:
        runtime = pd.to_numeric(df["runtime"], errors="coerce")
        df = df[runtime >= min_runtime]

    if max_runtime is not None and "runtime" in df.columns:
        runtime = pd.to_numeric(df["runtime"], errors="coerce")
        df = df[runtime <= max_runtime]

    if year_range:
        df = df[(df["release_year"] >= year_range[0]) & (df["release_year"] <= year_range[1])]

    if df.empty:
        return pd.DataFrame()

    # Compute fit score
    df["quality_score"] = compute_quality_score(df)

    # Genre match score (how many of the user's genres match)
    if genres:
        df["genre_match"] = df["genres"].apply(
            lambda g: len(set(g or []) & set(genres)) / len(genres)
            if isinstance(g, (list, np.ndarray)) and genres else 0
        )
    else:
        df["genre_match"] = 1.0

    # Popularity adjustment
    if "tmdb_popularity" in df.columns:
        pop = df["tmdb_popularity"].fillna(0)
        pop_norm = (pop - pop.min()) / (pop.max() - pop.min() + 1e-9)
        if popularity_weight < 0.5:
            # Prefer hidden gems (inverse popularity)
            pop_score = 1.0 - pop_norm
        else:
            pop_score = pop_norm
        pop_factor = 0.3 * abs(popularity_weight - 0.5) * 2  # Scale 0-0.3
    else:
        pop_score = 0.5
        pop_factor = 0

    # Combined fit score
    df["fit_score"] = (
        0.4 * df["quality_score"] +
        0.3 * df["genre_match"] +
        pop_factor * pop_score +
        (0.3 - pop_factor) * df.get("quality_score", 0.5)
    )

    # Generate why text
    def why_text(row):
        reasons = []
        if genres and row.get("genre_match", 0) > 0:
            matched = set(row.get("genres", []) or []) & set(genres)
            if matched:
                reasons.append(f"Matches genres: {', '.join(sorted(matched))}")
        if row.get("imdb_score") and row["imdb_score"] >= 7.5:
            reasons.append(f"Highly rated ({row['imdb_score']:.1f} IMDb)")
        if popularity_weight < 0.3 and row.get("imdb_votes", 0) < 50000:
            reasons.append("Hidden gem with fewer votes")
        elif popularity_weight > 0.7 and row.get("tmdb_popularity", 0) > 50:
            reasons.append("Popular and trending")
        return "; ".join(reasons) if reasons else "Good overall match"

    df["why_match"] = df.apply(why_text, axis=1)

    return df.nlargest(top_k, "fit_score")


def extract_vibe_signals(query_text, tag_names=None, keyword_list=None):
    """Extract detected thematic signals from user's vibe query.

    Returns list of detected theme strings.
    """
    query_lower = query_text.lower()
    signals = []

    # Genre-related patterns
    genre_patterns = {
        "thriller": ["thriller", "suspense", "tense", "edge of seat"],
        "comedy": ["funny", "comedy", "hilarious", "laugh", "humor"],
        "drama": ["drama", "dramatic", "emotional", "moving"],
        "horror": ["horror", "scary", "creepy", "terrifying", "spooky"],
        "romance": ["romance", "love story", "romantic", "love"],
        "sci-fi": ["sci-fi", "science fiction", "futuristic", "space", "alien"],
        "action": ["action", "explosive", "adventure", "fighting"],
        "documentary": ["documentary", "real story", "true story", "non-fiction"],
        "animation": ["animated", "animation", "cartoon", "anime"],
        "mystery": ["mystery", "detective", "whodunit", "puzzle"],
    }
    for theme, patterns in genre_patterns.items():
        if any(p in query_lower for p in patterns):
            signals.append(theme)

    # Mood/atmosphere patterns
    mood_patterns = {
        "atmospheric": ["atmospheric", "moody", "brooding", "dark"],
        "psychological": ["psychological", "mind-bending", "cerebral", "thought-provoking"],
        "feel-good": ["feel-good", "uplifting", "heartwarming", "wholesome"],
        "intense": ["intense", "gripping", "gritty", "raw"],
        "nostalgic": ["nostalgic", "retro", "classic", "old-school"],
        "slow-burn": ["slow-burn", "slow burn", "patient", "meditative"],
        "twist ending": ["twist", "surprise ending", "unexpected"],
        "visually stunning": ["visual", "beautiful", "cinematography", "stunning"],
    }
    for mood, patterns in mood_patterns.items():
        if any(p in query_lower for p in patterns):
            signals.append(mood)

    # Match against tag names from MovieLens genome
    if tag_names:
        for tag in tag_names:
            if tag.lower() in query_lower and tag.lower() not in [s.lower() for s in signals]:
                signals.append(tag)

    return signals[:8]  # Cap at 8 signals


def vibe_search(
    query_text, titles_df, desc_embeddings=None, desc_id_map=None,
    genome_vectors=None, genome_id_map=None, enriched_df=None,
    embed_model=None, scope="merged", top_k=15, min_imdb=None, year_range=None,
):
    """Hybrid vibe search combining semantic + keyword + genome + quality + awards.

    Weights: 0.35 desc embedding + 0.25 genre + 0.15 genome + 0.15 quality + 0.10 awards
    For titles without genome data, redistribute genome weight to desc + genre.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    df = titles_df.copy()

    # Apply optional filters
    if min_imdb:
        df = df[df["imdb_score"] >= min_imdb]
    if year_range:
        df = df[(df["release_year"] >= year_range[0]) & (df["release_year"] <= year_range[1])]

    if df.empty:
        return pd.DataFrame(), []

    # Extract signals
    tag_names = None
    if genome_id_map is not None:
        # Could load tag names here if needed
        pass
    signals = extract_vibe_signals(query_text, tag_names)

    # Compute quality scores
    df["quality_score"] = compute_quality_score(df)

    # Initialize score columns
    df["desc_score"] = 0.0
    df["genre_score"] = 0.0
    df["genome_score"] = 0.0
    df["quality_norm"] = 0.0
    df["awards_score"] = 0.0

    # 1. Description embedding similarity
    if desc_embeddings is not None and embed_model is not None and desc_id_map is not None:
        query_embedding = embed_model.encode([query_text])
        sims = cosine_similarity(query_embedding, desc_embeddings)[0]
        # Map to dataframe
        id_to_sim = dict(zip(desc_id_map["id"], sims))
        df["desc_score"] = df["id"].map(id_to_sim).fillna(0)
    else:
        # Fallback: simple keyword matching on description
        query_words = set(query_text.lower().split())
        df["desc_score"] = df["description"].fillna("").apply(
            lambda d: len(query_words & set(d.lower().split())) / max(len(query_words), 1)
        )

    # 2. Genre matching from signals
    genre_signals = [s for s in signals if s in [
        "thriller", "comedy", "drama", "horror", "romance",
        "sci-fi", "action", "documentary", "animation", "mystery",
    ]]
    if genre_signals:
        df["genre_score"] = df["genres"].apply(
            lambda g: len(set(g or []) & set(genre_signals)) / len(genre_signals)
            if isinstance(g, (list, np.ndarray)) else 0
        )

    # 3. Genome vector similarity (movies only)
    if genome_vectors is not None and genome_id_map is not None:
        genome_imdb_set = set(genome_id_map["imdb_id"])
        # Simple keyword matching against genome tags for now
        # Full embedding similarity would require query -> genome vector mapping
        if enriched_df is not None and "top_tags" in enriched_df.columns:
            query_words_lower = set(query_text.lower().split())
            tag_scores = {}
            for _, row in enriched_df[enriched_df["top_tags"].notna()].iterrows():
                tags = row["top_tags"]
                if isinstance(tags, (list, np.ndarray)):
                    tag_match = sum(1 for t in tags if any(w in t.lower() for w in query_words_lower))
                    tag_scores[row["id"]] = tag_match / max(len(tags), 1)
            if tag_scores:
                df["genome_score"] = df["id"].map(tag_scores).fillna(0)

    # 4. Keyword matching (tmdb_keywords + top_tags)
    if enriched_df is not None:
        query_words_lower = set(query_text.lower().split())
        keyword_scores = {}
        for _, row in enriched_df.iterrows():
            score = 0
            total_kw = 0
            if "tmdb_keywords" in enriched_df.columns:
                kws = row.get("tmdb_keywords")
                if isinstance(kws, (list, np.ndarray)):
                    total_kw += len(kws)
                    score += sum(1 for k in kws if any(w in k.lower() for w in query_words_lower))
            if total_kw > 0:
                keyword_scores[row["id"]] = score / total_kw
        if keyword_scores:
            kw_series = df["id"].map(keyword_scores).fillna(0)
            df["genre_score"] = (df["genre_score"] + kw_series) / 2

    # 5. Quality score (normalized 0-1)
    if "quality_score" in df.columns:
        qs = df["quality_score"]
        df["quality_norm"] = (qs - qs.min()) / (qs.max() - qs.min() + 1e-9)

    # 6. Awards boost
    if enriched_df is not None and "award_wins" in enriched_df.columns:
        award_map = enriched_df.set_index("id")["award_wins"].to_dict()
        awards = df["id"].map(award_map).fillna(0)
        max_awards = awards.max() if awards.max() > 0 else 1
        df["awards_score"] = awards / max_awards

    # Compute hybrid score with weight redistribution
    has_genome = df["genome_score"] > 0
    # For titles WITH genome data: standard weights
    # For titles WITHOUT genome data: redistribute genome weight
    desc_w = HYBRID_DESC_WEIGHT
    genre_w = HYBRID_GENRE_WEIGHT
    genome_w = HYBRID_GENOME_WEIGHT
    quality_w = HYBRID_QUALITY_WEIGHT
    awards_w = HYBRID_AWARDS_WEIGHT

    # Titles with genome
    df.loc[has_genome, "vibe_score"] = (
        desc_w * df.loc[has_genome, "desc_score"] +
        genre_w * df.loc[has_genome, "genre_score"] +
        genome_w * df.loc[has_genome, "genome_score"] +
        quality_w * df.loc[has_genome, "quality_norm"] +
        awards_w * df.loc[has_genome, "awards_score"]
    )
    # Titles without genome: redistribute
    redistrib_desc = desc_w + genome_w * 0.6
    redistrib_genre = genre_w + genome_w * 0.4
    df.loc[~has_genome, "vibe_score"] = (
        redistrib_desc * df.loc[~has_genome, "desc_score"] +
        redistrib_genre * df.loc[~has_genome, "genre_score"] +
        quality_w * df.loc[~has_genome, "quality_norm"] +
        awards_w * df.loc[~has_genome, "awards_score"]
    )

    result = df.nlargest(top_k, "vibe_score")
    return result, signals
