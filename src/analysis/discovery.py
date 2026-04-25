"""Analysis functions for the Discovery Engine page (Page 4).

Four recommendation modes:
  1. Similar to a title (enhanced similarity with why-similar explainer)
  2. Mood Board (mood-tile based matching via genome tags + TMDB keywords)
  3. Vibe search (NLP semantic + keyword matching + genome hybrid)
  4. Preference-based (legacy, kept for reference)
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

# ─── Mood Board Configuration ────────────────────────────────────────────────

MOOD_TILES = [
    {
        "emoji": "⚡", "label": "Edge of your seat",
        "genome_tags": ["suspenseful", "tense", "nail-biting", "gripping", "thrilling"],
        "tmdb_kws": ["suspense", "thriller", "tension", "thrill"],
        "genres": ["thriller", "horror", "crime", "mystery"],
    },
    {
        "emoji": "😢", "label": "Ugly cry",
        "genome_tags": ["tearjerker", "emotional", "moving", "heart wrenching", "sad", "touching"],
        "tmdb_kws": ["emotional", "tragedy", "tearjerker", "grief"],
        "genres": ["drama", "romance"],
    },
    {
        "emoji": "😊", "label": "Feel-good and warm",
        "genome_tags": ["feel-good", "heartwarming", "uplifting", "optimistic", "cheerful", "warm"],
        "tmdb_kws": ["feel-good", "heartwarming", "feel good", "uplifting", "feel good film"],
        "genres": ["comedy", "animation", "family"],
    },
    {
        "emoji": "🌀", "label": "Mind-bending",
        "genome_tags": ["mind bending", "psychological", "surreal", "thought provoking", "cerebral", "complex narrative"],
        "tmdb_kws": ["mindbender", "psychological", "mind-bender", "nonlinear narrative"],
        "genres": ["thriller", "sci-fi", "mystery"],
    },
    {
        "emoji": "🏔️", "label": "Epic adventure",
        "genome_tags": ["epic", "adventure", "grand", "spectacular", "journey", "quest"],
        "tmdb_kws": ["epic", "adventure", "quest", "epic journey"],
        "genres": ["action", "adventure", "fantasy", "sci-fi"],
    },
    {
        "emoji": "🌑", "label": "Dark and unsettling",
        "genome_tags": ["dark", "disturbing", "unsettling", "bleak", "gritty", "sinister", "dark themes"],
        "tmdb_kws": ["dark", "disturbing", "psychological horror", "nihilism", "bleak", "grim"],
        "genres": ["horror", "thriller", "crime", "drama"],
    },
    {
        "emoji": "😂", "label": "Laugh out loud",
        "genome_tags": ["funny", "hilarious", "comedy", "humor", "witty", "slapstick", "satirical"],
        "tmdb_kws": ["comedy", "humor", "funny", "satire", "parody"],
        "genres": ["comedy", "animation"],
    },
    {
        "emoji": "🕯️", "label": "Slow burn",
        "genome_tags": ["slow burn", "slow-burn", "atmospheric", "meditative", "deliberate pacing"],
        "tmdb_kws": ["slow burn", "atmospheric", "slow paced"],
        "genres": ["drama", "thriller", "mystery"],
    },
    {
        "emoji": "🏆", "label": "Inspiring true story",
        "genome_tags": ["inspirational", "based on true story", "uplifting", "real events", "biography"],
        "tmdb_kws": ["based on true story", "biography", "inspirational", "based on a true story"],
        "genres": ["drama", "documentation"],
    },
    {
        "emoji": "📼", "label": "Nostalgia trip",
        "genome_tags": ["nostalgic", "retro", "classic", "old school", "period piece"],
        "tmdb_kws": ["nostalgia", "retro", "80s", "90s", "period piece"],
        "genres": ["drama", "comedy", "romance"],
    },
    {
        "emoji": "❤️", "label": "Love story",
        "genome_tags": ["romance", "love story", "romantic", "relationship", "love"],
        "tmdb_kws": ["romance", "love story", "romantic", "love", "romantic comedy"],
        "genres": ["romance", "drama"],
    },
    {
        "emoji": "🔍", "label": "Mystery unraveling",
        "genome_tags": ["mystery", "whodunit", "detective", "puzzle", "twists", "investigation"],
        "tmdb_kws": ["mystery", "whodunit", "detective", "investigation"],
        "genres": ["mystery", "thriller", "crime"],
    },
    {
        "emoji": "💥", "label": "Action-packed",
        "genome_tags": ["action", "exciting", "fast paced", "explosive", "high octane"],
        "tmdb_kws": ["action", "action hero", "fight", "action thriller"],
        "genres": ["action", "thriller", "adventure"],
    },
    {
        "emoji": "🧠", "label": "Thought-provoking",
        "genome_tags": ["thought provoking", "philosophical", "deep", "meaningful", "social commentary"],
        "tmdb_kws": ["philosophical", "thought-provoking", "social commentary", "meditation"],
        "genres": ["drama", "sci-fi", "documentation"],
    },
    {
        "emoji": "👨‍👩‍👧", "label": "Family night",
        "genome_tags": ["family", "family friendly", "wholesome", "for kids", "all ages"],
        "tmdb_kws": ["family", "family film", "animation", "family friendly"],
        "genres": ["family", "animation", "comedy"],
    },
    {
        "emoji": "💎", "label": "Hidden gem",
        "genome_tags": ["underrated", "overlooked", "cult classic", "hidden gem", "cult"],
        "tmdb_kws": ["cult", "underrated", "cult film"],
        "genres": ["drama", "comedy", "thriller"],
    },
]

# Build a lookup dict for fast access by label
MOOD_TILE_BY_LABEL = {t["label"]: t for t in MOOD_TILES}

# Per-mood genre compatibility: titles with incompatible primary genres and no required genres
# receive a heavy penalty so e.g. Vikings doesn't appear in "Ugly cry" results
_MOOD_GENRE_COMPAT = {
    "Ugly cry": {
        "required_any": {"drama", "romance"},
        "incompatible": {"action", "war", "western", "sport", "adventure"},
    },
    "Love story": {
        "required_any": {"romance", "drama"},
        "incompatible": {"action", "war", "horror", "sci-fi", "sport"},
    },
    "Feel-good and warm": {
        "required_any": {"comedy", "animation", "family", "drama"},
        "incompatible": {"horror", "crime", "war"},
    },
}
_MOOD_GENRE_PENALTY = 0.2


def get_similar_with_explanation(
    title_id, titles_df, sim_df, credits_df=None, principals_df=None,
    enriched_df=None, scope="merged", top_k=10, min_imdb=6.0, min_votes=0,
):
    """Get similar titles with detailed 'why similar?' explanation.

    Returns list of dicts with similarity_score + explanation fields.
    """
    from src.analysis.similarity import get_similar_titles
    from src.data.loaders import deduplicate_titles

    similar = get_similar_titles(
        title_id, sim_df, titles_df,
        top_k=top_k, min_imdb=min_imdb, min_votes=min_votes,
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

    # Rescale similarity_score to [0.55, 0.90] so ranking is visually meaningful
    if len(results) > 1:
        scores = [r["similarity_score"] for r in results]
        lo, hi = min(scores), max(scores)
        span = hi - lo
        for r in results:
            if span > 1e-6:
                norm = (r["similarity_score"] - lo) / span
                r["similarity_score"] = 0.55 + norm * 0.35
            else:
                r["similarity_score"] = 0.72
    elif len(results) == 1:
        results[0]["similarity_score"] = 0.80

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
        "emotional": [
            "sad", "devastating", "heartbreaking", "grief", "tragedy", "tragic",
            "loss", "mourning", "depressing", "melancholic", "tearjerker",
            "gut-wrenching", "weep", "sorrowful", "sobbing", "devastating",
        ],
        "tense": ["tense", "tension", "anxiety", "dread", "foreboding"],
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
    min_votes=0,
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
    if min_votes and "imdb_votes" in df.columns:
        df = df[df["imdb_votes"] >= min_votes]
    if year_range:
        df = df[(df["release_year"] >= year_range[0]) & (df["release_year"] <= year_range[1])]

    # Tonal pre-filter: hard-exclude comedy/animation/music for emotional-distress queries
    # Checks ALL genres (not just primary) to catch stand-up specials in any genre order
    _DISTRESS_WORDS = {
        "sad", "devastating", "heartbreaking", "grief", "tragedy", "tragic",
        "loss", "mourning", "depressing", "melancholic",
    }
    _TONAL_EXCL_GENRES = {"comedy", "animation", "family", "talk-show", "reality", "music"}
    if set(query_text.lower().split()) & _DISTRESS_WORDS:
        def _has_excl_genre(g):
            gl = list(g) if isinstance(g, np.ndarray) else (g if isinstance(g, (list, tuple)) else [])
            return any(str(x).lower() in _TONAL_EXCL_GENRES for x in gl)
        df = df[~df["genres"].apply(_has_excl_genre)]

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
    # Emotional tone signal → boost drama/romance candidates
    if "emotional" in signals:
        for _g in ["drama", "romance"]:
            if _g not in genre_signals:
                genre_signals.append(_g)
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

    # Tone-based description re-ranking: boost sad/tragic descriptions, penalise comedy ones
    _SAD_DESC_KW = {
        "dies", "death", "loss", "grief", "tragedy", "devastating",
        "heartbreak", "sacrifice", "mourning", "sorrow", "tragic",
    }
    _COMEDY_DESC_KW = {"comedy", "funny", "laugh", "hilarious", "jokes", "comedian", "standup", "stand-up"}
    if set(query_text.lower().split()) & _DISTRESS_WORDS:
        def _desc_tone_mult(desc):
            ds = set(str(desc).lower().split())
            if ds & _SAD_DESC_KW:
                return 1.20
            if ds & _COMEDY_DESC_KW:
                return 0.50
            return 1.0
        df["vibe_score"] = df["vibe_score"] * df["description"].fillna("").apply(_desc_tone_mult)

    result = df.nlargest(top_k, "vibe_score")
    return result, signals


def mood_board_recommendations(
    titles_df, selected_mood_labels, content_type="Both",
    top_k=20, enriched_df=None, min_imdb=6.0, min_votes=0,
):
    """Match titles to selected mood tiles using genome tags, TMDB keywords, and quality.

    Args:
        selected_mood_labels: list of mood label strings (from MOOD_TILES)
        content_type: "Both", "Movie", or "Show"
        enriched_df: titles_enriched.parquet with top_tags and tmdb_keywords columns
        min_imdb: minimum IMDb score filter (default 6.0)
        min_votes: minimum IMDb vote count; 0 means no filter

    Returns:
        DataFrame with mood_match_pct (0–1), matched_moods (list), mood_score columns.
    """
    if not selected_mood_labels:
        return pd.DataFrame()

    df = titles_df.copy()

    # Content type filter
    if content_type != "Both":
        df = df[df["type"] == content_type]

    # Quality filters
    if min_imdb and "imdb_score" in df.columns:
        df = df[df["imdb_score"] >= min_imdb]
    if min_votes and "imdb_votes" in df.columns:
        df = df[df["imdb_votes"] >= min_votes]

    if df.empty:
        return pd.DataFrame()

    # Build per-mood config: label → {genome_tags, tmdb_kws, genres} (lowercased)
    mood_configs = []
    for label in selected_mood_labels:
        tile = MOOD_TILE_BY_LABEL.get(label)
        if tile:
            mood_configs.append({
                "label": label,
                "genome_tags": {t.lower() for t in tile["genome_tags"]},
                "tmdb_kws":    {k.lower() for k in tile["tmdb_kws"]},
                "genres":      {g.lower() for g in tile.get("genres", [])},
            })

    if not mood_configs:
        return pd.DataFrame()

    # Build enrichment lookups keyed by id (only for enriched columns)
    enr_tags = {}
    enr_kws = {}
    if enriched_df is not None:
        for _, erow in enriched_df.iterrows():
            eid = erow["id"]
            tags = erow.get("top_tags")
            if isinstance(tags, (list, np.ndarray)):
                enr_tags[eid] = {str(t).lower() for t in tags}
            kws = erow.get("tmdb_keywords")
            if isinstance(kws, (list, np.ndarray)):
                enr_kws[eid] = {str(k).lower() for k in kws}

    def _compute_mood_match(row):
        """Triple-signal weighted mood match: genre (40%) + keyword (40%) + tag (20%)."""
        tid = row["id"]
        title_tags   = enr_tags.get(tid, set())
        title_kws    = enr_kws.get(tid, set())
        title_genres = {g.lower() for g in (row.get("genres") or [])
                        if isinstance(g, str)}

        matched = []
        total_score = 0.0
        for mc in mood_configs:
            # Genre signal: fractional overlap — 100% of titles have genres
            genre_hit = (
                len(title_genres & mc["genres"]) / len(mc["genres"])
                if mc["genres"] else 0.0
            )
            # Keyword signal: fractional overlap of TMDB keywords (59.3% coverage)
            kw_hit = (
                min(len(title_kws & mc["tmdb_kws"]) / len(mc["tmdb_kws"]), 1.0)
                if mc["tmdb_kws"] else 0.0
            )
            # Tag signal: fractional overlap of genome tags (10.5% coverage)
            tag_hit = (
                min(len(title_tags & mc["genome_tags"]) / len(mc["genome_tags"]), 1.0)
                if mc["genome_tags"] else 0.0
            )

            mood_hit = 0.40 * genre_hit + 0.40 * kw_hit + 0.20 * tag_hit
            # Apply genre compatibility penalty for emotionally-typed moods
            compat = _MOOD_GENRE_COMPAT.get(mc["label"])
            if compat and mood_hit > 0:
                has_required = bool(title_genres & compat["required_any"])
                is_incompatible = bool(title_genres & compat["incompatible"])
                if is_incompatible and not has_required:
                    mood_hit *= _MOOD_GENRE_PENALTY
            # Rescale by available signal weight so genre-only titles aren't
            # unfairly penalised vs enriched titles. A title with no kw/tag data
            # can only earn at most 0.40 from genre; rescaling maps that to 1.0.
            available_weight = (0.40
                + (0.40 if title_kws else 0.0)
                + (0.20 if title_tags else 0.0))
            mood_hit_scaled = mood_hit / available_weight if available_weight > 0 else 0.0
            if mood_hit_scaled > 0:
                matched.append(mc["label"])
                total_score += mood_hit_scaled

        normalized = total_score / len(mood_configs) if mood_configs else 0.0
        return matched, normalized

    # Run match for each row (list comprehension avoids DataFrame.apply tuple pitfall)
    mood_results = [_compute_mood_match(row) for _, row in df.iterrows()]
    df["matched_moods"]  = [r[0] for r in mood_results]
    df["mood_match_pct"] = [r[1] for r in mood_results]

    # Quality score for tie-breaking — surfaces iconic titles
    df["quality_score"] = compute_quality_score(df)
    qs = df["quality_score"]
    df["quality_norm"] = (qs - qs.min()) / (qs.max() - qs.min() + 1e-9)

    # mood_match_pct is primary sort key; quality_norm is tiebreaker only
    df["mood_score"] = 0.85 * df["mood_match_pct"] + 0.15 * df["quality_norm"]

    # Keep only titles that match at least one mood; fallback to quality if none
    matched_df = df[df["mood_match_pct"] > 0]
    if matched_df.empty:
        matched_df = df  # fallback: quality-ranked results

    return matched_df.sort_values(
        ["mood_match_pct", "quality_norm"], ascending=[False, False]
    ).head(top_k)
