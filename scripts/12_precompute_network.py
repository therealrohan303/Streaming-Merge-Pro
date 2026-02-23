"""
Script 12: Precompute collaboration network.

Builds collaboration edges from imdb_principals.parquet (ALL role categories)
and credits data. Computes PageRank, Louvain community detection, and
per-person statistics.

Output:
  - data/precomputed/network/edges.parquet
  - data/precomputed/network/person_stats.parquet
"""

import sys
from pathlib import Path

import community as community_louvain
import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    ENRICHED_DIR,
    NETWORK_MIN_EDGE_WEIGHT,
    NETWORK_MIN_TITLES,
    PRECOMPUTED_DIR,
    PROCESSED_DIR,
)

NETWORK_DIR = PRECOMPUTED_DIR / "network"


def load_credits_and_principals():
    """Load and merge credits + IMDb principals into unified person-title data."""
    # Base credits (actor/director from platform data)
    credits = pd.read_parquet(PROCESSED_DIR / "all_platforms_credits.parquet")
    credits = credits.rename(columns={"id": "title_id"})
    print(f"Base credits: {len(credits):,} rows")

    # IMDb principals (writer/producer/composer/cinematographer)
    principals_path = ENRICHED_DIR / "imdb_principals.parquet"
    if principals_path.exists():
        principals = pd.read_parquet(principals_path)
        # Map imdb_id to title ids via titles table
        titles = pd.read_parquet(PROCESSED_DIR / "all_platforms_titles.parquet",
                                columns=["id", "imdb_id"])
        principals = principals.merge(
            titles[["id", "imdb_id"]].drop_duplicates(),
            left_on="imdb_id", right_on="imdb_id", how="inner",
        )
        principals = principals.rename(columns={"id": "title_id"})
        # Standardize column names to match credits
        principals["role"] = principals["category"].str.upper()
        principals = principals[["person_id", "title_id", "name", "role"]].copy()
        print(f"IMDb principals: {len(principals):,} rows")
    else:
        principals = pd.DataFrame(columns=["person_id", "title_id", "name", "role"])
        print("  IMDb principals not found, using credits only")

    # Combine — credits has actor/director, principals has writer/producer/etc.
    # Credits uses person_id + name columns already
    credits_slim = credits[["person_id", "title_id", "name", "role"]].copy()
    combined = pd.concat([credits_slim, principals], ignore_index=True)
    # Normalize person_id to string to avoid mixed int/str types in parquet
    combined["person_id"] = combined["person_id"].astype(str)
    combined = combined.drop_duplicates(subset=["person_id", "title_id", "role"])
    print(f"Combined: {len(combined):,} person-title-role links")
    return combined


def build_collaboration_edges(person_titles):
    """Build edges between people who worked on the same title."""
    print("\nBuilding collaboration edges ...")

    # Filter to people with minimum title count
    person_counts = person_titles.groupby("person_id")["title_id"].nunique()
    active_persons = set(person_counts[person_counts >= NETWORK_MIN_TITLES].index)
    filtered = person_titles[person_titles["person_id"].isin(active_persons)]
    print(f"  People with >= {NETWORK_MIN_TITLES} titles: {len(active_persons):,}")

    # Group by title to find co-workers
    title_groups = filtered.groupby("title_id")["person_id"].apply(set)

    edge_counts = {}
    for title_id, people in title_groups.items():
        people_list = sorted(people, key=str)
        for i in range(len(people_list)):
            for j in range(i + 1, len(people_list)):
                key = (people_list[i], people_list[j])
                edge_counts[key] = edge_counts.get(key, 0) + 1

    # Filter by minimum edge weight
    edges = [
        {"person_a": k[0], "person_b": k[1], "weight": v}
        for k, v in edge_counts.items()
        if v >= NETWORK_MIN_EDGE_WEIGHT
    ]
    edges_df = pd.DataFrame(edges)
    print(f"  Edges with weight >= {NETWORK_MIN_EDGE_WEIGHT}: {len(edges_df):,}")
    return edges_df


def compute_person_stats(person_titles, edges_df, titles_df):
    """Compute per-person statistics including PageRank and community."""
    print("\nComputing person statistics ...")

    # Build NetworkX graph
    G = nx.Graph()
    for _, row in edges_df.iterrows():
        G.add_edge(row["person_a"], row["person_b"], weight=row["weight"])
    print(f"  Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    # PageRank
    pagerank = nx.pagerank(G, weight="weight")
    print(f"  PageRank computed")

    # Louvain community detection
    communities = community_louvain.best_partition(G, weight="weight", random_state=42)
    n_communities = len(set(communities.values()))
    print(f"  Louvain communities: {n_communities}")

    # Pre-index title metadata (do this ONCE, not per person)
    titles_indexed = titles_df.drop_duplicates(subset="id").set_index("id")
    title_scores = titles_indexed["imdb_score"].to_dict()
    title_years = titles_indexed["release_year"].to_dict()
    title_platforms = titles_indexed["platform"].to_dict()

    title_genres_map = {}
    for tid, genres in titles_indexed["genres"].items():
        if isinstance(genres, (list, np.ndarray)):
            title_genres_map[tid] = list(genres)
        else:
            title_genres_map[tid] = []

    has_awards = "award_wins" in titles_indexed.columns
    title_awards = titles_indexed["award_wins"].to_dict() if has_awards else {}

    person_name_map = person_titles.drop_duplicates("person_id").set_index("person_id")["name"].to_dict()
    person_role_map = (
        person_titles.groupby("person_id")["role"]
        .apply(lambda x: x.mode().iloc[0] if len(x) > 0 else "UNKNOWN")
        .to_dict()
    )

    print("  Computing per-person stats ...")
    rows = []
    person_title_groups = person_titles.groupby("person_id")
    total = len(person_title_groups)

    for idx, (person_id, group) in enumerate(person_title_groups):
        title_ids = group["title_id"].unique()
        imdb_scores = [title_scores[tid] for tid in title_ids
                       if tid in title_scores and pd.notna(title_scores.get(tid))]
        avg_imdb = np.mean(imdb_scores) if imdb_scores else None

        # Career span
        years = [int(title_years[tid]) for tid in title_ids
                 if tid in title_years and pd.notna(title_years.get(tid))]
        career_start = min(years) if years else None
        career_end = max(years) if years else None

        # Top genre
        all_genres = []
        for tid in title_ids:
            all_genres.extend(title_genres_map.get(tid, []))
        top_genre = max(set(all_genres), key=all_genres.count) if all_genres else None

        # Platform breakdown
        platforms = list(set(title_platforms.get(tid) for tid in title_ids
                            if tid in title_platforms and title_platforms.get(tid)))

        # Awards context
        award_title_count = 0
        total_award_wins = 0
        if has_awards:
            for tid in title_ids:
                aw = title_awards.get(tid)
                if pd.notna(aw) and aw > 0:
                    award_title_count += 1
                    total_award_wins += int(aw)

        # Influence score
        pr = pagerank.get(person_id, 0)
        community_id = communities.get(person_id, -1)
        norm_awards = total_award_wins / max(len(title_ids), 1) if has_awards else 0
        influence_score = pr * (avg_imdb or 0) * (1 + norm_awards)

        rows.append({
            "person_id": person_id,
            "name": person_name_map.get(person_id, "Unknown"),
            "primary_role": person_role_map.get(person_id, "UNKNOWN"),
            "title_count": len(title_ids),
            "avg_imdb": round(avg_imdb, 2) if avg_imdb else None,
            "career_start": career_start,
            "career_end": career_end,
            "top_genre": top_genre,
            "pagerank": pr,
            "community_id": community_id,
            "influence_score": influence_score,
            "award_title_count": award_title_count,
            "total_award_wins": total_award_wins,
            "platform_list": platforms,
        })

        if (idx + 1) % 10000 == 0:
            print(f"    ... {idx + 1}/{total}")

    stats_df = pd.DataFrame(rows)
    stats_df = stats_df.sort_values("influence_score", ascending=False)
    print(f"  Person stats: {len(stats_df):,} people")
    return stats_df


def main():
    NETWORK_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    person_titles = load_credits_and_principals()
    titles_df = pd.read_parquet(PROCESSED_DIR / "all_platforms_titles.parquet")

    # Build edges
    edges_df = build_collaboration_edges(person_titles)
    edges_path = NETWORK_DIR / "edges.parquet"
    edges_df.to_parquet(edges_path, index=False)
    print(f"  Saved {edges_path}")

    # Compute person stats
    stats_df = compute_person_stats(person_titles, edges_df, titles_df)
    stats_path = NETWORK_DIR / "person_stats.parquet"
    stats_df.to_parquet(stats_path, index=False)
    print(f"  Saved {stats_path}")

    # Summary
    print(f"\nTop 10 by influence score:")
    for _, row in stats_df.head(10).iterrows():
        print(f"  {row['name']} ({row['primary_role']}) — "
              f"influence: {row['influence_score']:.6f}, "
              f"titles: {row['title_count']}, "
              f"avg IMDb: {row['avg_imdb']}")

    print("\nDone!")


if __name__ == "__main__":
    main()
