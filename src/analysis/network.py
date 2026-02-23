"""Analysis functions for the Cast & Crew Network page (Page 7).

Sections:
  1. Person Search & Profile
  2. Community Detection
  3. Influence Scoring
  4. Rankings
"""

import numpy as np
import pandas as pd

from src.config import PLATFORMS


def search_person(person_stats, query, role_filter=None, min_titles=1, top_k=20):
    """Search for a person by name."""
    if person_stats.empty or not query:
        return pd.DataFrame()

    mask = person_stats["name"].str.contains(query, case=False, na=False)
    results = person_stats[mask]

    if role_filter and role_filter != "All":
        results = results[results["primary_role"].str.upper() == role_filter.upper()]

    results = results[results["title_count"] >= min_titles]
    return results.head(top_k)


def get_person_profile(person_id, person_stats, edges_df, credits_df, titles_df, principals_df=None):
    """Build detailed profile for a person."""
    stats = person_stats[person_stats["person_id"] == person_id]
    if stats.empty:
        return None

    person = stats.iloc[0].to_dict()

    # Filmography from credits
    person_credits = credits_df[credits_df["person_id"] == person_id]
    if principals_df is not None and not principals_df.empty:
        person_principals = principals_df[principals_df["person_id"] == person_id]
        # Get title_ids from principals via imdb_id -> titles join
        if not person_principals.empty and "imdb_id" in titles_df.columns:
            imdb_to_id = titles_df.set_index("imdb_id")["id"].to_dict()
            principal_title_ids = person_principals["imdb_id"].map(imdb_to_id).dropna()
        else:
            principal_title_ids = pd.Series(dtype=str)
    else:
        principal_title_ids = pd.Series(dtype=str)

    # Combine title IDs
    all_title_ids = set(person_credits["title_id"].unique() if "title_id" in person_credits.columns
                        else person_credits.get("id", pd.Series()).unique())
    all_title_ids.update(principal_title_ids.unique())

    # Build filmography
    filmography = titles_df[titles_df["id"].isin(all_title_ids)].copy()
    if not filmography.empty:
        filmography = filmography.sort_values("release_year", ascending=False)

        # Add role info
        role_map = {}
        for _, row in person_credits.iterrows():
            tid = row.get("title_id") or row.get("id")
            if tid:
                role_map[tid] = row.get("role", "Unknown")
        if principals_df is not None:
            for _, row in principals_df[principals_df["person_id"] == person_id].iterrows():
                tid = imdb_to_id.get(row["imdb_id"]) if "imdb_id" in row else None
                if tid:
                    role_map[tid] = row.get("category", "Unknown").title()

        filmography["role"] = filmography["id"].map(role_map).fillna("Unknown")

        # Add character names from credits
        char_map = {}
        for _, row in person_credits.iterrows():
            tid = row.get("title_id") or row.get("id")
            if tid and pd.notna(row.get("character")):
                char_map[tid] = row["character"]
        filmography["character"] = filmography["id"].map(char_map)

    person["filmography"] = filmography

    # Top collaborators from edges
    if not edges_df.empty:
        collab_a = edges_df[edges_df["person_a"] == person_id][["person_b", "weight"]].rename(
            columns={"person_b": "collaborator_id"})
        collab_b = edges_df[edges_df["person_b"] == person_id][["person_a", "weight"]].rename(
            columns={"person_a": "collaborator_id"})
        collabs = pd.concat([collab_a, collab_b]).sort_values("weight", ascending=False)

        if not collabs.empty:
            collab_names = person_stats.set_index("person_id")["name"].to_dict()
            collab_roles = person_stats.set_index("person_id")["primary_role"].to_dict()
            collabs["name"] = collabs["collaborator_id"].map(collab_names)
            collabs["role"] = collabs["collaborator_id"].map(collab_roles)
            person["top_collaborators"] = collabs.head(5).to_dict("records")
        else:
            person["top_collaborators"] = []
    else:
        person["top_collaborators"] = []

    # Awards context
    if "award_wins" in filmography.columns:
        award_titles = filmography[filmography["award_wins"].notna() & (filmography["award_wins"] > 0)]
        if len(award_titles) > 0:
            person["award_titles"] = award_titles[["title", "award_wins"]].to_dict("records")
        else:
            person["award_titles"] = []
    else:
        person["award_titles"] = []

    # Career trend (IMDb over time)
    career_data = filmography[filmography["imdb_score"].notna() & filmography["release_year"].notna()]
    if len(career_data) >= 3:
        person["career_trend"] = career_data[["release_year", "imdb_score", "title"]].to_dict("records")
    else:
        person["career_trend"] = []

    return person


def get_community_details(person_stats, community_id, titles_df=None):
    """Get details about a specific community."""
    members = person_stats[person_stats["community_id"] == community_id]
    if members.empty:
        return None

    top_members = members.nlargest(10, "influence_score")
    dominant_genre = members["top_genre"].mode().iloc[0] if not members["top_genre"].mode().empty else "Unknown"

    # Platform breakdown
    platform_counts = {}
    if "platform_list" in members.columns:
        for platforms in members["platform_list"].dropna():
            if isinstance(platforms, str):
                # Handle stringified lists from parquet
                try:
                    platforms = eval(platforms)  # noqa: S307
                except Exception:
                    continue
            if isinstance(platforms, (list, np.ndarray)):
                for p in platforms:
                    platform_counts[p] = platform_counts.get(p, 0) + 1

    return {
        "community_id": community_id,
        "member_count": len(members),
        "avg_imdb": round(members["avg_imdb"].dropna().mean(), 2),
        "dominant_genre": dominant_genre,
        "top_members": top_members[["name", "primary_role", "influence_score", "title_count"]].to_dict("records"),
        "platform_breakdown": platform_counts,
    }


def get_cross_platform_bridges(person_stats, edges_df):
    """Find people who bridge Netflix-heavy and Max-heavy clusters."""
    if person_stats.empty or edges_df.empty:
        return pd.DataFrame()

    # Identify Netflix-heavy vs Max-heavy people
    def _parse_platform_list(platforms):
        if isinstance(platforms, str):
            try:
                platforms = eval(platforms)  # noqa: S307
            except Exception:
                return []
        if isinstance(platforms, (list, np.ndarray)):
            return list(platforms)
        return []

    def dominant_platform(platforms):
        platforms = _parse_platform_list(platforms)
        if not platforms:
            return None
        netflix_count = sum(1 for p in platforms if p == "netflix")
        max_count = sum(1 for p in platforms if p == "max")
        if netflix_count > max_count:
            return "netflix"
        elif max_count > netflix_count:
            return "max"
        return "both"

    person_stats = person_stats.copy()
    if "platform_list" not in person_stats.columns:
        return pd.DataFrame()
    person_stats["dominant"] = person_stats["platform_list"].apply(dominant_platform)

    # Find people connected to both Netflix-heavy and Max-heavy people
    netflix_people = set(person_stats[person_stats["dominant"] == "netflix"]["person_id"])
    max_people = set(person_stats[person_stats["dominant"] == "max"]["person_id"])

    bridges = []
    for _, row in edges_df.iterrows():
        a, b = row["person_a"], row["person_b"]
        if (a in netflix_people and b in max_people) or (a in max_people and b in netflix_people):
            bridges.append({"person_a": a, "person_b": b, "weight": row["weight"]})

    if not bridges:
        return pd.DataFrame()

    bridges_df = pd.DataFrame(bridges)
    # Find people who appear most as bridges
    all_bridge_people = list(bridges_df["person_a"]) + list(bridges_df["person_b"])
    bridge_counts = pd.Series(all_bridge_people).value_counts()
    top_bridges = bridge_counts.head(20)

    result = person_stats[person_stats["person_id"].isin(top_bridges.index)].copy()
    result["bridge_count"] = result["person_id"].map(bridge_counts)
    return result.sort_values("bridge_count", ascending=False)


def get_rankings(person_stats, role_filter="All", sort_by="influence_score", top_k=50):
    """Get ranked list of people by various criteria."""
    df = person_stats.copy()

    if role_filter != "All":
        df = df[df["primary_role"].str.upper() == role_filter.upper()]

    sort_col = {
        "influence_score": "influence_score",
        "most_titles": "title_count",
        "highest_avg_imdb": "avg_imdb",
        "most_popular": "title_count",  # proxy
    }.get(sort_by, "influence_score")

    df = df.dropna(subset=[sort_col])
    return df.nlargest(top_k, sort_col)
