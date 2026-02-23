"""Analysis functions for the Strategic Insights page (Page 5).

Sections:
  1. Merger Value Dashboard (headline KPIs)
  2. Prestige Index (awards per 1K titles)
  3. Content Overlap Analysis
  4. Gap Analysis with Decision Trace
  5. IP Synergy Map
  6. Competitive Positioning
  7. Market Impact Simulation
"""

import numpy as np
import pandas as pd

from src.config import (
    ALL_PLATFORMS,
    MERGED_PLATFORMS,
    PLATFORMS,
    WIKIDATA_COMPARISON_MIN_COVERAGE,
    WIKIDATA_MIN_COVERAGE,
    TMDB_MIN_COVERAGE,
)


def build_merged_entity(df):
    """Create merged Netflix+Max, deduplicating by id."""
    merged = df[df["platform"].isin(MERGED_PLATFORMS)].copy()
    merged = merged.drop_duplicates(subset="id", keep="first")
    merged["platform"] = "merged"
    return merged


def compute_merger_kpis(df):
    """Compute 4-5 headline KPIs for the merger value dashboard."""
    merged = build_merged_entity(df)
    netflix = df[df["platform"] == "netflix"]
    max_df = df[df["platform"] == "max"]

    kpis = {}

    # Genre gap coverage: genres Max brings that Netflix is weak in
    netflix_genres = netflix.explode("genres")["genres"].value_counts()
    max_genres = max_df.explode("genres")["genres"].value_counts()
    merged_genres = merged.explode("genres")["genres"].value_counts()
    all_genres = set(netflix_genres.index) | set(max_genres.index)

    # Netflix weak genres = genres where Netflix has < 3% share
    netflix_total = len(netflix)
    weak_genres = [g for g in all_genres
                   if netflix_genres.get(g, 0) / max(netflix_total, 1) < 0.03]
    filled_by_max = sum(1 for g in weak_genres if max_genres.get(g, 0) > 10)
    kpis["genre_gap_coverage"] = {
        "value": f"{filled_by_max}/{len(weak_genres)}",
        "label": "Genre Gaps Filled by Max",
        "detail": f"Max fills {filled_by_max} of Netflix's {len(weak_genres)} weak genres",
    }

    # Content overlap rate (by genre + type matching)
    netflix_profile = set(zip(
        netflix.explode("genres")["genres"],
        netflix.explode("genres")["type"],
    ))
    max_profile = set(zip(
        max_df.explode("genres")["genres"],
        max_df.explode("genres")["type"],
    ))
    overlap = netflix_profile & max_profile
    total = netflix_profile | max_profile
    overlap_rate = len(overlap) / max(len(total), 1)
    kpis["overlap_rate"] = {
        "value": f"{overlap_rate:.0%}",
        "label": "Content Profile Overlap",
        "detail": f"{len(overlap)} of {len(total)} genre-type combinations overlap",
    }

    # Quality lift
    netflix_avg = netflix["imdb_score"].mean()
    merged_avg = merged["imdb_score"].mean()
    lift = merged_avg - netflix_avg
    kpis["quality_lift"] = {
        "value": f"+{lift:.2f}",
        "label": "Avg IMDb Lift",
        "detail": f"Post-merger avg IMDb: {merged_avg:.2f} (was {netflix_avg:.2f})",
    }

    # Genre diversity (entropy)
    def genre_entropy(genre_counts):
        total = genre_counts.sum()
        if total == 0:
            return 0
        probs = genre_counts / total
        return -np.sum(probs * np.log2(probs + 1e-9))

    netflix_entropy = genre_entropy(netflix_genres)
    merged_entropy = genre_entropy(merged_genres)
    entropy_increase = merged_entropy - netflix_entropy
    kpis["genre_diversity"] = {
        "value": f"+{entropy_increase:.2f}",
        "label": "Genre Diversity Gain",
        "detail": f"Shannon entropy: {netflix_entropy:.2f} → {merged_entropy:.2f}",
    }

    # Optional: Prestige lift (if awards data available)
    if "award_wins" in df.columns:
        merged_awards = merged["award_wins"].fillna(0).sum()
        netflix_awards = netflix["award_wins"].fillna(0).sum()
        coverage = merged["award_wins"].notna().mean()
        if coverage >= WIKIDATA_MIN_COVERAGE:
            if netflix_awards > 0:
                prestige_lift = (merged_awards - netflix_awards) / netflix_awards
                kpis["prestige_lift"] = {
                    "value": f"+{prestige_lift:.0%}",
                    "label": "Award-Winning Titles Lift",
                    "detail": f"Award wins: {int(netflix_awards)} → {int(merged_awards)} ({coverage:.0%} coverage)",
                }

    return kpis


def compute_overlap_analysis(df):
    """Compute content overlap between Netflix and Max by genre, type, decade."""
    netflix = df[df["platform"] == "netflix"]
    max_df = df[df["platform"] == "max"]

    # Genre × type overlap intensity
    def profile(platform_df):
        exploded = platform_df.explode("genres")
        return exploded.groupby(["genres", "type"]).size()

    netflix_profile = profile(netflix)
    max_profile = profile(max_df)

    # Create overlap matrix
    all_keys = sorted(set(netflix_profile.index) | set(max_profile.index))
    rows = []
    for key in all_keys:
        n_count = netflix_profile.get(key, 0)
        m_count = max_profile.get(key, 0)
        overlap = min(n_count, m_count)
        rows.append({
            "genre": key[0],
            "type": key[1],
            "netflix_count": n_count,
            "max_count": m_count,
            "overlap": overlap,
            "confidence": min(n_count, m_count) / max(max(n_count, m_count), 1),
        })

    return pd.DataFrame(rows)


def compute_overlap_heatmap(overlap_df):
    """Build genre overlap heatmap data from overlap analysis."""
    if overlap_df.empty:
        return pd.DataFrame()

    # Pivot by genre
    heatmap = (
        overlap_df.groupby("genre")
        .agg({"overlap": "sum", "netflix_count": "sum", "max_count": "sum"})
        .reset_index()
    )
    heatmap["total"] = heatmap["netflix_count"] + heatmap["max_count"]
    heatmap["overlap_pct"] = heatmap["overlap"] / heatmap["total"].clip(lower=1)
    return heatmap.sort_values("overlap", ascending=False)


def compute_gap_analysis(df, perspective="merged", competitor=None):
    """Compute gaps with full decision trace fields.

    Returns DataFrame with columns: gap_type, genre, severity, coverage_pct,
    quality_benchmark, competitor_lead, confidence, box_office_tier, recommendation.
    """
    if perspective == "merged":
        base = build_merged_entity(df)
    else:
        base = df[df["platform"] == perspective]

    if competitor:
        comp = df[df["platform"] == competitor]
    else:
        competitors = [p for p in ALL_PLATFORMS if p not in MERGED_PLATFORMS]
        comp = df[df["platform"].isin(competitors)]

    base_exploded = base.explode("genres")
    comp_exploded = comp.explode("genres")
    base_total = len(base)
    comp_total = len(comp)

    base_genres = base_exploded["genres"].value_counts()
    comp_genres = comp_exploded["genres"].value_counts()

    rows = []
    for genre in set(base_genres.index) | set(comp_genres.index):
        base_count = base_genres.get(genre, 0)
        comp_count = comp_genres.get(genre, 0)
        base_share = base_count / max(base_total, 1)
        comp_share = comp_count / max(comp_total, 1)

        if comp_share <= base_share:
            continue

        ratio = comp_share / max(base_share, 0.001)

        # Quality benchmark
        base_genre_data = base_exploded[base_exploded["genres"] == genre]
        comp_genre_data = comp_exploded[comp_exploded["genres"] == genre]
        base_avg = base_genre_data["imdb_score"].mean()
        comp_avg = comp_genre_data["imdb_score"].mean()

        # Severity
        if ratio >= 3.0 or base_share < 0.03:
            severity = "High"
        elif ratio >= 1.5:
            severity = "Medium"
        else:
            severity = "Low"

        # Box office tier
        box_office_tier = "Unknown"
        if "box_office_usd" in comp_genre_data.columns:
            bo = comp_genre_data["box_office_usd"].dropna()
            if len(bo) > 5:
                median = bo.median()
                if median > 200e6:
                    box_office_tier = "High"
                elif median > 50e6:
                    box_office_tier = "Mid"
                else:
                    box_office_tier = "Low"

        # Data confidence
        confidence = "High"
        if "data_confidence" in comp_genre_data.columns:
            avg_conf = comp_genre_data["data_confidence"].mean()
            if avg_conf < 0.2:
                confidence = "Low"
            elif avg_conf < 0.5:
                confidence = "Medium"

        # Competitor leader
        if competitor:
            leader = competitor
        else:
            # Find which competitor leads most
            per_comp = {}
            for p in [p for p in ALL_PLATFORMS if p not in MERGED_PLATFORMS]:
                p_data = df[df["platform"] == p].explode("genres")
                p_count = len(p_data[p_data["genres"] == genre])
                per_comp[p] = p_count
            leader = max(per_comp, key=per_comp.get) if per_comp else "unknown"
            leader_count = per_comp.get(leader, 0)
            ratio = (leader_count / max(len(df[df["platform"] == leader]), 1)) / max(base_share, 0.001)

        # Recommendation
        target_count = min(max(int(comp_count * 0.3), 10), 50)
        min_imdb = max(round((comp_avg or 7.0) - 0.5, 1), 6.5)
        recommendation = f"Acquire {target_count}-{target_count + 20} titles, {genre}, {min_imdb}+ IMDb, 2018-2023"

        rows.append({
            "gap_type": "genre_share",
            "genre": genre,
            "severity": severity,
            "coverage_pct": round(base_share * 100, 1),
            "quality_benchmark": round(base_avg, 2) if pd.notna(base_avg) else None,
            "competitor_lead": f"{PLATFORMS.get(leader, {}).get('name', leader)} has {ratio:.1f}x",
            "confidence": confidence,
            "box_office_tier": box_office_tier,
            "recommendation": recommendation,
        })

    result = pd.DataFrame(rows)
    severity_order = {"High": 0, "Medium": 1, "Low": 2}
    if len(result) > 0:
        result["_sev_order"] = result["severity"].map(severity_order)
        result = result.sort_values("_sev_order").drop(columns="_sev_order")
    return result


def compute_ip_synergy(df):
    """Compute franchise/collection analysis pre/post merger.

    Only renders if TMDB collection coverage exceeds threshold.
    """
    if "collection_name" not in df.columns:
        return None, 0

    coverage = df["collection_name"].notna().mean()
    if coverage < TMDB_MIN_COVERAGE:
        return None, coverage

    netflix = df[df["platform"] == "netflix"]
    max_df = df[df["platform"] == "max"]
    merged = build_merged_entity(df)

    def franchise_stats(platform_df, label):
        fdf = platform_df[platform_df["collection_name"].notna()]
        franchise_counts = fdf["collection_name"].value_counts()
        franchise_quality = fdf.groupby("collection_name")["imdb_score"].mean()
        result = pd.DataFrame({
            "franchise": franchise_counts.index,
            "title_count": franchise_counts.values,
            "avg_imdb": [franchise_quality.get(f, None) for f in franchise_counts.index],
            "source": label,
        })
        # Quality score = title_count * avg_imdb
        result["quality_score"] = result["title_count"] * result["avg_imdb"].fillna(6.0)
        return result.head(15)

    netflix_franchises = franchise_stats(netflix, "Netflix")
    max_franchises = franchise_stats(max_df, "Max")
    merged_franchises = franchise_stats(merged, "Merged")

    return {
        "netflix": netflix_franchises,
        "max": max_franchises,
        "merged": merged_franchises,
        "coverage": coverage,
    }, coverage


def compute_market_simulation(df):
    """Compute market impact simulation metrics."""
    merged = build_merged_entity(df)
    platforms = {}

    for key in ALL_PLATFORMS:
        pdata = df[df["platform"] == key]
        platforms[key] = {
            "name": PLATFORMS[key]["name"],
            "catalog_size": len(pdata),
            "avg_imdb": pdata["imdb_score"].mean(),
            "quality_weighted": len(pdata) * pdata["imdb_score"].mean(),
        }

    # Add merged
    platforms["merged"] = {
        "name": "Netflix + Max",
        "catalog_size": len(merged),
        "avg_imdb": merged["imdb_score"].mean(),
        "quality_weighted": len(merged) * merged["imdb_score"].mean(),
    }

    # HHI calculation (using catalog size as proxy for market share)
    # Exclude netflix and max individually, include merged
    hhi_platforms = {k: v for k, v in platforms.items()
                     if k not in ("netflix", "max")}
    total_size = sum(v["catalog_size"] for v in hhi_platforms.values())
    hhi = sum((v["catalog_size"] / total_size * 100) ** 2
              for v in hhi_platforms.values())

    # Regulatory context
    if hhi > 2500:
        reg_label = "Highly concentrated"
    elif hhi > 1500:
        reg_label = "Moderately concentrated"
    else:
        reg_label = "Unconcentrated"

    return {
        "platforms": platforms,
        "hhi": round(hhi),
        "regulatory_label": reg_label,
        "total_catalog": total_size,
    }


def compute_competitive_positioning(df, competitor):
    """Compute competitive positioning vs a specific competitor."""
    merged = build_merged_entity(df)
    comp = df[df["platform"] == competitor]

    merged_exploded = merged.explode("genres")
    comp_exploded = comp.explode("genres")

    merged_genres = merged_exploded["genres"].value_counts()
    comp_genres = comp_exploded["genres"].value_counts()

    all_genres = set(merged_genres.index) | set(comp_genres.index)

    our_leads = []
    their_leads = []
    battlegrounds = []

    for genre in all_genres:
        m_count = merged_genres.get(genre, 0)
        c_count = comp_genres.get(genre, 0)
        m_share = m_count / max(len(merged), 1)
        c_share = c_count / max(len(comp), 1)

        if m_share > c_share * 1.5:
            m_avg = merged_exploded[merged_exploded["genres"] == genre]["imdb_score"].mean()
            our_leads.append({"genre": genre, "lead": f"{m_share/max(c_share, 0.001):.1f}x",
                            "count": m_count, "avg_imdb": round(m_avg, 2) if pd.notna(m_avg) else None})
        elif c_share > m_share * 1.5:
            c_avg = comp_exploded[comp_exploded["genres"] == genre]["imdb_score"].mean()
            their_leads.append({"genre": genre, "lead": f"{c_share/max(m_share, 0.001):.1f}x",
                              "count": c_count, "avg_imdb": round(c_avg, 2) if pd.notna(c_avg) else None})
        else:
            battlegrounds.append(genre)

    return {
        "our_leads": sorted(our_leads, key=lambda x: float(x["lead"].replace("x", "")), reverse=True)[:8],
        "their_leads": sorted(their_leads, key=lambda x: float(x["lead"].replace("x", "")), reverse=True)[:8],
        "battlegrounds": battlegrounds[:8],
        "merged_size": len(merged),
        "comp_size": len(comp),
        "merged_avg_imdb": round(merged["imdb_score"].mean(), 2),
        "comp_avg_imdb": round(comp["imdb_score"].mean(), 2),
    }
