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

import itertools

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
    filled_genres = [g for g in weak_genres if max_genres.get(g, 0) > 10]
    filled_by_max = len(filled_genres)
    filled_names = ", ".join(sorted(filled_genres)[:5])
    kpis["genre_gap_coverage"] = {
        "value": f"{filled_by_max}/{len(weak_genres)}",
        "label": "Genre Gaps Filled by Max",
        "detail": f"Genres added: {filled_names}" if filled_names else f"Max fills {filled_by_max} of Netflix's {len(weak_genres)} weak genres",
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

    # Catalog rank among all platforms
    from src.config import COMPETITOR_PLATFORMS
    merged_size = len(merged)
    comp_sizes = {p: len(df[df["platform"] == p]) for p in COMPETITOR_PLATFORMS}
    if comp_sizes:
        all_sizes = {**comp_sizes, "merged": merged_size}
        sorted_platforms = sorted(all_sizes, key=all_sizes.get, reverse=True)
        rank = sorted_platforms.index("merged") + 1
        above = [PLATFORMS.get(p, {}).get("name", p) for p in sorted_platforms[:rank - 1]]
        below = [PLATFORMS.get(p, {}).get("name", p) for p in sorted_platforms[rank:rank + 2]]
        above_str = f"Behind only {', '.join(above)}." if above else "Largest catalog in streaming."
        below_str = f" Ahead of {', '.join(below[:2])}." if below else ""
        kpis["catalog_rank"] = {
            "value": f"#{rank} by size",
            "label": "Catalog Rank",
            "detail": f"{merged_size:,} titles. {above_str}{below_str}",
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

    # Platforms that belong to the "perspective" entity — never counted as competitors
    own_platforms = MERGED_PLATFORMS if perspective == "merged" else [perspective]

    if competitor:
        if competitor == "merged":
            comp = build_merged_entity(df)
        else:
            comp = df[df["platform"] == competitor]
    else:
        # All Competitors = every platform except the perspective's own platform(s)
        comp_platforms = [p for p in ALL_PLATFORMS if p not in own_platforms]
        comp = df[df["platform"].isin(comp_platforms)]

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

        # Audience demand (from avg IMDb votes in this genre)
        if "imdb_votes" in base_genre_data.columns:
            avg_votes = base_genre_data["imdb_votes"].mean()
            if pd.isna(avg_votes) or avg_votes < 10_000:
                audience_demand = "Low"
            elif avg_votes < 100_000:
                audience_demand = "Medium"
            else:
                audience_demand = "High"
        else:
            audience_demand = "Unknown"

        # Data confidence (kept for completeness but hidden in cards when static)
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
            # Find which competitor leads most (excluding own platform(s))
            per_comp = {}
            for p in [p for p in ALL_PLATFORMS if p not in own_platforms]:
                p_data = df[df["platform"] == p].explode("genres")
                p_count = len(p_data[p_data["genres"] == genre])
                per_comp[p] = p_count
            leader = max(per_comp, key=per_comp.get) if per_comp else "unknown"
            leader_count = per_comp.get(leader, 0)
            ratio = (leader_count / max(len(df[df["platform"] == leader]), 1)) / max(base_share, 0.001)

        # Genre-specific addends for Low Priority / Narrow-gap recommendations
        _GENRE_REC_ADDENDS = {
            "romance": " K-drama and European romance catalog offer high-quality titles at lower acquisition costs.",
            "family": " Animated feature films and holiday specials are evergreen with high rewatch value.",
            "music": " Concert films and music biopics perform well with dedicated fanbase appeal.",
            "animation": " International animation (Studio Ghibli catalog, French animation) commands high engagement.",
            "documentary": " Award-circuit docs and investigative series align with prestige brand positioning.",
            "comedy": " Stand-up specials and international comedy (UK, Indian cinema) offer strong CPM value.",
            "sport": " Feature-length sports documentaries (not live rights) are high-demand and acquirable.",
            "western": " Classic western catalog collections anchor a distinct content identity for adult audiences.",
            "horror": " International horror (J-horror, Spanish horror) fills gaps at affordable acquisition rates.",
            "musical": " Broadway adaptation specials and concert films fit both family and prestige demographics.",
        }

        # Recommendation — dynamic based on gap characteristics
        target_count = min(max(int(comp_count * 0.3), 10), 50)
        min_imdb = max(round((comp_avg or 7.0) - 0.5, 1), 6.5)
        _g = genre.title()
        if base_share < 0.02:
            recommendation = (
                f"Build from scratch: acquire 40-60 {_g} titles targeting 7.0+ IMDb "
                f"classics and 2015+ catalog — no meaningful presence exists yet."
            )
        elif ratio >= 3.0 and (comp_avg or 0) > 7.0:
            recommendation = (
                f"Quality gap: competitor avg IMDb {comp_avg:.1f} vs merged — "
                f"target elevated {_g} content ({min_imdb}+) to close prestige gap, not just volume."
            )
        elif ratio < 1.5:
            addend = _GENRE_REC_ADDENDS.get(genre.lower(), "")
            recommendation = (
                f"Narrow gap ({ratio:.1f}x): 2-3 tentpole {_g} acquisitions sufficient."
                f"{addend}"
            )
        else:
            recommendation = (
                f"Acquire {target_count}-{target_count + 20} {_g} titles, "
                f"{min_imdb}+ IMDb, 2018-2023 to close the {ratio:.1f}x coverage gap."
            )

        rows.append({
            "gap_type": "genre_share",
            "genre": genre,
            "severity": severity,
            "coverage_pct": round(base_share * 100, 1),
            "quality_benchmark": round(base_avg, 2) if pd.notna(base_avg) else None,
            "competitor_lead": f"{PLATFORMS.get(leader, {}).get('name', leader)} has {ratio:.1f}x",
            "confidence": confidence,
            "audience_demand": audience_demand,
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
    battleground_dicts = []

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
            # Store counts so the page can show a closeness bar
            closeness = m_count / max(m_count + c_count, 1)
            battleground_dicts.append({
                "genre": genre,
                "merger_count": m_count,
                "comp_count": c_count,
                "closeness": round(closeness, 3),
            })

    # Sort battlegrounds by closeness to 0.5 (most evenly contested first)
    battleground_dicts.sort(key=lambda x: abs(x["closeness"] - 0.5))

    return {
        "our_leads": sorted(our_leads, key=lambda x: float(x["lead"].replace("x", "")), reverse=True)[:8],
        "their_leads": sorted(their_leads, key=lambda x: float(x["lead"].replace("x", "")), reverse=True)[:8],
        "battlegrounds": battleground_dicts[:8],
        "merged_size": len(merged),
        "comp_size": len(comp),
        "merged_avg_imdb": round(merged["imdb_score"].mean(), 2),
        "comp_avg_imdb": round(comp["imdb_score"].mean(), 2),
    }


# ─── New helpers added 2026-03-20 ────────────────────────────────────────────

def compute_quality_distribution(df):
    """Compute IMDb score distribution data for the Quality Ladder section.

    Returns a dict with per-platform score arrays and % above 7.0/8.0 thresholds.
    """
    result = {}
    display_platforms = ALL_PLATFORMS + ["merged"]
    for key in display_platforms:
        if key == "merged":
            pdata = build_merged_entity(df)
        else:
            pdata = df[df["platform"] == key]
        scores = pdata["imdb_score"].dropna().tolist()
        n = len(scores)
        above_7 = sum(1 for s in scores if s >= 7.0) / max(n, 1)
        above_8 = sum(1 for s in scores if s >= 8.0) / max(n, 1)
        result[key] = {
            "scores": scores,
            "n": n,
            "avg": round(float(np.mean(scores)), 2) if scores else None,
            "above_7": above_7,
            "above_8": above_8,
        }
    return result


def compute_temporal_momentum(df):
    """Compute avg IMDb score by decade for each platform.

    Returns a DataFrame: platform, decade, avg_imdb, title_count.
    Filters to decades with ≥10 titles to avoid noise.
    """
    rows = []
    for key in ALL_PLATFORMS + ["merged"]:
        if key == "merged":
            pdata = build_merged_entity(df)
        else:
            pdata = df[df["platform"] == key]
        if "decade" not in pdata.columns:
            continue
        grouped = (
            pdata[pdata["imdb_score"].notna()]
            .groupby("decade")
            .agg(avg_imdb=("imdb_score", "mean"), title_count=("imdb_score", "count"))
            .reset_index()
        )
        grouped = grouped[grouped["title_count"] >= 10]
        grouped["platform"] = key
        rows.append(grouped)
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame(columns=["platform", "decade", "avg_imdb", "title_count"])


def compute_alternative_scenario(df, platform_a, platform_b):
    """Compute merger KPIs for any two-platform hypothetical combination.

    Returns a dict with the same keys as compute_merger_kpis() where possible.
    """
    pa = df[df["platform"] == platform_a].copy()
    pb = df[df["platform"] == platform_b].copy()
    combined = pd.concat([pa, pb]).drop_duplicates(subset="id", keep="first")
    combined["platform"] = "alt_merged"

    kpis = {}

    kpis["catalog_size"] = {
        "value": f"{len(combined):,}",
        "label": "Combined Catalog Size",
        "detail": f"{PLATFORMS.get(platform_a,{}).get('name',platform_a)}: {len(pa):,} + {PLATFORMS.get(platform_b,{}).get('name',platform_b)}: {len(pb):,}",
    }

    avg_a = pa["imdb_score"].mean()
    avg_merged = combined["imdb_score"].mean()
    lift = avg_merged - avg_a if pd.notna(avg_a) and pd.notna(avg_merged) else 0
    kpis["avg_imdb"] = {
        "value": f"{avg_merged:.2f}" if pd.notna(avg_merged) else "N/A",
        "label": "Avg IMDb",
        "detail": f"Lift vs {PLATFORMS.get(platform_a,{}).get('name',platform_a)}: {lift:+.2f}",
    }

    # Genre diversity
    a_genres = pa.explode("genres")["genres"].value_counts()
    b_genres = pb.explode("genres")["genres"].value_counts()
    m_genres = combined.explode("genres")["genres"].value_counts()

    def _entropy(vc):
        t = vc.sum()
        if t == 0:
            return 0.0
        p = vc / t
        return float(-np.sum(p * np.log2(p + 1e-9)))

    kpis["genre_diversity"] = {
        "value": f"{_entropy(m_genres):.2f}",
        "label": "Genre Diversity (Shannon)",
        "detail": f"{PLATFORMS.get(platform_a,{}).get('name',platform_a)}: {_entropy(a_genres):.2f} → combined: {_entropy(m_genres):.2f}",
    }

    # Overlap rate
    a_profile = set(zip(pa.explode("genres")["genres"], pa.explode("genres")["type"]))
    b_profile = set(zip(pb.explode("genres")["genres"], pb.explode("genres")["type"]))
    overlap = a_profile & b_profile
    total = a_profile | b_profile
    overlap_rate = len(overlap) / max(len(total), 1)
    kpis["overlap_rate"] = {
        "value": f"{overlap_rate:.0%}",
        "label": "Content Profile Overlap",
        "detail": f"{len(overlap)} of {len(total)} genre-type pairs overlap",
    }

    # Prestige
    if "award_wins" in combined.columns:
        wins = combined["award_wins"].fillna(0).sum()
        a_wins = pa["award_wins"].fillna(0).sum()
        coverage = combined["award_wins"].notna().mean()
        if coverage >= WIKIDATA_MIN_COVERAGE and a_wins > 0:
            lift_pct = (wins - a_wins) / a_wins
            kpis["prestige_lift"] = {
                "value": f"+{lift_pct:.0%}",
                "label": "Award-Winning Titles Lift",
                "detail": f"Award wins: {int(a_wins)} → {int(wins)} ({coverage:.0%} coverage)",
            }
        kpis["_award_wins_raw"] = float(wins) if "award_wins" in combined.columns else 0.0
    else:
        kpis["_award_wins_raw"] = 0.0

    # Franchise depth — visible KPI + internal radar field
    if "collection_name" in combined.columns:
        franchise_depth = combined["collection_name"].notna().sum() / max(len(combined), 1)
    else:
        franchise_depth = 0.0
    kpis["franchise_depth"] = {
        "value": f"{franchise_depth:.0%}",
        "label": "Franchise Depth",
        "detail": "% of titles belonging to a named franchise or film collection",
    }
    kpis["_franchise_depth"] = franchise_depth

    # International reach — visible KPI + internal radar field
    # Proxy: titles whose primary production country is outside the major English-speaking markets
    _ENGLISH_COUNTRIES = {"US", "GB", "AU", "CA", "IE", "NZ"}
    if "production_countries" in combined.columns:
        def _is_intl(countries):
            if countries is None or (isinstance(countries, float)):
                return False
            lst = countries if isinstance(countries, list) else [str(countries)]
            return bool(lst) and lst[0] not in _ENGLISH_COUNTRIES
        intl_reach = combined["production_countries"].apply(_is_intl).sum() / max(len(combined), 1)
    else:
        intl_reach = 0.0
    kpis["intl_reach"] = {
        "value": f"{intl_reach:.0%}",
        "label": "International Reach",
        "detail": "% of titles primarily produced outside English-speaking markets (US/UK/AU/CA)",
    }
    kpis["_intl_reach"] = intl_reach

    # Raw numbers for radar normalization
    kpis["_catalog_size_raw"] = len(combined)
    kpis["_avg_imdb_raw"] = float(avg_merged) if pd.notna(avg_merged) else 0.0
    entropy_val = _entropy(m_genres)
    kpis["_entropy_raw"] = entropy_val

    return kpis


def compute_best_alternative_scenario(df):
    """Find the platform pair (excluding Netflix+Max) with highest quality-weighted catalog.

    Returns (platform_a, platform_b) tuple or None if computation fails.
    """
    best = None
    best_score = 0.0
    for pa, pb in itertools.combinations(ALL_PLATFORMS, 2):
        if {pa, pb} == set(MERGED_PLATFORMS):
            continue
        combined = pd.concat([
            df[df["platform"] == pa],
            df[df["platform"] == pb],
        ]).drop_duplicates(subset="id", keep="first")
        avg = combined["imdb_score"].mean()
        score = len(combined) * (float(avg) if pd.notna(avg) else 0.0)
        if score > best_score:
            best_score = score
            best = (pa, pb)
    return best


def compute_acquisition_shortlist(df, gap_df, n_per_gap=8, perspective="merged"):
    """Find real titles from competitor catalogs that fit each high-priority gap.

    For each High/Medium severity gap genre, finds actual titles on competitor
    platforms in that genre, sorted by Fit Score.

    Returns dict: {genre: DataFrame with title, year, platform, imdb_score, imdb_votes, fit_score}.
    """
    if gap_df.empty:
        return {}

    # Exclude own-platform titles so they never appear as acquisition candidates
    if perspective == "merged":
        own_ids = set(build_merged_entity(df)["id"])
    else:
        own_ids = set(df[df["platform"] == perspective]["id"].dropna())

    shortlist = {}

    high_priority = gap_df[gap_df["severity"].isin(["High", "Medium"])].head(5)

    for _, gap in high_priority.iterrows():
        genre = gap["genre"]
        # Find titles from competitor platforms in this genre
        comp_titles = df[~df["id"].isin(own_ids)].copy()
        comp_exploded = comp_titles.explode("genres")
        in_genre = comp_exploded[comp_exploded["genres"] == genre].copy()

        if in_genre.empty:
            continue

        # Compute fit score: quality (50%) + audience size (30%) + recency (20%)
        in_genre = in_genre[in_genre["imdb_score"].notna()].copy()
        if in_genre.empty:
            continue

        max_year = in_genre["release_year"].max() if "release_year" in in_genre.columns else 2023
        min_year = max(in_genre["release_year"].min(), 1980) if "release_year" in in_genre.columns else 1980

        in_genre["_quality"] = in_genre["imdb_score"] / 10.0
        if "imdb_votes" in in_genre.columns:
            in_genre["_audience"] = (in_genre["imdb_votes"].fillna(0).clip(upper=100000) / 100000)
        else:
            in_genre["_audience"] = 0.5
        if "release_year" in in_genre.columns:
            year_range = max(max_year - min_year, 1)
            in_genre["_recency"] = (in_genre["release_year"].fillna(min_year) - min_year) / year_range
        else:
            in_genre["_recency"] = 0.5

        in_genre["fit_score"] = (
            0.5 * in_genre["_quality"] +
            0.3 * in_genre["_audience"] +
            0.2 * in_genre["_recency"]
        )

        cols = ["id", "title", "release_year", "platform", "imdb_score", "fit_score"]
        if "imdb_votes" in in_genre.columns:
            cols.insert(5, "imdb_votes")

        result = (
            in_genre[[c for c in cols if c in in_genre.columns]]
            .drop_duplicates(subset=["title", "release_year"])
            .sort_values("fit_score", ascending=False)
            .head(n_per_gap)
            .copy()
        )
        result["fit_score"] = result["fit_score"].round(3)
        shortlist[genre] = result

    return shortlist
