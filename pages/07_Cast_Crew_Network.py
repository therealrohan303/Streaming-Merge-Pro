"""
Cast & Crew — Collaboration Network · Actor Wordle · Head-to-Head Arena
"""

import os
import random
import tempfile
from collections import deque

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Cast & Crew", page_icon="🎬", layout="wide")

# ─── Shared Constants ─────────────────────────────────────────────────────────
PLAT_COLORS = {
    "netflix":   "#E50914",
    "max":       "#002BE7",
    "prime":     "#FF9900",
    "disney":    "#8B31C7",
    "paramount": "#F5C518",
    "appletv":   "#888888",
}
PLAT_LABELS = {
    "netflix":   "Netflix",
    "max":       "Max",
    "prime":     "Prime Video",
    "disney":    "Disney+",
    "paramount": "Paramount+",
    "appletv":   "Apple TV+",
}
DEFAULT_COLOR = "#444466"
CARD_BG       = "#1E1E2E"
CARD_BORDER   = "#333"
CARD_TEXT     = "#ddd"

# Actor Wordle colours
AW_GREEN  = "#1db954"
AW_YELLOW = "#f0a500"
AW_RED    = "#c0392b"
AW_MAX_GUESSES = 6
AW_MIN_TITLES  = 5
AW_MIN_IMDB    = 6.0
AW_POOL_SIZE   = 1000
PERSON_STATS_PATH = "data/precomputed/network/person_stats.parquet"

# Head-to-Head Arena colours
ARENA_TEAL  = "#00b4a6"
ARENA_GOLD  = "#FFD700"
ARENA_MUTED = "#555566"
ARENA_MIN_TITLES = 3


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def _load_credits() -> pd.DataFrame:
    df = pd.read_parquet("data/processed/all_platforms_credits.parquet")
    df["person_id"] = df["person_id"].astype(str)
    return df


@st.cache_data
def _load_titles() -> pd.DataFrame:
    return pd.read_parquet("data/processed/all_platforms_titles.parquet")


@st.cache_data
def load_network_data():
    edges_df     = pd.read_parquet("data/precomputed/network/edges.parquet")
    person_stats = pd.read_parquet("data/precomputed/network/person_stats.parquet")
    person_stats["person_id"] = person_stats["person_id"].astype(str)
    edges_df["person_a"]      = edges_df["person_a"].astype(str)
    edges_df["person_b"]      = edges_df["person_b"].astype(str)
    return edges_df, person_stats


@st.cache_data
def compute_primary_platform():
    credits = _load_credits()
    titles  = _load_titles()
    t_slim  = titles[["id", "platform"]].rename(columns={"id": "title_id"})
    merged  = credits[["person_id", "title_id"]].merge(t_slim, on="title_id", how="left")
    return (
        merged.groupby(["person_id", "platform"]).size()
        .reset_index(name="cnt")
        .sort_values("cnt", ascending=False)
        .drop_duplicates("person_id")[["person_id", "platform"]]
        .rename(columns={"platform": "primary_platform"})
    )


@st.cache_data
def build_adjacency() -> dict:
    edges_df, _ = load_network_data()
    adj: dict = {}
    for pa, pb, w in zip(edges_df["person_a"], edges_df["person_b"], edges_df["weight"]):
        adj.setdefault(pa, {})[pb] = int(w)
        adj.setdefault(pb, {})[pa] = int(w)
    return adj


@st.cache_data
def build_arena_pool() -> pd.DataFrame:
    _, person_stats = load_network_data()
    credits         = _load_credits()
    titles          = _load_titles()

    person_stats = person_stats.copy()

    pool = person_stats[person_stats["title_count"] >= ARENA_MIN_TITLES].copy()

    t_slim  = titles[["id", "platform", "imdb_score", "genres"]].rename(columns={"id": "title_id"})
    cmerged = credits[["person_id", "title_id"]].merge(t_slim, on="title_id", how="left")

    primary_plat = (
        cmerged.groupby(["person_id", "platform"]).size()
        .reset_index(name="cnt").sort_values("cnt", ascending=False)
        .drop_duplicates("person_id")[["person_id", "platform"]]
        .rename(columns={"platform": "primary_platform"})
    )
    pool = pool.merge(primary_plat, on="person_id", how="left")

    best_title = (
        cmerged.groupby("person_id")["imdb_score"].max()
        .reset_index().rename(columns={"imdb_score": "best_title_imdb"})
    )
    pool = pool.merge(best_title, on="person_id", how="left")

    def to_list(g):
        if isinstance(g, (list, np.ndarray)):
            return [str(x) for x in g if x]
        return []

    cmerged["genres_list"] = cmerged["genres"].apply(to_list)
    exploded = cmerged.explode("genres_list").dropna(subset=["genres_list"])
    exploded = exploded[exploded["genres_list"] != ""]
    genre_div = (
        exploded.groupby("person_id")["genres_list"].nunique()
        .reset_index().rename(columns={"genres_list": "genre_diversity"})
    )
    pool = pool.merge(genre_div, on="person_id", how="left")

    pool["platform_diversity"] = pool["platform_list"].apply(
        lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0
    )

    plat_counts = cmerged.groupby(["person_id", "platform"]).size().reset_index(name="cnt")
    for plat_key, col_name in [("netflix", "netflix_titles"), ("max", "max_titles")]:
        pc = plat_counts[plat_counts["platform"] == plat_key][["person_id", "cnt"]].rename(columns={"cnt": col_name})
        pool = pool.merge(pc, on="person_id", how="left")
        pool[col_name] = pool[col_name].fillna(0).astype(int)

    pool["career_span"] = pool.apply(
        lambda r: int(r["career_end"]) - int(r["career_start"])
        if pd.notna(r.get("career_start")) and pd.notna(r.get("career_end")) and r["career_end"] > 0 else 0,
        axis=1,
    )
    pool["career_label"] = pool.apply(
        lambda r: f"{int(r['career_start'])}–{int(r['career_end'])}"
        if pd.notna(r.get("career_start")) and pd.notna(r.get("career_end")) and r["career_end"] > 0 else "N/A",
        axis=1,
    )

    pool["genre_diversity"]  = pool["genre_diversity"].fillna(0).astype(int)
    pool["best_title_imdb"]  = pool["best_title_imdb"].fillna(0.0).round(1)
    pool["primary_platform"] = pool["primary_platform"].fillna("N/A")
    return pool.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  NETWORK VISUALIZER HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def bfs_subgraph(seed_id: str, adj: dict, max_nodes: int = 100, max_hops: int = 2):
    visited: dict = {seed_id: 0}
    queue = deque([(seed_id, 0)])
    edges: set = set()
    while queue and len(visited) < max_nodes:
        node, hop = queue.popleft()
        if hop >= max_hops:
            continue
        for nbr, weight in sorted(adj.get(node, {}).items(), key=lambda x: -x[1]):
            if len(visited) >= max_nodes:
                break
            if nbr not in visited:
                visited[nbr] = hop + 1
                queue.append((nbr, hop + 1))
            if nbr in visited:
                key = (min(node, nbr), max(node, nbr))
                edges.add((*key, weight))
    return set(visited.keys()), edges


@st.cache_data(show_spinner=False)
def render_pyvis(node_ids: frozenset, edge_triples: frozenset, seed_id: str) -> str:
    from pyvis.network import Network

    _, person_stats = load_network_data()
    primary_plat    = compute_primary_platform()

    sub = person_stats[person_stats["person_id"].isin(node_ids)].copy()
    sub = sub.drop_duplicates("person_id")
    sub = sub.merge(primary_plat, on="person_id", how="left")
    sub = sub.drop_duplicates("person_id")
    sub["primary_platform"] = sub["primary_platform"].fillna("netflix")

    # Deduplicate by name (same person, two source IDs) — keep highest title_count
    sub = sub.sort_values("title_count", ascending=False)
    remap: dict = {}
    for name in sub[sub["name"].duplicated(keep=False)]["name"].unique():
        rows = sub[sub["name"] == name]
        kept = rows.iloc[0]["person_id"]
        for d in rows.iloc[1:]["person_id"].tolist():
            remap[d] = kept
    sub = sub.drop_duplicates("name", keep="first")

    remapped: set = set()
    for pa, pb, w in edge_triples:
        ra, rb = remap.get(pa, pa), remap.get(pb, pb)
        if ra == rb:
            continue
        remapped.add((min(ra, rb), max(ra, rb), w))
    edge_triples = remapped
    node_ids     = set(sub["person_id"].tolist())

    tc_max, tc_min = sub["title_count"].max(), sub["title_count"].min()

    def node_size(tc):
        return 22 if tc_max == tc_min else 10 + (tc - tc_min) / (tc_max - tc_min) * 38

    net = Network(height="650px", width="100%",
                  bgcolor="#1E1E2E", font_color="#cccccc", directed=False)
    net.barnes_hut(gravity=-9000, central_gravity=0.35,
                   spring_length=160, spring_strength=0.05, damping=0.09)

    for _, row in sub.iterrows():
        pid     = row["person_id"]
        color   = PLAT_COLORS.get(row["primary_platform"], DEFAULT_COLOR)
        is_seed = pid == seed_id
        tooltip = (
            f"<b>{row['name']}</b><br>"
            f"Role: {str(row['primary_role']).title()}<br>"
            f"Platform: {PLAT_LABELS.get(row['primary_platform'], row['primary_platform'])}<br>"
            f"Titles: {int(row['title_count'])}<br>"
            f"Avg IMDb: {float(row['avg_imdb']):.1f}"
        )
        net.add_node(pid, label=row["name"], title=tooltip,
                     size=node_size(row["title_count"]),
                     color={"background": color,
                            "border": "#FFD700" if is_seed else color,
                            "highlight": {"background": "#FFD700", "border": "#ffffff"}},
                     borderWidth=3 if is_seed else 1,
                     font={"size": 11, "color": "#dddddd"})

    for pa, pb, w in edge_triples:
        if pa in node_ids and pb in node_ids:
            net.add_edge(pa, pb, value=max(1, w), title=f"Shared titles: {w}",
                         color={"color": "rgba(255,255,255,0.12)",
                                "highlight": "rgba(255,215,0,0.6)"})

    net.set_options("""{
      "nodes": {"shadow": {"enabled": true, "size": 8}},
      "edges": {"smooth": {"type": "continuous"}},
      "physics": {
        "enabled": true,
        "stabilization": {"enabled": true, "iterations": 150, "updateInterval": 25}
      },
      "interaction": {"hover": true, "tooltipDelay": 80,
                      "navigationButtons": true, "keyboard": true}
    }""")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        net.save_graph(f.name)
        tmp = f.name
    with open(tmp, "r", encoding="utf-8") as f:
        html = f.read()
    os.unlink(tmp)
    return html


# ═══════════════════════════════════════════════════════════════════════════════
#  HEAD-TO-HEAD ARENA HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def arena_stat_rows(a: pd.Series, b: pd.Series) -> list[dict]:
    def row(label, val_a, val_b, display_a, display_b, higher_wins=True, is_text=False):
        if is_text:
            winner = "tie"
        elif val_a > val_b:
            winner = "a" if higher_wins else "b"
        elif val_b > val_a:
            winner = "b" if higher_wins else "a"
        else:
            winner = "tie"
        return {"label": label, "display_a": str(display_a), "display_b": str(display_b),
                "winner": winner}

    plat_a = PLAT_LABELS.get(a["primary_platform"], a["primary_platform"])
    plat_b = PLAT_LABELS.get(b["primary_platform"], b["primary_platform"])
    return [
        row("Primary Platform",   0, 0, plat_a, plat_b, is_text=True),
        row("Primary Role",       0, 0, str(a["primary_role"]).title(), str(b["primary_role"]).title(), is_text=True),
        row("Total Titles",       a["title_count"],        b["title_count"],        int(a["title_count"]),          int(b["title_count"])),
        row("Avg IMDb Score",     a["avg_imdb"],           b["avg_imdb"],           f"{float(a['avg_imdb']):.2f}",  f"{float(b['avg_imdb']):.2f}"),
        row("Best Single Title",  a["best_title_imdb"],    b["best_title_imdb"],    f"{float(a['best_title_imdb']):.1f}", f"{float(b['best_title_imdb']):.1f}"),
        row("Genre Diversity",    a["genre_diversity"],    b["genre_diversity"],    int(a["genre_diversity"]),       int(b["genre_diversity"])),
        row("Platform Diversity", a["platform_diversity"], b["platform_diversity"], int(a["platform_diversity"]),    int(b["platform_diversity"])),
        row("Career Span (yrs)",  a["career_span"],        b["career_span"],
            f"{a['career_label']} ({int(a['career_span'])} yrs)",
            f"{b['career_label']} ({int(b['career_span'])} yrs)"),
        row("Netflix Titles",     a["netflix_titles"],     b["netflix_titles"],     int(a["netflix_titles"]),        int(b["netflix_titles"])),
        row("Max Titles",         a["max_titles"],         b["max_titles"],         int(a["max_titles"]),            int(b["max_titles"])),
    ]


def arena_merger_insight(a: pd.Series, b: pd.Series, winner_name: str) -> str:
    a_name, b_name = a["name"], b["name"]
    if a["platform_diversity"] != b["platform_diversity"]:
        more = a_name if a["platform_diversity"] > b["platform_diversity"] else b_name
        return (f"**{more}** spans more platforms — higher risk/value in a merger "
                f"as their contracts touch more streaming ecosystems.")
    if abs(float(a["netflix_titles"]) + float(a["max_titles"]) -
           float(b["netflix_titles"]) - float(b["max_titles"])) > 2:
        more = a_name if (a["netflix_titles"] + a["max_titles"]) > (b["netflix_titles"] + b["max_titles"]) else b_name
        return (f"**{more}** has deeper presence across the two merging platforms (Netflix + Max) — "
                f"making them a more central figure in the combined catalog.")
    if a["genre_diversity"] != b["genre_diversity"]:
        more = a_name if a["genre_diversity"] > b["genre_diversity"] else b_name
        return (f"**{more}** covers more genres — their versatility makes them "
                f"harder to replace and more valuable to a merged platform.")
    if a["career_span"] != b["career_span"]:
        more = a_name if a["career_span"] > b["career_span"] else b_name
        return (f"**{more}** has a longer career span — their catalog longevity "
                f"provides the merged platform with deeper archival value.")
    return (f"**{winner_name}** edges out on overall stats — they represent "
            f"stronger cross-platform value in a Netflix × Max merger.")


def arena_stat_table(rows: list[dict]):
    for r in rows:
        col_a = ARENA_TEAL  if r["winner"] == "a" else (ARENA_MUTED if r["winner"] == "b" else CARD_TEXT)
        col_b = ARENA_TEAL  if r["winner"] == "b" else (ARENA_MUTED if r["winner"] == "a" else CARD_TEXT)
        wt_a  = "700" if r["winner"] == "a" else "400"
        wt_b  = "700" if r["winner"] == "b" else "400"
        tr_a  = " 🏆" if r["winner"] == "a" else ""
        tr_b  = "🏆 " if r["winner"] == "b" else ""
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 160px 1fr;
                    align-items:center;padding:10px 16px;border-bottom:1px solid #2a2a3e;">
            <div style="text-align:right;font-size:1em;font-weight:{wt_a};color:{col_a};">
                {r['display_a']}{tr_a}</div>
            <div style="text-align:center;font-size:0.75em;color:#666;
                        text-transform:uppercase;letter-spacing:1px;padding:0 12px;">
                {r['label']}</div>
            <div style="text-align:left;font-size:1em;font-weight:{wt_b};color:{col_b};">
                {tr_b}{r['display_b']}</div>
        </div>""", unsafe_allow_html=True)


def arena_winner_card(name: str, wins: int, total: int, insight: str):
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0d2d2a,#1a3a36);
                border:2px solid {ARENA_TEAL};border-radius:14px;
                padding:24px;text-align:center;margin-top:20px;">
        <div style="font-size:0.8em;color:{ARENA_TEAL};text-transform:uppercase;
                    letter-spacing:2px;margin-bottom:6px;">Overall Winner</div>
        <div style="font-size:2em;font-weight:900;color:#fff;margin-bottom:4px;">
            🏆 {name}</div>
        <div style="color:#aaa;font-size:0.9em;margin-bottom:16px;">Won {wins} of {total} stats</div>
        <div style="background:rgba(0,180,166,0.1);border:1px solid {ARENA_TEAL};
                    border-radius:8px;padding:12px;font-size:0.88em;color:{CARD_TEXT};text-align:left;">
            💡 <em>Merger insight:</em> {insight}</div>
    </div>""", unsafe_allow_html=True)


def arena_person_card(p: pd.Series, side: str):
    color = ARENA_TEAL if side == "left" else ARENA_GOLD
    plat  = PLAT_LABELS.get(p.get("primary_platform", "N/A"), p.get("primary_platform", "N/A"))
    st.markdown(f"""
    <div style="background:{CARD_BG};border:2px solid {color};border-radius:12px;
                padding:16px;text-align:center;">
        <div style="font-size:1.3em;font-weight:800;color:#fff;margin-bottom:8px;">{p['name']}</div>
        <div style="display:flex;gap:8px;flex-wrap:wrap;justify-content:center;">
            <span style="background:#2a2a3e;padding:4px 10px;border-radius:16px;
                         color:#aaa;font-size:0.8em;">{str(p['primary_role']).title()}</span>
            <span style="background:#2a2a3e;padding:4px 10px;border-radius:16px;
                         color:#aaa;font-size:0.8em;">{plat}</span>
            <span style="background:#2a2a3e;padding:4px 10px;border-radius:16px;
                         color:#aaa;font-size:0.8em;">{int(p['title_count'])} titles</span>
        </div>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  ACTOR WORDLE HELPERS  (exact logic from actor_wordle.py)
# ═══════════════════════════════════════════════════════════════════════════════

AW_CLUE_LABELS = [
    "Hint",
    "Number of titles",
    "Top genre",
    "Career era",
    "Platform(s)",
    "One title they appeared in",
]


@st.cache_data
def load_wordle_people() -> pd.DataFrame:
    df = pd.read_parquet(PERSON_STATS_PATH)
    df = df[
        (df["primary_role"] == "ACTOR")
        & (df["title_count"] >= AW_MIN_TITLES)
        & (df["avg_imdb"] >= AW_MIN_IMDB)
    ].copy()
    df["name"] = df["name"].astype(str)
    df = df.nlargest(AW_POOL_SIZE, "influence_score")
    return df.reset_index(drop=True)


@st.cache_data
def load_person_title_lookup() -> dict:
    """Returns {person_id_str: title_name} — uses already-cached credits/titles."""
    credits = _load_credits()
    titles  = _load_titles()
    merged = credits[["person_id", "title_id"]].merge(
        titles[["id", "title"]].rename(columns={"id": "title_id"}),
        on="title_id", how="left"
    ).dropna(subset=["title"])
    return merged.drop_duplicates("person_id").set_index("person_id")["title"].to_dict()


def pick_mystery(seed: int, people: pd.DataFrame, title_lookup: dict) -> dict:
    row = people.sample(1, random_state=seed % (2**31)).iloc[0]

    platforms_raw = row.get("platform_list", "")
    if isinstance(platforms_raw, (list, set)):
        platforms_str = ", ".join(sorted(str(p).capitalize() for p in platforms_raw))
    elif isinstance(platforms_raw, str) and platforms_raw:
        platforms_str = platforms_raw
    else:
        platforms_str = "Multiple platforms"

    start = row.get("career_start")
    if pd.notna(start) and start:
        decade = (int(start) // 10) * 10
        era = f"Active since the {decade}s"
    else:
        era = "Career era unknown"

    tc = int(row.get("title_count", 0))
    if tc <= 9:
        tc_range = "5–9 titles"
    elif tc <= 19:
        tc_range = "10–19 titles"
    elif tc <= 39:
        tc_range = "20–39 titles"
    else:
        tc_range = "40+ titles"

    one_title = title_lookup.get(str(row["person_id"]), "— title data unavailable —")

    clues = [
        "Guess the actor! Use the clues below to narrow it down.",
        f"{tc_range} in the catalog",
        str(row.get("top_genre", "Unknown")).title(),
        era,
        platforms_str,
        f'"{one_title}"',
    ]

    return {
        "name":         row["name"],
        "person_id":    row["person_id"],
        "clues":        clues,
        "avg_imdb":     row.get("avg_imdb", None),
        "title_count":  int(row.get("title_count", 0)),
        "primary_role": str(row.get("primary_role", "")),
        "top_genre":    str(row.get("top_genre", "")),
        "career_start": row.get("career_start", None),
        "platform_list": row.get("platform_list", []),
    }


def filter_candidates(mystery: dict, num_clues_shown: int, people: pd.DataFrame) -> list:
    filtered = people.copy()

    # Clue 2: title count bucket
    if num_clues_shown >= 2:
        tc = mystery["title_count"]
        if tc <= 9:
            filtered = filtered[(filtered["title_count"] >= 5) & (filtered["title_count"] <= 9)]
        elif tc <= 19:
            filtered = filtered[(filtered["title_count"] >= 10) & (filtered["title_count"] <= 19)]
        elif tc <= 39:
            filtered = filtered[(filtered["title_count"] >= 20) & (filtered["title_count"] <= 39)]
        else:
            filtered = filtered[filtered["title_count"] >= 40]

    # Clue 3: top genre
    if num_clues_shown >= 3:
        filtered = filtered[filtered["top_genre"].str.lower() == mystery["top_genre"].lower()]

    # Clue 4: career era (same decade)
    if num_clues_shown >= 4 and mystery["career_start"] and pd.notna(mystery["career_start"]):
        target_decade = (int(mystery["career_start"]) // 10) * 10
        def same_decade(val):
            try:
                return (int(val) // 10) * 10 == target_decade
            except Exception:
                return False
        filtered = filtered[filtered["career_start"].apply(same_decade)]

    # Clue 5: platform overlap
    if num_clues_shown >= 5:
        mystery_platforms = set()
        raw = mystery["platform_list"]
        if isinstance(raw, (list, set)):
            mystery_platforms = set(str(p).lower() for p in raw)
        elif isinstance(raw, str):
            mystery_platforms = set(p.strip().lower() for p in raw.split(","))

        def shares_platform(val):
            if isinstance(val, (list, set)):
                return bool(mystery_platforms & set(str(p).lower() for p in val))
            elif isinstance(val, str) and val:
                return bool(mystery_platforms & set(p.strip().lower() for p in val.split(",")))
            return False

        if mystery_platforms:
            filtered = filtered[filtered["platform_list"].apply(shares_platform)]

    return sorted(filtered["name"].unique().tolist())


# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD DATA (shared, cached)
# ═══════════════════════════════════════════════════════════════════════════════

with st.spinner("Loading data…"):
    edges_df, person_stats = load_network_data()
    primary_plat           = compute_primary_platform()
    adj                    = build_adjacency()
    arena_pool             = build_arena_pool()
    wordle_people          = load_wordle_people()
    title_lookup           = load_person_title_lookup()

# Actor Wordle session state
if "aw_seed" not in st.session_state:
    st.session_state.aw_seed = random.randint(0, 99999)
if "aw_mystery" not in st.session_state:
    st.session_state.aw_mystery = pick_mystery(st.session_state.aw_seed, wordle_people, title_lookup)
if "aw_guesses" not in st.session_state:
    st.session_state.aw_guesses = []
if "aw_won" not in st.session_state:
    st.session_state.aw_won = False
if "aw_lost" not in st.session_state:
    st.session_state.aw_lost = False

# Arena session state
if "arena_a" not in st.session_state:
    st.session_state.arena_a = None
if "arena_b" not in st.session_state:
    st.session_state.arena_b = None


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE HEADER + TABS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="padding:16px 0 4px;">
    <div style="font-size:2em;font-weight:900;color:#fff;">🎬 Cast & Crew</div>
    <div style="color:#888;font-size:0.9em;margin-top:4px;">Collaboration Network · Actor Wordle · Head-to-Head Arena</div>
</div>
""", unsafe_allow_html=True)

tab_net, tab_cw, tab_arena = st.tabs(["🕸️ Collaboration Network", "🎬 Actor Wordle", "⚔️ Head-to-Head Arena"])


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1: NETWORK VISUALIZER
# ═══════════════════════════════════════════════════════════════════════════════

with tab_net:
    st.markdown("##### Pick a person — explore their collaboration web up to 2 degrees of separation")

    # Default seed = most connected node
    connected_ids = set(adj.keys())
    default_seed  = (
        person_stats[person_stats["person_id"].isin(connected_ids)]
        .assign(_deg=lambda df: df["person_id"].map(lambda p: len(adj.get(p, {}))))
        .nlargest(1, "_deg")["person_id"].iloc[0]
    )

    seed_pool = (
        person_stats[person_stats["person_id"].isin(connected_ids)]
        .sort_values("title_count", ascending=False)
        [["person_id", "name", "primary_role", "title_count"]]
        .head(5000)
    )
    seed_ids = seed_pool["person_id"].tolist()

    def fmt_seed(pid):
        r = seed_pool[seed_pool["person_id"] == pid]
        if r.empty:
            return pid
        row = r.iloc[0]
        return f"{row['name']}  ({str(row['primary_role']).title()}, {int(row['title_count'])} titles)"

    ALL_ROLES = ["ACTOR", "DIRECTOR", "WRITER", "PRODUCER", "COMPOSER", "CINEMATOGRAPHER", "EDITOR"]

    col_s, col_d, col_m = st.columns([4, 1, 1])
    with col_s:
        seed_pid = st.selectbox(
            "Seed person", options=seed_ids, format_func=fmt_seed,
            index=seed_ids.index(default_seed) if default_seed in seed_ids else 0,
            placeholder="Type a name to search…", label_visibility="collapsed", key="net_seed",
        )
    with col_d:
        depth = st.selectbox("Degrees", [1, 2], index=1, key="net_depth")
    with col_m:
        max_nodes = st.selectbox("Max nodes", [50, 100, 150], index=1, key="net_max")

    role_filter = st.multiselect(
        "Filter by role", options=ALL_ROLES, default=ALL_ROLES,
        format_func=lambda r: r.title(), key="net_roles",
    )
    if not role_filter:
        role_filter = ALL_ROLES

    # Legend
    st.markdown(
        " &nbsp; ".join(
            f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;'
            f'background:{c};margin-right:3px;vertical-align:middle;"></span>'
            f'<span style="color:#aaa;font-size:0.78em;">{lbl}</span>'
            for lbl, c in [
                ("Netflix", PLAT_COLORS["netflix"]), ("Max", PLAT_COLORS["max"]),
                ("Prime Video", PLAT_COLORS["prime"]), ("Disney+", PLAT_COLORS["disney"]),
                ("Paramount+", PLAT_COLORS["paramount"]), ("Apple TV+", PLAT_COLORS["appletv"]),
                ("⭐ Seed", "#FFD700"),
            ]
        ), unsafe_allow_html=True,
    )

    with st.spinner("Building network…"):
        node_ids, edge_triples = bfs_subgraph(seed_pid, adj, max_nodes=max_nodes, max_hops=depth)

        selected_roles = set(role_filter)
        role_lookup    = person_stats.set_index("person_id")["primary_role"].to_dict()
        filtered_nodes = {
            pid for pid in node_ids
            if pid == seed_pid or role_lookup.get(pid, "").upper() in selected_roles
        }
        filtered_edges = {(pa, pb, w) for pa, pb, w in edge_triples
                          if pa in filtered_nodes and pb in filtered_nodes}

        net_html = render_pyvis(frozenset(filtered_nodes), frozenset(filtered_edges), seed_pid)

    components.html(net_html, height=670, scrolling=False)
    st.caption(
        f"**{len(filtered_nodes)} nodes** · **{len(filtered_edges)} edges** · "
        f"{depth}-hop neighbourhood · Node size = title count · Hover for details"
    )

    # Quick stats panel
    st.markdown("---")
    st.markdown("**Quick Stats — pick any visible node:**")
    visible = person_stats[person_stats["person_id"].isin(filtered_nodes)].sort_values("name")
    vis_ids = visible["person_id"].tolist()

    selected = st.selectbox(
        "Pick a node", options=vis_ids,
        format_func=lambda pid: visible[visible["person_id"] == pid].iloc[0]["name"]
                                if not visible[visible["person_id"] == pid].empty else pid,
        index=vis_ids.index(seed_pid) if seed_pid in vis_ids else 0,
        label_visibility="collapsed", key="net_stats",
    )

    if selected:
        r        = person_stats[person_stats["person_id"] == selected].iloc[0]
        pp_row   = primary_plat[primary_plat["person_id"] == selected]
        pplat    = pp_row.iloc[0]["primary_platform"] if not pp_row.empty else "N/A"
        pcolor   = PLAT_COLORS.get(pplat, DEFAULT_COLOR)
        plabel   = PLAT_LABELS.get(pplat, pplat)
        conns    = len(adj.get(selected, {}))

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        for col, lbl, val in [
            (c1, "Name",        r["name"]),
            (c2, "Role",        str(r["primary_role"]).title()),
            (c3, "Platform",    f'<span style="color:{pcolor}">{plabel}</span>'),
            (c4, "Titles",      int(r["title_count"])),
            (c5, "Avg IMDb",    f"{float(r['avg_imdb']):.1f}"),
            (c6, "Connections", conns),
        ]:
            with col:
                st.markdown(f"""
                <div style="background:{CARD_BG};border:1px solid {CARD_BORDER};
                            border-radius:8px;padding:12px;text-align:center;">
                    <div style="font-size:0.7em;color:#888;margin-bottom:4px;">{lbl}</div>
                    <div style="font-size:1em;font-weight:700;color:#ddd;">{val}</div>
                </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2: ACTOR WORDLE  (exact UI from actor_wordle.py)
# ═══════════════════════════════════════════════════════════════════════════════

with tab_cw:
    mystery = st.session_state.aw_mystery

    st.markdown("""
<div style="text-align:center;padding:24px 0 8px;">
    <span style="font-size:2.8rem;">🎬</span>
    <h1 style="margin:4px 0;font-size:2rem;letter-spacing:1px;">Actor Wordle</h1>
    <p style="color:#aaa;font-size:0.95rem;">Guess the mystery person from the streaming catalog.<br>
    A new clue is revealed after each wrong guess.</p>
</div>
""", unsafe_allow_html=True)

    st.divider()

    # ── STEP 1: handle input events FIRST so state is fresh before we render ──
    _num_g = len(st.session_state.aw_guesses)
    if not st.session_state.aw_won and not st.session_state.aw_lost and _num_g < AW_MAX_GUESSES:
        _filtered = filter_candidates(mystery, _num_g + 1, wordle_people)
        _filtered = [n for n in _filtered if n not in st.session_state.aw_guesses]

        st.markdown(
            f"### 🎯 Your Guess &nbsp;"
            f"<span style='color:#888;font-size:0.9rem;'>({AW_MAX_GUESSES - _num_g} remaining)</span>"
            f"&nbsp;<span style='color:#f39c12;font-size:0.85rem;'>({len(_filtered)} possible names)</span>",
            unsafe_allow_html=True,
        )
        _guess_input = st.selectbox(
            "Start typing a name...",
            options=[""] + _filtered,
            index=0,
            key=f"aw_guess_input_{_num_g}",
            label_visibility="collapsed",
        )
        if st.button("Submit Guess", type="primary", disabled=not _guess_input, key="aw_submit"):
            _guess = _guess_input.strip()
            st.session_state.aw_guesses = st.session_state.aw_guesses + [_guess]
            if _guess.lower() == mystery["name"].lower():
                st.session_state.aw_won = True
            elif len(st.session_state.aw_guesses) >= AW_MAX_GUESSES:
                st.session_state.aw_lost = True

        st.divider()

    # ── STEP 2: read fresh state and render results ───────────────────────────
    guesses     = st.session_state.aw_guesses
    num_guesses = len(guesses)
    clues_to_show = mystery["clues"][:num_guesses + 1]

    # Clues panel
    st.markdown("### 🕵️ Clues So Far")
    for i, clue_text in enumerate(clues_to_show):
        color  = "#1a472a" if i == 0 else "#1a1a2e"
        border = "#2ecc71" if i == 0 else "#4a4a8a"
        new_badge = (
            ' <span style="background:#e74c3c;color:white;font-size:0.7rem;'
            'padding:2px 6px;border-radius:4px;vertical-align:middle;">NEW</span>'
            if i == num_guesses else ""
        )
        st.markdown(f"""
    <div style="background:{color};border:1px solid {border};border-left:4px solid {border};
                border-radius:8px;padding:12px 16px;margin-bottom:8px;">
        <span style="color:#888;font-size:0.8rem;text-transform:uppercase;letter-spacing:1px;">
            Clue {i+1}: {AW_CLUE_LABELS[i]}
        </span>{new_badge}<br>
        <span style="color:white;font-size:1.1rem;font-weight:600;">{clue_text}</span>
    </div>
    """, unsafe_allow_html=True)

    # Locked future clues
    for j in range(len(clues_to_show), AW_MAX_GUESSES):
        st.markdown(f"""
    <div style="background:#111;border:1px solid #333;border-left:4px solid #333;
                border-radius:8px;padding:12px 16px;margin-bottom:8px;opacity:0.4;">
        <span style="color:#666;font-size:0.8rem;text-transform:uppercase;letter-spacing:1px;">
            Clue {j+1}: {AW_CLUE_LABELS[j]}
        </span><br>
        <span style="color:#444;font-size:1.1rem;">🔒 Unlocks after guess {j}</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Previous guesses
    if guesses:
        st.markdown("### ❌ Previous Guesses")
        for g in guesses:
            st.markdown(f"""
        <div style="background:#2d0000;border:1px solid #7f0000;border-radius:6px;
                    padding:8px 14px;margin-bottom:6px;color:#ff6b6b;">
            ✗ &nbsp; {g}
        </div>
        """, unsafe_allow_html=True)

    # Win / Lose state
    if st.session_state.aw_won:
        st.success(f"🎉 You got it! **{mystery['name']}** — in {num_guesses} guess{'es' if num_guesses != 1 else ''}!")
        st.markdown(f"""
    <div style="background:#1a2d1a;border:1px solid #2ecc71;border-radius:10px;padding:16px 20px;margin-top:8px;">
        <b style="font-size:1.2rem;color:#2ecc71;">{mystery['name']}</b><br>
        <span style="color:#aaa;">{str(mystery['primary_role']).title()} &nbsp;|&nbsp;
        {int(mystery['title_count'])} titles &nbsp;|&nbsp;
        Avg IMDb: {float(mystery['avg_imdb']):.1f}</span>
    </div>
    """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🎲 New Game", type="secondary", key="aw_new_game"):
            for k in ["aw_seed", "aw_mystery", "aw_guesses", "aw_won", "aw_lost"]:
                st.session_state.pop(k, None)

    elif st.session_state.aw_lost:
        st.error(f"😞 Out of guesses! The answer was **{mystery['name']}**.")
        st.markdown(f"""
    <div style="background:#2d1a1a;border:1px solid #e74c3c;border-radius:10px;padding:16px 20px;margin-top:8px;">
        <b style="font-size:1.2rem;color:#e74c3c;">{mystery['name']}</b><br>
        <span style="color:#aaa;">{str(mystery['primary_role']).title()} &nbsp;|&nbsp;
        {int(mystery['title_count'])} titles &nbsp;|&nbsp;
        Avg IMDb: {float(mystery['avg_imdb']):.1f}</span>
    </div>
    """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🎲 New Game", type="secondary", key="aw_new_game"):
            for k in ["aw_seed", "aw_mystery", "aw_guesses", "aw_won", "aw_lost"]:
                st.session_state.pop(k, None)

    st.markdown("""
<div style="text-align:center;color:#555;font-size:0.8rem;padding:24px 0 8px;">
    Prototype — using real data from the Streaming Merger catalog
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3: HEAD-TO-HEAD ARENA
# ═══════════════════════════════════════════════════════════════════════════════

with tab_arena:
    st.markdown("""
    <div style="padding:8px 0 12px;">
        <div style="font-size:1.5em;font-weight:800;color:#fff;">⚔️ Head-to-Head Arena</div>
        <div style="color:#888;font-size:0.85em;margin-top:4px;">
            Side-by-side stat battle between any two people in the catalog</div>
    </div>""", unsafe_allow_html=True)

    arena_pids = arena_pool["person_id"].tolist()

    def fmt_arena(pid):
        r = arena_pool[arena_pool["person_id"] == pid]
        if r.empty:
            return pid
        row = r.iloc[0]
        plat = PLAT_LABELS.get(row.get("primary_platform", ""), "")
        return f"{row['name']}  ({str(row['primary_role']).title()}, {plat}, {int(row['title_count'])} titles)"

    # Random Battle button
    col_rand, _ = st.columns([1, 5])
    with col_rand:
        if st.button("🎲 Random Battle", type="primary", key="arena_random"):
            picks = random.sample(arena_pids, 2)
            st.session_state.arena_a = picks[0]
            st.session_state.arena_b = picks[1]
            st.rerun()

    st.markdown("---")

    # Person selectors
    col_a, col_vs, col_b = st.columns([5, 1, 5])
    with col_a:
        st.markdown(f"<div style='color:{ARENA_TEAL};font-weight:700;margin-bottom:6px;'>Player A</div>",
                    unsafe_allow_html=True)
        pid_a = st.selectbox(
            "Person A", options=arena_pids, format_func=fmt_arena,
            index=arena_pids.index(st.session_state.arena_a)
                  if st.session_state.arena_a in arena_pids else 0,
            placeholder="Search for a person…", label_visibility="collapsed", key="arena_sel_a",
        )
        st.session_state.arena_a = pid_a

    with col_vs:
        st.markdown(f"""
        <div style="display:flex;align-items:center;justify-content:center;
                    height:100%;padding-top:28px;">
            <div style="background:#2a2a3e;border:2px solid #444;border-radius:50%;
                        width:48px;height:48px;display:flex;align-items:center;
                        justify-content:center;font-size:0.85em;font-weight:900;
                        color:#fff;">VS</div>
        </div>""", unsafe_allow_html=True)

    with col_b:
        st.markdown(f"<div style='color:{ARENA_GOLD};font-weight:700;margin-bottom:6px;'>Player B</div>",
                    unsafe_allow_html=True)
        pid_b = st.selectbox(
            "Person B", options=arena_pids, format_func=fmt_arena,
            index=arena_pids.index(st.session_state.arena_b)
                  if st.session_state.arena_b in arena_pids else min(1, len(arena_pids) - 1),
            placeholder="Search for a person…", label_visibility="collapsed", key="arena_sel_b",
        )
        st.session_state.arena_b = pid_b

    # Battle render
    if pid_a and pid_b:
        row_a = arena_pool[arena_pool["person_id"] == pid_a].iloc[0]
        row_b = arena_pool[arena_pool["person_id"] == pid_b].iloc[0]

        if pid_a == pid_b:
            st.warning("Pick two different people for a battle!")
        else:
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            c_a, _, c_b = st.columns([5, 1, 5])
            with c_a:
                arena_person_card(row_a, "left")
            with c_b:
                arena_person_card(row_b, "right")

            # Table header
            st.markdown(f"""
            <div style="display:grid;grid-template-columns:1fr 160px 1fr;
                        align-items:center;padding:10px 16px;margin-top:16px;
                        background:#16162a;border-radius:8px 8px 0 0;
                        border-bottom:2px solid {ARENA_TEAL};">
                <div style="text-align:right;font-size:0.9em;font-weight:700;
                            color:{ARENA_TEAL};">{row_a['name']}</div>
                <div style="text-align:center;font-size:0.75em;color:#555;
                            text-transform:uppercase;letter-spacing:1px;">Stat</div>
                <div style="text-align:left;font-size:0.9em;font-weight:700;
                            color:{ARENA_GOLD};">{row_b['name']}</div>
            </div>
            <div style="background:{CARD_BG};border:1px solid {CARD_BORDER};
                        border-top:none;border-radius:0 0 8px 8px;margin-bottom:8px;">
            """, unsafe_allow_html=True)

            stats = arena_stat_rows(row_a, row_b)
            arena_stat_table(stats)
            st.markdown("</div>", unsafe_allow_html=True)

            wins_a = sum(1 for s in stats if s["winner"] == "a")
            wins_b = sum(1 for s in stats if s["winner"] == "b")
            total  = len([s for s in stats if s["winner"] != "tie"])

            if wins_a > wins_b:
                w_name, w_wins = row_a["name"], wins_a
            elif wins_b > wins_a:
                w_name, w_wins = row_b["name"], wins_b
            else:
                w_name, w_wins = "Tie!", (wins_a + wins_b) // 2

            arena_winner_card(w_name, w_wins, total, arena_merger_insight(row_a, row_b, w_name))

            st.markdown(f"""
            <div style="display:flex;gap:8px;align-items:center;
                        margin-top:16px;font-size:0.85em;color:#888;">
                <span style="color:{ARENA_TEAL};font-weight:700;">{row_a['name']}: {wins_a} wins</span>
                <span style="flex:1;height:6px;background:#2a2a3e;border-radius:3px;overflow:hidden;">
                    <div style="width:{int(wins_a/(wins_a+wins_b+0.001)*100)}%;
                                height:100%;background:{ARENA_TEAL};border-radius:3px;"></div>
                </span>
                <span style="color:{ARENA_GOLD};font-weight:700;">{wins_b} wins: {row_b['name']}</span>
            </div>""", unsafe_allow_html=True)
