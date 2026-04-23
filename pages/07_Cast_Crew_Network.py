"""
Cast & Crew — Collaboration Network · Actor Wordle · Head-to-Head Arena
"""

import os
import random
import tempfile
from collections import deque

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

from src.data.loaders import load_all_platforms_credits, load_all_platforms_titles, load_person_stats

# ── Platform colours (shared across all tabs) ─────────────────────────────────
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
SEED_COLOR    = "#FFFFFF"   # white star — distinct from all platform yellows


def get_head_to_head_stats(pid, person_stats, credits_df, titles_df):
    pid = str(pid)
    row = person_stats[person_stats["person_id"] == pid]
    if row.empty:
        return None
    p = row.iloc[0]

    person_credits = credits_df[credits_df["person_id"] == pid].copy()
    titles_slim = titles_df[["id","title","imdb_score","genres","platform"]].rename(
        columns={"id": "title_id", "platform": "title_platform"}
    )
    merged = person_credits.merge(titles_slim, on="title_id", how="left")

    avg_imdb    = float(p.get("avg_imdb", 0) or 0)
    title_count = int(p.get("title_count", 0) or 0)
    influence   = float(p.get("influence_score", 0) or 0)

    best_row = merged.nlargest(1, "imdb_score") if "imdb_score" in merged.columns else pd.DataFrame()
    best_title = best_row.iloc[0]["title"] if not best_row.empty else None
    best_imdb  = round(float(best_row.iloc[0]["imdb_score"]), 1) if not best_row.empty else None

    genre_diversity = 0
    if not merged.empty and "genres" in merged.columns:
        all_genres = set()
        for g in merged["genres"].dropna():
            if isinstance(g, (list, set)):
                all_genres.update(str(x) for x in g)
            elif isinstance(g, str):
                all_genres.add(g)
        genre_diversity = len(all_genres)

    platform_list = p.get("platform_list", [])
    if platform_list is not None and len(platform_list) > 0:
        platform_diversity = len(set(str(x) for x in platform_list))
    else:
        platform_diversity = 0

    plat_counts = merged.groupby("title_platform").size() if (not merged.empty and "title_platform" in merged.columns) else pd.Series(dtype=int)
    primary_platform = plat_counts.idxmax() if not plat_counts.empty else "Unknown"
    netflix_count = int(plat_counts.get("netflix", 0))
    max_count     = int(plat_counts.get("max", 0))

    career_start = p.get("career_start")
    career_end   = p.get("career_end")
    career_span  = 0
    career_str   = "N/A"
    if pd.notna(career_start) and pd.notna(career_end) and career_end > 0:
        career_span = int(career_end) - int(career_start)
        career_str  = f"{int(career_start)}–{int(career_end)}"

    return {
        "name":               str(p["name"]),
        "primary_role":       str(p.get("primary_role", "Unknown")).title(),
        "primary_platform":   primary_platform,
        "avg_imdb":           avg_imdb,
        "title_count":        title_count,
        "genre_diversity":    genre_diversity,
        "platform_diversity": platform_diversity,
        "career_span":        career_span,
        "career_str":         career_str,
        "netflix_count":      netflix_count,
        "max_count":          max_count,
        "influence_score":    influence,
        "best_title":         best_title,
        "best_imdb":          best_imdb,
    }

from src.config import INITIAL_SIDEBAR_STATE, LAYOUT, PAGE_ICON, PAGE_TITLE
from src.ui.badges import page_header_html
st.set_page_config(page_title=f"Cast & Crew Network | {PAGE_TITLE}", page_icon=PAGE_ICON, layout=LAYOUT, initial_sidebar_state=INITIAL_SIDEBAR_STATE)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
PERSON_STATS_PATH = PROJECT_ROOT / "data" / "precomputed" / "network" / "person_stats.parquet"

# ── Shared data loading ────────────────────────────────────────────────────────
@st.cache_resource
def _load_credits():
    df = load_all_platforms_credits()
    df["person_id"] = df["person_id"].astype(str)
    return df

@st.cache_resource
def _load_titles():
    return load_all_platforms_titles()

@st.cache_resource
def _load_person_stats():
    return load_person_stats()

@st.cache_resource
def load_wordle_people():
    df = pd.read_parquet(PERSON_STATS_PATH)
    df = df[
        (df["primary_role"] == "ACTOR")
        & (df["title_count"] >= 5)
        & (df["avg_imdb"] >= 6.0)
    ].copy()
    df["name"] = df["name"].astype(str)
    df = df.nlargest(750, "influence_score")
    return df.reset_index(drop=True)

@st.cache_data
def get_cached_stats(pid, _person_stats, _credits, _titles):
    return get_head_to_head_stats(pid, _person_stats, _credits, _titles)


# ── Network-specific loaders ───────────────────────────────────────────────────
@st.cache_data
def _load_network_edges() -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "precomputed" / "network" / "edges.parquet"
    df = pd.read_parquet(path)
    df["person_a"] = df["person_a"].astype(str)
    df["person_b"] = df["person_b"].astype(str)
    return df

@st.cache_data
def _load_net_person_stats() -> pd.DataFrame:
    df = pd.read_parquet(PERSON_STATS_PATH)
    df["person_id"] = df["person_id"].astype(str)
    return df

@st.cache_data
def _compute_primary_platform() -> pd.DataFrame:
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
def _build_adjacency() -> dict:
    edges_df = _load_network_edges()
    adj: dict = {}
    for pa, pb, w in zip(edges_df["person_a"], edges_df["person_b"], edges_df["weight"]):
        adj.setdefault(pa, {})[pb] = int(w)
        adj.setdefault(pb, {})[pa] = int(w)
    return adj


def _bfs_subgraph(seed_id: str, adj: dict, max_nodes: int = 100, max_hops: int = 2):
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
def _render_pyvis(node_ids: frozenset, edge_triples: frozenset, seed_id: str) -> str:
    from pyvis.network import Network

    person_stats = _load_net_person_stats()
    primary_plat = _compute_primary_platform()
    credits      = _load_credits()
    titles       = _load_titles()

    sub = person_stats[person_stats["person_id"].isin(node_ids)].copy()
    sub = sub.drop_duplicates("person_id")
    sub = sub.merge(primary_plat, on="person_id", how="left")
    sub = sub.drop_duplicates("person_id")
    sub["primary_platform"] = sub["primary_platform"].fillna("unknown")

    # Deduplicate same name / two IDs — keep highest title_count
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

    # Subgraph-aware platform: colour each node by the platform of titles
    # it actually shares with neighbours in this subgraph (not globally).
    t_plat_map = (
        titles[["id", "platform"]].rename(columns={"id": "title_id"})
        .dropna(subset=["platform"])
        .set_index("title_id")["platform"].to_dict()
    )
    sub_credits = credits[credits["person_id"].isin(node_ids)][["person_id", "title_id"]]
    person_title_sets: dict = (
        sub_credits.groupby("person_id")["title_id"].apply(set).to_dict()
    )
    sub_adj: dict = {}
    for pa, pb, _ in edge_triples:
        sub_adj.setdefault(pa, set()).add(pb)
        sub_adj.setdefault(pb, set()).add(pa)

    def subgraph_platform(pid: str):
        my_titles = person_title_sets.get(pid, set())
        plat_counts: dict = {}
        for nbr in sub_adj.get(pid, set()):
            for tid in my_titles & person_title_sets.get(nbr, set()):
                plat = t_plat_map.get(tid)
                if plat:
                    plat_counts[plat] = plat_counts.get(plat, 0) + 1
        return max(plat_counts, key=plat_counts.get) if plat_counts else None

    tc_max = sub["title_count"].max()
    tc_min = sub["title_count"].min()

    def node_size(tc):
        return 22 if tc_max == tc_min else 10 + (tc - tc_min) / (tc_max - tc_min) * 38

    net = Network(height="650px", width="100%",
                  bgcolor="#1E1E2E", font_color="#cccccc", directed=False)
    net.barnes_hut(gravity=-9000, central_gravity=0.35,
                   spring_length=160, spring_strength=0.05, damping=0.09)

    for _, row in sub.iterrows():
        pid     = row["person_id"]
        is_seed = pid == seed_id
        sg_plat  = subgraph_platform(pid)
        plat_key = sg_plat if sg_plat else row["primary_platform"]
        color    = PLAT_COLORS.get(plat_key, DEFAULT_COLOR)
        plat_lbl = PLAT_LABELS.get(plat_key, plat_key.title() if plat_key else "Unknown")
        tooltip  = (
            f"{row['name']}\n"
            f"Role: {str(row['primary_role']).title()}\n"
            f"Platform: {plat_lbl}\n"
            f"Titles: {int(row['title_count'])}\n"
            f"Avg IMDb: {float(row['avg_imdb']):.1f}"
        )
        net.add_node(
            pid, label=row["name"], title=tooltip,
            size=node_size(row["title_count"]) * (0.8 if is_seed else 1),
            shape="star" if is_seed else "dot",
            color={"background": SEED_COLOR if is_seed else color,
                   "border":     SEED_COLOR if is_seed else color,
                   "highlight":  {"background": "#cccccc" if is_seed else "#FFD700",
                                  "border": "#ffffff"}},
            borderWidth=4 if is_seed else 1,
            font={"size": 11, "color": "#dddddd"},
        )

    for pa, pb, w in edge_triples:
        if pa in node_ids and pb in node_ids:
            net.add_edge(pa, pb, value=max(1, w), title=f"Shared titles: {w}",
                         color={"color": "rgba(255,255,255,0.12)",
                                "highlight": "rgba(255,255,255,0.6)"})

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


# ── Wordle helpers ─────────────────────────────────────────────────────────────
MAX_GUESSES = 6
CLUE_LABELS = [
    "Hint", "First name starts with", "Top genre",
    "Career era", "Streaming home", "One title they appeared in",
]

def _platform_clue(platform_list):
    """Return (clue_text, dominant_platform_key) based on title distribution."""
    if not isinstance(platform_list, (list, set)) or len(platform_list) == 0:
        return "Spread across multiple platforms", None
    counts = {}
    for p in platform_list:
        k = str(p).lower()
        counts[k] = counts.get(k, 0) + 1
    total = sum(counts.values())
    top = max(counts, key=counts.get)
    pct = counts[top] / total
    name = top.title()
    if pct >= 0.85:
        return f"Almost all titles on {name}", top
    elif pct >= 0.65:
        return f"Majority of titles on {name}", top
    elif pct >= 0.45:
        return f"Mostly on {name}, some other platforms", top
    else:
        return "Spread across multiple platforms", None

def pick_mystery(seed: int, people: pd.DataFrame) -> dict:
    row = people.sample(1, random_state=seed).iloc[0]
    platform_list = row.get("platform_list", [])
    platform_clue_text, dominant_platform = _platform_clue(platform_list)

    start = row.get("career_start")
    if pd.notna(start) and start:
        decade = (int(start) // 10) * 10
        era = f"Active since the {decade}s"
    else:
        era = "Career era unknown"

    first_letter = str(row["name"]).split()[0][0].upper()

    return {
        "name": row["name"],
        "person_id": row["person_id"],
        "clues": [
            "Guess the actor! Use the clues below to narrow it down.",
            f'"{first_letter}"',
            str(row.get("top_genre", "Unknown")).title(),
            era,
            platform_clue_text,
            "— revealed on final guess —",
        ],
        "avg_imdb":          row.get("avg_imdb", None),
        "title_count":       int(row.get("title_count", 0)),
        "primary_role":      str(row.get("primary_role", "")),
        "top_genre":         str(row.get("top_genre", "")),
        "career_start":      row.get("career_start", None),
        "platform_list":     platform_list,
        "dominant_platform": dominant_platform,
    }

def filter_candidates(mystery: dict, num_clues_shown: int, people: pd.DataFrame) -> list:
    filtered = people.copy()

    if num_clues_shown >= 2:
        first_letter = str(mystery["name"]).split()[0][0].upper()
        filtered = filtered[filtered["name"].str.split().str[0].str[0].str.upper() == first_letter]

    if num_clues_shown >= 3:
        filtered = filtered[filtered["top_genre"].str.lower() == mystery["top_genre"].lower()]

    if num_clues_shown >= 4 and mystery["career_start"] and pd.notna(mystery["career_start"]):
        target_decade = (int(mystery["career_start"]) // 10) * 10
        def same_decade(val):
            try:    return (int(val) // 10) * 10 == target_decade
            except: return False
        filtered = filtered[filtered["career_start"].apply(same_decade)]

    if num_clues_shown >= 5:
        target = mystery.get("dominant_platform")
        if target:
            def same_dominant(val):
                _, dom = _platform_clue(val)
                return dom == target
            filtered = filtered[filtered["platform_list"].apply(same_dominant)]

    return sorted(filtered["name"].unique().tolist())


# ── Load data ──────────────────────────────────────────────────────────────────
people       = load_wordle_people()
person_stats = _load_person_stats()
credits      = _load_credits()
titles       = _load_titles()
arena_pool   = person_stats[person_stats["title_count"] >= 3].copy()

# Network-specific (loaded once, cached)
net_person_stats = _load_net_person_stats()
adj              = _build_adjacency()
_role_lookup     = net_person_stats.set_index("person_id")["primary_role"].to_dict()

# ── Session state ──────────────────────────────────────────────────────────────
if "seed" not in st.session_state:
    st.session_state.seed = random.randint(0, 99999)
if "mystery" not in st.session_state:
    st.session_state.mystery = pick_mystery(st.session_state.seed, people)
if "guesses" not in st.session_state:
    st.session_state.guesses = []
if "won" not in st.session_state:
    st.session_state.won = False
if "lost" not in st.session_state:
    st.session_state.lost = False

# ── Page header + tabs ─────────────────────────────────────────────────────────
st.markdown(
    page_header_html(
        "Cast & Crew Network",
        "Explore collaboration networks, guess the mystery actor, and compare performers head-to-head.",
    ),
    unsafe_allow_html=True,
)

tab_net, tab_wordle, tab_arena = st.tabs(["Collaboration Network", "Actor Wordle", "Head-to-Head Arena"])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1: COLLABORATION NETWORK
# ══════════════════════════════════════════════════════════════════════════════

ALL_ROLES = ["ACTOR", "DIRECTOR", "WRITER", "PRODUCER", "COMPOSER", "CINEMATOGRAPHER", "EDITOR"]

with tab_net:
    st.markdown("##### Pick a person — explore their collaboration web up to 2 degrees of separation")

    connected_ids = set(adj.keys())
    default_seed  = (
        net_person_stats[net_person_stats["person_id"].isin(connected_ids)]
        .assign(_deg=lambda df: df["person_id"].map(lambda p: len(adj.get(p, {}))))
        .nlargest(1, "_deg")["person_id"].iloc[0]
    )

    seed_pool = (
        net_person_stats[net_person_stats["person_id"].isin(connected_ids)]
        .sort_values("title_count", ascending=False)
        [["person_id", "name", "primary_role", "title_count"]]
        .drop_duplicates("name", keep="first")
        .head(5000)
    )
    seed_ids = seed_pool["person_id"].tolist()
    _seed_label_map = {
        row.person_id: f"{row.name}  ({str(row.primary_role).title()}, {int(row.title_count)} titles)"
        for row in seed_pool.itertuples(index=False)
    }

    col_s, col_d, col_m = st.columns([4, 1, 1])
    with col_s:
        seed_pid = st.selectbox(
            "Seed person", options=seed_ids,
            format_func=lambda pid: _seed_label_map.get(pid, pid),
            index=seed_ids.index(default_seed) if default_seed in seed_ids else 0,
            placeholder="Type a name to search…", label_visibility="collapsed", key="net_seed",
        )
    with col_d:
        depth = st.selectbox("Degrees", [1, 2], index=1, key="net_depth",
            help="How many hops away from the seed person to include. 1 = direct collaborators only. 2 = collaborators of collaborators.")
    with col_m:
        max_nodes = st.selectbox("Max nodes", [50, 100, 150], index=1, key="net_max",
            help="Maximum number of people to show in the network. Lower = faster and less cluttered. Higher = more connections visible.")

    role_filter = st.multiselect(
        "Filter by role", options=ALL_ROLES, default=["ACTOR", "DIRECTOR"],
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
            ]
        )
        + " &nbsp; "
        + '<span style="color:#fff;font-size:1em;vertical-align:middle;">★</span>'
          '<span style="color:#aaa;font-size:0.78em;margin-left:3px;">Seed (white star)</span>',
        unsafe_allow_html=True,
    )

    with st.spinner("Building network…"):
        node_ids, edge_triples = _bfs_subgraph(seed_pid, adj, max_nodes=max_nodes, max_hops=depth)
        selected_roles  = set(role_filter)
        filtered_nodes  = {
            pid for pid in node_ids
            if pid == seed_pid or _role_lookup.get(pid, "").upper() in selected_roles
        }
        filtered_edges  = {(pa, pb, w) for pa, pb, w in edge_triples
                           if pa in filtered_nodes and pb in filtered_nodes}
        net_html = _render_pyvis(frozenset(filtered_nodes), frozenset(filtered_edges), seed_pid)

    components.html(net_html, height=670, scrolling=False)
    st.caption(
        f"**{len(filtered_nodes)} nodes** · **{len(filtered_edges)} edges** · "
        f"{depth}-hop neighbourhood · Node size = title count · Hover for details · "
        f"Color = platform of shared titles"
    )

    st.markdown("---")
    st.markdown("**Quick Stats — pick any visible node:**")
    visible = (
        net_person_stats[net_person_stats["person_id"].isin(filtered_nodes)]
        .sort_values("name")
        .drop_duplicates("name", keep="first")
    )
    vis_ids = visible["person_id"].tolist()
    _net_ps_idx = net_person_stats.set_index("person_id")
    _primary_plat_map = _compute_primary_platform().set_index("person_id")["primary_platform"].to_dict()

    selected_node = st.selectbox(
        "Pick a node", options=vis_ids,
        format_func=lambda pid: visible[visible["person_id"] == pid].iloc[0]["name"]
                                if not visible[visible["person_id"] == pid].empty else pid,
        index=vis_ids.index(seed_pid) if seed_pid in vis_ids else 0,
        label_visibility="collapsed", key="net_stats",
    )
    if selected_node:
        r     = _net_ps_idx.loc[selected_node]
        conns = len(adj.get(selected_node, {}))

        # Per-platform title count for this person
        person_credits = credits[credits["person_id"] == selected_node][["title_id"]]
        t_slim = titles[["id", "platform"]].rename(columns={"id": "title_id"})
        plat_counts = (
            person_credits.merge(t_slim, on="title_id", how="left")
            .groupby("platform").size()
            .sort_values(ascending=False)
            .to_dict()
        )

        # Top stat cards
        c1, c2, c3, c4 = st.columns(4)
        for col, lbl, val in [
            (c1, "Name",        r["name"]),
            (c2, "Role",        str(r["primary_role"]).title()),
            (c3, "Titles",      int(r["title_count"])),
            (c4, "Connections", conns),
        ]:
            col.markdown(
                f'<div style="background:#1E1E2E;border:1px solid #333;border-radius:8px;'
                f'padding:10px 14px;text-align:center;">'
                f'<div style="color:#666;font-size:0.7em;text-transform:uppercase;letter-spacing:1px;">{lbl}</div>'
                f'<div style="color:#fff;font-size:1em;font-weight:600;margin-top:4px;">{val}</div></div>',
                unsafe_allow_html=True,
            )

        # Platform breakdown legend
        if plat_counts:
            st.markdown("<div style='margin-top:10px;'>", unsafe_allow_html=True)
            plat_cols = st.columns(len(plat_counts))
            for i, (plat, count) in enumerate(plat_counts.items()):
                pkey   = str(plat).lower()
                pcolor = PLAT_COLORS.get(pkey, DEFAULT_COLOR)
                plabel = PLAT_LABELS.get(pkey, str(plat).title())
                plat_cols[i].markdown(
                    f'<div style="background:#1E1E2E;border:1px solid {pcolor}55;border-top:3px solid {pcolor};'
                    f'border-radius:8px;padding:10px 14px;text-align:center;">'
                    f'<div style="display:inline-block;width:10px;height:10px;border-radius:50%;'
                    f'background:{pcolor};margin-right:6px;vertical-align:middle;"></div>'
                    f'<span style="color:#aaa;font-size:0.78em;">{plabel}</span><br>'
                    f'<span style="color:#fff;font-size:1.1em;font-weight:700;">{count}</span>'
                    f'<span style="color:#666;font-size:0.75em;"> titles</span></div>',
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2: ACTOR WORDLE
# ══════════════════════════════════════════════════════════════════════════════

with tab_wordle:
    # Containers in visual order
    header_c = st.container()
    input_c  = st.container()  # guess input sits just below the header
    clues_c  = st.container()
    div1_c   = st.container()
    prev_c   = st.container()
    result_c = st.container()
    footer_c = st.container()

    # Process input FIRST so state is updated before render
    with input_c:
        mystery = st.session_state.mystery
        _num_g  = len(st.session_state.guesses)
        if not st.session_state.won and not st.session_state.lost and _num_g < MAX_GUESSES:
            _filt = filter_candidates(mystery, _num_g + 1, people)
            _filt = [n for n in _filt if n not in st.session_state.guesses]
            st.markdown(
                f"### Your Guess &nbsp;"
                f"<span style='color:#888;font-size:0.9rem;'>({MAX_GUESSES - _num_g} remaining)</span>"
                f"&nbsp;<span style='color:#f39c12;font-size:0.85rem;'>({len(_filt)} possible names)</span>",
                unsafe_allow_html=True,
            )
            _gi = st.selectbox(
                "Start typing a name...",
                options=[""] + _filt,
                index=0,
                key=f"guess_input_{_num_g}",
                label_visibility="collapsed",
            )
            if st.button("Submit Guess", type="primary", disabled=not _gi, key="wordle_submit"):
                _g = _gi.strip()
                st.session_state.guesses = st.session_state.guesses + [_g]
                if _g.lower() == mystery["name"].lower():
                    st.session_state.won = True
                elif len(st.session_state.guesses) >= MAX_GUESSES:
                    st.session_state.lost = True
                st.rerun()

    # Read fresh state
    mystery       = st.session_state.mystery
    guesses       = st.session_state.guesses
    num_guesses   = len(guesses)
    clues_to_show = mystery["clues"][:num_guesses + 1]

    with header_c:
        st.markdown("""
<div style="text-align:center;padding:24px 0 8px;">
    <h1 style="margin:4px 0;font-size:2rem;letter-spacing:1px;">Actor Wordle</h1>
    <p style="color:#aaa;font-size:0.95rem;">Guess the mystery person from the streaming catalog.<br>
    A new clue is revealed after each wrong guess.</p>
</div>
""", unsafe_allow_html=True)
        st.divider()

    with clues_c:
        st.markdown("### Clues So Far")
        for i, clue_text in enumerate(clues_to_show):
            if i == 5 and not st.session_state.won:
                try:
                    pid = str(mystery["person_id"])
                    person_titles = credits[credits["person_id"] == pid]
                    if not person_titles.empty:
                        match = titles[titles["id"] == person_titles.iloc[0]["title_id"]]
                        if not match.empty:
                            clue_text = f'"{match.iloc[0]["title"]}"'
                except Exception:
                    clue_text = "— title data unavailable —"

            if st.session_state.won:
                color  = "#0d3320"
                border = "#2ecc71"
                label_color = "#5dde8a"
            elif i == 0:
                color  = "#1a472a"
                border = "#2ecc71"
                label_color = "#888"
            else:
                color  = "#1a1a2e"
                border = "#4a4a8a"
                label_color = "#888"

            new_badge = ""
            glow = 'box-shadow:0 0 12px rgba(46,204,113,0.35);' if st.session_state.won else ''
            st.markdown(
                f'<div style="background:{color};border:1px solid {border};border-left:4px solid {border};'
                f'border-radius:8px;padding:12px 16px;margin-bottom:8px;{glow}">'
                f'<span style="color:{label_color};font-size:0.8rem;text-transform:uppercase;letter-spacing:1px;">'
                f'Clue {i+1}: {CLUE_LABELS[i]}</span>{new_badge}<br>'
                f'<span style="color:white;font-size:1.1rem;font-weight:600;">{clue_text}</span></div>',
                unsafe_allow_html=True,
            )
        if not st.session_state.won:
            for j in range(len(clues_to_show), MAX_GUESSES):
                st.markdown(
                    f'<div style="background:#111;border:1px solid #333;border-left:4px solid #333;'
                    f'border-radius:8px;padding:12px 16px;margin-bottom:8px;opacity:0.4;">'
                    f'<span style="color:#666;font-size:0.8rem;text-transform:uppercase;letter-spacing:1px;">'
                    f'Clue {j+1}: {CLUE_LABELS[j]}</span><br>'
                    f'<span style="color:#444;font-size:1.1rem;">🔒 Unlocks after guess {j}</span></div>',
                    unsafe_allow_html=True,
                )

    with div1_c:
        st.divider()

    with prev_c:
        wrong_guesses = guesses[:-1] if st.session_state.won else guesses
        if wrong_guesses:
            st.markdown("### ❌ Previous Guesses")
            for g in wrong_guesses:
                st.markdown(
                    f'<div style="background:#2d0000;border:1px solid #7f0000;border-radius:6px;'
                    f'padding:8px 14px;margin-bottom:6px;color:#ff6b6b;">✗ &nbsp; {g}</div>',
                    unsafe_allow_html=True,
                )

    with result_c:
        if st.session_state.won:
            if not st.session_state.get("_balloons_shown"):
                st.balloons()
                st.session_state["_balloons_shown"] = True
            guess_word = "guess" if num_guesses == 1 else "guesses"
            st.markdown(
                f'<div style="background:linear-gradient(135deg,#0a2e18,#0d4022);'
                f'border:2px solid #2ecc71;border-top:4px solid #2ecc71;border-radius:14px;'
                f'padding:32px 24px;margin-top:12px;text-align:center;'
                f'box-shadow:0 0 40px rgba(46,204,113,0.25);">'
                f''
                f'<div style="color:#2ecc71;font-size:0.75rem;text-transform:uppercase;letter-spacing:3px;margin-bottom:10px;">Correct!</div>'
                f'<div style="color:#ffffff;font-size:2rem;font-weight:900;letter-spacing:-0.5px;margin-bottom:6px;">{mystery["name"]}</div>'
                f'<div style="color:#5dde8a;font-size:1rem;margin-bottom:18px;">Solved in <b>{num_guesses}</b> {guess_word}</div>'
                f'<div style="display:inline-flex;gap:24px;justify-content:center;flex-wrap:wrap;'
                f'background:rgba(46,204,113,0.08);border-radius:10px;padding:12px 24px;">'
                f'<span style="color:#aaa;font-size:0.9rem;">{str(mystery["primary_role"]).title()}</span>'
                f'<span style="color:#aaa;font-size:0.9rem;">{int(mystery["title_count"])} titles</span>'
                f'<span style="color:#aaa;font-size:0.9rem;">Avg IMDb {float(mystery["avg_imdb"]):.1f}</span>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("New Game", type="primary", key="new_game_won"):
                for k in ["seed", "mystery", "guesses", "won", "lost", "_balloons_shown"]:
                    del st.session_state[k]
        elif st.session_state.lost:
            st.error(f"Out of guesses! The answer was **{mystery['name']}**.")
            st.markdown(
                f'<div style="background:#2d1a1a;border:1px solid #e74c3c;border-radius:10px;padding:16px 20px;margin-top:8px;">'
                f'<b style="font-size:1.2rem;color:#e74c3c;">{mystery["name"]}</b><br>'
                f'<span style="color:#aaa;">{str(mystery["primary_role"]).title()} &nbsp;|&nbsp;'
                f'{int(mystery["title_count"])} titles &nbsp;|&nbsp;'
                f'Avg IMDb: {float(mystery["avg_imdb"]):.1f}</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("New Game", type="secondary", key="new_game_lost"):
                for k in ["seed", "mystery", "guesses", "won", "lost", "_balloons_shown"]:
                    if k in st.session_state:
                        del st.session_state[k]

    with footer_c:
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3: HEAD-TO-HEAD ARENA
# ══════════════════════════════════════════════════════════════════════════════

with tab_arena:
    TEAL = "#00b4d8"
    GOLD = "#f39c12"

    st.markdown("""
<div style="text-align:center;padding:24px 0 8px;">
    <h1 style="margin:4px 0;font-size:2rem;letter-spacing:1px;">Head-to-Head Arena</h1>
    <p style="color:#aaa;font-size:0.95rem;">Pick any two people from the streaming catalog and see who wins.</p>
</div>
""", unsafe_allow_html=True)

    st.divider()

    if st.button("Random Battle", type="secondary", key="arena_random"):
        sample = arena_pool.sample(2, random_state=random.randint(0, 99999))
        st.session_state["arena_a"] = sample.iloc[0]["person_id"]
        st.session_state["arena_b"] = sample.iloc[1]["person_id"]
        st.session_state["arena_search_a"] = sample.iloc[0]["name"]
        st.session_state["arena_search_b"] = sample.iloc[1]["name"]

    _arena_ids   = arena_pool["person_id"].tolist()
    _arena_names = arena_pool.set_index("person_id")["name"].to_dict()

    col_a, col_vs, col_b = st.columns([5, 1, 5])
    with col_a:
        sel_a = st.selectbox(
            "Person A",
            options=[None] + _arena_ids,
            format_func=lambda pid: "Type to search…" if pid is None else _arena_names.get(pid, pid),
            key="arena_sel_a",
            placeholder="Type a name…",
        )
        if sel_a:
            st.session_state["arena_a"] = sel_a

    with col_vs:
        st.markdown("<div style='text-align:center;font-size:2.5rem;font-weight:900;color:#e74c3c;padding-top:28px;'>VS</div>", unsafe_allow_html=True)

    with col_b:
        sel_b = st.selectbox(
            "Person B",
            options=[None] + _arena_ids,
            format_func=lambda pid: "Type to search…" if pid is None else _arena_names.get(pid, pid),
            key="arena_sel_b",
            placeholder="Type a name…",
        )
        if sel_b:
            st.session_state["arena_b"] = sel_b

    pid_a = st.session_state.get("arena_a")
    pid_b = st.session_state.get("arena_b")

    if pid_a and pid_b and pid_a == pid_b:
        st.warning("Please select two different people.")

    elif pid_a and pid_b:
        stats_a = get_cached_stats(pid_a, person_stats, credits, titles)
        stats_b = get_cached_stats(pid_b, person_stats, credits, titles)

        if stats_a and stats_b:
            STAT_ICONS = {}

            def battle_row(label, val_a, val_b, higher_wins=True, is_str=False):
                if is_str or val_a is None or val_b is None:
                    win_a = win_b = False
                elif higher_wins:
                    win_a, win_b = val_a > val_b, val_b > val_a
                else:
                    win_a, win_b = val_a < val_b, val_b < val_a
                color_a = TEAL if win_a else ("#ccc" if not win_b else "#666")
                color_b = GOLD if win_b else ("#ccc" if not win_a else "#666")
                bg_a = "rgba(0,180,216,0.08)" if win_a else "transparent"
                bg_b = "rgba(243,156,18,0.08)" if win_b else "transparent"
                badge_a = ' <span style="font-size:0.75rem;color:#aaa;">W</span>' if win_a else ""
                badge_b = ' <span style="font-size:0.75rem;color:#aaa;">W</span>' if win_b else ""
                icon = STAT_ICONS.get(label, "")
                bar = ""
                if not is_str and val_a is not None and val_b is not None:
                    total = float(val_a or 0) + float(val_b or 0)
                    pct_a = (float(val_a) / total * 100) if total > 0 else 50
                    bar = (
                        f'<div style="height:3px;border-radius:2px;overflow:hidden;margin-top:6px;background:#1a1a1a;">'
                        f'<div style="width:{pct_a:.1f}%;height:100%;background:{TEAL};float:left;"></div>'
                        f'<div style="width:{100-pct_a:.1f}%;height:100%;background:{GOLD};float:left;"></div>'
                        f'</div>'
                    )
                return (
                    f'<tr>'
                    f'<td style="padding:14px 12px;border-bottom:1px solid #1e1e1e;text-align:center;vertical-align:middle;">'
                    f'<span style="color:#666;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.8px;">{label}</span></td>'
                    f'<td style="background:{bg_a};padding:14px 16px;border-bottom:1px solid #1e1e1e;text-align:right;vertical-align:middle;">'
                    f'<span style="color:{color_a};font-weight:{"700" if win_a else "400"};font-size:1rem;">{val_a}{badge_a}</span>{bar}</td>'
                    f'<td style="background:{bg_b};padding:14px 16px;border-bottom:1px solid #1e1e1e;text-align:left;vertical-align:middle;">'
                    f'<span style="color:{color_b};font-weight:{"700" if win_b else "400"};font-size:1rem;">{val_b}{badge_b}</span></td>'
                    f'</tr>'
                )

            comparisons = [
                ("Avg IMDb Score",    stats_a["avg_imdb"],       stats_b["avg_imdb"],       True),
                ("Total Titles",      stats_a["title_count"],    stats_b["title_count"],    True),
                ("Genre Diversity",   stats_a["genre_diversity"],stats_b["genre_diversity"],True),
                ("Platform Diversity",stats_a["platform_diversity"],stats_b["platform_diversity"],True),
                ("Career Span (yrs)", stats_a["career_span"],    stats_b["career_span"],    True),
                ("Netflix Titles",    stats_a["netflix_count"],  stats_b["netflix_count"],  True),
                ("Max Titles",        stats_a["max_count"],      stats_b["max_count"],      True),
                ("Influence Score",   round(stats_a["influence_score"]*10000,2), round(stats_b["influence_score"]*10000,2), True),
            ]

            wins_a = wins_b = 0
            for _, va, vb, hw in comparisons:
                wa = 1 if (va is not None and vb is not None and (va > vb if hw else va < vb)) else 0
                wb = 1 if (va is not None and vb is not None and (vb > va if hw else vb < va)) else 0
                wins_a += wa; wins_b += wb

            overall_winner = stats_a["name"] if wins_a > wins_b else (stats_b["name"] if wins_b > wins_a else "Tie")

            if stats_a["platform_diversity"] > stats_b["platform_diversity"]:
                merger_insight = f"<b>{stats_a['name']}</b> spans more platforms, making them higher-value in a merger context."
            elif stats_b["platform_diversity"] > stats_a["platform_diversity"]:
                merger_insight = f"<b>{stats_b['name']}</b> spans more platforms, making them higher-value in a merger context."
            elif stats_a["influence_score"] > stats_b["influence_score"]:
                merger_insight = f"<b>{stats_a['name']}</b> has a higher network influence score — more central to the talent graph."
            else:
                merger_insight = f"<b>{stats_b['name']}</b> has a higher network influence score — more central to the talent graph."

            rows_html = "".join([
                battle_row("Primary Role",     stats_a["primary_role"],  stats_b["primary_role"],  is_str=True),
                battle_row("Primary Platform", stats_a["primary_platform"], stats_b["primary_platform"], is_str=True),
                battle_row("Career",           stats_a["career_str"],    stats_b["career_str"],    is_str=True),
                battle_row("Best Title",
                    f"{stats_a['best_title']} ({stats_a['best_imdb']})" if stats_a["best_title"] else "—",
                    f"{stats_b['best_title']} ({stats_b['best_imdb']})" if stats_b["best_title"] else "—",
                    is_str=True),
                *[battle_row(lbl, va, vb, hw) for lbl, va, vb, hw in comparisons],
            ])

            score_badge_a = f'<div style="display:inline-block;background:rgba(0,180,216,0.15);color:{TEAL};font-size:0.75rem;padding:2px 8px;border-radius:12px;margin-top:4px;">{wins_a} wins</div>'
            score_badge_b = f'<div style="display:inline-block;background:rgba(243,156,18,0.15);color:{GOLD};font-size:0.75rem;padding:2px 8px;border-radius:12px;margin-top:4px;">{wins_b} wins</div>'

            table_html = (
                f'<table style="width:100%;border-collapse:collapse;margin-top:24px;border-radius:12px;overflow:hidden;border:1px solid #222;">'
                f'<thead><tr style="background:#0d1117;">'
                f'<th style="color:#555;font-size:0.7rem;text-transform:uppercase;letter-spacing:1px;padding:14px 12px;text-align:center;width:28%;border-bottom:2px solid #222;">Stat</th>'
                f'<th style="padding:14px 16px;text-align:right;width:36%;border-bottom:2px solid {TEAL}40;">'
                f'<div style="color:{TEAL};font-size:1.15rem;font-weight:700;">{stats_a["name"]}</div>{score_badge_a}</th>'
                f'<th style="padding:14px 16px;text-align:left;width:36%;border-bottom:2px solid {GOLD}40;">'
                f'<div style="color:{GOLD};font-size:1.15rem;font-weight:700;">{stats_b["name"]}</div>{score_badge_b}</th>'
                f'</tr></thead>'
                f'<tbody style="background:#0a0a0f;">{rows_html}</tbody>'
                f'</table>'
            )
            st.markdown(table_html, unsafe_allow_html=True)

            winner_color = TEAL if overall_winner == stats_a["name"] else (GOLD if overall_winner == stats_b["name"] else "#aaa")
            winner_label = overall_winner if overall_winner != "Tie" else "It's a Tie!"
            score_line   = f"{wins_a} – {wins_b} stat categories" if overall_winner != "Tie" else "Equal across all categories"
            st.markdown(
                f'<div style="background:linear-gradient(135deg,#0d1117,#0d1b2a);border:1px solid {winner_color}55;border-top:3px solid {winner_color};border-radius:12px;padding:24px;margin-top:20px;text-align:center;">'
                f'<div style="color:#555;font-size:0.7rem;text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;">Overall Winner</div>'
                f'<div style="color:{winner_color};font-size:2.2rem;font-weight:800;">{winner_label}</div>'
                f'<div style="color:#555;font-size:0.85rem;margin:8px 0 16px;">{score_line}</div>'
                f'<div style="color:#888;font-size:0.85rem;padding-top:14px;border-top:1px solid #1e1e1e;"><em>Merger Insight:</em> {merger_insight}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

# ── footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<div style="color:#555;font-size:0.8em;text-align:center;padding:8px 0 16px;">'
    'Hypothetical merger for academic analysis. '
    'Data is a snapshot (mid-2023). '
    'All insights are illustrative, not prescriptive. '
    'As of Feb 26, 2026, Netflix withdrew from this acquisition.'
    '</div>',
    unsafe_allow_html=True,
)
