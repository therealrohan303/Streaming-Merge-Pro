"""
Cast & Crew — New Sections Test Page
Combines: Collaboration Network Visualizer + CastleWordle
Run: streamlit run test_cast_crew.py
"""

import hashlib
import os
import tempfile
from collections import deque
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Cast & Crew — Test", page_icon="🎬", layout="wide")

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

# CastleWordle colour scheme
CW_GREEN  = "#1db954"
CW_YELLOW = "#f0a500"
CW_RED    = "#c0392b"
CW_MAX_GUESSES = 6
CW_MIN_TITLES  = 5


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

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
    credits = pd.read_parquet("data/processed/all_platforms_credits.parquet")
    titles  = pd.read_parquet("data/processed/all_platforms_titles.parquet")
    credits = credits.copy()
    credits["person_id"] = credits["person_id"].astype(str)
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
def build_adjacency(edges_df: pd.DataFrame) -> dict:
    adj: dict = {}
    for pa, pb, w in zip(edges_df["person_a"], edges_df["person_b"], edges_df["weight"]):
        adj.setdefault(pa, {})[pb] = int(w)
        adj.setdefault(pb, {})[pa] = int(w)
    return adj


@st.cache_data
def build_player_pool() -> pd.DataFrame:
    person_stats = pd.read_parquet("data/precomputed/network/person_stats.parquet")
    credits      = pd.read_parquet("data/processed/all_platforms_credits.parquet")
    titles       = pd.read_parquet("data/processed/all_platforms_titles.parquet")

    credits["person_id"]      = credits["person_id"].astype(str)
    person_stats["person_id"] = person_stats["person_id"].astype(str)

    pool = person_stats[person_stats["title_count"] >= CW_MIN_TITLES].copy()

    titles_slim = titles[["id", "platform", "genres"]].rename(columns={"id": "title_id"})
    cred_merged = credits[["person_id", "title_id"]].merge(titles_slim, on="title_id", how="left")

    primary_plat = (
        cred_merged.groupby(["person_id", "platform"]).size()
        .reset_index(name="cnt")
        .sort_values("cnt", ascending=False)
        .drop_duplicates("person_id")[["person_id", "platform"]]
        .rename(columns={"platform": "primary_platform"})
    )
    pool = pool.merge(primary_plat, on="person_id", how="left")

    def to_list(g):
        if isinstance(g, (list, np.ndarray)):
            return [str(x) for x in g if x]
        return []

    cred_merged["genres_list"] = cred_merged["genres"].apply(to_list)
    exploded = cred_merged.explode("genres_list").dropna(subset=["genres_list"])
    exploded = exploded[exploded["genres_list"] != ""]
    genre_sets = (
        exploded.groupby("person_id")["genres_list"]
        .apply(set).reset_index()
        .rename(columns={"genres_list": "genre_set"})
    )
    pool = pool.merge(genre_sets, on="person_id", how="left")
    pool["active_decade"] = pool["career_end"].apply(
        lambda y: f"{int(y) // 10 * 10}s" if pd.notna(y) and y > 0 else None
    )
    pool = pool.dropna(subset=["top_genre", "avg_imdb", "primary_platform",
                                "primary_role", "active_decade"])
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


def render_pyvis(node_ids: set, edge_triples: set,
                 person_stats: pd.DataFrame, primary_plat: pd.DataFrame,
                 seed_id: str) -> str:
    from pyvis.network import Network

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
#  CASTLEWORDLE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def get_daily_person(pool: pd.DataFrame) -> pd.Series:
    hash_int = int(hashlib.md5(date.today().isoformat().encode()).hexdigest(), 16)
    return pool.iloc[hash_int % len(pool)]


def cw_compare(guess: pd.Series, target: pd.Series) -> list[dict]:
    results = []

    # Platform
    match = guess["primary_platform"] == target["primary_platform"]
    results.append({"label": "Platform",
                    "value": PLAT_LABELS.get(guess["primary_platform"], guess["primary_platform"]),
                    "color": CW_GREEN if match else CW_RED, "hint": ""})

    # Role
    match = guess["primary_role"] == target["primary_role"]
    results.append({"label": "Role", "value": guess["primary_role"].title(),
                    "color": CW_GREEN if match else CW_RED, "hint": ""})

    # Genre
    g_top = str(guess["top_genre"]).lower()
    t_top = str(target["top_genre"]).lower()
    g_set = guess["genre_set"] if isinstance(guess["genre_set"], set) else set()
    t_set = target["genre_set"] if isinstance(target["genre_set"], set) else set()
    if g_top == t_top:
        gc, gh = CW_GREEN, ""
    elif g_set & t_set:
        gc, gh = CW_YELLOW, "~"
    else:
        gc, gh = CW_RED, ""
    results.append({"label": "Genre", "value": g_top.title(), "color": gc, "hint": gh})

    # Title Count
    diff = int(guess["title_count"]) - int(target["title_count"])
    if diff == 0:
        cc, ch = CW_GREEN, ""
    elif abs(diff) <= 5:
        cc, ch = CW_YELLOW, "⬆️" if diff < 0 else "⬇️"
    else:
        cc, ch = CW_RED, "⬆️" if diff < 0 else "⬇️"
    results.append({"label": "Titles", "value": str(int(guess["title_count"])),
                    "color": cc, "hint": ch})

    # IMDb
    diff_i = round(float(guess["avg_imdb"]), 1) - round(float(target["avg_imdb"]), 1)
    if abs(diff_i) <= 0.3:
        ic, ih = CW_GREEN, ""
    elif abs(diff_i) <= 1.0:
        ic, ih = CW_YELLOW, "⬆️" if diff_i < 0 else "⬇️"
    else:
        ic, ih = CW_RED, "⬆️" if diff_i < 0 else "⬇️"
    results.append({"label": "IMDb", "value": f"{round(float(guess['avg_imdb']), 1):.1f}",
                    "color": ic, "hint": ih})

    # Decade
    match = guess["active_decade"] == target["active_decade"]
    results.append({"label": "Decade", "value": guess["active_decade"],
                    "color": CW_GREEN if match else CW_RED, "hint": ""})

    return results


def render_guess_row(name: str, comparison: list[dict], is_win: bool = False):
    tiles = ""
    for a in comparison:
        tiles += f"""
        <div style="background:{a['color']};border-radius:6px;padding:8px 4px;
                    text-align:center;min-width:90px;flex:1;">
            <div style="font-size:0.65em;color:rgba(255,255,255,0.75);margin-bottom:2px;">
                {a['label']}</div>
            <div style="font-size:0.9em;font-weight:700;color:#fff;">
                {a['value']} {a['hint']}</div>
        </div>"""
    border = f"2px solid {CW_GREEN}" if is_win else f"1px solid {CARD_BORDER}"
    st.markdown(f"""
    <div style="margin-bottom:8px;">
        <div style="font-size:0.85em;color:{CARD_TEXT};margin-bottom:4px;padding-left:2px;">
            {"🎯 " if is_win else ""}<strong>{name}</strong></div>
        <div style="display:flex;gap:6px;flex-wrap:nowrap;background:{CARD_BG};
                    padding:8px;border-radius:8px;border:{border};">{tiles}</div>
    </div>""", unsafe_allow_html=True)


def render_profile_card(person: pd.Series, label: str = ""):
    plat_str = PLAT_LABELS.get(person["primary_platform"], person["primary_platform"])
    career   = (f"{int(person['career_start'])}–{int(person['career_end'])}"
                if pd.notna(person.get("career_start")) and pd.notna(person.get("career_end"))
                else "N/A")
    st.markdown(f"""
    <div style="background:{CARD_BG};border:1px solid {CW_GREEN};border-radius:12px;
                padding:20px;margin-top:12px;">
        <div style="font-size:0.75em;color:{CW_GREEN};text-transform:uppercase;
                    letter-spacing:1px;margin-bottom:4px;">{label}</div>
        <div style="font-size:1.5em;font-weight:800;color:#fff;margin-bottom:12px;">
            🎬 {person['name']}</div>
        <div style="display:flex;gap:12px;flex-wrap:wrap;">
            <span style="background:#2a2a3e;padding:6px 12px;border-radius:20px;
                         color:{CARD_TEXT};font-size:0.85em;">🎭 {str(person['primary_role']).title()}</span>
            <span style="background:#2a2a3e;padding:6px 12px;border-radius:20px;
                         color:{CARD_TEXT};font-size:0.85em;">📺 {plat_str}</span>
            <span style="background:#2a2a3e;padding:6px 12px;border-radius:20px;
                         color:{CARD_TEXT};font-size:0.85em;">🎞️ {int(person['title_count'])} titles</span>
            <span style="background:#2a2a3e;padding:6px 12px;border-radius:20px;
                         color:{CARD_TEXT};font-size:0.85em;">⭐ IMDb {float(person['avg_imdb']):.1f}</span>
            <span style="background:#2a2a3e;padding:6px 12px;border-radius:20px;
                         color:{CARD_TEXT};font-size:0.85em;">🎬 {str(person['top_genre']).title()}</span>
            <span style="background:#2a2a3e;padding:6px 12px;border-radius:20px;
                         color:{CARD_TEXT};font-size:0.85em;">📅 {career}</span>
        </div>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD DATA (shared, cached)
# ═══════════════════════════════════════════════════════════════════════════════

with st.spinner("Loading data…"):
    edges_df, person_stats = load_network_data()
    primary_plat           = compute_primary_platform()
    adj                    = build_adjacency(edges_df)
    pool                   = build_player_pool()

# CastleWordle session state — init/reset on new day
today_iso = date.today().isoformat()
if st.session_state.get("cw_date") != today_iso:
    st.session_state.cw_date      = today_iso
    st.session_state.cw_guesses   = []
    st.session_state.cw_game_over = False
    st.session_state.cw_won       = False

mystery = get_daily_person(pool)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE HEADER + TABS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="padding:16px 0 4px;">
    <div style="font-size:2em;font-weight:900;color:#fff;">🎬 Cast & Crew — New Sections</div>
    <div style="color:#888;font-size:0.9em;margin-top:4px;">Test build · Not yet in the main app</div>
</div>
""", unsafe_allow_html=True)

tab_net, tab_cw = st.tabs(["🕸️ Collaboration Network", "🏰 CastleWordle"])


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

        net_html = render_pyvis(filtered_nodes, filtered_edges, person_stats, primary_plat, seed_pid)

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
#  TAB 2: CASTLEWORDLE
# ═══════════════════════════════════════════════════════════════════════════════

with tab_cw:
    st.markdown("""
    <div style="text-align:center;padding:16px 0 8px;">
        <div style="font-size:2.2em;font-weight:900;color:#fff;letter-spacing:2px;">
            🏰 CastleWordle</div>
        <div style="color:#888;font-size:0.9em;margin-top:4px;">
            Guess today's mystery streaming talent — new puzzle every day</div>
    </div>""", unsafe_allow_html=True)

    with st.expander("How to Play"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
**Goal:** Guess the mystery person in 6 tries.

**After each guess**, each attribute is color-coded:
- 🟩 **Green** — exact match
- 🟨 **Yellow** — close (genre overlap, titles within 5, IMDb within 1.0)
- 🟥 **Red** — no match

**Attributes:** Platform · Role · Genre · Titles · IMDb · Decade
            """)
        with col_b:
            st.markdown("""
**Tips:**
- Mystery person has at least **5 titles** in the catalog
- **⬆️ / ⬇️** tell you if the answer is higher or lower
- New puzzle every day at midnight
- Use **Share Result** to copy your emoji grid
            """)

    st.divider()

    n_guesses = len(st.session_state.cw_guesses)
    game_over = st.session_state.cw_game_over
    won       = st.session_state.cw_won

    counter_color = CW_GREEN if won else (CW_RED if game_over else CARD_TEXT)
    st.markdown(f"""
    <div style="text-align:center;margin-bottom:12px;">
        <span style="font-size:1.1em;font-weight:700;color:{counter_color};">
            Guess {n_guesses}/{CW_MAX_GUESSES}</span>
    </div>""", unsafe_allow_html=True)

    # Previous guesses
    guesses_data = []
    for pid in st.session_state.cw_guesses:
        row = pool[pool["person_id"] == pid]
        if not row.empty:
            p    = row.iloc[0]
            comp = cw_compare(p, mystery)
            guesses_data.append((p, comp))
            render_guess_row(p["name"], comp, is_win=(p["person_id"] == mystery["person_id"]))

    # Input
    if not game_over:
        already   = set(st.session_state.cw_guesses)
        remaining = pool[~pool["person_id"].isin(already)]
        options   = remaining.sort_values("name")["person_id"].tolist()

        def fmt_option(pid):
            r = remaining[remaining["person_id"] == pid]
            if r.empty:
                return pid
            row = r.iloc[0]
            return (f"{row['name']}  ({row['primary_role'].title()}, "
                    f"{PLAT_LABELS.get(row['primary_platform'], row['primary_platform'])}, "
                    f"{int(row['title_count'])} titles)")

        col_sel, col_btn, col_give = st.columns([4, 1, 1])
        with col_sel:
            selected_pid = st.selectbox(
                "Search", options=options, format_func=fmt_option,
                index=None, placeholder="Type a name to search…",
                key="cw_select", label_visibility="collapsed",
            )
        with col_btn:
            submit = st.button("Guess", type="primary", use_container_width=True,
                               disabled=selected_pid is None)
        with col_give:
            give_up = st.button("Give Up", use_container_width=True)

        if submit and selected_pid:
            st.session_state.cw_guesses.append(selected_pid)
            if selected_pid == mystery["person_id"]:
                st.session_state.cw_won      = True
                st.session_state.cw_game_over = True
            elif len(st.session_state.cw_guesses) >= CW_MAX_GUESSES:
                st.session_state.cw_game_over = True
            st.rerun()

        if give_up:
            st.session_state.cw_game_over = True
            st.session_state.cw_won       = False
            st.rerun()

    # Win / Loss
    if game_over:
        if won:
            st.balloons()
            st.markdown(f"""
            <div style="text-align:center;padding:16px;background:rgba(29,185,84,0.15);
                        border:1px solid {CW_GREEN};border-radius:12px;margin:12px 0;">
                <div style="font-size:1.8em;font-weight:900;color:{CW_GREEN};">🎉 You got it!</div>
                <div style="color:#aaa;margin-top:4px;">
                    Solved in {len(st.session_state.cw_guesses)}/{CW_MAX_GUESSES} guesses</div>
            </div>""", unsafe_allow_html=True)
            render_profile_card(mystery, label="Today's Mystery Person")
        else:
            st.markdown(f"""
            <div style="text-align:center;padding:16px;background:rgba(192,57,43,0.15);
                        border:1px solid {CW_RED};border-radius:12px;margin:12px 0;">
                <div style="font-size:1.8em;font-weight:900;color:{CW_RED};">😔 Better luck tomorrow!</div>
            </div>""", unsafe_allow_html=True)
            render_profile_card(mystery, label="The Answer Was")

        COLOR_EMOJI = {CW_GREEN: "🟩", CW_YELLOW: "🟨", CW_RED: "🟥"}
        emoji_grid  = f"CastleWordle {today_iso}  {len(guesses_data)}/{CW_MAX_GUESSES}\n" + \
                      "\n".join("".join(COLOR_EMOJI.get(a["color"], "⬛") for a in comp)
                                for _, comp in guesses_data)
        st.code(emoji_grid, language=None)
        st.caption("Copy the grid above to share your result!")

        if st.button("Play Again (reset for testing)", type="secondary"):
            st.session_state.cw_date = ""
            st.rerun()

    st.divider()
    st.markdown(f"""
    <div style="display:flex;gap:16px;justify-content:center;flex-wrap:wrap;
                font-size:0.8em;color:#888;padding-bottom:12px;">
        <span>🟩 Exact match</span><span>🟨 Close / overlapping</span>
        <span>🟥 No match</span><span>⬆️ Answer is higher</span><span>⬇️ Answer is lower</span>
    </div>""", unsafe_allow_html=True)
