"""
Collaboration Network Visualizer — Standalone Test Page
Run: streamlit run test_network_visualizer.py
"""

import os
import tempfile
from collections import deque

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Network Visualizer Test", page_icon="🕸️", layout="wide")

# ─── Constants ───────────────────────────────────────────────────────────────
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


# ─── Data Loading & Precompute ────────────────────────────────────────────────
@st.cache_data
def load_data():
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
    primary = (
        merged.groupby(["person_id", "platform"]).size()
        .reset_index(name="cnt")
        .sort_values("cnt", ascending=False)
        .drop_duplicates("person_id")[["person_id", "platform"]]
        .rename(columns={"platform": "primary_platform"})
    )
    return primary


@st.cache_data
def build_adjacency(edges_df: pd.DataFrame) -> dict:
    adj: dict = {}
    for pa, pb, w in zip(edges_df["person_a"], edges_df["person_b"], edges_df["weight"]):
        adj.setdefault(pa, {})[pb] = int(w)
        adj.setdefault(pb, {})[pa] = int(w)
    return adj


# ─── Graph Helpers ────────────────────────────────────────────────────────────
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

    # Deduplicate by name: same real person can have two person_ids (platform vs IMDB source).
    # Keep the entry with the higher title_count; build a remap so edges still connect correctly.
    sub = sub.sort_values("title_count", ascending=False)
    duplicate_names = sub[sub["name"].duplicated(keep=False)]["name"].unique()
    remap: dict = {}   # dropped_pid -> kept_pid
    for name in duplicate_names:
        rows = sub[sub["name"] == name]
        kept_pid    = rows.iloc[0]["person_id"]   # highest title_count (already sorted)
        dropped_pids = rows.iloc[1:]["person_id"].tolist()
        for d in dropped_pids:
            remap[d] = kept_pid
    sub = sub.drop_duplicates("name", keep="first")

    # Remap edges: replace dropped person_ids with kept ones, then deduplicate
    remapped_edges: set = set()
    for pa, pb, w in edge_triples:
        ra = remap.get(pa, pa)
        rb = remap.get(pb, pb)
        if ra == rb:          # self-loop after remap — skip
            continue
        key = (min(ra, rb), max(ra, rb))
        remapped_edges.add((*key, w))
    edge_triples = remapped_edges
    node_ids     = set(sub["person_id"].tolist())

    tc_max = sub["title_count"].max()
    tc_min = sub["title_count"].min()

    def node_size(tc):
        if tc_max == tc_min:
            return 22
        return 10 + (tc - tc_min) / (tc_max - tc_min) * 38

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
        net.add_node(
            pid,
            label=row["name"],
            title=tooltip,
            size=node_size(row["title_count"]),
            color={
                "background": color,
                "border": "#FFD700" if is_seed else color,
                "highlight": {"background": "#FFD700", "border": "#ffffff"},
            },
            borderWidth=3 if is_seed else 1,
            font={"size": 11, "color": "#dddddd"},
        )

    for pa, pb, w in edge_triples:
        if pa in node_ids and pb in node_ids:
            net.add_edge(pa, pb, value=max(1, w),
                         title=f"Shared titles: {w}",
                         color={"color": "rgba(255,255,255,0.12)",
                                "highlight": "rgba(255,215,0,0.6)"})

    net.set_options("""{
      "nodes": {"shadow": {"enabled": true, "size": 8}},
      "edges": {"smooth": {"type": "continuous"}},
      "physics": {
        "enabled": true,
        "stabilization": {"enabled": true, "iterations": 150, "updateInterval": 25}
      },
      "interaction": {
        "hover": true, "tooltipDelay": 80,
        "navigationButtons": true, "keyboard": true
      }
    }""")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        net.save_graph(f.name)
        tmp = f.name
    with open(tmp, "r", encoding="utf-8") as f:
        html = f.read()
    os.unlink(tmp)
    return html


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE RENDER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="padding:20px 0 4px;">
    <div style="font-size:2em;font-weight:900;color:#fff;">🕸️ Collaboration Network Visualizer</div>
    <div style="color:#888;font-size:0.9em;margin-top:4px;">
        Pick a person — explore their collaboration web up to 2 degrees of separation
    </div>
</div>
""", unsafe_allow_html=True)

# Load data
with st.spinner("Loading network data…"):
    edges_df, person_stats = load_data()
    primary_plat           = compute_primary_platform()
    adj                    = build_adjacency(edges_df)

# Default seed = most connected node
connected_ids = set(adj.keys())
default_seed  = (
    person_stats[person_stats["person_id"].isin(connected_ids)]
    .assign(_deg=lambda df: df["person_id"].map(lambda p: len(adj.get(p, {}))))
    .nlargest(1, "_deg")["person_id"].iloc[0]
)

# Seed pool — top 5000 by title count for fast autocomplete
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

# ─── Controls ────────────────────────────────────────────────────────────────
ALL_ROLES = ["ACTOR", "DIRECTOR", "WRITER", "PRODUCER", "COMPOSER", "CINEMATOGRAPHER", "EDITOR"]

col_s, col_d, col_m = st.columns([4, 1, 1])
with col_s:
    seed_pid = st.selectbox(
        "Seed person",
        options=seed_ids,
        format_func=fmt_seed,
        index=seed_ids.index(default_seed) if default_seed in seed_ids else 0,
        placeholder="Type a name to search…",
        label_visibility="collapsed",
        key="net_seed",
    )
with col_d:
    depth = st.selectbox("Degrees", [1, 2], index=1, key="net_depth")
with col_m:
    max_nodes = st.selectbox("Max nodes", [50, 100, 150], index=1, key="net_max")

role_filter = st.multiselect(
    "Filter by role",
    options=ALL_ROLES,
    default=ALL_ROLES,
    format_func=lambda r: r.title(),
    key="net_roles",
)
if not role_filter:
    role_filter = ALL_ROLES   # fall back to all if user clears selection

# Platform legend
st.markdown(
    " &nbsp; ".join(
        f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;'
        f'background:{c};margin-right:3px;vertical-align:middle;"></span>'
        f'<span style="color:#aaa;font-size:0.78em;">{lbl}</span>'
        for lbl, c in [
            ("Netflix",     PLAT_COLORS["netflix"]),
            ("Max",         PLAT_COLORS["max"]),
            ("Prime Video", PLAT_COLORS["prime"]),
            ("Disney+",     PLAT_COLORS["disney"]),
            ("Paramount+",  PLAT_COLORS["paramount"]),
            ("Apple TV+",   PLAT_COLORS["appletv"]),
            ("⭐ Seed node", "#FFD700"),
        ]
    ),
    unsafe_allow_html=True,
)

# ─── Build & Render Graph ─────────────────────────────────────────────────────
with st.spinner("Building network…"):
    node_ids, edge_triples = bfs_subgraph(seed_pid, adj,
                                          max_nodes=max_nodes, max_hops=depth)

    # Apply role filter — always keep seed node visible
    selected_roles = set(role_filter)
    role_lookup    = person_stats.set_index("person_id")["primary_role"].to_dict()
    filtered_nodes = {
        pid for pid in node_ids
        if pid == seed_pid or role_lookup.get(pid, "").upper() in selected_roles
    }
    filtered_edges = {
        (pa, pb, w) for pa, pb, w in edge_triples
        if pa in filtered_nodes and pb in filtered_nodes
    }

    net_html = render_pyvis(filtered_nodes, filtered_edges,
                            person_stats, primary_plat, seed_pid)

components.html(net_html, height=670, scrolling=False)

st.caption(
    f"**{len(filtered_nodes)} nodes** · **{len(filtered_edges)} edges** · "
    f"{depth}-hop neighbourhood · Roles: {', '.join(r.title() for r in role_filter)} · "
    f"Node size = title count · Hover for details"
)

# ─── Quick Stats Panel ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("**Quick Stats — pick any visible node:**")

visible = person_stats[person_stats["person_id"].isin(filtered_nodes)].sort_values("name")
vis_ids = visible["person_id"].tolist()

selected = st.selectbox(
    "Pick a node",
    options=vis_ids,
    format_func=lambda pid: visible[visible["person_id"] == pid].iloc[0]["name"]
                            if not visible[visible["person_id"] == pid].empty else pid,
    index=vis_ids.index(seed_pid) if seed_pid in vis_ids else 0,
    label_visibility="collapsed",
    key="net_stats",
)

if selected:
    r       = person_stats[person_stats["person_id"] == selected].iloc[0]
    pp_row  = primary_plat[primary_plat["person_id"] == selected]
    pplat   = pp_row.iloc[0]["primary_platform"] if not pp_row.empty else "N/A"
    pcolor  = PLAT_COLORS.get(pplat, DEFAULT_COLOR)
    plabel  = PLAT_LABELS.get(pplat, pplat)
    career  = (f"{int(r['career_start'])}–{int(r['career_end'])}"
               if pd.notna(r.get("career_start")) and pd.notna(r.get("career_end")) else "N/A")
    connections = len(adj.get(selected, {}))

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    for col, label, value in [
        (c1, "Name",        r["name"]),
        (c2, "Role",        str(r["primary_role"]).title()),
        (c3, "Platform",    f'<span style="color:{pcolor}">{plabel}</span>'),
        (c4, "Titles",      int(r["title_count"])),
        (c5, "Avg IMDb",    f"{float(r['avg_imdb']):.1f}"),
        (c6, "Connections", connections),
    ]:
        with col:
            st.markdown(f"""
            <div style="background:{CARD_BG};border:1px solid {CARD_BORDER};
                        border-radius:8px;padding:12px;text-align:center;">
                <div style="font-size:0.7em;color:#888;margin-bottom:4px;">{label}</div>
                <div style="font-size:1em;font-weight:700;color:#ddd;">{value}</div>
            </div>
            """, unsafe_allow_html=True)
