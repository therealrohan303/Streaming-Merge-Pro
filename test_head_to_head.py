"""
Head-to-Head Arena — Standalone Test Page
Side-by-side stat battle between any two people in the streaming catalog.
Run: streamlit run test_head_to_head.py
"""

import random
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Head-to-Head Arena", page_icon="⚔️", layout="wide")

# ─── Constants ────────────────────────────────────────────────────────────────
MIN_TITLES = 3
PLAT_LABELS = {
    "netflix":   "Netflix",
    "max":       "Max",
    "prime":     "Prime Video",
    "disney":    "Disney+",
    "paramount": "Paramount+",
    "appletv":   "Apple TV+",
}
TEAL    = "#00b4a6"
MUTED   = "#555566"
CARD_BG = "#1E1E2E"
BORDER  = "#333"
TEXT    = "#ddd"
GOLD    = "#FFD700"


# ═══════════════════════════════════════════════════════════════════════════════
#  PRECOMPUTE ARENA STATS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def build_arena_pool() -> pd.DataFrame:
    person_stats = pd.read_parquet("data/precomputed/network/person_stats.parquet")
    credits      = pd.read_parquet("data/processed/all_platforms_credits.parquet")
    titles       = pd.read_parquet("data/processed/all_platforms_titles.parquet")

    person_stats["person_id"] = person_stats["person_id"].astype(str)
    credits["person_id"]      = credits["person_id"].astype(str)

    pool = person_stats[person_stats["title_count"] >= MIN_TITLES].copy()

    # ── Primary platform ──────────────────────────────────────────────────────
    t_slim  = titles[["id", "platform", "imdb_score", "genres"]].rename(columns={"id": "title_id"})
    cmerged = credits[["person_id", "title_id"]].merge(t_slim, on="title_id", how="left")

    primary_plat = (
        cmerged.groupby(["person_id", "platform"]).size()
        .reset_index(name="cnt").sort_values("cnt", ascending=False)
        .drop_duplicates("person_id")[["person_id", "platform"]]
        .rename(columns={"platform": "primary_platform"})
    )
    pool = pool.merge(primary_plat, on="person_id", how="left")

    # ── Best single title (highest IMDb they appeared in) ────────────────────
    best_title = (
        cmerged.groupby("person_id")["imdb_score"].max()
        .reset_index().rename(columns={"imdb_score": "best_title_imdb"})
    )
    pool = pool.merge(best_title, on="person_id", how="left")

    # ── Genre diversity (unique genres across all their titles) ───────────────
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

    # ── Platform diversity (number of platforms) ──────────────────────────────
    pool["platform_diversity"] = pool["platform_list"].apply(
        lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0
    )

    # ── Netflix & Max title counts ────────────────────────────────────────────
    plat_counts = (
        cmerged.groupby(["person_id", "platform"]).size()
        .reset_index(name="cnt")
    )
    for plat_key, col_name in [("netflix", "netflix_titles"), ("max", "max_titles")]:
        pc = plat_counts[plat_counts["platform"] == plat_key][["person_id", "cnt"]].rename(columns={"cnt": col_name})
        pool = pool.merge(pc, on="person_id", how="left")
        pool[col_name] = pool[col_name].fillna(0).astype(int)

    # ── Career span ───────────────────────────────────────────────────────────
    pool["career_span"] = pool.apply(
        lambda r: int(r["career_end"]) - int(r["career_start"])
        if pd.notna(r.get("career_start")) and pd.notna(r.get("career_end")) and r["career_end"] > 0
        else 0,
        axis=1,
    )
    pool["career_label"] = pool.apply(
        lambda r: f"{int(r['career_start'])}–{int(r['career_end'])}"
        if pd.notna(r.get("career_start")) and pd.notna(r.get("career_end")) and r["career_end"] > 0
        else "N/A",
        axis=1,
    )

    pool["genre_diversity"]  = pool["genre_diversity"].fillna(0).astype(int)
    pool["best_title_imdb"]  = pool["best_title_imdb"].fillna(0.0).round(1)
    pool["primary_platform"] = pool["primary_platform"].fillna("N/A")

    return pool.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  STAT COMPARISON LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def build_stat_rows(a: pd.Series, b: pd.Series) -> list[dict]:
    """Return list of stat dicts with winner info."""

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
        row("Primary Platform",     0, 0, plat_a, plat_b, is_text=True),
        row("Primary Role",         0, 0, str(a["primary_role"]).title(),
            str(b["primary_role"]).title(), is_text=True),
        row("Total Titles",         a["title_count"],        b["title_count"],
            int(a["title_count"]),          int(b["title_count"])),
        row("Avg IMDb Score",       a["avg_imdb"],           b["avg_imdb"],
            f"{float(a['avg_imdb']):.2f}",  f"{float(b['avg_imdb']):.2f}"),
        row("Best Single Title",    a["best_title_imdb"],    b["best_title_imdb"],
            f"{float(a['best_title_imdb']):.1f}", f"{float(b['best_title_imdb']):.1f}"),
        row("Genre Diversity",      a["genre_diversity"],    b["genre_diversity"],
            int(a["genre_diversity"]),       int(b["genre_diversity"])),
        row("Platform Diversity",   a["platform_diversity"], b["platform_diversity"],
            int(a["platform_diversity"]),    int(b["platform_diversity"])),
        row("Career Span (yrs)",    a["career_span"],        b["career_span"],
            f"{a['career_label']} ({int(a['career_span'])} yrs)",
            f"{b['career_label']} ({int(b['career_span'])} yrs)"),
        row("Netflix Titles",       a["netflix_titles"],     b["netflix_titles"],
            int(a["netflix_titles"]),        int(b["netflix_titles"])),
        row("Max Titles",           a["max_titles"],         b["max_titles"],
            int(a["max_titles"]),            int(b["max_titles"])),
    ]


def merger_insight(a: pd.Series, b: pd.Series, winner_name: str) -> str:
    """Generate a merger-critical insight line."""
    a_name = a["name"]
    b_name = b["name"]

    # Find the most differentiating stat for the winner
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


# ═══════════════════════════════════════════════════════════════════════════════
#  UI HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def stat_table(rows: list[dict]):
    for r in rows:
        col_a_color  = TEAL  if r["winner"] == "a"   else (MUTED if r["winner"] == "b" else TEXT)
        col_b_color  = TEAL  if r["winner"] == "b"   else (MUTED if r["winner"] == "a" else TEXT)
        a_weight     = "700" if r["winner"] == "a"   else "400"
        b_weight     = "700" if r["winner"] == "b"   else "400"
        a_trophy     = " 🏆" if r["winner"] == "a"   else ""
        b_trophy     = "🏆 " if r["winner"] == "b"   else ""

        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 160px 1fr;
                    align-items:center;padding:10px 16px;
                    border-bottom:1px solid #2a2a3e;">
            <div style="text-align:right;font-size:1em;font-weight:{a_weight};
                        color:{col_a_color};">{r['display_a']}{a_trophy}</div>
            <div style="text-align:center;font-size:0.75em;color:#666;
                        text-transform:uppercase;letter-spacing:1px;padding:0 12px;">
                {r['label']}</div>
            <div style="text-align:left;font-size:1em;font-weight:{b_weight};
                        color:{col_b_color};">{b_trophy}{r['display_b']}</div>
        </div>
        """, unsafe_allow_html=True)


def winner_card(name: str, wins: int, total: int, insight: str):
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0d2d2a,#1a3a36);
                border:2px solid {TEAL};border-radius:14px;
                padding:24px;text-align:center;margin-top:20px;">
        <div style="font-size:0.8em;color:{TEAL};text-transform:uppercase;
                    letter-spacing:2px;margin-bottom:6px;">Overall Winner</div>
        <div style="font-size:2em;font-weight:900;color:#fff;margin-bottom:4px;">
            🏆 {name}</div>
        <div style="color:#aaa;font-size:0.9em;margin-bottom:16px;">
            Won {wins} of {total} stats</div>
        <div style="background:rgba(0,180,166,0.1);border:1px solid {TEAL};
                    border-radius:8px;padding:12px;font-size:0.88em;color:{TEXT};
                    text-align:left;">
            💡 <em>Merger insight:</em> {insight}</div>
    </div>
    """, unsafe_allow_html=True)


def person_card(p: pd.Series, side: str):
    color = TEAL if side == "left" else GOLD
    plat  = PLAT_LABELS.get(p.get("primary_platform", "N/A"), p.get("primary_platform", "N/A"))
    st.markdown(f"""
    <div style="background:{CARD_BG};border:2px solid {color};border-radius:12px;
                padding:16px;text-align:center;">
        <div style="font-size:1.3em;font-weight:800;color:#fff;margin-bottom:8px;">
            {p['name']}</div>
        <div style="display:flex;gap:8px;flex-wrap:wrap;justify-content:center;">
            <span style="background:#2a2a3e;padding:4px 10px;border-radius:16px;
                         color:#aaa;font-size:0.8em;">{str(p['primary_role']).title()}</span>
            <span style="background:#2a2a3e;padding:4px 10px;border-radius:16px;
                         color:#aaa;font-size:0.8em;">{plat}</span>
            <span style="background:#2a2a3e;padding:4px 10px;border-radius:16px;
                         color:#aaa;font-size:0.8em;">{int(p['title_count'])} titles</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="padding:16px 0 8px;">
    <div style="font-size:2.2em;font-weight:900;color:#fff;letter-spacing:1px;">
        ⚔️ Head-to-Head Arena</div>
    <div style="color:#888;font-size:0.9em;margin-top:4px;">
        Pick any two people from the catalog for a side-by-side stat battle</div>
</div>
""", unsafe_allow_html=True)

with st.spinner("Loading arena data…"):
    pool = build_arena_pool()

pid_list   = pool["person_id"].tolist()
name_index = pool.set_index("person_id")["name"].to_dict()

def fmt_person(pid):
    r = pool[pool["person_id"] == pid]
    if r.empty:
        return pid
    row = r.iloc[0]
    plat = PLAT_LABELS.get(row.get("primary_platform", ""), "")
    return f"{row['name']}  ({str(row['primary_role']).title()}, {plat}, {int(row['title_count'])} titles)"

# ─── Session state ────────────────────────────────────────────────────────────
if "arena_a" not in st.session_state:
    st.session_state.arena_a = None
if "arena_b" not in st.session_state:
    st.session_state.arena_b = None

# ─── Random Battle button ─────────────────────────────────────────────────────
col_rand, _ = st.columns([1, 5])
with col_rand:
    if st.button("🎲 Random Battle", type="primary"):
        picks = random.sample(pid_list, 2)
        st.session_state.arena_a = picks[0]
        st.session_state.arena_b = picks[1]
        st.rerun()

st.markdown("---")

# ─── Person selectors ─────────────────────────────────────────────────────────
col_a, col_vs, col_b = st.columns([5, 1, 5])

with col_a:
    st.markdown(f"<div style='color:{TEAL};font-weight:700;margin-bottom:6px;'>Player A</div>",
                unsafe_allow_html=True)
    pid_a = st.selectbox(
        "Person A", options=pid_list, format_func=fmt_person,
        index=pid_list.index(st.session_state.arena_a)
               if st.session_state.arena_a in pid_list else 0,
        placeholder="Search for a person…", label_visibility="collapsed", key="sel_a",
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
    st.markdown(f"<div style='color:{GOLD};font-weight:700;margin-bottom:6px;'>Player B</div>",
                unsafe_allow_html=True)
    pid_b = st.selectbox(
        "Person B", options=pid_list, format_func=fmt_person,
        index=pid_list.index(st.session_state.arena_b)
               if st.session_state.arena_b in pid_list else min(1, len(pid_list) - 1),
        placeholder="Search for a person…", label_visibility="collapsed", key="sel_b",
    )
    st.session_state.arena_b = pid_b

# ─── Battle ───────────────────────────────────────────────────────────────────
if pid_a and pid_b:
    row_a = pool[pool["person_id"] == pid_a].iloc[0]
    row_b = pool[pool["person_id"] == pid_b].iloc[0]

    if pid_a == pid_b:
        st.warning("Pick two different people for a battle!")
    else:
        # Person cards
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        card_a, spacer, card_b = st.columns([5, 1, 5])
        with card_a:
            person_card(row_a, "left")
        with card_b:
            person_card(row_b, "right")

        # Stat table header
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 160px 1fr;
                    align-items:center;padding:10px 16px;margin-top:16px;
                    background:#16162a;border-radius:8px 8px 0 0;
                    border-bottom:2px solid {TEAL};">
            <div style="text-align:right;font-size:0.9em;font-weight:700;
                        color:{TEAL};">{row_a['name']}</div>
            <div style="text-align:center;font-size:0.75em;color:#555;
                        text-transform:uppercase;letter-spacing:1px;">Stat</div>
            <div style="text-align:left;font-size:0.9em;font-weight:700;
                        color:{GOLD};">{row_b['name']}</div>
        </div>
        <div style="background:{CARD_BG};border:1px solid {BORDER};
                    border-top:none;border-radius:0 0 8px 8px;margin-bottom:8px;">
        """, unsafe_allow_html=True)

        stats = build_stat_rows(row_a, row_b)
        stat_table(stats)

        st.markdown("</div>", unsafe_allow_html=True)

        # Tally wins
        wins_a = sum(1 for s in stats if s["winner"] == "a")
        wins_b = sum(1 for s in stats if s["winner"] == "b")
        total  = len([s for s in stats if s["winner"] != "tie"])

        if wins_a > wins_b:
            w_name, w_wins = row_a["name"], wins_a
        elif wins_b > wins_a:
            w_name, w_wins = row_b["name"], wins_b
        else:
            w_name, w_wins = "Tie!", (wins_a + wins_b) // 2

        insight = merger_insight(row_a, row_b, w_name)
        winner_card(w_name, w_wins, total, insight)

        # Win tally bar
        st.markdown(f"""
        <div style="display:flex;gap:8px;align-items:center;
                    margin-top:16px;font-size:0.85em;color:#888;">
            <span style="color:{TEAL};font-weight:700;">{row_a['name']}: {wins_a} wins</span>
            <span style="flex:1;height:6px;background:#2a2a3e;border-radius:3px;overflow:hidden;">
                <div style="width:{int(wins_a/(wins_a+wins_b+0.001)*100)}%;
                            height:100%;background:{TEAL};border-radius:3px;"></div>
            </span>
            <span style="color:{GOLD};font-weight:700;">{wins_b} wins: {row_b['name']}</span>
        </div>
        """, unsafe_allow_html=True)
