"""
CastleWordle — Test Page
A Wordle-style daily guessing game for streaming platform talent.
Run standalone: streamlit run test_castlewordle.py
"""

import hashlib
import numpy as np
import pandas as pd
import streamlit as st
from datetime import date

st.set_page_config(page_title="CastleWordle", page_icon="🎬", layout="wide")

# ─── Constants ───────────────────────────────────────────────────────────────
MAX_GUESSES = 6
MIN_TITLES = 5
PLATFORM_LABELS = {
    "netflix": "Netflix",
    "max": "Max",
    "prime": "Prime Video",
    "disney": "Disney+",
    "paramount": "Paramount+",
    "appletv": "Apple TV+",
}

GREEN  = "#1db954"
YELLOW = "#f0a500"
RED    = "#c0392b"
GRAY   = "#333344"
CARD_BG     = "#1E1E2E"
CARD_BORDER = "#333"
CARD_TEXT   = "#ddd"


# ─── Data Precompute ─────────────────────────────────────────────────────────
@st.cache_data
def build_player_pool() -> pd.DataFrame:
    person_stats = pd.read_parquet("data/precomputed/network/person_stats.parquet")
    credits      = pd.read_parquet("data/processed/all_platforms_credits.parquet")
    titles       = pd.read_parquet("data/processed/all_platforms_titles.parquet")

    credits["person_id"] = credits["person_id"].astype(str)
    person_stats["person_id"] = person_stats["person_id"].astype(str)

    pool = person_stats[person_stats["title_count"] >= MIN_TITLES].copy()

    # ── Primary platform (most credits on) ──────────────────────────────────
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

    # ── Genre set per person (for yellow overlap detection) ──────────────────
    def to_list(g):
        if isinstance(g, (list, np.ndarray)):
            return [str(x) for x in g if x]
        return []

    cred_merged["genres_list"] = cred_merged["genres"].apply(to_list)
    exploded = cred_merged.explode("genres_list").dropna(subset=["genres_list"])
    exploded = exploded[exploded["genres_list"] != ""]
    genre_sets = (
        exploded.groupby("person_id")["genres_list"]
        .apply(set)
        .reset_index()
        .rename(columns={"genres_list": "genre_set"})
    )
    pool = pool.merge(genre_sets, on="person_id", how="left")

    # ── Active decade from career_end ────────────────────────────────────────
    pool["active_decade"] = pool["career_end"].apply(
        lambda y: f"{int(y) // 10 * 10}s" if pd.notna(y) and y > 0 else None
    )

    # Drop rows missing any attribute needed for comparison
    pool = pool.dropna(subset=["top_genre", "avg_imdb", "primary_platform",
                                "primary_role", "active_decade"])
    return pool.reset_index(drop=True)


def get_daily_person(pool: pd.DataFrame) -> pd.Series:
    today_str = date.today().isoformat()
    hash_int  = int(hashlib.md5(today_str.encode()).hexdigest(), 16)
    return pool.iloc[hash_int % len(pool)]


# ─── Session State Init ───────────────────────────────────────────────────────
today_iso = date.today().isoformat()
if st.session_state.get("cw_date") != today_iso:
    st.session_state.cw_date     = today_iso
    st.session_state.cw_guesses  = []   # list of person_ids guessed
    st.session_state.cw_game_over = False
    st.session_state.cw_won      = False


# ─── Load Data ───────────────────────────────────────────────────────────────
with st.spinner("Loading CastleWordle..."):
    pool = build_player_pool()

mystery = get_daily_person(pool)


# ─── Helper: Compare Two People ───────────────────────────────────────────────
def compare(guess: pd.Series, target: pd.Series) -> list[dict]:
    """Return a list of attribute comparison dicts with color and display value."""
    results = []

    # 1. Primary Platform
    g_plat = PLATFORM_LABELS.get(guess["primary_platform"], guess["primary_platform"])
    match_plat = guess["primary_platform"] == target["primary_platform"]
    results.append({
        "label": "Platform",
        "value": g_plat,
        "color": GREEN if match_plat else RED,
        "hint": "",
    })

    # 2. Primary Role
    match_role = guess["primary_role"] == target["primary_role"]
    results.append({
        "label": "Role",
        "value": guess["primary_role"].title(),
        "color": GREEN if match_role else RED,
        "hint": "",
    })

    # 3. Primary Genre (top_genre + genre_set for yellow)
    g_top   = str(guess["top_genre"]).lower()
    t_top   = str(target["top_genre"]).lower()
    g_set   = guess["genre_set"] if isinstance(guess["genre_set"], set) else set()
    t_set   = target["genre_set"] if isinstance(target["genre_set"], set) else set()
    if g_top == t_top:
        genre_color = GREEN
    elif g_set & t_set:
        genre_color = YELLOW
    else:
        genre_color = RED
    results.append({
        "label": "Genre",
        "value": g_top.title(),
        "color": genre_color,
        "hint": "~" if genre_color == YELLOW else "",
    })

    # 4. Title Count
    g_cnt = int(guess["title_count"])
    t_cnt = int(target["title_count"])
    diff  = g_cnt - t_cnt
    if diff == 0:
        cnt_color = GREEN
        cnt_hint  = ""
    elif abs(diff) <= 5:
        cnt_color = YELLOW
        cnt_hint  = "⬆️" if diff < 0 else "⬇️"
    else:
        cnt_color = RED
        cnt_hint  = "⬆️" if diff < 0 else "⬇️"
    results.append({
        "label": "Titles",
        "value": str(g_cnt),
        "color": cnt_color,
        "hint": cnt_hint,
    })

    # 5. Avg IMDb Score
    g_imdb = round(float(guess["avg_imdb"]), 1)
    t_imdb = round(float(target["avg_imdb"]), 1)
    diff_i = g_imdb - t_imdb
    if abs(diff_i) <= 0.3:
        imdb_color = GREEN
        imdb_hint  = ""
    elif abs(diff_i) <= 1.0:
        imdb_color = YELLOW
        imdb_hint  = "⬆️" if diff_i < 0 else "⬇️"
    else:
        imdb_color = RED
        imdb_hint  = "⬆️" if diff_i < 0 else "⬇️"
    results.append({
        "label": "IMDb",
        "value": f"{g_imdb:.1f}",
        "color": imdb_color,
        "hint": imdb_hint,
    })

    # 6. Active Decade
    match_dec = guess["active_decade"] == target["active_decade"]
    results.append({
        "label": "Decade",
        "value": guess["active_decade"],
        "color": GREEN if match_dec else RED,
        "hint": "",
    })

    return results


# ─── Helper: Render a Guess Row ───────────────────────────────────────────────
def render_guess_row(person_name: str, comparison: list[dict], is_win_row: bool = False):
    tile_html = ""
    for attr in comparison:
        tile_html += f"""
        <div style="
            background:{attr['color']};
            border-radius:6px;
            padding:8px 4px;
            text-align:center;
            min-width:90px;
            flex:1;
        ">
            <div style="font-size:0.65em;color:rgba(255,255,255,0.75);margin-bottom:2px;">
                {attr['label']}
            </div>
            <div style="font-size:0.9em;font-weight:700;color:#fff;">
                {attr['value']} {attr['hint']}
            </div>
        </div>"""

    border = f"2px solid {GREEN}" if is_win_row else f"1px solid {CARD_BORDER}"
    st.markdown(f"""
    <div style="margin-bottom:8px;">
        <div style="
            font-size:0.85em;color:{CARD_TEXT};
            margin-bottom:4px;padding-left:2px;
        ">
            {"🎯 " if is_win_row else ""}<strong>{person_name}</strong>
        </div>
        <div style="display:flex;gap:6px;flex-wrap:nowrap;
                    background:{CARD_BG};padding:8px;
                    border-radius:8px;border:{border};">
            {tile_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Helper: Emoji Grid for Share ─────────────────────────────────────────────
def build_emoji_grid(guesses_data: list) -> str:
    COLOR_EMOJI = {GREEN: "🟩", YELLOW: "🟨", RED: "🟥"}
    lines = [f"CastleWordle {today_iso}  {len(guesses_data)}/{MAX_GUESSES}\n"]
    for _, comparison in guesses_data:
        row = "".join(COLOR_EMOJI.get(a["color"], "⬛") for a in comparison)
        lines.append(row)
    return "\n".join(lines)


# ─── Helper: Profile Card ─────────────────────────────────────────────────────
def render_profile_card(person: pd.Series, label: str = ""):
    plat_str = PLATFORM_LABELS.get(person["primary_platform"], person["primary_platform"])
    career   = f"{int(person['career_start'])}–{int(person['career_end'])}" \
               if pd.notna(person.get("career_start")) and pd.notna(person.get("career_end")) \
               else "N/A"
    st.markdown(f"""
    <div style="background:{CARD_BG};border:1px solid {GREEN};border-radius:12px;padding:20px;margin-top:12px;">
        <div style="font-size:0.75em;color:{GREEN};text-transform:uppercase;letter-spacing:1px;
                    margin-bottom:4px;">{label}</div>
        <div style="font-size:1.5em;font-weight:800;color:#fff;margin-bottom:12px;">
            🎬 {person['name']}
        </div>
        <div style="display:flex;gap:12px;flex-wrap:wrap;">
            <span style="background:#2a2a3e;padding:6px 12px;border-radius:20px;
                         color:{CARD_TEXT};font-size:0.85em;">
                🎭 {str(person['primary_role']).title()}
            </span>
            <span style="background:#2a2a3e;padding:6px 12px;border-radius:20px;
                         color:{CARD_TEXT};font-size:0.85em;">
                📺 {plat_str}
            </span>
            <span style="background:#2a2a3e;padding:6px 12px;border-radius:20px;
                         color:{CARD_TEXT};font-size:0.85em;">
                🎞️ {int(person['title_count'])} titles
            </span>
            <span style="background:#2a2a3e;padding:6px 12px;border-radius:20px;
                         color:{CARD_TEXT};font-size:0.85em;">
                ⭐ IMDb {float(person['avg_imdb']):.1f}
            </span>
            <span style="background:#2a2a3e;padding:6px 12px;border-radius:20px;
                         color:{CARD_TEXT};font-size:0.85em;">
                🎬 {str(person['top_genre']).title()}
            </span>
            <span style="background:#2a2a3e;padding:6px 12px;border-radius:20px;
                         color:{CARD_TEXT};font-size:0.85em;">
                📅 {career}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE RENDER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="text-align:center;padding:24px 0 8px;">
    <div style="font-size:2.5em;font-weight:900;color:#fff;letter-spacing:2px;">
        🏰 CastleWordle
    </div>
    <div style="color:#888;font-size:0.95em;margin-top:4px;">
        Guess today's mystery streaming talent — new puzzle every day
    </div>
</div>
""", unsafe_allow_html=True)

# ─── How to Play ─────────────────────────────────────────────────────────────
with st.expander("How to Play"):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
**Goal:** Guess the mystery person (actor, director, writer, etc.) in 6 tries.

**After each guess**, each attribute is color-coded:
- 🟩 **Green** — exact match
- 🟨 **Yellow** — close (genre overlap, title count within 5, IMDb within 1.0)
- 🟥 **Red** — no match

**Attributes compared:** Platform · Role · Genre · Titles · IMDb · Decade
        """)
    with col_b:
        st.markdown("""
**Tips:**
- The mystery person has at least **5 titles** in the catalog
- **⬆️ / ⬇️** arrows tell you if the answer is higher or lower
- A new puzzle drops every day at midnight
- Use **Share Result** to copy your emoji grid (like Wordle!)
        """)

st.divider()

# ─── Guess Counter ───────────────────────────────────────────────────────────
n_guesses  = len(st.session_state.cw_guesses)
game_over  = st.session_state.cw_game_over
won        = st.session_state.cw_won

counter_color = GREEN if won else (RED if game_over else CARD_TEXT)
st.markdown(f"""
<div style="text-align:center;margin-bottom:12px;">
    <span style="font-size:1.1em;font-weight:700;color:{counter_color};">
        Guess {n_guesses}/{MAX_GUESSES}
    </span>
</div>
""", unsafe_allow_html=True)

# ─── Previous Guesses ─────────────────────────────────────────────────────────
guesses_data = []   # list of (person_row, comparison)
for pid in st.session_state.cw_guesses:
    row = pool[pool["person_id"] == pid]
    if not row.empty:
        p    = row.iloc[0]
        comp = compare(p, mystery)
        guesses_data.append((p, comp))
        render_guess_row(p["name"], comp, is_win_row=(p["person_id"] == mystery["person_id"]))

# ─── Input Area ───────────────────────────────────────────────────────────────
if not game_over:
    already_guessed = set(st.session_state.cw_guesses)
    remaining_pool  = pool[~pool["person_id"].isin(already_guessed)]

    # Build display options sorted by name
    options = remaining_pool.sort_values("name")["person_id"].tolist()

    def fmt_option(pid):
        r = remaining_pool[remaining_pool["person_id"] == pid]
        if r.empty:
            return pid
        row = r.iloc[0]
        plat = PLATFORM_LABELS.get(row["primary_platform"], row["primary_platform"])
        return f"{row['name']}  ({row['primary_role'].title()}, {plat}, {int(row['title_count'])} titles)"

    col_sel, col_btn, col_give = st.columns([4, 1, 1])
    with col_sel:
        selected_pid = st.selectbox(
            "Search for a person",
            options=options,
            format_func=fmt_option,
            index=None,
            placeholder="Type a name to search...",
            key="cw_select",
            label_visibility="collapsed",
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
        elif len(st.session_state.cw_guesses) >= MAX_GUESSES:
            st.session_state.cw_game_over = True
        st.rerun()

    if give_up:
        st.session_state.cw_game_over = True
        st.session_state.cw_won       = False
        st.rerun()

# ─── Win / Loss State ─────────────────────────────────────────────────────────
if game_over:
    if won:
        st.balloons()
        st.markdown(f"""
        <div style="text-align:center;padding:16px;background:rgba(29,185,84,0.15);
                    border:1px solid {GREEN};border-radius:12px;margin:12px 0;">
            <div style="font-size:1.8em;font-weight:900;color:{GREEN};">🎉 You got it!</div>
            <div style="color:#aaa;margin-top:4px;">
                Solved in {len(st.session_state.cw_guesses)}/{MAX_GUESSES} guesses
            </div>
        </div>
        """, unsafe_allow_html=True)
        render_profile_card(mystery, label="Today's Mystery Person")
    else:
        st.markdown(f"""
        <div style="text-align:center;padding:16px;background:rgba(192,57,43,0.15);
                    border:1px solid {RED};border-radius:12px;margin:12px 0;">
            <div style="font-size:1.8em;font-weight:900;color:{RED};">😔 Better luck tomorrow!</div>
        </div>
        """, unsafe_allow_html=True)
        render_profile_card(mystery, label="The Answer Was")

    # Share button
    emoji_grid = build_emoji_grid(guesses_data)
    st.code(emoji_grid, language=None)
    st.caption("Copy the grid above to share your result!")

    if st.button("Play Again (reset for testing)", type="secondary"):
        st.session_state.cw_date      = ""   # force reset
        st.rerun()

# ─── Legend ──────────────────────────────────────────────────────────────────
st.divider()
st.markdown(f"""
<div style="display:flex;gap:16px;justify-content:center;flex-wrap:wrap;
            font-size:0.8em;color:#888;padding-bottom:12px;">
    <span>🟩 Exact match</span>
    <span>🟨 Close / overlapping</span>
    <span>🟥 No match</span>
    <span>⬆️ Answer is higher</span>
    <span>⬇️ Answer is lower</span>
</div>
""", unsafe_allow_html=True)
