"""Page: CinemaGuess — Guess the movie or show from a trailer clip and still frame.

Requires: data/precomputed/game_catalog.parquet
Run scripts/13_enrich_tmdb_game.py to generate the game catalog.

Game rules:
  - 6 guesses to identify the title.
  - A trailer clip and/or backdrop still are shown from the start.
  - A new clue is revealed after each wrong guess:
      Guess 1 wrong → genres
      Guess 2 wrong → release decade
      Guess 3 wrong → type (Movie / Show)
      Guess 4 wrong → runtime or season count
      Guess 5 wrong → first letter of title
"""

import hashlib
from datetime import date

import pandas as pd
import streamlit as st

from src.config import CARD_ACCENT, CARD_BG, CARD_BORDER, CARD_TEXT, PRECOMPUTED_DIR
from src.ui.badges import page_header_html, section_header_html
from src.ui.session import init_session_state

st.set_page_config(page_title="CinemaGuess", page_icon="🎬", layout="wide")
init_session_state()

GAME_CATALOG_PATH = PRECOMPUTED_DIR / "game_catalog.parquet"
MAX_GUESSES = 6

CLUE_DEFS = [
    (
        "Genre(s)",
        lambda r: (
            ", ".join(g.capitalize() for g in list(r["genres"])[:3])
            if r.get("genres") is not None and len(r["genres"]) > 0
            else "Unknown"
        ),
    ),
    (
        "Decade",
        lambda r: (
            f"{int(r['release_year'] // 10) * 10}s"
            if pd.notna(r.get("release_year"))
            else "Unknown"
        ),
    ),
    ("Type", lambda r: r.get("type", "Unknown")),
    (
        "Runtime / Seasons",
        lambda r: (
            f"{int(r['runtime'])} min"
            if r.get("type") == "Movie" and pd.notna(r.get("runtime"))
            else f"{int(r['seasons'])} season(s)"
            if pd.notna(r.get("seasons"))
            else "Unknown"
        ),
    ),
    (
        "First letter",
        lambda r: (
            "The " + r["title"][4].upper()
            if r.get("title", "").startswith("The ")
            else r.get("title", "?")[0].upper()
        ),
    ),
]


# ─── Data ────────────────────────────────────────────────────────────────────


@st.cache_data
def load_game_catalog():
    if not GAME_CATALOG_PATH.exists():
        return None
    df = pd.read_parquet(GAME_CATALOG_PATH)
    has_backdrop = df["backdrop_url"].notna()
    return df[has_backdrop].reset_index(drop=True)


game_df = load_game_catalog()

st.markdown(
    page_header_html(
        "CinemaGuess",
        "Guess the movie or show from a trailer clip and still frame — 6 chances.",
    ),
    unsafe_allow_html=True,
)

if game_df is None or game_df.empty:
    st.error(
        "Game catalog not found. Run `python scripts/13_enrich_tmdb_game.py` to generate it."
    )
    st.stop()


# ─── Game state helpers ───────────────────────────────────────────────────────


def _pick_daily(df):
    h = int(hashlib.md5(date.today().isoformat().encode()).hexdigest(), 16)
    return df.iloc[h % len(df)].to_dict()


def _pick_random(df):
    return df.sample(1).iloc[0].to_dict()


def _init_game(mode="random"):
    answer = _pick_daily(game_df) if mode == "daily" else _pick_random(game_df)
    st.session_state.cg = {
        "answer": answer,
        "guesses": [],
        "game_over": False,
        "won": False,
        "mode": mode,
    }


if "cg" not in st.session_state:
    _init_game("random")

cg = st.session_state.cg
answer = cg["answer"]
guesses = cg["guesses"]
num_guesses = len(guesses)
clues_revealed = num_guesses  # one new clue per wrong guess


# ─── Mode controls ───────────────────────────────────────────────────────────


col_mode, col_new = st.columns([3, 1])
with col_mode:
    mode_choice = st.radio(
        "Mode",
        ["Random", "Daily Challenge"],
        horizontal=True,
        key="cg_mode_radio",
        help="Daily Challenge: same title for everyone today.",
    )
with col_new:
    st.markdown("&nbsp;", unsafe_allow_html=True)
    if st.button("New Game", type="primary", use_container_width=True):
        _init_game("daily" if mode_choice == "Daily Challenge" else "random")
        st.rerun()


# ─── Visual assets ───────────────────────────────────────────────────────────

# Blur schedule: starts very heavy, eases off with each wrong guess.
# 0 guesses → 18px (text unreadable), fully unblurred only after game ends.
_BLUR_SCHEDULE = [18, 14, 10, 7, 4, 2, 0]


def _blur_px():
    """Current blur level in pixels. 0 once the game is over."""
    if cg["game_over"]:
        return 0
    return _BLUR_SCHEDULE[min(num_guesses, len(_BLUR_SCHEDULE) - 1)]


def _blurred_img_html(url, height=None):
    """Render an image with CSS blur so any title text is unreadable."""
    blur = _blur_px()
    style = f"filter: blur({blur}px); transition: filter 0.4s ease; width:100%; border-radius:6px; display:block;"
    if height:
        style += f" height:{height}px; object-fit:cover;"
    return f'<img src="{url}" style="{style}">'


has_backdrop = bool(answer.get("backdrop_url"))

if num_guesses == 0 and not cg["game_over"]:
    st.caption(f"Image starts blurred — unblurs a little with each guess. Current blur: {_blur_px()}px")

if has_backdrop:
    st.markdown(_blurred_img_html(answer["backdrop_url"]), unsafe_allow_html=True)
else:
    st.info("No visual assets available for this title — use the clues below.")


# ─── Clue cards ──────────────────────────────────────────────────────────────


if clues_revealed > 0:
    st.markdown(
        section_header_html("Clues", "Revealed after each wrong guess."),
        unsafe_allow_html=True,
    )
    n_clues = min(clues_revealed, len(CLUE_DEFS))
    clue_cols = st.columns(n_clues)
    for i in range(n_clues):
        label, fn = CLUE_DEFS[i]
        try:
            val = fn(answer)
        except Exception:
            val = "Unknown"
        with clue_cols[i]:
            st.markdown(
                f"""
                <div style="background:{CARD_BG};border:1px solid {CARD_BORDER};
                            border-top:3px solid {CARD_ACCENT};border-radius:6px;
                            padding:10px 12px;text-align:center;">
                    <div style="color:#888;font-size:0.72rem;margin-bottom:4px;">{label}</div>
                    <div style="color:{CARD_TEXT};font-weight:600;">{val}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ─── Previous guesses ────────────────────────────────────────────────────────


if guesses:
    st.markdown(section_header_html("Your Guesses"), unsafe_allow_html=True)
    for g in guesses:
        correct = g.strip().lower() == answer["title"].strip().lower()
        bg = "#1a5c38" if correct else "#5c1a1a"
        icon = "✓" if correct else "✗"
        st.markdown(
            f'<div style="background:{bg};border-radius:4px;padding:7px 14px;'
            f'margin-bottom:5px;color:#fff;">{icon} {g}</div>',
            unsafe_allow_html=True,
        )


# ─── Game over banner ─────────────────────────────────────────────────────────


def _result_card():
    year = answer.get("release_year")
    year_str = f" ({int(year)})" if pd.notna(year) else ""
    score = answer.get("imdb_score")
    genres = answer.get("genres", [])
    genre_str = ", ".join(g.capitalize() for g in list(genres)[:3]) if genres is not None and len(genres) > 0 else "N/A"

    if cg["won"]:
        st.success(f"Correct in {num_guesses} guess(es)!  **{answer['title']}**{year_str}")
    else:
        st.error(f"The answer was  **{answer['title']}**{year_str}")

    info_cols = st.columns(4)
    with info_cols[0]:
        st.metric("IMDb", f"{score:.1f}" if pd.notna(score) else "N/A")
    with info_cols[1]:
        st.metric("Type", answer.get("type", "N/A"))
    with info_cols[2]:
        st.metric("Genre(s)", genre_str if genre_str else "N/A")
    with info_cols[3]:
        st.metric("Year", int(year) if pd.notna(year) else "N/A")

    # Show unblurred backdrop in the result card
    if has_backdrop:
        st.markdown(_blurred_img_html(answer["backdrop_url"]), unsafe_allow_html=True)

    if st.button("Play Again", type="primary"):
        _init_game("daily" if mode_choice == "Daily Challenge" else "random")
        st.rerun()


if cg["game_over"]:
    _result_card()


# ─── Guess input ─────────────────────────────────────────────────────────────


elif not cg["game_over"]:
    st.divider()
    guesses_left = MAX_GUESSES - num_guesses
    st.markdown(
        f"**Guess {num_guesses + 1} of {MAX_GUESSES}** &nbsp;—&nbsp; "
        f"{guesses_left} guess(es) remaining",
        unsafe_allow_html=True,
    )

    all_titles = [""] + sorted(game_df["title"].dropna().unique().tolist())

    guess_val = st.selectbox(
        "Select a title",
        options=all_titles,
        key=f"cg_guess_{num_guesses}",
        label_visibility="collapsed",
    )

    col_submit, col_skip = st.columns([2, 1])
    with col_submit:
        if st.button(
            "Submit Guess",
            disabled=not guess_val,
            type="primary",
            use_container_width=True,
        ):
            if len(cg["guesses"]) == num_guesses:  # guard against double-submit
                cg["guesses"].append(guess_val)
                if guess_val.strip().lower() == answer["title"].strip().lower():
                    cg["game_over"] = True
                    cg["won"] = True
                elif len(cg["guesses"]) >= MAX_GUESSES:
                    cg["game_over"] = True
                    cg["won"] = False
            st.rerun()

    with col_skip:
        if st.button("Skip (count as wrong)", use_container_width=True):
            if len(cg["guesses"]) == num_guesses:  # guard against double-submit
                cg["guesses"].append("— skipped —")
                if len(cg["guesses"]) >= MAX_GUESSES:
                    cg["game_over"] = True
                    cg["won"] = False
            st.rerun()


# ─── Progress bar ─────────────────────────────────────────────────────────────


st.markdown("&nbsp;", unsafe_allow_html=True)
progress_pct = num_guesses / MAX_GUESSES
bar_color = CARD_ACCENT if not cg["game_over"] else ("#1a5c38" if cg["won"] else "#8b0000")
st.markdown(
    f"""
    <div style="background:#333;border-radius:4px;height:6px;margin-top:4px;">
        <div style="background:{bar_color};width:{progress_pct*100:.0f}%;height:6px;border-radius:4px;"></div>
    </div>
    <div style="color:#888;font-size:0.75rem;margin-top:4px;">
        {num_guesses} / {MAX_GUESSES} guesses used
    </div>
    """,
    unsafe_allow_html=True,
)
