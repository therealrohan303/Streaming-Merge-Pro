"""Global sidebar filters and quick stats panel."""

from collections import Counter

import streamlit as st

from src.ui.session import DEFAULTS, reset_filters


def render_sidebar_filters(df) -> dict:
    """Render sidebar filter widgets and return current filter values.

    Args:
        df: titles DataFrame used to derive dynamic options (genres, year range).

    Returns:
        dict with keys: platform_view, content_types, year_range, min_imdb, selected_genres
    """
    st.sidebar.header("Filters")

    # Platform view
    platform_options = {"Merged (Netflix + Max)": "merged",
                        "Netflix only": "netflix",
                        "Max only": "max",
                        "All 6 platforms": "all_platforms"}
    platform_label = st.sidebar.selectbox(
        "Platform view",
        options=list(platform_options.keys()),
        index=list(platform_options.values()).index(
            st.session_state.get("platform_view", DEFAULTS["platform_view"])
        ),
        help="Choose which platforms to include in the analysis",
    )
    platform_view = platform_options[platform_label]
    st.session_state["platform_view"] = platform_view

    # Content type
    content_types = st.sidebar.multiselect(
        "Content type",
        options=["Movie", "Show"],
        default=st.session_state.get("content_types", DEFAULTS["content_types"]),
        help="Filter by Movies, Shows, or both",
    )
    st.session_state["content_types"] = content_types

    # Year range
    min_year = int(df["release_year"].min()) if df["release_year"].notna().any() else 1940
    max_year = int(df["release_year"].max()) if df["release_year"].notna().any() else 2023
    default_range = st.session_state.get("year_range", DEFAULTS["year_range"])
    year_range = st.sidebar.slider(
        "Release year",
        min_value=min_year,
        max_value=max_year,
        value=(max(default_range[0], min_year), min(default_range[1], max_year)),
        help="Filter titles by release year range",
    )
    st.session_state["year_range"] = year_range

    # Minimum IMDb
    min_imdb = st.sidebar.slider(
        "Minimum IMDb score",
        min_value=0.0,
        max_value=10.0,
        value=st.session_state.get("min_imdb", DEFAULTS["min_imdb"]),
        step=0.5,
        help="Set minimum IMDb score — titles below this are hidden",
    )
    st.session_state["min_imdb"] = min_imdb

    # IMDb quick preset buttons
    p1, p2, p3 = st.sidebar.columns(3)
    if p1.button("6+", use_container_width=True):
        st.session_state["min_imdb"] = 6.0
        st.rerun()
    if p2.button("7+", use_container_width=True):
        st.session_state["min_imdb"] = 7.0
        st.rerun()
    if p3.button("8+", use_container_width=True):
        st.session_state["min_imdb"] = 8.0
        st.rerun()

    # Genre multi-select with title-cased labels and counts
    genre_counter = Counter()
    for genres in df["genres"].dropna():
        if isinstance(genres, list):
            genre_counter.update(genres)

    # Build display label → raw key mapping, sorted by count descending
    sorted_genres = sorted(genre_counter.items(), key=lambda x: (-x[1], x[0]))
    display_to_raw = {f"{g.title()} ({count:,})": g for g, count in sorted_genres}
    raw_to_display = {v: k for k, v in display_to_raw.items()}

    # Map any existing session state raw keys to display labels for defaults
    current_raw = st.session_state.get("selected_genres", [])
    default_display = [raw_to_display[g] for g in current_raw if g in raw_to_display]

    selected_display = st.sidebar.multiselect(
        "Genres",
        options=list(display_to_raw.keys()),
        default=default_display,
        help="Select one or more genres — titles matching any selected genre are shown",
    )
    # Convert display selections back to raw keys for storage and filtering
    selected_genres = [display_to_raw[d] for d in selected_display]
    st.session_state["selected_genres"] = selected_genres

    # Reset button
    if st.sidebar.button("Reset filters"):
        reset_filters()
        st.rerun()

    return {
        "platform_view": platform_view,
        "content_types": content_types,
        "year_range": year_range,
        "min_imdb": min_imdb,
        "selected_genres": selected_genres,
    }


def apply_filters(df, filters: dict):
    """Apply filter dict to a titles DataFrame.

    Returns filtered DataFrame (copy).
    """
    mask = True

    # Content type
    if filters.get("content_types"):
        mask = mask & df["type"].isin(filters["content_types"])

    # Year range
    yr = filters.get("year_range")
    if yr:
        mask = mask & df["release_year"].between(yr[0], yr[1])

    # Min IMDb
    min_imdb = filters.get("min_imdb", 0.0)
    if min_imdb > 0:
        mask = mask & (df["imdb_score"].fillna(0) >= min_imdb)

    # Genre filter (intersection: title must have at least one selected genre)
    selected = filters.get("selected_genres", [])
    if selected:
        selected_set = set(selected)
        mask = mask & df["genres"].apply(
            lambda g: bool(set(g) & selected_set) if isinstance(g, list) else False
        )

    return df[mask].reset_index(drop=True)


def render_quick_stats(df, total_count: int, total_avg_imdb: float):
    """Quick stats panel showing filtered title count and avg IMDb with deltas.

    Call inside a sidebar container context (e.g. `with placeholder:`)
    so elements render at the desired position.

    Args:
        df: filtered titles DataFrame
        total_count: total title count before filtering (for delta)
        total_avg_imdb: overall avg IMDb before filtering (for delta)
    """
    import math

    st.subheader("Current Selection")

    filtered_count = len(df)
    count_delta = filtered_count - total_count
    st.metric(
        "Titles Shown",
        f"{filtered_count:,}",
        delta=f"{count_delta:,} from total" if count_delta != 0 else "showing all",
        delta_color="off" if count_delta == 0 else "normal",
    )

    avg_imdb = df["imdb_score"].mean()
    if not math.isnan(avg_imdb):
        imdb_delta = avg_imdb - total_avg_imdb
        st.metric(
            "Avg IMDb Score",
            f"{avg_imdb:.2f}",
            delta=f"{imdb_delta:+.2f} from overall" if abs(imdb_delta) >= 0.005 else "same as overall",
            delta_color="normal" if abs(imdb_delta) >= 0.005 else "off",
        )
    else:
        st.metric("Avg IMDb Score", "N/A")

    st.markdown("---")
