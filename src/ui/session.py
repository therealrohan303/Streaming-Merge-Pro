"""Session state management for global filters."""

import streamlit as st

DEFAULTS = {
    "platform_view": "merged",
    "content_types": ["Movie", "Show"],
    "year_range": (1940, 2023),
    "min_imdb": 0.0,
    "selected_genres": [],
}


def init_session_state():
    """Set default filter values if keys are missing."""
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_filters():
    """Reset all filter keys to defaults."""
    for key, value in DEFAULTS.items():
        st.session_state[key] = value


def get_filter_state() -> dict:
    """Return current filter values as a plain dict."""
    return {key: st.session_state.get(key, default) for key, default in DEFAULTS.items()}
