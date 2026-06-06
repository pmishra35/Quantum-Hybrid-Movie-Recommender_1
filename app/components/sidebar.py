from __future__ import annotations

import streamlit as st


def render_sidebar() -> tuple[str, int, float]:
    st.sidebar.title("Quantum Hybrid Recommender")
    page = st.sidebar.radio(
        "Navigation",
        ["Home", "Recommend", "Evaluation", "About"],
        label_visibility="collapsed",
    )
    top_n = st.sidebar.selectbox("Recommendations", [5, 10], index=0)
    alpha = st.sidebar.select_slider("Hybrid alpha", options=[0.25, 0.5, 0.75, 1.0], value=0.5)
    st.sidebar.caption("alpha controls collaborative vs demographic weighting")
    return page, int(top_n), float(alpha)
