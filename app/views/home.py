from __future__ import annotations

import streamlit as st

from components.charts import metric_cards


def render(data) -> None:
    st.title("Quantum-Enhanced Hybrid Movie Recommendation System")
    st.caption("Collaborative filtering, demographics, SVD, and practical Qiskit-style simulation")

    metric_cards(data.ratings, data.users, data.movies)

    st.subheader("Project focus")
    st.write(
        "This M.Tech final-year project addresses cold-start and data sparsity in movie recommendations "
        "using MovieLens 1M. Existing users are served with collaborative signals, while new users can "
        "receive recommendations from demographic similarity."
    )

    st.subheader("Dataset snapshot")
    st.dataframe(data.movies[["movie_id", "title", "genres"]].head(10), use_container_width=True, hide_index=True)

