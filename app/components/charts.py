from __future__ import annotations

import pandas as pd
import streamlit as st


def metric_cards(ratings: pd.DataFrame, users: pd.DataFrame, movies: pd.DataFrame) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ratings", f"{len(ratings):,}")
    col2.metric("Users", f"{users['user_id'].nunique():,}")
    col3.metric("Movies", f"{movies['movie_id'].nunique():,}")
    sparsity = 1 - (len(ratings) / max(users["user_id"].nunique() * movies["movie_id"].nunique(), 1))
    col4.metric("Matrix sparsity", f"{sparsity:.1%}")


def evaluation_chart(results: pd.DataFrame) -> None:
    if results.empty:
        st.info("Run the pipeline to generate evaluation results.")
        return
    metrics = ["rmse", "mae", "precision_at_k", "recall_at_k"]
    available = [metric for metric in metrics if metric in results.columns]
    st.bar_chart(results.set_index("model")[available])


def recommendations_table(recommendations: pd.DataFrame) -> None:
    if recommendations.empty:
        st.warning("No recommendations were generated for this selection.")
        return
    display = recommendations.copy()
    if "predicted_rating" in display.columns:
        display["predicted_rating"] = display["predicted_rating"].astype(float).round(3)
    st.dataframe(display, use_container_width=True, hide_index=True)
