from __future__ import annotations

import streamlit as st

from components.charts import recommendations_table
from src.hybrid_model import HybridRecommender
from src.preprocessing import OCCUPATION_LABELS


def _fit_model(data, alpha: float) -> HybridRecommender:
    return HybridRecommender(alpha=alpha).fit(data.ratings, data.users, data.movies)


def render(data, top_n: int, alpha: float) -> None:
    st.title("Recommendations")
    mode = st.tabs(["Existing user", "Cold-start user"])
    model = _fit_model(data, alpha)

    with mode[0]:
        user_ids = sorted(data.users["user_id"].astype(int).unique())
        user_id = st.selectbox("User ID", user_ids, index=0)
        recs = model.recommend(user_id=int(user_id), top_n=top_n, exclude_seen=True)
        recommendations_table(recs)

    with mode[1]:
        col1, col2, col3 = st.columns(3)
        gender = col1.selectbox("Gender", ["M", "F"])
        age = col2.selectbox("Age group", [1, 18, 25, 35, 45, 50, 56], format_func=str)
        occupation = col3.selectbox(
            "Occupation",
            sorted(OCCUPATION_LABELS.keys()),
            format_func=lambda value: OCCUPATION_LABELS[value],
        )
        profile = {"gender": gender, "age": int(age), "occupation": int(occupation)}
        recs = model.recommend(user_id=None, profile=profile, top_n=top_n, exclude_seen=True)
        recommendations_table(recs)

