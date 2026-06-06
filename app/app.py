from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from components.sidebar import render_sidebar
from views import about, evaluation, home, recommend
from src.preprocessing import load_processed_data, prepare_data


@st.cache_data(show_spinner=False)
def get_data(sample_users: int = 500):
    processed = PROJECT_ROOT / "data" / "processed" / "cleaned_ratings.csv"
    if processed.exists() and processed.stat().st_size > 0:
        return load_processed_data()
    return prepare_data(sample_users=sample_users)


def main() -> None:
    st.set_page_config(page_title="Quantum Hybrid Recommender", page_icon="Q", layout="wide")
    page, top_n, alpha = render_sidebar()
    data = get_data()

    if page == "Home":
        home.render(data)
    elif page == "Recommend":
        recommend.render(data, top_n=top_n, alpha=alpha)
    elif page == "Evaluation":
        evaluation.render()
    else:
        about.render()


if __name__ == "__main__":
    main()
