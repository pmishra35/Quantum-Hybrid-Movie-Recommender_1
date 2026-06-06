from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from components.charts import evaluation_chart


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = PROJECT_ROOT / "reports" / "evaluation_results.csv"


def render() -> None:
    st.title("Evaluation Comparison")
    if not RESULTS_PATH.exists():
        st.info("Run `python run_pipeline.py` to generate evaluation metrics.")
        return

    results = pd.read_csv(RESULTS_PATH)
    st.dataframe(results, use_container_width=True, hide_index=True)
    evaluation_chart(results)

