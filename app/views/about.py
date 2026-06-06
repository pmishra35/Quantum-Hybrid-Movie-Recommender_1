from __future__ import annotations

import streamlit as st


def render() -> None:
    st.title("About")
    st.write(
        "The system combines collaborative filtering with demographic similarity to reduce cold-start "
        "and sparsity issues. Classical SVD provides matrix factorization, while the quantum-inspired "
        "module uses compact latent vectors and Qiskit statevector simulation when Qiskit is installed."
    )

    st.subheader("Models")
    st.markdown(
        """
- Collaborative Filtering: recommends movies from similar users.
- Demographic Similarity: supports new users using gender, age, and occupation.
- Hybrid Model: combines collaborative and demographic scores with alpha.
- Classical SVD: reconstructs sparse ratings through low-rank factors.
- Quantum-Inspired Hybrid: compares latent states through a small simulated circuit.
"""
    )

    st.subheader("Evaluation")
    st.write("RMSE and MAE measure rating error. Precision@K and Recall@K use ratings of 4 or 5 as relevant movies.")

