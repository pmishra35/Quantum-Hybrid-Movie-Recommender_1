from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

from src.collaborative_filtering import CollaborativeFilteringRecommender
from src.demographic_similarity import DemographicRecommender
from src.evaluation import evaluate_model, results_to_frame, train_test_split_by_user
from src.hybrid_model import HybridRecommender
from src.preprocessing import PROCESSED_DIR, RAW_DIR, prepare_data, save_processed_data
from src.quantum_model import QuantumInspiredRecommender
from src.svd_model import SVDRecommender


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def build_models(train: pd.DataFrame, users: pd.DataFrame, movies: pd.DataFrame, alpha: float) -> dict[str, object]:
    models: dict[str, object] = {
        "Collaborative Filtering": CollaborativeFilteringRecommender(top_neighbors=25).fit(train, movies),
        "Classical SVD": SVDRecommender(n_factors=20).fit(train, movies),
        "Demographic + CF Hybrid": HybridRecommender(alpha=alpha).fit(train, users, movies),
        "Quantum-Inspired Hybrid": QuantumInspiredRecommender(n_factors=4).fit(train, movies),
    }
    return models


def save_models(models: dict[str, object]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    mapping = {
        "Classical SVD": "svd_model.pkl",
        "Demographic + CF Hybrid": "hybrid_model.pkl",
        "Quantum-Inspired Hybrid": "quantum_model.pkl",
    }
    for name, filename in mapping.items():
        with (MODELS_DIR / filename).open("wb") as file:
            pickle.dump(models[name], file)


def save_evaluation_plot(results: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    metrics = ["rmse", "mae", "precision_at_k", "recall_at_k"]
    ax = results.set_index("model")[metrics].plot(kind="bar", figsize=(12, 6), rot=20)
    ax.set_title("Model Evaluation Comparison")
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_evaluation_comparison.png", dpi=160)
    plt.close()


def write_report(results: pd.DataFrame, sample_users: int, alpha: float, qiskit_used: bool) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    markdown_table = results.to_csv(index=False)
    lines = [
        "# Quantum-Enhanced Hybrid Movie Recommendation System",
        "",
        f"Sample users used for this run: {sample_users if sample_users else 'all users'}",
        f"Hybrid alpha: {alpha}",
        f"Qiskit statevector simulation used: {qiskit_used}",
        "",
        "## Evaluation Results",
        "",
        "```csv",
        markdown_table.strip(),
        "```",
        "",
        "Ratings >= 4 are treated as relevant for Precision@K and Recall@K.",
    ]
    (REPORTS_DIR / "final_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_pipeline(args: argparse.Namespace) -> pd.DataFrame:
    data = prepare_data(raw_dir=args.raw_dir, sample_users=args.sample_users, random_state=args.random_state)
    save_processed_data(data, PROCESSED_DIR)

    train, test = train_test_split_by_user(data.ratings, test_fraction=args.test_fraction, random_state=args.random_state)
    models = build_models(train, data.users, data.movies, alpha=args.alpha)
    results = [
        evaluate_model(name, model, train, test, k=args.top_k, max_users=args.max_eval_users)
        for name, model in models.items()
    ]
    results_frame = results_to_frame(results)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results_frame.to_csv(REPORTS_DIR / "evaluation_results.csv", index=False)
    save_models(models)
    save_evaluation_plot(results_frame)
    quantum_model = models["Quantum-Inspired Hybrid"]
    write_report(results_frame, args.sample_users, args.alpha, getattr(quantum_model, "qiskit_available", False))
    return results_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the quantum hybrid recommender pipeline.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--sample-users", type=int, default=500, help="Use 0 for the full MovieLens 1M user set.")
    parser.add_argument("--alpha", type=float, default=0.5, choices=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--max-eval-users", type=int, default=80)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    output = run_pipeline(parse_args())
    print(output.to_string(index=False))
