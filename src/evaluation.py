from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EvaluationResult:
    model: str
    rmse: float
    mae: float
    precision_at_k: float
    recall_at_k: float
    evaluated_ratings: int
    evaluated_users: int


def train_test_split_by_user(
    ratings: pd.DataFrame,
    test_fraction: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_state)
    train_parts = []
    test_parts = []
    for _, group in ratings.groupby("user_id"):
        if len(group) < 5:
            train_parts.append(group)
            continue
        mask = rng.random(len(group)) < test_fraction
        if not mask.any():
            mask[rng.integers(0, len(group))] = True
        if mask.all():
            mask[rng.integers(0, len(group))] = False
        test_parts.append(group[mask])
        train_parts.append(group[~mask])
    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True) if test_parts else ratings.iloc[0:0].copy()
    return train, test


def rating_errors(model, test_ratings: pd.DataFrame) -> tuple[float, float]:
    if test_ratings.empty:
        return float("nan"), float("nan")
    predictions = [
        model.predict(int(row.user_id), int(row.movie_id))
        for row in test_ratings.itertuples(index=False)
    ]
    actual = test_ratings["rating"].to_numpy(dtype=float)
    predicted = np.asarray(predictions, dtype=float)
    rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))
    mae = float(np.mean(np.abs(predicted - actual)))
    return rmse, mae


def precision_recall_at_k(
    model,
    train_ratings: pd.DataFrame,
    test_ratings: pd.DataFrame,
    k: int = 5,
    relevance_threshold: float = 4.0,
    max_users: int = 100,
) -> tuple[float, float, int]:
    precisions = []
    recalls = []
    users = list(test_ratings["user_id"].drop_duplicates().astype(int))[:max_users]
    for user_id in users:
        relevant = set(
            test_ratings.loc[
                (test_ratings["user_id"] == user_id) & (test_ratings["rating"] >= relevance_threshold),
                "movie_id",
            ].astype(int)
        )
        if not relevant:
            continue
        try:
            recs = model.recommend(user_id=user_id, top_n=k, exclude_seen=True)
        except TypeError:
            recs = model.recommend(user_id, top_n=k, exclude_seen=True)
        if recs.empty:
            continue
        recommended = set(recs["movie_id"].astype(int).head(k))
        hits = len(recommended & relevant)
        precisions.append(hits / k)
        recalls.append(hits / len(relevant))
    if not precisions:
        return 0.0, 0.0, 0
    return float(np.mean(precisions)), float(np.mean(recalls)), len(precisions)


def evaluate_model(
    model_name: str,
    model,
    train_ratings: pd.DataFrame,
    test_ratings: pd.DataFrame,
    k: int = 5,
    max_users: int = 100,
) -> EvaluationResult:
    rmse, mae = rating_errors(model, test_ratings)
    precision, recall, evaluated_users = precision_recall_at_k(
        model=model,
        train_ratings=train_ratings,
        test_ratings=test_ratings,
        k=k,
        max_users=max_users,
    )
    return EvaluationResult(
        model=model_name,
        rmse=rmse,
        mae=mae,
        precision_at_k=precision,
        recall_at_k=recall,
        evaluated_ratings=int(len(test_ratings)),
        evaluated_users=int(evaluated_users),
    )


def results_to_frame(results: list[EvaluationResult]) -> pd.DataFrame:
    return pd.DataFrame([result.__dict__ for result in results])
