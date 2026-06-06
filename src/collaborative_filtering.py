from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def _clip_rating(value: float) -> float:
    return float(np.clip(value, 1.0, 5.0))


@dataclass
class CollaborativeFilteringRecommender:
    top_neighbors: int = 30
    min_similarity: float = 0.0
    ratings: pd.DataFrame | None = field(default=None, init=False)
    movies: pd.DataFrame | None = field(default=None, init=False)
    matrix: pd.DataFrame | None = field(default=None, init=False)
    similarity: pd.DataFrame | None = field(default=None, init=False)
    user_means: pd.Series | None = field(default=None, init=False)
    item_means: pd.Series | None = field(default=None, init=False)
    global_mean: float = field(default=3.5, init=False)

    def fit(self, ratings: pd.DataFrame, movies: pd.DataFrame | None = None) -> "CollaborativeFilteringRecommender":
        self.ratings = ratings.copy()
        self.movies = movies.copy() if movies is not None else None
        self.global_mean = float(self.ratings["rating"].mean())
        self.user_means = self.ratings.groupby("user_id")["rating"].mean()
        self.item_means = self.ratings.groupby("movie_id")["rating"].mean()
        matrix = self.ratings.pivot_table(index="user_id", columns="movie_id", values="rating", aggfunc="mean")
        self.matrix = matrix

        centered = matrix.sub(matrix.mean(axis=1), axis=0).fillna(0.0)
        values = centered.to_numpy(dtype=float)
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        normalized = np.divide(values, norms, out=np.zeros_like(values), where=norms != 0)
        sim = normalized @ normalized.T
        self.similarity = pd.DataFrame(sim, index=matrix.index, columns=matrix.index)
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        self._ensure_fit()
        assert self.matrix is not None
        assert self.similarity is not None
        assert self.item_means is not None

        if movie_id not in self.matrix.columns:
            return _clip_rating(self.global_mean)
        if user_id not in self.matrix.index:
            return _clip_rating(float(self.item_means.get(movie_id, self.global_mean)))

        movie_ratings = self.matrix[movie_id].dropna()
        movie_ratings = movie_ratings.drop(index=user_id, errors="ignore")
        if movie_ratings.empty:
            return _clip_rating(float(self.item_means.get(movie_id, self.global_mean)))

        sims = self.similarity.loc[user_id, movie_ratings.index]
        sims = sims[sims > self.min_similarity].sort_values(ascending=False).head(self.top_neighbors)
        if sims.empty or np.isclose(float(sims.abs().sum()), 0.0):
            return _clip_rating(float(self.item_means.get(movie_id, self.global_mean)))

        neighbor_ratings = movie_ratings.loc[sims.index]
        return _clip_rating(float(np.average(neighbor_ratings, weights=sims)))

    def recommend(self, user_id: int, top_n: int = 5, exclude_seen: bool = True) -> pd.DataFrame:
        self._ensure_fit()
        assert self.matrix is not None

        candidates = list(self.matrix.columns)
        if exclude_seen and user_id in self.matrix.index:
            seen = set(self.matrix.loc[user_id].dropna().index)
            candidates = [movie_id for movie_id in candidates if movie_id not in seen]

        scored = [(movie_id, self.predict(user_id, int(movie_id))) for movie_id in candidates]
        return self._format_recommendations(scored, top_n, "Collaborative Filtering")

    def _format_recommendations(self, scored: list[tuple[int, float]], top_n: int, model_name: str) -> pd.DataFrame:
        rows = sorted(scored, key=lambda item: item[1], reverse=True)[:top_n]
        result = pd.DataFrame(rows, columns=["movie_id", "predicted_rating"])
        result["model"] = model_name
        if self.movies is not None and not result.empty:
            result = result.merge(self.movies[["movie_id", "title", "genres"]], on="movie_id", how="left")
            result = result[["movie_id", "title", "genres", "predicted_rating", "model"]]
        return result

    def _ensure_fit(self) -> None:
        if self.matrix is None or self.similarity is None:
            raise RuntimeError("CollaborativeFilteringRecommender.fit must be called first.")
