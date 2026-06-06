from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class SVDRecommender:
    n_factors: int = 20
    ratings: pd.DataFrame | None = field(default=None, init=False)
    movies: pd.DataFrame | None = field(default=None, init=False)
    user_ids: list[int] = field(default_factory=list, init=False)
    movie_ids: list[int] = field(default_factory=list, init=False)
    user_index: dict[int, int] = field(default_factory=dict, init=False)
    movie_index: dict[int, int] = field(default_factory=dict, init=False)
    reconstructed: np.ndarray | None = field(default=None, init=False)
    item_means: pd.Series | None = field(default=None, init=False)
    global_mean: float = field(default=3.5, init=False)
    user_factors: np.ndarray | None = field(default=None, init=False)
    item_factors: np.ndarray | None = field(default=None, init=False)

    def fit(self, ratings: pd.DataFrame, movies: pd.DataFrame | None = None) -> "SVDRecommender":
        self.ratings = ratings.copy()
        self.movies = movies.copy() if movies is not None else None
        self.global_mean = float(self.ratings["rating"].mean())
        self.item_means = self.ratings.groupby("movie_id")["rating"].mean()

        matrix = self.ratings.pivot_table(index="user_id", columns="movie_id", values="rating", aggfunc="mean")
        self.user_ids = [int(value) for value in matrix.index]
        self.movie_ids = [int(value) for value in matrix.columns]
        self.user_index = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        self.movie_index = {movie_id: idx for idx, movie_id in enumerate(self.movie_ids)}

        user_means = matrix.mean(axis=1).fillna(self.global_mean)
        filled = matrix.T.fillna(user_means).T.to_numpy(dtype=float)
        centered = filled - user_means.to_numpy(dtype=float)[:, None]

        u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        factors = min(self.n_factors, len(singular_values))
        u_k = u[:, :factors]
        s_k = singular_values[:factors]
        vt_k = vt[:factors, :]
        reconstructed = (u_k * s_k) @ vt_k + user_means.to_numpy(dtype=float)[:, None]

        self.reconstructed = np.clip(reconstructed, 1.0, 5.0)
        sqrt_s = np.sqrt(s_k)
        self.user_factors = u_k * sqrt_s
        self.item_factors = vt_k.T * sqrt_s
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        self._ensure_fit()
        assert self.reconstructed is not None
        assert self.item_means is not None

        if user_id in self.user_index and movie_id in self.movie_index:
            return float(self.reconstructed[self.user_index[user_id], self.movie_index[movie_id]])
        return float(np.clip(self.item_means.get(movie_id, self.global_mean), 1.0, 5.0))

    def recommend(self, user_id: int, top_n: int = 5, exclude_seen: bool = True) -> pd.DataFrame:
        self._ensure_fit()
        assert self.ratings is not None

        candidates = self.movie_ids
        if exclude_seen:
            seen = set(self.ratings.loc[self.ratings["user_id"] == user_id, "movie_id"])
            candidates = [movie_id for movie_id in candidates if movie_id not in seen]
        scored = [(movie_id, self.predict(user_id, movie_id)) for movie_id in candidates]
        rows = sorted(scored, key=lambda item: item[1], reverse=True)[:top_n]
        result = pd.DataFrame(rows, columns=["movie_id", "predicted_rating"])
        result["model"] = "Classical SVD"
        if self.movies is not None and not result.empty:
            result = result.merge(self.movies[["movie_id", "title", "genres"]], on="movie_id", how="left")
            result = result[["movie_id", "title", "genres", "predicted_rating", "model"]]
        return result

    def get_latent_features(self) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
        self._ensure_fit()
        assert self.user_factors is not None
        assert self.item_factors is not None
        return self.user_factors, self.item_factors, self.user_ids, self.movie_ids

    def _ensure_fit(self) -> None:
        if self.reconstructed is None:
            raise RuntimeError("SVDRecommender.fit must be called first.")
