from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .collaborative_filtering import CollaborativeFilteringRecommender
from .demographic_similarity import DemographicRecommender


@dataclass
class HybridRecommender:
    alpha: float = 0.5
    collaborative: CollaborativeFilteringRecommender = field(default_factory=CollaborativeFilteringRecommender)
    demographic: DemographicRecommender = field(default_factory=DemographicRecommender)
    ratings: pd.DataFrame | None = field(default=None, init=False)
    movies: pd.DataFrame | None = field(default=None, init=False)

    def fit(
        self,
        ratings: pd.DataFrame,
        users: pd.DataFrame,
        movies: pd.DataFrame | None = None,
    ) -> "HybridRecommender":
        self.alpha = float(np.clip(self.alpha, 0.0, 1.0))
        self.ratings = ratings.copy()
        self.movies = movies.copy() if movies is not None else None
        self.collaborative.fit(ratings, movies)
        self.demographic.fit(ratings, users, movies)
        return self

    def predict(self, user_id: int | None, movie_id: int, profile: dict | None = None) -> float:
        self._ensure_fit()
        demo_score = self.demographic.predict(user_id=user_id, movie_id=movie_id, profile=profile)
        if user_id is None:
            return demo_score
        try:
            cf_score = self.collaborative.predict(user_id, movie_id)
        except Exception:
            return demo_score
        return float(np.clip(self.alpha * cf_score + (1.0 - self.alpha) * demo_score, 1.0, 5.0))

    def recommend(
        self,
        user_id: int | None = None,
        profile: dict | None = None,
        top_n: int = 5,
        exclude_seen: bool = True,
    ) -> pd.DataFrame:
        self._ensure_fit()
        assert self.ratings is not None

        candidates = sorted(self.ratings["movie_id"].unique())
        if exclude_seen and user_id is not None:
            seen = set(self.ratings.loc[self.ratings["user_id"] == user_id, "movie_id"])
            candidates = [movie_id for movie_id in candidates if movie_id not in seen]

        scored = [(movie_id, self.predict(user_id, int(movie_id), profile)) for movie_id in candidates]
        rows = sorted(scored, key=lambda item: item[1], reverse=True)[:top_n]
        result = pd.DataFrame(rows, columns=["movie_id", "predicted_rating"])
        result["model"] = f"Hybrid alpha={self.alpha:.2f}"
        if self.movies is not None and not result.empty:
            result = result.merge(self.movies[["movie_id", "title", "genres"]], on="movie_id", how="left")
            result = result[["movie_id", "title", "genres", "predicted_rating", "model"]]
        return result

    def _ensure_fit(self) -> None:
        if self.ratings is None:
            raise RuntimeError("HybridRecommender.fit must be called first.")
