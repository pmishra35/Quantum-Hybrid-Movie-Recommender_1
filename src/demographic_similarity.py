from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class DemographicRecommender:
    neighbor_users: int = 100
    ratings: pd.DataFrame | None = field(default=None, init=False)
    users: pd.DataFrame | None = field(default=None, init=False)
    movies: pd.DataFrame | None = field(default=None, init=False)
    item_means: pd.Series | None = field(default=None, init=False)
    global_mean: float = field(default=3.5, init=False)
    user_profiles: pd.DataFrame | None = field(default=None, init=False)

    def fit(
        self,
        ratings: pd.DataFrame,
        users: pd.DataFrame,
        movies: pd.DataFrame | None = None,
    ) -> "DemographicRecommender":
        self.ratings = ratings.copy()
        self.users = users.copy()
        self.movies = movies.copy() if movies is not None else None
        self.global_mean = float(self.ratings["rating"].mean())
        self.item_means = self.ratings.groupby("movie_id")["rating"].mean()
        self.user_profiles = self.users.set_index("user_id")[["gender_encoded", "age_encoded", "occupation"]]
        return self

    def predict(self, user_id: int | None, movie_id: int, profile: dict | None = None) -> float:
        self._ensure_fit()
        assert self.ratings is not None
        assert self.item_means is not None

        neighbor_ids = self._neighbor_user_ids(user_id=user_id, profile=profile)
        scoped = self.ratings[(self.ratings["user_id"].isin(neighbor_ids)) & (self.ratings["movie_id"] == movie_id)]
        if not scoped.empty:
            return float(np.clip(scoped["rating"].mean(), 1.0, 5.0))
        return float(np.clip(self.item_means.get(movie_id, self.global_mean), 1.0, 5.0))

    def recommend(
        self,
        user_id: int | None = None,
        profile: dict | None = None,
        top_n: int = 5,
        exclude_seen: bool = True,
    ) -> pd.DataFrame:
        self._ensure_fit()
        assert self.ratings is not None

        neighbor_ids = self._neighbor_user_ids(user_id=user_id, profile=profile)
        scoped = self.ratings[self.ratings["user_id"].isin(neighbor_ids)]
        if scoped.empty:
            scoped = self.ratings

        scored = (
            scoped.groupby("movie_id")
            .agg(predicted_rating=("rating", "mean"), support=("rating", "size"))
            .reset_index()
        )
        if exclude_seen and user_id is not None:
            seen = set(self.ratings.loc[self.ratings["user_id"] == user_id, "movie_id"])
            scored = scored[~scored["movie_id"].isin(seen)]

        scored = scored.sort_values(["predicted_rating", "support"], ascending=[False, False]).head(top_n)
        scored["model"] = "Demographic Similarity"
        if self.movies is not None and not scored.empty:
            scored = scored.merge(self.movies[["movie_id", "title", "genres"]], on="movie_id", how="left")
            scored = scored[["movie_id", "title", "genres", "predicted_rating", "support", "model"]]
        return scored.reset_index(drop=True)

    def _neighbor_user_ids(self, user_id: int | None = None, profile: dict | None = None) -> list[int]:
        assert self.user_profiles is not None

        if profile is None and user_id is not None and user_id in self.user_profiles.index:
            target = self.user_profiles.loc[user_id].to_dict()
        elif profile is not None:
            target = self._normalize_profile(profile)
        else:
            return list(self.user_profiles.index[: self.neighbor_users])

        profiles = self.user_profiles.copy()
        gender_penalty = (profiles["gender_encoded"] != target["gender_encoded"]).astype(float)
        age_penalty = (profiles["age_encoded"] - target["age_encoded"]).abs() / max(float(profiles["age_encoded"].max()), 1.0)
        occupation_penalty = (profiles["occupation"] != target["occupation"]).astype(float)
        distance = gender_penalty + age_penalty + occupation_penalty
        if user_id is not None and user_id in distance.index:
            distance = distance.drop(index=user_id)
        return list(distance.sort_values().head(self.neighbor_users).index.astype(int))

    def _normalize_profile(self, profile: dict) -> dict:
        assert self.users is not None
        gender = profile.get("gender", profile.get("gender_encoded", "M"))
        gender_encoded = int(gender) if isinstance(gender, (int, np.integer)) else {"F": 0, "M": 1}.get(str(gender).upper(), 1)

        age_value = int(profile.get("age", profile.get("age_encoded", 25)))
        age_map = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}
        age_encoded = int(profile.get("age_encoded", age_map.get(age_value, 2)))
        occupation = int(profile.get("occupation", 0))
        return {"gender_encoded": gender_encoded, "age_encoded": age_encoded, "occupation": occupation}

    def _ensure_fit(self) -> None:
        if self.ratings is None or self.users is None or self.user_profiles is None:
            raise RuntimeError("DemographicRecommender.fit must be called first.")
