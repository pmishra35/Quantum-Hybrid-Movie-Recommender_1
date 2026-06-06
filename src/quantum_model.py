from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .svd_model import SVDRecommender


try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    QISKIT_AVAILABLE = True
except Exception:
    QuantumCircuit = None
    Statevector = None
    QISKIT_AVAILABLE = False


@dataclass
class QuantumInspiredRecommender:
    n_factors: int = 4
    svd_model: SVDRecommender = field(init=False)
    ratings: pd.DataFrame | None = field(default=None, init=False)
    movies: pd.DataFrame | None = field(default=None, init=False)
    qiskit_available: bool = field(default=QISKIT_AVAILABLE, init=False)
    _state_cache: dict[tuple[float, ...], object] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.svd_model = SVDRecommender(n_factors=self.n_factors)

    def fit(self, ratings: pd.DataFrame, movies: pd.DataFrame | None = None) -> "QuantumInspiredRecommender":
        self.ratings = ratings.copy()
        self.movies = movies.copy() if movies is not None else None
        self.svd_model.fit(ratings, movies)
        self._state_cache.clear()
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        self._ensure_fit()
        if user_id not in self.svd_model.user_index or movie_id not in self.svd_model.movie_index:
            return self.svd_model.predict(user_id, movie_id)

        user_vec = self.svd_model.user_factors[self.svd_model.user_index[user_id]]
        item_vec = self.svd_model.item_factors[self.svd_model.movie_index[movie_id]]
        similarity = self._quantum_similarity(user_vec, item_vec)
        baseline = self.svd_model.predict(user_id, movie_id)
        adjusted = 0.75 * baseline + 0.25 * (1.0 + 4.0 * similarity)
        return float(np.clip(adjusted, 1.0, 5.0))

    def recommend(self, user_id: int, top_n: int = 5, exclude_seen: bool = True) -> pd.DataFrame:
        self._ensure_fit()
        assert self.ratings is not None

        candidates = self.svd_model.movie_ids
        if exclude_seen:
            seen = set(self.ratings.loc[self.ratings["user_id"] == user_id, "movie_id"])
            candidates = [movie_id for movie_id in candidates if movie_id not in seen]
        scored = [(movie_id, self.predict(user_id, movie_id)) for movie_id in candidates]
        rows = sorted(scored, key=lambda item: item[1], reverse=True)[:top_n]
        result = pd.DataFrame(rows, columns=["movie_id", "predicted_rating"])
        result["model"] = "Quantum-Inspired Hybrid"
        result["qiskit_simulated"] = self.qiskit_available
        if self.movies is not None and not result.empty:
            result = result.merge(self.movies[["movie_id", "title", "genres"]], on="movie_id", how="left")
            result = result[["movie_id", "title", "genres", "predicted_rating", "model", "qiskit_simulated"]]
        return result

    def _quantum_similarity(self, user_vec: np.ndarray, item_vec: np.ndarray) -> float:
        user_norm = self._normalize(user_vec)
        item_norm = self._normalize(item_vec)
        if self.qiskit_available and QuantumCircuit is not None and Statevector is not None:
            try:
                user_state = self._state_for_vector(user_norm)
                item_state = self._state_for_vector(item_norm)
                return float(np.clip(abs(np.vdot(user_state.data, item_state.data)) ** 2, 0.0, 1.0))
            except Exception:
                self.qiskit_available = False
        return float(np.clip((np.dot(user_norm, item_norm) + 1.0) / 2.0, 0.0, 1.0))

    def _state_for_vector(self, vector: np.ndarray):
        assert Statevector is not None
        key = tuple(float(value) for value in np.round(vector[: self.n_factors], 8))
        if key not in self._state_cache:
            self._state_cache[key] = Statevector.from_instruction(self._angle_circuit(vector))
        return self._state_cache[key]

    def _angle_circuit(self, vector: np.ndarray):
        assert QuantumCircuit is not None
        dims = min(len(vector), self.n_factors)
        circuit = QuantumCircuit(dims)
        for idx, value in enumerate(vector[:dims]):
            angle = float((value + 1.0) * np.pi / 2.0)
            circuit.ry(angle, idx)
        return circuit

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=float)
        if vector.size == 0:
            return vector
        norm = np.linalg.norm(vector)
        if np.isclose(norm, 0.0):
            return np.zeros_like(vector)
        return np.clip(vector / norm, -1.0, 1.0)

    def _ensure_fit(self) -> None:
        if self.ratings is None:
            raise RuntimeError("QuantumInspiredRecommender.fit must be called first.")

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_state_cache"] = {}
        return state
