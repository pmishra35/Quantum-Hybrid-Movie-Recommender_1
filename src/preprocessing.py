from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ml-1m"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

AGE_LABELS = {
    1: "Under 18",
    18: "18-24",
    25: "25-34",
    35: "35-44",
    45: "45-49",
    50: "50-55",
    56: "56+",
}

OCCUPATION_LABELS = {
    0: "other",
    1: "academic/educator",
    2: "artist",
    3: "clerical/admin",
    4: "college/grad student",
    5: "customer service",
    6: "doctor/health care",
    7: "executive/managerial",
    8: "farmer",
    9: "homemaker",
    10: "K-12 student",
    11: "lawyer",
    12: "programmer",
    13: "retired",
    14: "sales/marketing",
    15: "scientist",
    16: "self-employed",
    17: "technician/engineer",
    18: "tradesman/craftsman",
    19: "unemployed",
    20: "writer",
}


@dataclass(frozen=True)
class PreparedData:
    ratings: pd.DataFrame
    users: pd.DataFrame
    movies: pd.DataFrame
    user_item_matrix: pd.DataFrame


def load_raw_data(raw_dir: Path | str = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load MovieLens 1M movies, ratings, and users files."""
    raw_dir = Path(raw_dir)
    movies_path = raw_dir / "movies.dat"
    ratings_path = raw_dir / "ratings.dat"
    users_path = raw_dir / "users.dat"

    missing = [path for path in (movies_path, ratings_path, users_path) if not path.exists()]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing MovieLens raw file(s): {names}")

    movies = pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )
    ratings = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )
    users = pd.read_csv(
        users_path,
        sep="::",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip_code"],
        encoding="latin-1",
    )
    return movies, ratings, users


def clean_movies(movies: pd.DataFrame) -> pd.DataFrame:
    cleaned = movies.copy()
    cleaned["genres"] = cleaned["genres"].fillna("Unknown")
    cleaned["year"] = cleaned["title"].str.extract(r"\((\d{4})\)\s*$").astype("float").astype("Int64")
    cleaned["title_clean"] = cleaned["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
    return cleaned.drop_duplicates("movie_id").sort_values("movie_id").reset_index(drop=True)


def clean_users(users: pd.DataFrame) -> pd.DataFrame:
    cleaned = users.copy()
    cleaned = cleaned.drop(columns=["zip_code"], errors="ignore")
    cleaned["gender_encoded"] = cleaned["gender"].map({"F": 0, "M": 1}).astype("int64")
    cleaned["age_label"] = cleaned["age"].map(AGE_LABELS).fillna("Unknown")
    cleaned["age_encoded"] = cleaned["age"].rank(method="dense").astype("int64") - 1
    cleaned["occupation_label"] = cleaned["occupation"].map(OCCUPATION_LABELS).fillna("other")
    return cleaned.drop_duplicates("user_id").sort_values("user_id").reset_index(drop=True)


def clean_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    cleaned = ratings.copy()
    cleaned = cleaned.drop_duplicates()
    cleaned = cleaned[(cleaned["rating"] >= 1) & (cleaned["rating"] <= 5)]
    cleaned["rating_normalized"] = (cleaned["rating"] - 1) / 4
    return cleaned.sort_values(["user_id", "movie_id"]).reset_index(drop=True)


def sample_by_users(
    ratings: pd.DataFrame,
    users: pd.DataFrame,
    sample_users: int | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Optionally sample users for fast local training while keeping real MovieLens records."""
    if sample_users is None or sample_users <= 0 or sample_users >= users["user_id"].nunique():
        return ratings.copy(), users.copy()

    rng = np.random.default_rng(random_state)
    selected_users = rng.choice(users["user_id"].to_numpy(), size=sample_users, replace=False)
    selected_users = set(int(user_id) for user_id in selected_users)
    sampled_users = users[users["user_id"].isin(selected_users)].copy()
    sampled_ratings = ratings[ratings["user_id"].isin(selected_users)].copy()
    return sampled_ratings.reset_index(drop=True), sampled_users.reset_index(drop=True)


def create_user_item_matrix(ratings: pd.DataFrame, fill_value: float = 0.0) -> pd.DataFrame:
    matrix = ratings.pivot_table(index="user_id", columns="movie_id", values="rating", aggfunc="mean")
    return matrix.fillna(fill_value).sort_index(axis=0).sort_index(axis=1)


def prepare_data(
    raw_dir: Path | str = RAW_DIR,
    sample_users: int | None = None,
    random_state: int = 42,
) -> PreparedData:
    movies, ratings, users = load_raw_data(raw_dir)
    movies = clean_movies(movies)
    ratings = clean_ratings(ratings)
    users = clean_users(users)
    ratings, users = sample_by_users(ratings, users, sample_users, random_state)
    matrix = create_user_item_matrix(ratings)
    return PreparedData(ratings=ratings, users=users, movies=movies, user_item_matrix=matrix)


def save_processed_data(data: PreparedData, processed_dir: Path | str = PROCESSED_DIR) -> None:
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    data.ratings.to_csv(processed_dir / "cleaned_ratings.csv", index=False)
    data.users.to_csv(processed_dir / "cleaned_users.csv", index=False)
    data.movies.to_csv(processed_dir / "cleaned_movies.csv", index=False)
    data.user_item_matrix.to_csv(processed_dir / "user_item_matrix.csv")


def load_processed_data(processed_dir: Path | str = PROCESSED_DIR) -> PreparedData:
    processed_dir = Path(processed_dir)
    ratings = pd.read_csv(processed_dir / "cleaned_ratings.csv")
    users = pd.read_csv(processed_dir / "cleaned_users.csv")
    movies_path = processed_dir / "cleaned_movies.csv"
    if movies_path.exists():
        movies = pd.read_csv(movies_path)
    else:
        movies, _, _ = load_raw_data()
        movies = clean_movies(movies)
    matrix = pd.read_csv(processed_dir / "user_item_matrix.csv", index_col=0)
    matrix.index = matrix.index.astype(int)
    matrix.columns = matrix.columns.astype(int)
    return PreparedData(ratings=ratings, users=users, movies=movies, user_item_matrix=matrix)
