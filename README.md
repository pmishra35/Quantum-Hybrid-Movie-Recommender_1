# Quantum-Enhanced Hybrid Movie Recommendation System

M.Tech final-year project using MovieLens 1M to demonstrate a recommendation system that combines collaborative filtering, demographic similarity, classical SVD, and a practical quantum-inspired simulation.

## Problem Statement

Traditional recommendation systems struggle with:

- Cold-start users with no rating history.
- Sparse user-item matrices where most movies are unrated by most users.

This project addresses both by combining behavioral ratings with user demographics and comparing classical and quantum-inspired recommendation approaches.

## Project Modules

- `src/preprocessing.py`: loads MovieLens 1M, cleans users/movies/ratings, and creates the user-item matrix.
- `src/collaborative_filtering.py`: user-based collaborative filtering.
- `src/demographic_similarity.py`: cold-start recommendation using gender, age, and occupation.
- `src/svd_model.py`: classical SVD matrix factorization.
- `src/hybrid_model.py`: combines collaborative and demographic scores with alpha.
- `src/quantum_model.py`: compact quantum-inspired recommender using Qiskit statevectors when available.
- `src/evaluation.py`: RMSE, MAE, Precision@K, and Recall@K.
- `run_pipeline.py`: end-to-end preprocessing, training, evaluation, model saving, and report generation.
- `app/app.py`: Streamlit dashboard and recommender demo.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the Pipeline

```powershell
python run_pipeline.py
```

The default run uses a real sample of 500 MovieLens users so it finishes quickly on a laptop. Use the full dataset with:

```powershell
python run_pipeline.py --sample-users 0
```

Generated outputs:

- `data/processed/cleaned_ratings.csv`
- `data/processed/cleaned_users.csv`
- `data/processed/cleaned_movies.csv`
- `data/processed/user_item_matrix.csv`
- `models/svd_model.pkl`
- `models/hybrid_model.pkl`
- `models/quantum_model.pkl`
- `reports/evaluation_results.csv`
- `reports/final_report.md`

## Run the App

```powershell
streamlit run app/app.py
```

The app supports existing-user recommendations and cold-start recommendations based on gender, age, and occupation.

## Evaluation

The pipeline compares:

- Collaborative Filtering
- Classical SVD
- Demographic + CF Hybrid
- Quantum-Inspired Hybrid

Metrics:

- RMSE
- MAE
- Precision@K
- Recall@K

Ratings of 4 or 5 are treated as relevant for Precision@K and Recall@K.

## Dataset Credit

MovieLens 1M is provided by GroupLens. Cite:

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems 5, 4.
