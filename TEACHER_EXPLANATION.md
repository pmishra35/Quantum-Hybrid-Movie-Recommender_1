# Detailed Explanation of the Quantum-Enhanced Hybrid Movie Recommendation System

## 1. Project Overview

This project is a movie recommendation system built as an M.Tech final-year project. The main goal of the project is to show how a recommender system can suggest movies to users by combining multiple recommendation techniques:

- Collaborative Filtering
- Demographic Similarity
- Hybrid Recommendation
- Classical SVD Matrix Factorization
- Quantum-Inspired Recommendation

The project is presented through a Streamlit web application. The website allows a user to view the dataset summary, generate recommendations for existing users, generate recommendations for new cold-start users, and compare model evaluation results.

The system focuses on two common problems in recommendation systems:

1. Cold-start problem
2. Data sparsity problem

The cold-start problem happens when a new user has no previous ratings, so the system cannot understand their personal movie taste from history. To handle this, the project uses demographic information such as gender, age group, and occupation.

The data sparsity problem happens because most users rate only a small number of movies. In a large user-movie matrix, most values are empty. To handle this, the project uses collaborative filtering, SVD, and hybrid methods.

## 2. Dataset Used

The dataset used in this project is the MovieLens 1M dataset.

It is not an artificially generated dataset. It is a real public research dataset provided by GroupLens.

The raw dataset files are stored in:

```text
data/raw/ml-1m/
```

The main raw files are:

```text
movies.dat
ratings.dat
users.dat
```

The MovieLens 1M dataset contains approximately:

- 1,000,209 ratings
- 3,900 movies
- 6,040 users

The dataset contains anonymous movie ratings made by real MovieLens users. The users joined MovieLens around the year 2000.

The project README also credits the dataset source:

```text
MovieLens 1M is provided by GroupLens.
```

Recommended citation:

```text
F. Maxwell Harper and Joseph A. Konstan. 2015.
The MovieLens Datasets: History and Context.
ACM Transactions on Interactive Intelligent Systems 5, 4.
```

## 3. Where the Dataset Is Loaded in the Code

The dataset loading logic is written in:

```text
src/preprocessing.py
```

The raw dataset directory is defined as:

```python
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ml-1m"
```

The function responsible for loading the raw data is:

```python
load_raw_data()
```

Inside this function, the project reads:

```python
movies_path = raw_dir / "movies.dat"
ratings_path = raw_dir / "ratings.dat"
users_path = raw_dir / "users.dat"
```

Then it loads the files using Pandas:

```python
pd.read_csv(...)
```

The files use `::` as the separator, so the code reads them using:

```python
sep="::"
engine="python"
encoding="latin-1"
```

The columns assigned to the files are:

Movies:

```text
movie_id, title, genres
```

Ratings:

```text
user_id, movie_id, rating, timestamp
```

Users:

```text
user_id, gender, age, occupation, zip_code
```

## 4. Data Preprocessing

After loading the raw MovieLens files, the project cleans and prepares the data.

The main preprocessing function is:

```python
prepare_data()
```

This function performs the following steps:

1. Load raw MovieLens data.
2. Clean movie records.
3. Clean rating records.
4. Clean user records.
5. Optionally sample users for faster execution.
6. Create a user-item matrix.
7. Return the prepared dataset.

### 4.1 Movie Cleaning

The movie cleaning function is:

```python
clean_movies()
```

It performs these tasks:

- Fills missing genres with `Unknown`
- Extracts the release year from the movie title
- Creates a cleaned title without the year
- Removes duplicate movie records
- Sorts movies by movie ID

For example, a title like:

```text
Toy Story (1995)
```

is processed into:

```text
title: Toy Story (1995)
year: 1995
title_clean: Toy Story
```

### 4.2 Ratings Cleaning

The ratings cleaning function is:

```python
clean_ratings()
```

It performs these tasks:

- Removes duplicate rating rows
- Keeps only ratings between 1 and 5
- Creates a normalized rating column

The normalized rating is calculated as:

```python
rating_normalized = (rating - 1) / 4
```

This converts ratings from the 1 to 5 scale into a 0 to 1 scale.

### 4.3 User Cleaning

The user cleaning function is:

```python
clean_users()
```

It performs these tasks:

- Removes the zip code column
- Encodes gender
- Converts age codes into readable age labels
- Encodes age into numerical form
- Converts occupation codes into readable occupation labels

For gender:

```text
F = 0
M = 1
```

For occupation, the MovieLens occupation codes are mapped to labels such as:

- academic/educator
- artist
- college/grad student
- programmer
- scientist
- writer

## 5. Processed Dataset Files

After preprocessing, the cleaned data is saved into:

```text
data/processed/
```

The generated processed files are:

```text
cleaned_ratings.csv
cleaned_users.csv
cleaned_movies.csv
user_item_matrix.csv
```

These files are created by the function:

```python
save_processed_data()
```

The pipeline saves them using:

```python
data.ratings.to_csv(...)
data.users.to_csv(...)
data.movies.to_csv(...)
data.user_item_matrix.to_csv(...)
```

The `user_item_matrix.csv` file is especially important because it represents users as rows and movies as columns. Each cell contains the rating given by a user to a movie. If the user did not rate a movie, the value is filled with 0.

## 6. Pipeline Flow

The full project pipeline is controlled by:

```text
run_pipeline.py
```

The main function is:

```python
run_pipeline()
```

The pipeline performs these steps:

1. Prepare the dataset.
2. Save processed data.
3. Split ratings into training and testing data.
4. Train all recommendation models.
5. Evaluate all models.
6. Save evaluation results.
7. Save trained model files.
8. Generate a final report.

The pipeline can be run using:

```powershell
python run_pipeline.py
```

By default, the project uses a real sample of 500 MovieLens users. This makes the project faster to run on a normal laptop.

To run the full dataset, the command is:

```powershell
python run_pipeline.py --sample-users 0
```

## 7. Train-Test Split

The evaluation module is:

```text
src/evaluation.py
```

The project splits the ratings by user. This means that each user's ratings are divided into training and testing portions.

The function used is:

```python
train_test_split_by_user()
```

The default test fraction is:

```text
20%
```

For users with enough ratings, some ratings are kept for testing and the rest are used for training. This helps evaluate whether the model can predict ratings that were hidden during training.

## 8. Recommendation Models

The project compares multiple recommendation models. Each model solves the recommendation problem in a different way.

## 8.1 Collaborative Filtering

File:

```text
src/collaborative_filtering.py
```

Collaborative filtering recommends movies based on similar users.

The basic idea is:

```text
If User A and User B rated many movies similarly, then movies liked by User B may also be liked by User A.
```

In this project, collaborative filtering works as follows:

1. Create a user-movie rating matrix.
2. Center ratings by subtracting each user's average rating.
3. Calculate similarity between users.
4. For a target user and movie, find similar users who rated that movie.
5. Predict the rating using a weighted average of neighbor ratings.

The model uses user similarity and selects the top similar users as neighbors.

If the model does not have enough information for a user or movie, it falls back to item average rating or global average rating.

## 8.2 Demographic Similarity

File:

```text
src/demographic_similarity.py
```

Demographic similarity is used mainly for cold-start recommendation.

Cold-start means the user has no previous rating history. In that case, collaborative filtering cannot work properly because there is no behavioral data for that user.

This model uses demographic attributes:

- Gender
- Age group
- Occupation

The model finds users who have similar demographic profiles. Then it recommends movies that those similar users rated highly.

For a new user, the system asks for:

- Gender
- Age group
- Occupation

Then it creates a profile and finds similar users from the dataset.

This allows the system to recommend movies even when the user has never rated anything before.

## 8.3 Hybrid Model

File:

```text
src/hybrid_model.py
```

The hybrid model combines:

- Collaborative Filtering
- Demographic Similarity

It uses a parameter called `alpha`.

The formula is:

```text
Final Score = alpha * Collaborative Score + (1 - alpha) * Demographic Score
```

If `alpha` is high, the system gives more importance to collaborative filtering.

If `alpha` is low, the system gives more importance to demographic similarity.

For example:

```text
alpha = 0.75
```

means:

```text
75% collaborative filtering
25% demographic similarity
```

For a new cold-start user, the system uses demographic similarity directly because no user rating history exists.

## 8.4 Classical SVD

File:

```text
src/svd_model.py
```

SVD stands for Singular Value Decomposition.

In recommendation systems, SVD is used for matrix factorization. It tries to reduce the large user-movie rating matrix into smaller hidden feature matrices.

The idea is that both users and movies can be represented using latent factors.

Examples of possible hidden factors could be:

- Preference for comedy
- Preference for action
- Preference for drama
- Preference for old movies
- Preference for animated movies

The model does not explicitly name these factors, but it mathematically discovers patterns from the rating matrix.

In this project, the SVD model:

1. Builds a user-item matrix.
2. Fills missing ratings using user averages.
3. Centers the matrix by subtracting user averages.
4. Applies NumPy SVD.
5. Reconstructs the rating matrix.
6. Predicts missing user-movie ratings.

The SVD model helps handle sparsity because it can estimate ratings for movies a user has not rated.

## 8.5 Quantum-Inspired Hybrid Model

File:

```text
src/quantum_model.py
```

The quantum-inspired model is not a full quantum computer implementation. It is a practical simulation inspired by quantum state comparison.

This model first uses the SVD recommender to create compact latent vectors for users and movies.

Then it compares the user vector and movie vector using a quantum-inspired similarity method.

If Qiskit is installed, the model can create small quantum circuits and use statevector simulation.

The basic flow is:

1. Train an SVD model.
2. Extract user and movie latent vectors.
3. Normalize the vectors.
4. Convert vectors into small simulated quantum states using rotation gates.
5. Compare the similarity between user and movie states.
6. Combine this similarity with the SVD baseline prediction.

The final prediction is calculated using:

```text
Adjusted Score = 0.75 * SVD Baseline + 0.25 * Quantum Similarity Score
```

If Qiskit is not available, the model falls back to a normal vector similarity calculation.

This makes the model practical because it can still run without requiring a real quantum computer.

## 9. Website Structure

The website is built using Streamlit.

The main app file is:

```text
app/app.py
```

The website can be started using:

```powershell
streamlit run app/app.py
```

The app has four main pages:

1. Home
2. Recommend
3. Evaluation
4. About

Navigation is handled through the sidebar.

The sidebar also allows the user to choose:

- Number of recommendations: 5 or 10
- Hybrid alpha value: 0.25, 0.5, 0.75, or 1.0

## 10. Home Page

File:

```text
app/views/home.py
```

The Home page introduces the project:

```text
Quantum-Enhanced Hybrid Movie Recommendation System
```

It displays:

- Total number of ratings
- Total number of users
- Total number of movies
- Dataset sparsity
- Project focus
- Dataset snapshot

The dataset snapshot shows the first few movie records with:

- movie_id
- title
- genres

This page is useful for explaining the dataset and project objective to the teacher.

## 11. Recommend Page

File:

```text
app/views/recommend.py
```

The Recommend page has two tabs:

1. Existing user
2. Cold-start user

### Existing User Recommendation

For an existing user, the app provides a dropdown of user IDs.

When a user ID is selected, the hybrid model recommends movies that the user has not already rated.

The recommendation output contains:

- movie_id
- title
- genres
- predicted_rating
- model

### Cold-Start User Recommendation

For a cold-start user, the app asks for:

- Gender
- Age group
- Occupation

The system then uses demographic similarity to recommend movies.

This is important because a new user has no rating history, so the system cannot rely on collaborative filtering alone.

## 12. Evaluation Page

File:

```text
app/views/evaluation.py
```

The Evaluation page displays model comparison results from:

```text
reports/evaluation_results.csv
```

It shows the performance of:

- Collaborative Filtering
- Classical SVD
- Demographic + CF Hybrid
- Quantum-Inspired Hybrid

The metrics shown are:

- RMSE
- MAE
- Precision@K
- Recall@K
- Evaluated ratings
- Evaluated users

The page also displays a chart for easier comparison.

## 13. About Page

File:

```text
app/views/about.py
```

The About page explains the purpose of the system and briefly describes the models.

It mentions that:

- Collaborative filtering recommends based on similar users.
- Demographic similarity supports new users.
- The hybrid model combines collaborative and demographic scores.
- Classical SVD reconstructs sparse ratings.
- The quantum-inspired model uses compact latent vectors and Qiskit statevector simulation when available.

## 14. Evaluation Metrics

The project uses four main metrics.

### 14.1 RMSE

RMSE means Root Mean Squared Error.

It measures how far the predicted ratings are from the actual ratings.

Lower RMSE is better.

### 14.2 MAE

MAE means Mean Absolute Error.

It measures the average absolute difference between predicted ratings and actual ratings.

Lower MAE is better.

### 14.3 Precision@K

Precision@K measures how many of the top K recommended movies are actually relevant.

In this project, ratings of 4 or 5 are treated as relevant.

Higher Precision@K is better.

### 14.4 Recall@K

Recall@K measures how many relevant movies were successfully recommended in the top K list.

Higher Recall@K is better.

## 15. Current Evaluation Results

The current evaluation results are stored in:

```text
reports/evaluation_results.csv
```

The current results are:

| Model | RMSE | MAE | Precision@K | Recall@K |
| --- | ---: | ---: | ---: | ---: |
| Collaborative Filtering | 1.0241 | 0.7964 | 0.0000 | 0.0000 |
| Classical SVD | 0.9937 | 0.7960 | 0.0400 | 0.0046 |
| Demographic + CF Hybrid | 1.0092 | 0.7879 | 0.0000 | 0.0000 |
| Quantum-Inspired Hybrid | 0.9898 | 0.8080 | 0.0200 | 0.0059 |

From these results, the Quantum-Inspired Hybrid has the best RMSE among the listed models in the current run. The Demographic + CF Hybrid has the best MAE in the current run.

The Precision@K and Recall@K values are low because the evaluation is done on a limited sample and only top recommendations are compared against hidden high ratings. This is common in small sampled recommendation experiments.

## 16. Why This Project Is Useful

This project is useful because it demonstrates a practical recommendation system using real data.

It shows how different recommendation strategies can be used for different situations:

- Existing users can be served using collaborative filtering.
- New users can be served using demographic similarity.
- Sparse data can be handled using SVD.
- Hybrid recommendation can combine multiple signals.
- Quantum-inspired simulation can be explored as an advanced comparison approach.

The project also gives a complete workflow:

```text
Raw Dataset -> Preprocessing -> Model Training -> Evaluation -> Web Application
```

This makes the project suitable for academic presentation because it includes both the technical implementation and the business problem.

## 17. Simple Explanation to Tell the Teacher

This website is a movie recommendation system built using the MovieLens 1M dataset. The system recommends movies to users by learning from past ratings, user demographics, and matrix factorization methods.

For existing users, it checks the user's rating history and finds similar users. Based on what similar users liked, it recommends movies.

For new users, it solves the cold-start problem by asking for gender, age group, and occupation. It then finds similar users from the dataset and recommends movies liked by those users.

The project also compares different models such as collaborative filtering, SVD, hybrid recommendation, and a quantum-inspired model. The evaluation page shows metrics like RMSE, MAE, Precision@K, and Recall@K.

The main aim is to demonstrate how a hybrid recommendation system can reduce cold-start and sparsity problems using a real-world dataset.

## 18. Technical File Map

Important files in the project:

```text
README.md
```

Explains the project, setup, pipeline, app, evaluation, and dataset credit.

```text
src/preprocessing.py
```

Loads MovieLens 1M, cleans the dataset, samples users, and creates the user-item matrix.

```text
src/collaborative_filtering.py
```

Implements user-based collaborative filtering.

```text
src/demographic_similarity.py
```

Implements demographic-based recommendation for cold-start users.

```text
src/hybrid_model.py
```

Combines collaborative filtering and demographic similarity using alpha.

```text
src/svd_model.py
```

Implements classical SVD matrix factorization.

```text
src/quantum_model.py
```

Implements the quantum-inspired recommender using SVD latent vectors and optional Qiskit simulation.

```text
src/evaluation.py
```

Implements train-test split and evaluation metrics.

```text
run_pipeline.py
```

Runs preprocessing, training, evaluation, model saving, and report generation.

```text
app/app.py
```

Main Streamlit application entry point.

```text
app/views/home.py
```

Home page of the website.

```text
app/views/recommend.py
```

Recommendation page for existing users and cold-start users.

```text
app/views/evaluation.py
```

Evaluation comparison page.

```text
app/views/about.py
```

About page explaining the models and evaluation.

## 19. How to Run the Project

First install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the pipeline:

```powershell
python run_pipeline.py
```

Run the website:

```powershell
streamlit run app/app.py
```

## 20. Presentation Flow

A good presentation flow for this project is:

1. Start with the problem: recommendation systems face cold-start and sparsity.
2. Explain the dataset: MovieLens 1M from GroupLens.
3. Explain preprocessing: cleaning movies, users, ratings, and creating the user-item matrix.
4. Explain models: collaborative filtering, demographic similarity, hybrid model, SVD, and quantum-inspired model.
5. Open the website and show the Home page.
6. Show existing-user recommendation.
7. Show cold-start recommendation.
8. Show evaluation comparison.
9. Conclude with how the hybrid approach helps combine user behavior and demographics.

## 21. Conclusion

The Quantum-Enhanced Hybrid Movie Recommendation System is a complete end-to-end recommendation project. It uses a real dataset, performs preprocessing, trains multiple models, evaluates them, and presents the results through a web application.

The most important contribution of the project is that it does not depend on only one recommendation method. It combines collaborative filtering and demographic similarity to support both existing users and new users.

The classical SVD model helps handle sparse rating data, and the quantum-inspired model adds an advanced experimental layer by comparing latent vectors using quantum-style state similarity.

Overall, the project demonstrates how recommendation systems can be built, evaluated, and explained in a practical business and technical context.
