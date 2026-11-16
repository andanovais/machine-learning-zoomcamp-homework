# Steam Game Recommendation & Rating Predictor

## Problem Description
With over 50,000 games on Steam, raw positive review percentages can be misleading when niche titles have only a handful of reviews. This project uses the **Game Recommendations on Steam** dataset (`antonkozyriev/game-recommendations-on-steam`) to combine Bayesian popularity-weighted approval ratios, Steam review tiers, pricing, and genre tags to recommend similar games and predict whether a player will recommend a title.

## Exploratory Data Analysis (EDA) & Feature Engineering
* **Class Imbalance:** Roughly 85% of user reviews in the dataset are positive (`is_recommended = 1`), making **ROC AUC** the ideal evaluation metric over raw accuracy.
* **Long-Tail Review Skew:** Most Steam titles have fewer than 189 reviews (75th percentile), while a small fraction of blockbuster titles accumulate hundreds of thousands of reviews.
* **Bayesian Weighted Positive Ratio:** Applied the IMDb weighted rating formula (`m = 189`) to pull low-review games toward the global mean approval rating.

## Models Evaluated & Results

| Model | Tuned Hyperparameters | Metric | Validation Score | Final Test Score |
| :--- | :--- | :--- | :--- | :--- |
| **Constant Baseline** | Training class mean | ROC AUC | `0.5000` | — |
| **Content-Based Hybrid (TF-IDF)** | `weight = 0.2` (20% similarity, 80% weighted ratio) | Hit Rate@10 | `0.0656` (`6.56%`) | — |
| **Random Forest Classifier** | `max_depth = 15`, `n_estimators = 50`, `min_samples_leaf = 10` | ROC AUC | `0.7294` | — |
| **Logistic Regression (Winner)** | `C = 0.01`, `solver = 'liblinear'` | ROC AUC | **`0.7297`** | **`0.7223`** |

## How to Run Locally & with Docker

### 1. Train & Export the Model
```bash
python train.py
```

### 2. Build & Run the Docker Container
```bash
docker build -t steam-predictor .
docker run -it --rm -p 9696:9696 steam-predictor
```

### 3. Test the Prediction Endpoint
In a separate terminal:
```bash
python test_predict.py
```