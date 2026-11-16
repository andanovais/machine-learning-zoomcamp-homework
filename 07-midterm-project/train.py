import os
import pickle
from pathlib import Path
import kagglehub
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def load_and_prepare_data():
    print("Downloading/loading Steam dataset via kagglehub...")
    path = kagglehub.dataset_download("antonkozyriev/game-recommendations-on-steam")

    df_games = pd.read_csv(os.path.join(path, "games.csv"))
    df_games.drop(
        columns=[
            "date_release", "win", "mac", "linux",
            "price_original", "discount", "steam_deck",
        ],
        inplace=True,
    )

    df_metadata = pd.read_json(os.path.join(path, "games_metadata.json"), lines=True)
    df_metadata.drop(columns=["description"], inplace=True)

    df_recommendations = pd.read_csv(
        os.path.join(path, "recommendations.csv"),
        skiprows=lambda i: i > 0 and i % 40 != 0,
        nrows=500000,
        usecols=["app_id", "user_id", "is_recommended"],
    )
    df_recommendations["is_recommended"] = df_recommendations["is_recommended"].astype(int)

    # Merge games and metadata, format tags, and calculate Bayesian weighted_positive_ratio
    df_games_with_tags = pd.merge(df_games, df_metadata, on="app_id", how="left")
    df_games_with_tags["tags"] = df_games_with_tags["tags"].apply(
        lambda x: ", ".join([t.lower().strip() for t in x])
        if isinstance(x, list) and len(x) > 0
        else "unknown"
    )

    v = df_games_with_tags["user_reviews"]
    m = 189
    R = df_games_with_tags["positive_ratio"]
    C = df_games_with_tags["positive_ratio"].mean()
    df_games_with_tags["weighted_positive_ratio"] = (v / (v + m)) * R + (m / (v + m)) * C

    # Sample 100k reviews and merge game features
    df_games_linear = df_games_with_tags[
        ["app_id", "weighted_positive_ratio", "price_final", "rating", "tags"]
    ].copy()
    df_recs_sampled = df_recommendations.sample(100000, random_state=1)
    del df_recs_sampled["user_id"]

    df_dataset = pd.merge(df_recs_sampled, df_games_linear, on="app_id", how="left")
    del df_dataset["app_id"]
    df_dataset = df_dataset.dropna().reset_index(drop=True)
    df_dataset["tags"] = df_dataset["tags"].str.split(",").str[0].str.strip()

    return df_dataset


def train():
    df_dataset = load_and_prepare_data()

    df_full_train, df_test = train_test_split(
        df_dataset, test_size=0.2, random_state=1
    )

    y_full_train = df_full_train["is_recommended"].values
    y_test = df_test["is_recommended"].values

    df_full_train_clean = df_full_train.drop(columns=["is_recommended"])
    df_test_clean = df_test.drop(columns=["is_recommended"])

    print("Vectorizing features and training LogisticRegression(C=0.01)...")
    dv = DictVectorizer(sparse=True)
    X_full_train = dv.fit_transform(df_full_train_clean.to_dict(orient="records"))
    X_test = dv.transform(df_test_clean.to_dict(orient="records"))

    model = LogisticRegression(solver="liblinear", C=0.01, max_iter=1000, random_state=1)
    model.fit(X_full_train, y_full_train)

    y_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    print(f"LogisticRegression (C=0.01) ---> Final Test ROC AUC = {auc:.4f}")

    project_dir = Path(__file__).resolve().parent
    output_file = project_dir / "model.bin"
    with open(output_file, "wb") as f_out:
        pickle.dump((dv, model), f_out)
    print(f"Saved model.bin successfully to: {output_file}")


if __name__ == "__main__":
    train()