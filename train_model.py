"""Train and save the sentiment model.

Run manually when retraining is required:
    py -3 train_model.py
"""

from __future__ import annotations

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from config import MODEL_DIR, MODEL_PATH, TWITTER_TRAIN_FILE, TWITTER_VAL_FILE, VECTORIZER_PATH, setup_logging
from utils.preprocess import clean_dataframe_text


logger = setup_logging()


def load_twitter_dataset(path: str) -> pd.DataFrame:
    """Load the Kaggle Twitter entity sentiment dataset."""
    df = pd.read_csv(
        path,
        names=["id", "entity", "sentiment", "text"],
        header=None,
        encoding="utf-8",
        on_bad_lines="skip",
    )
    df = df.dropna(subset=["text", "sentiment"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["sentiment"] = df["sentiment"].astype(str).str.strip()
    df = df[df["text"] != ""].reset_index(drop=True)
    return df


def train_and_save_model(max_features: int = 7000) -> dict[str, object]:
    """Train the classifier, evaluate it, and save artifacts."""
    logger.info("Loading training and validation datasets.")
    train_df = clean_dataframe_text(load_twitter_dataset(str(TWITTER_TRAIN_FILE)), "text")
    val_df = clean_dataframe_text(load_twitter_dataset(str(TWITTER_VAL_FILE)), "text")

    train_df = train_df[train_df["clean_text"].str.strip() != ""]
    val_df = val_df[val_df["clean_text"].str.strip() != ""]

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    x_train = vectorizer.fit_transform(train_df["clean_text"])
    x_val = vectorizer.transform(val_df["clean_text"])

    model = LogisticRegression(max_iter=1200, class_weight="balanced", n_jobs=None)
    model.fit(x_train, train_df["sentiment"])

    predictions = model.predict(x_val)
    accuracy = accuracy_score(val_df["sentiment"], predictions)
    report = classification_report(val_df["sentiment"], predictions, zero_division=0)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    logger.info("Saved model to %s", MODEL_PATH)
    logger.info("Saved vectorizer to %s", VECTORIZER_PATH)
    logger.info("Validation accuracy: %.4f", accuracy)
    logger.info("\n%s", report)

    return {"accuracy": accuracy, "report": report, "model": model, "vectorizer": vectorizer}


if __name__ == "__main__":
    results = train_and_save_model()
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(results["report"])
