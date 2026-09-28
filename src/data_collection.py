"""
data_collection.py
--------------------------------------------------------------------
PURPOSE:
    This module is responsible ONLY for loading raw datasets used by
    the Social Media Trend Monitoring Agent:

        1. News dataset        -> columns: title, source
        2. twitter_training.csv  -> used to TRAIN the sentiment model
        3. twitter_validation.csv -> used to TEST/VALIDATE the model

    All functions return clean pandas DataFrames with missing values
    already handled, so downstream modules (preprocessing, sentiment
    analysis, trend detection) can work directly on the output.
--------------------------------------------------------------------
"""

import os
import pandas as pd

# ------------------------------------------------------------------
# Default file locations (all datasets are expected inside data/)
# ------------------------------------------------------------------
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)

NEWS_FILE = os.path.join(DATA_DIR, "news_data.csv")
TWITTER_TRAIN_FILE = os.path.join(DATA_DIR, "twitter_training.csv")
TWITTER_VAL_FILE = os.path.join(DATA_DIR, "twitter_validation.csv")


# ==================================================================
# 1. LOAD NEWS DATASET
# ==================================================================
def load_news_data(file_path: str = NEWS_FILE) -> pd.DataFrame:
    """
    Load the News dataset which contains 'title' and 'source' columns.

    Handles missing values by:
        - Filling missing titles with an empty string, then dropping them
        - Filling missing sources with 'Unknown'
    """
    df = pd.read_csv(file_path)

    # Keep useful columns when they exist; the insights page can use the
    # richer metadata while still working with a minimal title/source file.
    expected_cols = ["title", "source", "published_at", "description"]
    available_cols = [c for c in expected_cols if c in df.columns]
    df = df[available_cols]

    # ---- Handle missing values ----
    if "title" in df.columns:
        df["title"] = df["title"].fillna("")
    if "source" in df.columns:
        df["source"] = df["source"].fillna("Unknown")

    # Drop rows where the title is empty (cannot be used for trend detection)
    df = df[df["title"].astype(str).str.strip() != ""]
    df = df.reset_index(drop=True)

    return df


# ==================================================================
# 2. LOAD TWITTER TRAINING DATASET
# ==================================================================
def load_twitter_training(file_path: str = TWITTER_TRAIN_FILE) -> pd.DataFrame:
    """
    Load twitter_training.csv.

    NOTE: The standard Kaggle "Twitter Entity Sentiment Analysis" dataset
    has NO header row and 4 columns in this order:
        [id, entity, sentiment, text]
    """
    df = pd.read_csv(
        file_path,
        names=["id", "entity", "sentiment", "text"],
        header=None,
        encoding="utf-8",
        on_bad_lines="skip"
    )
    df = _clean_twitter_df(df)
    return df


# ==================================================================
# 3. LOAD TWITTER VALIDATION DATASET
# ==================================================================
def load_twitter_validation(file_path: str = TWITTER_VAL_FILE) -> pd.DataFrame:
    """
    Load twitter_validation.csv.
    Same column structure as twitter_training.csv: [id, entity, sentiment, text]
    """
    df = pd.read_csv(
        file_path,
        names=["id", "entity", "sentiment", "text"],
        header=None,
        encoding="utf-8",
        on_bad_lines="skip"
    )
    df = _clean_twitter_df(df)
    return df


# ==================================================================
# HELPER: Clean a twitter dataframe (shared by train & validation)
# ==================================================================
def _clean_twitter_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values for twitter datasets:
        - Drop rows with missing text or sentiment
        - Strip extra whitespace
        - Remove rows that became empty after stripping
    """
    df = df.dropna(subset=["text", "sentiment"]).copy()

    df["text"] = df["text"].astype(str).str.strip()
    df["sentiment"] = df["sentiment"].astype(str).str.strip()

    df = df[df["text"] != ""]
    df = df.reset_index(drop=True)

    return df


# ==================================================================
# Quick manual test (only runs when this file is executed directly)
# ==================================================================
if __name__ == "__main__":
    news_df = load_news_data()
    train_df = load_twitter_training()
    val_df = load_twitter_validation()

    print("News data shape:", news_df.shape)
    print("Twitter training shape:", train_df.shape)
    print("Twitter validation shape:", val_df.shape)
