"""Text cleaning and trend extraction utilities."""

from __future__ import annotations

import re
import string
from collections import Counter

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


STOPWORDS = set(ENGLISH_STOP_WORDS)


def clean_text(text: object) -> str:
    """Normalize text for model prediction and keyword extraction."""
    value = "" if text is None else str(text)
    value = value.lower()
    value = re.sub(r"https?://\S+|www\.\S+", " ", value)
    value = value.translate(str.maketrans("", "", string.punctuation))
    value = re.sub(r"\d+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    words = [word for word in value.split() if word not in STOPWORDS and len(word) > 2]
    return " ".join(words)


def clean_dataframe_text(df: pd.DataFrame, source_col: str, target_col: str = "clean_text") -> pd.DataFrame:
    """Add a cleaned text column to a DataFrame."""
    cleaned = df.copy()
    cleaned[target_col] = cleaned[source_col].fillna("").astype(str).map(clean_text)
    return cleaned


def detect_trending_keywords(texts: pd.Series, top_n: int = 15) -> pd.DataFrame:
    """Return the most frequent useful keywords from a text series."""
    counter: Counter[str] = Counter()
    for text in texts.fillna("").astype(str):
        counter.update(clean_text(text).split())
    return pd.DataFrame(counter.most_common(top_n), columns=["keyword", "frequency"])


def remove_duplicate_articles(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate article rows using URL first and normalized title second."""
    if df.empty:
        return df
    deduped = df.copy()
    deduped["dedupe_title"] = deduped["title"].fillna("").astype(str).str.lower().str.strip()
    if "url" in deduped.columns:
        has_url = deduped["url"].fillna("").astype(str).str.strip() != ""
        with_url = deduped[has_url].drop_duplicates(subset=["url"], keep="first")
        without_url = deduped[~has_url]
        deduped = pd.concat([with_url, without_url], ignore_index=True)
    deduped = deduped.drop_duplicates(subset=["dedupe_title"], keep="first")
    return deduped.drop(columns=["dedupe_title"], errors="ignore").reset_index(drop=True)
