"""
preprocessing.py
--------------------------------------------------------------------
PURPOSE:
    Provides reusable text-cleaning functions used by BOTH:
        1. sentiment_analysis.py  (cleans tweet text)
        2. trend_detection.py     (cleans news titles)

    Cleaning steps performed (in order):
        1. Convert text to lowercase
        2. Remove URLs
        3. Remove punctuation
        4. Remove numbers
        5. Remove stopwords (using NLTK's English stopword list)
--------------------------------------------------------------------
"""

import re
import string
import nltk
import pandas as pd
from nltk.corpus import stopwords

# ------------------------------------------------------------------
# Make sure NLTK's stopword corpus is available.
# If it isn't downloaded yet, download it once (requires internet
# the FIRST time only; after that it is cached locally).
# ------------------------------------------------------------------
try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOPWORDS = set(stopwords.words("english"))


# ==================================================================
# 1. LOWERCASE
# ==================================================================
def to_lowercase(text: str) -> str:
    """Convert all characters in text to lowercase."""
    return text.lower()


# ==================================================================
# 2. REMOVE URLS
# ==================================================================
def remove_urls(text: str) -> str:
    """Remove http/https/www links from text."""
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    return url_pattern.sub("", text)


# ==================================================================
# 3. REMOVE PUNCTUATION
# ==================================================================
def remove_punctuation(text: str) -> str:
    """Remove all punctuation characters (!, ?, ., ,, # etc.)."""
    return text.translate(str.maketrans("", "", string.punctuation))


# ==================================================================
# 4. REMOVE NUMBERS
# ==================================================================
def remove_numbers(text: str) -> str:
    """Remove all numeric digits from text."""
    return re.sub(r"\d+", "", text)


# ==================================================================
# 5. REMOVE STOPWORDS
# ==================================================================
def remove_stopwords(text: str) -> str:
    """Remove common English stopwords (the, is, at, on, etc.) using NLTK."""
    words = text.split()
    filtered_words = [w for w in words if w not in STOPWORDS]
    return " ".join(filtered_words)


# ==================================================================
# FULL CLEANING PIPELINE (combines all steps above)
# ==================================================================
def clean_text_pipeline(text: str) -> str:
    """
    Run the complete text-cleaning pipeline on a single string.

    Order of operations:
        lowercase -> remove urls -> remove punctuation
        -> remove numbers -> remove stopwords -> trim extra spaces
    """
    if not isinstance(text, str):
        text = str(text)

    text = to_lowercase(text)
    text = remove_urls(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_stopwords(text)

    # Collapse multiple spaces into one and strip leading/trailing space
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ==================================================================
# DATAFRAME HELPER: Add a 'clean_text' column
# ==================================================================
def add_clean_text_column(df: pd.DataFrame, source_col: str, new_col: str = "clean_text") -> pd.DataFrame:
    """
    Apply clean_text_pipeline() to every row of `source_col` and
    store the cleaned result in a new column (default name: 'clean_text').
    """
    df = df.copy()
    df[new_col] = df[source_col].apply(clean_text_pipeline)
    return df


# ==================================================================
# Quick manual test
# ==================================================================
if __name__ == "__main__":
    sample = "Check this out!! http://example.com  This is AMAZING 123 :)"
    print("Original:", sample)
    print("Cleaned :", clean_text_pipeline(sample))
