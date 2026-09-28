"""
trend_detection.py
--------------------------------------------------------------------
PURPOSE:
    Detects trending topics/keywords from the News dataset by analyzing
    word frequency in news titles.

    PIPELINE:
        1. Clean each news title (lowercase, remove punctuation/urls/numbers)
        2. Remove stopwords
        3. Tokenize the cleaned title into individual words
        4. Count word frequencies across ALL titles using Counter
        5. Return the top N most frequent keywords (the "trends")
--------------------------------------------------------------------
"""

from collections import Counter
from itertools import islice

import pandas as pd

from src.preprocessing import clean_text_pipeline


# ==================================================================
# MAIN FUNCTION: Extract top trending keywords
# ==================================================================
def extract_trending_topics(news_df: pd.DataFrame, title_col: str = "title", top_n: int = 15):
    """
    Extract the top N trending keywords from news titles.

    Parameters
    ----------
    news_df   : pd.DataFrame containing a column with news titles
    title_col : name of the column containing the news titles
    top_n     : number of top keywords to return

    Returns
    -------
    list of tuples -> [(keyword, frequency), (keyword, frequency), ...]
    """
    all_words = []

    for title in news_df[title_col].astype(str):
        # Clean the title: lowercase, remove urls/punctuation/numbers/stopwords
        cleaned = clean_text_pipeline(title)

        # Tokenize into words
        words = cleaned.split()

        # Ignore very short words (length <= 2) since they add noise (e.g. "to", "in")
        words = [w for w in words if len(w) > 2]

        all_words.extend(words)

    # Count frequency of every word using Counter
    word_counts = Counter(all_words)

    # Get the top N most common keywords
    top_keywords = word_counts.most_common(top_n)

    return top_keywords


# ==================================================================
# HELPER: Convert trending topics into a DataFrame
# (useful for displaying tables / bar charts in Streamlit)
# ==================================================================
def trending_topics_to_dataframe(news_df: pd.DataFrame, title_col: str = "title", top_n: int = 15) -> pd.DataFrame:
    """
    Same as extract_trending_topics() but returns the result as a
    pandas DataFrame with columns: ['keyword', 'frequency'].
    """
    top_keywords = extract_trending_topics(news_df, title_col, top_n)
    df = pd.DataFrame(top_keywords, columns=["keyword", "frequency"])
    return df


def extract_trending_phrases(news_df: pd.DataFrame, title_col: str = "title", top_n: int = 10, ngram_size: int = 2) -> pd.DataFrame:
    """Return the most common short phrases from cleaned news titles."""
    phrase_counts = Counter()

    for title in news_df[title_col].astype(str):
        words = [w for w in clean_text_pipeline(title).split() if len(w) > 2]
        ngrams = zip(*(islice(words, i, None) for i in range(ngram_size)))
        phrase_counts.update(" ".join(ngram) for ngram in ngrams)

    return pd.DataFrame(phrase_counts.most_common(top_n), columns=["phrase", "frequency"])


def build_trend_insights(news_df: pd.DataFrame, title_col: str = "title", top_n: int = 15) -> dict:
    """Build dashboard-ready summaries for trend analysis."""
    trends_df = trending_topics_to_dataframe(news_df, title_col=title_col, top_n=top_n)
    phrases_df = extract_trending_phrases(news_df, title_col=title_col, top_n=min(top_n, 12))

    source_df = pd.DataFrame(columns=["source", "articles", "share"])
    if "source" in news_df.columns and not news_df.empty:
        source_df = (
            news_df["source"]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
            .rename_axis("source")
            .reset_index(name="articles")
        )
        source_df["share"] = source_df["articles"] / len(news_df)

    timeline_df = pd.DataFrame(columns=["date", "articles"])
    if "published_at" in news_df.columns:
        dated = news_df.copy()
        dated["published_at"] = pd.to_datetime(dated["published_at"], errors="coerce", utc=True)
        dated = dated.dropna(subset=["published_at"])
        if not dated.empty:
            timeline_df = (
                dated.assign(date=dated["published_at"].dt.date)
                .groupby("date", as_index=False)
                .size()
                .rename(columns={"size": "articles"})
                .sort_values("date")
            )

    examples = {}
    for keyword in trends_df["keyword"].head(8):
        mask = news_df[title_col].astype(str).str.contains(keyword, case=False, na=False)
        cols = [c for c in ["title", "source", "published_at"] if c in news_df.columns]
        examples[keyword] = news_df.loc[mask, cols].head(5)

    return {
        "keywords": trends_df,
        "phrases": phrases_df,
        "sources": source_df,
        "timeline": timeline_df,
        "examples": examples,
    }


# ==================================================================
# Quick manual test
# ==================================================================
if __name__ == "__main__":
    from src.data_collection import load_news_data

    news_data = load_news_data()
    trends = trending_topics_to_dataframe(news_data, top_n=10)
    print(trends)
