"""
ai_summary.py
--------------------------------------------------------------------
PURPOSE:
    LLM-based Trend Summary / Daily Briefing generation (Section 6.6
    of the SRS) using the Gemini API. Aggregates trending keywords,
    topic clusters, and sentiment results into a short, human-readable
    briefing. Fails gracefully (returns a templated fallback summary)
    if the API key is missing or the call errors out.
--------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd

from config import GEMINI_API_KEY, GEMINI_MODEL, setup_logging

logger = setup_logging()


def _fallback_summary(keywords_df: pd.DataFrame, sentiment_counts: dict, topics_df: pd.DataFrame) -> str:
    """Simple templated summary used when Gemini is unavailable."""
    top_kw = ", ".join(keywords_df["keyword"].head(5).tolist()) if not keywords_df.empty else "no strong keywords yet"
    total = sum(sentiment_counts.values()) or 1
    pos_pct = sentiment_counts.get("Positive", 0) / total * 100
    neg_pct = sentiment_counts.get("Negative", 0) / total * 100
    topic_line = ""
    if not topics_df.empty:
        topic_line = f" The strongest emerging topic cluster centers on: {topics_df.iloc[0]['Keywords']}."
    return (
        f"Trending discussion right now centers on {top_kw}. "
        f"Overall sentiment skews {'positive' if pos_pct > neg_pct else 'negative' if neg_pct > pos_pct else 'mixed'} "
        f"({pos_pct:.0f}% positive, {neg_pct:.0f}% negative).{topic_line}"
    )


def generate_trend_briefing(
    keywords_df: pd.DataFrame,
    sentiment_counts: dict,
    topics_df: pd.DataFrame | None = None,
    article_count: int = 0,
) -> str:
    """Generate a short AI trend briefing. Uses Gemini when configured, else a fallback template."""
    topics_df = topics_df if topics_df is not None else pd.DataFrame()

    if not GEMINI_API_KEY:
        return _fallback_summary(keywords_df, sentiment_counts, topics_df)

    try:
        import google.generativeai as genai

        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)

        top_keywords = ", ".join(keywords_df["keyword"].head(10).tolist()) if not keywords_df.empty else "N/A"
        topic_summary = (
            "; ".join(f"Topic {row.Topic}: {row.Keywords}" for row in topics_df.head(5).itertuples())
            if not topics_df.empty
            else "N/A"
        )
        sentiment_line = ", ".join(f"{label}: {count}" for label, count in sentiment_counts.items())

        prompt = (
            "You are an analyst writing a concise daily social media trend briefing.\n"
            f"Total articles analyzed: {article_count}\n"
            f"Top trending keywords: {top_keywords}\n"
            f"Topic clusters: {topic_summary}\n"
            f"Sentiment breakdown: {sentiment_line}\n\n"
            "Write a 4-6 sentence trend briefing in plain English covering: "
            "(1) what is trending, (2) how public sentiment looks, and "
            "(3) one actionable insight. No markdown headers, no bullet points."
        )

        response = model.generate_content(prompt)
        text = (response.text or "").strip()
        return text or _fallback_summary(keywords_df, sentiment_counts, topics_df)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Gemini summary generation failed, using fallback: %s", exc)
        return _fallback_summary(keywords_df, sentiment_counts, topics_df)
