"""Coordinator for all live data sources."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from config import DEFAULT_QUERY, NEWS_TRAINING_FILE, setup_logging
from fetchers.news_api import GNewsFetcher, NewsAPIFetcher, TwitterFetcher
from fetchers.reddit import RedditFetcher
from fetchers.rss import RSSFetcher
from utils.preprocess import remove_duplicate_articles


logger = setup_logging()


class LiveDataService:
    """Fetch, combine, deduplicate, and normalize live article data."""

    def __init__(self) -> None:
        self.fetchers = [
            NewsAPIFetcher(),
            GNewsFetcher(),
            RedditFetcher(),
            RSSFetcher(),
            TwitterFetcher(),
        ]

    def fetch(self, query: str = DEFAULT_QUERY) -> pd.DataFrame:
        """Fetch live data from all available sources with graceful fallback."""
        frames: list[pd.DataFrame] = []
        for fetcher in self.fetchers:
            try:
                if isinstance(fetcher, RSSFetcher):
                    frame = fetcher.fetch()
                else:
                    frame = fetcher.fetch(query)
                if not frame.empty:
                    frames.append(frame)
            except Exception as exc:
                logger.warning("%s failed: %s", fetcher.__class__.__name__, exc)

        if not frames:
            logger.info("No live API data returned; using local news_data.csv fallback.")
            frames.append(self._load_local_fallback())

        combined = pd.concat(frames, ignore_index=True)
        combined = self._normalize(combined)
        combined = remove_duplicate_articles(combined)
        logger.info("Fetched %s unique records.", len(combined))
        return combined

    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        normalized = df.copy()
        for column in ["title", "description", "source", "url", "category"]:
            if column not in normalized.columns:
                normalized[column] = ""
        if "published_at" not in normalized.columns:
            normalized["published_at"] = datetime.now(timezone.utc).isoformat()
        normalized["title"] = normalized["title"].fillna("").astype(str).str.strip()
        normalized["description"] = normalized["description"].fillna("").astype(str).str.strip()
        normalized["source"] = normalized["source"].fillna("Unknown").astype(str)
        normalized["published_at"] = pd.to_datetime(normalized["published_at"], errors="coerce", utc=True)
        normalized["published_at"] = normalized["published_at"].fillna(pd.Timestamp.now(tz="UTC"))
        normalized = normalized[normalized["title"] != ""]
        return normalized.reset_index(drop=True)

    @staticmethod
    def _load_local_fallback() -> pd.DataFrame:
        fallback = pd.read_csv(NEWS_TRAINING_FILE)
        if "url" not in fallback.columns:
            fallback["url"] = ""
        if "category" not in fallback.columns:
            fallback["category"] = "Local CSV"
        return fallback
