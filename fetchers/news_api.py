"""Fetch live news from NewsAPI, GNews, and Twitter/X when keys are available."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests

from config import GNEWS_API_KEY, NEWSAPI_KEY, TWITTER_BEARER_TOKEN, setup_logging


logger = setup_logging()


class NewsAPIFetcher:
    """Client for NewsAPI.org."""

    endpoint = "https://newsapi.org/v2/everything"

    def fetch(self, query: str, page_size: int = 50) -> pd.DataFrame:
        """Fetch articles from NewsAPI."""
        if not NEWSAPI_KEY:
            logger.info("NEWSAPI_KEY not configured; skipping NewsAPI fetch.")
            return pd.DataFrame()
        try:
            response = requests.get(
                self.endpoint,
                params={
                    "q": query,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": page_size,
                    "apiKey": NEWSAPI_KEY,
                },
                timeout=15,
            )
            response.raise_for_status()
            articles = response.json().get("articles", [])
            return _normalize_newsapi_articles(articles, "NewsAPI")
        except Exception as exc:
            logger.warning("NewsAPI fetch failed: %s", exc)
            return pd.DataFrame()


class GNewsFetcher:
    """Client for GNews."""

    endpoint = "https://gnews.io/api/v4/search"

    def fetch(self, query: str, page_size: int = 50) -> pd.DataFrame:
        """Fetch articles from GNews."""
        if not GNEWS_API_KEY:
            logger.info("GNEWS_API_KEY not configured; skipping GNews fetch.")
            return pd.DataFrame()
        try:
            response = requests.get(
                self.endpoint,
                params={"q": query, "lang": "en", "max": min(page_size, 100), "apikey": GNEWS_API_KEY},
                timeout=15,
            )
            response.raise_for_status()
            articles = response.json().get("articles", [])
            return _normalize_gnews_articles(articles)
        except Exception as exc:
            logger.warning("GNews fetch failed: %s", exc)
            return pd.DataFrame()


class TwitterFetcher:
    """Client for Twitter/X recent search when a bearer token is available."""

    endpoint = "https://api.twitter.com/2/tweets/search/recent"

    def fetch(self, query: str, page_size: int = 25) -> pd.DataFrame:
        """Fetch recent public tweets from Twitter/X."""
        if not TWITTER_BEARER_TOKEN:
            logger.info("TWITTER_BEARER_TOKEN not configured; skipping Twitter/X fetch.")
            return pd.DataFrame()
        try:
            response = requests.get(
                self.endpoint,
                headers={"Authorization": f"Bearer {TWITTER_BEARER_TOKEN}"},
                params={
                    "query": f"({query}) lang:en -is:retweet",
                    "max_results": min(max(page_size, 10), 100),
                    "tweet.fields": "created_at,public_metrics",
                },
                timeout=15,
            )
            response.raise_for_status()
            rows = []
            for item in response.json().get("data", []):
                rows.append(
                    {
                        "title": item.get("text", ""),
                        "description": item.get("text", ""),
                        "source": "Twitter/X",
                        "url": f"https://x.com/i/web/status/{item.get('id')}",
                        "published_at": item.get("created_at"),
                        "category": "Social",
                    }
                )
            return pd.DataFrame(rows)
        except Exception as exc:
            logger.warning("Twitter/X fetch failed: %s", exc)
            return pd.DataFrame()


def _normalize_newsapi_articles(articles: list[dict[str, Any]], source_name: str) -> pd.DataFrame:
    rows = []
    for article in articles:
        source = article.get("source") or {}
        rows.append(
            {
                "title": article.get("title", ""),
                "description": article.get("description", "") or article.get("content", ""),
                "source": source.get("name") or source_name,
                "url": article.get("url", ""),
                "published_at": article.get("publishedAt"),
                "category": "News",
            }
        )
    return pd.DataFrame(rows)


def _normalize_gnews_articles(articles: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for article in articles:
        source = article.get("source") or {}
        rows.append(
            {
                "title": article.get("title", ""),
                "description": article.get("description", "") or article.get("content", ""),
                "source": source.get("name") or "GNews",
                "url": article.get("url", ""),
                "published_at": article.get("publishedAt"),
                "category": "News",
            }
        )
    return pd.DataFrame(rows)


def fallback_timestamp() -> str:
    """Return an ISO timestamp for fallback records."""
    return datetime.now(timezone.utc).isoformat()
