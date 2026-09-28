"""Fetch articles from RSS feeds."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from config import RSS_FEEDS, setup_logging


logger = setup_logging()


class RSSFetcher:
    """RSS fetcher with a feedparser implementation and XML fallback."""

    def fetch(self, limit_per_feed: int = 25) -> pd.DataFrame:
        """Fetch entries from configured RSS feeds."""
        try:
            import feedparser

            rows = []
            for feed_url in RSS_FEEDS:
                parsed = feedparser.parse(feed_url)
                source = parsed.feed.get("title", feed_url)
                for entry in parsed.entries[:limit_per_feed]:
                    rows.append(
                        {
                            "title": entry.get("title", ""),
                            "description": entry.get("summary", ""),
                            "source": source,
                            "url": entry.get("link", ""),
                            "published_at": entry.get("published", datetime.now(timezone.utc).isoformat()),
                            "category": "RSS",
                        }
                    )
            return pd.DataFrame(rows)
        except Exception as exc:
            logger.warning("RSS fetch failed: %s", exc)
            return pd.DataFrame()
