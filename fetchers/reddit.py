"""Fetch Reddit posts through PRAW when credentials are available."""

from __future__ import annotations

import pandas as pd

from config import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT, setup_logging


logger = setup_logging()


class RedditFetcher:
    """Reddit fetcher using PRAW."""

    def fetch(self, query: str, limit: int = 50) -> pd.DataFrame:
        """Fetch matching Reddit submissions."""
        if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
            logger.info("Reddit credentials not configured; skipping Reddit fetch.")
            return pd.DataFrame()
        try:
            import praw

            reddit = praw.Reddit(
                client_id=REDDIT_CLIENT_ID,
                client_secret=REDDIT_CLIENT_SECRET,
                user_agent=REDDIT_USER_AGENT,
            )
            rows = []
            for post in reddit.subreddit("all").search(query, sort="new", limit=limit):
                rows.append(
                    {
                        "title": post.title,
                        "description": getattr(post, "selftext", ""),
                        "source": f"Reddit/r/{post.subreddit.display_name}",
                        "url": f"https://reddit.com{post.permalink}",
                        "published_at": pd.to_datetime(post.created_utc, unit="s", utc=True).isoformat(),
                        "category": "Social",
                    }
                )
            return pd.DataFrame(rows)
        except Exception as exc:
            logger.warning("Reddit fetch failed: %s", exc)
            return pd.DataFrame()
