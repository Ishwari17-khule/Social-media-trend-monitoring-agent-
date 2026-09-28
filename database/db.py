"""PostgreSQL persistence for real-time dashboard records."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

from config import DB_CONFIG, setup_logging


logger = setup_logging()


class DatabaseManager:
    """Manage PostgreSQL tables and article persistence."""

    def __init__(self, config: dict[str, str] | None = None) -> None:
        self.config = config or DB_CONFIG

    @contextmanager
    def connection(self) -> Iterator[psycopg2.extensions.connection]:
        """Yield a database connection and handle cleanup."""
        conn = psycopg2.connect(**self.config)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def is_available(self) -> bool:
        """Return True if PostgreSQL is reachable."""
        try:
            with self.connection():
                return True
        except Exception as exc:
            logger.warning("PostgreSQL unavailable: %s", exc)
            return False

    def create_tables(self) -> bool:
        """Create dashboard tables if they do not exist."""
        try:
            with self.connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        CREATE TABLE IF NOT EXISTS live_articles (
                            id SERIAL PRIMARY KEY,
                            title TEXT NOT NULL,
                            description TEXT,
                            source TEXT,
                            url TEXT UNIQUE,
                            published_at TIMESTAMPTZ,
                            category TEXT,
                            clean_text TEXT,
                            sentiment VARCHAR(50),
                            confidence DOUBLE PRECISION,
                            trend_score DOUBLE PRECISION,
                            inserted_at TIMESTAMPTZ DEFAULT NOW()
                        );
                        """
                    )
                    cursor.execute(
                        """
                        CREATE TABLE IF NOT EXISTS twitter_sentiment (
                            id SERIAL PRIMARY KEY,
                            text TEXT,
                            sentiment VARCHAR(50),
                            confidence DOUBLE PRECISION,
                            inserted_at TIMESTAMPTZ DEFAULT NOW()
                        );
                        """
                    )
            return True
        except Exception as exc:
            logger.warning("Could not create tables: %s", exc)
            return False

    def upsert_articles(self, df: pd.DataFrame) -> int:
        """Insert or update live article records by URL."""
        if df.empty:
            return 0
        if not self.create_tables():
            return 0

        rows = []
        for _, row in df.iterrows():
            url = row.get("url") or f"local://{abs(hash(row.get('title', '')))}"
            rows.append(
                (
                    row.get("title", ""),
                    row.get("description", ""),
                    row.get("source", "Unknown"),
                    url,
                    row.get("published_at"),
                    row.get("category", ""),
                    row.get("clean_text", ""),
                    row.get("sentiment", "Unknown"),
                    float(row.get("confidence", 0.0)),
                    float(row.get("trend_score", 0.0)),
                )
            )

        try:
            with self.connection() as conn:
                with conn.cursor() as cursor:
                    execute_values(
                        cursor,
                        """
                        INSERT INTO live_articles (
                            title, description, source, url, published_at, category,
                            clean_text, sentiment, confidence, trend_score
                        )
                        VALUES %s
                        ON CONFLICT (url) DO UPDATE SET
                            title = EXCLUDED.title,
                            description = EXCLUDED.description,
                            source = EXCLUDED.source,
                            published_at = EXCLUDED.published_at,
                            category = EXCLUDED.category,
                            clean_text = EXCLUDED.clean_text,
                            sentiment = EXCLUDED.sentiment,
                            confidence = EXCLUDED.confidence,
                            trend_score = EXCLUDED.trend_score;
                        """,
                        rows,
                    )
            logger.info("Upserted %s article rows.", len(rows))
            return len(rows)
        except Exception as exc:
            logger.warning("Could not upsert articles: %s", exc)
            return 0

    def fetch_articles(self, limit: int = 500) -> pd.DataFrame:
        """Fetch stored article records."""
        try:
            with self.connection() as conn:
                return pd.read_sql(
                    """
                    SELECT title, description, source, url, published_at, category,
                           clean_text, sentiment, confidence, trend_score, inserted_at
                    FROM live_articles
                    ORDER BY published_at DESC NULLS LAST
                    LIMIT %s;
                    """,
                    conn,
                    params=(limit,),
                )
        except Exception as exc:
            logger.warning("Could not fetch stored articles: %s", exc)
            return pd.DataFrame()

    def counts(self) -> dict[str, int]:
        """Return database table counts."""
        try:
            with self.connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT COUNT(*) FROM live_articles;")
                    live_count = cursor.fetchone()[0]
                    cursor.execute("SELECT COUNT(*) FROM twitter_sentiment;")
                    twitter_count = cursor.fetchone()[0]
            return {"live_articles": live_count, "twitter_sentiment": twitter_count}
        except Exception as exc:
            logger.warning("Could not read database counts: %s", exc)
            return {"live_articles": 0, "twitter_sentiment": 0}
