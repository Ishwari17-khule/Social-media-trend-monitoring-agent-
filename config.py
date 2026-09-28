"""Application configuration for the real-time trend dashboard."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"
ASSETS_DIR = BASE_DIR / "assets"

MODEL_PATH = MODEL_DIR / "sentiment_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"

NEWS_TRAINING_FILE = DATA_DIR / "news_data.csv"
TWITTER_TRAIN_FILE = DATA_DIR / "twitter_training.csv"
TWITTER_VAL_FILE = DATA_DIR / "twitter_validation.csv"

FETCH_INTERVAL_SECONDS = int(os.getenv("FETCH_INTERVAL_SECONDS", "60"))
DEFAULT_QUERY = os.getenv("DEFAULT_QUERY", "artificial intelligence OR technology OR startup")

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY", "")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "social-media-trend-agent/1.0")

RSS_FEEDS = [
    feed.strip()
    for feed in os.getenv(
        "RSS_FEEDS",
        "https://feeds.bbci.co.uk/news/technology/rss.xml,"
        "https://www.theverge.com/rss/index.xml,"
        "https://www.wired.com/feed/rss",
    ).split(",")
    if feed.strip()
]

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
    "database": os.getenv("POSTGRES_DB", "social_media_trend_db"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", ""),
}

# --- New: AI trend briefing (Gemini) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# --- New: Auth ---
AUTH_SECRET_KEY = os.getenv("AUTH_SECRET_KEY", "change-this-secret-key")

# --- New: FastAPI backend base URL (used by Streamlit if calling the API) ---
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# --- New: BERTopic ---
BERTOPIC_MIN_TOPIC_SIZE = int(os.getenv("BERTOPIC_MIN_TOPIC_SIZE", "3"))


def setup_logging() -> logging.Logger:
    """Configure application logging and return the shared logger."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("trend_agent")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    file_handler = logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger
