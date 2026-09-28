"""
FastAPI Backend (Section 4.1 System Architecture Diagram).

Exposes REST endpoints for auth, live data + sentiment, topic modeling
(BERTopic), and AI trend briefings (Gemini), backed by PostgreSQL.

Run with:  uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import DEFAULT_QUERY, setup_logging
from database.auth_db import AuthManager
from database.db import DatabaseManager
from fetchers.live_data import LiveDataService
from predict import SentimentPredictor
from src.ai_summary import generate_trend_briefing
from src.topic_modeling import extract_topics
from utils.preprocess import detect_trending_keywords

logger = setup_logging()

app = FastAPI(title="Social Media Trend Monitoring Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_predictor: Optional[SentimentPredictor] = None
_live_service: Optional[LiveDataService] = None
_database: Optional[DatabaseManager] = None
_auth: Optional[AuthManager] = None


def get_predictor() -> SentimentPredictor:
    global _predictor
    if _predictor is None:
        _predictor = SentimentPredictor()
    return _predictor


def get_live_service() -> LiveDataService:
    global _live_service
    if _live_service is None:
        _live_service = LiveDataService()
    return _live_service


def get_database() -> DatabaseManager:
    global _database
    if _database is None:
        _database = DatabaseManager()
    return _database


def get_auth() -> AuthManager:
    global _auth
    if _auth is None:
        _auth = AuthManager()
    return _auth


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str


class SummaryRequest(BaseModel):
    query: str = DEFAULT_QUERY
    top_n: int = 12


@app.get("/health")
def health() -> dict:
    """Basic liveness + PostgreSQL availability check."""
    return {"status": "ok", "database_available": get_database().is_available()}


@app.post("/auth/login")
def login(payload: LoginRequest) -> dict:
    """Use Case 1: User Login."""
    user = get_auth().verify_user(payload.username, payload.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    return {"authenticated": True, "user": user}


@app.post("/auth/register")
def register(payload: RegisterRequest) -> dict:
    """Create a new user account."""
    success, message = get_auth().register_user(payload.username, payload.password)
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {"success": True, "message": message}


@app.get("/trends")
def trends(query: str = DEFAULT_QUERY, top_n: int = 12) -> dict:
    """Fetch live data, score sentiment, and return trending keywords."""
    live_df = get_live_service().fetch(query)
    scored_df = get_predictor().predict_dataframe(live_df)
    keywords = detect_trending_keywords(scored_df["clean_text"], top_n=top_n)
    return {
        "article_count": len(scored_df),
        "sentiment_counts": scored_df["sentiment"].value_counts().to_dict(),
        "keywords": keywords.to_dict(orient="records"),
    }


@app.get("/topics")
def topics(query: str = DEFAULT_QUERY) -> dict:
    """BERTopic-based topic clustering (Section 6.5)."""
    live_df = get_live_service().fetch(query)
    scored_df = get_predictor().predict_dataframe(live_df)
    result = extract_topics(scored_df["clean_text"])
    return {
        "available": result.available,
        "coherence_score": result.coherence_score,
        "message": result.message,
        "topics": result.topics_df.to_dict(orient="records"),
    }


@app.post("/summary")
def summary(payload: SummaryRequest) -> dict:
    """AI Trend Briefing generation via Gemini (Section 6.6)."""
    live_df = get_live_service().fetch(payload.query)
    scored_df = get_predictor().predict_dataframe(live_df)
    keywords = detect_trending_keywords(scored_df["clean_text"], top_n=payload.top_n)
    topic_result = extract_topics(scored_df["clean_text"])
    briefing = generate_trend_briefing(
        keywords_df=keywords,
        sentiment_counts=scored_df["sentiment"].value_counts().to_dict(),
        topics_df=topic_result.topics_df,
        article_count=len(scored_df),
    )
    return {"briefing": briefing, "coherence_score": topic_result.coherence_score}


@app.get("/database/status")
def database_status() -> dict:
    return get_database().counts()
