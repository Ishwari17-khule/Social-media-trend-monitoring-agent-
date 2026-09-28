"""Prediction service that loads saved model artifacts once."""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import pandas as pd

from config import MODEL_PATH, VECTORIZER_PATH, setup_logging
from utils.preprocess import clean_text


logger = setup_logging()


@dataclass
class PredictionResult:
    """Single sentiment prediction result."""

    sentiment: str
    confidence: float


class SentimentPredictor:
    """Load saved model/vectorizer and predict new text sentiment."""

    def __init__(self, model_path=MODEL_PATH, vectorizer_path=VECTORIZER_PATH) -> None:
        if not model_path.exists() or not vectorizer_path.exists():
            raise FileNotFoundError(
                "Saved model artifacts are missing. Run `py -3 train_model.py` once before starting the app."
            )
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        logger.info("Loaded sentiment model and vectorizer from disk.")

    def predict(self, text: str) -> PredictionResult:
        """Predict sentiment and confidence for one text string."""
        cleaned = clean_text(text)
        features = self.vectorizer.transform([cleaned])
        sentiment = str(self.model.predict(features)[0])
        confidence = 0.0
        if hasattr(self.model, "predict_proba"):
            confidence = float(self.model.predict_proba(features).max())
        return PredictionResult(sentiment=sentiment, confidence=confidence)

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict sentiment for each article in a DataFrame."""
        if df.empty:
            return df
        scored = df.copy()
        text = scored["title"].fillna("").astype(str) + " " + scored["description"].fillna("").astype(str)
        scored["clean_text"] = text.map(clean_text)
        features = self.vectorizer.transform(scored["clean_text"])
        scored["sentiment"] = self.model.predict(features)
        if hasattr(self.model, "predict_proba"):
            scored["confidence"] = self.model.predict_proba(features).max(axis=1)
        else:
            scored["confidence"] = 0.0
        scored["trend_score"] = (scored["confidence"] * 100 + scored["clean_text"].str.split().str.len()).round(2)
        return scored


def predict_text(text: str) -> dict[str, float | str]:
    """Convenience function for simple one-off predictions."""
    result = SentimentPredictor().predict(text)
    return {"prediction": result.sentiment, "confidence": result.confidence}
