import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-sentiment-secret")
    MODEL_PATH = Path(os.environ.get("SENTIMENT_MODEL_PATH", BASE_DIR / "sentiment_model.pkl"))
    VECTORIZER_PATH = Path(os.environ.get("SENTIMENT_VECTORIZER_PATH", BASE_DIR / "vectorizer.pkl"))
    METADATA_PATH = Path(os.environ.get("SENTIMENT_METADATA_PATH", BASE_DIR / "model_metadata.json"))
    HISTORY_PATH = Path(os.environ.get("SENTIMENT_HISTORY_PATH", BASE_DIR / "analysis_history.json"))
    MAX_HISTORY_ITEMS = int(os.environ.get("SENTIMENT_MAX_HISTORY_ITEMS", "250"))
    POSITIVE_THRESHOLD = float(os.environ.get("SENTIMENT_POSITIVE_THRESHOLD", "0.58"))
    NEGATIVE_THRESHOLD = float(os.environ.get("SENTIMENT_NEGATIVE_THRESHOLD", "0.42"))
