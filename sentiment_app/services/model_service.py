import json
import math
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from sentiment_app.nlp.preprocessing import normalize_text


TOPIC_KEYWORDS = {
    "Delivery": ("delivery", "shipping", "package", "parcel", "courier", "late", "arrived"),
    "Support": ("support", "service", "agent", "help", "response", "refund", "return"),
    "Product": ("product", "quality", "item", "device", "app", "feature", "price"),
    "Experience": ("experience", "easy", "difficult", "slow", "fast", "broken", "works"),
}


class SentimentAnalyzer:
    def __init__(
        self,
        model_path,
        vectorizer_path,
        metadata_path=None,
        positive_threshold=0.58,
        negative_threshold=0.42,
    ):
        self.model_path = Path(model_path)
        self.vectorizer_path = Path(vectorizer_path)
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.model = None
        self.vectorizer = None
        self.metadata = {}
        self.load()

    @property
    def is_ready(self):
        return self.model is not None and self.vectorizer is not None

    @property
    def model_name(self):
        return type(self.model).__name__ if self.model is not None else None

    @property
    def vectorizer_features(self):
        return len(getattr(self.vectorizer, "vocabulary_", {})) if self.vectorizer else 0

    def load(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Missing model file: {self.model_path}")
        if not self.vectorizer_path.exists():
            raise FileNotFoundError(f"Missing vectorizer file: {self.vectorizer_path}")

        with self.model_path.open("rb") as file:
            self.model = pickle.load(file)
        with self.vectorizer_path.open("rb") as file:
            self.vectorizer = pickle.load(file)

        if self.metadata_path and self.metadata_path.exists():
            self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def analyze(self, text):
        text = " ".join(str(text or "").split())
        if not text:
            raise ValueError("Text is required for sentiment analysis.")
        if len(text) > 5000:
            raise ValueError("Text is too long. Please keep it under 5000 characters.")

        started = time.perf_counter()
        matrix = self.vectorizer.transform([text])
        prediction = int(self.model.predict(matrix)[0])
        probabilities = self._probabilities(matrix)
        sentiment, confidence = self._sentiment_from_probabilities(prediction, probabilities)

        return {
            "id": uuid4().hex,
            "text": text,
            "normalized_text": normalize_text(text),
            "sentiment": sentiment,
            "label": prediction,
            "confidence": round(confidence * 100, 2),
            "probabilities": {
                key: round(value * 100, 2) for key, value in probabilities.items()
            },
            "emotion": self._emotion(sentiment),
            "intent": self._intent(sentiment),
            "topics": self._topics(text),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
        }

    def _probabilities(self, matrix):
        if hasattr(self.model, "predict_proba"):
            raw = self.model.predict_proba(matrix)[0]
            by_class = {int(label): float(raw[index]) for index, label in enumerate(self.model.classes_)}
            return {
                "negative": by_class.get(0, 0.0),
                "positive": by_class.get(1, 0.0),
            }

        if hasattr(self.model, "decision_function"):
            score = float(np.ravel(self.model.decision_function(matrix))[0])
            positive = 1 / (1 + math.exp(-score))
            return {"negative": 1 - positive, "positive": positive}

        return {"negative": 0.5, "positive": 0.5}

    def _sentiment_from_probabilities(self, prediction, probabilities):
        positive_probability = probabilities.get("positive", 0.5)
        if positive_probability >= self.positive_threshold:
            return "Positive", positive_probability
        if positive_probability <= self.negative_threshold:
            return "Negative", 1 - positive_probability
        return "Neutral", max(positive_probability, 1 - positive_probability)

    def _topics(self, text):
        normalized = normalize_text(text)
        topics = [
            topic
            for topic, keywords in TOPIC_KEYWORDS.items()
            if any(keyword in normalized for keyword in keywords)
        ]
        return topics or ["General"]

    def _emotion(self, sentiment):
        return {
            "Positive": "Satisfied",
            "Neutral": "Mixed",
            "Negative": "Frustrated",
        }[sentiment]

    def _intent(self, sentiment):
        return {
            "Positive": "Praise",
            "Neutral": "Feedback",
            "Negative": "Complaint",
        }[sentiment]
