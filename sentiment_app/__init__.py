from pathlib import Path

from flask import Flask

from config import Config
from sentiment_app.routes import bp
from sentiment_app.services.history_store import HistoryStore
from sentiment_app.services.model_service import SentimentAnalyzer


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def create_app(config_object=Config):
    app = Flask(
        __name__,
        template_folder=str(PROJECT_ROOT / "templates"),
        static_folder=str(PROJECT_ROOT / "static"),
    )
    app.config.from_object(config_object)

    analyzer = SentimentAnalyzer(
        model_path=app.config["MODEL_PATH"],
        vectorizer_path=app.config["VECTORIZER_PATH"],
        metadata_path=app.config["METADATA_PATH"],
        positive_threshold=app.config["POSITIVE_THRESHOLD"],
        negative_threshold=app.config["NEGATIVE_THRESHOLD"],
    )
    history_store = HistoryStore(
        path=app.config["HISTORY_PATH"],
        max_items=app.config["MAX_HISTORY_ITEMS"],
    )

    app.extensions["sentiment_analyzer"] = analyzer
    app.extensions["history_store"] = history_store
    app.register_blueprint(bp)
    return app
