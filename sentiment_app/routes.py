from flask import Blueprint, current_app, jsonify, render_template, request


bp = Blueprint("sentiment", __name__)


def _json_payload():
    if request.is_json:
        return request.get_json(silent=True) or {}
    return request.form.to_dict()


@bp.get("/")
def index():
    return render_template("index.html")


@bp.post("/api/predict")
def predict():
    payload = _json_payload()
    text = payload.get("text", "")

    try:
        result = current_app.extensions["sentiment_analyzer"].analyze(text)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    current_app.extensions["history_store"].append(result)
    return jsonify(result)


@bp.post("/api/batch")
def batch_predict():
    payload = _json_payload()
    texts = payload.get("texts", [])
    if not isinstance(texts, list):
        return jsonify({"error": "texts must be a list"}), 400

    analyzer = current_app.extensions["sentiment_analyzer"]
    results = []
    for text in texts[:25]:
        try:
            results.append(analyzer.analyze(text))
        except ValueError:
            continue

    current_app.extensions["history_store"].extend(results)
    return jsonify({"results": results, "count": len(results)})


@bp.get("/api/history")
def history():
    return jsonify(current_app.extensions["history_store"].latest())


@bp.get("/api/stats")
def stats():
    return jsonify(current_app.extensions["history_store"].stats())


@bp.post("/api/clear-history")
def clear_history():
    current_app.extensions["history_store"].clear()
    return jsonify({"status": "cleared"})


@bp.get("/health")
def health():
    analyzer = current_app.extensions["sentiment_analyzer"]
    history_store = current_app.extensions["history_store"]
    return jsonify(
        {
            "status": "ok",
            "model_loaded": analyzer.is_ready,
            "model": analyzer.model_name,
            "vectorizer_features": analyzer.vectorizer_features,
            "history_items": history_store.count(),
            "metadata": analyzer.metadata,
        }
    )
