import json
from collections import Counter
from pathlib import Path
from threading import Lock


class HistoryStore:
    def __init__(self, path, max_items=250):
        self.path = Path(path)
        self.max_items = max_items
        self.lock = Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("[]", encoding="utf-8")

    def _read(self):
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _write(self, items):
        trimmed = items[-self.max_items :]
        self.path.write_text(json.dumps(trimmed, indent=2), encoding="utf-8")

    def append(self, entry):
        with self.lock:
            items = self._read()
            items.append(entry)
            self._write(items)

    def extend(self, entries):
        if not entries:
            return
        with self.lock:
            items = self._read()
            items.extend(entries)
            self._write(items)

    def latest(self, limit=12):
        with self.lock:
            return list(reversed(self._read()[-limit:]))

    def count(self):
        with self.lock:
            return len(self._read())

    def clear(self):
        with self.lock:
            self._write([])

    def stats(self):
        with self.lock:
            items = self._read()

        total = len(items)
        counts = Counter(item.get("sentiment", "Unknown") for item in items)
        if total == 0:
            return {
                "total": 0,
                "positive": 0,
                "neutral": 0,
                "negative": 0,
                "positive_pct": 0,
                "neutral_pct": 0,
                "negative_pct": 0,
                "average_confidence": 0,
                "nps_score": 0,
            }

        promoters = counts.get("Positive", 0)
        detractors = counts.get("Negative", 0)
        avg_confidence = sum(float(item.get("confidence", 0)) for item in items) / total

        return {
            "total": total,
            "positive": counts.get("Positive", 0),
            "neutral": counts.get("Neutral", 0),
            "negative": counts.get("Negative", 0),
            "positive_pct": round(counts.get("Positive", 0) * 100 / total, 1),
            "neutral_pct": round(counts.get("Neutral", 0) * 100 / total, 1),
            "negative_pct": round(counts.get("Negative", 0) * 100 / total, 1),
            "average_confidence": round(avg_confidence, 1),
            "nps_score": round((promoters - detractors) * 100 / total, 1),
        }
