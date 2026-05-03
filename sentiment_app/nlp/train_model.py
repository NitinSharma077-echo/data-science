import argparse
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

try:
    from sentiment_app.nlp.preprocessing import normalize_text
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    from sentiment_app.nlp.preprocessing import normalize_text


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT.parent / "training.1600000.processed.noemoticon.csv"


def load_sentiment140(path):
    df = pd.read_csv(
        path,
        encoding="latin-1",
        header=None,
        usecols=[0, 5],
        names=["target", "text"],
    )
    df["target"] = df["target"].replace({4: 1}).astype("int8")
    df["text"] = df["text"].fillna("").astype(str)
    return df


def balanced_sample(df, rows_per_class):
    if rows_per_class <= 0:
        return df.sample(frac=1, random_state=42).reset_index(drop=True)
    sampled = (
        df.groupby("target", group_keys=False)
        .sample(n=rows_per_class, random_state=42)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    return sampled


def build_vectorizer(max_features, min_df):
    return TfidfVectorizer(
        preprocessor=normalize_text,
        lowercase=False,
        token_pattern=r"(?u)\b[a-z][a-z0-9_']+\b",
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=0.95,
        max_features=max_features,
        sublinear_tf=True,
        strip_accents="unicode",
    )


def build_model(max_iter, c_value):
    return LogisticRegression(
        C=c_value,
        solver="saga",
        max_iter=max_iter,
        random_state=42,
    )


def train(args):
    started = time.perf_counter()
    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print(f"Loading dataset: {dataset_path}")
    df = load_sentiment140(dataset_path)
    df = balanced_sample(df, args.sample_per_class)
    print(f"Training rows: {len(df):,}")

    x_train, x_test, y_train, y_test = train_test_split(
        df["text"],
        df["target"],
        test_size=args.test_size,
        stratify=df["target"],
        random_state=42,
    )

    vectorizer = build_vectorizer(args.max_features, args.min_df)
    print("Vectorizing text...")
    train_matrix = vectorizer.fit_transform(x_train)
    test_matrix = vectorizer.transform(x_test)
    print(f"Vectorizer features: {train_matrix.shape[1]:,}")

    model = build_model(args.max_iter, args.c_value)
    print("Training classifier...")
    model.fit(train_matrix, y_train)

    predictions = model.predict(test_matrix)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)
    matrix = confusion_matrix(y_test, predictions).tolist()

    model_path = Path(args.model_out).resolve()
    vectorizer_path = Path(args.vectorizer_out).resolve()
    metadata_path = Path(args.metadata_out).resolve()

    with model_path.open("wb") as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)
    with vectorizer_path.open("wb") as file:
        pickle.dump(vectorizer, file, protocol=pickle.HIGHEST_PROTOCOL)

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "rows": int(len(df)),
        "test_size": args.test_size,
        "accuracy": round(float(accuracy), 6),
        "f1": round(float(f1), 6),
        "confusion_matrix": matrix,
        "classification_report": report,
        "model": {
            "type": "LogisticRegression",
            "solver": "saga",
            "max_iter": args.max_iter,
            "C": args.c_value,
        },
        "vectorizer": {
            "type": "TfidfVectorizer",
            "max_features": args.max_features,
            "actual_features": int(train_matrix.shape[1]),
            "ngram_range": [1, 2],
            "min_df": args.min_df,
            "max_df": 0.95,
            "sublinear_tf": True,
            "preprocessor": "sentiment_app.nlp.preprocessing.normalize_text",
        },
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 score: {f1:.4f}")
    print(f"Saved model: {model_path}")
    print(f"Saved vectorizer: {vectorizer_path}")
    print(f"Saved metadata: {metadata_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Sentiment140 Flask app model.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--sample-per-class", type=int, default=0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-features", type=int, default=120000)
    parser.add_argument("--min-df", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=150)
    parser.add_argument("--c-value", type=float, default=2.0)
    parser.add_argument("--model-out", default=str(PROJECT_ROOT / "sentiment_model.pkl"))
    parser.add_argument("--vectorizer-out", default=str(PROJECT_ROOT / "vectorizer.pkl"))
    parser.add_argument("--metadata-out", default=str(PROJECT_ROOT / "model_metadata.json"))
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
