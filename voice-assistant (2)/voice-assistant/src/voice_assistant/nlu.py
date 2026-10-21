"""Intent classification.

A scikit-learn Pipeline combining word n-grams (what was said) and character
n-grams (robust to Whisper misspellings such as "wether" or "alarm's"), with a
confidence threshold so uncertain predictions trigger a clarification instead
of a wrong action.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from .data import normalize_text

MODEL_FORMAT_VERSION = 1


def _features() -> FeatureUnion:
    return FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
        ]
    )


def candidate_models(seed: int = 42) -> dict[str, Pipeline]:
    """Models compared during evaluation. All share the same feature extraction,
    except Naive Bayes, which is the classic word-count baseline."""
    return {
        "Logistic Regression": Pipeline(
            [
                ("features", _features()),
                ("clf", LogisticRegression(C=10, max_iter=3000, class_weight="balanced")),
            ]
        ),
        "Linear SVM": Pipeline(
            [("features", _features()), ("clf", LinearSVC(C=1.0, random_state=seed))]
        ),
        "Naive Bayes (baseline)": Pipeline(
            [("features", TfidfVectorizer(ngram_range=(1, 2))), ("clf", MultinomialNB(alpha=0.1))]
        ),
        "Random Forest": Pipeline(
            [
                ("features", _features()),
                ("clf", RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)),
            ]
        ),
        "MLP": Pipeline(
            [
                ("features", _features()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(128,),
                        early_stopping=True,
                        max_iter=300,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


DEFAULT_MODEL = "Logistic Regression"


@dataclass(frozen=True)
class Prediction:
    intent: str
    confidence: float
    alternatives: tuple[tuple[str, float], ...]
    threshold: float

    @property
    def is_confident(self) -> bool:
        return self.confidence >= self.threshold


class IntentClassifier:
    def __init__(self, pipeline: Pipeline | None = None, threshold: float = 0.35):
        self.pipeline = pipeline or candidate_models()[DEFAULT_MODEL]
        self.threshold = threshold
        self.metadata: dict = {}

    @property
    def labels(self) -> list[str]:
        return list(self.pipeline.classes_)

    def fit(self, texts: Iterable[str], intents: Iterable[str]) -> IntentClassifier:
        texts = [normalize_text(t) for t in texts]
        intents = list(intents)
        self.pipeline.fit(texts, intents)
        self.metadata = {
            "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
            "n_examples": len(texts),
            "sklearn_version": sklearn.__version__,
        }
        return self

    def predict(self, text: str, top_k: int = 3) -> Prediction:
        probs = self.pipeline.predict_proba([normalize_text(text)])[0]
        order = np.argsort(probs)[::-1][:top_k]
        ranked = tuple((self.pipeline.classes_[i], float(probs[i])) for i in order)
        return Prediction(
            intent=ranked[0][0],
            confidence=ranked[0][1],
            alternatives=ranked[1:],
            threshold=self.threshold,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "format_version": MODEL_FORMAT_VERSION,
                "pipeline": self.pipeline,
                "threshold": self.threshold,
                "metadata": self.metadata,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, threshold: float | None = None) -> IntentClassifier:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No trained model at {path}. Run `va train` first.")
        bundle = joblib.load(path)
        if bundle.get("format_version") != MODEL_FORMAT_VERSION:
            raise ValueError("Model file was created by an incompatible version; retrain it.")
        model = cls(bundle["pipeline"], threshold or bundle["threshold"])
        model.metadata = bundle.get("metadata", {})
        return model
