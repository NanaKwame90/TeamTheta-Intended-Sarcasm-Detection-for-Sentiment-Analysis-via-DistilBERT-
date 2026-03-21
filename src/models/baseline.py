"""TF-IDF + LinearSVC baseline pipeline."""

from __future__ import annotations

from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.utils.config import TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, SVM_C


def build_baseline_pipeline(
    max_features: int = TFIDF_MAX_FEATURES,
    ngram_range: tuple = TFIDF_NGRAM_RANGE,
    C: float = SVM_C,
) -> Pipeline:
    """Return an sklearn Pipeline: TF-IDF → LinearSVC."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\w{1,}",
            stop_words=None,        # keep all tokens; task-specific stopwords hurt
        )),
        ("clf", LinearSVC(C=C, max_iter=2000)),
    ])
