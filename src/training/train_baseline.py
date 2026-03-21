"""Train and evaluate the TF-IDF + LinearSVC baseline."""

from __future__ import annotations

import os
from typing import List, Optional

from src.eval.metrics import (
    compute_metrics,
    log_report,
    save_metrics_json,
    save_confusion_matrix,
    append_results_csv,
)
from src.models.baseline import build_baseline_pipeline
from src.utils.config import OUTPUT_DIR, RESULTS_CSV
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

MODEL_NAME = "tfidf_linearsvc"


def run_baseline(
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    test_texts: Optional[List[str]] = None,
    test_labels: Optional[List[int]] = None,
) -> dict:
    """
    Fit the baseline on train, evaluate on val (and test if available).

    Saves:
        outputs/results_baseline.json
        outputs/confusion_baseline.png
        outputs/results.csv  (one row per split)
    """
    logger.info("Fitting TF-IDF + LinearSVC baseline …")
    pipeline = build_baseline_pipeline()
    pipeline.fit(train_texts, train_labels)

    results = {}

    for split_name, texts, labels in [
        ("val", val_texts, val_labels),
        ("test", test_texts, test_labels),
    ]:
        if texts is None or labels is None:
            continue

        preds = pipeline.predict(texts)
        metrics = compute_metrics(labels, preds)
        results[split_name] = metrics

        logger.info("[%s] %s", split_name.upper(), metrics)
        log_report(labels, preds, split=split_name)

        save_confusion_matrix(
            labels, preds,
            path=os.path.join(OUTPUT_DIR, f"confusion_baseline_{split_name}.png"),
            title=f"Baseline – {split_name}",
        )
        append_results_csv(MODEL_NAME, split_name, metrics, RESULTS_CSV)

    # Also save a backward-compatible single-file summary
    primary = results.get("val", results.get("test", {}))
    save_metrics_json(primary, os.path.join(OUTPUT_DIR, "results_baseline.json"))

    # Keep the old confusion_baseline.png pointing at val (or test if no val)
    _alias_confusion_matrix(results, val_texts, val_labels, pipeline)

    return results


def _alias_confusion_matrix(results, val_texts, val_labels, pipeline):
    """Write outputs/confusion_baseline.png from val predictions."""
    if val_texts and val_labels:
        preds = pipeline.predict(val_texts)
        save_confusion_matrix(
            val_labels, preds,
            path=os.path.join(OUTPUT_DIR, "confusion_baseline.png"),
            title="Baseline – val",
        )
