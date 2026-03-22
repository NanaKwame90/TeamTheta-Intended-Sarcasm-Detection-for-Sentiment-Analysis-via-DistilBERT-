"""
Shared evaluation helpers.

All models use this module to compute and persist metrics so that numbers
are always comparable and stored in a consistent format.
"""

from __future__ import annotations

import csv
import json
import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_LABEL_NAMES = ["non-sarcastic", "sarcastic"]


def compute_metrics(y_true: List[int], y_pred: List[int]) -> dict:
    """Return accuracy, macro-F1, and per-class F1."""
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_sarcastic": round(f1_score(y_true, y_pred, pos_label=1, average="binary",
                                        zero_division=0), 4),
    }


def log_report(y_true: List[int], y_pred: List[int], split: str = "eval") -> None:
    report = classification_report(y_true, y_pred, target_names=_LABEL_NAMES,
                                   zero_division=0)
    logger.info("Classification report (%s):\n%s", split, report)


def save_metrics_json(metrics: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", path)


def save_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    path: str,
    title: Optional[str] = None,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=_LABEL_NAMES)

    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    if title:
        ax.set_title(title)
    fig.tight_layout()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", path)


def append_results_csv(model_name: str, split: str, metrics: dict, csv_path: str) -> None:
    """Append one row to the shared results CSV (creates header on first write)."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)

    fieldnames = ["model", "split", "accuracy", "f1_macro", "f1_sarcastic"]
    row = {
        "model": model_name,
        "split": split,
        **{k: metrics.get(k, "") for k in fieldnames[2:]},
    }

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    logger.info("Results appended to %s", csv_path)
