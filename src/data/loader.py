"""
Data loader for SemEval 2022 Task 6 iSarcasmEval – English Subtask A.

Expected layout inside data/raw/iSarcasmEval/:
    train/train.En.csv          – columns: tweet, sarcastic, rephrase, sarcasm, irony, …
    test/task_A_En_test.csv     – columns: text, sarcastic (labels already included)

Note: the train file uses "tweet" as the text column; the test file uses "text".
Both are handled automatically.

Labels: 0 = non-sarcastic, 1 = sarcastic  (already binary in the official dataset).
"""

from __future__ import annotations

import os
from typing import List, Tuple, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import DATA_DIR, TRAIN_FILE, TEST_FILE, VAL_SPLIT, RANDOM_SEED
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_LABEL_COL = "sarcastic"
# The train file calls the text column "tweet"; the test file calls it "text".
_TEXT_COL_TRAIN = "tweet"
_TEXT_COL_TEST = "text"


def _read_csv(path: str) -> pd.DataFrame:
    """Read a CSV, trying comma then tab separators."""
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception:
        df = pd.read_csv(path, sep="\t", encoding="utf-8")
    return df


def load_splits(
    data_dir: str = DATA_DIR,
    train_file: str = TRAIN_FILE,
    test_file: str = TEST_FILE,
    val_split: float = VAL_SPLIT,
    seed: int = RANDOM_SEED,
) -> Tuple[
    List[str], List[int],   # train
    List[str], List[int],   # val
    Optional[List[str]], Optional[List[int]],  # test (None if unavailable)
]:
    """
    Load iSarcasmEval English Subtask A data.

    Returns
    -------
    train_texts, train_labels,
    val_texts,   val_labels,
    test_texts,  test_labels   <- test_labels is None when labels are not available
    """
    train_path = os.path.join(data_dir, train_file)
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Training file not found: {train_path}\n"
            "Please place the iSarcasmEval dataset at data/raw/iSarcasmEval/ "
            "(see README.md for instructions)."
        )

    train_df = _read_csv(train_path)
    _validate_columns(train_df, train_path, _TEXT_COL_TRAIN)
    train_df = train_df.dropna(subset=[_TEXT_COL_TRAIN, _LABEL_COL])

    texts = train_df[_TEXT_COL_TRAIN].astype(str).tolist()
    labels = train_df[_LABEL_COL].astype(int).tolist()

    # Stratified train/val split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels,
        test_size=val_split,
        random_state=seed,
        stratify=labels,
    )

    logger.info(
        "Train: %d  |  Val: %d  (%.0f%% held out)",
        len(train_texts), len(val_texts), val_split * 100,
    )

    # ── Test split (optional) ──────────────────────────────────────────────
    test_texts: Optional[List[str]] = None
    test_labels: Optional[List[int]] = None

    test_path = os.path.join(data_dir, test_file)
    if os.path.exists(test_path):
        test_df = _read_csv(test_path)
        # Detect text column: test file uses "text", train file uses "tweet"
        text_col = _TEXT_COL_TEST if _TEXT_COL_TEST in test_df.columns else _TEXT_COL_TRAIN
        if text_col in test_df.columns:
            test_texts = test_df[text_col].astype(str).tolist()
            if _LABEL_COL in test_df.columns:
                test_labels = test_df[_LABEL_COL].astype(int).tolist()
                logger.info("Test: %d (with labels)", len(test_texts))
            else:
                logger.info("Test: %d (no labels – evaluation skipped)", len(test_texts))
        else:
            logger.warning("Test file found but missing text column – skipping.")
    else:
        logger.info("No test file at %s – test evaluation skipped.", test_path)

    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels


def _validate_columns(df: pd.DataFrame, path: str, text_col: str) -> None:
    for col in (text_col, _LABEL_COL):
        if col not in df.columns:
            raise ValueError(
                f"Expected column '{col}' in {path}. "
                f"Found: {list(df.columns)}"
            )
