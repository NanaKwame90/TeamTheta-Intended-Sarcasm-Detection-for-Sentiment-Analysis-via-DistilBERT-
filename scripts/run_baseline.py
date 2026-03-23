#!/usr/bin/env python
"""Entry point: train and evaluate the TF-IDF + LinearSVC baseline.

Usage
-----
    python scripts/run_baseline.py
    python scripts/run_baseline.py --data-dir data/raw/iSarcasmEval
"""

import argparse
import sys
import os

# Ensure project root is on the path when running as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import load_splits
from src.training.train_baseline import run_baseline
from src.utils.config import DATA_DIR, RANDOM_SEED
from src.utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Run TF-IDF + LinearSVC baseline")
    p.add_argument("--data-dir", default=DATA_DIR,
                   help="Path to iSarcasmEval dataset directory")
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    print(f"Loading data from: {args.data_dir}")
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = (
        load_splits(data_dir=args.data_dir, seed=args.seed)
    )

    results = run_baseline(
        train_texts, train_labels,
        val_texts, val_labels,
        test_texts, test_labels,
    )

    print("\n=== Baseline Results ===")
    for split, m in results.items():
        print(f"  {split:4s}  accuracy={m['accuracy']:.4f}  "
              f"f1_macro={m['f1_macro']:.4f}  f1_sarcastic={m['f1_sarcastic']:.4f}")


if __name__ == "__main__":
    main()
