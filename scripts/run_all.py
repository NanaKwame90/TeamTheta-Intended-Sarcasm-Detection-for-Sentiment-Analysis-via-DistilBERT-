#!/usr/bin/env python
"""Run all experiments sequentially: baseline → BERT → DistilBERT.

Usage
-----
    python scripts/run_all.py
    python scripts/run_all.py --epochs 5 --batch-size 16
"""

import argparse
import subprocess
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import BATCH_SIZE, DATA_DIR, EPOCHS, LEARNING_RATE, RANDOM_SEED

TRANSFORMER_MODELS = ["bert-base-uncased", "distilbert-base-uncased"]


def parse_args():
    p = argparse.ArgumentParser(description="Run all experiments end-to-end")
    p.add_argument("--data-dir", default=DATA_DIR)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print(f"\n{'='*60}")
    print("Running:", " ".join(cmd))
    print("=" * 60)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[WARNING] Command exited with code {result.returncode}")


def main():
    args = parse_args()
    python = sys.executable
    scripts_dir = os.path.dirname(os.path.abspath(__file__))

    # 1) Baseline
    run([python, os.path.join(scripts_dir, "run_baseline.py"),
         "--data-dir", args.data_dir,
         "--seed", str(args.seed)])

    # 2) Transformer models
    for model in TRANSFORMER_MODELS:
        run([python, os.path.join(scripts_dir, "run_transformer.py"),
             "--model", model,
             "--data-dir", args.data_dir,
             "--epochs", str(args.epochs),
             "--batch-size", str(args.batch_size),
             "--lr", str(args.lr),
             "--seed", str(args.seed)])

    print("\nAll experiments complete. Results in outputs/results.csv")


if __name__ == "__main__":
    main()
