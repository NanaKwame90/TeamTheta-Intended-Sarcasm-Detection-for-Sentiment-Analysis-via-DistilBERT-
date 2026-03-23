#!/usr/bin/env python
"""Entry point: fine-tune a transformer model for sarcasm detection.

Usage
-----
    python scripts/run_transformer.py --model bert-base-uncased
    python scripts/run_transformer.py --model distilbert-base-uncased
    python scripts/run_transformer.py --model bert-base-uncased --epochs 5 --lr 3e-5
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import load_splits
from src.training.train_transformer import run_transformer
from src.utils.config import (
    BATCH_SIZE, DATA_DIR, EPOCHS, LEARNING_RATE, MAX_LENGTH, RANDOM_SEED
)
from src.utils.seed import set_seed

SUPPORTED_MODELS = ["bert-base-uncased", "distilbert-base-uncased"]


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune a transformer for sarcasm detection")
    p.add_argument(
        "--model", required=True,
        choices=SUPPORTED_MODELS,
        help="HuggingFace model name to fine-tune",
    )
    p.add_argument("--data-dir", default=DATA_DIR)
    p.add_argument("--max-length", type=int, default=MAX_LENGTH)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--lr", type=float, default=LEARNING_RATE, dest="learning_rate")
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    print(f"Model : {args.model}")
    print(f"Data  : {args.data_dir}")

    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = (
        load_splits(data_dir=args.data_dir, seed=args.seed)
    )

    results = run_transformer(
        model_name=args.model,
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        test_texts=test_texts,
        test_labels=test_labels,
        max_length=args.max_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )

    print(f"\n=== {args.model} Results ===")
    for split, m in results.items():
        print(f"  {split:4s}  accuracy={m['accuracy']:.4f}  "
              f"f1_macro={m['f1_macro']:.4f}  f1_sarcastic={m['f1_sarcastic']:.4f}")


if __name__ == "__main__":
    main()
