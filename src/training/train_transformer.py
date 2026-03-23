from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.eval.metrics import (
    append_results_csv,
    compute_metrics,
    log_report,
    save_confusion_matrix,
    save_metrics_json,
)
from src.models.transformer import load_model_and_tokenizer
from src.utils.config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MAX_LENGTH,
    OUTPUT_DIR,
    RESULTS_CSV,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _make_hf_dataset(
    texts: List[str],
    labels: Optional[List[int]],
    tokenizer,
    max_length: int,
) -> Dataset:
    data = {"text": texts}
    if labels is not None:
        data["label"] = labels
    ds = Dataset.from_dict(data)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    ds = ds.map(tokenize, batched=True, remove_columns=["text"])
    return ds


def _hf_compute_metrics(eval_pred):
    """Called by Trainer during evaluation."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return compute_metrics(labels.tolist(), preds.tolist())


def run_transformer(
    model_name: str,
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    test_texts: Optional[List[str]] = None,
    test_labels: Optional[List[int]] = None,
    max_length: int = MAX_LENGTH,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
) -> dict:
    """
    Fine-tune *model_name* and persist outputs.

    Saves:
        outputs/models/<model_slug>/          – best checkpoint
        outputs/results_<model_slug>.json
        outputs/confusion_<model_slug>_<split>.png
        outputs/results.csv                   – one row per split
    """
    model_slug = model_name.replace("/", "_")
    checkpoint_dir = os.path.join(OUTPUT_DIR, "models", model_slug)

    logger.info("Loading tokenizer & model: %s", model_name)
    tokenizer, model = load_model_and_tokenizer(model_name)

    train_ds = _make_hf_dataset(train_texts, train_labels, tokenizer, max_length)
    val_ds = _make_hf_dataset(val_texts, val_labels, tokenizer, max_length)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    steps_per_epoch = max(1, len(train_ds) // batch_size)
    warmup_steps = int(steps_per_epoch * epochs * WARMUP_RATIO)

    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        seed=42,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=_hf_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("Starting training for %d epoch(s) …", epochs)
    trainer.train()

    results = {}

    for split_name, texts, labels in [
        ("val", val_texts, val_labels),
        ("test", test_texts, test_labels),
    ]:
        if texts is None or labels is None:
            continue

        ds = _make_hf_dataset(texts, labels, tokenizer, max_length)
        output = trainer.predict(ds)
        preds = np.argmax(output.predictions, axis=-1).tolist()
        metrics = compute_metrics(labels, preds)
        results[split_name] = metrics

        logger.info("[%s] %s", split_name.upper(), metrics)
        log_report(labels, preds, split=split_name)

        save_confusion_matrix(
            labels, preds,
            path=os.path.join(OUTPUT_DIR, f"confusion_{model_slug}_{split_name}.png"),
            title=f"{model_slug} – {split_name}",
        )
        append_results_csv(model_slug, split_name, metrics, RESULTS_CSV)

    primary = results.get("val", results.get("test", {}))
    save_metrics_json(primary, os.path.join(OUTPUT_DIR, f"results_{model_slug}.json"))

    logger.info("Best checkpoint saved to %s", checkpoint_dir)
    return results
