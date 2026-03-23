"""
Thin wrapper around a Hugging Face sequence-classification model.

Supports any model whose tokenizer and AutoModelForSequenceClassification
are available on the Hub – in practice: bert-base-uncased, distilbert-base-uncased.
"""

from __future__ import annotations

from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_model_and_tokenizer(
    model_name: str,
    num_labels: int = 2,
):
    """
    Load a pre-trained tokenizer and sequence-classification head.

    Parameters
    ----------
    model_name : HuggingFace model ID, e.g. "bert-base-uncased"
    num_labels : number of output classes (2 for binary sarcasm detection)

    Returns
    -------
    tokenizer, model
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    return tokenizer, model
