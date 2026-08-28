# Efficient Intended Sarcasm Detection via DistilBERT & Hardware Acceleration

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1-EE4C2C.svg)](https://pytorch.org/)
[![Hugging Face Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow)](https://huggingface.co/docs/transformers/index)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end NLP pipeline for binary intended sarcasm detection on English tweets from **SemEval 2022 Task 6 (iSarcasmEval)**.

This repository evaluates the efficiency–performance trade-off between **BERT-base**, **DistilBERT**, and a **TF-IDF + LinearSVC** baseline, featuring cross-platform acceleration across **Apple Silicon (MPS)** and **NVIDIA CUDA (T4)** backends.

---

## 📊 Key Results

Under a severe **3:1 class imbalance** (2,600 non-sarcastic vs. 867 sarcastic instances), models are evaluated using **F1-Macro** and **F1-Sarcastic**:

| Model | Accuracy | F1-Macro | F1-Sarcastic | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **TF-IDF + LinearSVC** | **0.837** | 0.567 | 0.225 | Fast CPU baseline |
| **BERT-base (uncased)** | 0.811 | **0.615** | **0.340** | 110M params, 12 layers, d=768 |
| **DistilBERT (uncased)** | 0.784 | **0.600** | 0.328 | 66M params, retains **97.5%** of BERT F1 |

> **Takeaway:** DistilBERT retains **97.5%** of BERT-base macro F1 while reducing the parameter footprint by **40%**, making it suitable for resource-constrained deployment.

---

## ⚡ Hardware Acceleration & Infrastructure

- **Local Prototyping (Apple Silicon MPS):** Tokenization debugging and local runs via `torch.device("mps")` achieving **98.34 samples/sec** (~4.2 min/epoch).
- **Cloud Scaling (NVIDIA T4 CUDA):** High-throughput training on NVIDIA T4 GPUs reduced per-epoch time by ~40% (~1.7 min/epoch savings).
- **Data Engineering:** Custom **Apache Arrow** schema handling implemented to mitigate missing values and enforce UTF-8 integrity across tweet tokens.

---

## 📁 Repository Structure

```
teamtheta/
├── src/
│   ├── data/           # Dataset loading, custom schema fixes, & tokenization
│   ├── models/         # Model definitions (LinearSVC & Transformer wrappers)
│   ├── training/       # Training loops & HuggingFace Trainer configuration
│   ├── eval/           # Shared evaluation metrics (F1-Macro, confusion matrices)
│   └── utils/          # Config defaults, seed management, and loggers
├── scripts/
│   ├── run_baseline.py    # Run TF-IDF + LinearSVC baseline
│   ├── run_transformer.py # Run BERT / DistilBERT fine-tuning
│   └── run_all.py         # End-to-end evaluation pipeline
├── data/               # Gitignored (raw dataset storage)
├── outputs/            # Gitignored (generated figures & model checkpoints)
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Environment Setup

```bash
git clone https://github.com/NanaKwame90/TeamTheta.git
cd TeamTheta

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Dataset Setup

Download the [SemEval 2022 Task 6 dataset](https://github.com/iabufarha/iSarcasmEval) and place the training split at:

```
data/raw/iSarcasmEval/train/train.En.csv
```

### 3. Run Experiments

```bash
# Baseline (TF-IDF + LinearSVC)
python scripts/run_baseline.py

# Fine-tune DistilBERT (MPS / CUDA auto-detected)
python scripts/run_transformer.py --model distilbert-base-uncased --epochs 3 --batch-size 16 --lr 2e-5

# Fine-tune BERT-base
python scripts/run_transformer.py --model bert-base-uncased --epochs 3 --batch-size 16 --lr 2e-5

# Full evaluation pipeline
python scripts/run_all.py
```

---

## 🔬 Reproducibility

Global reproducibility is enforced in `src/utils/seed.py` by setting `RANDOM_SEED = 42` across Python, NumPy, PyTorch, and `TrainingArguments(seed=42)`. On CUDA backends, `torch.backends.cudnn.deterministic = True` is enabled.

---

## 4. Configuration Defaults

All defaults live in [`src/utils/config.py`](src/utils/config.py):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_LENGTH` | 128 | Token limit for transformer models |
| `BATCH_SIZE` | 32 | Training batch size |
| `EPOCHS` | 3 | Fine-tuning epochs |
| `LEARNING_RATE` | 2e-5 | AdamW learning rate |
| `VAL_SPLIT` | 0.15 | Fraction of training data used for validation |
| `RANDOM_SEED` | 42 | Global random seed |

---

## 📚 References

- Abu Farha et al. (2022). *SemEval-2022 Task 6: iSarcasmEval, Intended Sarcasm Detection in English and Arabic*. [ACL Anthology](https://aclanthology.org/2022.semeval-1.111)
- Sanh et al. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*. [arXiv:1910.01108](https://arxiv.org/abs/1910.01108)
- Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. [NAACL-HLT 2019](https://aclanthology.org/N19-1423/)
