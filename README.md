# TeamTheta – SemEval 2022 Task 6 iSarcasmEval (English, Subtask A)

Binary sarcasm detection on English tweets.
Three models: TF-IDF + LinearSVC (baseline), BERT, DistilBERT.

---

## Project structure

```
teamtheta/
├── src/
│   ├── data/           # Dataset loading & preprocessing
│   ├── models/         # Model definitions (baseline + transformer wrapper)
│   ├── training/       # Training loops
│   ├── eval/           # Shared metrics & evaluation helpers
│   └── utils/          # Config defaults, seed, logger
├── scripts/
│   ├── run_baseline.py
│   ├── run_transformer.py
│   └── run_all.py
├── data/       
├── outputs/   
├── requirements.txt
└── .gitignore
```

---

## 1. Set up the environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Python ≥ 3.10 recommended.

---

## 2. Obtain and place the dataset

1. Download the iSarcasmEval dataset from the official repository:
   <https://github.com/iabufarha/iSarcasmEval>

2. Unzip so that the following files exist:

   ```
   data/raw/iSarcasmEval/train/train.En.csv
   data/raw/iSarcasmEval/task_A_En_test.csv   ← optional (gold labels)
   ```

   The train CSV must have at minimum the columns **`tweet`** and **`sarcastic`** (0 / 1).
   If the test file is absent the scripts still run; test evaluation is simply skipped.

> **Note:** The `data/` directory is gitignored. Never commit raw data files.

---

## 3. Run the experiments

### Baseline (TF-IDF + LinearSVC)

```bash
python scripts/run_baseline.py
# optionally override dataset path
python scripts/run_baseline.py --data-dir data/raw/iSarcasmEval
```

### Transformer models

```bash
# BERT
python scripts/run_transformer.py --model bert-base-uncased

# DistilBERT
python scripts/run_transformer.py --model distilbert-base-uncased

# Additional options
python scripts/run_transformer.py --model bert-base-uncased \
    --epochs 5 --batch-size 16 --lr 3e-5
```

### Run everything sequentially

```bash
python scripts/run_all.py
```

---

## 4. Outputs

All outputs are written to `outputs/`:

| File | Description |
|------|-------------|
| `outputs/results.csv` | Shared comparison table (all models, all splits) |
| `outputs/results_baseline.json` | Baseline metrics (val split) |
| `outputs/results_<model>.json` | Transformer metrics (val split) |
| `outputs/confusion_baseline.png` | Confusion matrix – baseline (val) |
| `outputs/confusion_<model>_<split>.png` | Confusion matrix – transformer |
| `outputs/models/<model>/` | Best checkpoint saved by Trainer |

---

## 5. Configuration defaults

All defaults live in [src/utils/config.py](src/utils/config.py):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_LENGTH` | 128 | Token limit for transformer models |
| `BATCH_SIZE` | 32 | Training batch size |
| `EPOCHS` | 3 | Fine-tuning epochs |
| `LEARNING_RATE` | 2e-5 | AdamW learning rate |
| `VAL_SPLIT` | 0.15 | Fraction of training data used for validation |
| `RANDOM_SEED` | 42 | Global random seed |

Override any default by passing CLI flags (see §3).

---

## 6. Reproducibility

- `src/utils/seed.py` sets seeds for Python, NumPy, and PyTorch.
- The HuggingFace `Trainer` receives the same seed via `TrainingArguments(seed=42)`.
- `torch.backends.cudnn.deterministic = True` is set when CUDA is available.

---

## Label mapping

| Label | Meaning |
|-------|---------|
| 0 | non-sarcastic |
| 1 | sarcastic |

Defined once in `src/data/loader.py` (`_LABEL_COL = "sarcastic"`).

---

## References

- Abufarhah et al., *iSarcasmEval: Intended Sarcasm Detection in English and Arabic*, SemEval 2022 Task 6.
  <https://aclanthology.org/2022.semeval-1.111>
- Dataset: <https://github.com/iabufarha/iSarcasmEval>
