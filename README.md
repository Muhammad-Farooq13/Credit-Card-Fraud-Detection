# Credit Card Fraud Detection

[![CI](https://github.com/Muhammad-Farooq13/Credit-Card-Fraud-Detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Muhammad-Farooq13/Credit-Card-Fraud-Detection/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A complete MLOps pipeline for detecting fraudulent credit card transactions using **Logistic Regression with balanced class weights**. Handles extreme class imbalance (~0.17 % fraud rate) using stratified splits, manual class weighting, and PR-AUC as the primary evaluation metric.

---

## Live Demo

Run the Streamlit app locally:

```bash
pip install -r requirements.txt
python train_demo.py          # generate synthetic data + train demo model
streamlit run streamlit_app.py
```

---

## Problem Statement

Credit card fraud detection is a classic imbalanced classification problem:

- **Dataset**: 284,807 transactions, only 492 (0.17 %) are fraudulent
- **Features**: `Time`, `Amount`, and `V1`–`V28` (PCA-anonymised)
- **Challenge**: Standard accuracy is >99 % even for a dummy classifier; PR-AUC is the meaningful metric

---

## Project Structure

```
credit-card-fraud-detection/
├── .github/workflows/ci.yml     # CI: Python 3.11/3.12 matrix, ruff, black, codecov
├── configs/
│   └── train_config.yaml        # Pipeline paths configuration
├── data/
│   ├── raw/                     # Raw input CSVs (git-ignored)
│   └── processed/               # Preprocessed .npz splits (git-ignored)
├── artifacts/
│   ├── fraud_demo.pkl           # Streamlit demo bundle (model + metrics + curves)
│   └── eval_report.json         # Evaluation JSON report
├── src/
│   ├── data/load.py             # CSV loader
│   ├── data/preprocess.py       # Stratified 70/10/20 split + StandardScaler
│   ├── features/build_features.py  # Feature engineering (Amount_per_Time)
│   ├── models/train.py          # LogisticRegression training
│   ├── models/evaluate.py       # ROC-AUC / PR-AUC evaluation
│   └── visualization/plots.py   # Matplotlib plotting utilities
├── tests/                       # 5 unit tests (all passing ✅)
├── flask_app.py                 # REST API (/health, /predict)
├── mlops_pipeline.py            # CLI orchestrator (preprocess|train|evaluate)
├── train_demo.py                # Standalone demo trainer → fraud_demo.pkl
├── streamlit_app.py             # Interactive 4-tab Streamlit demo
├── requirements.txt             # Full dependencies (includes Streamlit + Plotly)
└── requirements-ci.txt          # Lean CI dependencies
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full MLOps pipeline

```bash
# Preprocess raw data
python mlops_pipeline.py --stage preprocess \
    --input data/raw/creditcard.csv \
    --output data/processed

# Train model
python mlops_pipeline.py --stage train \
    --config configs/train_config.yaml

# Evaluate on test set
python mlops_pipeline.py --stage evaluate \
    --model artifacts/model.joblib \
    --report artifacts/eval_report.json
```

### 3. Run Streamlit demo (no real data needed)

```bash
python train_demo.py            # generates synthetic data + fraud_demo.pkl
streamlit run streamlit_app.py
```

### 4. Start Flask REST API

```bash
python flask_app.py
# POST /predict  {"features": [[time, amount, v1, ..., v28]]}
# GET  /health
```

---

## Model Architecture

| Component | Details |
|---|---|
| **Algorithm** | Logistic Regression (`sklearn`) |
| **Class imbalance** | Manual balanced class weights |
| **Preprocessing** | `StandardScaler` (fit on train only) |
| **Split** | Stratified 70 % train / 10 % val / 20 % test |
| **Primary metric** | PR-AUC (Precision-Recall AUC) |
| **Secondary metric** | ROC-AUC |

### Why Logistic Regression?

- V1–V28 are already decorrelated PCA components → linear model is effective
- Coefficients provide direct interpretability of feature importance
- Fast inference suitable for real-time fraud screening
- Baseline before ensemble/deep learning methods

---

## Handling Class Imbalance

```python
# Compute class weights from training distribution
classes, counts = np.unique(y_train, return_counts=True)
class_weight = {
    int(c): len(y_train) / (len(classes) * cnt)
    for c, cnt in zip(classes, counts)
}
# Result: fraud transactions are up-weighted ~580× vs. legitimate
```

Stratified splits ensure the fraud ratio is preserved in all three sets.

---

## CI/CD

GitHub Actions runs on every push and pull request:

- **Python matrix**: 3.11 and 3.12
- **Linting**: `ruff check .` + `black --check .`
- **Pipeline**: synthetic data → preprocess → train → pytest
- **Coverage**: uploaded to Codecov via `codecov/codecov-action@v5`

---

## Flask REST API

```bash
# Health check
curl http://localhost:5000/health

# Fraud prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [[0.0, 149.62, -1.36, -0.07, 2.54, 1.38, -0.34, 0.46, 0.24, 0.10, -0.17, -0.45, -0.24, -0.49, -0.44, -0.03, -0.17, 0.06, 0.08, -0.40, 0.25, -0.19, 0.44, 0.00, 0.01, -0.01, -0.37, -0.07, -0.06, 378.66]]}'
```

Response: `{"fraud_probability": [0.023]}`

---

## Streamlit App Tabs

1. **📊 Overview** — dataset statistics, class imbalance chart, amount distributions
2. **🔍 Transaction Analyzer** — real-time fraud scoring with probability gauge (V1–V28 + Amount inputs)
3. **📈 Model Performance** — ROC curve, PR curve, confusion matrix, feature coefficients
4. **⚙️ Pipeline Details** — preprocessing code, model architecture, design rationale

---

## Tests

```bash
pytest tests/ -v
# 5 passed ✅
```

| Test | Description |
|---|---|
| `test_preprocess_outputs` | Verifies all 4 output files are created |
| `test_preprocess_shapes_consistent` | Train + val + test shapes sum to total |
| `test_preprocess_labels_binary` | Labels are strictly 0 or 1 |
| `test_preprocess_no_data_leakage` | Indices don't overlap across splits |
| `test_train_creates_model` | Model joblib artifact is created |

---

## Dataset

The genuine dataset is the [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset (ULB Machine Learning Group).

The demo uses a synthetic approximation generated by `train_demo.py` — no Kaggle account required to run the Streamlit app.

---

## License

MIT
