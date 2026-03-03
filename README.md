# Credit Card Fraud Detection

End-to-end fraud detection project focused on imbalanced classification and deployable ML inference.

## Project Snapshot
- **Problem**: Detect fraudulent transactions with high recall while controlling false positives.
- **Data**: Benchmark credit card transaction dataset (`creditcard.csv`-style schema).
- **Target**: `Class` (1 = fraud, 0 = legitimate).

## Business Questions
- How accurately can fraud be detected in highly imbalanced data?
- Which features contribute most to fraud risk?
- Can the model be deployed as a lightweight API for real-time scoring?

## Workflow
1. Ingest and validate transaction data
2. Handle imbalance and scale features
3. Train candidate models
4. Evaluate with fraud-focused metrics
5. Serve predictions through Flask/Docker

## Evaluation Focus
- ROC-AUC / PR-AUC
- Precision, Recall, F1
- Confusion matrix and threshold tuning

## Repository Structure
```
Credit-Card-Fraud-Detection/
├── src/                  # Data/features/models utilities
├── tests/                # Unit tests
├── notebooks/            # EDA and experiments
├── flask_app.py          # Prediction API
├── mlops_pipeline.py     # Pipeline steps
├── Dockerfile
└── requirements.txt
```

## Quick Start
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python mlops_pipeline.py
python flask_app.py
```

## Testing
```bash
pytest -q
```

## Why This Project Is Portfolio-Ready
- Demonstrates strong understanding of imbalanced ML problems.
- Shows model-to-production workflow with API and containerization.
- Uses reproducible project organization expected in industry teams.

## Author
- Muhammad Farooq
- GitHub: https://github.com/Muhammad-Farooq13
