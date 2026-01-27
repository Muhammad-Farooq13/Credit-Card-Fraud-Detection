# Credit Card Fraud Detection

## Project Overview
This project builds and deploys machine learning models to detect fraudulent credit card transactions. It follows MLOps best practices for reproducibility, testing, and deployment. The repository is structured for GitHub publishing and supports both local Flask serving and containerized deployment via Docker.

## Dataset Overview
- Source: `creditcard.csv` (Kaggle/benchmark credit card fraud dataset).
- Features: 30 anonymized numerical features (V1-V28), `Time`, `Amount`.
- Target: `Class` (1 = fraud, 0 = legitimate).
- Preprocessing: train/validation/test split with stratification, scaling (`StandardScaler` / `RobustScaler`), optional class imbalance handling (undersampling/SMOTE), outlier clipping, and feature standardization persisted with the model artifact.

## Repository Structure
```
.
├── data
│   ├── raw/                # Raw datasets (unmodified)
│   └── processed/          # Cleaned/processed datasets & artifacts
├── notebooks/              # EDA and experiments (keep lightweight)
├── src
│   ├── data/               # Data loading & preprocessing scripts
│   ├── features/           # Feature engineering scripts
│   ├── models/             # Model training, evaluation, persistence
│   ├── visualization/      # Plotting and reporting utilities
│   └── utils/              # Shared helpers (logging, config, paths)
├── tests/                  # Unit tests (pytest)
├── Dockerfile              # Container image for inference API
├── flask_app.py            # Flask app exposing inference endpoint
├── mlops_pipeline.py       # CI/CD style orchestration hooks
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Getting Started
### Prerequisites
- Python 3.10+
- pip / venv (or conda)
- Docker (optional for containerized run)

### Setup
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Data
Place the raw dataset at `data/raw/creditcard.csv`. Do not commit raw data (ignored via `.gitignore`).

## Experimentation
- Start in `notebooks/` for EDA and quick prototyping.
- Migrate stable code into `src/` modules for reuse.
- Track experiments (metrics, params) in a lightweight CSV/JSON under `artifacts/` or integrate with MLflow/Weights & Biases if desired.

## Model Development
- Baseline: logistic regression / random forest / gradient boosting.
- Evaluation metrics: ROC-AUC (primary), PR-AUC, recall@k, precision, F1, confusion matrix.
- Hyperparameter tuning: grid/random search with cross-validation; scripts/notebooks should log configs and results.
- Model selection: compare validation metrics and select best-performing model; persist model + preprocessor with `joblib`.

## Running the Pipeline
```bash
python mlops_pipeline.py --stage preprocess --input data/raw/creditcard.csv --output data/processed/train.pkl
python mlops_pipeline.py --stage train --config configs/train_config.yaml
python mlops_pipeline.py --stage evaluate --model artifacts/model.joblib --report artifacts/report.json
```

## Flask Inference API
### Local run
```bash
set FLASK_APP=flask_app.py
flask run --port 5000
```
Endpoint: `POST /predict` with JSON body `{ "features": [/* feature array */] }`.

### Dockerized run
```bash
docker build -t fraud-api .
docker run -p 5000:5000 fraud-api
```

## MLOps Practices
- Version control: Git for code and configs; ignore data and large artifacts.
- Automated testing: `pytest` in CI for `src/` and `tests/`.
- Reproducibility: pin dependencies in `requirements.txt`; capture random seeds; store train/val/test splits.
- Monitoring (future): log inference requests/latency; track drift via summary stats; schedule periodic re-training.
- Experiment tracking: integrate MLflow or Weights & Biases by wrapping `train_model` with run logging (params, metrics, artifacts). Configure credentials via env vars in CI/CD secrets.
- Drift monitoring: schedule a job to log feature/target summary stats from production data and compare against training stats (KS/PSI or population stability metrics).

## CI Badge & Secrets
- Set GitHub Action secrets as needed: `WANDB_API_KEY` or `MLFLOW_TRACKING_URI`/`MLFLOW_TRACKING_USERNAME`/`MLFLOW_TRACKING_PASSWORD` plus `MLFLOW_EXPERIMENT_NAME` if you enable tracking in `train_model`.
- Keep `data/raw/*`, `data/processed/*`, and artifact files out of git (patterns already in `.gitignore`).

## Maintainer
- Name: Muhammad Farooq
- Email: mfarooqshafee333@gmail.com
- GitHub: https://github.com/Muhammad-Farooq13

## Testing
Run unit tests:
```bash
pytest
```

## Deployment Notes
- The Dockerfile builds a minimal image with the trained model and Flask app.
- `mlops_pipeline.py` provides hooks for CI/CD (preprocess → train → evaluate → package). Wire it into GitHub Actions for push/PR triggers.
