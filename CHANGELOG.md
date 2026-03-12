# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-07-18

### Added
- `streamlit_app.py` — interactive 4-tab Streamlit demo (Overview, Transaction Analyzer, Model Performance, Pipeline Details)
- `train_demo.py` — standalone script that generates synthetic credit card fraud data and trains a demo model bundle (`artifacts/fraud_demo.pkl`) without requiring real Kaggle data
- `artifacts/fraud_demo.pkl` — pre-trained demo bundle containing model, scaler, metrics, ROC/PR curves, feature importances, and sample data for the Streamlit app
- `requirements-ci.txt` — lean dependency file for CI (excludes Streamlit/Plotly/Flask)
- `.streamlit/config.toml` — dark red theme (urgency signalling for fraud detection context)
- `runtime.txt` and `packages.txt` — Streamlit Cloud deployment configuration

### Changed
- `requirements.txt` — added `streamlit>=1.36.0` and `plotly>=5.18.0`
- `.github/workflows/ci.yml` — upgraded CI: Python matrix 3.11/3.12, added `ruff` lint, `black` format check, `codecov/codecov-action@v5` coverage upload, switched to `requirements-ci.txt`, added V1-V28 synthetic features to CI data preparation
- `.gitignore` — added `!artifacts/fraud_demo.pkl` exception to allow demo bundle to be versioned
- `README.md` — full rewrite: quickstart, MLOps pipeline, model architecture, class imbalance strategy, Flask API usage, test table, dataset attribution

### Technical Details
- **Model**: Logistic Regression with manually computed balanced class weights
- **Split**: Stratified 70 % train / 10 % val / 20 % test (SEED=42)
- **Preprocessing**: `StandardScaler` fit on train only (no data leakage)
- **Key metric**: PR-AUC (Precision-Recall) — more meaningful than ROC-AUC for severely imbalanced fraud data
- **Dataset**: ~0.17 % fraud rate (17 fraud per 10,000 transactions)
- **Tests**: 5/5 passing

## [0.1.0] - 2026-03-03

### Added
- Initial changelog for project tracking.
