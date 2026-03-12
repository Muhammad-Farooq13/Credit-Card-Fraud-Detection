"""
train_demo.py — Standalone script that generates synthetic credit card fraud data,
trains a LogisticRegression model, and bundles everything into artifacts/fraud_demo.pkl
for use by the Streamlit demo app (no GPU / real data required).
"""
from __future__ import annotations

import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.preprocessing import StandardScaler

SEED = 42
N_TOTAL = 12000
FRAUD_RATE = 0.0017          # ~0.17 % — matches Kaggle dataset
N_FRAUD = max(30, int(N_TOTAL * FRAUD_RATE))
N_LEGIT = N_TOTAL - N_FRAUD

np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Generate synthetic dataset
# ---------------------------------------------------------------------------
def _make_dataset() -> pd.DataFrame:
    n_features = 28

    # Legitimate transactions — centred at origin (V1-V28 PCA-like)
    legit_v = np.random.randn(N_LEGIT, n_features)
    legit_amount = np.random.lognormal(mean=3.0, sigma=1.5, size=N_LEGIT)
    legit_time = np.random.uniform(0, 172_792, N_LEGIT)

    # Fraudulent transactions — shifted features make classification tractable
    shift = np.random.choice([-3, 3], size=n_features)
    fraud_v = np.random.randn(N_FRAUD, n_features) * 1.2 + shift
    fraud_amount = np.random.lognormal(mean=4.5, sigma=0.8, size=N_FRAUD)
    fraud_time = np.random.uniform(0, 172_792, N_FRAUD)

    v_legit = {f"V{i+1}": legit_v[:, i] for i in range(n_features)}
    v_fraud = {f"V{i+1}": fraud_v[:, i] for i in range(n_features)}

    legit_df = pd.DataFrame(
        {"Time": legit_time, "Amount": legit_amount, **v_legit, "Class": 0}
    )
    fraud_df = pd.DataFrame(
        {"Time": fraud_time, "Amount": fraud_amount, **v_fraud, "Class": 1}
    )

    df = (
        pd.concat([legit_df, fraud_df])
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )
    return df


# ---------------------------------------------------------------------------
# Train + evaluate
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating synthetic dataset …")
    df = _make_dataset()
    print(f"  {len(df):,} rows  |  fraud: {df.Class.sum()} ({df.Class.mean() * 100:.2f} %)")

    feature_cols = [c for c in df.columns if c != "Class"]
    X = df[feature_cols].values
    y = df["Class"].values

    # Train / test split (80 / 20, stratified)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Class weights
    classes, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    class_weight = {int(c): total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}

    print("Training LogisticRegression …")
    model = LogisticRegression(
        class_weight=class_weight,
        max_iter=1000,
        random_state=SEED,
        solver="lbfgs",
    )
    model.fit(X_train_s, y_train)

    # Predictions
    y_prob = model.predict_proba(X_test_s)[:, 1]
    y_pred = model.predict(X_test_s)

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
    precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_test, y_prob)

    # Feature importances (LR coefficients)
    coef_abs = np.abs(model.coef_[0])
    top_idx = np.argsort(coef_abs)[::-1][:15]
    top_features = [(feature_cols[i], float(model.coef_[0][i])) for i in top_idx]

    print(f"  ROC-AUC : {roc_auc:.4f}")
    print(f"  PR-AUC  : {pr_auc:.4f}")

    # Bundle
    bundle = {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "metrics": {
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        },
        "curves": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "precision": precision_vals.tolist(),
            "recall": recall_vals.tolist(),
        },
        "top_features": top_features,
        "dataset_stats": {
            "n_total": int(len(df)),
            "n_fraud": int(df.Class.sum()),
            "n_legit": int((df.Class == 0).sum()),
            "fraud_rate": float(df.Class.mean()),
            "amount_mean": float(df.Amount.mean()),
            "amount_median": float(df.Amount.median()),
            "amount_std": float(df.Amount.std()),
            "fraud_amount_mean": float(df[df.Class == 1].Amount.mean()),
            "legit_amount_mean": float(df[df.Class == 0].Amount.mean()),
        },
        "sample_data": df.head(500).to_dict(orient="list"),
    }

    os.makedirs("artifacts", exist_ok=True)
    out_path = "artifacts/fraud_demo.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"\nBundle saved → {out_path}")
    print(json.dumps({"roc_auc": round(roc_auc, 4), "pr_auc": round(pr_auc, 4)}, indent=2))


if __name__ == "__main__":
    main()
