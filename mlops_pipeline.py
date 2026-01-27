"""
Lightweight pipeline orchestrator for CI/CD. Stages:
- preprocess: load raw data, clean, split, and persist processed datasets/artifacts
- train: train model from processed data and persist artifacts
- evaluate: load model, evaluate on holdout, emit report
Extend or integrate with GitHub Actions for automation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from src.data.load import load_raw_csv
from src.data.preprocess import preprocess_dataset
from src.models.train import train_model
from src.models.evaluate import evaluate_model


def load_config(config_path: str) -> dict:
    """Load a config file in YAML or JSON format."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")
    if path.suffix.lower() in {".yml", ".yaml"}:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def stage_preprocess(input_path: str, output_dir: str):
    df = load_raw_csv(input_path)
    artifacts = preprocess_dataset(df, output_dir)
    print(f"Preprocessing complete. Artifacts: {artifacts}")


def stage_train(config_path: str):
    config = load_config(config_path)
    model_path = train_model(config)
    print(f"Training complete. Model saved to {model_path}")


def stage_evaluate(model_path: str, report_path: str):
    metrics = evaluate_model(model_path, report_path)
    print(f"Evaluation complete. Report saved to {report_path} \nMetrics: {metrics}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLOps pipeline orchestrator")
    parser.add_argument("--stage", required=True, choices=["preprocess", "train", "evaluate"], help="Pipeline stage to run")
    parser.add_argument("--input", help="Input file path for preprocess stage")
    parser.add_argument("--output", help="Output directory for preprocess stage")
    parser.add_argument("--config", dest="config_path", help="Config path for training stage")
    parser.add_argument("--model", dest="model_path", help="Model path for evaluation stage")
    parser.add_argument("--report", dest="report_path", help="Report output path for evaluation stage")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.stage == "preprocess":
        if not args.input or not args.output:
            raise ValueError("--input and --output are required for preprocess stage")
        stage_preprocess(args.input, args.output)
    elif args.stage == "train":
        if not args.config_path:
            raise ValueError("--config is required for train stage")
        stage_train(args.config_path)
    elif args.stage == "evaluate":
        if not args.model_path or not args.report_path:
            raise ValueError("--model and --report are required for evaluate stage")
        stage_evaluate(args.model_path, args.report_path)


if __name__ == "__main__":
    main()
