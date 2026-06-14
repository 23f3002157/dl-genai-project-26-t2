"""
baseline_dummy.py
-----------------
Milestone 0 baseline. Predicts A B C for every question.
Logs a W&B run so you have your first tracked experiment.

Usage:
    python src/baseline_dummy.py --test_path data/raw/test.csv
"""

import os
import argparse
import random
import numpy as np
import pandas as pd
import wandb
from dotenv import load_dotenv
from src.utils import load_config, set_seed, map_at_3

load_dotenv()  # loads WANDB_API_KEY from .env


OPTION_LABELS = ["A", "B", "C", "D", "E"]


def dummy_predict_top3(strategy: str = "fixed") -> list:
    """
    strategy:
      fixed  → always predict A B C (pure dummy)
      random → random shuffle of options (slightly more honest baseline)
    """
    if strategy == "fixed":
        return ["A", "B", "C"]
    elif strategy == "random":
        opts = OPTION_LABELS.copy()
        random.shuffle(opts)
        return opts[:3]


def run(args):
    config = load_config(args.config)
    set_seed(config["training"]["seed"])

    # ── W&B Init ────────────────────────────────────────────
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    wandb.init(
        project=config["project"]["wandb_project"],
        name="dummy-baseline-run0",
        config={
            "model": "dummy-baseline",
            "strategy": args.strategy,
            "milestone": 0,
            "description": "Predict A B C for every question. No ML."
        }
    )

    # ── Load Test Data ───────────────────────────────────────
    test_df = pd.read_csv(args.test_path)
    print(f"Test samples: {len(test_df)}")
    id_col = "id" if "id" in test_df.columns else "ID"

    # ── Predict ──────────────────────────────────────────────
    all_preds = []
    pred_strings = []
    for _ in range(len(test_df)):
        top3 = dummy_predict_top3(args.strategy)
        all_preds.append(top3)
        pred_strings.append(" ".join(top3))

    # ── Compute MAP@3 on train set if available ──────────────
    # This gives you a real number to log, even if it's terrible
    train_map3 = None
    if os.path.exists(args.train_path):
        train_df = pd.read_csv(args.train_path)
        answer_col = "answer" if "answer" in train_df.columns else "Answer"
        train_preds = [dummy_predict_top3(args.strategy) for _ in range(len(train_df))]
        train_labels = train_df[answer_col].tolist()
        train_map3 = map_at_3(train_preds, train_labels)
        print(f"Dummy MAP@3 on train set: {train_map3:.4f}")

    # ── Log to W&B ───────────────────────────────────────────
    log_dict = {
        "milestone": 0,
        "strategy": args.strategy,
        "num_test_samples": len(test_df),
    }
    if train_map3 is not None:
        log_dict["train_map3"] = train_map3

    wandb.log(log_dict)
    wandb.summary["model_type"] = "dummy-baseline"
    wandb.summary["strategy"] = args.strategy
    if train_map3 is not None:
        wandb.summary["train_map3"] = train_map3
    wandb.finish()

    # ── Save Submission ──────────────────────────────────────
    submission = pd.DataFrame({
        "ID": test_df[id_col],
        "Prediction": pred_strings
    })
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"✅ Submission saved: {args.output}")
    print(submission.head(10))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path",  type=str, default="data/raw/test.csv")
    parser.add_argument("--train_path", type=str, default="data/raw/train.csv")
    parser.add_argument("--output",     type=str, default="submission.csv")
    parser.add_argument("--config",     type=str, default="configs/config.yaml")
    parser.add_argument("--strategy",   type=str, default="fixed",
                        choices=["fixed", "random"],
                        help="fixed=always ABC, random=shuffled options")
    args = parser.parse_args()
    run(args)