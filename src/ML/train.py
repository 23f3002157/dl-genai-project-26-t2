"""
train_tfidf_lr.py
-----------------
Training pipeline for TF-IDF + Logistic Regression MCQ model.
Uses processed data from data/processed/train_processed.csv.
Evaluates MAP@3, Top-1 Accuracy, Macro F1. Logs to W&B.
Saves model to models/tfidf_lr/

Location : src/base/train_tfidf_lr.py
Run      : python3 -m src.base.train_tfidf_lr
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import wandb
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils import map_at_3, load_config, set_seed
from src.ML.m import build_model

load_dotenv()
set_seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
CFG = {
    "model"              : "tfidf-lr",
    "tfidf_max_features" : 50000,
    "tfidf_ngram_range"  : (1, 2),
    "svd_components"     : 300,
    "lr_C"               : 1.0,
    "lr_max_iter"        : 1000,
}

OPTION_COLS  = ["A", "B", "C", "D", "E"]
LABEL_MAP    = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

TRAIN_PATH   = "data/raw/train.csv"
TEST_PATH    = "data/raw/test.csv"
SAVE_DIR     = "models/tfidf_lr"


# ── Input formatter ───────────────────────────────────────────────────────────
def build_input(row, option_col: str) -> str:
    """
    Use cleaned columns from preprocessed data.
    One string per (prompt, option) pair.
    """
    return str(row["prompt_clean"]) + " " + str(row[f"{option_col}_clean"])


# ── Flatten to one-vs-rest format ─────────────────────────────────────────────
def flatten_df(df: pd.DataFrame, is_test: bool = False):
    """
    Each row in train has 5 options → 5 training samples.
    Label = 1 if this option is correct, 0 otherwise.
    For inference we keep all 5 and rank by probability.
    """
    texts  = []
    labels = []
    row_ids = []

    for idx, row in df.iterrows():
        for col in OPTION_COLS:
            texts.append(build_input(row, col))
            if not is_test:
                labels.append(1 if row["answer"] == col else 0)
            row_ids.append((idx, col))

    return texts, labels, row_ids


# ── Predict top-3 per question ────────────────────────────────────────────────
def predict_top3(model, df: pd.DataFrame, is_test: bool = False):
    texts, _, row_ids = flatten_df(df, is_test=True)
    probs = model.predict_proba(texts)[:, 1]   # probability of class=1 (correct)

    results = []
    for i in range(len(df)):
        option_probs = {
            OPTION_COLS[j]: probs[i * 5 + j]
            for j in range(5)
        }
        ranked = sorted(option_probs, key=option_probs.get, reverse=True)
        results.append(ranked[:3])

    return results


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(top3_preds: list, labels: list):
    top1_preds = [p[0] for p in top3_preds]
    accuracy   = accuracy_score(labels, top1_preds)
    f1         = f1_score(
        labels, top1_preds,
        labels=OPTION_COLS,
        average="macro",
        zero_division=0,
    )
    map3 = map_at_3(top3_preds, labels)
    return {
        "top1_accuracy": round(accuracy, 4),
        "macro_f1"     : round(f1, 4),
        "map3"         : round(map3, 4),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    config = load_config()

    # ── Load processed data ───────────────────────────────────────────────────
    print("Loading processed data...")
    df = pd.read_csv(TRAIN_PATH)
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["answer"]
    )
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    # ── Flatten to binary classification format ───────────────────────────────
    print("\nBuilding training corpus...")
    train_texts, train_labels, _ = flatten_df(train_df)
    print(f"Training samples (5 per row): {len(train_texts)}")

    # ── Build & train model ───────────────────────────────────────────────────
    print("\nTraining TF-IDF + LR pipeline...")
    model = build_model(
        tfidf_max_features = CFG["tfidf_max_features"],
        tfidf_ngram_range  = CFG["tfidf_ngram_range"],
        svd_components     = CFG["svd_components"],
        lr_C               = CFG["lr_C"],
        lr_max_iter        = CFG["lr_max_iter"],
    )
    model.fit(train_texts, train_labels)
    print("Training complete.")

    # ── Evaluate on val set ───────────────────────────────────────────────────
    print("\nEvaluating on val set...")
    val_top3   = predict_top3(model, val_df)
    val_labels = val_df["answer"].tolist()
    val_metrics = compute_metrics(val_top3, val_labels)

    print("\n── Val Metrics ──────────────────────────────")
    for k, v in val_metrics.items():
        print(f"  {k:20s}: {v}")

    # ── Evaluate on train set ─────────────────────────────────────────────────
    train_top3     = predict_top3(model, train_df)
    train_labels_l = train_df["answer"].tolist()
    train_metrics  = compute_metrics(train_top3, train_labels_l)

    print("\n── Train Metrics ────────────────────────────")
    for k, v in train_metrics.items():
        print(f"  {k:20s}: {v}")

    # ── W&B ───────────────────────────────────────────────────────────────────
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    wandb.init(
        entity  = config["project"].get("wandb_entity"),
        project = config["project"]["wandb_project"],
        name    = "tfidf-lr-baseline",
        config  = {
            **CFG,
            "train_size"  : len(train_df),
            "val_size"    : len(val_df),
            "input_format": "prompt_clean + option_clean (preprocessed)",
        }
    )

    wandb.log({
        "val/map3"           : val_metrics["map3"],
        "val/top1_accuracy"  : val_metrics["top1_accuracy"],
        "val/macro_f1"       : val_metrics["macro_f1"],
        "train/map3"         : train_metrics["map3"],
        "train/top1_accuracy": train_metrics["top1_accuracy"],
        "train/macro_f1"     : train_metrics["macro_f1"],
    })
    wandb.summary["val_map3"]          = val_metrics["map3"]
    wandb.summary["val_macro_f1"]      = val_metrics["macro_f1"]
    wandb.summary["val_top1_accuracy"] = val_metrics["top1_accuracy"]
    wandb.summary["model_type"]        = "from-scratch-classical"
    wandb.finish()

    # ── Save model ────────────────────────────────────────────────────────────
    os.makedirs(SAVE_DIR, exist_ok=True)
    model_path = os.path.join(SAVE_DIR, "tfidf_lr_pipeline.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved: {model_path}")

    # ── Test submission ───────────────────────────────────────────────────────
    print("\nGenerating test submission...")
    test_df  = pd.read_csv(TEST_PATH)

    # Build clean columns for test set on the fly
    import re, string
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    def clean(text):
        text = str(text).lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text).strip()
        return " ".join([t for t in text.split() if t not in ENGLISH_STOP_WORDS])

    test_df["prompt_clean"] = test_df["prompt"].apply(clean)
    for col in OPTION_COLS:
        test_df[f"{col}_clean"] = test_df[col].apply(clean)

    test_top3 = predict_top3(model, test_df, is_test=True)
    predictions = [" ".join(p) for p in test_top3]

    submission = pd.DataFrame({
        "id"        : test_df["id"],
        "Prediction": predictions,
    })
    os.makedirs("outputs", exist_ok=True)
    submission.to_csv("outputs/submission_tfidf_lr.csv", index=False)

    print(f"\nSubmission saved: outputs/submission_tfidf_lr.csv")
    print(submission.head())
    print("\n── Summary ──────────────────────────────────")
    print(f"  val MAP@3    : {val_metrics['map3']}")
    print(f"  val Accuracy : {val_metrics['top1_accuracy']}")
    print(f"  val Macro F1 : {val_metrics['macro_f1']}")


if __name__ == "__main__":
    main()