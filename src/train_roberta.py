"""
train_roberta.py
----------------
RoBERTa baseline — zero-shot inference, no fine-tuning.
Uses roberta-base as a sequence classifier with random head.
Evaluates on 80/20 train/val split.
Logs MAP@3, Top-1 Accuracy, Macro F1 to W&B.

Location : src/train_roberta.py
Run      : python3 -m src.train_roberta
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

from src.utils import map_at_3, load_config, set_seed

load_dotenv()
set_seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME   = "roberta-base"
MAX_LENGTH   = 256
BATCH_SIZE   = 16
OPTION_COLS  = ["A", "B", "C", "D", "E"]
LABEL_MAP    = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
DEVICE       = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ── Input formatter ───────────────────────────────────────────────────────────
def build_input(row) -> str:
    prompt = str(row["prompt"])
    opts   = " | ".join([f"{c}: {str(row[c])}" for c in OPTION_COLS])
    return prompt + " " + opts


# ── Batch inference ───────────────────────────────────────────────────────────
def run_inference(model, tokenizer, texts: list, batch_size: int = BATCH_SIZE):
    all_probs = []
    model.eval()
    for i in tqdm(range(0, len(texts), batch_size), desc="Inference"):
        batch = texts[i: i + batch_size]
        enc   = tokenizer(
            batch,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(DEVICE)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        all_probs.extend(probs)
    return np.array(all_probs)


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(probs: np.ndarray, labels: list):
    top1_preds  = [IDX_TO_LABEL[int(np.argmax(p))] for p in probs]
    top3_preds  = [
        [IDX_TO_LABEL[i] for i in np.argsort(p)[::-1][:3]]
        for p in probs
    ]

    accuracy = accuracy_score(labels, top1_preds)
    f1       = f1_score(
        labels, top1_preds,
        labels=OPTION_COLS,
        average="macro",
        zero_division=0
    )
    map3     = map_at_3(top3_preds, labels)

    return {
        "top1_accuracy" : round(accuracy, 4),
        "macro_f1"      : round(f1, 4),
        "map3"          : round(map3, 4),
    }, top3_preds


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    config = load_config()
    print(f"Device : {DEVICE}")
    print(f"Model  : {MODEL_NAME}")

    # ── Load data ─────────────────────────────────────────────────────────────
    df = pd.read_csv("data/raw/train.csv")
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["answer"]
    )
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading RoBERTa...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=5,
        ignore_mismatched_sizes=True
    ).to(DEVICE)
    model.eval()

    # ── W&B init ──────────────────────────────────────────────────────────────
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    wandb.init(
        entity  = config["project"].get("wandb_entity"),
        project = config["project"]["wandb_project"],
        name    = "roberta-base-zeroshot",
        config  = {
            "model"      : MODEL_NAME,
            "mode"       : "zero-shot",
            "max_length" : MAX_LENGTH,
            "batch_size" : BATCH_SIZE,
            "train_size" : len(train_df),
            "val_size"   : len(val_df),
        }
    )

    # ── Inference on val set ──────────────────────────────────────────────────
    print("\nRunning inference on validation set...")
    val_texts  = [build_input(row) for _, row in val_df.iterrows()]
    val_labels = val_df["answer"].tolist()
    val_probs  = run_inference(model, tokenizer, val_texts)

    val_metrics, val_top3 = compute_metrics(val_probs, val_labels)
    print("\n── Val Metrics ──────────────────────────────")
    for k, v in val_metrics.items():
        print(f"  {k:20s}: {v}")

    # ── Inference on train set (for comparison) ───────────────────────────────
    print("\nRunning inference on train set...")
    train_texts  = [build_input(row) for _, row in train_df.iterrows()]
    train_labels = train_df["answer"].tolist()
    train_probs  = run_inference(model, tokenizer, train_texts)

    train_metrics, _ = compute_metrics(train_probs, train_labels)
    print("\n── Train Metrics ────────────────────────────")
    for k, v in train_metrics.items():
        print(f"  {k:20s}: {v}")

    # ── Log to W&B ────────────────────────────────────────────────────────────
    wandb.log({
        "val/map3"          : val_metrics["map3"],
        "val/top1_accuracy" : val_metrics["top1_accuracy"],
        "val/macro_f1"      : val_metrics["macro_f1"],
        "train/map3"        : train_metrics["map3"],
        "train/top1_accuracy": train_metrics["top1_accuracy"],
        "train/macro_f1"    : train_metrics["macro_f1"],
    })
    wandb.summary["val_map3"]     = val_metrics["map3"]
    wandb.summary["val_macro_f1"] = val_metrics["macro_f1"]
    wandb.summary["model_type"]   = "zero-shot"
    wandb.finish()

    # ── Generate submission on test set ───────────────────────────────────────
    print("\nGenerating test submission...")
    test_df    = pd.read_csv("data/raw/test.csv")
    test_texts = [build_input(row) for _, row in test_df.iterrows()]
    test_probs = run_inference(model, tokenizer, test_texts)

    predictions = [
        " ".join([IDX_TO_LABEL[i] for i in np.argsort(p)[::-1][:3]])
        for p in test_probs
    ]
    submission = pd.DataFrame({
        "id"         : test_df["id"],
        "Prediction" : predictions
    })
    os.makedirs("outputs", exist_ok=True)
    submission.to_csv("outputs/submission_roberta_zeroshot.csv", index=False)

    print(f"\nSubmission saved: outputs/submission_roberta_zeroshot.csv")
    print(submission.head())
    print("\n── Final Summary ────────────────────────────")
    print(f"  val MAP@3     : {val_metrics['map3']}")
    print(f"  val Accuracy  : {val_metrics['top1_accuracy']}")
    print(f"  val Macro F1  : {val_metrics['macro_f1']}")


if __name__ == "__main__":
    main()