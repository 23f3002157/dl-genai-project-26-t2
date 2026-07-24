"""
train_roberta_race.py
---------------------
Inference-only script using RoBERTa-large already fine-tuned on RACE MCQ dataset.
No training needed. Runs on CPU/MPS.
Evaluates MAP@3, Top-1 Accuracy, Macro F1 on val split.
Logs to W&B. Generates submission.csv.

Location : src/train_roberta_race.py
Run      : python3 -m src.train_roberta_race
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import wandb
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMultipleChoice
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
from src.utils import map_at_3, load_config, set_seed

load_dotenv()
set_seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
CFG = {
    "model"      : "LIAMF-USP/roberta-large-finetuned-race",
    "max_length" : 128,
    "batch_size" : 8,
}

OPTION_COLS  = ["A", "B", "C", "D", "E"]
LABEL_MAP    = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
DEVICE       = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ── Dataset ───────────────────────────────────────────────────────────────────
class RACEMCQDataset(Dataset):
    """
    AutoModelForMultipleChoice expects inputs shaped (batch, num_choices, seq_len).
    Each choice is tokenized as: [CLS] prompt [SEP] option [SEP]
    """
    def __init__(self, df, tokenizer, max_length, is_test=False):
        self.df         = df.reset_index(drop=True)
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.is_test    = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        prompt = str(row["prompt"])

        input_ids_list      = []
        attention_mask_list = []

        for col in OPTION_COLS:
            option = str(row[col])
            enc    = self.tokenizer(
                prompt,
                option,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids_list.append(enc["input_ids"].squeeze(0))
            attention_mask_list.append(enc["attention_mask"].squeeze(0))

        sample = {
            "input_ids"      : torch.stack(input_ids_list),       # (5, max_len)
            "attention_mask" : torch.stack(attention_mask_list),   # (5, max_len)
        }
        if not self.is_test:
            sample["labels"] = torch.tensor(
                LABEL_MAP[row["answer"]], dtype=torch.long
            )
        return sample


# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(model, loader, device, is_test=False):
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids      = input_ids,
                attention_mask = attention_mask,
            )
            logits = outputs.logits.cpu().numpy()   # (B, 5)
            all_logits.extend(logits)

            if not is_test:
                labels = [IDX_TO_LABEL[l.item()] for l in batch["labels"]]
                all_labels.extend(labels)

    return np.array(all_logits), all_labels


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(all_logits, labels):
    top1_preds = [IDX_TO_LABEL[int(np.argmax(l))] for l in all_logits]
    top3_preds = [
        [IDX_TO_LABEL[i] for i in np.argsort(l)[::-1][:3]]
        for l in all_logits
    ]
    accuracy = accuracy_score(labels, top1_preds)
    f1       = f1_score(
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
    }, top3_preds


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    config = load_config()
    print(f"Device : {DEVICE}")
    print(f"Model  : {CFG['model']}")

    # ── Data ─────────────────────────────────────────────────────────────────
    df = pd.read_csv("data/raw/train.csv")
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["answer"]
    )
    val_df = val_df.reset_index(drop=True)
    print(f"Val set: {len(val_df)} rows")

    # ── Model ────────────────────────────────────────────────────────────────
    print("Loading RoBERTa-large RACE...")
    tokenizer = AutoTokenizer.from_pretrained(CFG["model"])
    model     = AutoModelForMultipleChoice.from_pretrained(CFG["model"]).to(DEVICE)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")

    # ── Val inference ────────────────────────────────────────────────────────
    val_ds     = RACEMCQDataset(val_df, tokenizer, CFG["max_length"])
    val_loader = DataLoader(
        val_ds, batch_size=CFG["batch_size"], shuffle=False, num_workers=0
    )

    print("\nRunning inference on val set...")
    val_logits, val_labels = run_inference(model, val_loader, DEVICE, is_test=False)
    val_metrics, _         = compute_metrics(val_logits, val_labels)

    print("\n── Val Metrics ──────────────────────────────")
    for k, v in val_metrics.items():
        print(f"  {k:20s}: {v}")

    # ── W&B ──────────────────────────────────────────────────────────────────
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    wandb.init(
        entity  = config["project"].get("wandb_entity"),
        project = config["project"]["wandb_project"],
        name    = "roberta-large-race-pretrained",
        config  = {
            **CFG,
            "mode"         : "inference-only",
            "val_size"     : len(val_df),
            "total_params" : total_params,
            "input_format" : "[CLS] prompt [SEP] option [SEP]",
            "pretrained_on": "RACE MCQ dataset",
        }
    )

    wandb.log({
        "val/map3"          : val_metrics["map3"],
        "val/top1_accuracy" : val_metrics["top1_accuracy"],
        "val/macro_f1"      : val_metrics["macro_f1"],
    })
    wandb.summary["val_map3"]          = val_metrics["map3"]
    wandb.summary["val_macro_f1"]      = val_metrics["macro_f1"]
    wandb.summary["val_top1_accuracy"] = val_metrics["top1_accuracy"]
    wandb.summary["model_type"]        = "pretrained-race"
    wandb.finish()

    # ── Test submission ───────────────────────────────────────────────────────
    print("\nGenerating test submission...")
    test_df     = pd.read_csv("data/raw/test.csv")
    test_ds     = RACEMCQDataset(test_df, tokenizer, CFG["max_length"], is_test=True)
    test_loader = DataLoader(
        test_ds, batch_size=CFG["batch_size"], shuffle=False, num_workers=0
    )

    test_logits, _ = run_inference(model, test_loader, DEVICE, is_test=True)

    predictions = [
        " ".join([IDX_TO_LABEL[i] for i in np.argsort(l)[::-1][:3]])
        for l in test_logits
    ]
    submission = pd.DataFrame({
        "id"         : test_df["id"],
        "Prediction" : predictions,
    })
    os.makedirs("outputs", exist_ok=True)
    submission.to_csv("outputs/submission_roberta_race.csv", index=False)

    print(f"\nSubmission saved: outputs/submission_roberta_race.csv")
    print(submission.head())
    print("\n── Final Summary ────────────────────────────")
    print(f"  val MAP@3    : {val_metrics['map3']}")
    print(f"  val Accuracy : {val_metrics['top1_accuracy']}")
    print(f"  val Macro F1 : {val_metrics['macro_f1']}")


if __name__ == "__main__":
    main()