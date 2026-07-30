"""
train_deberta.py
----------------
Fine-tunes microsoft/deberta-v3-base on the MCQ dataset.
5-class classification — same format as BERT training.

Fixes applied vs previous broken run:
    - model.float() forces float32 — prevents MPS NaN loss
    - Single sequence format — no per-option batching
    - Standard cross-entropy — no binary scorer trick
    - Lower LR (1e-5) — DeBERTa is more sensitive than BERT

Location : src/deberta/train_deberta.py
Run      : python3 -m src.deberta.train_deberta
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils import map_at_3, load_config, set_seed
from src.deberta.deberta_dataset import DeBERTaMCQDataset, IDX_TO_LABEL, OPTION_COLS
from src.deberta.deberta_model import DeBERTaMCQClassifier

load_dotenv()
set_seed(42)

CFG = {
    "model_name"       : "microsoft/deberta-v3-base",
    "max_length"       : 256,
    "batch_size"       : 8,
    "grad_accum_steps" : 4,
    "epochs"           : 5,
    "lr"               : 1e-5,
    "weight_decay"     : 0.01,
    "warmup_ratio"     : 0.1,
    "dropout"          : 0.1,
    "patience"         : 2,
}

DEVICE   = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SAVE_DIR = "models/deberta"


def compute_metrics(all_logits, labels):
    top1_preds = [IDX_TO_LABEL[int(np.argmax(l))] for l in all_logits]
    top3_preds = [
        [IDX_TO_LABEL[i] for i in np.argsort(l)[::-1][:3]]
        for l in all_logits
    ]
    accuracy = accuracy_score(labels, top1_preds)
    f1       = f1_score(
        labels, top1_preds,
        labels=OPTION_COLS, average="macro", zero_division=0,
    )
    map3 = map_at_3(top3_preds, labels)
    return {
        "top1_accuracy": round(accuracy, 4),
        "macro_f1"     : round(f1, 4),
        "map3"         : round(map3, 4),
    }, top3_preds


def train_epoch(model, loader, optimizer, scheduler, device, grad_accum):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    for step, batch in enumerate(tqdm(loader, desc="Train", leave=False)):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)
        logits         = model(input_ids, attention_mask)
        loss           = F.cross_entropy(logits, labels) / grad_accum
        loss.backward()
        if (step + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        total_loss += loss.item() * grad_accum
    if (step + 1) % grad_accum != 0:
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return total_loss / len(loader)


def eval_epoch(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval ", leave=False):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits         = model(input_ids, attention_mask)
            all_logits.extend(logits.cpu().numpy())
            if "labels" in batch:
                all_labels.extend([IDX_TO_LABEL[l.item()] for l in batch["labels"]])
    return np.array(all_logits), all_labels


def main():
    config = load_config()
    print(f"Device : {DEVICE}")
    print(f"Model  : {CFG['model_name']}")

    df = pd.read_csv("data/raw/train.csv")
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["answer"]
    )
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CFG["model_name"])

    train_ds = DeBERTaMCQDataset(train_df, tokenizer, CFG["max_length"])
    val_ds   = DeBERTaMCQDataset(val_df,   tokenizer, CFG["max_length"])
    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"], shuffle=False, num_workers=0)

    print("Loading DeBERTa...")
    model = DeBERTaMCQClassifier(
        model_name=CFG["model_name"], num_labels=5, dropout=CFG["dropout"]
    ).to(DEVICE).float()

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params    : {total_params:,}")
    print(f"Trainable params: {train_params:,}")

    optimizer    = AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    total_steps  = (len(train_loader) // CFG["grad_accum_steps"]) * CFG["epochs"]
    warmup_steps = int(total_steps * CFG["warmup_ratio"])
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    wandb.init(
        entity  = config["project"].get("wandb_entity"),
        project = config["project"]["wandb_project"],
        name    = "deberta-v3-base-finetune",
        config  = {
            **CFG,
            "train_size"      : len(train_df),
            "val_size"        : len(val_df),
            "trainable_params": train_params,
            "effective_batch" : CFG["batch_size"] * CFG["grad_accum_steps"],
            "input_format"    : "Question: [prompt] A: ... B: ... C: ... D: ... E: ...",
            "precision"       : "float32",
        }
    )

    best_map3    = 0.0
    patience_ctr = 0
    os.makedirs(SAVE_DIR, exist_ok=True)

    for epoch in range(1, CFG["epochs"] + 1):
        train_loss             = train_epoch(
            model, train_loader, optimizer, scheduler, DEVICE, CFG["grad_accum_steps"]
        )
        val_logits, val_labels = eval_epoch(model, val_loader, DEVICE)
        val_metrics, _         = compute_metrics(val_logits, val_labels)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"map3={val_metrics['map3']} | "
            f"acc={val_metrics['top1_accuracy']} | "
            f"f1={val_metrics['macro_f1']}"
        )

        wandb.log({
            "epoch"             : epoch,
            "train/loss"        : train_loss,
            "val/map3"          : val_metrics["map3"],
            "val/top1_accuracy" : val_metrics["top1_accuracy"],
            "val/macro_f1"      : val_metrics["macro_f1"],
            "lr"                : optimizer.param_groups[0]["lr"],
        })

        if val_metrics["map3"] > best_map3:
            best_map3    = val_metrics["map3"]
            patience_ctr = 0
            model.deberta.save_pretrained(SAVE_DIR)
            tokenizer.save_pretrained(SAVE_DIR)
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "classifier.pt"))
            print(f"  ✅ Saved best model (map3={best_map3})")
        else:
            patience_ctr += 1
            if patience_ctr >= CFG["patience"]:
                print(f"  Early stopping at epoch {epoch}")
                break

    wandb.summary["best_val_map3"]     = best_map3
    wandb.summary["val_macro_f1"]      = val_metrics["macro_f1"]
    wandb.summary["val_top1_accuracy"] = val_metrics["top1_accuracy"]
    wandb.summary["model_type"]        = "pretrained-finetune"
    wandb.summary["trainable_params"]  = train_params
    wandb.finish()

    print("\nLoading best checkpoint...")
    model.load_state_dict(
        torch.load(os.path.join(SAVE_DIR, "classifier.pt"), map_location=DEVICE)
    )
    val_logits, val_labels = eval_epoch(model, val_loader, DEVICE)
    final_metrics, _       = compute_metrics(val_logits, val_labels)

    print("\n── Final Val Metrics ────────────────────────")
    for k, v in final_metrics.items():
        print(f"  {k:20s}: {v}")

    print("\nGenerating test submission...")
    test_df     = pd.read_csv("data/raw/test.csv")
    test_ds     = DeBERTaMCQDataset(test_df, tokenizer, CFG["max_length"], is_test=True)
    test_loader = DataLoader(test_ds, batch_size=CFG["batch_size"], shuffle=False, num_workers=0)

    test_logits, _ = eval_epoch(model, test_loader, DEVICE)

    predictions = [
        " ".join([IDX_TO_LABEL[i] for i in np.argsort(l)[::-1][:3]])
        for l in test_logits
    ]
    submission = pd.DataFrame({"id": test_df["id"], "Prediction": predictions})
    os.makedirs("outputs", exist_ok=True)
    submission.to_csv("outputs/submission_deberta.csv", index=False)

    print(f"\nSubmission saved: outputs/submission_deberta.csv")
    print(submission.head())
    print("\n── Summary ──────────────────────────────────")
    print(f"  Best val MAP@3   : {best_map3}")
    print(f"  Val Accuracy     : {final_metrics['top1_accuracy']}")
    print(f"  Val Macro F1     : {final_metrics['macro_f1']}")


if __name__ == "__main__":
    main()
