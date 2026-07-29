"""
train_bert.py
-------------
Fine-tunes bert-base-uncased on the MCQ dataset.
5-class classification: predict which option (A-E) is correct.

Training details:
    - AdamW optimizer with linear warmup + linear decay
    - Gradient clipping at 1.0
    - Early stopping on val MAP@3
    - float32 forced — avoids MPS precision issues
    - Logs train loss, val loss, MAP@3, Accuracy, Macro F1 to W&B

Location : src/bert/train_bert.py
Run      : python3 -m src.bert.train_bert
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
from src.bert.bert_dataset import BERTMCQDataset, IDX_TO_LABEL, OPTION_COLS
from src.bert.bert_model import BERTMCQClassifier

load_dotenv()
set_seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
CFG = {
    "model_name"       : "bert-base-uncased",
    "max_length"       : 256,
    "batch_size"       : 8,
    "grad_accum_steps" : 4,       # effective batch = 32
    "epochs"           : 5,
    "lr"               : 2e-5,
    "weight_decay"     : 0.01,
    "warmup_ratio"     : 0.1,
    "dropout"          : 0.1,
    "patience"         : 2,
}

DEVICE   = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SAVE_DIR = "models/bert"


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(all_logits: np.ndarray, labels: list) -> tuple:
    top1_preds = [IDX_TO_LABEL[int(np.argmax(l))] for l in all_logits]
    top3_preds = [
        [IDX_TO_LABEL[i] for i in np.argsort(l)[::-1][:3]]
        for l in all_logits
    ]
    accuracy = accuracy_score(labels, top1_preds)
    f1       = f1_score(
        labels, top1_preds,
        labels     = OPTION_COLS,
        average    = "macro",
        zero_division = 0,
    )
    map3 = map_at_3(top3_preds, labels)
    return {
        "top1_accuracy": round(accuracy, 4),
        "macro_f1"     : round(f1, 4),
        "map3"         : round(map3, 4),
    }, top3_preds


# ── Train one epoch ───────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, device, grad_accum):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(loader, desc="Train", leave=False)):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask, token_type_ids)
        loss   = F.cross_entropy(logits, labels) / grad_accum
        loss.backward()

        if (step + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum

    # handle leftover steps
    if (step + 1) % grad_accum != 0:
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return total_loss / len(loader)


# ── Eval one epoch ────────────────────────────────────────────────────────────
def eval_epoch(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval ", leave=False):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            logits = model(input_ids, attention_mask, token_type_ids)
            all_logits.extend(logits.cpu().numpy())

            if "labels" in batch:
                all_labels.extend([IDX_TO_LABEL[l.item()] for l in batch["labels"]])

    return np.array(all_logits), all_labels


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    config = load_config()
    print(f"Device : {DEVICE}")
    print(f"Model  : {CFG['model_name']}")

    # ── Data ─────────────────────────────────────────────────────────────────
    df = pd.read_csv("data/raw/train.csv")
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["answer"]
    )
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CFG["model_name"])

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_ds = BERTMCQDataset(train_df, tokenizer, CFG["max_length"])
    val_ds   = BERTMCQDataset(val_df,   tokenizer, CFG["max_length"])

    train_loader = DataLoader(
        train_ds, batch_size=CFG["batch_size"], shuffle=True,  num_workers=0
    )
    val_loader = DataLoader(
        val_ds,   batch_size=CFG["batch_size"], shuffle=False, num_workers=0
    )

    # ── Model — force float32 to avoid MPS precision issues ──────────────────
    print("Loading BERT...")
    model = BERTMCQClassifier(
        model_name = CFG["model_name"],
        num_labels = 5,
        dropout    = CFG["dropout"],
    ).to(DEVICE).float()

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params    : {total_params:,}")
    print(f"Trainable params: {train_params:,}")

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    optimizer    = AdamW(
        model.parameters(),
        lr           = CFG["lr"],
        weight_decay = CFG["weight_decay"],
    )
    total_steps  = (len(train_loader) // CFG["grad_accum_steps"]) * CFG["epochs"]
    warmup_steps = int(total_steps * CFG["warmup_ratio"])
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps,
    )

    # ── W&B ──────────────────────────────────────────────────────────────────
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    wandb.init(
        entity  = config["project"].get("wandb_entity"),
        project = config["project"]["wandb_project"],
        name    = "bert-base-uncased-finetune",
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

    # ── Training loop ─────────────────────────────────────────────────────────
    best_map3    = 0.0
    patience_ctr = 0
    os.makedirs(SAVE_DIR, exist_ok=True)

    for epoch in range(1, CFG["epochs"] + 1):
        train_loss             = train_epoch(
            model, train_loader, optimizer, scheduler,
            DEVICE, CFG["grad_accum_steps"]
        )
        val_logits, val_labels = eval_epoch(model, val_loader, DEVICE)
        val_metrics, _         = compute_metrics(val_logits, val_labels)

        # val loss
        val_texts_flat, val_labels_idx = [], []
        for batch in DataLoader(val_ds, batch_size=CFG["batch_size"]):
            val_texts_flat.append(batch)
            val_labels_idx.extend(batch["labels"].tolist())

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
            model.bert.save_pretrained(SAVE_DIR)
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

    # ── Final eval with best weights ─────────────────────────────────────────
    print("\nLoading best checkpoint...")
    model.load_state_dict(
        torch.load(os.path.join(SAVE_DIR, "classifier.pt"), map_location=DEVICE)
    )
    val_logits, val_labels = eval_epoch(model, val_loader, DEVICE)
    final_metrics, _       = compute_metrics(val_logits, val_labels)

    print("\n── Final Val Metrics ────────────────────────")
    for k, v in final_metrics.items():
        print(f"  {k:20s}: {v}")

    # ── Test submission ───────────────────────────────────────────────────────
    print("\nGenerating test submission...")
    test_df     = pd.read_csv("data/raw/test.csv")
    test_ds     = BERTMCQDataset(test_df, tokenizer, CFG["max_length"], is_test=True)
    test_loader = DataLoader(
        test_ds, batch_size=CFG["batch_size"], shuffle=False, num_workers=0
    )

    test_logits, _ = eval_epoch(model, test_loader, DEVICE)

    predictions = [
        " ".join([IDX_TO_LABEL[i] for i in np.argsort(l)[::-1][:3]])
        for l in test_logits
    ]
    submission = pd.DataFrame({
        "id"        : test_df["id"],
        "Prediction": predictions,
    })
    os.makedirs("outputs", exist_ok=True)
    submission.to_csv("outputs/submission_bert.csv", index=False)

    print(f"\nSubmission saved: outputs/submission_bert.csv")
    print(submission.head())
    print("\n── Summary ──────────────────────────────────")
    print(f"  Best val MAP@3   : {best_map3}")
    print(f"  Val Accuracy     : {final_metrics['top1_accuracy']}")
    print(f"  Val Macro F1     : {final_metrics['macro_f1']}")


if __name__ == "__main__":
    main()