"""
train_gru.py
------------
Full training pipeline for the from-scratch GRU MCQ model.
Builds vocabulary from raw train data (no preprocessing — raw text preserves
semantic content that GRU needs for sequential reasoning).
Trains with label smoothing + cosine LR schedule.
Evaluates MAP@3, Accuracy, Macro F1. Logs to W&B.

Location : src/gru/train_gru.py
Run      : python3 -m src.gru.train_gru
"""

import os
import sys
import json
import string
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from collections import Counter
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils import map_at_3, load_config, set_seed
from src.gru.gru_model import GRUMCQModel

load_dotenv()
set_seed(42)

# ── Hyperparameters ───────────────────────────────────────────────────────────
CFG = {
    "model"       : "gru-3",
    "embed_dim"   : 192,
    "hidden_dim"  : 384,
    "num_layers"  : 3,
    "num_heads"   : 4,
    "dropout"     : 0.35,
    "max_length"  : 160,
    "min_freq"    : 1,
    "batch_size"  : 32,
    "epochs"      : 15,
    "lr"          : 8e-4,
    "weight_decay": 1e-4,
    "patience"    : 4,
    "label_smooth": 0.1,
    "warmup_epochs": 2,
}

OPTION_COLS  = ["A", "B", "C", "D", "E"]
LABEL_MAP    = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
DEVICE       = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
PAD_IDX      = 0
UNK_IDX      = 1
SAVE_DIR     = "models/gru"


# ── Vocabulary ────────────────────────────────────────────────────────────────
class Vocabulary:
    def __init__(self, min_freq: int = 1):
        self.min_freq  = min_freq
        self.word2idx  = {"<PAD>": PAD_IDX, "<UNK>": UNK_IDX}
        self.idx2word  = {PAD_IDX: "<PAD>", UNK_IDX: "<UNK>"}

    def build(self, texts: list):
        counter = Counter()
        for text in texts:
            counter.update(self._tokenize(text))
        for word, freq in counter.items():
            if freq >= self.min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx]  = word
        print(f"Vocabulary size: {len(self.word2idx):,}")

    def _tokenize(self, text: str) -> list:
        # Keep raw text — no stopword removal for GRU
        text = str(text).lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text.split()

    def encode(self, text: str, max_length: int) -> list:
        tokens = self._tokenize(text)[:max_length]
        ids    = [self.word2idx.get(t, UNK_IDX) for t in tokens]
        ids   += [PAD_IDX] * (max_length - len(ids))
        return ids

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.word2idx, f)

    def __len__(self):
        return len(self.word2idx)


# ── Dataset ───────────────────────────────────────────────────────────────────
class MCQDataset(Dataset):
    def __init__(self, df, vocab, max_length):
        self.df         = df.reset_index(drop=True)
        self.vocab      = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        prompt = str(row["prompt"])
        ids    = []
        for col in OPTION_COLS:
            text    = prompt + " " + str(row[col])
            encoded = self.vocab.encode(text, self.max_length)
            ids.append(encoded)
        input_ids = torch.tensor(ids, dtype=torch.long)
        label     = torch.tensor(LABEL_MAP[row["answer"]], dtype=torch.long)
        return input_ids, label


class MCQTestDataset(Dataset):
    def __init__(self, df, vocab, max_length):
        self.df         = df.reset_index(drop=True)
        self.vocab      = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        prompt = str(row["prompt"])
        ids    = []
        for col in OPTION_COLS:
            text    = prompt + " " + str(row[col])
            encoded = self.vocab.encode(text, self.max_length)
            ids.append(encoded)
        return torch.tensor(ids, dtype=torch.long)


# ── Label smoothing loss ──────────────────────────────────────────────────────
class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing: float = 0.1, num_classes: int = 5):
        super().__init__()
        self.smoothing   = smoothing
        self.num_classes = num_classes

    def forward(self, logits, labels):
        confidence  = 1.0 - self.smoothing
        smooth_val  = self.smoothing / (self.num_classes - 1)
        one_hot     = torch.full_like(logits, smooth_val)
        one_hot.scatter_(1, labels.unsqueeze(1), confidence)
        log_probs   = F.log_softmax(logits, dim=-1)
        loss        = -(one_hot * log_probs).sum(dim=-1).mean()
        return loss


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


# ── Train epoch ───────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for input_ids, labels in tqdm(loader, desc="Train", leave=False):
        input_ids = input_ids.to(DEVICE)
        labels    = labels.to(DEVICE)
        optimizer.zero_grad()
        logits    = model(input_ids)
        loss      = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# ── Eval epoch ────────────────────────────────────────────────────────────────
def eval_epoch(model, loader):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for input_ids, labels in tqdm(loader, desc="Eval ", leave=False):
            input_ids = input_ids.to(DEVICE)
            logits    = model(input_ids).cpu().numpy()
            all_logits.extend(logits)
            all_labels.extend([IDX_TO_LABEL[l.item()] for l in labels])
    return np.array(all_logits), all_labels


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
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    # ── Vocabulary from raw train text ────────────────────────────────────────
    print("\nBuilding vocabulary from raw text...")
    all_texts = []
    for _, row in train_df.iterrows():
        all_texts.append(str(row["prompt"]))
        for col in OPTION_COLS:
            all_texts.append(str(row[col]))

    vocab = Vocabulary(min_freq=CFG["min_freq"])
    vocab.build(all_texts)

    # ── Datasets & loaders ────────────────────────────────────────────────────
    train_ds     = MCQDataset(train_df, vocab, CFG["max_length"])
    val_ds       = MCQDataset(val_df,   vocab, CFG["max_length"])
    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"], shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = GRUMCQModel(
        vocab_size = len(vocab),
        embed_dim  = CFG["embed_dim"],
        hidden_dim = CFG["hidden_dim"],
        num_layers = CFG["num_layers"],
        num_heads  = CFG["num_heads"],
        dropout    = CFG["dropout"],
        pad_idx    = PAD_IDX,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params    : {total_params:,}")
    print(f"Trainable params: {train_params:,}")

    # ── Loss, optimizer, scheduler ────────────────────────────────────────────
    criterion = LabelSmoothingCE(
        smoothing   = CFG["label_smooth"],
        num_classes = 5,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = CFG["lr"],
        weight_decay = CFG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max  = CFG["epochs"] - CFG["warmup_epochs"],
        eta_min= 1e-5,
    )
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor = 0.1,
        end_factor   = 1.0,
        total_iters  = CFG["warmup_epochs"],
    )

    # ── W&B ───────────────────────────────────────────────────────────────────
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    wandb.init(
        entity  = config["project"].get("wandb_entity"),
        project = config["project"]["wandb_project"],
        name    = "gru-from-scratch",
        config  = {
            **CFG,
            "vocab_size"      : len(vocab),
            "trainable_params": train_params,
            "optimizer"       : "AdamW",
            "scheduler"       : "CosineAnnealingLR + LinearWarmup",
            "loss"            : "LabelSmoothingCE",
        }
    )

    # ── Training ──────────────────────────────────────────────────────────────
    best_map3    = 0.0
    patience_ctr = 0
    os.makedirs(SAVE_DIR, exist_ok=True)

    for epoch in range(1, CFG["epochs"] + 1):
        train_loss             = train_epoch(model, train_loader, optimizer, criterion)
        val_logits, val_labels = eval_epoch(model, val_loader)
        val_metrics, _         = compute_metrics(val_logits, val_labels)

        # warmup first, then cosine
        if epoch <= CFG["warmup_epochs"]:
            warmup.step()
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d} | "
            f"loss={train_loss:.4f} | "
            f"map3={val_metrics['map3']} | "
            f"acc={val_metrics['top1_accuracy']} | "
            f"f1={val_metrics['macro_f1']} | "
            f"lr={current_lr:.6f}"
        )

        wandb.log({
            "epoch"             : epoch,
            "train/loss"        : train_loss,
            "val/map3"          : val_metrics["map3"],
            "val/top1_accuracy" : val_metrics["top1_accuracy"],
            "val/macro_f1"      : val_metrics["macro_f1"],
            "lr"                : current_lr,
        })

        if val_metrics["map3"] > best_map3:
            best_map3    = val_metrics["map3"]
            patience_ctr = 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pt"))
            print(f"  Saved best model (map3={best_map3})")
        else:
            patience_ctr += 1
            if patience_ctr >= CFG["patience"]:
                print(f"  Early stopping at epoch {epoch}")
                break

    wandb.summary["best_val_map3"]     = best_map3
    wandb.summary["val_macro_f1"]      = val_metrics["macro_f1"]
    wandb.summary["val_top1_accuracy"] = val_metrics["top1_accuracy"]
    wandb.summary["model_type"]        = "from-scratch"
    wandb.summary["trainable_params"]  = train_params
    wandb.finish()

    # ── Final eval ────────────────────────────────────────────────────────────
    print("\nLoading best model for final eval...")
    model.load_state_dict(
        torch.load(os.path.join(SAVE_DIR, "best_model.pt"), map_location=DEVICE)
    )
    val_logits, val_labels = eval_epoch(model, val_loader)
    final_metrics, _       = compute_metrics(val_logits, val_labels)

    print("\n── Final Val Metrics ────────────────────────")
    for k, v in final_metrics.items():
        print(f"  {k:20s}: {v}")

    # ── Save vocab ────────────────────────────────────────────────────────────
    vocab.save(os.path.join(SAVE_DIR, "vocab.json"))
    print(f"\nVocab saved: {SAVE_DIR}/vocab.json")

    # ── Test submission ───────────────────────────────────────────────────────
    print("\nGenerating test submission...")
    test_df     = pd.read_csv("data/raw/test.csv")
    test_ds     = MCQTestDataset(test_df, vocab, CFG["max_length"])
    test_loader = DataLoader(test_ds, batch_size=CFG["batch_size"], shuffle=False, num_workers=0)

    model.eval()
    all_test_logits = []
    with torch.no_grad():
        for input_ids in tqdm(test_loader, desc="Test"):
            logits = model(input_ids.to(DEVICE)).cpu().numpy()
            all_test_logits.extend(logits)

    predictions = [
        " ".join([IDX_TO_LABEL[i] for i in np.argsort(l)[::-1][:3]])
        for l in all_test_logits
    ]
    submission = pd.DataFrame({
        "id"        : test_df["id"],
        "Prediction": predictions,
    })
    os.makedirs("outputs", exist_ok=True)
    submission.to_csv("outputs/submission_gru.csv", index=False)

    print(f"\nSubmission saved: outputs/submission_gru.csv")
    print(submission.head())
    print("\n── Summary ──────────────────────────────────")
    print(f"  Best val MAP@3   : {best_map3}")
    print(f"  Val Accuracy     : {final_metrics['top1_accuracy']}")
    print(f"  Val Macro F1     : {final_metrics['macro_f1']}")
    print(f"  Vocab size       : {len(vocab):,}")
    print(f"  Trainable params : {train_params:,}")


if __name__ == "__main__":
    main()