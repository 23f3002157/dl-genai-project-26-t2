"""
train_lstm.py
-------------
Full training pipeline for the from-scratch LSTM MCQ model.
Builds vocabulary from train data, trains with CrossEntropyLoss,
evaluates MAP@3 / Accuracy / Macro F1, logs to W&B.

Location : src/train_lstm.py
Run      : python3 -m src.train_lstm
"""

import os
import re
import string
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
import pickle
from collections import Counter
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.utils import map_at_3, load_config, set_seed
from src.lstm_model import LSTMMCQModel

load_dotenv()
set_seed(42)

# ── Hyperparameters ───────────────────────────────────────────────────────────
CFG = {
    "model"       : "lstm-from-scratch",
    "embed_dim"   : 128,
    "hidden_dim"  : 256,
    "num_layers"  : 2,
    "dropout"     : 0.3,
    "max_length"  : 128,
    "min_freq"    : 2,
    "batch_size"  : 32,
    "epochs"      : 10,
    "lr"          : 1e-3,
    "weight_decay": 1e-4,
    "patience"    : 3,
}

OPTION_COLS  = ["A", "B", "C", "D", "E"]
LABEL_MAP    = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
DEVICE       = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
PAD_IDX      = 0
UNK_IDX      = 1


# ── Vocabulary ────────────────────────────────────────────────────────────────
class Vocabulary:
    def __init__(self, min_freq: int = 2):
        self.min_freq  = min_freq
        self.word2idx  = {"<PAD>": PAD_IDX, "<UNK>": UNK_IDX}
        self.idx2word  = {PAD_IDX: "<PAD>", UNK_IDX: "<UNK>"}

    def build(self, texts: list):
        counter = Counter()
        for text in texts:
            counter.update(self.tokenize(text))
        for word, freq in counter.items():
            if freq >= self.min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx]  = word
        print(f"Vocabulary size: {len(self.word2idx)}")

    def tokenize(self, text: str) -> list:
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text.split()

    def encode(self, text: str, max_length: int) -> list:
        tokens = self.tokenize(text)[:max_length]
        ids    = [self.word2idx.get(t, UNK_IDX) for t in tokens]
        ids   += [PAD_IDX] * (max_length - len(ids))
        return ids

    def __len__(self):
        return len(self.word2idx)


# ── Dataset ───────────────────────────────────────────────────────────────────
class MCQDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocab: Vocabulary, max_length: int):
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
            text      = prompt + " " + str(row[col])
            encoded   = self.vocab.encode(text, self.max_length)
            ids.append(encoded)
        input_ids = torch.tensor(ids, dtype=torch.long)       # (5, max_length)
        label     = torch.tensor(LABEL_MAP[row["answer"]], dtype=torch.long)
        return input_ids, label


class MCQTestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocab: Vocabulary, max_length: int):
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


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(all_logits: np.ndarray, labels: list):
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
        zero_division=0
    )
    map3 = map_at_3(top3_preds, labels)
    return {
        "top1_accuracy": round(accuracy, 4),
        "macro_f1"     : round(f1, 4),
        "map3"         : round(map3, 4),
    }, top3_preds


# ── Training loop ─────────────────────────────────────────────────────────────
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


def eval_epoch(model, loader):
    model.eval()
    all_logits = []
    all_labels = []
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

    # ── Load & split data ─────────────────────────────────────────────────────
    df       = pd.read_csv("data/raw/train.csv")
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["answer"]
    )
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    # ── Build vocabulary from train set only ──────────────────────────────────
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
    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"], shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = LSTMMCQModel(
        vocab_size  = len(vocab),
        embed_dim   = CFG["embed_dim"],
        hidden_dim  = CFG["hidden_dim"],
        num_layers  = CFG["num_layers"],
        dropout     = CFG["dropout"],
        pad_idx     = PAD_IDX,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params    : {total_params:,}")
    print(f"Trainable params: {train_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CFG["lr"],
        weight_decay=CFG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    # ── W&B ───────────────────────────────────────────────────────────────────
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    wandb.init(
        entity  = config["project"].get("wandb_entity"),
        project = config["project"]["wandb_project"],
        name    = "lstm-from-scratch",
        config  = {**CFG, "vocab_size": len(vocab), "trainable_params": train_params}
    )

    # ── Training ──────────────────────────────────────────────────────────────
    best_map3    = 0.0
    patience_ctr = 0
    os.makedirs("models/lstm", exist_ok=True)

    for epoch in range(1, CFG["epochs"] + 1):
        train_loss             = train_epoch(model, train_loader, optimizer, criterion)
        val_logits, val_labels = eval_epoch(model, val_loader)
        val_metrics, _         = compute_metrics(val_logits, val_labels)
        scheduler.step(val_metrics["map3"])

        print(
            f"Epoch {epoch:02d} | "
            f"loss={train_loss:.4f} | "
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
            torch.save(model.state_dict(), "models/lstm/best_model.pt")
            print(f"  ✅ Saved best model (map3={best_map3})")
        else:
            patience_ctr += 1
            if patience_ctr >= CFG["patience"]:
                print(f"Early stopping at epoch {epoch}")
                break

    wandb.summary["best_val_map3"]     = best_map3
    wandb.summary["val_macro_f1"]      = val_metrics["macro_f1"]
    wandb.summary["val_top1_accuracy"] = val_metrics["top1_accuracy"]
    wandb.summary["model_type"]        = "from-scratch"
    wandb.summary["trainable_params"]  = train_params

    # ── Final eval with best model ────────────────────────────────────────────
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load("models/lstm/best_model.pt", map_location=DEVICE))
    val_logits, val_labels = eval_epoch(model, val_loader)
    final_metrics, _       = compute_metrics(val_logits, val_labels)

    print("\n── Final Val Metrics ────────────────────────")
    for k, v in final_metrics.items():
        print(f"  {k:20s}: {v}")

    # ── Generate test submission ───────────────────────────────────────────────
    print("\nGenerating test submission...")
    test_df  = pd.read_csv("data/raw/test.csv")
    test_ds  = MCQTestDataset(test_df, vocab, CFG["max_length"])
    test_loader = DataLoader(test_ds, batch_size=CFG["batch_size"], shuffle=False)

    model.eval()
    all_test_logits = []
    with torch.no_grad():
        for input_ids in tqdm(test_loader, desc="Test inference"):
            logits = model(input_ids.to(DEVICE)).cpu().numpy()
            all_test_logits.extend(logits)

    predictions = [
        " ".join([IDX_TO_LABEL[i] for i in np.argsort(l)[::-1][:3]])
        for l in all_test_logits
    ]
    submission = pd.DataFrame({
        "id"         : test_df["id"],
        "Prediction" : predictions
    })
    os.makedirs("outputs", exist_ok=True)
    submission.to_csv("outputs/submission_lstm.csv", index=False)

    print(f"\nSubmission saved: outputs/submission_lstm.csv")
    print(submission.head())
    print("\n── Summary ──────────────────────────────────")
    print(f"  Best val MAP@3   : {best_map3}")
    print(f"  Val Accuracy     : {final_metrics['top1_accuracy']}")
    print(f"  Val Macro F1     : {final_metrics['macro_f1']}")
    print(f"  Vocab size       : {len(vocab)}")
    print(f"  Trainable params : {train_params:,}")

    with open("models/lstm/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    print("Vocab saved: models/lstm/vocab.pkl")
    wandb.finish()


if __name__ == "__main__":
    main()