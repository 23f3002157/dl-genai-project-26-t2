"""
train_electra_tfidf_nn.py
--------------------------
Training pipeline for the "model of choice": frozen ELECTRA (or MiniLM) sentence
embeddings + TF-IDF (reduced via SVD) features, fed into a small feed-forward NN
that scores each MCQ option. The transformer is used purely as a fixed feature
extractor here (not fine-tuned) -- that's what differentiates it from a fully
fine-tuned pretrained model. Mirrors the from-scratch LSTM pipeline: same data
paths, MAP@3 / Accuracy / Macro F1 metrics, W&B logging, Kaggle submission format.

Location : src/electra/train_electra.py
Run      : python3 -m src.electra.train_electra
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from src.utils import map_at_3, load_config, set_seed

load_dotenv()
set_seed(42)

# ── Paths (feed your own here) ─────────────────────────────────────────────────
TRAIN_CSV_PATH     = "data/raw/train.csv"
TEST_CSV_PATH      = "data/raw/test.csv"
MODEL_SAVE_DIR      = "models/electra_tfidf_nn"
FEATURE_CACHE_DIR  = "models/electra_tfidf_nn/feature_cache"
SUBMISSION_PATH    = "outputs/submission_electra_tfidf_nn.csv"

# ── Hyperparameters ────────────────────────────────────────────────────────────
CFG = {
    "model"              : "electra-mini-tfidf-nn",
    # Swap to "microsoft/MiniLM-L12-H384-uncased" if you'd rather use MiniLM.
    "transformer_name"   : "google/electra-small-discriminator",
    "max_length"         : 128,
    "tfidf_max_features" : 5000,
    "tfidf_svd_dim"      : 128,
    "hidden_dim"         : 256,
    "dropout"            : 0.3,
    "batch_size"         : 32,
    "epochs"             : 15,
    "lr"                 : 1e-3,
    "weight_decay"       : 1e-4,
    "patience"           : 3,
    "embed_batch_size"   : 64,   # batch size used only for frozen-embedding extraction
}

OPTION_COLS  = ["A", "B", "C", "D", "E"]
LABEL_MAP    = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)


# ── Text helper ─────────────────────────────────────────────────────────────────
def build_option_text(prompt: str, option: str) -> str:
    return f"{prompt} {option}"


# ── Frozen transformer embedding extractor ──────────────────────────────────────
class TransformerEmbedder:
    """Wraps a frozen pretrained transformer for mean-pooled sentence embeddings.
    Weights are NOT fine-tuned; used purely as a fixed feature extractor."""

    def __init__(self, model_name: str, max_length: int, device: torch.device):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.max_length = max_length
        self.device      = device
        self.embed_dim   = self.model.config.hidden_size

    @torch.no_grad()
    def encode(self, texts: list, batch_size: int) -> np.ndarray:
        all_embeds = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding", leave=False):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt"
            ).to(self.device)
            out    = self.model(**enc).last_hidden_state           # (B, T, H)
            mask   = enc["attention_mask"].unsqueeze(-1).float()   # (B, T, 1)
            summed = (out * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            mean_pooled = (summed / counts).cpu().numpy()          # (B, H)
            all_embeds.append(mean_pooled)
        return np.concatenate(all_embeds, axis=0)


# ── Feature builder (TF-IDF + transformer, cached to disk) ──────────────────────
def build_features(df: pd.DataFrame, tfidf: TfidfVectorizer, svd: TruncatedSVD,
                    embedder: TransformerEmbedder, cfg: dict, cache_path: str) -> np.ndarray:
    """Returns array of shape (N, 5, tfidf_svd_dim + embed_dim)."""
    if os.path.exists(cache_path):
        print(f"  Loading cached features: {cache_path}")
        return np.load(cache_path)

    texts_per_option = {col: [] for col in OPTION_COLS}
    for _, row in df.iterrows():
        prompt = str(row["prompt"])
        for col in OPTION_COLS:
            texts_per_option[col].append(build_option_text(prompt, str(row[col])))

    option_feats = []
    for col in OPTION_COLS:
        texts        = texts_per_option[col]
        tfidf_sparse = tfidf.transform(texts)
        tfidf_dense  = svd.transform(tfidf_sparse)                       # (N, svd_dim)
        embeds       = embedder.encode(texts, cfg["embed_batch_size"])   # (N, embed_dim)
        combined     = np.concatenate([tfidf_dense, embeds], axis=1)     # (N, feat_dim)
        option_feats.append(combined)

    features = np.stack(option_feats, axis=1)  # (N, 5, feat_dim)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, features)
    print(f"  Cached features: {cache_path}  shape={features.shape}")
    return features


# ── Dataset ────────────────────────────────────────────────────────────────────
class MCQFeatureDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: list = None):
        self.features = torch.tensor(features, dtype=torch.float32)  # (N, 5, feat_dim)
        self.labels = (
            torch.tensor([LABEL_MAP[l] for l in labels], dtype=torch.long)
            if labels is not None else None
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]


# ── Model ──────────────────────────────────────────────────────────────────────
class ElectraTfidfNN(nn.Module):
    """Small feed-forward scorer applied per-option: takes the concatenated
    [TF-IDF(SVD) | frozen transformer embedding] feature vector for one option
    and outputs a single relevance score. Run over all 5 options -> 5 logits."""

    def __init__(self, feat_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        # x: (B, 5, feat_dim)
        B, K, F = x.shape
        x = x.reshape(B * K, F)
        scores = self.net(x)          # (B*K, 1)
        return scores.view(B, K)      # (B, 5)


# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(all_logits: np.ndarray, labels: list):
    top1_preds = [IDX_TO_LABEL[int(np.argmax(l))] for l in all_logits]
    top3_preds = [
        [IDX_TO_LABEL[i] for i in np.argsort(l)[::-1][:3]]
        for l in all_logits
    ]
    accuracy = accuracy_score(labels, top1_preds)
    f1 = f1_score(labels, top1_preds, labels=OPTION_COLS, average="macro", zero_division=0)
    map3 = map_at_3(top3_preds, labels)
    return {
        "top1_accuracy": round(accuracy, 4),
        "macro_f1"     : round(f1, 4),
        "map3"         : round(map3, 4),
    }, top3_preds


# ── Training loop ──────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for features, labels in tqdm(loader, desc="Train", leave=False):
        features = features.to(DEVICE)
        labels   = labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(features)
        loss   = criterion(logits, labels)
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
        for features, labels in tqdm(loader, desc="Eval ", leave=False):
            features = features.to(DEVICE)
            logits   = model(features).cpu().numpy()
            all_logits.extend(logits)
            all_labels.extend([IDX_TO_LABEL[l.item()] for l in labels])
    return np.array(all_logits), all_labels


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    config = load_config()
    print(f"Device : {DEVICE}")

    # ── Load & split data ────────────────────────────────────────────────────
    df = pd.read_csv(TRAIN_CSV_PATH)
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["answer"]
    )
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    # ── Fit TF-IDF + SVD on TRAIN text only (avoid leakage) ─────────────────
    print("Fitting TF-IDF + SVD on train corpus...")
    train_corpus = []
    for _, row in train_df.iterrows():
        prompt = str(row["prompt"])
        for col in OPTION_COLS:
            train_corpus.append(build_option_text(prompt, str(row[col])))

    tfidf = TfidfVectorizer(max_features=CFG["tfidf_max_features"], ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(train_corpus)

    svd = TruncatedSVD(n_components=CFG["tfidf_svd_dim"], random_state=42)
    svd.fit(tfidf_matrix)

    # ── Frozen transformer embedder ──────────────────────────────────────────
    print(f"Loading frozen transformer: {CFG['transformer_name']}")
    embedder = TransformerEmbedder(CFG["transformer_name"], CFG["max_length"], DEVICE)
    feat_dim = CFG["tfidf_svd_dim"] + embedder.embed_dim
    print(f"Feature dim (tfidf_svd + transformer): {feat_dim}")

    # ── Build (or load cached) features ──────────────────────────────────────
    os.makedirs(FEATURE_CACHE_DIR, exist_ok=True)
    print("Building train features...")
    train_features = build_features(
        train_df, tfidf, svd, embedder, CFG,
        os.path.join(FEATURE_CACHE_DIR, "train_features.npy")
    )
    print("Building val features...")
    val_features = build_features(
        val_df, tfidf, svd, embedder, CFG,
        os.path.join(FEATURE_CACHE_DIR, "val_features.npy")
    )

    # ── Datasets & loaders ────────────────────────────────────────────────────
    train_ds = MCQFeatureDataset(train_features, train_df["answer"].tolist())
    val_ds   = MCQFeatureDataset(val_features,   val_df["answer"].tolist())
    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"], shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ElectraTfidfNN(
        feat_dim   = feat_dim,
        hidden_dim = CFG["hidden_dim"],
        dropout    = CFG["dropout"],
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params    : {total_params:,}")
    print(f"Trainable params: {train_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    # ── W&B ───────────────────────────────────────────────────────────────────
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    wandb.init(
        entity  = config["project"].get("wandb_entity"),
        project = config["project"]["wandb_project"],
        name    = "electra-tfidf-nn",
        config  = {**CFG, "feat_dim": feat_dim, "trainable_params": train_params}
    )

    # ── Training ──────────────────────────────────────────────────────────────
    best_map3    = 0.0
    patience_ctr = 0
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    for epoch in range(1, CFG["epochs"] + 1):
        train_loss              = train_epoch(model, train_loader, optimizer, criterion)
        val_logits, val_labels  = eval_epoch(model, val_loader)
        val_metrics, _          = compute_metrics(val_logits, val_labels)
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
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "best_model.pt"))
            print(f"  ✅ Saved best model (map3={best_map3})")
        else:
            patience_ctr += 1
            if patience_ctr >= CFG["patience"]:
                print(f"Early stopping at epoch {epoch}")
                break

    wandb.summary["best_val_map3"]     = best_map3
    wandb.summary["val_macro_f1"]      = val_metrics["macro_f1"]
    wandb.summary["val_top1_accuracy"] = val_metrics["top1_accuracy"]
    wandb.summary["model_type"]        = "electra-tfidf-nn"
    wandb.summary["trainable_params"]  = train_params

    # ── Final eval with best model ────────────────────────────────────────────
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_DIR, "best_model.pt"), map_location=DEVICE))
    val_logits, val_labels = eval_epoch(model, val_loader)
    final_metrics, _       = compute_metrics(val_logits, val_labels)

    print("\n── Final Val Metrics ────────────────────────")
    for k, v in final_metrics.items():
        print(f"  {k:20s}: {v}")

    # ── Generate test submission ─────────────────────────────────────────────
    print("\nGenerating test submission...")
    test_df = pd.read_csv(TEST_CSV_PATH)
    test_features = build_features(
        test_df, tfidf, svd, embedder, CFG,
        os.path.join(FEATURE_CACHE_DIR, "test_features.npy")
    )
    test_ds     = MCQFeatureDataset(test_features)
    test_loader = DataLoader(test_ds, batch_size=CFG["batch_size"], shuffle=False)

    model.eval()
    all_test_logits = []
    with torch.no_grad():
        for features in tqdm(test_loader, desc="Test inference"):
            logits = model(features.to(DEVICE)).cpu().numpy()
            all_test_logits.extend(logits)

    predictions = [
        " ".join([IDX_TO_LABEL[i] for i in np.argsort(l)[::-1][:3]])
        for l in all_test_logits
    ]
    submission = pd.DataFrame({
        "id"        : test_df["id"],
        "Prediction": predictions
    })
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)

    print(f"\nSubmission saved: {SUBMISSION_PATH}")
    print(submission.head())
    print("\n── Summary ──────────────────────────────────")
    print(f"  Best val MAP@3   : {best_map3}")
    print(f"  Val Accuracy     : {final_metrics['top1_accuracy']}")
    print(f"  Val Macro F1     : {final_metrics['macro_f1']}")
    print(f"  Feature dim      : {feat_dim}")
    print(f"  Trainable params : {train_params:,}")

    # ── Save vectorizer / SVD for reproducible inference ────────────────────
    with open(os.path.join(MODEL_SAVE_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(tfidf, f)
    with open(os.path.join(MODEL_SAVE_DIR, "svd.pkl"), "wb") as f:
        pickle.dump(svd, f)
    print(f"TF-IDF vectorizer + SVD saved to: {MODEL_SAVE_DIR}")

    wandb.finish()


if __name__ == "__main__":
    main()