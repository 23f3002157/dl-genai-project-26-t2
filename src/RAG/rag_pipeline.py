"""
rag_pipeline.py
---------------
Full RAG pipeline for MCQ:
    1. Retrieve  — FAISS bi-encoder retrieves top-k KB entries
    2. Rerank    — CrossEncoder picks the single best doc
    3. Augment   — prepend context to prompt
    4. Read      — reader model scores all 5 options
    5. Evaluate  — MAP@3, Accuracy, Macro F1 on val split

Reader model is configurable — see READER_MODEL below.
Switch between options to compare performance.

Location : src/rag/rag_pipeline.py
Run      : python3 -m src.rag.rag_pipeline
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
import wandb
import faiss
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForMultipleChoice
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils import map_at_3, load_config, set_seed

load_dotenv()
set_seed(42)

# ── Reader model options ──────────────────────────────────────────────────────
# Switch READER_MODEL to compare performance across readers.
# All use AutoModelForMultipleChoice — no fine-tuning needed.

# READER_MODEL = "LIAMF-USP/roberta-large-finetuned-race"
# READER_MODEL = "LIAMF-USP/roberta-large-finetuned-race"   # RoBERTa-large, RACE MCQ — strong baseline
# READER_MODEL = "potsawee/longformer-large-4096-answering-race"  # Longformer, handles long context
# READER_MODEL = "mrm8488/bert-large-finetuned-squadv2"     # BERT-large, SQuAD2 QA
# READER_MODEL = "easonnie/deberta-v3-base-race"            # DeBERTa-base, RACE MCQ
READER_MODEL = "ALBERT/albert-xxlarge-v2"                 # ALBERT xxlarge, strong on MCQ

# ── Config ────────────────────────────────────────────────────────────────────
CFG = {
    "bi_encoder"   : "sentence-transformers/all-MiniLM-L6-v2",
    "cross_encoder": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "reader_model" : READER_MODEL,
    "retrieve_k"   : 5,
    "max_length"   : 128,
    "batch_size"   : 8,
}

OPTION_COLS  = ["A", "B", "C", "D", "E"]
LABEL_MAP    = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
RAG_DIR      = "models/rag"
DEVICE       = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ── Load KB ───────────────────────────────────────────────────────────────────
def load_kb():
    index = faiss.read_index(os.path.join(RAG_DIR, "faiss_index.bin"))
    with open(os.path.join(RAG_DIR, "kb.pkl"), "rb") as f:
        kb = pickle.load(f)
    with open(os.path.join(RAG_DIR, "meta.json")) as f:
        meta = json.load(f)
    print(f"KB: {len(kb)} entries | FAISS: {index.ntotal} vectors")
    return index, kb, meta


# ── Step 1: Retrieve ──────────────────────────────────────────────────────────
def retrieve(prompt: str, index, bi_encoder: SentenceTransformer, k: int) -> list:
    emb          = bi_encoder.encode([prompt], convert_to_numpy=True).astype("float32")
    _, idxs      = index.search(emb, k)
    return idxs[0].tolist()


# ── Step 2: Rerank ────────────────────────────────────────────────────────────
def rerank(prompt: str, docs: list, cross_encoder: CrossEncoder) -> str:
    pairs  = [[prompt, doc] for doc in docs]
    scores = cross_encoder.predict(pairs)
    return docs[int(np.argmax(scores))]


# ── Step 3 + 4: Augment + Score ───────────────────────────────────────────────
def score_options(
    model, tokenizer, prompt: str, context: str,
    row, device, max_length: int
) -> np.ndarray:
    """
    Prepend retrieved context to prompt.
    AutoModelForMultipleChoice takes (prompt, option) pairs.
    Returns logits of shape (5,).
    """
    augmented_prompt    = f"Context: {context} {prompt}"
    input_ids_list      = []
    attention_mask_list = []

    for col in OPTION_COLS:
        enc = tokenizer(
            augmented_prompt,
            str(row[col]),
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids_list.append(enc["input_ids"].squeeze(0))
        attention_mask_list.append(enc["attention_mask"].squeeze(0))

    input_ids      = torch.stack(input_ids_list).unsqueeze(0).to(device)
    attention_mask = torch.stack(attention_mask_list).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits[0].cpu().numpy()   # (5,)

    return logits


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


# ── Pipeline loop ─────────────────────────────────────────────────────────────
def run_pipeline(df, index, kb, bi_encoder, cross_encoder,
                 reader_model, tokenizer, device, is_test=False):
    all_logits = []
    all_labels = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="RAG"):
        prompt = str(row["prompt"])

        retrieved_idxs = retrieve(prompt, index, bi_encoder, CFG["retrieve_k"])
        retrieved_docs = [kb[i]["text"] for i in retrieved_idxs]
        best_doc       = rerank(prompt, retrieved_docs, cross_encoder)
        logits         = score_options(
            reader_model, tokenizer, prompt, best_doc,
            row, device, CFG["max_length"]
        )
        all_logits.append(logits)

        if not is_test:
            all_labels.append(row["answer"])

    return np.array(all_logits), all_labels


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    config = load_config()
    print(f"Device       : {DEVICE}")
    print(f"Reader model : {CFG['reader_model']}")

    # ── KB ────────────────────────────────────────────────────────────────────
    index, kb, meta = load_kb()

    # ── Models ───────────────────────────────────────────────────────────────
    print("\nLoading bi-encoder...")
    bi_encoder = SentenceTransformer(CFG["bi_encoder"])

    print("Loading cross-encoder...")
    cross_encoder = CrossEncoder(CFG["cross_encoder"])

    print("Loading reader model...")
    tokenizer    = AutoTokenizer.from_pretrained(CFG["reader_model"])
    reader_model = AutoModelForMultipleChoice.from_pretrained(
        CFG["reader_model"]
    ).to(DEVICE)
    reader_model.eval()

    # ── Data ─────────────────────────────────────────────────────────────────
    df = pd.read_csv("data/raw/train.csv")
    _, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["answer"]
    )
    val_df = val_df.reset_index(drop=True)
    print(f"Val set: {len(val_df)} rows")

    # ── W&B ───────────────────────────────────────────────────────────────────
    run_name = "rag-" + CFG["reader_model"].split("/")[-1]
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    wandb.init(
        entity  = config["project"].get("wandb_entity"),
        project = config["project"]["wandb_project"],
        name    = run_name,
        config  = {
            **CFG,
            "kb_size"  : len(kb),
            "val_size" : len(val_df),
            "pipeline" : "FAISS → CrossEncoder → MultipleChoice reader",
        }
    )

    # ── Val inference ─────────────────────────────────────────────────────────
    print("\nRunning RAG pipeline on val set...")
    val_logits, val_labels = run_pipeline(
        val_df, index, kb, bi_encoder, cross_encoder,
        reader_model, tokenizer, DEVICE
    )
    val_metrics, _ = compute_metrics(val_logits, val_labels)

    print("\n── Val Metrics ──────────────────────────────")
    for k, v in val_metrics.items():
        print(f"  {k:20s}: {v}")

    wandb.log({
        "val/map3"          : val_metrics["map3"],
        "val/top1_accuracy" : val_metrics["top1_accuracy"],
        "val/macro_f1"      : val_metrics["macro_f1"],
    })
    wandb.summary["val_map3"]          = val_metrics["map3"]
    wandb.summary["val_macro_f1"]      = val_metrics["macro_f1"]
    wandb.summary["val_top1_accuracy"] = val_metrics["top1_accuracy"]
    wandb.summary["model_type"]        = "rag-pipeline"
    wandb.summary["reader_model"]      = CFG["reader_model"]
    wandb.finish()

    # ── Test submission ───────────────────────────────────────────────────────
    print("\nGenerating test submission...")
    test_df = pd.read_csv("data/raw/test.csv").reset_index(drop=True)

    test_logits, _ = run_pipeline(
        test_df, index, kb, bi_encoder, cross_encoder,
        reader_model, tokenizer, DEVICE, is_test=True
    )

    predictions = [
        " ".join([IDX_TO_LABEL[i] for i in np.argsort(l)[::-1][:3]])
        for l in test_logits
    ]
    submission = pd.DataFrame({
        "id"        : test_df["id"],
        "Prediction": predictions,
    })
    os.makedirs("outputs", exist_ok=True)
    slug = CFG["reader_model"].split("/")[-1]
    out_path = f"outputs/submission_rag_{slug}.csv"
    submission.to_csv(out_path, index=False)

    print(f"\nSubmission saved: {out_path}")
    print(submission.head())
    print("\n── Summary ──────────────────────────────────")
    print(f"  Reader       : {CFG['reader_model']}")
    print(f"  val MAP@3    : {val_metrics['map3']}")
    print(f"  val Accuracy : {val_metrics['top1_accuracy']}")
    print(f"  val Macro F1 : {val_metrics['macro_f1']}")


if __name__ == "__main__":
    main()