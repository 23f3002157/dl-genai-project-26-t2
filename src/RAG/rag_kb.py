"""
rag_kb.py
---------
Builds the FAISS knowledge base from train.csv.
Each KB entry = the correct answer text for one training row.
Saves FAISS index + KB metadata to models/rag/

Location : src/rag/rag_kb.py
Run      : python3 -m src.rag.rag_kb
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils import set_seed

set_seed(42)

ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TRAIN_PATH   = "data/raw/train.csv"
SAVE_DIR     = "models/rag"
OPTION_COLS  = ["A", "B", "C", "D", "E"]


def build_kb(df: pd.DataFrame) -> list:
    kb = []
    for _, row in df.iterrows():
        correct_letter = row["answer"]
        kb.append({
            "text"  : str(row[correct_letter]),
            "answer": correct_letter,
            "id"    : int(row["id"]),
        })
    return kb


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Loading train data...")
    df = pd.read_csv(TRAIN_PATH)
    print(f"Shape: {df.shape}")

    print("\nBuilding knowledge base...")
    kb = build_kb(df)
    print(f"KB entries: {len(kb)}")

    print("\nLoading encoder...")
    encoder = SentenceTransformer(ENCODER_NAME)

    print("\nEncoding KB entries...")
    texts      = [entry["text"] for entry in kb]
    embeddings = encoder.encode(
        texts,
        show_progress_bar=True,
        batch_size=64,
        convert_to_numpy=True,
    ).astype("float32")

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"FAISS index size : {index.ntotal}")
    print(f"Embedding dim    : {dim}")

    faiss.write_index(index, os.path.join(SAVE_DIR, "faiss_index.bin"))

    with open(os.path.join(SAVE_DIR, "kb.pkl"), "wb") as f:
        pickle.dump(kb, f)

    with open(os.path.join(SAVE_DIR, "meta.json"), "w") as f:
        json.dump({
            "encoder"  : ENCODER_NAME,
            "kb_size"  : len(kb),
            "embed_dim": int(dim),
        }, f, indent=2)

    print(f"\n✅ Saved to {SAVE_DIR}/")
    print("   faiss_index.bin | kb.pkl | meta.json")


if __name__ == "__main__":
    main()