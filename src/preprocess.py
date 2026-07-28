"""
preprocess.py
-------------
Cleans train.csv and test.csv, saves processed files to data/processed/.
Also exports utility functions for use across other scripts.

Location : src/preprocess.py
Run      : python3 -m src.preprocess
"""

import os
import re
import string
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

OPTION_COLS = ["A", "B", "C", "D", "E"]
LABEL_MAP   = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

TRAIN_IN  = "data/raw/train.csv"
TEST_IN   = "data/raw/test.csv"
TRAIN_OUT = "data/processed/train_processed.csv"
TEST_OUT  = "data/processed/test_processed.csv"


# ── Utility functions (importable by other scripts) ───────────────────────────

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [t for t in text.split() if t not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)


def get_combined_texts(df: pd.DataFrame) -> list:
    texts = list(df["prompt"])
    for col in OPTION_COLS:
        texts.extend(df[col].tolist())
    return [str(t) for t in texts if pd.notna(t)]


def check_missing(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isnull().sum()
    pct     = (missing / len(df) * 100).round(2)
    summary = pd.DataFrame({"missing_count": missing, "missing_pct": pct})
    return summary[summary["missing_count"] > 0]


def load_data(train_path: str, test_path: str = None):
    train_df = pd.read_csv(train_path)
    print(f"Train shape  : {train_df.shape}")
    if test_path:
        test_df = pd.read_csv(test_path)
        print(f"Test shape   : {test_df.shape}")
        return train_df, test_df
    return train_df


# ── Processing pipeline ───────────────────────────────────────────────────────

def process(df: pd.DataFrame, is_test: bool = False) -> pd.DataFrame:
    out = df.copy()

    out["prompt_clean"] = out["prompt"].apply(clean_text)
    for col in OPTION_COLS:
        out[f"{col}_clean"] = out[col].apply(clean_text)

    out["input_text"] = out.apply(
        lambda r: r["prompt_clean"] + " " +
                  " ".join([r[f"{c}_clean"] for c in OPTION_COLS]),
        axis=1
    )

    if not is_test and "answer" in out.columns:
        out["label"] = out["answer"].map(LABEL_MAP)

    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("data/processed", exist_ok=True)

    print("Loading data...")
    train_df = pd.read_csv(TRAIN_IN)
    test_df  = pd.read_csv(TEST_IN)
    print(f"Train : {train_df.shape}")
    print(f"Test  : {test_df.shape}")

    print("\nPreprocessing...")
    train_proc = process(train_df, is_test=False)
    test_proc  = process(test_df,  is_test=True)

    train_proc.to_csv(TRAIN_OUT, index=False)
    test_proc.to_csv(TEST_OUT,   index=False)

    print(f"\nSaved: {TRAIN_OUT}")
    print(f"Saved: {TEST_OUT}")
    print(f"\nColumns: {train_proc.columns.tolist()}")
    print(f"\nSample:")
    print(f"  Original : {train_df.iloc[0]['prompt'][:80]}")
    print(f"  Cleaned  : {train_proc.iloc[0]['prompt_clean'][:80]}")


if __name__ == "__main__":
    main()