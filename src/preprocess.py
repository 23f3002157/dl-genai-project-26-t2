"""
preprocess.py
-------------
Reusable text cleaning and preprocessing pipeline.
Used across all milestones. Import functions from here — do not duplicate logic.

Usage:
    from src.preprocess import clean_text, remove_stopwords, load_data
"""

import string
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

OPTION_COLS = ["A", "B", "C", "D", "E"]


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_data(train_path: str, test_path: str = None):
    """Load train and optionally test CSV files."""
    train_df = pd.read_csv(train_path)
    print(f"Train shape : {train_df.shape}")
    print(f"Train columns: {train_df.columns.tolist()}")
    if test_path:
        test_df = pd.read_csv(test_path)
        print(f"Test shape  : {test_df.shape}")
        return train_df, test_df
    return train_df


# ── Text Cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Lowercase and remove standard punctuation from a string.
    Does NOT remove stopwords — that's a separate step.

    Steps:
        1. Lowercase
        2. Remove all characters in string.punctuation
        3. Strip leading/trailing whitespace
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.strip()
    return text


def tokenize(text: str) -> list:
    """Split cleaned text by whitespace into tokens."""
    return text.split()


def remove_stopwords(tokens: list) -> list:
    """Remove sklearn's standard English stopwords from a token list."""
    return [t for t in tokens if t not in ENGLISH_STOP_WORDS]


def clean_and_tokenize(text: str, remove_stops: bool = False) -> list:
    """Full pipeline: clean → tokenize → optionally remove stopwords."""
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    if remove_stops:
        tokens = remove_stopwords(tokens)
    return tokens


# ── Feature Construction ──────────────────────────────────────────────────────

def get_combined_texts(df: pd.DataFrame) -> list:
    """
    For TF-IDF fitting: combine prompt + all 5 options per row
    into a single flat list of strings.
    Returns one string per cell (prompt and each option separately).
    """
    texts = []
    texts.extend(df["prompt"].tolist())
    for col in OPTION_COLS:
        texts.extend(df[col].tolist())
    return [str(t) for t in texts if pd.notna(t)]


def get_row_texts(row) -> dict:
    """
    For a single row, return a dict of cleaned text per field.
    Keys: 'prompt', 'A', 'B', 'C', 'D', 'E'
    """
    return {
        "prompt": clean_text(str(row["prompt"])),
        "A": clean_text(str(row["A"])),
        "B": clean_text(str(row["B"])),
        "C": clean_text(str(row["C"])),
        "D": clean_text(str(row["D"])),
        "E": clean_text(str(row["E"])),
    }


# ── Missing Data Handling ─────────────────────────────────────────────────────

def check_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary of missing values per column."""
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    summary = pd.DataFrame({"missing_count": missing, "missing_pct": pct})
    return summary[summary["missing_count"] > 0]


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing option text with empty string."""
    df = df.copy()
    for col in OPTION_COLS + ["prompt"]:
        if col in df.columns:
            df[col] = df[col].fillna("")
    return df


# ── Run standalone to verify ──────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data("data/raw/train.csv")

    print("\n── Missing values ──")
    missing = check_missing(df)
    print(missing if len(missing) > 0 else "None")

    print("\n── Sample clean_text ──")
    sample = df["prompt"].iloc[0]
    print(f"Original : {sample[:80]}...")
    print(f"Cleaned  : {clean_text(sample)[:80]}...")

    print("\n── Sample tokenize + stopword removal ──")
    tokens = clean_and_tokenize(sample, remove_stops=False)
    tokens_no_stop = clean_and_tokenize(sample, remove_stops=True)
    print(f"Tokens (raw)     : {len(tokens)}")
    print(f"Tokens (no stop) : {len(tokens_no_stop)}")