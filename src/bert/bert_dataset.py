"""
bert_dataset.py
---------------
Tokenization and Dataset classes for BERT fine-tuning on MCQ.

Input format:
    "Question: [prompt] A: [optA] B: [optB] C: [optC] D: [optD] E: [optE]"

Single sequence → 5-class classification.
This format is stable on MPS and avoids per-option batching issues.

Location: src/bert/bert_dataset.py
"""

import torch
import pandas as pd
from torch.utils.data import Dataset

OPTION_COLS  = ["A", "B", "C", "D", "E"]
LABEL_MAP    = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}


def format_input(row) -> str:
    """
    Build a single string encoding the full MCQ:
        Question: <prompt> A: <optA> B: <optB> C: <optC> D: <optD> E: <optE>

    Keeps all 5 options in one sequence so BERT's self-attention can
    compare them directly against the question in one forward pass.
    """
    prompt = str(row["prompt"])
    opts   = " ".join([f"{c}: {str(row[c])}" for c in OPTION_COLS])
    return f"Question: {prompt} {opts}"


class BERTMCQDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int, is_test: bool = False):
        self.df         = df.reset_index(drop=True)
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.is_test    = is_test
        self.texts      = [format_input(self.df.iloc[i]) for i in range(len(self.df))]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc  = self.tokenizer(
            text,
            max_length     = self.max_length,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt",
        )
        sample = {
            "input_ids"      : enc["input_ids"].squeeze(0),
            "attention_mask" : enc["attention_mask"].squeeze(0),
        }
        if "token_type_ids" in enc:
            sample["token_type_ids"] = enc["token_type_ids"].squeeze(0)

        if not self.is_test:
            sample["labels"] = torch.tensor(
                LABEL_MAP[self.df.iloc[idx]["answer"]], dtype=torch.long
            )
        return sample