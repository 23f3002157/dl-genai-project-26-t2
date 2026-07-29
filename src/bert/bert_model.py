"""
bert_model.py
-------------
BERT model wrapper for MCQ 5-class sequence classification.
Loads bert-base-uncased with a classification head on top of [CLS] token.

Architecture:
    BERT encoder → [CLS] representation → dropout → Linear(768, 5)

Location: src/bert/bert_model.py
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class BERTMCQClassifier(nn.Module):
    """
    Wraps bert-base-uncased for 5-class MCQ classification.
    Uses [CLS] token representation as the sequence embedding.
    Classification head: Linear(hidden_size, 5).
    """

    def __init__(
        self,
        model_name : str   = "bert-base-uncased",
        num_labels : int   = 5,
        dropout    : float = 0.1,
    ):
        super().__init__()
        self.bert       = AutoModel.from_pretrained(model_name)
        hidden_size     = self.bert.config.hidden_size   # 768 for bert-base
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
        )
        # [CLS] token is always at position 0
        cls_output = outputs.last_hidden_state[:, 0, :]   # (B, 768)
        cls_output = self.dropout(cls_output)
        logits     = self.classifier(cls_output)           # (B, 5)
        return logits