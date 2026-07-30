"""
deberta_model.py
----------------
DeBERTa model wrapper for MCQ 5-class sequence classification.
Uses [CLS] token representation → dropout → Linear(hidden_size, 5).

Key difference from BERT:
    DeBERTa uses disentangled attention (separate content + position embeddings)
    which gives it stronger contextual understanding, especially for long inputs.
    hidden_size = 768 for deberta-v3-base.

Location: src/deberta/deberta_model.py
"""

import torch.nn as nn
from transformers import AutoModel


class DeBERTaMCQClassifier(nn.Module):
    def __init__(
        self,
        model_name : str   = "microsoft/deberta-v3-base",
        num_labels : int   = 5,
        dropout    : float = 0.1,
    ):
        super().__init__()
        self.deberta    = AutoModel.from_pretrained(model_name)
        hidden_size     = self.deberta.config.hidden_size
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs    = self.deberta(
            input_ids      = input_ids,
            attention_mask = attention_mask,
        )
        # DeBERTa returns last_hidden_state — take [CLS] at position 0
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits     = self.classifier(cls_output)
        return logits
