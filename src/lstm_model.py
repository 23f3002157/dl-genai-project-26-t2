"""
lstm_model.py
-------------
LSTM-based MCQ scorer built from scratch in PyTorch.
No pretrained weights — vocabulary and embeddings learned from data.

Architecture:
    1. Embedding layer (vocab built from train data)
    2. Bidirectional LSTM
    3. Attention pooling over LSTM outputs
    4. Scoring head: outputs a scalar score per option
    5. Top-3 options ranked by score → MAP@3

Location: src/lstm_model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_out):
        # lstm_out: (batch, seq_len, hidden*2)
        weights = torch.softmax(self.attn(lstm_out), dim=1)  # (batch, seq_len, 1)
        context = (weights * lstm_out).sum(dim=1)             # (batch, hidden*2)
        return context


class LSTMScorer(nn.Module):
    """
    Scores a single (prompt + option) concatenated string.
    Used per option — call 5 times per question, rank results.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int   = 128,
        hidden_dim: int  = 256,
        num_layers: int  = 2,
        dropout: float   = 0.3,
        pad_idx: int     = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm      = nn.LSTM(
            input_size    = embed_dim,
            hidden_size   = hidden_dim,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if num_layers > 1 else 0.0,
        )
        self.attention = Attention(hidden_dim)
        self.dropout   = nn.Dropout(dropout)
        self.scorer    = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids):
        # input_ids: (batch, seq_len)
        x           = self.dropout(self.embedding(input_ids))
        lstm_out, _ = self.lstm(x)
        context     = self.attention(lstm_out)
        context     = self.dropout(context)
        score       = self.scorer(context).squeeze(-1)   # (batch,)
        return score


class LSTMMCQModel(nn.Module):
    """
    Wraps LSTMScorer to handle all 5 options per question.
    Input : (batch, num_options, seq_len)
    Output: (batch, num_options) logits
    """
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256,
                 num_layers=2, dropout=0.3, pad_idx=0):
        super().__init__()
        self.scorer = LSTMScorer(
            vocab_size  = vocab_size,
            embed_dim   = embed_dim,
            hidden_dim  = hidden_dim,
            num_layers  = num_layers,
            dropout     = dropout,
            pad_idx     = pad_idx,
        )

    def forward(self, input_ids):
        # input_ids: (batch, 5, seq_len)
        batch_size, num_options, seq_len = input_ids.shape
        flat    = input_ids.view(batch_size * num_options, seq_len)
        scores  = self.scorer(flat)                          # (batch*5,)
        logits  = scores.view(batch_size, num_options)       # (batch, 5)
        return logits