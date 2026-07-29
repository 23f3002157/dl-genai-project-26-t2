"""
gru_model.py
------------
GRU-based MCQ scorer built from scratch in PyTorch.
No pretrained weights — vocabulary and embeddings learned from data.

Architecture:
    1. Embedding layer (vocab built from train data)
    2. Bidirectional GRU
    3. Multi-head attention pooling over GRU outputs
    4. Residual scoring head: outputs a scalar score per option
    5. Top-3 options ranked by score → MAP@3

Differences from LSTM:
    - GRU has 2 gates (reset, update) vs LSTM's 3 (input, forget, output)
    - Fewer parameters → less overfitting on small datasets
    - Faster to train, often matches or beats LSTM on short sequences
    - Multi-head attention pooling instead of single-head

Location: src/gru/gru_model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads  = num_heads
        self.attn_heads = nn.ModuleList([
            nn.Linear(hidden_dim * 2, 1) for _ in range(num_heads)
        ])
        self.proj = nn.Linear(hidden_dim * 2 * num_heads, hidden_dim * 2)

    def forward(self, gru_out):
        # gru_out: (batch, seq_len, hidden*2)
        head_contexts = []
        for attn in self.attn_heads:
            weights = torch.softmax(attn(gru_out), dim=1)   # (B, seq, 1)
            context = (weights * gru_out).sum(dim=1)         # (B, hidden*2)
            head_contexts.append(context)
        concat  = torch.cat(head_contexts, dim=-1)           # (B, hidden*2*heads)
        out     = self.proj(concat)                          # (B, hidden*2)
        return out


class GRUScorer(nn.Module):
    """
    Scores a single (prompt + option) concatenated string.
    Called per option — 5 times per question.
    """
    def __init__(
        self,
        vocab_size : int,
        embed_dim  : int   = 192,
        hidden_dim : int   = 384,
        num_layers : int   = 3,
        num_heads  : int   = 4,
        dropout    : float = 0.35,
        pad_idx    : int   = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.embed_drop = nn.Dropout(dropout)

        self.gru = nn.GRU(
            input_size    = embed_dim,
            hidden_size   = hidden_dim,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if num_layers > 1 else 0.0,
        )
        self.attention = MultiHeadAttentionPooling(hidden_dim, num_heads)
        self.dropout   = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
        )

    def forward(self, input_ids):
        # input_ids: (batch, seq_len)
        x           = self.embed_drop(self.embedding(input_ids))
        gru_out, _  = self.gru(x)
        context     = self.attention(gru_out)
        context     = self.layer_norm(context)
        context     = self.dropout(context)
        score       = self.scorer(context).squeeze(-1)   # (batch,)
        return score


class GRUMCQModel(nn.Module):
    """
    Wraps GRUScorer to handle all 5 options per question.
    Input : (batch, num_options, seq_len)
    Output: (batch, num_options) logits
    """
    def __init__(
        self,
        vocab_size : int,
        embed_dim  : int   = 192,
        hidden_dim : int   = 384,
        num_layers : int   = 3,
        num_heads  : int   = 4,
        dropout    : float = 0.35,
        pad_idx    : int   = 0,
    ):
        super().__init__()
        self.scorer = GRUScorer(
            vocab_size = vocab_size,
            embed_dim  = embed_dim,
            hidden_dim = hidden_dim,
            num_layers = num_layers,
            num_heads  = num_heads,
            dropout    = dropout,
            pad_idx    = pad_idx,
        )

    def forward(self, input_ids):
        # input_ids: (batch, 5, seq_len)
        batch_size, num_options, seq_len = input_ids.shape
        flat   = input_ids.view(batch_size * num_options, seq_len)
        scores = self.scorer(flat)                             # (batch*5,)
        logits = scores.view(batch_size, num_options)          # (batch, 5)
        return logits