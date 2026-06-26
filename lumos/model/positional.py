from typing import Literal

import math
import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    """Base class for positional embeddings."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self._init_embedding()

    def _init_embedding(self):
        raise NotImplementedError

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"Input sequence length ({seq_len}) exceeds maximum ({self.max_len})")
        return x + self.pe[:, :seq_len]

    def get_embedding(self, start_pos: int, length: int):
        if start_pos + length > self.max_len:
            raise ValueError(f"Positions {start_pos}..{start_pos + length - 1} exceed max length {self.max_len}")
        return self.pe[:, start_pos : start_pos + length]


class SinusoidalEmbedding(PositionalEmbedding):
    """Fixed sinusoidal embeddings (Vaswani et al., 2017)."""

    def _init_embedding(self):
        if self.d_model % 2 != 0:
            raise ValueError(f"Cannot use sinusoidal embedding with odd d_model ({self.d_model})")

        pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))


class TapeEmbedding(PositionalEmbedding):
    """Fixed sinusoidal embeddings scaled by dimension and length."""

    def _init_embedding(self):
        if self.d_model % 2 != 0:
            raise ValueError(f"Cannot use TAPE embedding with odd d_model ({self.d_model})")

        pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = (
            torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
            * self.d_model
            / self.max_len
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))


class LearnedEmbedding(PositionalEmbedding):
    """Learned positional embeddings."""

    def _init_embedding(self):
        self.pe = nn.Parameter(torch.randn(1, self.max_len, self.d_model) * 0.02)


class AbsEmbedding(PositionalEmbedding):
    """Simple linear absolute position embedding scaled by max_len."""

    def _init_embedding(self):
        pe = torch.zeros(self.max_len, self.d_model)
        position_scaled = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1) / self.max_len
        pe = pe + position_scaled
        self.register_buffer("pe", pe.unsqueeze(0))


class NoEmbedding(nn.Module):
    """Identity layer — no positional embedding."""

    def __init__(self, d_model: int = None, max_len: int = None):
        super().__init__()

    def forward(self, x):
        return x

    def get_embedding(self, start_pos: int, length: int):
        return 0.0


def get_positional_embedding(
    position_embedding_type: Literal["none", "sinusoidal", "learned", "tape", "abs"],
    d_model: int,
    max_len: int = 5000,
) -> nn.Module:
    """Factory function to create the specified positional embedding."""
    embedding_map = {
        "none": lambda: NoEmbedding(),
        "sinusoidal": lambda: SinusoidalEmbedding(d_model, max_len),
        "learned": lambda: LearnedEmbedding(d_model, max_len),
        "tape": lambda: TapeEmbedding(d_model, max_len),
        "abs": lambda: AbsEmbedding(d_model, max_len),
    }
    if position_embedding_type not in embedding_map:
        raise ValueError(f"Unknown position_embedding_type: {position_embedding_type}")
    return embedding_map[position_embedding_type]()
