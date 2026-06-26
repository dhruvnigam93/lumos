import math
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):

    def __init__(self, d_q: int, d_kv: int, d_attn: int, n_heads: int):
        super().__init__()
        assert d_attn % n_heads == 0, "d_attn must be divisible by n_heads"
        self.d_attn = d_attn
        self.n_heads = n_heads
        self.d_k = d_attn // n_heads

        self.W_q = nn.Linear(d_q, d_attn)
        self.W_k = nn.Linear(d_kv, d_attn)
        self.W_v = nn.Linear(d_kv, d_attn)
        self.W_o = nn.Linear(d_attn, d_attn)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask=None):
        B, len_q, _ = Q.shape
        _, len_kv, _ = K.shape

        q_proj = self.W_q(Q).reshape(B, len_q, self.n_heads, self.d_k).transpose(1, 2)
        k_proj = self.W_k(K).reshape(B, len_kv, self.n_heads, self.d_k).transpose(1, 2)
        v_proj = self.W_v(V).reshape(B, len_kv, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)

        out = torch.matmul(attn_weights, v_proj)
        out = out.transpose(1, 2).contiguous().reshape(B, len_q, self.d_attn)
        out = self.W_o(out)

        return out, attn_weights
