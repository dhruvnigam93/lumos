import torch.nn as nn

from lumos.model.attention import MultiHeadAttention


class EncoderLayer(nn.Module):

    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_q=d_model, d_kv=d_model, d_attn=d_model, n_heads=n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(inplace=True),
            nn.Linear(dim_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.self_attn(x, x, x, mask=mask)
        x = self.norm1(x + self.dropout1(attn_out))

        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_out))

        return x


class DecoderLayer(nn.Module):

    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = MultiHeadAttention(d_q=d_model, d_kv=d_model, d_attn=d_model, n_heads=n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(inplace=True),
            nn.Linear(dim_ff, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_out, self_mask=None, cross_mask=None):
        attn_out_cross, _ = self.cross_attn(Q=x, K=enc_out, V=enc_out, mask=cross_mask)
        x = self.norm2(x + self.dropout2(attn_out_cross))

        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout3(ff_out))

        return x


class TransformerEncoder(nn.Module):

    def __init__(self, d_model: int, n_heads: int, dim_ff: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, dim_ff, dropout) for _ in range(n_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x


class TransformerDecoder(nn.Module):

    def __init__(self, d_model: int, n_heads: int, dim_ff: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, dim_ff, dropout) for _ in range(n_layers)])

    def forward(self, x, enc_out, self_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, enc_out, self_mask=self_mask, cross_mask=cross_mask)
        return x
