from typing import Literal

import torch
import torch.nn as nn

from lumos.model.embeddings import ConfigurableEmbedding
from lumos.model.layers import TransformerEncoder, TransformerDecoder
from lumos.model.positional import NoEmbedding, get_positional_embedding


class LUMOS(nn.Module):
    """Large User MOdel Series — cross-attention transformer for multi-task user behavior prediction.

    Architecture:
        Encoder processes historical user activity sequences concatenated with static user
        features and event context. Decoder cross-attends from future event context queries
        to the encoded user history, producing per-timestep or aggregated predictions.
    """

    def __init__(
        self,
        dim_activity_raw: int,
        dim_static_raw: int,
        dim_event_context_raw: int,
        dim_activity_embed: int,
        dim_static_embed: int,
        dim_event_context_embed: int,
        d_model: int,
        n_heads: int,
        dim_ff: int,
        n_enc_layers: int,
        n_dec_layers: int,
        seq_len_hist: int,
        prediction_mode: Literal["sequence", "aggregated"],
        seq_len_future: int,
        n_targets: int = 1,
        position_embedding_type: Literal["none", "sinusoidal", "learned", "tape", "abs"] = "none",
        embedding_depth: int = 1,
        token_to_encoder_depth: int = 1,
        output_depth: int = 1,
    ):
        super().__init__()

        self.seq_len_hist = seq_len_hist
        self.prediction_mode = prediction_mode
        self.seq_len_future = seq_len_future
        self.dim_activity_raw = dim_activity_raw
        self.dim_static_raw = dim_static_raw
        self.dim_event_context_raw = dim_event_context_raw
        self.dim_activity_embed = dim_activity_embed
        self.dim_static_embed = dim_static_embed
        self.dim_event_context_embed = dim_event_context_embed
        self.d_model = d_model
        self.n_targets = n_targets
        self.position_embedding_type = position_embedding_type
        self.embedding_depth = embedding_depth
        self.token_to_encoder_depth = token_to_encoder_depth
        self.output_depth = output_depth

        hidden = lambda d, depth: [d] * (depth - 1) if depth > 1 else []
        self.activity_embed = ConfigurableEmbedding(dim_activity_raw, hidden(dim_activity_embed, embedding_depth), dim_activity_embed)
        self.static_embed = ConfigurableEmbedding(dim_static_raw, hidden(dim_static_embed, embedding_depth), dim_static_embed)
        self.event_context_embed = ConfigurableEmbedding(dim_event_context_raw, hidden(dim_event_context_embed, embedding_depth), dim_event_context_embed)

        dim_context_in = dim_activity_embed + dim_static_embed + dim_event_context_embed

        hist_proj_layers = [nn.Linear(dim_context_in, d_model), nn.ReLU()]
        for _ in range(token_to_encoder_depth - 1):
            hist_proj_layers.extend([nn.Linear(d_model, d_model), nn.ReLU()])
        self.hist_proj = nn.Sequential(*hist_proj_layers)

        total_seq_len = seq_len_hist + seq_len_future
        self.pos_embed = get_positional_embedding(position_embedding_type, d_model, total_seq_len)

        self.encoder = TransformerEncoder(d_model, n_heads, dim_ff, n_enc_layers)
        self.decoder = TransformerDecoder(d_model, n_heads, dim_ff, n_dec_layers)

        future_proj_layers = [nn.Linear(dim_event_context_embed, d_model), nn.ReLU()]
        for _ in range(token_to_encoder_depth - 1):
            future_proj_layers.extend([nn.Linear(d_model, d_model), nn.ReLU()])
        self.future_proj = nn.Sequential(*future_proj_layers)

        if prediction_mode == "aggregated":
            out_input_dim = d_model * seq_len_future
        elif prediction_mode == "sequence":
            out_input_dim = d_model
        else:
            raise ValueError(f"Invalid prediction mode: {prediction_mode}")

        out_layers = [nn.Linear(out_input_dim, d_model), nn.ReLU()]
        for _ in range(output_depth - 1):
            out_layers.extend([nn.Linear(d_model, d_model), nn.ReLU()])
        out_layers.append(nn.Linear(d_model, n_targets))
        self.out = nn.Sequential(*out_layers)

    @classmethod
    def from_paper_config(cls) -> "LUMOS":
        """Instantiate LUMOS with the configuration from the paper (arXiv:2512.08957)."""
        return cls(
            dim_activity_raw=13,
            dim_static_raw=9,
            dim_event_context_raw=32,
            dim_activity_embed=128,
            dim_static_embed=64,
            dim_event_context_embed=128,
            d_model=512,
            n_heads=8,
            dim_ff=1024,
            n_enc_layers=4,
            n_dec_layers=2,
            seq_len_hist=360,
            seq_len_future=7,
            prediction_mode="aggregated",
            n_targets=5,
            position_embedding_type="sinusoidal",
            embedding_depth=1,
            token_to_encoder_depth=1,
            output_depth=1,
        )

    def forward(
        self,
        activity_history: torch.Tensor,
        event_context_history: torch.Tensor,
        static_features: torch.Tensor,
        future_event_context: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            activity_history: [B, seq_len_hist, dim_activity_raw]
            event_context_history: [B, seq_len_hist, dim_event_context_raw]
            static_features: [B, dim_static_raw]
            future_event_context: [B, seq_len_future, dim_event_context_raw]

        Returns:
            Predictions with shape [B, n_targets] (aggregated) or [B, seq_len_future, n_targets] (sequence).
        """
        B, T_hist, _ = activity_history.shape
        _, T_future, _ = future_event_context.shape

        ah = self.activity_embed(activity_history)
        eh = self.event_context_embed(event_context_history)
        st = self.static_embed(static_features.unsqueeze(1)).repeat(1, T_hist, 1)

        hist_concat = torch.cat([ah, st, eh], dim=-1)
        hist_enc_in = self.hist_proj(hist_concat)

        if isinstance(self.pos_embed, NoEmbedding):
            hist_enc_in_with_pos = hist_enc_in
        else:
            hist_pos_embed = self.pos_embed.get_embedding(0, T_hist)
            hist_enc_in_with_pos = hist_enc_in + hist_pos_embed

        enc_out = self.encoder(hist_enc_in_with_pos)

        fs = self.event_context_embed(future_event_context)
        dec_in = self.future_proj(fs)

        if isinstance(self.pos_embed, NoEmbedding):
            dec_in_with_pos = dec_in
        else:
            future_pos_embed = self.pos_embed.get_embedding(T_hist, T_future)
            dec_in_with_pos = dec_in + future_pos_embed

        dec_out = self.decoder(x=dec_in_with_pos, enc_out=enc_out)

        if self.prediction_mode == "aggregated":
            dec_out = dec_out.reshape(B, -1)

        return self.out(dec_out)

    def event_context_embedding(self, event_context_raw: torch.Tensor) -> torch.Tensor:
        """Extract learned event context embeddings.

        Args:
            event_context_raw: [B, dim_event_context_raw]

        Returns:
            Embeddings of shape [B, dim_event_context_embed].
        """
        return self.event_context_embed(event_context_raw)

    def historical_user_embedding(
        self,
        activity_history: torch.Tensor,
        event_context_history: torch.Tensor,
        static_features: torch.Tensor,
        reduction: Literal["mean", "max", "expavg", "last"] = "mean",
        exp_alpha: float = 1.0,
        embedding_stage: Literal["post_proj", "post_encoder"] = "post_encoder",
    ) -> torch.Tensor:
        """Extract a single user-state vector per sample.

        Args:
            activity_history: [B, T, dim_activity_raw]
            event_context_history: [B, T, dim_event_context_raw]
            static_features: [B, dim_static_raw]
            reduction: Pooling strategy over the time dimension.
            exp_alpha: Temperature for exponential-average reduction.
            embedding_stage: Whether to extract after projection or after the encoder.

        Returns:
            User embedding of shape [B, d_model].
        """
        B, T, D = activity_history.shape
        ah = self.activity_embed(activity_history)
        eh = self.event_context_embed(event_context_history)
        st = self.static_embed(static_features.unsqueeze(1)).repeat(1, T, 1)
        hist_concat = torch.cat([ah, st, eh], dim=-1)
        hist_enc_in = self.hist_proj(hist_concat)

        to_reduce = hist_enc_in
        if embedding_stage == "post_encoder":
            if not isinstance(self.pos_embed, NoEmbedding):
                hist_pos_embed = self.pos_embed.get_embedding(0, T)
                to_reduce = hist_enc_in + hist_pos_embed.to(hist_enc_in.device)
            to_reduce = self.encoder(to_reduce)
        elif embedding_stage != "post_proj":
            raise ValueError(f"Unknown embedding_stage '{embedding_stage}'")

        if reduction == "mean":
            return to_reduce.mean(dim=1)
        elif reduction == "max":
            return to_reduce.max(dim=1).values
        elif reduction == "expavg":
            t = torch.linspace(0, 1, T, dtype=torch.float32, device=to_reduce.device)
            weights = torch.softmax(exp_alpha * t, dim=0).unsqueeze(0).unsqueeze(-1)
            return (to_reduce * weights).sum(dim=1)
        elif reduction == "last":
            return to_reduce[:, -1, :]
        else:
            raise ValueError(f"Unknown reduction '{reduction}'")
