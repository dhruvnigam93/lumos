from dataclasses import dataclass, field
from typing import Literal


@dataclass
class LUMOSConfig:
    """Model architecture configuration."""

    dim_activity_raw: int
    dim_static_raw: int
    dim_event_context_raw: int
    dim_activity_embed: int = 128
    dim_static_embed: int = 64
    dim_event_context_embed: int = 128
    d_model: int = 512
    n_heads: int = 8
    dim_ff: int = 1024
    n_enc_layers: int = 4
    n_dec_layers: int = 2
    seq_len_hist: int = 360
    seq_len_future: int = 7
    prediction_mode: Literal["sequence", "aggregated"] = "aggregated"
    n_targets: int = 5
    position_embedding_type: Literal["none", "sinusoidal", "learned", "tape", "abs"] = "sinusoidal"
    embedding_depth: int = 1
    token_to_encoder_depth: int = 1
    output_depth: int = 1


@dataclass
class TaskConfig:
    """Configuration for a single prediction task."""

    name: str
    task_type: Literal["binary", "continuous"]
    loss_weight: float = 1.0


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    batch_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    adam_betas: tuple[float, float] = (0.9, 0.999)
    epochs: int = 10
    val_split: float = 0.1
    log_interval: int = 100
    checkpoint_interval: int = 1000
    loss_aggregation: Literal["sum", "uncertainty_weighted", "learned_weights"] = "sum"
    lr_scheduler: Literal["constant", "cosine", "cosine_warmup", "step"] = "constant"
    lr_scheduler_kwargs: dict = field(default_factory=dict)
    tasks: list[TaskConfig] = field(default_factory=list)
