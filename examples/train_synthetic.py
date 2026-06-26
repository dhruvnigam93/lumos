"""Train LUMOS on synthetic data — demonstrates the full training pipeline."""

import tempfile

import torch
from torch.utils.data import DataLoader

from lumos.model import LUMOS
from lumos.data.synthetic import generate_synthetic_dataset
from lumos.data.dataset import UserSequenceDataset
from lumos.training import LUMOSTrainer

N_USERS = 200
N_DAYS = 50
LOOKBACK = 20
LOOKAHEAD = 5
BATCH_SIZE = 16
EPOCHS = 3

print("Generating synthetic data...")
df = generate_synthetic_dataset(n_users=N_USERS, n_days=N_DAYS, seed=42)
dataset = UserSequenceDataset(df, lookback=LOOKBACK, lookahead=LOOKAHEAD)
print(f"Dataset size: {len(dataset)} samples")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

model = LUMOS(
    dim_activity_raw=13,
    dim_static_raw=9,
    dim_event_context_raw=32,
    dim_activity_embed=64,
    dim_static_embed=32,
    dim_event_context_embed=64,
    d_model=128,
    n_heads=4,
    dim_ff=256,
    n_enc_layers=2,
    n_dec_layers=1,
    seq_len_hist=LOOKBACK,
    seq_len_future=LOOKAHEAD,
    prediction_mode="aggregated",
    n_targets=5,
    position_embedding_type="sinusoidal",
)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

with tempfile.TemporaryDirectory() as tmpdir:
    trainer = LUMOSTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        target_types=["binary", "binary", "continuous", "continuous", "continuous"],
        target_names=["churned", "transact", "engagement", "revenue", "sessions"],
        learning_rate=1e-3,
        loss_aggregation="sum",
        checkpoint_dir=f"{tmpdir}/checkpoints",
        log_interval=50,
    )

    print("Training...")
    history = trainer.train(epochs=EPOCHS)

    for epoch, (tl, vl) in enumerate(zip(history["train_loss"], history["val_loss"])):
        vl_str = f"{vl:.4f}" if vl is not None else "N/A"
        print(f"  Epoch {epoch}: train_loss={tl:.4f}, val_loss={vl_str}")

print("Training complete.")
