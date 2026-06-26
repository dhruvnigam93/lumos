"""LUMOS Quickstart — instantiate the model and run a forward pass."""

import torch
from lumos.model import LUMOS

model = LUMOS.from_paper_config()
print(f"LUMOS model: {sum(p.numel() for p in model.parameters()):,} parameters")

batch_size = 4
activity_history = torch.randn(batch_size, 360, 13)
event_context_history = torch.randn(batch_size, 360, 32)
static_features = torch.randn(batch_size, 9)
future_event_context = torch.randn(batch_size, 7, 32)

predictions = model(activity_history, event_context_history, static_features, future_event_context)
print(f"Predictions shape: {predictions.shape}")  # [4, 5]

user_embeddings = model.historical_user_embedding(
    activity_history, event_context_history, static_features, reduction="mean"
)
print(f"User embeddings shape: {user_embeddings.shape}")  # [4, 512]

print("Quickstart complete.")
