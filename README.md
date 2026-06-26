# LUMOS: Large User MOdel Series

A PyTorch framework for training transformer-based user behavior models that learn unified user representations from sequential activity data.

**Paper:** [LUMOS: Large User MOdel Series for Scalable User Representation Learning](https://arxiv.org/abs/2512.08957)

## Architecture

LUMOS uses a cross-attention encoder-decoder transformer:

- **Encoder** processes historical user activity sequences concatenated with static user features and contextual event information
- **Decoder** cross-attends from future event context queries to the encoded user history
- Supports both **sequence-level** and **aggregated** prediction modes
- Multi-task learning with configurable loss weighting (sum, uncertainty-weighted, learned weights)

The trained encoder produces dense user embeddings that transfer to downstream tasks like churn prediction, engagement scoring, and user segmentation.

## Installation

```bash
pip install git+https://github.com/dhruvnigam93/lumos-user-model.git
```

With training extras (TensorBoard, system monitoring):

```bash
pip install "lumos-user-model[training] @ git+https://github.com/dhruvnigam93/lumos-user-model.git"
```

## Quickstart

```python
import torch
from lumos.model import LUMOS

# Instantiate with paper configuration
model = LUMOS.from_paper_config()

# Forward pass
predictions = model(
    activity_history=torch.randn(4, 360, 13),      # [batch, seq_len, activity_features]
    event_context_history=torch.randn(4, 360, 32),  # [batch, seq_len, context_features]
    static_features=torch.randn(4, 9),              # [batch, static_features]
    future_event_context=torch.randn(4, 7, 32),     # [batch, future_len, context_features]
)
# predictions.shape: [4, 5]

# Extract user embeddings
embeddings = model.historical_user_embedding(
    activity_history=torch.randn(4, 360, 13),
    event_context_history=torch.randn(4, 360, 32),
    static_features=torch.randn(4, 9),
    reduction="mean",  # or "max", "expavg", "last"
)
# embeddings.shape: [4, 512]
```

## Training on Synthetic Data

```python
from torch.utils.data import DataLoader
from lumos.model import LUMOS
from lumos.data import generate_synthetic_dataset, UserSequenceDataset
from lumos.training import LUMOSTrainer

# Generate data
df = generate_synthetic_dataset(n_users=1000, n_days=50)
dataset = UserSequenceDataset(df, lookback=30, lookahead=7)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Build model
model = LUMOS(
    dim_activity_raw=13, dim_static_raw=9, dim_event_context_raw=32,
    dim_activity_embed=128, dim_static_embed=64, dim_event_context_embed=128,
    d_model=512, n_heads=8, dim_ff=1024,
    n_enc_layers=4, n_dec_layers=2,
    seq_len_hist=30, seq_len_future=7,
    prediction_mode="aggregated", n_targets=5,
    position_embedding_type="sinusoidal",
)

# Train
trainer = LUMOSTrainer(
    model=model,
    train_loader=train_loader,
    target_types=["binary", "binary", "continuous", "continuous", "continuous"],
    learning_rate=1e-4,
    loss_aggregation="uncertainty_weighted",
)
history = trainer.train(epochs=10)
```

## Extracting User Embeddings

```python
from lumos.embeddings import UserEmbeddingExtractor

extractor = UserEmbeddingExtractor(model)
embeddings = extractor.extract_from_dataloader(dataloader, reduction="mean")

# Find similar users
similar = UserEmbeddingExtractor.find_similar_users(embeddings, query_idx=0, top_k=10)
```

## Project Structure

```
lumos/
  model/         # LUMOS transformer architecture
  training/      # Trainer, losses, schedulers, monitoring
  data/          # Dataset, synthetic data generator, preprocessing
  embeddings/    # User embedding extraction and similarity
  evaluation/    # Binary and continuous evaluation metrics
  config.py      # Configuration dataclasses
examples/        # Quickstart, training, and embedding extraction scripts
tests/           # Comprehensive test suite
```

## Citation

```bibtex
@article{nigam2024lumos,
  title={LUMOS: Large User MOdel Series for Scalable User Representation Learning},
  author={Nigam, Dhruv and Saha, Susmit and Tatte, Palash},
  journal={arXiv preprint arXiv:2512.08957},
  year={2024}
}
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
