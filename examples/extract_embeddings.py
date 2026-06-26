"""Extract user embeddings from a LUMOS model and compute similarity."""

import torch
from torch.utils.data import DataLoader

from lumos.model import LUMOS
from lumos.data.synthetic import generate_synthetic_dataset
from lumos.data.dataset import UserSequenceDataset
from lumos.embeddings import UserEmbeddingExtractor

print("Setting up model and data...")
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
    seq_len_hist=20,
    seq_len_future=5,
    prediction_mode="aggregated",
    n_targets=5,
    position_embedding_type="sinusoidal",
)

df = generate_synthetic_dataset(n_users=50, n_days=30, seed=42)
dataset = UserSequenceDataset(df, lookback=20, lookahead=5)
loader = DataLoader(dataset, batch_size=16)

print("Extracting embeddings...")
extractor = UserEmbeddingExtractor(model)
embeddings = extractor.extract_from_dataloader(loader, reduction="mean")
print(f"Extracted {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")

print("\nMost similar users to user 0:")
similar = UserEmbeddingExtractor.find_similar_users(embeddings, query_idx=0, top_k=5)
for idx, score in similar:
    print(f"  User {idx}: similarity = {score:.4f}")

sim_matrix = UserEmbeddingExtractor.cosine_similarity(embeddings[:10])
print(f"\nSimilarity matrix (10x10) stats: min={sim_matrix.min():.3f}, max={sim_matrix.max():.3f}")
