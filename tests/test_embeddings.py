import numpy as np
import torch

from lumos.model import LUMOS
from lumos.embeddings import UserEmbeddingExtractor


class TestUserEmbeddingExtractor:
    def setup_method(self):
        self.model = LUMOS(
            dim_activity_raw=8,
            dim_static_raw=4,
            dim_event_context_raw=16,
            dim_activity_embed=32,
            dim_static_embed=16,
            dim_event_context_embed=32,
            d_model=64,
            n_heads=4,
            dim_ff=128,
            n_enc_layers=2,
            n_dec_layers=1,
            seq_len_hist=20,
            seq_len_future=5,
            prediction_mode="aggregated",
            n_targets=3,
            position_embedding_type="sinusoidal",
        )
        self.extractor = UserEmbeddingExtractor(self.model)

    def test_extract_embeddings(self):
        emb = self.extractor.extract_user_embeddings(
            torch.randn(4, 20, 8),
            torch.randn(4, 20, 16),
            torch.randn(4, 4),
        )
        assert emb.shape == (4, 64)

    def test_cosine_similarity(self):
        embeddings = np.random.randn(10, 64).astype(np.float32)
        sim = UserEmbeddingExtractor.cosine_similarity(embeddings)
        assert sim.shape == (10, 10)
        np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-5)

    def test_find_similar_users(self):
        embeddings = np.random.randn(20, 64).astype(np.float32)
        similar = UserEmbeddingExtractor.find_similar_users(embeddings, query_idx=0, top_k=5)
        assert len(similar) == 5
        assert all(idx != 0 for idx, _ in similar)
        assert all(-1 <= score <= 1 for _, score in similar)
