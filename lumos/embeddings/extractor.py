from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader

from lumos.model.lumos import LUMOS


class UserEmbeddingExtractor:
    """Extract user embeddings from a trained LUMOS model.

    Provides methods for batch extraction, similarity computation, and visualization.
    """

    def __init__(self, model: LUMOS, device: str | torch.device | None = None):
        self.model = model
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def extract_user_embeddings(
        self,
        activity_history: torch.Tensor,
        event_context_history: torch.Tensor,
        static_features: torch.Tensor,
        reduction: Literal["mean", "max", "expavg", "last"] = "mean",
        exp_alpha: float = 1.0,
        embedding_stage: Literal["post_proj", "post_encoder"] = "post_encoder",
    ) -> torch.Tensor:
        """Extract user embeddings for a batch of users.

        Args:
            activity_history: [B, T, dim_activity_raw]
            event_context_history: [B, T, dim_event_context_raw]
            static_features: [B, dim_static_raw]
            reduction: Pooling strategy over time.
            exp_alpha: Temperature for exponential-average reduction.
            embedding_stage: "post_proj" or "post_encoder".

        Returns:
            User embeddings of shape [B, d_model].
        """
        return self.model.historical_user_embedding(
            activity_history.to(self.device),
            event_context_history.to(self.device),
            static_features.to(self.device),
            reduction=reduction,
            exp_alpha=exp_alpha,
            embedding_stage=embedding_stage,
        )

    @torch.no_grad()
    def extract_from_dataloader(
        self,
        dataloader: DataLoader,
        reduction: Literal["mean", "max", "expavg", "last"] = "mean",
        embedding_stage: Literal["post_proj", "post_encoder"] = "post_encoder",
    ) -> np.ndarray:
        """Extract embeddings for all samples in a DataLoader.

        Args:
            dataloader: A DataLoader yielding dicts with keys
                "activity_history", "event_context_history", "static_features".
            reduction: Pooling strategy.
            embedding_stage: "post_proj" or "post_encoder".

        Returns:
            Numpy array of shape [N, d_model].
        """
        all_embeddings = []
        for batch in dataloader:
            emb = self.extract_user_embeddings(
                batch["activity_history"],
                batch["event_context_history"],
                batch["static_features"],
                reduction=reduction,
                embedding_stage=embedding_stage,
            )
            all_embeddings.append(emb.cpu().numpy())
        return np.concatenate(all_embeddings, axis=0)

    @staticmethod
    def cosine_similarity(embeddings: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine similarity matrix.

        Args:
            embeddings: Array of shape [N, D].

        Returns:
            Similarity matrix of shape [N, N].
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = embeddings / norms
        return normalized @ normalized.T

    @staticmethod
    def find_similar_users(embeddings: np.ndarray, query_idx: int, top_k: int = 10) -> list[tuple[int, float]]:
        """Find the most similar users to a query user.

        Args:
            embeddings: Array of shape [N, D].
            query_idx: Index of the query user.
            top_k: Number of similar users to return.

        Returns:
            List of (user_index, similarity_score) tuples, sorted by descending similarity.
        """
        query = embeddings[query_idx : query_idx + 1]
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        query_norm = np.maximum(np.linalg.norm(query, keepdims=True), 1e-8)

        similarities = ((embeddings / norms) @ (query / query_norm).T).squeeze()
        top_indices = np.argsort(similarities)[::-1][1 : top_k + 1]
        return [(int(idx), float(similarities[idx])) for idx in top_indices]
