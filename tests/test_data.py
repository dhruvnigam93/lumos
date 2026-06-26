import tempfile
import os

import numpy as np
import torch

from lumos.data.synthetic import generate_synthetic_dataset
from lumos.data.dataset import UserSequenceDataset
from lumos.data.preprocessing import MinMaxScaler, LogMinMaxScaler, build_sequences


class TestSyntheticData:
    def test_basic_generation(self):
        df = generate_synthetic_dataset(n_users=10, n_days=5)
        assert len(df) == 50
        assert "user_id" in df.columns
        assert "day" in df.columns
        assert "session_count" in df.columns
        assert "target_is_churned" in df.columns

    def test_parquet_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.parquet")
            df = generate_synthetic_dataset(n_users=5, n_days=3, output_path=path)
            assert os.path.exists(path)
            assert len(df) == 15

    def test_reproducibility(self):
        df1 = generate_synthetic_dataset(n_users=10, n_days=5, seed=42)
        df2 = generate_synthetic_dataset(n_users=10, n_days=5, seed=42)
        assert df1.equals(df2)


class TestUserSequenceDataset:
    def test_basic(self):
        df = generate_synthetic_dataset(n_users=10, n_days=20)
        ds = UserSequenceDataset(df, lookback=10, lookahead=3)
        assert len(ds) > 0
        sample = ds[0]
        assert sample["activity_history"].shape == (10, 13)
        assert sample["event_context_history"].shape == (10, 32)
        assert sample["static_features"].shape == (9,)
        assert sample["future_event_context"].shape == (3, 32)
        assert sample["targets"].shape == (5,)

    def test_from_parquet(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.parquet")
            generate_synthetic_dataset(n_users=10, n_days=20, output_path=path)
            ds = UserSequenceDataset(path, lookback=10, lookahead=3)
            assert len(ds) > 0

    def test_all_tensors_finite(self):
        df = generate_synthetic_dataset(n_users=5, n_days=20)
        ds = UserSequenceDataset(df, lookback=10, lookahead=3)
        sample = ds[0]
        for key, tensor in sample.items():
            assert torch.isfinite(tensor).all(), f"Non-finite values in {key}"


class TestScalers:
    def test_minmax(self):
        data = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(data)
        assert scaled.min() >= 0
        assert scaled.max() <= 1

    def test_minmax_serialization(self):
        data = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = MinMaxScaler().fit(data)
        params = scaler.get_params()
        restored = MinMaxScaler.from_params(params)
        assert np.allclose(scaler.transform(data), restored.transform(data))

    def test_logminmax(self):
        data = np.array([[1, 100], [10, 1000], [100, 10000]])
        scaler = LogMinMaxScaler()
        scaled = scaler.fit_transform(data)
        assert scaled.min() >= 0
        assert scaled.max() <= 1

    def test_build_sequences(self):
        df = generate_synthetic_dataset(n_users=5, n_days=10)
        sequences = build_sequences(df, "user_id", "day", ["session_count", "click_count"], lookback=5)
        assert len(sequences) == 5
        for uid, seq in sequences.items():
            assert seq.shape == (5, 2)
