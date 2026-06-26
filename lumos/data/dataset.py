import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


_DEFAULT_ACTIVITY_COLS = [
    "session_count",
    "total_duration_min",
    "actions_count",
    "unique_items",
    "purchase_count",
    "purchase_amount",
    "search_count",
    "click_count",
    "view_count",
    "share_count",
    "feedback_count",
    "login_count",
    "notification_interactions",
]

_DEFAULT_STATIC_COLS = [
    "tenure_days",
    "age_bucket",
    "gender_encoded",
    "region_encoded",
    "device_type",
    "signup_channel",
    "has_premium",
    "account_tier",
    "referral_source",
]

_DEFAULT_EVENT_CONTEXT_COLS = [f"event_context_{i}" for i in range(32)]

_DEFAULT_TARGET_COLS = [
    "target_is_churned",
    "target_will_transact",
    "target_engagement_score",
    "target_revenue",
    "target_sessions_next_period",
]


class UserSequenceDataset(Dataset):
    """PyTorch Dataset that constructs user activity sequences from a flat DataFrame.

    Expects a DataFrame (or parquet path) with columns: user_id, day, activity features,
    static features, event context features, and target columns.

    Each sample produces:
        - activity_history: [lookback, n_activity_features]
        - event_context_history: [lookback, n_event_context_features]
        - static_features: [n_static_features]
        - future_event_context: [lookahead, n_event_context_features]
        - targets: [n_targets]
    """

    def __init__(
        self,
        data: str | pd.DataFrame,
        lookback: int = 30,
        lookahead: int = 7,
        activity_cols: list[str] | None = None,
        static_cols: list[str] | None = None,
        event_context_cols: list[str] | None = None,
        target_cols: list[str] | None = None,
        user_col: str = "user_id",
        day_col: str = "day",
    ):
        if isinstance(data, str):
            data = pd.read_parquet(data)
        self.df = data.sort_values([user_col, day_col]).reset_index(drop=True)

        self.lookback = lookback
        self.lookahead = lookahead
        self.activity_cols = activity_cols or _DEFAULT_ACTIVITY_COLS
        self.static_cols = static_cols or _DEFAULT_STATIC_COLS
        self.event_context_cols = event_context_cols or _DEFAULT_EVENT_CONTEXT_COLS
        self.target_cols = target_cols or _DEFAULT_TARGET_COLS
        self.user_col = user_col
        self.day_col = day_col

        self._build_index()

    def _build_index(self):
        self.samples = []
        for uid, group in self.df.groupby(self.user_col):
            group = group.sort_values(self.day_col).reset_index(drop=True)
            n_days = len(group)
            min_required = self.lookback + self.lookahead
            if n_days >= min_required:
                for start in range(n_days - min_required + 1):
                    self.samples.append((uid, group, start))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        uid, group, start = self.samples[idx]
        hist_end = start + self.lookback
        future_end = hist_end + self.lookahead

        hist_slice = group.iloc[start:hist_end]
        future_slice = group.iloc[hist_end:future_end]

        activity_history = torch.tensor(hist_slice[self.activity_cols].values, dtype=torch.float32)
        event_context_history = torch.tensor(hist_slice[self.event_context_cols].values, dtype=torch.float32)
        static_features = torch.tensor(hist_slice[self.static_cols].iloc[0].values, dtype=torch.float32)
        future_event_context = torch.tensor(future_slice[self.event_context_cols].values, dtype=torch.float32)

        targets_row = future_slice[self.target_cols].iloc[-1]
        targets = torch.tensor(targets_row.values, dtype=torch.float32)

        return {
            "activity_history": activity_history,
            "event_context_history": event_context_history,
            "static_features": static_features,
            "future_event_context": future_event_context,
            "targets": targets,
        }
