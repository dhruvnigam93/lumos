import numpy as np
import pandas as pd


_ACTIVITY_FEATURES = [
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

_STATIC_FEATURES = [
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

_EVENT_CONTEXT_FEATURES = [f"event_context_{i}" for i in range(32)]

_TARGET_NAMES = [
    "target_is_churned",
    "target_will_transact",
    "target_engagement_score",
    "target_revenue",
    "target_sessions_next_period",
]

_USER_ARCHETYPES = {
    "power_user": {"activity_scale": 3.0, "churn_prob": 0.05, "transact_prob": 0.9},
    "regular": {"activity_scale": 1.0, "churn_prob": 0.15, "transact_prob": 0.6},
    "casual": {"activity_scale": 0.4, "churn_prob": 0.3, "transact_prob": 0.3},
    "at_risk": {"activity_scale": 0.15, "churn_prob": 0.7, "transact_prob": 0.1},
    "new_user": {"activity_scale": 0.6, "churn_prob": 0.25, "transact_prob": 0.4},
}


def generate_synthetic_dataset(
    n_users: int = 1000,
    n_days: int = 30,
    output_path: str | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic user behavior dataset for LUMOS.

    Creates realistic-looking multi-day user activity data with five user archetypes
    (power_user, regular, casual, at_risk, new_user) to produce varied behavioral patterns.

    Args:
        n_users: Number of synthetic users.
        n_days: Number of days of history per user.
        output_path: If provided, write the DataFrame as a parquet file.
        seed: Random seed for reproducibility.

    Returns:
        A DataFrame with columns: user_id, day, activity features, static features,
        event context features, and target columns.
    """
    rng = np.random.default_rng(seed)

    archetype_names = list(_USER_ARCHETYPES.keys())
    user_archetypes = rng.choice(archetype_names, size=n_users)

    rows = []
    for uid in range(n_users):
        arch = _USER_ARCHETYPES[user_archetypes[uid]]
        scale = arch["activity_scale"]

        static = {
            "tenure_days": rng.integers(1, 1000),
            "age_bucket": rng.integers(0, 6),
            "gender_encoded": rng.integers(0, 3),
            "region_encoded": rng.integers(0, 10),
            "device_type": rng.integers(0, 3),
            "signup_channel": rng.integers(0, 5),
            "has_premium": rng.integers(0, 2),
            "account_tier": rng.integers(0, 4),
            "referral_source": rng.integers(0, 5),
        }

        for day in range(n_days):
            row = {"user_id": uid, "day": day}

            row["session_count"] = max(0, rng.poisson(2 * scale))
            row["total_duration_min"] = max(0, rng.exponential(15 * scale))
            row["actions_count"] = max(0, rng.poisson(10 * scale))
            row["unique_items"] = max(0, rng.poisson(5 * scale))
            row["purchase_count"] = max(0, rng.poisson(0.5 * scale))
            row["purchase_amount"] = max(0, rng.exponential(20 * scale))
            row["search_count"] = max(0, rng.poisson(3 * scale))
            row["click_count"] = max(0, rng.poisson(8 * scale))
            row["view_count"] = max(0, rng.poisson(12 * scale))
            row["share_count"] = max(0, rng.poisson(0.3 * scale))
            row["feedback_count"] = max(0, rng.poisson(0.2 * scale))
            row["login_count"] = max(0, rng.poisson(1.5 * scale))
            row["notification_interactions"] = max(0, rng.poisson(2 * scale))

            for k, v in static.items():
                row[k] = v

            event_ctx = rng.standard_normal(32) * 0.5 + rng.standard_normal(32) * (0.3 * (day / n_days))
            for i, val in enumerate(event_ctx):
                row[f"event_context_{i}"] = val

            row["target_is_churned"] = float(rng.random() < arch["churn_prob"])
            row["target_will_transact"] = float(rng.random() < arch["transact_prob"])
            row["target_engagement_score"] = max(0, rng.exponential(5 * scale))
            row["target_revenue"] = max(0, rng.exponential(10 * scale))
            row["target_sessions_next_period"] = max(0, rng.poisson(3 * scale))

            rows.append(row)

    df = pd.DataFrame(rows)

    if output_path:
        df.to_parquet(output_path, index=False)

    return df
