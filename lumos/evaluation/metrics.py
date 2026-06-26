import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    auc,
    r2_score,
)


def compute_ece(y_true, y_pred, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) for binary classification.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred: Predicted probabilities in [0, 1].
        n_bins: Number of uniform bins.

    Returns:
        The ECE value.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_pred.min() < 0 or y_pred.max() > 1:
        raise ValueError("Predicted probabilities must be in [0, 1].")
    if len(y_true) == 0:
        raise ValueError("Empty inputs.")

    n = len(y_true)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred, bin_edges) - 1

    ece = 0.0
    for b in range(n_bins):
        mask = bin_indices == b
        count = np.sum(mask)
        if count > 0:
            ece += (count / n) * abs(np.mean(y_pred[mask]) - np.mean(y_true[mask]))
    return ece


def evaluate_binary(y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    """Comprehensive evaluation metrics for binary classification.

    Args:
        y_pred: Predicted probabilities in [0, 1].
        y_true: Ground truth binary labels (0 or 1).

    Returns:
        Dictionary containing AUC, AP, F1, TPR@k, FPR@thresholds, ECE, and more.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    n = len(y_true)
    positives = y_true.sum()

    results = {"n_samples": n, "n_positives": int(positives), "positive_ratio": positives / n if n > 0 else 0}

    try:
        results["auc"] = roc_auc_score(y_true, y_pred)
    except ValueError:
        results["auc"] = float("nan")

    try:
        results["ap"] = average_precision_score(y_true, y_pred)
    except ValueError:
        results["ap"] = float("nan")

    try:
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        precision = np.nan_to_num(precision)
        recall = np.nan_to_num(recall)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        best_idx = np.argmax(f1)
        results["best_f1"] = float(f1[best_idx])
        results["best_threshold"] = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
        results["precision_at_best_f1"] = float(precision[best_idx])
        results["recall_at_best_f1"] = float(recall[best_idx])
    except ValueError:
        results["best_f1"] = float("nan")
        results["best_threshold"] = float("nan")

    try:
        percentiles = np.argsort(np.argsort(y_pred)) / max(n - 1, 1) * 100
        for cutoff, label in [(90, "10"), (80, "20"), (50, "50")]:
            top_mask = percentiles >= cutoff
            if positives > 0:
                results[f"tpr@{label}"] = float(y_true[top_mask].sum() / positives)
            else:
                results[f"tpr@{label}"] = 0.0
    except Exception:
        for label in ["10", "20", "50"]:
            results[f"tpr@{label}"] = 0.0

    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
        above = y_pred > threshold
        false_pos = ((above) & (y_true == 0)).sum()
        total_above = above.sum()
        results[f"fpr@{threshold}"] = float(false_pos / max(total_above, 1))

    try:
        results["ece"] = compute_ece(y_true, y_pred)
    except ValueError:
        results["ece"] = float("nan")

    return results


def evaluate_continuous(y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    """Comprehensive evaluation metrics for continuous/regression targets.

    Args:
        y_pred: Predicted values.
        y_true: Ground truth values.

    Returns:
        Dictionary containing RMSE, MAE, MAPE, R2, within-k%, error percentiles, etc.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    errors = y_true - y_pred
    abs_errors = np.abs(errors)

    non_zero = y_true != 0
    pct_errors = np.zeros_like(errors)
    if non_zero.any():
        pct_errors[non_zero] = (errors[non_zero] / y_true[non_zero]) * 100
    abs_pct_errors = np.abs(pct_errors)

    results = {
        "n_samples": len(y_true),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae": float(np.mean(abs_errors)),
        "mean_error": float(np.mean(errors)),
        "median_error": float(np.median(errors)),
        "std_error": float(np.std(errors)),
    }

    if non_zero.any():
        results["mape"] = float(np.mean(abs_pct_errors[non_zero]))
    else:
        results["mape"] = float("nan")

    try:
        results["r2"] = float(r2_score(y_true, y_pred))
    except ValueError:
        results["r2"] = float("nan")

    for pct in [10, 20, 50]:
        results[f"within_{pct}pct"] = float(np.mean(abs_pct_errors <= pct))

    for p in [90, 95, 99]:
        results[f"error_p{p}"] = float(np.percentile(abs_errors, p))
        results[f"pct_error_p{p}"] = float(np.percentile(abs_pct_errors, p))

    results["actual_mean"] = float(np.mean(y_true))
    results["pred_mean"] = float(np.mean(y_pred))
    results["actual_std"] = float(np.std(y_true))
    results["pred_std"] = float(np.std(y_pred))

    return results


def count_parameters(model) -> int:
    """Count trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
