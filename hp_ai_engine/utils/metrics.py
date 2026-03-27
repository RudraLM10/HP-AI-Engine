"""
Evaluation metrics for HP AI Engine.

All functions accept numpy arrays or PyTorch tensors and return float scalars.
"""

from __future__ import annotations

import numpy as np
import torch


def _to_numpy(x: np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert input to numpy array if it's a torch tensor."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def mape(y_true: np.ndarray | torch.Tensor, y_pred: np.ndarray | torch.Tensor) -> float:
    """
    Mean Absolute Percentage Error.

    Excludes zero-valued actuals to avoid division by zero.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        MAPE as a percentage (0-100+).
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    mask = y_true != 0
    if not mask.any():
        return 0.0

    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def rmse(y_true: np.ndarray | torch.Tensor, y_pred: np.ndarray | torch.Tensor) -> float:
    """Root Mean Squared Error."""
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray | torch.Tensor, y_pred: np.ndarray | torch.Tensor) -> float:
    """Mean Absolute Error."""
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def r_squared(y_true: np.ndarray | torch.Tensor, y_pred: np.ndarray | torch.Tensor) -> float:
    """
    Coefficient of determination (R²).

    Returns:
        R² score. 1.0 = perfect prediction, 0.0 = predicting the mean,
        negative = worse than predicting the mean.
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0

    return float(1.0 - ss_res / ss_tot)


def calibration_score(
    y_true: np.ndarray | torch.Tensor,
    lower: np.ndarray | torch.Tensor,
    upper: np.ndarray | torch.Tensor,
) -> float:
    """
    Calibration score: fraction of actuals falling within a confidence interval.

    A well-calibrated 90% CI should contain ~90% of actuals.

    Args:
        y_true: Ground truth values.
        lower: Lower bound of the confidence interval.
        upper: Upper bound of the confidence interval.

    Returns:
        Coverage fraction in [0.0, 1.0].
    """
    y_true = _to_numpy(y_true)
    lower = _to_numpy(lower)
    upper = _to_numpy(upper)

    within = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(within))


def smape(y_true: np.ndarray | torch.Tensor, y_pred: np.ndarray | torch.Tensor) -> float:
    """
    Symmetric Mean Absolute Percentage Error.

    More robust than MAPE for values near zero.

    Returns:
        sMAPE as a percentage (0-200).
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denominator != 0

    if not mask.any():
        return 0.0

    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100)


def compute_all_metrics(
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
    lower: np.ndarray | torch.Tensor | None = None,
    upper: np.ndarray | torch.Tensor | None = None,
) -> dict[str, float]:
    """
    Compute all available metrics in one call.

    Returns:
        Dict with keys: mape, rmse, mae, r_squared, smape.
        If lower/upper provided, also includes calibration_score.
    """
    results = {
        "mape": mape(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r_squared": r_squared(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }

    if lower is not None and upper is not None:
        results["calibration_score"] = calibration_score(y_true, lower, upper)

    return results
