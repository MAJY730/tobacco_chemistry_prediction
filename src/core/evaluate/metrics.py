"""
Evaluation metrics for multi-output regression.
Supports per-target and average metrics: MAE, RMSE, MAPE, R2.
"""

import numpy as np
import pandas as pd

TARGET_NAMES = ["total_sugar", "reducing_sugar", "nicotine", "total_nitrogen", "potassium"]


def _mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred), axis=0)


def _rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))


def _mape(y_true, y_pred, epsilon=1e-8):
    denominator = np.abs(y_true)
    denominator[denominator < epsilon] = epsilon
    return np.mean(np.abs((y_true - y_pred) / denominator), axis=0) * 100.0


def _r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    r2 = 1.0 - ss_res / ss_tot
    # Handle case where ss_tot is 0
    r2[ss_tot < 1e-10] = 0.0
    return r2


def compute_all_metrics(y_true, y_pred) -> pd.DataFrame:
    """
    Compute MAE, RMSE, MAPE, R2 for each target, plus Avg column.

    Args:
        y_true: (N, 5) ground truth
        y_pred: (N, 5) predictions

    Returns:
        DataFrame with columns [total_sugar, reducing_sugar, nicotine,
                               total_nitrogen, potassium, Avg]
        and rows [MAE, RMSE, MAPE, R2]
    """
    mae_vals = _mae(y_true, y_pred)
    rmse_vals = _rmse(y_true, y_pred)
    mape_vals = _mape(y_true, y_pred)
    r2_vals = _r2(y_true, y_pred)

    data = {
        "metric": ["MAE", "RMSE", "MAPE", "R2"],
        TARGET_NAMES[0]: [round(float(mae_vals[0]), 4), round(float(rmse_vals[0]), 4),
                           round(float(mape_vals[0]), 4), round(float(r2_vals[0]), 4)],
        TARGET_NAMES[1]: [round(float(mae_vals[1]), 4), round(float(rmse_vals[1]), 4),
                           round(float(mape_vals[1]), 4), round(float(r2_vals[1]), 4)],
        TARGET_NAMES[2]: [round(float(mae_vals[2]), 4), round(float(rmse_vals[2]), 4),
                           round(float(mape_vals[2]), 4), round(float(r2_vals[2]), 4)],
        TARGET_NAMES[3]: [round(float(mae_vals[3]), 4), round(float(rmse_vals[3]), 4),
                           round(float(mape_vals[3]), 4), round(float(r2_vals[3]), 4)],
        TARGET_NAMES[4]: [round(float(mae_vals[4]), 4), round(float(rmse_vals[4]), 4),
                           round(float(mape_vals[4]), 4), round(float(r2_vals[4]), 4)],
        "Avg": [
            round(float(np.mean(mae_vals)), 4),
            round(float(np.mean(rmse_vals)), 4),
            round(float(np.mean(mape_vals)), 4),
            round(float(np.mean(r2_vals)), 4),
        ],
    }
    return pd.DataFrame(data)
