"""Public reproduction entry extracted from the verified Grade mean implementation
used in Experiment 02. Scientific prediction and evaluation logic is preserved;
diagnostic and audit-only routines are omitted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = ["total_sugar", "reducing_sugar", "nicotine", "total_nitrogen", "potassium"]
EPSILON = 1e-8
NUM_GRADES = 16


def metrics_raw(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Return unrounded per-target and average metrics using the formal definitions."""
    mae = np.mean(np.abs(y_true - y_pred), axis=0)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    denominator = np.maximum(np.abs(y_true), EPSILON)
    mape = np.mean(np.abs((y_true - y_pred) / denominator), axis=0) * 100.0
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    r2 = np.where(ss_tot < 1e-10, 0.0, 1.0 - ss_res / ss_tot)
    rows = []
    for metric, values in (("MAE", mae), ("RMSE", rmse), ("MAPE", mape), ("R2", r2)):
        row = {"metric": metric}
        row.update({target: float(values[i]) for i, target in enumerate(TARGETS)})
        row["Avg"] = float(np.mean(values))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train-only Grade mean baseline")
    parser.add_argument("--y", type=Path, required=True)
    parser.add_argument("--grade_ids", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("grade_mean_metrics.csv"))
    args = parser.parse_args()
    y_raw = np.load(args.y, allow_pickle=False)
    grade_ids = np.load(args.grade_ids, allow_pickle=False)
    with args.split.open(encoding="utf-8") as handle:
        split = json.load(handle)
    if y_raw.ndim != 2 or y_raw.shape[1] != 5:
        raise ValueError("y must have shape (N, 5)")
    if grade_ids.ndim != 1 or len(grade_ids) != len(y_raw):
        raise ValueError("Grade IDs must have shape (N,) and match y")
    if grade_ids.dtype.kind not in "iu":
        raise ValueError("Grade IDs must be integers")
    if np.any((grade_ids < 0) | (grade_ids >= NUM_GRADES)):
        raise ValueError("Grade IDs must be in 0..15")
    indices = {}
    for key in ("train_idx", "test_idx"):
        if key not in split:
            raise ValueError(f"Split is missing {key}")
        idx = np.asarray(split[key])
        if idx.ndim != 1 or idx.size == 0 or idx.dtype.kind not in "iu":
            raise ValueError(f"{key} must be a nonempty integer array")
        if np.any(idx < 0) or np.any(idx >= len(y_raw)):
            raise ValueError(f"{key} contains out-of-range indices")
        if len(np.unique(idx)) != len(idx):
            raise ValueError(f"{key} contains duplicate indices")
        indices[key] = idx
    train_idx, test_idx = indices["train_idx"], indices["test_idx"]
    if np.intersect1d(train_idx, test_idx).size:
        raise ValueError("Train and test indices overlap")
    grade_means = {}
    for grade_id in range(NUM_GRADES):
        mask = grade_ids[train_idx] == grade_id
        if not mask.any():
            raise RuntimeError("A formal category has no training records")
        values = y_raw[train_idx][mask]
        means = values.mean(axis=0)
        grade_means[grade_id] = means
    grade_mean_pred = np.vstack([grade_means[int(g)] for g in grade_ids[test_idx]])
    grade_mean_metrics = metrics_raw(y_raw[test_idx], grade_mean_pred)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    grade_mean_metrics.to_csv(args.output, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
