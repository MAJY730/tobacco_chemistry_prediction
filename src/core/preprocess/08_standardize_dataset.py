"""
Standardize X and y using only train set fit.
Avoids data leakage by fitting scaler exclusively on train_idx.
"""

import sys
import json
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocess.utils import ensure_dir, save_report_to_excel

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
STANDARDIZED_DIR = PROJECT_ROOT / "data" / "processed" / "standardized"
SCALERS_DIR = PROJECT_ROOT / "data" / "processed" / "scalers"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"

INPUT_FILES = {
    "original6": "X_original6.npy",
    "wind6": "X_wind6.npy",
    "wind7": "X_wind7.npy",
}

OUTPUT_FILES = {
    "original6": "X_original6_scaled.npy",
    "wind6": "X_wind6_scaled.npy",
    "wind7": "X_wind7_scaled.npy",
}

SCALER_FILES = {
    "original6": "x_original6_scaler.pkl",
    "wind6": "x_wind6_scaler.pkl",
    "wind7": "x_wind7_scaler.pkl",
}

FEATURE_NAMES = {
    "original6": ["Tmax", "Tmin", "Tavg", "Rain", "Sunshine", "Pressure"],
    "wind6": ["Tmax", "Tmin", "Rain", "Sunshine", "Pressure", "Wind"],
    "wind7": ["Tmax", "Tmin", "Tavg", "Rain", "Sunshine", "Pressure", "Wind"],
}

Y_FEATURE_NAMES = ["total_sugar", "reducing_sugar", "nicotine", "total_nitrogen", "potassium"]


def standardize_3d(X_train, X_val, X_test):
    """
    Standardize 3D array (N, T, F) using per-feature mean/std from train set.
    Returns scaled arrays and scaler params dict.
    """
    N_train, T, F = X_train.shape
    mean = np.zeros(F, dtype=np.float32)
    std = np.zeros(F, dtype=np.float32)

    for f in range(F):
        vals = X_train[:, :, f].ravel()
        mean[f] = np.mean(vals)
        std[f] = np.std(vals)
        if std[f] < 1e-10:
            std[f] = 1.0  # avoid division by zero

    def transform(X):
        X_scaled = X.copy()
        for f in range(F):
            X_scaled[:, :, f] = (X[:, :, f] - mean[f]) / std[f]
        return X_scaled

    return transform(X_train), transform(X_val), transform(X_test), {"mean": mean, "std": std}


def standardize_2d(y_train, y_val, y_test):
    """Standardize 2D (N, F) using per-column mean/std from train set."""
    N_train, F = y_train.shape
    mean = np.zeros(F, dtype=np.float32)
    std = np.zeros(F, dtype=np.float32)

    for f in range(F):
        vals = y_train[:, f]
        mean[f] = np.mean(vals)
        std[f] = np.std(vals)
        if std[f] < 1e-10:
            std[f] = 1.0

    def transform(y):
        y_scaled = y.copy()
        for f in range(F):
            y_scaled[:, f] = (y[:, f] - mean[f]) / std[f]
        return y_scaled

    return transform(y_train), transform(y_val), transform(y_test), {"mean": mean, "std": std}


def main():
    print("=" * 60)
    print("Dataset Standardization (08_standardize_dataset.py)")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    ensure_dir(STANDARDIZED_DIR)
    ensure_dir(SCALERS_DIR)
    ensure_dir(RESULTS_DIR)

    # Load split indices
    with open(PROCESSED_DIR / "split_indices.json", "r") as f:
        split = json.load(f)
    train_idx = split["train_idx"]
    val_idx = split["val_idx"]
    test_idx = split["test_idx"]

    print(f"train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")

    # Load y
    y = np.load(str(PROCESSED_DIR / "y.npy"))
    y_train_raw = y[train_idx]
    y_val_raw = y[val_idx]
    y_test_raw = y[test_idx]

    # Standardize y
    print("\n--- Standardizing y ---")
    y_train_scaled, y_val_scaled, y_test_scaled, y_params = standardize_2d(y_train_raw, y_val_raw, y_test_raw)

    # Reconstruct full y_scaled in original order
    y_scaled = np.zeros_like(y)
    y_scaled[train_idx] = y_train_scaled
    y_scaled[val_idx] = y_val_scaled
    y_scaled[test_idx] = y_test_scaled
    np.save(str(STANDARDIZED_DIR / "y_scaled.npy"), y_scaled)
    print(f"y_scaled saved: {y_scaled.shape}")

    with open(SCALERS_DIR / "y_scaler.pkl", "wb") as f:
        pickle.dump(y_params, f)
    print(f"y_scaler saved: {SCALERS_DIR / 'y_scaler.pkl'}")

    scaler_check_rows = []

    for yf, name in enumerate(Y_FEATURE_NAMES):
        scaler_check_rows.append({
            "dataset": "y",
            "feature": name,
            "train_mean": round(float(y_params["mean"][yf]), 4),
            "train_std": round(float(y_params["std"][yf]), 4),
            "scaled_train_mean": round(float(np.mean(y_scaled[train_idx, yf])), 6),
            "scaled_train_std": round(float(np.std(y_scaled[train_idx, yf])), 6),
        })

    # Standardize each X
    for input_type in ["original6", "wind6", "wind7"]:
        print(f"\n--- Standardizing {input_type} ---")
        X = np.load(str(PROCESSED_DIR / INPUT_FILES[input_type]))
        X_train_raw = X[train_idx]
        X_val_raw = X[val_idx]
        X_test_raw = X[test_idx]

        X_train_sc, X_val_sc, X_test_sc, x_params = standardize_3d(X_train_raw, X_val_raw, X_test_raw)

        # Reconstruct full X_scaled
        X_scaled = np.zeros_like(X)
        X_scaled[train_idx] = X_train_sc
        X_scaled[val_idx] = X_val_sc
        X_scaled[test_idx] = X_test_sc
        np.save(str(STANDARDIZED_DIR / OUTPUT_FILES[input_type]), X_scaled)
        print(f"  {OUTPUT_FILES[input_type]}: {X_scaled.shape}")

        with open(SCALERS_DIR / SCALER_FILES[input_type], "wb") as f:
            pickle.dump(x_params, f)
        print(f"  {SCALER_FILES[input_type]} saved")

        names = FEATURE_NAMES[input_type]
        for xf, name in enumerate(names):
            scaler_check_rows.append({
                "dataset": input_type,
                "feature": name,
                "train_mean": round(float(x_params["mean"][xf]), 4),
                "train_std": round(float(x_params["std"][xf]), 4),
                "scaled_train_mean": round(float(np.mean(X_scaled[train_idx, :, xf])), 6),
                "scaled_train_std": round(float(np.std(X_scaled[train_idx, :, xf])), 6),
            })

    # Save scaler check report
    df_check = pd.DataFrame(scaler_check_rows)
    save_report_to_excel({"scaler_check": df_check}, str(RESULTS_DIR / "stage3_scaler_check.xlsx"))
    print(f"\nScaler check report saved: {RESULTS_DIR / 'stage3_scaler_check.xlsx'}")

    print("\n" + "=" * 60)
    print("Standardization complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
