"""Build Original-5 by removing Tavg from Original-6 and verify all common channels."""

from __future__ import annotations

import hashlib
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
STANDARDIZED_DIR = PROCESSED_DIR / "standardized"
SCALERS_DIR = PROCESSED_DIR / "scalers"
OUTPUT_DIR = EXPERIMENT_DIR / "data"

FEATURES = ["Tmax", "Tmin", "Rain", "Sunshine", "Pressure"]
O5_FROM_O6 = [0, 1, 3, 4, 5]
O5_FROM_W6 = [0, 1, 2, 3, 4]
O5_FROM_W7 = [0, 1, 3, 4, 5]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def standardize_independently(
    values: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray
) -> tuple[np.ndarray, dict]:
    mean = np.zeros(values.shape[2], dtype=np.float32)
    std = np.zeros(values.shape[2], dtype=np.float32)
    train = values[train_idx]
    for feature_index in range(values.shape[2]):
        flattened = train[:, :, feature_index].ravel()
        mean[feature_index] = np.mean(flattened)
        std[feature_index] = np.std(flattened)
        if std[feature_index] < 1e-10:
            std[feature_index] = 1.0

    def transform(part: np.ndarray) -> np.ndarray:
        scaled = part.copy()
        for feature_index in range(part.shape[2]):
            scaled[:, :, feature_index] = (
                part[:, :, feature_index] - mean[feature_index]
            ) / std[feature_index]
        return scaled

    scaled_all = np.zeros_like(values)
    scaled_all[train_idx] = transform(values[train_idx])
    scaled_all[val_idx] = transform(values[val_idx])
    scaled_all[test_idx] = transform(values[test_idx])
    return scaled_all, {"mean": mean, "std": std}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "split": PROCESSED_DIR / "split_indices.json",
        "o6_raw": PROCESSED_DIR / "X_original6.npy",
        "w6_raw": PROCESSED_DIR / "X_wind6.npy",
        "w7_raw": PROCESSED_DIR / "X_wind7.npy",
        "o6_scaled": STANDARDIZED_DIR / "X_original6_scaled.npy",
        "w6_scaled": STANDARDIZED_DIR / "X_wind6_scaled.npy",
        "w7_scaled": STANDARDIZED_DIR / "X_wind7_scaled.npy",
        "o6_scaler": SCALERS_DIR / "x_original6_scaler.pkl",
        "w6_scaler": SCALERS_DIR / "x_wind6_scaler.pkl",
        "w7_scaler": SCALERS_DIR / "x_wind7_scaler.pkl",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")

    split = json.loads(paths["split"].read_text(encoding="utf-8"))
    if split.get("config", {}).get("random_state") != 42:
        raise ValueError("Split is not seed=42")
    train_idx = np.asarray(split["train_idx"], dtype=np.int64)
    val_idx = np.asarray(split["val_idx"], dtype=np.int64)
    test_idx = np.asarray(split["test_idx"], dtype=np.int64)
    if (len(train_idx), len(val_idx), len(test_idx)) != (1952, 418, 420):
        raise ValueError("Unexpected split sizes")

    o6_raw = np.load(paths["o6_raw"])
    w6_raw = np.load(paths["w6_raw"])
    w7_raw = np.load(paths["w7_raw"])
    o6_scaled = np.load(paths["o6_scaled"])
    w6_scaled = np.load(paths["w6_scaled"])
    w7_scaled = np.load(paths["w7_scaled"])
    o5_raw = np.ascontiguousarray(o6_raw[:, :, O5_FROM_O6])
    o5_scaled, o5_scaler = standardize_independently(o5_raw, train_idx, val_idx, test_idx)

    checks = {
        "shape_is_2790_153_5": o5_raw.shape == (2790, 153, 5),
        "raw_equals_o6_without_tavg": np.array_equal(o5_raw, o6_raw[:, :, O5_FROM_O6]),
        "raw_equals_w6_without_wind": np.array_equal(o5_raw, w6_raw[:, :, O5_FROM_W6]),
        "raw_equals_w7_without_tavg_wind": np.array_equal(o5_raw, w7_raw[:, :, O5_FROM_W7]),
        "scaled_equals_o6_without_tavg": np.array_equal(
            o5_scaled, o6_scaled[:, :, O5_FROM_O6]
        ),
        "scaled_equals_w6_without_wind": np.array_equal(
            o5_scaled, w6_scaled[:, :, O5_FROM_W6]
        ),
        "scaled_equals_w7_without_tavg_wind": np.array_equal(
            o5_scaled, w7_scaled[:, :, O5_FROM_W7]
        ),
        "scaled_train_mean_zero": bool(
            np.allclose(
                o5_scaled[train_idx].astype(np.float64).mean(axis=(0, 1)),
                0.0,
                atol=2e-6,
            )
        ),
        "scaled_train_std_one": bool(
            np.allclose(
                o5_scaled[train_idx].astype(np.float64).std(axis=(0, 1)),
                1.0,
                atol=1e-6,
            )
        ),
    }
    with paths["o6_scaler"].open("rb") as handle:
        o6_scaler = pickle.load(handle)
    with paths["w6_scaler"].open("rb") as handle:
        w6_scaler = pickle.load(handle)
    with paths["w7_scaler"].open("rb") as handle:
        w7_scaler = pickle.load(handle)
    checks.update(
        {
            "mean_equals_o6_subset": np.array_equal(
                o5_scaler["mean"], o6_scaler["mean"][O5_FROM_O6]
            ),
            "std_equals_o6_subset": np.array_equal(
                o5_scaler["std"], o6_scaler["std"][O5_FROM_O6]
            ),
            "mean_equals_w6_subset": np.array_equal(
                o5_scaler["mean"], w6_scaler["mean"][O5_FROM_W6]
            ),
            "std_equals_w6_subset": np.array_equal(
                o5_scaler["std"], w6_scaler["std"][O5_FROM_W6]
            ),
            "mean_equals_w7_subset": np.array_equal(
                o5_scaler["mean"], w7_scaler["mean"][O5_FROM_W7]
            ),
            "std_equals_w7_subset": np.array_equal(
                o5_scaler["std"], w7_scaler["std"][O5_FROM_W7]
            ),
        }
    )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Original-5 validation failed: {failed}")

    raw_output = OUTPUT_DIR / "X_original5.npy"
    scaled_output = OUTPUT_DIR / "X_original5_scaled.npy"
    scaler_output = OUTPUT_DIR / "x_original5_scaler.pkl"
    metadata_output = OUTPUT_DIR / "original5_preparation_metadata.json"
    np.save(raw_output, o5_raw)
    np.save(scaled_output, o5_scaled)
    with scaler_output.open("wb") as handle:
        pickle.dump(o5_scaler, handle)
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "configuration": "Original-5",
        "features": FEATURES,
        "construction": "Original-6 with Tavg channel removed; no Wind added",
        "shape": list(o5_raw.shape),
        "split": {"seed": 42, "train": 1952, "validation": 418, "test": 420},
        "scaler_fit_scope": "training indices only",
        "scaler": {
            "mean": o5_scaler["mean"].tolist(), "std": o5_scaler["std"].tolist()
        },
        "validation_checks": checks,
        "source_sha256": {name: sha256(path) for name, path in paths.items()},
        "output_sha256": {
            "X_original5.npy": sha256(raw_output),
            "X_original5_scaled.npy": sha256(scaled_output),
            "x_original5_scaler.pkl": sha256(scaler_output),
        },
    }
    metadata_output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("ORIGINAL-5 PREPARATION: PASSED")
    print(f"features={FEATURES}")
    print(f"shape={o5_raw.shape}")
    print(f"mean={o5_scaler['mean']}")
    print(f"std={o5_scaler['std']}")
    for name, passed in checks.items():
        print(f"{name}={passed}")
    print(f"outputs={OUTPUT_DIR}")


if __name__ == "__main__":
    main()
