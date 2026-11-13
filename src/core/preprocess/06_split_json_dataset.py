"""
Create a shared train / val / test split.
Read sample_info.csv, create a random split, and save split_indices.json.
All subsequent model training must use the same split_indices.json.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocess.utils import ensure_dir

with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RANDOM_STATE = config["split"]["random_state"]  # 42
TRAIN_RATIO = config["split"]["train_ratio"]    # 0.7
VAL_RATIO = config["split"]["val_ratio"]        # 0.15
TEST_RATIO = config["split"]["test_ratio"]      # 0.15


def main():
    print("=" * 60)
    print("Dataset split (06_split_json_dataset.py)")
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"random_state: {RANDOM_STATE}")
    print(f"Split ratios: train={TRAIN_RATIO}, val={VAL_RATIO}, test={TEST_RATIO}")
    print("=" * 60)

    # Read sample_info.
    info_path = PROCESSED_DIR / "sample_info.csv"
    if not info_path.exists():
        print(f"[ERROR] sample_info.csv not found: {info_path}")
        print("Run 05_build_from_original_json.py first.")
        sys.exit(1)

    df = pd.read_csv(info_path)
    N = len(df)
    print(f"\nTotal record count: {N}")

    # Check that split ratios sum to one.
    assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 0.01, \
        f"Split ratios do not sum to one: {TRAIN_RATIO}+{VAL_RATIO}+{TEST_RATIO}={TRAIN_RATIO+VAL_RATIO+TEST_RATIO}"

    # Generate a random permutation.
    rng = np.random.RandomState(RANDOM_STATE)
    indices = rng.permutation(N)

    # Split the permutation.
    n_train = int(N * TRAIN_RATIO)
    n_val = int(N * VAL_RATIO)
    n_test = N - n_train - n_val  # Keep the total record count unchanged.

    train_idx = sorted(indices[:n_train].tolist())
    val_idx = sorted(indices[n_train:n_train + n_val].tolist())
    test_idx = sorted(indices[n_train + n_val:].tolist())

    print(f"train: {len(train_idx)} ({len(train_idx)/N*100:.1f}%)")
    print(f"val:   {len(val_idx)} ({len(val_idx)/N*100:.1f}%)")
    print(f"test:  {len(test_idx)} ({len(test_idx)/N*100:.1f}%)")

    # Check for disjoint splits.
    train_set = set(train_idx)
    val_set = set(val_idx)
    test_set = set(test_idx)
    assert len(train_set & val_set) == 0, "train and val overlap."
    assert len(train_set & test_set) == 0, "train and test overlap."
    assert len(val_set & test_set) == 0, "val and test overlap."
    assert len(train_set | val_set | test_set) == N, "Record count mismatch."
    print("No overlap check passed.")

    # Save split indices.
    split_data = {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "config": {
            "random_state": RANDOM_STATE,
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "test_ratio": TEST_RATIO,
            "total_samples": N,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_test": len(test_idx),
        },
        "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    split_path = PROCESSED_DIR / "split_indices.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_data, f, ensure_ascii=False, indent=2)

    print(f"\nSplit file saved: {split_path}")
    print("\n" + "=" * 60)
    print("Dataset split complete. Do not rerun this script.")
    print("All subsequent training runs must use the same split_indices.json.")
    print("=" * 60)


if __name__ == "__main__":
    main()
