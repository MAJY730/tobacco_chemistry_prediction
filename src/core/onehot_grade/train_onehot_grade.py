"""Public reproduction entry extracted from the verified Climate + One-hot
implementation used in Experiment 02. Model, training, model-selection, and
evaluation logic are preserved; non-core reporting routines are omitted.
"""
from __future__ import annotations

import json
import sys
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXPERIMENT_DIR))

from evaluate.metrics import TARGET_NAMES, compute_all_metrics
from onehot_grade_tsmixer import OneHotGradeTSMixer

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
STANDARDIZED_DIR = PROCESSED_DIR / "standardized"
SCALERS_DIR = PROCESSED_DIR / "scalers"
TABLES_DIR = EXPERIMENT_DIR / "results" / "seed42" / "tables"

SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 300
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 30
SCHEDULER_PATIENCE = 10
SCHEDULER_FACTOR = 0.5
TORCH_NUM_THREADS = 8
NUM_GRADES = 16


def set_seed(seed: int) -> None:
    torch.set_num_threads(TORCH_NUM_THREADS)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def inverse_y(values: np.ndarray, scaler: dict) -> np.ndarray:
    return values * scaler["std"] + scaler["mean"]


def load_and_validate_data() -> dict:
    paths = {
        "split": PROCESSED_DIR / "split_indices.json",
        "wind7": STANDARDIZED_DIR / "X_wind7_scaled.npy",
        "grade_ids": PROCESSED_DIR / "grade_ids.npy",
        "grade_mapping": PROCESSED_DIR / "grade_mapping.json",
        "y_scaled": STANDARDIZED_DIR / "y_scaled.npy",
        "y_scaler": SCALERS_DIR / "y_scaler.pkl",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required files missing: {missing}")

    with paths["split"].open("r", encoding="utf-8") as handle:
        split = json.load(handle)
    with paths["grade_mapping"].open("r", encoding="utf-8") as handle:
        mapping = json.load(handle)
    with paths["y_scaler"].open("rb") as handle:
        y_scaler = pickle.load(handle)

    for key in ("train_idx", "val_idx", "test_idx"):
        if key not in split:
            raise ValueError(f"Split is missing {key}")
        idx = np.asarray(split[key])
        if idx.ndim != 1 or idx.size == 0 or idx.dtype.kind not in "iu":
            raise ValueError(f"{key} must be a nonempty integer array")
        if np.any(idx < 0) or np.any(idx >= 2790):
            raise ValueError(f"{key} contains out-of-range indices")
    train_idx = np.asarray(split["train_idx"], dtype=np.int64)
    val_idx = np.asarray(split["val_idx"], dtype=np.int64)
    test_idx = np.asarray(split["test_idx"], dtype=np.int64)
    all_idx = np.concatenate([train_idx, val_idx, test_idx])
    climate = np.load(paths["wind7"])
    grade_ids = np.load(paths["grade_ids"])
    raw_path = PROCESSED_DIR / "y.npy"
    y_raw = np.load(raw_path) if raw_path.exists() else None
    y_scaled = np.load(paths["y_scaled"])

    if split.get("config", {}).get("random_state") != SEED:
        raise ValueError("Fixed split is not seed=42")
    if (len(train_idx), len(val_idx), len(test_idx)) != (1952, 418, 420):
        raise ValueError("Unexpected split sizes")
    if len(np.unique(all_idx)) != 2790 or sorted(all_idx.tolist()) != list(range(2790)):
        raise ValueError("Split indices overlap or do not cover exactly 0..2789")
    if climate.shape != (2790, 153, 7):
        raise ValueError(f"Unexpected Wind-7 shape: {climate.shape}")
    if grade_ids.shape != (2790,) or y_scaled.shape != (2790, 5):
        raise ValueError("Unexpected Grade or target array shape")
    if grade_ids.dtype.kind not in "iu" or grade_ids.min() != 0 or grade_ids.max() != 15:
        raise ValueError("Grade IDs are not integer IDs spanning 0..15")
    if mapping.get("num_grades") != NUM_GRADES:
        raise ValueError("Grade mapping is not the expected 16-class mapping")
    expected_targets = [
        "total_sugar", "reducing_sugar", "nicotine", "total_nitrogen", "potassium"
    ]
    if list(TARGET_NAMES) != expected_targets:
        raise ValueError("Target order differs from the existing metric implementation")
    if y_raw is not None:
        if y_raw.shape != (2790, 5):
            raise ValueError("Unexpected raw target shape")
        if not np.allclose(inverse_y(y_scaled, y_scaler), y_raw, rtol=1e-5, atol=1e-5):
            raise ValueError("y_scaled and y_scaler do not reconstruct raw targets")
    else:
        print("Optional raw-target consistency check not performed: y.npy is absent.")

    return {
        "paths": paths,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "climate": climate,
        "grade_ids": grade_ids,
        "y_scaled": y_scaled,
        "y_scaler": y_scaler,
    }


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for climate, grade_ids in loader:
            outputs.append(model(climate.to(device), grade_ids.to(device)).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def main() -> None:
    set_seed(SEED)
    data = load_and_validate_data()
    train_idx, val_idx, test_idx = data["train_idx"], data["val_idx"], data["test_idx"]
    climate, grade_ids, y_scaled = data["climate"], data["grade_ids"], data["y_scaled"]
    train_ds = TensorDataset(
        torch.as_tensor(climate[train_idx], dtype=torch.float32),
        torch.as_tensor(y_scaled[train_idx], dtype=torch.float32),
        torch.as_tensor(grade_ids[train_idx], dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.as_tensor(climate[val_idx], dtype=torch.float32),
        torch.as_tensor(y_scaled[val_idx], dtype=torch.float32),
        torch.as_tensor(grade_ids[val_idx], dtype=torch.long),
    )
    test_ds = TensorDataset(
        torch.as_tensor(climate[test_idx], dtype=torch.float32),
        torch.as_tensor(grade_ids[test_idx], dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OneHotGradeTSMixer().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if parameter_count != 179477:
        raise ValueError(f"Unexpected parameter count: {parameter_count}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE
    )
    criterion = nn.MSELoss()
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_count = 0
    history_rows = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_sum = 0.0
        for batch_climate, batch_targets, batch_grades in train_loader:
            batch_climate = batch_climate.to(device)
            batch_targets = batch_targets.to(device)
            batch_grades = batch_grades.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_climate, batch_grades), batch_targets)
            loss.backward()
            optimizer.step()
            train_sum += loss.item() * batch_climate.size(0)
        train_loss = train_sum / len(train_loader.dataset)

        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for batch_climate, batch_targets, batch_grades in val_loader:
                batch_climate = batch_climate.to(device)
                batch_targets = batch_targets.to(device)
                batch_grades = batch_grades.to(device)
                val_sum += criterion(
                    model(batch_climate, batch_grades), batch_targets
                ).item() * batch_climate.size(0)
        val_loss = val_sum / len(val_loader.dataset)
        scheduler.step(val_loss)
        history_rows.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
             "learning_rate": optimizer.param_groups[0]["lr"]}
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
        if epoch == 1 or epoch % 10 == 0:
            message = f"epoch={epoch} train_loss={train_loss:.8f} val_loss={val_loss:.8f}"
            print(message, flush=True)
        if patience_count >= EARLY_STOPPING_PATIENCE:
            break

    stop_epoch = history_rows[-1]["epoch"]
    if best_state is None:
        raise RuntimeError("Training did not produce a best state")
    model.load_state_dict(best_state)

    y_pred_scaled = predict(model, test_loader, device)
    y_true = inverse_y(y_scaled[test_idx], data["y_scaler"])
    y_pred = inverse_y(y_pred_scaled, data["y_scaler"])
    metrics = compute_all_metrics(y_true, y_pred)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(TABLES_DIR / "onehot_grade_tsmixer_seed42_metrics.csv",
                   index=False, encoding="utf-8-sig")
    print(f"Best epoch: {best_epoch}; stop epoch: {stop_epoch}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
