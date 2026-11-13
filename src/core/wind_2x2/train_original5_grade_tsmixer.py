"""Train only the Original-5 Grade-TSMixer default experiment at seed=42."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import pickle
import platform
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluate.metrics import TARGET_NAMES, compute_all_metrics
from models.top_ts_models import build_model

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
O5_DATA_DIR = EXPERIMENT_DIR / "data"
RESULTS_DIR = EXPERIMENT_DIR / "results" / "original5" / "seed42"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
LOGS_DIR = RESULTS_DIR / "logs"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 300
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 30
SCHEDULER_PATIENCE = 10
SCHEDULER_FACTOR = 0.5
TORCH_NUM_THREADS = 8
EXPECTED_PARAMETERS = 178965


def set_seed(seed: int) -> None:
    torch.set_num_threads(TORCH_NUM_THREADS)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def environment_info() -> dict:
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": package_version("scikit-learn"),
        "matplotlib": package_version("matplotlib"),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "torch_num_threads": torch.get_num_threads(),
    }


def inverse_y(values: np.ndarray, scaler: dict) -> np.ndarray:
    return values * scaler["std"] + scaler["mean"]


def load_and_validate_data() -> dict:
    paths = {
        "o5_raw": O5_DATA_DIR / "X_original5.npy",
        "o5_scaled": O5_DATA_DIR / "X_original5_scaled.npy",
        "o5_scaler": O5_DATA_DIR / "x_original5_scaler.pkl",
        "o5_metadata": O5_DATA_DIR / "original5_preparation_metadata.json",
        "split": PROCESSED_DIR / "split_indices.json",
        "grade_ids": PROCESSED_DIR / "grade_ids.npy",
        "grade_mapping": PROCESSED_DIR / "grade_mapping.json",
        "y_raw": PROCESSED_DIR / "y.npy",
        "y_scaled": PROCESSED_DIR / "standardized" / "y_scaled.npy",
        "y_scaler": PROCESSED_DIR / "scalers" / "y_scaler.pkl",
        "metrics_code": PROJECT_ROOT / "evaluate" / "metrics.py",
        "model_code": PROJECT_ROOT / "models" / "top_ts_models.py",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")
    split = json.loads(paths["split"].read_text(encoding="utf-8"))
    metadata = json.loads(paths["o5_metadata"].read_text(encoding="utf-8"))
    mapping = json.loads(paths["grade_mapping"].read_text(encoding="utf-8"))
    with paths["y_scaler"].open("rb") as handle:
        y_scaler = pickle.load(handle)
    train_idx = np.asarray(split["train_idx"], dtype=np.int64)
    val_idx = np.asarray(split["val_idx"], dtype=np.int64)
    test_idx = np.asarray(split["test_idx"], dtype=np.int64)
    all_idx = np.concatenate([train_idx, val_idx, test_idx])
    climate = np.load(paths["o5_scaled"])
    grade_ids = np.load(paths["grade_ids"])
    y_raw = np.load(paths["y_raw"])
    y_scaled = np.load(paths["y_scaled"])

    if split.get("config", {}).get("random_state") != SEED:
        raise ValueError("Fixed split is not seed=42")
    if (len(train_idx), len(val_idx), len(test_idx)) != (1952, 418, 420):
        raise ValueError("Unexpected split sizes")
    if len(np.unique(all_idx)) != 2790 or sorted(all_idx.tolist()) != list(range(2790)):
        raise ValueError("Split indices do not cover exactly 0..2789")
    if climate.shape != (2790, 153, 5):
        raise ValueError(f"Unexpected Original-5 shape: {climate.shape}")
    if metadata.get("features") != ["Tmax", "Tmin", "Rain", "Sunshine", "Pressure"]:
        raise ValueError("Original-5 feature order is incorrect")
    if not all(metadata.get("validation_checks", {}).values()):
        raise ValueError("Original-5 preparation validation did not fully pass")
    if sha256(paths["o5_scaled"]) != metadata["output_sha256"]["X_original5_scaled.npy"]:
        raise ValueError("Original-5 standardized data hash changed")
    if grade_ids.shape != (2790,) or grade_ids.min() != 0 or grade_ids.max() != 15:
        raise ValueError("Unexpected Grade IDs")
    if mapping.get("num_grades") != 16:
        raise ValueError("Expected 16 Grade classes")
    if y_raw.shape != (2790, 5) or y_scaled.shape != (2790, 5):
        raise ValueError("Unexpected target shape")
    if list(TARGET_NAMES) != [
        "total_sugar", "reducing_sugar", "nicotine", "total_nitrogen", "potassium"
    ]:
        raise ValueError("Target order differs from formal metrics code")
    if not np.allclose(inverse_y(y_scaled, y_scaler), y_raw, rtol=1e-5, atol=1e-5):
        raise ValueError("y_scaled and y_scaler do not reconstruct raw targets")
    return {
        "paths": paths,
        "file_hashes": {name: sha256(path) for name, path in paths.items()},
        "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
        "climate": climate, "grade_ids": grade_ids,
        "y_scaled": y_scaled, "y_scaler": y_scaler,
    }


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    predictions = []
    with torch.no_grad():
        for climate, grade_ids in loader:
            predictions.append(model(climate.to(device), grade_ids.to(device)).cpu().numpy())
    return np.concatenate(predictions, axis=0)


def plot_training_curve(history: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history["epoch"], history["train_loss"], label="Train loss")
    ax.plot(history["epoch"], history["val_loss"], label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Standardized-target MSE")
    ax.set_title("Grade-TSMixer Original-5 training curve (seed=42)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    for directory in (CHECKPOINTS_DIR, LOGS_DIR, TABLES_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "grade_tsmixer_original5_default_seed42.log"
    logging.basicConfig(
        filename=log_path, filemode="w", level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger("original5")
    started_at = datetime.now().astimezone()
    set_seed(SEED)
    environment = environment_info()
    data = load_and_validate_data()
    logger.info("Environment: %s", environment)
    logger.info("Original-5 validation passed")

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

    device = torch.device(environment["device"])
    model = build_model(
        "grade_tsmixer", input_dim=5, seq_len=153, variant="default", num_grades=16
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if parameter_count != EXPECTED_PARAMETERS:
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
            batch_climate, batch_targets, batch_grades = (
                batch_climate.to(device), batch_targets.to(device), batch_grades.to(device)
            )
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
                batch_climate, batch_targets, batch_grades = (
                    batch_climate.to(device), batch_targets.to(device), batch_grades.to(device)
                )
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
            logger.info(message)
        if patience_count >= EARLY_STOPPING_PATIENCE:
            break

    stop_epoch = history_rows[-1]["epoch"]
    if best_state is None:
        raise RuntimeError("Training did not produce a best state")
    model.load_state_dict(best_state)
    history = pd.DataFrame(history_rows)
    checkpoint_path = CHECKPOINTS_DIR / "grade_tsmixer_original5_default_seed42_best.pth"
    torch.save(
        {
            "model_state_dict": best_state, "model_name": "grade_tsmixer",
            "input_type": "original5", "variant": "default", "input_dim": 5,
            "best_epoch": best_epoch, "stop_epoch": stop_epoch,
            "best_val_loss": best_val_loss, "seed": SEED, "target_names": TARGET_NAMES,
        },
        checkpoint_path,
    )
    y_pred_scaled = predict(model, test_loader, device)
    y_true = inverse_y(y_scaled[test_idx], data["y_scaler"])
    y_pred = inverse_y(y_pred_scaled, data["y_scaler"])
    metrics = compute_all_metrics(y_true, y_pred)

    history_path = TABLES_DIR / "grade_tsmixer_original5_default_seed42_training_history.csv"
    metrics_csv_path = TABLES_DIR / "grade_tsmixer_original5_default_seed42_metrics.csv"
    metrics_json_path = TABLES_DIR / "grade_tsmixer_original5_default_seed42_metrics.json"
    long_path = TABLES_DIR / "grade_tsmixer_original5_default_seed42_predictions_long.csv"
    wide_path = TABLES_DIR / "grade_tsmixer_original5_default_seed42_predictions_wide.csv"
    metadata_path = TABLES_DIR / "grade_tsmixer_original5_default_seed42_run_metadata.json"
    curve_path = FIGURES_DIR / "grade_tsmixer_original5_default_seed42_training_curve.png"
    history.to_csv(history_path, index=False, encoding="utf-8-sig")
    metrics.to_csv(metrics_csv_path, index=False, encoding="utf-8-sig")
    metrics.to_json(metrics_json_path, orient="records", indent=2, force_ascii=False)

    long_rows, wide_rows = [], []
    for sample_index, original_index in enumerate(test_idx):
        wide_row = {
            "sample_index": sample_index, "original_index": int(original_index),
            "grade_id": int(grade_ids[original_index]), "seed": SEED,
        }
        for target_index, target_name in enumerate(TARGET_NAMES):
            true_value = float(y_true[sample_index, target_index])
            pred_value = float(y_pred[sample_index, target_index])
            long_rows.append(
                {"sample_index": sample_index, "original_index": int(original_index),
                 "grade_id": int(grade_ids[original_index]), "target_name": target_name,
                 "y_true": true_value, "y_pred": pred_value, "input_type": "original5",
                 "model_name": "grade_tsmixer", "variant": "default", "seed": SEED}
            )
            wide_row[f"true_{target_name}"] = true_value
            wide_row[f"pred_{target_name}"] = pred_value
        wide_rows.append(wide_row)
    pd.DataFrame(long_rows).to_csv(long_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(wide_rows).to_csv(wide_path, index=False, encoding="utf-8-sig")
    plot_training_curve(history, curve_path)

    finished_at = datetime.now().astimezone()
    run_metadata = {
        "experiment": "Original-5 Grade-TSMixer default",
        "started_at": started_at.isoformat(), "finished_at": finished_at.isoformat(),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "seed": SEED, "features": ["Tmax", "Tmin", "Rain", "Sunshine", "Pressure"],
        "parameter_count": parameter_count,
        "training": {
            "optimizer": "AdamW", "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS, "loss": "MSE on standardized targets",
            "scheduler": "ReduceLROnPlateau", "scheduler_factor": SCHEDULER_FACTOR,
            "scheduler_patience": SCHEDULER_PATIENCE,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "best_epoch": best_epoch, "stop_epoch": stop_epoch,
            "best_val_loss": best_val_loss,
            "last_train_loss": float(history.iloc[-1]["train_loss"]),
            "last_val_loss": float(history.iloc[-1]["val_loss"]),
        },
        "environment": environment,
        "data": {
            "n_total": 2790, "n_train": 1952, "n_validation": 418, "n_test": 420,
            "input_shape": [2790, 153, 5], "target_names": TARGET_NAMES,
            "input_files": {name: str(path) for name, path in data["paths"].items()},
            "input_sha256": data["file_hashes"],
        },
        "outputs": {
            "checkpoint": str(checkpoint_path), "metrics_csv": str(metrics_csv_path),
            "metrics_json": str(metrics_json_path), "predictions_wide": str(wide_path),
            "predictions_long": str(long_path), "training_history": str(history_path),
            "training_curve": str(curve_path), "log": str(log_path),
        },
    }
    metadata_path.write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Completed: best_epoch=%d stop_epoch=%d metrics=%s", best_epoch, stop_epoch, metrics.to_dict())
    print("\nDATA VALIDATION: PASSED", flush=True)
    print(f"PARAMETERS: {parameter_count}", flush=True)
    print(f"BEST EPOCH: {best_epoch}", flush=True)
    print(f"STOP EPOCH: {stop_epoch}", flush=True)
    print(metrics.to_string(index=False), flush=True)
    print(f"\nRESULTS: {RESULTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
