"""
Train a single baseline model on a single input type.
CLI: python train/train_baselines.py --input_type original6 --model ridge
"""

import sys
import json
import pickle
import argparse
import logging
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.baselines import (
    build_ridge, build_random_forest, build_xgboost,
    build_lstm, build_gru, build_cnn_lstm,
    flatten_X, HAS_XGBOOST,
)
from evaluate.metrics import compute_all_metrics, TARGET_NAMES
from preprocess.utils import ensure_dir, save_report_to_excel

STANDARDIZED_DIR = PROJECT_ROOT / "data" / "processed" / "standardized"
SCALERS_DIR = PROJECT_ROOT / "data" / "processed" / "scalers"
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
LOGS_DIR = RESULTS_DIR / "logs"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"

INPUT_MAP = {
    "original6": "X_original6_scaled.npy",
    "wind6": "X_wind6_scaled.npy",
    "wind7": "X_wind7_scaled.npy",
}


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(input_type):
    """Load standardized data, y, splits, scalers."""
    X = np.load(str(STANDARDIZED_DIR / INPUT_MAP[input_type]))
    y_scaled = np.load(str(STANDARDIZED_DIR / "y_scaled.npy"))

    with open(PROJECT_ROOT / "data" / "processed" / "split_indices.json", "r") as f:
        split = json.load(f)

    with open(SCALERS_DIR / "y_scaler.pkl", "rb") as f:
        y_scaler = pickle.load(f)

    train_idx = split["train_idx"]
    val_idx = split["val_idx"]
    test_idx = split["test_idx"]

    return (
        X[train_idx], X[val_idx], X[test_idx],
        y_scaled[train_idx], y_scaled[val_idx], y_scaled[test_idx],
        y_scaler, train_idx, val_idx, test_idx,
    )


def inverse_transform_y(y_scaled, y_scaler):
    """Inverse standardize y (N, 5)."""
    mean = y_scaler["mean"]
    std = y_scaler["std"]
    return y_scaled * std + mean


def train_ml_model(model, X_train, y_train, model_name):
    """Train ML model with flattened input."""
    X_flat = flatten_X(X_train)
    model.fit(X_flat, y_train)
    return model


def predict_ml_model(model, X):
    """Predict with flattened input."""
    return model.predict(flatten_X(X))


def train_dl_model(model, X_train, y_train, X_val, y_val,
                   epochs=200, batch_size=32, lr=1e-3, patience=20, device="cpu"):
    """Train DL model with early stopping."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val), torch.FloatTensor(y_val))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)
        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                pred = model(batch_X)
                loss = criterion(pred, batch_y)
                val_loss += loss.item() * batch_X.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    model.load_state_dict(best_state)
    return model, train_losses, val_losses, epoch + 1 - patience_counter


def predict_dl_model(model, X, batch_size=32, device="cpu"):
    """Predict with DL model."""
    model.eval()
    model = model.to(device)
    dataset = TensorDataset(torch.FloatTensor(X))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions = []
    with torch.no_grad():
        for (batch_X,) in loader:
            batch_X = batch_X.to(device)
            pred = model(batch_X)
            predictions.append(pred.cpu().numpy())
    return np.concatenate(predictions, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_type", type=str, required=True,
                        choices=["original6", "wind6", "wind7"])
    parser.add_argument("--model", type=str, required=True,
                        choices=["ridge", "rf", "xgboost", "lstm", "gru", "cnn_lstm"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ensure_dir(TABLES_DIR)
    ensure_dir(LOGS_DIR)
    ensure_dir(CHECKPOINT_DIR)

    # Setup logging
    log_file = LOGS_DIR / f"{args.input_type}_{args.model}_train.log"
    logging.basicConfig(
        filename=str(log_file), level=logging.INFO,
        format="%(asctime)s %(message)s", filemode="w")
    logger = logging.getLogger()
    logger.info(f"Training {args.model} on {args.input_type}")
    logger.info(f"Args: {args}")

    # Load data
    X_train, X_val, X_test, y_train_s, y_val_s, y_test_s, y_scaler, train_idx, val_idx, test_idx = load_data(args.input_type)
    F = X_train.shape[2]
    logger.info(f"Data loaded: X_train={X_train.shape}, y_train_s={y_train_s.shape}")
    logger.info(f"X_val={X_val.shape}, X_test={X_test.shape}")

    # Build model
    if args.model == "ridge":
        model = build_ridge()
        is_dl = False
    elif args.model == "rf":
        model = build_random_forest()
        is_dl = False
    elif args.model == "xgboost":
        if not HAS_XGBOOST:
            logger.warning("XGBoost not available, skipping")
            print("[WARNING] XGBoost not installed, skipping.")
            return
        model = build_xgboost()
        is_dl = False
    elif args.model == "lstm":
        model = build_lstm(F)
        is_dl = True
    elif args.model == "gru":
        model = build_gru(F)
        is_dl = True
    elif args.model == "cnn_lstm":
        model = build_cnn_lstm(F)
        is_dl = True
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Train
    print(f"\nTraining {args.model} on {args.input_type} ...")
    best_epoch = None

    if is_dl:
        model, train_losses, val_losses, best_epoch = train_dl_model(
            model, X_train, y_train_s, X_val, y_val_s,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, patience=20, device=device,
        )
        logger.info(f"Best epoch: {best_epoch}, best val loss: {min(val_losses):.6f}")
        print(f"  Best epoch: {best_epoch}, best val loss: {min(val_losses):.6f}")

        # Save checkpoint
        ckpt_path = CHECKPOINT_DIR / f"{args.input_type}_{args.model}.pth"
        torch.save({
            "model_state_dict": model.state_dict(),
            "input_size": F,
            "best_epoch": best_epoch,
        }, str(ckpt_path))
        print(f"  Checkpoint saved: {ckpt_path}")

        # Predict
        y_train_pred_s = predict_dl_model(model, X_train, device=device)
        y_val_pred_s = predict_dl_model(model, X_val, device=device)
        y_test_pred_s = predict_dl_model(model, X_test, device=device)
    else:
        model = train_ml_model(model, X_train, y_train_s, args.model)
        logger.info("ML model trained successfully")
        print("  ML model trained successfully")

        # Save checkpoint
        ckpt_path = CHECKPOINT_DIR / f"{args.input_type}_{args.model}.pkl"
        with open(str(ckpt_path), "wb") as f:
            pickle.dump(model, f)
        print(f"  Checkpoint saved: {ckpt_path}")

        y_train_pred_s = predict_ml_model(model, X_train)
        y_val_pred_s = predict_ml_model(model, X_val)
        y_test_pred_s = predict_ml_model(model, X_test)

    # Inverse transform
    y_train_true = inverse_transform_y(y_train_s, y_scaler)
    y_val_true = inverse_transform_y(y_val_s, y_scaler)
    y_test_true = inverse_transform_y(y_test_s, y_scaler)
    y_train_pred = inverse_transform_y(y_train_pred_s, y_scaler)
    y_val_pred = inverse_transform_y(y_val_pred_s, y_scaler)
    y_test_pred = inverse_transform_y(y_test_pred_s, y_scaler)

    # Metrics on test set
    metrics_df = compute_all_metrics(y_test_true, y_test_pred)
    print("\nTest Metrics:")
    print(metrics_df.to_string(index=False))

    # Save metrics
    metrics_path = TABLES_DIR / f"{args.input_type}_{args.model}_metrics.xlsx"
    save_report_to_excel({"metrics": metrics_df}, str(metrics_path))
    logger.info(f"Metrics saved: {metrics_path}")

    # Save predictions
    pred_df = pd.DataFrame(
        np.hstack([y_test_true, y_test_pred]),
        columns=[f"true_{n}" for n in TARGET_NAMES] + [f"pred_{n}" for n in TARGET_NAMES],
    )
    pred_df.to_csv(
        str(TABLES_DIR / f"{args.input_type}_{args.model}_predictions.csv"),
        index=False, encoding="utf-8-sig",
    )

    print(f"\nResults saved to: {TABLES_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
