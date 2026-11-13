"""
Train a single Stage 5 top-tier time series model.
CLI: python train/train_top_ts_model.py --model tsmixer --input_type wind7 --variant default
"""

import sys
import json
import pickle
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.top_ts_models import build_model
from evaluate.metrics import compute_all_metrics, TARGET_NAMES
from preprocess.utils import ensure_dir, save_report_to_excel

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STANDARDIZED_DIR = PROJECT_ROOT / "data" / "processed" / "standardized"
SCALERS_DIR = PROJECT_ROOT / "data" / "processed" / "scalers"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
LOGS_DIR = RESULTS_DIR / "logs"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
FIGURES_DIR = RESULTS_DIR / "figures"

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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(input_type):
    X = np.load(str(STANDARDIZED_DIR / INPUT_MAP[input_type]))
    y_s = np.load(str(STANDARDIZED_DIR / "y_scaled.npy"))
    with open(PROCESSED_DIR / "split_indices.json", "r") as f:
        split = json.load(f)
    with open(SCALERS_DIR / "y_scaler.pkl", "rb") as f:
        y_scaler = pickle.load(f)
    return (
        X[split["train_idx"]], X[split["val_idx"]], X[split["test_idx"]],
        y_s[split["train_idx"]], y_s[split["val_idx"]], y_s[split["test_idx"]],
        y_scaler,
    )


def inverse_y(y_s, y_scaler):
    return y_s * y_scaler["std"] + y_scaler["mean"]


def plot_training_curve(train_losses, val_losses, save_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="Train Loss", alpha=0.7)
    ax.plot(val_losses, label="Val Loss", alpha=0.7)
    ax.set_xlabel("Epoch"), ax.set_ylabel("MSE Loss")
    ax.set_title("Training Curve"), ax.legend(), ax.grid(True, alpha=0.3)
    fig.tight_layout(), fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["tsmixer", "patchtst", "grade_tsmixer", "grade_patchtst"])
    parser.add_argument("--input_type", type=str, required=True,
                        choices=["original6", "wind6", "wind7"])
    parser.add_argument("--variant", type=str, default="default",
                        choices=["default", "light"])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ensure_dir(TABLES_DIR); ensure_dir(LOGS_DIR)
    ensure_dir(CHECKPOINT_DIR); ensure_dir(FIGURES_DIR)

    tag = f"stage5_{args.model}_{args.input_type}_{args.variant}_seed{args.seed}"
    log_file = LOGS_DIR / f"{tag}.log"
    logging.basicConfig(filename=str(log_file), level=logging.INFO,
                        format="%(asctime)s %(message)s", filemode="w")
    logger = logging.getLogger()
    logger.info(f"Training {tag}, args: {args}")

    # Load data
    X_train, X_val, X_test, yt_s, yv_s, ytest_s, y_scaler = load_data(args.input_type)
    F = X_train.shape[2]
    logger.info(f"X_train={X_train.shape}, F={F}")

    # Load grade_ids if needed
    is_grade = args.model.startswith("grade_")
    grade_ids_train = grade_ids_val = grade_ids_test = None
    if is_grade:
        grade_ids = np.load(str(PROCESSED_DIR / "grade_ids.npy"))
        with open(PROCESSED_DIR / "split_indices.json", "r") as f:
            split = json.load(f)
        grade_ids_train = torch.LongTensor(grade_ids[split["train_idx"]])
        grade_ids_val = torch.LongTensor(grade_ids[split["val_idx"]])
        grade_ids_test = torch.LongTensor(grade_ids[split["test_idx"]])
        logger.info(f"Grade IDs loaded, num_grades from mapping")

    # Build model
    model = build_model(args.model, F, seq_len=153, variant=args.variant, num_grades=None)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Params: {n_params:,}")
    print(f"Model: {tag}, params: {n_params:,}")

    # DataLoaders
    if is_grade:
        train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(yt_s), grade_ids_train)
        val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(yv_s), grade_ids_val)
        test_ds = TensorDataset(torch.FloatTensor(X_test), grade_ids_test)
    else:
        train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(yt_s))
        val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(yv_s))
        test_ds = TensorDataset(torch.FloatTensor(X_test))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience_count = 0
    train_losses, val_losses, best_epoch = [], [], 0

    for epoch in range(args.epochs):
        model.train(); tl = 0.0
        for batch in train_loader:
            if is_grade:
                bx, by, bg = batch; bx, by, bg = bx.to(device), by.to(device), bg.to(device)
            else:
                bx, by = batch; bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx, bg) if is_grade else model(bx)
            loss = criterion(pred, by)
            loss.backward(); optimizer.step()
            tl += loss.item() * bx.size(0)
        tl /= len(train_loader.dataset); train_losses.append(tl)

        model.eval(); vl = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if is_grade:
                    bx, by, bg = batch; bx, by, bg = bx.to(device), by.to(device), bg.to(device)
                else:
                    bx, by = batch; bx, by = bx.to(device), by.to(device)
                pred = model(bx, bg) if is_grade else model(bx)
                vl += criterion(pred, by).item() * bx.size(0)
        vl /= len(val_loader.dataset); val_losses.append(vl)
        scheduler.step(vl)

        if vl < best_val_loss:
            best_val_loss = vl; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0; best_epoch = epoch + 1
        else:
            patience_count += 1

        if (epoch + 1) % 20 == 0:
            logger.info(f"Epoch {epoch+1}: train={tl:.6f}, val={vl:.6f}")
            print(f"  Epoch {epoch+1}/{args.epochs}: train={tl:.6f}, val={vl:.6f}")

        if patience_count >= args.patience:
            logger.info(f"Early stop at epoch {epoch+1}")
            print(f"  Early stop at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    logger.info(f"Best epoch: {best_epoch}, val_loss: {best_val_loss:.6f}")

    # Save checkpoint
    ckpt_path = CHECKPOINT_DIR / f"{tag}.pth"
    torch.save({"state": best_state, "best_epoch": best_epoch, "args": vars(args)}, str(ckpt_path))

    # Training curve
    plot_training_curve(train_losses, val_losses, FIGURES_DIR / f"{tag}_training_curve.png")

    # Test evaluation
    model.eval()
    test_preds = []
    with torch.no_grad():
        for batch in test_loader:
            if is_grade:
                bx, bg = batch; bx, bg = bx.to(device), bg.to(device)
                pred = model(bx, bg)
            else:
                (bx,) = batch; bx = bx.to(device)
                pred = model(bx)
            test_preds.append(pred.cpu().numpy())
    y_pred_s = np.concatenate(test_preds, axis=0)

    y_true = inverse_y(ytest_s, y_scaler)
    y_pred = inverse_y(y_pred_s, y_scaler)

    md = compute_all_metrics(y_true, y_pred)
    print("\nTest Metrics:\n" + md.to_string(index=False))

    save_report_to_excel({"metrics": md}, str(TABLES_DIR / f"{tag}_metrics.xlsx"))
    logger.info(f"Metrics saved")

    # Predictions CSV
    rows = []
    for i in range(len(y_true)):
        for j, nm in enumerate(TARGET_NAMES):
            rows.append({"sample_index": i, "target_name": nm, "y_true": y_true[i, j],
                         "y_pred": y_pred[i, j], "input_type": args.input_type,
                         "model_name": args.model, "variant": args.variant, "seed": args.seed})
    import pandas as pd
    pd.DataFrame(rows).to_csv(str(TABLES_DIR / f"{tag}_predictions.csv"), index=False, encoding="utf-8-sig")

    print(f"\nResults saved for {tag}")
    print("Done!")


if __name__ == "__main__":
    main()
