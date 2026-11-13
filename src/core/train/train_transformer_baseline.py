"""
Train a single Transformer-based baseline.
CLI: python train/train_transformer_baseline.py --model transformer --input_type wind7
"""

import sys, json, pickle, argparse, logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.transformer_baselines import build_transformer
from evaluate.metrics import compute_all_metrics, TARGET_NAMES
from preprocess.utils import ensure_dir, save_report_to_excel

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

STANDARDIZED_DIR = PROJECT_ROOT / "data" / "processed" / "standardized"
SCALERS_DIR = PROJECT_ROOT / "data" / "processed" / "scalers"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"; LOGS_DIR = RESULTS_DIR / "logs"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"; FIGURES_DIR = RESULTS_DIR / "figures"

INPUT_MAP = {"original6": "X_original6_scaled.npy", "wind6": "X_wind6_scaled.npy", "wind7": "X_wind7_scaled.npy"}


def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def inverse_y(y_s, s):
    return y_s * s["std"] + s["mean"]


def plot_curve(train_l, val_l, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_l, label="Train", alpha=0.7); ax.plot(val_l, label="Val", alpha=0.7)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE"); ax.set_title("Training Curve")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(str(path), dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["transformer","grade_transformer","transformer_lstm","grade_transformer_lstm"])
    parser.add_argument("--input_type", type=str, default="wind7",
                        choices=["original6","wind6","wind7"])
    parser.add_argument("--variant", type=str, default="default", choices=["default","light"])
    parser.add_argument("--epochs", type=int, default=300); parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001); parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=30); parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for d in [TABLES_DIR, LOGS_DIR, CHECKPOINT_DIR, FIGURES_DIR]: ensure_dir(d)

    tag = f"stage8_{args.model}_{args.input_type}_{args.variant}_seed{args.seed}"
    logging.basicConfig(filename=str(LOGS_DIR/f"{tag}.log"), level=logging.INFO,
                        format="%(asctime)s %(message)s", filemode="w")
    logger = logging.getLogger(); logger.info(f"Training {tag}")

    # Data
    X_all = np.load(str(STANDARDIZED_DIR / INPUT_MAP[args.input_type]))
    y_s_all = np.load(str(STANDARDIZED_DIR / "y_scaled.npy"))
    with open(PROCESSED_DIR / "split_indices.json") as f: split = json.load(f)
    with open(SCALERS_DIR / "y_scaler.pkl", "rb") as f: y_scaler = pickle.load(f)

    tr, va, te = split["train_idx"], split["val_idx"], split["test_idx"]
    X_tr, X_va, X_te = X_all[tr], X_all[va], X_all[te]
    yt_s, yv_s, ytest_s = y_s_all[tr], y_s_all[va], y_s_all[te]

    is_grade = args.model.startswith("grade_")
    g_ids = None
    if is_grade:
        g_ids = np.load(str(PROCESSED_DIR / "grade_ids.npy"))
    g_tr = torch.LongTensor(g_ids[tr]) if is_grade else None
    g_va = torch.LongTensor(g_ids[va]) if is_grade else None
    g_te = torch.LongTensor(g_ids[te]) if is_grade else None

    F = X_tr.shape[2]; logger.info(f"F={F}")
    model = build_transformer(args.model, F, seq_len=153, variant=args.variant).to(device)
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Params: {n_p:,}"); print(f"Model: {tag}, params: {n_p:,}")

    # DataLoaders
    if is_grade:
        tr_ds = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(yt_s), g_tr)
        va_ds = TensorDataset(torch.FloatTensor(X_va), torch.FloatTensor(yv_s), g_va)
        te_ds = TensorDataset(torch.FloatTensor(X_te), g_te)
    else:
        tr_ds = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(yt_s))
        va_ds = TensorDataset(torch.FloatTensor(X_va), torch.FloatTensor(yv_s))
        te_ds = TensorDataset(torch.FloatTensor(X_te))
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False)
    te_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)
    crit = nn.MSELoss()

    best_vl, best_state, patience_cnt = float("inf"), None, 0
    tr_losses, vl_losses, best_ep = [], [], 0

    for ep in range(args.epochs):
        model.train(); tl = 0.0
        for batch in tr_loader:
            if is_grade:
                bx, by, bg = batch; bx, by, bg = bx.to(device), by.to(device), bg.to(device)
            else:
                bx, by = batch; bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            pred = model(bx, bg) if is_grade else model(bx)
            loss = crit(pred, by); loss.backward(); opt.step()
            tl += loss.item() * bx.size(0)
        tl /= len(tr_loader.dataset); tr_losses.append(tl)

        model.eval(); vl = 0.0
        with torch.no_grad():
            for batch in va_loader:
                if is_grade:
                    bx, by, bg = batch; bx, by, bg = bx.to(device), by.to(device), bg.to(device)
                else:
                    bx, by = batch; bx, by = bx.to(device), by.to(device)
                pred = model(bx, bg) if is_grade else model(bx)
                vl += crit(pred, by).item() * bx.size(0)
        vl /= len(va_loader.dataset); vl_losses.append(vl); sch.step(vl)

        if vl < best_vl:
            best_vl = vl; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0; best_ep = ep + 1
        else:
            patience_cnt += 1

        if (ep+1) % 30 == 0:
            logger.info(f"Epoch {ep+1}: train={tl:.6f}, val={vl:.6f}")
            print(f"  Epoch {ep+1}/{args.epochs}: train={tl:.6f}, val={vl:.6f}")
        if patience_cnt >= args.patience:
            logger.info(f"Early stop at {ep+1}"); print(f"  Early stop at {ep+1}"); break

    model.load_state_dict(best_state)
    torch.save({"state": best_state, "best_epoch": best_ep, "args": vars(args)},
               str(CHECKPOINT_DIR / f"{tag}.pth"))
    plot_curve(tr_losses, vl_losses, FIGURES_DIR / f"{tag}_training_curve.png")

    model.eval(); preds = []
    with torch.no_grad():
        for batch in te_loader:
            if is_grade:
                bx, bg = batch; bx, bg = bx.to(device), bg.to(device)
                pred = model(bx, bg)
            else:
                (bx,) = batch; bx = bx.to(device)
                pred = model(bx)
            preds.append(pred.cpu().numpy())
    yp_s = np.concatenate(preds, axis=0)
    yt = inverse_y(ytest_s, y_scaler); yp = inverse_y(yp_s, y_scaler)

    md = compute_all_metrics(yt, yp)
    print("\nTest Metrics:\n" + md.to_string(index=False))
    save_report_to_excel({"metrics": md}, str(TABLES_DIR / f"{tag}_metrics.xlsx"))

    rows = []
    for i in range(len(yt)):
        for j, nm in enumerate(TARGET_NAMES):
            rows.append({"sample_index": i, "target_name": nm, "y_true": yt[i, j],
                         "y_pred": yp[i, j], "input_type": args.input_type,
                         "model_name": args.model, "variant": args.variant, "seed": args.seed})
    pd.DataFrame(rows).to_csv(str(TABLES_DIR / f"{tag}_predictions.csv"), index=False, encoding="utf-8-sig")
    print(f"\nResults saved for {tag}\nDone!")


if __name__ == "__main__":
    main()
