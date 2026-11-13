"""
Compact trainer for Stage 9: trains a single model with configurable data paths.
Supports group_split and more_rules data sources.
"""

import sys, json, pickle, argparse, logging, time
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from evaluate.metrics import compute_all_metrics, TARGET_NAMES
from preprocess.utils import ensure_dir, save_report_to_excel

RESULTS_DIR = PROJECT_ROOT / "results"

# Model builders
def build_model(name, input_dim):
    if name == "gru":
        from models.baselines import build_gru
        return build_gru(input_dim)
    elif name == "cnn_lstm":
        from models.baselines import build_cnn_lstm
        return build_cnn_lstm(input_dim)
    elif name == "grade_gru":
        from models.grade_aware_baselines import build_grade_aware_baseline
        return build_grade_aware_baseline(name, input_dim)
    elif name == "grade_transformer":
        from models.transformer_baselines import build_transformer
        return build_transformer(name, input_dim)
    elif name == "grade_tsmixer":
        from models.top_ts_models import build_model
        return build_model(name, input_dim)
    elif name == "grade_patchtst":
        from models.top_ts_models import build_model
        return build_model(name, input_dim)
    raise ValueError(name)


def inverse(y_s, scaler):
    return y_s * scaler["std"] + scaler["mean"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input_type", type=str, required=True, choices=["original6","wind6","wind7"])
    parser.add_argument("--data_source", type=str, required=True, choices=["main","group","more_rules"])
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32); parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.001); parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--tag_prefix", type=str, default="")
    args = parser.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve data paths
    if args.data_source == "main":
        ddir = PROJECT_ROOT / "data" / "processed"
        std = ddir / "standardized"; scl = ddir / "scalers"
        spath = ddir / "split_indices.json"; gpath = ddir / "grade_ids.npy"
        mpath = ddir / "grade_mapping.json"
    elif args.data_source == "group":
        ddir = PROJECT_ROOT / "data" / "processed"
        std = ddir / "group_standardized"; scl = ddir / "group_scalers"
        spath = ddir / "group_split_indices.json"; gpath = ddir / "grade_ids.npy"
        mpath = ddir / "grade_mapping.json"
    else:  # more_rules
        ddir = PROJECT_ROOT / "data" / "processed_more"
        std = ddir / "standardized"; scl = ddir / "scalers"
        spath = ddir / "split_indices.json"; gpath = ddir / "grade_ids.npy"
        mpath = ddir / "grade_mapping.json"

    xmap = {"original6":"X_original6","wind6":"X_wind6","wind7":"X_wind7"}
    xname = xmap[args.input_type]
    xfile = f"{xname}_scaled.npy" if args.data_source != "group" else f"{xname}_group_scaled.npy"
    yfile = "y_scaled.npy" if args.data_source != "group" else "y_group_scaled.npy"
    sfile = "y_scaler.pkl" if args.data_source != "group" else "y_group_scaler.pkl"
    xfile_scl = f"{xname}_scaler.pkl" if args.data_source != "group" else f"{xname}_group_scaler.pkl"

    X_all = np.load(str(std / xfile)); y_all = np.load(str(std / yfile))
    with open(spath) as f: split = json.load(f)
    with open(scl / sfile, "rb") as f: y_scaler = pickle.load(f)

    tr, va, te = split["train_idx"], split["val_idx"], split["test_idx"]
    X_tr, X_va, X_te = X_all[tr], X_all[va], X_all[te]
    yt_s, yv_s, yte_s = y_all[tr], y_all[va], y_all[te]

    is_grade = args.model.startswith("grade_")
    g_all = np.load(str(gpath)) if is_grade else None
    g_tr = torch.LongTensor(g_all[tr]) if is_grade else None
    g_va = torch.LongTensor(g_all[va]) if is_grade else None
    g_te = torch.LongTensor(g_all[te]) if is_grade else None

    F = X_tr.shape[2]
    tag = f"{args.tag_prefix}_{args.model}_{args.input_type}_seed{args.seed}" if args.tag_prefix else f"stage9_{args.data_source}_{args.model}_{args.input_type}_seed{args.seed}"
    print(f"Training: {tag} (F={F}, Ntr={len(tr)})")

    model = build_model(args.model, F).to(device)

    if is_grade:
        tr_ds = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(yt_s), g_tr)
        va_ds = TensorDataset(torch.FloatTensor(X_va), torch.FloatTensor(yv_s), g_va)
        te_ds = TensorDataset(torch.FloatTensor(X_te), g_te)
    else:
        tr_ds = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(yt_s))
        va_ds = TensorDataset(torch.FloatTensor(X_va), torch.FloatTensor(yv_s))
        te_ds = TensorDataset(torch.FloatTensor(X_te))
    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
    va_ld = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False)
    te_ld = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)
    crit = nn.MSELoss()
    best_vl, best_state, patience_cnt = float("inf"), None, 0
    best_ep = 0

    for ep in range(args.epochs):
        model.train(); tl = 0.0
        for batch in tr_ld:
            if is_grade: bx, by, bg = batch; bx, by, bg = bx.to(device), by.to(device), bg.to(device)
            else: bx, by = batch; bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            pred = model(bx, bg) if is_grade else model(bx)
            loss = crit(pred, by); loss.backward(); opt.step()
            tl += loss.item() * bx.size(0)
        tl /= len(tr_ld.dataset)

        model.eval(); vl = 0.0
        with torch.no_grad():
            for batch in va_ld:
                if is_grade: bx, by, bg = batch; bx, by, bg = bx.to(device), by.to(device), bg.to(device)
                else: bx, by = batch; bx, by = bx.to(device), by.to(device)
                vl += crit(model(bx, bg) if is_grade else model(bx), by).item() * bx.size(0)
        vl /= len(va_ld.dataset); sch.step(vl)

        if vl < best_vl:
            best_vl = vl; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0; best_ep = ep + 1
        else: patience_cnt += 1
        if (ep+1) % 50 == 0: print(f"  Ep {ep+1}: tr={tl:.4f} vl={vl:.4f}")
        if patience_cnt >= args.patience: print(f"  Early stop {ep+1}"); break

    model.load_state_dict(best_state)

    ckpt_dir = RESULTS_DIR / "checkpoints"
    ensure_dir(ckpt_dir)
    torch.save({"state": best_state, "best_epoch": best_ep, "args": vars(args)}, str(ckpt_dir / f"{tag}.pth"))

    model.eval(); preds = []
    with torch.no_grad():
        for batch in te_ld:
            if is_grade: bx, bg = batch; bx, bg = bx.to(device), bg.to(device); pred = model(bx, bg)
            else: (bx,) = batch; bx = bx.to(device); pred = model(bx)
            preds.append(pred.cpu().numpy())
    yp_s = np.concatenate(preds); yt = inverse(yte_s, y_scaler); yp = inverse(yp_s, y_scaler)

    md = compute_all_metrics(yt, yp)
    print("Test:\n" + md.to_string(index=False))

    tables_dir = RESULTS_DIR / "tables"; ensure_dir(tables_dir)
    save_report_to_excel({"metrics": md}, str(tables_dir / f"{tag}_metrics.xlsx"))
    rows = []
    for i in range(len(yt)):
        for j, nm in enumerate(TARGET_NAMES):
            rows.append({"sample_index": i, "target_name": nm, "y_true": yt[i,j], "y_pred": yp[i,j],
                         "input_type": args.input_type, "model_name": args.model, "seed": args.seed})
    pd.DataFrame(rows).to_csv(str(tables_dir / f"{tag}_predictions.csv"), index=False, encoding="utf-8-sig")
    print(f"Done: {tag}")
if __name__ == "__main__": main()
